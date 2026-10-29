from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable, Iterator
from urllib.parse import parse_qs

import httpx
import pytest

from authfi import ConfigurationError, TransportError
from authfi._clock import SystemClock
from authfi._protocol.http import Attempt, HttpRequest, Retry
from authfi._protocol.ids import is_valid_request_id
from authfi._protocol.oauth import ClientAuth, client_credentials_request
from authfi._protocol.retry import RetryPolicy, parse_retry_after
from authfi._transport import AsyncTransport, SyncTransport, Timeout, user_agent
from authfi._version import __version__
from tests.support import FakeClock

URL = "https://acme.authfi.io/oauth/token"
Fail = Callable[[httpx.Request], httpx.Response]


def connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("refused", request=request)


def read_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadError("reset after send", request=request)


def status(code: int, retry_after: str | None = None) -> Fail:
    def answer(_request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": retry_after} if retry_after else {}
        return httpx.Response(code, json={"error": "x"}, headers=headers)

    return answer


class Server:
    """Fails the first ``failures`` requests with ``fail``, then answers 200."""

    def __init__(self, fail: Fail, failures: int = 1) -> None:
        self.fail = fail
        self.failures = failures
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if len(self.requests) <= self.failures:
            return self.fail(request)
        return httpx.Response(200, json={"ok": True})


def transport(server: Callable[[httpx.Request], httpx.Response], clock: FakeClock, retries: int = 2) -> SyncTransport:
    return SyncTransport(
        client=httpx.Client(transport=httpx.MockTransport(server)),
        timeout=Timeout(),
        policy=RetryPolicy(retries=retries, rng=lambda _lo, hi: hi),
        agent=user_agent(),
        logger=logging.getLogger("authfi"),
        clock=clock,
    )


def request(retry: Retry, method: str = "POST") -> HttpRequest:
    return HttpRequest(method, URL, what="the token endpoint", form={"a": "b"}, retry=retry)  # type: ignore[arg-type]


# (class, failure, retried?)
TABLE: list[tuple[Retry, str, Fail, bool]] = [
    (Retry.SAFE, "refused connection", connect_error, True),
    (Retry.SAFE, "reset after send", read_error, True),
    (Retry.SAFE, "429", status(429), True),
    (Retry.SAFE, "503 without Retry-After", status(503), True),
    (Retry.SAFE, "503 with Retry-After", status(503, "1"), True),
    (Retry.SAFE, "502", status(502), True),
    (Retry.SAFE, "504", status(504), True),
    (Retry.SAFE, "400 refusal", status(400), False),
    (Retry.SAFE, "401 refusal", status(401), False),
    (Retry.NEVER_AFTER_SEND, "refused connection", connect_error, True),
    (Retry.NEVER_AFTER_SEND, "reset after send", read_error, False),
    (Retry.NEVER_AFTER_SEND, "429", status(429), True),
    (Retry.NEVER_AFTER_SEND, "503 without Retry-After", status(503), False),
    (Retry.NEVER_AFTER_SEND, "503 with Retry-After", status(503, "1"), True),
    (Retry.NEVER_AFTER_SEND, "502", status(502), False),
    (Retry.NEVER_AFTER_SEND, "400 refusal", status(400), False),
]


@pytest.mark.parametrize(("retry", "name", "fail", "retried"), TABLE, ids=[f"{t[0].value}-{t[1]}" for t in TABLE])
def test_the_retry_table(retry: Retry, name: str, fail: Fail, retried: bool, clock: FakeClock) -> None:
    server = Server(fail)
    t = transport(server, clock)
    if retried:
        assert t.send(request(retry)).status == 200
        assert len(server.requests) == 2
    elif fail in (connect_error, read_error):
        with pytest.raises(TransportError) as caught:
            t.send(request(retry))
        assert not caught.value.retryable
        assert len(server.requests) == 1
    else:
        assert t.send(request(retry)).status != 200
        assert len(server.requests) == 1


@pytest.mark.parametrize(("retry", "name", "fail", "retried"), TABLE, ids=[f"{t[0].value}-{t[1]}" for t in TABLE])
async def test_the_retry_table_async(retry: Retry, name: str, fail: Fail, retried: bool, clock: FakeClock) -> None:
    server = Server(fail)
    t = AsyncTransport(
        client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        timeout=Timeout(),
        policy=RetryPolicy(retries=2, rng=lambda _lo, hi: hi),
        agent=user_agent(),
        logger=logging.getLogger("authfi"),
        clock=clock,
    )
    if retried:
        assert (await t.send(request(retry))).status == 200
        assert len(server.requests) == 2
    elif fail in (connect_error, read_error):
        with pytest.raises(TransportError) as caught:
            await t.send(request(retry))
        assert not caught.value.retryable
        assert len(server.requests) == 1
    else:
        assert (await t.send(request(retry))).status != 200
        assert len(server.requests) == 1


def test_an_idempotency_key_makes_a_request_safe_to_repeat(clock: FakeClock) -> None:
    server = Server(read_error)
    req = HttpRequest(
        "POST", URL, what="x", headers={"Idempotency-Key": "k-1"}, form={"a": "b"}, retry=Retry.NEVER_AFTER_SEND
    )
    assert transport(server, clock).send(req).status == 200
    assert [r.headers["Idempotency-Key"] for r in server.requests] == ["k-1", "k-1"]


def test_retries_stop_at_the_limit_and_report_the_failure(clock: FakeClock) -> None:
    server = Server(connect_error, failures=10)
    with pytest.raises(TransportError) as caught:
        transport(server, clock, retries=2).send(request(Retry.SAFE))
    assert len(server.requests) == 3
    assert caught.value.code == "connection_failed"
    assert caught.value.retryable


def test_a_server_error_after_the_last_retry_is_returned_for_the_caller_to_read(clock: FakeClock) -> None:
    server = Server(status(503), failures=10)
    response = transport(server, clock, retries=1).send(request(Retry.SAFE))
    assert response.status == 503
    assert len(server.requests) == 2


def test_backoff_is_full_jitter_and_retry_after_is_honoured(clock: FakeClock) -> None:
    server = Server(status(429, "3"), failures=1)
    transport(server, clock).send(request(Retry.SAFE))
    assert clock.sleeps == [3.0]
    clock.sleeps.clear()
    server = Server(connect_error, failures=2)
    transport(server, clock).send(request(Retry.SAFE))
    assert clock.sleeps == [0.2, 0.4]


def test_a_retry_after_beyond_the_deadline_is_not_waited_for(clock: FakeClock) -> None:
    server = Server(status(429, "120"), failures=1)
    response = transport(server, clock).send(request(Retry.SAFE))
    assert response.status == 429
    assert len(server.requests) == 1


def test_one_request_id_per_logical_call_the_same_on_every_retry(clock: FakeClock) -> None:
    server = Server(connect_error, failures=2)
    t = transport(server, clock)
    t.send(request(Retry.SAFE))
    ids = {r.headers["X-Request-Id"] for r in server.requests}
    assert len(ids) == 1
    assert is_valid_request_id(ids.pop())
    t.send(request(Retry.SAFE))
    assert server.requests[-1].headers["X-Request-Id"] != server.requests[0].headers["X-Request-Id"]


def test_every_attempt_gets_its_own_fresh_credential(clock: FakeClock) -> None:
    made: list[int] = []

    def per_attempt() -> Attempt:
        made.append(len(made))
        return Attempt(form={"client_assertion": f"assertion-{len(made)}"})

    server = Server(connect_error, failures=2)
    req = HttpRequest("POST", URL, what="mint", form={"grant_type": "client_credentials"}, per_attempt=per_attempt)
    transport(server, clock).send(req)
    sent = [r.content.decode() for r in server.requests]
    assert sent == [
        "grant_type=client_credentials&client_assertion=assertion-1",
        "grant_type=client_credentials&client_assertion=assertion-2",
        "grant_type=client_credentials&client_assertion=assertion-3",
    ]


def test_a_client_assertion_is_signed_afresh_for_every_attempt_of_a_token_request(clock: FakeClock) -> None:
    """A retried token request must not replay its assertion: the server records each ``jti`` once."""
    count = 0

    def sign(_audience: str) -> str:
        nonlocal count
        count += 1
        return f"assertion-{count}"

    auth = ClientAuth.private_key_jwt("agt_1", sign)
    server = Server(status(503), failures=1)
    response = transport(server, clock).send(client_credentials_request(URL, auth))
    assert response.status == 200
    assertions = [parse_qs(r.content.decode())["client_assertion"] for r in server.requests]
    assert assertions == [["assertion-1"], ["assertion-2"]]


def test_the_user_agent_names_the_sdk(clock: FakeClock) -> None:
    server = Server(status(200), failures=0)
    transport(server, clock).send(request(Retry.SAFE))
    agent = server.requests[0].headers["User-Agent"]
    assert agent.startswith(f"authfi-python/{__version__} (python 3.")
    assert "httpx " in agent
    assert user_agent("myapp/1.0").endswith(" myapp/1.0")
    assert "traceparent" in server.requests[0].headers


def test_redirects_are_refused_not_followed(clock: FakeClock) -> None:
    server = Server(lambda r: httpx.Response(302, headers={"Location": "https://evil.example/"}), failures=1)
    with pytest.raises(TransportError) as caught:
        transport(server, clock).send(request(Retry.SAFE))
    assert caught.value.code == "redirect_refused"
    assert len(server.requests) == 1


@pytest.mark.parametrize("url", ["http://acme.authfi.io/x", "ftp://acme.authfi.io/x"])
def test_plain_http_is_refused_off_loopback(url: str, clock: FakeClock) -> None:
    server = Server(status(200), failures=0)
    with pytest.raises(ConfigurationError):
        transport(server, clock).send(HttpRequest("GET", url, what="x"))
    assert server.requests == []


def test_http_to_loopback_is_allowed(clock: FakeClock) -> None:
    server = Server(status(200), failures=0)
    assert transport(server, clock).send(HttpRequest("GET", "http://127.0.0.1:8080/x", what="x")).status == 200


def test_an_oversized_answer_is_refused(clock: FakeClock) -> None:
    big = b"x" * 2048
    server = Server(lambda r: httpx.Response(200, content=big), failures=1)
    with pytest.raises(TransportError) as caught:
        transport(server, clock).send(HttpRequest("GET", URL, what="x", max_bytes=1024))
    assert caught.value.code == "response_too_large"

    def streamed(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=iter([b"x" * 600, b"x" * 600]))

    with pytest.raises(TransportError):
        transport(Server(streamed, failures=1), clock).send(HttpRequest("GET", URL, what="x", max_bytes=1024))


async def _one(chunk: bytes) -> AsyncIterator[bytes]:
    yield chunk


def _async_transport(
    server: Callable[[httpx.Request], httpx.Response], clock: object, timeout: Timeout
) -> AsyncTransport:
    return AsyncTransport(
        client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        timeout=timeout,
        policy=RetryPolicy(retries=0),
        agent=user_agent(),
        logger=logging.getLogger("authfi"),
        clock=clock,  # type: ignore[arg-type]
    )


def test_a_body_dripped_past_the_total_deadline_is_refused(clock: FakeClock) -> None:
    """A server that sends one byte a second cannot hold the call past ``Timeout.total``."""
    sent = 0

    def drip() -> Iterator[bytes]:
        nonlocal sent
        for _ in range(270):
            sent += 1
            clock.advance(1.0)
            yield b"x"

    server = Server(lambda _r: httpx.Response(200, content=drip()), failures=1)
    t = SyncTransport(
        client=httpx.Client(transport=httpx.MockTransport(server)),
        timeout=Timeout(total=5.0),
        policy=RetryPolicy(retries=0),
        agent=user_agent(),
        logger=logging.getLogger("authfi"),
        clock=clock,
    )
    with pytest.raises(TransportError) as caught:
        t.send(HttpRequest("GET", URL, what="the token endpoint"))
    assert caught.value.code == "timeout"
    assert not caught.value.retryable
    assert sent <= 6


async def test_an_async_body_dripped_past_the_total_deadline_is_refused(clock: FakeClock) -> None:
    sent = 0

    async def drip() -> AsyncIterator[bytes]:
        nonlocal sent
        for _ in range(270):
            sent += 1
            clock.advance(1.0)
            yield b"x"

    server = Server(lambda _r: httpx.Response(200, content=drip()), failures=1)
    t = _async_transport(server, clock, Timeout(total=5.0))
    with pytest.raises(TransportError) as caught:
        await t.send(HttpRequest("GET", URL, what="the token endpoint"))
    assert caught.value.code == "timeout"
    assert sent <= 6


async def test_an_async_body_that_stops_arriving_is_cut_at_the_deadline() -> None:
    """No chunk arrives to check the clock on: the event loop's own timer ends the wait."""

    async def stalls() -> AsyncIterator[bytes]:
        yield b"x"
        await asyncio.sleep(30)
        yield b"y"  # pragma: no cover

    server = Server(lambda _r: httpx.Response(200, content=stalls()), failures=1)
    t = _async_transport(server, SystemClock(), Timeout(connect=1.0, read=1.0, total=0.3))
    started = time.monotonic()
    with pytest.raises(TransportError) as caught:
        await t.send(HttpRequest("GET", URL, what="the token endpoint"))
    assert caught.value.code == "timeout"
    assert time.monotonic() - started < 5, "the wait ran on to the server's own pace instead of the deadline"


def test_a_body_that_cannot_be_decoded_is_a_transport_error_not_an_httpx_one(clock: FakeClock) -> None:
    def bad_gzip(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=iter([b"this is not gzip"]))

    with pytest.raises(TransportError) as caught:
        transport(Server(bad_gzip, failures=1), clock).send(HttpRequest("GET", URL, what="the token endpoint"))
    assert caught.value.code == "response_invalid"
    assert not caught.value.retryable


async def test_an_async_body_that_cannot_be_decoded_is_a_transport_error(clock: FakeClock) -> None:
    def bad_gzip(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=_one(b"this is not gzip"))

    t = _async_transport(Server(bad_gzip, failures=1), clock, Timeout())
    with pytest.raises(TransportError) as caught:
        await t.send(HttpRequest("GET", URL, what="the token endpoint"))
    assert caught.value.code == "response_invalid"


def test_timeouts_are_passed_per_phase_and_capped_by_the_deadline(clock: FakeClock) -> None:
    server = Server(status(200), failures=0)
    t = transport(server, clock)
    t.send(request(Retry.SAFE))
    timeout = server.requests[0].extensions["timeout"]
    assert timeout["connect"] == 5.0
    assert timeout["read"] == 10.0


def test_a_deprecation_is_logged_once_per_url(clock: FakeClock, caplog: pytest.LogCaptureFixture) -> None:
    def deprecated(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={},
            headers=[
                ("Deprecation", "@1760000000"),
                ("Link", '<https://acme.authfi.io/oauth/token>; rel="successor-version"'),
            ],
        )

    t = transport(Server(deprecated, failures=5), clock)
    with caplog.at_level(logging.WARNING, logger="authfi"):
        for _ in range(3):
            t.send(request(Retry.SAFE))
    messages = [r.getMessage() for r in caplog.records if "deprecated" in r.getMessage()]
    assert len(messages) == 1
    assert "successor is https://acme.authfi.io/oauth/token" in messages[0]


def test_a_traceparent_an_edge_dropped_is_named_once(clock: FakeClock, caplog: pytest.LogCaptureFixture) -> None:
    def new_trace(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={}, headers={"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"})

    t = transport(Server(new_trace, failures=5), clock)
    with caplog.at_level(logging.WARNING, logger="authfi"):
        first = t.send(request(Retry.SAFE))
        t.send(request(Retry.SAFE))
    assert first.dropped_header == "traceparent"
    assert len([r for r in caplog.records if "did not arrive" in r.getMessage()]) == 1


def test_a_dpop_challenge_on_a_request_that_carried_a_proof_names_the_dropped_header(clock: FakeClock) -> None:
    def challenge(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(401, headers={"WWW-Authenticate": 'DPoP error="invalid_token"'})

    req = HttpRequest("GET", URL, what="x", headers={"DPoP": "proof"})
    assert transport(Server(challenge, failures=5), clock).send(req).dropped_header == "DPoP"


def test_an_emdt013_on_a_request_that_carried_an_idempotency_key_names_the_dropped_header(clock: FakeClock) -> None:
    def refused(_r: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "EMDT013"})

    req = HttpRequest("POST", URL, what="x", headers={"Idempotency-Key": "k"})
    assert transport(Server(refused, failures=5), clock).send(req).dropped_header == "Idempotency-Key"


def test_the_request_repr_shows_no_body() -> None:
    req = HttpRequest("POST", URL + "?q=1", what="x", form={"client_secret": "MARKER"})
    assert "MARKER" not in repr(req)
    assert "q=1" not in repr(req)


@pytest.mark.parametrize("value", [0, -1, "5", True])
def test_timeout_values_are_checked(value: object) -> None:
    with pytest.raises(ConfigurationError):
        Timeout.of(value)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        Timeout(connect=value)  # type: ignore[arg-type]


def test_a_number_is_the_total_deadline() -> None:
    assert Timeout.of(3) == Timeout(connect=3.0, read=3.0, total=3.0)
    assert Timeout.of(60) == Timeout(connect=5.0, read=10.0, total=60.0)


def test_retry_after_dates_and_nonsense() -> None:
    assert parse_retry_after("7", 0) == 7.0
    assert parse_retry_after(None, 0) is None
    assert parse_retry_after("soon", 0) is None
    assert parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT", 1445412470.0) == 10.0


async def test_an_async_transport_belongs_to_one_event_loop(clock: FakeClock) -> None:
    import asyncio
    import threading

    server = Server(status(200), failures=0)
    t = AsyncTransport(
        client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
        timeout=Timeout(),
        policy=RetryPolicy(),
        agent=user_agent(),
        logger=logging.getLogger("authfi"),
        clock=clock,
    )
    await t.send(request(Retry.SAFE))
    errors: list[BaseException] = []

    def other_loop() -> None:
        try:
            asyncio.run(t.send(request(Retry.SAFE)))
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=other_loop)
    thread.start()
    thread.join()
    assert len(errors) == 1
    assert isinstance(errors[0], ConfigurationError)
    assert "event loop" in str(errors[0])
