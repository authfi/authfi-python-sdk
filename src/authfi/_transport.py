"""The one HTTP core: httpx under the protocol layer, sync and async.

Every request the SDK makes goes through here, so every request gets the same rules:

- connect, read and total-deadline timeouts (:class:`Timeout`);
- the retry table of :mod:`authfi._protocol.retry`, with a fresh client assertion per attempt;
- no redirects: a 3xx from an OAuth or API endpoint is an error, never followed;
- answers read up to a size limit;
- a named ``User-Agent`` (the edge refuses anonymous clients), one ``X-Request-Id`` per logical
  call, the same on every retry, and ``traceparent``;
- a ``Deprecation`` answer logged once per URL, and a header an edge dropped reported once per
  origin;
- plain ``http`` refused off loopback.
"""

from __future__ import annotations

import asyncio
import logging
import platform
from collections.abc import Generator
from dataclasses import dataclass
from typing import TypeVar, cast
from urllib.parse import urlsplit

import httpx

from authfi._clock import Clock
from authfi._log import warn_once
from authfi._protocol.http import Attempt, Effect, HttpRequest, HttpResponse, Retry, Sleep, merge_headers
from authfi._protocol.ids import inject_traceparent, request_id, trace_id_of
from authfi._protocol.retry import Failure, RetryPolicy
from authfi._protocol.urls import is_loopback, origin_of
from authfi._version import __version__
from authfi.errors import ConfigurationError, TransportError

__all__ = ["DEFAULT_TIMEOUT", "AsyncTransport", "SyncTransport", "Timeout", "user_agent"]

T = TypeVar("T")

_LINK_TARGET = "successor-version"


@dataclass(frozen=True, slots=True)
class Timeout:
    """How long the SDK waits.

    Attributes:
        connect: seconds to establish a connection.
        read: seconds to wait for each read of the answer.
        total: the deadline for one logical call, retries included.

    """

    connect: float = 5.0
    read: float = 10.0
    total: float = 30.0

    def __post_init__(self) -> None:
        for name in ("connect", "read", "total"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
                raise ConfigurationError(f"Timeout.{name} must be a positive number of seconds, not {value!r}")

    @classmethod
    def of(cls, value: Timeout | float) -> Timeout:
        """``value`` as a :class:`Timeout`; a number is the total deadline, with the phases inside it."""
        if isinstance(value, Timeout):
            return value
        number = cast(object, value)
        if not isinstance(number, int | float) or isinstance(number, bool) or number <= 0:
            raise ConfigurationError(f"timeout must be a Timeout or a positive number of seconds, not {number!r}")
        total = float(number)
        return cls(connect=min(5.0, total), read=min(10.0, total), total=total)


#: Connect 5 s, read 10 s, 30 s per logical call.
DEFAULT_TIMEOUT = Timeout()


def user_agent(suffix: str | None = None) -> str:
    """``authfi-python/<version> (python <x.y>; httpx <version>)`` plus the caller's suffix."""
    python = ".".join(platform.python_version_tuple()[:2])
    value = f"authfi-python/{__version__} (python {python}; httpx {httpx.__version__})"
    return f"{value} {suffix}" if suffix else value


def _check_scheme(url: str) -> None:
    parts = urlsplit(url)
    if parts.scheme == "https" or (parts.scheme == "http" and is_loopback(parts.hostname)):
        return
    raise ConfigurationError(f"refusing to send a request to {origin_of(url)}: only https, or http to loopback")


def _form_of(request: HttpRequest, attempt: Attempt) -> dict[str, str] | None:
    if request.form is None and not attempt.form:
        return None
    return {**(request.form or {}), **attempt.form}


def _failure_of(exc: httpx.TransportError) -> Failure:
    if isinstance(exc, httpx.ConnectError | httpx.ConnectTimeout | httpx.PoolTimeout):
        return Failure.NOT_SENT
    return Failure.AFTER_SEND


def _transport_error(exc: httpx.TransportError, request: HttpRequest, rid: str, sent_safe: bool) -> TransportError:
    code = "timeout" if isinstance(exc, httpx.TimeoutException) else "connection_failed"
    retryable = sent_safe or _failure_of(exc) is Failure.NOT_SENT
    return TransportError(
        f"{request.what} at {origin_of(request.url)} could not be reached: {type(exc).__name__}",
        code=code,
        retryable=retryable,
        request_id=rid,
    )


def _successor(links: list[str]) -> str | None:
    for value in links:
        for part in value.split(","):
            if _LINK_TARGET in part and "<" in part and ">" in part:
                return part[part.index("<") + 1 : part.index(">")]
    return None


class _Base:
    """What the sync and async drivers share: request preparation, diagnostics, the retry policy."""

    def __init__(
        self,
        *,
        timeout: Timeout,
        policy: RetryPolicy,
        agent: str,
        logger: logging.Logger,
        clock: Clock,
    ) -> None:
        self.timeout = timeout
        self.policy = policy
        self.agent = agent
        self.log = logger
        self.clock = clock

    def _httpx_timeout(self, deadline: float) -> httpx.Timeout:
        remaining = max(0.001, deadline - self.clock.monotonic())
        connect = min(self.timeout.connect, remaining)
        read = min(self.timeout.read, remaining)
        return httpx.Timeout(connect=connect, read=read, write=read, pool=connect)

    def _headers(self, request: HttpRequest, attempt: Attempt, rid: str) -> dict[str, str]:
        base = {"User-Agent": self.agent, "X-Request-Id": rid}
        headers = merge_headers(base, request.headers, attempt.headers)
        inject_traceparent(headers)
        return headers

    def _build(
        self, client: httpx.Client | httpx.AsyncClient, request: HttpRequest, rid: str, deadline: float
    ) -> httpx.Request:
        # No local here names the form or the headers: an error reporter that captures frame locals
        # would record the client secret, refresh token and assertion they carry.
        attempt = request.per_attempt() if request.per_attempt is not None else Attempt()
        return client.build_request(
            request.method,
            request.url,
            headers=self._headers(request, attempt, rid),
            data=_form_of(request, attempt),
            json=request.json_body,
            timeout=self._httpx_timeout(deadline),
        )

    def _deadline(self) -> float:
        return self.clock.monotonic() + self.timeout.total

    def _diagnose(self, sent: httpx.Request, response: HttpResponse) -> HttpResponse:
        """Log a deprecation, and name a header the answer shows an edge removed."""
        if response.header("Deprecation") is not None:
            url = response.url.split("?", 1)[0]
            successor = _successor(response.header_values("Link"))
            warn_once(
                self.log,
                ("deprecation", url),
                "%s is deprecated (Deprecation: %s)%s",
                url,
                response.header("Deprecation"),
                f"; its successor is {successor}" if successor else "",
            )
        dropped = self._dropped(sent, response)
        if dropped is None:
            return response
        warn_once(
            self.log,
            ("dropped", f"{origin_of(response.url)} {dropped}"),
            "a %s header sent to %s did not arrive: an edge or proxy on the path removed it",
            dropped,
            origin_of(response.url),
        )
        return HttpResponse(
            response.status, response.headers, response.body, response.url, response.request_id, dropped
        )

    @staticmethod
    def _dropped(sent: httpx.Request, response: HttpResponse) -> str | None:
        if "dpop" in sent.headers and response.status == 401:
            challenge = (response.header("WWW-Authenticate") or "").lower()
            if "dpop" in challenge or "token_dpop_proof_required" in challenge:
                return "DPoP"
        if "idempotency-key" in sent.headers and 400 <= response.status < 500:
            body = response.json_object() or {}
            if body.get("error") == "EMDT013" or body.get("code") == "EMDT013":
                return "Idempotency-Key"
        sent_trace = trace_id_of(sent.headers.get("traceparent"))
        answered = trace_id_of(response.header("traceparent"))
        if sent_trace and answered and sent_trace != answered:
            return "traceparent"
        return None

    def _answer(self, sent: httpx.Request, raw: httpx.Response, body: bytes, rid: str) -> HttpResponse:
        response = HttpResponse(raw.status_code, tuple(raw.headers.multi_items()), body, str(sent.url), rid)
        if 300 <= raw.status_code < 400:
            raise TransportError(
                f"{origin_of(str(sent.url))} answered a redirect (HTTP {raw.status_code}); the SDK follows none",
                code="redirect_refused",
                status=raw.status_code,
                request_id=rid,
            )
        return self._diagnose(sent, response)

    @staticmethod
    def _read_timeout(request: HttpRequest, rid: str) -> TransportError:
        return TransportError(
            f"{request.what} at {origin_of(request.url)} did not finish its answer within the deadline",
            code="timeout",
            request_id=rid,
        )

    @staticmethod
    def _undecodable(request: HttpRequest, rid: str, exc: httpx.DecodingError) -> TransportError:
        return TransportError(
            f"{request.what} at {origin_of(request.url)} answered a body that could not be decoded: "
            f"{type(exc).__name__}",
            code="response_invalid",
            request_id=rid,
        )

    @staticmethod
    def _too_large(request: HttpRequest, rid: str) -> TransportError:
        return TransportError(
            f"{request.what} at {origin_of(request.url)} answered more than {request.max_bytes} bytes",
            code="response_too_large",
            request_id=rid,
        )

    def _log_retry(self, request: HttpRequest, why: str, delay: float, rid: str) -> None:
        self.log.info(
            "retrying %s at %s in %.2fs after %s (request %s)", request.what, origin_of(request.url), delay, why, rid
        )


class SyncTransport(_Base):
    """The blocking driver, on ``httpx.Client``."""

    def __init__(
        self,
        *,
        client: httpx.Client | None,
        timeout: Timeout,
        policy: RetryPolicy,
        agent: str,
        logger: logging.Logger,
        clock: Clock,
    ) -> None:
        super().__init__(timeout=timeout, policy=policy, agent=agent, logger=logger, clock=clock)
        self._owned = client is None
        self._client = client

    @property
    def client(self) -> httpx.Client:
        """The HTTP client: the caller's, or one this transport owns and makes on first use."""
        if self._client is None:
            self._client = httpx.Client(follow_redirects=False)
        return self._client

    def send(self, request: HttpRequest) -> HttpResponse:
        """Send one logical call, retrying as its class allows, and return the final answer."""
        _check_scheme(request.url)
        rid = request_id()
        deadline = self._deadline()
        attempt = 0
        while True:
            sent = self._build(self.client, request, rid, deadline)
            try:
                raw = self.client.send(sent, stream=True, follow_redirects=False)
            except httpx.TransportError as exc:
                delay = self.policy.after_failure(request, _failure_of(exc), attempt)
                if delay is None or self.clock.monotonic() + delay >= deadline:
                    self.log.warning(
                        "%s at %s failed: %s (request %s)",
                        request.what,
                        origin_of(request.url),
                        type(exc).__name__,
                        rid,
                    )
                    raise _transport_error(exc, request, rid, self.policy.retry_class(request) is Retry.SAFE) from exc
                self._log_retry(request, type(exc).__name__, delay, rid)
                self.clock.sleep(delay)
                attempt += 1
                continue
            try:
                body = self._read(raw, request, rid, deadline)
            finally:
                raw.close()
            response = self._answer(sent, raw, body, rid)
            delay = self.policy.after_response(request, response, attempt, self.clock.time())
            if delay is None:
                return response
            if self.clock.monotonic() + delay >= deadline:
                return response
            self._log_retry(request, f"HTTP {response.status}", delay, rid)
            self.clock.sleep(delay)
            attempt += 1

    def _read(self, raw: httpx.Response, request: HttpRequest, rid: str, deadline: float) -> bytes:
        declared = raw.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > request.max_bytes:
            raise self._too_large(request, rid)
        chunks: list[bytes] = []
        size = 0
        try:
            for chunk in raw.iter_bytes():
                size += len(chunk)
                if size > request.max_bytes:
                    raise self._too_large(request, rid)
                chunks.append(chunk)
                if self.clock.monotonic() >= deadline:
                    raise self._read_timeout(request, rid)
        except httpx.DecodingError as exc:
            raise self._undecodable(request, rid, exc) from exc
        except httpx.TransportError as exc:
            raise _transport_error(exc, request, rid, False) from exc
        return b"".join(chunks)

    def run(self, flow: Generator[Effect, HttpResponse | None, T]) -> T:
        """Drive a protocol flow to its result."""
        try:
            effect = next(flow)
            while True:
                if isinstance(effect, Sleep):
                    self.clock.sleep(effect.seconds)
                    effect = flow.send(None)
                else:
                    effect = flow.send(self.send(effect))
        except StopIteration as done:
            result: T = done.value
            return result
        finally:
            flow.close()

    def close(self) -> None:
        """Close the client when this transport owns it."""
        if self._owned and self._client is not None:
            self._client.close()
            self._client = None

    def after_fork(self) -> None:
        """In a forked child: drop an owned client without closing the parent's sockets."""
        if self._owned:
            self._client = None


class AsyncTransport(_Base):
    """The asyncio driver, on ``httpx.AsyncClient``. Bound to the event loop that first uses it."""

    def __init__(
        self,
        *,
        client: httpx.AsyncClient | None,
        timeout: Timeout,
        policy: RetryPolicy,
        agent: str,
        logger: logging.Logger,
        clock: Clock,
    ) -> None:
        super().__init__(timeout=timeout, policy=policy, agent=agent, logger=logger, clock=clock)
        self._owned = client is None
        self._client = client
        self._loop: asyncio.AbstractEventLoop | None = None

    def check_loop(self) -> None:
        """Bind to the running loop on first use; refuse any other loop afterwards."""
        loop = asyncio.get_running_loop()
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise ConfigurationError(
                "this AsyncAuthFI belongs to another event loop; create one AsyncAuthFI per event loop",
                code="wrong_event_loop",
            )

    @property
    def client(self) -> httpx.AsyncClient:
        """The HTTP client: the caller's, or one this transport owns and makes on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(follow_redirects=False)
        return self._client

    async def send(self, request: HttpRequest) -> HttpResponse:
        """Send one logical call, retrying as its class allows, and return the final answer."""
        self.check_loop()
        _check_scheme(request.url)
        rid = request_id()
        deadline = self._deadline()
        attempt = 0
        while True:
            sent = self._build(self.client, request, rid, deadline)
            try:
                raw = await self.client.send(sent, stream=True, follow_redirects=False)
            except httpx.TransportError as exc:
                delay = self.policy.after_failure(request, _failure_of(exc), attempt)
                if delay is None or self.clock.monotonic() + delay >= deadline:
                    self.log.warning(
                        "%s at %s failed: %s (request %s)",
                        request.what,
                        origin_of(request.url),
                        type(exc).__name__,
                        rid,
                    )
                    raise _transport_error(exc, request, rid, self.policy.retry_class(request) is Retry.SAFE) from exc
                self._log_retry(request, type(exc).__name__, delay, rid)
                await self.clock.asleep(delay)
                attempt += 1
                continue
            try:
                body = await self._read(raw, request, rid, deadline)
            finally:
                await raw.aclose()
            response = self._answer(sent, raw, body, rid)
            delay = self.policy.after_response(request, response, attempt, self.clock.time())
            if delay is None or self.clock.monotonic() + delay >= deadline:
                return response
            self._log_retry(request, f"HTTP {response.status}", delay, rid)
            await self.clock.asleep(delay)
            attempt += 1

    async def _read(self, raw: httpx.Response, request: HttpRequest, rid: str, deadline: float) -> bytes:
        declared = raw.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > request.max_bytes:
            raise self._too_large(request, rid)
        chunks: list[bytes] = []
        size = 0
        try:
            async with asyncio.timeout(max(0.0, deadline - self.clock.monotonic())):
                async for chunk in raw.aiter_bytes():
                    size += len(chunk)
                    if size > request.max_bytes:
                        raise self._too_large(request, rid)
                    chunks.append(chunk)
                    if self.clock.monotonic() >= deadline:
                        raise self._read_timeout(request, rid)
        except TimeoutError as exc:
            raise self._read_timeout(request, rid) from exc
        except httpx.DecodingError as exc:
            raise self._undecodable(request, rid, exc) from exc
        except httpx.TransportError as exc:
            raise _transport_error(exc, request, rid, False) from exc
        return b"".join(chunks)

    async def run(self, flow: Generator[Effect, HttpResponse | None, T]) -> T:
        """Drive a protocol flow to its result."""
        try:
            effect = next(flow)
            while True:
                if isinstance(effect, Sleep):
                    await self.clock.asleep(effect.seconds)
                    effect = flow.send(None)
                else:
                    effect = flow.send(await self.send(effect))
        except StopIteration as done:
            result: T = done.value
            return result
        finally:
            flow.close()

    async def aclose(self) -> None:
        """Close the client when this transport owns it."""
        if self._owned and self._client is not None:
            await self._client.aclose()
            self._client = None

    def after_fork(self) -> None:
        """In a forked child: drop an owned client, and the loop binding."""
        if self._owned:
            self._client = None
        self._loop = None
