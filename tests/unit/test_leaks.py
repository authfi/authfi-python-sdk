"""The secret-leak suite (design §1.7, §6.1).

An error reporter that captures frame locals (Sentry's default, Rich's ``show_locals``) must never
record a credential. Each test raises inside a path that handles one, and searches what both
reporters capture for a marker that stands for the credential, in the forms it can take.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from urllib.parse import quote_plus

import httpx
import pytest

from authfi import Secret
from authfi._protocol.http import HttpRequest
from authfi._protocol.oauth import (
    ClientAuth,
    authorization_code_request,
    client_credentials_request,
    refresh_request,
    revocation_request,
)
from authfi._protocol.retry import RetryPolicy
from authfi._transport import AsyncTransport, SyncTransport, Timeout, user_agent
from tests.leaks import captured, sdk_frames
from tests.support import FakeClock

TOKEN_URL = "https://acme.authfi.io/oauth/token"
MARKER = "LEAKMARKER4f9c1d"


def forms_of(marker: str) -> list[str]:
    """The ways a credential can appear in a frame: as is, form-encoded, and base64 in a Basic header."""
    return [
        marker,
        quote_plus(marker),
        base64.b64encode(f"app:{marker}".encode()).decode().rstrip("="),
        base64.b64encode(f"app:{quote_plus(marker)}".encode()).decode().rstrip("="),
    ]


def assert_clean(exc: BaseException, *, sdk_only: bool = False) -> None:
    text = sdk_frames(exc) if sdk_only else captured(exc)
    assert text.strip()
    for form in forms_of(MARKER):
        assert form not in text, f"{form!r} was captured from a frame"


class Boom(Exception):
    """Raised by the fake server or the signing callback, as any error in the path would be."""


def sync_transport(handler: Callable[[httpx.Request], httpx.Response]) -> SyncTransport:
    return SyncTransport(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        timeout=Timeout(),
        policy=RetryPolicy(retries=0),
        agent=user_agent(),
        logger=logging.getLogger("authfi"),
        clock=FakeClock(),
    )


def explode(_request: httpx.Request) -> httpx.Response:
    raise Boom("the server side blew up")


def requests() -> list[tuple[str, HttpRequest]]:
    secret = Secret(MARKER)
    post = ClientAuth.secret_post("app", secret)
    basic = ClientAuth.secret_basic("app", secret)
    jwt = ClientAuth.private_key_jwt("app", lambda _aud: MARKER)
    return [
        ("client_credentials, secret_post", client_credentials_request(TOKEN_URL, post)),
        ("client_credentials, secret_basic", client_credentials_request(TOKEN_URL, basic)),
        ("client_credentials, private_key_jwt", client_credentials_request(TOKEN_URL, jwt)),
        (
            "authorization_code, code_verifier",
            authorization_code_request(
                TOKEN_URL, ClientAuth.none("app"), code="c", redirect_uri="https://app/cb", code_verifier=secret
            ),
        ),
        ("refresh_token", refresh_request(TOKEN_URL, ClientAuth.none("app"), refresh_token=secret)),
        ("revocation", revocation_request(TOKEN_URL, ClientAuth.none("app"), token=secret)),
    ]


@pytest.mark.parametrize(("name", "request_"), requests(), ids=[n for n, _ in requests()])
def test_a_failure_in_the_server_call_captures_no_credential(name: str, request_: HttpRequest) -> None:
    with pytest.raises(Boom) as caught:
        sync_transport(explode).send(request_)
    assert_clean(caught.value)


@pytest.mark.parametrize(("name", "request_"), requests(), ids=[n for n, _ in requests()])
async def test_a_failure_in_the_async_server_call_captures_no_credential(name: str, request_: HttpRequest) -> None:
    transport = AsyncTransport(
        client=httpx.AsyncClient(transport=httpx.MockTransport(explode)),
        timeout=Timeout(),
        policy=RetryPolicy(retries=0),
        agent=user_agent(),
        logger=logging.getLogger("authfi"),
        clock=FakeClock(),
    )
    with pytest.raises(Boom) as caught:
        await transport.send(request_)
    assert_clean(caught.value)


@pytest.mark.parametrize(("name", "request_"), requests(), ids=[n for n, _ in requests()])
def test_a_failure_while_the_request_is_built_captures_no_credential(
    name: str, request_: HttpRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(*_args: object, **_kwargs: object) -> httpx.Request:
        raise Boom("the HTTP library refused to build the request")

    transport = sync_transport(explode)
    monkeypatch.setattr(transport.client, "build_request", refuse)
    with pytest.raises(Boom) as caught:
        transport.send(request_)
    assert_clean(caught.value, sdk_only=True)


def test_a_failure_in_the_callers_signing_function_captures_no_credential() -> None:
    secret = Secret(MARKER)

    def sign(_audience: str) -> str:
        raise Boom("the key store is down")

    auth = ClientAuth.private_key_jwt("app", sign)
    request = refresh_request(TOKEN_URL, auth, refresh_token=secret)
    with pytest.raises(Boom) as caught:
        sync_transport(explode).send(request)
    assert_clean(caught.value)
