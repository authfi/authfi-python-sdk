from __future__ import annotations

import asyncio
import base64
import logging
import threading

import httpx
import pytest

from authfi import (
    AsyncAuthFI,
    AuthFI,
    ConfigurationError,
    NotSupportedError,
    OAuthError,
    Secret,
)
from tests.support import FakeClock, Platform, form_of, json_response

API = "https://api.example.com"


class Mint:
    """The token endpoint: counts mints and answers a fresh token each time."""

    def __init__(self, expires_in: int | None = 900) -> None:
        self.count = 0
        self.expires_in = expires_in
        self.gate: threading.Event | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.gate is not None:
            self.gate.wait(5)
        self.count += 1
        body: dict[str, object] = {"access_token": f"token-{self.count}", "token_type": "Bearer"}
        if self.expires_in is not None:
            body["expires_in"] = self.expires_in
        return json_response(body)


@pytest.fixture
def mint(platform: Platform) -> Mint:
    m = Mint()
    platform.route("POST", "/oauth/token", m)
    return m


def test_a_token_for_an_api_with_basic_authentication(af: AuthFI, platform: Platform, mint: Mint) -> None:
    m2m = af.client_credentials("app_reports", Secret("s3cret"), resource="api_orders", send_to=[API], scope=["r"])
    assert m2m.token() == "token-1"
    request = platform.calls("POST", "/oauth/token")[0]
    assert form_of(request) == {"grant_type": "client_credentials", "resource": "api_orders", "scope": "r"}
    assert base64.b64decode(request.headers["Authorization"].split()[1]) == b"app_reports:s3cret"
    assert "s3cret" not in repr(m2m)


def test_client_secret_post(af: AuthFI, platform: Platform, mint: Mint) -> None:
    m2m = af.client_credentials(
        "app_reports", Secret("s3cret"), resource="api_orders", send_to=[API], auth_method="client_secret_post"
    )
    m2m.token()
    form = form_of(platform.calls("POST", "/oauth/token")[0])
    assert form["client_id"] == "app_reports"
    assert form["client_secret"] == "s3cret"


def test_the_token_is_cached_and_renewed_at_two_thirds(af: AuthFI, mint: Mint, clock: FakeClock) -> None:
    m2m = af.client_credentials("app", Secret("s"), resource="api_orders", send_to=[API])
    assert m2m.token() == "token-1"
    clock.advance(500)
    assert m2m.token() == "token-1"
    clock.advance(101)
    assert m2m.token() == "token-2"
    assert m2m.token(force=True) == "token-3"


def test_concurrent_callers_share_one_mint(platform: Platform, clock: FakeClock) -> None:
    mint = Mint()
    mint.gate = threading.Event()
    platform.route("POST", "/oauth/token", mint)
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    m2m = af.client_credentials("app", Secret("s"), resource="api_orders", send_to=[API])
    af.metadata()
    tokens: list[str] = []
    threads = [threading.Thread(target=lambda: tokens.append(m2m.token())) for _ in range(8)]
    for t in threads:
        t.start()
    mint.gate.set()
    for t in threads:
        t.join()
    assert tokens == ["token-1"] * 8
    assert mint.count == 1


def test_a_customer_api_token_goes_only_to_its_api(af: AuthFI, mint: Mint) -> None:
    m2m = af.client_credentials("app", Secret("s"), resource="api_orders", send_to=[API])
    assert m2m.send_to == (API,)
    assert m2m.headers("GET", API + "/orders")["Authorization"] == "Bearer token-1"
    with pytest.raises(ConfigurationError):
        m2m.headers("GET", "https://acme.authfi.io/apis/regional/v1/users")
    with pytest.raises(ConfigurationError, match="send_to is required"):
        af.client_credentials("app", Secret("s"), resource="api_orders")


def test_a_token_for_authfi_itself_goes_to_the_issuer(af: AuthFI, mint: Mint, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="authfi"):
        m2m = af.client_credentials("app", Secret("s"))
        af.client_credentials("app", Secret("s"))
    assert m2m.send_to == ("https://acme.authfi.io",)
    assert len([r for r in caplog.records if "no resource" in r.getMessage()]) == 1
    console = af.client_credentials("app", Secret("s"), resource="authfi-console")
    assert console.send_to == ("https://acme.authfi.io",)


def test_private_key_jwt_is_not_offered_to_a_machine_client(af: AuthFI) -> None:
    with pytest.raises(NotSupportedError) as caught:
        af.client_credentials("app", Secret("s"), auth_method="private_key_jwt")  # type: ignore[arg-type]
    assert caught.value.feature == "private_key_jwt"
    with pytest.raises(ConfigurationError):
        af.client_credentials("app", Secret("s"), auth_method="basic")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        af.client_credentials("app", "plain", resource="api_orders", send_to=[API])  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        af.client_credentials("app", Secret("s"), scope="a b")


def test_a_project_without_the_grant_says_so(platform: Platform, af: AuthFI) -> None:
    platform.discovery["grant_types_supported"] = ["authorization_code"]
    with pytest.raises(NotSupportedError):
        af.client_credentials("app", Secret("s"), resource="api_orders", send_to=[API]).token()


def test_a_refusal_is_raised_and_not_retried(af: AuthFI, platform: Platform) -> None:
    platform.route("POST", "/oauth/token", json_response({"error": "invalid_client"}, status=401))
    with pytest.raises(OAuthError) as caught:
        af.client_credentials("app", Secret("s"), resource="api_orders", send_to=[API]).token()
    assert caught.value.code == "invalid_client"
    assert len(platform.calls("POST", "/oauth/token")) == 1


def test_a_token_with_no_lifetime_is_kept_briefly(platform: Platform, af: AuthFI, clock: FakeClock) -> None:
    platform.route("POST", "/oauth/token", Mint(expires_in=None))
    m2m = af.client_credentials("app", Secret("s"), resource="api_orders", send_to=[API])
    assert m2m.token() == "token-1"
    clock.advance(61)
    assert m2m.token() == "token-2"


def test_the_httpx_auth_carries_the_token(af: AuthFI, mint: Mint) -> None:
    seen: list[str] = []

    def api(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["authorization"])
        return httpx.Response(200)

    m2m = af.client_credentials("app", Secret("s"), resource="api_orders", send_to=[API])
    with httpx.Client(transport=httpx.MockTransport(api), auth=m2m.httpx_auth()) as client:
        client.get(API + "/orders")
    assert seen == ["Bearer token-1"]


async def test_the_async_client(aaf: AsyncAuthFI, platform: Platform, clock: FakeClock) -> None:
    mint = Mint()
    platform.route("POST", "/oauth/token", mint)
    m2m = aaf.client_credentials("app", Secret("s"), resource="api_orders", send_to=[API])
    tokens = await asyncio.gather(*(m2m.token() for _ in range(5)))
    assert set(tokens) == {"token-1"}
    assert mint.count == 1
    clock.advance(601)
    assert await m2m.token() == "token-2"
    assert (await m2m.headers("GET", API))["Authorization"] == "Bearer token-2"
    assert m2m.client_id == "app"
    assert m2m.resource == "api_orders"
    assert "AsyncClientCredentials" in repr(m2m)
