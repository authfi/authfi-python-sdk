from __future__ import annotations

import json

import httpx
import pytest

from authfi import APIError, AsyncAuthFI, AuthFI, CheckResult, ConfigurationError, Secret
from tests.support import FakeClock, Platform, json_response


def answer(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    allowed = body["object"] == "doc:1"
    return json_response({"data": {"allowed": allowed, "requires_approval": body["object"] == "doc:2", "reason": ""}})


def test_a_live_check_with_a_held_token(af: AuthFI, platform: Platform) -> None:
    platform.route("POST", "/apis/regional/v1/check", answer)
    result = af.check(Secret("tok"), subject="user:u1", relation="viewer", object="doc:1", context={"ip": "1.2.3.4"})
    assert result == CheckResult(allowed=True, approval_required=False, reason=None)
    sent = platform.calls("POST", "/apis/regional/v1/check")[0]
    assert sent.headers["authorization"] == "Bearer tok"
    assert json.loads(sent.content) == {
        "subject": "user:u1",
        "relation": "viewer",
        "object": "doc:1",
        "context": {"ip": "1.2.3.4"},
    }
    held = af.check(Secret("tok"), subject="user:u1", relation="editor", object="doc:2")
    assert held.approval_required
    assert not held.allowed


def test_a_check_with_a_credential_source(af: AuthFI, platform: Platform) -> None:
    platform.route("POST", "/apis/regional/v1/check", answer)
    platform.route("POST", "/oauth/token", json_response({"access_token": "m2m", "expires_in": 900}))
    m2m = af.client_credentials("app", Secret("s"))
    assert af.check(m2m, subject="app:x", relation="r", object="doc:1").allowed
    assert platform.calls("POST", "/apis/regional/v1/check")[0].headers["authorization"] == "Bearer m2m"


def test_a_credential_for_another_api_is_never_sent_to_authfi(af: AuthFI, platform: Platform) -> None:
    platform.route("POST", "/oauth/token", json_response({"access_token": "m2m", "expires_in": 900}))
    m2m = af.client_credentials("app", Secret("s"), resource="api_orders", send_to=["https://api.example.com"])
    with pytest.raises(ConfigurationError, match="refusing to send"):
        af.check(m2m, subject="app:x", relation="r", object="doc:1")


def test_a_plain_string_and_empty_arguments_are_refused(af: AuthFI) -> None:
    with pytest.raises(ConfigurationError):
        af.check("raw-token", subject="u", relation="r", object="o")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        af.check(Secret("t"), subject="", relation="r", object="o")


def test_a_refusal_is_an_api_error(af: AuthFI, platform: Platform) -> None:
    platform.route("POST", "/apis/regional/v1/check", json_response({"error": "unauthorized"}, status=401))
    with pytest.raises(APIError) as caught:
        af.check(Secret("t"), subject="u", relation="r", object="o")
    assert caught.value.code == "unauthorized"
    platform.route("POST", "/apis/regional/v1/check", json_response({"nodata": 1}))
    with pytest.raises(APIError):
        af.check(Secret("t"), subject="u", relation="r", object="o")


async def test_the_async_check(aaf: AsyncAuthFI, platform: Platform) -> None:
    platform.route("POST", "/apis/regional/v1/check", answer)
    assert (await aaf.check(Secret("t"), subject="u", relation="r", object="doc:1")).allowed
    with pytest.raises(ConfigurationError):
        await aaf.check("raw", subject="u", relation="r", object="o")  # type: ignore[arg-type]


def test_a_cold_check_on_a_custom_domain_asks_the_machine_api_host(clock: FakeClock) -> None:
    platform = Platform(issuer="https://login.acme.com")
    platform.discovery["backchannel_authentication_endpoint"] = (
        "https://acme.authfi.io/apis/regional/v1/ai/auth/backchannel-authorize"
    )
    platform.route("POST", "/apis/regional/v1/check", answer)
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        assert af.check(Secret("tok"), subject="user:u1", relation="viewer", object="doc:1").allowed
    assert [r.url.host for r in platform.calls("POST", "/apis/regional/v1/check")] == ["acme.authfi.io"]


async def test_a_cold_async_check_on_a_custom_domain_asks_the_machine_api_host(clock: FakeClock) -> None:
    platform = Platform(issuer="https://login.acme.com")
    platform.discovery["backchannel_authentication_endpoint"] = (
        "https://acme.authfi.io/apis/regional/v1/ai/auth/backchannel-authorize"
    )
    platform.route("POST", "/apis/regional/v1/check", answer)
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as af:
        assert (await af.check(Secret("tok"), subject="user:u1", relation="viewer", object="doc:1")).allowed
    assert [r.url.host for r in platform.calls("POST", "/apis/regional/v1/check")] == ["acme.authfi.io"]
