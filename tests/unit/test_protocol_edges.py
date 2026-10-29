"""The protocol layer's less travelled paths, each with the outcome it must have."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import httpx
import pytest

from authfi import AuthFI, ConfigurationError, DiscoveryError, InvalidToken, Secret, TransportError
from authfi._protocol import discovery, http, ids, jwks, oauth, urls
from authfi._protocol import verify as rules
from authfi._protocol.errors import with_answer
from authfi._protocol.jwks import PublicKey
from authfi.errors import APIError
from tests.support import TENANT, FakeClock, Platform, b64url, json_response


def test_ttl_without_max_age_and_the_rfc_8414_grant_default() -> None:
    assert discovery.ttl_from_headers("public") == discovery.DEFAULT_TTL
    md = discovery.ServerMetadata(issuer="https://x", token_endpoint="https://x/t", jwks_uri="https://x/j")
    assert md.supports_grant("authorization_code")
    assert not md.supports_grant("client_credentials")


def test_a_discovery_document_that_is_not_json(platform: Platform, clock: FakeClock) -> None:
    platform.route("GET", "/.well-known/openid-configuration", httpx.Response(200, content=b"<html>"))
    with pytest.raises(DiscoveryError, match="not JSON"):
        AuthFI(platform.issuer, http_client=platform.client(), clock=clock).metadata()


def test_a_key_set_that_fails_or_is_not_json(platform: Platform, clock: FakeClock) -> None:
    platform.route("GET", "/.well-known/jwks.json", httpx.Response(500))
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock, retries=0)
    with pytest.raises(DiscoveryError, match="HTTP 500"):
        af._key("x")
    platform.route("GET", "/.well-known/jwks.json", httpx.Response(200, content=b"nope"))
    with pytest.raises(DiscoveryError, match="not JSON"):
        AuthFI(platform.issuer, http_client=platform.client(), clock=clock)._key("x")


def test_a_key_that_does_not_load_is_left_out() -> None:
    keys = jwks.parse_jwks({"keys": [{"kty": "RSA", "kid": "bad", "n": "!!", "e": "AQAB"}, "not-a-key"]}, "t")
    assert len(keys) == 0


def test_an_answer_keeps_its_status_on_the_error() -> None:
    response = http.HttpResponse(418, (), b"", "https://x", "afr-1")
    error = with_answer(APIError("x"), response)
    assert error.status == 418
    assert repr(response) == "HttpResponse(418 https://x)"


def test_a_flow_driver_that_sends_no_response_is_a_bug() -> None:
    flow = http.call(http.HttpRequest("GET", "https://x", what="x"))
    next(flow)
    with pytest.raises(RuntimeError):
        flow.send(None)


def test_pause_and_header_merging() -> None:
    assert list(http.pause(3)) == [http.Sleep(3)]
    assert http.merge_headers({"Accept": "a"}, {"accept": "b"}) == {"accept": "b"}


def test_opentelemetry_context_is_propagated_when_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def inject(carrier: dict[str, str]) -> None:
        carrier["traceparent"] = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"
        carrier["tracestate"] = "vendor=1"

    otel = types.ModuleType("opentelemetry")
    propagate = types.ModuleType("opentelemetry.propagate")
    propagate.inject = inject  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", otel)
    monkeypatch.setitem(sys.modules, "opentelemetry.propagate", propagate)
    headers: dict[str, str] = {}
    ids.inject_traceparent(headers)
    assert headers["traceparent"].endswith("-01")
    assert headers["tracestate"] == "vendor=1"


def test_opentelemetry_with_no_active_context_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    propagate = types.ModuleType("opentelemetry.propagate")
    propagate.inject = lambda carrier: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", types.ModuleType("opentelemetry"))
    monkeypatch.setitem(sys.modules, "opentelemetry.propagate", propagate)
    headers: dict[str, str] = {}
    ids.inject_traceparent(headers)
    assert ids.trace_id_of(headers["traceparent"]) is not None


def test_client_auth_built_by_hand_without_its_credential() -> None:
    with pytest.raises(ConfigurationError, match="signs the assertion"):
        oauth.ClientAuth("private_key_jwt", "agt_1").attempt("https://x")
    with pytest.raises(ConfigurationError, match="client secret"):
        oauth.ClientAuth("client_secret_post", "app").attempt("https://x")


def test_a_refresh_can_narrow_the_scope() -> None:
    request = oauth.refresh_request(
        "https://x/t", oauth.ClientAuth.none("app"), refresh_token=Secret("r"), scope=["openid"]
    )
    assert request.form is not None
    assert request.form["scope"] == "openid"


def test_url_edges() -> None:
    with pytest.raises(ConfigurationError, match="no host"):
        urls.validate_issuer("https://")
    assert not urls.is_under("http://[::1", "https://x")


def segment(value: object) -> str:
    return b64url(json.dumps(value).encode())


def test_unverified_token_repr_shows_no_token() -> None:
    raw = f"{segment({'alg': 'RS256', 'typ': 'at+jwt', 'kid': 'k'})}.{segment({})}.sig"
    token = rules.inspect_token(raw)
    assert repr(token) == "Unverified(kid='k', alg='RS256')"


@pytest.mark.parametrize("crit", [[], ["exp"], [1], "exp"])
def test_crit_is_refused_unless_understood(crit: object) -> None:
    raw = f"{segment({'alg': 'RS256', 'typ': 'at+jwt', 'kid': 'k', 'crit': crit})}.{segment({})}.sig"
    with pytest.raises(InvalidToken) as caught:
        rules.inspect_token(raw)
    assert caught.value.code == "token_claims_invalid"


@pytest.fixture
def rsa_key() -> PublicKey:
    from tests.support import shared_keys

    keys = jwks.parse_jwks({"keys": [shared_keys().jwk("RS256")]}, "t")
    key = keys.get("rsa-1")
    assert key is not None
    return key


def test_signature_layer_refusals(rsa_key: PublicKey) -> None:
    from tests.support import shared_keys

    good = shared_keys().sign({"sub": "x"})
    token = rules.inspect_token(good)
    assert rules.verify_signature(token, rsa_key)["sub"] == "x"
    header, payload, _ = good.split(".")
    bad_b64 = rules.inspect_token(f"{header}.{payload}.@@@")
    with pytest.raises(InvalidToken) as caught:
        rules.verify_signature(bad_b64, rsa_key)
    assert caught.value.code in ("token_malformed", "token_signature_invalid")


@pytest.mark.parametrize("payload", [b"not json", b"[1,2]"])
def test_a_signed_payload_that_is_not_a_json_object(payload: bytes, rsa_key: PublicKey) -> None:
    import jwt as pyjwt

    from tests.support import shared_keys

    raw = pyjwt.api_jws.PyJWS().encode(
        payload, shared_keys().rsa_key, algorithm="RS256", headers={"kid": "rsa-1", "typ": "at+jwt"}
    )
    with pytest.raises(InvalidToken) as caught:
        rules.verify_signature(rules.inspect_token(raw), rsa_key)
    assert caught.value.code == "token_malformed"


def test_an_act_that_is_not_an_actor_claim() -> None:
    config = rules.VerifyConfig("https://x", frozenset({"a"}), 0)
    claims: dict[str, Any] = {"iss": "https://x", "aud": "a", "exp": 2e9, "sub": "s", "tenant_id": TENANT, "act": "agt"}
    with pytest.raises(InvalidToken) as caught:
        rules.check_claims(claims, config, 1e9)
    assert caught.value.code == "token_claims_invalid"


def test_a_token_naming_no_issuer_is_not_routed_by_a_multi_issuer_verifier(af: AuthFI, platform: Platform) -> None:
    from authfi import MultiIssuerVerifier

    verifier = MultiIssuerVerifier({platform.issuer: (af.verifier(audience=["a"]), TENANT)})
    assert verifier.issuers == (platform.issuer,)
    assert verifier.logger is af.logger
    raw = platform.keys.sign({"sub": "x"})
    with pytest.raises(InvalidToken) as caught:
        verifier.verify(raw)
    assert caught.value.code == "token_issuer_not_accepted"


async def test_the_async_multi_issuer_verifier_refuses_strangers(platform: Platform, clock: FakeClock) -> None:
    from authfi import AsyncAuthFI, AsyncMultiIssuerVerifier

    aaf = AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock)
    verifier = AsyncMultiIssuerVerifier({platform.issuer: (aaf.verifier(audience=["a"]), TENANT)})
    assert verifier.issuers == (platform.issuer,)
    assert verifier.logger is aaf.logger
    stranger = Platform("https://stranger.authfi.io").access_token()
    with pytest.raises(InvalidToken) as caught:
        await verifier.verify(stranger)
    assert caught.value.code == "token_issuer_not_accepted"


def test_a_single_role_string_claim_is_read(af: AuthFI, platform: Platform) -> None:
    token = af.verifier(audience=["api_orders"]).verify(platform.access_token(roles="admin"))
    assert token.roles == frozenset({"admin"})
    with pytest.raises(ConfigurationError):
        token.require_role("")


def test_a_transport_error_names_no_secret(platform: Platform, clock: FakeClock) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    platform.route("POST", "/oauth/token", refuse)
    platform.route("GET", "/.well-known/openid-configuration", json_response(platform.discovery))
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock, retries=0)
    m2m = af.client_credentials("app", Secret("MARKER-SECRET"), resource="api", send_to=["https://api.example.com"])
    with pytest.raises(TransportError) as caught:
        m2m.token()
    assert "MARKER-SECRET" not in str(caught.value)
    assert caught.value.retryable
