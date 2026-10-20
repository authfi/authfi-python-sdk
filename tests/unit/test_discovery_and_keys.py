from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from authfi import AsyncAuthFI, AuthFI, DiscoveryError
from authfi._protocol.discovery import parse_metadata, ttl_from_headers
from authfi._protocol.jwks import parse_jwks
from tests.support import FakeClock, Platform, fixture_json, json_response, shared_keys


def test_the_live_platform_beta_document_parses() -> None:
    doc = fixture_json("platform-beta/openid-configuration.json")
    md = parse_metadata("https://platform-beta.authfi.io", doc)
    assert md.token_endpoint == "https://platform-beta.authfi.io/oauth/token"
    assert md.supports_grant("urn:openid:params:grant-type:ciba")
    assert md.id_token_signing_alg_values_supported == ("RS256",)
    assert md.backchannel_authentication_endpoint is not None
    with pytest.raises(TypeError):
        md.raw["issuer"] = "x"  # type: ignore[index]


def test_a_document_naming_another_issuer_is_refused() -> None:
    doc = fixture_json("platform-beta/openid-configuration.json")
    with pytest.raises(DiscoveryError) as caught:
        parse_metadata("https://acme.authfi.io", doc)
    assert caught.value.code == "issuer_mismatch"


def test_an_issuer_with_a_trailing_slash_in_the_document_is_another_issuer() -> None:
    doc = dict(fixture_json("platform-beta/openid-configuration.json"), issuer="https://platform-beta.authfi.io/")
    with pytest.raises(DiscoveryError):
        parse_metadata("https://platform-beta.authfi.io", doc)


def test_an_http_endpoint_is_refused() -> None:
    doc = dict(fixture_json("platform-beta/openid-configuration.json"), token_endpoint="http://evil.example/token")
    with pytest.raises(DiscoveryError, match="not an https URL"):
        parse_metadata("https://platform-beta.authfi.io", doc)


def test_missing_required_fields_are_named() -> None:
    doc = fixture_json("platform-beta/openid-configuration.json")
    del doc["jwks_uri"]
    with pytest.raises(DiscoveryError, match="jwks_uri"):
        parse_metadata("https://platform-beta.authfi.io", doc)
    with pytest.raises(DiscoveryError):
        parse_metadata("https://x", [])


@pytest.mark.parametrize(
    ("header", "ttl"),
    [
        (None, 300.0),
        ("public, max-age=3600", 3600.0),
        ("max-age=5", 60.0),
        ("max-age=999999", 86400.0),
        ("no-store", 60.0),
    ],
)
def test_cache_lifetimes_follow_cache_control_within_bounds(header: str | None, ttl: float) -> None:
    assert ttl_from_headers(header) == ttl


def test_metadata_is_fetched_once_and_kept_for_its_lifetime(af: AuthFI, platform: Platform, clock: FakeClock) -> None:
    af.metadata()
    af.metadata()
    assert len(platform.calls("GET", "/.well-known/openid-configuration")) == 1
    clock.advance(301)
    af.metadata()
    assert len(platform.calls("GET", "/.well-known/openid-configuration")) == 2
    af.metadata(refresh=True)
    assert len(platform.calls("GET", "/.well-known/openid-configuration")) == 3


def test_the_oauth_authorization_server_document_is_the_fallback(platform: Platform, clock: FakeClock) -> None:
    doc = platform.discovery
    platform.route("GET", "/.well-known/openid-configuration", httpx.Response(404))
    platform.route("GET", "/.well-known/oauth-authorization-server", json_response(doc))
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    assert af.metadata().issuer == platform.issuer


def test_a_failure_is_never_cached(platform: Platform, clock: FakeClock) -> None:
    calls = {"n": 0}

    def flaky(_r: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(500) if calls["n"] == 1 else json_response(platform.discovery)

    platform.route("GET", "/.well-known/openid-configuration", flaky)
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock, retries=0)
    with pytest.raises(DiscoveryError):
        af.metadata()
    assert af.metadata().issuer == platform.issuer


def test_concurrent_callers_share_one_fetch(platform: Platform, clock: FakeClock) -> None:
    gate = threading.Event()
    inner = platform.routes[("GET", "/.well-known/openid-configuration")]

    def slow(request: httpx.Request) -> httpx.Response:
        gate.wait(5)
        return inner(request)

    platform.route("GET", "/.well-known/openid-configuration", slow)
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    results: list[str] = []
    threads = [threading.Thread(target=lambda: results.append(af.metadata().issuer)) for _ in range(8)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join()
    assert results == [platform.issuer] * 8
    assert len(platform.calls("GET", "/.well-known/openid-configuration")) == 1


async def test_concurrent_tasks_share_one_fetch(aaf: AsyncAuthFI, platform: Platform) -> None:
    results = await asyncio.gather(*(aaf.metadata() for _ in range(8)))
    assert {r.issuer for r in results} == {platform.issuer}
    assert len(platform.calls("GET", "/.well-known/openid-configuration")) == 1


async def test_a_cancelled_waiter_does_not_cancel_the_shared_fetch(aaf: AsyncAuthFI, platform: Platform) -> None:
    first = asyncio.ensure_future(aaf.metadata())
    second = asyncio.ensure_future(aaf.metadata())
    await asyncio.sleep(0)
    first.cancel()
    assert (await second).issuer == platform.issuer


def test_the_api_origin_comes_from_discovery_for_a_custom_domain(clock: FakeClock) -> None:
    platform = Platform(issuer="https://login.acme.com")
    platform.discovery["backchannel_authentication_endpoint"] = (
        "https://acme.authfi.io/apis/regional/v1/ai/auth/backchannel-authorize"
    )
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    assert af.api_origin == "https://login.acme.com"
    af.metadata()
    assert af.api_origin == "https://acme.authfi.io"
    assert af.authfi_origins == ("https://login.acme.com", "https://acme.authfi.io")
    pinned = AuthFI(platform.issuer, api_origin="https://api.acme.com", http_client=platform.client(), clock=clock)
    pinned.metadata()
    assert pinned.api_origin == "https://api.acme.com"


def jwks_with(*extra: dict[str, Any]) -> dict[str, Any]:
    return {"keys": [*shared_keys().jwks()["keys"], *extra]}


def test_unusable_keys_are_left_out() -> None:
    good = shared_keys().jwk("RS256")
    small = rsa.generate_private_key(public_exponent=65537, key_size=1024)  # noqa: S505 - the weak key under test
    small_jwk = json.loads(RSAAlgorithm.to_jwk(small.public_key()))
    keys = parse_jwks(
        jwks_with(
            dict(good, kid="burned", d="AAAA"),
            dict(good, kid="enc", use="enc"),
            dict(good, kid="mislabelled", alg="PS256"),
            dict(small_jwk, kid="small"),
            {"kty": "oct", "kid": "sym", "k": "AAAA"},
            {"kty": "EC", "crv": "P-384", "kid": "p384", "x": "AA", "y": "AA"},
            dict(good, kid=""),
        ),
        "test",
    )
    assert set(keys.keys) == {"rsa-1", "ed-1", "ec-1"}
    assert keys.get("rsa-1") is not None and keys.get("rsa-1").alg == "RS256"  # type: ignore[union-attr]
    assert len(keys) == 3


def test_the_live_key_set_has_an_okp_and_an_rsa_key() -> None:
    keys = parse_jwks(fixture_json("platform-beta/jwks.json"), "platform-beta")
    assert {k.alg for k in keys.keys.values()} == {"EdDSA", "RS256"}


def test_a_document_that_is_not_a_key_set_is_refused() -> None:
    with pytest.raises(DiscoveryError):
        parse_jwks({"nokeys": []}, "x")


def test_an_unknown_kid_refetches_once_per_30_seconds(af: AuthFI, platform: Platform, clock: FakeClock) -> None:
    assert af._key("rsa-1") is not None
    assert len(platform.calls("GET", "/.well-known/jwks.json")) == 1
    clock.advance(31)
    assert af._key("nope") is None
    assert len(platform.calls("GET", "/.well-known/jwks.json")) == 2
    for _ in range(20):
        assert af._key(f"invented-{_}") is None
    assert len(platform.calls("GET", "/.well-known/jwks.json")) == 2
    clock.advance(31)
    af._key("still-nope")
    assert len(platform.calls("GET", "/.well-known/jwks.json")) == 3


def test_a_rotated_key_is_found_by_the_refetch(af: AuthFI, platform: Platform, clock: FakeClock) -> None:
    af._key("rsa-1")
    clock.advance(31)
    new = dict(shared_keys().jwk("EdDSA"), kid="ed-2")
    platform.route("GET", "/.well-known/jwks.json", json_response(jwks_with(new)))
    assert af._key("ed-2") is not None


async def test_async_kid_refetch(aaf: AsyncAuthFI, platform: Platform, clock: FakeClock) -> None:
    assert await aaf._key("ed-1") is not None
    clock.advance(31)
    assert await aaf._key("nope") is None
    assert await aaf._key("nope2") is None
    assert len(platform.calls("GET", "/.well-known/jwks.json")) == 2
