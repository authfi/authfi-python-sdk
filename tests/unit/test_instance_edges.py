"""Instance sign-in: the wire parsers and the asyncio credential, edge by edge."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import jwt
import pytest

import authfi
from authfi import ConfigurationError, NotSupportedError
from authfi._protocol.http import HttpResponse
from authfi._protocol.instance import (
    jwk_thumbprint,
    parse_attestation,
    parse_challenge,
    refusal_with_fix,
)
from authfi.errors import APIError, OAuthError
from authfi.instance import AsyncInstanceAuth, Evidence, InstanceAuth
from tests.agent_platform import AGENT
from tests.instance_platform import CONFIG, CONFIG_REF, InstancePlatform


def reply(status: int, body: Any, *, raw: bytes | None = None) -> HttpResponse:
    import json

    return HttpResponse(status, (), raw if raw is not None else json.dumps(body).encode(), "https://x.example/a", "r")


# --- the parsers ---------------------------------------------------------------------------------------------


def test_jwk_thumbprint_knows_only_the_keys_this_sdk_makes() -> None:
    assert jwk_thumbprint({"kty": "RSA", "n": "x", "e": "AQ"}) is None
    assert jwk_thumbprint({"kty": "EC", "crv": "P-256", "x": "a"}) is None
    assert jwk_thumbprint({"kty": "OKP", "crv": "Ed25519", "x": "a"})


def test_a_refusal_other_than_the_two_the_route_gives_is_passed_through() -> None:
    plain = refusal_with_fix(reply(403, {"error": "something_new", "error_description": "no"}), "the route")
    assert isinstance(plain, APIError) and plain.code == "something_new" and plain.description == "no"
    html = refusal_with_fix(reply(502, None, raw=b"<html>bad gateway</html>"), "the route")
    assert not isinstance(html, APIError)


def test_the_challenge_and_the_attestation_answers_are_checked() -> None:
    with pytest.raises(OAuthError, match="JSON object"):
        parse_challenge(reply(200, None, raw=b"nope"))
    with pytest.raises(APIError, match="invalid_request"):
        parse_challenge(reply(400, {"error": "invalid_request"}))
    with pytest.raises(OAuthError, match="no attestation"):
        parse_attestation(reply(200, {}), agent_id=AGENT, key_thumbprint="t", now=time.time())
    with pytest.raises(OAuthError, match="not a JWT"):
        parse_attestation(reply(200, {"attestation": "nope"}), agent_id=AGENT, key_thumbprint="t", now=time.time())


def attestation_with(platform: InstancePlatform, **claims: Any) -> HttpResponse:
    from authfi._dpop import DPoPKey

    key = DPoPKey()
    body = {"agent_id": AGENT, "evidence": {"kind": "k8s"}, "config": CONFIG}
    raw = platform._attestation(body, key.public_jwk())  # pyright: ignore[reportPrivateUsage]
    parts = jwt.decode(raw, options={"verify_signature": False})
    parts.update(claims)
    parts = {k: v for k, v in parts.items() if v is not None}
    token = jwt.encode(parts, platform.keys.ed_key, algorithm="EdDSA", headers={"typ": "oauth-client-attestation+jwt"})
    response = reply(200, {"attestation": token})
    object.__setattr__(response, "url", key.thumbprint)
    return response


def test_an_attestation_without_a_lifetime_or_already_over_is_refused() -> None:
    platform = InstancePlatform()
    for changes, says in (({"exp": None}, "no lifetime"), ({"exp": 1, "iat": 0}, "expired")):
        response = attestation_with(platform, **changes)
        with pytest.raises(OAuthError, match=says):
            parse_attestation(response, agent_id=AGENT, key_thumbprint=response.url, now=time.time())


# --- the credential, without a network ------------------------------------------------------------------------------


@pytest.fixture
def platform() -> InstancePlatform:
    return InstancePlatform()


@pytest.fixture
def token_file(tmp_path: Path, platform: InstancePlatform) -> Path:
    path = tmp_path / "token"
    path.write_text(platform.k8s_token())
    return path


def env_for(platform: InstancePlatform, token_file: Path, **extra: str) -> dict[str, str]:
    return {
        "AUTHFI_ISSUER": platform.issuer,
        "AUTHFI_AGENT_ID": AGENT,
        "AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE": str(token_file),
        "AUTHFI_CONFIG_VERSION": str(CONFIG["version"]),
        "AUTHFI_CONFIG_DIGEST": str(CONFIG["digest"]),
        **extra,
    }


def blocking(platform: InstancePlatform, token_file: Path, **extra: str) -> InstanceAuth:
    credential = authfi.connect(env_for(platform, token_file, **extra), http_client=platform.client())
    assert isinstance(credential, InstanceAuth)
    return credential


def test_the_instance_key_signs_nothing_an_agent_key_would(platform: InstancePlatform, token_file: Path) -> None:
    from authfi import _dpop

    agent = blocking(platform, token_file)
    assert agent.key.alg == _dpop.run_key().alg and agent.key.kid
    assert agent.key.public_jwk() == _dpop.run_key().public_jwk()
    with pytest.raises(NotSupportedError, match="token exchange"):
        agent.key.sign(b"x")
    assert repr(agent) == f"InstanceAuth(agent_id='{AGENT}', evidence='k8s')"
    agent.project.close()


def test_unusable_evidence_is_named(platform: InstancePlatform, tmp_path: Path) -> None:
    project = authfi.AuthFI(platform.issuer, http_client=platform.client())
    config = CONFIG_REF
    missing = InstanceAuth(project, AGENT, Evidence("device", str(tmp_path / "none"), "k"), config=config)
    with pytest.raises(ConfigurationError, match="AUTHFI_DEVICE_KEY_KID"):
        missing.token()
    empty = tmp_path / "empty"
    empty.write_text("")
    with pytest.raises(ConfigurationError, match="is empty"):
        InstanceAuth(project, AGENT, Evidence("k8s", str(empty)), config=config).token()
    junk = tmp_path / "junk"
    junk.write_text("not a jwt")
    with pytest.raises(ConfigurationError, match="service-account JWT"):
        InstanceAuth(project, AGENT, Evidence("k8s", str(junk)), config=config).token()
    project.close()


def test_a_string_audience_is_accepted_and_the_image_digest_is_sent(platform: InstancePlatform, tmp_path: Path) -> None:
    now = int(time.time())
    claims = {
        "aud": platform.issuer,
        "iat": now,
        "exp": now + 600,
        "jti": "j1",
        "kubernetes.io": {"pod": {"name": "p", "uid": "u"}, "node": {"name": "n"}},
    }
    path = tmp_path / "t"
    path.write_text(jwt.encode(claims, platform.cluster_key, algorithm="EdDSA"))
    agent = blocking(platform, path, AUTHFI_IMAGE_DIGEST="sha256:" + "ef" * 32)
    assert agent.token()
    assert platform.attests[0]["image_digest"] == "sha256:" + "ef" * 32
    agent.project.close()


def test_an_environment_with_no_agent_or_no_evidence_is_named(platform: InstancePlatform, token_file: Path) -> None:
    project = authfi.AuthFI(platform.issuer, http_client=platform.client())
    with pytest.raises(ConfigurationError, match="AUTHFI_AGENT_ID"):
        InstanceAuth.from_environment(project, {"AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE": str(token_file)})
    with pytest.raises(ConfigurationError, match="no evidence"):
        InstanceAuth.from_environment(
            project, {"AUTHFI_DEVICE_KEY_FILE": ""} | {"AUTHFI_AGENT_ID": AGENT}, resource=None
        )
    project.close()


def test_a_project_without_the_grant_is_told_so(platform: InstancePlatform, token_file: Path) -> None:
    platform.discovery["grant_types_supported"] = ["authorization_code"]
    agent = blocking(platform, token_file)
    with pytest.raises(NotSupportedError, match="client credentials"):
        agent.token()
    agent.project.close()


def test_a_customer_apis_token_is_bound_and_goes_only_where_named(platform: InstancePlatform, token_file: Path) -> None:
    agent = blocking(platform, token_file)
    api = agent.for_resource("api_orders", send_to=["https://api.example.com"])
    headers = api.headers("GET", "https://api.example.com/orders")
    assert headers["Authorization"].startswith("DPoP ") and headers["DPoP"]
    with pytest.raises(ConfigurationError, match="refusing to send"):
        api.headers("GET", "https://evil.example/orders")
    with pytest.raises(ConfigurationError, match="resource is required"):
        agent.for_resource(" ")
    agent.project.close()


# --- asyncio --------------------------------------------------------------------------------------------------------


def instance_of(agent: AsyncInstanceAuth) -> str | None:
    return agent.instance_id


async def test_the_asyncio_credential_in_full(platform: InstancePlatform, token_file: Path) -> None:
    platform.require_dpop_nonce = "n-1"
    agent = authfi.connect_async(env_for(platform, token_file), http_client=platform.async_client())
    assert isinstance(agent, AsyncInstanceAuth)
    assert instance_of(agent) is None
    await agent.token()
    assert (instance_of(agent) or "").startswith("inst_")
    await agent.token(force=True)  # the held attestation is reused
    assert len(platform.attests) == 1
    api = agent.for_resource("api_orders", send_to=["https://api.example.com"])
    assert (await api.headers("GET", "https://api.example.com/x"))["Authorization"].startswith("DPoP ")
    with pytest.raises(ConfigurationError, match="resource is required"):
        agent.for_resource("")
    with pytest.raises(NotSupportedError):
        agent.key.sign(b"x")
    assert repr(agent) == f"AsyncInstanceAuth(agent_id='{AGENT}', evidence='k8s')"
    await agent.project.aclose()


async def test_the_asyncio_credential_names_what_is_missing(platform: InstancePlatform, token_file: Path) -> None:
    project = authfi.AsyncAuthFI(platform.issuer, http_client=platform.async_client())
    with pytest.raises(ConfigurationError, match="no evidence"):
        AsyncInstanceAuth.from_environment(project, {"AUTHFI_AGENT_ID": AGENT, "AUTHFI_DEVICE_KEY_FILE": ""})
    platform.discovery["grant_types_supported"] = ["authorization_code"]
    agent = AsyncInstanceAuth(project, AGENT, Evidence("k8s", str(token_file)), config=CONFIG_REF)
    with pytest.raises(NotSupportedError, match="client credentials"):
        await agent.token()
    with pytest.raises(NotSupportedError):
        agent._assertion("aud")  # pyright: ignore[reportPrivateUsage]
    await project.aclose()
