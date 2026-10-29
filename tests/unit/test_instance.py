"""Instance sign-in (C3, client side) against a faithful fake of the platform."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jwt
import pytest

import authfi
from authfi import AgentKey, ConfigurationError, NotSupportedError
from authfi.errors import APIError, OAuthError
from authfi.instance import AsyncInstanceAuth, Evidence, InstanceAuth, find_evidence
from tests.agent_platform import AGENT
from tests.instance_platform import ATTEST, ATTEST_FIELDS, CHALLENGE, CONFIG, InstancePlatform
from tests.support import FakeClock, json_response


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


def connect(platform: InstancePlatform, token_file: Path, clock: FakeClock | None = None, **extra: str) -> InstanceAuth:
    env = env_for(platform, token_file, **extra)
    if clock is None:
        credential = authfi.connect(env, http_client=platform.client())
    else:
        credential = authfi.connect(env, http_client=platform.client(), clock=clock)
    assert isinstance(credential, InstanceAuth)
    return credential


# --- signing in -----------------------------------------------------------------------------------


def test_connect_signs_in_by_attestation_with_no_key(platform: InstancePlatform, token_file: Path) -> None:
    agent = connect(platform, token_file)
    token = agent.token()
    claims = jwt.decode(token, options={"verify_signature": False})
    assert claims["sub"] == AGENT
    assert claims["cnf"]["jkt"]
    assert agent.sign_in == "instance"
    assert agent.instance_id is not None and agent.instance_id.startswith("inst_")
    assert len(platform.attests) == 1
    body = platform.attests[0]
    assert body["evidence"]["kind"] == "k8s"
    assert body["config"] == CONFIG
    assert set(body) <= ATTEST_FIELDS, "the route refuses a field it does not know"
    assert "instance_key" in body and "run_key" not in body
    assert "d" not in body["instance_key"], "only the public half of the key is sent"
    assert "private" not in str(body).lower()
    request = platform.token_requests[0]
    assert {"oauth-client-attestation", "oauth-client-attestation-pop", "dpop"} <= set(request.headers)
    assert "client_assertion" not in request.content.decode()
    agent.project.close()


def test_the_token_is_bound_to_the_instance_key_and_calls_carry_a_proof(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = connect(platform, token_file)
    url = platform.issuer + "/instance/echo"
    headers = agent.headers("GET", url)
    assert headers["Authorization"].startswith("DPoP ")
    answer = platform.client().get(url, headers=headers)
    assert answer.json() == {"ok": True, "sub": AGENT}
    # the same token as a Bearer is refused (C4), and so is a proof for another URL
    bearer = platform.client().get(url, headers={"Authorization": "Bearer " + agent.token()})
    assert bearer.status_code == 401
    other = agent.headers("GET", platform.issuer + "/elsewhere")
    assert platform.client().get(url, headers={**headers, "DPoP": other["DPoP"]}).status_code == 400
    agent.project.close()


def test_a_second_token_reuses_the_attestation_and_a_fresh_challenge(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = connect(platform, token_file)
    agent.token()
    agent.token(force=True)
    assert len(platform.attests) == 1
    assert len(platform.token_requests) == 2
    pops = [r.headers["oauth-client-attestation-pop"] for r in platform.token_requests]
    assert len(set(pops)) == 2
    agent.project.close()


def test_the_attestation_is_renewed_before_it_expires_with_a_rotated_token(
    platform: InstancePlatform, token_file: Path
) -> None:
    clock = FakeClock()
    agent = connect(platform, token_file, clock)
    agent.token()
    clock.advance(900 - 30)
    token_file.write_text(platform.k8s_token())
    agent.token(force=True)
    assert len(platform.attests) == 2
    agent.project.close()


def test_an_unrotated_service_account_token_is_never_sent_twice(platform: InstancePlatform, token_file: Path) -> None:
    clock = FakeClock()
    agent = connect(platform, token_file, clock)
    agent.token()
    clock.advance(900 - 30)
    with pytest.raises(ConfigurationError, match="expirationSeconds: 600"):
        agent.token(force=True)
    assert len(platform.attests) == 1
    agent.project.close()


def test_a_server_nonce_is_asked_for_once_and_used(platform: InstancePlatform, token_file: Path) -> None:
    platform.require_dpop_nonce = "n-1"
    agent = connect(platform, token_file)
    assert agent.token()
    assert len(platform.token_requests) == 2
    agent.project.close()


async def test_the_asyncio_credential_signs_in_the_same_way(platform: InstancePlatform, token_file: Path) -> None:
    agent = authfi.connect_async(env_for(platform, token_file), http_client=platform.async_client())
    assert isinstance(agent, AsyncInstanceAuth)
    headers = await agent.headers("GET", platform.issuer + "/instance/echo")
    assert headers["Authorization"].startswith("DPoP ")
    assert len(platform.attests) == 1
    await agent.project.aclose()


def test_a_device_key_signs_the_nonce(platform: InstancePlatform, tmp_path: Path) -> None:
    key = AgentKey.generate(kid="dev-1")
    path = tmp_path / "device.key"
    key.write(str(path))
    platform.binding_kind = "device"
    platform.device_jwk = {**key.public_jwk(), "alg": "EdDSA"}
    env = {"AUTHFI_ISSUER": platform.issuer, "AUTHFI_AGENT_ID": AGENT, "AUTHFI_DEVICE_KEY_FILE": str(path)}
    env["AUTHFI_DEVICE_KEY_KID"] = "dev-1"
    env["AUTHFI_CONFIG_VERSION"] = str(CONFIG["version"])
    env["AUTHFI_CONFIG_DIGEST"] = str(CONFIG["digest"])
    agent = authfi.connect(env, http_client=platform.client())
    assert isinstance(agent, InstanceAuth)
    assert agent.token()
    evidence = platform.attests[0]["evidence"]
    assert evidence["kind"] == "device"
    assert "." not in evidence["token"], "the evidence is the bare signature, not a JWT"
    assert platform.attests[0]["config"] == CONFIG
    agent.project.close()


def test_the_device_signature_is_over_the_nonce_the_agent_and_the_instance_key(tmp_path: Path) -> None:
    from cryptography.hazmat.primitives.asymmetric import ed25519

    from authfi._protocol.oauth import b64url
    from authfi.instance import _device_signature  # pyright: ignore[reportPrivateUsage]

    private = ed25519.Ed25519PrivateKey.generate()

    class Device:
        alg = "EdDSA"
        kid = "dev-1"

        def sign(self, data: bytes) -> bytes:
            return private.sign(data)

    token = _device_signature(Device(), "agt_test1", "n-1", "jkt-1")
    signature = base64_decode(token)
    private.public_key().verify(signature, b"authfi-agent-attest" + b"\n" + b"n-1\nagt_test1\njkt-1")
    assert token == b64url(signature)
    with pytest.raises(Exception, match=""):
        private.public_key().verify(signature, b"authfi-agent-attest" + b"\n" + b"n-1\nagt_test1\njkt-2")


def base64_decode(value: str) -> bytes:
    import base64

    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


# --- which credential ----------------------------------------------------------------------------


def test_evidence_beats_a_key_file_and_the_key_is_the_labelled_fallback(
    platform: InstancePlatform, token_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    key_path = tmp_path / "agent.key"
    AgentKey.generate(kid="k1").write(str(key_path))
    both = env_for(platform, token_file, AUTHFI_AGENT_KEY_FILE=str(key_path))
    assert isinstance(authfi.connect(both, http_client=platform.client()), InstanceAuth)
    legacy = {"AUTHFI_ISSUER": platform.issuer, "AUTHFI_AGENT_ID": AGENT, "AUTHFI_AGENT_KEY_FILE": str(key_path)}
    legacy["AUTHFI_AGENT_KID"] = "k1"
    with caplog.at_level("INFO", logger="authfi"):
        credential = authfi.connect(legacy, http_client=platform.client())
    assert not isinstance(credential, InstanceAuth)
    assert "legacy" in caplog.text


def test_an_agent_with_no_way_in_is_told_both_ways(platform: InstancePlatform, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("authfi.instance.K8S_TOKEN_PATH", "/nonexistent/token")
    env = {"AUTHFI_ISSUER": platform.issuer, "AUTHFI_AGENT_ID": AGENT}
    with pytest.raises(ConfigurationError, match=r"AUTHFI_DEVICE_KEY_FILE.*AUTHFI_AGENT_KEY_FILE"):
        authfi.connect(env, http_client=platform.client())
    with pytest.raises(ConfigurationError, match=r"AUTHFI_DEVICE_KEY_FILE.*AUTHFI_AGENT_KEY_FILE"):
        authfi.connect_async(env, http_client=platform.async_client())


def test_find_evidence_prefers_the_device_key_and_reads_only_what_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("authfi.instance.K8S_TOKEN_PATH", str(tmp_path / "none"))
    assert find_evidence({}) is None
    assert find_evidence({"AUTHFI_DEVICE_KEY_FILE": "/d"}) == Evidence("device", "/d")
    present = tmp_path / "token"
    present.write_text("x")
    monkeypatch.setattr("authfi.instance.K8S_TOKEN_PATH", str(present))
    assert find_evidence({}) == Evidence("k8s", str(present))


def test_a_half_written_configuration_is_refused(platform: InstancePlatform, token_file: Path) -> None:
    env = env_for(platform, token_file, AUTHFI_CONFIG_DIGEST="abc")
    with pytest.raises(ConfigurationError, match="sha256"):
        authfi.connect(env, http_client=platform.client())


# --- every refusal is the same answer, with what to check -----------------------------------------


def _unapproved(p: InstancePlatform) -> None:
    p.approved_config = {"version": 8, "digest": "sha256:" + "cd" * 32}


def _pod_gone(p: InstancePlatform) -> None:
    p.pod_live = False


def _bound_to_a_device(p: InstancePlatform) -> None:
    p.binding_kind = "device"


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (_unapproved, "config_not_approved"),
        (_pod_gone, "pod_not_live"),
        (_bound_to_a_device, "evidence_kind_mismatch"),
    ],
)
def test_every_platform_refusal_is_one_401_that_says_where_the_reason_is(
    platform: InstancePlatform, token_file: Path, setup: Callable[[InstancePlatform], None], reason: str
) -> None:
    setup(platform)
    agent = connect(platform, token_file)
    with pytest.raises(APIError) as caught:
        agent.token()
    assert platform.refused == [reason], "the platform knows why; the answer does not say"
    assert caught.value.status == 401
    assert reason not in str(caught.value)
    assert "audit tape" in str(caught.value)
    assert not platform.token_requests
    agent.project.close()


def test_a_replayed_evidence_token_is_refused(platform: InstancePlatform, tmp_path: Path) -> None:
    token = platform.k8s_token(jti="same")
    platform.evidence_jtis.add("same")
    path = tmp_path / "t"
    path.write_text(token)
    agent = connect(platform, path)
    with pytest.raises(APIError) as caught:
        agent.token()
    assert platform.refused == ["token_reused"]
    assert caught.value.status == 401
    agent.project.close()


def test_a_kind_of_workload_the_agent_is_not_bound_to_is_refused(platform: InstancePlatform, token_file: Path) -> None:
    config = authfi.instance.Configuration(CONFIG["version"], CONFIG["digest"])
    agent = InstanceAuth(
        authfi.AuthFI(platform.issuer, http_client=platform.client()),
        AGENT,
        Evidence("gcp", str(token_file)),
        config=config,
    )
    with pytest.raises(APIError) as caught:
        agent.token()
    assert platform.refused == ["evidence_kind_mismatch"]
    assert caught.value.status == 401
    agent.project.close()


def test_a_platform_that_cannot_decide_is_a_503_that_says_to_sign_in_again(
    platform: InstancePlatform, token_file: Path
) -> None:
    platform.unavailable = True
    agent = connect(platform, token_file)
    with pytest.raises(APIError) as caught:
        agent.token()
    assert caught.value.code == "temporarily_unavailable"
    assert caught.value.status == 503
    assert "Sign in again" in str(caught.value)
    assert len(platform.attests) == 1, "an attestation that may have spent the evidence is not sent again"
    agent.project.close()


def test_a_token_for_another_audience_is_refused_before_it_is_sent(platform: InstancePlatform, tmp_path: Path) -> None:
    path = tmp_path / "t"
    path.write_text(platform.k8s_token(aud="https://kubernetes.default.svc"))
    agent = connect(platform, path)
    with pytest.raises(ConfigurationError, match=platform.issuer):
        agent.token()
    assert not platform.attests
    agent.project.close()


def test_an_unreadable_evidence_file_names_the_path(platform: InstancePlatform, tmp_path: Path) -> None:
    agent = connect(platform, tmp_path / "missing")
    with pytest.raises(ConfigurationError, match="missing"):
        agent.token()
    agent.project.close()


# --- what comes back is checked --------------------------------------------------------------------


def _answer_with(platform: InstancePlatform, **changes: object) -> None:
    """Make the attest route answer an attestation altered by ``changes``."""
    original = platform._attestation  # pyright: ignore[reportPrivateUsage]

    def altered(body: dict[str, object], run_jwk: dict[str, object]) -> str:
        raw = original(body, run_jwk)
        claims = jwt.decode(raw, options={"verify_signature": False})
        header = jwt.get_unverified_header(raw)
        claims.update(changes)
        return jwt.encode(claims, platform.keys.ed_key, algorithm="EdDSA", headers=header)

    platform._attestation = altered  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]


@pytest.mark.parametrize(
    ("changes", "says"),
    [
        ({"sub": "agt_other"}, "agt_other"),
        ({"cnf": {"jwk": {"kty": "EC", "crv": "P-256", "x": "AA", "y": "AA"}}}, "different key"),
        ({"exp": 4_000_000_000, "iat": 1}, "longer than 900"),
        ({"authfi_instance": {"instance_id": "inst_nope"}}, "instance_id"),
    ],
)
def test_an_attestation_for_another_agent_or_key_is_refused(
    platform: InstancePlatform, token_file: Path, changes: dict[str, object], says: str
) -> None:
    _answer_with(platform, **changes)
    agent = connect(platform, token_file)
    with pytest.raises(OAuthError, match=says):
        agent.token()
    assert not platform.token_requests
    agent.project.close()


def test_an_attestation_of_the_wrong_type_is_refused(platform: InstancePlatform, token_file: Path) -> None:
    platform.route(
        "POST",
        ATTEST,
        lambda _r: json_response({"attestation": jwt.encode({"sub": AGENT}, "k" * 32, algorithm="HS256")}),
    )
    agent = connect(platform, token_file)
    with pytest.raises(OAuthError, match="typ"):
        agent.token()
    agent.project.close()


def test_a_challenge_without_a_nonce_is_refused(platform: InstancePlatform, token_file: Path) -> None:
    platform.route("POST", "/apis/regional/v1/ai/agents/attest/challenge", lambda _r: json_response({}))
    agent = connect(platform, token_file)
    with pytest.raises(OAuthError, match="nonce"):
        agent.token()
    agent.project.close()


def test_a_reply_wrapped_in_data_is_not_the_contract_and_is_not_read(platform: InstancePlatform, token_file: Path) -> None:
    platform.route(
        "POST",
        "/apis/regional/v1/ai/agents/attest/challenge",
        lambda _r: json_response({"data": {"nonce": "n-1"}}),
    )
    agent = connect(platform, token_file)
    with pytest.raises(OAuthError, match="nonce"):
        agent.token()
    assert not platform.attests
    agent.project.close()


def test_an_instance_without_its_approved_configuration_is_refused_before_anything_is_sent(
    platform: InstancePlatform, token_file: Path
) -> None:
    env = env_for(platform, token_file)
    del env["AUTHFI_CONFIG_VERSION"], env["AUTHFI_CONFIG_DIGEST"]
    with pytest.raises(ConfigurationError, match="AUTHFI_CONFIG_VERSION"):
        authfi.connect(env, http_client=platform.client())
    assert not platform.attests


@pytest.mark.parametrize("digest", ["sha256:abc", "sha256:" + "AB" * 32, "ab" * 32, "md5:" + "ab" * 16])
def test_an_image_digest_that_is_not_sha256_hex_is_refused(
    platform: InstancePlatform, token_file: Path, digest: str
) -> None:
    with pytest.raises(ConfigurationError, match="AUTHFI_IMAGE_DIGEST"):
        authfi.connect(env_for(platform, token_file, AUTHFI_IMAGE_DIGEST=digest), http_client=platform.client())


def test_the_proof_and_the_pop_are_signed_in_the_algorithm_of_the_instance_key(
    platform: InstancePlatform, token_file: Path
) -> None:
    from authfi import _dpop

    agent = connect(platform, token_file)
    agent.token()
    alg = _dpop.run_key().alg
    assert jwt.get_unverified_header(platform.attests[0]["proof"])["alg"] == alg
    pop = platform.token_requests[0].headers["oauth-client-attestation-pop"]
    assert jwt.get_unverified_header(pop)["alg"] == alg
    assert "challenge" not in jwt.decode(pop, options={"verify_signature": False})
    agent.project.close()


def test_the_token_request_asks_for_no_challenge_of_its_own(platform: InstancePlatform, token_file: Path) -> None:
    calls: list[int] = []
    inner = platform._challenge  # pyright: ignore[reportPrivateUsage]
    platform.route("POST", CHALLENGE, lambda r: calls.append(1) or inner(r))
    agent = connect(platform, token_file)
    agent.token()
    agent.token(force=True)
    assert len(calls) == 1, "one challenge, for the attestation; a token request needs none"
    agent.project.close()


# --- what the instance does not do, and what never leaks -------------------------------------------


def test_asking_a_person_with_an_instance_is_refused_with_the_fix(platform: InstancePlatform, token_file: Path) -> None:
    agent = connect(platform, token_file)
    with pytest.raises(NotSupportedError, match="token exchange"):
        agent.request_user_token("priya@acme.com", binding_message="Refund")
    agent.project.close()


def test_nothing_secret_is_in_a_repr_or_a_log(
    platform: InstancePlatform, token_file: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level("DEBUG"):
        agent = connect(platform, token_file)
        token = agent.token()
    secrets = [token, token_file.read_text(), platform.attests[0]["proof"]]
    shown = repr(agent) + repr(agent._core.attestation) + caplog.text  # pyright: ignore[reportPrivateUsage]
    for secret in secrets:
        assert secret not in shown
    agent.project.close()


def test_a_forked_child_does_not_reuse_the_parents_attestation(platform: InstancePlatform, token_file: Path) -> None:
    agent = connect(platform, token_file)
    agent.token()
    held = agent._core.attestation  # pyright: ignore[reportPrivateUsage]
    assert held is not None
    from authfi import _dpop

    _dpop._after_fork_in_child()  # pyright: ignore[reportPrivateUsage]
    assert agent._core.reusable(agent.project.clock.time(), 60) is None  # pyright: ignore[reportPrivateUsage]
    agent.project.close()
