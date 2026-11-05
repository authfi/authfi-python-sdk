"""A workload-bound agent asks a person and opens approvals with its instance token, and can ask to be observed.

The fake platform takes the credential as agent-identity does: on the CIBA start, the approval request and the
status read, the instance's access token with a DPoP proof by its key for that method and URL (``ath`` of the
token), and no assertion; at the token endpoint the CIBA grant is authenticated like the mint.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path

import jwt
import pytest

import authfi
from authfi import ConfigurationError, _dpop
from authfi._protocol.instance import observe_only_refusal
from authfi.agents.agent import CibaRequest
from authfi.errors import APIError, AuthFIError, OAuthError
from authfi.instance import AsyncInstanceAuth, Configuration, Evidence, InstanceAuth
from authfi.secret import Secret
from tests.agent_platform import AGENT
from tests.instance_platform import CONFIG, InstancePlatform
from tests.support import FakeClock

START = "/apis/regional/v1/ai/auth/backchannel-authorize"
APPROVE = "/apis/regional/v1/ai/auth/request-approval"
OBSERVE_BODY = {"version": 0, "digest": ""}


@pytest.fixture(autouse=True)
def fresh_nonces(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server nonce one test learned must not be sent by the next."""
    monkeypatch.setattr(_dpop, "nonces", _dpop.NonceCache())


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


def observing_env(platform: InstancePlatform, token_file: Path, **extra: str) -> dict[str, str]:
    env = env_for(platform, token_file, **extra)
    del env["AUTHFI_CONFIG_VERSION"], env["AUTHFI_CONFIG_DIGEST"]
    return env


def connect(platform: InstancePlatform, token_file: Path, env: dict[str, str] | None = None) -> InstanceAuth:
    agent = authfi.connect(
        env or env_for(platform, token_file), http_client=platform.client(), clock=FakeClock()
    )
    assert isinstance(agent, InstanceAuth)
    return agent


def ath(token: str) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(token.encode()).digest()).rstrip(b"=").decode()


def open_request(agent: InstanceAuth | AsyncInstanceAuth) -> CibaRequest:
    """A CIBA request the platform holds, as ``request_user_token`` would have returned it."""
    deadline = agent._clock.monotonic() + 300  # pyright: ignore[reportPrivateUsage]
    return CibaRequest(Secret("req-123"), 300, 5.0, None, deadline)


def claims_of(proof: str) -> dict[str, object]:
    return jwt.decode(proof, options={"verify_signature": False})


# --- asking a person -----------------------------------------------------------------------------------------


def test_an_instance_asks_a_person_with_its_token_and_a_proof_and_no_assertion(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = connect(platform, token_file)
    request = agent.request_user_token("priya@acme.com", binding_message="Refund order 4411")
    assert request.auth_req_id.reveal() == "req-123"
    sent = [r for r in platform.instance_calls if r.url.path == START]
    assert len(sent) == 1
    scheme, _, token = sent[0].headers["authorization"].partition(" ")
    assert scheme == "DPoP" and token == agent.token()
    proof = claims_of(sent[0].headers["dpop"])
    assert proof["htm"] == "POST"
    assert proof["htu"] == platform.issuer + START
    assert proof["ath"] == ath(token)
    assert "client_assertion" not in sent[0].content.decode()
    form = platform.ciba_starts[0]
    assert form["client_id"] == AGENT and form["login_hint"] == "priya@acme.com"
    agent.project.close()


def test_a_proof_by_another_key_is_refused_and_nothing_is_asked(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = connect(platform, token_file)
    token = agent.token()
    other = _dpop.DPoPKey()

    def wrong_key() -> Callable[[str, str], dict[str, str]]:
        return lambda method, url: dict(_dpop.authorization_headers(token, other, method, url, clock=time.time))

    agent._presenter = wrong_key  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(AuthFIError) as caught:
        agent.request_user_token("priya@acme.com", binding_message="Refund")
    assert caught.value.status == 401
    assert platform.holder_refusals == ["invalid_dpop_proof"]
    assert not platform.ciba_starts
    agent.project.close()


@pytest.mark.parametrize("defect", ["bearer", "no_ath", "other_method", "other_url"])
def test_the_platform_refuses_a_credential_that_is_not_proven_for_this_request(
    platform: InstancePlatform, token_file: Path, defect: str
) -> None:
    agent = connect(platform, token_file)
    token = agent.token()
    key = _dpop.run_key()

    def headers(method: str, url: str) -> dict[str, str]:
        if defect == "bearer":
            return {"Authorization": "Bearer " + token}
        if defect == "no_ath":
            return {"Authorization": "DPoP " + token, "DPoP": key.proof(method, url, now=time.time())}
        if defect == "other_method":
            proof = key.proof("GET", url, now=time.time(), access_token=token)
            return {"Authorization": "DPoP " + token, "DPoP": proof}
        return {
            "Authorization": "DPoP " + token,
            "DPoP": key.proof(method, url + "/other", now=time.time(), access_token=token),
        }

    agent._presenter = lambda: headers  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(AuthFIError) as caught:
        agent.request_user_token("priya@acme.com", binding_message="Refund")
    assert caught.value.status == 401
    assert not platform.ciba_starts
    agent.project.close()


def test_a_server_nonce_is_used_on_the_one_retry(platform: InstancePlatform, token_file: Path) -> None:
    agent = connect(platform, token_file)
    agent.token()
    platform.require_dpop_nonce = "n-1"
    agent.request_user_token("priya@acme.com", binding_message="Refund")
    posts = [r for r in platform.requests if r.url.path == START]
    assert len(posts) == 2, "refused once for want of the nonce, then accepted"
    assert claims_of(posts[1].headers["dpop"])["nonce"] == "n-1"
    agent.project.close()


def test_an_instance_redeems_with_its_attestation_and_gets_a_token_bound_to_its_key(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = connect(platform, token_file)
    platform.ciba_answers = ["pending", "ok"]
    delegated = agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
    assert len(platform.ciba_redeems) == 2
    redeem = platform.ciba_redeems[-1]
    assert {"oauth-client-attestation", "oauth-client-attestation-pop", "dpop"} <= set(redeem.headers)
    assert "client_assertion" not in redeem.content.decode()
    form = platform.ciba_polls[-1]
    assert form["client_id"] == AGENT and form["auth_req_id"] == "req-123"
    assert str(delegated.subject) == "user:usr_person"
    key = delegated.bound_key()
    assert key is not None and key.thumbprint == _dpop.run_key().thumbprint
    assert len(platform.attests) == 1, "the held attestation is presented, not renewed"
    agent.project.close()


def test_a_wait_that_outlives_the_attestation_renews_it_before_the_poll(
    platform: InstancePlatform, token_file: Path
) -> None:
    platform.attestation_lifetime = 100
    agent = connect(platform, token_file)
    agent.token()
    first = platform.token_requests[0].headers["oauth-client-attestation"]
    request = agent.request_user_token("priya@acme.com", binding_message="Refund")
    token_file.write_text(platform.k8s_token())  # the kubelet rotated it
    agent.await_user_token(request, poll_interval=50)
    assert len(platform.attests) == 2
    assert platform.ciba_redeems[0].headers["oauth-client-attestation"] != first
    agent.project.close()


def test_a_refusal_at_the_redeem_is_the_persons_decision_not_a_credential_failure(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = connect(platform, token_file)
    platform.ciba_answers = ["denied"]
    with pytest.raises(authfi.ConsentDenied):
        agent.token_as_user("priya@acme.com", binding_message="Refund")
    agent.project.close()


# --- approvals -----------------------------------------------------------------------------------------------


def test_an_instance_opens_and_reads_an_approval_with_its_token(platform: InstancePlatform, token_file: Path) -> None:
    agent = connect(platform, token_file)
    approval = agent.request_approval("refund order 4411", context={"order": 4411})
    assert approval.approval_id == "apr-1"
    opened = platform.approvals_opened[0]
    assert "client_assertion" not in opened and opened["agent_id"] == AGENT
    sent = next(r for r in platform.instance_calls if r.url.path == APPROVE)
    token = agent.token()
    assert sent.headers["authorization"] == "DPoP " + token
    proof = claims_of(sent.headers["dpop"])
    assert (proof["htm"], proof["htu"], proof["ath"]) == ("POST", platform.issuer + APPROVE, ath(token))
    platform.approval_status = "approved"
    assert agent.wait_for_approval("apr-1").decided
    read = [r for r in platform.instance_calls if r.method == "GET"][-1]
    assert claims_of(read.headers["dpop"])["htm"] == "GET"
    agent.project.close()


def test_an_approval_request_by_the_wrong_key_is_refused(platform: InstancePlatform, token_file: Path) -> None:
    agent = connect(platform, token_file)
    token = agent.token()
    other = _dpop.DPoPKey()
    agent._presenter = lambda: (  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
        lambda method, url: dict(_dpop.authorization_headers(token, other, method, url, clock=time.time))
    )
    with pytest.raises(AuthFIError) as caught:
        agent.request_approval("refund")
    assert caught.value.status == 401
    assert not platform.approvals_opened
    agent.project.close()


# --- the asyncio credential ----------------------------------------------------------------------------------


async def test_the_asyncio_instance_asks_a_person_and_opens_an_approval(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = authfi.connect_async(
        env_for(platform, token_file), http_client=platform.async_client(), clock=FakeClock()
    )
    assert isinstance(agent, AsyncInstanceAuth)
    delegated = await agent.token_as_user("priya@acme.com", binding_message="Refund")
    assert delegated.bound_key() is not None
    assert [r for r in platform.instance_calls if r.url.path == START]
    assert (await agent.request_approval("refund")).approval_id == "apr-1"
    platform.approval_status = "approved"
    assert (await agent.wait_for_approval("apr-1")).decided
    assert len(platform.attests) == 1
    await agent.project.aclose()


async def test_the_asyncio_instance_with_the_wrong_key_is_refused(platform: InstancePlatform, token_file: Path) -> None:
    agent = authfi.connect_async(
        env_for(platform, token_file), http_client=platform.async_client(), clock=FakeClock()
    )
    assert isinstance(agent, AsyncInstanceAuth)
    token = await agent.token()
    other = _dpop.DPoPKey()

    async def wrong_key() -> Callable[[str, str], dict[str, str]]:
        return lambda method, url: dict(_dpop.authorization_headers(token, other, method, url, clock=time.time))

    agent._presenter = wrong_key  # type: ignore[method-assign]  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(AuthFIError) as caught:
        await agent.request_user_token("priya@acme.com", binding_message="Refund")
    assert caught.value.status == 401
    await agent.project.aclose()


# --- observe mode --------------------------------------------------------------------------------------------


def test_an_instance_names_no_version_only_when_it_asks_to_be_observed(
    platform: InstancePlatform, token_file: Path
) -> None:
    platform.observe_window = True
    agent = connect(platform, token_file)
    agent.token()
    assert platform.attests[0]["config"] == CONFIG, "an instance that names a version sends it"
    asked = connect(platform, token_file, observing_env(platform, token_file, AUTHFI_CONFIG_OBSERVE="1"))
    platform.approved_config = dict(CONFIG)
    token_file.write_text(platform.k8s_token())
    asked.token()
    assert platform.attests[1]["config"] == OBSERVE_BODY
    assert jwt.decode(asked.token(), options={"verify_signature": False})["cfg_obs"] is True
    agent.project.close()
    asked.project.close()


@pytest.mark.parametrize("how", ["env", "argument", "configuration"])
def test_every_explicit_way_to_ask_to_be_observed_sends_version_0_and_no_digest(
    platform: InstancePlatform, token_file: Path, how: str
) -> None:
    platform.observe_window = True
    if how == "env":
        agent = connect(platform, token_file, observing_env(platform, token_file, AUTHFI_CONFIG_OBSERVE="true"))
    elif how == "argument":
        built = authfi.connect(
            observing_env(platform, token_file), http_client=platform.client(), clock=FakeClock(), observe=True
        )
        assert isinstance(built, InstanceAuth)
        agent = built
    else:
        project = authfi.AuthFI(platform.issuer, http_client=platform.client(), clock=FakeClock())
        agent = InstanceAuth(
            project, AGENT, Evidence("k8s", str(token_file)), config=Configuration.observe()
        )
    agent.token()
    assert platform.attests[0]["config"] == OBSERVE_BODY
    agent.project.close()


def test_naming_no_version_is_still_refused_unless_observing_is_asked_for(
    platform: InstancePlatform, token_file: Path
) -> None:
    platform.observe_window = True
    with pytest.raises(ConfigurationError, match="AUTHFI_CONFIG_OBSERVE"):
        authfi.connect(observing_env(platform, token_file), http_client=platform.client())
    for off in ("", "0", "false"):
        with pytest.raises(ConfigurationError, match="AUTHFI_CONFIG_VERSION"):
            authfi.connect(
                observing_env(platform, token_file, AUTHFI_CONFIG_OBSERVE=off), http_client=platform.client()
            )
    assert not platform.attests, "nothing was sent"


def test_a_version_and_a_request_to_be_observed_are_not_both_accepted(
    platform: InstancePlatform, token_file: Path
) -> None:
    with pytest.raises(ConfigurationError, match="not both"):
        authfi.connect(
            env_for(platform, token_file, AUTHFI_CONFIG_OBSERVE="1"), http_client=platform.client()
        )
    with pytest.raises(ConfigurationError, match="AUTHFI_CONFIG_OBSERVE is 1"):
        authfi.connect(
            observing_env(platform, token_file, AUTHFI_CONFIG_OBSERVE="maybe"), http_client=platform.client()
        )
    assert not platform.attests


def test_observing_is_for_an_instance_not_for_a_key_file(platform: InstancePlatform, tmp_path: Path) -> None:
    env = {"AUTHFI_ISSUER": platform.issuer, "AUTHFI_AGENT_ID": AGENT, "AUTHFI_AGENT_KEY_FILE": str(tmp_path / "k")}
    with pytest.raises(ConfigurationError, match="observe is for an agent that signs in as an instance"):
        authfi.connect(env, http_client=platform.client(), observe=True)


def test_a_closed_window_says_what_to_do_in_one_sentence(platform: InstancePlatform, token_file: Path) -> None:
    agent = connect(platform, token_file, observing_env(platform, token_file, AUTHFI_CONFIG_OBSERVE="1"))
    with pytest.raises(APIError) as caught:
        agent.token()
    assert platform.refused == ["config_not_approved"]
    assert caught.value.status == 401
    text = str(caught.value)
    assert "ask a person to open one in the console" in text
    assert "AUTHFI_CONFIG_VERSION and AUTHFI_CONFIG_DIGEST" in text
    assert "observation window" in text
    agent.project.close()


def test_a_refused_version_is_not_told_to_open_a_window(platform: InstancePlatform, token_file: Path) -> None:
    platform.approved_config = {"version": 8, "digest": CONFIG["digest"]}
    agent = connect(platform, token_file)
    with pytest.raises(APIError) as caught:
        agent.token()
    assert "observation window" not in str(caught.value)
    agent.project.close()


def test_an_instance_admitted_only_to_be_observed_is_told_the_fix_when_it_asks_a_person(
    platform: InstancePlatform, token_file: Path
) -> None:
    platform.observe_window = True
    agent = connect(platform, token_file, observing_env(platform, token_file, AUTHFI_CONFIG_OBSERVE="1"))
    with pytest.raises(OAuthError) as asked:
        agent.request_user_token("priya@acme.com", binding_message="Refund")
    assert asked.value.code == "unauthorized_client" and asked.value.status == 403
    assert "admitted to be observed" in str(asked.value), "the platform's own words are kept"
    assert "Approve a configuration version" in str(asked.value)
    assert "AUTHFI_CONFIG_VERSION" in str(asked.value)
    redeeming = open_request(agent)
    with pytest.raises(OAuthError) as redeemed:
        agent.await_user_token(redeeming)
    assert redeemed.value.code == "observe_only_instance" and redeemed.value.status == 403
    assert "Approve a configuration version" in str(redeemed.value)
    assert not platform.ciba_starts
    agent.project.close()


async def test_the_asyncio_instance_admitted_only_to_be_observed_is_told_the_fix(
    platform: InstancePlatform, token_file: Path
) -> None:
    platform.observe_window = True
    agent = authfi.connect_async(
        observing_env(platform, token_file),
        http_client=platform.async_client(),
        clock=FakeClock(),
        observe=True,
    )
    assert isinstance(agent, AsyncInstanceAuth)
    with pytest.raises(OAuthError, match="Approve a configuration version"):
        await agent.request_user_token("priya@acme.com", binding_message="Refund")
    redeeming = open_request(agent)
    with pytest.raises(OAuthError, match="Approve a configuration version"):
        await agent.await_user_token(redeeming)
    await agent.project.aclose()


def test_a_refusal_that_is_not_the_observe_one_is_left_as_the_platform_said_it(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = connect(platform, token_file)
    platform.ciba_answers = ["invalid_grant"]
    request = agent.request_user_token("priya@acme.com", binding_message="Refund")
    with pytest.raises(OAuthError) as caught:
        agent.await_user_token(request)
    assert caught.value.code == "invalid_grant"
    assert "Approve a configuration version" not in str(caught.value)
    agent.project.close()


def test_the_signed_request_body_of_the_default_config_is_unchanged(
    platform: InstancePlatform, token_file: Path
) -> None:
    agent = connect(platform, token_file)
    agent.token()
    assert json.dumps(platform.attests[0]["config"], sort_keys=True) == json.dumps(CONFIG, sort_keys=True)
    agent.project.close()


@pytest.mark.parametrize(
    "refusal",
    [
        OAuthError("insufficient_scope", "the scope is not held", status=403),
        OAuthError("unauthorized_client", "this client may not use the grant", status=403),
        OAuthError("observe_only_instance", "admitted to be observed", status=400),
        APIError("forbidden", "not yours", status=403),
        APIError("observe_only_instance", "admitted to be observed", status=401),
    ],
)
def test_only_the_observe_refusals_get_the_observe_fix(refusal: AuthFIError) -> None:
    assert observe_only_refusal(refusal) is refusal


@pytest.mark.parametrize("kind", [OAuthError, APIError])
def test_an_observe_refusal_keeps_its_code_status_and_ids_and_gains_the_fix(kind: type[OAuthError | APIError]) -> None:
    said = kind("observe_only_instance", "admitted to be observed.", status=403)
    said.request_id, said.run_id = "req-1", "run-1"
    named = observe_only_refusal(said)
    assert type(named) is kind and named is not said
    kept = (named.code, named.status, named.request_id, named.run_id)
    assert kept == ("observe_only_instance", 403, "req-1", "run-1")
    assert str(named).count("admitted to be observed") == 1
    assert "Approve a configuration version" in str(named)


def test_an_instance_token_is_never_given_to_an_address_off_the_project(
    platform: InstancePlatform, token_file: Path
) -> None:
    """The instance's token and a proof of its key are made only for AuthFI's own addresses."""
    agent = connect(platform, token_file)
    with pytest.raises(AuthFIError):
        agent.headers("POST", "https://evil.example/ciba")
    good = agent.headers("POST", platform.issuer + START)
    assert good["Authorization"].startswith("DPoP ")
    agent.project.close()
