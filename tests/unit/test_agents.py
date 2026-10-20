from __future__ import annotations

import asyncio
import json
import pickle
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest

from authfi import (
    AgentKey,
    AsyncAuthFI,
    AuthFI,
    AuthFISecurityWarning,
    ConfigurationError,
    ConsentDenied,
    ConsentExpired,
    NotSupportedError,
    OAuthError,
    Principal,
    Secret,
)
from authfi.agents import AgentAuth, AsyncAgentAuth, DelegatedToken
from tests.agent_platform import AGENT, CIBA, AgentPlatform
from tests.support import FakeClock, form_of

API = "https://api.example.com"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate("ES256", kid="k1")


@pytest.fixture
def platform(key: AgentKey) -> AgentPlatform:
    return AgentPlatform(agent_jwk=key.public_jwk())


@pytest.fixture
def af(platform: AgentPlatform, clock: FakeClock) -> Iterator[AuthFI]:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        yield project


@pytest.fixture
def agent(af: AuthFI, key: AgentKey) -> AgentAuth:
    return af.agent(AGENT, key)


def claims_of(raw: str) -> dict[str, Any]:
    return jwt.decode(raw, options={"verify_signature": False})


# --- its own token ---------------------------------------------------------------------------------


def test_the_agent_mints_with_a_signed_assertion_and_no_secret(agent: AgentAuth, platform: AgentPlatform) -> None:
    token = agent.token()
    assert claims_of(token)["sub"] == AGENT
    form = form_of(platform.calls("POST", "/oauth/token")[0])
    assert form["grant_type"] == "client_credentials"
    assert form["client_id"] == AGENT
    assert form["client_assertion_type"] == "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
    assert "client_secret" not in form
    assert "resource" not in form
    assertion = claims_of(form["client_assertion"])
    assert assertion["aud"] == platform.issuer + "/oauth/token"
    assert assertion["exp"] - assertion["iat"] == 300


def test_the_token_is_cached_per_resource_and_renewed_before_it_expires(
    agent: AgentAuth, platform: AgentPlatform, clock: FakeClock
) -> None:
    first = agent.token()
    assert agent.token() == first
    assert platform.minted == 1
    agent.token(resource="api_orders")
    assert platform.minted == 2
    assert form_of(platform.calls("POST", "/oauth/token")[-1])["resource"] == "api_orders"
    clock.advance(platform.expires_in - agent.refresh_skew + 1)
    agent.token()
    assert platform.minted == 3
    agent.token(force=True)
    assert platform.minted == 4


def test_many_callers_share_one_mint(af: AuthFI, key: AgentKey, platform: AgentPlatform) -> None:
    agent = af.agent(AGENT, key)
    af.metadata()
    tokens: list[str] = []
    threads = [threading.Thread(target=lambda: tokens.append(agent.token())) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(tokens)) == 1
    assert platform.minted == 1


def test_capabilities_come_with_the_token(agent: AgentAuth, platform: AgentPlatform) -> None:
    assert agent.capabilities() == frozenset(platform.capabilities)


def test_a_refusal_is_raised_once(agent: AgentAuth, platform: AgentPlatform) -> None:
    platform.agent_jwk = AgentKey.generate(kid="k1").public_jwk()  # the project holds another key
    with pytest.raises(OAuthError) as caught:
        agent.token()
    assert caught.value.code == "invalid_client"
    assert len(platform.calls("POST", "/oauth/token")) == 1


@pytest.mark.parametrize(
    ("field", "value", "feature"),
    [
        ("token_endpoint_auth_methods_supported", ["client_secret_basic"], "private_key_jwt"),
        ("token_endpoint_auth_signing_alg_values_supported", ["EdDSA"], "ES256"),
        ("grant_types_supported", ["authorization_code"], "client_credentials"),
    ],
)
def test_a_project_that_would_refuse_the_agent_says_so_up_front(
    agent: AgentAuth, platform: AgentPlatform, field: str, value: list[str], feature: str
) -> None:
    platform.discovery[field] = value
    with pytest.raises(NotSupportedError) as caught:
        agent.token()
    assert caught.value.feature == feature
    assert platform.calls("POST", "/oauth/token") == []


def test_the_agent_token_goes_only_to_authfi_unless_told_otherwise(agent: AgentAuth, af: AuthFI, key: AgentKey) -> None:
    assert agent.send_to == (af.issuer,)
    assert agent.headers("POST", af.issuer + "/mcp")["Authorization"].startswith("Bearer ")
    with pytest.raises(ConfigurationError):
        agent.headers("POST", API + "/x")
    api = agent.for_resource("api_orders", send_to=[API])
    assert api.send_to == (API,)
    assert claims_of(api.token())["aud"] == "api_orders"
    with pytest.raises(ConfigurationError, match="send_to is required"):
        agent.for_resource("api_orders")
    with pytest.raises(ConfigurationError, match="send_to is required"):
        af.agent(AGENT, key, resource="api_orders")


def test_construction_is_checked(af: AuthFI, key: AgentKey) -> None:
    with pytest.raises(ConfigurationError, match="agent_id"):
        af.agent("", key)
    with pytest.raises(ConfigurationError, match="no secret fallback"):
        af.agent(AGENT, "a-client-secret")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        af.agent(AGENT, key, refresh_skew=-1)
    assert "kid='k1'" in repr(af.agent(AGENT, key))


# --- from the environment --------------------------------------------------------------------------


def private_file(tmp_path: Path, key: AgentKey) -> Path:
    path = tmp_path / "agent.key"
    key.write(path)
    return path


def test_from_the_environment_with_a_key_file(af: AuthFI, key: AgentKey, tmp_path: Path) -> None:
    path = private_file(tmp_path, key)
    agent = af.agent_from_env(
        {
            "AUTHFI_AGENT_ID": AGENT,
            "AUTHFI_AGENT_KEY_FILE": str(path),
            "AUTHFI_AGENT_KID": "k1",
            "AUTHFI_AGENT_RESOURCE": "https://acme.authfi.io/mcp",
        }
    )
    assert agent.agent_id == AGENT
    assert agent.key.public_jwk() == key.public_jwk()
    assert agent.resource == "https://acme.authfi.io/mcp"


def test_a_key_in_the_environment_warns_and_the_file_wins(af: AuthFI, key: AgentKey, tmp_path: Path) -> None:
    other = AgentKey.generate("ES256", kid="k2")
    pem_path = tmp_path / "other.key"
    other.write(pem_path)
    pem = pem_path.read_text()
    with pytest.warns(AuthFISecurityWarning, match="AUTHFI_AGENT_KEY_FILE"):
        agent = af.agent_from_env({"AUTHFI_AGENT_ID": AGENT, "AUTHFI_AGENT_PRIVATE_KEY": pem, "AUTHFI_AGENT_KID": "k2"})
    assert agent.key.kid == "k2"
    path = private_file(tmp_path, key)
    with pytest.warns(AuthFISecurityWarning, match="using the file"):
        agent = af.agent_from_env(
            {
                "AUTHFI_AGENT_ID": AGENT,
                "AUTHFI_AGENT_KEY_FILE": str(path),
                "AUTHFI_AGENT_PRIVATE_KEY": pem,
                "AUTHFI_AGENT_KID": "k1",
            }
        )
    assert agent.key.kid == "k1"
    with pytest.warns(AuthFISecurityWarning), pytest.raises(ConfigurationError, match="AUTHFI_AGENT_KID"):
        af.agent_from_env({"AUTHFI_AGENT_ID": AGENT, "AUTHFI_AGENT_PRIVATE_KEY": pem})


def test_missing_settings_are_named(af: AuthFI) -> None:
    with pytest.raises(ConfigurationError, match="AUTHFI_AGENT_ID"):
        af.agent_from_env({})
    with pytest.raises(ConfigurationError, match="AUTHFI_AGENT_KEY_FILE"):
        af.agent_from_env({"AUTHFI_AGENT_ID": AGENT})


# --- acting for a person: CIBA ---------------------------------------------------------------------


def test_ciba_asks_with_a_binding_message_and_the_agents_own_credential(
    agent: AgentAuth, platform: AgentPlatform
) -> None:
    request = agent.request_user_token(
        "priya@acme.com", binding_message="Refund order 4411", scope=["send_email"], expires_in=120
    )
    assert "req-123" not in repr(request)
    form = form_of(platform.calls("POST", "/apis/regional/v1/ai/auth/backchannel-authorize")[0])
    assert form["login_hint"] == "priya@acme.com"
    assert form["binding_message"] == "Refund order 4411"
    assert form["scope"] == "send_email"
    assert form["requested_expiry"] == "120"
    assert form["client_id"] == AGENT
    with pytest.raises(TypeError):
        pickle.dumps(request)


@pytest.mark.parametrize(("hint", "message"), [("", "x"), ("priya@acme.com", ""), ("priya@acme.com", "  ")])
def test_ciba_needs_someone_to_ask_and_what_to_tell_them(agent: AgentAuth, hint: str, message: str) -> None:
    with pytest.raises(ConfigurationError):
        agent.request_user_token(hint, binding_message=message)


def test_the_first_poll_waits_the_interval_and_slow_down_widens_it(
    agent: AgentAuth, platform: AgentPlatform, clock: FakeClock
) -> None:
    platform.ciba_answers = ["pending", "slow_down", "pending", "ok"]
    request = agent.request_user_token("priya@acme.com", binding_message="Refund order 4411")
    delegated = agent.await_user_token(request)
    assert clock.sleeps == [5, 5, 10, 10]
    assert [p["grant_type"] for p in platform.ciba_polls] == [CIBA] * 4
    assertions = {p["client_assertion"] for p in platform.ciba_polls}
    assert len(assertions) == 4
    assert delegated.via == "ciba"
    assert delegated.subject == Principal("user", "usr_person")
    assert delegated.actor == Principal("agent", AGENT)
    assert delegated.scopes == frozenset({"send_email"})


@pytest.mark.parametrize(("answer", "error"), [("denied", ConsentDenied), ("expired", ConsentExpired)])
def test_a_decision_stops_the_wait_at_once(
    agent: AgentAuth, platform: AgentPlatform, answer: str, error: type[Exception]
) -> None:
    platform.ciba_answers = [answer]
    with pytest.raises(error):
        agent.token_as_user("priya@acme.com", binding_message="Refund")
    assert len(platform.ciba_polls) == 1


def test_the_wait_gives_up_at_its_timeout_and_the_request_stays_open(
    agent: AgentAuth, platform: AgentPlatform, clock: FakeClock
) -> None:
    platform.ciba_answers = ["pending"] * 10
    request = agent.request_user_token("priya@acme.com", binding_message="Refund")
    with pytest.raises(OAuthError) as caught:
        agent.await_user_token(request, timeout=12)
    assert caught.value.code == "authorization_pending"
    assert len(platform.ciba_polls) == 2
    platform.ciba_answers = ["ok"]
    assert agent.await_user_token(request).via == "ciba"


def test_a_ciba_token_is_never_renewed(agent: AgentAuth, platform: AgentPlatform, clock: FakeClock) -> None:
    delegated = agent.token_as_user("priya@acme.com", binding_message="Refund")
    assert claims_of(delegated.token())["sub"] == "user:usr_person"
    clock.advance(600 - agent.refresh_skew + 1)
    with pytest.raises(ConsentExpired):
        delegated.token()


def test_a_delegated_token_does_not_print_or_pickle(agent: AgentAuth) -> None:
    delegated = agent.token_as_user("priya@acme.com", binding_message="Refund")
    raw = delegated.token()
    assert raw not in repr(delegated)
    assert "user:usr_person" in repr(delegated)
    with pytest.raises(TypeError, match="checkpoint"):
        pickle.dumps(delegated)
    assert not hasattr(delegated, "__dict__")
    with pytest.raises(TypeError):
        json.dumps(delegated)


# --- acting for a person: token exchange ------------------------------------------------------------


def person_token(platform: AgentPlatform, lifetime: int = 900) -> Secret:
    return Secret(platform.access_token(sub="usr_7", exp=int(time.time()) + lifetime))


def test_the_exchange_carries_no_client_authentication(agent: AgentAuth, platform: AgentPlatform) -> None:
    delegated = agent.exchange(person_token(platform), resource="api_orders", scope=["orders:read"], send_to=[API])
    form = platform.exchanges[0]
    assert "client_id" not in form
    assert "client_assertion" not in form
    assert claims_of(form["actor_token"])["sub"] == AGENT
    assert form["subject_token_type"] == "urn:ietf:params:oauth:token-type:access_token"
    assert form["resource"] == "api_orders"
    assert form["scope"] == "orders:read"
    assert delegated.subject == Principal("user", "usr_7")
    assert delegated.actor == Principal("agent", AGENT)
    assert delegated.send_to == (API,)


def test_an_exchanged_token_is_exchanged_again_while_the_person_is_signed_in(
    agent: AgentAuth, platform: AgentPlatform, clock: FakeClock
) -> None:
    delegated = agent.exchange(person_token(platform, lifetime=3600))
    clock.advance(300 - agent.refresh_skew + 1)
    delegated.token()
    assert len(platform.exchanges) == 2


def test_an_exchange_stops_when_the_persons_token_has_expired(
    agent: AgentAuth, platform: AgentPlatform, clock: FakeClock
) -> None:
    delegated = agent.exchange(person_token(platform, lifetime=250))
    clock.advance(260)
    with pytest.raises(ConsentExpired, match="sign in again"):
        delegated.token()


def test_the_exchange_follows_a_refreshing_session(agent: AgentAuth, platform: AgentPlatform) -> None:
    from authfi import TokenSource

    class Session(TokenSource):
        def __init__(self) -> None:
            self.calls = 0

        @property
        def send_to(self) -> tuple[str, ...]:
            return (platform.issuer,)

        def token(self, *, force: bool = False) -> str:
            self.calls += 1
            return person_token(platform).reveal()

    session = Session()
    agent.exchange(session).token(force=True)
    assert session.calls == 2


def test_an_exchange_the_policy_refuses(agent: AgentAuth, platform: AgentPlatform) -> None:
    platform.exchange_refusal = "access_denied"
    with pytest.raises(OAuthError) as caught:
        agent.exchange(person_token(platform))
    assert caught.value.code == "access_denied"
    with pytest.raises(ConfigurationError):
        agent.exchange("raw-token")  # type: ignore[arg-type]


# --- the action gate and approvals -----------------------------------------------------------------


def test_freeze_an_action_then_ask_the_gate(agent: AgentAuth, platform: AgentPlatform) -> None:
    txn = agent.transaction_token("refund", resource="order:4411", context={"amount": 120})
    assert "txn-token" not in repr(txn)
    sent = json.loads(platform.calls("POST", "/apis/regional/v1/ai/auth/transaction-token")[0].content)
    assert sent == {"action": "refund", "resource": "order:4411", "context": {"amount": 120}}
    assert agent.authorize_action(txn).decision == "allow"
    platform.decision = "approval_required"
    held = agent.authorize_action(txn)
    assert held.decision == "approval_required"
    assert held.approval_id == "apr-1"
    gate = json.loads(platform.calls("POST", "/apis/regional/v1/ai/auth/authorize")[0].content)
    assert gate["transaction_token"] == "txn-token.refund"
    assert gate["agent_id"] == AGENT


def test_an_approval_is_opened_with_the_agents_own_credential(agent: AgentAuth, platform: AgentPlatform) -> None:
    approval = agent.request_approval("refund order 4411", context={"amount": 120}, expires_in=120)
    assert approval.approval_id == "apr-1"
    assert approval.status == "pending"
    assert not approval.decided
    opened = platform.approvals_opened[0]
    assert opened["timeout_seconds"] == 120
    assert claims_of(opened["client_assertion"])["aud"] == platform.issuer


def test_reading_an_approval_the_project_does_not_let_agents_read(agent: AgentAuth, platform: AgentPlatform) -> None:
    platform.approval_status = 403
    with pytest.raises(NotSupportedError) as caught:
        agent.approval("apr-1")
    assert caught.value.feature == "agent_approval_status"
    with pytest.raises(NotSupportedError):
        agent.wait_for_approval("apr-1")
    assert platform.approval_reads == 2


def test_waiting_for_an_approval(agent: AgentAuth, platform: AgentPlatform, clock: FakeClock) -> None:
    platform.approval_status = "approved"
    assert agent.wait_for_approval("apr-1").status == "approved"
    platform.approval_status = "pending"
    with pytest.raises(OAuthError) as caught:
        agent.wait_for_approval("apr-1", timeout=12)
    assert caught.value.code == "authorization_pending"
    assert clock.sleeps == [5.0, 5.0]


def test_a_project_without_agent_configuration_offers_no_approvals(agent: AgentAuth, platform: AgentPlatform) -> None:
    platform.route("GET", "/.well-known/agent-configuration", httpx.Response(404))
    with pytest.raises(NotSupportedError):
        agent.request_approval("refund")


# --- async parity ----------------------------------------------------------------------------------


async def test_the_async_agent(platform: AgentPlatform, key: AgentKey, clock: FakeClock) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as aaf:
        agent = aaf.agent(AGENT, key)
        assert isinstance(agent, AsyncAgentAuth)
        tokens = await asyncio.gather(*(agent.token() for _ in range(5)))
        assert len(set(tokens)) == 1
        assert platform.minted == 1
        assert await agent.capabilities() == frozenset(platform.capabilities)
        platform.ciba_answers = ["pending", "ok"]
        delegated = await agent.token_as_user("priya@acme.com", binding_message="Refund")
        assert delegated.subject == Principal("user", "usr_person")
        assert clock.sleeps == [5, 5]
        exchanged = await agent.exchange(person_token(platform))
        assert (await exchanged.headers("POST", platform.issuer + "/mcp"))["Authorization"].startswith("Bearer ")
        txn = await agent.transaction_token("refund")
        assert (await agent.authorize_action(txn)).decision == "allow"
        assert (await agent.request_approval("refund")).approval_id == "apr-1"
        platform.approval_status = "denied"
        assert (await agent.wait_for_approval("apr-1")).status == "denied"
        resource = agent.for_resource("api_orders", send_to=[API])
        assert claims_of(await resource.token())["aud"] == "api_orders"
        clock.advance(10_000)
        with pytest.raises(ConsentExpired):
            await delegated.token()


async def test_an_async_ciba_wait_can_be_cancelled(platform: AgentPlatform, key: AgentKey) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client()) as aaf:
        agent = aaf.agent(AGENT, key)
        platform.ciba_answers = ["pending"] * 100
        request = await agent.request_user_token("priya@acme.com", binding_message="Refund")
        waiting = asyncio.ensure_future(agent.await_user_token(request))
        await asyncio.sleep(0.05)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert platform.ciba_polls == []


def test_delegated_token_is_a_token_source() -> None:
    from authfi import TokenSource

    assert issubclass(DelegatedToken, TokenSource)
