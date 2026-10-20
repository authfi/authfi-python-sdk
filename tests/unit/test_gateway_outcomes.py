"""The gateway helpers' types and their edges: the outcome union, one hold contract, cold custom domains."""

from __future__ import annotations

import dataclasses
import gc
from collections.abc import Iterator

import pytest

from authfi import (
    APPROVAL_ARGUMENT,
    HOLD_MESSAGE,
    AgentKey,
    AsyncAuthFI,
    AuthFI,
    AuthFIError,
    ConfigurationError,
    ToolAllowed,
    ToolApprovalRequired,
    ToolOutcome,
    ToolRefused,
)
from authfi._protocol.gateway import MAX_CLOUD_SECONDS, cloud_request, read_result
from authfi.agents import AgentAuth
from tests.agent_platform import AGENT, AgentPlatform
from tests.support import FakeClock, fixture_json, json_response

HOLD_ID = "7f1c2a9e-5d1b-4f3a-9a77-2b0c8e1d4f00"
GATEWAY_HOLD = fixture_json("mcp/hold-result.json")["content"][0]["text"]


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def platform(key: AgentKey) -> AgentPlatform:
    return AgentPlatform(agent_jwk=key.public_jwk())


@pytest.fixture
def agent(platform: AgentPlatform, key: AgentKey, clock: FakeClock) -> Iterator[AgentAuth]:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        yield af.agent(AGENT, key)


# --- the outcome is three kinds, and the type checker can tell them apart -------------------------------


def describe(outcome: ToolOutcome) -> str:
    match outcome:
        case ToolAllowed(result=result):
            return f"allowed {result}"
        case ToolRefused(code=code):
            return f"refused {code}"
        case ToolApprovalRequired(approval_id=approval_id):
            return f"hold {approval_id.upper()}"  # approval_id is a str here, with no cast


def test_the_three_kinds_narrow_in_a_match() -> None:
    allowed = read_result({"content": [], "isError": False})
    refused = read_result(fixture_json("mcp/refusal-result.json"))
    held = read_result(fixture_json("mcp/hold-result.json"))
    assert [type(o) for o in (allowed, refused, held)] == [ToolAllowed, ToolRefused, ToolApprovalRequired]
    assert describe(refused) == "refused ETOOL003"
    assert describe(held) == f"hold {HOLD_ID.upper()}"
    assert [o.kind for o in (allowed, refused, held)] == ["allowed", "refused", "approval_required"]


def test_a_refusal_keeps_the_gateways_own_answer() -> None:
    answer = fixture_json("mcp/refusal-result.json")
    refused = read_result(answer)
    assert isinstance(refused, ToolRefused)
    assert refused.result is answer
    assert isinstance(read_result(fixture_json("mcp/hold-result.json")), ToolApprovalRequired)


def test_the_outcomes_are_frozen_and_slotted() -> None:
    held = ToolApprovalRequired(HOLD_ID, "approval_required")
    with pytest.raises(dataclasses.FrozenInstanceError):
        held.approval_id = "other"  # type: ignore[misc]
    assert not hasattr(held, "__dict__")
    assert not hasattr(ToolAllowed("x"), "__dict__")
    assert not hasattr(ToolRefused(), "__dict__")


def test_what_a_model_may_read() -> None:
    held = ToolApprovalRequired(HOLD_ID, "approval_required")
    assert held.model_text == HOLD_MESSAGE
    assert HOLD_ID not in held.model_text and APPROVAL_ARGUMENT not in held.model_text
    assert ToolAllowed({"content": [{"type": "text", "text": "ok"}]}).model_text == "ok"
    refused = ToolRefused(code="ETOOL003", reason="no grant", required_scope="mcp:a.b")
    assert refused.model_text == "Refused by AuthFI (ETOOL003): no grant. The call needs the scope mcp:a.b."


def test_the_model_facing_wait_is_the_same_text_in_every_sdk() -> None:
    assert HOLD_MESSAGE == "This action is waiting for a person's approval in AuthFI."


# --- one contract for the gateway's hold text ------------------------------------------------------------


def held(text: str) -> ToolOutcome:
    return read_result({"content": [{"type": "text", "text": text}], "isError": True})


def test_the_gateways_exact_hold_text_is_a_hold() -> None:
    outcome = held(GATEWAY_HOLD)
    assert isinstance(outcome, ToolApprovalRequired)
    assert (outcome.approval_id, outcome.code) == (HOLD_ID, "approval_required")


@pytest.mark.parametrize(
    "text",
    [
        "Waiting for approval a.b (approval_required). A person must approve this action in the AuthFI console.",
        "Waiting for approval a:b (approval_required). A person must approve this action in the AuthFI console.",
        "Waiting for approval  (approval_required). A person must approve this action in the AuthFI console.",
        "Waiting for approval x (Approval). A person must approve this action in the AuthFI console.",
        "Waiting for approval x (approval_required). Nobody needs to do anything.",
        "Waiting for approval x (approval_required).",
        "Please note: Waiting for approval x (approval_required). A person must approve this action",
    ],
    ids=["dot", "colon", "no id", "capital code", "other second sentence", "first sentence only", "not at the start"],
)
def test_text_that_is_not_the_gateways_hold_is_the_tools_own_error(text: str) -> None:
    assert isinstance(held(text), ToolAllowed)


def test_only_an_error_result_can_be_a_hold() -> None:
    assert isinstance(read_result({"content": [{"type": "text", "text": GATEWAY_HOLD}], "isError": False}), ToolAllowed)


# --- cloud credentials ------------------------------------------------------------------------------------


def test_cloud_credentials_cannot_be_asked_for_longer_than_the_platform_allows() -> None:
    assert MAX_CLOUD_SECONDS == 900
    with pytest.raises(ConfigurationError, match="1 to 900 seconds"):
        cloud_request("https://x", {}, account="a", role="r", expires_in=901, session_policy=None, approval_id=None)
    with pytest.raises(ConfigurationError, match="1 to 900 seconds"):
        cloud_request("https://x", {}, account="a", role="r", expires_in=0, session_policy=None, approval_id=None)
    cloud_request("https://x", {}, account="a", role="r", expires_in=900, session_policy=None, approval_id=None)


def test_only_201_and_202_are_an_answer_from_the_cloud_door(agent: AgentAuth, platform: AgentPlatform) -> None:
    platform.route(
        "POST", "/mcp/v1/cloud/credentials", json_response({"data": {"credential_id": "c", "provider": "aws"}}, 200)
    )
    with pytest.raises(AuthFIError) as caught:
        agent.gateway().cloud_credentials("prod", "admin")
    assert caught.value.status == 200


def custom_domain(key: AgentKey) -> AgentPlatform:
    platform = AgentPlatform(issuer="https://login.acme.com", agent_jwk=key.public_jwk())
    platform.discovery["backchannel_authentication_endpoint"] = (
        "https://acme.authfi.io/apis/regional/v1/ai/auth/backchannel-authorize"
    )
    return platform


def hosts(platform: AgentPlatform, method: str, path: str) -> list[str]:
    return [r.url.host for r in platform.calls(method, path)]


def test_a_cold_agent_on_a_custom_domain_asks_the_machine_api_host(key: AgentKey, clock: FakeClock) -> None:
    platform = custom_domain(key)
    platform.route(
        "POST",
        "/mcp/v1/cloud/credentials",
        json_response({"data": {"credential_id": "c", "provider": "aws", "access_key_id": "AKIA"}}, 201),
    )
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        outcome = af.agent(AGENT, key).gateway().cloud_credentials("prod", "admin")
    assert outcome.kind == "issued"
    assert hosts(platform, "POST", "/mcp/v1/cloud/credentials") == ["acme.authfi.io"]


def test_a_cold_agent_on_a_custom_domain_freezes_actions_at_the_machine_api_host(
    key: AgentKey, clock: FakeClock
) -> None:
    platform = custom_domain(key)
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        agent = af.agent(AGENT, key)
        txn = agent.transaction_token("refund")
        assert agent.authorize_action(txn).decision == "allow"
    assert hosts(platform, "POST", "/apis/regional/v1/ai/auth/transaction-token") == ["acme.authfi.io"]


async def test_a_cold_async_agent_on_a_custom_domain_asks_the_machine_api_host(key: AgentKey, clock: FakeClock) -> None:
    platform = custom_domain(key)
    platform.route(
        "POST",
        "/mcp/v1/cloud/credentials",
        json_response({"data": {"credential_id": "c", "provider": "aws"}}, 201),
    )
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as aaf:
        agent = aaf.agent(AGENT, key)
        await agent.gateway().cloud_credentials("prod", "admin")
        await agent.transaction_token("refund")
    assert hosts(platform, "POST", "/mcp/v1/cloud/credentials") == ["acme.authfi.io"]
    assert hosts(platform, "POST", "/apis/regional/v1/ai/auth/transaction-token") == ["acme.authfi.io"]


# --- one gateway per agent -------------------------------------------------------------------------------


def test_an_agent_has_one_gateway_and_asking_for_it_leaks_nothing(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    assert agent.gateway() is gateway
    project = agent.project
    before = len(project._closables)  # pyright: ignore[reportPrivateUsage]
    for _ in range(2000):
        agent.gateway().http_client()
    assert len(project._closables) == before  # pyright: ignore[reportPrivateUsage]
    assert agent.gateway().http_client() is gateway.http_client()


def test_the_projects_list_of_things_to_close_drops_what_was_collected(agent: AgentAuth) -> None:
    from authfi.gateway import Gateway

    project = agent.project
    for _ in range(50):
        Gateway(project, agent)
    gc.collect()
    Gateway(project, agent)
    assert len(project._closables) < 10  # pyright: ignore[reportPrivateUsage]


async def test_an_async_agent_has_one_gateway(key: AgentKey, platform: AgentPlatform, clock: FakeClock) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as aaf:
        agent = aaf.agent(AGENT, key)
        assert agent.gateway() is agent.gateway()
