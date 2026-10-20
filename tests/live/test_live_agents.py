"""Agents and the gateway against a real project, as a probe administrator (see ``console.py``).

Registers one agent with a key the SDK makes, and removes it, and the delegation it grants.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from authfi import AgentKey, APIError, AsyncAuthFI, AuthFI, ConsentDenied, NotSupportedError, OAuthError, Secret
from authfi.agents import AgentAuth
from tests.live.console import ISSUER, RUN, Cleanups, console_token, eventually, gql, needs_console

pytestmark = [pytest.mark.live, needs_console]


def _claims(token: str) -> dict[str, Any]:
    payload = token.split(".")[1]
    return dict(json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))))


@pytest.fixture(scope="module")
def me() -> dict[str, str]:
    return dict(gql("{ me { id email } }")["me"])


@pytest.fixture(scope="module")
def cleanups() -> Iterator[Cleanups]:
    steps = Cleanups()
    yield steps
    steps.run()


@pytest.fixture(scope="module")
def af() -> Iterator[AuthFI]:
    with AuthFI(ISSUER, user_agent_suffix=f"sdk-live/{RUN}") as project:
        yield project


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid=f"sdk-live-{RUN}")


@pytest.fixture(scope="module")
def agent(af: AuthFI, key: AgentKey, me: dict[str, str], cleanups: Cleanups) -> AgentAuth:
    registered = gql(
        "mutation($i:RegisterAgentInput!){ registerAgent(input:$i){ id } }",
        {
            "i": {
                "name": f"sdk-live-{RUN}",
                "type": "delegated",
                "publicJwk": json.dumps(key.public_jwk()),
                "jwkKid": key.kid,
                "ownerUserId": me["id"],
            }
        },
    )["registerAgent"]["id"]
    cleanups.add("agent", lambda: gql("mutation($id:ID!){ deleteAgent(id:$id) }", {"id": registered}))
    return af.agent(registered, key)


def test_the_agent_mints_its_own_token_with_its_key(af: AuthFI, agent: AgentAuth) -> None:
    token = agent.token()
    claims = _claims(token)
    assert claims["iss"] == ISSUER
    assert claims["sub"] == agent.agent_id
    assert isinstance(claims.get("tenant_id"), str)
    assert agent.token() == token  # cached
    header = json.loads(base64.urlsafe_b64decode(token.split(".")[0] + "=="))
    assert af._key(header["kid"]) is not None  # signed by a key the project publishes
    print(f"  agent token ok: alg={header['alg']} aud={claims['aud']} capabilities={len(agent.capabilities())}")


def test_a_key_the_project_never_registered_is_refused(af: AuthFI, agent: AgentAuth, key: AgentKey) -> None:
    impostor = af.agent(agent.agent_id, AgentKey.generate(kid=key.kid))
    with pytest.raises(OAuthError) as caught:
        impostor.token()
    assert caught.value.code == "invalid_client"


def test_the_gateway_answers_an_ungranted_call_as_a_refusal(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    client = gateway.http_client()

    def rpc(body: dict[str, Any]) -> dict[str, Any]:
        response = client.post(gateway.mcp_url, json=body, headers={"Accept": "application/json, text/event-stream"})
        return dict(response.json())

    tools = rpc({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}).get("result", {}).get("tools", [])
    print(f"  tools/list: {len(tools)} tools; scopes {[gateway.scope_of(t) for t in tools[:3]]}")
    if not tools:
        return
    answer = rpc(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tools[0]["name"], "arguments": {}}}
    )
    outcome = gateway.read_result(answer.get("result"))
    print(f"  tools/call {tools[0]['name']}: {outcome.kind} ({outcome.error}, {outcome.code})")
    assert outcome.kind != "allowed"


def test_a_cloud_credential_for_an_ungranted_role_is_refused_with_a_code(agent: AgentAuth) -> None:
    try:
        outcome = agent.gateway().cloud_credentials(f"sdk-live-{RUN}", "none")
    except APIError as error:
        print(f"  cloud credentials: APIError {error.code} ({error.status})")
        return
    assert outcome.kind == "approval_required"


def test_token_exchange_needs_a_delegation(agent: AgentAuth, cleanups: Cleanups) -> None:
    person_token = Secret(console_token())
    with pytest.raises(OAuthError) as caught:
        agent.exchange(person_token)
    assert caught.value.code == "access_denied"
    sub = str(_claims(person_token.reveal())["sub"])
    person = sub if sub.startswith("user:") else "user:" + sub
    gql("mutation($a:ID!,$o:String!){ grantXaaDelegation(agentId:$a, object:$o) }", {"a": agent.agent_id, "o": person})
    cleanups.add(
        "delegation",
        lambda: gql(
            "mutation($a:ID!,$o:String!){ revokeXaaDelegation(agentId:$a, object:$o) }",
            {"a": agent.agent_id, "o": person},
        ),
    )

    def attempt() -> Any:
        try:
            return agent.exchange(person_token)
        except OAuthError as error:
            if error.code == "access_denied":
                return None
            raise

    delegated = eventually(attempt, tries=5)
    assert delegated is not None
    assert delegated.via == "exchange"
    assert str(delegated.actor) == f"agent:{agent.agent_id}"
    assert str(delegated.subject) == person


def _decide(agent: AgentAuth, binding_message: str, decision: str) -> bool:
    def find() -> Any:
        requests = gql("{ myAgentRequests { authReqId agentId bindingMessage } }")["myAgentRequests"]
        return next(
            (r for r in requests if r["agentId"] == agent.agent_id and r["bindingMessage"] == binding_message), None
        )

    mine = eventually(find)
    if mine is None:
        return False
    gql(
        "mutation($id:ID!,$d:ApprovalDecision!){ decideMyAgentRequest(authReqId:$id, decision:$d){ status } }",
        {"id": mine["authReqId"], "d": decision},
    )
    return True


def test_ciba_the_person_approves_on_their_own_channel(agent: AgentAuth, me: dict[str, str]) -> None:
    binding = f"SDK live {RUN}: let the test agent act for you"
    request = agent.request_user_token(me["email"], binding_message=binding, expires_in=300)
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(agent.await_user_token, request, timeout=120)
        assert _decide(agent, binding, "APPROVE")
        token = waiting.result(timeout=150)
    assert token.via == "ciba"
    assert str(token.actor) == f"agent:{agent.agent_id}"
    assert token.subject.kind == "user"


def test_ciba_a_person_who_says_no_is_consent_denied(agent: AgentAuth, me: dict[str, str]) -> None:
    binding = f"SDK live {RUN}: deny this one"
    request = agent.request_user_token(me["email"], binding_message=binding, expires_in=300)
    with ThreadPoolExecutor(1) as pool:
        waiting = pool.submit(agent.await_user_token, request, timeout=120)
        assert _decide(agent, binding, "DENY")
        with pytest.raises(ConsentDenied):
            waiting.result(timeout=150)


def test_an_approval_opens_and_its_status_is_typed(agent: AgentAuth) -> None:
    approval = agent.request_approval(f"sdk-live-{RUN}.check", context={"run": RUN}, expires_in=60)
    assert approval.status == "pending"
    try:
        status = agent.approval(approval.approval_id).status
    except NotSupportedError as error:
        status = f"NotSupportedError {error.feature}"
    print(f"  approval {approval.status}; status read: {status}")


def test_the_action_gate_answers_with_a_typed_decision(agent: AgentAuth) -> None:
    try:
        decision = agent.authorize_action(
            agent.transaction_token(f"sdk-live-{RUN}.read", resource="sdk-live", context={"n": 1})
        )
    except APIError as error:
        print(f"  action gate: APIError {error.code}")
        return
    print(f"  action gate: {decision.decision} ({decision.reason})")
    assert decision.decision in ("allow", "deny", "approval_required")


async def test_the_asyncio_agent_mints_too(key: AgentKey, agent: AgentAuth) -> None:
    async with AsyncAuthFI(ISSUER) as af:
        token = await af.agent(agent.agent_id, key).token()
    assert _claims(token)["sub"] == agent.agent_id
