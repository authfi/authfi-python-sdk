"""Sub-agents: an agent of its own, made by a running agent, holding no more than its parent."""

from __future__ import annotations

import json
import pickle
from collections.abc import Iterator
from typing import Any

import httpx
import jwt
import pytest

from authfi import AgentKey, APIError, AsyncAuthFI, AuthFI, ConfigurationError
from authfi.agents import AgentAuth, AsyncAgentAuth, AsyncSubAgent, SubAgent
from tests.agent_platform import AGENT, AgentPlatform
from tests.support import FakeClock, json_response

SPAWN = "/apis/regional/v1/ai/agents/subagents"


class Tree:
    """The platform's side of a delegation: who holds what, and the rule that a child never holds more."""

    def __init__(self, platform: AgentPlatform) -> None:
        self.platform = platform
        self.held: dict[str, list[str]] = {AGENT: list(platform.capabilities)}
        self.parent: dict[str, str] = {}
        self.minted = 0
        platform.route("POST", SPAWN, self._spawn)

    def _caller(self, request: httpx.Request) -> str | None:
        raw = request.headers.get("authorization", "").split(" ", 1)[-1]
        try:
            return str(jwt.decode(raw, options={"verify_signature": False}).get("sub"))
        except jwt.PyJWTError:
            return None

    def _chain(self, agent: str) -> list[str]:
        out: list[str] = []
        while agent in self.parent:
            agent = self.parent[agent]
            out.append(agent)
        return out

    def _answer(self, child: str, name: str, status: int) -> httpx.Response:
        self.minted += 1
        chain = self._chain(child)
        token = self.platform.access_token(
            sub=child, client_id=child, parent_agent=self.parent[child], agent_chain=chain
        )
        return json_response(
            {
                "data": {
                    "agent_id": child,
                    "wimse_id": f"spiffe://proj.example/ai/agent/{child}",
                    "name": name,
                    "parent_agent_id": self.parent[child],
                    "root_agent_id": chain[-1],
                    "depth": len(chain),
                    "chain": chain,
                    "capabilities": self.held[child],
                    "scope": " ".join(self.held[child]),
                    "access_token": token,
                    "token_type": "Bearer",
                    "expires_in": 900,
                }
            },
            status=status,
        )

    def _spawn(self, request: httpx.Request) -> httpx.Response:
        caller = self._caller(request)
        if caller is None or caller not in self.held:
            return json_response({"error": {"code": "unauthorized", "message": "no"}}, status=401)
        body = json.loads(request.content)
        missing = [p for p in body["permissions"] if p not in self.held[caller]]
        if missing:
            return json_response(
                {"error": {"code": "ESUBAGENT002", "message": "a sub-agent cannot hold " + " ".join(missing)}},
                status=403,
            )
        child = f"agt_sub{len(self.parent) + 1}"
        self.parent[child] = caller
        self.held[child] = list(body["permissions"])
        self.platform.route("POST", f"{SPAWN}/{child}/token", self.handle_token)
        self.last_body = body
        return self._answer(child, body["name"], 201)

    def handle_token(self, request: httpx.Request) -> httpx.Response:
        caller = self._caller(request)
        child = request.url.path.split("/")[-2]
        if self.parent.get(child) != caller:
            return json_response(
                {"error": {"code": "ESUBAGENT006", "message": "no such sub-agent of yours"}}, status=403
            )
        # What the child holds now is what its parent still holds.
        self.held[child] = [p for p in self.held[child] if p in self.held[caller]]
        return self._answer(child, child, 200)


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate("ES256", kid="k1")


@pytest.fixture
def platform(key: AgentKey) -> AgentPlatform:
    return AgentPlatform(agent_jwk=key.public_jwk(), capabilities=["mcp:crm.read", "mcp:crm.write", "send_email"])


@pytest.fixture
def tree(platform: AgentPlatform) -> Tree:
    return Tree(platform)


@pytest.fixture
def af(platform: AgentPlatform, tree: Tree, clock: FakeClock) -> Iterator[AuthFI]:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        yield project


@pytest.fixture
def agent(af: AuthFI, key: AgentKey) -> AgentAuth:
    return af.agent(AGENT, key)


def claims_of(raw: str) -> dict[str, Any]:
    return jwt.decode(raw, options={"verify_signature": False})


def test_a_sub_agent_has_an_identity_of_its_own(agent: AgentAuth, tree: Tree) -> None:
    sub = agent.sub_agent("crm lookup", permissions=["mcp:crm.read"], framework="langgraph", run_id="run_1")
    assert isinstance(sub, SubAgent)
    assert sub.agent_id.startswith("agt_") and sub.agent_id != AGENT
    assert sub.parent_agent_id == AGENT and sub.root_agent_id == AGENT and sub.depth == 1
    assert sub.chain == (AGENT,)
    assert sub.permissions == ("mcp:crm.read",)
    assert sub.wimse_id.endswith(sub.agent_id)
    assert sub.name == "crm lookup"
    # The credential is the sub-agent's: its calls name it, not the parent.
    assert claims_of(sub.token())["sub"] == sub.agent_id
    assert tree.last_body == {
        "name": "crm lookup",
        "permissions": ["mcp:crm.read"],
        "run_id": "run_1",
        "framework": "langgraph",
    }


def test_a_request_the_parent_cannot_cover_is_refused_whole(agent: AgentAuth, tree: Tree) -> None:
    with pytest.raises(APIError) as caught:
        agent.sub_agent("greedy", permissions=["mcp:crm.read", "mcp:billing.refund"])
    assert caught.value.code == "ESUBAGENT002"
    assert "mcp:billing.refund" in str(caught.value)
    assert tree.parent == {}, "a refused request made a sub-agent"


def test_three_levels_each_inside_the_one_above(agent: AgentAuth) -> None:
    one = agent.sub_agent("one", permissions=["mcp:crm.read", "mcp:crm.write"])
    two = one.sub_agent("two", permissions=["mcp:crm.read"])
    three = two.sub_agent("three", permissions=["mcp:crm.read"])
    assert (one.depth, two.depth, three.depth) == (1, 2, 3)
    assert three.chain == (two.agent_id, one.agent_id, AGENT)
    assert three.root_agent_id == AGENT and three.parent_agent_id == two.agent_id
    assert claims_of(three.token())["agent_chain"] == [two.agent_id, one.agent_id, AGENT]
    with pytest.raises(APIError) as caught:
        two.sub_agent("wider", permissions=["mcp:crm.write"])  # level one held it; level two never did
    assert caught.value.code == "ESUBAGENT002"


def test_a_token_is_renewed_from_what_the_parent_holds_now(agent: AgentAuth, tree: Tree, clock: FakeClock) -> None:
    sub = agent.sub_agent("kid", permissions=["mcp:crm.read", "mcp:crm.write"])
    first = sub.token()
    tree.held[AGENT].remove("mcp:crm.write")
    assert sub.token() == first and tree.minted == 1, "a token still in date was renewed early"
    clock.advance(1000)
    assert sub.token() and tree.minted == 2, "an ended token was not renewed"
    assert sub.permissions == ("mcp:crm.read",), "a grant the parent lost is still on the sub-agent"


def test_the_headers_carry_the_sub_agents_token_to_authfi_only(agent: AgentAuth, platform: AgentPlatform) -> None:
    sub = agent.sub_agent("kid", permissions=["mcp:crm.read"])
    h = sub.headers("POST", platform.issuer + "/mcp")
    assert claims_of(h["Authorization"].split(" ", 1)[1])["sub"] == sub.agent_id
    with pytest.raises(ConfigurationError):
        sub.headers("GET", "https://elsewhere.example/api")


def test_a_closed_sub_agent_forgets_its_credential(agent: AgentAuth) -> None:
    with agent.sub_agent("kid", permissions=["mcp:crm.read"]) as sub:
        assert sub.token()
    with pytest.raises(ConfigurationError):
        sub.token()
    with pytest.raises(TypeError):
        pickle.dumps(sub)


def test_a_request_with_no_permission_or_a_space_is_refused_before_it_is_sent(
    agent: AgentAuth, platform: AgentPlatform
) -> None:
    for bad in ([], ["mcp:crm read"], ""):
        with pytest.raises(ConfigurationError):
            agent.sub_agent("x", permissions=bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError):
        agent.sub_agent("  ", permissions=["mcp:crm.read"])
    assert platform.calls("POST", SPAWN) == []


async def test_the_asyncio_agent_makes_the_same_sub_agent(
    platform: AgentPlatform, tree: Tree, key: AgentKey, clock: FakeClock
) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as project:
        agent: AsyncAgentAuth = project.agent(AGENT, key)
        sub = await agent.sub_agent("crm", permissions=["mcp:crm.read"])
        assert isinstance(sub, AsyncSubAgent)
        assert sub.parent_agent_id == AGENT and sub.depth == 1
        assert claims_of(await sub.token())["sub"] == sub.agent_id
        deeper = await sub.sub_agent("deeper", permissions=["mcp:crm.read"])
        assert deeper.chain == (sub.agent_id, AGENT)
        with pytest.raises(APIError) as caught:
            await sub.sub_agent("wider", permissions=["mcp:crm.write"])
        assert caught.value.code == "ESUBAGENT002"
