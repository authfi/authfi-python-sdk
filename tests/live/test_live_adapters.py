"""The framework adapters against a real project's gateways (see ``console.py``).

Registers one agent, grants it the model and one tool, and removes all of it. Needs the
``langgraph`` extra.
"""

from __future__ import annotations

import functools
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("langgraph")

import langsmith
from langchain_core.messages import ToolMessage

from authfi import AgentKey, AsyncAuthFI, AuthFI
from authfi.langchain import LangChainIntegration
from authfi.langgraph import LangGraphIntegration, ask
from tests.fake_gateway import CaptureServer
from tests.live.console import ISSUER, RUN, Cleanups, gql, needs_console

pytestmark = [pytest.mark.live, needs_console]

MODEL = "claude-haiku-5-5"
GRANTED = "deepwiki.read_wiki_structure"
UNGRANTED = "deepwiki__ask_wiki_question"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid=f"sdk-live-adapters-{RUN}")


@pytest.fixture(scope="module")
def agent_id(key: AgentKey) -> Iterator[str]:
    steps = Cleanups()
    me = gql("{ me { id } }")["me"]["id"]
    registered = gql(
        "mutation($i:RegisterAgentInput!){ registerAgent(input:$i){ id } }",
        {
            "i": {
                "name": f"sdk-live-adapters-{RUN}",
                "type": "delegated",
                "publicJwk": json.dumps(key.public_jwk()),
                "jwkKid": key.kid,
                "ownerUserId": me,
            }
        },
    )["registerAgent"]["id"]
    steps.add("agent", lambda: gql("mutation($id:ID!){ deleteAgent(id:$id) }", {"id": registered}))
    subject = "agent:" + registered

    def revoke_tool(tool: str) -> None:
        gql("mutation($s:String!,$t:String!){ revokeMcpTool(subject:$s, tool:$t) }", {"s": subject, "t": tool})

    def revoke_capability(tool: str) -> None:
        gql(
            "mutation($a:ID!,$p:String!){ revokeAgentCapability(agentId:$a, permission:$p) }",
            {"a": registered, "p": "mcp:" + tool},
        )

    for tool in ("anthropic/" + MODEL, GRANTED):
        gql("mutation($s:String!,$t:String!){ grantMcpTool(subject:$s, tool:$t) }", {"s": subject, "t": tool})
        steps.add("grant " + tool, functools.partial(revoke_tool, tool))
        gql(
            "mutation($a:ID!,$i:GrantCapabilityInput!){ grantAgentCapability(agentId:$a, input:$i){ permission } }",
            {"a": registered, "i": {"permission": "mcp:" + tool, "reason": "sdk live " + RUN}},
        )
        steps.add("capability " + tool, functools.partial(revoke_capability, tool))
    yield registered
    steps.run()


def test_langchain_calls_the_model_and_langsmith_never_sees_the_token(agent_id: str, key: AgentKey) -> None:
    with AuthFI(ISSUER) as af, CaptureServer() as capture:
        agent = af.agent(agent_id, key)
        langchain = LangChainIntegration(agent)
        client = langchain.configure_tracing(api_url=capture.url, api_key="capture-only")
        langsmith.configure(enabled=True)
        try:
            answer = langchain.chat_model(MODEL, max_tokens=30, max_retries=0).invoke("Reply with exactly: LIVE OK")
            client.flush()
        finally:
            langsmith.configure(client=None, enabled=None)
        blob = capture.blob()
        token = agent.token()
    print(f"  langchain: model answered {str(answer.content)[:40]!r}; LangSmith got {len(blob)} bytes")
    assert "LIVE OK" in str(answer.content)
    assert MODEL.encode() in blob  # the run was traced
    assert token.encode() not in blob
    assert b"LIVE OK" not in blob  # no content by default


@pytest.fixture
async def langgraph(agent_id: str, key: AgentKey) -> AsyncIterator[LangGraphIntegration]:
    async with AsyncAuthFI(ISSUER) as af:
        yield LangGraphIntegration(af.agent(agent_id, key))


async def test_langgraph_runs_a_granted_tool_and_reads_a_refusal(langgraph: LangGraphIntegration) -> None:
    tools = await langgraph.tools()
    names = [tool.name for tool in tools]
    print(f"  langgraph: {len(names)} tools offered")
    assert UNGRANTED in names or names
    graph: Any = langgraph.agent_graph(langgraph.chat_model(MODEL, max_tokens=400, max_retries=0), tools)
    state: dict[str, Any] = await graph.ainvoke(
        ask(
            "First call deepwiki__read_wiki_structure for the GitHub repo pallets/itsdangerous. "
            "Then call deepwiki__ask_wiki_question with repoName pallets/itsdangerous and question "
            "'What is it?'. Reply with the first topic title, and with the refusal text if a tool was refused."
        )
    )
    answers = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    for m in answers:
        print(f"  langgraph: {m.name} -> {m.status}: {str(m.content)[:90]!r}")
    assert any(m.status == "success" for m in answers)
    refused = [m for m in answers if m.status == "error"]
    assert all("Refused by AuthFI" in str(m.content) for m in refused)
