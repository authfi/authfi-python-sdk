"""authfi.langgraph with the platform's agent security on: DPoP on every call, tasks, and a held cloud request.

Real LangGraph, the real MCP client and the real Anthropic client, against the agent-security fake gateway.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from authfi import AgentKey, AsyncAuthFI
from authfi.langgraph import LangGraphIntegration, ask
from tests.fake_agent_gateway import OUTSIDE, AgentGateway
from tests.fake_gateway import ALLOWED
from tests.unit.test_agent_capabilities import person

AGENT = "agt_lgsec1"
CONFIG: Any = {"configurable": {"thread_id": "t1"}}


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def gw() -> Iterator[AgentGateway]:
    with AgentGateway(nonce=True) as gateway:
        yield gateway


@pytest.fixture
async def project(gw: AgentGateway) -> AsyncIterator[AsyncAuthFI]:
    async with AsyncAuthFI(gw.issuer) as af:
        yield af


def _use(name: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": name, "input": {"repoName": "pallets/flask", **args}}]


async def _run(langgraph: LangGraphIntegration, gw: AgentGateway, script: list[list[dict[str, Any]]]) -> dict[str, Any]:
    tools = await langgraph.tools()
    graph: Any = langgraph.agent_graph(
        langgraph.chat_model("claude-haiku-5-5", max_tokens=64), tools, checkpointer=InMemorySaver()
    )
    gw.model_script += script
    state: dict[str, Any] = await graph.ainvoke(ask("Summarise the wiki"), CONFIG)
    return state


async def test_the_model_and_the_tools_prove_the_key_on_every_call(
    gw: AgentGateway, project: AsyncAuthFI, key: AgentKey
) -> None:
    langgraph = LangGraphIntegration(project.agent(AGENT, key))
    await langgraph.gateway.source.token()  # the nonce is learned at mint
    gw._nonce = "nonce-rotated"  # the gateway rotates it between calls: one challenge, answered once

    state = await _run(langgraph, gw, [_use(ALLOWED), [{"type": "text", "text": "Done."}]])

    assert state["messages"][-1].content == "Done."
    assert gw.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    gateway_calls = [r for r in gw.requests if r["path"] == "/mcp" or r["path"].endswith("/v1/messages")]
    assert gateway_calls and all(r["headers"]["authorization"].startswith("DPoP ") for r in gateway_calls)
    assert all("dpop" in r["headers"] for r in gateway_calls)
    assert gw.nonce_challenges >= 2  # the mint, and the rotation
    assert gw.refused_proofs == []  # every proof the SDK sent after the challenges was good
    proofs = [p for p in gw.proofs if p["path"] in {"/mcp", "/mcp/v1/messages"}]
    assert {p["path"] for p in proofs} == {"/mcp", "/mcp/v1/messages"}  # the tools and the model are both proved
    assert len({p["claims"]["jti"] for p in proofs}) == len(proofs)  # no proof is used twice


async def test_given_a_task_a_call_outside_it_reaches_the_model_with_what_it_can_still_do(
    gw: AgentGateway, project: AsyncAuthFI, key: AgentKey
) -> None:
    agent = project.agent(AGENT, key)
    gw.model_script += [_use(OUTSIDE), [{"type": "text", "text": "I will stop here."}]]

    async with await agent.task("support.refund", subject_token=person(), params={"order": "4411"}) as task:
        langgraph = LangGraphIntegration(task)  # the task is the credential: one line
        tools = await langgraph.tools()
        assert OUTSIDE in {t.name for t in tools}  # the gateway lists it; the task is what refuses it
        graph: Any = langgraph.agent_graph(
            langgraph.chat_model("claude-haiku-5-5", max_tokens=64), tools, checkpointer=InMemorySaver()
        )
        state = await graph.ainvoke(ask("Ask the wiki a question"), CONFIG)

    (refusal,) = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert refusal.status == "error"
    assert "EMDT005" in str(refusal.content) and "Use a tool the task allows" in str(refusal.content)
    model_read = json.dumps([r["json"] for r in gw.model_requests()])
    assert "Use a tool the task allows" in model_read and "new task" not in model_read.lower()
    # The tool calls and the model's calls all carried the task's credential, never the agent's own.
    after = [r for r in gw.requests if r["path"] in {"/mcp", "/mcp/v1/messages"}]
    carried = {r["headers"]["authorization"].split(" ", 1)[1] for r in after}
    assert {gw.issued[t]["kind"] for t in carried} == {"task"}
    assert task.mandate_id in {gw.issued[t].get("mandate_id") for t in carried}


async def test_a_held_cloud_request_surfaces_as_any_held_tool_call_does(
    gw: AgentGateway, project: AsyncAuthFI, key: AgentKey
) -> None:
    """The gateway's own ``cloud__credentials`` tool is an MCP tool: its hold is the adapter's hold."""
    gw.add_cloud_tool()
    langgraph = LangGraphIntegration(project.agent(AGENT, key))
    tools = await langgraph.tools()
    assert "cloud__credentials" in {t.name for t in tools}
    graph: Any = langgraph.agent_graph(
        langgraph.chat_model("claude-haiku-5-5", max_tokens=64), tools, checkpointer=InMemorySaver()
    )
    gw.model_script += [
        [{"type": "tool_use", "name": "cloud__credentials", "input": {"account": "prod", "role": "admin"}}],
        [{"type": "text", "text": "Deployed."}],
    ]

    state = await graph.ainvoke(ask("Deploy"), CONFIG)

    (waiting,) = state["__interrupt__"]
    (approval,) = waiting.value["authfi_approvals"]
    assert approval["tool"] == "cloud__credentials" and approval["args"] == {"account": "prod", "role": "admin"}
    assert "approval_id" not in json.dumps([r["json"] for r in gw.model_requests()])  # the id never reaches the model

    gw.decide(approval["approval_id"], "approved")
    state = await graph.ainvoke(Command(resume=True), CONFIG)

    assert state["messages"][-1].content == "Deployed."
    (issued,) = [m for m in state["messages"] if isinstance(m, ToolMessage) and m.status == "success"]
    assert "ASIAFAKE" in str(issued.content)  # the gateway's tool result is the tool's answer: the model reads it
