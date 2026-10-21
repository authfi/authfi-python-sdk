"""authfi.langgraph: real LangGraph and the real MCP client, against a real-HTTP stand-in for the gateway."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from authfi import AgentKey, AsyncAuthFI, AuthFI, ConfigurationError, GatewayTokenRefused
from authfi.gateway import APPROVAL_ARGUMENT, HOLD_MESSAGE
from authfi.langgraph import APPROVAL_NODE, AuthFIState, Hold, LangGraphIntegration, ask, route_after_tools
from tests.fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway

AGENT = "agt_langgraph1"
CONFIG: Any = {"configurable": {"thread_id": "t1"}}


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
async def langgraph(fake: FakeGateway, key: AgentKey) -> AsyncIterator[LangGraphIntegration]:
    async with AsyncAuthFI(fake.issuer) as af:
        yield LangGraphIntegration(af.agent(AGENT, key))


async def _graph(langgraph: LangGraphIntegration, **kwargs: Any) -> Any:
    tools = await langgraph.tools()
    model = langgraph.chat_model("claude-haiku-5-5", max_tokens=64)
    return langgraph.agent_graph(model, tools, checkpointer=InMemorySaver(), **kwargs)


def _use(name: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": name, "input": {"repoName": "pallets/flask", **args}}]


def _tool_messages(state: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def _everything_the_model_read(fake: FakeGateway) -> str:
    return json.dumps([r["json"] for r in fake.model_requests()])


# --- tools ------------------------------------------------------------------------------------------


async def test_tools_keep_the_gateways_names_and_leave_out_never_allowed(langgraph: LangGraphIntegration) -> None:
    names = [tool.name for tool in await langgraph.tools()]
    assert names == [ALLOWED, UNGRANTED, STEP_UP]
    every = [tool.name for tool in await langgraph.tools(include_never_allowed=True)]
    assert NEVER in every
    assert langgraph.client() is langgraph.client()


async def test_an_allowed_call_carries_the_agents_token(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    graph = await _graph(langgraph)
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Done."}]]

    state = await graph.ainvoke(ask("Summarise the wiki"), CONFIG)

    assert state["messages"][-1].content == "Done."
    assert fake.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    for headers, _ in fake.tool_requests():
        assert headers["authorization"].startswith("Bearer tok-minted-")
        assert headers["x-request-id"].startswith("afr-")
    (answer,) = _tool_messages(state)
    assert answer.status == "success"
    assert "answered for pallets/flask" in str(answer.content)


async def test_a_refusal_is_the_tools_error_and_the_run_carries_on(
    fake: FakeGateway, langgraph: LangGraphIntegration
) -> None:
    graph = await _graph(langgraph)
    fake.model_script += [_use(UNGRANTED), [{"type": "text", "text": "I cannot."}]]

    state = await graph.ainvoke(ask("Ask the wiki"), CONFIG)

    (refusal,) = _tool_messages(state)
    assert refusal.status == "error"
    assert "ETOOL003" in str(refusal.content)
    assert "mcp:deepwiki.ask_question" in str(refusal.content)
    assert state["messages"][-1].content == "I cannot."


# --- approvals --------------------------------------------------------------------------------------


async def test_a_held_call_waits_and_runs_once_approved(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    graph = await _graph(langgraph)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Read it."}]]

    state = await graph.ainvoke(ask("Read the wiki"), CONFIG)

    (waiting,) = state["__interrupt__"]
    (approval,) = waiting.value["authfi_approvals"]
    assert approval["tool"] == STEP_UP
    assert approval["args"] == {"repoName": "pallets/flask"}
    assert fake.pending() == [approval["approval_id"]]
    assert _tool_messages(state)[0].content == HOLD_MESSAGE

    fake.decide(approval["approval_id"], "approved")
    state = await graph.ainvoke(Command(resume=True), CONFIG)

    assert "__interrupt__" not in state
    assert state["messages"][-1].content == "Read it."
    (answer,) = _tool_messages(state)  # the placeholder was replaced, not added to
    assert "answered for pallets/flask" in str(answer.content)
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]
    assert approval["approval_id"] not in _everything_the_model_read(fake)
    assert "Waiting for approval" not in _everything_the_model_read(fake)


async def test_a_call_still_waiting_is_held_again(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    graph = await _graph(langgraph)
    fake.model_script += [_use(STEP_UP)]
    state = await graph.ainvoke(ask("Read the wiki"), CONFIG)
    first = state["__interrupt__"][0].value["authfi_approvals"][0]["approval_id"]

    state = await graph.ainvoke(Command(resume=True), CONFIG)

    assert state["__interrupt__"][0].value["authfi_approvals"][0]["approval_id"] == first
    assert state["authfi_holds"][0]["approval_id"] == first
    assert fake.tool_calls == []


async def test_a_denied_call_reaches_the_model_as_refused(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    graph = await _graph(langgraph)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Denied."}]]
    state = await graph.ainvoke(ask("Read the wiki"), CONFIG)
    fake.decide(state["__interrupt__"][0].value["authfi_approvals"][0]["approval_id"], "denied")

    state = await graph.ainvoke(Command(resume=True), CONFIG)

    (refusal,) = _tool_messages(state)
    assert refusal.status == "error"
    assert "approval_denied" in str(refusal.content)
    assert state["messages"][-1].content == "Denied."


async def test_stopping_gives_up_without_sending_again(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    graph = await _graph(langgraph)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Stopped."}]]
    await graph.ainvoke(ask("Read the wiki"), CONFIG)
    sent = len(fake.tool_requests())

    state = await graph.ainvoke(Command(resume={"stop": True}), CONFIG)

    assert len(fake.tool_requests()) == sent
    (stopped,) = _tool_messages(state)
    assert stopped.status == "error"
    assert "Not run" in str(stopped.content)


async def test_an_approval_the_model_writes_is_removed(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    graph = await _graph(langgraph)
    fake.model_script += [_use(STEP_UP, **{APPROVAL_ARGUMENT: "stolen-approval"})]

    state = await graph.ainvoke(ask("Read the wiki"), CONFIG)

    assert "__interrupt__" in state
    for _, params in fake.tool_requests():
        assert APPROVAL_ARGUMENT not in params["arguments"]


async def test_the_interceptor_removes_a_model_approval_and_adds_only_its_own(
    langgraph: LangGraphIntegration,
) -> None:
    from langchain_mcp_adapters.interceptors import MCPToolCallRequest
    from mcp.types import CallToolResult, TextContent

    from authfi import langgraph as module

    seen: list[dict[str, Any]] = []

    async def handler(request: MCPToolCallRequest) -> CallToolResult:
        seen.append(request.args)
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)

    request = MCPToolCallRequest(
        name=STEP_UP, args={"repoName": "x", APPROVAL_ARGUMENT: "stolen"}, server_name="authfi"
    )
    await langgraph.interceptor(request, handler)
    token = module._RESUMING.set(module._Resume("granted", "call-1"))
    try:
        await langgraph.interceptor(request, handler)
    finally:
        module._RESUMING.reset(token)

    assert seen == [{"repoName": "x"}, {"repoName": "x", APPROVAL_ARGUMENT: "granted"}]


async def test_another_interceptors_answer_passes_through(langgraph: LangGraphIntegration) -> None:
    from langchain_mcp_adapters.interceptors import MCPToolCallRequest

    answer = ToolMessage("cached", tool_call_id="c1")

    async def handler(request: MCPToolCallRequest) -> ToolMessage:
        return answer

    request = MCPToolCallRequest(name=ALLOWED, args={}, server_name="authfi")
    assert await langgraph.interceptor(request, handler) is answer


async def test_other_failures_are_not_token_refusals(langgraph: LangGraphIntegration) -> None:
    from langchain_mcp_adapters.interceptors import MCPToolCallRequest

    async def handler(request: MCPToolCallRequest) -> ToolMessage:
        raise RuntimeError("upstream broke")

    with pytest.raises(RuntimeError, match="upstream broke"):
        await langgraph.interceptor(MCPToolCallRequest(name=ALLOWED, args={}, server_name="authfi"), handler)


# --- outside a graph, and failures ------------------------------------------------------------------


async def test_a_tool_called_directly_never_shows_the_hold(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    tools = {tool.name: tool for tool in await langgraph.tools()}

    held = await tools[STEP_UP].ainvoke({"repoName": "x"})
    refused = await tools[UNGRANTED].ainvoke({"repoName": "x"})

    assert HOLD_MESSAGE in str(held)
    assert "Waiting for approval" not in str(held)
    assert "ETOOL003" in str(refused)


async def test_a_refused_token_ends_the_run(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    tools = {tool.name: tool for tool in await langgraph.tools()}
    fake.revoked = True

    with pytest.raises(GatewayTokenRefused):
        await tools[ALLOWED].ainvoke({"repoName": "x"})


async def test_the_resume_needs_the_integrations_tools(langgraph: LangGraphIntegration) -> None:
    hold: Hold = {"approval_id": "a", "tool": "other__tool", "args": {}, "tool_call_id": "c", "message_id": "m"}
    with pytest.raises(KeyError, match=r"LangGraphIntegration\.tools"):
        await langgraph._send_again(hold)


async def test_a_model_only_graph_and_a_prompt(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    graph: Any = langgraph.agent_graph(langgraph.chat_model("claude-haiku-5-5", max_tokens=64), [], prompt="Be brief.")
    state = await graph.ainvoke(ask("Hello"))
    assert state["messages"][-1].content == "Done."
    assert fake.model_requests()[0]["json"]["system"] == "Be brief."


async def test_the_approval_node_with_nothing_held(langgraph: LangGraphIntegration) -> None:
    assert await langgraph.approval_node({"messages": [], "authfi_holds": []}) == {}
    assert route_after_tools({"authfi_holds": []}) == "agent"
    assert route_after_tools({"authfi_holds": [{"approval_id": "a"}]}) == APPROVAL_NODE
    assert set(AuthFIState.__annotations__) == {"messages", "authfi_holds"}


def test_a_blocking_credential_is_refused(fake: FakeGateway, key: AgentKey) -> None:
    with AuthFI(fake.issuer) as af, pytest.raises(ConfigurationError, match="asyncio"):
        LangGraphIntegration(af.agent(AGENT, key))  # type: ignore[arg-type]


def test_tracing_is_langchains(langgraph: LangGraphIntegration) -> None:
    from authfi.langchain import LangChainIntegration

    assert langgraph.configure_tracing is LangChainIntegration.configure_tracing
