"""authfi.langgraph: real LangGraph and the real MCP client, against a real-HTTP stand-in for the gateway."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableConfig  # module level: a tool's annotations are resolved by name
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
        assert headers["x-request-id"].startswith(f"{langgraph.run_id}.")
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


async def test_export_steps_is_the_langchain_integrations_and_the_run_is_shared(
    fake: FakeGateway, langgraph: LangGraphIntegration
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    import langsmith
    from opentelemetry.sdk.trace import TracerProvider

    assert langgraph.run_id == langgraph.langchain.run_id
    assert langgraph.run_id.startswith("run-")
    provider = TracerProvider()
    try:
        assert langgraph.export_steps(tracer_provider=provider) is provider
    finally:
        langsmith.configure(client=None, enabled=None)
        provider.shutdown()


# --- every gateway call names the step that made it -------------------------------------------------------


def _step_of(headers: dict[str, str]) -> tuple[str, str, str]:
    """``(trace id, span id, flags)`` of a request's ``traceparent``."""
    _, trace_id, span_id, flags = headers["traceparent"].split("-")
    return trace_id, span_id, flags


async def test_a_model_call_and_a_tool_call_name_the_run_that_made_them(
    fake: FakeGateway, langgraph: LangGraphIntegration
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    import asyncio

    import langsmith
    from langsmith import run_trees
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    try:
        langgraph.export_steps(tracer_provider=provider)
        graph = await _graph(langgraph)
        fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Done."}]]

        await graph.ainvoke(ask("Summarise the wiki"), CONFIG)
        await asyncio.to_thread(run_trees.get_cached_client().flush)  # LangSmith makes the spans from a thread
        await asyncio.to_thread(provider.force_flush)

        exported = {(s["trace_id"], s["span_id"]): s["name"] for s in fake.exported_spans()}
        models = [_step_of(r["headers"]) for r in fake.model_requests()]
        calls = [_step_of(headers) for headers, _ in fake.tool_requests()]
        assert len(models) == 2 and len(calls) == 1
        for trace_id, span_id, flags in models:
            # the chat model's own run, not the graph node that LangChain leaves current
            assert exported[(trace_id, span_id)] == "AuthFIChatAnthropic" and flags == "01"
        assert models[0][1] != models[1][1]
        assert exported[calls[0][:2]] == ALLOWED and calls[0][2] == "01"
    finally:
        langsmith.configure(client=None, enabled=None)
        provider.shutdown()


async def test_a_streamed_model_call_names_its_run_too(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    import asyncio

    import langsmith
    from langsmith import run_trees
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    try:
        langgraph.export_steps(tracer_provider=provider)
        model = langgraph.chat_model("claude-haiku-5-5", max_tokens=64)
        fake.model_script += [[{"type": "text", "text": "Done."}]]

        chunks = [chunk async for chunk in model.astream("hi")]
        await asyncio.to_thread(run_trees.get_cached_client().flush)
        await asyncio.to_thread(provider.force_flush)

        assert chunks
        trace_id, span_id, flags = _step_of(fake.model_requests()[0]["headers"])
        exported = {(s["trace_id"], s["span_id"]): s["name"] for s in fake.exported_spans()}
        assert exported[(trace_id, span_id)] == "AuthFIChatAnthropic" and flags == "01"
    finally:
        langsmith.configure(client=None, enabled=None)
        provider.shutdown()


# --- the agent behind a step is the graph above it ---------------------------------------------------------


async def _export(fake: FakeGateway, langgraph: LangGraphIntegration, run: Any) -> list[dict[str, Any]]:
    """Export steps, call ``run()``, and return the spans AuthFI received."""
    import asyncio

    import langsmith
    from langsmith import run_trees
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    try:
        langgraph.export_steps(tracer_provider=provider)
        await run()
        await asyncio.to_thread(run_trees.get_cached_client().flush)  # LangSmith makes the spans from a thread
        await asyncio.to_thread(provider.force_flush)
        return fake.exported_spans()
    finally:
        langsmith.configure(client=None, enabled=None)
        provider.shutdown()


def _agent_above(spans: list[dict[str, Any]], step: dict[str, Any]) -> str | None:
    """The one rule: the nearest ``gen_ai.agent.name`` at or above ``step``."""
    by_id = {(s["trace_id"], s["span_id"]): s for s in spans}
    node: dict[str, Any] | None = step
    while node is not None:
        name = node["attributes"].get("gen_ai.agent.name")
        if name:
            return str(name)
        node = by_id.get((node["trace_id"], node["parent_span_id"]))
    return None


def _step_for(spans: list[dict[str, Any]], headers: dict[str, str]) -> dict[str, Any]:
    trace_id, span_id, _ = _step_of(headers)
    (found,) = [s for s in spans if (s["trace_id"], s["span_id"]) == (trace_id, span_id)]
    return found


async def _supervisor(langgraph: LangGraphIntegration, sub_name: str | None, parent_name: str | None) -> Any:
    """A parent graph whose only tool runs a second graph, which holds the gateway's tools."""
    from langchain_core.tools import StructuredTool

    model = langgraph.chat_model("claude-haiku-5-5", max_tokens=64)
    researcher: Any = langgraph.agent_graph(model, await langgraph.tools(), name=sub_name)

    async def ask_wiki_researcher(request: str, config: RunnableConfig) -> str:
        """Hand a request to the wiki researcher."""
        out = await researcher.ainvoke(ask(request), config)
        return str(out["messages"][-1].content)

    delegate = StructuredTool.from_function(coroutine=ask_wiki_researcher, name="ask_wiki_researcher", description="x")
    return langgraph.agent_graph(model, [delegate], name=parent_name)


async def test_the_agent_behind_a_call_is_the_graph_whose_run_made_it(
    fake: FakeGateway, langgraph: LangGraphIntegration
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    graph = await _supervisor(langgraph, "wiki_researcher", "support")
    fake.model_script += [
        _use("ask_wiki_researcher", request="r"),  # support asks its sub-agent
        _use(ALLOWED),  # the sub-agent calls the gateway tool
        [{"type": "text", "text": "Sections."}],
        [{"type": "text", "text": "Done."}],
    ]

    async def run() -> None:
        await graph.ainvoke(ask("Read the wiki"), CONFIG)

    spans = await _export(fake, langgraph, run)

    (tool_call,) = fake.tool_requests()
    assert _agent_above(spans, _step_for(spans, tool_call[0])) == "wiki_researcher"
    models = [_agent_above(spans, _step_for(spans, r["headers"])) for r in fake.model_requests()]
    assert models == ["support", "wiki_researcher", "wiki_researcher", "support"]
    graphs = [s for s in spans if s["attributes"].get("gen_ai.agent.name")]
    assert {(s["name"], s["attributes"]["gen_ai.operation.name"]) for s in graphs} == {
        ("support", "invoke_agent"),
        ("wiki_researcher", "invoke_agent"),
    }


async def test_a_graph_given_no_name_is_still_an_agent_and_has_langgraphs_default(
    fake: FakeGateway, langgraph: LangGraphIntegration
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    graph = await _supervisor(langgraph, None, None)
    fake.model_script += [_use("ask_wiki_researcher", request="r"), _use(ALLOWED)]

    async def run() -> None:
        await graph.ainvoke(ask("Read the wiki"), CONFIG)

    spans = await _export(fake, langgraph, run)

    (tool_call,) = fake.tool_requests()
    assert _agent_above(spans, _step_for(spans, tool_call[0])) == "LangGraph"
    assert [s["name"] for s in spans if s["attributes"].get("gen_ai.agent.name")] == ["LangGraph", "LangGraph"]


async def test_only_a_graphs_run_is_an_agent_not_its_nodes_tools_or_models(
    fake: FakeGateway, langgraph: LangGraphIntegration
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    graph = await _graph(langgraph, name="support")
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Done."}]]

    async def run() -> None:
        await graph.ainvoke(ask("Read the wiki"), CONFIG)

    spans = await _export(fake, langgraph, run)

    assert [s["name"] for s in spans if s["attributes"].get("gen_ai.agent.name")] == ["support"]
    assert {s["name"] for s in spans} >= {"agent", "tools", ALLOWED, "AuthFIChatAnthropic"}


async def test_a_graph_that_is_a_node_of_another_is_one_agent_not_two(
    fake: FakeGateway, langgraph: LangGraphIntegration
) -> None:
    """A subgraph added as a node runs as the node and as its own graph; only the graph is the agent."""
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph  # pyright: ignore[reportMissingTypeStubs]

    await langgraph.tools()  # the credential has been used, so the exporter can find its event loop

    async def do_work(state: MessagesState) -> dict[str, Any]:
        return {"messages": [AIMessage("worked")]}

    inner: Any = StateGraph(MessagesState)
    inner.add_node("work", do_work)
    inner.add_edge(START, "work")
    inner.add_edge("work", END)
    outer: Any = StateGraph(MessagesState)
    outer.add_node("researcher", inner.compile(name="researcher"))  # the node and its graph share a name
    outer.add_edge(START, "researcher")
    outer.add_edge("researcher", END)
    graph: Any = outer.compile(name="support")

    async def run() -> None:
        given: Any = {"messages": [("user", "hi")]}
        await graph.ainvoke(given)

    spans = await _export(fake, langgraph, run)

    named = [s for s in spans if s["name"] == "researcher"]
    assert len(named) == 2
    (agent,) = [s for s in named if s["attributes"].get("gen_ai.agent.name")]
    children = {s["name"] for s in spans if s["parent_span_id"] == agent["span_id"]}
    assert children == {"work"}  # the graph run is the one the inner node hangs under
    (step,) = [s for s in spans if s["name"] == "work"]
    assert _agent_above(spans, step) == "researcher"


async def test_without_export_steps_no_run_is_named(fake: FakeGateway, langgraph: LangGraphIntegration) -> None:
    from authfi._protocol.ids import step_traceparent

    graph = await _graph(langgraph)
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Done."}]]

    await graph.ainvoke(ask("Summarise the wiki"), CONFIG)

    assert step_traceparent(langgraph.source) is None  # nothing is exported, so there is no step to name
    # (what the call carries then is the active OpenTelemetry span, if the application has one, else a fresh value)
    assert all("traceparent" in r["headers"] for r in fake.model_requests())


def test_a_langsmith_whose_bridge_moved_is_refused() -> None:
    from types import SimpleNamespace

    from authfi import IntegrationError
    from authfi._langsmith_steps import use_run_ids

    with pytest.raises(IntegrationError, match="langsmith"):
        use_run_ids(SimpleNamespace(otel_exporter=None))


async def test_the_models_client_does_not_set_a_traceparent_of_its_own(langgraph: LangGraphIntegration) -> None:
    """The Anthropic SDK's own propagation replaces any traceparent with that of its parentless span."""
    import anthropic

    model = langgraph.chat_model("claude-haiku-5-5")
    client = model._async_client  # pyright: ignore[reportPrivateUsage]
    if not hasattr(client, "open_telemetry"):
        pytest.skip("this anthropic does not trace")
    assert client.open_telemetry["propagation"] is False
    assert isinstance(client, anthropic.AsyncAnthropic)
