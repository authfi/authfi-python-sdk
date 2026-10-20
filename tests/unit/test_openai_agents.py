"""authfi.openai_agents: the SDK's own Runner, LiteLLM and MCP client against a real-HTTP stand-in for the gateway."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("agents")

from agents import Agent, Runner, RunResult
from agents.items import ToolCallOutputItem
from agents.tracing import TracingProcessor
from agents.tracing.spans import Span
from agents.tracing.traces import Trace

from authfi import AgentKey, AsyncAuthFI, AuthFI, ConfigurationError, GatewayTokenRefused
from authfi.gateway import APPROVAL_ARGUMENT, HOLD_MESSAGE
from authfi.openai_agents import AuthFIMCPServer, AuthFITraceProcessor, OpenAIAgentsIntegration
from tests.fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway

AGENT = "agt_openai1"
PROMPT = "Look up the wiki; my email is ada@example.com"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
async def integration(fake: FakeGateway, key: AgentKey) -> AsyncIterator[OpenAIAgentsIntegration]:
    async with AsyncAuthFI(fake.issuer) as af:
        integration = OpenAIAgentsIntegration(af.agent(AGENT, key))
        integration.configure_tracing()
        yield integration


@pytest.fixture
async def tools(integration: OpenAIAgentsIntegration) -> AsyncIterator[AuthFIMCPServer]:
    async with integration.mcp_server() as server:
        yield server


def _agent(integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer | None = None) -> Agent[Any]:
    return Agent(
        name="research",
        model=integration.model("claude-haiku-5-5"),
        mcp_servers=[tools] if tools else [],
    )


def _use(name: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": name, "input": {"repoName": "pallets/flask", **args}}]


def _outputs(result: RunResult) -> list[str]:
    return [json.dumps(item.output, default=str) for item in result.new_items if isinstance(item, ToolCallOutputItem)]


def _everything_the_model_read(fake: FakeGateway) -> str:
    return json.dumps([r["json"] for r in fake.model_requests()])


# --- the model --------------------------------------------------------------------------------------


async def test_the_model_carries_the_agents_token_and_no_openai_key(
    fake: FakeGateway, integration: OpenAIAgentsIntegration
) -> None:
    result = await Runner.run(_agent(integration), PROMPT, run_config=integration.run_config())

    assert result.final_output == "Done."
    (request,) = fake.model_requests()
    assert request["path"] == "/mcp/v1/messages"
    assert request["headers"]["authorization"] == "Bearer tok-minted-1"
    assert request["headers"].get("x-api-key") in (None, "authfi-sets-the-token-per-request")


async def test_an_openai_compatible_model_uses_the_gateways_client(
    fake: FakeGateway, integration: OpenAIAgentsIntegration
) -> None:
    agent = Agent(name="research", model=integration.model("gpt-5-mini", api="openai"))
    result = await Runner.run(agent, PROMPT, run_config=integration.run_config())

    assert result.final_output == "Done."
    chat = [r for r in fake.requests if r["path"].endswith("/v1/chat/completions")]
    assert chat[0]["path"] == "/mcp/v1/chat/completions"
    assert chat[0]["headers"]["authorization"] == "Bearer tok-minted-1"


async def test_a_streamed_run_carries_the_token_too(fake: FakeGateway, integration: OpenAIAgentsIntegration) -> None:
    result = Runner.run_streamed(_agent(integration), PROMPT, run_config=integration.run_config())
    async for _ in result.stream_events():
        pass
    assert result.final_output == "Done."
    assert fake.model_requests()[0]["headers"]["authorization"] == "Bearer tok-minted-1"
    assert fake.model_requests()[0]["json"]["stream"] is True


async def test_a_refused_token_ends_a_streamed_run(fake: FakeGateway, integration: OpenAIAgentsIntegration) -> None:
    fake.revoked = True
    result = Runner.run_streamed(_agent(integration), PROMPT, run_config=integration.run_config())
    with pytest.raises(GatewayTokenRefused):
        async for _ in result.stream_events():
            pass


def test_only_the_gateways_apis(integration: OpenAIAgentsIntegration) -> None:
    with pytest.raises(ConfigurationError, match="gemini"):
        integration.model("x", api="gemini")  # type: ignore[arg-type]


async def test_a_refused_token_ends_the_run(fake: FakeGateway, integration: OpenAIAgentsIntegration) -> None:
    fake.revoked = True
    with pytest.raises(GatewayTokenRefused):
        await Runner.run(_agent(integration), PROMPT, run_config=integration.run_config())


# --- tools ------------------------------------------------------------------------------------------


async def test_tools_keep_the_gateways_names_and_leave_out_never_allowed(tools: AuthFIMCPServer) -> None:
    names = [tool.name for tool in await tools.list_tools()]
    assert names == [ALLOWED, UNGRANTED, STEP_UP]
    assert NEVER not in names


async def test_never_allowed_tools_on_request(integration: OpenAIAgentsIntegration) -> None:
    async with integration.mcp_server(include_never_allowed=True) as server:
        assert NEVER in [tool.name for tool in await server.list_tools()]


async def test_an_allowed_call(fake: FakeGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer) -> None:
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]

    result = await Runner.run(_agent(integration, tools), PROMPT, run_config=integration.run_config())

    assert result.final_output == "Read it."
    assert fake.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    for headers, _ in fake.tool_requests():
        assert headers["authorization"].startswith("Bearer tok-minted-")


async def test_a_refusal_reaches_the_model_naming_why(
    fake: FakeGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer
) -> None:
    fake.model_script += [_use(UNGRANTED), [{"type": "text", "text": "I cannot."}]]

    result = await Runner.run(_agent(integration, tools), PROMPT, run_config=integration.run_config())

    (output,) = _outputs(result)
    assert "ETOOL003" in output
    assert "mcp:deepwiki.ask_question" in output
    assert "An error occurred" not in output
    assert result.final_output == "I cannot."


# --- approvals --------------------------------------------------------------------------------------


async def _paused(fake: FakeGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer, **args: Any) -> Any:
    fake.model_script += [_use(STEP_UP, **args), [{"type": "text", "text": "Finished."}]]
    agent = _agent(integration, tools)
    result = await Runner.run(agent, PROMPT, run_config=integration.run_config())
    assert result.interruptions
    assert fake.tool_requests() == []  # paused before the call
    return agent, result


async def test_a_call_that_may_need_approval_pauses_and_runs_once_approved(
    fake: FakeGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer
) -> None:
    agent, result = await _paused(fake, integration, tools)
    state = result.to_state()

    (waiting,) = await tools.resolve(state, result.interruptions)
    assert waiting.tool == STEP_UP
    assert waiting.args == {"repoName": "pallets/flask"}
    assert fake.pending() == [waiting.approval_id]
    assert await tools.resolve(state, result.interruptions) != []  # nobody decided: still waiting

    fake.decide(waiting.approval_id, "approved")
    assert await tools.resolve(state, result.interruptions) == []
    result = await Runner.run(agent, state, run_config=integration.run_config())

    assert result.final_output == "Finished."
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]  # once
    assert "answered for pallets/flask" in _outputs(result)[0]
    assert waiting.approval_id not in _everything_the_model_read(fake)
    assert "Waiting for approval" not in _everything_the_model_read(fake)


async def test_a_denied_call_reaches_the_model_as_refused(
    fake: FakeGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer
) -> None:
    agent, result = await _paused(fake, integration, tools)
    state = result.to_state()
    (waiting,) = await tools.resolve(state, result.interruptions)
    fake.decide(waiting.approval_id, "denied")

    assert await tools.resolve(state, result.interruptions) == []
    result = await Runner.run(agent, state, run_config=integration.run_config())

    assert "approval_denied" in _outputs(result)[0]
    assert fake.tool_calls == []


async def test_a_model_written_approval_is_removed(
    fake: FakeGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer
) -> None:
    _, result = await _paused(fake, integration, tools, **{APPROVAL_ARGUMENT: "stolen"})
    await tools.resolve(result.to_state(), result.interruptions)
    for _, params in fake.tool_requests():
        assert APPROVAL_ARGUMENT not in params["arguments"]


async def test_a_hold_the_tool_did_not_announce(fake: FakeGateway, tools: AuthFIMCPServer) -> None:
    await tools.list_tools()
    held = await tools.call_tool(STEP_UP, {"repoName": "x"})
    assert held.isError
    assert held.content[0].text == HOLD_MESSAGE  # type: ignore[union-attr]

    # The tool now pauses the run before its next call, and a resolve carries the approval.
    unannounced = (await tools.list_tools())[0].model_copy(update={"name": STEP_UP, "annotations": None})
    assert tools._needs_approval(None, None, unannounced)  # type: ignore[arg-type]
    (approval_id,) = fake.pending()
    fake.decide(approval_id, "approved")
    answered = await tools.call_tool(STEP_UP, {"repoName": "x"})
    assert not answered.isError
    assert fake.tool_calls == [(STEP_UP, {"repoName": "x"})]


async def test_resolve_ignores_what_is_not_its_own(tools: AuthFIMCPServer) -> None:
    from agents.items import ToolApprovalItem

    other = ToolApprovalItem(agent=Agent(name="a"), raw_item={"name": "lookup", "arguments": "{}"})
    assert await tools.resolve(None, [other, "not an item"]) == []  # type: ignore[arg-type]


async def test_resolve_skips_calls_it_cannot_read(tools: AuthFIMCPServer) -> None:
    from agents.items import ToolApprovalItem

    await tools.list_tools()
    broken = [
        ToolApprovalItem(agent=Agent(name="a"), raw_item={"name": ALLOWED, "arguments": "{not json"}),
        ToolApprovalItem(agent=Agent(name="a"), raw_item={"name": ALLOWED, "arguments": "[1]"}),
        ToolApprovalItem(agent=Agent(name="a"), raw_item={"name": ALLOWED}),
    ]
    assert await tools.resolve(None, broken) == []  # type: ignore[arg-type]


async def test_a_refused_token_on_a_tool_call(fake: FakeGateway, tools: AuthFIMCPServer) -> None:
    await tools.list_tools()
    fake.revoked = True
    with pytest.raises(GatewayTokenRefused):
        await tools.call_tool(ALLOWED, {"repoName": "x"})


def test_the_integration_sets_the_servers_approval_and_address(integration: OpenAIAgentsIntegration) -> None:
    for argument in ("params", "require_approval"):
        given: dict[str, Any] = {argument: "x"}
        with pytest.raises(ConfigurationError, match=argument):
            integration.mcp_server(**given)


# --- tracing ----------------------------------------------------------------------------------------


class Capture(TracingProcessor):
    def __init__(self) -> None:
        self.exported: list[str] = []

    def on_trace_start(self, trace: Trace) -> None:
        self.exported.append(json.dumps(trace.export(), default=str))

    def on_trace_end(self, trace: Trace) -> None:
        pass

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        self.exported.append(json.dumps(span.export(), default=str))

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


def test_the_trace_processor_hands_everything_on() -> None:
    calls: list[str] = []

    class Inner(Capture):
        def on_trace_end(self, trace: Trace) -> None:
            calls.append("trace_end")

        def on_span_start(self, span: Span[Any]) -> None:
            calls.append("span_start")

        def shutdown(self) -> None:
            calls.append("shutdown")

        def force_flush(self) -> None:
            calls.append("flush")

    wrapped = AuthFITraceProcessor(Inner())
    wrapped.on_trace_end(None)  # type: ignore[arg-type]
    wrapped.on_span_start(None)  # type: ignore[arg-type]
    wrapped.force_flush()
    wrapped.shutdown()
    assert calls == ["trace_end", "span_start", "flush", "shutdown"]


@pytest.mark.parametrize("capture_content", [False, True])
async def test_traces_carry_no_token_and_content_only_when_asked(
    fake: FakeGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer, capture_content: bool
) -> None:
    capture = Capture()
    integration.configure_tracing([capture], capture_content=capture_content)
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]

    await Runner.run(
        _agent(integration, tools), PROMPT, run_config=integration.run_config(capture_content=capture_content)
    )
    integration.configure_tracing()
    exported = "\n".join(capture.exported)

    assert AGENT in exported  # the trace was recorded, naming the agent
    for token in fake.minted:
        assert token not in exported
    assert ("ada@example.com" in exported) is capture_content


def test_tracing_off_without_processors_and_a_jwt_key_is_named(
    integration: OpenAIAgentsIntegration, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "eyJhbGciOi.x.y")
    assert integration.configure_tracing() == []
    assert "OPENAI_API_KEY holds a JWT" in caplog.text
    wrapped = AuthFITraceProcessor(Capture())
    assert integration.configure_tracing([wrapped]) == [wrapped]
    integration.configure_tracing()


def test_a_blocking_credential_is_refused(fake: FakeGateway, key: AgentKey) -> None:
    with AuthFI(fake.issuer) as af, pytest.raises(ConfigurationError, match="asyncio"):
        OpenAIAgentsIntegration(af.agent(AGENT, key))  # type: ignore[arg-type]


def test_run_config_names_the_agent(integration: OpenAIAgentsIntegration) -> None:
    config = integration.run_config(trace_metadata={"case": "1"})
    assert config.trace_metadata == {"case": "1", "authfi_agent_id": AGENT}
    assert config.trace_include_sensitive_data is False
