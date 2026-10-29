"""authfi.pydantic_ai: Pydantic AI's own agent, Anthropic model and MCP toolset against the gateway stand-in."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("pydantic_ai")

from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from authfi import AgentKey, AsyncAuthFI, AuthFI, ConfigurationError, GatewayTokenRefused
from authfi.gateway import APPROVAL_ARGUMENT, HOLD_MESSAGE
from authfi.pydantic_ai import AuthFIModel, AuthFIToolset, PydanticAIIntegration
from tests.fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway

AGENT = "agt_pydantic1"
PROMPT = "Look up the wiki; my email is ada@example.com"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
async def integration(fake: FakeGateway, key: AgentKey) -> AsyncIterator[PydanticAIIntegration]:
    async with AsyncAuthFI(fake.issuer) as af:
        yield PydanticAIIntegration(af.agent(AGENT, key))


@pytest.fixture
async def tools(integration: PydanticAIIntegration) -> AsyncIterator[AuthFIToolset]:
    async with integration.toolset() as server:
        yield server


def _agent(integration: PydanticAIIntegration, tools: AuthFIToolset | None = None) -> Agent[None, Any]:
    return Agent(
        integration.model("claude-haiku-5-5"),
        toolsets=[tools] if tools else [],
        output_type=[str, DeferredToolRequests],
    )


def _use(name: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": name, "input": {"repoName": "pallets/flask", **args}}]


def _returns(result: Any) -> list[str]:
    return [
        str(part.content)
        for message in result.all_messages()
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]


def _everything_the_model_read(fake: FakeGateway) -> str:
    return json.dumps([r["json"] for r in fake.model_requests()])


# --- the model --------------------------------------------------------------------------------------


async def test_the_model_carries_the_agents_token_and_no_anthropic_key(
    fake: FakeGateway, integration: PydanticAIIntegration
) -> None:
    result = await _agent(integration).run(PROMPT)

    assert result.output == "Done."
    (request,) = [r for r in fake.requests if r["path"].endswith("/v1/messages")]
    assert request["headers"]["authorization"].startswith("Bearer tok-minted-")
    assert "sk-ant" not in json.dumps(request["headers"])


async def test_a_streamed_run_carries_the_token_too(fake: FakeGateway, integration: PydanticAIIntegration) -> None:
    async with _agent(integration).run_stream(PROMPT) as stream:
        assert await stream.get_output() == "Done."
    assert all(
        r["headers"]["authorization"].startswith("Bearer tok-minted-")
        for r in fake.requests
        if r["path"].endswith("/v1/messages")
    )


async def test_a_refused_token_ends_the_run(fake: FakeGateway, integration: PydanticAIIntegration) -> None:
    fake.revoked = True
    with pytest.raises(GatewayTokenRefused):
        await _agent(integration).run(PROMPT)


async def test_a_refused_token_ends_a_streamed_run(fake: FakeGateway, integration: PydanticAIIntegration) -> None:
    fake.revoked = True
    with pytest.raises(GatewayTokenRefused):
        async with _agent(integration).run_stream(PROMPT) as stream:
            await stream.get_output()


# --- tools ------------------------------------------------------------------------------------------


async def test_tools_keep_the_gateways_names_and_leave_out_never_allowed(
    integration: PydanticAIIntegration, tools: AuthFIToolset
) -> None:
    names = sorted(d.name for d in await _tool_defs(tools))
    assert names == sorted([ALLOWED, UNGRANTED, STEP_UP])
    assert NEVER not in names


async def _tool_defs(toolset: Any) -> list[Any]:
    from pydantic_ai import RunContext
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.usage import RunUsage

    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    return [t.tool_def for t in (await toolset.get_tools(ctx)).values()]


async def test_never_allowed_tools_on_request(integration: PydanticAIIntegration) -> None:
    async with integration.toolset(include_never_allowed=True) as server:
        assert NEVER in [d.name for d in await _tool_defs(server)]


async def test_an_allowed_call(fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset) -> None:
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]

    result = await _agent(integration, tools).run(PROMPT)

    assert result.output == "Read it."
    assert fake.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    assert _returns(result) == [f"{ALLOWED} answered for pallets/flask"]
    for headers, _ in fake.tool_requests():
        assert headers["authorization"].startswith("Bearer tok-minted-")


async def test_a_refusal_reaches_the_model_naming_why(
    fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset
) -> None:
    fake.model_script += [_use(UNGRANTED), [{"type": "text", "text": "I cannot."}]]

    result = await _agent(integration, tools).run(PROMPT)

    (output,) = _returns(result)
    assert "ETOOL003" in output
    assert "mcp:deepwiki.ask_question" in output
    assert result.output == "I cannot."
    assert fake.tool_calls == []


async def test_a_refused_token_on_connecting(fake: FakeGateway, integration: PydanticAIIntegration) -> None:
    fake.tools_revoked = True
    with pytest.raises(GatewayTokenRefused):
        async with integration.toolset() as server:
            await _tool_defs(server)


async def test_a_refused_token_on_a_tool_call_ends_the_run(
    fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset
) -> None:
    await _tool_defs(tools)  # lists the tools once; the toolset keeps the list
    fake.tools_revoked = True
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]
    with pytest.raises(GatewayTokenRefused):
        await _agent(integration, tools).run(PROMPT)


async def test_an_error_that_is_not_a_refusal_is_left_alone() -> None:
    from pydantic_ai.models.function import FunctionModel

    def boom(messages: Any, info: Any) -> Any:
        raise ValueError("the model broke")

    with pytest.raises(ValueError, match="the model broke"):
        await Agent(AuthFIModel(FunctionModel(boom))).run(PROMPT)


# --- approvals --------------------------------------------------------------------------------------


async def _deferred(
    fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset, **args: Any
) -> tuple[Agent[None, Any], Any]:
    fake.model_script += [_use(STEP_UP, **args), [{"type": "text", "text": "Finished."}]]
    agent = _agent(integration, tools)
    result = await agent.run(PROMPT)
    assert isinstance(result.output, DeferredToolRequests)
    assert fake.tool_requests() == []  # deferred before the call
    return agent, result


async def _resume(agent: Agent[None, Any], result: Any, decisions: DeferredToolResults) -> Any:
    return await agent.run(message_history=result.all_messages(), deferred_tool_results=decisions)


async def test_a_call_that_may_need_approval_defers_and_runs_once_approved(
    fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset
) -> None:
    agent, result = await _deferred(fake, integration, tools)
    decisions = DeferredToolResults()

    (waiting,) = await tools.resolve(result.output, decisions)
    assert waiting.tool == STEP_UP
    assert waiting.args == {"repoName": "pallets/flask"}
    assert waiting.call.tool_name == STEP_UP
    assert fake.pending() == [waiting.approval_id]
    assert decisions.approvals == {}
    assert await tools.resolve(result.output, decisions) != []  # nobody decided: still waiting
    assert len(fake.approvals) == 1  # the same approval, not a second one

    fake.decide(waiting.approval_id, "approved")
    assert await tools.resolve(result.output, decisions) == []
    final = await _resume(agent, result, decisions)

    assert final.output == "Finished."
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]  # once
    assert "answered for pallets/flask" in _returns(final)[0]
    assert waiting.approval_id not in _everything_the_model_read(fake)
    assert "Waiting for approval" not in _everything_the_model_read(fake)


async def test_a_denied_call_reaches_the_model_as_refused(
    fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset
) -> None:
    agent, result = await _deferred(fake, integration, tools)
    decisions = DeferredToolResults()
    (waiting,) = await tools.resolve(result.output, decisions)
    fake.decide(waiting.approval_id, "denied")

    assert await tools.resolve(result.output, decisions) == []
    final = await _resume(agent, result, decisions)

    assert "approval_denied" in _returns(final)[0]
    assert fake.tool_calls == []


async def test_a_model_written_approval_is_removed(
    fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset
) -> None:
    _, result = await _deferred(fake, integration, tools, **{APPROVAL_ARGUMENT: "stolen"})
    await tools.resolve(result.output, DeferredToolResults())
    assert fake.tool_requests()
    for _, params in fake.tool_requests():
        assert APPROVAL_ARGUMENT not in params["arguments"]


async def test_an_approval_given_without_resolve_does_not_skip_the_gateway(
    fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset
) -> None:
    agent, result = await _deferred(fake, integration, tools)
    decisions = DeferredToolResults(approvals={result.output.approvals[0].tool_call_id: True})

    final = await _resume(agent, result, decisions)

    # The gateway, not the application's say-so, decides: it held the call, and the model reads only that.
    assert fake.tool_calls == []
    assert len(fake.pending()) == 1
    assert _returns(final) == [HOLD_MESSAGE]
    assert fake.pending()[0] not in _everything_the_model_read(fake)

    # The tool now defers before its next call, and a resolve carries the approval the gateway opened.
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Finished."}]]
    again = await agent.run(PROMPT)
    assert isinstance(again.output, DeferredToolRequests)
    decisions = DeferredToolResults()
    (waiting,) = await tools.resolve(again.output, decisions)
    assert [waiting.approval_id] == fake.pending()


async def test_resolve_ignores_what_is_not_its_own(tools: AuthFIToolset) -> None:
    from pydantic_ai import ToolCallPart

    other = DeferredToolRequests(approvals=[ToolCallPart("lookup", {}, "call-1")])
    decisions = DeferredToolResults()
    assert await tools.resolve(other, decisions) == []
    assert decisions.approvals == {}


def test_the_integration_sets_the_servers_address_and_client(integration: PydanticAIIntegration) -> None:
    for argument in ("client", "http_client", "headers", "auth", "verify", "process_tool_call"):
        given: dict[str, Any] = {argument: "x"}
        with pytest.raises(ConfigurationError, match=argument):
            integration.toolset(**given)


def test_a_blocking_credential_is_refused(fake: FakeGateway, key: AgentKey) -> None:
    with AuthFI(fake.issuer) as af, pytest.raises(ConfigurationError, match="asyncio"):
        PydanticAIIntegration(af.agent(AGENT, key))  # type: ignore[arg-type]


def test_the_run_is_named(integration: PydanticAIIntegration) -> None:
    assert integration.agent_id == AGENT
    assert integration.run_id == getattr(integration.source, "run_id", "")


# --- steps ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("capture_content", [False, True])
async def test_export_steps_sends_the_agents_steps_to_authfi(
    fake: FakeGateway, integration: PydanticAIIntegration, tools: AuthFIToolset, capture_content: bool
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    from opentelemetry.sdk.trace import TracerProvider

    provider = TracerProvider()
    integration.export_steps(tracer_provider=provider, capture_content=capture_content)
    try:
        fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]
        await _agent(integration, tools).run(PROMPT)
    finally:
        Agent.instrument_all(False)
    await asyncio.to_thread(provider.force_flush)
    await asyncio.to_thread(provider.shutdown)

    steps = fake.exported_spans()
    names = [s["name"] for s in steps]
    assert any(n.startswith("invoke_agent") or n == "agent run" for n in names)
    assert any(n.startswith("chat") for n in names)
    assert any(n.startswith("execute_tool") or n == "running tool" for n in names)
    for step in steps:
        assert step["attributes"]["authfi.agent_id"] == AGENT
        assert step["attributes"]["authfi.run_id"] == integration.run_id
        assert step["headers"]["authorization"].startswith("Bearer tok-minted-")
    posted = b"".join(r["body"] for r in fake.requests if r["path"] == "/otlp/v1/traces")
    assert not any(t.encode() in posted for t in fake.minted)
    assert (b"ada@example.com" in posted) is capture_content
