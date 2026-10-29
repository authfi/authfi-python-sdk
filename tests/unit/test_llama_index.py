"""authfi.llama_index: LlamaIndex's own FunctionAgent, Anthropic LLM and MCP tools against the gateway stand-in."""

# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportDeprecated=false
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("llama_index.core")
pytest.importorskip("llama_index.llms.anthropic")
pytest.importorskip("llama_index.tools.mcp")

import anthropic
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import FunctionTool

from authfi import AgentKey, AsyncAuthFI, AuthFI, ConfigurationError, GatewayTokenRefused
from authfi.gateway import APPROVAL_ARGUMENT
from authfi.llama_index import AuthFIMCPClient, LlamaIndexIntegration, PendingApproval
from tests.fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway

AGENT = "agt_llama1"
PROMPT = "Look up the wiki; my email is ada@example.com"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
async def integration(fake: FakeGateway, key: AgentKey) -> AsyncIterator[LlamaIndexIntegration]:
    async with AsyncAuthFI(fake.issuer) as af:
        yield LlamaIndexIntegration(af.agent(AGENT, key))


def _agent(integration: LlamaIndexIntegration, tools: list[FunctionTool] | None = None) -> FunctionAgent:
    return FunctionAgent(tools=list(tools or []), llm=integration.llm("claude-haiku-5-5"), system_prompt="Be brief.")


async def _run(agent: FunctionAgent) -> Any:
    return await agent.run(user_msg=PROMPT)


def _use(name: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": name, "input": {"repoName": "pallets/flask", **args}}]


def _everything_the_model_read(fake: FakeGateway) -> str:
    return json.dumps([r["json"] for r in fake.model_requests()])


# --- the model --------------------------------------------------------------------------------------


async def test_the_model_carries_the_agents_token(fake: FakeGateway, integration: LlamaIndexIntegration) -> None:
    answer = await _agent(integration).run(user_msg=PROMPT)

    assert "Done." in str(answer)
    (request, *_) = [r for r in fake.requests if r["path"].endswith("/v1/messages")]
    assert request["headers"]["authorization"].startswith("Bearer tok-minted-")
    assert request["headers"]["x-request-id"].startswith(integration.run_id + ".")


async def test_a_refused_token_ends_the_run(fake: FakeGateway, integration: LlamaIndexIntegration) -> None:
    fake.revoked = True
    with pytest.raises(GatewayTokenRefused):
        async with integration.guard():
            await _agent(integration).run(user_msg=PROMPT)


async def test_a_guarded_run_that_goes_well_raises_nothing(
    fake: FakeGateway, integration: LlamaIndexIntegration
) -> None:
    async with integration.guard():
        answer = await _agent(integration).run(user_msg=PROMPT)
    assert "Done." in str(answer)


async def test_a_blocking_call_says_why_it_cannot(integration: LlamaIndexIntegration) -> None:
    with pytest.raises(anthropic.APIConnectionError) as refused:
        integration.llm("claude-haiku-5-5").complete("hello")
    assert isinstance(refused.value.__cause__, ConfigurationError)
    assert "asyncio" in str(refused.value.__cause__)


def test_the_integration_sets_the_models_address_and_key(integration: LlamaIndexIntegration) -> None:
    for argument in ("base_url", "api_key", "region", "project_id", "aws_region"):
        given: dict[str, Any] = {argument: "x"}
        with pytest.raises(ConfigurationError, match=argument):
            integration.llm("claude-haiku-5-5", **given)


# --- tools ------------------------------------------------------------------------------------------


async def test_tools_keep_the_gateways_names_and_leave_out_never_allowed(integration: LlamaIndexIntegration) -> None:
    names = [t.metadata.name for t in await integration.tools()]
    assert names == [ALLOWED, UNGRANTED, STEP_UP]
    assert NEVER not in names


async def test_never_allowed_tools_on_request(integration: LlamaIndexIntegration) -> None:
    assert NEVER in [t.metadata.name for t in await integration.tools(include_never_allowed=True)]


async def test_an_allowed_call(fake: FakeGateway, integration: LlamaIndexIntegration) -> None:
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]

    answer = await _agent(integration, await integration.tools()).run(user_msg=PROMPT)

    assert "Read it." in str(answer)
    assert fake.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    for headers, _ in fake.tool_requests():
        assert headers["authorization"].startswith("Bearer tok-minted-")
    assert "answered for pallets/flask" in _everything_the_model_read(fake)


async def test_a_refusal_reaches_the_model_naming_why(fake: FakeGateway, integration: LlamaIndexIntegration) -> None:
    fake.model_script += [_use(UNGRANTED), [{"type": "text", "text": "I cannot."}]]

    answer = await _agent(integration, await integration.tools()).run(user_msg=PROMPT)

    assert "I cannot." in str(answer)
    read = _everything_the_model_read(fake)
    assert "ETOOL003" in read
    assert "mcp:deepwiki.ask_question" in read
    assert fake.tool_calls == []


async def test_a_refused_token_on_listing(fake: FakeGateway, integration: LlamaIndexIntegration) -> None:
    fake.tools_revoked = True
    with pytest.raises(GatewayTokenRefused):
        async with integration.guard():
            await integration.tools()


async def test_a_refused_token_on_a_tool_call_ends_the_run_even_though_llamaindex_hides_it(
    fake: FakeGateway, integration: LlamaIndexIntegration
) -> None:
    tools = await integration.tools()
    fake.tools_revoked = True
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]
    with pytest.raises(GatewayTokenRefused):
        async with integration.guard():
            await _agent(integration, tools).run(user_msg=PROMPT)


# --- approvals --------------------------------------------------------------------------------------


async def test_a_held_call_waits_for_the_person_and_runs_once_approved(
    fake: FakeGateway, integration: LlamaIndexIntegration
) -> None:
    told: list[PendingApproval] = []
    tools = await integration.tools(on_approval=told.append, poll_interval=0.05)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Finished."}]]

    run = asyncio.create_task(_run(_agent(integration, tools)))
    for _ in range(200):
        if told:
            break
        await asyncio.sleep(0.05)
    assert [t.tool for t in told] == [STEP_UP]
    assert told[0].args == {"repoName": "pallets/flask"}
    assert fake.pending() == [told[0].approval_id]
    assert fake.tool_calls == []
    assert not run.done()

    fake.decide(told[0].approval_id, "approved")
    answer = await asyncio.wait_for(run, 30)

    assert "Finished." in str(answer)
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]  # once
    assert len(fake.approvals) == 1  # the same approval asked again, never a second one
    assert len(told) == 1
    read = _everything_the_model_read(fake)
    assert told[0].approval_id not in read
    assert "Waiting for approval" not in read


async def test_a_denied_call_reaches_the_model_as_refused(
    fake: FakeGateway, integration: LlamaIndexIntegration
) -> None:
    told: list[PendingApproval] = []
    tools = await integration.tools(on_approval=told.append, poll_interval=0.05)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Denied."}]]

    run = asyncio.create_task(_run(_agent(integration, tools)))
    for _ in range(200):
        if told:
            break
        await asyncio.sleep(0.05)
    fake.decide(told[0].approval_id, "denied")
    await asyncio.wait_for(run, 30)

    assert "approval_denied" in _everything_the_model_read(fake)
    assert fake.tool_calls == []


async def test_an_approval_that_does_not_come_in_time_is_not_a_yes(
    fake: FakeGateway, integration: LlamaIndexIntegration
) -> None:
    tools = await integration.tools(approval_timeout=0.2, poll_interval=0.05)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Gave up."}]]

    await _agent(integration, tools).run(user_msg=PROMPT)

    assert fake.tool_calls == []
    assert len(fake.pending()) == 1  # still open for the person
    assert "did not come in time" in _everything_the_model_read(fake)


async def test_a_model_written_approval_is_removed(fake: FakeGateway, integration: LlamaIndexIntegration) -> None:
    client = AuthFIMCPClient(integration.gateway, approval_timeout=0.1, poll_interval=0.05)

    await client.call_tool(STEP_UP, {"repoName": "x", APPROVAL_ARGUMENT: "stolen"})

    assert fake.tool_requests()
    for _, params in fake.tool_requests():
        assert params["arguments"].get(APPROVAL_ARGUMENT) != "stolen"


async def test_an_async_callback_is_awaited(fake: FakeGateway, integration: LlamaIndexIntegration) -> None:
    told: list[str] = []

    async def tell(pending: PendingApproval) -> None:
        told.append(pending.approval_id)

    client = AuthFIMCPClient(integration.gateway, on_approval=tell, approval_timeout=0.1, poll_interval=0.05)
    await client.call_tool(STEP_UP, {"repoName": "x"})
    assert told == fake.pending()


def test_a_blocking_credential_is_refused(fake: FakeGateway, key: AgentKey) -> None:
    with AuthFI(fake.issuer) as af, pytest.raises(ConfigurationError, match="asyncio"):
        LlamaIndexIntegration(af.agent(AGENT, key))  # type: ignore[arg-type]


# --- steps ------------------------------------------------------------------------------------------


async def test_export_steps_sends_the_agents_steps_to_authfi_without_the_words(
    fake: FakeGateway, integration: LlamaIndexIntegration
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    from opentelemetry.sdk.trace import TracerProvider

    from authfi.llama_index import AuthFIStepsHandler

    provider = TracerProvider()
    integration.export_steps(tracer_provider=provider)
    try:
        fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]
        await _agent(integration, await integration.tools()).run(user_msg=PROMPT)
    finally:
        AuthFIStepsHandler.attach(provider).detach()
    await asyncio.to_thread(provider.force_flush)
    await asyncio.to_thread(provider.shutdown)

    steps = fake.exported_spans()
    names = [s["name"] for s in steps]
    assert any(n.startswith("invoke_agent ") for n in names)
    assert any(n.startswith("execute_tool ") for n in names)
    assert any(n.startswith("chat ") for n in names)
    for step in steps:
        assert step["attributes"]["authfi.agent_id"] == AGENT
        assert step["attributes"]["authfi.run_id"] == integration.run_id
        assert step["headers"]["authorization"].startswith("Bearer tok-minted-")
    posted = b"".join(r["body"] for r in fake.requests if r["path"] == "/otlp/v1/traces")
    assert b"ada@example.com" not in posted
    assert not any(t.encode() in posted for t in fake.minted)
