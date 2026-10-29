"""authfi.crewai: CrewAI's own Crew, Anthropic LLM and tools against the real-HTTP gateway stand-in."""

# CrewAI cannot share an environment with the type checkers (see [tool.uv] conflicts), so mypy and pyright
# skip this file; the crewai CI job checks it with CrewAI installed.
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("crewai")

from crewai import Agent, Crew, Task

from authfi import AgentKey, AsyncAuthFI, AuthFI, ConfigurationError, GatewayTokenRefused
from authfi.crewai import AuthFITool, CrewAIIntegration, PendingApproval
from authfi.gateway import APPROVAL_ARGUMENT
from tests.fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway

AGENT = "agt_crew1"
PROMPT = "Look up the wiki; my email is ada@example.com"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
async def integration(fake: FakeGateway, key: AgentKey) -> AsyncIterator[CrewAIIntegration]:
    async with AsyncAuthFI(fake.issuer) as af:
        yield CrewAIIntegration(af.agent(AGENT, key))


def _crew(integration: CrewAIIntegration, tools: list[AuthFITool] | None = None) -> Crew:
    agent = Agent(
        role="Researcher",
        goal="Describe a repository's wiki",
        backstory="You read documentation.",
        llm=integration.llm("claude-haiku-5-5"),
        tools=list(tools or []),
        allow_delegation=False,
    )
    task = Task(description=PROMPT, expected_output="A summary", agent=agent)
    return Crew(agents=[agent], tasks=[task])


def _use(name: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": name, "input": {"repoName": "pallets/flask", **args}}]


def _everything_the_model_read(fake: FakeGateway) -> str:
    return json.dumps([r["json"] for r in fake.model_requests()])


# --- the model --------------------------------------------------------------------------------------


async def test_the_model_carries_the_agents_token(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    result = await _crew(integration).akickoff()

    assert "Done." in str(result)
    (request, *_) = [r for r in fake.requests if r["path"].endswith("/v1/messages")]
    assert request["headers"]["authorization"].startswith("Bearer tok-minted-")
    assert request["headers"]["x-request-id"].startswith(integration.run_id + ".")


async def test_a_refused_token_ends_the_run(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    fake.revoked = True
    with pytest.raises(GatewayTokenRefused):
        async with integration.guard():
            await _crew(integration).akickoff()


async def test_a_refused_token_on_a_tool_call_ends_the_run_even_though_crewai_hides_it(
    fake: FakeGateway, integration: CrewAIIntegration
) -> None:
    tools = await integration.tools()
    fake.tools_revoked = True
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]
    with pytest.raises(GatewayTokenRefused):
        async with integration.guard():
            await _crew(integration, tools).akickoff()


async def test_a_guarded_run_that_goes_well_raises_nothing(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    async with integration.guard():
        result = await _crew(integration).akickoff()
    assert "Done." in str(result)


def test_the_integration_sets_the_models_address_key_and_interceptor(integration: CrewAIIntegration) -> None:
    for argument in ("base_url", "api_key", "interceptor"):
        given: dict[str, Any] = {argument: "x"}
        with pytest.raises(ConfigurationError, match=argument):
            integration.llm("claude-haiku-5-5", **given)


# --- tools ------------------------------------------------------------------------------------------


async def test_tools_keep_the_gateways_names_and_leave_out_never_allowed(integration: CrewAIIntegration) -> None:
    names = [t.name for t in await integration.tools()]
    assert names == [ALLOWED, UNGRANTED, STEP_UP]
    assert NEVER not in names


async def test_never_allowed_tools_on_request(integration: CrewAIIntegration) -> None:
    assert NEVER in [t.name for t in await integration.tools(include_never_allowed=True)]


async def test_an_allowed_call(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]

    result = await _crew(integration, await integration.tools()).akickoff()

    assert "Read it." in str(result)
    assert fake.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    for headers, _ in fake.tool_requests():
        assert headers["authorization"].startswith("Bearer tok-minted-")
    assert "answered for pallets/flask" in _everything_the_model_read(fake)


async def test_a_refusal_reaches_the_model_naming_why(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    fake.model_script += [_use(UNGRANTED), [{"type": "text", "text": "I cannot."}]]

    result = await _crew(integration, await integration.tools()).akickoff()

    assert "I cannot." in str(result)
    read = _everything_the_model_read(fake)
    assert "ETOOL003" in read
    assert "mcp:deepwiki.ask_question" in read
    assert fake.tool_calls == []


async def test_a_refused_token_on_listing(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    fake.tools_revoked = True
    with pytest.raises(GatewayTokenRefused):
        await integration.tools()


async def test_the_blocking_run_is_refused_with_the_reason(integration: CrewAIIntegration) -> None:
    (tool, *_) = await integration.tools()
    with pytest.raises(ConfigurationError, match="akickoff"):
        tool._run(repoName="x")


# --- approvals --------------------------------------------------------------------------------------


async def test_a_held_call_waits_for_the_person_and_runs_once_approved(
    fake: FakeGateway, integration: CrewAIIntegration
) -> None:
    told: list[PendingApproval] = []
    tools = await integration.tools(on_approval=told.append, poll_interval=0.05)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Finished."}]]

    run = asyncio.create_task(_crew(integration, tools).akickoff())
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
    result = await asyncio.wait_for(run, 30)

    assert "Finished." in str(result)
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]  # once
    assert len(fake.approvals) == 1  # the same approval asked again, never a second one
    assert len(told) == 1
    read = _everything_the_model_read(fake)
    assert told[0].approval_id not in read
    assert "Waiting for approval" not in read


async def test_a_denied_call_reaches_the_model_as_refused(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    told: list[PendingApproval] = []
    tools = await integration.tools(on_approval=told.append, poll_interval=0.05)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Denied."}]]

    run = asyncio.create_task(_crew(integration, tools).akickoff())
    for _ in range(200):
        if told:
            break
        await asyncio.sleep(0.05)
    fake.decide(told[0].approval_id, "denied")
    await asyncio.wait_for(run, 30)

    assert "approval_denied" in _everything_the_model_read(fake)
    assert fake.tool_calls == []


async def test_an_approval_that_does_not_come_in_time_is_not_a_yes(
    fake: FakeGateway, integration: CrewAIIntegration
) -> None:
    tools = await integration.tools(approval_timeout=0.2, poll_interval=0.05)
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Gave up."}]]

    await _crew(integration, tools).akickoff()

    assert fake.tool_calls == []
    assert len(fake.pending()) == 1  # still open for the person
    assert "did not come in time" in _everything_the_model_read(fake)


async def test_a_model_written_approval_is_removed(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    tools = await integration.tools(approval_timeout=0.1, poll_interval=0.05)
    (tool,) = [t for t in tools if t.name == STEP_UP]

    await tool._arun(repoName="x", **{APPROVAL_ARGUMENT: "stolen"})

    assert fake.tool_requests()
    for _, params in fake.tool_requests():
        assert params["arguments"].get(APPROVAL_ARGUMENT) != "stolen"


async def test_an_async_callback_is_awaited(fake: FakeGateway, integration: CrewAIIntegration) -> None:
    told: list[str] = []

    async def tell(pending: PendingApproval) -> None:
        told.append(pending.approval_id)

    tools = await integration.tools(on_approval=tell, approval_timeout=0.1, poll_interval=0.05)
    (tool,) = [t for t in tools if t.name == STEP_UP]
    await tool._arun(repoName="x")
    assert told == fake.pending()


def test_a_blocking_credential_is_refused(fake: FakeGateway, key: AgentKey) -> None:
    with AuthFI(fake.issuer) as af, pytest.raises(ConfigurationError, match="asyncio"):
        CrewAIIntegration(af.agent(AGENT, key))  # type: ignore[arg-type]


# --- steps ------------------------------------------------------------------------------------------


async def test_export_steps_sends_the_agents_steps_to_authfi_without_the_words(
    fake: FakeGateway, integration: CrewAIIntegration
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    from opentelemetry.sdk.trace import TracerProvider

    from authfi.crewai import AuthFIStepsListener

    provider = TracerProvider()
    integration.export_steps(tracer_provider=provider)
    try:
        fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Read it."}]]
        await _crew(integration, await integration.tools()).akickoff()
        await asyncio.sleep(0.5)  # CrewAI delivers events from its own threads
    finally:
        AuthFIStepsListener.attach(provider).detach()
    await asyncio.to_thread(provider.force_flush)
    await asyncio.to_thread(provider.shutdown)

    steps = fake.exported_spans()
    names = [s["name"] for s in steps]
    assert any(n.startswith("workflow ") for n in names)
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
