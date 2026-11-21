"""authfi.langchain_hitl: LangChain's real create_agent and MCP tools, the person deciding in AuthFI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

pytest.importorskip("langchain.agents")
pytest.importorskip("langchain_mcp_adapters")

from langchain.agents import create_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_core.messages import ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from authfi import AgentKey, ApprovalUnavailable, AsyncAuthFI, ConfigurationError
from authfi.gateway import APPROVAL_ARGUMENT, AsyncGateway
from authfi.langchain import LangChainIntegration
from authfi.langchain_hitl import AuthFIApprovalMiddleware
from tests.fake_gateway import ALLOWED, STEP_UP, FakeGateway

AGENT = "agt_hitl1"
CONFIG: Any = {"configurable": {"thread_id": "t1"}}
ASK: Any = {"messages": [{"role": "user", "content": "Read the wiki"}]}


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway(approvals_readable=True) as gateway:
        yield gateway


def _use(name: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": name, "input": {"repoName": "pallets/flask", **args}}]


def _said(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}]


def _model_read(fake: FakeGateway) -> str:
    return json.dumps([r["json"] for r in fake.model_requests()])


async def _build(
    fake: FakeGateway, key: AgentKey, saver: InMemorySaver, **options: Any
) -> tuple[AsyncAuthFI, AuthFIApprovalMiddleware, Any]:
    """A process: a project, the agent, the middleware and a LangChain agent on the shared checkpointer."""
    project = AsyncAuthFI(fake.issuer)
    langchain = LangChainIntegration(project.agent(AGENT, key))
    gateway = langchain.gateway
    assert isinstance(gateway, AsyncGateway)
    client = MultiServerMCPClient(
        {
            "authfi": {
                "transport": "streamable_http",
                "url": gateway.mcp_url,
                "httpx_client_factory": gateway.mcp_client_factory(httpx),
            }
        }
    )
    tools = await client.get_tools(server_name="authfi")
    middleware = AuthFIApprovalMiddleware(langchain, **options)
    agent = create_agent(
        langchain.chat_model("claude-haiku-5-5", max_tokens=64), tools, middleware=[middleware], checkpointer=saver
    )
    return project, middleware, agent


def _tool_messages(state: dict[str, Any]) -> list[ToolMessage]:
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def _interrupt_value(state: dict[str, Any]) -> dict[str, Any]:
    (waiting,) = state["__interrupt__"]
    return dict(waiting.value)


async def _until_pending(fake: FakeGateway) -> str:
    for _ in range(500):
        if fake.pending():
            return fake.pending()[0]
        await asyncio.sleep(0.01)
    raise AssertionError("no approval was raised")


async def test_a_held_call_interrupts_and_runs_once_a_person_approves_in_authfi(
    fake: FakeGateway, key: AgentKey
) -> None:
    fake.model_script += [_use(STEP_UP), _said("Read it.")]
    project, _, agent = await _build(fake, key, InMemorySaver())
    async with project:
        state = await agent.ainvoke(ASK, CONFIG)
        value = _interrupt_value(state)
        assert value["type"] == "tool_approval"
        assert value["name"] == STEP_UP
        assert fake.pending() == [value["authfi_approval_id"]]
        assert fake.tool_calls == []

        fake.decide(value["authfi_approval_id"], "approved")
        state = await agent.ainvoke(Command(resume=True), CONFIG)

    assert "__interrupt__" not in state
    assert state["messages"][-1].content == "Read it."
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]
    assert len(fake.approvals) == 1  # the replay after the resume did not raise a second approval
    assert value["authfi_approval_id"] not in _model_read(fake)
    assert "Waiting for approval" not in _model_read(fake)


async def test_a_local_click_decides_nothing(fake: FakeGateway, key: AgentKey) -> None:
    fake.model_script += [_use(STEP_UP)]
    project, _, agent = await _build(fake, key, InMemorySaver())
    async with project:
        first = _interrupt_value(await agent.ainvoke(ASK, CONFIG))
        state = await agent.ainvoke(Command(resume={"type": "approve"}), CONFIG)  # nobody approved it in AuthFI

    assert _interrupt_value(state)["authfi_approval_id"] == first["authfi_approval_id"]
    assert fake.tool_calls == []
    assert len(fake.approvals) == 1


async def test_a_denied_call_reaches_the_model_as_refused_without_the_approval(
    fake: FakeGateway, key: AgentKey
) -> None:
    fake.model_script += [_use(STEP_UP), _said("Not done.")]
    project, _, agent = await _build(fake, key, InMemorySaver())
    async with project:
        value = _interrupt_value(await agent.ainvoke(ASK, CONFIG))
        fake.decide(value["authfi_approval_id"], "denied")
        state = await agent.ainvoke(Command(resume=True), CONFIG)

    (answer,) = _tool_messages(state)
    assert "approval_denied" in str(answer.content)
    assert value["authfi_approval_id"] not in _model_read(fake)
    assert fake.tool_calls == []


async def test_stopping_gives_up_and_the_model_reads_that_it_did_not_run(fake: FakeGateway, key: AgentKey) -> None:
    fake.model_script += [_use(STEP_UP), _said("Gave up.")]
    project, _, agent = await _build(fake, key, InMemorySaver())
    async with project:
        await agent.ainvoke(ASK, CONFIG)
        state = await agent.ainvoke(Command(resume={"stop": True}), CONFIG)

    (answer,) = _tool_messages(state)
    assert "Not run" in str(answer.content)
    assert fake.tool_calls == []


async def test_a_model_written_approval_is_removed(fake: FakeGateway, key: AgentKey) -> None:
    fake.model_script += [[{"type": "tool_use", "name": STEP_UP, "input": {"repoName": "a/b", APPROVAL_ARGUMENT: "x"}}]]
    project, _, agent = await _build(fake, key, InMemorySaver())
    async with project:
        state = await agent.ainvoke(ASK, CONFIG)
    assert "__interrupt__" in state  # the call was held: the model's "approval" spent nothing
    assert fake.tool_calls == []


async def test_the_gateway_waiting_for_a_person_is_polled_and_the_run_continues(
    fake: FakeGateway, key: AgentKey
) -> None:
    fake.model_script += [_use(STEP_UP), _said("Read it.")]
    project, _, agent = await _build(fake, key, InMemorySaver(), approval_poll=True, poll_interval=0.01)
    async with project:
        running = asyncio.ensure_future(agent.ainvoke(ASK, CONFIG))
        approval_id = await _until_pending(fake)
        assert not running.done()
        fake.decide(approval_id, "approved")
        state = await asyncio.wait_for(running, 10)

    assert "__interrupt__" not in state
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]


async def test_nobody_deciding_in_time_still_interrupts(fake: FakeGateway, key: AgentKey) -> None:
    fake.model_script += [_use(STEP_UP)]
    project, _, agent = await _build(
        fake, key, InMemorySaver(), approval_poll=True, poll_interval=0.01, approval_timeout=0.05
    )
    async with project:
        state = await agent.ainvoke(ASK, CONFIG)
    assert "__interrupt__" in state


async def test_an_agent_revoked_while_waiting_stops_the_run(fake: FakeGateway, key: AgentKey) -> None:
    fake.model_script += [_use(STEP_UP)]
    project, _, agent = await _build(fake, key, InMemorySaver(), approval_poll=True, poll_interval=0.01)
    async with project:
        running = asyncio.ensure_future(agent.ainvoke(ASK, CONFIG))
        await _until_pending(fake)
        fake.revoked = True
        with pytest.raises(ApprovalUnavailable):
            await asyncio.wait_for(running, 10)


async def test_a_restarted_process_resumes_from_the_checkpoint_and_the_saved_approval(
    fake: FakeGateway, key: AgentKey
) -> None:
    saver = InMemorySaver()
    fake.model_script += [_use(STEP_UP), _said("Read it.")]
    first, middleware, agent = await _build(fake, key, saver)
    async with first:
        value = _interrupt_value(await agent.ainvoke(ASK, CONFIG))
        saved = json.dumps(middleware.held())
    assert value["authfi_approval_id"] in saved
    fake.decide(value["authfi_approval_id"], "approved")

    second, middleware, agent = await _build(fake, key, saver)  # a new process: only the checkpoint and `saved`
    middleware.adopt(json.loads(saved))
    async with second:
        state = await agent.ainvoke(Command(resume=True), CONFIG)

    assert state["messages"][-1].content == "Read it."
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]
    assert len(fake.approvals) == 1
    assert value["authfi_approval_id"] not in _model_read(fake)


async def test_a_project_binding_approvals_to_the_run_says_to_raise_the_call_again(key: AgentKey) -> None:
    with FakeGateway(approvals_readable=True, resume_binding="run") as gateway:
        saver = InMemorySaver()
        gateway.model_script += [_use(STEP_UP), _said("Asking again.")]
        first, middleware, agent = await _build(gateway, key, saver)
        async with first:
            value = _interrupt_value(await agent.ainvoke(ASK, CONFIG))
            saved = middleware.held()
        gateway.decide(value["authfi_approval_id"], "approved")

        second, middleware, agent = await _build(gateway, key, saver)
        middleware.adopt(saved)
        async with second:
            state = await agent.ainvoke(Command(resume=True), CONFIG)

        (answer,) = _tool_messages(state)
        assert "Raise the call again" in str(answer.content)
        assert gateway.tool_calls == []
        assert value["authfi_approval_id"] not in _model_read(gateway)


async def test_a_tool_the_gateway_allows_is_untouched(fake: FakeGateway, key: AgentKey) -> None:
    fake.model_script += [_use(ALLOWED), _said("Done.")]
    project, _, agent = await _build(fake, key, InMemorySaver())
    async with project:
        state = await agent.ainvoke(ASK, CONFIG)
    assert "__interrupt__" not in state
    assert fake.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]


async def test_a_listed_tool_asks_a_person_first_and_runs_once_approved(fake: FakeGateway, key: AgentKey) -> None:
    fake.model_script += [_use(ALLOWED), _said("Done.")]
    project, _, agent = await _build(fake, key, InMemorySaver(), request=[ALLOWED])
    async with project:
        value = _interrupt_value(await agent.ainvoke(ASK, CONFIG))
        assert fake.tool_calls == []
        fake.decide(value["authfi_approval_id"], "approved")
        state = await agent.ainvoke(Command(resume=True), CONFIG)

    assert state["messages"][-1].content == "Done."
    assert fake.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    assert len(fake.approvals) == 1


async def test_a_listed_tool_a_person_denied_does_not_run(fake: FakeGateway, key: AgentKey) -> None:
    fake.model_script += [_use(ALLOWED), _said("Not done.")]
    project, _, agent = await _build(fake, key, InMemorySaver(), request=[ALLOWED])
    async with project:
        value = _interrupt_value(await agent.ainvoke(ASK, CONFIG))
        fake.decide(value["authfi_approval_id"], "denied")
        state = await agent.ainvoke(Command(resume=True), CONFIG)

    (answer,) = _tool_messages(state)
    assert "denied" in str(answer.content)
    assert fake.tool_calls == []


async def test_asking_needs_the_agent_itself(fake: FakeGateway, key: AgentKey) -> None:
    class Holder:  # a credential that cannot ask
        pass

    async with AsyncAuthFI(fake.issuer) as project:
        langchain = LangChainIntegration(project.agent(AGENT, key))
        assert isinstance(langchain.gateway, AsyncGateway)
        langchain.gateway.source = Holder()  # type: ignore[assignment]
        with pytest.raises(ConfigurationError):
            AuthFIApprovalMiddleware(langchain, request=[ALLOWED])
        with pytest.raises(ConfigurationError):
            AuthFIApprovalMiddleware(object())
