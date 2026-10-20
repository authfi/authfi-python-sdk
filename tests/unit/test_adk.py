"""authfi.adk: ADK's own runner, LiteLlm and McpToolset against a real-HTTP stand-in for the gateway."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from authfi import AgentKey, AsyncAuthFI, ConfigurationError, GatewayTokenRefused, IntegrationError
from authfi.adk import ADKIntegration, _no_google_mtls
from authfi.gateway import APPROVAL_ARGUMENT, HOLD_MESSAGE
from tests.fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway

AGENT = "agt_adk1"
ARGS = {"repoName": "pallets/flask"}


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
async def adk(fake: FakeGateway, key: AgentKey) -> AsyncIterator[ADKIntegration]:
    async with AsyncAuthFI(fake.issuer) as af:
        yield ADKIntegration(af.agent(AGENT, key))


class Run:
    def __init__(self, adk: ADKIntegration) -> None:
        self.adk = adk
        agent = adk.agent(name="research", instruction="Answer briefly.", model="claude-haiku-5-5")
        self.runner = InMemoryRunner(agent=agent, app_name="authfi-test")
        self.session_id = ""

    async def send(self, message: types.Content) -> list[Event]:
        if not self.session_id:
            session = await self.runner.session_service.create_session(app_name="authfi-test", user_id="u")
            self.session_id = session.id
        return [
            event async for event in self.runner.run_async(user_id="u", session_id=self.session_id, new_message=message)
        ]

    async def ask(self, text: str) -> list[Event]:
        return await self.send(types.Content(role="user", parts=[types.Part(text=text)]))


def _use(tool: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": tool, "input": {**ARGS, **args}}]


def _final(events: list[Event]) -> str:
    texts = [p.text for e in events if e.content and e.content.parts for p in e.content.parts if p.text]
    return texts[-1] if texts else ""


def _responses(events: list[Event]) -> list[dict[str, Any]]:
    return [
        dict(r.response or {})
        for e in events
        for r in e.get_function_responses()
        if r.name != "adk_request_confirmation"
    ]


def _everything_the_model_read(fake: FakeGateway) -> str:
    return json.dumps([r["json"] for r in fake.model_requests()])


# --- model and tools --------------------------------------------------------------------------------


async def test_the_model_and_an_allowed_tool_carry_the_agents_token(fake: FakeGateway, adk: ADKIntegration) -> None:
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "It has an overview."}]]

    events = await Run(adk).ask("Describe the wiki")

    assert _final(events) == "It has an overview."
    assert fake.tool_calls == [(ALLOWED, ARGS)]
    for request in fake.model_requests():
        assert request["headers"]["authorization"].startswith("Bearer tok-minted-")
    for headers, _ in fake.tool_requests():
        assert headers["authorization"].startswith("Bearer tok-minted-")


async def test_never_allowed_tools_are_not_offered(fake: FakeGateway, adk: ADKIntegration) -> None:
    offered = {tool.name for tool in await adk.toolset().get_tools()}
    assert offered == {ALLOWED, UNGRANTED, STEP_UP}
    assert NEVER in [tool.name for tool in await adk.toolset(include_never_allowed=True).get_tools()]
    assert [tool.name for tool in await adk.toolset(tool_filter=[ALLOWED]).get_tools()] == [ALLOWED]

    def step_up_only(tool: Any, readonly_context: Any = None) -> bool:
        return bool(tool.name == STEP_UP)

    only_step_up = adk.toolset(tool_filter=step_up_only)
    assert [tool.name for tool in await only_step_up.get_tools()] == [STEP_UP]


async def test_a_refusal_reaches_the_model_naming_why(fake: FakeGateway, adk: ADKIntegration) -> None:
    fake.model_script += [_use(UNGRANTED), [{"type": "text", "text": "I cannot."}]]

    events = await Run(adk).ask("Ask the wiki")

    (response,) = _responses(events)
    assert "ETOOL003" in response["error"]
    assert _final(events) == "I cannot."


async def test_a_refused_token_on_the_model_ends_the_run(fake: FakeGateway, adk: ADKIntegration) -> None:
    fake.revoked = True
    with pytest.raises(GatewayTokenRefused):
        await Run(adk).ask("Describe the wiki")


async def test_a_refused_token_on_a_tool_ends_the_run(fake: FakeGateway, adk: ADKIntegration) -> None:
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Should not get here."}]]
    run = Run(adk)
    agent: Any = run.runner.agent
    await agent.tools[0].get_tools()
    fake.tools_revoked = True
    with pytest.raises(GatewayTokenRefused):
        await run.ask("Describe the wiki")


# --- approvals --------------------------------------------------------------------------------------


async def test_a_held_call_runs_once_the_person_has_decided(fake: FakeGateway, adk: ADKIntegration) -> None:
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Read it."}]]
    run = Run(adk)

    events = await run.ask("Read the wiki")
    (pending,) = adk.held_approvals(events)
    assert pending.tool == STEP_UP
    assert pending.args == ARGS
    assert fake.pending() == [pending.approval_id]

    fake.decide(pending.approval_id, "approved")
    events = await run.send(adk.answer(pending, confirmed=True))

    assert _final(events) == "Read it."
    assert fake.tool_calls == [(STEP_UP, ARGS)]
    assert pending.approval_id not in _everything_the_model_read(fake)
    assert "Waiting for approval" not in _everything_the_model_read(fake)


async def test_a_call_confirmed_too_early_asks_again_under_the_same_approval(
    fake: FakeGateway, adk: ADKIntegration
) -> None:
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Still waiting."}], _use(STEP_UP)]
    run = Run(adk)
    (first,) = adk.held_approvals(await run.ask("Read the wiki"))

    events = await run.send(adk.answer(first, confirmed=True))  # nobody decided yet
    assert HOLD_MESSAGE in json.dumps(_responses(events))
    events = await run.ask("Try again")

    (again,) = adk.held_approvals(events)
    assert again.approval_id == first.approval_id
    assert fake.pending() == [first.approval_id]  # no second approval
    assert fake.tool_calls == []


async def test_a_declined_call_is_not_sent(fake: FakeGateway, adk: ADKIntegration) -> None:
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Not done."}]]
    run = Run(adk)
    (pending,) = adk.held_approvals(await run.ask("Read the wiki"))
    sent = len(fake.tool_requests())

    events = await run.send(adk.answer(pending, confirmed=False))

    assert len(fake.tool_requests()) == sent
    assert "Not run" in json.dumps(_responses(events))


async def test_a_model_written_approval_is_removed(fake: FakeGateway, adk: ADKIntegration) -> None:
    fake.model_script += [_use(STEP_UP, **{APPROVAL_ARGUMENT: "stolen"})]
    await Run(adk).ask("Read the wiki")
    for _, params in fake.tool_requests():
        assert APPROVAL_ARGUMENT not in params["arguments"]


def test_held_approvals_ignores_other_calls() -> None:
    call = types.FunctionCall(id="c1", name="adk_request_confirmation", args={"toolConfirmation": {"payload": "x"}})
    other = types.FunctionCall(id="c2", name="lookup", args={})
    event = Event(
        author="a",
        content=types.Content(role="model", parts=[types.Part(function_call=call), types.Part(function_call=other)]),
    )
    assert ADKIntegration.held_approvals([event]) == []


# --- what the integration guards ------------------------------------------------------------------------


async def test_google_mtls_is_kept_off_the_gateway(adk: ADKIntegration, monkeypatch: pytest.MonkeyPatch) -> None:
    import google.auth

    def probed(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Google credentials were probed for the AuthFI gateway")

    monkeypatch.setattr(google.auth, "default", probed)
    toolset = adk.toolset()
    assert await toolset._mcp_session_manager._get_mtls_transport() is None
    assert len(await toolset.get_tools()) == 3


def test_a_moved_mtls_seam_fails_loudly() -> None:
    with pytest.raises(IntegrationError, match="tested with"):
        _no_google_mtls(SimpleNamespace())  # type: ignore[arg-type]
    with pytest.raises(IntegrationError):
        _no_google_mtls(SimpleNamespace(_mcp_session_manager=SimpleNamespace()))  # type: ignore[arg-type]


def test_tracing_has_no_content_unless_asked(adk: ADKIntegration, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.name=x,authfi.agent_id=old")
    adk.configure_tracing()
    import os

    assert os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] == "false"
    assert os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] == "NO_CONTENT"
    assert os.environ["OTEL_RESOURCE_ATTRIBUTES"] == f"service.name=x,authfi.agent_id={AGENT}"
    adk.configure_tracing(capture_content=True)
    assert os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] == "true"
    assert "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT" not in os.environ
    monkeypatch.delenv("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS")


@pytest.fixture(scope="module")
def spans() -> Any:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return exporter


@pytest.mark.parametrize("capture_content", [False, True])
async def test_spans_carry_no_token_and_content_only_when_asked(
    fake: FakeGateway, adk: ADKIntegration, spans: Any, monkeypatch: pytest.MonkeyPatch, capture_content: bool
) -> None:
    for name in ("ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS", "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"):
        monkeypatch.delenv(name, raising=False)
    adk.configure_tracing(capture_content=capture_content)
    spans.clear()
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "Done."}]]

    await Run(adk).ask("my email is ada@example.com")
    exported = json.dumps([dict(span.attributes or {}) for span in spans.get_finished_spans()], default=str)

    assert "claude-haiku-5-5" in exported  # the model call was traced
    for token in fake.minted:
        assert token not in exported
    assert ("ada@example.com" in exported) is capture_content


async def test_a_blocking_credential_is_refused(fake: FakeGateway, key: AgentKey) -> None:
    from authfi import AuthFI

    with AuthFI(fake.issuer) as af, pytest.raises(ConfigurationError, match="asyncio"):
        ADKIntegration(af.agent(AGENT, key))  # type: ignore[arg-type]


def test_the_credential_arguments_are_the_integrations(adk: ADKIntegration) -> None:
    with pytest.raises(ConfigurationError, match="api_key"):
        adk.model("m", api_key="x")


async def test_your_callbacks_run_after_these(adk: ADKIntegration) -> None:
    async def mine(*args: Any) -> None:
        return None

    agent = adk.agent(name="a", instruction="i", model="m", before_tool_callback=mine, after_tool_callback=[mine])
    assert agent.before_tool_callback == [adk.before_tool_callback, mine]
    assert agent.after_tool_callback == [adk.after_tool_callback, mine]


async def test_with_google_credentials_present_the_gateway_still_gets_the_agents_token(
    fake: FakeGateway, adk: ADKIntegration, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Application Default Credentials ADK swaps in Google's mTLS client, which sends no AuthFI token."""
    import google.auth
    import httpx
    from google.adk.tools.mcp_tool import mcp_session_manager as manager

    through_google: list[httpx.Request] = []

    class GoogleTransport(httpx.AsyncBaseTransport):
        def __init__(self, auth_session: Any) -> None:
            pass

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            through_google.append(request)
            return httpx.Response(500, request=request)

    class MtlsSession:
        is_mtls = True

        def __init__(self, credentials: Any) -> None:
            pass

        async def configure_mtls_channel(self) -> None:
            pass

    def default_credentials(**_: Any) -> tuple[object, str]:
        return object(), "a-project"

    def refreshable(*_: Any, **__: Any) -> object:
        return object()

    monkeypatch.setattr(google.auth, "default", default_credentials)
    monkeypatch.setattr(manager, "_AIO_SUPPORTED", True)
    monkeypatch.setattr(manager, "AsyncAuthorizedSession", MtlsSession)
    monkeypatch.setattr(manager, "_RefreshableAsyncCredentials", refreshable)
    monkeypatch.setattr(manager, "_GoogleAuthAsyncTransport", GoogleTransport)
    monkeypatch.setenv("GOOGLE_API_USE_CLIENT_CERTIFICATE", "true")

    tools = await adk.toolset().get_tools()

    assert len(tools) == 3
    assert through_google == []  # nothing went through Google's client
    listed = [r for r in fake.requests if r["path"].endswith("/mcp")]
    assert listed
    assert all(r["headers"]["authorization"].startswith("Bearer tok-minted-") for r in listed)


REFUSED = "MCP tool execution failed: The AuthFI gateway refused this credential"


async def test_a_refused_token_is_read_from_the_calls_own_response_not_from_shared_state(
    adk: ADKIntegration,
) -> None:
    """ADK may run tool calls in parallel: call B's success clears the gateway's field before A's callback runs."""
    adk.gateway.token_refused = None  # what a parallel success leaves behind
    tool = SimpleNamespace(name=ALLOWED)
    context = SimpleNamespace(tool_confirmation=None)
    with pytest.raises(GatewayTokenRefused):
        await adk.after_tool_callback(tool, dict(ARGS), context, {"error": REFUSED})  # type: ignore[arg-type]


async def test_an_unrelated_tool_error_does_not_raise_a_stale_refusal(adk: ADKIntegration) -> None:
    adk.gateway.token_refused = GatewayTokenRefused(adk.gateway.mcp_url, None)  # left by another call
    tool = SimpleNamespace(name=ALLOWED)
    context = SimpleNamespace(tool_confirmation=None)
    assert await adk.after_tool_callback(tool, dict(ARGS), context, {"error": "repo not found"}) is None  # type: ignore[arg-type]


def test_the_pending_approval_extends_the_one_vocabulary() -> None:
    from authfi.adk import PendingApproval
    from authfi.gateway import PendingApproval as Base

    assert issubclass(PendingApproval, Base)
    assert list(PendingApproval.__dataclass_fields__) == ["approval_id", "tool", "args", "call_id"]
