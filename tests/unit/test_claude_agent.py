"""authfi.claude_agent: the permission callback, the in-process MCP server and the options.

The gateway is the real-HTTP stand-in; Claude Code itself is not started.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import PermissionResultAllow, PermissionResultDeny, ToolPermissionContext

from authfi import (
    AgentKey,
    AsyncAuthFI,
    AuthFI,
    ConfigurationError,
    GatewayTokenRefused,
    NotSupportedError,
)
from authfi.claude_agent import CONTENT_SWITCHES, PLACEHOLDER_KEY_COMMAND, ClaudeAgentIntegration, PendingApproval
from authfi.gateway import APPROVAL_ARGUMENT, HOLD_MESSAGE
from tests.fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway
from tests.support import FakeClock

AGENT = "agt_claude1"
CONTEXT = ToolPermissionContext(tool_use_id="toolu_1")
ARGS = {"repoName": "pallets/flask"}


def _name(tool: str) -> str:
    return "mcp__authfi__" + tool


@pytest.fixture(scope="module")
def key_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("keys") / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    return path.resolve()


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
async def af(fake: FakeGateway) -> AsyncIterator[AsyncAuthFI]:
    async with AsyncAuthFI(fake.issuer) as project:
        yield project


@pytest.fixture
def claude(af: AsyncAuthFI, key_file: Path) -> ClaudeAgentIntegration:
    return ClaudeAgentIntegration(af.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))


def _handler(claude: ClaudeAgentIntegration, tool: str) -> Any:
    return claude._handler(tool)


# --- who may use it ---------------------------------------------------------------------------------


async def test_a_token_held_for_a_person_is_not_supported(claude: ClaudeAgentIntegration) -> None:
    from authfi.agents.agent import AsyncDelegatedToken

    delegated = object.__new__(AsyncDelegatedToken)
    with pytest.raises(NotSupportedError) as caught:
        ClaudeAgentIntegration(delegated)  # type: ignore[arg-type]
    assert caught.value.feature == "claude_agent_delegated"


def test_the_agent_must_be_asyncio_and_its_key_in_a_file(fake: FakeGateway, af: AsyncAuthFI, key_file: Path) -> None:
    with AuthFI(fake.issuer) as blocking, pytest.raises(ConfigurationError, match="AsyncAuthFI"):
        ClaudeAgentIntegration(blocking.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="key file"):
        ClaudeAgentIntegration(af.agent(AGENT, AgentKey.generate(kid="k2")))


# --- tools and the permission callback ----------------------------------------------------------------


async def test_tools_leave_out_never_allowed(claude: ClaudeAgentIntegration) -> None:
    assert [tool.name for tool in await claude.tools()] == [ALLOWED, UNGRANTED, STEP_UP]
    assert NEVER in [tool.name for tool in await claude.tools(include_never_allowed=True)]


async def test_an_allowed_call_is_sent_once(fake: FakeGateway, claude: ClaudeAgentIntegration) -> None:
    decision = await claude.can_use_tool(_name(ALLOWED), ARGS, CONTEXT)
    assert isinstance(decision, PermissionResultAllow)
    assert decision.updated_input == ARGS
    assert decision.updated_permissions is None

    answer = await _handler(claude, ALLOWED)(ARGS)

    assert answer["is_error"] is False
    assert "answered for pallets/flask" in answer["content"][0]["text"]
    assert fake.tool_calls == [(ALLOWED, ARGS)]
    for headers, _ in fake.tool_requests():
        assert headers["authorization"].startswith("Bearer tok-minted-")


async def test_a_refusal_reaches_the_model_naming_why(fake: FakeGateway, claude: ClaudeAgentIntegration) -> None:
    assert isinstance(await claude.can_use_tool(_name(UNGRANTED), ARGS, CONTEXT), PermissionResultAllow)
    answer = await _handler(claude, UNGRANTED)(ARGS)
    assert answer["is_error"] is True
    assert "ETOOL003" in answer["content"][0]["text"]
    assert len(fake.tool_requests()) == 1


async def test_a_held_call_waits_for_the_decision(fake: FakeGateway, af: AsyncAuthFI, key_file: Path) -> None:
    seen: list[PendingApproval] = []

    async def until_decided(pending: PendingApproval) -> None:
        seen.append(pending)
        fake.decide(pending.approval_id, "approved")

    claude = ClaudeAgentIntegration(
        af.agent(AGENT, AgentKey.from_file(key_file, kid="k1")), wait_for_decision=until_decided
    )
    decision = await claude.can_use_tool(_name(STEP_UP), ARGS, CONTEXT)

    assert isinstance(decision, PermissionResultAllow)
    (pending,) = seen
    assert pending.tool == STEP_UP
    assert pending.args == ARGS
    answer = await _handler(claude, STEP_UP)(ARGS)
    assert "answered for pallets/flask" in answer["content"][0]["text"]
    assert fake.tool_calls == [(STEP_UP, ARGS)]
    assert pending.approval_id not in json.dumps(answer)


async def test_a_denied_call_reaches_the_model_as_refused(fake: FakeGateway, af: AsyncAuthFI, key_file: Path) -> None:
    def deny(pending: PendingApproval) -> None:
        fake.decide(pending.approval_id, "denied")

    claude = ClaudeAgentIntegration(af.agent(AGENT, AgentKey.from_file(key_file, kid="k1")), wait_for_decision=deny)
    assert isinstance(await claude.can_use_tool(_name(STEP_UP), ARGS, CONTEXT), PermissionResultAllow)
    answer = await _handler(claude, STEP_UP)(ARGS)
    assert "approval_denied" in answer["content"][0]["text"]
    assert fake.tool_calls == []


async def test_without_a_wait_a_held_call_is_refused_for_now(fake: FakeGateway, claude: ClaudeAgentIntegration) -> None:
    first = await claude.can_use_tool(_name(STEP_UP), ARGS, CONTEXT)
    assert isinstance(first, PermissionResultDeny)
    assert first.message == HOLD_MESSAGE
    assert first.interrupt is False
    (approval_id,) = fake.pending()

    fake.decide(approval_id, "approved")
    again = await claude.can_use_tool(_name(STEP_UP), ARGS, CONTEXT)

    assert isinstance(again, PermissionResultAllow)
    await _handler(claude, STEP_UP)(ARGS)
    assert fake.tool_calls == [(STEP_UP, ARGS)]
    assert fake.tool_requests()[-1][1]["arguments"][APPROVAL_ARGUMENT] == approval_id


async def test_a_model_written_approval_is_removed(fake: FakeGateway, claude: ClaudeAgentIntegration) -> None:
    decision = await claude.can_use_tool(_name(STEP_UP), {**ARGS, APPROVAL_ARGUMENT: "stolen"}, CONTEXT)
    assert isinstance(decision, PermissionResultDeny)
    for _, params in fake.tool_requests():
        assert APPROVAL_ARGUMENT not in params["arguments"]


async def test_the_server_alone_never_shows_the_hold(fake: FakeGateway, claude: ClaudeAgentIntegration) -> None:
    answer = await _handler(claude, STEP_UP)(ARGS)  # as if can_use_tool had not been asked
    assert answer == {"content": [{"type": "text", "text": HOLD_MESSAGE}], "is_error": True}
    (approval_id,) = fake.pending()
    fake.decide(approval_id, "approved")
    answer = await _handler(claude, STEP_UP)(ARGS)
    assert answer["is_error"] is False


async def test_other_tools_are_refused_unless_you_decide_them(claude: ClaudeAgentIntegration) -> None:
    refused = await claude.can_use_tool("Bash", {"command": "ls"}, CONTEXT)
    assert isinstance(refused, PermissionResultDeny)
    assert "not one of AuthFI's tools" in refused.message

    async def mine(name: str, data: dict[str, Any], context: ToolPermissionContext) -> PermissionResultAllow:
        return PermissionResultAllow()

    await claude.options(model="claude-haiku-5-5", can_use_tool=mine)
    assert isinstance(await claude.can_use_tool("Bash", {"command": "ls"}, CONTEXT), PermissionResultAllow)


async def test_a_refused_token_stops_the_run(fake: FakeGateway, claude: ClaudeAgentIntegration) -> None:
    fake.revoked = True

    decision = await claude.can_use_tool(_name(ALLOWED), ARGS, CONTEXT)

    assert isinstance(decision, PermissionResultDeny)
    assert decision.interrupt is True
    assert isinstance(claude.token_refused, GatewayTokenRefused)
    answer = await _handler(claude, ALLOWED)(ARGS)
    assert "refused this credential" in answer["content"][0]["text"]
    with pytest.raises(GatewayTokenRefused):
        await claude.tools()


async def test_the_ask_hook_sends_only_authfi_tools_to_the_permission_flow(claude: ClaudeAgentIntegration) -> None:
    no_context: Any = None
    governed: Any = {"hook_event_name": "PreToolUse", "tool_name": _name(ALLOWED)}
    other: Any = {"hook_event_name": "PreToolUse", "tool_name": "Read"}
    asked: Any = await claude._ask(governed, "t", no_context)
    assert asked["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert await claude._ask(other, "t", no_context) == {}


# --- options ------------------------------------------------------------------------------------------


async def test_the_options_carry_no_credential(
    fake: FakeGateway, claude: ClaudeAgentIntegration, key_file: Path
) -> None:
    options = await claude.options(model="claude-haiku-5-5", env={"EXTRA": "1"}, settings='{"theme": "dark"}')

    env = dict(options.env)
    assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")  # the proxy, which sends the call to the gateway
    assert fake.issuer not in env["ANTHROPIC_BASE_URL"]
    assert env["AUTHFI_AGENT_ID"] == AGENT
    assert env["AUTHFI_AGENT_KEY_FILE"] == str(key_file)
    assert env["EXTRA"] == "1"
    for name in CONTENT_SWITCHES:
        assert env[name] == "0"
    settings = json.loads(str(options.settings))
    assert settings["theme"] == "dark"
    assert settings["apiKeyHelper"] == PLACEHOLDER_KEY_COMMAND  # no token is minted for a process that cannot prove it
    assert options.permission_mode == "default"
    assert options.mcp_servers["authfi"]["type"] == "sdk"  # type: ignore[index]
    assert options.hooks is not None
    assert options.hooks["PreToolUse"]
    dumped = json.dumps({"env": env, "settings": settings}, default=str)
    assert "tok-" not in dumped
    assert "PRIVATE KEY" not in dumped


async def test_content_is_captured_only_when_asked(claude: ClaudeAgentIntegration) -> None:
    options = await claude.options(model="claude-haiku-5-5", capture_content=True, settings={"a": 1})
    assert all(dict(options.env)[name] == "1" for name in CONTENT_SWITCHES)


@pytest.mark.parametrize(
    ("given", "match"),
    [
        ({"permission_mode": "bypassPermissions"}, "bypassPermissions"),
        ({"env": {"ANTHROPIC_API_KEY": "x"}}, "ANTHROPIC_API_KEY"),
        ({"mcp_servers": {"authfi": {}}}, "name yours"),
        ({"settings": "/path/to/settings.json"}, "settings"),
        ({"settings": 3}, "settings"),
    ],
)
async def test_options_that_would_bypass_the_integration(
    claude: ClaudeAgentIntegration, given: dict[str, Any], match: str
) -> None:
    with pytest.raises(ConfigurationError, match=match):
        await claude.options(model="claude-haiku-5-5", **given)


def test_tracing_records_the_agent_and_not_content(
    claude: ClaudeAgentIntegration, monkeypatch: pytest.MonkeyPatch
) -> None:
    langsmith = pytest.importorskip("langsmith")  # optional: the claude-agent extra does not install it
    integration = pytest.importorskip("langsmith.integrations.claude_agent_sdk")

    seen: dict[str, Any] = {}

    def configure(**kwargs: Any) -> bool:
        seen.update(kwargs)
        return True

    monkeypatch.setattr(integration, "configure_claude_agent_sdk", configure)
    try:
        assert claude.configure_tracing(project_name="p", metadata={"case": "1"}) is True
        client = langsmith.run_trees.get_cached_client()
        assert client._hide_inputs is True
    finally:
        langsmith.configure(client=None)
    assert seen == {"metadata": {"case": "1", "authfi_agent_id": AGENT}, "project_name": "p"}


def test_tracing_without_langsmith_is_a_no_op_that_says_so(
    claude: ClaudeAgentIntegration, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "langsmith", None)
    monkeypatch.setitem(sys.modules, "langsmith.integrations.claude_agent_sdk", None)
    assert claude.configure_tracing() is False


# --- the caller cannot redirect the agent's token, and the parent's credentials do not reach Claude Code ---


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("ANTHROPIC_BASE_URL", "another host"),
        ("OTEL_LOG_USER_PROMPTS", "capture_content"),
        ("OTEL_LOG_RAW_API_BODIES", "capture_content"),
        ("AUTHFI_ISSUER", "set by the integration"),
        ("AUTHFI_AGENT_KEY_FILE", "set by the integration"),
        ("CLAUDE_CODE_API_KEY_HELPER_TTL_MS", "set by the integration"),
        ("ANTHROPIC_AUTH_TOKEN", "fixed credential"),
    ],
)
async def test_the_callers_env_cannot_override_what_the_integration_sets(
    claude: ClaudeAgentIntegration, name: str, match: str
) -> None:
    for spelled in (name, name.lower()):  # Windows environment names are case-insensitive
        with pytest.raises(ConfigurationError, match=match):
            await claude.options(model="claude-haiku-5-5", env={spelled: "https://elsewhere.example"})
        with pytest.raises(ConfigurationError, match=match):
            await claude.options(model="claude-haiku-5-5", settings={"env": {spelled: "x"}})


async def test_the_callers_other_env_is_kept_and_the_integrations_wins(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    options = await claude.options(model="claude-haiku-5-5", env={"HTTPS_PROXY": "http://proxy.example:3128"})
    env = dict(options.env)
    assert env["HTTPS_PROXY"] == "http://proxy.example:3128"
    assert env["ANTHROPIC_BASE_URL"].startswith("http://127.0.0.1:")


async def test_credentials_in_this_processes_environment_do_not_reach_claude_code(
    claude: ClaudeAgentIntegration, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import os

    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AUTHFI_AGENT_PRIVATE_KEY"):
        monkeypatch.setenv(name, "sk-inherited")
    with caplog.at_level("WARNING", logger="authfi"):
        options = await claude.options(model="claude-haiku-5-5")
    env = dict(options.env)
    assert [env[n] for n in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AUTHFI_AGENT_PRIVATE_KEY")] == ["", "", ""]
    # What the SDK starts Claude Code with: the parent's environment, then these options.
    child = {**os.environ, **env}
    assert "sk-inherited" not in child.values()
    assert "ANTHROPIC_API_KEY" in caplog.text and "sk-inherited" not in caplog.text


async def test_without_inherited_credentials_nothing_is_logged(
    claude: ClaudeAgentIntegration, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AUTHFI_AGENT_PRIVATE_KEY"):
        monkeypatch.delenv(name, raising=False)
    with caplog.at_level("WARNING", logger="authfi"):
        options = await claude.options(model="claude-haiku-5-5")
    assert dict(options.env)["ANTHROPIC_API_KEY"] == ""
    assert "is set in this process" not in caplog.text


# --- a decided call is served to the run that made it, once, soon ----------------------------------------


@pytest.fixture
async def timed(fake: FakeGateway, key_file: Path) -> AsyncIterator[tuple[ClaudeAgentIntegration, FakeClock]]:
    clock = FakeClock()
    async with AsyncAuthFI(fake.issuer, clock=clock) as project:
        yield ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1"))), clock


async def test_an_answer_nobody_collected_is_not_served_to_a_later_call(
    fake: FakeGateway, timed: tuple[ClaudeAgentIntegration, FakeClock]
) -> None:
    claude, clock = timed
    assert isinstance(await claude.can_use_tool(_name(ALLOWED), ARGS, CONTEXT), PermissionResultAllow)
    clock.advance(121)  # two minutes: Claude Code never ran the call: a hook denied it, the run stopped
    await claude.can_use_tool(_name(ALLOWED), ARGS, CONTEXT)
    answer = await _handler(claude, ALLOWED)(ARGS)
    assert answer["is_error"] is False
    assert len(fake.tool_calls) == 2  # the second call was sent afresh, not answered from the first


async def test_an_answer_is_served_within_its_time(
    fake: FakeGateway, timed: tuple[ClaudeAgentIntegration, FakeClock]
) -> None:
    claude, clock = timed
    await claude.can_use_tool(_name(ALLOWED), ARGS, CONTEXT)
    clock.advance(119)
    await _handler(claude, ALLOWED)(ARGS)
    assert len(fake.tool_calls) == 1


async def test_two_runs_with_the_same_call_each_execute(
    fake: FakeGateway, timed: tuple[ClaudeAgentIntegration, FakeClock]
) -> None:
    claude, _ = timed
    await claude.can_use_tool(_name(ALLOWED), ARGS, CONTEXT)  # run one, which ends before the handler runs
    claude.reset()
    await claude.can_use_tool(_name(ALLOWED), ARGS, CONTEXT)  # run two
    await _handler(claude, ALLOWED)(ARGS)
    assert len(fake.tool_calls) == 2


async def test_a_held_call_is_forgotten_after_its_time_too(
    fake: FakeGateway, timed: tuple[ClaudeAgentIntegration, FakeClock]
) -> None:
    claude, clock = timed
    first = await claude.can_use_tool(_name(STEP_UP), ARGS, CONTEXT)
    assert isinstance(first, PermissionResultDeny)
    clock.advance(121)
    fake_requests = len(fake.tool_calls)
    again = await claude.can_use_tool(_name(STEP_UP), ARGS, CONTEXT)
    assert isinstance(again, PermissionResultDeny)
    sent_with_old_approval = [args for _, args in fake.tool_calls[fake_requests:] if APPROVAL_ARGUMENT in args]
    assert sent_with_old_approval == []


def test_the_pending_approval_is_the_one_every_adapter_uses() -> None:
    from authfi.gateway import PendingApproval as Base

    assert PendingApproval is Base


# --- the helper is told where the machine APIs are ----------------------------------------------------


async def test_the_helper_is_told_the_machine_api_host_of_a_custom_domain_project(key_file: Path) -> None:
    with FakeGateway(api_origin="https://acme.authfi.io") as gateway:
        async with AsyncAuthFI(gateway.issuer) as project:
            claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
            assert "AUTHFI_API_ORIGIN" not in claude.helper_env()  # discovery has not been read yet
            options = await claude.options(model="claude-haiku-5-5")
    assert dict(options.env)["AUTHFI_API_ORIGIN"] == "https://acme.authfi.io"


async def test_the_helper_is_not_pinned_to_an_api_origin_when_there_is_nothing_to_say(
    claude: ClaudeAgentIntegration,
) -> None:
    options = await claude.options(model="claude-haiku-5-5")
    assert "AUTHFI_API_ORIGIN" not in dict(options.env)


# --- the agent's steps ----------------------------------------------------------------------------------


async def test_without_export_steps_claude_code_exports_nothing_to_authfi(claude: ClaudeAgentIntegration) -> None:
    options = await claude.options(model="claude-haiku-5-5")
    env = dict(options.env)
    assert env["OTEL_RESOURCE_ATTRIBUTES"] == f"authfi.agent_id={AGENT}"  # this gateway names no run
    assert "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT" not in env
    assert "otelHeadersHelper" not in json.loads(str(options.settings))


async def test_export_steps_points_claude_codes_trace_at_authfi_with_a_refreshing_token(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    pytest.importorskip("opentelemetry.proto")  # the proxy reads the export with the protocol types of extra otel
    options = await claude.options(model="claude-haiku-5-5", export_steps=True)
    env = dict(options.env)
    # through the loopback proxy, which names each span's agent and sends the export on to AuthFI
    assert env["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] == env["ANTHROPIC_BASE_URL"] + "/otlp/v1/traces"
    assert env["OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"] == "http/protobuf"
    assert env["OTEL_TRACES_EXPORTER"] == "otlp"
    assert env["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"
    assert env["CLAUDE_CODE_ENHANCED_TELEMETRY_BETA"] == "1"
    # Claude Code has under two seconds to export at shutdown, and its first export runs the headers helper (about a
    # second): spans go out while it runs, and the helper's answer is kept for as long as a token lasts.
    assert env["OTEL_TRACES_EXPORT_INTERVAL"] == "1000"
    assert env["CLAUDE_CODE_OTEL_HEADERS_HELPER_DEBOUNCE_MS"] == "240000"
    # another system's OTLP credential, inherited from the environment, must not go to AuthFI
    assert env["OTEL_EXPORTER_OTLP_HEADERS"] == ""
    assert env["OTEL_EXPORTER_OTLP_TRACES_HEADERS"] == ""
    assert all(env[name] == "0" for name in CONTENT_SWITCHES)  # prompts and tool content stay off
    settings = json.loads(str(options.settings))
    assert settings["otelHeadersHelper"].endswith("-m authfi headers")  # run again on every refresh
    assert "tok-" not in json.dumps({"env": env, "settings": settings}, default=str)


@pytest.mark.parametrize(
    "name",
    [
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_EXPORTER_OTLP_HEADERS",
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS",
        "OTEL_TRACES_EXPORTER",
        "OTEL_TRACES_EXPORT_INTERVAL",
        "CLAUDE_CODE_OTEL_HEADERS_HELPER_DEBOUNCE_MS",
        "CLAUDE_CODE_ENABLE_TELEMETRY",
        "CLAUDE_CODE_PROPAGATE_TRACEPARENT",
    ],
)
async def test_with_export_steps_the_trace_settings_are_the_integrations(
    claude: ClaudeAgentIntegration, name: str
) -> None:
    for spelled in (name, name.lower()):
        with pytest.raises(ConfigurationError, match="set by the integration"):
            await claude.options(model="claude-haiku-5-5", export_steps=True, env={spelled: "x"})
        with pytest.raises(ConfigurationError, match="set by the integration"):
            await claude.options(model="claude-haiku-5-5", export_steps=True, settings={"env": {spelled: "x"}})
    # without export_steps they are the caller's own
    options = await claude.options(model="claude-haiku-5-5", env={name: "mine"})
    assert dict(options.env)[name] == "mine"


async def test_export_steps_owns_the_headers_helper(claude: ClaudeAgentIntegration) -> None:
    with pytest.raises(ConfigurationError, match="otelHeadersHelper"):
        await claude.options(model="claude-haiku-5-5", export_steps=True, settings={"otelHeadersHelper": "/bin/x"})
    options = await claude.options(model="claude-haiku-5-5", settings={"otelHeadersHelper": "/bin/mine"})
    assert json.loads(str(options.settings))["otelHeadersHelper"] == "/bin/mine"


def test_a_helper_command_is_quoted_for_the_shell_that_runs_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from authfi import claude_agent

    path = r"C:\Program Files\Python\python.exe"
    monkeypatch.setattr("os.name", "nt")
    assert claude_agent._shell_word(path) == f'"{path}"'
    assert claude_agent._shell_word(r"C:\Python\python.exe") == r"C:\Python\python.exe"
    monkeypatch.setattr("os.name", "posix")
    assert claude_agent._shell_word(path) == f"'{path}'"


# --- the agent behind each exported span ------------------------------------------------------------------


def _export(*spans: dict[str, str]) -> bytes:
    """An OTLP trace export of spans with these string attributes, as Claude Code sends it."""
    pytest.importorskip("opentelemetry.proto")
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

    request = ExportTraceServiceRequest()
    scope = request.resource_spans.add().scope_spans.add()
    for number, attributes in enumerate(spans, start=1):
        span = scope.spans.add(name="claude_code.llm_request", span_id=number.to_bytes(8, "big"))
        span.trace_id = b"\x01" * 16
        for key, value in attributes.items():
            span.attributes.append(KeyValue(key=key, value=AnyValue(string_value=value)))
    return bytes(request.SerializeToString())


MAIN = {"session.id": "s1", "gen_ai.conversation.id": "s1"}
WIKI = {"session.id": "s1", "agent_id": "a1", "gen_ai.conversation.id": "a1"}
WIKI_BY_CONVERSATION = {"session.id": "s1", "gen_ai.conversation.id": "a1"}  # a span that does not say agent_id
STRANGER = {"session.id": "s1", "agent_id": "a9", "gen_ai.conversation.id": "a9"}
TRACES = "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"


async def _send_export(url: str, body: bytes, **headers: str) -> Any:
    import httpx

    async with httpx.AsyncClient() as http:
        return await http.post(
            url,
            content=body,
            headers={"Content-Type": "application/x-protobuf", "Authorization": "Bearer tok-claude-code", **headers},
        )


def _agent_names(fake: FakeGateway) -> list[str | None]:
    """``gen_ai.agent.name`` of each span AuthFI received, ``None`` where a span has none."""
    return [s["attributes"].get("gen_ai.agent.name") or None for s in fake.exported_spans()]


def _received(fake: FakeGateway) -> list[dict[str, Any]]:
    return [r for r in fake.requests if r["path"] == "/otlp/v1/traces"]


async def test_each_exported_span_names_its_agent_a_sub_agent_by_its_type_and_the_main_loop_by_its_name(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    options = await claude.options(model="claude-haiku-5-5", export_steps=True, agent_name="support")
    started: Any = options.hooks["SubagentStart"][0].hooks[0]  # type: ignore[index]
    await started({"hook_event_name": "SubagentStart", "agent_id": "a1", "agent_type": "wiki_researcher"}, None, {})

    answer = await _send_export(
        dict(options.env)[TRACES], _export(MAIN, WIKI, WIKI_BY_CONVERSATION), **{"X-Request-Id": "run-1"}
    )

    assert answer.status_code == 200
    assert _agent_names(fake) == ["support", "wiki_researcher", "wiki_researcher"]
    (sent,) = _received(fake)
    assert sent["headers"]["authorization"] == "Bearer tok-claude-code"  # Claude Code's own token, to AuthFI only
    assert sent["headers"]["x-request-id"] == "run-1"


async def test_the_main_loop_is_named_claude_code_unless_the_application_names_it(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    options = await claude.options(model="claude-haiku-5-5", export_steps=True)

    await _send_export(dict(options.env)[TRACES], _export(MAIN))

    assert _agent_names(fake) == ["claude_code"]
    with pytest.raises(ConfigurationError, match="agent_name"):
        await claude.options(model="claude-haiku-5-5", export_steps=True, agent_name=" ")


async def test_a_sub_agent_whose_start_was_never_seen_is_not_named_for_its_parent(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    options = await claude.options(model="claude-haiku-5-5", export_steps=True, agent_name="support")

    await _send_export(dict(options.env)[TRACES], _export(MAIN, STRANGER))

    assert _agent_names(fake) == [
        "support",
        "a9",
    ]  # its id, which is true; "support" would read its calls as the parent's


async def test_a_tool_hook_inside_a_sub_agent_teaches_its_type_when_its_start_was_missed(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    options = await claude.options(model="claude-haiku-5-5", export_steps=True)
    ask: Any = options.hooks["PreToolUse"][-1].hooks[0]  # type: ignore[index]

    await ask({"tool_name": "Read", "agent_id": "a1", "agent_type": "wiki_researcher"}, None, {})
    await _send_export(dict(options.env)[TRACES], _export(WIKI))

    assert _agent_names(fake) == ["wiki_researcher"]


async def test_a_span_that_already_names_its_agent_keeps_it(fake: FakeGateway, claude: ClaudeAgentIntegration) -> None:
    options = await claude.options(model="claude-haiku-5-5", export_steps=True)

    await _send_export(dict(options.env)[TRACES], _export({**MAIN, "gen_ai.agent.name": "theirs"}))

    assert _agent_names(fake) == ["theirs"]


async def test_a_compressed_export_is_read_and_sent_on_uncompressed(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    import gzip

    options = await claude.options(model="claude-haiku-5-5", export_steps=True, agent_name="support")

    await _send_export(dict(options.env)[TRACES], gzip.compress(_export(MAIN)), **{"Content-Encoding": "gzip"})

    assert _agent_names(fake) == ["support"]
    assert "content-encoding" not in _received(fake)[0]["headers"]


@pytest.mark.parametrize("encoding", ["", "br", "gzip"])
async def test_an_export_that_cannot_be_read_still_reaches_authfi_as_it_was(
    fake: FakeGateway, claude: ClaudeAgentIntegration, encoding: str
) -> None:
    options = await claude.options(model="claude-haiku-5-5", export_steps=True)
    extra = {"Content-Encoding": encoding} if encoding else {}

    answer = await _send_export(dict(options.env)[TRACES], b"\xff\xfe not an export", **extra)

    assert answer.status_code == 200
    (sent,) = _received(fake)
    assert sent["body"] == b"\xff\xfe not an export"
    assert sent["headers"].get("content-encoding", "") == encoding


async def test_without_export_steps_the_proxy_does_not_serve_the_trace_route(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    options = await claude.options(model="claude-haiku-5-5")
    base = dict(options.env)["ANTHROPIC_BASE_URL"]

    answer = await _send_export(base + "/otlp/v1/traces", _export(MAIN))

    assert answer.status_code == 404
    assert not _received(fake)


async def test_only_the_trace_export_is_served_beside_the_model_call(
    fake: FakeGateway, claude: ClaudeAgentIntegration
) -> None:
    options = await claude.options(model="claude-haiku-5-5", export_steps=True)
    base = dict(options.env)["ANTHROPIC_BASE_URL"]

    refused = [
        await _send_export(base + "/otlp/v1/metrics", _export(MAIN)),
        await _send_export(base + "/otlp/v1/traces?x=1", _export(MAIN)),
        await _send_export(base + "/otlp/v1/traces/", _export(MAIN)),
        await _send_export(base.rsplit("/", 1)[0] + "/wrong/otlp/v1/traces", _export(MAIN)),
    ]

    assert [r.status_code for r in refused] == [404] * 4
    assert not [r for r in fake.requests if r["path"].startswith("/otlp")]


async def test_a_hook_input_that_is_not_an_agent_and_its_type_teaches_nothing(claude: ClaudeAgentIntegration) -> None:
    options = await claude.options(model="claude-haiku-5-5", export_steps=True)
    started: Any = options.hooks["SubagentStart"][0].hooks[0]  # type: ignore[index]

    for given in (
        {},
        {"agent_id": "a1"},
        {"agent_type": "x"},
        {"agent_id": 1, "agent_type": "x"},
        {"agent_id": "", "agent_type": "x"},
    ):
        await started(given, None, {})

    assert claude._names.name_of("a1") == "a1"  # pyright: ignore[reportPrivateUsage]


async def test_without_the_protocol_types_the_trace_goes_to_authfi_unchanged_and_the_log_says_so(
    fake: FakeGateway, claude: ClaudeAgentIntegration, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("authfi.claude_agent.agent_names_available", lambda: False)

    with caplog.at_level("WARNING", logger="authfi"):
        options = await claude.options(model="claude-haiku-5-5", export_steps=True)

    assert dict(options.env)[TRACES] == fake.issuer + "/otlp/v1/traces"
    assert "pip install 'authfi[otel] @ git+https://github.com/authfi/authfi-python-sdk@main'" in caplog.text


def test_an_export_is_read_whatever_encoding_the_exporter_chose() -> None:
    import gzip
    import zlib

    from authfi._claude_steps import AgentNames

    names = AgentNames("support")
    body = _export(MAIN)

    for encoded, encoding in (
        (body, ""),
        (body, "identity"),
        (gzip.compress(body), "GZIP"),
        (zlib.compress(body), "deflate"),
    ):
        rewritten = names.rewrite(encoded, encoding)
        assert rewritten is not None and b"support" in rewritten
    assert names.rewrite(gzip.compress(body), "br") is None  # an encoding it does not read
    assert names.rewrite(b"not gzip", "gzip") is None
    assert names.rewrite(b"\xff\xfe", "") is None


def test_it_remembers_a_bounded_number_of_sub_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    from authfi import _claude_steps
    from authfi._claude_steps import AgentNames

    monkeypatch.setattr(_claude_steps, "_MAX_AGENTS", 2)
    names = AgentNames()
    for number in range(3):
        names.learn(f"a{number}", f"type{number}")

    assert [names.name_of(f"a{number}") for number in range(3)] == ["a0", "type1", "type2"]  # the oldest is forgotten
