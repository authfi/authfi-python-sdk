"""authfi.claude_agent: the permission callback, the in-process MCP server and the options.

The gateway is the real-HTTP stand-in; Claude Code itself is not started.
"""

from __future__ import annotations

import sys
import json
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
from authfi.claude_agent import CONTENT_SWITCHES, ClaudeAgentIntegration, PendingApproval
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
    assert env["ANTHROPIC_BASE_URL"] == fake.issuer + "/mcp"
    assert env["AUTHFI_AGENT_ID"] == AGENT
    assert env["AUTHFI_AGENT_KEY_FILE"] == str(key_file)
    assert env["EXTRA"] == "1"
    for name in CONTENT_SWITCHES:
        assert env[name] == "0"
    settings = json.loads(str(options.settings))
    assert settings["theme"] == "dark"
    assert settings["apiKeyHelper"].endswith("-m authfi token")
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
    assert env["ANTHROPIC_BASE_URL"] == fake.issuer + "/mcp"


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
