"""authfi.claude_agent: the permission callback, the in-process MCP server and the options.

The gateway is the real-HTTP stand-in; Claude Code itself is not started.
"""

from __future__ import annotations

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
    import langsmith
    import langsmith.integrations.claude_agent_sdk as integration  # pyright: ignore[reportMissingTypeStubs]

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
