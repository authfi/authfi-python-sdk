"""The Claude Agent SDK: approvals through ``can_use_tool``, credentials that renew themselves, content off.

``pip install "authfi[claude-agent] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from claude_agent_sdk import ClaudeSDKClient
    from authfi import AsyncAuthFI
    from authfi.claude_agent import ClaudeAgentIntegration

    af = AsyncAuthFI.from_env()
    claude = ClaudeAgentIntegration(af.agent_from_env(), wait_for_decision=until_decided)
    options = await claude.options(model="claude-haiku-5-5")
    async with ClaudeSDKClient(options) as client:
        await client.query("Describe pallets/flask's wiki")
        async for message in client.receive_response():
            ...

- **The model.** Claude Code calls the AI gateway with a token from ``python -m authfi token`` (its
  ``apiKeyHelper``), run again before the token expires. The helper signs as the agent from the
  agent's key file; neither the key nor a token is put in Claude Code's options or environment.
- **The tools.** The gateway's tools are served to Claude Code by an MCP server in this process,
  which calls the gateway with the current token on every request. Tool names are the gateway's,
  under the server's name (``mcp__authfi__<server>__<tool>``).
- **Approvals.** Claude Code asks ``can_use_tool`` before each of the gateway's tools runs; the
  options set the permission mode and a hook that make sure it does. ``can_use_tool`` sends the
  call to the gateway. When the gateway holds it for a person's approval, ``wait_for_decision`` is
  awaited, and the call is sent again with the approval; the gateway decides. The approval id never
  reaches the model.
- **Content.** Claude Code's OpenTelemetry content logging (prompts, answers, tool details and
  content, raw API bodies) is off unless ``capture_content``.

A token held for a person is not supported here: Claude Code could only receive it as a fixed value
in its environment.
"""

from __future__ import annotations

import inspect
import json
import os
import shlex
import sys
from collections import OrderedDict
from collections.abc import AsyncGenerator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

try:
    import httpx
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        HookContext,
        HookInput,
        HookJSONOutput,
        HookMatcher,
        McpSdkServerConfig,
        PermissionResultAllow,
        PermissionResultDeny,
        SdkMcpTool,
        ToolPermissionContext,
        create_sdk_mcp_server,
    )
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    from mcp.types import CallToolResult, Tool
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.claude_agent needs the Claude Agent SDK: "
        "pip install 'authfi[claude-agent] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import async_gateway_for, configure_langsmith
from authfi._protocol.urls import origin_of
from authfi.agents.agent import AsyncAgentAuth, AsyncDelegatedToken, DelegatedToken
from authfi.errors import ConfigurationError, GatewayTokenRefused, NotSupportedError
from authfi.gateway import HOLD_MESSAGE, PendingApproval, ToolApprovalRequired, ToolRefused, token_refusal
from authfi.tokens import AsyncTokenSource

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI

__all__ = ["ClaudeAgentIntegration", "PendingApproval"]

#: Claude Code's OpenTelemetry switches for content. Each is set, so an inherited "1" does not turn one on.
CONTENT_SWITCHES = (
    "OTEL_LOG_USER_PROMPTS",
    "OTEL_LOG_ASSISTANT_RESPONSES",
    "OTEL_LOG_TOOL_CONTENT",
    "OTEL_LOG_TOOL_DETAILS",
    "OTEL_LOG_RAW_API_BODIES",
)

#: How long Claude Code keeps a token from its ``apiKeyHelper`` before running it again: well inside
#: an agent token's lifetime of minutes.
KEY_HELPER_TTL_MS = 240_000

#: Environment that would hand Claude Code a fixed credential; the helper is the only way in. Claude
#: Code inherits this process's environment, so each is also blanked in the environment it is given.
_FIXED_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AUTHFI_AGENT_PRIVATE_KEY")

#: Environment the integration sets and the caller may not: a different value would send the agent's
#: token to another host, run another helper, or switch content on.
_OWNED = (
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_API_KEY_HELPER_TTL_MS",
    "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
    "OTEL_RESOURCE_ATTRIBUTES",
)
_OWNED_PREFIXES = ("AUTHFI_", "OTEL_LOG_")

#: How many held or decided calls the integration remembers until Claude Code runs them.
MEMORY = 10_000

#: How long it remembers one. A call nobody ran is forgotten, so a later identical call is sent afresh.
MEMORY_SECONDS = 120.0

#: Returns, or resolves, once the person has decided; its value is not read: the gateway decides.
WaitForDecision = Callable[["PendingApproval"], object]
CanUseTool = Callable[
    [str, dict[str, Any], ToolPermissionContext], Awaitable[PermissionResultAllow | PermissionResultDeny]
]


@dataclass(frozen=True, slots=True)
class _Held:
    approval_id: str


@dataclass(frozen=True, slots=True)
class _Decided:
    answer: dict[str, Any]


def _key(tool: str, args: Mapping[str, Any]) -> str:
    return tool + "\x00" + json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


def _text(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "is_error": True}


class ClaudeAgentIntegration:
    """The AuthFI gateways for the Claude Agent SDK.

    Args:
        source: the agent, ``AsyncAuthFI.agent(...)`` with its key loaded from a file
            (``AgentKey.from_file`` or ``AUTHFI_AGENT_KEY_FILE``): Claude Code runs
            ``python -m authfi token``, which signs with that file.
        wait_for_decision: awaited with a :class:`PendingApproval` when the gateway holds a call;
            return once the person has decided in AuthFI. Without it, a held call is refused for now
            with "This action is waiting for a person's approval in AuthFI.", and the same call made later is
            sent with the approval.
        project: the project, for a credential of your own that does not name it.
        server_name: the name of the gateway's MCP server in Claude Code.

    """

    def __init__(
        self,
        source: AsyncAgentAuth,
        *,
        wait_for_decision: WaitForDecision | None = None,
        project: AsyncAuthFI | None = None,
        server_name: str = "authfi",
    ) -> None:
        given = cast(object, source)
        if isinstance(given, DelegatedToken | AsyncDelegatedToken):
            raise NotSupportedError(
                "claude_agent_delegated",
                "a token held for a person cannot be given to Claude Code: it would be a fixed value in its "
                "environment. Run Claude Code as the agent itself.",
            )
        if not isinstance(given, AsyncAgentAuth):
            raise ConfigurationError("ClaudeAgentIntegration takes the agent from AsyncAuthFI.agent(...)")
        if getattr(source.key, "path", None) is None:
            raise ConfigurationError(
                "Claude Code gets the agent's token by running `python -m authfi token`, which signs with the "
                "agent's key file. Load the key with AgentKey.from_file (or AUTHFI_AGENT_KEY_FILE); the SDK does "
                "not copy a private key into another process."
            )
        self.source = source
        self.gateway = async_gateway_for(cast(AsyncTokenSource, source), project)
        self.agent_id = source.agent_id
        self.wait_for_decision = wait_for_decision
        self.server_name = server_name
        self.prefix = f"mcp__{server_name}__"
        #: The gateway's refusal of the token, once it happened: the run was stopped.
        self.token_refused: GatewayTokenRefused | None = None
        self._calls: OrderedDict[str, tuple[_Held | _Decided, float]] = OrderedDict()
        self._other: CanUseTool | None = None

    # --- the gateway's tools --------------------------------------------------------------------------

    @asynccontextmanager
    async def _session(self) -> AsyncGenerator[ClientSession, None]:
        factory = self.gateway.mcp_client_factory(httpx)
        async with (
            factory(None, httpx.Timeout(30.0, read=300.0)) as http_client,
            streamable_http_client(self.gateway.mcp_url, http_client=http_client) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            yield session

    async def tools(self, *, include_never_allowed: bool = False) -> list[Tool]:
        """The gateway's tools for this agent; those marked ``never_allowed`` are left out unless asked."""
        try:
            async with self._session() as session:
                listed = await session.list_tools()
        except Exception as error:
            raise self._refused(error) from error
        out: list[Tool] = []
        for tool in listed.tools:
            if self.gateway.governance_of(tool) == "never_allowed" and not include_never_allowed:
                self.gateway.project.logger.info("leaving out %s: never allowed for this agent", tool.name)
                continue
            out.append(tool)
        return out

    def _refused(self, error: Exception) -> Exception:
        refused = token_refusal(error)
        if refused is None:
            return error
        self.token_refused = refused
        return refused

    async def _send(self, tool: str, args: dict[str, Any], approval_id: str | None) -> _Held | _Decided:
        sent = self.gateway.with_approval(args, approval_id) if approval_id else args
        try:
            async with self._session() as session:
                result: CallToolResult = await session.call_tool(tool, sent)
        except Exception as error:
            raise self._refused(error) from error
        outcome = self.gateway.read_result(result)
        if isinstance(outcome, ToolApprovalRequired):
            self.gateway.project.logger.info("%s waits for approval %s", tool, outcome.approval_id)
            return _Held(outcome.approval_id)
        if isinstance(outcome, ToolRefused):
            self.gateway.project.logger.info("the gateway refused %s: %s", tool, outcome.code or outcome.error)
            return _Decided(_text(outcome.model_text))
        content = [block.model_dump(by_alias=True, exclude_none=True) for block in result.content]
        return _Decided({"content": content, "is_error": bool(result.isError)})

    def _remember(self, key: str, entry: _Held | _Decided) -> None:
        self._calls[key] = (entry, self.gateway.project.clock.monotonic())
        self._calls.move_to_end(key)
        while len(self._calls) > MEMORY:
            self._calls.popitem(last=False)

    def _recall(self, key: str, *, take: bool = False) -> _Held | _Decided | None:
        """What was remembered for this call, unless it is older than :data:`MEMORY_SECONDS`."""
        found = self._calls.pop(key, None) if take else self._calls.get(key)
        if found is None:
            return None
        entry, at = found
        if self.gateway.project.clock.monotonic() - at > MEMORY_SECONDS:
            self._calls.pop(key, None)
            return None
        return entry

    def reset(self) -> None:
        """Forget every remembered call. Call it when a run ends, so the next run never inherits an answer."""
        self._calls.clear()

    def _handler(self, tool: str) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
        async def run(arguments: dict[str, Any]) -> dict[str, Any]:
            args = self.gateway.strip_approval_argument(arguments)
            key = _key(tool, args)
            known = self._recall(key, take=True)
            if isinstance(known, _Decided):  # can_use_tool sent it and the gateway decided
                return known.answer
            try:
                entry = await self._send(tool, args, known.approval_id if known else None)
            except GatewayTokenRefused as refused:
                return _text(str(refused))
            if isinstance(entry, _Held):
                self._remember(key, entry)
                return _text(HOLD_MESSAGE)
            return entry.answer

        return run

    async def mcp_server(self, *, include_never_allowed: bool = False) -> McpSdkServerConfig:
        """The gateway's tools as an MCP server in this process, for ``ClaudeAgentOptions.mcp_servers``."""
        tools: list[SdkMcpTool[Any]] = [
            SdkMcpTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=dict(tool.inputSchema),
                handler=self._handler(tool.name),
            )
            for tool in await self.tools(include_never_allowed=include_never_allowed)
        ]
        return create_sdk_mcp_server(self.server_name, tools=tools)

    # --- approvals ------------------------------------------------------------------------------------

    async def can_use_tool(
        self, tool_name: str, tool_input: dict[str, Any], context: ToolPermissionContext
    ) -> PermissionResultAllow | PermissionResultDeny:
        """Claude Code's permission callback; :meth:`options` installs it.

        For one of the gateway's tools, the call is sent to the gateway now. Once the gateway has
        decided (at once, or after ``wait_for_decision``), the call is allowed and Claude Code receives
        the gateway's answer, the tool's result or the refusal, without the call being sent again. A
        call still waiting is refused for now. Other tools go to the ``can_use_tool`` given to
        :meth:`options`, or are refused. A refused token stops the run.
        """
        if not tool_name.startswith(self.prefix):
            if self._other is not None:
                return await self._other(tool_name, tool_input, context)
            return PermissionResultDeny(
                message=f"{tool_name} is not one of AuthFI's tools, and nothing here decides it"
            )
        tool = tool_name[len(self.prefix) :]
        args = self.gateway.strip_approval_argument(tool_input)
        key = _key(tool, args)
        known = self._recall(key)
        try:
            if not isinstance(known, _Decided):
                known = await self._send(tool, args, known.approval_id if known else None)
            if isinstance(known, _Held) and self.wait_for_decision is not None:
                decided = self.wait_for_decision(PendingApproval(known.approval_id, tool, args))
                if inspect.isawaitable(decided):
                    await decided
                known = await self._send(tool, args, known.approval_id)
        except GatewayTokenRefused as refused:
            return PermissionResultDeny(message=str(refused), interrupt=True)
        self._remember(key, known)
        if isinstance(known, _Held):
            return PermissionResultDeny(message=HOLD_MESSAGE)
        return PermissionResultAllow(updated_input=args)

    async def _ask(self, input_data: HookInput, tool_use_id: str | None, context: HookContext) -> HookJSONOutput:
        """Send the gateway's tools to the permission flow, so no allow rule skips ``can_use_tool``."""
        name = cast(Mapping[str, Any], input_data).get("tool_name")
        if not isinstance(name, str) or not name.startswith(self.prefix):
            return {}
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "ask",
                "permissionDecisionReason": "AuthFI governs this tool",
            }
        }

    # --- options ----------------------------------------------------------------------------------------

    def helper_env(self) -> dict[str, str]:
        """What ``python -m authfi token`` needs to sign as this agent: names and a path, never the key.

        The API origin is the one discovery has taught the project: :meth:`options` reads discovery first.
        """
        project = self.gateway.project
        key_path = cast(object, getattr(self.source.key, "path", None))
        env = {
            "AUTHFI_ISSUER": project.issuer,
            "AUTHFI_AGENT_ID": self.agent_id,
            "AUTHFI_AGENT_KEY_FILE": str(key_path),
            "AUTHFI_AGENT_KID": self.source.key.kid,
        }
        if project.api_origin != origin_of(project.issuer):
            # A project on a custom domain serves its machine APIs elsewhere; the helper is told where.
            env["AUTHFI_API_ORIGIN"] = project.api_origin
        if self.source.resource:
            env["AUTHFI_AGENT_RESOURCE"] = self.source.resource
        return env

    async def options(
        self, *, model: str, capture_content: bool = False, include_never_allowed: bool = False, **claude_options: Any
    ) -> ClaudeAgentOptions:
        """``ClaudeAgentOptions`` for this agent on the gateways.

        ``claude_options`` are ``ClaudeAgentOptions``'s. ``env``, ``settings`` (a dict or JSON),
        ``hooks`` and ``mcp_servers`` are merged with the integration's. ``can_use_tool`` decides the
        tools that are not AuthFI's (by default they are refused). A credential in ``env`` is refused,
        and so is ``permission_mode="bypassPermissions"``, which would skip approvals.
        """
        await self.gateway.project.resolve_api_origin()
        if claude_options.get("permission_mode") == "bypassPermissions":
            raise ConfigurationError(
                "permission_mode bypassPermissions skips can_use_tool, and with it AuthFI approvals"
            )
        given_env = dict(claude_options.pop("env", None) or {})
        _refuse_owned(given_env, "env")
        inherited = [name for name in _FIXED_CREDENTIALS if os.environ.get(name)]
        if inherited:
            self.gateway.project.logger.warning(
                "%s is set in this process's environment; Claude Code is not given it: the agent's token comes "
                "from the key file",
                ", ".join(inherited),
            )
        env = {
            **given_env,
            # Claude Code starts with this process's environment under these: blank, so none reaches it.
            **dict.fromkeys(_FIXED_CREDENTIALS, ""),
            "ANTHROPIC_BASE_URL": self.gateway.model_base_url("anthropic"),
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
            "ANTHROPIC_SMALL_FAST_MODEL": model,
            "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": str(KEY_HELPER_TTL_MS),
            # The AI gateway serves the models the project configures; Anthropic's experimental
            # beta headers are not passed through to them.
            "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            "OTEL_RESOURCE_ATTRIBUTES": f"authfi.agent_id={self.agent_id}",
            **self.helper_env(),
            **{name: "1" if capture_content else "0" for name in CONTENT_SWITCHES},
        }
        settings = _settings(claude_options.pop("settings", None))
        _refuse_owned(settings.get("env"), "settings['env']")
        settings["apiKeyHelper"] = f"{shlex.quote(sys.executable)} -m authfi token"
        servers = dict(claude_options.pop("mcp_servers", None) or {})
        if self.server_name in servers:
            raise ConfigurationError(
                f"the MCP server {self.server_name!r} is the integration's; name yours differently"
            )
        servers[self.server_name] = await self.mcp_server(include_never_allowed=include_never_allowed)
        hooks = dict(claude_options.pop("hooks", None) or {})
        hooks["PreToolUse"] = [*hooks.get("PreToolUse", []), HookMatcher(matcher=None, hooks=[self._ask])]
        self._other = claude_options.pop("can_use_tool", None)
        claude_options.setdefault("permission_mode", "default")
        return ClaudeAgentOptions(
            model=model,
            env=env,
            settings=json.dumps(settings),
            mcp_servers=servers,
            hooks=hooks,
            can_use_tool=self.can_use_tool,
            **claude_options,
        )

    def configure_tracing(self, *, capture_content: bool = False, **kwargs: Any) -> bool:
        """Turn on LangSmith's Claude Agent SDK tracing, recording metadata and not content.

        LangSmith's integration records the prompt, the system prompt and every tool's input and
        result by default. This installs a LangSmith client that hides inputs and outputs unless
        ``capture_content``, and puts ``authfi_agent_id`` in each run's metadata. ``kwargs`` go to
        ``configure_claude_agent_sdk`` (``project_name``, ``tags``, ``name``). Returns ``False`` when LangSmith
        (or its Claude Agent SDK integration) is not installed.
        """
        try:
            from langsmith.integrations.claude_agent_sdk import (  # pyright: ignore[reportMissingTypeStubs]
                configure_claude_agent_sdk,  # pyright: ignore[reportUnknownVariableType]
            )
        except ImportError:
            return False  # LangSmith is optional: without it nothing is traced to it

        metadata = dict(kwargs.pop("metadata", None) or {})
        metadata["authfi_agent_id"] = self.agent_id
        configure_langsmith(capture_content=capture_content)
        return bool(configure_claude_agent_sdk(metadata=metadata, **kwargs))


def _refuse_owned(env: object, where: str) -> None:
    """Refuse an environment variable that would turn the agent's token, or the content switches, elsewhere."""
    if not isinstance(env, Mapping):
        return
    for given in cast(Mapping[str, object], env):
        name = str(given)
        upper = name.upper()  # Windows environment names are case-insensitive, so a lowercase spelling is the same
        if upper in _FIXED_CREDENTIALS:
            raise ConfigurationError(
                f"{name} in {where} is a fixed credential; Claude Code gets the agent's token itself"
            )
        if upper == "ANTHROPIC_BASE_URL":
            raise ConfigurationError(
                f"{name} in {where} would send the agent's token to another host: Claude Code is "
                "pointed at the AuthFI gateway by this integration"
            )
        if upper.startswith("OTEL_LOG_"):
            raise ConfigurationError(f"{name} in {where} is set from capture_content, not by hand")
        if upper in _OWNED or upper.startswith(_OWNED_PREFIXES):
            raise ConfigurationError(f"{name} in {where} is set by the integration")


def _settings(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(cast(Mapping[str, Any], value))
    if isinstance(value, str):
        try:
            parsed: object = json.loads(value)
        except ValueError:
            parsed = None
        if isinstance(parsed, Mapping):
            return dict(cast(Mapping[str, Any], parsed))
    raise ConfigurationError("settings must be a dict or a JSON object to merge apiKeyHelper into")
