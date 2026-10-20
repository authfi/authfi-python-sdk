"""The Claude Agent SDK on AuthFI: approvals through can_use_tool, credentials that refresh, content off.

``pip install 'authfi[claude-agent]'``

::

    from authfi import AgentAuth, AgentKey
    from authfi.claude_agent import AuthFIClaudeClient, claude_options

    agent = AgentAuth.from_env()           # the key must be in a file: AUTHFI_AGENT_KEY_FILE
    options = claude_options(agent, model="claude-haiku-5-5", wait_for_decision=my_wait)
    async with AuthFIClaudeClient(options, agent=agent) as client:
        await client.query("Describe pallets/flask's wiki")
        async for message in client.receive_response():
            ...

:func:`claude_options` builds ``ClaudeAgentOptions`` for the project:

- **Model credential.** Claude Code calls the AI gateway (``<project>/mcp``)
  with a token from ``authfi token`` as its ``apiKeyHelper``, re-run before
  the token expires. The helper signs as the agent from the agent's key file;
  the key itself is never handed to Claude Code.
- **MCP credential.** The gateway is an MCP server whose headers come from
  ``authfi headers`` (``headersHelper``). Claude Code runs it when it
  connects, and :class:`AuthFIClaudeClient` reconnects the server before the
  token expires, so the header is fresh for a long session.
- **Approvals.** Claude Code consults ``can_use_tool`` only when a permission
  mode asks it to; with none, its own rules decide and the approval code never
  runs. The options set ``permission_mode="default"`` and a ``PreToolUse``
  hook that sends the gateway's tools to the permission flow.
  :class:`AuthFIApprovals` notes a held call in ``PostToolUse`` /
  ``PostToolUseFailure``; when the model calls it again, ``can_use_tool``
  waits for the person's decision and lets the call go with the approval id.
- **Content.** Claude Code's OpenTelemetry content logging (prompts,
  answers, tool details and content, raw API bodies) is off unless
  ``capture_content``; the agent id and run id are resource attributes.
  :func:`trace_with_langsmith` does the same for LangSmith's integration.
- **Run id.** Every gateway call carries ``X-Request-Id`` set to the run id.

A token acting for a person (a :class:`authfi.DelegatedToken` from CIBA) is
passed as a fixed value: it cannot be renewed anyway, because the person's
approval was for that token.
"""

import asyncio
import inspect
import json
import logging
import os
import shlex
import sys
import time
import uuid

try:
    from claude_agent_sdk import (
        ClaudeAgentOptions,
        ClaudeSDKClient,
        HookMatcher,
        PermissionResultAllow,
        PermissionResultDeny,
    )
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("authfi.claude_agent needs the Claude Agent SDK: pip install 'authfi[claude-agent]'") from e

from . import APPROVAL_PENDING, AgentAuth, AsyncAgentAuth, AuthFIError, DelegatedToken
from ._tracing import langsmith_metadata_only
from .gateway import call_digest, credential, model_base_url, read_result, request_id, with_approval, without_approval

_log = logging.getLogger("authfi")

# Claude Code's OpenTelemetry switches for content. Each defaults to off in
# Claude Code; they are set here so an inherited "1" does not turn one on.
_CONTENT_SWITCHES = ("OTEL_LOG_USER_PROMPTS", "OTEL_LOG_ASSISTANT_RESPONSES", "OTEL_LOG_TOOL_CONTENT",
                     "OTEL_LOG_TOOL_DETAILS", "OTEL_LOG_RAW_API_BODIES")

# How long Claude Code keeps the apiKeyHelper's token before running it again.
# Agent tokens live for minutes; this keeps the one in use well inside that.
_KEY_HELPER_TTL_MS = 240000

_DECIDED_NO = ("denied", "expired")


def _helper(command):
    return "%s -m authfi %s" % (shlex.quote(sys.executable), command)


def _helper_env(agent, run_id):
    """What ``authfi token`` / ``authfi headers`` need to sign as ``agent``, without the key itself."""
    env = {"AUTHFI_PLATFORM_URL": agent.platform_url, "AUTHFI_AGENT_ID": agent.agent_id,
           "AUTHFI_AGENT_KID": agent.key.key_id, "AUTHFI_RUN_ID": run_id}
    if agent.key.path:
        env["AUTHFI_AGENT_KEY_FILE"] = agent.key.path
    elif not os.environ.get("AUTHFI_AGENT_PRIVATE_KEY"):
        raise AuthFIError(
            "Claude Code gets the agent's token by running `authfi token`, which signs with the "
            "agent's key file. Load the key with AgentKey.from_file (or AUTHFI_AGENT_KEY_FILE); a key "
            "built in memory cannot be reached from that process, and this SDK does not copy a "
            "private key into another process's environment.",
            status=400,
        )
    if agent.audience:
        env["AUTHFI_AGENT_AUDIENCE"] = agent.audience
    return env


def claude_options(agent, model="claude-haiku-5-5", url=None, server_name="authfi", approvals=None,
                   wait_for_decision=None, capture_content=False, run_id=None, **options):
    """``ClaudeAgentOptions`` for an agent on the project's AuthFI gateway.

    ``agent`` is an :class:`authfi.AgentAuth` (or ``AsyncAgentAuth``), or a
    :class:`authfi.DelegatedToken` to act for a person. ``approvals`` is an
    :class:`AuthFIApprovals`; by default one is made with
    ``wait_for_decision``. ``run_id`` names this run (a new UUID by
    default); it is sent as ``X-Request-Id`` and stamped on telemetry.
    Every other keyword is ``ClaudeAgentOptions``'s; ``env``, ``hooks`` and
    ``mcp_servers`` you pass are merged with these, and ``env`` you pass wins.
    """
    run_id = request_id(run_id) or str(uuid.uuid4())
    if isinstance(agent, AsyncAgentAuth):
        agent = agent.agent
    base_url = url
    env = {
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
        "ANTHROPIC_SMALL_FAST_MODEL": model,
        # The project's gateway serves the models it is configured for, and the
        # edge does not pass Anthropic's experimental beta headers through.
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        "ANTHROPIC_CUSTOM_HEADERS": "X-Request-Id: %s" % run_id,
    }
    attributes = ["authfi.run_id=%s" % run_id]
    server = {"type": "http"}
    settings = _settings(options.pop("settings", None))
    if isinstance(agent, AgentAuth):
        base_url = model_base_url(credential(agent), url)
        env.update(_helper_env(agent, run_id))
        env["CLAUDE_CODE_API_KEY_HELPER_TTL_MS"] = str(_KEY_HELPER_TTL_MS)
        settings["apiKeyHelper"] = _helper("token")
        server["headersHelper"] = _helper("headers")
        attributes.insert(0, "authfi.agent_id=%s" % agent.agent_id)
    elif isinstance(agent, DelegatedToken):
        if not url:
            raise AuthFIError("url is required with a DelegatedToken: https://<project>.authfi.io/mcp", status=400)
        token = agent.token()
        env["ANTHROPIC_AUTH_TOKEN"] = token
        server["headers"] = {"Authorization": "Bearer " + token, "X-Request-Id": run_id}
    else:
        raise AuthFIError(
            "claude_options takes an AgentAuth (Claude Code then refreshes the token itself) or a "
            "DelegatedToken; got %s" % type(agent).__name__,
            status=400,
        )
    env["ANTHROPIC_BASE_URL"] = base_url
    server["url"] = base_url
    if not capture_content:
        env.update({name: "0" for name in _CONTENT_SWITCHES})
    env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(attributes)
    env.update(options.pop("env", None) or {})

    if approvals is None:
        approvals = AuthFIApprovals(wait_for_decision=wait_for_decision, agent=agent if isinstance(agent, AgentAuth)
                                    else None, server_name=server_name)
    hooks = dict(options.pop("hooks", None) or {})
    for event, matchers in approvals.hooks().items():
        hooks[event] = list(hooks.get(event) or []) + matchers
    mcp_servers = dict(options.pop("mcp_servers", None) or {})
    mcp_servers[server_name] = server
    options.setdefault("permission_mode", "default")
    if options["permission_mode"] == "bypassPermissions":
        _log.warning("permission_mode bypassPermissions: Claude Code will not ask can_use_tool, so AuthFI "
                     "approvals cannot pause a held call")
    return ClaudeAgentOptions(model=model, env=env, settings=json.dumps(settings) if settings else None,
                              mcp_servers=mcp_servers, can_use_tool=approvals.can_use_tool, hooks=hooks, **options)


def _settings(value):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(value)
    except ValueError:
        raise AuthFIError("settings must be a dict or a JSON object to merge apiKeyHelper into; "
                          "a settings file path cannot be merged", status=400)
    if not isinstance(parsed, dict):
        raise AuthFIError("settings must be a JSON object", status=400)
    return parsed


class AuthFIApprovals:
    """Approvals for the gateway's tools through Claude Code's own permission hook.

    The gateway decides a held call when it arrives, inside the tool's
    execution; ``can_use_tool`` runs before execution. So the work is split:

    1. ``PostToolUse`` / ``PostToolUseFailure`` read the gateway's answer and,
       for a hold, note the approval for that exact tool and arguments, and
       tell the model to call the tool again;
    2. when it does, ``can_use_tool`` awaits ``wait_for_decision(approval)``
       and lets the call go with the approval id (``updated_input``). The
       model never has to copy the id.

    ``wait_for_decision(approval)`` receives ``{"approval_id", "tool",
    "args"}`` and returns once a person has decided: ``True``, or the
    approval record. A record or ``False`` that says denied or expired denies
    the call here; anything else lets the gateway decide. Without one, the
    agent reads its approval itself (``wait_for_approval``), which needs the
    project to let agents read their own approvals.

    ``other_tools`` decides calls to tools that are not the gateway's:
    ``"deny"`` (the default, so nothing runs that nobody decided), ``"allow"``,
    or a ``can_use_tool`` function of your own.
    """

    def __init__(self, wait_for_decision=None, agent=None, server_name="authfi", other_tools="deny", timeout=600):
        if wait_for_decision is None and agent is None:
            raise AuthFIError("AuthFIApprovals needs wait_for_decision, or the agent to read its approvals with",
                              status=400)
        self.wait_for_decision = wait_for_decision
        self.agent = AsyncAgentAuth(agent) if isinstance(agent, AgentAuth) else agent
        self.prefix = "mcp__%s__" % server_name
        self.other_tools = other_tools
        self.timeout = timeout
        self.held = {}

    def governed(self, tool_name):
        return tool_name.startswith(self.prefix)

    def hooks(self):
        return {
            "PreToolUse": [HookMatcher(matcher=None, hooks=[self.pre_tool])],
            "PostToolUse": [HookMatcher(matcher=None, hooks=[self.post_tool])],
            "PostToolUseFailure": [HookMatcher(matcher=None, hooks=[self.post_tool])],
        }

    async def pre_tool(self, input_data, tool_use_id, context):
        """Send every gateway tool to the permission flow, so no allow rule skips can_use_tool."""
        if not self.governed(input_data.get("tool_name") or ""):
            return {}
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "ask",
                                       "permissionDecisionReason": "AuthFI governs this tool"}}

    async def post_tool(self, input_data, tool_use_id, context):
        name = input_data.get("tool_name") or ""
        if not self.governed(name):
            return {}
        event = input_data.get("hook_event_name")
        if event == "PostToolUseFailure":
            answer = read_result(str(input_data.get("error") or ""), is_error=True)
        else:
            answer = read_result(_response_as_result(input_data.get("tool_response")))
        if not answer.held:
            return {}
        args = without_approval(input_data.get("tool_input"))
        self.held[call_digest(name, args)] = answer.approval_id
        _log.info("%s is held for approval %s", name, answer.approval_id)
        return {"hookSpecificOutput": {"hookEventName": event, "additionalContext": (
            "AuthFI is holding this call for a person's approval (approval %s). It has not run. Call the "
            "tool again with the same arguments: the run then waits until they decide." % answer.approval_id)}}

    async def can_use_tool(self, tool_name, tool_input, context):
        if not self.governed(tool_name):
            return await self._other(tool_name, tool_input, context)
        args = without_approval(tool_input)
        approval_id = self.held.pop(call_digest(tool_name, args), None)
        if approval_id is None:
            return PermissionResultAllow()
        approval = {"approval_id": approval_id, "tool": tool_name, "args": args}
        try:
            decision = await self._wait(approval)
        except AuthFIError as e:
            _log.warning("could not wait for approval %s: %s", approval_id, e)
            return PermissionResultDeny(message="AuthFI could not learn the decision on approval %s: %s"
                                        % (approval_id, e), interrupt=False)
        status = decision.get("status") if isinstance(decision, dict) else (None if decision else "denied")
        if status in _DECIDED_NO:
            return PermissionResultDeny(message="A person %s this call in AuthFI (approval %s)."
                                        % ("declined" if status == "denied" else "did not approve in time",
                                           approval_id), interrupt=False)
        return PermissionResultAllow(updated_input=with_approval(args, approval_id))

    async def _wait(self, approval):
        if self.wait_for_decision is not None:
            decision = self.wait_for_decision(approval)
            return await decision if inspect.isawaitable(decision) else decision
        record = await self.agent.wait_for_approval(approval["approval_id"], timeout=self.timeout)
        if record.get("status") == APPROVAL_PENDING:
            raise AuthFIError("approval %s is still pending" % approval["approval_id"], status=408)
        return record

    async def _other(self, tool_name, tool_input, context):
        if callable(self.other_tools):
            return await self.other_tools(tool_name, tool_input, context)
        if self.other_tools == "allow":
            return PermissionResultAllow()
        return PermissionResultDeny(message="%s is not an AuthFI tool, and nothing here decides it" % tool_name,
                                    interrupt=False)


def _response_as_result(response):
    if isinstance(response, dict):
        return response
    if isinstance(response, list):
        return {"content": response, "isError": False}
    return {"content": [{"type": "text", "text": str(response or "")}], "isError": False}


class AuthFIClaudeClient(ClaudeSDKClient):
    """``ClaudeSDKClient`` that keeps the gateway's MCP credential fresh.

    Claude Code reads an MCP server's headers when it connects. While this
    client is open it reconnects the gateway server a while before the token
    it connected with expires, so ``authfi headers`` runs again. ``agent`` is
    the ``AgentAuth`` the options were built from; ``refresh_every`` overrides
    the interval in seconds. With a ``DelegatedToken`` nothing is refreshed:
    that token cannot be renewed.
    """

    def __init__(self, options=None, agent=None, server_name="authfi", refresh_every=None, **kwargs):
        super().__init__(options=options, **kwargs)
        if isinstance(agent, AsyncAgentAuth):
            agent = agent.agent
        self._authfi_agent = agent if isinstance(agent, AgentAuth) else None
        self._authfi_server = server_name
        self._authfi_every = refresh_every
        self._authfi_task = None

    async def connect(self, *args, **kwargs):
        await super().connect(*args, **kwargs)
        if self._authfi_agent is not None and self._authfi_task is None:
            every = self._authfi_every or await _refresh_interval(self._authfi_agent)
            self._authfi_task = asyncio.ensure_future(self._refresh(every))

    async def _refresh(self, every):
        while True:
            await asyncio.sleep(every)
            try:
                await self.reconnect_mcp_server(self._authfi_server)
                _log.info("reconnected MCP server %s with a fresh token", self._authfi_server)
            except Exception as e:  # the session may be closing; the next turn reports a real failure
                _log.warning("could not reconnect MCP server %s to refresh its token: %s", self._authfi_server, e)

    async def disconnect(self):
        if self._authfi_task is not None:
            self._authfi_task.cancel()
            try:
                await self._authfi_task
            except (asyncio.CancelledError, Exception):
                pass
            self._authfi_task = None
        await super().disconnect()


async def _refresh_interval(agent):
    """Seconds between MCP reconnects: two minutes inside the agent's token lifetime, at least 30."""
    await AsyncAgentAuth(agent).token()
    lifetime = agent._expires_at - time.time()
    return max(30.0, lifetime - 120.0)


def trace_with_langsmith(agent, capture_content=False, run_id=None, client_kwargs=None, **kwargs):
    """Turn on LangSmith's Claude Agent SDK tracing, recording metadata and not content.

    LangSmith's integration records the user prompt, the system prompt, tool
    arguments and full tool results by default. This installs a LangSmith
    client that hides inputs and outputs unless ``capture_content``, and
    stamps ``authfi_agent_id`` and ``authfi_run_id`` into every run's
    metadata. ``client_kwargs`` go to ``langsmith.Client``; ``kwargs`` to
    ``configure_claude_agent_sdk`` (``project_name``, ``tags``, ``name``). Call
    it before the client is created.
    """
    from langsmith.integrations.claude_agent_sdk import configure_claude_agent_sdk

    if isinstance(agent, AsyncAgentAuth):
        agent = agent.agent
    metadata = dict(kwargs.pop("metadata", None) or {})
    if getattr(agent, "agent_id", None):
        metadata["authfi_agent_id"] = agent.agent_id
    if run_id:
        metadata["authfi_run_id"] = run_id
    langsmith_metadata_only(capture_content, **(client_kwargs or {}))
    return configure_claude_agent_sdk(metadata=metadata, **kwargs)
