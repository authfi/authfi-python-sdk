"""The OpenAI Agents SDK on AuthFI: tracing that carries no token and no content, and governed tools.

``pip install 'authfi[openai-agents]'``

::

    from agents import Agent, Runner
    from authfi import AgentAuth, AsyncAgentAuth
    from authfi.openai_agents import AuthFIMCPServer, AuthFIModel, configure_tracing, run_config

    configure_tracing()                  # first: no default exporter, no content in spans
    agent_auth = AsyncAgentAuth(AgentAuth.from_env())
    async with AuthFIMCPServer(agent_auth) as tools:
        agent = Agent(name="research", model=AuthFIModel(agent_auth), mcp_servers=[tools])
        result = await Runner.run(agent, "Describe pallets/flask's wiki", run_config=run_config(agent_auth))

**Tracing.** The SDK's default exporter sends every trace to OpenAI with
``Authorization: Bearer <the OpenAI key>``. When the AuthFI token is used as
that key (``OPENAI_API_KEY``, ``set_default_openai_key``, or a client set with
``use_for_tracing`` left on) every export carries the agent's token; and
``trace_include_sensitive_data`` defaults to on, so prompts, answers and tool
results go too. :func:`configure_tracing` removes the default exporter, keeps
content out of spans unless asked, and sends traces only to the processors you
name, each wrapped so that content is stripped before it is handed on.

**Model.** :class:`AuthFIModel` is LiteLLM on the project's AI gateway. It
sends the current token as ``Authorization: Bearer`` on every call; the
OpenAI key is never set, so no exporter can pick the token up.

**Tools.** :class:`AuthFIMCPServer` is the SDK's streamable HTTP MCP client
on the project's MCP gateway: a current token on every request, provider-legal
tool names, ``never_allowed`` tools left out, refusals handed to the model as
the tool's error (the older gateway's HTTP 403 included), and holds bridged to
the SDK's own tool approval.

**Approvals.** The SDK decides whether a call needs approval before sending
it; the gateway decides when the call arrives. So a held call is answered to
the model as waiting, with an instruction to call again; the repeat call
pauses the run as a ``ToolApprovalItem`` in ``result.interruptions``. Once the
person has decided, ``state.approve(item)`` and run again: the call is sent
with the approval id and the gateway decides. :meth:`AuthFIMCPServer.approval_for`
names the approval behind an interruption.
"""

import dataclasses
import json
import logging
import os

try:
    from agents import RunConfig
    from agents.extensions.models.litellm_model import LitellmModel
    from agents.mcp import MCPServerStreamableHttp
    from agents.mcp import _compat as _mcp_compat
    from agents.tracing import (
        TracingProcessor,
        get_current_span,
        get_current_trace,
        set_trace_processors,
        set_tracing_disabled,
    )
    from agents.tracing.span_data import FunctionSpanData, GenerationSpanData, ResponseSpanData
    from mcp.types import CallToolResult, TextContent
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("authfi.openai_agents needs the OpenAI Agents SDK: pip install 'authfi[openai-agents]'") from e

from .gateway import (
    REQUEST_ID_HEADER,
    REQUEST_ID_META,
    ToolNames,
    call_digest,
    credential,
    mcp_http_client_factory,
    model_base_url,
    never_allowed,
    read_result,
    request_id,
    with_approval,
)

_log = logging.getLogger("authfi")

# LiteLLM will not send an Anthropic request without an api_key, and sends it as
# x-api-key. The gateway reads the Authorization header first and drops x-api-key,
# so this stands in and is never a credential.
_PLACEHOLDER = "authfi-sets-the-token-per-request"


# --------------------------------------------------------------------- tracing


def configure_tracing(processors=(), capture_content=False):
    """Trace without the token and, unless asked, without content. Call it before the first run.

    - The SDK's default exporter, which sends traces to OpenAI with the
      OpenAI key, is removed. ``processors`` replace it, each wrapped in
      :class:`AuthFITraceProcessor`; with none, tracing is turned off.
    - ``capture_content=False`` makes ``trace_include_sensitive_data`` default
      to off for every ``RunConfig`` built afterwards, and the wrapper strips
      prompts, answers and tool inputs and outputs from any span that still
      carries them.
    """
    if (os.environ.get("OPENAI_API_KEY") or "").startswith("eyJ"):
        # The leak this function exists for: a JWT as the OpenAI key rides every export
        # of any processor that sends to OpenAI with the default key.
        _log.warning("OPENAI_API_KEY holds a JWT. If it is an AuthFI token, unset it: AuthFIModel "
                     "does not need it, and an exporter that sends to OpenAI would send it.")
    os.environ["OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA"] = "true" if capture_content else "false"
    wrapped = [p if isinstance(p, AuthFITraceProcessor) else AuthFITraceProcessor(p, capture_content)
               for p in processors]
    set_trace_processors(wrapped)
    set_tracing_disabled(not wrapped)
    return wrapped


class AuthFITraceProcessor(TracingProcessor):
    """Hands spans to another processor, without their content unless ``capture_content``.

    The prompt, the answer and tool inputs and outputs are removed from
    generation, response and function spans before the wrapped processor
    sees them. Names, timings, ids, usage and the trace metadata (with the
    agent id from :func:`run_config`) are kept.
    """

    def __init__(self, inner, capture_content=False):
        self.inner = inner
        self.capture_content = capture_content

    def on_trace_start(self, trace):
        self.inner.on_trace_start(trace)

    def on_trace_end(self, trace):
        self.inner.on_trace_end(trace)

    def on_span_start(self, span):
        self.inner.on_span_start(span)

    def on_span_end(self, span):
        if not self.capture_content:
            _strip_content(span.span_data)
        self.inner.on_span_end(span)

    def shutdown(self):
        self.inner.shutdown()

    def force_flush(self):
        self.inner.force_flush()


def _strip_content(data):
    if isinstance(data, (GenerationSpanData, FunctionSpanData)):
        data.input = None
        data.output = None
    elif isinstance(data, ResponseSpanData):
        data.input = None


def run_config(agent, capture_content=False, **kwargs):
    """A ``RunConfig`` whose trace names the agent and holds no content unless asked.

    ``agent`` is the credential (an ``AgentAuth`` and so on); its agent id goes
    in ``trace_metadata`` as ``authfi_agent_id``. ``kwargs`` are ``RunConfig``'s.
    """
    cred = credential(agent)
    metadata = dict(kwargs.pop("trace_metadata", None) or {})
    if cred.agent_id:
        metadata["authfi_agent_id"] = cred.agent_id
    kwargs.setdefault("trace_include_sensitive_data", capture_content)
    return RunConfig(trace_metadata=metadata or None, **kwargs)


def _run_request_id():
    """The id a model call is sent with: the trace's id, which the exported trace carries."""
    trace = get_current_trace()
    return request_id(getattr(trace, "trace_id", None))


def _call_request_id():
    """The id a tool call is sent with: its function span's id, else the trace's."""
    span = get_current_span()
    return request_id(getattr(span, "span_id", None)) or _run_request_id()


# --------------------------------------------------------------------- the model


class AuthFIModel(LitellmModel):
    """LiteLLM on the project's AI gateway, with the current AuthFI token on every call.

    ``agent`` is the credential; ``model`` the model's name at the gateway
    (``claude-haiku-5-5``) and ``provider`` its LiteLLM provider prefix.
    The token is added to the call's headers, which the SDK does not put in
    spans; the trace id goes as ``X-Request-Id``.
    """

    def __init__(self, agent, model="claude-haiku-5-5", base_url=None, provider="anthropic", **kwargs):
        self.credential = credential(agent)
        super().__init__(model="%s/%s" % (provider, model), base_url=model_base_url(self.credential, base_url),
                         api_key=_PLACEHOLDER, **kwargs)

    async def _with_token(self, model_settings):
        headers = dict(model_settings.extra_headers or {})
        headers["Authorization"] = "Bearer " + await self.credential.atoken()
        rid = _run_request_id()
        if rid:
            headers[REQUEST_ID_HEADER] = rid
        return dataclasses.replace(model_settings, extra_headers=headers)

    async def get_response(self, system_instructions, input, model_settings, *args, **kwargs):
        return await super().get_response(system_instructions, input, await self._with_token(model_settings),
                                          *args, **kwargs)

    async def stream_response(self, system_instructions, input, model_settings, *args, **kwargs):
        settings = await self._with_token(model_settings)
        async for event in super().stream_response(system_instructions, input, settings, *args, **kwargs):
            yield event


# --------------------------------------------------------------------- the tools


class AuthFIMCPServer(MCPServerStreamableHttp):
    """The project's MCP gateway as an OpenAI Agents SDK MCP server.

    ``agent`` is the credential. ``url`` defaults to ``<project>/mcp``. Every
    other argument is ``MCPServerStreamableHttp``'s, except ``params`` and
    ``require_approval``, which this class sets.
    """

    def __init__(self, agent, url=None, name="authfi", include_never_allowed=False,
                 client_session_timeout_seconds=60, **kwargs):
        self.credential = credential(agent)
        params = {
            "url": model_base_url(self.credential, url),
            "httpx_client_factory": mcp_http_client_factory(self.credential, _mcp_compat.MCP_HTTPX),
            "timeout": 60,
            "sse_read_timeout": 300,
        }
        self.names = ToolNames()
        self.include_never_allowed = include_never_allowed
        # digest of a held call (tool and arguments) -> the approval it waits for
        self._held = {}
        super().__init__(params=params, name=name, require_approval=self._needs_approval,
                         client_session_timeout_seconds=client_session_timeout_seconds, **kwargs)

    async def list_tools(self, run_context=None, agent=None):
        out = []
        for tool in await super().list_tools(run_context, agent):
            if never_allowed(tool) and not self.include_never_allowed:
                _log.info("leaving out %s: the gateway marks it never_allowed for this agent", tool.name)
                continue
            out.append(tool.model_copy(update={"name": self.names.provider(tool.name)}))
        return out

    def _needs_approval(self, run_context, agent, tool):
        gateway_name = self.names.gateway(tool.name)
        return any(name == gateway_name for name, _ in self._held.values())

    async def call_tool(self, tool_name, arguments, meta=None):
        name = self.names.gateway(tool_name)
        args = dict(arguments or {})
        digest = call_digest(name, args)
        approval = self._held.get(digest, (None, None))[1]
        meta = dict(meta or {})
        rid = _call_request_id()
        if rid:
            meta[REQUEST_ID_META] = rid
        result = await super().call_tool(name, with_approval(args, approval) if approval else args, meta or None)
        answer = read_result(result)
        if answer.held:
            self._held[digest] = (name, answer.approval_id)
            _log.info("%s is held for approval %s", name, answer.approval_id)
            return CallToolResult(content=[TextContent(type="text", text=_held_for_sdk(answer))], isError=True)
        self._held.pop(digest, None)
        return result

    def approval_for(self, item):
        """The AuthFI approval behind a ``ToolApprovalItem`` from ``result.interruptions``.

        Returns ``{"approval_id", "tool", "args"}``, or None when the item is
        not a call this server holds.
        """
        raw = getattr(item, "raw_item", None)
        provider_name = getattr(item, "tool_name", None) or getattr(raw, "name", None)
        arguments = getattr(raw, "arguments", None)
        if provider_name is None or arguments is None:
            return None
        args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
        name = self.names.gateway(provider_name)
        held = self._held.get(call_digest(name, args))
        if not held:
            return None
        return {"approval_id": held[1], "tool": name, "args": args}


def _held_for_sdk(answer):
    return ("This call is waiting for a person's approval in AuthFI (approval %s). It has not run. "
            "Call this tool again with the same arguments: the run then pauses until they decide."
            % answer.approval_id)
