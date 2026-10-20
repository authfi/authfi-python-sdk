"""Google ADK on AuthFI: the model credential, governed MCP tools, approvals through tool confirmation.

``pip install 'authfi[adk]'``

::

    from authfi import AgentAuth, AsyncAgentAuth
    from authfi.adk import AuthFIADK, configure_telemetry

    agent_auth = AsyncAgentAuth(AgentAuth.from_env())
    configure_telemetry(agent_auth)              # before the Runner is built
    authfi = AuthFIADK(agent_auth)
    agent = authfi.agent(name="research", instruction="...", model="claude-haiku-5-5")
    # run it with an ADK Runner; held calls arrive as adk_request_confirmation:
    for held in authfi.held_approvals(events):   # once the person has decided
        message = authfi.answer(held, confirmed=True)

- **Model credential.** ADK's ``LiteLlm`` sends ``api_key`` as ``x-api-key``
  and refuses to send without one. The model gets a placeholder key, and a
  ``before_model_callback`` puts the current token in
  ``Authorization: Bearer`` on every call, with the invocation id as
  ``X-Request-Id``.
- **MCP.** ``McpToolset`` on the project's gateway, with an HTTP client that
  sends the current token on every request, so a session ADK opens outside a
  run (``get_tools()``) is authenticated too. A ``header_provider`` adds the
  invocation id. Tools the gateway marks ``never_allowed`` are left out, and
  the tool list is cached instead of being listed before every model step.
- **Refusals.** A refused call reaches the model as ``{"error": "<why>"}``
  instead of "MCP tool execution failed: Server returned an error response".
- **Approvals.** An ``after_tool_callback`` turns a held call into ADK's tool
  confirmation (``request_confirmation``), with the approval id as payload; a
  ``before_tool_callback`` sends the confirmed call with the approval id. ADK
  asks for a confirmation once per function call, so answer it only once the
  person has decided. A call confirmed too early and held again is answered
  to the model as still waiting; its repeat goes under the same approval and
  asks for confirmation again, instead of the run ending held.
- **Telemetry.** :func:`configure_telemetry` turns ADK's content capture in
  spans off unless asked, and names the agent in the OpenTelemetry resource.
"""

import logging
import os

try:
    from google.adk.agents import LlmAgent
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from google.genai import types
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("authfi.adk needs Google ADK: pip install 'authfi[adk]'") from e

from .gateway import (
    APPROVAL_ARGUMENT,
    call_digest,
    credential,
    mcp_http_client_factory,
    model_base_url,
    never_allowed,
    read_result,
    request_id_header,
    without_approval,
)

_log = logging.getLogger("authfi")

#: The function call ADK uses to ask the application for a tool confirmation.
CONFIRMATION_CALL = "adk_request_confirmation"

#: The confirmation payload key that carries the AuthFI approval id.
PAYLOAD_KEY = "authfi_approval_id"

# LiteLLM will not send an Anthropic request without an api_key, and sends it as
# x-api-key. The gateway reads the Authorization header first and drops x-api-key,
# so this stands in and is never a credential.
_PLACEHOLDER = "authfi-sets-the-token-per-request"

# ADK lists the toolset's tools before every model step unless the list is cached.
_TOOL_LIST_TTL = 60.0


def configure_telemetry(agent=None, capture_content=False):
    """ADK telemetry the AuthFI way: metadata, not content, unless ``capture_content``.

    ADK puts full prompts and tool results in its spans by default
    (``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS``). This turns that off, keeps the
    OpenTelemetry GenAI content mode at ``NO_CONTENT``, and adds
    ``authfi.agent_id`` to ``OTEL_RESOURCE_ATTRIBUTES``. ADK reads these when
    its telemetry and tracer provider are created, so call it first.
    """
    os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true" if capture_content else "false"
    if not capture_content:
        os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "NO_CONTENT"
    agent_id = credential(agent).agent_id if agent is not None else None
    if agent_id:
        attributes = [a for a in (os.environ.get("OTEL_RESOURCE_ATTRIBUTES") or "").split(",")
                      if a and not a.startswith("authfi.agent_id=")]
        attributes.append("authfi.agent_id=%s" % agent_id)
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(attributes)


class AuthFIADK:
    """The project's AuthFI gateway as an ADK model, toolset and callbacks.

    ``agent`` is an :class:`authfi.AgentAuth`, :class:`authfi.AsyncAgentAuth`,
    :class:`authfi.DelegatedToken` or a function returning the token. ``url``
    defaults to ``<project>/mcp``, which serves the AI gateway and the MCP
    gateway.
    """

    def __init__(self, agent, url=None):
        self.credential = credential(agent)
        self.url = model_base_url(self.credential, url)
        # digest of a held call (tool and arguments) -> the approval it waits for
        self._held = {}

    # --- the model

    def model(self, model="claude-haiku-5-5", provider="anthropic", **kwargs):
        """``LiteLlm`` on the AI gateway. Use it with :meth:`before_model`, which sends the token."""
        return LiteLlm(model="%s/%s" % (provider, model), api_base=self.url, api_key=_PLACEHOLDER, **kwargs)

    async def before_model(self, callback_context, llm_request):
        """``before_model_callback``: the current token and the invocation id on this model call."""
        config = llm_request.config or types.GenerateContentConfig()
        options = config.http_options or types.HttpOptions()
        headers = dict(options.headers or {})
        headers.update(await self.credential.aheaders())
        headers.update(request_id_header(getattr(callback_context, "invocation_id", None)))
        options.headers = headers
        config.http_options = options
        llm_request.config = config
        return None

    # --- the tools

    def toolset(self, tool_filter=None, include_never_allowed=False, tool_list_cache_ttl_seconds=_TOOL_LIST_TTL,
                **kwargs):
        """``McpToolset`` on the gateway: the current token on every request, the invocation id per call."""
        params = StreamableHTTPConnectionParams(url=self.url, timeout=30.0,
                                                httpx_client_factory=mcp_http_client_factory(self.credential))
        toolset = McpToolset(connection_params=params, header_provider=_invocation_header,
                             tool_filter=_ToolFilter(tool_filter, include_never_allowed),
                             tool_list_cache_ttl_seconds=tool_list_cache_ttl_seconds, **kwargs)
        _no_google_mtls(toolset)
        return toolset

    # --- approvals and refusals

    def before_tool(self, tool, args, tool_context):
        """``before_tool_callback``: send a confirmed call with its approval id, stop a declined one.

        A call the gateway is already holding is sent with that approval's id
        even when it was not confirmed (a repeat of a call still waiting), so
        the gateway answers about the same approval instead of opening another.
        """
        digest = call_digest(tool.name, args)
        confirmation = tool_context.tool_confirmation
        payload = confirmation.payload if confirmation is not None and isinstance(confirmation.payload, dict) else {}
        approval_id = payload.get(PAYLOAD_KEY)
        if approval_id and not confirmation.confirmed:
            self._held.pop(digest, None)
            return {"error": "A person declined this call in AuthFI (approval %s). It did not run; do not "
                             "try it again." % approval_id}
        approval_id = approval_id or self._held.get(digest)
        if approval_id:
            args[APPROVAL_ARGUMENT] = approval_id
        return None

    def after_tool(self, tool, args, tool_context, tool_response):
        """``after_tool_callback``: a hold becomes ADK's confirmation; a refusal names its reason.

        ADK surfaces a confirmation only for a call it has not resumed yet. A
        resumed call the gateway holds again (it was confirmed before anyone
        decided) is answered to the model as still waiting; its repeat is held
        under the same approval and asks for confirmation again.
        """
        answer = read_result(tool_response)
        digest = call_digest(tool.name, args)
        if not answer.held:
            self._held.pop(digest, None)
            return {"error": answer.text} if answer.refused else None
        self._held[digest] = answer.approval_id
        _log.info("%s is held for approval %s", tool.name, answer.approval_id)
        if tool_context.tool_confirmation is not None:
            return {"error": "This call is still waiting for a person's approval in AuthFI (approval %s). It "
                             "has not run. Call the tool again with the same arguments to wait for their "
                             "decision." % answer.approval_id}
        tool_context.request_confirmation(
            hint="AuthFI is holding %s for a person's approval (approval %s). Answer once they have "
                 "decided in the AuthFI console." % (tool.name, answer.approval_id),
            payload={PAYLOAD_KEY: answer.approval_id, "tool": tool.name, "args": without_approval(args)})
        tool_context.actions.skip_summarization = True
        return {"status": "waiting_for_approval", PAYLOAD_KEY: answer.approval_id}

    # --- an agent with all of it

    def agent(self, name, instruction, model="claude-haiku-5-5", tools=None, **kwargs):
        """An ``LlmAgent`` on the gateway: its model, its toolset and the callbacks above.

        ``tools`` are added to the gateway's toolset. Callbacks you pass in
        ``kwargs`` run after these.
        """
        callbacks = {"before_model_callback": self.before_model, "before_tool_callback": self.before_tool,
                     "after_tool_callback": self.after_tool}
        for key, ours in callbacks.items():
            theirs = kwargs.pop(key, None)
            callbacks[key] = [ours] + (list(theirs) if isinstance(theirs, (list, tuple)) else
                                       [theirs] if theirs else [])
        return LlmAgent(name=name, instruction=instruction, model=self.model(model),
                        tools=[self.toolset()] + list(tools or []), **callbacks, **kwargs)

    # --- for the application

    @staticmethod
    def held_approvals(events):
        """The AuthFI holds among ADK events: ``{"approval_id", "tool", "args", "call"}`` each.

        ``call`` is ADK's ``adk_request_confirmation`` function call, which
        :meth:`answer` replies to.
        """
        out = []
        for event in events:
            for part in (event.content.parts if event.content and event.content.parts else ()):
                call = part.function_call
                if call is None or call.name != CONFIRMATION_CALL:
                    continue
                confirmation = (call.args or {}).get("toolConfirmation") or {}
                payload = confirmation.get("payload") or {}
                if payload.get(PAYLOAD_KEY):
                    out.append({"approval_id": payload[PAYLOAD_KEY], "tool": payload.get("tool"),
                                "args": payload.get("args"), "call": call})
        return out

    @staticmethod
    def answer(held, confirmed=True):
        """The message that answers ADK's confirmation for ``held``, to send once the person has decided."""
        call = held["call"]
        payload = {PAYLOAD_KEY: held["approval_id"]}
        return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
            id=call.id, name=CONFIRMATION_CALL, response={"confirmed": bool(confirmed), "payload": payload}))])


def _no_google_mtls(toolset):
    """Keep ADK's Google mTLS client off the AuthFI gateway connection.

    For every streamable HTTP MCP server ADK looks for Google Application
    Default Credentials and, when it finds them, swaps the configured HTTP
    client for a Google-authenticated one: the AuthFI token would no longer be
    sent and a Google token would be, to a server that is not Google's. Off
    Google Cloud the probe costs about 12 seconds per event loop. Neither is
    wanted for this connection, so this toolset's session manager is told
    there is no mTLS transport. Other toolsets are unaffected.
    """
    manager = getattr(toolset, "_mcp_session_manager", None)
    if manager is None or not hasattr(manager, "_get_mtls_transport"):
        return

    async def no_mtls_transport():
        return None
    manager._get_mtls_transport = no_mtls_transport


def _invocation_header(readonly_context):
    return request_id_header(getattr(readonly_context, "invocation_id", None))


class _ToolFilter:
    """ADK's tool predicate: leave out ``never_allowed`` tools, then apply the caller's filter."""

    def __init__(self, inner, include_never_allowed):
        self.inner = inner
        self.include_never_allowed = include_never_allowed

    def __call__(self, tool, readonly_context=None):
        raw = getattr(tool, "raw_mcp_tool", None)
        if not self.include_never_allowed and raw is not None and never_allowed(raw):
            _log.info("leaving out %s: the gateway marks it never_allowed for this agent", tool.name)
            return False
        if self.inner is None:
            return True
        if isinstance(self.inner, (list, tuple, set)):
            return tool.name in self.inner
        return bool(self.inner(tool, readonly_context))
