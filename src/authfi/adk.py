"""Google ADK: the model's credential, the gateway's tools, approvals through ADK's tool confirmation.

``pip install "authfi[adk] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from google.adk.runners import InMemoryRunner
    from authfi import AsyncAuthFI
    from authfi.adk import ADKIntegration

    af = AsyncAuthFI.from_env()
    adk = ADKIntegration(af.agent_from_env())
    adk.configure_tracing()                    # before the runner is built
    agent = adk.agent(name="research", instruction="...", model="claude-haiku-5-5")
    # run it with an ADK runner; a held call arrives as ADK's tool confirmation:
    for pending in adk.held_approvals(events):  # once the person has decided in AuthFI
        message = adk.answer(pending, confirmed=True)

- **The model** is LiteLLM on the AI gateway. It puts the current token on each call itself,
  where ADK's traces do not record it; LiteLLM's required key is a placeholder.
- **The tools** are ADK's ``McpToolset`` on the gateway, with the gateway's HTTP client, so every
  request carries the current token, even a session ADK opens outside a run. ADK replaces an MCP
  connection's client with a Google mTLS client when it finds Google credentials, which would send
  no AuthFI token; that is turned off for this toolset only.
- **Refusals** reach the model as ``{"error": "<why>"}``. A refused token raises
  :class:`~authfi.GatewayTokenRefused`.
- **Approvals.** A held call becomes ADK's tool confirmation. Answer it once the person has decided
  in AuthFI; the call is then sent with the approval, and the gateway decides. The approval id stays
  in the confirmation, which ADK keeps from the model.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import AsyncGenerator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

try:
    import httpx
    from google.adk.agents import LlmAgent
    from google.adk.agents.readonly_context import ReadonlyContext
    from google.adk.events import Event
    from google.adk.models.lite_llm import LiteLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.tools.base_tool import BaseTool
    from google.adk.tools.mcp_tool.mcp_session_manager import StreamableHTTPConnectionParams
    from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
    from google.adk.tools.tool_context import ToolContext
    from google.genai import types
    from pydantic import PrivateAttr
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.adk needs Google ADK: pip install 'authfi[adk] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import PLACEHOLDER_KEY, agent_id_of, async_gateway_for
from authfi.errors import ConfigurationError, GatewayTokenRefused, IntegrationError
from authfi.gateway import (
    HOLD_MESSAGE,
    TOKEN_REFUSED_MESSAGE,
    AsyncGateway,
    ToolApprovalRequired,
    ToolRefused,
    token_refusal,
)
from authfi.gateway import PendingApproval as BasePendingApproval
from authfi.tokens import AsyncTokenSource

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI

__all__ = ["ADKIntegration", "AuthFILiteLlm", "PendingApproval"]

#: The function call ADK asks the application with for a tool confirmation.
CONFIRMATION_CALL = "adk_request_confirmation"

#: The confirmation payload key that carries the AuthFI approval id.
PAYLOAD_KEY = "authfi_approval_id"

#: The google-adk versions whose private MCP seam this module is tested with.
TESTED = "google-adk 2.12 and 2.13"

#: How many held calls the integration remembers until they are confirmed and run again.
MEMORY = 10_000


@dataclass(frozen=True, slots=True)
class PendingApproval(BasePendingApproval):
    """A call the gateway holds until a person approves it; ``call_id`` is ADK's confirmation request."""

    call_id: str


def _key(tool: str, args: Mapping[str, Any]) -> str:
    return tool + "\x00" + json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


class AuthFILiteLlm(LiteLlm):
    """LiteLLM on the AI gateway that puts the current token on each call. Make it with ``model()``.

    The token goes in the request's HTTP headers, which ADK leaves out of its traces. A refused
    token raises :class:`~authfi.GatewayTokenRefused`.
    """

    _authfi_gateway: AsyncGateway | None = PrivateAttr(default=None)

    @classmethod
    def on_gateway(cls, gateway: AsyncGateway, model: str, **kwargs: Any) -> AuthFILiteLlm:
        """Claude ``model`` on ``gateway``; ``kwargs`` are ``LiteLlm``'s."""
        params: dict[str, Any] = {
            **kwargs,
            "model": f"anthropic/{model}",
            "api_base": gateway.model_base_url("anthropic"),
            "api_key": PLACEHOLDER_KEY,
        }
        llm = cls(**params)
        llm._authfi_gateway = gateway
        return llm

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        gateway = cast(AsyncGateway, self._authfi_gateway)
        config = llm_request.config.model_copy()
        options = (config.http_options or types.HttpOptions()).model_copy()
        url = gateway.model_base_url("anthropic") + "/v1/messages"
        options.headers = {**(options.headers or {}), **(await gateway.source.headers("POST", url))}
        config.http_options = options
        llm_request.config = config
        try:
            async for response in super().generate_content_async(llm_request, stream=stream):
                yield response
        except Exception as error:
            refused = token_refusal(error)
            if refused is not None:
                raise refused from error
            raise


def _no_google_mtls(toolset: McpToolset) -> None:
    # ADK swaps a streamable HTTP MCP connection's client for a Google mTLS one when it finds Google
    # credentials: the AuthFI token would not be sent, and off Google Cloud the probe costs seconds per
    # event loop. ADK has no setting for one toolset, so the probe is answered "none" here.
    manager = getattr(toolset, "_mcp_session_manager", None)
    if manager is None or not callable(getattr(manager, "_get_mtls_transport", None)):
        raise IntegrationError(
            "authfi.adk keeps Google's mTLS client off the AuthFI gateway through "
            f"McpToolset._mcp_session_manager._get_mtls_transport, which this google-adk does not have. It is "
            f"tested with {TESTED}."
        )

    async def no_mtls_transport() -> None:
        return None

    manager._get_mtls_transport = no_mtls_transport


class ADKIntegration:
    """The AuthFI gateways for Google ADK.

    Args:
        source: an asyncio credential (``AsyncAuthFI.agent(...)``, or the ``AsyncDelegatedToken``
            it holds for a person). ADK runs on asyncio.
        project: the project, for a credential of your own that does not name it.

    """

    def __init__(self, source: AsyncTokenSource, *, project: AsyncAuthFI | None = None) -> None:
        self.source = source
        self.gateway = async_gateway_for(source, project)
        self.agent_id = agent_id_of(source)
        self._held: OrderedDict[str, str] = OrderedDict()

    # --- model and tools --------------------------------------------------------------------------------

    def model(self, model: str, **kwargs: Any) -> AuthFILiteLlm:
        """Claude on the AI gateway, through LiteLLM. ``kwargs`` are ``LiteLlm``'s, except the credential ones."""
        for refused in ("api_key", "api_base", "base_url", "extra_headers"):
            if refused in kwargs:
                raise ConfigurationError(f"{refused} is set by the integration; the credential is the agent's")
        return AuthFILiteLlm.on_gateway(self.gateway, model, **kwargs)

    def toolset(self, *, include_never_allowed: bool = False, **kwargs: Any) -> McpToolset:
        """ADK's ``McpToolset`` on the MCP gateway: the current token on every request.

        A tool the gateway marks ``never_allowed`` is left out unless ``include_never_allowed``; the
        tool list is cached for a minute unless ``tool_list_cache_ttl_seconds`` says otherwise.
        ``kwargs`` are ``McpToolset``'s.
        """
        params = StreamableHTTPConnectionParams(
            url=self.gateway.mcp_url,
            timeout=30.0,
            httpx_client_factory=self.gateway.mcp_client_factory(httpx),
        )
        given = kwargs.pop("tool_filter", None)

        def offered(tool: BaseTool, readonly_context: ReadonlyContext | None = None) -> bool:
            raw = getattr(tool, "raw_mcp_tool", None)
            if not include_never_allowed and self.gateway.governance_of(raw) == "never_allowed":
                self.gateway.project.logger.info("leaving out %s: never allowed for this agent", tool.name)
                return False
            if given is None:
                return True
            if isinstance(given, list | tuple | set):
                return tool.name in cast(Iterable[str], given)
            return bool(given(tool, readonly_context))

        kwargs.setdefault("tool_list_cache_ttl_seconds", 60.0)
        toolset = McpToolset(connection_params=params, tool_filter=offered, **kwargs)
        _no_google_mtls(toolset)
        return toolset

    # --- the callbacks -----------------------------------------------------------------------------------

    def _remember(self, key: str, approval_id: str) -> None:
        self._held[key] = approval_id
        self._held.move_to_end(key)
        while len(self._held) > MEMORY:
            self._held.popitem(last=False)

    async def before_tool_callback(
        self, tool: BaseTool, args: dict[str, Any], tool_context: ToolContext
    ) -> dict[str, Any] | None:
        """Remove an approval the model wrote, and send a confirmed or repeated held call with its approval.

        A call the person declined in your application is not sent.
        """
        clean = self.gateway.strip_approval_argument(args)
        args.clear()
        args.update(clean)
        key = _key(tool.name, clean)
        confirmation = tool_context.tool_confirmation
        payload: object = confirmation.payload if confirmation is not None else None
        given = cast(Mapping[str, Any], payload).get(PAYLOAD_KEY) if isinstance(payload, Mapping) else None
        approval_id = self._held.get(key) or (given if isinstance(given, str) else None)
        if confirmation is not None and not confirmation.confirmed:
            self._held.pop(key, None)
            return {"error": "Not run: the action was not approved."}
        if approval_id:
            args.update(self.gateway.with_approval({}, approval_id))
        return None

    async def after_tool_callback(
        self, tool: BaseTool, args: dict[str, Any], tool_context: ToolContext, tool_response: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Turn a refusal into the tool's error and a hold into ADK's tool confirmation.

        ADK hands every MCP failure to the model as text; when the failure was the gateway refusing
        the token, this raises :class:`~authfi.GatewayTokenRefused` instead, ending the run. The
        refusal is read from this call's own response, never from state the gateway shares between
        calls, which ADK may run in parallel.
        """
        error = tool_response.get("error")
        if isinstance(error, str) and TOKEN_REFUSED_MESSAGE in error:
            raise GatewayTokenRefused(self.gateway.mcp_url, None)
        clean = self.gateway.strip_approval_argument(args)
        key = _key(tool.name, clean)
        outcome = self.gateway.read_result(tool_response)
        logger = self.gateway.project.logger
        if not isinstance(outcome, ToolApprovalRequired | ToolRefused):
            self._held.pop(key, None)
            return None
        if isinstance(outcome, ToolRefused):
            self._held.pop(key, None)
            logger.info("the gateway refused %s: %s", tool.name, outcome.code or outcome.error)
            return {"error": outcome.model_text}
        approval_id = outcome.approval_id
        self._remember(key, approval_id)
        logger.info("%s waits for approval %s", tool.name, approval_id)
        if tool_context.tool_confirmation is None:
            tool_context.request_confirmation(
                hint="A person must approve this action in AuthFI. Answer once they have decided.",
                payload={PAYLOAD_KEY: approval_id, "tool": tool.name, "args": clean},
            )
            tool_context.actions.skip_summarization = True
        # Confirmed before the person decided: ADK confirms a call once, so the model hears it waits,
        # and the same call made again goes under this approval and asks again.
        return {"status": HOLD_MESSAGE}

    def agent(self, *, name: str, instruction: str, model: str, tools: Sequence[Any] = (), **kwargs: Any) -> LlmAgent:
        """An ``LlmAgent`` with this model, the gateway's toolset and the tool callbacks.

        ``tools`` are added after the gateway's toolset. Tool callbacks you pass run after these.
        """
        callbacks: dict[str, list[Any]] = {
            "before_tool_callback": [self.before_tool_callback],
            "after_tool_callback": [self.after_tool_callback],
        }
        for name_, ours in callbacks.items():
            theirs = kwargs.pop(name_, None)
            if isinstance(theirs, list | tuple):
                ours.extend(cast(Sequence[Any], theirs))
            elif theirs is not None:
                ours.append(theirs)
        options: dict[str, Any] = {
            **kwargs,
            **callbacks,
            "name": name,
            "instruction": instruction,
            "model": self.model(model),
            "tools": [self.toolset(), *tools],
        }
        return LlmAgent(**options)

    # --- for the application -----------------------------------------------------------------------------

    @staticmethod
    def held_approvals(events: Iterable[Event]) -> list[PendingApproval]:
        """The calls the gateway holds, among ADK events: its confirmation requests carrying an approval."""
        out: list[PendingApproval] = []
        for event in events:
            for call in event.get_function_calls():
                if call.name != CONFIRMATION_CALL or not call.id:
                    continue
                confirmation = cast(Mapping[str, Any], call.args or {}).get("toolConfirmation")
                payload = (
                    cast(Mapping[str, Any], confirmation).get("payload") if isinstance(confirmation, Mapping) else None
                )
                if not isinstance(payload, Mapping):
                    continue
                detail = cast(Mapping[str, Any], payload)
                approval_id, tool, args = detail.get(PAYLOAD_KEY), detail.get("tool"), detail.get("args")
                if isinstance(approval_id, str) and isinstance(tool, str):
                    given = cast(dict[str, Any], args) if isinstance(args, dict) else {}
                    out.append(PendingApproval(approval_id, tool, given, call.id))
        return out

    @staticmethod
    def answer(pending: PendingApproval, *, confirmed: bool) -> types.Content:
        """The message that answers ADK's confirmation for ``pending``: send it once the person has decided."""
        response = {"confirmed": confirmed, "payload": {PAYLOAD_KEY: pending.approval_id}}
        part = types.Part(
            function_response=types.FunctionResponse(id=pending.call_id, name=CONFIRMATION_CALL, response=response)
        )
        return types.Content(role="user", parts=[part])

    def configure_tracing(self, *, capture_content: bool = False) -> None:
        """ADK telemetry with metadata and no content, unless ``capture_content``. Call it before the runner.

        ADK puts prompts and tool results in its spans by default
        (``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS``). This turns that off, sets the OpenTelemetry GenAI
        content mode to ``NO_CONTENT``, and adds ``authfi.agent_id`` to ``OTEL_RESOURCE_ATTRIBUTES``.
        Turning content on is the tenant's policy decision.
        """
        os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"] = "true" if capture_content else "false"
        if capture_content:
            os.environ.pop("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", None)
        else:
            os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"] = "NO_CONTENT"
        if self.agent_id:
            attributes = [
                a
                for a in (os.environ.get("OTEL_RESOURCE_ATTRIBUTES") or "").split(",")
                if a and not a.startswith("authfi.agent_id=")
            ]
            attributes.append(f"authfi.agent_id={self.agent_id}")
            os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(attributes)
