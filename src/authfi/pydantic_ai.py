"""Pydantic AI: Claude on the AI gateway, the gateway's governed tools, approvals through deferred tools.

``pip install "authfi[pydantic-ai] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults
    from authfi import AsyncAuthFI
    from authfi.pydantic_ai import PydanticAIIntegration

    af = AsyncAuthFI.from_env()
    pai = PydanticAIIntegration(af.agent_from_env())
    pai.export_steps()                           # optional (extra "otel"): the run's steps, never its words

    async with pai.toolset() as tools:
        agent = Agent(pai.model("claude-haiku-5-5"), toolsets=[tools], output_type=[str, DeferredToolRequests])
        result = await agent.run("Describe pallets/flask's wiki")
        while isinstance(result.output, DeferredToolRequests):   # a tool call needs a person's approval
            decisions = DeferredToolResults()
            if waiting := await tools.resolve(result.output, decisions):
                ...                                              # show them; once decided, resolve again
                continue
            result = await agent.run(
                message_history=result.all_messages(), deferred_tool_results=decisions
            )

- **Model.** :meth:`PydanticAIIntegration.model` is Pydantic AI's own ``AnthropicModel`` over the AI gateway
  with the agent's current token on every request; it never needs an Anthropic key.
- **Tools.** The gateway's MCP tools through Pydantic AI's ``MCPToolset``, under the gateway's own names
  (``<server>__<tool>``). Tools the gateway says are never allowed for this agent are left out. A refusal
  reaches the model as the tool's result, naming why, and the run carries on. A token the gateway refuses
  outright (401) ends the run with :class:`~authfi.GatewayTokenRefused`.
- **Approvals.** A tool the gateway says may need approval defers the call as Pydantic AI's own
  ``ApprovalRequired``. :meth:`AuthFIToolset.resolve` sends the deferred call to the gateway, which opens the
  approval, and approves it in Pydantic AI only once the gateway has decided. The approval id never reaches
  the model, and one the model writes is removed.
- **Steps.** :meth:`PydanticAIIntegration.export_steps` turns on Pydantic AI's OpenTelemetry instrumentation
  with content off, and sends it to AuthFI tagged with the agent and the run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self, cast

from typing_extensions import override

try:
    from anthropic import AsyncAnthropic
    from fastmcp.client.transports import StreamableHttpTransport
    from pydantic_ai import Agent, ApprovalRequired, DeferredToolRequests, DeferredToolResults, RunContext, ToolCallPart
    from pydantic_ai.mcp import MCPToolset
    from pydantic_ai.messages import ModelMessage, ModelResponse
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.models.wrapper import WrapperModel
    from pydantic_ai.providers.anthropic import AnthropicProvider
    from pydantic_ai.settings import ModelSettings
    from pydantic_ai.toolsets.abstract import ToolsetTool
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.pydantic_ai needs Pydantic AI with MCP: "
        "pip install 'authfi[pydantic-ai] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import PLACEHOLDER_KEY, agent_id_of, async_gateway_for, httpx_of
from authfi._holds import Decided, Held, HeldCalls, gateway_refusal, text_of
from authfi.errors import ConfigurationError
from authfi.gateway import HOLD_MESSAGE, AsyncGateway
from authfi.gateway import PendingApproval as BasePendingApproval
from authfi.tokens import AsyncTokenSource

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from opentelemetry.sdk.trace import TracerProvider
    from pydantic_ai.models import StreamedResponse

    from authfi.client import AsyncAuthFI

__all__ = ["AuthFIModel", "AuthFIToolset", "PendingApproval", "PydanticAIIntegration"]

#: The name the gateway's MCP server has in Pydantic AI; tools keep the gateway's own names.
SERVER_ID = "authfi"


class PydanticAIIntegration:
    """The AuthFI gateways for Pydantic AI.

    Args:
        source: an asyncio credential (``AsyncAuthFI.agent(...)``, or the ``AsyncDelegatedToken`` it holds
            for a person). Pydantic AI's MCP client runs on asyncio.
        project: the project, for a credential of your own that does not name it.

    """

    def __init__(self, source: AsyncTokenSource, *, project: AsyncAuthFI | None = None) -> None:
        self.source = source
        self.gateway = async_gateway_for(source, project)
        self.agent_id = agent_id_of(source)

    @property
    def run_id(self) -> str:
        """The credential's run: on every gateway call (``X-Request-Id``) and every exported step."""
        return cast(str, getattr(self.source, "run_id", None) or "")

    def model(self, model: str) -> AuthFIModel:
        """Claude on the AI gateway, called with the agent's token. Pass it as ``Agent(model, ...)``."""
        client = AsyncAnthropic(
            base_url=self.gateway.model_base_url("anthropic"),
            api_key=PLACEHOLDER_KEY,
            http_client=self.gateway.http_client(httpx_of("anthropic._base_client")),
        )
        return AuthFIModel(AnthropicModel(model, provider=AnthropicProvider(anthropic_client=client)))

    def toolset(self, *, include_never_allowed: bool = False, **kwargs: Any) -> AuthFIToolset:
        """The MCP gateway as a Pydantic AI toolset. Use it with ``async with``; ``kwargs`` are ``MCPToolset``'s."""
        return AuthFIToolset(self.gateway, include_never_allowed=include_never_allowed, **kwargs)

    def export_steps(self, *, capture_content: bool = False, **options: Any) -> TracerProvider:
        """Send this agent's steps to AuthFI (extra ``otel``): Pydantic AI's own OpenTelemetry spans.

        Turns on Pydantic AI's instrumentation for every agent (``Agent.instrument_all``), with no prompts,
        answers or tool arguments unless ``capture_content``, and exports the spans tagged ``authfi.agent_id``
        and ``authfi.run_id``. Call it once, before the first run. ``options`` are those of
        :func:`authfi.otel.register`.
        """
        from authfi import otel

        provider = otel.register(self.source, capture_content=capture_content, **options)
        Agent.instrument_all(InstrumentationSettings(tracer_provider=provider, include_content=capture_content))
        return provider


class AuthFIModel(WrapperModel):
    """Pydantic AI's Anthropic model on the AI gateway; a token the gateway refuses ends the run.

    Make it with :meth:`PydanticAIIntegration.model`.
    """

    @override
    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        try:
            return await super().request(messages, model_settings, model_request_parameters)
        except Exception as error:
            refused = gateway_refusal(error)
            if refused is not None:
                raise refused from error
            raise

    @override
    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[Any] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        try:
            async with super().request_stream(
                messages, model_settings, model_request_parameters, run_context
            ) as stream:
                yield stream
        except Exception as error:
            refused = gateway_refusal(error)
            if refused is not None:
                raise refused from error
            raise


@dataclass(frozen=True, slots=True)
class PendingApproval(BasePendingApproval):
    """A deferred tool call whose approval a person has not decided yet; ``call`` is Pydantic AI's own."""

    call: ToolCallPart


def _http_module() -> Any:
    """The httpx package FastMCP's HTTP transport is built on: ``httpx2`` from FastMCP 4, ``httpx`` before."""
    return httpx_of("fastmcp.client.transports.http")


def _client_factory(gateway: AsyncGateway) -> Callable[..., Any]:
    """FastMCP's HTTP client factory for the gateway: its clients (no redirects, the token on every request)."""
    inner = gateway.mcp_client_factory(_http_module())

    def factory(headers: Mapping[str, str] | None = None, timeout: Any = None, auth: Any = None, **_: Any) -> Any:
        return inner(headers=headers, timeout=timeout, auth=auth)

    return factory


def _outlook(tool_def: object) -> str | None:
    metadata = cast("Mapping[str, Any]", getattr(tool_def, "metadata", None) or {})
    return AsyncGateway.governance_of({"annotations": metadata.get("annotations"), "_meta": metadata.get("meta")})


def _for_the_model(entry: Decided) -> Any:
    if entry.refusal is not None:
        return entry.refusal
    result = entry.result
    text = text_of(result)
    structured = getattr(result, "structured_content", None)
    return structured if structured is not None and not text else text


class AuthFIToolset(MCPToolset[Any]):
    """The MCP gateway as a Pydantic AI toolset. Make it with :meth:`PydanticAIIntegration.toolset`.

    Every request carries the current token. Tool names are the gateway's. A tool the gateway marks
    ``never_allowed`` is left out, unless ``include_never_allowed``. A refusal reaches the model as the
    tool's result, naming why. A tool the gateway marks ``may_require_approval`` defers the call for
    approval (see :meth:`resolve`).
    """

    def __init__(self, gateway: AsyncGateway, *, include_never_allowed: bool = False, **kwargs: Any) -> None:
        for refused in ("client", "http_client", "headers", "auth", "verify", "process_tool_call"):
            if refused in kwargs:
                raise ConfigurationError(f"{refused} is set by the integration")
        kwargs.setdefault("id", SERVER_ID)
        transport = StreamableHttpTransport(gateway.mcp_url, httpx_client_factory=_client_factory(gateway))
        super().__init__(transport, **kwargs)
        self.gateway = gateway
        self.include_never_allowed = include_never_allowed
        self._held = HeldCalls(gateway)
        self._listed: set[str] = set()

    @override
    async def __aenter__(self) -> Self:
        try:
            return await super().__aenter__()
        except Exception as error:
            refused = gateway_refusal(error)
            if refused is not None:
                raise refused from error
            raise

    @override
    async def get_tools(self, ctx: RunContext[Any]) -> dict[str, ToolsetTool[Any]]:
        try:
            tools = await super().get_tools(ctx)
        except Exception as error:
            refused = gateway_refusal(error)
            if refused is not None:
                raise refused from error
            raise
        out: dict[str, ToolsetTool[Any]] = {}
        for name, tool in tools.items():
            if _outlook(tool.tool_def) == "never_allowed" and not self.include_never_allowed:
                self.gateway.project.logger.info("leaving out %s: never allowed for this agent", name)
                continue
            out[name] = tool
            self._listed.add(name)
        return out

    def _needs_approval(self, name: str, tool_def: object) -> bool:
        return _outlook(tool_def) == "may_require_approval" or self._held.holds(name)

    async def _raw(self, name: str, args: dict[str, Any]) -> Any:
        return await self.client.call_tool(name=name, arguments=args, raise_on_error=False)

    async def _send(self, name: str, args: dict[str, Any], approval_id: str | None) -> Held | Decided:
        async def call(sent: dict[str, Any]) -> Any:
            return await self._raw(name, sent)

        return await self._held.send(call, name, args, approval_id)

    @override
    async def call_tool(
        self, name: str, tool_args: dict[str, Any], ctx: RunContext[Any], tool: ToolsetTool[Any]
    ) -> Any:
        args = self.gateway.strip_approval_argument(tool_args)
        if self._needs_approval(name, tool.tool_def) and not ctx.tool_call_approved:
            raise ApprovalRequired
        known = self._held.take(name, args)
        if isinstance(known, Decided):  # resolve() already sent it and the gateway decided
            return _for_the_model(known)
        entry = await self._send(name, args, known.approval_id if known else None)
        if isinstance(entry, Held):
            self._held.remember(name, args, entry)
            return HOLD_MESSAGE
        return _for_the_model(entry)

    async def resolve(self, requests: DeferredToolRequests, results: DeferredToolResults) -> list[PendingApproval]:
        """Approve in Pydantic AI each deferred call of this toolset that the gateway has decided.

        Each call is sent to the gateway with its approval, if it has one: the first time, the gateway opens
        the approval (or decides at once). A call the gateway decided is approved in ``results``, and the
        run gets the gateway's answer, the tool's result or the refusal, without sending it again. The
        calls still waiting for a person are returned; resolve again once they decide. Pass the same
        ``results`` to ``agent.run(deferred_tool_results=...)`` when none are waiting.
        """
        waiting: list[PendingApproval] = []
        for call in requests.approvals:
            if call.tool_name not in self._listed:
                continue
            args = self.gateway.strip_approval_argument(call.args_as_dict())
            known = self._held.get(call.tool_name, args)
            if not isinstance(known, Decided):
                known = await self._send(call.tool_name, args, known.approval_id if known else None)
                self._held.remember(call.tool_name, args, known)
            if isinstance(known, Held):
                waiting.append(PendingApproval(known.approval_id, call.tool_name, args, call))
                continue
            results.approvals[call.tool_call_id] = True
        return waiting
