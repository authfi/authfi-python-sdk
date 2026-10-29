"""LlamaIndex: Claude on the AI gateway, the gateway's governed tools, approvals a workflow waits for, steps.

``pip install "authfi[llama-index] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from llama_index.core.agent.workflow import FunctionAgent
    from authfi import AsyncAuthFI
    from authfi.llama_index import LlamaIndexIntegration

    af = AsyncAuthFI.from_env()
    llama = LlamaIndexIntegration(af.agent_from_env())
    llama.export_steps()                       # optional (extra "otel"): the run's steps, never its words

    agent = FunctionAgent(tools=await llama.tools(), llm=llama.llm("claude-haiku-5-5"))
    async with llama.guard():                  # a token the gateway refuses ends the run
        answer = await agent.run("Describe pallets/flask's wiki")

- **Model.** :meth:`LlamaIndexIntegration.llm` is LlamaIndex's own ``Anthropic`` LLM over the AI gateway with
  the agent's current token on every request; it never needs an Anthropic key. It runs on asyncio (workflows
  and agents call ``achat``), and a blocking call is refused with a message that says so.
- **Tools.** The gateway's MCP tools through LlamaIndex's own ``McpToolSpec`` and ``BasicMCPClient``, under
  the gateway's own names (``<server>__<tool>``). Tools the gateway says are never allowed for this agent
  are left out. A refusal reaches the model as the tool's error result, naming why, and the run carries on.
- **Approvals.** A tool call the gateway holds for a person waits for that person: the tool tells
  ``on_approval`` once, asks the gateway again every ``poll_interval`` seconds, and returns the tool's
  result when the person approves, or the refusal when they deny it. After ``approval_timeout`` seconds the
  model reads that the approval did not come, and the approval stays open. The approval id never reaches
  the model, and one the model writes is removed.
- **Token refused.** LlamaIndex hands a failed tool call to the model as text, so :meth:`LlamaIndexIntegration.guard`
  turns the gateway's refusal of the token (401) into :class:`~authfi.GatewayTokenRefused` when the run ends.
- **Steps.** :meth:`LlamaIndexIntegration.export_steps` turns LlamaIndex's instrumentation spans into
  OpenTelemetry GenAI spans (the workflow, each model call and each tool call) tagged with the agent and the
  run. Words are never read.
"""

# pyright: reportPrivateImportUsage=false, reportMissingTypeStubs=false, reportUnknownMemberType=false
# LlamaIndex re-exports its instrumentation API without __all__ and ships its MCP package untyped.
from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

from typing_extensions import override

try:
    import anthropic
    import httpx
    import llama_index.core.instrumentation as instrumentation
    from llama_index.core.base.llms.types import LLMMetadata
    from llama_index.core.instrumentation.span.base import BaseSpan
    from llama_index.core.instrumentation.span_handlers.base import BaseSpanHandler
    from llama_index.llms.anthropic import Anthropic
    from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
    from mcp.types import CallToolResult, ListToolsResult, TextContent
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.llama_index needs LlamaIndex with its Anthropic and MCP packages: "
        "pip install 'authfi[llama-index] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import PLACEHOLDER_KEY, agent_id_of, async_gateway_for, httpx_of
from authfi._holds import Decided, Held, HeldCalls, guard, wait_for_decision
from authfi.errors import ConfigurationError
from authfi.gateway import AsyncGateway
from authfi.gateway import PendingApproval as BasePendingApproval
from authfi.tokens import AsyncTokenSource

if TYPE_CHECKING:
    from llama_index.core.tools import FunctionTool
    from opentelemetry.sdk.trace import TracerProvider

    from authfi.client import AsyncAuthFI

__all__ = ["AuthFILLM", "AuthFIMCPClient", "AuthFIStepsHandler", "LlamaIndexIntegration", "PendingApproval"]

#: How long a held call waits for a person, and how often the gateway is asked again.
APPROVAL_TIMEOUT = 900.0
POLL_INTERVAL = 5.0

#: Told once when a tool call starts waiting for a person; may be a plain function or a coroutine function.
OnApproval = Callable[["PendingApproval"], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class PendingApproval(BasePendingApproval):
    """A tool call the gateway holds until a person decides ``approval_id``."""


class LlamaIndexIntegration:
    """The AuthFI gateways for LlamaIndex.

    Args:
        source: an asyncio credential (``AsyncAuthFI.agent(...)``, or the ``AsyncDelegatedToken`` it holds for a
            person). LlamaIndex's agents and workflows run on asyncio.
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

    def llm(self, model: str, **kwargs: Any) -> AuthFILLM:
        """Claude on the AI gateway as LlamaIndex's ``Anthropic`` LLM, called with the agent's token."""
        for refused in ("base_url", "api_key", "region", "project_id", "aws_region"):
            if refused in kwargs:
                raise ConfigurationError(f"{refused} is set by the integration")
        return AuthFILLM(self.gateway, model=model, **kwargs)

    async def tools(
        self,
        *,
        include_never_allowed: bool = False,
        on_approval: OnApproval | None = None,
        approval_timeout: float = APPROVAL_TIMEOUT,
        poll_interval: float = POLL_INTERVAL,
        **kwargs: Any,
    ) -> list[FunctionTool]:
        """The MCP gateway's tools as LlamaIndex tools, for ``FunctionAgent(tools=...)``.

        A tool the gateway marks ``never_allowed`` is left out, unless ``include_never_allowed``. A tool call the
        gateway holds for approval waits for the person (see the module notes); ``on_approval`` is told once,
        with the :class:`PendingApproval`, so the application can say where to decide it. ``kwargs`` are
        ``McpToolSpec``'s.
        """
        client = AuthFIMCPClient(
            self.gateway,
            include_never_allowed=include_never_allowed,
            on_approval=on_approval,
            approval_timeout=approval_timeout,
            poll_interval=poll_interval,
        )
        return await McpToolSpec(client, **kwargs).to_tool_list_async()

    def guard(self) -> AbstractAsyncContextManager[None]:
        """End the run with :class:`~authfi.GatewayTokenRefused` if the gateway refuses the token while it runs."""
        return guard(self.gateway)

    def export_steps(self, *, capture_content: bool = False, **options: Any) -> TracerProvider:
        """Send this agent's steps to AuthFI (extra ``otel``): LlamaIndex's spans as OpenTelemetry GenAI spans.

        The spans name the workflow, each model call and each tool call with timings and status, and never the
        words (``capture_content`` is accepted for the exporter's own policy and adds no content here: the
        bridge reads none). Call it once, before the first run. ``options`` are those of
        :func:`authfi.otel.register`.
        """
        from authfi import otel

        provider = otel.register(self.source, capture_content=capture_content, **options)
        AuthFIStepsHandler.attach(provider)
        return provider


def _refusing_client(package: ModuleType) -> Any:
    """A blocking HTTP client (of the package the provider SDK uses) that refuses every request."""

    def refuse(request: Any) -> Any:
        raise ConfigurationError(
            "the AuthFI credential runs on asyncio: call the LLM with achat(), astream_chat() or an agent's run()"
        )

    return package.Client(transport=package.MockTransport(refuse))


class AuthFILLM(Anthropic):
    """LlamaIndex's Anthropic LLM on the AI gateway. Make it with :meth:`LlamaIndexIntegration.llm`."""

    def __init__(self, gateway: AsyncGateway, **kwargs: Any) -> None:
        kwargs["base_url"] = gateway.model_base_url("anthropic")
        kwargs["api_key"] = PLACEHOLDER_KEY
        super().__init__(**kwargs)
        # LlamaIndex builds its provider clients itself and takes no HTTP client, so these two are replaced:
        # the asyncio one carries the token, the blocking one says why it cannot.
        package = httpx_of("anthropic._base_client")
        timeout, retries = kwargs.get("timeout"), kwargs.get("max_retries", 10)
        self._aclient = anthropic.AsyncAnthropic(
            api_key=PLACEHOLDER_KEY,
            base_url=gateway.model_base_url("anthropic"),
            timeout=timeout,
            max_retries=retries,
            http_client=gateway.http_client(package),
        )
        self._client = anthropic.Anthropic(
            api_key=PLACEHOLDER_KEY,
            base_url=gateway.model_base_url("anthropic"),
            timeout=timeout,
            max_retries=0,
            http_client=_refusing_client(package),
        )

    @property
    @override
    def metadata(self) -> LLMMetadata:
        """LlamaIndex's metadata, with a default window for a model its table lags behind the gateway on."""
        try:
            return super().metadata
        except ValueError:
            return LLMMetadata(
                context_window=UNLISTED_CONTEXT_WINDOW,
                num_output=self.max_tokens,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=self.model,
            )


#: The context window assumed for a model LlamaIndex's table does not list yet (the gateway may serve newer ones).
UNLISTED_CONTEXT_WINDOW = 200_000


def _text_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


class AuthFIMCPClient(BasicMCPClient):
    """LlamaIndex's MCP client for the gateway. Make it through :meth:`LlamaIndexIntegration.tools`.

    Every request carries the current token. A tool the gateway marks ``never_allowed`` is left out of the
    list. A refusal comes back as the tool's error result; a hold waits for the person.
    """

    def __init__(
        self,
        gateway: AsyncGateway,
        *,
        include_never_allowed: bool = False,
        on_approval: OnApproval | None = None,
        approval_timeout: float = APPROVAL_TIMEOUT,
        poll_interval: float = POLL_INTERVAL,
    ) -> None:
        super().__init__(gateway.mcp_url, http_client=gateway.mcp_client_factory(httpx)())
        self.gateway = gateway
        self.include_never_allowed = include_never_allowed
        self.on_approval = on_approval
        self.approval_timeout = approval_timeout
        self.poll_interval = poll_interval
        self._held = HeldCalls(gateway)

    @override
    async def list_tools(self) -> ListToolsResult:
        listed = await super().list_tools()
        if self.include_never_allowed:
            return listed
        kept = []
        for tool in listed.tools:
            if self.gateway.governance_of(tool) == "never_allowed":
                self.gateway.project.logger.info("leaving out %s: never allowed for this agent", tool.name)
                continue
            kept.append(tool)
        return listed.model_copy(update={"tools": kept})

    async def _send(self, tool: str, args: dict[str, Any], approval_id: str | None) -> Held | Decided:
        async def call(sent: dict[str, Any]) -> CallToolResult:
            return await super(AuthFIMCPClient, self).call_tool(tool, sent)

        return await self._held.send(call, tool, args, approval_id)

    @override
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None = None, *args: Any, **kwargs: Any
    ) -> Any:
        clean = self.gateway.strip_approval_argument(arguments)

        async def told(approval_id: str) -> None:
            if self.on_approval is not None:
                answer = self.on_approval(PendingApproval(approval_id, tool_name, clean))
                if answer is not None:
                    await answer

        decided = await wait_for_decision(
            lambda approval_id: self._send(tool_name, clean, approval_id),
            on_held=told,
            timeout=self.approval_timeout,
            poll_interval=self.poll_interval,
        )
        if decided is None:
            return _text_result(
                f"The approval for {tool_name} did not come in time, so the action did not run. "
                "Tell the person it still waits for their decision."
            )
        if decided.refusal is not None:
            return _text_result(decided.refusal)
        return decided.result


# --- steps ------------------------------------------------------------------------------------------


class _Step(BaseSpan):
    """One LlamaIndex span, and the OpenTelemetry span it became."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    otel: Any = None


_ATTACHED: dict[int, AuthFIStepsHandler] = {}
_LOCK = threading.Lock()


class AuthFIStepsHandler(BaseSpanHandler[_Step]):
    """LlamaIndex's instrumentation spans as OpenTelemetry GenAI spans on a provider: steps, never words.

    A span is named for what LlamaIndex ran (the workflow, a model call, a tool call). No input, output,
    prompt or tool argument is read, so none can reach a span. A span is a child of the span LlamaIndex says
    is its parent, so a gateway call made inside it joins the steps' trace.
    """

    def __init__(self, provider: TracerProvider) -> None:
        super().__init__()
        self._tracer = provider.get_tracer("authfi.llama_index")

    @classmethod
    def attach(cls, provider: TracerProvider) -> AuthFIStepsHandler:
        """Listen on LlamaIndex's root dispatcher for ``provider``; a second call for it adds nothing."""
        with _LOCK:
            found = _ATTACHED.get(id(provider))
            if found is None:
                found = cls(provider)
                instrumentation.get_dispatcher().add_span_handler(found)
                _ATTACHED[id(provider)] = found
            return found

    def detach(self) -> None:
        """Stop listening."""
        root = instrumentation.get_dispatcher()
        handlers = cast("list[Any]", root.span_handlers)
        handlers[:] = [h for h in handlers if h is not self]
        with _LOCK:
            for key, value in list(_ATTACHED.items()):
                if value is self:
                    del _ATTACHED[key]

    @classmethod
    @override
    def class_name(cls) -> str:
        return "AuthFIStepsHandler"

    @override
    def new_span(
        self,
        id_: str,
        bound_args: Any,
        instance: Any = None,
        parent_span_id: str | None = None,
        tags: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _Step | None:
        described = _describe(id_, instance)
        if described is None:
            return None
        name, attrs = described
        from opentelemetry.trace import set_span_in_context

        parent = self.open_spans.get(parent_span_id) if parent_span_id else None
        context = set_span_in_context(parent.otel) if parent is not None and parent.otel is not None else None
        step = _Step(id_=id_, parent_id=parent_span_id)
        step.otel = self._tracer.start_span(name, context=context, attributes=attrs)
        return step

    @override
    def prepare_to_exit_span(
        self, id_: str, bound_args: Any, instance: Any = None, result: Any = None, **kwargs: Any
    ) -> _Step | None:
        step = self.open_spans.get(id_)
        if step is not None and step.otel is not None:
            step.otel.end()
        return step

    @override
    def prepare_to_drop_span(
        self, id_: str, bound_args: Any, instance: Any = None, err: BaseException | None = None, **kwargs: Any
    ) -> _Step | None:
        step = self.open_spans.get(id_)
        if step is not None and step.otel is not None:
            from opentelemetry.trace import Status, StatusCode

            step.otel.set_status(Status(StatusCode.ERROR))
            step.otel.end()
        return step


def _describe(id_: str, instance: Any) -> tuple[str, dict[str, str | int]] | None:
    """The OpenTelemetry GenAI name and attributes of one LlamaIndex span, or ``None`` for one not worth a step."""
    function = id_[:-37] if len(id_) > 37 and id_[-37] == "-" else id_  # "Owner.method-<uuid>"
    owner, _, method = function.partition(".")
    if method in {"achat", "astream_chat", "achat_with_tools", "astream_chat_with_tools", "chat", "stream_chat"}:
        model = getattr(instance, "model", None)
        attrs: dict[str, str | int] = {"gen_ai.operation.name": "chat"}
        if isinstance(model, str) and model:
            attrs["gen_ai.request.model"] = model
        return (f"chat {model}" if isinstance(model, str) and model else "chat", attrs)
    if owner in {"FunctionTool", "AsyncBaseTool", "BaseTool"} and method in {"acall", "call"}:
        tool = getattr(getattr(instance, "metadata", None), "name", None) or "tool"
        return (f"execute_tool {tool}", {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": str(tool)})
    if method == "run" and owner.endswith(("Workflow", "Agent")):
        return (f"invoke_agent {owner}", {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": owner})
    return None
