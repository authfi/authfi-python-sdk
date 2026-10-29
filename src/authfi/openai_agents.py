"""The OpenAI Agents SDK: traces with no token and no content, governed tools, approvals through the SDK.

``pip install "authfi[openai-agents] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from agents import Agent, Runner
    from authfi import AsyncAuthFI
    from authfi.openai_agents import OpenAIAgentsIntegration

    af = AsyncAuthFI.from_env()
    openai_agents = OpenAIAgentsIntegration(af.agent_from_env())
    openai_agents.configure_tracing()            # first: no default exporter, no content in spans

    async with openai_agents.mcp_server() as tools:
        agent = Agent(name="research", model=openai_agents.model("claude-haiku-5-5"), mcp_servers=[tools])
        result = await Runner.run(agent, "Describe pallets/flask's wiki", run_config=openai_agents.run_config())
        while result.interruptions:              # a tool call needs a person's approval
            state = result.to_state()
            waiting = await tools.resolve(state, result.interruptions)
            if waiting:
                ...                              # show them; once the person has decided, resolve again
                continue
            result = await Runner.run(agent, state, run_config=openai_agents.run_config())

**Tracing.** The SDK's default exporter sends every trace to OpenAI with ``Authorization: Bearer
<the OpenAI key>``, and ``trace_include_sensitive_data`` defaults to on. Where the AuthFI token
doubled as the OpenAI key, each export carried it, with the prompts and tool results.
:meth:`OpenAIAgentsIntegration.configure_tracing` removes that exporter and keeps content out of
spans; the model here never needs an OpenAI key.

**Approvals.** The SDK asks for approval before a call; the gateway decides when the call arrives.
A tool the gateway says may need approval pauses the run as the SDK's own ``ToolApprovalItem``.
:meth:`AuthFIMCPServer.resolve` sends the paused call to the gateway, which opens the approval, and
approves the item in the SDK only once the gateway has decided. The approval id never reaches the
model.
"""

from __future__ import annotations

import dataclasses
import json
import os
from collections import OrderedDict
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, cast

from typing_extensions import override

try:
    import httpx
    from agents import AgentBase, Model, ModelSettings, RunConfig, RunContextWrapper, RunState
    from agents.extensions.models.litellm_model import LitellmModel
    from agents.items import ToolApprovalItem
    from agents.mcp import MCPServerStreamableHttp
    from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
    from agents.tracing import TracingProcessor, set_trace_processors, set_tracing_disabled
    from agents.tracing.span_data import (
        CustomSpanData,
        FunctionSpanData,
        GenerationSpanData,
        ResponseSpanData,
        SpeechGroupSpanData,
        SpeechSpanData,
        TranscriptionSpanData,
    )
    from agents.tracing.spans import Span
    from agents.tracing.traces import Trace
    from mcp.types import CallToolResult, TextContent
    from mcp.types import Tool as MCPTool
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.openai_agents needs the OpenAI Agents SDK: "
        "pip install 'authfi[openai-agents] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import PLACEHOLDER_KEY, agent_id_of, async_gateway_for, httpx_of
from authfi.errors import ConfigurationError
from authfi.gateway import HOLD_MESSAGE, AsyncGateway, ToolApprovalRequired, ToolRefused, token_refusal
from authfi.gateway import PendingApproval as BasePendingApproval
from authfi.tokens import AsyncTokenSource

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI

__all__ = [
    "AuthFIMCPServer",
    "AuthFIModel",
    "AuthFITraceProcessor",
    "OpenAIAgentsIntegration",
    "PendingApproval",
]

#: How many held or decided calls a server remembers until the run comes back for them.
MEMORY = 10_000


class OpenAIAgentsIntegration:
    """The AuthFI gateways for the OpenAI Agents SDK.

    Args:
        source: an asyncio credential (``AsyncAuthFI.agent(...)``, or the ``AsyncDelegatedToken``
            it holds for a person). The SDK runs on asyncio.
        project: the project, for a credential of your own that does not name it.

    """

    def __init__(self, source: AsyncTokenSource, *, project: AsyncAuthFI | None = None) -> None:
        self.source = source
        self.gateway = async_gateway_for(source, project)
        self.agent_id = agent_id_of(source)

    def model(self, model: str, *, api: Literal["anthropic", "openai"] = "anthropic") -> Model:
        """A model on the AI gateway, called with the agent's token.

        ``api="anthropic"`` (Claude) goes through LiteLLM's Anthropic client; ``api="openai"`` is the
        SDK's own Chat Completions model, for a project with an OpenAI-compatible provider. Neither
        needs an OpenAI key.
        """
        if api == "openai":
            client = AsyncOpenAI(
                base_url=self.gateway.model_base_url("openai"),
                api_key=PLACEHOLDER_KEY,
                http_client=self.gateway.http_client(httpx_of("openai._base_client")),
            )
            return OpenAIChatCompletionsModel(model=model, openai_client=client)
        if api != "anthropic":
            raise ConfigurationError(f"the AI gateway speaks the anthropic and openai APIs, not {api!r}")
        return AuthFIModel(self.gateway, model)

    def mcp_server(self, *, include_never_allowed: bool = False, **kwargs: Any) -> AuthFIMCPServer:
        """The MCP gateway as the SDK's MCP server. Use it with ``async with``; ``kwargs`` are the server's."""
        return AuthFIMCPServer(self.gateway, include_never_allowed=include_never_allowed, **kwargs)

    def run_config(self, *, capture_content: bool = False, **kwargs: Any) -> RunConfig:
        """A ``RunConfig`` whose trace names the agent (``authfi_agent_id``) and holds no content unless asked."""
        metadata = dict(kwargs.pop("trace_metadata", None) or {})
        if self.agent_id:
            metadata["authfi_agent_id"] = self.agent_id
        kwargs.setdefault("trace_include_sensitive_data", capture_content)
        return RunConfig(trace_metadata=metadata or None, **kwargs)

    def configure_tracing(
        self, processors: Sequence[TracingProcessor] = (), *, capture_content: bool = False
    ) -> list[TracingProcessor]:
        """Trace without the token and, unless ``capture_content``, without content. Call it before the first run.

        This replaces every trace processor the application registered before it, not only the SDK's
        default exporter (to OpenAI, with the OpenAI key). ``processors`` take their place, each
        wrapped in :class:`AuthFITraceProcessor`: to keep one the application already has, pass it
        here. With none, tracing is off. A run made without :meth:`run_config` still reaches the
        processors without content, because the wrapper removes it; ``capture_content`` turns it on,
        which is the tenant's policy decision.
        """
        if (os.environ.get("OPENAI_API_KEY") or "").startswith("eyJ"):
            self.gateway.project.logger.warning(
                "OPENAI_API_KEY holds a JWT; if it is an AuthFI token, unset it: the gateway models do not need "
                "it, and an exporter that sends to OpenAI sends it"
            )
        wrapped: list[TracingProcessor] = [
            p if isinstance(p, AuthFITraceProcessor) else AuthFITraceProcessor(p, capture_content=capture_content)
            for p in processors
        ]
        set_trace_processors(wrapped)
        set_tracing_disabled(not wrapped)
        return wrapped


#: Settings that can carry a credential or a user's data, dropped from a span's ``model_config``.
_MODEL_CONFIG_SECRETS = ("extra_headers", "extra_args", "extra_query")


def _without_secrets(model_config: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if model_config is None:
        return None
    return {k: v for k, v in model_config.items() if k not in _MODEL_CONFIG_SECRETS}


class AuthFITraceProcessor(TracingProcessor):
    """Hands spans to another processor without their content, unless ``capture_content``.

    Prompts, answers and tool inputs and outputs are removed from generation, response, function,
    custom, speech and transcription spans before ``inner`` sees them, and so are the model settings
    that can hold a header or a query (``extra_headers``, ``extra_args``, ``extra_query``), which this
    adapter puts the token in. Names, timings, ids, usage, the other settings and the trace's metadata
    are kept.
    """

    def __init__(self, inner: TracingProcessor, *, capture_content: bool = False) -> None:
        self.inner = inner
        self.capture_content = capture_content

    @override
    def on_trace_start(self, trace: Trace) -> None:
        self.inner.on_trace_start(trace)

    @override
    def on_trace_end(self, trace: Trace) -> None:
        self.inner.on_trace_end(trace)

    @override
    def on_span_start(self, span: Span[Any]) -> None:
        self.inner.on_span_start(span)

    @override
    def on_span_end(self, span: Span[Any]) -> None:
        data = span.span_data
        if isinstance(data, GenerationSpanData | SpeechSpanData | TranscriptionSpanData):
            # A header is never content, so these come off whether content is captured or not.
            data.model_config = _without_secrets(data.model_config)
        if not self.capture_content:
            self._scrub(data)
        self.inner.on_span_end(span)

    @staticmethod
    def _scrub(data: object) -> None:
        if isinstance(data, GenerationSpanData):
            data.input = None
            data.output = None
        elif isinstance(data, FunctionSpanData):
            data.input = None
            data.output = None
        elif isinstance(data, ResponseSpanData):
            if data.response is not None:  # keep the id, which the span exports, and drop the answer
                data._response_id = data.response.id  # pyright: ignore[reportPrivateUsage]
            data.response = None
            data.input = None
        elif isinstance(data, CustomSpanData):
            data.data = {}
        elif isinstance(data, SpeechSpanData | TranscriptionSpanData):
            data.input = None
            data.output = None
        elif isinstance(data, SpeechGroupSpanData):
            data.input = None

    @override
    def shutdown(self) -> None:
        self.inner.shutdown()

    @override
    def force_flush(self) -> None:
        self.inner.force_flush()


class AuthFIModel(LitellmModel):
    """Claude on the AI gateway, through LiteLLM, with the current token on each call.

    The token is added to one call's headers, which the SDK does not record; the agent's model
    settings are not changed. LiteLLM insists on an API key, which it sends as ``x-api-key``; a
    placeholder stands in, as the gateway reads ``Authorization``.
    """

    def __init__(self, gateway: AsyncGateway, model: str) -> None:
        super().__init__(
            model=f"anthropic/{model}", base_url=gateway.model_base_url("anthropic"), api_key=PLACEHOLDER_KEY
        )
        self.gateway = gateway

    async def _with_token(self, settings: ModelSettings) -> ModelSettings:
        url = self.gateway.model_base_url("anthropic") + "/v1/messages"
        headers = {**(settings.extra_headers or {}), **(await self.gateway.source.headers("POST", url))}
        return dataclasses.replace(settings, extra_headers=headers)

    @override
    async def get_response(
        self, system_instructions: str | None, input: Any, model_settings: ModelSettings, *args: Any, **kwargs: Any
    ) -> Any:
        settings = await self._with_token(model_settings)
        try:
            return await super().get_response(system_instructions, input, settings, *args, **kwargs)
        except Exception as error:
            refused = token_refusal(error)
            if refused is not None:
                raise refused from error
            raise

    @override
    async def stream_response(
        self, system_instructions: str | None, input: Any, model_settings: ModelSettings, *args: Any, **kwargs: Any
    ) -> AsyncIterator[Any]:
        settings = await self._with_token(model_settings)
        try:
            async for event in super().stream_response(system_instructions, input, settings, *args, **kwargs):
                yield event
        except Exception as error:
            refused = token_refusal(error)
            if refused is not None:
                raise refused from error
            raise


@dataclass(frozen=True, slots=True)
class PendingApproval(BasePendingApproval):
    """A paused tool call whose approval a person has not decided yet; ``item`` is the SDK's own."""

    item: ToolApprovalItem


@dataclass(frozen=True, slots=True)
class _Held:
    approval_id: str


@dataclass(frozen=True, slots=True)
class _Decided:
    result: CallToolResult


def _key(tool: str, args: Mapping[str, Any]) -> str:
    return tool + "\x00" + json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


def _text_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)


class AuthFIMCPServer(MCPServerStreamableHttp):
    """The MCP gateway as the SDK's streamable HTTP MCP server. Make it with ``mcp_server()``.

    Every request carries the current token. Tool names are the gateway's. A tool the gateway marks
    ``never_allowed`` is left out, unless ``include_never_allowed``. A refusal reaches the model as
    the tool's error, naming why. A tool the gateway marks ``may_require_approval`` pauses the run
    for approval (see :meth:`resolve`).
    """

    def __init__(self, gateway: AsyncGateway, *, include_never_allowed: bool = False, **kwargs: Any) -> None:
        for refused in ("params", "require_approval"):
            if refused in kwargs:
                raise ConfigurationError(f"{refused} is set by the integration")

        kwargs.setdefault("name", "authfi")
        kwargs.setdefault("client_session_timeout_seconds", 60)
        super().__init__(
            params={
                "url": gateway.mcp_url,
                "httpx_client_factory": gateway.mcp_client_factory(httpx),
                "timeout": 60,
                "sse_read_timeout": 300,
            },
            require_approval=self._needs_approval,
            **kwargs,
        )
        self.gateway = gateway
        self.include_never_allowed = include_never_allowed
        self._calls: OrderedDict[str, _Held | _Decided] = OrderedDict()
        self._listed: set[str] = set()

    @override
    async def list_tools(
        self, run_context: RunContextWrapper[Any] | None = None, agent: AgentBase | None = None
    ) -> list[MCPTool]:
        out: list[MCPTool] = []
        for tool in await super().list_tools(run_context, agent):
            if self.gateway.governance_of(tool) == "never_allowed" and not self.include_never_allowed:
                self.gateway.project.logger.info("leaving out %s: never allowed for this agent", tool.name)
                continue
            out.append(tool)
            self._listed.add(tool.name)
        return out

    def _needs_approval(self, run_context: RunContextWrapper[Any], agent: AgentBase, tool: MCPTool) -> bool:
        if self.gateway.governance_of(tool) == "may_require_approval":
            return True
        prefix = tool.name + "\x00"
        return any(key.startswith(prefix) and isinstance(entry, _Held) for key, entry in self._calls.items())

    def _remember(self, key: str, entry: _Held | _Decided) -> None:
        self._calls[key] = entry
        self._calls.move_to_end(key)
        while len(self._calls) > MEMORY:
            self._calls.popitem(last=False)

    async def _send(
        self, tool: str, args: dict[str, Any], approval_id: str | None, meta: dict[str, Any] | None
    ) -> _Held | _Decided:
        sent = self.gateway.with_approval(args, approval_id) if approval_id else args
        try:
            result = await super().call_tool(tool, sent, meta)
        except Exception as error:
            refused = token_refusal(error)
            if refused is not None:
                raise refused from error
            raise
        outcome = self.gateway.read_result(result)
        if isinstance(outcome, ToolApprovalRequired):
            self.gateway.project.logger.info("%s waits for approval %s", tool, outcome.approval_id)
            return _Held(outcome.approval_id)
        if isinstance(outcome, ToolRefused):
            self.gateway.project.logger.info("the gateway refused %s: %s", tool, outcome.code or outcome.error)
            return _Decided(_text_result(outcome.model_text))
        return _Decided(result)

    @override
    async def call_tool(
        self, tool_name: str, arguments: dict[str, Any] | None, meta: dict[str, Any] | None = None
    ) -> CallToolResult:
        args = self.gateway.strip_approval_argument(arguments)
        key = _key(tool_name, args)
        known = self._calls.pop(key, None)
        if isinstance(known, _Decided):  # resolve() already sent it and the gateway decided
            return known.result
        entry = await self._send(tool_name, args, known.approval_id if known else None, meta)
        if isinstance(entry, _Held):
            self._remember(key, entry)
            return _text_result(HOLD_MESSAGE)
        return entry.result

    async def resolve(self, state: RunState[Any, Any], interruptions: Sequence[Any]) -> list[PendingApproval]:
        """Approve in the SDK each paused call of this server that the gateway has decided.

        Each call is sent to the gateway with its approval, if it has one: the first time, the gateway
        opens the approval (or decides at once). A call the gateway decided is approved in ``state``,
        and the run gets the gateway's answer, the tool's result or the refusal, without sending it
        again. The calls still waiting for a person are returned; resolve again once they decide.
        """
        waiting: list[PendingApproval] = []
        for item in interruptions:
            if not isinstance(item, ToolApprovalItem):
                continue
            call = self._call_of(item)
            if call is None:
                continue
            tool, args = call
            key = _key(tool, args)
            known = self._calls.get(key)
            if not isinstance(known, _Decided):
                known = await self._send(tool, args, known.approval_id if known else None, None)
                self._remember(key, known)
            if isinstance(known, _Held):
                waiting.append(PendingApproval(known.approval_id, tool, args, item))
                continue
            state.approve(item)
        return waiting

    def _call_of(self, item: ToolApprovalItem) -> tuple[str, dict[str, Any]] | None:
        raw: object = item.raw_item
        name = item.tool_name or _attribute(raw, "name")
        arguments = _attribute(raw, "arguments")
        if name not in self._listed or arguments is None:
            return None
        try:
            parsed: object = json.loads(arguments) if isinstance(arguments, str) else arguments
        except ValueError:
            return None
        if not isinstance(parsed, Mapping):
            return None
        return name, self.gateway.strip_approval_argument(cast(Mapping[str, Any], parsed))


def _attribute(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value).get(name)
    return getattr(value, name, None)
