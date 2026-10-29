"""CrewAI: Claude on the AI gateway, the gateway's governed tools, approvals a crew waits for, steps over OTLP.

``pip install "authfi[crewai] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from crewai import Agent, Crew, Task
    from authfi import AsyncAuthFI
    from authfi.crewai import CrewAIIntegration

    af = AsyncAuthFI.from_env()
    crewai = CrewAIIntegration(af.agent_from_env())
    crewai.export_steps()                       # optional (extra "otel"): the run's steps, never its words

    researcher = Agent(
        role="Researcher",
        goal="Describe a repository's wiki",
        backstory="You read documentation.",
        llm=crewai.llm("claude-haiku-5-5"),
        tools=await crewai.tools(),
    )
    task = Task(description="Describe pallets/flask", expected_output="A summary", agent=researcher)
    crew = Crew(agents=[researcher], tasks=[task])
    async with crewai.guard():              # a token the gateway refuses ends the run
        result = await crew.akickoff()

CrewAI runs on asyncio through ``akickoff()``; the tools and the model here are asyncio-only, as the other
AuthFI adapters are, and refuse the blocking ``kickoff()`` with a message that says so.

- **Model.** :meth:`CrewAIIntegration.llm` is CrewAI's own ``LLM`` (its Anthropic provider) over the AI gateway.
  CrewAI's transport interceptor puts the agent's current token on every request, so no Anthropic key is
  needed and a renewed token is never stale.
- **Tools.** The gateway's MCP tools as CrewAI ``BaseTool`` objects, under the gateway's own names
  (``<server>__<tool>``). Tools the gateway says are never allowed for this agent are left out. A refusal
  reaches the model as the tool's result, naming why, and the crew carries on. A token the gateway refuses
  outright (401) ends the run with :class:`~authfi.GatewayTokenRefused` when the kickoff runs in
  :meth:`CrewAIIntegration.guard`.
- **Approvals.** CrewAI cannot pause a run and resume it, so a tool the gateway holds for a person waits for
  that person: the tool tells ``on_approval`` once that a call waits, asks the gateway again every
  ``poll_interval`` seconds, and returns the tool's result when the person approves, or the refusal when
  they deny it. After ``approval_timeout`` seconds the model reads that the approval did not come, and the
  approval stays open. The approval id never reaches the model, and one the model writes is removed.
- **Steps.** :meth:`CrewAIIntegration.export_steps` turns CrewAI's event bus into OpenTelemetry GenAI spans
  (the crew, each agent, each model call and each tool call) tagged with the agent and the run. Words are
  never read. CrewAI's own anonymous telemetry is separate: set ``CREWAI_DISABLE_TELEMETRY=true`` to turn it off.
"""

# CrewAI cannot share an environment with the type checkers (see [tool.uv] conflicts), so mypy and pyright
# skip this file; the crewai CI job checks it with CrewAI installed.
from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

from typing_extensions import override

try:
    import httpx
    from crewai import LLM
    from crewai.events import (
        AgentExecutionCompletedEvent,
        AgentExecutionErrorEvent,
        AgentExecutionStartedEvent,
        CrewKickoffCompletedEvent,
        CrewKickoffFailedEvent,
        CrewKickoffStartedEvent,
        LLMCallCompletedEvent,
        LLMCallFailedEvent,
        LLMCallStartedEvent,
        ToolUsageErrorEvent,
        ToolUsageFinishedEvent,
        ToolUsageStartedEvent,
        crewai_event_bus,
    )
    from crewai.events.base_events import BaseEvent
    from crewai.llms.hooks.base import BaseInterceptor
    from crewai.tools import BaseTool
    from crewai.utilities.pydantic_schema_utils import create_model_from_schema
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client
    from mcp.types import Tool as MCPTool
    from pydantic import PrivateAttr
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.crewai needs CrewAI with MCP: "
        "pip install 'authfi[crewai] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import PLACEHOLDER_KEY, agent_id_of, async_gateway_for
from authfi._holds import Decided, Held, HeldCalls, gateway_refusal, guard, text_of, wait_for_decision
from authfi.errors import ConfigurationError, GatewayTokenRefused
from authfi.gateway import AsyncGateway
from authfi.gateway import PendingApproval as BasePendingApproval
from authfi.tokens import AsyncTokenSource

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.trace import Span as OtelSpan

    from authfi.client import AsyncAuthFI

__all__ = ["AuthFIInterceptor", "AuthFIStepsListener", "AuthFITool", "CrewAIIntegration", "PendingApproval"]

_T = TypeVar("_T")

#: How long a held call waits for a person, and how often the gateway is asked again.
APPROVAL_TIMEOUT = 900.0
POLL_INTERVAL = 5.0

#: Told once when a tool call starts waiting for a person; may be a plain function or a coroutine function.
OnApproval = Callable[["PendingApproval"], Awaitable[None] | None]


@dataclass(frozen=True, slots=True)
class PendingApproval(BasePendingApproval):
    """A tool call the gateway holds until a person decides ``approval_id``."""


class CrewAIIntegration:
    """The AuthFI gateways for CrewAI.

    Args:
        source: an asyncio credential (``AsyncAuthFI.agent(...)``, or the ``AsyncDelegatedToken`` it holds for a
            person). CrewAI's ``akickoff()`` runs on asyncio.
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

    def llm(self, model: str, **kwargs: Any) -> LLM:
        """Claude on the AI gateway as CrewAI's ``LLM``, called with the agent's token. ``kwargs`` are the LLM's."""
        for refused in ("base_url", "api_key", "interceptor"):
            if refused in kwargs:
                raise ConfigurationError(f"{refused} is set by the integration")
        return LLM(
            model=f"anthropic/{model}",
            base_url=self.gateway.model_base_url("anthropic"),
            api_key=PLACEHOLDER_KEY,
            interceptor=AuthFIInterceptor(self.gateway, _loop_of(self.source)),
            **kwargs,
        )

    async def tools(
        self,
        *,
        include_never_allowed: bool = False,
        on_approval: OnApproval | None = None,
        approval_timeout: float = APPROVAL_TIMEOUT,
        poll_interval: float = POLL_INTERVAL,
    ) -> list[AuthFITool]:
        """The MCP gateway's tools as CrewAI tools, for ``Agent(tools=...)``.

        A tool the gateway marks ``never_allowed`` is left out, unless ``include_never_allowed``. A tool
        call the gateway holds for approval waits for the person (see the module notes); ``on_approval`` is told
        once, with the :class:`PendingApproval`, so the application can say where to decide it.
        """
        listed = await _list_tools(self.gateway)
        tools: list[AuthFITool] = []
        for tool in listed:
            if self.gateway.governance_of(tool) == "never_allowed" and not include_never_allowed:
                self.gateway.project.logger.info("leaving out %s: never allowed for this agent", tool.name)
                continue
            tools.append(
                AuthFITool.from_mcp(
                    self.gateway,
                    tool,
                    on_approval=on_approval,
                    approval_timeout=approval_timeout,
                    poll_interval=poll_interval,
                )
            )
        return tools

    def guard(self) -> AbstractAsyncContextManager[None]:
        """End the run with :class:`~authfi.GatewayTokenRefused` if the gateway refuses the token while it runs.

        CrewAI hands a failed model call or tool call to the model as text, or buries it in its own error, so
        a revoked agent would look like a crew that fails for no reason. Wrap the kickoff::

            async with crewai.guard():
                result = await crew.akickoff()
        """
        return guard(self.gateway)

    def export_steps(self, *, capture_content: bool = False, **options: Any) -> TracerProvider:
        """Send this agent's steps to AuthFI (extra ``otel``): CrewAI's events as OpenTelemetry GenAI spans.

        The spans name the crew, each agent, each tool and each model call with timings, status and usage, and
        never the words (``capture_content`` is accepted for the exporter's own policy and adds no content here:
        the bridge reads none). Call it once, before the first run. ``options`` are those of
        :func:`authfi.otel.register`.
        """
        from authfi import otel

        provider = otel.register(self.source, capture_content=capture_content, **options)
        AuthFIStepsListener.attach(provider)
        return provider


def _on_loop(loop: asyncio.AbstractEventLoop | None, coro: Coroutine[Any, Any, _T]) -> _T:
    """Run ``coro`` on the credential's event loop from CrewAI's worker thread, and wait for its answer.

    Under ``akickoff()`` CrewAI runs the model call and each tool in a worker thread, while the credential
    belongs to the event loop ``akickoff()`` runs on. Called on that loop's own thread, or with no loop
    running, this refuses with the way out instead of deadlocking.
    """
    try:
        here = asyncio.get_running_loop()
    except RuntimeError:
        here = None
    if loop is None or loop.is_closed() or not loop.is_running() or here is loop:
        coro.close()
        raise ConfigurationError(
            "the AuthFI credential runs on asyncio: run the crew with `await crew.akickoff()` from the event loop "
            "that made the credential, not with kickoff()"
        )
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


def _loop_of(source: object) -> asyncio.AbstractEventLoop | None:
    """The event loop the credential belongs to: the running one, else the one its project is bound to."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        project = getattr(source, "project", None)
        bound = getattr(getattr(project, "_transport", None), "bound_loop", None)
        return cast("asyncio.AbstractEventLoop | None", bound)


class AuthFIInterceptor(BaseInterceptor[httpx.Request, httpx.Response]):
    """CrewAI's transport interceptor that carries the agent's current token to the AI gateway.

    Under ``akickoff()`` CrewAI makes the model call in a worker thread, so the token is fetched on the
    credential's event loop. A 401 from the gateway is recorded on the gateway for :meth:`CrewAIIntegration.guard`.
    """

    def __init__(self, gateway: AsyncGateway, loop: asyncio.AbstractEventLoop | None = None) -> None:
        self.gateway = gateway
        self._loop = loop

    def _stamp(self, message: httpx.Request, headers: Mapping[str, str]) -> httpx.Request:
        message.headers.update(headers)
        self.gateway._stamp(message)  # pyright: ignore[reportPrivateUsage]
        return message

    def _saw(self, message: httpx.Response) -> httpx.Response:
        """Record the gateway's answer to the token: CrewAI's model errors would otherwise hide a 401."""
        if message.status_code == 401:
            self.gateway.token_refused = GatewayTokenRefused(
                self.gateway.model_base_url("anthropic"),  # the transport's response has no request yet
                message.headers.get("www-authenticate"),
                request_id=message.headers.get("x-request-id"),
            )
        return message

    @override
    def on_outbound(self, message: httpx.Request) -> httpx.Request:
        loop = self._loop or _loop_of(self.gateway.source)
        headers = _on_loop(loop, self.gateway.source.headers(message.method, str(message.url)))
        return self._stamp(message, headers)

    @override
    def on_inbound(self, message: httpx.Response) -> httpx.Response:
        return self._saw(message)

    @override
    async def aon_outbound(self, message: httpx.Request) -> httpx.Request:
        return self._stamp(message, await self.gateway.source.headers(message.method, str(message.url)))

    @override
    async def aon_inbound(self, message: httpx.Response) -> httpx.Response:
        return self._saw(message)


def _client_factory(gateway: AsyncGateway) -> Callable[..., Any]:
    return gateway.mcp_client_factory(httpx)


async def _list_tools(gateway: AsyncGateway) -> list[MCPTool]:
    try:
        async with (
            streamablehttp_client(gateway.mcp_url, httpx_client_factory=_client_factory(gateway)) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            return list((await session.list_tools()).tools)
    except Exception as error:
        refused = gateway_refusal(error)
        if refused is not None:
            raise refused from error
        raise


class AuthFITool(BaseTool):
    """One of the gateway's tools as a CrewAI tool. Make them with :meth:`CrewAIIntegration.tools`."""

    # Not pydantic fields: the gateway, the held-call memory and the approval policy ride on the instance.
    _gateway: AsyncGateway = PrivateAttr()
    _held: HeldCalls = PrivateAttr()
    _on_approval: OnApproval | None = PrivateAttr(default=None)
    _approval_timeout: float = PrivateAttr(default=APPROVAL_TIMEOUT)
    _poll_interval: float = PrivateAttr(default=POLL_INTERVAL)
    _loop: asyncio.AbstractEventLoop | None = PrivateAttr(default=None)

    @classmethod
    def from_mcp(
        cls,
        gateway: AsyncGateway,
        tool: MCPTool,
        *,
        on_approval: OnApproval | None,
        approval_timeout: float,
        poll_interval: float,
    ) -> AuthFITool:
        """The CrewAI tool for one ``tools/list`` entry."""
        schema = dict(tool.inputSchema or {"type": "object", "properties": {}})
        made = cls(
            name=tool.name,
            description=tool.description or f"The {tool.name} tool",
            args_schema=create_model_from_schema(schema, model_name=f"{tool.name}_args"),
        )
        made._gateway = gateway
        made._held = HeldCalls(gateway)
        made._on_approval = on_approval
        made._approval_timeout = approval_timeout
        made._poll_interval = poll_interval
        made._loop = asyncio.get_running_loop()
        return made

    @override
    def _run(self, *args: Any, **kwargs: Any) -> Any:
        return _on_loop(self._loop, self._arun(**kwargs))

    @override
    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        args_in = self._gateway.strip_approval_argument(kwargs)
        pending: PendingApproval | None = None

        async def told(approval_id: str) -> None:
            nonlocal pending
            pending = PendingApproval(approval_id, self.name, args_in)
            if self._on_approval is not None:
                answer = self._on_approval(pending)
                if answer is not None:
                    await answer

        decided = await wait_for_decision(
            lambda approval_id: self._send(args_in, approval_id),
            on_held=told,
            timeout=self._approval_timeout,
            poll_interval=self._poll_interval,
        )
        if decided is None:
            return (
                f"The approval for {self.name} did not come in time, so the action did not run. "
                "Tell the person it still waits for their decision."
            )
        return decided.refusal if decided.refusal is not None else text_of(decided.result)

    async def _send(self, args: dict[str, Any], approval_id: str | None) -> Held | Decided:
        async def call(sent: dict[str, Any]) -> Any:
            return await _call_tool(self._gateway, self.name, sent)

        return await self._held.send(call, self.name, args, approval_id)


async def _call_tool(gateway: AsyncGateway, name: str, args: dict[str, Any]) -> Any:
    async with (
        streamablehttp_client(gateway.mcp_url, httpx_client_factory=_client_factory(gateway)) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        return await session.call_tool(name, args)


# --- steps ------------------------------------------------------------------------------------------


class AuthFIStepsListener:
    """CrewAI's events as OpenTelemetry GenAI spans on a provider: the crew's steps, never its words.

    No prompt, answer, tool argument or tool output is read, so none can reach a span. A span is a child of the
    span of the event that CrewAI says is its parent, so a gateway call made inside it joins the steps' trace.
    """

    _attached: ClassVar[dict[int, AuthFIStepsListener]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, provider: TracerProvider) -> None:
        self._tracer = provider.get_tracer("authfi.crewai")
        self._open: dict[str, OtelSpan] = {}
        self._guard = threading.Lock()
        self._handlers: list[tuple[type[BaseEvent], Callable[..., None]]] = []

    @classmethod
    def attach(cls, provider: TracerProvider) -> AuthFIStepsListener:
        """Listen on CrewAI's event bus for ``provider``; a second call for the same provider adds nothing."""
        with cls._lock:
            found = cls._attached.get(id(provider))
            if found is None:
                found = cls(provider)
                found._register()
                cls._attached[id(provider)] = found
            return found

    def detach(self) -> None:
        """Stop listening, and end the spans still open."""
        for event_type, handler in self._handlers:
            crewai_event_bus.off(event_type, handler)
        self._handlers = []
        with self._guard:
            leftover, self._open = list(self._open.values()), {}
        for span in leftover:
            span.end()
        with self._lock:
            for key, value in list(self._attached.items()):
                if value is self:
                    del self._attached[key]

    def _register(self) -> None:
        begin: list[tuple[type[BaseEvent], Callable[[Any], tuple[str, dict[str, str | int]]]]] = [
            (CrewKickoffStartedEvent, _crew_step),
            (AgentExecutionStartedEvent, _agent_step),
            (LLMCallStartedEvent, _chat_step),
            (ToolUsageStartedEvent, _tool_step),
        ]
        end: list[tuple[type[BaseEvent], bool]] = [
            (CrewKickoffCompletedEvent, False),
            (CrewKickoffFailedEvent, True),
            (AgentExecutionCompletedEvent, False),
            (AgentExecutionErrorEvent, True),
            (LLMCallCompletedEvent, False),
            (LLMCallFailedEvent, True),
            (ToolUsageFinishedEvent, False),
            (ToolUsageErrorEvent, True),
        ]
        for event_type, describe in begin:
            self._on(event_type, self._starter(describe))
        for event_type, failed in end:
            self._on(event_type, self._ender(failed))

    def _on(self, event_type: type[BaseEvent], handler: Callable[[Any, Any], None]) -> None:
        crewai_event_bus.on(event_type)(handler)
        self._handlers.append((event_type, handler))

    def _starter(self, describe: Callable[[Any], tuple[str, dict[str, str | int]]]) -> Callable[[Any, Any], None]:
        def start(_source: object, event: Any) -> None:
            name, attrs = describe(event)
            with self._guard:
                parent = self._open.get(event.parent_event_id) if event.parent_event_id else None
            from opentelemetry.trace import set_span_in_context

            context = set_span_in_context(parent) if parent is not None else None
            span = self._tracer.start_span(
                name, context=context, attributes=dict(attrs), start_time=_nanos(event.timestamp)
            )
            with self._guard:
                self._open[event.event_id] = span

        return start

    def _ender(self, failed: bool) -> Callable[[Any, Any], None]:
        def end(_source: object, event: Any) -> None:
            with self._guard:
                span = self._open.pop(event.started_event_id, None) if event.started_event_id else None
            if span is None:
                return
            usage = getattr(event, "usage", None)
            if isinstance(usage, Mapping):
                for source, target in _USAGE:
                    value = cast("Mapping[str, Any]", usage).get(source)
                    if isinstance(value, int):
                        span.set_attribute(target, value)
            if failed:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR))
            span.end(end_time=_nanos(event.timestamp))

        return end


_USAGE = (
    ("input_tokens", "gen_ai.usage.input_tokens"),
    ("prompt_tokens", "gen_ai.usage.input_tokens"),
    ("output_tokens", "gen_ai.usage.output_tokens"),
    ("completion_tokens", "gen_ai.usage.output_tokens"),
)


def _nanos(stamp: datetime | None) -> int | None:
    return int(stamp.timestamp() * 1_000_000_000) if stamp is not None else None


def _op(name: str) -> dict[str, str | int]:
    return {"gen_ai.operation.name": name}


def _crew_step(event: Any) -> tuple[str, dict[str, str | int]]:
    return f"workflow {getattr(event, 'crew_name', None) or 'crew'}", _op("invoke_workflow")


def _agent_step(event: Any) -> tuple[str, dict[str, str | int]]:
    role = event.agent_role or "agent"
    return f"invoke_agent {role}", {**_op("invoke_agent"), "gen_ai.agent.name": role}


def _chat_step(event: Any) -> tuple[str, dict[str, str | int]]:
    model = getattr(event, "model", None)
    if not model:
        return "chat", _op("chat")
    return f"chat {model}", {**_op("chat"), "gen_ai.request.model": model}


def _tool_step(event: Any) -> tuple[str, dict[str, str | int]]:
    return f"execute_tool {event.tool_name}", {**_op("execute_tool"), "gen_ai.tool.name": event.tool_name}
