"""LangGraph: the gateway's governed tools, refusals the model reads, and approvals through ``interrupt()``.

``pip install "authfi[langgraph] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from authfi import AsyncAuthFI
    from authfi.langgraph import LangGraphIntegration, ask

    af = AsyncAuthFI.from_env()
    langgraph = LangGraphIntegration(af.agent_from_env())
    tools = await langgraph.tools()
    graph = langgraph.agent_graph(langgraph.chat_model("claude-haiku-5-5"), tools, checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}}
    state = await graph.ainvoke(ask("Summarise pallets/flask's wiki"), config)
    while "__interrupt__" in state:          # a call waits for a person's approval in AuthFI
        ...                                  # once they have decided:
        state = await graph.ainvoke(Command(resume=True), config)

- **Tools.** The gateway's MCP tools, under the gateway's own names (``<server>__<tool>``), with
  the current token on every request. Tools the gateway says are never allowed for this agent are
  left out.
- **Refusals** reach the model as the tool's error, naming why, and the run carries on. A token
  the gateway refuses outright (401) ends the run with :class:`~authfi.GatewayTokenRefused`.
- **Approvals.** A held call is recorded in the graph's state (``authfi_holds``) and the model
  reads only that the action waits for a person. :meth:`LangGraphIntegration.approval_node` calls
  ``interrupt()`` before anything else, so LangGraph re-running it on resume repeats nothing; then
  it sends each held call again with its approval, and the gateway decides. The approval id never
  passes through the model, and one the model writes is removed.
"""

from __future__ import annotations

import contextvars
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Any, TypedDict, cast

try:
    import httpx
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, ToolMessage
    from langchain_core.tools import BaseTool
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.interceptors import MCPToolCallRequest
    from langchain_mcp_adapters.sessions import StreamableHttpConnection
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph import END, START, StateGraph  # pyright: ignore[reportMissingTypeStubs]
    from langgraph.graph.message import add_messages  # pyright: ignore[reportMissingTypeStubs]
    from langgraph.graph.state import CompiledStateGraph  # pyright: ignore[reportMissingTypeStubs]
    from langgraph.prebuilt import ToolNode
    from langgraph.types import Command, interrupt
    from mcp.types import CallToolResult, TextContent
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.langgraph needs LangGraph: "
        "pip install 'authfi[langgraph] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import async_gateway_for
from authfi.gateway import HOLD_MESSAGE, AsyncGateway, ToolAllowed, ToolRefused, token_refusal
from authfi.langchain import AuthFIChatAnthropic, LangChainIntegration
from authfi.tokens import AsyncTokenSource

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from authfi.client import AsyncAuthFI

__all__ = [
    "APPROVAL_NODE",
    "AuthFIState",
    "Hold",
    "LangGraphIntegration",
    "ask",
    "route_after_tools",
]

#: The node that waits for approvals, in graphs built by :meth:`LangGraphIntegration.agent_graph`.
APPROVAL_NODE = "authfi_approval"

#: What a tool interceptor answers: the MCP result, or a LangChain message or LangGraph command.
ToolCallResult = CallToolResult | ToolMessage | Command[Any]

#: The name the gateway's MCP server has in the client; tools keep the gateway's own names.
SERVER_NAME = "authfi"


class Hold(TypedDict):
    """A tool call the gateway holds until a person approves it."""

    approval_id: str
    tool: str
    args: dict[str, Any]
    tool_call_id: str
    message_id: str


@dataclass(frozen=True, slots=True)
class _Resume:
    approval_id: str
    tool_call_id: str


# Set by the approval node while it sends a held call again: the approval to send it with.
_RESUMING: contextvars.ContextVar[_Resume | None] = contextvars.ContextVar("authfi_resuming", default=None)


def _holds(current: list[Hold] | None, update: list[Hold] | Mapping[str, list[Hold]] | None) -> list[Hold]:
    """Reducer for ``authfi_holds``: tool calls add holds; the approval node replaces the list."""
    if isinstance(update, Mapping):
        return list(update.get("replace") or [])
    if not update:
        return []
    return [*(current or []), *update]


class AuthFIState(TypedDict):
    """LangGraph's messages, plus the tool calls waiting for a person's approval."""

    messages: Annotated[list[AnyMessage], add_messages]
    authfi_holds: Annotated[list[Hold], _holds]


def ask(text: str) -> dict[str, Any]:
    """The input that starts a run on :class:`AuthFIState` with one message from the user."""
    return {"messages": [("user", text)], "authfi_holds": []}


def route_after_tools(state: Mapping[str, Any]) -> str:
    """After the tools node and the approval node: wait while any call is held, else back to the model."""
    return APPROVAL_NODE if state.get("authfi_holds") else "agent"


def _route_after_agent(state: Mapping[str, Any]) -> str:
    messages = cast(Sequence[BaseMessage], state["messages"])
    last = messages[-1] if messages else None
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


class LangGraphIntegration:
    """The AuthFI gateway for LangGraph: model, tools, and the approval node.

    Args:
        source: an asyncio credential (``AsyncAuthFI.agent(...)``, or the ``AsyncDelegatedToken``
            it holds for a person). MCP tools run on asyncio.
        project: the project, for a credential of your own that does not name it.

    """

    def __init__(self, source: AsyncTokenSource, *, project: AsyncAuthFI | None = None) -> None:
        if not isinstance(cast(object, source), AsyncTokenSource):
            async_gateway_for(source, project)  # raises, naming the asyncio credential to use
        self.source = source
        self.langchain = LangChainIntegration(source, project=project)
        self.gateway = cast(AsyncGateway, self.langchain.gateway)
        self.agent_id = self.langchain.agent_id
        self._client: MultiServerMCPClient | None = None
        self._tools: dict[str, BaseTool] = {}

    # --- the model ------------------------------------------------------------------------------------

    def chat_model(self, model: str, **kwargs: Any) -> AuthFIChatAnthropic:
        """A Claude model on the AI gateway, as :meth:`authfi.langchain.LangChainIntegration.chat_model`."""
        return self.langchain.chat_model(model, **kwargs)

    configure_tracing = staticmethod(LangChainIntegration.configure_tracing)

    @property
    def run_id(self) -> str:
        """The credential's run, as :attr:`authfi.langchain.LangChainIntegration.run_id`."""
        return self.langchain.run_id

    def export_steps(self, *, capture_content: bool = False, **options: Any) -> TracerProvider:
        """Send this agent's steps to AuthFI; see :meth:`authfi.langchain.LangChainIntegration.export_steps`."""
        return self.langchain.export_steps(capture_content=capture_content, **options)

    # --- tools ----------------------------------------------------------------------------------------

    def connection(self) -> StreamableHttpConnection:
        """The ``langchain-mcp-adapters`` connection for the gateway, with this agent's HTTP client."""
        return {
            "transport": "streamable_http",
            "url": self.gateway.mcp_url,
            "httpx_client_factory": self.gateway.mcp_client_factory(httpx),
        }

    def client(self) -> MultiServerMCPClient:
        """The MCP client for the gateway, made once per integration."""
        if self._client is None:
            self._client = MultiServerMCPClient(
                {SERVER_NAME: self.connection()},
                tool_interceptors=[self.interceptor],
                tool_name_prefix=False,
            )
        return self._client

    async def tools(self, *, include_never_allowed: bool = False) -> list[BaseTool]:
        """The gateway's tools as LangChain tools, under the gateway's names.

        Tools the gateway marks ``never_allowed`` for this agent are left out, unless
        ``include_never_allowed``; the gateway still decides every call.
        """
        out: list[BaseTool] = []
        for tool in await self.client().get_tools(server_name=SERVER_NAME):
            self._tools[tool.name] = tool
            if self.gateway.governance_of(tool) == "never_allowed" and not include_never_allowed:
                self.gateway.project.logger.info("leaving out %s: never allowed for this agent", tool.name)
                continue
            out.append(tool)
        return out

    async def interceptor(
        self,
        request: MCPToolCallRequest,
        handler: Callable[[MCPToolCallRequest], Awaitable[ToolCallResult]],
    ) -> ToolCallResult:
        """The tool interceptor that speaks to the gateway; :meth:`client` installs it.

        It removes an approval the model wrote, adds the approval when the approval node sends a held
        call again, turns a refusal into the tool's error and a hold into graph state, and raises
        :class:`~authfi.GatewayTokenRefused` when the gateway refuses the token.
        """
        resuming = _RESUMING.get()
        args = self.gateway.strip_approval_argument(request.args)
        if resuming is not None:
            args = self.gateway.with_approval(args, resuming.approval_id)
        try:
            result = await handler(request.override(args=args))
        except Exception as error:
            refused = token_refusal(error)
            if refused is not None:
                raise refused from error
            raise
        if not isinstance(result, CallToolResult):
            return result  # another interceptor answered
        outcome = self.gateway.read_result(result)
        if isinstance(outcome, ToolAllowed):
            return result
        tool_call_id = getattr(request.runtime, "tool_call_id", None) or (resuming and resuming.tool_call_id)
        logger = self.gateway.project.logger
        if isinstance(outcome, ToolRefused):
            logger.info("the gateway refused %s: %s", request.name, outcome.code or outcome.error)
            if not isinstance(tool_call_id, str):
                return _error_result(outcome.model_text)
            return ToolMessage(outcome.model_text, tool_call_id=tool_call_id, name=request.name, status="error")
        approval_id = outcome.approval_id
        logger.info("%s waits for approval %s", request.name, approval_id)
        if not isinstance(tool_call_id, str):
            return _error_result(HOLD_MESSAGE)
        if resuming is not None:  # still waiting: the approval node holds it again
            return ToolMessage(HOLD_MESSAGE, tool_call_id=tool_call_id, name=request.name, artifact=_held(approval_id))
        hold: Hold = {
            "approval_id": approval_id,
            "tool": request.name,
            "args": dict(args),
            "tool_call_id": tool_call_id,
            "message_id": "authfi-hold-" + tool_call_id,
        }
        placeholder = ToolMessage(HOLD_MESSAGE, tool_call_id=tool_call_id, name=request.name, id=hold["message_id"])
        return Command(update={"messages": [placeholder], "authfi_holds": [hold]})

    # --- approvals ------------------------------------------------------------------------------------

    async def approval_node(self, state: AuthFIState) -> dict[str, Any]:
        """The node that waits for a person: ``interrupt()`` first, then each held call again.

        The interrupt's value is ``{"authfi_approvals": [{"approval_id", "tool", "args"}]}`` for your
        application to show. Resume once the person has decided in AuthFI; the gateway, not the
        resume value, decides each call. A call still waiting is held again. Resume with
        ``{"stop": True}`` to give up: each held call then reaches the model as not run.
        """
        holds = state.get("authfi_holds") or []
        if not holds:
            return {}
        decision = interrupt(
            {
                "authfi_approvals": [
                    {"approval_id": h["approval_id"], "tool": h["tool"], "args": h["args"]} for h in holds
                ]
            }
        )
        stop = isinstance(decision, Mapping) and cast(Mapping[str, Any], decision).get("stop") is True
        messages: list[ToolMessage] = []
        waiting: list[Hold] = []
        for hold in holds:
            if stop:
                messages.append(
                    ToolMessage(
                        "Not run: the action was not approved in time.",
                        tool_call_id=hold["tool_call_id"],
                        name=hold["tool"],
                        id=hold["message_id"],
                        status="error",
                    )
                )
                continue
            sent = await self._send_again(hold)
            still = _held_id(sent)
            if still is not None:
                waiting.append({**hold, "approval_id": still})
                continue
            messages.append(
                ToolMessage(
                    sent.content,
                    tool_call_id=hold["tool_call_id"],
                    name=hold["tool"],
                    id=hold["message_id"],
                    status=sent.status,
                )
            )
        return {"messages": messages, "authfi_holds": {"replace": waiting}}

    async def _send_again(self, hold: Hold) -> ToolMessage:
        tool = self._tools.get(hold["tool"])
        if tool is None:
            raise KeyError(
                f"no tool {hold['tool']!r} from this gateway: build the graph from LangGraphIntegration.tools()"
            )
        token = _RESUMING.set(_Resume(hold["approval_id"], hold["tool_call_id"]))
        try:
            sent = await tool.ainvoke(
                {"type": "tool_call", "id": hold["tool_call_id"], "name": tool.name, "args": hold["args"]}
            )
        finally:
            _RESUMING.reset(token)
        if isinstance(sent, ToolMessage):
            return sent
        return ToolMessage(str(sent), tool_call_id=hold["tool_call_id"], name=hold["tool"])

    # --- a ready-made agent ---------------------------------------------------------------------------

    def agent_graph(
        self,
        model: BaseChatModel,
        tools: Sequence[BaseTool],
        *,
        checkpointer: BaseCheckpointSaver[Any] | bool | None = None,
        prompt: str | None = None,
    ) -> CompiledStateGraph[Any, Any, Any, Any]:
        """A tool-calling agent: the model, LangGraph's tool node, and the approval node after it.

        ``model`` is :meth:`chat_model` (or any chat model), ``tools`` the list from :meth:`tools`.
        Approvals need a checkpointer: ``interrupt()`` pauses the run in it.
        """
        bound = model.bind_tools(list(tools)) if tools else model

        async def agent(state: AuthFIState) -> dict[str, Any]:
            messages: list[Any] = list(state["messages"])
            if prompt:
                messages = [("system", prompt), *messages]
            return {"messages": [await bound.ainvoke(messages)]}

        graph = StateGraph(AuthFIState)
        graph.add_node("agent", agent)  # pyright: ignore[reportUnknownMemberType]
        graph.add_edge(START, "agent")
        if tools:
            graph.add_node("tools", ToolNode(list(tools)))  # pyright: ignore[reportUnknownMemberType]
            graph.add_node(APPROVAL_NODE, self.approval_node)  # pyright: ignore[reportUnknownMemberType]
            graph.add_conditional_edges("agent", _route_after_agent, ["tools", END])
            graph.add_conditional_edges("tools", route_after_tools, [APPROVAL_NODE, "agent"])
            graph.add_conditional_edges(APPROVAL_NODE, route_after_tools, [APPROVAL_NODE, "agent"])
        else:
            graph.add_edge("agent", END)
        return graph.compile(checkpointer=checkpointer)  # pyright: ignore[reportUnknownMemberType]


_HELD_KEY = "authfi_held_approval_id"


def _held(approval_id: str) -> dict[str, str]:
    return {_HELD_KEY: approval_id}


def _held_id(message: ToolMessage) -> str | None:
    artifact = message.artifact
    if isinstance(artifact, Mapping):
        found = cast(Mapping[str, Any], artifact).get(_HELD_KEY)
        return found if isinstance(found, str) else None
    return None


def _error_result(text: str) -> CallToolResult:
    # Outside a LangGraph tool node there is no tool call to answer: the tool's error says only this.
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=True)
