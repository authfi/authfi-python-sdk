"""LangGraph on AuthFI: governed MCP tools, refusals the model reads, approvals through interrupt.

``pip install 'authfi[langgraph]'``

::

    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command
    from authfi import AgentAuth
    from authfi.langgraph import AuthFIGateway, ask

    gateway = AuthFIGateway(AgentAuth.from_env())
    tools = await gateway.tools()
    graph = gateway.agent_graph(gateway.chat_model("claude-haiku-5-5"), tools, checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "t1"}}
    state = await graph.ainvoke(ask("Summarise pallets/flask's wiki"), config)
    if "__interrupt__" in state:          # a call is waiting for a person's approval
        ...                               # once they have decided:
        state = await graph.ainvoke(Command(resume=True), config)

What it does for you:

- **MCP auth.** The tools reach ``<project>/mcp`` with the agent's current
  token on every request.
- **Tool names.** Each tool is offered to the model under a name every
  provider accepts (``deepwiki__read_wiki_structure``) and called under the
  gateway's own. Tools the gateway marks ``never_allowed`` are left out.
- **Refusals.** A refused call reaches the model as the tool's error, naming
  why, so the run carries on. The older gateway's HTTP 403 is turned into that
  same tool error. A token the gateway will not take at all (HTTP 401) ends the
  run with :class:`authfi.gateway.GatewayRefusedToken`, not an ExceptionGroup.
- **Approvals.** A held call is recorded in the graph's state and the tools
  node finishes. The ``authfi_approval`` node then calls ``interrupt()`` before
  anything else. On resume it sends the held call again with the approval id,
  and the gateway decides. Calling ``interrupt()`` inside the tool call instead
  would send the call again on resume, because LangGraph re-runs a node from
  its start, and open a second approval that nobody uses.
- **Linking.** Each tool call is sent with its LangChain run id as
  ``X-Request-Id``, and each model call with its own, so a trace joins the
  gateway's record of the call.
"""

import contextvars
import logging
from typing import Annotated

try:
    from langchain_core.messages import AIMessage, ToolMessage
    from langchain_core.runnables.config import var_child_runnable_config
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.types import Command, interrupt
except ImportError as e:  # pragma: no cover - exercised only without the extra
    raise ImportError("authfi.langgraph needs LangGraph: pip install 'authfi[langgraph]'") from e

from .gateway import (
    REQUEST_ID_HEADER,
    ToolNames,
    credential,
    held_text,
    mcp_http_client_factory,
    model_base_url,
    never_allowed,
    read_result,
    request_id,
    token_refusal,
    with_approval,
)
from .langchain import AuthFIChatAnthropic

_log = logging.getLogger("authfi")

#: The node that waits for approvals, in graphs built by :meth:`AuthFIGateway.agent_graph`.
APPROVAL_NODE = "authfi_approval"

# Set by the approval node while it sends a held call again: the approval to send it with.
_approval_for_retry = contextvars.ContextVar("authfi_approval_for_retry", default=None)


def _holds(current, update):
    """Reducer for ``authfi_holds``: tool calls append; the approval node replaces.

    Several tool calls in one step may each be held, so a tool's update is
    added to the list. The approval node returns ``{"replace": [...]}`` with
    the calls still waiting, or ``[]`` when none are.
    """
    if isinstance(update, dict) and "replace" in update:
        return list(update["replace"])
    if not update:
        return []
    return list(current or []) + list(update)


class AuthFIState(MessagesState):
    """``MessagesState`` plus the tool calls waiting for a person's approval."""

    authfi_holds: Annotated[list, _holds]


def ask(text):
    """The input that starts a run on :class:`AuthFIState` with one user message."""
    return {"messages": [("user", text)], "authfi_holds": []}


class AuthFIGateway:
    """The project's AuthFI gateway, as LangGraph tools and a model.

    ``agent`` is an :class:`authfi.AgentAuth`, :class:`authfi.AsyncAgentAuth`,
    :class:`authfi.DelegatedToken` or a function returning the token. ``url``
    defaults to ``<project>/mcp``, which serves both the MCP gateway and the
    AI gateway.
    """

    def __init__(self, agent, url=None, server_name="authfi"):
        self.credential = credential(agent)
        self.url = model_base_url(self.credential, url)
        self.server_name = server_name
        self.names = ToolNames()
        self._tools = {}

    # --- the model

    def chat_model(self, model, **kwargs):
        """:class:`authfi.langchain.AuthFIChatAnthropic` on this gateway, with this agent's token."""
        return AuthFIChatAnthropic(agent=self.credential, base_url=self.url, model=model, **kwargs)

    # --- the tools

    def connection(self):
        """The ``langchain-mcp-adapters`` connection for the gateway."""
        return {"transport": "streamable_http", "url": self.url,
                "httpx_client_factory": mcp_http_client_factory(self.credential)}

    def client(self):
        """A ``MultiServerMCPClient`` for the gateway, with this module's interceptor."""
        return MultiServerMCPClient({self.server_name: self.connection()}, tool_interceptors=[self.interceptor])

    async def tools(self, include_never_allowed=False):
        """The gateway's tools as LangChain tools, ready to bind to a model.

        Each is renamed to a provider-legal name; calls still go to the
        gateway under its own name. Tools the gateway marks ``never_allowed``
        are left out unless ``include_never_allowed``.
        """
        out = []
        for tool in await self.client().get_tools():
            if never_allowed(tool) and not include_never_allowed:
                _log.info("leaving out %s: the gateway marks it never_allowed for this agent", tool.name)
                continue
            gateway_name = tool.name
            tool.name = self.names.provider(gateway_name)
            self._tools[gateway_name] = tool
            out.append(tool)
        return out

    async def interceptor(self, request, handler):
        """The ``langchain-mcp-adapters`` tool interceptor that speaks to the gateway.

        It sends the call's run id as ``X-Request-Id``, adds the approval id
        when the approval node sends a held call again, turns a hold into
        graph state, and raises a token the gateway refuses as
        :class:`authfi.gateway.GatewayRefusedToken`.
        """
        rid = _tool_run_id(request)
        if rid:
            request = request.override(headers=dict(request.headers or {}, **{REQUEST_ID_HEADER: rid}))
        approval = _approval_for_retry.get()
        if approval:
            request = request.override(args=with_approval(request.args, approval))
        try:
            result = await handler(request)
        except Exception as e:
            refused = token_refusal(e)
            if refused is not None:
                raise refused from e
            raise
        if approval or isinstance(result, (ToolMessage, Command)):
            return result
        answer = read_result(result)
        tool_call_id = getattr(request.runtime, "tool_call_id", None)
        if not answer.held or tool_call_id is None:
            return result
        hold = {"approval_id": answer.approval_id, "tool": request.name, "args": dict(request.args or {}),
                "tool_call_id": tool_call_id, "message_id": "authfi-hold-" + tool_call_id, "text": answer.text}
        _log.info("%s is held for approval %s", request.name, answer.approval_id)
        placeholder = ToolMessage(content=held_text(answer), tool_call_id=tool_call_id, id=hold["message_id"])
        return Command(update={"messages": [placeholder], "authfi_holds": [hold]})

    # --- approvals

    async def approval_node(self, state):
        """The node that waits for a person: ``interrupt()`` first, then the held calls again.

        Nothing runs before ``interrupt()``, so re-running this node on resume
        has no side effect. The interrupt's value lists each held call
        (``approval_id``, ``tool``, ``args``, ``text``). Resume with any value
        once the person has decided; the gateway, not the resume value,
        decides each call. Resume with ``{"stop": True}`` to give up: each held
        call then reaches the model as not run. A call that is still waiting is
        held again and the node interrupts again.
        """
        holds = state.get("authfi_holds") or []
        if not holds:
            return {}
        decision = interrupt({"authfi_approvals": [
            {"approval_id": h["approval_id"], "tool": h["tool"], "args": h["args"], "text": h["text"]}
            for h in holds]})
        stop = isinstance(decision, dict) and decision.get("stop")
        messages, waiting = [], []
        for hold in holds:
            if stop:
                messages.append(ToolMessage(
                    content="Not run: nobody waited for approval %s." % hold["approval_id"],
                    tool_call_id=hold["tool_call_id"], id=hold["message_id"], status="error"))
                continue
            sent = await self._send_again(hold)
            answer = read_result(_message_text(sent), is_error=getattr(sent, "status", None) == "error")
            if answer.held:
                waiting.append(dict(hold, approval_id=answer.approval_id, text=answer.text))
                continue
            messages.append(ToolMessage(content=sent.content, tool_call_id=hold["tool_call_id"],
                                        id=hold["message_id"], status=getattr(sent, "status", "success")))
        return {"messages": messages, "authfi_holds": {"replace": waiting}}

    async def _send_again(self, hold):
        tool = self._tools.get(hold["tool"])
        if tool is None:
            raise KeyError("no tool %r from this gateway: build the graph from AuthFIGateway.tools()" % hold["tool"])
        token = _approval_for_retry.set(hold["approval_id"])
        try:
            return await tool.ainvoke({"type": "tool_call", "id": hold["tool_call_id"], "name": tool.name,
                                       "args": hold["args"]})
        finally:
            _approval_for_retry.reset(token)

    # --- a ready-made agent

    def agent_graph(self, model, tools, checkpointer=None, prompt=None):
        """A tool-calling agent: model, tools, and the approval node between tools and the model.

        ``model`` is a chat model (:meth:`chat_model`), ``tools`` the list from
        :meth:`tools`. A checkpointer is needed for approvals: ``interrupt()``
        pauses the run in it.
        """
        bound = model.bind_tools(tools) if tools else model

        async def agent(state):
            messages = state["messages"]
            if prompt:
                messages = [("system", prompt)] + list(messages)
            return {"messages": [await bound.ainvoke(messages)]}

        graph = StateGraph(AuthFIState)
        graph.add_node("agent", agent)
        graph.add_edge(START, "agent")
        if tools:
            graph.add_node("tools", ToolNode(tools))
            graph.add_node(APPROVAL_NODE, self.approval_node)
            graph.add_conditional_edges("agent", _route_after_agent, ["tools", END])
            graph.add_conditional_edges("tools", route_after_tools, [APPROVAL_NODE, "agent"])
            graph.add_conditional_edges(APPROVAL_NODE, route_after_tools, [APPROVAL_NODE, "agent"])
        else:
            graph.add_edge("agent", END)
        return graph.compile(checkpointer=checkpointer)


def route_after_tools(state, next_node="agent"):
    """After the tools node (and the approval node): wait for approvals while any call is held."""
    return APPROVAL_NODE if state.get("authfi_holds") else next_node


def _route_after_agent(state):
    last = state["messages"][-1]
    return "tools" if isinstance(last, AIMessage) and last.tool_calls else END


def _tool_run_id(request):
    """The LangChain run id of this tool call, else the model's tool call id."""
    config = var_child_runnable_config.get() or {}
    manager = config.get("callbacks")
    rid = request_id(getattr(manager, "parent_run_id", None))
    if rid:
        return rid
    return request_id(getattr(request.runtime, "tool_call_id", None))


def _message_text(message):
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "\n".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content or ())
