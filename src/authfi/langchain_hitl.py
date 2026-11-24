"""LangChain's human-in-the-loop pause, decided by AuthFI: a middleware for ``langchain.agents.create_agent``.

``pip install "authfi[langchain-hitl]"``

::

    from langchain.agents import create_agent
    from authfi.langchain import LangChainIntegration
    from authfi.langchain_hitl import AuthFIApprovalMiddleware

    langchain = LangChainIntegration(af.agent_from_env())          # an asyncio agent
    agent = create_agent(
        model, tools,
        middleware=[AuthFIApprovalMiddleware(langchain, request=["send_email"])],
        checkpointer=checkpointer,
    )

LangChain pauses a run with ``interrupt()`` (``HumanInTheLoopMiddleware``) and a person decides in the application.
Here the person decides in AuthFI, and the application only says where:

- **A call the gateway holds** (a risky tool, DLP, over budget, a changed tool pin) becomes an ``interrupt()`` whose
  value is LangChain's ``tool_approval`` request plus ``authfi_approval_id``. The tool does not run.
- **A tool listed in** ``request`` **asks a person first** (``agent.request_approval``) and becomes the same interrupt.
- With ``approval_poll`` the middleware first waits (``poll_interval`` seconds apart, up to ``approval_timeout``) for
  the person's decision with the agent's own token, and interrupts only if nobody decided in time.

Resuming the interrupt decides nothing: whatever the resume value says, the middleware asks AuthFI (a call is sent
again with its approval and the gateway decides; a request is read from the approval). A call still waiting
interrupts again. Resume with ``{"stop": True}`` to give up: the model reads that the action did not run.

The approval id lives in the interrupt value and the checkpoint, never in the model's messages: it is removed from
anything the model reads, and one the model writes into a call is removed. A resume in the same process finds the
approval in memory; a restarted process gives the saved ids to :meth:`AuthFIApprovalMiddleware.adopt` so it does not
raise a second approval. A project that binds approvals to the run answers a resumed call with
"approval was raised by another run": the model reads that the call must be raised again.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from authfi._holds import Held, HeldCalls, call_key
from authfi._protocol.gateway import read_result
from authfi.errors import ApprovalUnavailable, ConfigurationError
from authfi.gateway import OTHER_RUN_REFUSAL, AsyncGateway, ToolApprovalRequired, token_refusal

try:
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage
    from langgraph.types import Command, interrupt
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("authfi.langchain_hitl needs langchain 1.x: pip install 'authfi[langchain-hitl]'") from exc

if TYPE_CHECKING:
    from langgraph.prebuilt.tool_node import ToolCallRequest

__all__ = ["AuthFIApprovalMiddleware"]

_OTHER_RUN_TEXT = (
    "That approval belongs to an earlier run, so the action did not run. Raise the call again: a person will be "
    "asked afresh."
)

Handler = Callable[["ToolCallRequest"], Awaitable["ToolMessage | Command[Any]"]]


def _text_of(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        parts = cast("Sequence[object]", content)
        return "\n".join(
            str(cast("Mapping[str, Any]", p).get("text", "")) if isinstance(p, Mapping) else str(p) for p in parts
        )
    return str(content)


class AuthFIApprovalMiddleware(AgentMiddleware):
    """LangChain's tool-call pause, decided by AuthFI. See the module notes.

    Args:
        integration: the :class:`~authfi.langchain.LangChainIntegration` (or LangGraph's) of an asyncio agent.
        request: tool names that ask a person (``request_approval``) before they run, besides the calls the
            gateway holds. Needs the agent itself as the credential.
        approval_poll: wait for the person's decision before interrupting (see the module notes).
        approval_timeout: seconds to wait; ``poll_interval``: seconds between polls.
    """

    def __init__(
        self,
        integration: Any,
        *,
        request: Iterable[str] = (),
        approval_poll: bool = False,
        approval_timeout: float = 900.0,
        poll_interval: float = 3.0,
    ) -> None:
        super().__init__()
        gateway = getattr(integration, "gateway", None)
        if not isinstance(gateway, AsyncGateway):
            raise ConfigurationError("AuthFIApprovalMiddleware needs the integration of an asyncio agent")
        self.gateway = gateway
        self.request = frozenset(request)
        if self.request and not callable(getattr(gateway.source, "request_approval", None)):
            raise ConfigurationError("request= needs the agent itself: a token held for a person cannot ask")
        self.approval_poll = approval_poll
        self.approval_timeout = approval_timeout
        self.poll_interval = poll_interval
        self._held = HeldCalls(gateway)
        self._asked: dict[str, str] = {}

    # --- for the application ---------------------------------------------------------------------------

    def held(self) -> list[dict[str, Any]]:
        """The calls waiting for a person, as data to save with the checkpoint (``approval_id``, ``tool``, ``args``)."""
        return self._held.saved()

    def adopt(self, held: Sequence[Mapping[str, Any]]) -> None:
        """Remember calls saved by :meth:`held` in another process, so a resume does not raise a second approval."""
        self._held.adopt(held)

    # --- the tool call ---------------------------------------------------------------------------------

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Handler) -> ToolMessage | Command[Any]:
        """Run the tool; a call the gateway holds, or one listed in ``request``, waits for a person in AuthFI."""
        name = request.tool_call["name"]
        args = self.gateway.strip_approval_argument(request.tool_call["args"])
        if name in self.request:
            refused = await self._ask_first(request, args)
            if refused is not None:
                return refused
        while True:
            known = self._held.get(name, args)
            approval_id = known.approval_id if isinstance(known, Held) else None
            sent = self.gateway.with_approval(args, approval_id) if approval_id else args
            result = await self._run(request.override(tool_call={**request.tool_call, "args": sent}), handler)
            if not isinstance(result, ToolApprovalRequired):
                self._held.take(name, args)
                return self._clean(result, approval_id)
            held = result
            self._held.remember(name, args, Held(held.approval_id))
            self.gateway.project.logger.info("%s waits for approval %s", name, held.approval_id)
            stopped = await self._wait_for_a_person(request, name, args, held.approval_id)
            if stopped is not None:
                self._held.take(name, args)
                return stopped

    async def _run(
        self, request: ToolCallRequest, handler: Handler
    ) -> ToolMessage | Command[Any] | ToolApprovalRequired:
        try:
            result = await handler(request)
        except Exception as error:
            refused = token_refusal(error)
            if refused is not None:
                raise refused from error
            outcome = read_result(str(error), is_error=True)
            if isinstance(outcome, ToolApprovalRequired):
                return outcome
            raise
        if isinstance(result, ToolMessage):
            outcome = read_result(_text_of(result.content), is_error=result.status == "error")
            if isinstance(outcome, ToolApprovalRequired):
                return outcome
        return result

    def _clean(self, result: ToolMessage | Command[Any], approval_id: str | None) -> ToolMessage | Command[Any]:
        """The answer without the approval id, and with the instruction to raise the call again if another run's."""
        if not isinstance(result, ToolMessage):
            return result
        text = _text_of(result.content)
        if OTHER_RUN_REFUSAL in text:
            return result.model_copy(update={"content": _OTHER_RUN_TEXT, "status": "error"})
        if approval_id and approval_id in text:
            return result.model_copy(update={"content": text.replace(approval_id, "an approval")})
        return result

    async def _wait_for_a_person(
        self, request: ToolCallRequest, name: str, args: dict[str, Any], approval_id: str
    ) -> ToolMessage | None:
        """Wait until the approval is decided (then the caller sends the call again), or give up: a ToolMessage."""
        if self.approval_poll and await self._poll(approval_id) != "pending":
            return None
        decision = interrupt(self._value(request, name, args, approval_id, "A person must approve this in AuthFI."))
        if isinstance(decision, Mapping) and cast("Mapping[str, Any]", decision).get("stop") is True:
            return self._not_run(request, "the action was not approved in time")
        return None

    async def _poll(self, approval_id: str) -> str:
        try:
            return await self.gateway.await_approval(
                approval_id, timeout=self.approval_timeout, poll_interval=self.poll_interval
            )
        except ApprovalUnavailable:
            self.gateway.project.logger.warning("approval %s cannot be read", approval_id)
            raise

    # --- a person asked before the tool runs ------------------------------------------------------------

    async def _ask_first(self, request: ToolCallRequest, args: dict[str, Any]) -> ToolMessage | None:
        name = request.tool_call["name"]
        key = call_key(name, args)
        agent = cast("Any", self.gateway.source)
        approval_id = self._asked.get(key)
        if approval_id is None:
            approval = await agent.request_approval(f"Run {name}", context={"tool": name, "args": args})
            approval_id = approval.approval_id
            self._asked[key] = approval_id
        while True:
            status = await self._poll_once_or_wait(approval_id)
            if status == "approved":
                self._asked.pop(key, None)
                return None
            if status != "pending":
                self._asked.pop(key, None)
                return self._not_run(request, f"the approval was {status}")
            decision = interrupt(self._value(request, name, args, approval_id, f"A person must approve {name}."))
            if isinstance(decision, Mapping) and cast("Mapping[str, Any]", decision).get("stop") is True:
                self._asked.pop(key, None)
                return self._not_run(request, "the action was not approved in time")

    async def _poll_once_or_wait(self, approval_id: str) -> str:
        timeout = self.approval_timeout if self.approval_poll else 0.0
        return await self.gateway.await_approval(approval_id, timeout=timeout, poll_interval=self.poll_interval)

    # --- what the application and the model see ---------------------------------------------------------

    @staticmethod
    def _value(
        request: ToolCallRequest, name: str, args: dict[str, Any], approval_id: str, text: str
    ) -> dict[str, Any]:
        """LangChain's ``tool_approval`` interrupt value, plus the AuthFI approval the application should point at."""
        return {
            "type": "tool_approval",
            "tool_call_id": request.tool_call["id"],
            "name": name,
            "args": args,
            "description": text,
            "authfi_approval_id": approval_id,
        }

    @staticmethod
    def _not_run(request: ToolCallRequest, why: str) -> ToolMessage:
        return ToolMessage(
            content=f"Not run: {why}.",
            name=request.tool_call["name"],
            tool_call_id=request.tool_call["id"] or "",
            status="error",
        )
