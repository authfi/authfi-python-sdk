"""The calls the gateway holds for a person, and the ones it has decided, until a framework comes back for them.

Pydantic AI, CrewAI and LlamaIndex pause a run for approval in their own way, and all three need the same
memory behind it: a call the gateway held (its approval id), and a call the gateway already answered (so it
is not sent twice). This is that memory; the framework's pause and resume stay in the framework's adapter.
"""

from __future__ import annotations

import asyncio
import json
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, cast

from authfi.gateway import AsyncGateway, ToolApprovalRequired, ToolRefused, token_refusal

__all__ = [
    "MEMORY",
    "Decided",
    "Held",
    "HeldCalls",
    "call_key",
    "guard",
    "text_of",
    "wait_for_decision",
]

#: How many held or decided calls are remembered until the run comes back for them.
MEMORY = 10_000


@dataclass(frozen=True, slots=True)
class Held:
    """The gateway holds this call until a person approves ``approval_id``."""

    approval_id: str


@dataclass(frozen=True, slots=True)
class Decided:
    """The gateway answered this call: ``result`` is the tool's answer, or ``refusal`` the words of a refusal."""

    result: Any = None
    refusal: str | None = None


def text_of(result: object) -> str:
    """The text blocks of an MCP tool result, joined: what a framework shows its model of a tool's answer."""
    blocks = cast("list[Any]", getattr(result, "content", None) or [])
    return "\n".join(text for block in blocks if isinstance(text := getattr(block, "text", None), str))


def call_key(tool: str, args: Mapping[str, Any]) -> str:
    """One name for one call: the tool and its arguments."""
    return tool + "\x00" + json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


class HeldCalls:
    """What an adapter remembers about calls the gateway held or decided.

    ``ttl`` (seconds) makes a remembered call forgettable: one nobody came back for is dropped, so a later
    identical call is sent afresh. Without it a call is kept until it is taken or pushed out by :data:`MEMORY`.
    """

    def __init__(self, gateway: AsyncGateway, *, ttl: float | None = None) -> None:
        self.gateway = gateway
        self._ttl = ttl
        self._calls: OrderedDict[str, tuple[Held | Decided, float]] = OrderedDict()

    def _now(self) -> float:
        return self.gateway.project.clock.monotonic() if self._ttl is not None else 0.0

    def _fresh(self, key: str, found: tuple[Held | Decided, float] | None) -> Held | Decided | None:
        if found is None:
            return None
        entry, at = found
        if self._ttl is not None and self._now() - at > self._ttl:
            self._calls.pop(key, None)
            return None
        return entry

    def get(self, tool: str, args: Mapping[str, Any]) -> Held | Decided | None:
        """The remembered call, left in place."""
        key = call_key(tool, args)
        return self._fresh(key, self._calls.get(key))

    def take(self, tool: str, args: Mapping[str, Any]) -> Held | Decided | None:
        """The remembered call, forgotten: a decided call is answered once."""
        key = call_key(tool, args)
        return self._fresh(key, self._calls.pop(key, None))

    def remember(self, tool: str, args: Mapping[str, Any], entry: Held | Decided) -> None:
        """Keep ``entry`` for the call, forgetting the oldest beyond :data:`MEMORY`."""
        key = call_key(tool, args)
        self._calls[key] = (entry, self._now())
        self._calls.move_to_end(key)
        while len(self._calls) > MEMORY:
            self._calls.popitem(last=False)

    def clear(self) -> None:
        """Forget every remembered call."""
        self._calls.clear()

    def holds(self, tool: str) -> bool:
        """Whether any call of ``tool`` is held, so its next call must wait for a person even if unannounced."""
        prefix = tool + "\x00"
        return any(key.startswith(prefix) and isinstance(entry, Held) for key, (entry, _) in self._calls.items())

    async def send(
        self,
        call: Callable[[dict[str, Any]], Awaitable[Any]],
        tool: str,
        args: Mapping[str, Any],
        approval_id: str | None = None,
    ) -> Held | Decided:
        """Send the call through ``call`` (the framework's own MCP call), with the approval if it has one.

        Returns :class:`Held` when the gateway holds it, :class:`Decided` with the gateway's refusal or the
        tool's answer otherwise. A credential the gateway refuses outright raises ``GatewayTokenRefused``.
        """
        sent = self.gateway.with_approval(args, approval_id) if approval_id else dict(args)
        try:
            result = await call(sent)
        except Exception as error:
            refused = token_refusal(error)
            if refused is not None:
                raise refused from error
            raise
        outcome = self.gateway.read_result(result)
        if isinstance(outcome, ToolApprovalRequired):
            self.gateway.project.logger.info("%s waits for approval %s", tool, outcome.approval_id)
            return Held(outcome.approval_id)
        if isinstance(outcome, ToolRefused):
            self.gateway.project.logger.info("the gateway refused %s: %s", tool, outcome.code or outcome.error)
            return Decided(refusal=outcome.model_text)
        return Decided(result=result)


class guard:
    """End a run with :class:`~authfi.GatewayTokenRefused` if the gateway refuses the token while it runs.

    A framework that hands a failed tool call to the model as text, or wraps a failed model call in its own
    error, would make a revoked agent look like a run that fails for no reason. The gateway's clients record the
    refusal on ``gateway.token_refused``; this raises it when the run ends, or replaces the framework's error
    with it.
    """

    def __init__(self, gateway: AsyncGateway) -> None:
        self._gateway = gateway

    async def __aenter__(self) -> None:
        self._gateway.token_refused = None

    async def __aexit__(self, exc_type: object, error: BaseException | None, traceback: object) -> None:
        refused = self._gateway.token_refused
        if error is None:
            if refused is not None:
                raise refused
            return
        if isinstance(error, Exception):
            refused = refused or token_refusal(error)
            if refused is not None:
                raise refused from error


async def wait_for_decision(
    send: Callable[[str | None], Awaitable[Held | Decided]],
    *,
    on_held: Callable[[str], Awaitable[None]],
    timeout: float,
    poll_interval: float,
) -> Decided | None:
    """Send a call until the gateway decides it: the answer for a framework that cannot pause a run.

    ``send`` sends the call, with the approval id once there is one. ``on_held`` is told once, with the
    approval id, when the gateway first holds the call. Returns ``None`` when ``timeout`` seconds pass first;
    the approval stays open for the person.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    approval_id: str | None = None
    while True:
        entry = await send(approval_id)
        if isinstance(entry, Decided):
            return entry
        if approval_id is None:
            await on_held(entry.approval_id)
        approval_id = entry.approval_id
        if loop.time() + poll_interval > deadline:
            return None
        await asyncio.sleep(poll_interval)
