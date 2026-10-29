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

from authfi.errors import GatewayTokenRefused
from authfi.gateway import AsyncGateway, ToolApprovalRequired, ToolRefused, token_refusal

__all__ = [
    "MEMORY",
    "Decided",
    "Held",
    "HeldCalls",
    "call_key",
    "gateway_refusal",
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


def gateway_refusal(error: BaseException) -> GatewayTokenRefused | None:
    """The gateway's refusal of the token, found in ``error``, whichever MCP SDK raised it.

    :func:`~authfi.gateway.token_refusal` reads the JSON-RPC error the 1.x SDK keeps in ``error.error``; the
    2.x SDK keeps ``data`` on the error itself, which this reads the same way.
    """
    refused = token_refusal(error)
    if refused is not None:
        return refused
    seen: set[int] = set()
    stack: list[BaseException | None] = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        data = getattr(current, "data", None)
        if isinstance(data, Mapping):
            detail = cast("Mapping[str, Any]", data)
            if detail.get("authfi_refusal") == "gateway_token_refused":
                challenge, url = detail.get("www_authenticate"), detail.get("url")
                return GatewayTokenRefused(
                    url if isinstance(url, str) else "", challenge if isinstance(challenge, str) else None
                )
        stack.extend(getattr(current, "exceptions", None) or ())
        stack.extend((current.__cause__, current.__context__))
    return None


def text_of(result: object) -> str:
    """The text blocks of an MCP tool result, joined: what a framework shows its model of a tool's answer."""
    blocks = cast("list[Any]", getattr(result, "content", None) or [])
    return "\n".join(text for block in blocks if isinstance(text := getattr(block, "text", None), str))


def call_key(tool: str, args: Mapping[str, Any]) -> str:
    """One name for one call: the tool and its arguments."""
    return tool + "\x00" + json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)


class HeldCalls:
    """What an adapter remembers about calls the gateway held or decided."""

    def __init__(self, gateway: AsyncGateway) -> None:
        self.gateway = gateway
        self._calls: OrderedDict[str, Held | Decided] = OrderedDict()

    def get(self, tool: str, args: Mapping[str, Any]) -> Held | Decided | None:
        """The remembered call, left in place."""
        return self._calls.get(call_key(tool, args))

    def take(self, tool: str, args: Mapping[str, Any]) -> Held | Decided | None:
        """The remembered call, forgotten: a decided call is answered once."""
        return self._calls.pop(call_key(tool, args), None)

    def remember(self, tool: str, args: Mapping[str, Any], entry: Held | Decided) -> None:
        """Keep ``entry`` for the call, forgetting the oldest beyond :data:`MEMORY`."""
        key = call_key(tool, args)
        self._calls[key] = entry
        self._calls.move_to_end(key)
        while len(self._calls) > MEMORY:
            self._calls.popitem(last=False)

    def holds(self, tool: str) -> bool:
        """Whether any call of ``tool`` is held, so its next call must wait for a person even if unannounced."""
        prefix = tool + "\x00"
        return any(key.startswith(prefix) and isinstance(entry, Held) for key, entry in self._calls.items())

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
            refused = gateway_refusal(error)
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
            refused = refused or gateway_refusal(error)
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
