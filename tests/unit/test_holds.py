"""authfi._holds: what the Pydantic AI, CrewAI and LlamaIndex adapters share, without any of the frameworks."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from typing import Any

import pytest

from authfi import GatewayTokenRefused
from authfi._holds import MEMORY, Decided, Held, HeldCalls, call_key, guard, text_of, wait_for_decision
from authfi.gateway import AsyncGateway

HOLD = (
    "Waiting for approval appr-1 (approval_required). A person must approve this action in the AuthFI console. "
    'Retry this call with an argument "_authfi_approval_id": "appr-1" once it is approved.'
)


class Result:
    def __init__(self, text: str, *, is_error: bool = False, meta: dict[str, Any] | None = None) -> None:
        self.content = [SimpleNamespace(text=text)]
        self.isError = is_error
        self.meta = meta


def _gateway() -> Any:
    stub = SimpleNamespace(
        project=SimpleNamespace(logger=logging.getLogger("test")),
        with_approval=AsyncGateway.with_approval,
        read_result=AsyncGateway.read_result,
        token_refused=None,
    )
    return stub


# --- the memory -------------------------------------------------------------------------------------


def test_a_call_is_named_by_its_tool_and_arguments_whatever_their_order() -> None:
    assert call_key("t", {"a": 1, "b": 2}) == call_key("t", {"b": 2, "a": 1})
    assert call_key("t", {"a": 1}) != call_key("u", {"a": 1})


def test_a_decided_call_is_answered_once() -> None:
    held = HeldCalls(_gateway())
    held.remember("t", {"a": 1}, Decided(result="x"))
    assert held.get("t", {"a": 1}) == Decided(result="x")
    assert held.take("t", {"a": 1}) == Decided(result="x")
    assert held.take("t", {"a": 1}) is None


def test_a_tool_with_a_held_call_is_known_to_wait() -> None:
    held = HeldCalls(_gateway())
    held.remember("t", {"a": 1}, Held("appr-1"))
    held.remember("u", {"a": 1}, Decided(result="x"))
    assert held.holds("t")
    assert not held.holds("u")
    assert not held.holds("tt")


def test_the_oldest_calls_are_forgotten_beyond_the_memory() -> None:
    held = HeldCalls(_gateway())
    for n in range(MEMORY + 3):
        held.remember("t", {"n": n}, Decided(result=n))
    assert held.get("t", {"n": 0}) is None
    assert held.get("t", {"n": MEMORY + 2}) == Decided(result=MEMORY + 2)


async def test_a_call_is_sent_and_read_the_way_the_gateway_answers() -> None:
    held = HeldCalls(_gateway())
    sent: list[dict[str, Any]] = []

    async def answer(arguments: dict[str, Any]) -> Result:
        sent.append(arguments)
        return Result(HOLD, is_error=True)

    assert await held.send(answer, "t", {"a": 1}) == Held("appr-1")

    async def refuse(arguments: dict[str, Any]) -> Result:
        sent.append(arguments)
        return Result(
            "Refused (insufficient_scope, ETOOL003): no grant.",
            is_error=True,
            meta={"io.authfi/refusal": {"code": "ETOOL003", "reason": "no grant"}},
        )

    refused = await held.send(refuse, "t", {"a": 1}, "appr-1")
    assert isinstance(refused, Decided)
    assert refused.refusal is not None
    assert "ETOOL003" in refused.refusal
    assert sent[1] == {"a": 1, "_authfi_approval_id": "appr-1"}

    async def allow(arguments: dict[str, Any]) -> Result:
        return Result("fine")

    allowed = await held.send(allow, "t", {"a": 1})
    assert isinstance(allowed, Decided)
    assert allowed.result.content[0].text == "fine"


async def test_an_error_that_is_not_a_refusal_passes_through() -> None:
    held = HeldCalls(_gateway())

    async def boom(arguments: dict[str, Any]) -> Result:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        await held.send(boom, "t", {})


async def test_a_token_the_gateway_refuses_ends_the_send() -> None:
    held = HeldCalls(_gateway())

    async def revoked(arguments: dict[str, Any]) -> Result:
        raise RpcError({"authfi_refusal": "gateway_token_refused", "url": "https://x/mcp"})

    with pytest.raises(GatewayTokenRefused):
        await held.send(revoked, "t", {})


# --- the guard ----------------------------------------------------------------------------------------


async def test_a_guarded_run_with_no_refusal_changes_nothing() -> None:
    gateway = _gateway()
    async with guard(gateway):
        pass
    with pytest.raises(ValueError, match="mine"):
        async with guard(gateway):
            raise ValueError("mine")


async def test_a_refusal_the_framework_hid_is_raised_when_the_run_ends() -> None:
    gateway = _gateway()
    with pytest.raises(GatewayTokenRefused):
        async with guard(gateway):
            gateway.token_refused = GatewayTokenRefused("https://x/mcp", None)


async def _fail_with_the_refusal_recorded(gateway: Any) -> None:
    gateway.token_refused = GatewayTokenRefused("https://x/mcp", None)
    raise RuntimeError("tool failed")


async def test_a_framework_error_is_replaced_by_the_refusal_behind_it() -> None:
    gateway = _gateway()
    with pytest.raises(GatewayTokenRefused):
        async with guard(gateway):
            await _fail_with_the_refusal_recorded(gateway)
    with pytest.raises(GatewayTokenRefused):
        async with guard(gateway):
            raise RpcError({"authfi_refusal": "gateway_token_refused"})


async def test_a_refusal_left_by_an_earlier_run_does_not_end_this_one() -> None:
    gateway = _gateway()
    gateway.token_refused = GatewayTokenRefused("https://x/mcp", None)
    async with guard(gateway):
        pass


async def test_a_cancellation_is_never_replaced() -> None:
    gateway = _gateway()
    gateway.token_refused = GatewayTokenRefused("https://x/mcp", None)
    with pytest.raises(asyncio.CancelledError):
        async with guard(gateway):
            raise asyncio.CancelledError


# --- waiting for a person ---------------------------------------------------------------------------


async def test_a_call_decided_at_once_is_not_waited_for() -> None:
    told: list[str] = []

    async def send(approval_id: str | None) -> Held | Decided:
        return Decided(result="done")

    async def on_held(approval_id: str) -> None:
        told.append(approval_id)

    assert await wait_for_decision(send, on_held=on_held, timeout=1, poll_interval=0.01) == Decided(result="done")
    assert told == []


async def test_a_held_call_is_told_once_and_asked_again_with_its_approval() -> None:
    told: list[str] = []
    asked: list[str | None] = []

    async def send(approval_id: str | None) -> Held | Decided:
        asked.append(approval_id)
        return Held("appr-1") if len(asked) < 3 else Decided(result="approved")

    async def on_held(approval_id: str) -> None:
        told.append(approval_id)

    assert await wait_for_decision(send, on_held=on_held, timeout=5, poll_interval=0.01) == Decided(result="approved")
    assert told == ["appr-1"]
    assert asked == [None, "appr-1", "appr-1"]


async def test_a_call_nobody_decides_is_given_up_on_not_approved() -> None:
    async def send(approval_id: str | None) -> Held | Decided:
        return Held("appr-1")

    async def on_held(approval_id: str) -> None:
        return None

    assert await wait_for_decision(send, on_held=on_held, timeout=0.05, poll_interval=0.02) is None


def test_the_text_of_a_result_is_its_text_blocks() -> None:
    assert text_of(Result("a")) == "a"
    assert (
        text_of(SimpleNamespace(content=[SimpleNamespace(text="a"), SimpleNamespace(), SimpleNamespace(text="b")]))
        == "a\nb"
    )
    assert text_of(SimpleNamespace(content=None)) == ""
    assert text_of(object()) == ""
