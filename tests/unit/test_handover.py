"""Handing a whole task to another process: ``Task.hand_over`` and ``AgentAuth.take_over``.

The gateway (:mod:`tests.fake_agent_gateway`) derives its answers from the bytes it receives: the key move's
signature, its holder, its single use, and which key may use a root. A second process has a second run key, so the
tests give the receiving side a key of its own by replacing the one ``_dpop.key_for`` picks (a fork would do it on
a platform that has one; the SDK's own run key is untouched). Each refusal has the allowed case beside it.
"""

from __future__ import annotations

import json
import pickle
import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import jwt
import pytest

from authfi import (
    AgentKey,
    AsyncAuthFI,
    AuthFI,
    ConfigurationError,
    Handover,
    KeyMoveRefused,
    KeyMoveSpent,
    OAuthError,
    Secret,
    TakeOverRefused,
    TakeOverSlowDown,
    Task,
    TaskEnded,
    TaskMoved,
    TaskNotYours,
    TemporarilyUnavailable,
    _dpop,
)
from authfi._protocol.errors import take_over_error
from authfi._protocol.http import HttpResponse
from authfi.agents import AgentAuth
from tests.fake_agent_gateway import AgentGateway, thumbprint
from tests.support import FakeClock, as_process

AGENT = "agt_move1"
OTHER = "agt_move2"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture(autouse=True)
def _fresh_nonces() -> Iterator[None]:
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def gw() -> Iterator[AgentGateway]:
    with AgentGateway() as gateway:
        yield gateway


def person() -> Secret:
    return Secret(jwt.encode({"sub": "usr_1", "exp": int(time.time()) + 3600}, "k" * 32, algorithm="HS256"))


def start(agent: AgentAuth) -> Task:
    return agent.task("support.refund", subject_token=person(), params={"order": "4411"})


def tools_list(gateway: AgentGateway, task: Task) -> int:
    return (
        task.gateway()
        .http_client()
        .post(gateway.mcp_url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        .status_code
    )


def old_root_status(gateway: AgentGateway, root: str, old_key: _dpop.DPoPKey) -> int:
    """What the platform answers the sender's root, presented as the sender's key proves it."""
    headers = _dpop.authorization_headers(root, old_key, "POST", gateway.mcp_url, clock=time.time)
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    return httpx.post(gateway.mcp_url, json=body, headers=dict(headers)).status_code


def test_the_whole_task_moves_to_the_receiving_process_and_the_senders_root_stops(
    gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    sender_key = _dpop.run_key()
    receiver_key = _dpop.DPoPKey()
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        task = start(agent)
        root = task.token()
        assert old_root_status(gw, root, sender_key) == 200  # the control: the sender's root works until the move
        with as_process(receiver_key, monkeypatch):
            to = agent.run_key_thumbprint()
        assert to == receiver_key.thumbprint != sender_key.thumbprint

        handover = task.hand_over(to=to)
        assert (handover.mandate_id, handover.to) == (task.mandate_id, to)
        # Nothing has moved yet: the platform moves the task when the receiver asks.
        assert gw.holders[task.mandate_id] == sender_key.thumbprint
        with as_process(receiver_key, monkeypatch):
            taken = agent.take_over(handover.to_wire())
        assert tools_list(gw, taken) == 200
        assert taken.mandate_id == task.mandate_id and taken.bound_key() is receiver_key

    assert gw.holders[task.mandate_id] == receiver_key.thumbprint
    assert [r["how"] for r in gw.renewals] == ["signed_by_previous"]
    # The sender's root stops at once, with life left in it, and its key has no way back.
    assert old_root_status(gw, root, sender_key) == 401
    # The move was signed with the sender's key and carries its public half only.
    (renewal,) = gw.renewals
    header = jwt.get_unverified_header(renewal["authfi_key_move"])
    assert header["typ"] == "mandate-key-move+jwt" and thumbprint(header["jwk"]) == sender_key.thumbprint
    assert "d" not in header["jwk"]
    claims = jwt.decode(renewal["authfi_key_move"], options={"verify_signature": False})
    assert claims["jkt"] == receiver_key.thumbprint and claims["exp"] - claims["iat"] == 30


def test_the_sender_holds_nothing_after_it_hands_over(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        task = start(af.agent(AGENT, key))
        task.hand_over(to=_dpop.DPoPKey().thumbprint)
        for use in (lambda: task.token(), lambda: task.headers("POST", gw.mcp_url)):
            with pytest.raises(ConfigurationError, match="handed over") as caught:
                use()
            assert caught.value.code == "task_closed"
        with pytest.raises(ConfigurationError) as again:
            task.hand_over(to=_dpop.DPoPKey().thumbprint)  # a move is one use: there is no second
        assert again.value.code == "task_closed"


def test_hand_over_refuses_what_cannot_be_a_receiving_key(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        task = start(af.agent(AGENT, key))
        for bad, code in (
            ("", "handover_target_invalid"),
            ("not a thumbprint", "handover_target_invalid"),
            (_dpop.run_key().thumbprint, "handover_to_self"),
        ):
            with pytest.raises(ConfigurationError) as caught:
                task.hand_over(to=bad)
            assert caught.value.code == code
        assert tools_list(gw, task) == 200  # none of the refusals closed it
        task.hand_over(to=_dpop.DPoPKey().thumbprint)  # the control: a real key is let through


def test_a_handover_made_for_another_key_is_refused_before_it_is_sent(
    gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        task = start(agent)
        handover = task.hand_over(to=_dpop.DPoPKey().thumbprint)
        before = len(gw.exchanges)
        with as_process(_dpop.DPoPKey(), monkeypatch), pytest.raises(ConfigurationError) as caught:
            agent.take_over(handover)
    assert caught.value.code == "handover_for_another_key"
    assert len(gw.exchanges) == before


def move_for(task: Task, signer: _dpop.DPoPKey, to: str, *, iat: int | None = None, life: int = 30) -> Handover:
    """A handover signed by ``signer``, who may not be the key that holds the task."""
    now = int(time.time()) if iat is None else iat
    move = signer.sign_jwt(
        {"mandate_id": task.mandate_id, "jkt": to, "iat": now, "exp": now + life, "jti": uuid.uuid4().hex},
        typ="mandate-key-move+jwt",
        carry_public_key=True,
    )
    return Handover(task.mandate_id, to, Secret(task.token()), Secret(move), float(now + life))


def take(
    gw: AgentGateway, key: AgentKey, handover: Handover, receiver: _dpop.DPoPKey, patch: pytest.MonkeyPatch
) -> Task:
    with AuthFI(gw.issuer) as af, as_process(receiver, patch):
        return af.agent(AGENT, key).take_over(handover)


def test_a_move_not_signed_by_the_holder_is_refused_and_the_holders_is_taken(
    gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver = _dpop.DPoPKey()
    with AuthFI(gw.issuer) as af:
        task = start(af.agent(AGENT, key))
        forged = move_for(task, _dpop.DPoPKey(), receiver.thumbprint)  # signed by a key that is not the holder
        with pytest.raises(KeyMoveRefused, match="not signed by the key that holds the task") as caught:
            take(gw, key, forged, receiver, monkeypatch)
        assert caught.value.code == "invalid_grant" and caught.value.status == 400
        assert gw.holders[task.mandate_id] == _dpop.run_key().thumbprint  # nothing moved
        genuine = move_for(task, _dpop.run_key(), receiver.thumbprint)  # the control
        assert take(gw, key, genuine, receiver, monkeypatch).mandate_id == task.mandate_id


def test_a_move_for_another_task_or_that_has_run_out_is_refused(
    gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver = _dpop.DPoPKey()
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        task = start(agent)
        stale = move_for(task, _dpop.run_key(), receiver.thumbprint, iat=int(time.time()) - 120)
        with pytest.raises(KeyMoveRefused, match="expired"):
            take(gw, key, stale, receiver, monkeypatch)
        elsewhere = move_for(task, _dpop.run_key(), _dpop.DPoPKey().thumbprint)
        retargeted = Handover(
            task.mandate_id, receiver.thumbprint, elsewhere.root, elsewhere.move, elsewhere.expires_at
        )
        with pytest.raises(KeyMoveRefused, match="another task or another key"):
            take(gw, key, retargeted, receiver, monkeypatch)
        taken = take(gw, key, move_for(task, _dpop.run_key(), receiver.thumbprint), receiver, monkeypatch)
        assert taken.mandate_id == task.mandate_id


def test_a_handover_the_sdk_knows_has_run_out_is_refused_without_asking(
    gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock = FakeClock(wall=time.time())
    receiver = _dpop.DPoPKey()
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        handover = start(agent).hand_over(to=receiver.thumbprint)
        clock.advance(31)
        before = len(gw.exchanges)
        with as_process(receiver, monkeypatch), pytest.raises(KeyMoveRefused, match="expired"):
            agent.take_over(handover)
    assert len(gw.exchanges) == before


def test_a_task_that_moved_first_is_taken_by_one_process_only(
    gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    first, second = _dpop.DPoPKey(), _dpop.DPoPKey()
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        task = start(agent)
        a = task.hand_over(to=first.thumbprint)
        with as_process(first, monkeypatch):
            agent.take_over(a)
        # The same handover again, from another process that was given the text: the root is the old key's now.
        again = Handover(a.mandate_id, second.thumbprint, a.root, a.move, a.expires_at)
        with as_process(second, monkeypatch), pytest.raises(TaskMoved) as caught:
            agent.take_over(again)
    assert caught.value.status == 400


def test_an_ended_task_and_another_agents_root_are_refused(
    gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver = _dpop.DPoPKey()
    with AuthFI(gw.issuer) as af:
        task = start(af.agent(AGENT, key))
        handover = task.hand_over(to=receiver.thumbprint)
        with as_process(receiver, monkeypatch), pytest.raises(TaskNotYours):
            af.agent(OTHER, key).take_over(handover)
        gw.end_task(task.mandate_id)
        with as_process(receiver, monkeypatch), pytest.raises(TaskEnded):
            af.agent(AGENT, key).take_over(handover)


def test_every_refusal_the_platform_documents_has_its_type() -> None:
    """The answers of ``mandateissue/renew.go`` and ``agent/mandate.go`` on authfi-regional-agent-identity main."""

    def refused(code: str, why: str, status: int = 400) -> OAuthError:
        body = json.dumps({"error": code, "error_description": why}).encode()
        error = take_over_error(HttpResponse(status, (), body, "https://x/oauth/token", "r1"), "the token endpoint")
        assert isinstance(error, OAuthError) and error.code == code and error.description == why
        return error

    table: dict[str, type[TakeOverRefused]] = {
        "the key move has already been used": KeyMoveSpent,
        "the key move is too large": KeyMoveRefused,
        "the key move is not a JWS": KeyMoveRefused,
        "the key move is not a mandate-key-move JWS carrying its key": KeyMoveRefused,
        "the key move is not signed by the key that holds the task": KeyMoveRefused,
        "the key move's signature does not verify": KeyMoveRefused,
        "the key move is unreadable": KeyMoveRefused,
        "the key move is for another task or another key": KeyMoveRefused,
        "the key move carries no usable jti": KeyMoveRefused,
        "the key move is expired or lives longer than a minute": KeyMoveRefused,
        "the root was minted for a key that no longer holds this task": TaskMoved,
        "the task moved to another key first": TaskMoved,
        "this key already holds the task": TaskMoved,
        "the task is not active": TaskEnded,
        "the task cannot be taken over now": TaskEnded,
        "there is no such task": TaskNotYours,
        "the root is not one this agent may renew": TaskNotYours,
        # Refusals with no type of their own are the base, with the platform's words.
        "the actor assertion does not name the key that signed the request": TakeOverRefused,
        "the approval has already been used": TakeOverRefused,
        "the takeover was approved for another key": TakeOverRefused,
    }
    for why, kind in table.items():
        assert type(refused("invalid_grant", why)) is kind, why
    assert type(refused("slow_down", "a move of this task is already asked for, or too many were asked this hour")) is (
        TakeOverSlowDown
    )
    assert type(refused("invalid_client", "client authentication failed", 401)) is OAuthError
    assert type(refused("unsupported_grant_type", "mandates are not enabled on this deployment")) is OAuthError
    unavailable = take_over_error(
        HttpResponse(
            503,
            (),
            b'{"error":"temporarily_unavailable","error_description":"the root could not be checked"}',
            "https://x/oauth/token",
            "r1",
        ),
        "the token endpoint",
    )
    assert isinstance(unavailable, TemporarilyUnavailable)


def test_the_wire_form_is_the_root_and_the_move_and_nothing_else(gw: AgentGateway, key: AgentKey) -> None:
    receiver = _dpop.DPoPKey()
    with AuthFI(gw.issuer) as af:
        handover = start(af.agent(AGENT, key)).hand_over(to=receiver.thumbprint)
    wire = handover.to_wire()
    assert set(json.loads(wire)) == {"v", "mandate_id", "to", "root", "move", "expires_at"}
    again = Handover.from_wire(wire)
    assert (again.mandate_id, again.to, again.expires_at) == (handover.mandate_id, handover.to, handover.expires_at)
    assert again.root.reveal() == handover.root.reveal() and again.move.reveal() == handover.move.reveal()
    assert handover.root.reveal() not in repr(handover) and handover.move.reveal() not in repr(handover)
    with pytest.raises(TypeError):
        pickle.dumps(handover)
    for bad in (
        "",
        "not json",
        "[]",
        "{}",
        json.dumps({**json.loads(wire), "d": "x"}),
        json.dumps({**json.loads(wire), "v": 2}),
    ):
        with pytest.raises(ConfigurationError) as caught:
            Handover.from_wire(bad)
        assert caught.value.code == "handover_invalid"


def test_a_task_cannot_be_taken_over_inside_a_task(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        handover = start(agent).hand_over(to=_dpop.DPoPKey().thumbprint)
        with start(agent), pytest.raises(ConfigurationError) as caught:
            agent.take_over(handover)
    assert caught.value.code == "task_nested"


async def test_take_over_in_asyncio_code(gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch) -> None:
    receiver = _dpop.DPoPKey()
    async with AsyncAuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        task: Any = await agent.task("support.refund", subject_token=person(), params={"order": "4411"})
        with as_process(receiver, monkeypatch):
            to = await agent.run_key_thumbprint()
        assert to == receiver.thumbprint
        handover = task.hand_over(to=to)
        with as_process(receiver, monkeypatch):
            taken = await agent.take_over(handover.to_wire())
        assert taken.mandate_id == task.mandate_id and taken.bound_key() is receiver
        with as_process(receiver, monkeypatch), pytest.raises(TaskMoved):
            await agent.take_over(handover)
    assert gw.holders[task.mandate_id] == receiver.thumbprint


def test_an_answer_for_another_task_than_the_one_handed_over_is_refused(
    gw: AgentGateway, key: AgentKey, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver = _dpop.DPoPKey()
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        task = start(agent)
        handover = task.hand_over(to=receiver.thumbprint)
        gw.mandates[task.mandate_id] = {**gw.mandates[task.mandate_id], "id": "mdt_other"}
        with as_process(receiver, monkeypatch), pytest.raises(OAuthError) as caught:
            agent.take_over(handover)
    assert caught.value.code == "invalid_response"
