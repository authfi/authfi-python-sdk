"""Gateway.await_approval: the blocking twin of AsyncGateway's wait for a held call's approval."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from authfi import AgentKey, ApprovalUnavailable, AuthFI, NotSupportedError
from authfi.agents.agent import AgentAuth
from tests.fake_gateway import STEP_UP, FakeGateway

AGENT = "agt_poll2"


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway(approvals_readable=True) as gateway:
        yield gateway


@pytest.fixture
def af(fake: FakeGateway) -> Iterator[AuthFI]:
    with AuthFI(fake.issuer) as project:
        yield project


@pytest.fixture
def agent(af: AuthFI, tmp_path: Path) -> AgentAuth:
    path = tmp_path / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    return af.agent(AGENT, AgentKey.from_file(path, kid="k1"))


def _hold(fake: FakeGateway) -> str:
    approval_id = "apr-poll-2"
    fake.approvals[approval_id] = {"status": "pending", "tool": STEP_UP, "args": {}}
    return approval_id


@pytest.mark.parametrize("status", ["approved", "denied", "expired", "cancelled"])
def test_a_decision_ends_the_wait_with_that_status(fake: FakeGateway, agent: AgentAuth, status: str) -> None:
    approval_id = _hold(fake)
    decider = threading.Timer(0.1, fake.decide, args=(approval_id, status))
    decider.start()
    try:
        assert agent.gateway().await_approval(approval_id, timeout=5, poll_interval=0.01) == status
    finally:
        decider.cancel()


def test_no_decision_in_time_is_pending_and_the_approval_stays_open(fake: FakeGateway, agent: AgentAuth) -> None:
    approval_id = _hold(fake)
    assert agent.gateway().await_approval(approval_id, timeout=0.05, poll_interval=0.01) == "pending"
    assert fake.pending() == [approval_id]


def test_a_revoked_agent_stops_the_wait_at_once(fake: FakeGateway, agent: AgentAuth) -> None:
    approval_id = _hold(fake)
    fake.revoked = True
    with pytest.raises(ApprovalUnavailable) as caught:
        agent.gateway().await_approval(approval_id, timeout=5, poll_interval=0.01)
    assert caught.value.reason == "agent_refused"


def test_an_unknown_approval_stops_the_wait_at_once(fake: FakeGateway, agent: AgentAuth) -> None:
    with pytest.raises(ApprovalUnavailable) as caught:
        agent.gateway().await_approval("no-such", timeout=5, poll_interval=0.01)
    assert caught.value.reason == "approval_not_found"


def test_a_credential_that_is_not_the_agents_own_cannot_poll(fake: FakeGateway, af: AuthFI) -> None:
    class Person:  # a token held for a person: the approval route refuses it
        pass

    gateway = af.agent(AGENT, AgentKey.generate(kid="k1")).gateway()
    gateway.source = Person()  # type: ignore[assignment]
    with pytest.raises(NotSupportedError):
        gateway.await_approval("x")
