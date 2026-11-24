"""AsyncGateway.await_approval: polling a held call's approval with the agent's own token."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from authfi import AgentKey, ApprovalUnavailable, AsyncAuthFI, NotSupportedError
from authfi.agents.agent import AsyncAgentAuth
from tests.fake_gateway import STEP_UP, FakeGateway

AGENT = "agt_poll1"


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway(approvals_readable=True) as gateway:
        yield gateway


@pytest.fixture
async def af(fake: FakeGateway) -> AsyncIterator[AsyncAuthFI]:
    async with AsyncAuthFI(fake.issuer) as project:
        yield project


@pytest.fixture
def agent(af: AsyncAuthFI, tmp_path: Path) -> AsyncAgentAuth:
    path = tmp_path / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    return af.agent(AGENT, AgentKey.from_file(path, kid="k1"))


async def _hold(fake: FakeGateway, agent: AsyncAgentAuth) -> str:
    approval_id = "apr-poll-1"
    fake.approvals[approval_id] = {"status": "pending", "tool": STEP_UP, "args": {}}
    return approval_id


@pytest.mark.parametrize("status", ["approved", "denied", "expired", "cancelled"])
async def test_a_decision_ends_the_wait_with_that_status(fake: FakeGateway, agent: AsyncAgentAuth, status: str) -> None:
    approval_id = await _hold(fake, agent)
    waiting = asyncio.ensure_future(agent.gateway().await_approval(approval_id, timeout=5, poll_interval=0.01))
    await asyncio.sleep(0.05)
    assert not waiting.done()
    fake.decide(approval_id, status)
    assert await waiting == status


async def test_no_decision_in_time_is_pending_and_the_approval_stays_open(
    fake: FakeGateway, agent: AsyncAgentAuth
) -> None:
    approval_id = await _hold(fake, agent)
    assert await agent.gateway().await_approval(approval_id, timeout=0.05, poll_interval=0.01) == "pending"
    assert fake.pending() == [approval_id]


async def test_a_revoked_agent_stops_the_wait_at_once(fake: FakeGateway, agent: AsyncAgentAuth) -> None:
    approval_id = await _hold(fake, agent)
    fake.revoked = True
    with pytest.raises(ApprovalUnavailable) as caught:
        await agent.gateway().await_approval(approval_id, timeout=5, poll_interval=0.01)
    assert caught.value.reason == "agent_refused"


async def test_an_unknown_approval_stops_the_wait_at_once(fake: FakeGateway, agent: AsyncAgentAuth) -> None:
    with pytest.raises(ApprovalUnavailable) as caught:
        await agent.gateway().await_approval("no-such", timeout=5, poll_interval=0.01)
    assert caught.value.reason == "approval_not_found"


async def test_a_credential_that_is_not_the_agents_own_cannot_poll(fake: FakeGateway, af: AsyncAuthFI) -> None:
    class Person:  # a token held for a person: the approval route refuses it
        pass

    gateway = af.agent(AGENT, AgentKey.generate(kid="k1")).gateway()
    gateway.source = Person()  # type: ignore[assignment]
    with pytest.raises(NotSupportedError):
        await gateway.await_approval("x")
