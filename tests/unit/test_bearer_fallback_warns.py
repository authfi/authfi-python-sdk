"""An agent token that falls back to Bearer says so at warning level, once.

A project can require sender-constrained agent tokens (and a new one does): the platform then refuses to mint an unbound
token. The fallback used to be one INFO line, so the first sign of it was a refused mint on the day the project moved.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from authfi import AgentKey, AuthFI
from tests.fake_agent_gateway import AgentGateway

AGENT = "agt_bearerwarn"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def gw() -> Iterator[AgentGateway]:
    with AgentGateway(nonce=True, model_proof=True) as gateway:
        yield gateway


def bearer_warnings(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [r for r in caplog.records if r.levelno >= logging.WARNING and "Bearer" in r.getMessage()]


def test_opting_out_of_dpop_warns_once_at_warning_level(
    gw: AgentGateway, key: AgentKey, caplog: pytest.LogCaptureFixture
) -> None:
    with AuthFI(gw.issuer) as af, caplog.at_level(logging.INFO):
        agent = af.agent(AGENT, key, dpop=False)
        agent.token(force=True)
        agent.token(force=True)
    warned = bearer_warnings(caplog)
    assert len(warned) == 1, [r.getMessage() for r in caplog.records]
    assert "sender-constrained" in warned[0].getMessage()


def test_a_bound_agent_makes_no_bearer_warning(gw: AgentGateway, key: AgentKey, caplog: pytest.LogCaptureFixture) -> None:
    """The control leg: the default agent binds its token and has nothing to warn about."""
    with AuthFI(gw.issuer) as af, caplog.at_level(logging.INFO):
        af.agent(AGENT, key).token(force=True)
    assert bearer_warnings(caplog) == []
