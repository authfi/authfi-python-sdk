"""authfi.claude_agent with the platform's agent security on.

Claude Code itself is not started. Its own model calls go through ``authfi token``, a helper process that cannot hold
the agent's DPoP key, so that command asks for a Bearer token; the in-process MCP tools carry the key's proof.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

pytest.importorskip("claude_agent_sdk")

import jwt

from authfi import AgentKey, AsyncAuthFI, NotSupportedError
from authfi import cli as cli_module
from authfi.claude_agent import ClaudeAgentIntegration
from tests.fake_agent_gateway import AgentGateway
from tests.fake_gateway import ALLOWED
from tests.unit.test_agent_capabilities import person

AGENT = "agt_clsec1"
ARGS = {"repoName": "pallets/flask"}


@pytest.fixture(scope="module")
def key_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("keys") / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    return path.resolve()


@pytest.fixture
def gw() -> Iterator[AgentGateway]:
    with AgentGateway(nonce=True) as gateway:
        yield gateway


@pytest.fixture
async def project(gw: AgentGateway) -> AsyncIterator[AsyncAuthFI]:
    async with AsyncAuthFI(gw.issuer) as af:
        yield af


async def test_the_in_process_tools_prove_the_key_and_answer_a_nonce(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path
) -> None:
    agent = project.agent(AGENT, AgentKey.from_file(key_file, kid="k1"))
    claude = ClaudeAgentIntegration(agent)
    await agent.token()
    gw._nonce = "nonce-rotated"

    answer = await claude._handler(ALLOWED)(ARGS)  # pyright: ignore[reportPrivateUsage]

    assert "answered for pallets/flask" in str(answer)
    calls = [r for r in gw.requests if r["path"] == "/mcp"]
    assert calls and all(r["headers"]["authorization"].startswith("DPoP ") for r in calls)
    assert gw.nonce_challenges >= 2 and gw.refused_proofs == []


async def test_a_task_is_not_supported_because_claude_codes_model_calls_cannot_be_held_to_it(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path
) -> None:
    agent = project.agent(AGENT, AgentKey.from_file(key_file, kid="k1"))
    async with await agent.task("support.refund", subject_token=person(), params={"order": "4411"}) as task:
        with pytest.raises(NotSupportedError) as caught:
            ClaudeAgentIntegration(task)  # type: ignore[arg-type]
    assert caught.value.feature == "claude_agent_task"


def test_the_token_command_gives_claude_code_a_bearer_token_it_can_use(
    gw: AgentGateway, key_file: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("AUTHFI_ISSUER", gw.issuer)
    monkeypatch.setenv("AUTHFI_AGENT_ID", AGENT)
    monkeypatch.setenv("AUTHFI_AGENT_KEY_FILE", str(key_file))
    monkeypatch.setenv("AUTHFI_AGENT_KID", "k1")

    assert cli_module.main(["token"]) == 0

    token = capsys.readouterr().out.strip()
    assert "cnf" not in jwt.decode(token, options={"verify_signature": False})  # not bound to a key it cannot prove
    assert gw.proofs == []  # and no proof was sent to ask for one
