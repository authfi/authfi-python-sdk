"""authfi.claude_agent end to end: the real Claude Code CLI, bundled with the SDK, against the gateway stand-in."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("claude_agent_sdk")

import claude_agent_sdk
from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, ResultMessage, TextBlock

from authfi import AgentKey, AsyncAuthFI
from authfi.claude_agent import ClaudeAgentIntegration, PendingApproval
from tests.fake_gateway import ALLOWED, STEP_UP, FakeGateway

AGENT = "agt_claude_cli1"
CLI = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude"

pytestmark = pytest.mark.skipif(not os.access(CLI, os.X_OK), reason="the SDK's bundled Claude Code is not here")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def key_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("cli") / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    return path.resolve()


async def _run(claude: ClaudeAgentIntegration, prompt: str, cwd: Path) -> tuple[list[str], ResultMessage | None]:
    options = await claude.options(model="claude-haiku-5-5", cwd=str(cwd), max_turns=4)
    texts: list[str] = []
    result: ResultMessage | None = None
    async with ClaudeSDKClient(options) as client:
        await client.query(prompt)
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                texts += [block.text for block in message.content if isinstance(block, TextBlock)]
            elif isinstance(message, ResultMessage):
                result = message
    return texts, result


def _use(tool: str) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": "mcp__authfi__" + tool, "input": {"repoName": "pallets/flask"}}]


async def test_claude_code_calls_the_model_and_a_tool_as_the_agent(
    fake: FakeGateway, key_file: Path, tmp_path: Path
) -> None:
    fake.model_script += [_use(ALLOWED), [{"type": "text", "text": "The wiki has an overview."}]]
    async with AsyncAuthFI(fake.issuer) as af:
        claude = ClaudeAgentIntegration(af.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
        texts, result = await _run(claude, "Describe the wiki", tmp_path)

    assert result is not None
    assert "The wiki has an overview." in texts
    assert fake.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    model_calls = fake.model_requests()
    assert model_calls
    for request in model_calls:  # the apiKeyHelper's token, minted in Claude Code's own process
        assert request["headers"]["authorization"].startswith("Bearer tok-minted-")


async def test_claude_code_waits_for_an_approval(fake: FakeGateway, key_file: Path, tmp_path: Path) -> None:
    fake.model_script += [_use(STEP_UP), [{"type": "text", "text": "Read it."}]]
    seen: list[PendingApproval] = []

    async def until_decided(pending: PendingApproval) -> None:
        seen.append(pending)
        fake.decide(pending.approval_id, "approved")

    async with AsyncAuthFI(fake.issuer) as af:
        agent = af.agent(AGENT, AgentKey.from_file(key_file, kid="k1"))
        claude = ClaudeAgentIntegration(agent, wait_for_decision=until_decided)
        texts, _ = await _run(claude, "Read the wiki", tmp_path)

    assert "Read it." in texts
    assert len(seen) == 1
    assert fake.tool_calls == [(STEP_UP, {"repoName": "pallets/flask"})]
    everything_the_model_read = str([r["json"] for r in fake.model_requests()])
    assert seen[0].approval_id not in everything_the_model_read
