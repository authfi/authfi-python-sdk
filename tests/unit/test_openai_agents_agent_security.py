"""authfi.openai_agents with the platform's agent security on: DPoP on the model and the tools, and tasks.

The SDK's own Runner, LiteLLM and MCP client, against :class:`tests.fake_agent_gateway.AgentGateway`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("agents")

from agents import Agent, Runner
from agents.items import ToolCallOutputItem

from authfi import AgentKey, AsyncAuthFI
from authfi.openai_agents import AuthFIMCPServer, OpenAIAgentsIntegration
from tests.fake_agent_gateway import OUTSIDE, AgentGateway
from tests.fake_gateway import ALLOWED
from tests.unit.test_agent_capabilities import person

AGENT = "agt_oasec1"
PROMPT = "Look up the wiki"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def gw() -> Iterator[AgentGateway]:
    with AgentGateway(nonce=True) as gateway:
        yield gateway


@pytest.fixture
async def project(gw: AgentGateway) -> AsyncIterator[AsyncAuthFI]:
    async with AsyncAuthFI(gw.issuer) as af:
        yield af


@pytest.fixture
async def integration(project: AsyncAuthFI, key: AgentKey) -> OpenAIAgentsIntegration:
    return OpenAIAgentsIntegration(project.agent(AGENT, key))


@pytest.fixture
async def tools(integration: OpenAIAgentsIntegration) -> AsyncIterator[AuthFIMCPServer]:
    async with integration.mcp_server() as server:
        yield server


def _agent(integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer) -> Agent[Any]:
    return Agent(name="research", model=integration.model("claude-haiku-5-5"), mcp_servers=[tools])


def _use(name: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": name, "input": {"repoName": "pallets/flask", **args}}]


async def test_the_model_and_the_tools_prove_the_key_and_answer_a_nonce(
    gw: AgentGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer
) -> None:
    await integration.gateway.source.token()
    gw._nonce = "nonce-rotated"  # asked for once on the model call and once on the tool call
    gw.model_script += [_use(ALLOWED), [{"type": "text", "text": "Done."}]]

    result = await Runner.run(_agent(integration, tools), PROMPT, run_config=integration.run_config())

    assert result.final_output == "Done."
    assert gw.tool_calls == [(ALLOWED, {"repoName": "pallets/flask"})]
    model = gw.model_requests()
    assert model and all(r["headers"]["authorization"].startswith("DPoP ") and "dpop" in r["headers"] for r in model)
    assert gw.nonce_challenges >= 2 and gw.refused_proofs == []
    paths = {p["path"] for p in gw.proofs}
    assert {"/mcp", "/mcp/v1/messages"} <= paths


async def test_a_model_call_that_is_the_first_to_meet_a_new_nonce_answers_it_once(
    gw: AgentGateway, integration: OpenAIAgentsIntegration
) -> None:
    """LiteLLM is given the proof in a header, so the adapter answers the nonce itself."""
    await integration.gateway.source.token()
    challenges = gw.nonce_challenges
    gw._nonce = "nonce-rotated"

    result = await Runner.run(
        Agent(name="research", model=integration.model("claude-haiku-5-5")),
        PROMPT,
        run_config=integration.run_config(),
    )

    assert result.final_output == "Done."
    assert gw.nonce_challenges == challenges + 1 and gw.refused_proofs == []
    assert len(gw.model_requests()) == 2  # the challenge, then the call with the nonce


async def test_a_streamed_model_call_answers_the_nonce_too(
    gw: AgentGateway, integration: OpenAIAgentsIntegration, tools: AuthFIMCPServer
) -> None:
    await integration.gateway.source.token()
    gw._nonce = "nonce-rotated"

    result = Runner.run_streamed(_agent(integration, tools), PROMPT, run_config=integration.run_config())
    async for _ in result.stream_events():
        pass

    assert result.final_output == "Done."
    assert gw.refused_proofs == [] and gw.nonce_challenges >= 2


async def test_inside_a_task_a_call_outside_it_reaches_the_model_with_what_it_can_still_do(
    gw: AgentGateway, project: AsyncAuthFI, key: AgentKey
) -> None:
    agent = project.agent(AGENT, key)
    gw.model_script += [_use(OUTSIDE), [{"type": "text", "text": "I will stop here."}]]

    async with await agent.task("support.refund", subject_token=person(), params={"order": "4411"}) as task:
        integration = OpenAIAgentsIntegration(task)  # the task is the credential: one line
        async with integration.mcp_server() as tools:
            result = await Runner.run(_agent(integration, tools), PROMPT, run_config=integration.run_config())

    (output,) = [json.dumps(i.output, default=str) for i in result.new_items if isinstance(i, ToolCallOutputItem)]
    assert "EMDT005" in output and "Use a tool the task allows" in output
    read = json.dumps([r["json"] for r in gw.model_requests()])
    assert "Use a tool the task allows" in read and "new task" not in read.lower()
    assert result.final_output == "I will stop here."
    carried = {
        r["headers"]["authorization"].split(" ", 1)[1] for r in gw.requests if r["path"] in {"/mcp", "/mcp/v1/messages"}
    }
    assert {gw.issued[t]["kind"] for t in carried} == {"task"}
