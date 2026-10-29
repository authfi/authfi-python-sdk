"""authfi.adk with the platform's agent security on: DPoP on the model and the tools, and tasks.

ADK's own runner, LiteLlm and McpToolset, against :class:`tests.fake_agent_gateway.AgentGateway`.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

pytest.importorskip("google.adk")

from google.adk.events import Event
from google.adk.models.llm_request import LlmRequest
from google.adk.runners import InMemoryRunner
from google.genai import types

from authfi import AgentKey, AsyncAuthFI
from authfi.adk import ADKIntegration
from tests.fake_agent_gateway import OUTSIDE, AgentGateway
from tests.fake_gateway import ALLOWED
from tests.unit.test_agent_capabilities import person

AGENT = "agt_adksec1"
ARGS = {"repoName": "pallets/flask"}


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


async def _ask(adk: ADKIntegration, text: str) -> list[Event]:
    agent = adk.agent(name="research", instruction="Answer briefly.", model="claude-haiku-5-5")
    runner = InMemoryRunner(agent=agent, app_name="authfi-test")
    session = await runner.session_service.create_session(app_name="authfi-test", user_id="u")
    message = types.Content(role="user", parts=[types.Part(text=text)])
    return [e async for e in runner.run_async(user_id="u", session_id=session.id, new_message=message)]


def _final(events: list[Event]) -> str:
    texts = [p.text for e in events if e.content and e.content.parts for p in e.content.parts if p.text]
    return texts[-1] if texts else ""


def _use(tool: str, **args: Any) -> list[dict[str, Any]]:
    return [{"type": "tool_use", "name": tool, "input": {**ARGS, **args}}]


async def test_the_model_and_the_tools_prove_the_key_and_answer_a_nonce(
    gw: AgentGateway, project: AsyncAuthFI, key: AgentKey
) -> None:
    agent = project.agent(AGENT, key)
    adk = ADKIntegration(agent)
    await agent.token()
    gw._nonce = "nonce-rotated"
    gw.model_script += [_use(ALLOWED), [{"type": "text", "text": "It has an overview."}]]

    events = await _ask(adk, "Describe the wiki")

    assert _final(events) == "It has an overview."
    assert gw.tool_calls == [(ALLOWED, ARGS)]
    model = gw.model_requests()
    assert model and all(r["headers"]["authorization"].startswith("DPoP ") and "dpop" in r["headers"] for r in model)
    assert gw.nonce_challenges >= 2 and gw.refused_proofs == []
    assert {"/mcp", "/mcp/v1/messages"} <= {p["path"] for p in gw.proofs}


async def test_a_model_call_that_is_the_first_to_meet_a_new_nonce_answers_it_once(
    gw: AgentGateway, project: AsyncAuthFI, key: AgentKey
) -> None:
    """LiteLLM is given the proof in a header, so the adapter answers the nonce itself."""
    agent = project.agent(AGENT, key)
    await agent.token()
    challenges = gw.nonce_challenges
    gw._nonce = "nonce-rotated"
    request = LlmRequest(model="claude-haiku-5-5", contents=[types.Content(role="user", parts=[types.Part(text="hi")])])

    answers = [r async for r in ADKIntegration(agent).model("claude-haiku-5-5").generate_content_async(request)]

    assert answers and gw.nonce_challenges == challenges + 1 and gw.refused_proofs == []
    assert len(gw.model_requests()) == 2  # the challenge, then the call with the nonce


async def test_inside_a_task_a_call_outside_it_reaches_the_model_with_what_it_can_still_do(
    gw: AgentGateway, project: AsyncAuthFI, key: AgentKey
) -> None:
    agent = project.agent(AGENT, key)
    gw.model_script += [_use(OUTSIDE), [{"type": "text", "text": "I will stop here."}]]

    async with await agent.task("support.refund", subject_token=person(), params={"order": "4411"}) as task:
        events = await _ask(ADKIntegration(task), "Ask the wiki a question")  # the task is the credential

    assert _final(events) == "I will stop here."
    read = json.dumps([r["json"] for r in gw.model_requests()])
    assert "EMDT005" in read and "Use a tool the task allows" in read and "new task" not in read.lower()
    carried = {
        r["headers"]["authorization"].split(" ", 1)[1] for r in gw.requests if r["path"] in {"/mcp", "/mcp/v1/messages"}
    }
    assert {gw.issued[t]["kind"] for t in carried} == {"task"}
