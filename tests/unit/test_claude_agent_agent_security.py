"""authfi.claude_agent with the platform's agent security on.

Claude Code itself is not started. It cannot hold the agent's DPoP key, so its model calls go to a loopback proxy in
this process, which sends them with the key's proof, like the in-process MCP tools; ``authfi headers`` (the helper of
its trace export) cannot hold the key either, so the token commands ask for a Bearer token.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

pytest.importorskip("claude_agent_sdk")

import jwt

from authfi import AgentKey, AsyncAuthFI, GatewayTokenRefused, NotSupportedError
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


# --- one Claude Code process is one run of the key ---------------------------------------------------------

#: What Claude Code sends the base URL it is given: the key its apiKeyHelper printed, as both headers it uses.
CLAUDE_CODES_KEY = "authfi-loopback"
MODEL_REQUEST: dict[str, Any] = {"model": "claude-haiku-5-5", "max_tokens": 8, "stream": True, "messages": []}


async def _model_call(base_url: str, path: str = "/v1/messages?beta=true") -> httpx.Response:
    headers = {"Authorization": f"Bearer {CLAUDE_CODES_KEY}", "x-api-key": CLAUDE_CODES_KEY}
    async with httpx.AsyncClient() as http:
        return await http.post(base_url + path, headers=headers, json=MODEL_REQUEST)


async def test_claude_codes_model_calls_and_the_tools_are_one_run_of_the_key(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path
) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    options = await claude.options(model="claude-haiku-5-5")
    base_url = dict(options.env)["ANTHROPIC_BASE_URL"]

    answer = await _model_call(base_url)
    await claude._handler(ALLOWED)(ARGS)  # pyright: ignore[reportPrivateUsage]

    assert answer.status_code == 200
    assert b"event: message_start" in answer.content  # the stream came through
    (model,) = [call for call in gw.run_calls if call[0] == "/mcp/v1/messages"]
    tools = [call for call in gw.run_calls if call[0] == "/mcp"]
    assert tools
    assert {run for _, run, _ in [model, *tools]} == {model[1]}  # one run, the tools' and the model's
    assert {scheme for _, _, scheme in [model, *tools]} == {"dpop"}  # of the key: no Bearer call is on the agent's run
    assert claude.gateway_run_id == model[1]


async def test_the_gateway_never_sees_what_claude_code_sent_as_its_key(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path
) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    base_url = dict((await claude.options(model="claude-haiku-5-5")).env)["ANTHROPIC_BASE_URL"]

    await _model_call(base_url)

    (request,) = [r for r in gw.requests if r["path"] == "/mcp/v1/messages"]
    assert "x-api-key" not in request["headers"]
    assert CLAUDE_CODES_KEY not in str(request["headers"])
    assert request["headers"]["authorization"].startswith("DPoP ")
    assert request["headers"]["dpop"]
    assert request["headers"]["x-request-id"].startswith(claude.run_id + ".")  # the call joins the process's run


async def test_a_model_call_passes_a_gateway_that_enforces_the_proof(key_file: Path) -> None:
    with AgentGateway(model_proof=True) as strict:
        async with AsyncAuthFI(strict.issuer) as af:
            claude = ClaudeAgentIntegration(af.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
            base_url = dict((await claude.options(model="claude-haiku-5-5")).env)["ANTHROPIC_BASE_URL"]
            answer = await _model_call(base_url)
    assert answer.status_code == 200
    assert strict.unproven_model_calls == []


async def test_the_trace_belongs_to_the_run_the_gateway_named(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path
) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    options = await claude.options(model="claude-haiku-5-5", export_steps=True)

    run = claude.gateway_run_id
    assert run is not None
    assert run.startswith("run_")
    assert dict(options.env)["OTEL_RESOURCE_ATTRIBUTES"] == f"authfi.agent_id={AGENT},authfi.run_id={run}"


async def test_only_the_model_routes_are_forwarded(gw: AgentGateway, project: AsyncAuthFI, key_file: Path) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    base_url = dict((await claude.options(model="claude-haiku-5-5")).env)["ANTHROPIC_BASE_URL"]
    origin, _, secret = base_url.rpartition("/")
    before = len(gw.requests)

    refused = [
        await _model_call(origin + "/not-the-secret"),  # another process on this machine
        await _model_call(origin, "/v1/messages"),  # no secret at all
        await _model_call(base_url, "/mcp"),  # the tool door
        await _model_call(base_url, "/v1/../mcp"),  # a way round to it
        await _model_call(base_url, "/oauth/token"),
    ]

    assert [r.status_code for r in refused] == [404] * 5
    assert len(gw.requests) == before  # none of them was sent on
    assert secret not in "".join(r.text for r in refused)


async def test_a_refused_token_on_a_model_call_is_the_runs_refusal(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path
) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    base_url = dict((await claude.options(model="claude-haiku-5-5")).env)["ANTHROPIC_BASE_URL"]
    gw.revoked = True

    answer = await _model_call(base_url)

    assert answer.status_code == 401
    assert isinstance(claude.token_refused, GatewayTokenRefused)


async def test_closing_the_project_stops_the_proxy(gw: AgentGateway, key_file: Path) -> None:
    async with AsyncAuthFI(gw.issuer) as af:
        claude = ClaudeAgentIntegration(af.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
        base_url = dict((await claude.options(model="claude-haiku-5-5")).env)["ANTHROPIC_BASE_URL"]
    with pytest.raises(httpx.ConnectError):
        await _model_call(base_url)
