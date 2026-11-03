"""authfi.claude_agent with the platform's agent security on.

Claude Code itself is not started. It cannot hold the agent's DPoP key, so its model calls go to a loopback proxy in
this process, which sends them with the key's proof, like the in-process MCP tools; ``authfi headers`` (the helper of
its trace export) cannot hold the key either, so the token commands ask for a Bearer token.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

pytest.importorskip("claude_agent_sdk")

import jwt

from authfi import AgentKey, AsyncAuthFI, GatewayTokenRefused, NotSupportedError
from authfi import _model_proxy as model_proxy
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


async def _proxy_address(claude: ClaudeAgentIntegration) -> tuple[str, int, str]:
    base_url = dict((await claude.options(model="claude-haiku-5-5")).env)["ANTHROPIC_BASE_URL"]
    origin, _, secret = base_url.rpartition("/")
    host, _, port = origin.removeprefix("http://").partition(":")
    return host, int(port), secret


async def _raw(host: str, port: int, request: bytes, *, wait: float = 5.0) -> bytes:
    """What the proxy answers to these exact bytes: a client library would tidy the target before sending."""
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(request)
        await writer.drain()
        return await asyncio.wait_for(reader.read(65536), wait)
    except ConnectionError:
        return b""
    finally:
        writer.close()


def _post(target: str, extra: str = "", body: bytes = b"{}") -> bytes:
    head = f"POST {target} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
    return (head + extra + (f"Content-Length: {len(body)}\r\n" if "Content-Length" not in extra else "")).encode() + (
        b"\r\n" + body
    )


async def test_only_the_model_call_is_forwarded(gw: AgentGateway, project: AsyncAuthFI, key_file: Path) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    base_url = dict((await claude.options(model="claude-haiku-5-5")).env)["ANTHROPIC_BASE_URL"]
    before = len(gw.requests)

    refused = [
        await _model_call(base_url, "/v1/cloud/credentials"),  # the cloud credential door, with the agent's proof
        await _model_call(base_url, "/v1/messages/count_tokens"),
        await _model_call(base_url, "/v1/chat/completions"),
        await _model_call(base_url, "/mcp"),  # the tool door
        await _model_call(base_url, "/oauth/token"),
        await _model_call(base_url, "/v1/messages?" + "a" * 500),  # a query that is not one of Claude Code's
    ]
    async with httpx.AsyncClient() as http:
        refused.append(await http.get(base_url + "/v1/messages"))
        refused.append(await http.request("DELETE", base_url + "/v1/messages"))

    assert [r.status_code for r in refused] == [404] * 8
    assert len(gw.requests) == before  # none of them was sent on


@pytest.mark.parametrize(
    "target",
    [
        "/{wrong}/v1/messages",  # a secret of the right length that is not ours
        "/{prefix}/v1/messages",  # the start of ours
        "//v1/messages",  # none
        "/v1/messages",
        "/{secret}/v1/../mcp",  # dot segments, which a client library would have removed
        "/{secret}/v1/messages/../../mcp",
        "/{secret}/v1/./messages/../cloud/credentials",
        "/{secret}/v1%2Fmessages",  # an encoded slash, and
        "/{secret}/v1/%6Dessages",  # an encoded letter
        "/{secret}/v1/messages/",
        "/{secret}/v1//messages",
        "/{secret}/mcp",
        "/{secret}/v1/cloud/credentials",
        "http://127.0.0.1/{secret}/v1/messages",  # absolute-form
    ],
)
async def test_what_is_not_the_call_under_the_secret_is_never_sent_on(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path, target: str
) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    host, port, secret = await _proxy_address(claude)
    before = len(gw.requests)
    target = target.format(wrong="x" * len(secret), prefix=secret[:10], secret=secret)

    answer = await _raw(host, port, _post(target))

    assert answer.startswith((b"HTTP/1.1 404", b"HTTP/1.1 400"))
    assert len(gw.requests) == before
    assert secret.encode() not in answer


@pytest.mark.parametrize(
    "extra",
    [
        "Content-Length: -5\r\n",
        "Content-Length: +5\r\n",
        "Content-Length: 1_0\r\n",
        "Content-Length: \u00b2\r\n",
        "Content-Length: 99999999999\r\n",
        "Content-Length: 2\r\nContent-Length: 2\r\n",
        "Content-Length: 2\r\nTransfer-Encoding: chunked\r\n",
    ],
)
async def test_a_malformed_length_is_a_bad_request_not_a_failure(
    gw: AgentGateway,
    project: AsyncAuthFI,
    key_file: Path,
    caplog: pytest.LogCaptureFixture,
    extra: str,
) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    host, port, secret = await _proxy_address(claude)
    before = len(gw.requests)
    request = f"POST /{secret}/v1/messages HTTP/1.1\r\nHost: x\r\n{extra}\r\n".encode("latin-1", "replace") + b"{}"

    with caplog.at_level("DEBUG", logger="authfi"):
        answer = await _raw(host, port, request)

    assert answer.startswith((b"HTTP/1.1 400", b"HTTP/1.1 413"))
    assert len(gw.requests) == before
    assert not [
        r for r in caplog.records if r.levelname == "ERROR" or r.exc_info
    ]  # no traceback for a client's mistake


@pytest.mark.parametrize("size", ["-5", "+5", "1_0", "\u00b2", "g", "", "f" * 9, "a" * 70000])
async def test_a_malformed_chunk_is_a_bad_request_not_a_failure(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path, caplog: pytest.LogCaptureFixture, size: str
) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    host, port, secret = await _proxy_address(claude)
    before = len(gw.requests)
    head = f"POST /{secret}/v1/messages HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
    request = head.encode() + size.encode("utf-8") + b"\r\n{}\r\n0\r\n\r\n"

    with caplog.at_level("DEBUG", logger="authfi"):
        answer = await _raw(host, port, request)

    assert answer.startswith(b"HTTP/1.1 400")
    assert len(gw.requests) == before
    assert not [r for r in caplog.records if r.levelname == "ERROR" or r.exc_info]


async def test_a_chunked_model_call_is_forwarded(gw: AgentGateway, project: AsyncAuthFI, key_file: Path) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    host, port, secret = await _proxy_address(claude)
    body = json.dumps(MODEL_REQUEST).encode()
    head = f"POST /{secret}/v1/messages HTTP/1.1\r\nHost: x\r\nTransfer-Encoding: chunked\r\n\r\n"
    request = head.encode() + f"{len(body):x}\r\n".encode() + body + b"\r\n0\r\n\r\n"

    answer = await _raw(host, port, request)

    assert answer.startswith(b"HTTP/1.1 200")
    assert [r for r in gw.run_calls if r[0] == "/mcp/v1/messages"]


async def test_a_head_that_never_ends_is_closed(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_proxy, "_HEAD_SECONDS", 0.3)
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    host, port, _ = await _proxy_address(claude)

    started = time.monotonic()
    answer = await _raw(host, port, b"POST /x HTTP/1.1\r\nHost: x\r\n", wait=3.0)  # and then nothing

    assert time.monotonic() - started < 2.0
    assert answer.startswith(b"HTTP/1.1 408")


async def test_a_body_that_never_ends_is_closed(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_proxy, "_BODY_SECONDS", 0.3)
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    host, port, secret = await _proxy_address(claude)
    before = len(gw.requests)

    answer = await _raw(host, port, _post(f"/{secret}/v1/messages", "Content-Length: 100\r\n", b"{"), wait=3.0)

    assert answer.startswith(b"HTTP/1.1 408")
    assert len(gw.requests) == before


async def test_sockets_held_open_without_a_request_cannot_keep_the_real_caller_out(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_proxy, "_HEAD_SECONDS", 60.0)  # it is not the timeout that makes room
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    host, port, secret = await _proxy_address(claude)

    holders = [await asyncio.open_connection(host, port) for _ in range(model_proxy._MAX_UNKNOWN)]  # pyright: ignore[reportPrivateUsage]
    try:
        await asyncio.sleep(0.2)
        answer = await _raw(host, port, _post(f"/{secret}/v1/messages", body=json.dumps(MODEL_REQUEST).encode()))

        assert answer.startswith(b"HTTP/1.1 200")  # the real caller took the place of the oldest
        oldest = holders[0][0]
        assert await asyncio.wait_for(oldest.read(), 2.0) == b""  # which was closed
    finally:
        for _, writer in holders:
            writer.close()


async def test_a_connection_cancelled_before_its_first_step_leaves_nothing_behind(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_proxy, "_MAX_UNKNOWN", 1)
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    await _proxy_address(claude)
    proxy = claude._proxy  # pyright: ignore[reportPrivateUsage]
    stub: Any = SimpleNamespace(transport=SimpleNamespace(abort=lambda: None), close=lambda: None)
    no_reader: Any = None  # the task is cancelled before it reads

    proxy._accept(no_reader, stub)  # pyright: ignore[reportPrivateUsage]
    await proxy.aclose()  # before the loop has run the connection's task

    assert proxy._connections == set()  # pyright: ignore[reportPrivateUsage]
    assert proxy._preauth == {}  # pyright: ignore[reportPrivateUsage]
    # the same object, started again, takes calls as at first
    host, port, secret = await _proxy_address(claude)
    held = await asyncio.open_connection(host, port)
    try:
        answer = await _raw(host, port, _post(f"/{secret}/v1/messages", body=json.dumps(MODEL_REQUEST).encode()))
        assert answer.startswith(b"HTTP/1.1 200")
    finally:
        held[1].close()


async def test_the_callers_user_agent_is_the_calls_client(
    gw: AgentGateway, project: AsyncAuthFI, key_file: Path
) -> None:
    claude = ClaudeAgentIntegration(project.agent(AGENT, AgentKey.from_file(key_file, kid="k1")))
    base_url = dict((await claude.options(model="claude-haiku-5-5")).env)["ANTHROPIC_BASE_URL"]
    agent = "claude-cli/2.1.316 (external, sdk-py, agent-sdk/0.2.176)"
    async with httpx.AsyncClient() as http:
        await http.post(base_url + "/v1/messages", headers={"User-Agent": agent}, json=MODEL_REQUEST)
        await http.post(base_url + "/v1/messages", headers={"User-Agent": "authfi-loopback"}, json=MODEL_REQUEST)

    seen = [r["headers"]["user-agent"] for r in gw.requests if r["path"] == "/mcp/v1/messages"]
    assert seen == [agent, "authfi-loopback"]


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
