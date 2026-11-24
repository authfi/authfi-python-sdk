"""authfi.claude_code: the interactive Claude Code through the loopback proxy, as the agent the environment names."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import httpx
import pytest

from authfi import AgentKey, claude_code, cli
from authfi.errors import ConfigurationError
from tests.fake_gateway import FakeGateway

AGENT = "agt_claude_code1"


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def key_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("claude-code") / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    return path.resolve()


def _agent_env(fake: FakeGateway, key_file: Path) -> dict[str, str]:
    return {
        "AUTHFI_ISSUER": fake.issuer,
        "AUTHFI_AGENT_ID": AGENT,
        "AUTHFI_AGENT_KEY_FILE": str(key_file),
        "AUTHFI_AGENT_KID": "k1",
        "PATH": "",
    }


def test_the_environment_points_claude_code_at_the_proxy_and_names_what_it_overrides() -> None:
    base = {
        "ANTHROPIC_API_KEY": "sk-person",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "ANTHROPIC_AUTH_TOKEN": "",
        "AUTHFI_AGENT_ID": AGENT,
        "AUTHFI_DEVICE_KEY_FILE": "/home/p/.authfi/device.key",
        "HOME": "/home/p",
    }
    env, overridden = claude_code.child_environment(base, "http://127.0.0.1:1/s", "claude-haiku-5-5")

    assert overridden == ["ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL"]  # an empty one was not the person's value
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1/s"
    assert env["ANTHROPIC_API_KEY"] == env["ANTHROPIC_AUTH_TOKEN"] == env["AUTHFI_AGENT_PRIVATE_KEY"] == ""
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert env["CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS"] == "1"
    assert env["ANTHROPIC_SMALL_FAST_MODEL"] == env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "claude-haiku-5-5"
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
    assert env["HOME"] == "/home/p"
    assert "AUTHFI_AGENT_ID" not in env and "AUTHFI_DEVICE_KEY_FILE" not in env


def test_a_persons_own_attribution_setting_is_kept() -> None:
    env, _ = claude_code.child_environment({"CLAUDE_CODE_ATTRIBUTION_HEADER": "1"}, "http://x", "m")
    assert env["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "1"


async def test_claude_codes_model_call_leaves_with_the_agents_credential(fake: FakeGateway, key_file: Path) -> None:
    fake.model_script += [[{"type": "text", "text": "Hello from the gateway."}]]
    seen: dict[str, object] = {}

    async def as_claude_code(argv: Sequence[str], env: Mapping[str, str]) -> int:
        seen["argv"], seen["env"] = list(argv), dict(env)
        async with httpx.AsyncClient() as client:  # what Claude Code sends: the placeholder key, to the proxy
            answer = await client.post(
                env["ANTHROPIC_BASE_URL"] + "/v1/messages",
                headers={"x-api-key": "authfi-loopback", "anthropic-version": "2023-06-01"},
                json={"model": "claude-haiku-5-5", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]},
            )
        seen["answer"] = answer.json()
        return 7

    code = await claude_code.run(
        ["--resume", "abc"],
        env=_agent_env(fake, key_file),
        claude="claude",
        spawn=as_claude_code,
        warn=lambda _text: None,
    )

    assert code == 7
    argv = cast(list[str], seen["argv"])
    assert argv[0] == "claude" and argv[-2:] == ["--resume", "abc"]
    assert json.loads(argv[argv.index("--settings") + 1]) == {"apiKeyHelper": "echo authfi-loopback"}
    assert argv[argv.index("--model") + 1] == "claude-haiku-5-5"
    [request] = fake.model_requests()
    authorization = request["headers"]["authorization"]
    assert authorization.startswith("Bearer tok-minted-")  # the agent's own token, minted in this process
    assert "authfi-loopback" not in json.dumps(request["headers"])  # the placeholder never left
    assert "Hello from the gateway." in json.dumps(seen["answer"])


async def test_the_proxy_is_closed_when_claude_code_exits(fake: FakeGateway, key_file: Path) -> None:
    urls: list[str] = []

    async def exits(_argv: Sequence[str], env: Mapping[str, str]) -> int:
        urls.append(env["ANTHROPIC_BASE_URL"])
        return 0

    await claude_code.run(env=_agent_env(fake, key_file), claude="claude", spawn=exits, warn=lambda _text: None)

    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError):
            await client.post(urls[0] + "/v1/messages", json={})


async def test_a_persons_own_anthropic_key_is_overridden_and_named(fake: FakeGateway, key_file: Path) -> None:
    said: list[str] = []
    given: dict[str, str] = {}

    async def exits(_argv: Sequence[str], env: Mapping[str, str]) -> int:
        given.update(env)
        return 0

    env = {**_agent_env(fake, key_file), "ANTHROPIC_API_KEY": "sk-person"}
    await claude_code.run(env=env, claude="claude", spawn=exits, warn=said.append)

    assert given["ANTHROPIC_API_KEY"] == ""
    assert said and "ANTHROPIC_API_KEY" in said[0]


async def test_a_machine_client_is_refused(fake: FakeGateway) -> None:
    env = {"AUTHFI_ISSUER": fake.issuer, "AUTHFI_CLIENT_ID": "app1", "AUTHFI_CLIENT_SECRET": "s"}

    async def never(_argv: Sequence[str], _env: Mapping[str, str]) -> int:
        raise AssertionError("Claude Code must not start")

    with pytest.raises(ConfigurationError, match="machine client"):
        await claude_code.run(env=env, claude="claude", spawn=never)


async def test_claude_code_not_on_path_is_refused(fake: FakeGateway, key_file: Path) -> None:
    with pytest.raises(ConfigurationError, match="not on PATH"):
        await claude_code.run(env=_agent_env(fake, key_file))


def test_the_command_line_passes_what_follows_the_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], str, str]] = []

    def record(args: Sequence[str], *, model: str, small_model: str) -> int:
        calls.append((list(args), model, small_model))
        return 0

    monkeypatch.setattr(claude_code, "command", record)

    assert cli.main(["claude", "--model", "claude-sonnet-5-5", "--", "--resume", "abc"]) == 0
    assert calls == [(["--resume", "abc"], "claude-sonnet-5-5", claude_code.DEFAULT_MODEL)]


async def test_what_the_gateway_does_not_serve_is_refused_cleanly_and_never_sent(
    fake: FakeGateway, key_file: Path
) -> None:
    answers: dict[str, httpx.Response] = {}

    async def as_claude_code(_argv: Sequence[str], env: Mapping[str, str]) -> int:
        base = env["ANTHROPIC_BASE_URL"]
        async with httpx.AsyncClient() as client:  # the calls Claude Code makes besides the model call
            body = {"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "hi"}]}
            answers["count"] = await client.post(base + "/v1/messages/count_tokens?beta=true", json=body)
            answers["models"] = await client.get(base + "/v1/models?limit=1000")
            answers["hello"] = await client.head(base + "/api/hello")
        return 0

    await claude_code.run(env=_agent_env(fake, key_file), claude="claude", spawn=as_claude_code, warn=lambda _t: None)

    assert {name: r.status_code for name, r in answers.items()} == {"count": 404, "models": 404, "hello": 404}
    assert answers["count"].json()["error"]["type"] == "not_found_error"  # Claude Code then estimates the count
    assert not answers["hello"].content
    assert not [r for r in fake.requests if r["path"].startswith("/mcp/v1")]
