"""authfi.launcher (``authfi run``): any coding tool through the loopback proxy, as the agent the environment names."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import cast

import httpx
import pytest

from authfi import AgentKey, cli, launcher
from authfi.errors import ConfigurationError
from tests.fake_gateway import FakeGateway

AGENT = "agt_run_tool1"
CHAT = {"model": "gpt-x", "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture(scope="module")
def key_file(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("run-tool") / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    return path.resolve()


@pytest.fixture
def bin_dir(tmp_path: Path) -> Path:
    """A directory holding one executable called ``codextool``, for ``shutil.which`` to find."""
    name = "codextool.exe" if os.name == "nt" else "codextool"
    tool = tmp_path / name
    tool.write_text("")
    tool.chmod(0o755)
    return tmp_path


def _agent_env(fake: FakeGateway, key_file: Path, bin_dir: Path) -> dict[str, str]:
    return {
        "AUTHFI_ISSUER": fake.issuer,
        "AUTHFI_AGENT_ID": AGENT,
        "AUTHFI_AGENT_KEY_FILE": str(key_file),
        "AUTHFI_AGENT_KID": "k1",
        "PATH": str(bin_dir),
    }


def test_the_openai_environment_points_the_tool_at_the_proxy_and_names_what_it_overrides() -> None:
    base = {
        "OPENAI_API_KEY": "sk-person",
        "OPENAI_BASE_URL": "https://api.openai.com/v1",
        "OPENAI_API_BASE": "",
        "AUTHFI_AGENT_ID": AGENT,
        "AUTHFI_DEVICE_KEY_FILE": "/home/p/.authfi/device.key",
        "HOME": "/home/p",
    }
    env, overridden = launcher.child_environment(base, "http://127.0.0.1:1/s", "openai")

    assert overridden == ["OPENAI_API_KEY", "OPENAI_BASE_URL"]  # an empty one was not the person's value
    assert env["OPENAI_BASE_URL"] == env["OPENAI_API_BASE"] == "http://127.0.0.1:1/s/v1"
    assert env["OPENAI_API_KEY"] == launcher.PLACEHOLDER_KEY
    assert env["HOME"] == "/home/p"
    assert not [name for name in env if name.startswith("AUTHFI_")]


def test_the_anthropic_environment_blanks_the_other_credential() -> None:
    env, overridden = launcher.child_environment(
        {"ANTHROPIC_AUTH_TOKEN": "tok-person", "AUTHFI_X": "1"}, "http://127.0.0.1:1/s", "anthropic"
    )

    assert overridden == ["ANTHROPIC_AUTH_TOKEN"]
    assert env["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:1/s"  # Anthropic's SDK adds /v1 itself
    assert env["ANTHROPIC_API_KEY"] == launcher.PLACEHOLDER_KEY and env["ANTHROPIC_AUTH_TOKEN"] == ""
    assert "OPENAI_API_KEY" not in env and "AUTHFI_X" not in env


def test_a_differently_cased_variable_does_not_survive_beside_ours() -> None:
    env, overridden = launcher.child_environment({"openai_api_key": "sk-person"}, "http://x/s", "openai")
    assert overridden == ["openai_api_key"]
    assert [v for n, v in env.items() if n.upper() == "OPENAI_API_KEY"] == [launcher.PLACEHOLDER_KEY]


def test_an_unknown_format_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="format"):
        launcher.child_environment({}, "http://x", "grpc")


async def test_an_openai_tools_calls_leave_with_the_agents_credential(
    fake: FakeGateway, key_file: Path, bin_dir: Path
) -> None:
    fake.model_script += [[{"type": "text", "text": "chat answer"}], [{"type": "text", "text": "responses answer"}]]
    seen: dict[str, object] = {}

    async def as_tool(argv: Sequence[str], env: Mapping[str, str]) -> int:
        seen["argv"] = list(argv)
        async with httpx.AsyncClient() as client:  # what an OpenAI SDK sends: base URL + path, the placeholder key
            headers = {"authorization": "Bearer " + env["OPENAI_API_KEY"]}
            chat = await client.post(env["OPENAI_BASE_URL"] + "/chat/completions", headers=headers, json=CHAT)
            responses = await client.post(
                env["OPENAI_BASE_URL"] + "/responses", headers=headers, json={"model": "gpt-x", "input": "hi"}
            )
        seen["chat"], seen["responses"] = chat.json(), responses.json()
        return 3

    code = await launcher.run(
        ["codextool", "--flag"],
        api="openai",
        env=_agent_env(fake, key_file, bin_dir),
        spawn=as_tool,
        warn=lambda _t: None,
    )

    assert code == 3
    assert cast(list[str], seen["argv"])[1:] == ["--flag"]
    assert "chat answer" in json.dumps(seen["chat"]) and "responses answer" in json.dumps(seen["responses"])
    [chat_request] = fake.model_calls("/v1/chat/completions")
    [responses_request] = fake.model_calls("/v1/responses")
    for request in (chat_request, responses_request):
        assert request["headers"]["authorization"].startswith("Bearer tok-minted-")  # the agent's, not the placeholder
        assert launcher.PLACEHOLDER_KEY not in json.dumps(request["headers"])


async def test_the_openai_proxy_forwards_no_other_route_and_says_so_in_openai_shape(
    fake: FakeGateway, key_file: Path, bin_dir: Path
) -> None:
    answers: list[httpx.Response] = []

    async def as_tool(_argv: Sequence[str], env: Mapping[str, str]) -> int:
        root = env["OPENAI_BASE_URL"].removesuffix("/v1")
        async with httpx.AsyncClient() as client:
            answers.append(await client.post(root + "/v1/messages", json={}))  # the other format's route
            answers.append(await client.get(root + "/v1/models"))  # the model door lists them; this route does not
            answers.append(await client.post(root + "/v1/cloud/credentials", json={}))
        return 0

    await launcher.run(
        ["codextool"], api="openai", env=_agent_env(fake, key_file, bin_dir), spawn=as_tool, warn=lambda _t: None
    )

    assert [a.status_code for a in answers] == [404, 404, 404]
    assert answers[0].json()["error"]["type"] == "invalid_request_error"
    assert not fake.model_calls("/v1/messages") and not fake.model_calls("/v1/models")


async def test_the_anthropic_format_serves_messages_and_refuses_counting_tokens_cleanly(
    fake: FakeGateway, key_file: Path, bin_dir: Path
) -> None:
    fake.model_script += [[{"type": "text", "text": "anthropic answer"}]]
    answers: dict[str, httpx.Response] = {}

    async def as_tool(_argv: Sequence[str], env: Mapping[str, str]) -> int:
        base = env["ANTHROPIC_BASE_URL"]
        async with httpx.AsyncClient() as client:
            headers = {"x-api-key": env["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"}
            body = {"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
            answers["messages"] = await client.post(base + "/v1/messages?beta=true", headers=headers, json=body)
            answers["count"] = await client.post(
                base + "/v1/messages/count_tokens?beta=true", headers=headers, json=body
            )
            answers["chat"] = await client.post(base + "/v1/chat/completions", headers=headers, json=CHAT)
            answers["hello"] = await client.head(base + "/api/hello")
        return 0

    await launcher.run(
        ["codextool"], api="anthropic", env=_agent_env(fake, key_file, bin_dir), spawn=as_tool, warn=lambda _t: None
    )

    assert "anthropic answer" in answers["messages"].text
    assert answers["count"].status_code == 404
    assert answers["count"].json() == {
        "type": "error",
        "error": {"type": "not_found_error", "message": "not offered through the AuthFI gateway"},
    }
    assert answers["chat"].status_code == 404 and answers["hello"].status_code == 404
    assert not answers["hello"].content
    assert len(fake.model_requests()) == 1 and not fake.model_calls("/v1/chat/completions")


async def test_a_persons_own_key_is_named_and_the_template_is_filled(
    fake: FakeGateway, key_file: Path, bin_dir: Path
) -> None:
    said: list[str] = []
    given: dict[str, object] = {}

    async def exits(argv: Sequence[str], env: Mapping[str, str]) -> int:
        given["argv"], given["env"] = list(argv), dict(env)
        return 0

    env = {**_agent_env(fake, key_file, bin_dir), "OPENAI_API_KEY": "sk-person"}
    await launcher.run(["codextool", "-c", 'openai_base_url="{base_url}"'], env=env, spawn=exits, warn=said.append)

    tool_env = cast(dict[str, str], given["env"])
    assert said and "OPENAI_API_KEY" in said[0] and "codextool" in said[0]
    assert tool_env["OPENAI_API_KEY"] == launcher.PLACEHOLDER_KEY
    assert cast(list[str], given["argv"])[2] == f'openai_base_url="{tool_env["OPENAI_BASE_URL"]}"'
    assert not [name for name in tool_env if name.startswith("AUTHFI_")]


async def test_the_proxy_is_closed_when_the_tool_exits(fake: FakeGateway, key_file: Path, bin_dir: Path) -> None:
    urls: list[str] = []

    async def exits(_argv: Sequence[str], env: Mapping[str, str]) -> int:
        urls.append(env["OPENAI_BASE_URL"])
        return 0

    await launcher.run(["codextool"], env=_agent_env(fake, key_file, bin_dir), spawn=exits, warn=lambda _t: None)

    async with httpx.AsyncClient() as client:
        with pytest.raises(httpx.ConnectError):
            await client.post(urls[0] + "/chat/completions", json=CHAT)


async def test_a_machine_client_a_missing_tool_and_no_tool_are_refused(
    fake: FakeGateway, key_file: Path, bin_dir: Path
) -> None:
    async def never(_argv: Sequence[str], _env: Mapping[str, str]) -> int:
        raise AssertionError("the tool must not start")

    machine = {"AUTHFI_ISSUER": fake.issuer, "AUTHFI_CLIENT_ID": "app1", "AUTHFI_CLIENT_SECRET": "s", "PATH": ""}
    with pytest.raises(ConfigurationError, match="not on PATH"):
        await launcher.run(["codextool"], env=machine, spawn=never)
    with pytest.raises(ConfigurationError, match="machine client"):
        await launcher.run(["codextool"], env={**machine, "PATH": str(bin_dir)}, spawn=never)
    with pytest.raises(ConfigurationError, match="name the tool"):
        await launcher.run([], env=_agent_env(fake, key_file, bin_dir), spawn=never)


def test_the_command_line_passes_the_format_and_what_follows_the_separator(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], str]] = []

    def record(argv: Sequence[str], *, api: str) -> int:
        calls.append((list(argv), api))
        return 0

    monkeypatch.setattr(launcher, "command", record)

    assert cli.main(["run", "--format", "anthropic", "--", "opencode", "--model", "x"]) == 0
    assert cli.main(["run", "--", "codex"]) == 0
    assert calls == [(["opencode", "--model", "x"], "anthropic"), (["codex"], "openai")]
