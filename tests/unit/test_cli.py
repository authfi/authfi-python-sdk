from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from authfi import AgentKey, AuthFI
from authfi import cli as cli_module
from tests.agent_platform import AGENT, AgentPlatform


@pytest.fixture
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, key: AgentKey) -> AgentPlatform:
    platform = AgentPlatform(agent_jwk=key.public_jwk())
    path = tmp_path / "agent.key"
    key.write(path)
    monkeypatch.setenv("AUTHFI_ISSUER", platform.issuer)
    monkeypatch.setenv("AUTHFI_AGENT_ID", AGENT)
    monkeypatch.setenv("AUTHFI_AGENT_KEY_FILE", str(path))
    monkeypatch.setenv("AUTHFI_AGENT_KID", "k1")
    monkeypatch.delenv("AUTHFI_RUN_ID", raising=False)

    class Project(AuthFI):
        @classmethod
        def from_env(cls, *args: Any, **kwargs: Any) -> Any:
            kwargs["http_client"] = platform.client()
            return super().from_env(*args, **kwargs)

    monkeypatch.setattr(cli_module, "AuthFI", Project)
    return platform


def test_token_prints_the_agents_current_token(environment: AgentPlatform, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli_module.main(["token"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.count(".") == 2
    assert environment.minted == 1


def test_headers_prints_json_with_the_run_id(
    environment: AgentPlatform, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AUTHFI_RUN_ID", "run-42")
    assert cli_module.main(["headers"]) == 0
    headers = json.loads(capsys.readouterr().out)
    assert headers["Authorization"].startswith("Bearer ")
    assert headers["X-Request-Id"] == "run-42"
    monkeypatch.setenv("AUTHFI_RUN_ID", "not a valid id")
    assert cli_module.main(["headers"]) == 0
    assert "X-Request-Id" not in json.loads(capsys.readouterr().out)


def test_a_failure_prints_nothing_on_stdout(
    environment: AgentPlatform, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("AUTHFI_AGENT_ID")
    assert cli_module.main(["token"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "AUTHFI_AGENT_ID" in captured.err


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_keygen_writes_a_private_file_and_prints_the_public_half(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "new.key"
    assert cli_module.main(["keygen", "--out", str(out), "--alg", "ES256", "--kid", "k7"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["public_jwk"]["kid"] == "k7"
    assert printed["public_jwk"]["crv"] == "P-256"
    assert "d" not in printed["public_jwk"]
    assert AgentKey.from_file(out, kid="k7").public_jwk() == printed["public_jwk"]
    assert cli_module.main(["keygen", "--out", str(out)]) == 1
    assert "exists" in capsys.readouterr().err


def test_usage_errors_exit_2() -> None:
    with pytest.raises(SystemExit) as caught:
        cli_module.main([])
    assert caught.value.code == 2
