"""A workload-bound instance asks for a task with its own access token, and the task is held by its own key.

An instance holds no key it can sign a client assertion with, so the start's ``actor_token`` is its access token
(``actor_token_type`` the access-token type), bound to the key that signs the request's proof; agent-identity
verifies it as ``VerifyInstanceActor`` does and the root it answers is held by that same key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import jwt
import pytest

import authfi
from authfi import _dpop
from authfi.agents.task import AsyncTask, Task
from authfi.instance import AsyncInstanceAuth, InstanceAuth
from authfi.secret import Secret
from tests.agent_platform import EXCHANGE
from tests.instance_platform import InstancePlatform
from tests.support import FakeClock, form_of, json_response

MANDATE = "mdt_" + "1" * 32
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


@dataclass
class TaskPlatform(InstancePlatform):
    """The instance routes plus the task start the way agent-identity answers an instance's actor."""

    starts: list[dict[str, str]] = field(default_factory=list[dict[str, str]])

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = form_of(request)
        if form.get("grant_type") != EXCHANGE or "authorization_details" not in form:
            return super()._token(request)
        self.starts.append(form)
        if form.get("actor_token_type") != ACCESS_TOKEN_TYPE:
            return self.refuse("invalid_client", 401)
        actor = jwt.decode(form["actor_token"], options={"verify_signature": False})
        refusal = self.check_dpop(request, actor["cnf"]["jkt"], access_token=None)
        if refusal is not None:
            return refusal
        root = self.agent_token(
            self.issuer + "/mcp", sub="user:usr_person", cnf=actor["cnf"], act={"sub": actor["sub"]}, scope=""
        )
        details = [
            {
                "type": "authfi_mandate",
                "mandate_id": MANDATE,
                "template": "support.refund",
                "tools": ["orders.read"],
                "budgets": {"usd_micros": 100},
                "expires_at": "2026-12-01T00:00:00Z",
            }
        ]
        return json_response(
            {"access_token": root, "token_type": "DPoP", "expires_in": 300, "authorization_details": details}
        )


@pytest.fixture(autouse=True)
def fresh_nonces(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_dpop, "nonces", _dpop.NonceCache())


@pytest.fixture
def platform() -> TaskPlatform:
    return TaskPlatform()


@pytest.fixture
def token_file(tmp_path: Path, platform: TaskPlatform) -> Path:
    path = tmp_path / "token"
    path.write_text(platform.k8s_token())
    return path


def env_for(platform: TaskPlatform, token_file: Path) -> dict[str, str]:
    from tests.instance_platform import CONFIG
    from tests.unit.test_instance_ciba import env_for as base

    env = base(platform, token_file)
    env.setdefault("AUTHFI_CONFIG_VERSION", str(CONFIG["version"]))
    return env


def started_task(platform: TaskPlatform, token_file: Path) -> tuple[InstanceAuth, Task]:
    agent = authfi.connect(env_for(platform, token_file), http_client=platform.client(), clock=FakeClock())
    assert isinstance(agent, InstanceAuth)
    task = agent.request_task("support.refund", subject_token=Secret("person-token"), params={"order": "4411"})
    assert isinstance(task, Task)
    return agent, task


def test_an_instance_starts_a_task_with_its_own_token_as_the_actor(platform: TaskPlatform, token_file: Path) -> None:
    agent, task = started_task(platform, token_file)
    form = platform.starts[0]
    assert form["actor_token"] == agent.token()
    assert form["actor_token_type"] == ACCESS_TOKEN_TYPE
    assert "client_assertion" not in form and "client_id" not in form
    assert json.loads(form["authorization_details"])[0]["template"] == "support.refund"
    assert task.mandate_id == MANDATE
    agent.project.close()


def test_the_task_is_held_by_the_instances_own_key(platform: TaskPlatform, token_file: Path) -> None:
    agent, task = started_task(platform, token_file)
    assert task.bound_key().thumbprint == _dpop.run_key().thumbprint
    agent.project.close()


def test_a_start_whose_proof_is_by_another_key_is_refused(platform: TaskPlatform, token_file: Path) -> None:
    agent = authfi.connect(env_for(platform, token_file), http_client=platform.client(), clock=FakeClock())
    assert isinstance(agent, InstanceAuth)
    agent._task_key = lambda metadata: _dpop.DPoPKey()  # type: ignore[method-assign]  # pyright: ignore
    with pytest.raises(authfi.AuthFIError):
        agent.request_task("support.refund", subject_token=Secret("person-token"))
    agent.project.close()


def test_an_agent_that_signs_its_own_assertion_keeps_the_assertion_type() -> None:
    """The control: a key agent's start is not changed, only an instance's."""
    from authfi._protocol.agent import TASK_ACTOR_ASSERTION
    from authfi.agents.agent import AgentAuth, AsyncAgentAuth

    assert AgentAuth._task_actor_type == TASK_ACTOR_ASSERTION  # pyright: ignore[reportPrivateUsage]
    assert AsyncAgentAuth._task_actor_type == TASK_ACTOR_ASSERTION  # pyright: ignore[reportPrivateUsage]
    assert TASK_ACTOR_ASSERTION != ACCESS_TOKEN_TYPE


async def test_an_async_instance_starts_a_task_the_same_way(platform: TaskPlatform, token_file: Path) -> None:
    agent = authfi.connect_async(env_for(platform, token_file), http_client=platform.async_client(), clock=FakeClock())
    assert isinstance(agent, AsyncInstanceAuth)
    task = await agent.request_task("support.refund", subject_token=Secret("person-token"))
    assert isinstance(task, AsyncTask)
    form = platform.starts[0]
    assert form["actor_token_type"] == ACCESS_TOKEN_TYPE
    assert form["actor_token"] == await agent.token()
    await agent.project.aclose()
