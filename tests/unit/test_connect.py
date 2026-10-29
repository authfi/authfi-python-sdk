from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import authfi
from authfi import (
    AgentAuth,
    AgentKey,
    AsyncAgentAuth,
    AsyncAuthFI,
    AsyncClientCredentials,
    AsyncGateway,
    AuthFI,
    ClientCredentials,
    ConfigurationError,
    Gateway,
)
from authfi._adapters import async_gateway_for, gateway_for
from authfi._frameworks import new_run_id, valid_run_id
from tests.agent_platform import AGENT, AgentPlatform
from tests.support import FakeClock, Platform, json_response


@pytest.fixture
def key_file(tmp_path: Path) -> Path:
    path = tmp_path / "agent.key"
    AgentKey.generate(kid="k1").write(str(path))
    return path


@pytest.fixture
def platform(key_file: Path) -> AgentPlatform:
    return AgentPlatform(agent_jwk=AgentKey.from_file(str(key_file), kid="k1").public_jwk())


def agent_env(platform: Platform, key_file: Path) -> dict[str, str]:
    return {
        "AUTHFI_ISSUER": platform.issuer,
        "AUTHFI_AGENT_ID": AGENT,
        "AUTHFI_AGENT_KEY_FILE": str(key_file),
        "AUTHFI_AGENT_KID": "k1",
    }


def client_env(platform: Platform) -> dict[str, str]:
    return {"AUTHFI_ISSUER": platform.issuer, "AUTHFI_CLIENT_ID": "app_batch", "AUTHFI_CLIENT_SECRET": "s3cret"}


# --- which credential the environment names --------------------------------------------------------


def test_the_agent_key_in_the_environment_is_an_agent(platform: Platform, key_file: Path) -> None:
    af = authfi.connect(agent_env(platform, key_file), http_client=platform.client())
    assert isinstance(af, AgentAuth)
    assert af.agent_id == AGENT
    assert af.project.issuer == platform.issuer
    af.project.close()


def test_the_client_id_and_secret_are_a_machine_client_for_the_gateway(platform: Platform) -> None:
    af = authfi.connect(client_env(platform), http_client=platform.client())
    assert isinstance(af, ClientCredentials)
    assert af.client_id == "app_batch"
    assert af.resource == platform.issuer + "/mcp"
    assert "s3cret" not in repr(af)
    af.project.close()


def test_a_resource_in_the_environment_or_the_call_names_what_the_client_is_for(platform: Platform) -> None:
    env = client_env(platform) | {"AUTHFI_CLIENT_RESOURCE": platform.issuer + "/mcp/other"}
    named = authfi.connect(env, http_client=platform.client())
    assert isinstance(named, ClientCredentials)
    assert named.resource == platform.issuer + "/mcp/other"
    argued = authfi.connect(env, resource=platform.issuer + "/mcp", http_client=platform.client())
    assert isinstance(argued, ClientCredentials)
    assert argued.resource == platform.issuer + "/mcp"


def test_an_agent_wins_when_both_are_named(platform: Platform, key_file: Path) -> None:
    env = agent_env(platform, key_file) | {"AUTHFI_CLIENT_ID": "app_batch", "AUTHFI_CLIENT_SECRET": "s"}
    assert isinstance(authfi.connect(env, http_client=platform.client()), AgentAuth)


async def test_the_asyncio_twin_builds_the_asyncio_credentials(platform: Platform, key_file: Path) -> None:
    agent = authfi.connect_async(agent_env(platform, key_file), http_client=platform.async_client())
    assert isinstance(agent, AsyncAgentAuth)
    machine = authfi.connect_async(client_env(platform), http_client=platform.async_client())
    assert isinstance(machine, AsyncClientCredentials)
    await agent.project.aclose()
    await machine.project.aclose()


# --- what is missing is named ----------------------------------------------------------------------


def test_no_issuer_names_the_variable(key_file: Path) -> None:
    with pytest.raises(ConfigurationError, match="AUTHFI_ISSUER"):
        authfi.connect({"AUTHFI_AGENT_ID": AGENT})
    with pytest.raises(ConfigurationError, match="AUTHFI_ISSUER"):
        authfi.connect_async({"AUTHFI_AGENT_ID": AGENT})


def test_no_credential_names_both_ways_to_have_one(platform: Platform) -> None:
    for build in (authfi.connect, authfi.connect_async):
        with pytest.raises(ConfigurationError) as caught:
            build({"AUTHFI_ISSUER": platform.issuer})
        assert "AUTHFI_AGENT_ID" in caught.value.message
        assert "AUTHFI_CLIENT_ID" in caught.value.message


def test_an_agent_without_its_key_names_the_key_variable(platform: Platform) -> None:
    env = {"AUTHFI_ISSUER": platform.issuer, "AUTHFI_AGENT_ID": AGENT}
    with pytest.raises(ConfigurationError, match="AUTHFI_AGENT_KEY_FILE"):
        authfi.connect(env)


def test_a_client_without_its_secret_names_the_secret_variable(platform: Platform) -> None:
    env = {"AUTHFI_ISSUER": platform.issuer, "AUTHFI_CLIENT_ID": "app_batch"}
    with pytest.raises(ConfigurationError, match="AUTHFI_CLIENT_SECRET"):
        authfi.connect(env)
    with pytest.raises(ConfigurationError, match="AUTHFI_CLIENT_SECRET"):
        authfi.connect_async(env)


def test_a_secret_without_its_client_id_is_not_a_credential(platform: Platform) -> None:
    env = {"AUTHFI_ISSUER": platform.issuer, "AUTHFI_CLIENT_SECRET": "s3cret"}
    with pytest.raises(ConfigurationError, match="AUTHFI_CLIENT_ID"):
        authfi.connect(env)


def test_the_environment_is_not_read_when_one_is_given(
    platform: Platform, key_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, value in agent_env(platform, key_file).items():
        monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match="AUTHFI_ISSUER"):
        authfi.connect({})
    assert isinstance(authfi.connect(http_client=platform.client()), AgentAuth)


# --- the run ---------------------------------------------------------------------------------------


def test_a_run_has_an_id_that_stays_and_can_be_chosen(platform: Platform) -> None:
    af = authfi.connect(client_env(platform), http_client=platform.client(), run_id="nightly-2026-10-29")
    assert af.run_id == "nightly-2026-10-29"
    assert af.run_id == "nightly-2026-10-29"
    other = authfi.connect(client_env(platform), http_client=platform.client())
    assert valid_run_id(other.run_id)
    assert other.run_id.startswith("run-")
    assert other.run_id != af.run_id


def test_a_run_id_that_the_gateway_would_not_keep_is_refused(platform: Platform) -> None:
    for bad in ("", "has space", "x" * 97, "run/1", "ünï"):
        with pytest.raises(ConfigurationError, match="run id"):
            authfi.connect(client_env(platform), http_client=platform.client(), run_id=bad)
        with pytest.raises(ConfigurationError, match="run id"):
            authfi.connect_async(client_env(platform), http_client=platform.async_client(), run_id=bad)
    assert not valid_run_id(new_run_id() + " ")


def test_starting_a_run_restarts_the_count(platform: Platform) -> None:
    af = authfi.connect(client_env(platform), http_client=platform.client(), run_id="r1")
    assert [af.next_request_id(), af.next_request_id()] == ["r1.1", "r1.2"]
    assert af.start_run("r2") == "r2"
    assert af.next_request_id() == "r2.1"
    assert valid_run_id(af.start_run())


# --- a machine client on the gateways --------------------------------------------------------------


class Gateways:
    """The gateway end of a call: records what the credential sent."""

    def __init__(self, platform: Platform) -> None:
        self.platform = platform
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp":
            self.seen.append(request)
            return json_response({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
        return self.platform(request)


def mint(platform: Platform) -> None:
    platform.route(
        "POST",
        "/oauth/token",
        lambda _r: json_response({"access_token": "m2m-token", "token_type": "Bearer", "expires_in": 900}),
    )


def test_a_machine_client_has_one_gateway_and_every_adapter_shares_it(platform: Platform, clock: FakeClock) -> None:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        m2m = project.client_credentials("app_batch", authfi.Secret("s"), resource=platform.issuer + "/mcp")
        gateway = m2m.gateway()
        assert isinstance(gateway, Gateway)
        assert m2m.gateway() is gateway
        assert gateway_for(m2m) is gateway
        assert gateway.source is m2m


async def test_the_asyncio_machine_client_has_one_asyncio_gateway(platform: Platform, clock: FakeClock) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as project:
        m2m = project.client_credentials("app_batch", authfi.Secret("s"), resource=platform.issuer + "/mcp")
        gateway = m2m.gateway()
        assert isinstance(gateway, AsyncGateway)
        assert async_gateway_for(m2m) is gateway
        assert gateway_for(m2m) is gateway


def test_a_machine_clients_calls_say_so_and_carry_the_run(platform: Platform, clock: FakeClock) -> None:
    mint(platform)
    server = Gateways(platform)
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        m2m = project.client_credentials("app_batch", authfi.Secret("s"), resource=platform.issuer + "/mcp")
        m2m.start_run("batch-7")
        gateway = m2m.gateway()
        client = gateway.http_client()
        client._transport = httpx.MockTransport(server)
        client.post(gateway.mcp_url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        client.post(gateway.mcp_url, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    first, second = server.seen
    assert first.headers["authorization"] == "Bearer m2m-token"
    assert first.headers["x-request-id"] == "batch-7.1"
    assert second.headers["x-request-id"] == "batch-7.2"
    assert first.headers["user-agent"].startswith("authfi-python/")
    assert first.headers["user-agent"].endswith("authfi-credential/client-credentials")
    assert "traceparent" in first.headers


def test_an_agents_calls_carry_the_run_and_no_client_label(platform: AgentPlatform, key_file: Path) -> None:
    server = Gateways(platform)
    af = authfi.connect(agent_env(platform, key_file), http_client=platform.client(), run_id="agent-run")
    assert isinstance(af, AgentAuth)
    gateway = af.gateway()
    client = gateway.http_client()
    client._transport = httpx.MockTransport(server)
    client.post(gateway.mcp_url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    af.project.close()
    sent = server.seen[0]
    assert sent.headers["x-request-id"] == "agent-run.1"
    assert "client-credentials" not in sent.headers["user-agent"]


def test_a_caller_chosen_request_id_is_left_alone(platform: Platform, clock: FakeClock) -> None:
    mint(platform)
    server = Gateways(platform)
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        m2m = project.client_credentials("app_batch", authfi.Secret("s"), resource=platform.issuer + "/mcp")
        gateway = m2m.gateway()
        client = gateway.http_client()
        client._transport = httpx.MockTransport(server)
        client.post(gateway.mcp_url, json={}, headers={"X-Request-Id": "mine-1"})
    assert server.seen[0].headers["x-request-id"] == "mine-1"


def test_a_credential_without_a_run_still_gets_a_request_id(platform: Platform, clock: FakeClock) -> None:
    class Bare(authfi.TokenSource):
        @property
        def send_to(self) -> tuple[str, ...]:
            return (platform.issuer,)

        def token(self, *, force: bool = False) -> str:
            return "bare"

    server = Gateways(platform)
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        gateway = Gateway(project, Bare())
        client = gateway.http_client()
        client._transport = httpx.MockTransport(server)
        client.post(gateway.mcp_url, json={})
    assert server.seen[0].headers["x-request-id"].startswith("afr-")
    assert "client-credentials" not in server.seen[0].headers["user-agent"]


# --- frameworks ------------------------------------------------------------------------------------


def test_a_framework_that_is_not_installed_names_its_extra(platform: Platform, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real = builtins.__import__

    def refuse(name: str, *args: object, **kwargs: object) -> object:
        if name == "langchain_core":
            raise ImportError("No module named 'langchain_core'")
        return real(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.delitem(__import__("sys").modules, "authfi.langchain", raising=False)
    monkeypatch.delitem(__import__("sys").modules, "langchain_core", raising=False)
    monkeypatch.setattr(builtins, "__import__", refuse)
    af = authfi.connect(client_env(platform), http_client=platform.client())
    with pytest.raises(ConfigurationError, match=r"authfi\[langchain\]"):
        af.langchain()


def test_the_factories_build_the_integration_on_the_same_credential(platform: Platform) -> None:
    for framework in ("langchain_anthropic", "langgraph", "agents", "claude_agent_sdk", "google.adk"):
        pytest.importorskip(framework)
    af = authfi.connect(client_env(platform), http_client=platform.client(), run_id="fw-run")
    langchain = af.langchain()
    assert langchain.source is af
    assert langchain.gateway is af.gateway()
    assert langchain.run_id == "fw-run"
    with pytest.raises(ConfigurationError, match="asyncio"):
        af.adk()
    with pytest.raises(ConfigurationError, match="asyncio"):
        af.openai_agents()
    with pytest.raises(ConfigurationError, match="asyncio"):
        af.langgraph()
    with pytest.raises(ConfigurationError, match=r"AsyncAuthFI\.agent"):
        af.claude_agent()
