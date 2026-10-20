"""The agent module's refusals and less travelled paths, each with the outcome it must have."""

from __future__ import annotations

import pickle
import time
from collections.abc import Iterator

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from authfi import (
    AgentKey,
    APIError,
    AsyncAuthFI,
    AsyncTokenSource,
    AuthFI,
    AuthFIDeprecationWarning,
    ConfigurationError,
    ConsentExpired,
    DiscoveryError,
    NotSupportedError,
    OAuthError,
    Secret,
    WaitTimeout,
    assert_public_jwk,
)
from authfi._protocol import agent as wire
from authfi._protocol.http import HttpRequest, HttpResponse
from authfi._protocol.oauth import ClientAuth
from authfi.agents import AgentAuth
from authfi.agents.agent import Approval, _capabilities, _legacy_ciba_request
from authfi.agents.keys import _alg_of
from tests.agent_platform import AGENT, AgentPlatform
from tests.support import FakeClock, json_response

ISSUER = "https://acme.authfi.io"


def answer(status: int, body: bytes) -> HttpResponse:
    return HttpResponse(status, (), body, ISSUER + "/x", "afr-1")


# --- the wire --------------------------------------------------------------------------------------


def test_agent_configuration_refusals() -> None:
    with pytest.raises(DiscoveryError):
        wire.parse_agent_configuration(ISSUER, [])
    with pytest.raises(DiscoveryError) as caught:
        wire.parse_agent_configuration(ISSUER, {"issuer": "https://other"})
    assert caught.value.code == "issuer_mismatch"
    assert wire.parse_agent_configuration(ISSUER, {"endpoints": "x"}) == wire.AgentEndpoints(None, None, None)


def test_ciba_request_and_answer_edges() -> None:
    auth = ClientAuth.none("agt")
    with pytest.raises(ConfigurationError, match="one string"):
        wire.ciba_start_request(
            "https://x", auth, "a", login_hint="p", binding_message="m", scope="a b", expires_in=None
        )
    with pytest.raises(ConfigurationError, match="positive"):
        wire.ciba_start_request("https://x", auth, "a", login_hint="p", binding_message="m", scope=None, expires_in=0)
    with pytest.raises(OAuthError) as refused:
        wire.parse_ciba_start(answer(401, b'{"error":"invalid_client"}'))
    assert refused.value.code == "invalid_client"
    with pytest.raises(OAuthError, match="auth_req_id"):
        wire.parse_ciba_start(answer(200, b"{}"))
    start = wire.parse_ciba_start(answer(200, b'{"auth_req_id":"r"}'))
    assert (start.expires_in, start.interval) == (300, 5.0)
    assert repr(start) == "CibaStart(expires_in=300, interval=5.0)"


def test_ciba_poll_names_the_resource() -> None:
    flow = wire.ciba_poll_flow(
        ISSUER + "/oauth/token",
        ClientAuth.none("agt"),
        auth_req_id=Secret("r"),
        interval=1,
        wait_deadline=100,
        request_deadline=100,
        now=lambda: 0,
        resource="api_orders",
    )
    next(flow)  # the wait
    request = flow.send(None)
    assert isinstance(request, HttpRequest)
    assert request.form is not None
    assert request.form["resource"] == "api_orders"


def test_api_answers_without_data_or_with_a_refusal() -> None:
    with pytest.raises(APIError) as refused:
        wire.parse_transaction_token(answer(403, b'{"error":"EAGENT010"}'))
    assert refused.value.code == "EAGENT010"
    with pytest.raises(APIError, match="no data"):
        wire.parse_transaction_token(answer(200, b"{}"))
    with pytest.raises(APIError, match="no token"):
        wire.parse_transaction_token(answer(200, b'{"data":{}}'))
    with pytest.raises(APIError, match="decision"):
        wire.parse_action_decision(answer(200, b'{"data":{"decision":"maybe"}}'))
    with pytest.raises(APIError, match="approval_id"):
        wire.parse_approval(answer(200, b'{"data":{}}'), reading=True)


def test_request_builders_refuse_empty_values() -> None:
    with pytest.raises(ConfigurationError):
        wire.transaction_token_request(ISSUER, "t", action=" ", resource=None, context=None)
    with pytest.raises(ConfigurationError):
        wire.approval_request(ISSUER, agent_id="a", sign=str, audience=ISSUER, action="", context=None, expires_in=10)
    with pytest.raises(ConfigurationError):
        wire.approval_request(ISSUER, agent_id="a", sign=str, audience=ISSUER, action="act", context=None, expires_in=0)
    with pytest.raises(DiscoveryError):
        wire.read_approval_request(ISSUER + "/approvals", "a", "t")
    with pytest.raises(ConfigurationError):
        wire.read_approval_request(ISSUER + "/approvals/{id}", " ", "t")
    assert wire.label_claims("not-a-token") == {}


def test_approvals_with_odd_timestamps_and_capabilities_from_scope() -> None:
    record = wire.ApprovalRecord("a", "approved", "x", "not a time", None, "2026-10-20T10:00:00Z")
    approval = Approval.of(record)
    assert approval.created_at is None
    assert approval.resolved_at is not None
    assert approval.decided
    from authfi._protocol.oauth import TokenResponse

    assert _capabilities(TokenResponse(Secret("t"), "Bearer", 60, scope=frozenset({"a"}))) == frozenset({"a"})


def test_old_ciba_request_shapes() -> None:
    assert _legacy_ciba_request("req", 0).auth_req_id == Secret("req")
    with pytest.raises(ConfigurationError):
        _legacy_ciba_request({}, 0)


# --- keys ------------------------------------------------------------------------------------------


def test_key_refusals() -> None:
    weak_exponent = rsa.generate_private_key(public_exponent=3, key_size=2048)
    with pytest.raises(ConfigurationError, match="65537"):
        AgentKey(weak_exponent, "k")
    with pytest.raises(ConfigurationError, match="JSON object"):
        assert_public_jwk([])  # type: ignore[type-var]
    with pytest.raises(ConfigurationError):
        _alg_of("a string")
    with pytest.raises(ConfigurationError, match="JSON object"):
        AgentKey.from_jwk("[1]")
    with pytest.raises(ConfigurationError, match="not a usable"):
        AgentKey.from_jwk({"kty": "EC", "crv": "P-256", "x": "AA", "y": "AA", "d": "AA"})
    assert AgentKey(ec.generate_private_key(ec.SECP256R1()), "k").alg == "ES256"


# --- the agent client ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def platform(key: AgentKey) -> AgentPlatform:
    return AgentPlatform(agent_jwk=key.public_jwk())


@pytest.fixture
def agent(platform: AgentPlatform, key: AgentKey, clock: FakeClock) -> Iterator[AgentAuth]:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        yield af.agent(AGENT, key)


def test_what_the_project_does_not_offer(agent: AgentAuth, platform: AgentPlatform) -> None:
    del platform.discovery["backchannel_authentication_endpoint"]
    with pytest.raises(NotSupportedError) as ciba:
        agent.request_user_token("p", binding_message="m")
    assert ciba.value.feature == "ciba"
    platform.discovery["grant_types_supported"] = ["client_credentials"]
    agent.project.metadata(refresh=True)
    with pytest.raises(NotSupportedError) as exchange:
        agent.exchange(Secret("t"))
    assert exchange.value.feature == "token_exchange"


def test_resource_views_and_delegated_details(agent: AgentAuth, platform: AgentPlatform) -> None:
    view = agent.for_resource("https://acme.authfi.io/mcp")
    assert repr(view) == "AgentAuth.for_resource('https://acme.authfi.io/mcp')"
    with pytest.raises(ConfigurationError):
        agent.for_resource(" ")
    delegated = agent.token_as_user("p", binding_message="m")
    assert delegated.scopes == frozenset({"send_email"})
    assert delegated.expires_at.timestamp() > time.time()


def test_a_broken_agent_configuration(agent: AgentAuth, platform: AgentPlatform) -> None:
    platform.route("GET", "/.well-known/agent-configuration", httpx.Response(500))
    with pytest.raises(DiscoveryError):
        agent.request_approval("x")


def test_approval_status_without_an_endpoint(agent: AgentAuth, platform: AgentPlatform) -> None:
    del platform.agent_configuration["endpoints"]["approval"]
    with pytest.raises(NotSupportedError):
        agent.approval("apr-1")


def test_the_0_1_agent_settings(platform: AgentPlatform, key: AgentKey, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.warns(AuthFIDeprecationWarning):
        legacy = AgentAuth("agt_1", key, platform.issuer, None, None, None, None, 30)  # pyright: ignore[reportDeprecated]
    assert legacy.refresh_skew == 30.0
    with pytest.warns(AuthFIDeprecationWarning):
        assert legacy.audience is None  # pyright: ignore[reportDeprecated]
    with pytest.warns(AuthFIDeprecationWarning):
        assert legacy.platform_url == platform.issuer  # pyright: ignore[reportDeprecated]
    with pytest.raises(ConfigurationError, match="send_to"), pytest.warns(AuthFIDeprecationWarning):
        AgentAuth(agent_id="agt_1", key=key, platform_url=platform.issuer, audience="api_orders")  # pyright: ignore[reportDeprecated]


def test_0_1_environment_names_are_read_with_a_warning(
    key: AgentKey, tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pathlib import Path

    path = Path(str(tmp_path)) / "k.pem"
    key.write(path)
    for name in ("AUTHFI_ISSUER", "AUTHFI_AGENT_RESOURCE", "AUTHFI_AGENT_PRIVATE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AUTHFI_AGENT_ID", "agt_1")
    monkeypatch.setenv("AUTHFI_AGENT_KEY_FILE", str(path))
    monkeypatch.setenv("AUTHFI_AGENT_KID", "k1")
    monkeypatch.setenv("AUTHFI_PLATFORM_URL", "https://acme.authfi.io")
    monkeypatch.setenv("AUTHFI_AGENT_AUDIENCE", "https://acme.authfi.io/mcp")
    with pytest.warns(AuthFIDeprecationWarning) as caught:
        agent = AgentAuth.from_env()  # pyright: ignore[reportDeprecated]
    messages = " ".join(str(w.message) for w in caught)
    assert "AUTHFI_PLATFORM_URL is AUTHFI_ISSUER" in messages
    assert "AUTHFI_AGENT_AUDIENCE is AUTHFI_AGENT_RESOURCE" in messages
    assert agent.project.issuer == "https://acme.authfi.io"
    assert agent.resource == "https://acme.authfi.io/mcp"
    monkeypatch.delenv("AUTHFI_PLATFORM_URL")
    with pytest.raises(ConfigurationError, match="AUTHFI_ISSUER"), pytest.warns(AuthFIDeprecationWarning):
        AgentAuth.from_env()  # pyright: ignore[reportDeprecated]


def test_backchannel_endpoint_of_0_1(agent: AgentAuth) -> None:
    with pytest.warns(AuthFIDeprecationWarning):
        assert agent.backchannel_endpoint is not None  # pyright: ignore[reportDeprecated]


async def test_async_edges(platform: AgentPlatform, key: AgentKey, clock: FakeClock) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as aaf:
        agent = aaf.agent(AGENT, key)
        view = agent.for_resource("https://acme.authfi.io/mcp")
        assert view.send_to == (platform.issuer,)
        assert repr(view) == "AsyncAgentAuth.for_resource('https://acme.authfi.io/mcp')"
        with pytest.raises(ConfigurationError):
            agent.for_resource("")

        class Session(AsyncTokenSource):
            @property
            def send_to(self) -> tuple[str, ...]:
                return (platform.issuer,)

            async def token(self, *, force: bool = False) -> str:
                return platform.access_token(sub="usr_7")

        exchanged = await agent.exchange(Session())
        assert exchanged.scopes == frozenset()
        assert exchanged.expires_at.timestamp() > 0
        clock.advance(300)
        assert await exchanged.token()
        assert len(platform.exchanges) == 2
        assert "usr_7" in repr(exchanged)
        with pytest.raises(TypeError):
            pickle.dumps(exchanged)
        expired = Secret(platform.access_token(sub="usr_7", exp=int(time.time()) - 10))
        with pytest.raises(ConsentExpired):
            await agent.exchange(expired)
        with pytest.raises(ConfigurationError):
            await agent.exchange("raw")  # type: ignore[arg-type]
        platform.approval_status = "pending"
        with pytest.raises(WaitTimeout) as timed_out:
            await agent.wait_for_approval("apr-1", timeout=1)
        assert timed_out.value.pending_id == "apr-1"
        del platform.agent_configuration["endpoints"]["approval"]
        del platform.agent_configuration["endpoints"]["request_approval"]
        agent2 = aaf.agent(AGENT, key)
        with pytest.raises(NotSupportedError):
            await agent2.request_approval("x")
        with pytest.raises(NotSupportedError):
            await agent2.approval("apr-1")
        platform.route("GET", "/.well-known/agent-configuration", httpx.Response(404))
        agent3 = aaf.agent(AGENT, key)
        with pytest.raises(NotSupportedError):
            await agent3.request_approval("x")
        platform.route("GET", "/.well-known/agent-configuration", json_response({}, status=500))
        agent4 = aaf.agent(AGENT, key)
        with pytest.raises(DiscoveryError):
            await agent4.approval("x")
