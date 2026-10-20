"""The agent's rules that a revert once left unnoticed: where its token goes, what a retry may repeat."""

from __future__ import annotations

import copy
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
from typing import Any
from urllib.parse import urlsplit

import httpx
import jwt
import pytest

from authfi import (
    AgentKey,
    AsyncAuthFI,
    AuthFI,
    AuthFIDeprecationWarning,
    ConfigurationError,
    ConsentExpired,
    Secret,
    TransportError,
)
from authfi.agents import AgentAuth, AsyncAgentAuth, AsyncDelegatedToken, DelegatedToken
from tests.agent_platform import AGENT, AgentPlatform
from tests.support import FakeClock, form_of

API = "https://api.example.com"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate("ES256", kid="k1")


@pytest.fixture
def platform(key: AgentKey) -> AgentPlatform:
    return AgentPlatform(agent_jwk=key.public_jwk())


@pytest.fixture
def af(platform: AgentPlatform, clock: FakeClock) -> Iterator[AuthFI]:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        yield project


@pytest.fixture
async def aaf(platform: AgentPlatform, clock: FakeClock) -> AsyncIterator[AsyncAuthFI]:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as project:
        yield project


def audience_of(raw: str) -> Any:
    return jwt.decode(raw, options={"verify_signature": False})["aud"]


# --- the agent's own token goes to AuthFI, a customer API's token goes nowhere of AuthFI's --------------


def test_an_agent_cannot_hold_a_customer_apis_token_by_default(af: AuthFI, aaf: AsyncAuthFI, key: AgentKey) -> None:
    with pytest.raises(ConfigurationError, match=r"for_resource\('api_orders', send_to"):
        af.agent(AGENT, key, resource="api_orders")
    with pytest.raises(ConfigurationError, match="not AuthFI's own"):
        aaf.agent(AGENT, key, resource=API)
    with pytest.raises(ConfigurationError, match="not AuthFI's own"):
        AgentAuth(af, AGENT, key, resource="api_orders")
    # send_to belongs to for_resource: the agent's own token goes only to AuthFI
    with pytest.raises(ConfigurationError, match="for_resource"):
        AgentAuth(af, AGENT, key, send_to=[API])  # type: ignore[call-overload]  # pyright: ignore[reportCallIssue]
    with pytest.raises(TypeError):
        af.agent(AGENT, key, send_to=[API])  # type: ignore[call-arg]  # pyright: ignore[reportCallIssue]
    assert af.agent(AGENT, key, resource=af.issuer + "/mcp").resource == af.issuer + "/mcp"


def test_authfi_endpoints_get_the_authfi_token_even_after_a_customer_token_was_minted(
    af: AuthFI, key: AgentKey, platform: AgentPlatform
) -> None:
    agent = af.agent(AGENT, key)
    orders = agent.for_resource("api_orders", send_to=[API])
    assert audience_of(orders.token()) == "api_orders"

    agent.transaction_token("refund")
    agent.approval("apr-1")
    agent.exchange(Secret(platform.access_token(sub="usr_7")))

    assert platform.bearer_audiences == [af.issuer + "/mcp", af.issuer + "/mcp"]
    assert platform.actor_audiences == [af.issuer + "/mcp"]


def test_a_0_1_agent_made_for_a_customer_url_still_sends_authfi_its_own_token(
    af: AuthFI, key: AgentKey, platform: AgentPlatform
) -> None:
    with pytest.warns(AuthFIDeprecationWarning):
        legacy = AgentAuth(AGENT, key, platform.issuer, API)  # pyright: ignore[reportDeprecated]
    legacy.project._transport._client = platform.client()  # pyright: ignore[reportPrivateUsage]
    assert audience_of(legacy.token()) == API
    legacy.transaction_token("refund")
    legacy.approval("apr-1")
    legacy.exchange(Secret(platform.access_token(sub="usr_7")), send_to=[API])
    assert platform.bearer_audiences == [platform.issuer + "/mcp", platform.issuer + "/mcp"]
    assert platform.actor_audiences == [platform.issuer + "/mcp"]


async def test_the_async_agent_sends_authfi_its_own_token_too(
    aaf: AsyncAuthFI, key: AgentKey, platform: AgentPlatform
) -> None:
    agent = aaf.agent(AGENT, key)
    assert audience_of(await agent.for_resource("api_orders", send_to=[API]).token()) == "api_orders"
    await agent.transaction_token("refund")
    await agent.approval("apr-1")
    await agent.exchange(Secret(platform.access_token(sub="usr_7")))
    assert platform.bearer_audiences == [aaf.issuer + "/mcp", aaf.issuer + "/mcp"]
    assert platform.actor_audiences == [aaf.issuer + "/mcp"]


# --- an approval request signs a new assertion every time it is sent ----------------------------------


def test_a_retried_approval_request_carries_a_new_assertion(af: AuthFI, key: AgentKey, platform: AgentPlatform) -> None:
    agent = af.agent(AGENT, key)
    real = platform.routes[("POST", "/apis/regional/v1/ai/auth/request-approval")]
    attempts: list[dict[str, Any]] = []

    def throttled_once(request: httpx.Request) -> httpx.Response:
        import json

        attempts.append(json.loads(request.content))
        if len(attempts) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"}, json={"error": "slow_down"})
        return real(request)

    platform.route("POST", "/apis/regional/v1/ai/auth/request-approval", throttled_once)
    approval = agent.request_approval("refund order 4411")
    assert approval.approval_id == "apr-1"
    assert len(attempts) == 2
    first, second = (a["client_assertion"] for a in attempts)
    assert first != second
    assert (
        jwt.decode(first, options={"verify_signature": False})["jti"]
        != jwt.decode(second, options={"verify_signature": False})["jti"]
    )
    assert all(a["agent_id"] == AGENT and a["requesting_action"] == "refund order 4411" for a in attempts)


# --- each request that may create state is sent once -----------------------------------------------


def _reset(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadError("connection reset after the request was sent", request=request)


def _unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"error": "temporarily_unavailable"})


def _ciba(agent: AgentAuth) -> object:
    return agent.request_user_token("priya@acme.com", binding_message="Refund")


def _exchange(agent: AgentAuth) -> object:
    return agent.exchange(Secret(jwt.encode({"sub": "usr_7", "exp": 9_999_999_999}, "k" * 32, algorithm="HS256")))


def _transaction(agent: AgentAuth) -> object:
    return agent.transaction_token("refund")


def _authorize(agent: AgentAuth) -> object:
    token = agent.transaction_token("refund")
    return agent.authorize_action(token)


def _approval(agent: AgentAuth) -> object:
    return agent.request_approval("refund")


# (name, path, call): the paths of requests that may create state
STATEFUL = [
    ("ciba start", "/apis/regional/v1/ai/auth/backchannel-authorize", _ciba),
    ("token exchange", "/oauth/token", _exchange),
    ("transaction token", "/apis/regional/v1/ai/auth/transaction-token", _transaction),
    ("authorize", "/apis/regional/v1/ai/auth/authorize", _authorize),
    ("approval request", "/apis/regional/v1/ai/auth/request-approval", _approval),
]


@pytest.mark.parametrize(("name", "path", "call"), STATEFUL, ids=[s[0] for s in STATEFUL])
def test_a_reset_after_the_request_was_sent_is_never_retried(
    name: str, path: str, call: Callable[[AgentAuth], object], af: AuthFI, key: AgentKey, platform: AgentPlatform
) -> None:
    agent = af.agent(AGENT, key)
    agent.token()  # the agent's own token is minted before the call under test
    if name == "token exchange":
        platform.route("POST", path, _exchange_reset(platform))
    else:
        platform.route("POST", path, _reset)
    before = len(platform.calls("POST", path))
    with pytest.raises(TransportError) as caught:
        call(agent)
    assert not caught.value.retryable
    assert len(platform.calls("POST", path)) - before == 1


def _exchange_reset(platform: AgentPlatform) -> Callable[[httpx.Request], httpx.Response]:
    """The token endpoint is shared with the agent's own mint: reset only the exchange."""
    real = platform.routes[("POST", "/oauth/token")]

    def handler(request: httpx.Request) -> httpx.Response:
        if form_of(request).get("grant_type") == "urn:ietf:params:oauth:grant-type:token-exchange":
            return _reset(request)
        return real(request)

    return handler


@pytest.mark.parametrize(("name", "path", "call"), STATEFUL, ids=[s[0] for s in STATEFUL])
def test_a_503_without_retry_after_is_not_retried(
    name: str, path: str, call: Callable[[AgentAuth], object], af: AuthFI, key: AgentKey, platform: AgentPlatform
) -> None:
    agent = af.agent(AGENT, key)
    agent.token()
    real = platform.routes[("POST", path)]

    def only_the_call_under_test(request: httpx.Request) -> httpx.Response:
        if path == "/oauth/token" and form_of(request).get("grant_type") != (
            "urn:ietf:params:oauth:grant-type:token-exchange"
        ):
            return real(request)
        return _unavailable(request)

    platform.route("POST", path, only_the_call_under_test)
    before = len(platform.calls("POST", path))
    with pytest.raises(Exception) as caught:
        call(agent)
    assert not isinstance(caught.value, TransportError) or not caught.value.retryable
    assert len(platform.calls("POST", path)) - before == 1


# --- an approval id is data, never a piece of the URL ---------------------------------------------------


def test_an_approval_id_cannot_rewrite_the_path_or_query(af: AuthFI, key: AgentKey, platform: AgentPlatform) -> None:
    agent = af.agent(AGENT, key)
    agent.approval("../x?y")
    read = [r for r in platform.requests if r.method == "GET" and "/approvals/" in r.url.raw_path.decode()][-1]
    assert read.url.raw_path.decode().endswith("/approvals/..%2Fx%3Fy")
    assert read.url.query == b""
    assert urlsplit(str(read.url)).path.count("/approvals/") == 1


# --- a subject token that is not a JWT is the platform's to judge -------------------------------------


def test_an_opaque_subject_token_is_exchanged_not_called_expired(
    af: AuthFI, key: AgentKey, platform: AgentPlatform
) -> None:
    delegated = af.agent(AGENT, key).exchange(Secret("an-opaque-session-handle"))
    assert delegated.subject.id == "usr_opaque"
    assert len(platform.exchanges) == 1


def test_a_jwt_whose_exp_has_passed_is_still_expired(af: AuthFI, key: AgentKey, platform: AgentPlatform) -> None:
    gone = Secret(platform.access_token(sub="usr_7", exp=1))
    with pytest.raises(ConsentExpired, match="sign in again"):
        af.agent(AGENT, key).exchange(gone)
    assert platform.exchanges == []


# --- a framework that copies its config must not crash on the agent ------------------------------------


def test_deepcopy_gives_back_the_same_agent_key_agent_and_tokens(
    af: AuthFI, aaf: AsyncAuthFI, key: AgentKey, platform: AgentPlatform
) -> None:
    agent = af.agent(AGENT, key)
    assert copy.deepcopy(key) is key
    assert copy.copy(key) is key
    assert copy.deepcopy(agent) is agent
    assert copy.deepcopy({"agent": agent})["agent"] is agent
    assert copy.deepcopy(aaf.agent(AGENT, key)).agent_id == AGENT
    delegated = agent.exchange(Secret(platform.access_token(sub="usr_7")))
    assert isinstance(delegated, DelegatedToken)
    assert copy.deepcopy(delegated) is delegated
    assert copy.copy(delegated) is delegated


async def test_deepcopy_of_the_async_surfaces(aaf: AsyncAuthFI, key: AgentKey, platform: AgentPlatform) -> None:
    agent = aaf.agent(AGENT, key)
    assert isinstance(agent, AsyncAgentAuth)
    assert copy.deepcopy(agent) is agent
    delegated = await agent.exchange(Secret(platform.access_token(sub="usr_7")))
    assert isinstance(delegated, AsyncDelegatedToken)
    assert copy.deepcopy(delegated) is delegated


def test_warnings_are_not_raised_by_the_new_form(af: AuthFI, key: AgentKey) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        af.agent(AGENT, key)
