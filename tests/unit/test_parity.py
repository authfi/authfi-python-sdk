"""The blocking and asyncio surfaces have the same names and parameters, and never mix kinds."""

from __future__ import annotations

import inspect

import pytest

import authfi
from authfi import agents

PAIRS: list[tuple[type, type]] = [
    (authfi.AuthFI, authfi.AsyncAuthFI),
    (authfi.TokenVerifier, authfi.AsyncTokenVerifier),
    (authfi.MultiIssuerVerifier, authfi.AsyncMultiIssuerVerifier),
    (authfi.ClientCredentials, authfi.AsyncClientCredentials),
    (agents.AgentAuth, agents.AsyncAgentAuth),
    (agents.DelegatedToken, agents.AsyncDelegatedToken),
    (authfi.TokenSource, authfi.AsyncTokenSource),
    (authfi.InMemoryTokenStore, authfi.AsyncInMemoryTokenStore),
    (authfi.Gateway, authfi.AsyncGateway),
]

# Names that differ on purpose: 0.1's blocking-only names, the close/aclose pair, and asyncio-only names.
LEGACY_OR_RENAMED = {
    "close",
    "aclose",
    "__enter__",
    "__exit__",
    "__aenter__",
    "__aexit__",
    # 0.1's AuthFI and AgentAuth names (blocking only, deprecated)
    "tenant",
    "api_url",
    "application_id",
    "auto_sync",
    "jwks_ttl",
    "base_domain",
    "jwks_url",
    "verify_token",
    "require",
    "require_role",
    "current_user",
    "register_permission",
    "sync",
    "cloud_credentials",
    "cloud_token",
    "audience",
    "platform_url",
    "token_endpoint",
    "backchannel_endpoint",
    "discovery",
    "auth_header",
    "can",
    "request_user_approval",
    "from_env",
    # MCP clients run on asyncio only
    "mcp_client_factory",
}

# Parameters only the blocking form has: 0.1's spellings.
LEGACY_PARAMETERS = {"audience", "sleep"}

# Parameters the blocking form still takes by position because 0.1 did, and the asyncio form only by name.
LEGACY_POSITIONAL = {("AgentAuth", "token", "force"), ("AgentAuth", "await_user_token", "timeout")}

# Methods that build or describe and do no I/O: plain functions on both sides.
PLAIN_ON_BOTH = {
    "agent",
    "agent_from_env",
    "client_credentials",
    "for_resource",
    "gateway",
    "governance_of",
    "http_client",
    "httpx_auth",
    "model_base_url",
    "names_authfi",
    "new_http_client",
    "pinned",
    "read_result",
    "scope_of",
    "strip_approval_argument",
    "verifier",
    "with_approval",
}


def public(cls: type) -> set[str]:
    return {name for name in dir(cls) if not name.startswith("_")}


@pytest.mark.parametrize(("sync", "asyn"), PAIRS, ids=[p[0].__name__ for p in PAIRS])
def test_the_pair_has_the_same_names(sync: type, asyn: type) -> None:
    assert public(sync) - LEGACY_OR_RENAMED == public(asyn) - LEGACY_OR_RENAMED


@pytest.mark.parametrize(("sync", "asyn"), PAIRS, ids=[p[0].__name__ for p in PAIRS])
def test_the_pair_has_the_same_parameters_and_the_async_side_is_coroutines(sync: type, asyn: type) -> None:
    for name in public(sync) - LEGACY_OR_RENAMED:
        left = inspect.getattr_static(sync, name)
        if not callable(left) or isinstance(left, type | property):
            continue
        sync_fn = getattr(sync, name)
        async_fn = getattr(asyn, name)
        where = f"{sync.__name__}.{name}"
        sync_params = inspect.signature(sync_fn).parameters
        async_params = inspect.signature(async_fn).parameters
        assert sorted(set(sync_params) - LEGACY_PARAMETERS) == sorted(set(async_params) - LEGACY_PARAMETERS), where
        for param in sorted(set(async_params) - LEGACY_PARAMETERS):
            left_param, right_param = sync_params[param], async_params[param]
            assert (left_param.default is inspect.Parameter.empty) == (
                right_param.default is inspect.Parameter.empty
            ), f"{where}({param}): required on one side only"
            if (sync.__name__, name, param) not in LEGACY_POSITIONAL:
                assert left_param.kind == right_param.kind, f"{where}({param}): {left_param.kind} vs {right_param.kind}"
        # The asyncio side does its I/O in coroutines; the blocking side never is one.
        assert not inspect.iscoroutinefunction(sync_fn), f"{where} is a coroutine"
        if name not in PLAIN_ON_BOTH:
            assert inspect.iscoroutinefunction(async_fn), f"{where} is not a coroutine on the asyncio side"


def test_from_env_exists_on_both_projects() -> None:
    assert (
        inspect.signature(authfi.AuthFI.from_env).parameters.keys()
        == inspect.signature(authfi.AsyncAuthFI.from_env).parameters.keys()
    )
