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
        sync_params = sorted(set(inspect.signature(sync_fn).parameters) - LEGACY_PARAMETERS)
        async_params = sorted(set(inspect.signature(async_fn).parameters) - LEGACY_PARAMETERS)
        assert sync_params == async_params, f"{sync.__name__}.{name}"
        if inspect.iscoroutinefunction(async_fn):
            assert not inspect.iscoroutinefunction(sync_fn), f"{sync.__name__}.{name} is a coroutine"


def test_from_env_exists_on_both_projects() -> None:
    assert (
        inspect.signature(authfi.AuthFI.from_env).parameters.keys()
        == inspect.signature(authfi.AsyncAuthFI.from_env).parameters.keys()
    )
