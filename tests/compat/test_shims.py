# pyright: reportDeprecated=false
"""The 0.1 names: each still works where it is safe, warns, and points at its successor."""

from __future__ import annotations

import warnings
from typing import Any, cast

import pytest
from flask import Flask

from authfi import USER_AGENT, AuthFI, AuthFIDeprecationWarning, ConfigurationError, NotSupportedError, OAuthError
from tests.support import FakeClock, Platform

AUD = ["api_orders"]


def legacy(platform: Platform, clock: FakeClock, **kwargs: Any) -> AuthFI:
    with pytest.warns(AuthFIDeprecationWarning):
        return AuthFI(tenant="acme", api_key="sk_live_x", http_client=platform.client(), clock=clock, **kwargs)


def test_the_new_form_does_not_warn() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        AuthFI("https://acme.authfi.io")


def test_the_old_form_builds_the_project_from_the_tenant(platform: Platform, clock: FakeClock) -> None:
    af = legacy(platform, clock)
    assert af.issuer == "https://acme.authfi.io"
    with pytest.warns(AuthFIDeprecationWarning, match="issuer"):
        positional = AuthFI("acme", "sk_live_x")
    assert positional.issuer == "https://acme.authfi.io"
    with pytest.warns(AuthFIDeprecationWarning):
        custom = AuthFI(tenant="acme", issuer="https://login.acme.com/")
    assert custom.issuer == "https://login.acme.com"
    with pytest.raises(ConfigurationError, match="address"):
        AuthFI(api_key="x")


def test_old_attributes_warn_and_new_projects_do_not_have_them(platform: Platform, clock: FakeClock) -> None:
    af = legacy(platform, clock, application_id="app", client_secret="s3cret", jwks_ttl=120)
    with pytest.warns(AuthFIDeprecationWarning):
        assert af.application_id == "app"
    with pytest.warns(AuthFIDeprecationWarning):
        assert af.jwks_ttl == 120
    with pytest.warns(AuthFIDeprecationWarning):
        assert af.base_domain == "authfi.io"
    assert "s3cret" not in repr(af)
    with pytest.raises(AttributeError), pytest.warns(AuthFIDeprecationWarning):
        _ = AuthFI("https://acme.authfi.io").tenant


def test_jwks_ttl_still_sets_how_long_the_key_set_is_kept(platform: Platform, clock: FakeClock) -> None:
    af = legacy(platform, clock, jwks_ttl=1000, audience=AUD)
    with pytest.warns(AuthFIDeprecationWarning):
        af.verify_token(platform.access_token())
    clock.advance(600)
    with pytest.warns(AuthFIDeprecationWarning):
        af.verify_token(platform.access_token())
    assert len(platform.calls("GET", "/.well-known/jwks.json")) == 1


def test_verify_token_needs_an_audience_and_then_returns_read_only_claims(platform: Platform, clock: FakeClock) -> None:
    af = legacy(platform, clock)
    with pytest.raises(ConfigurationError, match="audience"), pytest.warns(AuthFIDeprecationWarning):
        af.verify_token(platform.access_token())
    af = legacy(platform, clock, audience=AUD)
    with pytest.warns(AuthFIDeprecationWarning):
        claims = af.verify_token(platform.access_token())
    assert claims["sub"] == "usr_1"
    with pytest.raises(TypeError):
        cast(Any, claims)["sub"] = "x"


def test_retired_calls_say_what_replaces_them(platform: Platform, clock: FakeClock) -> None:
    af = legacy(platform, clock)
    with pytest.raises(NotSupportedError), pytest.warns(AuthFIDeprecationWarning):
        af.register_permission("x")
    with pytest.raises(NotSupportedError), pytest.warns(AuthFIDeprecationWarning):
        af.sync()
    with pytest.raises(NotSupportedError), pytest.warns(AuthFIDeprecationWarning):
        af.cloud_credentials("t", "aws")
    with pytest.raises(NotSupportedError), pytest.warns(AuthFIDeprecationWarning):
        af.cloud_token("t", "aud")
    assert platform.requests == []


def test_the_old_flask_decorators_answer_401_and_403_not_500(platform: Platform, clock: FakeClock) -> None:
    af = legacy(platform, clock, audience=AUD)
    app = Flask(__name__)

    with pytest.warns(AuthFIDeprecationWarning):

        @app.get("/users")
        @af.require("read:users")
        def users() -> dict[str, Any]:
            with pytest.warns(AuthFIDeprecationWarning):
                current = af.current_user()
            return {"sub": current["sub"] if current else None}

    with pytest.warns(AuthFIDeprecationWarning):

        @app.get("/admin")
        @af.require_role("admin", "owner")
        def admin() -> dict[str, str]:
            return {}

    client = app.test_client()
    assert client.get("/users").status_code == 401
    denied = client.get("/users", headers={"Authorization": "Bearer " + platform.access_token(permissions=["x"])})
    assert denied.status_code == 403
    assert denied.get_json() == {"error": "forbidden"}
    ok = client.get("/users", headers={"Authorization": "Bearer " + platform.access_token(permissions=["read:users"])})
    assert ok.get_json() == {"sub": "usr_1"}
    assert (
        client.get("/admin", headers={"Authorization": "Bearer " + platform.access_token(roles=["owner"])}).status_code
        == 200
    )


def test_agent_auth_error_is_oauth_error() -> None:
    import authfi

    with pytest.warns(AuthFIDeprecationWarning):
        alias = authfi.AgentAuthError
    assert alias is OAuthError
    error = OAuthError("invalid_grant", "used", 400)
    assert error.status == 400
    with pytest.warns(AuthFIDeprecationWarning):
        assert error.error == "invalid_grant"
    with pytest.raises(AttributeError):
        _ = cast(Any, authfi).NoSuchName


def test_user_agent_constant() -> None:
    assert USER_AGENT.startswith("authfi-python/")
