"""FastAPI and Flask run the same cases, and answer every refusal the same way, byte for byte."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, cast

import httpx
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from flask import Flask

from authfi import AccessToken, AsyncAuthFI, AuthFI, ConfigurationError
from authfi._resource import INVALID_TOKEN_DESCRIPTION, Requirements
from authfi.fastapi import AuthFIBearer
from authfi.flask import AuthFIFlask
from tests.support import TENANT, FakeClock, Platform, json_response

AUD = ["api_orders"]


@dataclass
class Answer:
    status: int
    challenge: str | None
    body: dict[str, Any]
    request_id: str | None
    cache_control: str | None = None
    retry_after: str | None = None


Call = Callable[[dict[str, str]], Answer]


def fastapi_app(platform: Platform, clock: FakeClock, rules: dict[str, Any], **options: Any) -> Iterator[Call]:
    aaf = AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock)
    auth = AuthFIBearer(aaf.verifier(audience=AUD), **options)
    app = FastAPI()
    auth.install(app)
    dependency = auth.require(**rules) if rules else auth.optional()

    @app.get("/r")
    async def route(token: AccessToken | None = Depends(dependency)) -> dict[str, Any]:  # noqa: B008
        return {"who": str(token.principal) if token else None}

    with TestClient(app) as client:

        def call(headers: dict[str, str]) -> Answer:
            r = cast(httpx.Client, client).get("/r", headers=headers)
            body = cast(dict[str, Any], r.json())
            return Answer(
                r.status_code,
                r.headers.get("www-authenticate"),
                body,
                r.headers.get("x-request-id"),
                r.headers.get("cache-control"),
                r.headers.get("retry-after"),
            )

        yield call


def flask_app(platform: Platform, clock: FakeClock, rules: dict[str, Any], **options: Any) -> Iterator[Call]:
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    app = Flask(__name__)
    guard = AuthFIFlask(app, af.verifier(audience=AUD), **options)

    @app.get("/r")
    @guard.protect(**rules) if rules else guard.protect(optional=True)
    def route() -> dict[str, Any]:
        token = guard.token
        return {"who": str(token.principal) if token else None}

    client = app.test_client()

    def call(headers: dict[str, str]) -> Answer:
        r = client.get("/r", headers=headers)
        body = cast(dict[str, Any], r.get_json())
        return Answer(
            r.status_code,
            r.headers.get("WWW-Authenticate"),
            body,
            r.headers.get("X-Request-Id"),
            r.headers.get("Cache-Control"),
            r.headers.get("Retry-After"),
        )

    yield call


FRAMEWORKS = {"fastapi": fastapi_app, "flask": flask_app}


@pytest.fixture(params=sorted(FRAMEWORKS))
def framework(request: pytest.FixtureRequest) -> str:
    return str(request.param)


_LIVE: list[Iterator[Call]] = []


@pytest.fixture(autouse=True)
def _close_apps() -> Iterator[None]:
    """Keep each app's test client open for the whole test, then close it."""
    yield
    while _LIVE:
        _LIVE.pop().close()  # type: ignore[attr-defined]


def make(framework: str, platform: Platform, clock: FakeClock, rules: dict[str, Any], **options: Any) -> Call:
    gen = FRAMEWORKS[framework](platform, clock, rules, **options)
    _LIVE.append(gen)
    return next(gen)


# One table of guard cases, run in every adapter, each both ways.
GUARDS: list[tuple[dict[str, Any], dict[str, Any], int]] = [
    ({"permissions": ["a", "b"]}, {"permissions": ["a", "b"]}, 200),
    ({"permissions": ["a", "b"]}, {"permissions": ["a"]}, 403),
    ({"any_permission": ["a", "c"]}, {"permissions": ["c"]}, 200),
    ({"any_permission": ["a", "c"]}, {"permissions": ["d"]}, 403),
    ({"roles": ["r1", "r2"]}, {"roles": ["r2"]}, 200),
    ({"roles": ["r1"]}, {"roles": ["r2"]}, 403),
    ({"scopes": ["s1", "s2"]}, {"scope": "s1 s2"}, 200),
    ({"scopes": ["s1", "s2"]}, {"scope": "s1"}, 403),
    ({"permissions": ["a"], "roles": ["r1"]}, {"permissions": ["a"], "roles": ["r1"]}, 200),
    ({"permissions": ["a"], "roles": ["r1"]}, {"permissions": ["a"]}, 403),
]


@pytest.mark.parametrize(("rules", "grants", "status"), GUARDS)
def test_the_guard_table(
    framework: str, platform: Platform, clock: FakeClock, rules: dict[str, Any], grants: dict[str, Any], status: int
) -> None:
    call = make(framework, platform, clock, rules)
    answer = call({"Authorization": "Bearer " + platform.access_token(**grants)})
    assert answer.status == status
    if status == 200:
        assert answer.body == {"who": "user:usr_1"}
    elif "scopes" in rules:
        assert answer.body == {"error": "insufficient_scope"}
        assert answer.challenge == f'Bearer error="insufficient_scope", scope="{" ".join(rules["scopes"])}"'
    else:
        assert answer.body == {"error": "forbidden"}
        assert answer.challenge is None


def bad_tokens(platform: Platform) -> dict[str, str]:
    now = int(time.time())
    return {
        "token_expired": platform.access_token(exp=now - 3600),
        "token_audience_mismatch": platform.access_token(aud="other"),
        "token_not_access_token": platform.keys.sign(
            {"iss": platform.issuer, "sub": "u", "aud": "api_orders", "exp": now + 60, "tenant_id": TENANT}, typ="JWT"
        ),
        "token_issuer_not_accepted": platform.access_token(iss="https://evil.authfi.io"),
        "token_no_tenant": platform.access_token(tenant_id=None),
        "token_key_unknown": platform.keys.sign({"iss": platform.issuer}, kid="nope"),
        "token_malformed": "not.a-token",
        "token_dpop_proof_required": platform.access_token(cnf={"jkt": "x"}),
        "token_act_chain_unsupported": platform.access_token(act={"sub": "agt_1", "act": {"sub": "agt_2"}}),
    }


@pytest.mark.parametrize(
    "code",
    [
        "token_expired",
        "token_audience_mismatch",
        "token_not_access_token",
        "token_issuer_not_accepted",
        "token_no_tenant",
        "token_key_unknown",
        "token_malformed",
        "token_dpop_proof_required",
        "token_act_chain_unsupported",
    ],
)
def test_every_refusal_is_the_same_opaque_answer_in_every_adapter(
    code: str, platform: Platform, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    token = bad_tokens(platform)[code]
    answers: dict[str, Answer] = {}
    with caplog.at_level(logging.INFO, logger="authfi"):
        for name in FRAMEWORKS:
            answers[name] = make(name, platform, clock, {"permissions": ["a"]})({"Authorization": "Bearer " + token})
    fastapi_answer, flask_answer = answers["fastapi"], answers["flask"]
    assert fastapi_answer == flask_answer
    assert fastapi_answer.status == 401
    assert fastapi_answer.body == {"error": "invalid_token", "error_description": INVALID_TOKEN_DESCRIPTION}
    assert fastapi_answer.challenge == f'Bearer error="invalid_token", error_description="{INVALID_TOKEN_DESCRIPTION}"'
    assert code not in str(fastapi_answer)
    logged = [r.getMessage() for r in caplog.records if r.getMessage().startswith("refused GET")]
    assert len(logged) == 2
    assert all(code in line for line in logged)
    assert all(token not in line for line in logged)


def test_every_answer_to_a_refused_request_is_marked_no_store(
    framework: str, platform: Platform, clock: FakeClock
) -> None:
    call = make(framework, platform, clock, {"permissions": ["a"]})
    assert call({}).cache_control == "no-store"
    assert call({"Authorization": "Bearer junk"}).cache_control == "no-store"
    assert call({"Authorization": "Bearer " + platform.access_token(permissions=["b"])}).cache_control == "no-store"
    assert call({"Authorization": "Bearer " + platform.access_token(permissions=["a"])}).cache_control != "no-store"


def unreachable_jwks(platform: Platform) -> None:
    platform.route("GET", "/.well-known/jwks.json", json_response({"error": "down"}, status=500))


def unreachable_discovery(platform: Platform) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    platform.route("GET", "/.well-known/openid-configuration", refuse)


@pytest.mark.parametrize("outage", [unreachable_jwks, unreachable_discovery], ids=["key set", "discovery"])
def test_a_token_that_cannot_be_checked_is_a_503_to_retry_not_a_500(
    outage: Callable[[Platform], None],
    platform: Platform,
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    outage(platform)
    token = platform.access_token()
    answers: dict[str, Answer] = {}
    with caplog.at_level(logging.INFO, logger="authfi"):
        for name in FRAMEWORKS:
            answers[name] = make(name, platform, clock, {"permissions": ["a"]})(
                {"Authorization": "Bearer " + token, "X-Request-Id": "trace-123"}
            )
    assert answers["fastapi"] == answers["flask"]
    answer = answers["fastapi"]
    assert answer.status == 503
    assert answer.body == {"error": "service_unavailable"}
    assert answer.retry_after == "5"
    assert answer.cache_control == "no-store"
    assert answer.challenge is None
    assert answer.request_id == "trace-123"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and "could not check" in r.getMessage()]
    assert len(errors) == 2
    assert all("trace-123" in r.getMessage() and token not in r.getMessage() for r in errors)


def test_a_missing_token_is_logged_at_info_and_any_other_refusal_at_warning(
    framework: str, platform: Platform, clock: FakeClock, caplog: pytest.LogCaptureFixture
) -> None:
    call = make(framework, platform, clock, {"permissions": ["a"]})
    with caplog.at_level(logging.DEBUG, logger="authfi"):
        call({})
        call({"Authorization": "Bearer junk"})
    levels = {
        r.getMessage().split(": ")[1].split(" ")[0]: r.levelno
        for r in caplog.records
        if r.getMessage().startswith("refused GET")
    }
    assert levels == {"token_missing": logging.INFO, "token_malformed": logging.WARNING}


def test_no_credential_is_a_challenge_without_an_error(framework: str, platform: Platform, clock: FakeClock) -> None:
    call = make(
        framework,
        platform,
        clock,
        {"permissions": ["a"]},
        resource_metadata="https://api.example.com/.well-known/oauth-protected-resource",
    )
    answer = call({})
    assert answer.status == 401
    assert answer.body == {"error": "unauthorized"}
    assert answer.challenge == 'Bearer resource_metadata="https://api.example.com/.well-known/oauth-protected-resource"'


def test_a_dpop_request_is_answered_with_the_dpop_scheme(framework: str, platform: Platform, clock: FakeClock) -> None:
    call = make(framework, platform, clock, {"permissions": ["a"]})
    answer = call({"Authorization": "DPoP " + platform.access_token(cnf={"jkt": "x"}), "DPoP": "proof"})
    assert answer.status == 401
    assert answer.challenge is not None and answer.challenge.startswith("DPoP ")


def test_the_code_on_the_wire_is_an_explicit_choice(framework: str, platform: Platform, clock: FakeClock) -> None:
    call = make(framework, platform, clock, {"permissions": ["a"]}, expose_refusal_code=True)
    answer = call({"Authorization": "Bearer " + platform.access_token(exp=int(time.time()) - 3600)})
    assert answer.body["code"] == "token_expired"


def test_step_up_raised_in_a_flask_view_answers_rfc_9470(platform: Platform, clock: FakeClock) -> None:
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    app = Flask(__name__)
    guard = AuthFIFlask(app, af.verifier(audience=AUD))

    @app.get("/r")
    @guard.protect(permissions=["a"])
    def route() -> dict[str, str]:
        token = guard.token
        assert token is not None
        token.require_acr("urn:authfi:acr:mfa", max_age=300)
        return {}

    r = app.test_client().get("/r", headers={"Authorization": "Bearer " + platform.access_token(permissions=["a"])})
    assert r.status_code == 401
    assert r.get_json() == {"error": "insufficient_user_authentication"}
    assert r.headers["WWW-Authenticate"] == (
        'Bearer error="insufficient_user_authentication", acr_values="urn:authfi:acr:mfa", max_age=300'
    )


def test_step_up_raised_in_a_fastapi_route_answers_rfc_9470(platform: Platform, clock: FakeClock) -> None:
    aaf = AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock)
    auth = AuthFIBearer(aaf.verifier(audience=AUD))
    app = FastAPI()
    auth.install(app)

    @app.get("/r")
    async def route(token: AccessToken = Depends(auth.require(permissions=["a"]))) -> dict[str, str]:  # noqa: B008
        token.require_acr("urn:authfi:acr:passkey")
        return {}

    with TestClient(app) as client:
        api = cast(httpx.Client, client)
        r = api.get("/r", headers={"Authorization": "Bearer " + platform.access_token(permissions=["a"])})
    assert r.status_code == 401
    assert r.json() == {"error": "insufficient_user_authentication"}
    assert r.headers["www-authenticate"] == (
        'Bearer error="insufficient_user_authentication", acr_values="urn:authfi:acr:passkey"'
    )


def test_the_request_id_is_passed_back(framework: str, platform: Platform, clock: FakeClock) -> None:
    call = make(framework, platform, clock, {"permissions": ["a"]})
    assert call({"X-Request-Id": "trace-123"}).request_id == "trace-123"
    assert call({"X-Request-Id": "bad id!"}).request_id is None


def test_an_optional_route(framework: str, platform: Platform, clock: FakeClock) -> None:
    call = make(framework, platform, clock, {})
    assert call({}).body == {"who": None}
    assert call({"Authorization": "Bearer " + platform.access_token()}).body == {"who": "user:usr_1"}
    assert call({"Authorization": "Bearer junk"}).status == 401


def test_flask_does_not_leak_the_previous_requests_token(platform: Platform, clock: FakeClock) -> None:
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    app = Flask(__name__)
    guard = AuthFIFlask(app, af.verifier(audience=AUD))

    @app.get("/me")
    @guard.protect(optional=True)
    def me() -> dict[str, Any]:
        token = guard.token
        return {"who": str(token.principal) if token else None}

    client = app.test_client()
    assert client.get("/me", headers={"Authorization": "Bearer " + platform.access_token()}).get_json() == {
        "who": "user:usr_1"
    }
    assert client.get("/me").get_json() == {"who": None}


@pytest.mark.parametrize("bad", [{"permissions": []}, {"roles": "admin"}, {"scopes": [""]}])
def test_an_empty_requirement_is_a_configuration_error(bad: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        Requirements.of(**bad)


def test_each_adapter_takes_the_verifier_its_framework_needs(platform: Platform, clock: FakeClock) -> None:
    af = AuthFI(platform.issuer, clock=clock)
    aaf = AsyncAuthFI(platform.issuer, clock=clock)
    with pytest.raises(ConfigurationError, match="AsyncAuthFI"):
        AuthFIBearer(af.verifier(audience=AUD))  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="AuthFI"):
        AuthFIFlask(None, aaf.verifier(audience=AUD))  # type: ignore[arg-type]
    guard = AuthFIFlask(None, af.verifier(audience=AUD))
    with pytest.raises(ConfigurationError):
        guard.protect(optional=True, permissions=["a"])


def test_a_request_path_cannot_forge_a_log_line_or_fill_the_log(caplog: pytest.LogCaptureFixture) -> None:
    """Flask decodes %0a in a path: it is logged escaped, on one line, and capped."""
    from authfi import InvalidToken
    from authfi._resource import log_refusal, log_unavailable

    logger = logging.getLogger("authfi")
    forged = "/a\nWARNING authfi: forged"
    with caplog.at_level(logging.INFO, logger="authfi"):
        log_refusal(logger, InvalidToken("token_expired", "x"), "GET", forged, "r1")
        log_unavailable(logger, RuntimeError("x"), "GET", forged, "r1")
        log_refusal(logger, InvalidToken("token_expired", "x"), "GET", "/" + "p" * 5000, "r1")
    messages = [r.getMessage() for r in caplog.records]
    assert len(messages) == 3
    assert all("\n" not in m for m in messages)
    escaped = "/a" + chr(92) + "nWARNING authfi: forged"
    assert escaped in messages[0] and escaped in messages[1]
    assert len(messages[2]) < 400
