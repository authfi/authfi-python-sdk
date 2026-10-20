"""People signing in to a server-rendered app: the authorize URL, the sealed pending sign-in, the callback,
the ID token, refresh, userinfo, sign-out and the session, against a fake AuthFI project."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import secrets
import threading
from collections.abc import Iterator
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
import pytest

from authfi import (
    AsyncAuthFI,
    AsyncInMemoryTokenStore,
    AuthFI,
    ConfigurationError,
    ConsentDenied,
    InMemoryTokenStore,
    InvalidCallback,
    InvalidIdToken,
    InvalidPendingSignIn,
    NotSupportedError,
    OAuthError,
    PendingSignIn,
    Secret,
    StepUpRequired,
    TokenSet,
)
from authfi.signin import WebApp
from tests.support import FakeClock, Platform, b64url, json_response

CLIENT = "app_web"
REDIRECT = "https://app.example.com/callback"
STATE_KEY = Secret(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())


class OIDC:
    """The sign-in half of a fake project: remembers the authorize request and answers the token endpoint."""

    def __init__(self, platform: Platform, clock: FakeClock) -> None:
        self.platform = platform
        self.clock = clock
        self.authorize: dict[str, str] = {}
        self.codes: dict[str, dict[str, str]] = {}
        self.acr = "urn:authfi:acr:pwd"
        self.auth_age = 0
        self.id_claims: dict[str, Any] = {}
        self.id_alg = "RS256"
        self.id_typ: str | None = "JWT"
        self.id_kid: str | None = None
        self.refreshes = 0
        self.revoked: list[str] = []
        self.refresh_answer: dict[str, Any] | None = None
        platform.route("POST", "/oauth/token", self.token)
        platform.route("GET", "/oauth/userinfo", self.userinfo)
        platform.route("POST", "/oauth/token/revoke", self.revoke)

    def browser(self, url: str, *, iss: str | None = None, state: str | None = None, **extra: str) -> str:
        """What the browser comes back with after the person signs in at ``url``."""
        query = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        self.authorize = query
        code = "code-" + secrets.token_hex(4)
        self.codes[code] = query
        back = {
            "code": code,
            "state": state if state is not None else query["state"],
            "iss": iss or self.platform.issuer,
        }
        back.update(extra)
        return REDIRECT + "?" + urlencode({k: v for k, v in back.items() if v != ""})

    def id_token(self, access: str, nonce: str | None, **claims: Any) -> str:
        now = int(self.clock.time())
        body: dict[str, Any] = {
            "iss": self.platform.issuer,
            "sub": "usr_ada",
            "aud": CLIENT,
            "exp": now + 300,
            "iat": now,
            "auth_time": now - self.auth_age,
            "acr": self.acr,
            "amr": ["pwd"],
            "email": "ada@example.com",
            "email_verified": True,
            "name": "Ada",
            "at_hash": b64url(hashlib.sha256(access.encode()).digest()[:16]),
        }
        if nonce is not None:
            body["nonce"] = nonce
        body.update(self.id_claims)
        body.update(claims)
        return self.platform.keys.sign(body, alg=self.id_alg, typ=self.id_typ, kid=self.id_kid)

    def token(self, request: httpx.Request) -> httpx.Response:
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        if form["grant_type"] == "authorization_code":
            asked = self.codes.pop(form["code"], None)
            if asked is None:
                return json_response({"error": "invalid_grant"}, 400)
            challenge = b64url(hashlib.sha256(form["code_verifier"].encode()).digest())
            if challenge != asked["code_challenge"] or form["redirect_uri"] != REDIRECT:
                return json_response({"error": "invalid_grant"}, 400)
            access = "at-" + secrets.token_hex(8)
            return json_response(
                {
                    "access_token": access,
                    "token_type": "Bearer",
                    "expires_in": 900,
                    "refresh_token": "rt-1",
                    "id_token": self.id_token(access, asked["nonce"]),
                    "scope": asked["scope"],
                }
            )
        if form["grant_type"] == "refresh_token":
            self.refreshes += 1
            if self.refresh_answer is not None:
                return json_response(self.refresh_answer, 400 if "error" in self.refresh_answer else 200)
            access = f"at-refreshed-{self.refreshes}"
            return json_response(
                {
                    "access_token": access,
                    "token_type": "Bearer",
                    "expires_in": 900,
                    "refresh_token": f"rt-{self.refreshes + 1}",
                }
            )
        return json_response({"error": "unsupported_grant_type"}, 400)

    def userinfo(self, request: httpx.Request) -> httpx.Response:
        if not request.headers.get("authorization", "").startswith("Bearer at-"):
            return json_response({"error": "invalid_token"}, 401)
        return json_response(
            {"sub": self.id_claims.get("userinfo_sub", "usr_ada"), "email": "ada@example.com", "name": "Ada"}
        )

    def revoke(self, request: httpx.Request) -> httpx.Response:
        self.revoked.append(parse_qs(request.content.decode())["token"][0])
        return httpx.Response(200)


@pytest.fixture
def platform() -> Platform:
    return Platform()


@pytest.fixture
def oidc(platform: Platform, clock: FakeClock) -> OIDC:
    return OIDC(platform, clock)


@pytest.fixture
def af(platform: Platform, clock: FakeClock) -> Iterator[AuthFI]:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        yield project


@pytest.fixture
def web(af: AuthFI) -> WebApp:
    return af.web_app(client_id=CLIENT, client_secret=Secret("s3cret"), redirect_uri=REDIRECT, state_key=STATE_KEY)


def _sign_in(web: WebApp, oidc: OIDC, **start: Any) -> Any:
    pending = web.start_sign_in(**start)
    restored = web.load_pending(pending.dump())
    return web.finish_sign_in(oidc.browser(pending.url), restored)


# --- the round trip ---------------------------------------------------------------------------------


def test_a_person_signs_in(web: WebApp, oidc: OIDC, platform: Platform) -> None:
    pending = web.start_sign_in(login_hint="ada@example.com", prompt="login")
    query = parse_qs(urlsplit(pending.url).query)
    assert pending.url.startswith(platform.issuer + "/oauth/authorize?")
    assert query["response_type"] == ["code"]
    assert query["code_challenge_method"] == ["S256"]
    assert query["scope"] == ["openid profile email"]
    assert query["login_hint"] == ["ada@example.com"]
    assert len(query["state"][0]) >= 43
    assert len(query["nonce"][0]) >= 43

    blob = pending.dump()
    signed_in = web.finish_sign_in(oidc.browser(pending.url), web.load_pending(blob))

    assert signed_in.user.sub == "usr_ada"
    assert signed_in.user.email == "ada@example.com"
    assert signed_in.user.email_verified is True
    assert signed_in.tokens.refresh_token == Secret("rt-1")
    assert signed_in.tokens.id_token is not None
    (exchange,) = platform.calls("POST", "/oauth/token")
    assert exchange.headers["authorization"].startswith("Basic ")


def test_the_callback_may_be_its_query_parameters(web: WebApp, oidc: OIDC) -> None:
    pending = web.start_sign_in()
    back = {k: v[0] for k, v in parse_qs(urlsplit(oidc.browser(pending.url)).query).items()}
    assert web.finish_sign_in(back, pending).user.sub == "usr_ada"


def test_a_public_client_uses_pkce_alone(af: AuthFI, oidc: OIDC, platform: Platform) -> None:
    public = af.web_app(client_id=CLIENT, client_secret=None, redirect_uri=REDIRECT, state_key=STATE_KEY)
    _sign_in(public, oidc)
    (exchange,) = platform.calls("POST", "/oauth/token")
    assert "authorization" not in exchange.headers
    assert parse_qs(exchange.content.decode())["client_id"] == [CLIENT]


# --- the pending sign-in ------------------------------------------------------------------------------


def test_the_pending_sign_in_is_opaque_and_bound(web: WebApp, af: AuthFI, clock: FakeClock) -> None:
    pending = web.start_sign_in()
    blob = pending.dump()
    assert pending.state not in blob
    assert "PendingSignIn(client_id='app_web')" == repr(pending)

    other = af.web_app(client_id="app_other", client_secret=None, redirect_uri=REDIRECT, state_key=STATE_KEY)
    with pytest.raises(InvalidPendingSignIn) as caught:
        other.load_pending(blob)
    assert caught.value.code == "pending_sign_in_tampered"

    tampered = blob[:-4] + ("AAAA" if not blob.endswith("AAAA") else "BBBB")
    with pytest.raises(InvalidPendingSignIn):
        web.load_pending(tampered)
    with pytest.raises(InvalidPendingSignIn, match="malformed"):
        web.load_pending("not-a-blob")

    clock.advance(601)
    with pytest.raises(InvalidPendingSignIn) as expired:
        web.load_pending(blob)
    assert expired.value.code == "pending_sign_in_expired"


def test_a_pending_sign_in_does_not_pickle(web: WebApp) -> None:
    import pickle

    with pytest.raises(TypeError, match="dump"):
        pickle.dumps(web.start_sign_in())


def test_load_without_the_app(web: WebApp, clock: FakeClock) -> None:
    blob = web.start_sign_in().dump()
    restored = PendingSignIn.load(blob, key=STATE_KEY, client_id=CLIENT, redirect_uri=REDIRECT, now=clock.time())
    assert restored.state


@pytest.mark.parametrize(
    ("key", "match"),
    [
        (Secret(base64.urlsafe_b64encode(b"sixteen bytes!!!").decode()), "16 bytes, not 32"),
        (Secret("!!!not base64!!!"), "base64|bytes"),
        ("plain string", "Secret"),
    ],
)
def test_a_state_key_must_be_32_random_bytes(af: AuthFI, key: Any, match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        af.web_app(client_id=CLIENT, client_secret=None, redirect_uri=REDIRECT, state_key=key)


@pytest.mark.parametrize(
    "redirect", ["http://app.example.com/callback", "https://app.example.com/cb#frag", "/callback"]
)
def test_the_redirect_uri_is_checked(af: AuthFI, redirect: str) -> None:
    with pytest.raises(ConfigurationError, match="redirect_uri"):
        af.web_app(client_id=CLIENT, client_secret=None, redirect_uri=redirect, state_key=STATE_KEY)


def test_a_loopback_redirect_is_allowed(af: AuthFI) -> None:
    app = af.web_app(client_id=CLIENT, client_secret=None, redirect_uri="http://127.0.0.1:8000/cb", state_key=STATE_KEY)
    assert app.client_id == CLIENT


def test_other_settings_are_checked(af: AuthFI, web: WebApp) -> None:
    with pytest.raises(ConfigurationError, match="client_id"):
        af.web_app(client_id=" ", client_secret=None, redirect_uri=REDIRECT, state_key=STATE_KEY)
    with pytest.raises(ConfigurationError, match="auth_method"):
        af.web_app(
            client_id=CLIENT,
            client_secret=Secret("s"),
            redirect_uri=REDIRECT,
            state_key=STATE_KEY,
            auth_method="x",  # type: ignore[arg-type]
        )
    with pytest.raises(ConfigurationError, match="step_up_acr"):
        af.web_app(client_id=CLIENT, client_secret=None, redirect_uri=REDIRECT, state_key=STATE_KEY, step_up_acr="x")
    with pytest.raises(ConfigurationError, match="scope"):
        web.start_sign_in(scope="openid profile")
    with pytest.raises(ConfigurationError, match="acr_values"):
        web.start_sign_in(acr_values="urn:x")
    with pytest.raises(NotSupportedError):
        web.start_sign_in(prompt="none")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="prompt"):
        web.start_sign_in(prompt="select_account")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="max_age"):
        web.start_sign_in(max_age=-1)
    assert "openid" in web.start_sign_in(scope=("profile",)).url


# --- the callback ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "code"),
    [
        ({"iss": "https://evil.example"}, "callback_iss_mismatch"),
        ({"state": "someone-elses"}, "callback_state_mismatch"),
    ],
)
def test_the_callback_is_refused_before_the_code_is_used(
    web: WebApp, oidc: OIDC, platform: Platform, change: dict[str, str], code: str
) -> None:
    pending = web.start_sign_in()
    with pytest.raises(InvalidCallback) as caught:
        web.finish_sign_in(oidc.browser(pending.url, **change), pending)
    assert caught.value.code == code
    assert platform.calls("POST", "/oauth/token") == []


def test_a_callback_without_iss_is_refused(web: WebApp, oidc: OIDC) -> None:
    pending = web.start_sign_in()
    url = oidc.browser(pending.url)
    no_iss = url.replace("&iss=" + oidc.platform.issuer.replace(":", "%3A").replace("/", "%2F"), "")
    with pytest.raises(InvalidCallback) as caught:
        web.finish_sign_in(no_iss, pending)
    assert caught.value.code == "callback_iss_missing"


def test_a_refusal_in_the_callback(web: WebApp, oidc: OIDC) -> None:
    pending = web.start_sign_in()
    with pytest.raises(ConsentDenied):
        web.finish_sign_in(oidc.browser(pending.url, error="access_denied", code=""), pending)
    with pytest.raises(OAuthError) as caught:
        web.finish_sign_in(oidc.browser(pending.url, error="login_required", error_description="x", code=""), pending)
    assert caught.value.code == "login_required"
    with pytest.raises(InvalidCallback, match="code"):
        web.finish_sign_in(oidc.browser(pending.url, code=""), pending)


# --- the ID token -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("setup", "code"),
    [
        ({"id_claims": {"nonce": "another"}}, "id_token_nonce_mismatch"),
        ({"id_claims": {"aud": "app_other"}}, "id_token_audience_mismatch"),
        ({"id_claims": {"aud": [CLIENT, "app_other"]}}, "id_token_audience_mismatch"),
        ({"id_claims": {"azp": "app_other"}}, "id_token_audience_mismatch"),
        ({"id_claims": {"iss": "https://evil.example"}}, "id_token_issuer_mismatch"),
        ({"id_claims": {"at_hash": "AAAA"}}, "id_token_at_hash_mismatch"),
        ({"id_claims": {"exp": 1}}, "id_token_expired"),
        ({"id_claims": {"iat": 4_000_000_000}}, "id_token_claims_invalid"),
        ({"id_claims": {"sub": ""}}, "id_token_claims_invalid"),
        ({"id_typ": "at+jwt"}, "id_token_not_id_token"),
        ({"id_typ": "JOSE"}, "id_token_not_id_token"),
        ({"id_alg": "EdDSA"}, "id_token_alg_refused"),
    ],
)
def test_an_id_token_is_refused(web: WebApp, oidc: OIDC, setup: dict[str, Any], code: str) -> None:
    for name, value in setup.items():
        setattr(oidc, name, value)
    with pytest.raises(InvalidIdToken) as caught:
        _sign_in(web, oidc)
    assert caught.value.code == code


def test_an_id_token_signed_by_an_unknown_key(web: WebApp, oidc: OIDC, platform: Platform) -> None:
    oidc.id_kid = "rsa-unknown"
    with pytest.raises(InvalidIdToken) as caught:
        _sign_in(web, oidc)
    assert caught.value.code == "id_token_key_unknown"


def test_a_token_answer_without_an_id_token(web: WebApp, oidc: OIDC, platform: Platform) -> None:
    platform.route("POST", "/oauth/token", json_response({"access_token": "at-1", "token_type": "Bearer"}))
    with pytest.raises(InvalidIdToken, match="openid"):
        _sign_in(web, oidc)


def test_an_acr_short_of_what_was_asked_is_a_step_up(web: WebApp, oidc: OIDC) -> None:
    with pytest.raises(StepUpRequired) as caught:
        _sign_in(web, oidc, acr_values=["urn:authfi:acr:mfa"])
    assert caught.value.acr_values == ("urn:authfi:acr:mfa",)
    oidc.acr = "urn:authfi:acr:mfa"
    assert _sign_in(web, oidc, acr_values=["urn:authfi:acr:mfa"]).user.acr == "urn:authfi:acr:mfa"


def test_max_age_is_checked_against_auth_time(web: WebApp, oidc: OIDC) -> None:
    oidc.auth_age = 3600
    with pytest.raises(StepUpRequired) as caught:
        _sign_in(web, oidc, max_age=300)
    assert caught.value.max_age == 300
    oidc.id_claims = {"auth_time": None}
    with pytest.raises(InvalidIdToken, match="auth_time"):
        _sign_in(web, oidc, max_age=300)
    oidc.id_claims = {}
    oidc.auth_age = 10
    assert _sign_in(web, oidc, max_age=300).user.auth_time is not None


# --- after sign-in --------------------------------------------------------------------------------------


def test_refresh_is_single_flight(web: WebApp, oidc: OIDC) -> None:
    tokens = _sign_in(web, oidc).tokens
    results: list[TokenSet] = []
    barrier = threading.Barrier(4)

    def refresh() -> None:
        barrier.wait()
        results.append(web.refresh(tokens))

    threads = [threading.Thread(target=refresh) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert oidc.refreshes == 1
    assert {r.access_token.reveal() for r in results} == {"at-refreshed-1"}
    assert results[0].refresh_token == Secret("rt-2")
    assert results[0].id_token == tokens.id_token  # kept from the sign-in


def test_a_refresh_answering_an_id_token_must_name_the_same_person(web: WebApp, oidc: OIDC) -> None:
    tokens = _sign_in(web, oidc).tokens
    oidc.refresh_answer = {
        "access_token": "at-r",
        "token_type": "Bearer",
        "expires_in": 900,
        "id_token": oidc.id_token("at-r", None, sub="usr_mallory"),
    }
    with pytest.raises(InvalidIdToken) as caught:
        web.refresh(tokens)
    assert caught.value.code == "id_token_subject_changed"


def test_a_refresh_needing_step_up(af: AuthFI, oidc: OIDC) -> None:
    web = af.web_app(
        client_id=CLIENT,
        client_secret=None,
        redirect_uri=REDIRECT,
        state_key=STATE_KEY,
        step_up_acr=["urn:authfi:acr:mfa"],
    )
    tokens = _sign_in(web, oidc).tokens
    oidc.refresh_answer = {"error": "insufficient_user_authentication", "error_description": "acr urn:authfi:acr:mfa"}
    with pytest.raises(StepUpRequired) as caught:
        web.refresh(tokens)
    assert caught.value.acr_values == ("urn:authfi:acr:mfa",)


def test_a_refresh_needs_a_refresh_token(web: WebApp, oidc: OIDC) -> None:
    import dataclasses

    tokens = dataclasses.replace(_sign_in(web, oidc).tokens, refresh_token=None)
    with pytest.raises(ConfigurationError, match="offline_access"):
        web.refresh(tokens)


def test_userinfo_must_name_the_person_who_signed_in(web: WebApp, oidc: OIDC) -> None:
    tokens = _sign_in(web, oidc).tokens
    info = web.userinfo(tokens)
    assert (info.sub, info.email, info.name) == ("usr_ada", "ada@example.com", "Ada")
    oidc.id_claims = {"userinfo_sub": "usr_mallory"}
    with pytest.raises(InvalidIdToken, match="another person"):
        web.userinfo(tokens)


def test_sign_out_revokes_the_refresh_token_first(web: WebApp, oidc: OIDC, platform: Platform) -> None:
    tokens = _sign_in(web, oidc).tokens

    url = web.sign_out(tokens, post_logout_redirect_uri="https://app.example.com/bye", state="s1")

    assert oidc.revoked == ["rt-1"]
    query = parse_qs(urlsplit(url).query)
    assert url.startswith(platform.issuer + "/oauth/logout?")
    assert query == {
        "client_id": [CLIENT],
        "post_logout_redirect_uri": ["https://app.example.com/bye"],
        "state": ["s1"],
    }
    assert "id_token_hint" not in query


def test_a_failed_revocation_is_reported(web: WebApp, oidc: OIDC, platform: Platform) -> None:
    tokens = _sign_in(web, oidc).tokens
    platform.route("POST", "/oauth/token/revoke", json_response({"error": "server_error"}, 400))
    with pytest.raises(Exception, match="server_error"):
        web.sign_out(tokens)


def test_the_session_refreshes_once_across_workers(web: WebApp, oidc: OIDC, clock: FakeClock) -> None:
    tokens = _sign_in(web, oidc).tokens
    store = InMemoryTokenStore()
    session = web.session(tokens, store=store, key="sess-1", send_to=["https://api.example.com"])

    assert session.token() == tokens.access_token.reveal()
    clock.advance(900)
    assert session.token() == "at-refreshed-1"
    assert session.token() == "at-refreshed-1"
    assert oidc.refreshes == 1
    assert "https://api.example.com" in session.send_to
    assert session.headers("GET", "https://api.example.com/orders") == {"Authorization": "Bearer at-refreshed-1"}


def test_a_session_with_nothing_stored(web: WebApp, oidc: OIDC) -> None:
    tokens = _sign_in(web, oidc).tokens
    store = InMemoryTokenStore()
    session = web.session(tokens, store=store, key="k")
    store._sets.clear()  # the person signed out elsewhere
    with pytest.raises(ConfigurationError, match="sign in again"):
        session.token()


# --- asyncio ----------------------------------------------------------------------------------------------


async def test_asyncio_sign_in_refresh_and_sign_out(platform: Platform, oidc: OIDC, clock: FakeClock) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as af:
        web = af.web_app(client_id=CLIENT, client_secret=Secret("s3cret"), redirect_uri=REDIRECT, state_key=STATE_KEY)
        pending = await web.start_sign_in(max_age=600, acr_values=[oidc.acr])
        signed_in = await web.finish_sign_in(oidc.browser(pending.url), web.load_pending(pending.dump()))
        assert signed_in.user.sub == "usr_ada"

        fresh = await asyncio.gather(*(web.refresh(signed_in.tokens) for _ in range(3)))
        assert oidc.refreshes == 1
        assert {t.access_token.reveal() for t in fresh} == {"at-refreshed-1"}

        assert (await web.userinfo(signed_in.tokens)).sub == "usr_ada"
        store = AsyncInMemoryTokenStore()
        session = await web.session(signed_in.tokens, store=store, key="k")
        clock.advance(900)
        assert await session.token() == "at-refreshed-2"
        url = await web.sign_out(signed_in.tokens)
        assert url.startswith(platform.issuer + "/oauth/logout?client_id=")
        assert oidc.revoked == ["rt-1"]


def test_discovery_without_the_endpoints(platform: Platform, web: WebApp, oidc: OIDC) -> None:
    tokens = _sign_in(web, oidc).tokens
    for name in ("authorization_endpoint", "userinfo_endpoint", "end_session_endpoint", "revocation_endpoint"):
        platform.discovery.pop(name)
    web.project.metadata(refresh=True)
    with pytest.raises(NotSupportedError):
        web.start_sign_in()
    with pytest.raises(NotSupportedError):
        web.userinfo(tokens)
    with pytest.raises(NotSupportedError):
        web.sign_out(tokens)


def test_client_secret_post(af: AuthFI, oidc: OIDC, platform: Platform) -> None:
    web = af.web_app(
        client_id=CLIENT,
        client_secret=Secret("s3cret"),
        redirect_uri=REDIRECT,
        state_key=STATE_KEY,
        auth_method="client_secret_post",
    )
    _sign_in(web, oidc)
    (exchange,) = platform.calls("POST", "/oauth/token")
    assert parse_qs(exchange.content.decode())["client_secret"] == ["s3cret"]
    assert "authorization" not in exchange.headers


def test_userinfo_refusals(web: WebApp, oidc: OIDC, platform: Platform) -> None:
    import dataclasses

    tokens = _sign_in(web, oidc).tokens
    platform.route("GET", "/oauth/userinfo", json_response({"error": "invalid_token"}, 401))
    with pytest.raises(Exception, match="invalid_token"):
        web.userinfo(tokens)
    platform.route("GET", "/oauth/userinfo", json_response(["not", "an", "object"]))
    with pytest.raises(ConfigurationError, match="JSON object"):
        web.userinfo(tokens)
    platform.route("GET", "/oauth/userinfo", json_response({"sub": "anyone"}))
    without_id_token = dataclasses.replace(tokens, id_token=None)
    assert web.userinfo(without_id_token).sub == "anyone"


class LosingStore(InMemoryTokenStore):
    """A store where another worker refreshes first."""

    def __init__(self, winner: TokenSet) -> None:
        super().__init__()
        self.winner = winner
        self.lost = False

    def compare_and_set(self, key: str, old: TokenSet | None, new: TokenSet) -> bool:
        if old is not None and not self.lost:
            self.lost = True
            super().compare_and_set(key, old, self.winner)
            return False
        return super().compare_and_set(key, old, new)


def test_a_session_that_loses_the_race_uses_the_winners_tokens(web: WebApp, oidc: OIDC, clock: FakeClock) -> None:
    import dataclasses

    tokens = _sign_in(web, oidc).tokens
    winner = dataclasses.replace(
        tokens, access_token=Secret("at-winner"), expires_at=tokens.expires_at.replace(year=2100)
    )
    store = LosingStore(winner)
    session = web.session(tokens, store=store, key="k")
    clock.advance(900)
    assert session.token() == "at-winner"
    assert repr(session) == "PersonSession(key='k')"


async def test_asyncio_refusals(platform: Platform, oidc: OIDC, clock: FakeClock) -> None:
    import dataclasses

    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as af:
        web = af.web_app(client_id=CLIENT, client_secret=None, redirect_uri=REDIRECT, state_key=STATE_KEY)
        assert repr(web) == "AsyncWebApp(client_id='app_web')"
        pending = await web.start_sign_in()
        signed_in = await web.finish_sign_in(oidc.browser(pending.url), pending)
        tokens = signed_in.tokens

        with pytest.raises(ConfigurationError, match="offline_access"):
            await web.refresh(dataclasses.replace(tokens, refresh_token=None))
        oidc.refresh_answer = {
            "access_token": "at-r",
            "token_type": "Bearer",
            "id_token": oidc.id_token("at-r", None),
        }
        assert (await web.refresh(tokens)).access_token == Secret("at-r")

        store = AsyncInMemoryTokenStore()
        session = await web.session(tokens, store=store, key="k", send_to=["https://api.example.com"])
        assert "AsyncPersonSession" in repr(session)
        assert await session.token() == tokens.access_token.reveal()
        store._sets.clear()
        with pytest.raises(ConfigurationError, match="sign in again"):
            await session.token()

        platform.route("POST", "/oauth/token/revoke", json_response({"error": "server_error"}, 400))
        with pytest.raises(Exception, match="server_error"):
            await web.sign_out(tokens)

        platform.route("POST", "/oauth/token", json_response({"access_token": "at-1", "token_type": "Bearer"}))
        pending = await web.start_sign_in()
        with pytest.raises(InvalidIdToken, match="openid"):
            await web.finish_sign_in(oidc.browser(pending.url), pending)
