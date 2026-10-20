"""People signing in to a server-rendered web app: OpenID Connect code flow with PKCE.

::

    web = af.web_app(
        client_id="app_web",
        client_secret=Secret(os.environ["AUTHFI_CLIENT_SECRET"]),
        redirect_uri="https://app.example.com/callback",
        state_key=Secret(os.environ["AUTHFI_STATE_KEY"]),   # 32 random bytes, base64
    )

    pending = web.start_sign_in()                  # on /login
    response.set_cookie("authfi_pending", pending.dump(), httponly=True, secure=True, samesite="lax")
    redirect(pending.url)

    pending = web.load_pending(request.cookies["authfi_pending"])     # on /callback
    signed_in = web.finish_sign_in(str(request.url), pending)
    signed_in.user.sub, signed_in.user.email, signed_in.tokens

The pending sign-in holds the PKCE verifier and the nonce. ``dump()`` encrypts it (AES-256-GCM
under ``state_key``, bound to the client and redirect URI, valid for ten minutes), so it is safe in
any cookie or session; a signed-only cookie session would hand the verifier to the browser.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast

from authfi._protocol.discovery import ServerMetadata
from authfi._protocol.errors import api_error
from authfi._protocol.http import HttpRequest, HttpResponse, Retry
from authfi._protocol.oauth import (
    ClientAuth,
    authorization_code_request,
    parse_token_response,
    refresh_request,
    revocation_request,
)
from authfi._protocol.signin import (
    IdTokenClaims,
    PendingState,
    authorize_url,
    check_redirect_uri,
    end_session_url,
    inspect_id_token,
    new_pending,
    read_callback,
    seal,
    state_key_bytes,
    unseal,
    verify_id_token,
)
from authfi._protocol.verify import unverified_claims
from authfi.errors import APIError, ConfigurationError, InvalidIdToken, NotSupportedError
from authfi.secret import Secret
from authfi.tokens import (
    AsyncRefreshGate,
    AsyncTokenSource,
    AsyncTokenStore,
    Destinations,
    RefreshGate,
    TokenSet,
    TokenSource,
    TokenStore,
)

if TYPE_CHECKING:
    from authfi._protocol.oauth import TokenResponse
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = [
    "AsyncPersonSession",
    "AsyncWebApp",
    "IdTokenClaims",
    "PendingSignIn",
    "PersonSession",
    "SignedIn",
    "UserInfo",
    "WebApp",
]

Prompt = Literal["login", "consent"]

#: A person's access token is renewed this many seconds before it expires.
REFRESH_SKEW = 60.0


class PendingSignIn:
    """A sign-in waiting for its callback. ``url`` is where to send the browser.

    Keep it between the two requests with :meth:`dump`, which encrypts it; never as a dict in a
    signed cookie. It does not print, pickle or compare its secrets.
    """

    __slots__ = ("_client_id", "_key", "_redirect_uri", "_state", "url")

    def __init__(self, state: PendingState, url: str, *, key: Secret, client_id: str, redirect_uri: str) -> None:
        self._state = state
        self.url = url
        self._key = key
        self._client_id = client_id
        self._redirect_uri = redirect_uri

    @property
    def state(self) -> str:
        """The ``state`` the callback must carry."""
        return self._state.state

    def dump(self) -> str:
        """An opaque string, encrypted under the app's ``state_key``: safe in a cookie or a session."""
        return seal(self._state, state_key_bytes(self._key), self._client_id, self._redirect_uri)

    @classmethod
    def load(cls, blob: str, *, key: Secret, client_id: str, redirect_uri: str, now: float) -> PendingSignIn:
        """The pending sign-in in ``blob``.

        Raises :class:`~authfi.InvalidPendingSignIn` when it was tampered with, made for another
        client or redirect URI, or is older than ten minutes.
        """
        state = unseal(blob, state_key_bytes(key), client_id, redirect_uri, now)
        return cls(state, "", key=key, client_id=client_id, redirect_uri=redirect_uri)

    def __repr__(self) -> str:
        return f"PendingSignIn(client_id={self._client_id!r})"

    def __reduce__(self) -> NoReturn:
        raise TypeError("A PendingSignIn cannot be pickled: keep pending.dump(), which is encrypted.")


@dataclass(frozen=True, slots=True)
class SignedIn:
    """A finished sign-in: the person's tokens and who they are (from the verified ID token)."""

    tokens: TokenSet
    user: IdTokenClaims


@dataclass(frozen=True, slots=True)
class UserInfo:
    """The person's claims from the userinfo endpoint. ``raw`` holds every claim."""

    sub: str
    email: str | None
    email_verified: bool | None
    name: str | None
    raw: Mapping[str, Any]

    @classmethod
    def of(cls, claims: Mapping[str, Any]) -> UserInfo:
        """The typed view of a userinfo answer."""
        email, name, verified = claims.get("email"), claims.get("name"), claims.get("email_verified")
        return cls(
            sub=str(claims.get("sub")),
            email=email if isinstance(email, str) else None,
            email_verified=verified if isinstance(verified, bool) else None,
            name=name if isinstance(name, str) else None,
            raw=dict(claims),
        )


class _Config:
    def __init__(
        self,
        project: AuthFI | AsyncAuthFI,
        client_id: str,
        client_secret: Secret | None,
        redirect_uri: str,
        state_key: Secret,
        auth_method: str,
        step_up_acr: Sequence[str],
    ) -> None:
        self.project = project
        self.client_id = str(client_id or "").strip()
        if not self.client_id:
            raise ConfigurationError("client_id is required: the app's client id (app_...)")
        self.redirect_uri = check_redirect_uri(redirect_uri)
        state_key_bytes(state_key)  # refuse a key of the wrong length now; the bytes are derived where used
        self.state_key = state_key
        if client_secret is None:
            self.auth = ClientAuth.none(self.client_id)
        elif auth_method == "client_secret_basic":
            self.auth = ClientAuth.secret_basic(self.client_id, client_secret)
        elif auth_method == "client_secret_post":
            self.auth = ClientAuth.secret_post(self.client_id, client_secret)
        else:
            raise ConfigurationError(
                f"auth_method must be client_secret_basic or client_secret_post, not {auth_method!r}"
            )
        if isinstance(cast(object, step_up_acr), str):
            raise ConfigurationError("step_up_acr is a sequence of acr values, not one string")
        self.step_up_acr = tuple(step_up_acr)

    def start(
        self,
        metadata: ServerMetadata,
        scope: Sequence[str],
        acr_values: Sequence[str] | None,
        max_age: int | None,
        login_hint: str | None,
        prompt: str | None,
    ) -> PendingSignIn:
        if metadata.authorization_endpoint is None:
            raise NotSupportedError(
                "authorization_code", "This AuthFI project's discovery names no authorization endpoint."
            )
        if isinstance(cast(object, acr_values), str):
            raise ConfigurationError("acr_values is a sequence of acr values, not one string")
        state = new_pending(self.project.clock.time(), acr_values or (), max_age)
        url = authorize_url(
            metadata.authorization_endpoint,
            client_id=self.client_id,
            redirect_uri=self.redirect_uri,
            pending=state,
            scope=scope,
            login_hint=login_hint,
            prompt=prompt,
        )
        return PendingSignIn(state, url, key=self.state_key, client_id=self.client_id, redirect_uri=self.redirect_uri)

    def load(self, blob: str) -> PendingSignIn:
        return PendingSignIn.load(
            blob,
            key=self.state_key,
            client_id=self.client_id,
            redirect_uri=self.redirect_uri,
            now=self.project.clock.time(),
        )

    def code(self, callback: Mapping[str, str] | str, pending: PendingSignIn) -> str:
        return read_callback(callback, issuer=self.project.issuer, state=pending.state)

    def revocation(self, metadata: ServerMetadata, tokens: TokenSet) -> HttpRequest:
        """Revoke the refresh token; with none, the access token, under its own hint."""
        if metadata.revocation_endpoint is None:
            raise NotSupportedError("revocation", "This AuthFI project's discovery names no revocation endpoint.")
        if tokens.refresh_token is not None:
            return revocation_request(metadata.revocation_endpoint, self.auth, token=tokens.refresh_token)
        return revocation_request(
            metadata.revocation_endpoint, self.auth, token=tokens.access_token, token_type_hint="access_token"
        )

    def end_session(self, metadata: ServerMetadata, post_logout_redirect_uri: str | None, state: str | None) -> str:
        if metadata.end_session_endpoint is None:
            raise NotSupportedError("end_session", "This AuthFI project's discovery names no end-session endpoint.")
        if post_logout_redirect_uri is not None:
            check_redirect_uri(post_logout_redirect_uri)
        return end_session_url(
            metadata.end_session_endpoint,
            client_id=self.client_id,
            post_logout_redirect_uri=post_logout_redirect_uri,
            state=state,
        )

    def userinfo_request(self, metadata: ServerMetadata, tokens: TokenSet) -> HttpRequest:
        if metadata.userinfo_endpoint is None:
            raise NotSupportedError("userinfo", "This AuthFI project's discovery names no userinfo endpoint.")
        return HttpRequest(
            "GET",
            metadata.userinfo_endpoint,
            what="the userinfo endpoint",
            headers={"Accept": "application/json", "Authorization": f"Bearer {tokens.access_token.reveal()}"},
            retry=Retry.SAFE,
        )

    def userinfo(self, response: HttpResponse, tokens: TokenSet) -> UserInfo:
        if response.status != 200:
            raise api_error(response, "the userinfo endpoint")
        body = response.json_object()
        if body is None:
            raise APIError("invalid_response", "the userinfo endpoint did not answer a JSON object", response.status)
        info = UserInfo.of(body)
        signed_in = _subject(tokens)
        if signed_in is not None and info.sub != signed_in:
            raise InvalidIdToken("userinfo_subject_mismatch", "the userinfo answer names another person")
        return info


def _subject(tokens: TokenSet) -> str | None:
    if tokens.id_token is None:
        return None
    sub = unverified_claims(tokens.id_token.reveal()).get("sub")  # verified when the set was made
    return sub if isinstance(sub, str) else None


def _needs_refresh(tokens: TokenSet) -> Secret:
    if tokens.refresh_token is None:
        raise ConfigurationError("this TokenSet has no refresh token: ask for the offline_access scope when signing in")
    return tokens.refresh_token


class WebApp:
    """A server-rendered web app signing people in, for blocking code. Make it with :meth:`AuthFI.web_app`."""

    def __init__(
        self,
        project: AuthFI,
        *,
        client_id: str,
        client_secret: Secret | None,
        redirect_uri: str,
        state_key: Secret,
        auth_method: Literal["client_secret_basic", "client_secret_post"] = "client_secret_basic",
        step_up_acr: Sequence[str] = (),
    ) -> None:
        self._config = _Config(project, client_id, client_secret, redirect_uri, state_key, auth_method, step_up_acr)
        self.project = project
        self._gate = RefreshGate(project.clock)

    @property
    def client_id(self) -> str:
        """The app's client id."""
        return self._config.client_id

    def start_sign_in(
        self,
        scope: Sequence[str] = ("openid", "profile", "email"),
        *,
        acr_values: Sequence[str] | None = None,
        max_age: int | None = None,
        login_hint: str | None = None,
        prompt: Prompt | None = None,
    ) -> PendingSignIn:
        """A new sign-in: PKCE S256, ``state`` and ``nonce``. Send the browser to ``pending.url``."""
        return self._config.start(self.project.metadata(), scope, acr_values, max_age, login_hint, prompt)

    def load_pending(self, blob: str) -> PendingSignIn:
        """The pending sign-in ``dump()`` made, checked as :meth:`PendingSignIn.load` does."""
        return self._config.load(blob)

    def finish_sign_in(self, callback: Mapping[str, str] | str, pending: PendingSignIn) -> SignedIn:
        """Finish the sign-in from its callback (the URL, or its query parameters).

        Requires ``iss`` and checks it (RFC 9207), checks ``state``, exchanges the code with the PKCE
        verifier, and verifies the ID token: alg from discovery, ``aud``, ``nonce``, ``at_hash``,
        ``max_age`` and ``acr``. An ``acr`` short of what was asked raises :class:`~authfi.StepUpRequired`.
        """
        code = self._config.code(callback, pending)
        metadata = self.project.metadata()
        request = authorization_code_request(
            metadata.token_endpoint,
            self._config.auth,
            code=code,
            redirect_uri=self._config.redirect_uri,
            code_verifier=pending._state.verifier,  # pyright: ignore[reportPrivateUsage]
        )
        response = self._exchange(request)
        tokens = TokenSet.from_response(response, self.project.clock.time())
        user = self._verify(metadata, response, nonce=pending._state.nonce, pending=pending._state)  # pyright: ignore[reportPrivateUsage]
        self.project.logger.info("signed in %s to %s", user.sub, self.client_id)
        return SignedIn(tokens, user)

    def _exchange(self, request: HttpRequest) -> TokenResponse:
        transport = self.project._transport  # pyright: ignore[reportPrivateUsage]
        return parse_token_response(transport.send(request), "the token endpoint", step_up_acr=self._config.step_up_acr)

    def _verify(
        self,
        metadata: ServerMetadata,
        response: TokenResponse,
        *,
        nonce: Secret | None,
        pending: PendingState | None = None,
        subject: str | None = None,
    ) -> IdTokenClaims:
        if response.id_token is None:
            raise InvalidIdToken(
                "id_token_missing", "the token endpoint answered no ID token: ask for the openid scope"
            )
        token = inspect_id_token(response.id_token.reveal(), metadata.id_token_signing_alg_values_supported)
        key = self.project._key(token.kid)  # pyright: ignore[reportPrivateUsage]
        return verify_id_token(
            token,
            key,
            issuer=self.project.issuer,
            client_id=self.client_id,
            now=self.project.clock.time(),
            leeway=self.project.leeway,
            nonce=nonce,
            access_token=response.access_token,
            max_age=pending.max_age if pending else None,
            acr_values=pending.acr_values if pending else (),
            subject=subject,
        )

    def refresh(self, tokens: TokenSet) -> TokenSet:
        """New tokens for ``tokens``: one refresh per refresh token across threads, its result shared.

        A refresh token is single-use; a second use revokes its family and signs the person out, so
        concurrent and late callers get the first refresh's result. Across workers, use
        :meth:`session` with a shared :class:`~authfi.TokenStore`.
        """
        refresh_token = _needs_refresh(tokens)

        def refresh() -> TokenSet:
            metadata = self.project.metadata()
            request = refresh_request(metadata.token_endpoint, self._config.auth, refresh_token=refresh_token)
            response = self._exchange(request)
            if response.id_token is not None:
                self._verify(metadata, response, nonce=None, subject=_subject(tokens))
            return TokenSet.from_response(response, self.project.clock.time(), previous=tokens)

        return self._gate.run(refresh_token, refresh)

    def userinfo(self, tokens: TokenSet) -> UserInfo:
        """The person's claims from the userinfo endpoint; it must name the person who signed in."""
        metadata = self.project.metadata()
        transport = self.project._transport  # pyright: ignore[reportPrivateUsage]
        return self._config.userinfo(transport.send(self._config.userinfo_request(metadata, tokens)), tokens)

    def sign_out(
        self, tokens: TokenSet, *, post_logout_redirect_uri: str | None = None, state: str | None = None
    ) -> str:
        """Revoke the refresh token, then return the end-session URL to send the browser to.

        Ending the session at AuthFI does not revoke the refresh token, so it is revoked here first;
        a person with no refresh token has the access token revoked. A project whose discovery names no
        revocation endpoint raises :class:`~authfi.NotSupportedError` rather than sign the person out
        with their tokens still alive.
        """
        metadata = self.project.metadata()
        url = self._config.end_session(metadata, post_logout_redirect_uri, state)
        request = self._config.revocation(metadata, tokens)
        response = self.project._transport.send(request)  # pyright: ignore[reportPrivateUsage]
        if response.status != 200:
            raise api_error(response, "the revocation endpoint")
        return url

    def session(self, tokens: TokenSet, *, store: TokenStore, key: str, send_to: Sequence[str] = ()) -> PersonSession:
        """The person's access token as a :class:`~authfi.TokenSource`, renewed through ``store``.

        ``tokens`` is stored under ``key`` unless something already is. ``send_to`` names the
        origins of the APIs the token is for.
        """
        store.compare_and_set(key, None, tokens)
        return PersonSession(self, store, key, send_to)

    def __repr__(self) -> str:
        return f"WebApp(client_id={self.client_id!r})"


class PersonSession(TokenSource):
    """A signed-in person's access token, renewed before it expires and shared through a TokenStore."""

    __slots__ = ("_destinations", "_key", "_store", "_web")

    def __init__(self, web: WebApp, store: TokenStore, key: str, send_to: Sequence[str]) -> None:
        self._web = web
        self._store = store
        self._key = key
        self._destinations = Destinations(web.project, None, send_to)

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`."""
        return self._destinations.origins

    def token(self, *, force: bool = False) -> str:
        """The person's current access token; refreshed, once across workers, when it is about to expire."""
        current = self._store.load(self._key)
        if current is None:
            raise ConfigurationError(f"no tokens are stored under {self._key!r}: the person must sign in again")
        if not force and not current.expires_within(REFRESH_SKEW, self._web.project.clock.time()):
            return current.access_token.reveal()
        fresh = self._web.refresh(current)
        if not self._store.compare_and_set(self._key, current, fresh):
            winner = self._store.load(self._key)
            if winner is not None:
                return winner.access_token.reveal()
        return fresh.access_token.reveal()

    def __repr__(self) -> str:
        return f"PersonSession(key={self._key!r})"


class AsyncWebApp:
    """A server-rendered web app signing people in, for asyncio code. Make it with :meth:`AsyncAuthFI.web_app`."""

    def __init__(
        self,
        project: AsyncAuthFI,
        *,
        client_id: str,
        client_secret: Secret | None,
        redirect_uri: str,
        state_key: Secret,
        auth_method: Literal["client_secret_basic", "client_secret_post"] = "client_secret_basic",
        step_up_acr: Sequence[str] = (),
    ) -> None:
        self._config = _Config(project, client_id, client_secret, redirect_uri, state_key, auth_method, step_up_acr)
        self.project = project
        self._gate = AsyncRefreshGate(project.clock)

    @property
    def client_id(self) -> str:
        """The app's client id."""
        return self._config.client_id

    async def start_sign_in(
        self,
        scope: Sequence[str] = ("openid", "profile", "email"),
        *,
        acr_values: Sequence[str] | None = None,
        max_age: int | None = None,
        login_hint: str | None = None,
        prompt: Prompt | None = None,
    ) -> PendingSignIn:
        """See :meth:`WebApp.start_sign_in`."""
        return self._config.start(await self.project.metadata(), scope, acr_values, max_age, login_hint, prompt)

    def load_pending(self, blob: str) -> PendingSignIn:
        """See :meth:`WebApp.load_pending`."""
        return self._config.load(blob)

    async def finish_sign_in(self, callback: Mapping[str, str] | str, pending: PendingSignIn) -> SignedIn:
        """See :meth:`WebApp.finish_sign_in`."""
        code = self._config.code(callback, pending)
        metadata = await self.project.metadata()
        request = authorization_code_request(
            metadata.token_endpoint,
            self._config.auth,
            code=code,
            redirect_uri=self._config.redirect_uri,
            code_verifier=pending._state.verifier,  # pyright: ignore[reportPrivateUsage]
        )
        response = await self._exchange(request)
        tokens = TokenSet.from_response(response, self.project.clock.time())
        user = await self._verify(metadata, response, nonce=pending._state.nonce, pending=pending._state)  # pyright: ignore[reportPrivateUsage]
        self.project.logger.info("signed in %s to %s", user.sub, self.client_id)
        return SignedIn(tokens, user)

    async def _exchange(self, request: HttpRequest) -> TokenResponse:
        transport = self.project._transport  # pyright: ignore[reportPrivateUsage]
        return parse_token_response(
            await transport.send(request), "the token endpoint", step_up_acr=self._config.step_up_acr
        )

    async def _verify(
        self,
        metadata: ServerMetadata,
        response: TokenResponse,
        *,
        nonce: Secret | None,
        pending: PendingState | None = None,
        subject: str | None = None,
    ) -> IdTokenClaims:
        if response.id_token is None:
            raise InvalidIdToken(
                "id_token_missing", "the token endpoint answered no ID token: ask for the openid scope"
            )
        token = inspect_id_token(response.id_token.reveal(), metadata.id_token_signing_alg_values_supported)
        key = await self.project._key(token.kid)  # pyright: ignore[reportPrivateUsage]
        return verify_id_token(
            token,
            key,
            issuer=self.project.issuer,
            client_id=self.client_id,
            now=self.project.clock.time(),
            leeway=self.project.leeway,
            nonce=nonce,
            access_token=response.access_token,
            max_age=pending.max_age if pending else None,
            acr_values=pending.acr_values if pending else (),
            subject=subject,
        )

    async def refresh(self, tokens: TokenSet) -> TokenSet:
        """See :meth:`WebApp.refresh`; one refresh per refresh token across tasks."""
        refresh_token = _needs_refresh(tokens)

        async def refresh() -> TokenSet:
            metadata = await self.project.metadata()
            request = refresh_request(metadata.token_endpoint, self._config.auth, refresh_token=refresh_token)
            response = await self._exchange(request)
            if response.id_token is not None:
                await self._verify(metadata, response, nonce=None, subject=_subject(tokens))
            return TokenSet.from_response(response, self.project.clock.time(), previous=tokens)

        return await self._gate.run(refresh_token, refresh)

    async def userinfo(self, tokens: TokenSet) -> UserInfo:
        """See :meth:`WebApp.userinfo`."""
        metadata = await self.project.metadata()
        transport = self.project._transport  # pyright: ignore[reportPrivateUsage]
        return self._config.userinfo(await transport.send(self._config.userinfo_request(metadata, tokens)), tokens)

    async def sign_out(
        self, tokens: TokenSet, *, post_logout_redirect_uri: str | None = None, state: str | None = None
    ) -> str:
        """See :meth:`WebApp.sign_out`."""
        metadata = await self.project.metadata()
        url = self._config.end_session(metadata, post_logout_redirect_uri, state)
        request = self._config.revocation(metadata, tokens)
        response = await self.project._transport.send(request)  # pyright: ignore[reportPrivateUsage]
        if response.status != 200:
            raise api_error(response, "the revocation endpoint")
        return url

    async def session(
        self, tokens: TokenSet, *, store: AsyncTokenStore, key: str, send_to: Sequence[str] = ()
    ) -> AsyncPersonSession:
        """See :meth:`WebApp.session`."""
        await store.compare_and_set(key, None, tokens)
        return AsyncPersonSession(self, store, key, send_to)

    def __repr__(self) -> str:
        return f"AsyncWebApp(client_id={self.client_id!r})"


class AsyncPersonSession(AsyncTokenSource):
    """See :class:`PersonSession`."""

    __slots__ = ("_destinations", "_key", "_store", "_web")

    def __init__(self, web: AsyncWebApp, store: AsyncTokenStore, key: str, send_to: Sequence[str]) -> None:
        self._web = web
        self._store = store
        self._key = key
        self._destinations = Destinations(web.project, None, send_to)

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`."""
        return self._destinations.origins

    async def token(self, *, force: bool = False) -> str:
        """See :meth:`PersonSession.token`."""
        current = await self._store.load(self._key)
        if current is None:
            raise ConfigurationError(f"no tokens are stored under {self._key!r}: the person must sign in again")
        if not force and not current.expires_within(REFRESH_SKEW, self._web.project.clock.time()):
            return current.access_token.reveal()
        fresh = await self._web.refresh(current)
        if not await self._store.compare_and_set(self._key, current, fresh):
            winner = await self._store.load(self._key)
            if winner is not None:
                return winner.access_token.reveal()
        return fresh.access_token.reveal()

    def __repr__(self) -> str:
        return f"AsyncPersonSession(key={self._key!r})"
