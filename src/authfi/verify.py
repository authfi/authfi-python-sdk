"""Protecting an API: verify the access tokens that reach it.

::

    verifier = af.verifier(audience=["api_orders"])        # your API's client id
    token = verifier.verify_request(request.headers.get("Authorization"))
    token.require_permission("orders:read")

One verifier serves every token that reaches an API: a person's, a machine client's, an agent's,
and a token an agent holds for a person. :class:`AccessToken` gives each the same shape, with a
normalised :class:`Principal` and, for a delegated token, the :attr:`AccessToken.actor`.

Authorization is in ``permissions`` and ``roles``. On a person's or a machine client's token,
``scope`` is what the client asked for; check it only for scopes your API defines itself, and on
agent tokens, whose scope is what they were granted.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, cast

from authfi._protocol.urls import FIRST_PARTY_CLIENTS
from authfi._protocol.verify import (
    Unverified,
    VerifyConfig,
    audiences_of,
    check_claims,
    inspect_token,
    unverified_claims,
    verify_signature,
)
from authfi.errors import (
    ConfigurationError,
    InsufficientAuthentication,
    InsufficientPermission,
    InsufficientScope,
    InvalidToken,
)
from authfi.secret import Secret

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = [
    "FIRST_PARTY_CLIENTS",
    "AccessToken",
    "AsyncMultiIssuerVerifier",
    "AsyncTokenVerifier",
    "CheckResult",
    "MultiIssuerVerifier",
    "Principal",
    "TokenVerifier",
]

PrincipalKind = Literal["user", "app", "agent"]
TokenKind = Literal["person", "machine", "agent", "delegated"]

_log = logging.getLogger("authfi")


@dataclass(frozen=True, slots=True)
class Principal:
    """A party: a person (``user``), a machine client (``app``) or an AI agent (``agent``).

    ``id`` is never typed: ``user:abc`` becomes ``Principal("user", "abc")``. ``str()`` gives the
    typed form back. Compare principals, never ``sub`` strings: tokens of different shapes write
    the same party differently.
    """

    kind: PrincipalKind
    id: str

    def __str__(self) -> str:
        return f"{self.kind}:{self.id}"


_TYPED = {"user:": "user", "app:": "app", "agent:": "agent"}


def _parse_party(value: str, default: PrincipalKind) -> Principal:
    for prefix, kind in _TYPED.items():
        if value.startswith(prefix):
            return Principal(cast(PrincipalKind, kind), value[len(prefix) :])
    if value.startswith("agt_"):
        return Principal("agent", value)
    return Principal(default, value)


def _strings(claims: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = claims.get(name)
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(v for v in cast(list[object], value) if isinstance(v, str))
    return ()


def _time(value: object) -> datetime | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return datetime.fromtimestamp(float(value), UTC)
    return None


def _required(names: tuple[str, ...], what: str) -> tuple[str, ...]:
    out = names
    if not out or not all(isinstance(n, str) and n for n in cast(tuple[object, ...], out)):
        raise ConfigurationError(f"{what} needs at least one name; an empty requirement would require nothing")
    return out


@dataclass(frozen=True, slots=True)
class AccessToken:
    """A verified access token. It keeps no raw token.

    ``principal`` is who the token is for; ``actor`` who is acting for them (an agent) on a
    delegated token. ``token_kind`` names the shape: ``person``, ``machine``, ``agent`` or
    ``delegated``.
    """

    principal: Principal
    actor: Principal | None
    token_kind: TokenKind
    client_id: str
    tenant_id: str
    issuer: str
    audience: tuple[str, ...]
    scopes: frozenset[str]
    permissions: frozenset[str]
    roles: frozenset[str]
    groups: frozenset[str]
    org_id: str | None
    acr: str | None
    amr: tuple[str, ...]
    auth_time: datetime | None
    expires_at: datetime
    jti: str | None
    claims: Mapping[str, Any] = field(repr=False)

    @classmethod
    def from_claims(cls, claims: Mapping[str, Any]) -> AccessToken:
        """The token's shape, read from claims that already verified."""
        sub = str(claims.get("sub", ""))
        client_id = claims.get("client_id")
        client = client_id if isinstance(client_id, str) else ""
        act = claims.get("act")
        actor_sub = cast(Mapping[str, Any], act).get("sub") if isinstance(act, dict) else None
        actor = _parse_party(actor_sub, "app") if isinstance(actor_sub, str) else None
        typed = sub.startswith(("user:", "app:", "agent:"))
        principal: Principal
        kind: TokenKind
        if sub.startswith("user:") or (actor is not None and not typed):
            principal = _parse_party(sub, "user")
            kind = "delegated" if actor is not None else "person"
        elif typed:
            # A typed agent or app token keeps its own kind, with or without an actor.
            principal = _parse_party(sub, "user")
            kind = "agent" if principal.kind == "agent" else "machine"
        elif sub.startswith("agt_"):
            principal, kind = _parse_party(sub, "agent"), "agent"
        elif client and sub == client:
            principal, kind = _parse_party(sub, "app"), "machine"
        else:
            principal, kind = _parse_party(sub, "user"), "person"
        custom = claims.get("custom")
        org = cast(Mapping[str, Any], custom).get("org_id") if isinstance(custom, dict) else None
        scope = claims.get("scope")
        acr = claims.get("acr")
        jti = claims.get("jti")
        expires = _time(claims.get("exp"))
        return cls(
            principal=principal,
            actor=actor,
            token_kind=kind,
            client_id=client,
            tenant_id=str(claims.get("tenant_id", "")),
            issuer=str(claims.get("iss", "")),
            audience=audiences_of(claims),
            scopes=frozenset(scope.split()) if isinstance(scope, str) else frozenset(_strings(claims, "scope")),
            permissions=frozenset(_strings(claims, "permissions")),
            roles=frozenset(_strings(claims, "roles")),
            groups=frozenset(_strings(claims, "groups")),
            org_id=org if isinstance(org, str) else None,
            acr=acr if isinstance(acr, str) else None,
            amr=_strings(claims, "amr"),
            auth_time=_time(claims.get("auth_time")),
            expires_at=expires if expires is not None else datetime.fromtimestamp(0, UTC),
            jti=jti if isinstance(jti, str) else None,
            claims=types.MappingProxyType(dict(claims)),
        )

    def require_permission(self, *names: str) -> None:
        """Every one of ``names`` must be in ``permissions``, or :class:`InsufficientPermission`."""
        required = _required(names, "require_permission")
        missing = [n for n in required if n not in self.permissions]
        if missing:
            raise InsufficientPermission(required, f"missing permission(s) {', '.join(missing)}")

    def require_any_permission(self, *names: str) -> None:
        """At least one of ``names`` must be in ``permissions``."""
        required = _required(names, "require_any_permission")
        if not self.permissions.intersection(required):
            raise InsufficientPermission(required, f"none of the permissions {', '.join(required)}")

    def require_role(self, *names: str) -> None:
        """At least one of ``names`` must be in ``roles``."""
        required = _required(names, "require_role")
        if not self.roles.intersection(required):
            raise InsufficientPermission(required, f"none of the roles {', '.join(required)}")

    def require_scope(self, *names: str) -> None:
        """Every one of ``names`` must be in ``scopes``, or :class:`InsufficientScope`.

        Meaningful for agent tokens and for scopes your API defines; on a person's or a machine
        client's token, ``scope`` is what the client asked for.
        """
        required = _required(names, "require_scope")
        if not set(required) <= self.scopes:
            raise InsufficientScope(required, "missing scope(s)")

    def require_acr(self, *values: str, max_age: int | None = None, now: datetime | None = None) -> None:
        """``acr`` must be one of ``values``, and ``auth_time`` within ``max_age`` seconds when given."""
        required = _required(values, "require_acr")
        if self.acr not in required:
            raise InsufficientAuthentication(required, max_age, f"acr {self.acr!r} is not one of {required}")
        if max_age is not None:
            moment = now or datetime.now(UTC)
            if self.auth_time is None or (moment - self.auth_time).total_seconds() > max_age:
                raise InsufficientAuthentication(required, max_age, "the authentication is too old")

    def require_agent(self) -> None:
        """The token must be an agent's own, or one an agent holds for a person; else ``token_no_agent``."""
        if self.principal.kind == "agent" or (self.actor is not None and self.actor.kind == "agent"):
            return
        raise InvalidToken("token_no_agent", "the token names no agent")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """A live authorization answer (``POST /apis/regional/v1/check``)."""

    allowed: bool
    approval_required: bool
    reason: str | None


def check_audiences(project: AuthFI | AsyncAuthFI, audience: Sequence[str]) -> frozenset[str]:
    """``audience`` as the set a verifier accepts, refusing AuthFI's own audiences."""
    if isinstance(audience, str):
        raise ConfigurationError('audience is a sequence of client ids, such as ["api_orders"], not one string')
    given = cast(Sequence[object], audience)
    values = frozenset(a.strip() for a in given if isinstance(a, str) and a.strip())
    if not values:
        raise ConfigurationError(
            "audience is required: your API's client id (mint machine and agent tokens with resource=<it>), "
            "or the client ids of the front ends whose people call it. There is no way to skip it."
        )
    for value in values:
        if project.names_authfi(value):
            raise ConfigurationError(
                f"audience {value!r} is AuthFI's own: accepting it would let every token meant for the AuthFI "
                "gateway or console be replayed at your API. Mint the token with resource=<your API's client "
                "id> and verify that audience."
            )
    return values


def _scheme(authorization: str | None) -> tuple[str, str]:
    if not authorization or not authorization.strip():
        raise InvalidToken("token_missing", "no Authorization header")
    scheme, _, value = authorization.strip().partition(" ")
    scheme = scheme.lower()
    value = value.strip()
    if scheme not in ("bearer", "dpop") or not value:
        raise InvalidToken("token_missing", "no Bearer or DPoP credential")
    return scheme, value


def _raw(token: Secret | str) -> str:
    return token.reveal() if isinstance(token, Secret) else token


def _log_refused(logger: logging.Logger, error: InvalidToken) -> None:
    """A refusal, with its precise code: info when no token was sent, else warning (as the kernel logs it)."""
    logger.log(logging.INFO if error.code == "token_missing" else logging.WARNING, "refused a token: %s", error.code)


class _VerifierBase:
    def __init__(
        self,
        project: AuthFI | AsyncAuthFI,
        audience: Sequence[str],
        *,
        expected_tenant_id: str | None,
        leeway: float | None,
    ) -> None:
        if leeway is not None and not 0 <= leeway <= 300:
            raise ConfigurationError(f"leeway must be from 0 to 300 seconds, not {leeway!r}")
        if expected_tenant_id is not None and not expected_tenant_id.strip():
            raise ConfigurationError(
                "expected_tenant_id must not be empty: leave it out for no pin. An empty value read from the "
                "environment would otherwise switch the tenant check off."
            )
        self._config = VerifyConfig(
            issuer=project.issuer,
            audiences=check_audiences(project, audience),
            leeway=project.leeway if leeway is None else float(leeway),
            expected_tenant_id=expected_tenant_id,
        )
        self._clock = project.clock
        self._log = project.logger
        self._origin_checked = False

    def _check_learned_origin(self, project: AuthFI | AsyncAuthFI) -> None:
        """Once discovery has taught the API origin, refuse an audience under it, as construction could not."""
        if not self._origin_checked:
            check_audiences(project, sorted(self._config.audiences))
            self._origin_checked = True

    @property
    def issuer(self) -> str:
        """The one issuer this verifier accepts."""
        return self._config.issuer

    @property
    def audiences(self) -> frozenset[str]:
        """The audiences this verifier accepts."""
        return self._config.audiences

    @property
    def expected_tenant_id(self) -> str | None:
        """The tenant this verifier is pinned to, if any."""
        return self._config.expected_tenant_id

    @property
    def logger(self) -> logging.Logger:
        """The project's logger, where refusals are recorded with their codes."""
        return self._log

    @staticmethod
    def _dpop_refusal(claims: Mapping[str, Any]) -> InvalidToken:
        if "cnf" in claims or "authfi_mandate" in claims:
            return InvalidToken("token_dpop_unsupported", "this verifier checks no DPoP proof yet")
        return InvalidToken("token_dpop_unbound", "the DPoP scheme with a token bound to no key")

    def _refused(self, error: InvalidToken) -> InvalidToken:
        _log_refused(self._log, error)
        return error

    def __repr__(self) -> str:
        return f"{type(self).__name__}(issuer={self.issuer!r}, audience={sorted(self.audiences)!r})"


class TokenVerifier(_VerifierBase):
    """Verifies access tokens for one issuer, in blocking code. Make it with :meth:`AuthFI.verifier`."""

    def __init__(
        self,
        project: AuthFI,
        audience: Sequence[str],
        *,
        expected_tenant_id: str | None = None,
        leeway: float | None = None,
    ) -> None:
        super().__init__(project, audience, expected_tenant_id=expected_tenant_id, leeway=leeway)
        self._project = project

    def verify(self, raw: Secret | str) -> AccessToken:
        """Verify a token presented with the Bearer scheme."""
        return self._verify(_raw(raw), self._config, bearer=True)

    def verify_request(
        self,
        authorization: str | None,
        *,
        dpop: str | None = None,
        method: str | None = None,
        url: str | None = None,
    ) -> AccessToken:
        """Verify a request's ``Authorization`` header (``Bearer`` or ``DPoP``, any case).

        ``dpop``, ``method`` and ``url`` are the DPoP proof and what it must bind. A token
        presented with the DPoP scheme is refused (``token_dpop_unsupported``) until this verifier
        checks proofs.
        """
        return self._request(authorization, self._config)

    def _request(self, authorization: str | None, config: VerifyConfig) -> AccessToken:
        try:
            scheme, raw = _scheme(authorization)
        except InvalidToken as error:
            raise self._refused(error) from None
        if scheme == "bearer":
            return self._verify(raw, config, bearer=True)
        claims = self._claims(raw, config, bearer=False)
        raise self._refused(self._dpop_refusal(claims))

    def _verify(self, raw: str, config: VerifyConfig, *, bearer: bool) -> AccessToken:
        return AccessToken.from_claims(self._claims(raw, config, bearer=bearer))

    def _claims(self, raw: str, config: VerifyConfig, *, bearer: bool) -> Mapping[str, Any]:
        try:
            token: Unverified = inspect_token(raw)
            self._project.resolve_api_origin()
            self._check_learned_origin(self._project)
            claims = verify_signature(token, self._project._key(token.kid))  # pyright: ignore[reportPrivateUsage]
            check_claims(claims, config, self._clock.time(), bearer=bearer)
        except InvalidToken as error:
            raise self._refused(error) from None
        return claims

    def pinned(self, tenant_id: str) -> TokenVerifier:
        """This verifier, pinned to ``tenant_id``."""
        clone = object.__new__(TokenVerifier)
        clone.__dict__.update(self.__dict__)
        clone._config = VerifyConfig(self._config.issuer, self._config.audiences, self._config.leeway, tenant_id)
        return clone


class AsyncTokenVerifier(_VerifierBase):
    """Verifies access tokens for one issuer, in asyncio code. Make it with :meth:`AsyncAuthFI.verifier`."""

    def __init__(
        self,
        project: AsyncAuthFI,
        audience: Sequence[str],
        *,
        expected_tenant_id: str | None = None,
        leeway: float | None = None,
    ) -> None:
        super().__init__(project, audience, expected_tenant_id=expected_tenant_id, leeway=leeway)
        self._project = project

    async def verify(self, raw: Secret | str) -> AccessToken:
        """See :meth:`TokenVerifier.verify`."""
        return await self._verify(_raw(raw), self._config, bearer=True)

    async def verify_request(
        self,
        authorization: str | None,
        *,
        dpop: str | None = None,
        method: str | None = None,
        url: str | None = None,
    ) -> AccessToken:
        """See :meth:`TokenVerifier.verify_request`."""
        return await self._request(authorization, self._config)

    async def _request(self, authorization: str | None, config: VerifyConfig) -> AccessToken:
        try:
            scheme, raw = _scheme(authorization)
        except InvalidToken as error:
            raise self._refused(error) from None
        if scheme == "bearer":
            return await self._verify(raw, config, bearer=True)
        claims = await self._claims(raw, config, bearer=False)
        raise self._refused(self._dpop_refusal(claims))

    async def _verify(self, raw: str, config: VerifyConfig, *, bearer: bool) -> AccessToken:
        return AccessToken.from_claims(await self._claims(raw, config, bearer=bearer))

    async def _claims(self, raw: str, config: VerifyConfig, *, bearer: bool) -> Mapping[str, Any]:
        try:
            token = inspect_token(raw)
            await self._project.resolve_api_origin()
            self._check_learned_origin(self._project)
            key = await self._project._key(token.kid)  # pyright: ignore[reportPrivateUsage]
            claims = verify_signature(token, key)
            check_claims(claims, config, self._clock.time(), bearer=bearer)
        except InvalidToken as error:
            raise self._refused(error) from None
        return claims

    def pinned(self, tenant_id: str) -> AsyncTokenVerifier:
        """This verifier, pinned to ``tenant_id``."""
        clone = object.__new__(AsyncTokenVerifier)
        clone.__dict__.update(self.__dict__)
        clone._config = VerifyConfig(self._config.issuer, self._config.audiences, self._config.leeway, tenant_id)
        return clone


def _issuer_of(raw: str) -> str:
    claims = unverified_claims(raw)
    iss = claims.get("iss")
    if not isinstance(iss, str):
        raise InvalidToken("token_issuer_not_accepted", "the token names no issuer")
    return iss


def _pinned_entries(
    entries: Mapping[str, tuple[Any, str]],
) -> dict[str, Any]:
    if not entries:
        raise ConfigurationError("MultiIssuerVerifier needs at least one issuer")
    out: dict[str, Any] = {}
    for issuer, given in entries.items():
        entry = cast(object, given)
        if not isinstance(entry, tuple) or len(cast(tuple[object, ...], entry)) != 2:
            raise ConfigurationError(f"{issuer}: give (verifier, tenant_id); each issuer is pinned to its tenant")
        verifier, tenant = cast(tuple[Any, object], entry)
        if not isinstance(tenant, str) or not tenant:
            raise ConfigurationError(f"{issuer}: a tenant id is required for every issuer")
        if verifier.issuer != issuer:
            raise ConfigurationError(f"the verifier filed under {issuer} verifies {verifier.issuer}")
        if verifier.expected_tenant_id not in (None, tenant):
            raise ConfigurationError(f"{issuer}: the verifier is pinned to another tenant than {tenant}")
        out[issuer] = verifier.pinned(tenant)
    return out


class MultiIssuerVerifier:
    """Verifies tokens from an allow-list of issuers, each pinned to its tenant (blocking code).

    ::

        verifier = MultiIssuerVerifier({
            "https://acme.authfi.io": (acme.verifier(audience=["api_orders"]), "4e46e0c4-..."),
            "https://globex.authfi.io": (globex.verifier(audience=["api_orders"]), "9a1d..."),
        })

    The verifier is chosen by the token's unverified ``iss``, only from the list; a token whose
    ``tenant_id`` is not the tenant held for its issuer is refused (``token_tenant_not_issuers``).
    """

    def __init__(self, entries: Mapping[str, tuple[TokenVerifier, str]]) -> None:
        self._verifiers: dict[str, TokenVerifier] = _pinned_entries(entries)

    @property
    def logger(self) -> logging.Logger:
        """The logger of the first issuer's project."""
        return next(iter(self._verifiers.values())).logger

    @property
    def issuers(self) -> tuple[str, ...]:
        """The accepted issuers."""
        return tuple(self._verifiers)

    def _pick(self, raw: str) -> TokenVerifier:
        try:
            verifier = self._verifiers.get(_issuer_of(raw))
            if verifier is None:
                raise InvalidToken("token_issuer_not_accepted", "the token's issuer is not on the allow-list")
        except InvalidToken as error:
            _log_refused(self.logger, error)
            raise
        return verifier

    def verify(self, raw: Secret | str) -> AccessToken:
        """See :meth:`TokenVerifier.verify`."""
        value = _raw(raw)
        return self._pick(value).verify(value)

    def verify_request(
        self,
        authorization: str | None,
        *,
        dpop: str | None = None,
        method: str | None = None,
        url: str | None = None,
    ) -> AccessToken:
        """See :meth:`TokenVerifier.verify_request`."""
        try:
            _, raw = _scheme(authorization)
        except InvalidToken as error:
            _log_refused(self.logger, error)
            raise
        return self._pick(raw).verify_request(authorization, dpop=dpop, method=method, url=url)


class AsyncMultiIssuerVerifier:
    """See :class:`MultiIssuerVerifier`; for asyncio code."""

    def __init__(self, entries: Mapping[str, tuple[AsyncTokenVerifier, str]]) -> None:
        self._verifiers: dict[str, AsyncTokenVerifier] = _pinned_entries(entries)

    @property
    def logger(self) -> logging.Logger:
        """The logger of the first issuer's project."""
        return next(iter(self._verifiers.values())).logger

    @property
    def issuers(self) -> tuple[str, ...]:
        """The accepted issuers."""
        return tuple(self._verifiers)

    def _pick(self, raw: str) -> AsyncTokenVerifier:
        try:
            verifier = self._verifiers.get(_issuer_of(raw))
            if verifier is None:
                raise InvalidToken("token_issuer_not_accepted", "the token's issuer is not on the allow-list")
        except InvalidToken as error:
            _log_refused(self.logger, error)
            raise
        return verifier

    async def verify(self, raw: Secret | str) -> AccessToken:
        """See :meth:`TokenVerifier.verify`."""
        value = _raw(raw)
        return await self._pick(value).verify(value)

    async def verify_request(
        self,
        authorization: str | None,
        *,
        dpop: str | None = None,
        method: str | None = None,
        url: str | None = None,
    ) -> AccessToken:
        """See :meth:`TokenVerifier.verify_request`."""
        try:
            _, raw = _scheme(authorization)
        except InvalidToken as error:
            _log_refused(self.logger, error)
            raise
        return await self._pick(raw).verify_request(authorization, dpop=dpop, method=method, url=url)
