"""What ``main``'s single-module SDK offered, kept working for the 0.x series with warnings.

Only names that ``main`` exported get a path here. Each is typed, marked ``@deprecated`` (so type
checkers and IDEs flag it) and warns :class:`~authfi.errors.AuthFIDeprecationWarning` once per
call site. Insecure behaviour is not kept: a token is never verified without an audience, and a
retired route is never called.
"""

from __future__ import annotations

import functools
import types
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NoReturn, ParamSpec, TypeVar, cast

from typing_extensions import deprecated

from authfi.errors import AuthFIDeprecationWarning, ConfigurationError, NotSupportedError
from authfi.secret import Secret

if TYPE_CHECKING:
    from authfi.flask import AuthFIFlask
    from authfi.verify import TokenVerifier

__all__ = ["LEGACY_KEYWORDS", "LegacyAuthFIMixin", "LegacyProject", "is_legacy_construction", "legacy_project"]

P = ParamSpec("P")
R = TypeVar("R")

#: Keywords only ``main``'s ``AuthFI(tenant, api_key, ...)`` took.
LEGACY_KEYWORDS = frozenset(
    {
        "tenant",
        "api_key",
        "api_url",
        "application_id",
        "client_secret",
        "auto_sync",
        "jwks_ttl",
        "base_domain",
        "audience",
    }
)
_LEGACY_ORDER = (
    "tenant",
    "api_key",
    "api_url",
    "application_id",
    "client_secret",
    "auto_sync",
    "jwks_ttl",
    "issuer",
    "base_domain",
)

_MIGRATION = "see docs/migration-0.2.md"


def is_legacy_construction(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    """``main``'s form: ``tenant=`` or another legacy keyword, or two or more positional arguments.

    ``audience=`` alone is not the old form: it is what a 0.2 user writes at the wrong place, and
    reading it as the old form would take their issuer for a tenant name.
    """
    if len(args) >= 2 or (LEGACY_KEYWORDS - {"audience"}).intersection(kwargs):
        return True
    if "audience" in kwargs:
        raise ConfigurationError("audience= belongs to AuthFI(tenant, ...); use af.verifier(audience=[...])")
    return False


@dataclass(frozen=True, slots=True)
class LegacyProject:
    """What ``main``'s constructor was given, kept for the deprecated attributes that read it."""

    tenant: str | None
    api_url: str
    application_id: str | None
    client_secret: Secret | None
    auto_sync: bool
    jwks_ttl: int
    base_domain: str
    audience: tuple[str, ...] | None


def legacy_project(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[str, LegacyProject, dict[str, Any]]:
    """The issuer, the legacy settings and the remaining new-style options of ``main``'s form."""
    if len(args) > len(_LEGACY_ORDER):
        raise ConfigurationError(f"AuthFI takes at most {len(_LEGACY_ORDER)} positional arguments in its old form")
    given: dict[str, Any] = dict(zip(_LEGACY_ORDER, args, strict=False))
    for name in _LEGACY_ORDER:
        if name in kwargs:
            if name in given:
                raise ConfigurationError(f"AuthFI got {name} twice")
            given[name] = kwargs.pop(name)
    audience = kwargs.pop("audience", None)
    tenant = given.get("tenant")
    base_domain = str(given.get("base_domain") or "authfi.io")
    issuer_given = given.get("issuer")
    if issuer_given:
        issuer = str(issuer_given).rstrip("/")
    elif tenant:
        issuer = f"https://{tenant}.{base_domain}"
    else:
        raise ConfigurationError("AuthFI needs your project's address: AuthFI(issuer='https://<project>.authfi.io')")
    warnings_ = [
        "AuthFI(tenant, api_key, ...) is replaced by AuthFI(issuer); pass issuer="
        f"{issuer!r} and nothing else ({_MIGRATION})",
    ]
    if given.get("api_key") is not None or "api_url" in given:
        warnings_.append("api_key and api_url are not used: a project is named by its issuer alone")
    if "auto_sync" in given:
        warnings_.append("auto_sync is not used: permissions are managed in the console")
    for message in warnings_:
        warnings.warn(message, AuthFIDeprecationWarning, stacklevel=4)
    secret = given.get("client_secret")
    legacy = LegacyProject(
        tenant=str(tenant) if tenant else None,
        api_url=str(given.get("api_url") or "https://api.authfi.io"),
        application_id=given.get("application_id"),
        client_secret=Secret(secret) if isinstance(secret, str | bytes) and secret else None,
        auto_sync=bool(given.get("auto_sync", True)),
        jwks_ttl=int(given.get("jwks_ttl") or 300),
        base_domain=base_domain,
        audience=tuple(audience)
        if audience is not None and not isinstance(audience, str)
        else ((audience,) if isinstance(audience, str) else None),
    )
    return issuer, legacy, kwargs


def _retired(feature: str, message: str) -> NotSupportedError:
    return NotSupportedError(feature, message)


class LegacyAuthFIMixin:
    """``main``'s ``AuthFI`` methods and attributes, on the new ``AuthFI``. Each one is deprecated."""

    _legacy: LegacyProject | None = None
    _legacy_guard_cache: AuthFIFlask | None = None

    if TYPE_CHECKING:  # what the mixin reads from AuthFI
        issuer: str
        verifier: Callable[..., TokenVerifier]

    def _legacy_settings(self) -> LegacyProject:
        if self._legacy is None:
            raise AttributeError("this AuthFI was built from its issuer; the 0.1 attributes do not exist on it")
        return self._legacy

    @property
    @deprecated("AuthFI.tenant is gone: the project is named by its issuer", category=AuthFIDeprecationWarning)
    def tenant(self) -> str | None:
        """The 0.1 tenant slug."""
        return self._legacy_settings().tenant

    @property
    @deprecated("AuthFI.api_url is not used", category=AuthFIDeprecationWarning)
    def api_url(self) -> str:
        """The 0.1 API host, which nothing calls."""
        return self._legacy_settings().api_url

    @property
    @deprecated("AuthFI.application_id is not used", category=AuthFIDeprecationWarning)
    def application_id(self) -> str | None:
        """The 0.1 application id."""
        return self._legacy_settings().application_id

    @property
    @deprecated(
        "AuthFI.auto_sync is not used: permissions are managed in the console", category=AuthFIDeprecationWarning
    )
    def auto_sync(self) -> bool:
        """The 0.1 auto-sync setting, which syncs nothing."""
        return self._legacy_settings().auto_sync

    @property
    @deprecated(
        "AuthFI.jwks_ttl: the key set is kept for its Cache-Control lifetime", category=AuthFIDeprecationWarning
    )
    def jwks_ttl(self) -> int:
        """The 0.1 key-set lifetime, in seconds."""
        return self._legacy_settings().jwks_ttl

    @property
    @deprecated("AuthFI.base_domain is gone: pass the issuer", category=AuthFIDeprecationWarning)
    def base_domain(self) -> str:
        """The 0.1 base domain."""
        return self._legacy_settings().base_domain

    @property
    @deprecated("AuthFI.jwks_url: the key set comes from discovery's jwks_uri", category=AuthFIDeprecationWarning)
    def jwks_url(self) -> str:
        """Where 0.1 read the key set: the issuer's well-known JWKS."""
        return self.issuer + "/.well-known/jwks.json"

    @deprecated(
        "AuthFI.verify_token(token) is replaced by AuthFI.verifier(audience=[...]).verify(token)",
        category=AuthFIDeprecationWarning,
    )
    def verify_token(self, token: str | Secret) -> Mapping[str, Any]:
        """Verify ``token`` with the strict verifier and return its claims, read-only.

        It needs the audience, given as ``AuthFI(..., audience=[...])``: a token verified for no
        audience is one any API of the project would accept.
        """
        return self._legacy_verifier().verify(token).claims

    def _legacy_audience(self) -> tuple[str, ...]:
        legacy = self._legacy
        if legacy is None or not legacy.audience:
            raise ConfigurationError(
                "verify_token needs your API's audience: AuthFI(..., audience=['<your API client id>']), or "
                "af.verifier(audience=[...]).verify(token). Without one, a token for any API of the project "
                "would be accepted."
            )
        return legacy.audience

    def _legacy_verifier(self) -> TokenVerifier:
        return self.verifier(audience=self._legacy_audience())

    def _legacy_guard(self) -> AuthFIFlask:
        if self._legacy_guard_cache is None:
            from authfi.flask import AuthFIFlask  # Flask is an optional extra

            self._legacy_guard_cache = AuthFIFlask(None, self._legacy_verifier())
        return self._legacy_guard_cache

    @deprecated("AuthFI.require is replaced by authfi.flask.AuthFIFlask.protect", category=AuthFIDeprecationWarning)
    def require(self, *permissions: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """A Flask view decorator requiring ALL of ``permissions``."""
        return _flask_decorator(self, permissions=permissions or None)

    @deprecated(
        "AuthFI.require_role is replaced by authfi.flask.AuthFIFlask.protect", category=AuthFIDeprecationWarning
    )
    def require_role(self, *roles: str) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """A Flask view decorator requiring ANY of ``roles``."""
        return _flask_decorator(self, roles=roles or None)

    @deprecated("AuthFI.current_user is replaced by AuthFIFlask.token", category=AuthFIDeprecationWarning)
    def current_user(self) -> Mapping[str, Any] | None:
        """The claims of the current Flask request's token, or ``None``."""
        try:
            from flask import g  # Flask is an optional extra
        except ImportError:  # pragma: no cover - without the extra
            return None
        try:
            token = g.get("authfi_token")
        except RuntimeError:  # outside a request
            return None
        claims: object = getattr(token, "claims", None)
        return types.MappingProxyType(dict(cast(Mapping[str, Any], claims))) if isinstance(claims, Mapping) else None

    @deprecated(
        "AuthFI.register_permission is retired: create permissions in the console", category=AuthFIDeprecationWarning
    )
    def register_permission(self, name: str, description: str | None = None) -> NoReturn:
        """Retired: there is no permission sync route."""
        raise _retired("permission_sync", "Permissions are created and assigned in the AuthFI console.")

    @deprecated("AuthFI.sync is retired: create permissions in the console", category=AuthFIDeprecationWarning)
    def sync(self) -> NoReturn:
        """Retired: there is no permission sync route."""
        raise _retired("permission_sync", "Permissions are created and assigned in the AuthFI console.")

    @deprecated(
        "AuthFI.cloud_credentials is replaced by agent.gateway().cloud_credentials()", category=AuthFIDeprecationWarning
    )
    def cloud_credentials(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Retired: an agent gets cloud credentials through the AuthFI gateway."""
        raise _retired(
            "cloud_credentials",
            "Cloud credentials are issued to agents through the AuthFI gateway: agent.gateway().cloud_credentials().",
        )

    @deprecated(
        "AuthFI.cloud_token is replaced by agent.gateway().cloud_credentials()", category=AuthFIDeprecationWarning
    )
    def cloud_token(self, *args: Any, **kwargs: Any) -> NoReturn:
        """Retired: an agent gets cloud credentials through the AuthFI gateway."""
        raise _retired(
            "cloud_credentials",
            "Cloud credentials are issued to agents through the AuthFI gateway: agent.gateway().cloud_credentials().",
        )


def _flask_decorator(
    project: LegacyAuthFIMixin,
    *,
    permissions: Sequence[str] | None = None,
    roles: Sequence[str] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorate(view: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(view)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            from authfi.flask import AuthFIRefusalError, refusal_response

            guard = project._legacy_guard()  # pyright: ignore[reportPrivateUsage]
            try:
                return guard.protect(permissions=permissions, roles=roles)(view)(*args, **kwargs)
            except AuthFIRefusalError as error:
                # A Flask view may return a Response whatever its declared type.
                return cast(R, refusal_response(error.refusal))

        return wrapper

    return decorate
