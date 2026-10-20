"""What every API framework adapter shares: the requirement rules and the wire answer.

The wire answer is opaque by design. The precise refusal code goes to the log; the caller gets
RFC 6750 ``invalid_token`` with one fixed description, so it cannot tell a wrong key from a wrong
audience or a wrong account. ``expose_refusal_code=True`` adds the code to the body, for an
internal API whose callers are trusted to see it.

============================  =========================================================
Refusal                       Answer
============================  =========================================================
no credential                 401, ``Bearer`` (no ``error``), ``{"error":"unauthorized"}``
any other token refusal       401, ``Bearer error="invalid_token"``, fixed description
keys or discovery unreachable 503, ``Retry-After: 5``, ``{"error":"service_unavailable"}``
missing scope                 403, ``Bearer error="insufficient_scope", scope="..."``
missing permission or role    403, no challenge, ``{"error":"forbidden"}``
authentication too weak       401, ``Bearer error="insufficient_user_authentication"``
============================  =========================================================
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from authfi._protocol.ids import is_valid_request_id
from authfi.errors import (
    ConfigurationError,
    DiscoveryError,
    InsufficientAuthentication,
    InsufficientPermission,
    InsufficientScope,
    InvalidToken,
    TemporarilyUnavailable,
    TransportError,
)
from authfi.verify import AccessToken

__all__ = [
    "INVALID_TOKEN_DESCRIPTION",
    "UNAVAILABLE",
    "Refusal",
    "Requirements",
    "log_refusal",
    "log_unavailable",
    "refusal_for",
    "unavailable_refusal",
]

#: What it means when the token could not be checked at all: the issuer's keys or discovery are
#: unreachable. The caller's token may be fine, so this is a 503 to retry, not a 401.
UNAVAILABLE = (TransportError, DiscoveryError, TemporarilyUnavailable)

#: The one description every token refusal carries on the wire.
INVALID_TOKEN_DESCRIPTION = "The access token is not valid."


def _names(value: Sequence[str] | None, what: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        raise ConfigurationError(f"{what} is a list of names, such as [{value!r}], not one string")
    names = tuple(value)
    if not names or not all(isinstance(n, str) and n for n in cast(tuple[object, ...], names)):
        raise ConfigurationError(f"{what} is empty: an empty requirement would require nothing; leave it out instead")
    return names


@dataclass(frozen=True, slots=True)
class Requirements:
    """What a route requires. Each kind given must pass (AND).

    ``permissions``: all of them. ``any_permission``: at least one. ``roles``: at least one.
    ``scopes``: all of them. An empty list is a configuration error, never "no requirement".
    """

    permissions: tuple[str, ...] = ()
    any_permission: tuple[str, ...] = ()
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()

    @classmethod
    def of(
        cls,
        *,
        permissions: Sequence[str] | None = None,
        any_permission: Sequence[str] | None = None,
        roles: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> Requirements:
        """Checked requirements."""
        return cls(
            _names(permissions, "permissions"),
            _names(any_permission, "any_permission"),
            _names(roles, "roles"),
            _names(scopes, "scopes"),
        )

    def check(self, token: AccessToken) -> None:
        """Raise :class:`InsufficientPermission` or :class:`InsufficientScope` unless every rule passes."""
        if self.permissions:
            token.require_permission(*self.permissions)
        if self.any_permission:
            token.require_any_permission(*self.any_permission)
        if self.roles:
            token.require_role(*self.roles)
        if self.scopes:
            token.require_scope(*self.scopes)


@dataclass(frozen=True, slots=True)
class Refusal:
    """An HTTP answer to a refused request."""

    status: int
    headers: dict[str, str]
    body: dict[str, Any]


def _quote(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _challenge(scheme: str, params: list[tuple[str, str]]) -> str:
    if not params:
        return scheme
    return scheme + " " + ", ".join(f'{k}="{_quote(v)}"' if k != "max_age" else f"{k}={v}" for k, v in params)


def refusal_for(
    error: InvalidToken,
    *,
    dpop: bool = False,
    resource_metadata: str | None = None,
    expose_refusal_code: bool = False,
    request_id: str | None = None,
) -> Refusal:
    """The answer for ``error``: identical whatever framework sends it."""
    scheme = "DPoP" if dpop else "Bearer"
    meta = [("resource_metadata", resource_metadata)] if resource_metadata else []
    headers = {"Content-Type": "application/json", "Cache-Control": "no-store"}
    if request_id and is_valid_request_id(request_id):
        headers["X-Request-Id"] = request_id
    status: int
    body: dict[str, Any]
    if isinstance(error, InsufficientPermission):
        status, body = 403, {"error": "forbidden"}
    elif isinstance(error, InsufficientScope):
        status, body = 403, {"error": "insufficient_scope"}
        headers["WWW-Authenticate"] = _challenge(
            scheme, [("error", "insufficient_scope"), ("scope", " ".join(error.required))]
        )
    elif isinstance(error, InsufficientAuthentication):
        status, body = 401, {"error": "insufficient_user_authentication"}
        params = [("error", "insufficient_user_authentication"), ("acr_values", " ".join(error.acr_values))]
        if error.max_age is not None:
            params.append(("max_age", str(error.max_age)))
        headers["WWW-Authenticate"] = _challenge(scheme, params)
    elif error.code == "token_missing":
        status, body = 401, {"error": "unauthorized"}
        headers["WWW-Authenticate"] = _challenge(scheme, meta)
    else:
        status = 401
        body = {"error": "invalid_token", "error_description": INVALID_TOKEN_DESCRIPTION}
        headers["WWW-Authenticate"] = _challenge(
            scheme, [("error", "invalid_token"), ("error_description", INVALID_TOKEN_DESCRIPTION), *meta]
        )
    if expose_refusal_code:
        body["code"] = error.code
    return Refusal(status, headers, body)


def _rid(request_id: str | None) -> str:
    return request_id if request_id and is_valid_request_id(request_id) else "-"


def log_refusal(logger: logging.Logger, error: InvalidToken, method: str, path: str, request_id: str | None) -> None:
    """Log a refusal with its precise code, which the wire answer does not carry.

    Info for a request that sent no token, warning for any other refusal (as the platform's kernel does).
    """
    level = logging.INFO if error.code == "token_missing" else logging.WARNING
    logger.log(level, "refused %s %s: %s (request %s)", method, path, error.code, _rid(request_id))


def unavailable_refusal(request_id: str | None = None) -> Refusal:
    """The answer when the token could not be checked: 503, retry shortly."""
    headers = {"Content-Type": "application/json", "Cache-Control": "no-store", "Retry-After": "5"}
    if request_id and is_valid_request_id(request_id):
        headers["X-Request-Id"] = request_id
    return Refusal(503, headers, {"error": "service_unavailable"})


def log_unavailable(logger: logging.Logger, error: Exception, method: str, path: str, request_id: str | None) -> None:
    """Log that a request was not checked because the issuer's keys or discovery were unreachable."""
    logger.error(
        "could not check the token of %s %s: %s: %s (request %s)",
        method,
        path,
        type(error).__name__,
        getattr(error, "code", "-"),
        _rid(request_id),
    )
