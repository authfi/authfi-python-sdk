"""Flask: protect views with AuthFI access tokens.

``pip install 'authfi[flask]'``

::

    from flask import Flask
    from authfi import AuthFI
    from authfi.flask import AuthFIFlask

    af = AuthFI.from_env()
    app = Flask(__name__)
    guard = AuthFIFlask(app, af.verifier(audience=["api_orders"]))

    @app.get("/orders")
    @guard.protect(permissions=["orders:read"])
    def orders():
        return {"for": str(guard.token.principal)}

The verified token lives on ``flask.g`` for one request only. A refused request is answered as
RFC 6750 says, with one opaque description; the precise code is logged.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Sequence
from typing import ParamSpec, TypeVar, cast

try:
    from flask import Flask, Response, g, jsonify, request
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("authfi.flask needs Flask: pip install 'authfi[flask]'") from exc

from authfi._resource import (
    UNAVAILABLE,
    Refusal,
    Requirements,
    log_refusal,
    log_unavailable,
    refusal_for,
    unavailable_refusal,
)
from authfi.errors import ConfigurationError, InvalidToken
from authfi.verify import AccessToken, AsyncMultiIssuerVerifier, AsyncTokenVerifier, MultiIssuerVerifier, TokenVerifier

__all__ = ["AuthFIFlask", "AuthFIRefusalError"]

P = ParamSpec("P")
R = TypeVar("R")

_G_KEY = "authfi_token"


class AuthFIRefusalError(Exception):
    """A refused request; the guard's error handler answers it."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(refusal.status)
        self.refusal = refusal


class AuthFIFlask:
    """Verifies the request's token and applies each view's requirements.

    Args:
        app: the Flask app, or ``None`` to call :meth:`init_app` later.
        verifier: a :class:`~authfi.verify.TokenVerifier` from ``AuthFI.verifier()``, or a
            :class:`~authfi.verify.MultiIssuerVerifier`.
        resource_metadata: your API's RFC 9728 metadata URL, sent in the 401 challenge.
        expose_refusal_code: add the precise refusal code to the answer body; see
            :class:`authfi.fastapi.AuthFIBearer`.

    """

    def __init__(
        self,
        app: Flask | None,
        verifier: TokenVerifier | MultiIssuerVerifier,
        *,
        resource_metadata: str | None = None,
        expose_refusal_code: bool = False,
    ) -> None:
        if isinstance(cast(object, verifier), AsyncTokenVerifier | AsyncMultiIssuerVerifier):
            raise ConfigurationError("Flask views are blocking: build the verifier with AuthFI(...).verifier(...)")
        self._verifier = verifier
        self._resource_metadata = resource_metadata
        self._expose = expose_refusal_code
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        """Register the error handlers that answer refusals.

        A refusal raised inside a view (``token.require_acr(...)``, ``token.require_scope(...)``)
        is answered the same way as one the guard raised, never as a 500.
        """

        def answer(refusal: Refusal) -> Response:
            response = jsonify(refusal.body)
            response.status_code = refusal.status
            for name, value in refusal.headers.items():
                response.headers[name] = value
            return response

        def refused(error: AuthFIRefusalError) -> Response:
            return answer(error.refusal)

        def raised_in_view(error: InvalidToken) -> Response:
            return answer(self._refuse(error).refusal)

        app.register_error_handler(AuthFIRefusalError, refused)
        app.register_error_handler(InvalidToken, raised_in_view)

    @property
    def token(self) -> AccessToken | None:
        """The verified token of the current request, or ``None``."""
        value: object = g.get(_G_KEY)
        return value if isinstance(value, AccessToken) else None

    def protect(
        self,
        *,
        permissions: Sequence[str] | None = None,
        any_permission: Sequence[str] | None = None,
        roles: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
        optional: bool = False,
    ) -> Callable[[Callable[P, R]], Callable[P, R]]:
        """A view decorator: verify the token, apply the requirements, put the token on ``flask.g``.

        ``permissions`` and ``scopes``: all of them. ``any_permission`` and ``roles``: at least one.
        Several kinds: each must pass. With ``optional``, a request with no token reaches the view
        with :attr:`token` ``None``; a bad token is still refused.
        """
        rules = Requirements.of(permissions=permissions, any_permission=any_permission, roles=roles, scopes=scopes)
        if optional and rules != Requirements():
            raise ConfigurationError("an optional route cannot also require permissions, roles or scopes")

        def decorate(view: Callable[P, R]) -> Callable[P, R]:
            @functools.wraps(view)
            def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                g.pop(_G_KEY, None)
                if optional and not request.headers.get("Authorization"):
                    return view(*args, **kwargs)
                try:
                    token = self._verifier.verify_request(
                        request.headers.get("Authorization"),
                        dpop=request.headers.get("DPoP"),
                        method=request.method,
                        url=request.url,
                    )
                    rules.check(token)
                except InvalidToken as error:
                    raise self._refuse(error) from None
                except UNAVAILABLE as error:
                    rid = request.headers.get("X-Request-Id")
                    log_unavailable(self._verifier.logger, error, request.method, request.path, rid)
                    raise AuthFIRefusalError(unavailable_refusal(rid)) from None
                setattr(g, _G_KEY, token)
                return view(*args, **kwargs)

            return wrapper

        return decorate

    def _refuse(self, error: InvalidToken) -> AuthFIRefusalError:
        rid = request.headers.get("X-Request-Id")
        log_refusal(self._verifier.logger, error, request.method, request.path, rid)
        scheme = (request.headers.get("Authorization") or "").split(" ", 1)[0].lower()
        return AuthFIRefusalError(
            refusal_for(
                error,
                dpop=scheme == "dpop",
                resource_metadata=self._resource_metadata,
                expose_refusal_code=self._expose,
                request_id=rid,
            )
        )
