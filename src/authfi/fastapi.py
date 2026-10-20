"""FastAPI: protect routes with AuthFI access tokens.

``pip install 'authfi[fastapi]'``

::

    from fastapi import Depends, FastAPI
    from authfi import AsyncAuthFI
    from authfi.fastapi import AuthFIBearer

    af = AsyncAuthFI.from_env()
    auth = AuthFIBearer(af.verifier(audience=["api_orders"]))
    app = FastAPI()
    auth.install(app)

    @app.get("/orders")
    async def orders(token: AccessToken = Depends(auth.require(permissions=["orders:read"]))):
        return {"for": str(token.principal)}

It takes the async verifier, so fetching the project's keys never blocks the event loop. A refused
request is answered as RFC 6750 says, with one opaque description; the precise code is logged.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import cast

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("authfi.fastapi needs FastAPI: pip install 'authfi[fastapi]'") from exc

from authfi._resource import Refusal, Requirements, log_refusal, refusal_for
from authfi.errors import ConfigurationError, InvalidToken
from authfi.verify import AccessToken, AsyncMultiIssuerVerifier, AsyncTokenVerifier, MultiIssuerVerifier, TokenVerifier

__all__ = ["AuthFIBearer", "AuthFIRefusal"]


class AuthFIRefusal(HTTPException):
    """A refused request. :meth:`AuthFIBearer.install` answers it with its exact body."""

    def __init__(self, refusal: Refusal) -> None:
        super().__init__(status_code=refusal.status, detail=refusal.body, headers=refusal.headers)
        self.refusal = refusal


class AuthFIBearer:
    """Route dependencies that verify the request's token and apply a route's requirements.

    Args:
        verifier: an :class:`~authfi.verify.AsyncTokenVerifier` from ``AsyncAuthFI.verifier()``,
            or an :class:`~authfi.verify.AsyncMultiIssuerVerifier`.
        resource_metadata: your API's RFC 9728 metadata URL, sent in the 401 challenge.
        expose_refusal_code: add the precise refusal code to the answer body. Off by default:
            the code tells a caller why its token failed, which an attacker probing your API can
            use. Turn it on only for an internal API.

    """

    def __init__(
        self,
        verifier: AsyncTokenVerifier | AsyncMultiIssuerVerifier,
        *,
        resource_metadata: str | None = None,
        expose_refusal_code: bool = False,
    ) -> None:
        if isinstance(cast(object, verifier), TokenVerifier | MultiIssuerVerifier):
            raise ConfigurationError(
                "AuthFIBearer takes the asyncio verifier, so a key fetch never blocks the event loop: build it "
                "with AsyncAuthFI(...).verifier(...)"
            )
        self._verifier = verifier
        self._resource_metadata = resource_metadata
        self._expose = expose_refusal_code

    def install(self, app: FastAPI) -> None:
        """Answer refusals with their exact RFC 6750 body (without it, FastAPI wraps it in ``detail``).

        A refusal raised inside a route (``token.require_acr(...)``) is answered the same way, never
        as a 500.
        """

        async def handler(request: Request, exc: Exception) -> JSONResponse:
            if isinstance(exc, AuthFIRefusal):
                refusal = exc.refusal
            elif isinstance(exc, InvalidToken):
                refusal = self._refuse(request, exc).refusal
            else:  # pragma: no cover - FastAPI routes only the two classes registered below
                raise exc
            return JSONResponse(refusal.body, status_code=refusal.status, headers=refusal.headers)

        app.add_exception_handler(AuthFIRefusal, handler)
        app.add_exception_handler(InvalidToken, handler)

    def require(
        self,
        *,
        permissions: Sequence[str] | None = None,
        any_permission: Sequence[str] | None = None,
        roles: Sequence[str] | None = None,
        scopes: Sequence[str] | None = None,
    ) -> Callable[[Request], Awaitable[AccessToken]]:
        """A dependency that returns the verified token, or refuses the request.

        ``permissions`` and ``scopes``: all of them. ``any_permission`` and ``roles``: at least one.
        Several kinds: each must pass. An empty list is a configuration error.
        """
        rules = Requirements.of(permissions=permissions, any_permission=any_permission, roles=roles, scopes=scopes)

        async def dependency(request: Request) -> AccessToken:
            token = await self._verify(request)
            try:
                rules.check(token)
            except InvalidToken as error:
                raise self._refuse(request, error) from None
            return token

        return dependency

    def optional(self) -> Callable[[Request], Awaitable[AccessToken | None]]:
        """A dependency that returns the verified token, ``None`` without one, and refuses a bad one."""

        async def dependency(request: Request) -> AccessToken | None:
            if not request.headers.get("authorization"):
                return None
            return await self._verify(request)

        return dependency

    async def _verify(self, request: Request) -> AccessToken:
        try:
            return await self._verifier.verify_request(
                request.headers.get("authorization"),
                dpop=request.headers.get("dpop"),
                method=request.method,
                url=str(request.url),
            )
        except InvalidToken as error:
            raise self._refuse(request, error) from None

    def _refuse(self, request: Request, error: InvalidToken) -> AuthFIRefusal:
        rid = request.headers.get("x-request-id")
        log_refusal(self._verifier.logger, error, request.method, request.url.path, rid)
        scheme = (request.headers.get("authorization") or "").split(" ", 1)[0].lower()
        return AuthFIRefusal(
            refusal_for(
                error,
                dpop=scheme == "dpop",
                resource_metadata=self._resource_metadata,
                expose_refusal_code=self._expose,
                request_id=rid,
            )
        )
