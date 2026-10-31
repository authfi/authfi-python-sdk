"""Auth for a graph deployed on LangSmith Deployment (LangGraph Platform): AuthFI tokens in, AuthFI decisions on.

In ``langgraph.json``::

    { "graphs": { "agent": "./agent.py:graph" }, "auth": { "path": "./auth.py:auth" } }

and ``auth.py``::

    from authfi.langgraph_auth import auth_from_env
    auth = auth_from_env(audience="api_graphs")          # AUTHFI_ISSUER, then the deployment's own credential

``@auth.authenticate`` verifies the caller's access token (the project's JWKS, audience, tenant and delegation
chain) and, for a DPoP-bound token such as an agent instance's, the proof of the key it is bound to. A bound token
presented as Bearer is refused. ``@auth.on`` asks the project's policy engine about every thread, assistant, run,
cron and store call: *may this principal ``<action>`` ``<resource>:<id>``*. The caller is ``user:usr_1``,
``agent:agt_1`` or ``app:app_1``; for a person's token an agent holds, the person (the agent is in ``extra``).

The decision is asked at ``POST /apis/regional/v1/check`` (the SDK's :meth:`~authfi.AsyncAuthFI.check`). AuthZEN's
``/access/v1/evaluation`` is not exposed by the project, so it is not used.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from authfi._dpop_check import ProofReplayCache, check_proof
from authfi.client import AsyncAuthFI
from authfi.errors import ConfigurationError, InvalidToken
from authfi.tokens import AsyncTokenSource
from authfi.verify import AccessToken, AsyncTokenVerifier

try:
    from langgraph_sdk import Auth
except ImportError as error:  # pragma: no cover - exercised by installing the core package alone
    raise ConfigurationError(
        "authfi.langgraph_auth needs langgraph-sdk: "
        f"pip install 'authfi[langgraph-auth] @ git+https://github.com/authfi/authfi-python-sdk@main' ({error})"
    ) from error

__all__ = ["auth_from_env", "create_auth", "resource_of"]

_ACTIONS = {"create_run": "create_run"}


def resource_of(resource: str, value: Mapping[str, Any]) -> str:
    """The object a call is about, as the policy engine names it: ``threads:<id>``, ``assistants:*`` and so on."""
    ids = {
        "threads": value.get("thread_id"),
        "assistants": value.get("assistant_id"),
        "crons": value.get("cron_id"),
        "runs": value.get("run_id") or value.get("thread_id"),
    }
    named = ids.get(resource)
    if resource == "store":
        namespace = value.get("namespace")
        named = "/".join(str(p) for p in cast("Sequence[object]", namespace)) if namespace else None
    return f"langgraph:{resource}:{named}" if named else f"langgraph:{resource}:*"


def _refusal(error: InvalidToken, scheme: str) -> Any:
    # one opaque answer, as the platform's APIs give: the precise code goes to the log, not the caller
    return Auth.exceptions.HTTPException(
        status_code=401, detail="invalid_token", headers={"WWW-Authenticate": f'{scheme} error="invalid_token"'}
    )


def _split(authorization: str | None) -> tuple[str, str]:
    if not authorization or not authorization.strip():
        raise InvalidToken("token_missing", "no Authorization header")
    scheme, _, value = authorization.strip().partition(" ")
    if scheme.lower() not in ("bearer", "dpop") or not value.strip():
        raise InvalidToken("token_missing", "no Bearer or DPoP credential")
    return scheme.lower(), value.strip()


class _Authenticator:
    def __init__(self, verifier: AsyncTokenVerifier, public_origin: str | None, replay: ProofReplayCache) -> None:
        self._verifier = verifier
        self._public_origin = public_origin
        self._replay = replay

    async def verify(
        self, authorization: str | None, dpop: str | None, method: str, url: str, now: float
    ) -> AccessToken:
        scheme, raw = _split(authorization)
        if scheme == "bearer":
            return await self._verifier.verify(raw)  # a bound token is refused here: token_dpop_proof_required
        verifier = self._verifier
        try:
            claims = await verifier._claims(raw, verifier._config, bearer=False)  # pyright: ignore[reportPrivateUsage]
            cnf = claims.get("cnf")
            jkt = cast("Mapping[str, Any]", cnf).get("jkt") if isinstance(cnf, dict) else None
            check_proof(
                dpop,
                method=method,
                url=url,
                access_token=raw,
                jkt=jkt if isinstance(jkt, str) else None,
                now=now,
                replay=self._replay,
            )
        except InvalidToken as error:
            verifier.logger.warning("refused a DPoP request to %s: %s", url, error.code)
            raise
        return AccessToken.from_claims(claims)

    def url_of(self, request: Any) -> str:
        url = request.url
        base = f"{url.scheme}://{url.netloc}" if self._public_origin is None else self._public_origin.rstrip("/")
        return str(base + url.path)


def _user(token: AccessToken) -> dict[str, Any]:
    subject = token.principal
    extra: dict[str, Any] = {"tenant_id": token.tenant_id, "token_kind": token.token_kind, "client_id": token.client_id}
    if token.actor is not None:
        extra["actor"] = str(token.actor)
    return {
        "identity": str(subject),
        "permissions": sorted(token.permissions | token.scopes),
        "display_name": str(subject),
        **extra,
    }


def create_auth(
    project: AsyncAuthFI,
    credential: AsyncTokenSource,
    *,
    audience: Sequence[str],
    public_origin: str | None = None,
    expected_tenant_id: str | None = None,
) -> Any:
    """An ``Auth()`` object for ``langgraph.json``.

    ``credential`` is the deployment's own token for the policy check (an agent or a machine client of the
    project). ``audience`` is what the callers' tokens are for. Behind a proxy, give ``public_origin`` (the origin
    callers use) so the DPoP proof's URL matches.
    """
    verifier = project.verifier(list(audience), expected_tenant_id=expected_tenant_id)
    authenticator = _Authenticator(verifier, public_origin, ProofReplayCache())
    auth = Auth()

    @auth.authenticate
    async def authenticate(request: Any, method: str, authorization: str | None, headers: dict[bytes, bytes]) -> Any:
        dpop = headers.get(b"dpop", b"").decode("latin-1") or None
        try:
            token = await authenticator.verify(authorization, dpop, method, authenticator.url_of(request), time.time())
        except InvalidToken as error:
            scheme = "DPoP" if (authorization or "").lower().startswith("dpop") else "Bearer"
            raise _refusal(error, scheme) from None
        return _user(token)

    @auth.on
    async def authorize(ctx: Any, value: Any) -> Any:
        action = _ACTIONS.get(str(ctx.action), str(ctx.action))
        obj = resource_of(str(ctx.resource), cast("Mapping[str, Any]", value) if isinstance(value, Mapping) else {})
        subject = str(ctx.user.identity)
        answer = await project.check(
            credential,
            subject=subject,
            relation=action,
            object=obj,
            context=_context(ctx.user),
        )
        if answer.allowed:
            return None
        if answer.approval_required:
            raise Auth.exceptions.HTTPException(status_code=403, detail="needs_approval")
        raise Auth.exceptions.HTTPException(status_code=403, detail="forbidden")

    return auth


def _context(user: Any) -> dict[str, Any] | None:
    """What the policy engine is told besides the question: the agent, when it acts for the person."""
    try:
        actor = user["actor"] if "actor" in user else None
    except (TypeError, KeyError):  # a Studio user is not a dict
        return None
    return {"actor": actor} if isinstance(actor, str) else None


def auth_from_env(*, audience: Sequence[str], env: Mapping[str, str] | None = None, **options: Any) -> Any:
    """:func:`create_auth` for the deployment's environment: ``AUTHFI_ISSUER`` and the credential it names."""
    import authfi

    credential = authfi.connect_async(env)
    return create_auth(credential.project, credential, audience=audience, **options)
