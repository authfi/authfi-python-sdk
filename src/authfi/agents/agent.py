"""An AI agent: its own token, acting for a person with their consent, the action gate and approvals.

An agent is an OAuth client with a key. Its token comes from the same token endpoint and the same
code as a machine client's; an API verifies it with the same verifier as a person's.

::

    agent = af.agent("agt_123", AgentKey.from_file("/run/secrets/agent.key", kid="k1"))
    agent.token()                                   # its own token, for <issuer>/mcp by default

    request = agent.request_user_token("priya@acme.com", binding_message="Refund order 4411")
    as_priya = agent.await_user_token(request)      # the person approved on their own device

    as_her = agent.exchange(signed_in_person)       # the person signed in to your app

Two ways of asking a person, two names: ``request_user_token`` / ``await_user_token`` end in a
token the agent uses to act for the person (CIBA); ``request_approval`` / ``wait_for_approval``
end in a decision about one action, and no token.
"""

from __future__ import annotations

import asyncio
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, NoReturn, cast, overload

from typing_extensions import deprecated

from authfi._cache import AsyncCache, SyncCache
from authfi._protocol.agent import (
    CIBA_GRANT,
    TOKEN_EXCHANGE_GRANT,
    AgentEndpoints,
    ApprovalRecord,
    CibaStart,
    approval_request,
    authorize_request,
    ciba_poll_flow,
    ciba_start_request,
    exchange_request,
    label_claims,
    parse_action_decision,
    parse_agent_configuration,
    parse_approval,
    parse_ciba_start,
    parse_transaction_token,
    read_approval_request,
    transaction_token_request,
)
from authfi._protocol.discovery import ServerMetadata, ttl_from_headers
from authfi._protocol.http import DOCUMENT_LIMIT, HttpRequest, HttpResponse, Retry
from authfi._protocol.oauth import ClientAuth, TokenResponse, client_credentials_request, parse_token_response
from authfi.agents._legacy import LegacyAgentMixin, is_legacy_agent, legacy_agent_arguments
from authfi.agents.keys import Signer, client_assertion
from authfi.errors import (
    AuthFIDeprecationWarning,
    ConfigurationError,
    ConsentExpired,
    DiscoveryError,
    NotSupportedError,
    OAuthError,
)
from authfi.gateway import AsyncGateway, Gateway
from authfi.secret import Secret
from authfi.tokens import AsyncTokenSource, Destinations, TokenSource
from authfi.verify import Principal

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = [
    "ActionDecision",
    "AgentAuth",
    "Approval",
    "AsyncAgentAuth",
    "AsyncDelegatedToken",
    "CibaRequest",
    "DelegatedToken",
    "TransactionToken",
]

_APPROVAL_POLL = 5.0
_PENDING = "pending"


@dataclass(frozen=True, slots=True, repr=False)
class CibaRequest:
    """A person was asked; poll it with ``await_user_token``. ``auth_req_id`` is a bearer handle."""

    auth_req_id: Secret
    expires_in: int
    interval: float
    approval_url: str | None
    deadline: float = field(compare=False)

    def __repr__(self) -> str:
        return f"CibaRequest(expires_in={self.expires_in}, interval={self.interval})"

    def __reduce__(self) -> NoReturn:
        raise TypeError("A CibaRequest cannot be pickled: its auth_req_id is a bearer handle")


@dataclass(frozen=True, slots=True, repr=False)
class TransactionToken:
    """A frozen action: 60 seconds in which the gate can decide exactly this action, and no other."""

    token: Secret
    txn: str
    action: str
    resource: str | None
    context: Mapping[str, Any]
    expires_at: datetime

    def __repr__(self) -> str:
        return f"TransactionToken(txn={self.txn!r}, action={self.action!r}, expires_at={self.expires_at.isoformat()})"


@dataclass(frozen=True, slots=True)
class ActionDecision:
    """The gate's decision. ``approval_required`` means a person must approve; ``approval_id`` names it."""

    decision: Literal["allow", "deny", "approval_required"]
    approval_id: str | None
    reason: str | None


def _when(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True, slots=True)
class Approval:
    """A person's approval of one action. ``status``: ``pending``, ``approved``, ``denied``, ``expired``."""

    approval_id: str
    status: str
    action: str
    created_at: datetime | None
    expires_at: datetime | None
    resolved_at: datetime | None
    context: Mapping[str, Any]

    @classmethod
    def of(cls, record: ApprovalRecord) -> Approval:
        """From the platform's record."""
        return cls(
            record.approval_id,
            record.status,
            record.action,
            _when(record.created_at),
            _when(record.expires_at),
            _when(record.resolved_at),
            record.context,
        )

    @property
    def decided(self) -> bool:
        """``True`` once it is no longer pending."""
        return self.status != _PENDING


@dataclass(frozen=True, slots=True, repr=False)
class _Minted:
    access_token: Secret
    capabilities: frozenset[str]
    expires_at: float


def _capabilities(response: TokenResponse) -> frozenset[str]:
    names = response.extra.get("capabilities")
    if isinstance(names, list):
        return frozenset(n for n in cast(list[object], names) if isinstance(n, str))
    return response.scope or frozenset()


def _check_agent_grant(metadata: ServerMetadata, signer: Signer) -> None:
    if not metadata.supports_grant("client_credentials"):
        raise NotSupportedError("client_credentials", "This AuthFI project does not offer agent tokens.")
    methods = metadata.token_endpoint_auth_methods_supported
    if methods and "private_key_jwt" not in methods:
        raise NotSupportedError("private_key_jwt", "This AuthFI project does not accept a signed client assertion.")
    algs = metadata.token_endpoint_auth_signing_alg_values_supported
    if algs and signer.alg not in algs:
        raise NotSupportedError(signer.alg, f"This AuthFI project does not accept {signer.alg} assertions.")


def _ciba_endpoint(metadata: ServerMetadata) -> str:
    endpoint = metadata.backchannel_authentication_endpoint
    if not endpoint or not metadata.supports_grant(CIBA_GRANT):
        raise NotSupportedError("ciba", "This AuthFI project does not offer acting for a person without a browser.")
    return endpoint


def _exchange_supported(metadata: ServerMetadata) -> None:
    if not metadata.supports_grant(TOKEN_EXCHANGE_GRANT):
        raise NotSupportedError("token_exchange", "This AuthFI project does not offer token exchange.")


def _labels(raw: str, agent_id: str) -> tuple[Principal, Principal, frozenset[str]]:
    claims = label_claims(raw)
    sub = claims.get("sub")
    act = claims.get("act")
    actor_sub = cast(Mapping[str, Any], act).get("sub") if isinstance(act, dict) else None
    subject = Principal.parse(sub, "user") if isinstance(sub, str) else Principal("user", "")
    actor = Principal.parse(actor_sub, "agent") if isinstance(actor_sub, str) else Principal("agent", agent_id)
    scope = claims.get("scope")
    return subject, actor, frozenset(scope.split()) if isinstance(scope, str) else frozenset()


def _secret_alive(secret: Secret, now: float) -> bool:
    exp = label_claims(secret.reveal()).get("exp")
    return isinstance(exp, int | float) and not isinstance(exp, bool) and exp > now


class _AgentBase:
    """Configuration and rules shared by the sync and async agent clients."""

    def __init__(
        self,
        project: AuthFI | AsyncAuthFI,
        agent_id: str,
        key: Signer,
        *,
        resource: str | None,
        send_to: Sequence[str],
        refresh_skew: float,
    ) -> None:
        agent = str(agent_id or "").strip()
        if not agent:
            raise ConfigurationError("agent_id is required (agt_...)")
        if not isinstance(cast(object, key), Signer):
            raise ConfigurationError(
                "key must be an AgentKey (or another Signer): an agent authenticates with a key it holds, and "
                "there is no secret fallback"
            )
        if not 0 <= refresh_skew <= 600:
            raise ConfigurationError("refresh_skew must be from 0 to 600 seconds")
        self.agent_id = agent
        self.key = key
        self.resource = resource.strip() if resource else None
        self._destinations = Destinations(project, self.resource, send_to)
        self.refresh_skew = float(refresh_skew)
        self._project_ref = project
        self._auth = ClientAuth.private_key_jwt(agent, self._assertion)
        self._clock = project.clock
        self._log = project.logger

    def _assertion(self, audience: str) -> str:
        return client_assertion(self.key, self.agent_id, audience, now=self._clock.time())

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`."""
        return self._destinations.origins

    def _resource(self, resource: str | None) -> str | None:
        return resource.strip() if resource else self.resource

    def _minted(self, response: TokenResponse) -> tuple[_Minted, float]:
        lifetime = float(response.expires_in or 300)
        minted = _Minted(response.access_token, _capabilities(response), self._clock.time() + lifetime)
        return minted, max(1.0, lifetime - self.refresh_skew)

    def _ciba_request(self, start: CibaStart, now: float) -> CibaRequest:
        return CibaRequest(
            start.auth_req_id, start.expires_in, start.interval, start.approval_url, now + start.expires_in
        )

    def _check_destination_args(self, resource: str | None, send_to: Sequence[str]) -> Destinations:
        return Destinations(self._project_ref, resource, send_to)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(agent_id={self.agent_id!r}, kid={self.key.kid!r})"


class _AgentResource(TokenSource):
    """The agent's own token for one resource, sent only to that resource's origins."""

    __slots__ = ("_agent", "_destinations", "_resource")

    def __init__(self, agent: AgentAuth, resource: str, destinations: Destinations) -> None:
        self._agent = agent
        self._resource = resource
        self._destinations = destinations

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`."""
        return self._destinations.origins

    def token(self, *, force: bool = False) -> str:
        """The agent's token for this resource."""
        return self._agent.token(resource=self._resource, force=force)

    def __repr__(self) -> str:
        return f"AgentAuth.for_resource({self._resource!r})"


class DelegatedToken(TokenSource):
    """A token an agent holds to act for a person: ``sub`` is the person, ``act.sub`` the agent.

    - From CIBA (``via == "ciba"``) it is never renewed: the person's approval was for this token.
      ``refresh_skew`` before it expires, :meth:`token` raises :class:`ConsentExpired`, so no call
      is made with a token about to die. Acting for them again means asking them again.
    - From token exchange (``via == "exchange"``) it is exchanged again while the person's own
      token lives, and raises :class:`ConsentExpired` once it does not.

    It does not print its token and does not pickle: a checkpoint stores an id, never a token.
    """

    __slots__ = ("_current", "_destinations", "_lock", "_renew", "_skew", "_time", "actor", "project", "subject", "via")

    def __init__(
        self,
        *,
        response: TokenResponse,
        via: Literal["ciba", "exchange"],
        agent_id: str,
        destinations: Destinations,
        renew: Callable[[], TokenResponse] | None,
        refresh_skew: float,
        time: Callable[[], float],
        project: Any = None,
    ) -> None:
        #: The project that issued the token: what the gateway helpers and adapters attach to.
        self.project = project
        self.via: Literal["ciba", "exchange"] = via
        self._renew = renew
        self._skew = refresh_skew
        self._time = time
        self._destinations = destinations
        self._lock = threading.Lock()
        self.subject: Principal
        self.actor: Principal
        self._current: tuple[Secret, float, frozenset[str]]
        self._take(response, agent_id)

    def _take(self, response: TokenResponse, agent_id: str) -> None:
        subject, actor, scopes = _labels(response.access_token.reveal(), agent_id)
        self.subject, self.actor = subject, actor
        lifetime = float(response.expires_in or 300)
        self._current = (response.access_token, self._time() + lifetime, response.scope or scopes)

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`."""
        return self._destinations.origins

    @property
    def scopes(self) -> frozenset[str]:
        """What the token may do: the person's approval intersected with the agent's grants."""
        return self._current[2]

    @property
    def expires_at(self) -> datetime:
        """When the current token expires."""
        return datetime.fromtimestamp(self._current[1], UTC)

    def token(self, *, force: bool = False) -> str:
        """The current token; renewed (exchange) or refused (:class:`ConsentExpired`) near its expiry."""
        with self._lock:
            token, expires_at, _ = self._current
            if not force and self._time() < expires_at - self._skew:
                return token.reveal()
            if self._renew is None:
                raise ConsentExpired(
                    "expired_token",
                    f"the token acting for {self.subject} has run out, and only they can approve again",
                )
            self._take(self._renew(), self.actor.id)
            return self._current[0].reveal()

    def __repr__(self) -> str:
        return (
            f"DelegatedToken(subject={str(self.subject)!r}, actor={str(self.actor)!r}, via={self.via!r}, "
            f"expires_at={self.expires_at.isoformat()})"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError(
            "A DelegatedToken cannot be pickled: a checkpoint would store a person's token. Store the person's id "
            "and ask again, or exchange again."
        )


class AsyncDelegatedToken(AsyncTokenSource):
    """See :class:`DelegatedToken`; for asyncio code."""

    __slots__ = ("_current", "_destinations", "_lock", "_renew", "_skew", "_time", "actor", "project", "subject", "via")

    def __init__(
        self,
        *,
        response: TokenResponse,
        via: Literal["ciba", "exchange"],
        agent_id: str,
        destinations: Destinations,
        renew: Callable[[], Any] | None,
        refresh_skew: float,
        time: Callable[[], float],
        project: Any = None,
    ) -> None:
        #: The project that issued the token: what the gateway helpers and adapters attach to.
        self.project = project
        self.via: Literal["ciba", "exchange"] = via
        self._renew = renew
        self._skew = refresh_skew
        self._time = time
        self._destinations = destinations
        self._lock = asyncio.Lock()
        self.subject: Principal
        self.actor: Principal
        self._current: tuple[Secret, float, frozenset[str]]
        self._take(response, agent_id)

    def _take(self, response: TokenResponse, agent_id: str) -> None:
        subject, actor, scopes = _labels(response.access_token.reveal(), agent_id)
        self.subject, self.actor = subject, actor
        lifetime = float(response.expires_in or 300)
        self._current = (response.access_token, self._time() + lifetime, response.scope or scopes)

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`."""
        return self._destinations.origins

    @property
    def scopes(self) -> frozenset[str]:
        """See :attr:`DelegatedToken.scopes`."""
        return self._current[2]

    @property
    def expires_at(self) -> datetime:
        """See :attr:`DelegatedToken.expires_at`."""
        return datetime.fromtimestamp(self._current[1], UTC)

    async def token(self, *, force: bool = False) -> str:
        """See :meth:`DelegatedToken.token`."""
        async with self._lock:
            token, expires_at, _ = self._current
            if not force and self._time() < expires_at - self._skew:
                return token.reveal()
            if self._renew is None:
                raise ConsentExpired(
                    "expired_token",
                    f"the token acting for {self.subject} has run out, and only they can approve again",
                )
            self._take(await self._renew(), self.actor.id)
            return self._current[0].reveal()

    def __repr__(self) -> str:
        return (
            f"AsyncDelegatedToken(subject={str(self.subject)!r}, actor={str(self.actor)!r}, via={self.via!r}, "
            f"expires_at={self.expires_at.isoformat()})"
        )

    def __reduce__(self) -> NoReturn:
        raise TypeError("An AsyncDelegatedToken cannot be pickled: a checkpoint would store a person's token.")


def _legacy_ciba_request(request: CibaRequest | Mapping[str, Any] | str, now: float) -> CibaRequest:
    if isinstance(request, CibaRequest):
        return request
    if isinstance(request, str):
        return CibaRequest(Secret(request), 300, 5.0, None, now + 300)
    auth_req_id = request.get("auth_req_id")
    if not isinstance(auth_req_id, str) or not auth_req_id:
        raise ConfigurationError("no auth_req_id to wait on")
    interval = request.get("interval")
    expires_in = request.get("expires_in")
    lifetime = expires_in if isinstance(expires_in, int) and expires_in > 0 else 300
    wait = float(interval) if isinstance(interval, int | float) and interval > 0 else 5.0
    return CibaRequest(Secret(auth_req_id), lifetime, wait, None, now + lifetime)


def _subject_value(subject_token: TokenSource | Secret) -> str:
    if isinstance(subject_token, Secret):
        return subject_token.reveal()
    if isinstance(subject_token, TokenSource):  # pyright: ignore[reportUnnecessaryIsInstance]
        return subject_token.token()
    raise ConfigurationError("subject_token must be a TokenSource or a Secret, never a plain string")


class AgentAuth(LegacyAgentMixin, _AgentBase, TokenSource):
    """An AI agent, for blocking code. Make it with :meth:`authfi.AuthFI.agent`.

    It is a :class:`~authfi.TokenSource` for its own token (``<issuer>/mcp`` unless ``resource``
    names another), so the gateway helpers and framework adapters take it directly.
    """

    @overload
    def __init__(
        self,
        project: AuthFI,
        agent_id: str,
        key: Signer,
        *,
        resource: str | None = None,
        send_to: Sequence[str] = (),
        refresh_skew: float = 60.0,
    ) -> None: ...

    @overload
    @deprecated(
        "AgentAuth(agent_id, key, platform_url=...) is AuthFI(platform_url).agent(agent_id, key)",
        category=AuthFIDeprecationWarning,
    )
    def __init__(
        self,
        agent_id: str,
        key: Signer,
        platform_url: str | None = None,
        audience: str | None = None,
        token_endpoint: str | None = None,
        backchannel_endpoint: str | None = None,
        discovery_url: str | None = None,
        refresh_skew: float = 60,
    ) -> None: ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if is_legacy_agent(args, kwargs):
            project, agent_id, key, options = legacy_agent_arguments(args, kwargs)
            self._init(project, agent_id, key, **options)
        else:
            self._init(*args, **kwargs)

    def _init(
        self,
        project: AuthFI,
        agent_id: str,
        key: Signer,
        *,
        resource: str | None = None,
        send_to: Sequence[str] = (),
        refresh_skew: float = 60.0,
    ) -> None:
        super().__init__(project, agent_id, key, resource=resource, send_to=send_to, refresh_skew=refresh_skew)
        self.project = project
        self._tokens: SyncCache[_Minted] = SyncCache(project.clock, max_entries=32)
        self._endpoints: SyncCache[AgentEndpoints] = SyncCache(project.clock, max_entries=1)

    def _send(self, request: HttpRequest) -> HttpResponse:
        return self.project._transport.send(request)  # pyright: ignore[reportPrivateUsage]

    # --- its own token -------------------------------------------------------------------------------

    def _mint(self, resource: str | None) -> tuple[_Minted, float]:
        metadata = self.project.metadata()
        _check_agent_grant(metadata, self.key)
        request = client_credentials_request(metadata.token_endpoint, self._auth, resource=resource)
        response = parse_token_response(self._send(request), "the token endpoint")
        self._log.debug("minted a token for agent %s", self.agent_id)
        return self._minted(response)

    def _own(self, resource: str | None, force: bool) -> _Minted:
        chosen = self._resource(resource)
        return self._tokens.get(chosen or "", lambda: self._mint(chosen), refresh=force)

    @overload
    def token(self, *, resource: str | None = None, force: bool = False) -> str: ...

    @overload
    @deprecated("AgentAuth.token(audience) is AgentAuth.token(resource=audience)", category=AuthFIDeprecationWarning)
    def token(self, audience: str | None, force: bool = False) -> str: ...

    def token(self, audience: str | None = None, force: bool = False, *, resource: str | None = None) -> str:
        """The agent's own token: cached, renewed ``refresh_skew`` before it expires, one mint at a time."""
        if audience is not None:
            warnings.warn(
                "AgentAuth.token(audience) is AgentAuth.token(resource=audience)",
                AuthFIDeprecationWarning,
                stacklevel=2,
            )
            resource = resource or audience
        return self._own(resource, force).access_token.reveal()

    def capabilities(self, *, resource: str | None = None) -> frozenset[str]:
        """The capability names granted to the agent, as its token carries them."""
        return self._own(resource, False).capabilities

    def for_resource(self, resource: str, *, send_to: Sequence[str] = ()) -> TokenSource:
        """The agent's own token for ``resource``, sent only to ``send_to`` (or AuthFI, for its own)."""
        if not resource or not resource.strip():
            raise ConfigurationError("resource is required")
        return _AgentResource(self, resource.strip(), self._check_destination_args(resource.strip(), send_to))

    # --- acting for a person -------------------------------------------------------------------------

    def request_user_token(
        self,
        login_hint: str,
        *,
        binding_message: str,
        scope: Sequence[str] | None = None,
        expires_in: int | None = None,
    ) -> CibaRequest:
        """Ask a person, on their own device, to let this agent act for them (CIBA).

        ``binding_message`` is what they read ("Refund order 4411"). The agent never sees or relays a
        code. Wait for their decision with :meth:`await_user_token`.
        """
        metadata = self.project.metadata()
        endpoint = _ciba_endpoint(metadata)
        request = ciba_start_request(
            endpoint,
            self._auth,
            metadata.token_endpoint,
            login_hint=login_hint,
            binding_message=binding_message,
            scope=scope,
            expires_in=expires_in,
        )
        start = parse_ciba_start(self._send(request))
        self._log.info("agent %s asked a person to approve acting for them", self.agent_id)
        return self._ciba_request(start, self._clock.monotonic())

    @overload
    def await_user_token(
        self,
        request: CibaRequest,
        *,
        timeout: float = 300.0,
        resource: str | None = None,
        send_to: Sequence[str] = (),
    ) -> DelegatedToken: ...

    @overload
    @deprecated("await_user_token takes the CibaRequest request_user_token returns", category=AuthFIDeprecationWarning)
    def await_user_token(
        self,
        request: Mapping[str, Any] | str,
        timeout: float = 300.0,
        audience: str | None = None,
        sleep: object = None,
    ) -> DelegatedToken: ...

    def await_user_token(
        self,
        request: CibaRequest | Mapping[str, Any] | str,
        timeout: float = 300.0,
        audience: str | None = None,
        sleep: object = None,
        *,
        resource: str | None = None,
        send_to: Sequence[str] = (),
    ) -> DelegatedToken:
        """Wait for the person's decision, then return the token that acts for them.

        Polls at the request's interval, waiting it before the first poll. ``timeout`` (seconds) is
        how long to wait; the request stays open, and waiting on it again continues it. A refusal is
        :class:`ConsentDenied`; an expired request :class:`ConsentExpired`.
        """
        if not isinstance(request, CibaRequest) or audience is not None or sleep is not None:
            warnings.warn(
                "await_user_token takes the CibaRequest request_user_token returns; audience is resource, and "
                "sleep is not used",
                AuthFIDeprecationWarning,
                stacklevel=2,
            )
            request = _legacy_ciba_request(request, self._clock.monotonic())
            resource = resource or audience
        chosen = self._resource(resource)
        destinations = self._check_destination_args(chosen, send_to)
        metadata = self.project.metadata()
        flow = ciba_poll_flow(
            metadata.token_endpoint,
            self._auth,
            auth_req_id=request.auth_req_id,
            interval=request.interval,
            deadline=min(self._clock.monotonic() + timeout, request.deadline),
            now=self._clock.monotonic,
            resource=chosen,
        )
        response = self.project._transport.run(flow)  # pyright: ignore[reportPrivateUsage]
        self._log.info("agent %s now acts for a person (CIBA)", self.agent_id)
        return DelegatedToken(
            response=response,
            via="ciba",
            agent_id=self.agent_id,
            destinations=destinations,
            renew=None,
            refresh_skew=self.refresh_skew,
            time=self._clock.time,
            project=self.project,
        )

    def token_as_user(
        self,
        login_hint: str,
        *,
        binding_message: str,
        scope: Sequence[str] | None = None,
        expires_in: int | None = None,
        timeout: float = 300.0,
        resource: str | None = None,
        send_to: Sequence[str] = (),
    ) -> DelegatedToken:
        """:meth:`request_user_token`, then :meth:`await_user_token`."""
        request = self.request_user_token(
            login_hint, binding_message=binding_message, scope=scope, expires_in=expires_in
        )
        return self.await_user_token(request, timeout=timeout, resource=resource, send_to=send_to)

    def exchange(
        self,
        subject_token: TokenSource | Secret,
        *,
        resource: str | None = None,
        scope: Sequence[str] | None = None,
        send_to: Sequence[str] = (),
    ) -> DelegatedToken:
        """Act for the person whose token this is (RFC 8693): ``sub`` = them, ``act.sub`` = this agent.

        ``subject_token`` is their session (a :class:`~authfi.TokenSource`) or their access token held
        as a :class:`~authfi.Secret`. The project's policy decides whether this agent may act for
        them; a refusal is :class:`~authfi.OAuthError` ``access_denied``. The result is exchanged
        again near its expiry, while their own token lives.
        """
        chosen = self._resource(resource)
        destinations = self._check_destination_args(chosen, send_to)
        metadata = self.project.metadata()
        _exchange_supported(metadata)

        def mint() -> TokenResponse:
            if isinstance(subject_token, Secret) and not _secret_alive(subject_token, self._clock.time()):
                raise ConsentExpired("expired_token", "the person's own token has expired; they must sign in again")
            request = exchange_request(
                metadata.token_endpoint,
                subject_token=_subject_value(subject_token),
                actor_token=self.token(),
                resource=chosen,
                scope=scope,
            )
            return parse_token_response(self._send(request), "the token endpoint")

        delegated = DelegatedToken(
            response=mint(),
            via="exchange",
            agent_id=self.agent_id,
            destinations=destinations,
            renew=mint,
            refresh_skew=self.refresh_skew,
            time=self._clock.time,
            project=self.project,
        )
        self._log.info("agent %s now acts for a person (token exchange)", self.agent_id)
        return delegated

    # --- the gateways --------------------------------------------------------------------------------

    def gateway(self) -> Gateway:
        """The AuthFI gateways (MCP tools, models, cloud credentials) with this agent's own token."""
        return Gateway(self.project, self)

    # --- the action gate -----------------------------------------------------------------------------

    def transaction_token(
        self,
        action: str,
        *,
        resource: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> TransactionToken:
        """Freeze ``action`` (with ``resource`` and ``context``) for 60 seconds, for the gate to decide.

        Constraints are checked before the token is made, so an action over a limit never gets one.
        This is AuthFI's own action gate, named after the IETF Transaction Tokens draft without
        following it.
        """
        request = transaction_token_request(
            self.project.api_origin, self.token(), action=action, resource=resource, context=context
        )
        token, txn, expires_in = parse_transaction_token(self._send(request))
        expires_at = datetime.fromtimestamp(self._clock.time() + expires_in, UTC)
        return TransactionToken(token, txn, action, resource, dict(context or {}), expires_at)

    def authorize_action(self, txn: TransactionToken) -> ActionDecision:
        """The gate's decision on a frozen action: ``allow``, ``deny`` or ``approval_required``."""
        request = authorize_request(
            self._agent_endpoints().authorize,
            self.project.api_origin,
            agent_id=self.agent_id,
            transaction_token=txn.token,
            action=txn.action,
            resource=txn.resource,
            context=txn.context,
        )
        decision, approval_id, reason = parse_action_decision(self._send(request))
        self._log.info("the gate decided %s for agent %s: %s", txn.action, self.agent_id, decision)
        return ActionDecision(cast(Any, decision), approval_id, reason)

    # --- approvals -----------------------------------------------------------------------------------

    def _agent_endpoints(self) -> AgentEndpoints:
        def fetch() -> tuple[AgentEndpoints, float]:
            request = HttpRequest(
                "GET",
                self.project.issuer + "/.well-known/agent-configuration",
                what="the agent configuration",
                headers={"Accept": "application/json"},
                retry=Retry.SAFE,
                max_bytes=DOCUMENT_LIMIT,
            )
            response = self._send(request)
            if response.status == 404:
                return AgentEndpoints(None, None, None), ttl_from_headers(None)
            if response.status != 200:
                raise DiscoveryError(f"the agent configuration answered HTTP {response.status}", status=response.status)
            doc = response.json_object()
            return parse_agent_configuration(self.project.issuer, doc), ttl_from_headers(
                response.header("Cache-Control")
            )

        return self._endpoints.get("endpoints", fetch)

    def request_approval(
        self,
        action: str,
        *,
        context: Mapping[str, Any] | None = None,
        expires_in: int = 300,
    ) -> Approval:
        """Ask a person to approve one ``action``. No token results; wait with :meth:`wait_for_approval`."""
        endpoint = self._agent_endpoints().request_approval
        if endpoint is None:
            raise NotSupportedError("agent_approval", "This AuthFI project does not offer approvals to agents.")
        request = approval_request(
            endpoint,
            agent_id=self.agent_id,
            assertion=self._assertion(self.project.issuer),
            action=action,
            context=context,
            expires_in=expires_in,
        )
        approval = Approval.of(parse_approval(self._send(request), reading=False))
        self._log.info("agent %s asked for approval %s", self.agent_id, approval.approval_id)
        return approval

    def approval(self, approval_id: str) -> Approval:
        """The current state of an approval this agent raised."""
        template = self._agent_endpoints().approval
        if template is None:
            raise NotSupportedError(
                "agent_approval_status", "This AuthFI project does not offer approval status to agents."
            )
        response = self._send(read_approval_request(template, approval_id, self.token()))
        return Approval.of(parse_approval(response, reading=True))

    def wait_for_approval(self, approval_id: str, *, timeout: float = 300.0) -> Approval:
        """Poll :meth:`approval` until a person decides; returns it whatever they decided.

        At ``timeout`` it raises ``authorization_pending``; the approval stays open.
        """
        deadline = self._clock.monotonic() + timeout
        while True:
            current = self.approval(approval_id)
            if current.decided:
                return current
            if self._clock.monotonic() + _APPROVAL_POLL > deadline:
                raise OAuthError("authorization_pending", f"approval {approval_id} is still pending")
            self._clock.sleep(_APPROVAL_POLL)


class AsyncAgentAuth(_AgentBase, AsyncTokenSource):
    """An AI agent, for asyncio code. Make it with :meth:`authfi.AsyncAuthFI.agent`. See :class:`AgentAuth`."""

    def __init__(
        self,
        project: AsyncAuthFI,
        agent_id: str,
        key: Signer,
        *,
        resource: str | None = None,
        send_to: Sequence[str] = (),
        refresh_skew: float = 60.0,
    ) -> None:
        super().__init__(project, agent_id, key, resource=resource, send_to=send_to, refresh_skew=refresh_skew)
        self.project = project
        self._tokens: AsyncCache[_Minted] = AsyncCache(project.clock, max_entries=32)
        self._endpoints: AsyncCache[AgentEndpoints] = AsyncCache(project.clock, max_entries=1)

    async def _send(self, request: HttpRequest) -> HttpResponse:
        return await self.project._transport.send(request)  # pyright: ignore[reportPrivateUsage]

    async def _mint(self, resource: str | None) -> tuple[_Minted, float]:
        metadata = await self.project.metadata()
        _check_agent_grant(metadata, self.key)
        request = client_credentials_request(metadata.token_endpoint, self._auth, resource=resource)
        response = parse_token_response(await self._send(request), "the token endpoint")
        self._log.debug("minted a token for agent %s", self.agent_id)
        return self._minted(response)

    async def _own(self, resource: str | None, force: bool) -> _Minted:
        chosen = self._resource(resource)
        return await self._tokens.get(chosen or "", lambda: self._mint(chosen), refresh=force)

    async def token(self, *, resource: str | None = None, force: bool = False) -> str:
        """See :meth:`AgentAuth.token`."""
        return (await self._own(resource, force)).access_token.reveal()

    async def capabilities(self, *, resource: str | None = None) -> frozenset[str]:
        """See :meth:`AgentAuth.capabilities`."""
        return (await self._own(resource, False)).capabilities

    def for_resource(self, resource: str, *, send_to: Sequence[str] = ()) -> AsyncTokenSource:
        """See :meth:`AgentAuth.for_resource`."""
        if not resource or not resource.strip():
            raise ConfigurationError("resource is required")
        return _AsyncAgentResource(self, resource.strip(), self._check_destination_args(resource.strip(), send_to))

    async def request_user_token(
        self,
        login_hint: str,
        *,
        binding_message: str,
        scope: Sequence[str] | None = None,
        expires_in: int | None = None,
    ) -> CibaRequest:
        """See :meth:`AgentAuth.request_user_token`."""
        metadata = await self.project.metadata()
        endpoint = _ciba_endpoint(metadata)
        request = ciba_start_request(
            endpoint,
            self._auth,
            metadata.token_endpoint,
            login_hint=login_hint,
            binding_message=binding_message,
            scope=scope,
            expires_in=expires_in,
        )
        start = parse_ciba_start(await self._send(request))
        self._log.info("agent %s asked a person to approve acting for them", self.agent_id)
        return self._ciba_request(start, self._clock.monotonic())

    async def await_user_token(
        self,
        request: CibaRequest,
        *,
        timeout: float = 300.0,
        resource: str | None = None,
        send_to: Sequence[str] = (),
    ) -> AsyncDelegatedToken:
        """See :meth:`AgentAuth.await_user_token`. Cancel the task to stop waiting."""
        chosen = self._resource(resource)
        destinations = self._check_destination_args(chosen, send_to)
        metadata = await self.project.metadata()
        flow = ciba_poll_flow(
            metadata.token_endpoint,
            self._auth,
            auth_req_id=request.auth_req_id,
            interval=request.interval,
            deadline=min(self._clock.monotonic() + timeout, request.deadline),
            now=self._clock.monotonic,
            resource=chosen,
        )
        response = await self.project._transport.run(flow)  # pyright: ignore[reportPrivateUsage]
        self._log.info("agent %s now acts for a person (CIBA)", self.agent_id)
        return AsyncDelegatedToken(
            response=response,
            via="ciba",
            agent_id=self.agent_id,
            destinations=destinations,
            renew=None,
            refresh_skew=self.refresh_skew,
            time=self._clock.time,
            project=self.project,
        )

    async def token_as_user(
        self,
        login_hint: str,
        *,
        binding_message: str,
        scope: Sequence[str] | None = None,
        expires_in: int | None = None,
        timeout: float = 300.0,
        resource: str | None = None,
        send_to: Sequence[str] = (),
    ) -> AsyncDelegatedToken:
        """See :meth:`AgentAuth.token_as_user`."""
        request = await self.request_user_token(
            login_hint, binding_message=binding_message, scope=scope, expires_in=expires_in
        )
        return await self.await_user_token(request, timeout=timeout, resource=resource, send_to=send_to)

    async def exchange(
        self,
        subject_token: AsyncTokenSource | Secret,
        *,
        resource: str | None = None,
        scope: Sequence[str] | None = None,
        send_to: Sequence[str] = (),
    ) -> AsyncDelegatedToken:
        """See :meth:`AgentAuth.exchange`."""
        chosen = self._resource(resource)
        destinations = self._check_destination_args(chosen, send_to)
        metadata = await self.project.metadata()
        _exchange_supported(metadata)

        async def mint() -> TokenResponse:
            if isinstance(subject_token, Secret):
                if not _secret_alive(subject_token, self._clock.time()):
                    raise ConsentExpired("expired_token", "the person's own token has expired; they must sign in again")
                subject = subject_token.reveal()
            elif isinstance(subject_token, AsyncTokenSource):  # pyright: ignore[reportUnnecessaryIsInstance]
                subject = await subject_token.token()
            else:
                raise ConfigurationError("subject_token must be an AsyncTokenSource or a Secret, never a plain string")
            request = exchange_request(
                metadata.token_endpoint,
                subject_token=subject,
                actor_token=await self.token(),
                resource=chosen,
                scope=scope,
            )
            return parse_token_response(await self._send(request), "the token endpoint")

        delegated = AsyncDelegatedToken(
            response=await mint(),
            via="exchange",
            agent_id=self.agent_id,
            destinations=destinations,
            renew=mint,
            refresh_skew=self.refresh_skew,
            time=self._clock.time,
            project=self.project,
        )
        self._log.info("agent %s now acts for a person (token exchange)", self.agent_id)
        return delegated

    def gateway(self) -> AsyncGateway:
        """See :meth:`AgentAuth.gateway`."""
        return AsyncGateway(self.project, self)

    async def transaction_token(
        self,
        action: str,
        *,
        resource: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> TransactionToken:
        """See :meth:`AgentAuth.transaction_token`."""
        request = transaction_token_request(
            self.project.api_origin, await self.token(), action=action, resource=resource, context=context
        )
        token, txn, expires_in = parse_transaction_token(await self._send(request))
        expires_at = datetime.fromtimestamp(self._clock.time() + expires_in, UTC)
        return TransactionToken(token, txn, action, resource, dict(context or {}), expires_at)

    async def authorize_action(self, txn: TransactionToken) -> ActionDecision:
        """See :meth:`AgentAuth.authorize_action`."""
        request = authorize_request(
            (await self._agent_endpoints()).authorize,
            self.project.api_origin,
            agent_id=self.agent_id,
            transaction_token=txn.token,
            action=txn.action,
            resource=txn.resource,
            context=txn.context,
        )
        decision, approval_id, reason = parse_action_decision(await self._send(request))
        self._log.info("the gate decided %s for agent %s: %s", txn.action, self.agent_id, decision)
        return ActionDecision(cast(Any, decision), approval_id, reason)

    async def _agent_endpoints(self) -> AgentEndpoints:
        async def fetch() -> tuple[AgentEndpoints, float]:
            request = HttpRequest(
                "GET",
                self.project.issuer + "/.well-known/agent-configuration",
                what="the agent configuration",
                headers={"Accept": "application/json"},
                retry=Retry.SAFE,
                max_bytes=DOCUMENT_LIMIT,
            )
            response = await self._send(request)
            if response.status == 404:
                return AgentEndpoints(None, None, None), ttl_from_headers(None)
            if response.status != 200:
                raise DiscoveryError(f"the agent configuration answered HTTP {response.status}", status=response.status)
            doc = response.json_object()
            return parse_agent_configuration(self.project.issuer, doc), ttl_from_headers(
                response.header("Cache-Control")
            )

        return await self._endpoints.get("endpoints", fetch)

    async def request_approval(
        self,
        action: str,
        *,
        context: Mapping[str, Any] | None = None,
        expires_in: int = 300,
    ) -> Approval:
        """See :meth:`AgentAuth.request_approval`."""
        endpoint = (await self._agent_endpoints()).request_approval
        if endpoint is None:
            raise NotSupportedError("agent_approval", "This AuthFI project does not offer approvals to agents.")
        request = approval_request(
            endpoint,
            agent_id=self.agent_id,
            assertion=self._assertion(self.project.issuer),
            action=action,
            context=context,
            expires_in=expires_in,
        )
        approval = Approval.of(parse_approval(await self._send(request), reading=False))
        self._log.info("agent %s asked for approval %s", self.agent_id, approval.approval_id)
        return approval

    async def approval(self, approval_id: str) -> Approval:
        """See :meth:`AgentAuth.approval`."""
        template = (await self._agent_endpoints()).approval
        if template is None:
            raise NotSupportedError(
                "agent_approval_status", "This AuthFI project does not offer approval status to agents."
            )
        response = await self._send(read_approval_request(template, approval_id, await self.token()))
        return Approval.of(parse_approval(response, reading=True))

    async def wait_for_approval(self, approval_id: str, *, timeout: float = 300.0) -> Approval:
        """See :meth:`AgentAuth.wait_for_approval`. Cancel the task to stop waiting."""
        deadline = self._clock.monotonic() + timeout
        while True:
            current = await self.approval(approval_id)
            if current.decided:
                return current
            if self._clock.monotonic() + _APPROVAL_POLL > deadline:
                raise OAuthError("authorization_pending", f"approval {approval_id} is still pending")
            await self._clock.asleep(_APPROVAL_POLL)


class _AsyncAgentResource(AsyncTokenSource):
    """See :class:`_AgentResource`."""

    __slots__ = ("_agent", "_destinations", "_resource")

    def __init__(self, agent: AsyncAgentAuth, resource: str, destinations: Destinations) -> None:
        self._agent = agent
        self._resource = resource
        self._destinations = destinations

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`."""
        return self._destinations.origins

    async def token(self, *, force: bool = False) -> str:
        """The agent's token for this resource."""
        return await self._agent.token(resource=self._resource, force=force)

    def __repr__(self) -> str:
        return f"AsyncAgentAuth.for_resource({self._resource!r})"
