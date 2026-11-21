"""A sub-agent: an agent of its own that a running agent makes for one part of its work.

::

    agent = af.agent(AGENT_ID, key)
    with agent.sub_agent("crm lookup", permissions=["mcp:crm.read"]) as crm:
        integration = LangGraphIntegration(crm)      # the sub-agent is the credential: its calls are its own
        ...

A sub-agent is a credential, like the agent's own token, and an adapter takes it in the agent's place. It has an
agent id and a WIMSE id of its own, so the console and the audit tape name it and not its parent, and every call it
makes is held to what it holds. It holds no key, so only the agent that made it can get a token for it, and it holds
only what that agent holds: asking for more raises :class:`~authfi.APIError` with the code ``ESUBAGENT002`` and
nothing is made. A sub-agent can make sub-agents the same way, up to eight generations below the registered agent.

Leaving the ``with`` forgets the credential. The platform's record of the sub-agent stays, as evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, NoReturn, Self

from authfi import _dpop
from authfi._protocol.http import HttpRequest, HttpResponse
from authfi._protocol.ids import link_step_owner
from authfi._protocol.subagent import (
    SUB_AGENT_PATH,
    SubAgentGrant,
    parse_sub_agent,
    sub_agent_request,
    sub_agent_token_request,
)
from authfi.errors import ConfigurationError
from authfi.secret import Secret
from authfi.tokens import AsyncTokenSource, TokenSource, check_destination

if TYPE_CHECKING:
    from authfi.agents.agent import AgentAuth, AsyncAgentAuth
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = ["AsyncSubAgent", "SubAgent"]

#: A token is renewed when it has this many seconds left.
_RENEW_SKEW = 30.0


def _proved_key(parent: object) -> _dpop.DPoPKey | None:
    """The key the parent proved on the request that made the sub-agent: the one its token is bound to."""
    if isinstance(parent, _SubAgentState):
        return parent._key  # pyright: ignore[reportPrivateUsage]
    key: object = getattr(parent, "_last_key", None)
    return key if isinstance(key, _dpop.DPoPKey) else None


def _framework_of(parent: object) -> str | None:
    value = getattr(parent, "framework", None)
    return value if isinstance(value, str) and value else None


class _SubAgentState:
    """What a sub-agent is: the platform's record of it, its token, and the key that token is bound to."""

    grant: SubAgentGrant
    #: Whoever made it (an agent or another sub-agent), the registered agent at the top, and the project's connection.
    #: Typed in each subclass: blocking and asyncio code never share a parent.
    parent: Any
    agent: Any
    project: Any

    def _setup(self, grant: SubAgentGrant, key: _dpop.DPoPKey | None, time: Callable[[], float]) -> None:
        self.grant = grant
        self._token: Secret | None = grant.access_token
        self._closed = False
        self._key = key
        self._time = time
        self._expires = time() + float(grant.expires_in)
        # The framework exports its steps for the credential it was started with; this one names the step through it.
        link_step_owner(self, self.parent)

    @property
    def agent_id(self) -> str:
        """The sub-agent's own agent id (``agt_...``): the one on every call it makes and on the audit tape."""
        return self.grant.agent_id

    @property
    def wimse_id(self) -> str:
        """The sub-agent's WIMSE identifier, empty when the platform recorded none."""
        return self.grant.wimse_id

    @property
    def name(self) -> str:
        """The name it was given, which people read in the console."""
        return self.grant.name

    @property
    def parent_agent_id(self) -> str:
        """The agent that made it."""
        return self.grant.parent_agent_id

    @property
    def root_agent_id(self) -> str:
        """The registered agent at the top of its tree."""
        return self.grant.root_agent_id

    @property
    def depth(self) -> int:
        """How many generations below the registered agent it is: 1 for a sub-agent of the registered agent."""
        return self.grant.depth

    @property
    def chain(self) -> tuple[str, ...]:
        """The agents above it: its parent first, the registered agent last."""
        return self.grant.chain

    @property
    def permissions(self) -> tuple[str, ...]:
        """What it holds now: no more than its parent holds."""
        return self.grant.capabilities

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`: AuthFI's origins, as the agent's own token."""
        return self.agent.send_to

    def bound_key(self) -> _dpop.DPoPKey | None:
        """The key the token is bound to (the one its parent proved), or ``None`` for a Bearer token."""
        return self._key

    def close(self) -> None:
        """Forget the credential. A closed sub-agent is not renewed: asking it for a token is an error."""
        self._token = None
        self._closed = True

    def _expiring(self) -> bool:
        return not self._closed and self._time() >= self._expires - _RENEW_SKEW

    def _live(self) -> str:
        if self._token is None:
            raise ConfigurationError("this sub-agent is closed: its credential was forgotten when its with block ended")
        return self._token.reveal()

    def _take(self, grant: SubAgentGrant) -> None:
        self.grant = SubAgentGrant(
            agent_id=self.grant.agent_id,
            wimse_id=self.grant.wimse_id,
            name=self.grant.name,
            parent_agent_id=self.grant.parent_agent_id,
            root_agent_id=self.grant.root_agent_id,
            depth=self.grant.depth,
            chain=self.grant.chain,
            # What it holds is what the platform says now: a grant its parent lost is not in the new token.
            capabilities=grant.capabilities,
            access_token=grant.access_token,
            token_type=grant.token_type,
            expires_in=grant.expires_in,
        )
        self._token = grant.access_token
        self._expires = self._time() + float(grant.expires_in)

    def _bearing(self, method: str, url: str) -> _dpop.AuthHeaders:
        check_destination(url, self.send_to)
        return _dpop.authorization_headers(self._live(), self._key, method, url, clock=self._time)

    def _renew_path(self) -> str:
        return SUB_AGENT_PATH + "/" + self.grant.agent_id + "/token"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(agent_id={self.agent_id!r}, parent={self.parent_agent_id!r}, depth={self.depth})"

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self

    def __reduce__(self) -> NoReturn:
        raise TypeError("A sub-agent cannot be pickled: its credential is held by a key that belongs to this process")


class SubAgent(_SubAgentState, TokenSource):
    """A sub-agent in blocking code: made with ``agent.sub_agent(...)``, taken by an adapter in the agent's place."""

    parent: AgentAuth | SubAgent
    agent: AgentAuth
    project: AuthFI

    def __init__(
        self,
        parent: AgentAuth | SubAgent,
        grant: SubAgentGrant,
        key: _dpop.DPoPKey | None,
        time: Callable[[], float],
    ) -> None:
        self.parent = parent
        self.agent = parent.agent if isinstance(parent, SubAgent) else parent
        self.project = self.agent.project
        self._setup(grant, key, time)

    def token(self, *, force: bool = False) -> str:
        """The sub-agent's token, renewed from its parent's grants as they are now when it is about to end."""
        if force or self._expiring():
            self._renew()
        return self._live()

    def headers(self, method: str, url: str) -> dict[str, str]:
        """The sub-agent's credential to ``url``: ``Authorization: DPoP`` and a fresh proof."""
        check_destination(url, self.send_to)
        if self._expiring():
            self._renew()
        return self._bearing(method, url)

    def retry_headers(
        self, method: str, url: str, status: int, www_authenticate: str | None, nonce: str | None
    ) -> dict[str, str] | None:
        """The one retry a server nonce asks for."""
        if self._key is not None and _dpop.learn_nonce(url, status, www_authenticate, nonce):
            return self.headers(method, url)
        return None

    def _renew(self) -> None:
        url = self.project.resolve_api_origin() + self._renew_path()
        response = self.parent._send_own(  # pyright: ignore[reportPrivateUsage]
            "POST", url, lambda credential: sub_agent_token_request(url, credential, audience=None)
        )
        self._take(parse_sub_agent(response, "the sub-agent token endpoint"))

    def _send_own(self, method: str, url: str, build: Callable[[Mapping[str, str]], HttpRequest]) -> HttpResponse:
        """Call AuthFI's own endpoint as this sub-agent (to make one below it)."""
        response = self.agent._send(build(self.headers(method, url)))  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        if self._key is not None and _dpop.learn_nonce(
            url, response.status, response.header("WWW-Authenticate"), response.header("DPoP-Nonce")
        ):
            response = self.agent._send(build(self.headers(method, url)))  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        return response

    def sub_agent(
        self,
        name: str,
        *,
        permissions: Sequence[str],
        audience: str | None = None,
        run_id: str | None = None,
        framework: str | None = None,
    ) -> SubAgent:
        """A sub-agent of this one: see :meth:`authfi.agents.AgentAuth.sub_agent`."""
        return spawn(self, name, permissions, audience, run_id, framework)


class AsyncSubAgent(_SubAgentState, AsyncTokenSource):
    """A sub-agent, in asyncio code. Make it with ``await agent.sub_agent(...)``."""

    parent: AsyncAgentAuth | AsyncSubAgent
    agent: AsyncAgentAuth
    project: AsyncAuthFI

    def __init__(
        self,
        parent: AsyncAgentAuth | AsyncSubAgent,
        grant: SubAgentGrant,
        key: _dpop.DPoPKey | None,
        time: Callable[[], float],
    ) -> None:
        self.parent = parent
        self.agent = parent.agent if isinstance(parent, AsyncSubAgent) else parent
        self.project = self.agent.project
        self._setup(grant, key, time)

    async def token(self, *, force: bool = False) -> str:
        """See :meth:`SubAgent.token`."""
        if force or self._expiring():
            await self._renew()
        return self._live()

    async def headers(self, method: str, url: str) -> dict[str, str]:
        """See :meth:`SubAgent.headers`."""
        check_destination(url, self.send_to)
        if self._expiring():
            await self._renew()
        return self._bearing(method, url)

    async def retry_headers(
        self, method: str, url: str, status: int, www_authenticate: str | None, nonce: str | None
    ) -> dict[str, str] | None:
        """The one retry a server nonce asks for."""
        if self._key is not None and _dpop.learn_nonce(url, status, www_authenticate, nonce):
            return await self.headers(method, url)
        return None

    async def _renew(self) -> None:
        url = await self.project.resolve_api_origin() + self._renew_path()
        response = await self.parent._send_own(  # pyright: ignore[reportPrivateUsage]
            "POST", url, lambda credential: sub_agent_token_request(url, credential, audience=None)
        )
        self._take(parse_sub_agent(response, "the sub-agent token endpoint"))

    async def _send_own(self, method: str, url: str, build: Callable[[Mapping[str, str]], HttpRequest]) -> HttpResponse:
        """See :meth:`SubAgent._send_own`."""
        response = await self.agent._send(build(await self.headers(method, url)))  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        if self._key is not None and _dpop.learn_nonce(
            url, response.status, response.header("WWW-Authenticate"), response.header("DPoP-Nonce")
        ):
            response = await self.agent._send(build(await self.headers(method, url)))  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]
        return response

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    async def sub_agent(
        self,
        name: str,
        *,
        permissions: Sequence[str],
        audience: str | None = None,
        run_id: str | None = None,
        framework: str | None = None,
    ) -> AsyncSubAgent:
        """See :meth:`SubAgent.sub_agent`."""
        return await spawn_async(self, name, permissions, audience, run_id, framework)


def spawn(
    parent: AgentAuth | SubAgent,
    name: str,
    permissions: Sequence[str],
    audience: str | None,
    run_id: str | None,
    framework: str | None,
) -> SubAgent:
    """Make a sub-agent of ``parent`` (an agent or a sub-agent) and its first token."""
    url = parent.project.resolve_api_origin() + SUB_AGENT_PATH
    response = parent._send_own(  # pyright: ignore[reportPrivateUsage]
        "POST",
        url,
        lambda credential: sub_agent_request(
            url,
            credential,
            name=name,
            permissions=permissions,
            audience=audience,
            run_id=run_id,
            framework=framework or _framework_of(parent),
        ),
    )
    grant = parse_sub_agent(response)
    return SubAgent(parent, grant, _proved_key(parent), parent.project.clock.time)


async def spawn_async(
    parent: AsyncAgentAuth | AsyncSubAgent,
    name: str,
    permissions: Sequence[str],
    audience: str | None,
    run_id: str | None,
    framework: str | None,
) -> AsyncSubAgent:
    """See :func:`spawn`."""
    url = await parent.project.resolve_api_origin() + SUB_AGENT_PATH
    response = await parent._send_own(  # pyright: ignore[reportPrivateUsage]
        "POST",
        url,
        lambda credential: sub_agent_request(
            url,
            credential,
            name=name,
            permissions=permissions,
            audience=audience,
            run_id=run_id,
            framework=framework or _framework_of(parent),
        ),
    )
    grant = parse_sub_agent(response)
    return AsyncSubAgent(parent, grant, _proved_key(parent), parent.project.clock.time)
