"""A2A 1.0 client helpers: trust a remote agent's card, call it with this agent's token, read its task state.

::

    remote = authfi.a2a.remote(af, "https://research.partner.example", trust_roots=[R1_PEM])
    status = remote.send("Summarise the Q3 incidents")
    if status.needs_input:   ...     # TASK_STATE_INPUT_REQUIRED: a person must answer
    if status.needs_auth:    ...     # TASK_STATE_AUTH_REQUIRED: consent or credentials are missing

- **The card is verified or refused.** ``trust_roots`` are the roots you trust (an AuthFI agent's chain ends at Root
  R1); ``keys`` are public JWKs by ``kid`` for agents that sign with a plain key. A card with no valid signature is
  refused unless you pass ``allow_unsigned=True``, which says so in its name.
- **The token goes where the card says, and only there.** The agent's DPoP-bound token (or its Bearer token, for a
  credential that is not bound) is sent to the card's own origin, or to an origin you list in ``allow_origins``; a
  card that points elsewhere is refused. The token is for ``resource`` (the interface's origin unless you name another).
- **The card's security scheme must be one this SDK speaks:** OAuth 2, OpenID Connect or HTTP bearer.
- :func:`status_for` and :func:`status_for_outcome` turn an AuthFI decision into the state a remote caller reads.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from authfi import _dpop
from authfi._protocol.a2a import (
    A2AError,
    AgentCard,
    Interface,
    TaskStatus,
    card_request,
    message_body,
    message_request,
    message_url,
    parse_card,
    parse_task,
    status_for,
    status_for_outcome,
    verify_card,
)
from authfi._protocol.http import HttpResponse
from authfi._protocol.urls import origin_of
from authfi.agents.agent import AgentAuth, AsyncAgentAuth
from authfi.tokens import AsyncTokenSource, TokenSource

__all__ = [
    "A2AError",
    "AgentCard",
    "AsyncRemoteAgent",
    "RemoteAgent",
    "TaskStatus",
    "async_remote",
    "remote",
    "status_for",
    "status_for_outcome",
]

_SPOKEN = ("oauth2SecurityScheme", "openIdConnectSecurityScheme", "httpAuthSecurityScheme")


def _checked(
    card: AgentCard,
    *,
    trust_roots: Sequence[bytes | str],
    keys: Mapping[str, Mapping[str, Any]] | None,
    allow_unsigned: bool,
) -> AgentCard:
    if allow_unsigned and not card.raw.get("signatures"):
        return card
    return verify_card(card, trust_roots=trust_roots, keys=keys)


def _interface_for(card: AgentCard, origin: str, allow_origins: Sequence[str]) -> Interface:
    interface = card.interface()
    target = origin_of(interface.url)
    if target != origin and target not in allow_origins:
        raise A2AError(
            "interface_origin",
            f"{card.name}'s card points its interface at {target}, not {origin}. Pass allow_origins=[{target!r}] if "
            "you trust that, or the token will not be sent.",
        )
    if not any(scheme in _SPOKEN for scheme in card.schemes):
        raise A2AError(
            "no_scheme",
            f"{card.name} accepts {sorted(card.schemes) or 'no declared authentication'}; this SDK presents an OAuth 2 "
            "or HTTP bearer token",
        )
    return interface


def _nonce_retry(response: HttpResponse, url: str) -> bool:
    nonce = response.header("DPoP-Nonce")
    if nonce and response.status in (400, 401):
        _dpop.nonces.remember(url, nonce)
        return True
    return False


class RemoteAgent:
    """A remote A2A agent whose card is verified, for blocking code. Make it with :func:`remote`."""

    def __init__(self, project: Any, card: AgentCard, interface: Interface, source: TokenSource) -> None:
        self.card = card
        self.interface = interface
        self._project = project
        self._source = source

    def send(self, text: str, *, context_id: str | None = None, task_id: str | None = None) -> TaskStatus:
        """Send one text message; return the task's state and the agent's words."""
        body = message_body(text, context_id=context_id, task_id=task_id)
        send = self._project._transport.send  # pyright: ignore[reportPrivateUsage]
        url = message_url(self.interface)
        response = send(message_request(self.interface, body, self._source.headers("POST", url)))
        if _nonce_retry(response, url):  # the remote asked for its nonce: once more, with it
            response = send(message_request(self.interface, body, self._source.headers("POST", url)))
        return parse_task(response)

    def __repr__(self) -> str:
        return f"RemoteAgent({self.card.name!r}, verified={self.card.verified})"


class AsyncRemoteAgent:
    """:class:`RemoteAgent` for asyncio code. Make it with :func:`async_remote`."""

    def __init__(self, project: Any, card: AgentCard, interface: Interface, source: AsyncTokenSource) -> None:
        self.card = card
        self.interface = interface
        self._project = project
        self._source = source

    async def send(self, text: str, *, context_id: str | None = None, task_id: str | None = None) -> TaskStatus:
        """See :meth:`RemoteAgent.send`."""
        body = message_body(text, context_id=context_id, task_id=task_id)
        send = self._project._transport.send  # pyright: ignore[reportPrivateUsage]
        url = message_url(self.interface)
        response = await send(message_request(self.interface, body, await self._source.headers("POST", url)))
        if _nonce_retry(response, url):  # the remote asked for its nonce: once more, with it
            response = await send(message_request(self.interface, body, await self._source.headers("POST", url)))
        return parse_task(response)

    def __repr__(self) -> str:
        return f"AsyncRemoteAgent({self.card.name!r}, verified={self.card.verified})"


def remote(
    agent: AgentAuth,
    url: str,
    *,
    trust_roots: Sequence[bytes | str] = (),
    keys: Mapping[str, Mapping[str, Any]] | None = None,
    allow_unsigned: bool = False,
    allow_origins: Sequence[str] = (),
    resource: str | None = None,
) -> RemoteAgent:
    """Fetch and verify the card at ``url``'s origin, and return the agent to call."""
    project = agent.project
    origin = origin_of(url)
    response = project._transport.send(card_request(origin))  # pyright: ignore[reportPrivateUsage]
    card = _checked(parse_card(response), trust_roots=trust_roots, keys=keys, allow_unsigned=allow_unsigned)
    interface = _interface_for(card, origin, allow_origins)
    source = agent.for_resource(resource or origin_of(interface.url), send_to=[origin_of(interface.url)])
    return RemoteAgent(project, card, interface, source)


async def async_remote(
    agent: AsyncAgentAuth,
    url: str,
    *,
    trust_roots: Sequence[bytes | str] = (),
    keys: Mapping[str, Mapping[str, Any]] | None = None,
    allow_unsigned: bool = False,
    allow_origins: Sequence[str] = (),
    resource: str | None = None,
) -> AsyncRemoteAgent:
    """See :func:`remote`."""
    project = agent.project
    origin = origin_of(url)
    response = await project._transport.send(card_request(origin))  # pyright: ignore[reportPrivateUsage]
    card = _checked(parse_card(response), trust_roots=trust_roots, keys=keys, allow_unsigned=allow_unsigned)
    interface = _interface_for(card, origin, allow_origins)
    source = agent.for_resource(resource or origin_of(interface.url), send_to=[origin_of(interface.url)])
    return AsyncRemoteAgent(project, card, interface, source)
