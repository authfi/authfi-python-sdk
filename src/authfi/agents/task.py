"""A task: the bounded authority a person gives an agent for one piece of work (a mandate).

::

    with agent.task("support.refund", subject_token=signed_in_person, params={"order": "4411"}) as task:
        integration = LangGraphIntegration(task)     # the task is the credential: the agent's calls are held to it
        ...

A task is a credential, like the agent's own token, and an adapter takes it in the agent's place. The gateway
then holds each call to the task: a tool outside it, a spend over it and an ended task are refused (see
:class:`~authfi.ToolRefused`), and the refusal says what the agent can still do.

Nothing swaps the agent's own token for the task's behind your back. A framework may keep an MCP session
open from before the task, and a call through it would go with the agent's token, wider than the task: so the
task is passed in, and a session is made with it.

A task is one level deep. Nothing a task does can widen it or start another: a ``task`` inside a ``with task``
is refused, and a refused call has no "ask for more" answer. Only a person gives a task. Leaving the ``with``
forgets the credential.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from datetime import UTC, datetime
from types import TracebackType
from typing import TYPE_CHECKING, Any, NoReturn, Self, cast

from authfi import _dpop
from authfi._protocol.agent import TaskDetails, task_details
from authfi._protocol.oauth import TokenResponse
from authfi.errors import ConfigurationError, ConsentExpired
from authfi.gateway import AsyncCloudAccess, AsyncGateway, CloudAccess, Gateway
from authfi.secret import Secret
from authfi.tokens import AsyncTokenSource, TokenSource, check_destination

if TYPE_CHECKING:
    from authfi.agents.agent import AgentAuth, AsyncAgentAuth
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = ["AsyncTask", "Task"]

#: Whether the running code is inside a ``with`` task: only to refuse a second one.
_INSIDE: contextvars.ContextVar[bool] = contextvars.ContextVar("authfi_inside_task", default=False)


def refuse_nested() -> None:
    """A task is one level deep: nothing started inside one can be a task."""
    if _INSIDE.get():
        raise ConfigurationError(
            "a task is one level deep: you are inside one already. Nothing a task does can start another or "
            "widen it. Finish this task, and ask the person for the next.",
            code="task_nested",
        )


class _TaskState:
    """What a task is: its root token, the key that holds it, and what the platform granted."""

    details: TaskDetails

    def __init__(
        self,
        agent: AgentAuth | AsyncAgentAuth,
        response: TokenResponse,
        key: _dpop.DPoPKey,
        time: Callable[[], float],
    ) -> None:
        self.agent = agent
        #: The project, which is what an adapter attaches a credential to.
        self.project = agent.project
        #: The agent that holds the task (an adapter labels its traces with it).
        self.agent_id = agent.agent_id
        self.details = task_details(response)
        self._root: Secret | None = response.access_token
        self._key = key
        self._time = time
        self._expires = time() + float(response.expires_in or 300)
        self._reset: list[contextvars.Token[bool]] = []

    @property
    def mandate_id(self) -> str:
        """The task's id, which the platform's record of it carries."""
        return self.details.mandate_id

    @property
    def template(self) -> str:
        """The template the task was made from, such as ``support.refund@3``."""
        return self.details.template

    @property
    def tools(self) -> tuple[str, ...]:
        """The tools the task allows."""
        return self.details.tools

    @property
    def expires_at(self) -> datetime:
        """When the task's credential expires."""
        return datetime.fromtimestamp(self._expires, UTC)

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`authfi.TokenSource.send_to`: AuthFI's origins, as the agent's own token."""
        return self.agent.send_to

    def close(self) -> None:
        """Forget the credential. The platform's record of the task ends with its deadline or a person's decision."""
        self._root = None

    def _live(self) -> str:
        if self._root is None:
            raise ConfigurationError("this task is closed: its credential was forgotten when its with block ended")
        if self._time() >= self._expires:
            raise ConsentExpired("expired_token", "the task has run out; only a person can give the agent another")
        return self._root.reveal()

    def _headers(self, method: str, url: str) -> dict[str, str]:
        check_destination(url, self.send_to)
        return _dpop.authorization_headers(self._live(), self._key, method, url, clock=self._time)

    def bound_key(self) -> _dpop.DPoPKey:
        """The key that holds the task."""
        return self._key

    def _enter(self) -> Self:
        refuse_nested()
        self._reset.append(_INSIDE.set(True))
        return self

    def _exit(self) -> None:
        _INSIDE.reset(self._reset.pop())
        if not self._reset:
            self.close()

    def __enter__(self) -> Self:
        return self._enter()

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._exit()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(mandate_id={self.mandate_id!r}, template={self.template!r})"

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        return self

    def __reduce__(self) -> NoReturn:
        raise TypeError("A task cannot be pickled: its credential is held by a key that belongs to this process")


class Task(_TaskState, TokenSource):
    """A task, in blocking code. It is a :class:`~authfi.TokenSource`: give it to an adapter or a client."""

    def __init__(
        self, agent: AgentAuth, response: TokenResponse, key: _dpop.DPoPKey, time: Callable[[], float]
    ) -> None:
        super().__init__(agent, response, key, time)
        self._gateway: Gateway | None = None

    def gateway(self) -> Gateway:
        """The gateways (tools, models, cloud credentials) with this task as the credential."""
        if self._gateway is None:
            self._gateway = Gateway(cast("AuthFI", self.project), self)
        return self._gateway

    @property
    def cloud(self) -> CloudAccess:
        """Cloud credentials held to this task: ``task.cloud.credentials(account, role)``."""
        return self.gateway().cloud

    def token(self, *, force: bool = False) -> str:
        """The task's root token. A reveal point: keep it brief."""
        return self._live()

    def headers(self, method: str, url: str) -> dict[str, str]:
        """``Authorization: DPoP`` and a fresh proof; refuses an origin that is not AuthFI's."""
        return self._headers(method, url)

    def retry_headers(
        self, method: str, url: str, status: int, www_authenticate: str | None, nonce: str | None
    ) -> dict[str, str] | None:
        """Headers for the one retry a server nonce asks for."""
        if _dpop.learn_nonce(url, status, www_authenticate, nonce):
            return self._headers(method, url)
        return None


class AsyncTask(_TaskState, AsyncTokenSource):
    """A task, in asyncio code. ``with`` and ``async with`` both enter it."""

    def __init__(
        self, agent: AsyncAgentAuth, response: TokenResponse, key: _dpop.DPoPKey, time: Callable[[], float]
    ) -> None:
        super().__init__(agent, response, key, time)
        self._gateway: AsyncGateway | None = None

    def gateway(self) -> AsyncGateway:
        """See :meth:`Task.gateway`."""
        if self._gateway is None:
            self._gateway = AsyncGateway(cast("AsyncAuthFI", self.project), self)
        return self._gateway

    @property
    def cloud(self) -> AsyncCloudAccess:
        """See :attr:`Task.cloud`."""
        return self.gateway().cloud

    async def token(self, *, force: bool = False) -> str:
        """See :meth:`Task.token`."""
        return self._live()

    async def headers(self, method: str, url: str) -> dict[str, str]:
        """See :meth:`Task.headers`."""
        return self._headers(method, url)

    async def retry_headers(
        self, method: str, url: str, status: int, www_authenticate: str | None, nonce: str | None
    ) -> dict[str, str] | None:
        """See :meth:`Task.retry_headers`."""
        if _dpop.learn_nonce(url, status, www_authenticate, nonce):
            return self._headers(method, url)
        return None

    async def __aenter__(self) -> Self:
        return self._enter()

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._exit()
