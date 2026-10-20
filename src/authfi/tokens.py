"""Credentials: ``TokenSource``, the one interface every credential answers, and the person's ``TokenSet``.

A machine client, an agent, a token an agent holds for a person, and a person's session are all
a :class:`TokenSource` (or, in asyncio code, an :class:`AsyncTokenSource`). Framework adapters,
the gateway helpers and HTTP clients take any of them. Each source knows where its credential may
go (:attr:`TokenSource.send_to`), and refuses to put it anywhere else.

``token()`` and ``headers()`` are the two places a credential leaves its :class:`~authfi.Secret`.
Call them inside the request that needs the value, and keep no reference to what they return.
"""

from __future__ import annotations

import abc
import asyncio
import hashlib
import threading
from collections.abc import AsyncGenerator, Awaitable, Callable, Generator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, ClassVar, Literal, NoReturn, Protocol, runtime_checkable

import httpx

from authfi._cache import AsyncCache, SyncCache
from authfi._clock import Clock
from authfi._protocol.oauth import TokenResponse
from authfi._protocol.urls import origin_of, validate_origin
from authfi.errors import ConfigurationError
from authfi.secret import Secret

__all__ = [
    "AsyncInMemoryTokenStore",
    "AsyncTokenSource",
    "AsyncTokenStore",
    "InMemoryTokenStore",
    "TokenSet",
    "TokenSource",
    "TokenStore",
]


def check_destination(url: str, send_to: Sequence[str]) -> None:
    """Refuse to send a credential to ``url`` unless its origin is one of ``send_to``."""
    origin = origin_of(url)
    if origin not in send_to:
        raise ConfigurationError(
            f"refusing to send this credential to {origin}: it may go only to {', '.join(send_to) or 'nowhere'}. "
            "Name the API's origin in send_to when you create the credential.",
            code="destination_not_allowed",
        )


def normalise_send_to(values: Sequence[str]) -> tuple[str, ...]:
    """``values`` as exact origins, each checked; a bare string is refused."""
    if isinstance(values, str):
        raise ConfigurationError("send_to is a sequence of origins, such as ('https://api.example.com',)")
    return tuple(dict.fromkeys(validate_origin(v) for v in values))


class TokenSource(abc.ABC):
    """A credential that blocking code sends: ``token()``, ``headers()``, and an ``httpx.Auth``.

    Subclass it to plug in a credential of your own. It is a nominal base class, not a protocol:
    an :class:`AsyncTokenSource` is never an instance of it, so code that receives either can
    dispatch on ``isinstance`` (or on :attr:`is_async`) without calling a coroutine as if it were
    a function.
    """

    is_async: ClassVar[Literal[False]] = False

    @property
    @abc.abstractmethod
    def send_to(self) -> tuple[str, ...]:
        """The origins this credential may be sent to."""

    @abc.abstractmethod
    def token(self, *, force: bool = False) -> str:
        """The current token, renewed when due (or now, with ``force``). A reveal point: keep it brief."""

    def headers(self, method: str, url: str) -> dict[str, str]:
        """The headers that carry this credential to ``url``. Refuses an origin not in :attr:`send_to`."""
        check_destination(url, self.send_to)
        return {"Authorization": "Bearer " + self.token()}

    def httpx_auth(self) -> httpx.Auth:
        """An ``httpx.Auth`` for ``httpx.Client``: a current credential on every request, origin-checked."""
        return _SyncSourceAuth(self)


class AsyncTokenSource(abc.ABC):
    """A credential that asyncio code sends. The same as :class:`TokenSource`, with coroutines."""

    is_async: ClassVar[Literal[True]] = True

    @property
    @abc.abstractmethod
    def send_to(self) -> tuple[str, ...]:
        """The origins this credential may be sent to."""

    @abc.abstractmethod
    async def token(self, *, force: bool = False) -> str:
        """See :meth:`TokenSource.token`."""

    async def headers(self, method: str, url: str) -> dict[str, str]:
        """See :meth:`TokenSource.headers`."""
        check_destination(url, self.send_to)
        return {"Authorization": "Bearer " + await self.token()}

    def httpx_auth(self) -> httpx.Auth:
        """An ``httpx.Auth`` for ``httpx.AsyncClient``: a current credential on every request, origin-checked."""
        return _AsyncSourceAuth(self)


class _SyncSourceAuth(httpx.Auth):
    def __init__(self, source: TokenSource) -> None:
        self._source = source

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers.update(self._source.headers(request.method, str(request.url)))
        yield request

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        raise ConfigurationError(
            "this credential is for blocking code; with httpx.AsyncClient use the AsyncAuthFI version of it"
        )
        yield request  # type: ignore[unreachable]  # pragma: no cover - makes this an async generator

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._source!r})"


class _AsyncSourceAuth(httpx.Auth):
    def __init__(self, source: AsyncTokenSource) -> None:
        self._source = source

    def sync_auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        raise ConfigurationError(
            "this credential is for asyncio code; with httpx.Client use the AuthFI (blocking) version of it"
        )
        yield request  # type: ignore[unreachable]  # pragma: no cover - makes this a generator

    async def async_auth_flow(self, request: httpx.Request) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers.update(await self._source.headers(request.method, str(request.url)))
        yield request

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._source!r})"


@dataclass(frozen=True, slots=True, repr=False, eq=False)
class TokenSet:
    """A person's tokens from the token endpoint.

    The tokens are :class:`~authfi.Secret`; the set does not pickle (a checkpoint stores a session
    id, never tokens). Two sets are equal when they hold the same access and refresh tokens,
    compared in constant time.
    """

    access_token: Secret
    expires_at: datetime
    refresh_token: Secret | None = None
    id_token: Secret | None = None
    scope: frozenset[str] = field(default_factory=frozenset[str])
    token_type: str = "Bearer"  # noqa: S105 - a token type name, not a secret

    @classmethod
    def from_response(cls, response: TokenResponse, now: float, previous: TokenSet | None = None) -> TokenSet:
        """A set from a token answer received at wall-clock ``now``.

        ``previous`` supplies what a refresh answer may leave out: the refresh token when it was not
        rotated, the ID token, and the scope.
        """
        lifetime = response.expires_in if response.expires_in is not None else 0
        return cls(
            access_token=response.access_token,
            expires_at=datetime.fromtimestamp(now + lifetime, UTC),
            refresh_token=response.refresh_token or (previous.refresh_token if previous else None),
            id_token=response.id_token or (previous.id_token if previous else None),
            scope=response.scope if response.scope is not None else (previous.scope if previous else frozenset()),
            token_type=response.token_type,
        )

    def expires_within(self, seconds: float, now: float) -> bool:
        """``True`` when the access token expires within ``seconds`` of wall-clock ``now``."""
        return self.expires_at.timestamp() - seconds <= now

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TokenSet):
            return NotImplemented
        return self.access_token == other.access_token and self.refresh_token == other.refresh_token

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        refresh = "yes" if self.refresh_token else "no"
        return f"TokenSet(expires_at={self.expires_at.isoformat()}, scope={sorted(self.scope)!r}, refresh={refresh})"

    def __copy__(self) -> TokenSet:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> TokenSet:
        return self

    def __reduce__(self) -> NoReturn:
        raise TypeError("A TokenSet cannot be pickled: store a session id, and keep the tokens in a TokenStore.")


@runtime_checkable
class TokenStore(Protocol):
    """Where a person's :class:`TokenSet` lives between requests, shared by every worker.

    ``compare_and_set`` makes a refresh atomic across workers: only the worker that replaces the
    set it read stores its result; the others load the winner's. A refresh token is single-use,
    so two workers refreshing the same one would sign the person out.
    """

    def load(self, key: str) -> TokenSet | None:
        """The set stored under ``key``, or ``None``."""
        ...

    def compare_and_set(self, key: str, old: TokenSet | None, new: TokenSet) -> bool:
        """Store ``new`` if the current value is ``old`` (``None``: absent). ``True`` when stored."""
        ...


@runtime_checkable
class AsyncTokenStore(Protocol):
    """The asyncio form of :class:`TokenStore`."""

    async def load(self, key: str) -> TokenSet | None:
        """See :meth:`TokenStore.load`."""
        ...

    async def compare_and_set(self, key: str, old: TokenSet | None, new: TokenSet) -> bool:
        """See :meth:`TokenStore.compare_and_set`."""
        ...


class InMemoryTokenStore:
    """A :class:`TokenStore` in this process's memory: for one process only (tests, a single worker)."""

    def __init__(self) -> None:
        self._sets: dict[str, TokenSet] = {}
        self._lock = threading.Lock()

    def load(self, key: str) -> TokenSet | None:
        """See :meth:`TokenStore.load`."""
        with self._lock:
            return self._sets.get(key)

    def compare_and_set(self, key: str, old: TokenSet | None, new: TokenSet) -> bool:
        """See :meth:`TokenStore.compare_and_set`."""
        with self._lock:
            current = self._sets.get(key)
            if current is not old and current != old:
                return False
            self._sets[key] = new
            return True

    def __repr__(self) -> str:
        return f"InMemoryTokenStore({len(self._sets)} sets)"


class AsyncInMemoryTokenStore:
    """An :class:`AsyncTokenStore` in this process's memory: for one process only."""

    def __init__(self) -> None:
        self._sets: dict[str, TokenSet] = {}
        self._lock = asyncio.Lock()

    async def load(self, key: str) -> TokenSet | None:
        """See :meth:`TokenStore.load`."""
        async with self._lock:
            return self._sets.get(key)

    async def compare_and_set(self, key: str, old: TokenSet | None, new: TokenSet) -> bool:
        """See :meth:`TokenStore.compare_and_set`."""
        async with self._lock:
            current = self._sets.get(key)
            if current is not old and current != old:
                return False
            self._sets[key] = new
            return True

    def __repr__(self) -> str:
        return f"AsyncInMemoryTokenStore({len(self._sets)} sets)"


# How long a finished refresh is remembered, so a caller still holding the old refresh token gets
# the new set instead of spending the old token a second time (which would revoke the family).
_REMEMBER_REFRESH = 120.0


def _refresh_key(refresh_token: Secret) -> str:
    return hashlib.sha256(refresh_token.reveal().encode("utf-8")).hexdigest()


class RefreshGate:
    """One refresh per refresh token at a time across threads, and its result shared."""

    def __init__(self, clock: Clock) -> None:
        self._cache: SyncCache[TokenSet] = SyncCache(clock, max_entries=1024)

    def run(self, refresh_token: Secret, refresh: Callable[[], TokenSet]) -> TokenSet:
        """``refresh()`` once for ``refresh_token``; concurrent and late callers get its result."""
        return self._cache.get(_refresh_key(refresh_token), lambda: (refresh(), _REMEMBER_REFRESH))

    def after_fork(self) -> None:
        """See :meth:`authfi._cache.SyncCache.after_fork`."""
        self._cache.after_fork()


class AsyncRefreshGate:
    """One refresh per refresh token at a time across tasks, and its result shared."""

    def __init__(self, clock: Clock) -> None:
        self._cache: AsyncCache[TokenSet] = AsyncCache(clock, max_entries=1024)

    async def run(self, refresh_token: Secret, refresh: Callable[[], Awaitable[TokenSet]]) -> TokenSet:
        """See :meth:`RefreshGate.run`; ``refresh`` is a coroutine function."""

        async def fetch() -> tuple[TokenSet, float]:
            return await refresh(), _REMEMBER_REFRESH

        return await self._cache.get(_refresh_key(refresh_token), fetch)

    def after_fork(self) -> None:
        """See :meth:`authfi._cache.AsyncCache.after_fork`."""
        self._cache.after_fork()
