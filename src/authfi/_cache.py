"""Single-flight caches: one fetch at a time per key, kept for its lifetime, failures never kept.

Concurrent callers needing the same value share one in-flight fetch: N threads (or tasks) at a
cold start make one discovery request, and N callers at a token's expiry make one mint. A fetch
that fails is not cached: the next caller tries again.

The sync cache uses a ``threading`` lock and futures; the async one runs a task per fetch and
shields it, so a caller that is cancelled does not cancel the fetch the others wait on. Both reset
their lock and in-flight map in a forked child.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from authfi._clock import Clock

__all__ = ["AsyncCache", "SyncCache"]

V = TypeVar("V")


class _Store(Generic[V]):
    """Values with their expiry and fetch time, least recently used first."""

    def __init__(self, clock: Clock, max_entries: int) -> None:
        self.clock = clock
        self.max_entries = max_entries
        self.values: OrderedDict[str, tuple[V, float, float]] = OrderedDict()

    def fresh(self, key: str) -> V | None:
        entry = self.values.get(key)
        if entry is None or entry[1] <= self.clock.monotonic():
            return None
        self.values.move_to_end(key)
        return entry[0]

    def put(self, key: str, value: V, ttl: float) -> None:
        now = self.clock.monotonic()
        self.values[key] = (value, now + ttl, now)
        self.values.move_to_end(key)
        while len(self.values) > self.max_entries:
            self.values.popitem(last=False)

    def fetched_at(self, key: str) -> float | None:
        entry = self.values.get(key)
        return entry[2] if entry is not None else None

    def drop(self, key: str) -> None:
        self.values.pop(key, None)


class SyncCache(Generic[V]):
    """A thread-safe single-flight cache."""

    def __init__(self, clock: Clock, *, max_entries: int = 256) -> None:
        self._store: _Store[V] = _Store(clock, max_entries)
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future[V]] = {}

    def get(self, key: str, fetch: Callable[[], tuple[V, float]], *, refresh: bool = False) -> V:
        """The value for ``key``: cached, joined to an in-flight fetch, or fetched now.

        ``fetch`` returns the value and its lifetime in seconds. ``refresh`` skips the cached value
        (an in-flight fetch is still shared).
        """
        with self._lock:
            if not refresh:
                hit = self._store.fresh(key)
                if hit is not None:
                    return hit
            pending = self._inflight.get(key)
            leader = pending is None
            future: concurrent.futures.Future[V] = concurrent.futures.Future() if pending is None else pending
            if leader:
                self._inflight[key] = future
        if not leader:
            return future.result()
        try:
            value, ttl = fetch()
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
            future.set_exception(exc)
            raise
        with self._lock:
            self._store.put(key, value, ttl)
            self._inflight.pop(key, None)
        future.set_result(value)
        return value

    def fetched_at(self, key: str) -> float | None:
        """When ``key`` was last fetched (monotonic), or ``None``."""
        with self._lock:
            return self._store.fetched_at(key)

    def drop(self, key: str) -> None:
        """Forget ``key``."""
        with self._lock:
            self._store.drop(key)

    def after_fork(self) -> None:
        """In a forked child: a new lock (never released: it may be held), no in-flight fetches."""
        self._lock = threading.Lock()
        self._inflight = {}


class AsyncCache(Generic[V]):
    """A single-flight cache for one event loop."""

    def __init__(self, clock: Clock, *, max_entries: int = 256) -> None:
        self._store: _Store[V] = _Store(clock, max_entries)
        self._inflight: dict[str, asyncio.Task[V]] = {}

    async def get(self, key: str, fetch: Callable[[], Awaitable[tuple[V, float]]], *, refresh: bool = False) -> V:
        """See :meth:`SyncCache.get`."""
        if not refresh:
            hit = self._store.fresh(key)
            if hit is not None:
                return hit
        task = self._inflight.get(key)
        if task is None:
            task = asyncio.ensure_future(self._fetch(key, fetch))
            task.add_done_callback(_retrieve)
            self._inflight[key] = task
        return await asyncio.shield(task)

    async def _fetch(self, key: str, fetch: Callable[[], Awaitable[tuple[V, float]]]) -> V:
        try:
            value, ttl = await fetch()
            self._store.put(key, value, ttl)
            return value
        finally:
            self._inflight.pop(key, None)

    def fetched_at(self, key: str) -> float | None:
        """When ``key`` was last fetched (monotonic), or ``None``."""
        return self._store.fetched_at(key)

    def drop(self, key: str) -> None:
        """Forget ``key``."""
        self._store.drop(key)

    def after_fork(self) -> None:
        """In a forked child: no in-flight fetches (their tasks belong to the parent's loop)."""
        self._inflight = {}


def _retrieve(task: asyncio.Task[object]) -> None:
    # A fetch whose every waiter was cancelled would otherwise log "exception never retrieved".
    if not task.cancelled():
        task.exception()
