"""Time, in one injectable place.

Deadlines, backoff and cache lifetimes use the monotonic clock, so a wall-clock step cannot move
them. Only token claims (``exp``, ``iat``, ``nbf``) and the times this SDK reports use the wall
clock.
"""

from __future__ import annotations

import asyncio
import time
from typing import Protocol, runtime_checkable

__all__ = ["Clock", "SystemClock"]


@runtime_checkable
class Clock(Protocol):
    """The clock the SDK reads and sleeps on. Tests pass one that advances on demand."""

    def monotonic(self) -> float:
        """Seconds on a clock that never goes back, for deadlines and lifetimes."""
        ...

    def time(self) -> float:
        """Seconds since the epoch, for token claims."""
        ...

    def sleep(self, seconds: float) -> None:
        """Block the calling thread for ``seconds``."""
        ...

    async def asleep(self, seconds: float) -> None:
        """Suspend the calling task for ``seconds``."""
        ...


class SystemClock:
    """The real clock."""

    def monotonic(self) -> float:
        """See :meth:`Clock.monotonic`."""
        return time.monotonic()

    def time(self) -> float:
        """See :meth:`Clock.time`."""
        return time.time()

    def sleep(self, seconds: float) -> None:
        """See :meth:`Clock.sleep`."""
        if seconds > 0:
            time.sleep(seconds)

    async def asleep(self, seconds: float) -> None:
        """See :meth:`Clock.asleep`."""
        await asyncio.sleep(max(0.0, seconds))
