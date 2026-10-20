"""When a request is sent again, and after how long. No I/O.

At most ``retries`` more attempts, with exponential backoff and full jitter (200 ms base, 2 s
cap). ``Retry-After`` is honoured, within the logical call's deadline. A refusal (a 4xx other than
429) is never retried.

====================  ===========================================================================
Class                 Retried on
====================  ===========================================================================
``SAFE``              a connection failure of any kind; 429, 502, 503, 504. A request carrying an
                      ``Idempotency-Key`` is in this class whatever it was built as
``NEVER_AFTER_SEND``  only a failure that provably never sent the request (refused connection,
                      connect timeout, no pooled connection); 429, and 503 with ``Retry-After``
====================  ===========================================================================
"""

from __future__ import annotations

import email.utils
import enum
import random
from collections.abc import Callable
from dataclasses import dataclass

from authfi._protocol.http import HttpRequest, HttpResponse, Retry

__all__ = ["Failure", "RetryPolicy", "parse_retry_after"]

_RETRY_STATUS_SAFE = frozenset({429, 502, 503, 504})


class Failure(enum.Enum):
    """How an attempt failed before an answer came back."""

    NOT_SENT = "not_sent"  # the request provably never left: refused connection, connect timeout
    AFTER_SEND = "after_send"  # it may have reached the server: read timeout, reset, protocol error


def parse_retry_after(value: str | None, now: float) -> float | None:
    """Seconds to wait from a ``Retry-After`` value (delta-seconds or an HTTP date), else ``None``."""
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    return max(0.0, when.timestamp() - now)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """The retry rules, shared by the sync and async drivers."""

    retries: int = 2
    base: float = 0.2
    cap: float = 2.0
    rng: Callable[[float, float], float] = random.uniform

    def backoff(self, attempt: int) -> float:
        """The full-jitter delay before attempt ``attempt + 1`` (``attempt`` counts from 0)."""
        return self.rng(0.0, min(self.cap, self.base * (2**attempt)))

    @staticmethod
    def retry_class(request: HttpRequest) -> Retry:
        """The request's class; one that carries an ``Idempotency-Key`` is safe to repeat."""
        if any(name.lower() == "idempotency-key" for name in request.headers):
            return Retry.SAFE
        return request.retry

    def after_failure(self, request: HttpRequest, failure: Failure, attempt: int) -> float | None:
        """The delay before trying again after ``failure``, or ``None`` to give up."""
        if attempt >= self.retries:
            return None
        if failure is Failure.NOT_SENT or self.retry_class(request) is Retry.SAFE:
            return self.backoff(attempt)
        return None

    def after_response(self, request: HttpRequest, response: HttpResponse, attempt: int, now: float) -> float | None:
        """The delay before trying again after ``response``, or ``None`` to accept it."""
        if attempt >= self.retries:
            return None
        status = response.status
        retry_after = parse_retry_after(response.header("Retry-After"), now)
        if self.retry_class(request) is Retry.SAFE:
            if status not in _RETRY_STATUS_SAFE:
                return None
        elif not (status == 429 or (status == 503 and retry_after is not None)):
            return None
        return retry_after if retry_after is not None else self.backoff(attempt)
