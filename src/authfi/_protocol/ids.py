"""Request ids, idempotency keys and W3C trace context.

- A request id is ``afr-`` and a 26-character ULID: sortable, and inside the alphabet the gateway
  keeps on its call record (``[A-Za-z0-9._:-]{1,128}``; no underscore). One is made per logical
  call and sent unchanged on every retry.
- An idempotency key is a UUIDv7, made by the caller per logical operation.
- ``traceparent`` comes from the active OpenTelemetry context when ``opentelemetry-api`` is
  installed, and is otherwise fresh with the sampled flag off.
"""

from __future__ import annotations

import importlib
import os
import re
import time
import uuid
from collections.abc import Callable, MutableMapping

__all__ = [
    "REQUEST_ID_PATTERN",
    "idempotency_key",
    "inject_traceparent",
    "is_valid_request_id",
    "request_id",
    "trace_id_of",
]

#: The request ids the gateway keeps on its call record.
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-[0-9a-f]{16}-[0-9a-f]{2}$")


def _ulid() -> str:
    millis = time.time_ns() // 1_000_000
    value = (millis << 80) | int.from_bytes(os.urandom(10), "big")
    out: list[str] = []
    for _ in range(26):
        out.append(_CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(out))


def request_id() -> str:
    """A fresh request id: ``afr-`` and a ULID."""
    return "afr-" + _ulid()


def is_valid_request_id(value: str) -> bool:
    """``True`` when the gateway would keep ``value`` as a request id."""
    return REQUEST_ID_PATTERN.fullmatch(value) is not None


def idempotency_key() -> str:
    """A fresh ``Idempotency-Key`` value: a UUIDv7 (RFC 9562), time-ordered."""
    millis = time.time_ns() // 1_000_000
    rand = int.from_bytes(os.urandom(10), "big")
    rand_a = rand >> 68  # 12 bits
    rand_b = rand & ((1 << 62) - 1)  # 62 bits
    value = (millis & ((1 << 48) - 1)) << 80 | 0x7 << 76 | rand_a << 64 | 0b10 << 62 | rand_b
    return str(uuid.UUID(int=value))


def _fresh_traceparent() -> str:
    trace = os.urandom(16).hex()
    span = os.urandom(8).hex()
    return f"00-{trace}-{span}-00"


def inject_traceparent(headers: MutableMapping[str, str]) -> None:
    """Set ``traceparent`` (and ``tracestate``) from OpenTelemetry, or a fresh unsampled value."""
    if "traceparent" in {k.lower() for k in headers}:
        return
    try:
        propagate = importlib.import_module("opentelemetry.propagate")
    except ImportError:
        headers["traceparent"] = _fresh_traceparent()
        return
    inject: Callable[[dict[str, str]], None] = propagate.inject
    carrier: dict[str, str] = {}
    inject(carrier)
    if "traceparent" in carrier:
        headers.update(carrier)
    else:
        headers["traceparent"] = _fresh_traceparent()


def trace_id_of(traceparent: str | None) -> str | None:
    """The trace id in a ``traceparent`` value, or ``None`` when it is not one."""
    if not traceparent:
        return None
    match = _TRACEPARENT.match(traceparent.strip().lower())
    return match.group(1) if match else None
