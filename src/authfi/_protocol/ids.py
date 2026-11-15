"""Request ids, idempotency keys and W3C trace context.

- A request id is ``afr-`` and a 26-character ULID: sortable, and inside the alphabet the gateway
  keeps on its call record (``[A-Za-z0-9._:-]{1,128}``; no underscore). One is made per logical
  call and sent unchanged on every retry.
- An idempotency key is a UUIDv7, made by the caller per logical operation.
- ``traceparent`` names the framework step executing when the call is made. A framework whose tracer is not
  OpenTelemetry context registers a step source (:func:`register_step_source`) for the credential it exports
  steps for; the active OpenTelemetry span is the step otherwise (when ``opentelemetry-api`` is installed), and with
  no step active the value is fresh with the sampled flag off. A header the caller already set is kept.
"""

from __future__ import annotations

import importlib
import logging
import os
import re
import threading
import time
import uuid
import weakref
from collections.abc import Callable, MutableMapping
from contextlib import AbstractContextManager
from functools import partial
from typing import Any

__all__ = [
    "REQUEST_ID_PATTERN",
    "Step",
    "StepSource",
    "idempotency_key",
    "inject_traceparent",
    "is_valid_request_id",
    "link_step_owner",
    "register_step_source",
    "request_id",
    "step_traceparent",
    "trace_id_of",
    "valid_traceparent",
]

#: The request ids the gateway keeps on its call record.
REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{1,128}")

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_TRACEPARENT = re.compile(r"^[0-9a-f]{2}-([0-9a-f]{32})-([0-9a-f]{16})-[0-9a-f]{2}$")

#: A framework step as it is exported: its trace id (32 lowercase hex) and its span id (16).
Step = tuple[str, str]
#: Answers with the step executing in the calling context, or ``None`` when none of its framework's is.
StepSource = Callable[[], Step | None]

_LOG = logging.getLogger("authfi")

#: Owner id -> (the owner, weakly where a class allows it, else held; its sources). An id is what is looked up
#: because some credentials are slotted classes that cannot be weakly referenced or hashed by value.
_sources: dict[int, tuple[object, list[StepSource]]] = {}
#: Owner id -> the credential its steps are exported for. A task or a sub-agent calls with a credential of its own
#: while the framework's steps are exported for the agent that holds it, so its calls name the step through it.
_parents: dict[int, tuple[object, object]] = {}
_sources_lock = threading.RLock()  # a weakref callback (_forget) can run while it is held, in the same thread


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


def register_step_source(owner: object, source: StepSource) -> None:
    """Have ``source`` name the step for gateway calls made with ``owner``, the credential its steps are exported for.

    A framework that traces with its own tracer has no OpenTelemetry context active when a call goes out; its
    adapter registers how to read its current step here when it starts exporting steps. Registering the same
    source twice adds nothing, and the registration ends with ``owner``.
    """
    with _sources_lock:
        held = _sources.get(id(owner))
        if held is None:
            key = id(owner)
            try:
                ref: object = weakref.ref(owner, partial(_forget, key, _sources, _parents, _sources_lock))
            except TypeError:  # a slotted class without __weakref__: kept for the life of the process
                ref = owner
            held = _sources[key] = (ref, [])
        if source not in held[1]:
            held[1].append(source)


def link_step_owner(owner: object, parent: object) -> None:
    """Have ``owner`` name the step ``parent``'s sources say is executing, after any source of its own.

    A credential made from another one (a task from its agent) signs the calls its code makes, while the framework
    exports its steps for the credential it was started with. The link ends with ``owner``. Linking a credential to
    itself, or in a loop, adds nothing.
    """
    if owner is parent:
        return
    with _sources_lock:
        key = id(owner)
        if key in _parents:
            _parents[key] = (_parents[key][0], parent)
            return
        try:
            ref: object = weakref.ref(owner, partial(_forget, key, _sources, _parents, _sources_lock))
        except TypeError:  # a slotted class without __weakref__: kept for the life of the process
            ref = owner
        _parents[key] = (ref, parent)


def _forget(
    key: int,
    sources: dict[int, tuple[object, list[StepSource]]],
    parents: dict[int, tuple[object, object]],
    lock: AbstractContextManager[Any],
    _ref: object = None,
) -> None:
    """Drop what was kept for a credential that is gone. The tables are passed in: at exit the module's are cleared."""
    with lock:
        sources.pop(key, None)
        parents.pop(key, None)


def _step_of(owner: object | None) -> Step | None:
    seen: set[int] = set()
    while owner is not None and id(owner) not in seen:
        seen.add(id(owner))
        with _sources_lock:
            held = _sources.get(id(owner))
            sources = list(held[1]) if held is not None else []
            linked = _parents.get(id(owner))
        for source in sources:
            try:
                step = source()
            except Exception:  # a framework that cannot say what is executing must not stop the call it is making
                _LOG.warning("a step source failed; the call carries no step", exc_info=True)
                continue
            if step is not None:
                return step
        owner = linked[1] if linked is not None else None
    return None


def step_traceparent(owner: object | None = None) -> str | None:
    """The ``traceparent`` of the step a source registered for ``owner`` says is executing, or ``None``.

    The step is exported, so the flag says sampled.
    """
    step = _step_of(owner)
    return f"00-{step[0]}-{step[1]}-01" if step is not None else None


def inject_traceparent(headers: MutableMapping[str, str], owner: object | None = None) -> None:
    """Set ``traceparent`` (and ``tracestate``): the step executing, else a fresh unsampled value.

    The step is the one ``owner``'s registered source names, else the active OpenTelemetry span. A header already
    in ``headers`` is kept.
    """
    if "traceparent" in {k.lower() for k in headers}:
        return
    stepped = step_traceparent(owner)
    if stepped is not None:
        headers["traceparent"] = stepped
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


def valid_traceparent(value: object) -> str | None:
    """``value`` as a lowercase ``traceparent`` when it is one, else ``None``: a value read from a message."""
    if not isinstance(value, str):
        return None
    clean = value.strip().lower()
    return clean if _TRACEPARENT.match(clean) else None
