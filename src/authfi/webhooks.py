"""Verify the webhooks AuthFI sends to your endpoint.

::

    from authfi import Secret
    from authfi.webhooks import verify

    event = verify(request_body_bytes, request.headers, Secret(os.environ["AUTHFI_WEBHOOK_SECRET"]))
    if event.type == "user.created":
        ...

Each delivery is a ``POST`` of a JSON body with four headers:

=========================  =================================================================================
Header                     Meaning
=========================  =================================================================================
``X-AuthFI-Signature-256`` ``sha256=`` and the hex HMAC-SHA256, keyed by the subscription's secret, of the
                           timestamp, a ``.`` and the exact body bytes
``X-AuthFI-Timestamp``     Unix seconds when the delivery was signed
``X-AuthFI-Delivery``      the delivery's id (the same on a redelivery)
``X-AuthFI-Event``         the event name (not signed: :func:`verify` reads the name from the signed body)
=========================  =================================================================================

Pass :func:`verify` the body exactly as received (bytes, before any JSON parsing): re-serialised JSON signs
differently. The signature binds the timestamp, so a captured delivery cannot be replayed outside the
tolerance window; inside it, a replay is the same ``delivery_id``, which your endpoint should record.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final, Literal, cast

from authfi.errors import ConfigurationError, WebhookVerificationError
from authfi.secret import Secret

__all__ = [
    "DEFAULT_TOLERANCE",
    "EVENT_TYPES",
    "SIGNATURE_HEADER",
    "TIMESTAMP_HEADER",
    "WebhookEvent",
    "WebhookEventType",
    "sign",
    "verify",
]

SIGNATURE_HEADER: Final = "X-AuthFI-Signature-256"
TIMESTAMP_HEADER: Final = "X-AuthFI-Timestamp"
DELIVERY_HEADER: Final = "X-AuthFI-Delivery"

#: How far the timestamp may be from now, in seconds, in either direction.
DEFAULT_TOLERANCE: Final = 300

WebhookEventType = Literal[
    "user.created",
    "user.updated",
    "user.deleted",
    "user.blocked",
    "user.mfa_enrolled",
    "group.member_added",
    "group.member_removed",
    "organization.created",
    "role.assigned",
    "session.revoked",
    "login.success",
    "login.failed",
]

#: Every event the platform delivers today. The set grows: keep a default arm for a name not listed.
EVENT_TYPES: Final[tuple[WebhookEventType, ...]] = (
    "user.created",
    "user.updated",
    "user.deleted",
    "user.blocked",
    "user.mfa_enrolled",
    "group.member_added",
    "group.member_removed",
    "organization.created",
    "role.assigned",
    "session.revoked",
    "login.success",
    "login.failed",
)

_PREFIX = "sha256="
_DIGITS = frozenset("0123456789")
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class WebhookEvent:
    """A verified delivery.

    Attributes:
        id: the id of the platform request that caused the event.
        type: the event name, from the signed body (``WebhookEventType | str``: the set grows).
        timestamp: when the platform built the event.
        actor: who did it (``user:<id>``, ``svc:<name>``...), when known.
        subject: whom it happened to, when known.
        resource_type: the kind of thing it happened to, when known.
        resource_id: the id of that thing, when known.
        delivery_id: ``X-AuthFI-Delivery``: unsigned, so use it to deduplicate, never to decide.
        signed_at: the timestamp the delivery was signed with.
        body: the whole decoded body, for a field this release does not name.

    """

    id: str
    type: WebhookEventType | str
    timestamp: datetime | None
    actor: str | None
    subject: str | None
    resource_type: str | None
    resource_id: str | None
    delivery_id: str | None
    signed_at: datetime
    body: Mapping[str, Any]


def _lowered(headers: Mapping[str, str] | Iterable[tuple[str, str]]) -> dict[str, str]:
    """The headers by lower-case name, the first value of each, from a mapping or a list of pairs."""
    pairs: Iterable[tuple[str, str]] = (
        cast(Mapping[str, str], headers).items() if isinstance(headers, Mapping) else headers
    )
    out: dict[str, str] = {}
    for key, value in pairs:
        out.setdefault(key.lower(), value)
    return out


def _bytes(body: bytes | bytearray | memoryview | str) -> bytes:
    if isinstance(body, str):
        return body.encode("utf-8")
    if isinstance(body, bytes | bytearray | memoryview):  # pyright: ignore[reportUnnecessaryIsInstance]
        return bytes(body)
    raise ConfigurationError("the webhook body is the raw request body (bytes), not parsed JSON")


def _secrets(secret: Secret | str | Sequence[Secret | str]) -> list[bytes]:
    values: Sequence[Secret | str] = [secret] if isinstance(secret, Secret | str) else list(secret)
    out: list[bytes] = []
    for value in values:
        text = value.reveal() if isinstance(value, Secret) else value
        if not text:
            raise ConfigurationError("the webhook secret is empty: pass the subscription's signing secret")
        out.append(text.encode("utf-8"))
    if not out:
        raise ConfigurationError("no webhook secret was given")
    return out


def _mac(secret: bytes, timestamp: str, body: bytes) -> str:
    mac = hmac.new(secret, digestmod=hashlib.sha256)
    mac.update(timestamp.encode("ascii"))
    mac.update(b".")
    mac.update(body)
    return mac.hexdigest()


def sign(secret: Secret | str, body: bytes | str, *, timestamp: int | None = None) -> dict[str, str]:
    """The signature and timestamp headers for ``body``: what the platform sends, for testing your endpoint."""
    key = _secrets(secret)[0]
    raw = _bytes(body)
    stamp = str(int(time.time()) if timestamp is None else timestamp)
    return {SIGNATURE_HEADER: _PREFIX + _mac(key, stamp, raw), TIMESTAMP_HEADER: stamp}


def _refuse(code: str, message: str) -> WebhookVerificationError:
    return WebhookVerificationError(message, code=code)


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _when(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def verify(
    body: bytes | bytearray | memoryview | str,
    headers: Mapping[str, str] | Iterable[tuple[str, str]],
    secret: Secret | str | Sequence[Secret | str],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
    now: Callable[[], float] = time.time,
) -> WebhookEvent:
    """Check a delivery's signature and age, and return the event.

    Args:
        body: the request body exactly as received.
        headers: the request headers (any mapping, case-insensitive, or a list of pairs).
        secret: the subscription's signing secret. A list is accepted for a rotation: the delivery verifies
            when any of them signed it.
        tolerance: how many seconds the timestamp may differ from now, in either direction. Must be positive.
        now: the clock, in Unix seconds (for tests).

    Raises:
        WebhookVerificationError: with ``code`` one of ``webhook_signature_missing``,
            ``webhook_signature_malformed``, ``webhook_timestamp_missing``, ``webhook_timestamp_invalid``,
            ``webhook_signature_mismatch``, ``webhook_timestamp_outside_tolerance`` or ``webhook_body_invalid``.
            The signature is checked before the age, so a forged delivery is a mismatch, never "too old".
        ConfigurationError: the secret is empty or the tolerance is not positive: a mistake in your code,
            not in a delivery.

    """
    if not tolerance > 0:
        raise ConfigurationError("tolerance is a positive number of seconds; a zero window would refuse every delivery")
    keys = _secrets(secret)
    raw = _bytes(body)

    given = _lowered(headers)
    header = given.get(SIGNATURE_HEADER.lower())
    if not header:
        raise _refuse("webhook_signature_missing", f"the {SIGNATURE_HEADER} header is missing")
    signature = header.strip()
    if not signature.startswith(_PREFIX):
        raise _refuse("webhook_signature_malformed", f"the {SIGNATURE_HEADER} header does not start with sha256=")
    digest = signature[len(_PREFIX) :].lower()
    if len(digest) != 64 or not set(digest) <= _HEX:
        raise _refuse("webhook_signature_malformed", f"the {SIGNATURE_HEADER} header is not a SHA-256 digest")

    stamp = (given.get(TIMESTAMP_HEADER.lower()) or "").strip()
    if not stamp:
        raise _refuse("webhook_timestamp_missing", f"the {TIMESTAMP_HEADER} header is missing")
    if not stamp.isascii() or len(stamp) > 11 or not set(stamp) <= _DIGITS:
        raise _refuse("webhook_timestamp_invalid", f"the {TIMESTAMP_HEADER} header is not Unix seconds")

    matched = False
    for key in keys:  # every key is tried, so the time taken does not say which one signed
        matched |= hmac.compare_digest(_mac(key, stamp, raw), digest)
    if not matched:
        raise _refuse(
            "webhook_signature_mismatch", "the signature does not match the body: wrong secret or altered body"
        )

    signed_at = int(stamp)
    if abs(now() - signed_at) > tolerance:
        raise _refuse(
            "webhook_timestamp_outside_tolerance",
            f"the delivery was signed more than {tolerance:g} seconds from now",
        )

    try:
        decoded: object = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        decoded = None
    if not isinstance(decoded, dict):
        raise _refuse("webhook_body_invalid", "the body is not a JSON object")
    parsed = cast(dict[str, Any], decoded)
    event = _text(parsed.get("event"))
    if event is None:
        raise _refuse("webhook_body_invalid", "the body names no event")
    data_value = parsed.get("data")
    data: dict[str, Any] = cast(dict[str, Any], data_value) if isinstance(data_value, dict) else {}
    return WebhookEvent(
        id=_text(parsed.get("id")) or "",
        type=event,
        timestamp=_when(parsed.get("timestamp")),
        actor=_text(data.get("actor")),
        subject=_text(data.get("subject")),
        resource_type=_text(data.get("resource_type")),
        resource_id=_text(data.get("resource_id")),
        delivery_id=given.get(DELIVERY_HEADER.lower()),
        signed_at=datetime.fromtimestamp(signed_at, UTC),
        body=parsed,
    )
