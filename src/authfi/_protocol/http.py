"""The request and response descriptions the protocol layer builds and reads, with no I/O.

A flow is a generator: it yields an :class:`HttpRequest` (or a :class:`Sleep`), and the driver
sends back the :class:`HttpResponse`. The sync and async clients run the same flows on
``httpx.Client`` and ``httpx.AsyncClient``, which is what keeps the two surfaces in step.
"""

from __future__ import annotations

import enum
import json
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, TypeVar
from urllib.parse import urlsplit

__all__ = [
    "API_LIMIT",
    "DOCUMENT_LIMIT",
    "Attempt",
    "Effect",
    "Flow",
    "HttpRequest",
    "HttpResponse",
    "Retry",
    "Sleep",
    "call",
    "merge_headers",
    "pause",
]

#: The largest JSON answer read from an OAuth endpoint or an API (1 MiB).
API_LIMIT = 1 << 20
#: The largest discovery document or key set read (256 KiB).
DOCUMENT_LIMIT = 256 << 10

Method: TypeAlias = Literal["GET", "POST", "PUT", "PATCH", "DELETE"]


class Retry(enum.Enum):
    """Which failures a request may be sent again after.

    ``SAFE``: a read, a mint whose credential is made fresh per attempt, or a request carrying an
    ``Idempotency-Key``. Retried on a connection failure, 429, 502, 503 and 504.

    ``NEVER_AFTER_SEND``: the request may have created state (a code redeemed, an approval opened,
    a refresh token rotated). Retried only when it provably never left, and on 429 and on 503 with
    ``Retry-After``, which the platform answers before processing.
    """

    SAFE = "safe"
    NEVER_AFTER_SEND = "never_after_send"


@dataclass(frozen=True, slots=True)
class Attempt:
    """What is made fresh for each attempt: a client assertion, a DPoP proof."""

    form: Mapping[str, str] = field(default_factory=dict[str, str])
    headers: Mapping[str, str] = field(default_factory=dict[str, str])


@dataclass(frozen=True, slots=True, repr=False)
class HttpRequest:
    """One logical HTTP call. Its ``repr`` names the method and URL only, never the body.

    ``per_attempt`` is called before every attempt, so a retried mint carries a fresh assertion
    (and a fresh ``jti``) and never replays one the server may already have seen.
    """

    method: Method
    url: str
    what: str
    headers: Mapping[str, str] = field(default_factory=dict[str, str])
    form: Mapping[str, str] | None = None
    json_body: Any = None
    retry: Retry = Retry.SAFE
    max_bytes: int = API_LIMIT
    per_attempt: Callable[[], Attempt] | None = None

    def __repr__(self) -> str:
        parts = urlsplit(self.url)
        return f"HttpRequest({self.method} {parts.scheme}://{parts.netloc}{parts.path})"


@dataclass(frozen=True, slots=True, repr=False)
class HttpResponse:
    """An answer, read in full (up to the request's size limit)."""

    status: int
    headers: tuple[tuple[str, str], ...]
    body: bytes
    url: str
    request_id: str
    dropped_header: str | None = None

    def header(self, name: str) -> str | None:
        """The first value of header ``name`` (case-insensitive), or ``None``."""
        lower = name.lower()
        for key, value in self.headers:
            if key.lower() == lower:
                return value
        return None

    def header_values(self, name: str) -> list[str]:
        """Every value of header ``name``."""
        lower = name.lower()
        return [value for key, value in self.headers if key.lower() == lower]

    def json(self) -> Any:
        """The body as JSON. Raises ``ValueError`` when it is not JSON."""
        return json.loads(self.body.decode("utf-8"))

    def json_object(self) -> dict[str, Any] | None:
        """The body as a JSON object, or ``None`` when it is anything else."""
        try:
            value = self.json()
        except (ValueError, UnicodeDecodeError):
            return None
        return value if isinstance(value, dict) else None  # pyright: ignore[reportUnknownVariableType]

    @property
    def answer_request_id(self) -> str | None:
        """The request id the platform answered with, else the one sent."""
        return self.header("AuthFI-Request-Id") or self.header("X-Request-Id") or self.request_id

    def __repr__(self) -> str:
        return f"HttpResponse({self.status} {self.url})"


@dataclass(frozen=True, slots=True)
class Sleep:
    """Wait ``seconds`` before the flow goes on (a CIBA poll interval)."""

    seconds: float


T = TypeVar("T")

Effect: TypeAlias = HttpRequest | Sleep
#: A protocol flow that returns ``T``: it yields effects and is sent responses.
Flow: TypeAlias = Generator[Effect, HttpResponse | None, T]


def call(request: HttpRequest) -> Generator[Effect, HttpResponse | None, HttpResponse]:
    """Yield ``request`` and return the response the driver sends back (use with ``yield from``)."""
    response = yield request
    if response is None:  # a driver bug, not a server answer
        raise RuntimeError(f"no response was sent for {request!r}")
    return response


def pause(seconds: float) -> Generator[Effect, HttpResponse | None, None]:
    """Yield a :class:`Sleep` of ``seconds`` (use with ``yield from``)."""
    yield Sleep(seconds)


def merge_headers(*sources: Mapping[str, str]) -> dict[str, str]:
    """Headers from ``sources``, later ones winning, compared case-insensitively."""
    out: dict[str, str] = {}
    index: dict[str, str] = {}
    for source in sources:
        for key, value in source.items():
            previous = index.get(key.lower())
            if previous is not None:
                del out[previous]
            out[key] = value
            index[key.lower()] = key
    return out
