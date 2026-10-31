"""The same spans to LangSmith and to AuthFI: LangSmith's OpenTelemetry mapping, and no secret in a span.

LangSmith reads an OpenTelemetry span through these attributes: ``langsmith.span.kind`` (the run type:
``llm``, ``chain``, ``tool``, ``retriever``, ``embedding``, ``prompt``, ``parser``), ``langsmith.trace.name``
(the run's name), ``langsmith.metadata.<key>``, and the span's OpenTelemetry parent (the run tree). It reads
the ``gen_ai.*`` attributes too. :class:`LangSmithAttributes` sets the first three where the framework's own
instrumentation left them out, so one span carries what both products read; :func:`authfi.otel.register` adds
it. A customer who also exports to LangSmith adds LangSmith's exporter to the same provider and changes no
code.

:func:`redact_span` runs on every export, whether or not content capture is on: a value shaped like a token or
a key, and any attribute named like a credential, leaves as ``[redacted]``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from authfi.errors import ConfigurationError

try:
    from opentelemetry.sdk.trace import Event, ReadableSpan, Span, SpanProcessor
except ImportError as error:  # pragma: no cover - exercised by installing the core package alone
    raise ConfigurationError(
        "authfi.tracing needs the OpenTelemetry SDK: "
        f"pip install 'authfi[otel] @ git+https://github.com/authfi/authfi-python-sdk@main' ({error})"
    ) from error

if TYPE_CHECKING:
    from opentelemetry.context import Context

    from authfi.tokens import AsyncTokenSource, TokenSource

__all__ = ["REDACTED", "LangSmithAttributes", "redact_span", "run_type_of"]

REDACTED = "[redacted]"

_SECRET_VALUE = re.compile(
    r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"  # a JWT
    r"|\b(?:Bearer|DPoP|Basic)\s+[A-Za-z0-9._~+/=-]{8,}"  # an Authorization value
    r"|\bsk-[A-Za-z0-9_-]{16,}"  # a provider key
    r"|\b(?:ghp|gho|ghs|github_pat)_[A-Za-z0-9_]{16,}"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"  # an AWS key id
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    r"|\bxox[abprs]-[A-Za-z0-9-]{10,}",
)
_SECRET_NAME = re.compile(
    r"(?:^|[._-])(?:authorization|api[_-]?key|password|passwd|secret|private[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|client[_-]?secret|credentials?|cookie|session[_-]?token)s?(?:$|[._-])",
    re.IGNORECASE,
)

#: ``gen_ai.operation.name`` values and the LangSmith run type each one is.
_RUN_TYPES = {
    "chat": "llm",
    "text_completion": "llm",
    "completion": "llm",
    "generate_content": "llm",
    "embeddings": "embedding",
    "embedding": "embedding",
    "execute_tool": "tool",
    "retrieval": "retriever",
    "invoke_agent": "chain",
    "create_agent": "chain",
}


def run_type_of(attributes: Mapping[str, Any]) -> str:
    """The LangSmith run type of a span with these attributes (``chain`` when nothing says otherwise)."""
    named = attributes.get("langsmith.span.kind")
    if isinstance(named, str) and named:
        return named
    operation = attributes.get("gen_ai.operation.name")
    if isinstance(operation, str) and operation in _RUN_TYPES:
        return _RUN_TYPES[operation]
    if "gen_ai.tool.name" in attributes:
        return "tool"
    return "chain"


class LangSmithAttributes(SpanProcessor):
    """Sets ``langsmith.span.kind``, ``langsmith.trace.name`` (on a root span) and the AuthFI run as metadata."""

    def __init__(self, source: TokenSource | AsyncTokenSource | None = None) -> None:
        self._source = source

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Set what the span does not already carry."""
        current = span.attributes or {}
        if "langsmith.span.kind" not in current:
            span.set_attribute("langsmith.span.kind", run_type_of(current))
        if span.parent is None and "langsmith.trace.name" not in current:
            span.set_attribute("langsmith.trace.name", span.name)
        source = self._source
        run_id = getattr(source, "run_id", None)
        agent_id = getattr(source, "agent_id", None)
        if isinstance(run_id, str) and "langsmith.metadata.authfi_run_id" not in current:
            span.set_attribute("langsmith.metadata.authfi_run_id", run_id)
        if isinstance(agent_id, str) and "langsmith.metadata.authfi_agent_id" not in current:
            span.set_attribute("langsmith.metadata.authfi_agent_id", agent_id)

    def on_end(self, span: ReadableSpan) -> None:
        """Nothing to do at the end."""

    def shutdown(self) -> None:
        """Nothing to release."""

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Nothing is buffered."""
        return True


def _clean_value(value: Any) -> Any:
    if isinstance(value, str):
        return _SECRET_VALUE.sub(REDACTED, value)
    if isinstance(value, tuple):
        return tuple(_clean_value(v) for v in value)  # pyright: ignore[reportUnknownVariableType]
    return value


def _clean(attributes: Mapping[str, Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in (attributes or {}).items():
        out[key] = REDACTED if _SECRET_NAME.search(key) else _clean_value(value)
    return out


def redact_span(span: ReadableSpan) -> ReadableSpan:
    """``span`` with every secret-shaped value and credential-named attribute replaced; ``span`` itself if clean."""
    attributes = dict(span.attributes or {})
    cleaned = _clean(attributes)
    events = tuple(
        Event(e.name, _clean(e.attributes), e.timestamp) if _clean(e.attributes) != dict(e.attributes or {}) else e
        for e in span.events
    )
    name = _SECRET_VALUE.sub(REDACTED, span.name)
    status = span.status
    if status.description and _SECRET_VALUE.search(status.description):
        status = type(status)(status.status_code, _SECRET_VALUE.sub(REDACTED, status.description))
    if cleaned == attributes and events == tuple(span.events) and name == span.name and status is span.status:
        return span
    return ReadableSpan(
        name=name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=cleaned,
        events=events,
        links=span.links,
        kind=span.kind,
        status=status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )
