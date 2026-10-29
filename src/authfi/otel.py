"""An agent's steps to AuthFI: an OpenTelemetry exporter for ``<issuer origin>/otlp/v1/traces``.

Install the ``otel`` extra, connect, and make one call for the framework::

    af = authfi.connect_async()
    graph = af.langgraph()
    graph.export_steps()                      # or openai_agents(), adk(), claude_agent(), langchain()

or, for any OpenTelemetry-instrumented code, register the exporter yourself::

    authfi.otel.register(af)

The exporter sends OTLP/HTTP protobuf with the credential's *current* token on every export, so a token
that is renewed is never stale in a header. AuthFI keeps each span as the token's agent's, whatever
the payload says, and drops the content attributes; this module drops them first unless
``capture_content`` is set (the tenant's policy decision). Each span is tagged ``authfi.agent_id`` (or
``authfi.client_id``) and ``authfi.run_id``, and carries LangSmith's run type and trace name
(:mod:`authfi.tracing`), so the same spans go to LangSmith and to AuthFI. A secret never leaves in a span, with
content capture on or off. A gateway call made inside a span carries that span's
``traceparent``, so the gateway's record and the steps share a trace id.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

from authfi._protocol.urls import origin_of
from authfi.errors import ConfigurationError
from authfi.tokens import AsyncTokenSource, TokenSource
from authfi.tracing import LangSmithAttributes, redact_span

try:
    import requests
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult
    from requests.auth import AuthBase
except ImportError as error:  # pragma: no cover - exercised by installing the core package alone
    raise ConfigurationError(
        "authfi.otel needs the OpenTelemetry SDK and OTLP exporter: "
        f"pip install 'authfi[otel] @ git+https://github.com/authfi/authfi-python-sdk@main' ({error})"
    ) from error

if TYPE_CHECKING:
    from opentelemetry.context import Context

__all__ = ["CONTENT_ATTRIBUTES", "AuthFISpanExporter", "StepAttributes", "endpoint_for", "exporter", "register"]

_LOG = logging.getLogger("authfi.otel")

TRACES_PATH = "/otlp/v1/traces"

#: GenAI attributes that carry what was said; AuthFI drops them on receipt, and this module before sending.
CONTENT_ATTRIBUTES = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.prompt",
        "gen_ai.completion",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    }
)
_CONTENT_PREFIXES = ("gen_ai.prompt.", "gen_ai.completion.", "llm.input_messages", "llm.output_messages")

#: How long an export waits for a token held by an asyncio credential.
_TOKEN_WAIT = 10.0


def endpoint_for(source: TokenSource | AsyncTokenSource) -> str:
    """``<issuer origin>/otlp/v1/traces`` for the project behind ``source``."""
    project = getattr(source, "project", None)
    issuer = getattr(project, "issuer", None)
    if not isinstance(issuer, str):
        raise ConfigurationError("this credential does not name its AuthFI project: use connect() or af.agent(...)")
    return origin_of(issuer) + TRACES_PATH


def _is_content(key: str) -> bool:
    return key in CONTENT_ATTRIBUTES or key.startswith(_CONTENT_PREFIXES)


def _without_content(span: ReadableSpan) -> ReadableSpan:
    attributes = dict(span.attributes or {})
    events = tuple(e for e in span.events if not e.name.startswith("gen_ai."))
    clean = {k: v for k, v in attributes.items() if not _is_content(k)}
    if len(clean) == len(attributes) and len(events) == len(span.events):
        return span
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=clean,
        events=events,
        links=span.links,
        kind=span.kind,
        status=span.status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


class _CurrentToken(AuthBase):
    """Puts the credential's current token on each export, and refuses an origin the credential may not reach."""

    def __init__(self, source: TokenSource | AsyncTokenSource, loop: asyncio.AbstractEventLoop | None) -> None:
        self._source = source
        self._loop = loop

    def _loop_for(self) -> asyncio.AbstractEventLoop:
        loop = self._loop
        if loop is None:
            loop = cast("Any", getattr(self._source, "project", None))._transport.bound_loop
        if loop is None or loop.is_closed():
            raise ConfigurationError(
                "the credential's event loop is not running, so its token cannot be renewed for this export"
            )
        return loop

    def _headers(self, url: str) -> dict[str, str]:
        source = self._source
        if isinstance(source, TokenSource):
            return source.headers("POST", url)
        future = asyncio.run_coroutine_threadsafe(source.headers("POST", url), self._loop_for())
        return future.result(_TOKEN_WAIT)

    def retry_headers(self, url: str, response: requests.Response) -> dict[str, str] | None:
        """The headers to send ``url`` once more after ``response``, when it asked for the server's DPoP nonce."""
        challenge, nonce = response.headers.get("www-authenticate"), response.headers.get("dpop-nonce")
        source = self._source
        if isinstance(source, TokenSource):
            return source.retry_headers("POST", url, response.status_code, challenge, nonce)
        future = asyncio.run_coroutine_threadsafe(
            source.retry_headers("POST", url, response.status_code, challenge, nonce), self._loop_for()
        )
        return future.result(_TOKEN_WAIT)

    def __call__(self, r: requests.PreparedRequest) -> requests.PreparedRequest:
        r.headers.update(self._headers(str(r.url)))
        return r

    def __repr__(self) -> str:
        return f"_CurrentToken({self._source!r})"


class _NonceSession(requests.Session):
    """A session that sends an export once more when AuthFI answered 401 asking for its DPoP nonce."""

    def __init__(self, auth: _CurrentToken) -> None:
        super().__init__()
        self.auth = auth
        self._current = auth

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        response = super().send(request, **kwargs)
        if response.status_code != 401:
            return response
        headers = self._current.retry_headers(str(request.url), response)
        if headers is None:
            return response
        response.close()
        again = request.copy()
        again.headers.update(headers)
        return super().send(again, **kwargs)


class AuthFISpanExporter(SpanExporter):
    """OTLP/HTTP protobuf to AuthFI, authenticated as ``source`` on every export, content removed unless allowed."""

    def __init__(
        self,
        source: TokenSource | AsyncTokenSource,
        *,
        capture_content: bool = False,
        loop: asyncio.AbstractEventLoop | None = None,
        timeout: float = 10.0,
        endpoint: str | None = None,
    ) -> None:
        self._capture_content = capture_content
        url = endpoint or endpoint_for(source)
        session = _NonceSession(_CurrentToken(source, loop))
        self._inner = OTLPSpanExporter(endpoint=url, session=session, timeout=timeout)
        self.endpoint = url

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Send ``spans``; a failure (including a token that could not be renewed) is logged and reported."""
        batch = [redact_span(s if self._capture_content else _without_content(s)) for s in spans]
        try:
            return self._inner.export(batch)
        except Exception as error:  # a span exporter must not raise into the application
            _LOG.warning("steps were not exported to %s: %s", self.endpoint, error)
            return SpanExportResult.FAILURE

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Nothing is buffered here."""
        return self._inner.force_flush(timeout_millis)

    def shutdown(self) -> None:
        """Close the HTTP session."""
        cast("Any", self._inner).shutdown()


class StepAttributes(SpanProcessor):
    """Tags every span ``authfi.agent_id`` (or ``authfi.client_id``) and the credential's current ``authfi.run_id``."""

    def __init__(self, source: TokenSource | AsyncTokenSource) -> None:
        self._source = source

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        """Set the identity attributes the span does not already have."""
        source = self._source
        agent_id = getattr(source, "agent_id", None)
        client_id = getattr(source, "client_id", None)
        run_id = getattr(source, "run_id", None)
        current = span.attributes or {}
        if isinstance(agent_id, str) and "authfi.agent_id" not in current:
            span.set_attribute("authfi.agent_id", agent_id)
        elif isinstance(client_id, str) and "authfi.client_id" not in current:
            span.set_attribute("authfi.client_id", client_id)
        if isinstance(run_id, str) and "authfi.run_id" not in current:
            span.set_attribute("authfi.run_id", run_id)

    def on_end(self, span: ReadableSpan) -> None:
        """Nothing to do at the end."""

    def shutdown(self) -> None:
        """Nothing to release."""

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        """Nothing is buffered."""
        return True


def exporter(
    source: TokenSource | AsyncTokenSource,
    *,
    capture_content: bool = False,
    loop: asyncio.AbstractEventLoop | None = None,
    endpoint: str | None = None,
) -> AuthFISpanExporter:
    """A span exporter to AuthFI for ``source`` (an agent, or a machine client), for a ``BatchSpanProcessor``.

    ``loop`` is the event loop an asyncio credential belongs to; it is found from the credential once it
    has been used, so it is rarely needed.
    """
    return AuthFISpanExporter(source, capture_content=capture_content, loop=loop, endpoint=endpoint)


#: The sources already registered on a provider, so a second call (a second integration) adds nothing.
_registered: weakref.WeakKeyDictionary[Any, weakref.WeakSet[Any]] = weakref.WeakKeyDictionary()


def register(
    source: TokenSource | AsyncTokenSource,
    *,
    capture_content: bool = False,
    tracer_provider: TracerProvider | None = None,
    loop: asyncio.AbstractEventLoop | None = None,
    resource_attributes: Mapping[str, str] | None = None,
) -> TracerProvider:
    """Export this process's OpenTelemetry spans to AuthFI as ``source`` and return the provider.

    Uses ``tracer_provider`` if given, else the global SDK provider if there is one, else makes one and
    makes it the global provider. Registering the same credential on a provider twice adds nothing.
    """
    provider = tracer_provider
    if provider is None:
        current = trace.get_tracer_provider()
        if isinstance(current, TracerProvider):
            provider = current
        elif current.__class__.__name__ == "ProxyTracerProvider":
            provider = TracerProvider(resource=Resource.create(dict(resource_attributes or {})))
            trace.set_tracer_provider(provider)
        else:
            raise ConfigurationError(
                f"the tracer provider is a {type(current).__name__}, not the OpenTelemetry SDK's: pass tracer_provider="
            )
    done = _registered.setdefault(provider, weakref.WeakSet())
    if source in done:
        return provider
    provider.add_span_processor(StepAttributes(source))
    provider.add_span_processor(LangSmithAttributes(source))
    provider.add_span_processor(BatchSpanProcessor(exporter(source, capture_content=capture_content, loop=loop)))
    done.add(source)
    return provider
