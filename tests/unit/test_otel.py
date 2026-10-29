from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import httpx
import pytest
from opentelemetry import trace
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import authfi
from authfi import AsyncAuthFI, AuthFI, ConfigurationError, Secret
from authfi import otel as authfi_otel
from tests.support import FakeClock, Platform, json_response


class Collector:
    """A local OTLP/HTTP endpoint: records what each export carried, answers with ``status``."""

    def __init__(self) -> None:
        self.exports: list[tuple[dict[str, str], ExportTraceServiceRequest]] = []
        self.status = 200
        collector = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                request = ExportTraceServiceRequest()
                if self.headers.get("Content-Type") == "application/x-protobuf":
                    request.ParseFromString(body)
                collector.exports.append(({k.lower(): v for k, v in self.headers.items()}, request))
                self.send_response(collector.status)
                self.send_header("Content-Type", "application/x-protobuf")
                self.end_headers()

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def spans(self) -> list[Any]:
        return [
            span
            for _, request in self.exports
            for resource in request.resource_spans
            for scope in resource.scope_spans
            for span in scope.spans
        ]


@pytest.fixture
def collector() -> Iterator[Collector]:
    c = Collector()
    yield c
    c.close()


@pytest.fixture
def platform(collector: Collector) -> Platform:
    p = Platform(issuer=collector.origin)
    tokens = iter(f"token-{n}" for n in range(1, 50))
    p.route(
        "POST",
        "/oauth/token",
        lambda _r: json_response({"access_token": next(tokens), "token_type": "Bearer", "expires_in": 900}),
    )
    return p


def provider_with(source: Any, **options: Any) -> tuple[TracerProvider, InMemorySpanExporter]:
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(authfi_otel.StepAttributes(source))
    provider.add_span_processor(SimpleSpanProcessor(memory))
    return provider, memory


def one_span(provider: TracerProvider, **attributes: Any) -> Any:
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("invoke_agent research", attributes=attributes) as span:
        return span


# --- where it sends, and as whom ----------------------------------------------------------------------


def test_the_endpoint_is_the_issuers_origin_and_the_otlp_traces_path(platform: Platform, clock: FakeClock) -> None:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        assert authfi_otel.endpoint_for(m2m) == platform.issuer + "/otlp/v1/traces"
        assert authfi_otel.exporter(m2m).endpoint == platform.issuer + "/otlp/v1/traces"


def test_a_credential_that_does_not_name_its_project_has_no_endpoint() -> None:
    class Bare(authfi.TokenSource):
        @property
        def send_to(self) -> tuple[str, ...]:
            return ()

        def token(self, *, force: bool = False) -> str:
            return "x"

    with pytest.raises(ConfigurationError, match="project"):
        authfi_otel.endpoint_for(Bare())


def test_each_export_carries_the_current_token_and_the_agents_spans(
    platform: Platform, collector: Collector, clock: FakeClock
) -> None:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app_batch", Secret("s"), resource=platform.issuer + "/mcp")
        m2m.start_run("run-a")
        provider = TracerProvider()
        provider.add_span_processor(authfi_otel.StepAttributes(m2m))
        exporter = authfi_otel.exporter(m2m)
        memory = InMemorySpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(memory))
        one_span(provider)
        assert exporter.export(memory.get_finished_spans()) is SpanExportResult.SUCCESS
        clock.advance(2000)  # the token expires and is renewed
        assert exporter.export(memory.get_finished_spans()) is SpanExportResult.SUCCESS
        exporter.shutdown()
    first, second = (headers for headers, _ in collector.exports)
    assert first["authorization"] == "Bearer token-1"
    assert second["authorization"] == "Bearer token-2"  # the token is read per export, not frozen in at the start
    assert first["content-type"] == "application/x-protobuf"
    span = collector.spans()[0]
    attributes = {kv.key: kv.value.string_value for kv in span.attributes}
    assert attributes["authfi.client_id"] == "app_batch"
    assert attributes["authfi.run_id"] == "run-a"
    assert span.name == "invoke_agent research"


def test_the_agents_id_tags_its_spans(platform: Platform, collector: Collector) -> None:
    key = authfi.AgentKey.generate(kid="k1")
    with AuthFI(platform.issuer, http_client=platform.client()) as af:
        agent = af.agent("agt_one", key)
        agent.start_run("run-b")
        provider, memory = provider_with(agent)
        one_span(provider)
        (finished,) = memory.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["authfi.agent_id"] == "agt_one"
    assert finished.attributes["authfi.run_id"] == "run-b"
    assert "authfi.client_id" not in finished.attributes


def test_a_span_that_names_its_own_agent_keeps_it_out_of_our_hands(platform: Platform, clock: FakeClock) -> None:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        provider, memory = provider_with(m2m)
        one_span(provider, **{"authfi.run_id": "framework-run", "authfi.client_id": "framework"})
        (finished,) = memory.get_finished_spans()
    assert finished.attributes is not None
    assert finished.attributes["authfi.run_id"] == "framework-run"  # a framework may set its own run


# --- content ------------------------------------------------------------------------------------------

CONTENT = {
    "gen_ai.input.messages": "[secret question]",
    "gen_ai.output.messages": "[secret answer]",
    "gen_ai.system_instructions": "be secret",
    "gen_ai.prompt.0.content": "old spelling",
    "gen_ai.tool.call.arguments": '{"q": "secret"}',
    "gen_ai.tool.call.result": "secret result",
    "llm.input_messages.0.content": "vendor spelling",
}


def export_with_content(platform: Platform, clock: FakeClock, *, capture_content: bool) -> Any:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        provider, memory = provider_with(m2m)
        tracer = provider.get_tracer("test")
        with tracer.start_as_current_span("chat", attributes={**CONTENT, "gen_ai.request.model": "claude"}) as span:
            span.add_event("gen_ai.client.inference.operation.details", {"gen_ai.input.messages": "[secret]"})
            span.add_event("retry", {"attempt": 2})
        exporter = authfi_otel.exporter(m2m, capture_content=capture_content)
        assert exporter.export(memory.get_finished_spans()) is SpanExportResult.SUCCESS
        exporter.shutdown()
    return memory.get_finished_spans()[0]


def test_content_is_not_sent_unless_the_tenant_allows_it(
    platform: Platform, collector: Collector, clock: FakeClock
) -> None:
    original = export_with_content(platform, clock, capture_content=False)
    (span,) = collector.spans()
    keys = {kv.key for kv in span.attributes}
    assert keys.isdisjoint(CONTENT)
    assert "gen_ai.request.model" in keys  # the step itself is kept
    assert [e.name for e in span.events] == ["retry"]
    # what the application's own exporters see is not changed
    assert original.attributes is not None
    assert "gen_ai.input.messages" in original.attributes


def test_content_goes_when_the_tenant_allows_it(platform: Platform, collector: Collector, clock: FakeClock) -> None:
    export_with_content(platform, clock, capture_content=True)
    (span,) = collector.spans()
    assert {kv.key for kv in span.attributes} >= set(CONTENT)
    assert len(span.events) == 2


def test_a_span_without_content_is_exported_as_it_is() -> None:
    provider = TracerProvider()
    memory = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(memory))
    one_span(provider, **{"gen_ai.request.model": "claude"})
    (span,) = memory.get_finished_spans()
    assert authfi_otel._without_content(span) is span


# --- failures never reach the application ---------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403, 503])
def test_a_refusal_from_authfi_is_a_failed_export_not_an_exception(
    platform: Platform, collector: Collector, clock: FakeClock, status: int
) -> None:
    collector.status = status
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        provider, memory = provider_with(m2m)
        one_span(provider)
        exporter = authfi_otel.exporter(m2m)
        assert exporter.export(memory.get_finished_spans()) is SpanExportResult.FAILURE
        exporter.shutdown()
    assert collector.exports  # it was sent; the answer was the refusal


def test_the_token_is_never_sent_to_a_host_the_credential_may_not_reach(
    platform: Platform, collector: Collector, clock: FakeClock
) -> None:
    elsewhere = Collector()
    try:
        with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
            m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
            provider, memory = provider_with(m2m)
            one_span(provider)
            exporter = authfi_otel.exporter(m2m, endpoint=elsewhere.origin + "/otlp/v1/traces")
            assert exporter.export(memory.get_finished_spans()) is SpanExportResult.FAILURE
            exporter.shutdown()
        assert not elsewhere.exports  # it is listening, and was sent nothing
    finally:
        elsewhere.close()
    assert not collector.exports


def test_an_export_without_a_token_is_a_failed_export(
    platform: Platform, collector: Collector, clock: FakeClock
) -> None:
    platform.route("POST", "/oauth/token", json_response({"error": "invalid_client"}, status=401))
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        provider, memory = provider_with(m2m)
        one_span(provider)
        exporter = authfi_otel.exporter(m2m)
        assert exporter.export(memory.get_finished_spans()) is SpanExportResult.FAILURE
        exporter.shutdown()
    assert not collector.exports  # no token, nothing is sent


# --- asyncio credentials ----------------------------------------------------------------------------------


async def test_an_asyncio_credential_renews_its_token_on_its_own_loop(
    platform: Platform, collector: Collector, clock: FakeClock
) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        await m2m.token()  # the credential's loop is now known
        provider, memory = provider_with(m2m)
        one_span(provider)
        exporter = authfi_otel.exporter(m2m)
        # the batch processor exports from its own thread, as this does
        result = await asyncio.to_thread(exporter.export, memory.get_finished_spans())
        assert result is SpanExportResult.SUCCESS
        exporter.shutdown()
    assert collector.exports[0][0]["authorization"] == "Bearer token-1"


async def test_an_asyncio_credential_that_has_not_run_cannot_be_asked_for_a_token(
    platform: Platform, collector: Collector, clock: FakeClock
) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        provider, memory = provider_with(m2m)
        one_span(provider)
        exporter = authfi_otel.exporter(m2m)
        assert await asyncio.to_thread(exporter.export, memory.get_finished_spans()) is SpanExportResult.FAILURE
        exporter.shutdown()
    assert not collector.exports


async def test_an_asyncio_credential_whose_loop_has_closed_cannot_export(
    platform: Platform, collector: Collector, clock: FakeClock
) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        provider, memory = provider_with(m2m)
        one_span(provider)
        closed = asyncio.new_event_loop()
        closed.close()
        exporter = authfi_otel.exporter(m2m, loop=closed)
        assert await asyncio.to_thread(exporter.export, memory.get_finished_spans()) is SpanExportResult.FAILURE
        exporter.shutdown()
    assert not collector.exports


# --- registering --------------------------------------------------------------------------------------------


def test_register_adds_the_exporter_to_the_given_provider_once(platform: Platform, clock: FakeClock) -> None:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        provider = TracerProvider()
        assert authfi_otel.register(m2m, tracer_provider=provider) is provider
        processors = len(provider._active_span_processor._span_processors)
        assert processors == 2
        assert authfi_otel.register(m2m, tracer_provider=provider) is provider
        assert len(provider._active_span_processor._span_processors) == processors
        provider.shutdown()


def test_register_sends_a_spans_to_authfi(platform: Platform, collector: Collector, clock: FakeClock) -> None:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        m2m.start_run("run-c")
        provider = authfi_otel.register(m2m, tracer_provider=TracerProvider())
        one_span(provider, **{"gen_ai.input.messages": "secret"})
        provider.force_flush()
        provider.shutdown()
    (span,) = collector.spans()
    attributes = {kv.key: kv.value.string_value for kv in span.attributes}
    assert attributes["authfi.run_id"] == "run-c"
    assert "gen_ai.input.messages" not in attributes


def test_register_uses_the_global_sdk_provider_or_makes_one(
    platform: Platform, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder: list[Any] = [trace.ProxyTracerProvider()]

    def set_provider(provider: Any) -> None:
        holder[0] = provider

    monkeypatch.setattr(trace, "get_tracer_provider", lambda: holder[0])
    monkeypatch.setattr(trace, "set_tracer_provider", set_provider)
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        made = authfi_otel.register(m2m, resource_attributes={"service.name": "research"})
        assert isinstance(made, TracerProvider)
        assert holder[0] is made
        assert made.resource.attributes["service.name"] == "research"
        again = authfi_otel.register(m2m)
        assert again is made
        made.shutdown()


def test_register_refuses_a_provider_it_cannot_add_a_processor_to(
    platform: Platform, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Foreign:
        pass

    monkeypatch.setattr(trace, "get_tracer_provider", lambda: Foreign())
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        with pytest.raises(ConfigurationError, match="tracer_provider"):
            authfi_otel.register(m2m)


# --- the trace the gateway records is the trace of the step -----------------------------------------------


def test_a_gateway_call_inside_a_step_carries_that_steps_trace(platform: Platform, clock: FakeClock) -> None:
    seen: list[httpx.Request] = []

    def server(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp":
            seen.append(request)
            return json_response({"jsonrpc": "2.0", "id": 1, "result": {}})
        return platform(request)

    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        m2m = af.client_credentials("app", Secret("s"), resource=platform.issuer + "/mcp")
        gateway = m2m.gateway()
        client = gateway.http_client()
        client._transport = httpx.MockTransport(server)
        provider = TracerProvider()
        with provider.get_tracer("test").start_as_current_span("execute_tool search") as span:
            client.post(gateway.mcp_url, json={})
        trace_id = format(span.get_span_context().trace_id, "032x")
    assert seen[0].headers["traceparent"].split("-")[1] == trace_id
