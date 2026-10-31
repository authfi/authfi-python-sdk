"""LangSmith's OpenTelemetry mapping, decisions as child spans, and no secret in a span."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import Status, StatusCode

import authfi
from authfi import otel as authfi_otel
from authfi._decision_span import DECISION_META, decision_attributes, record_decision
from authfi._protocol.gateway import REFUSAL_META, ToolAllowed, ToolApprovalRequired, read_result
from authfi.tracing import REDACTED, LangSmithAttributes, redact_span, run_type_of
from tests.agent_platform import AgentPlatform


class Source(authfi.TokenSource):
    run_id = "run-abc"
    agent_id = "agt_1"

    @property
    def send_to(self) -> tuple[str, ...]:
        return ()

    def token(self, *, force: bool = False) -> str:
        return "t"


def provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    p = TracerProvider()
    memory = InMemorySpanExporter()
    p.add_span_processor(LangSmithAttributes(Source()))
    p.add_span_processor(SimpleSpanProcessor(memory))
    return p, memory


# --- LangSmith's mapping ---------------------------------------------------------------------------


def test_a_root_span_names_the_trace_and_every_span_has_a_run_type() -> None:
    p, memory = provider()
    tracer = p.get_tracer("t")
    with tracer.start_as_current_span("research"):
        with tracer.start_as_current_span("chat claude", attributes={"gen_ai.operation.name": "chat"}):
            pass
        with tracer.start_as_current_span("embed", attributes={"gen_ai.operation.name": "embeddings"}):
            pass
        with tracer.start_as_current_span("lookup", attributes={"gen_ai.tool.name": "wiki"}):
            pass
    by_name = {s.name: dict(s.attributes or {}) for s in memory.get_finished_spans()}
    assert by_name["research"]["langsmith.span.kind"] == "chain"
    assert by_name["research"]["langsmith.trace.name"] == "research"
    assert by_name["chat claude"]["langsmith.span.kind"] == "llm"
    assert by_name["embed"]["langsmith.span.kind"] == "embedding"
    assert by_name["lookup"]["langsmith.span.kind"] == "tool"
    assert "langsmith.trace.name" not in by_name["lookup"]  # only a root names the trace
    assert by_name["research"]["langsmith.metadata.authfi_run_id"] == "run-abc"
    assert by_name["research"]["langsmith.metadata.authfi_agent_id"] == "agt_1"


def test_what_the_framework_already_set_is_kept() -> None:
    p, memory = provider()
    with p.get_tracer("t").start_as_current_span(
        "x", attributes={"langsmith.span.kind": "retriever", "langsmith.trace.name": "mine"}
    ):
        pass
    attributes = dict(memory.get_finished_spans()[0].attributes or {})
    assert attributes["langsmith.span.kind"] == "retriever"
    assert attributes["langsmith.trace.name"] == "mine"
    assert run_type_of({"langsmith.span.kind": "parser"}) == "parser"
    assert run_type_of({}) == "chain"


def test_register_adds_the_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    platform = AgentPlatform()
    with authfi.AuthFI(platform.issuer, http_client=platform.client()) as af:
        m2m = af.client_credentials("app", authfi.Secret("s"), resource=platform.issuer + "/mcp")
        p = TracerProvider()
        authfi_otel.register(m2m, tracer_provider=p)
        kinds = [type(x).__name__ for x in p._active_span_processor._span_processors]  # pyright: ignore[reportPrivateUsage]
        assert "LangSmithAttributes" in kinds


# --- decisions as child runs --------------------------------------------------------------------------


def test_attributes_of_each_outcome() -> None:
    allowed = decision_attributes(ToolAllowed({"content": [{"text": "SECRET ANSWER"}]}), tool="wiki.read")
    assert allowed["authfi.decision.outcome"] == "allow"
    assert allowed["langsmith.span.kind"] == "tool"
    assert allowed["gen_ai.tool.name"] == "wiki.read"
    assert "SECRET ANSWER" not in str(allowed)
    held = decision_attributes(ToolApprovalRequired("apr_1", "approval_required"))
    assert held["authfi.decision.outcome"] == "needs_approval"
    assert held["authfi.decision.approval_id"] == "apr_1"
    result = {
        "isError": True,
        "_meta": {
            REFUSAL_META: {
                "code": "no_grant",
                "reason": "no grant for send_email",
                "rule": "deny-external",
                "decision_id": "dec_9",
            },
            DECISION_META: {"mandate_id": "man_3", "grant": "grt_2"},
        },
    }
    denied = decision_attributes(read_result(result), decision_id="dec_override")
    assert denied["authfi.decision.outcome"] == "deny"
    assert denied["authfi.decision.id"] == "dec_override"
    assert denied["authfi.decision.rule"] == "deny-external"
    assert denied["authfi.decision.grant"] == "grt_2"
    assert denied["authfi.mandate.id"] == "man_3"
    assert denied["authfi.decision.code"] == "no_grant"


def test_a_gateway_decision_is_a_tool_child_span_of_the_step() -> None:
    p, memory = provider()
    from opentelemetry import trace

    tracer = p.get_tracer("t")
    # the SDK records through the global provider, as instrumented code does
    original = trace.get_tracer

    def fixed(*_a: object, **_k: object) -> Any:
        return tracer

    trace.get_tracer = fixed
    try:
        with tracer.start_as_current_span("step") as step:
            gateway = authfi.Gateway.read_result
            gateway({"isError": True, "_meta": {REFUSAL_META: {"code": "no_grant", "decision_id": "dec_1"}}})
            step_id = step.get_span_context().span_id
    finally:
        trace.get_tracer = original
    decision = next(s for s in memory.get_finished_spans() if s.name.startswith("authfi.decision"))
    assert decision.parent is not None and decision.parent.span_id == step_id
    assert decision.attributes is not None
    assert decision.attributes["langsmith.span.kind"] == "tool"
    assert decision.attributes["authfi.decision.id"] == "dec_1"


def test_no_current_span_records_nothing() -> None:
    p, memory = provider()
    record_decision(ToolAllowed("x"))
    assert not memory.get_finished_spans()
    _ = p


# --- no secret in a span ---------------------------------------------------------------------------------


def test_redact_span_removes_every_secret_shape() -> None:
    p, memory = provider()
    jwt_like = "eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiJhZ3RfMSJ9.c2lnbmF0dXJlLWJ5dGVz"
    with p.get_tracer("t").start_as_current_span(
        "call " + jwt_like,
        attributes={
            "http.request.header.authorization": "anything",
            "tool.api_key": "abc",
            "gen_ai.tool.call.arguments": f"curl -H 'Authorization: Bearer {jwt_like}' sk-ant-0123456789abcdef0123",
            "aws": "AKIAABCDEFGHIJKLMNOP and ghp_0123456789abcdefghij",
            "key": "-----BEGIN PRIVATE KEY-----",
            "gen_ai.usage.output_tokens": 12,
            "note": "fine",
        },
    ) as span:
        span.add_event("tool.result", {"result": f"DPoP {jwt_like}", "ok": "yes"})
        span.set_status(Status(StatusCode.ERROR, f"failed with Bearer {jwt_like}"))
    clean = redact_span(memory.get_finished_spans()[0])
    shown = repr(dict(clean.attributes or {})) + clean.name + repr([dict(e.attributes or {}) for e in clean.events])
    shown += clean.status.description or ""
    for secret in (
        jwt_like,
        "sk-ant-0123456789abcdef0123",
        "AKIAABCDEFGHIJKLMNOP",
        "ghp_0123456789abcdefghij",
        "BEGIN PRIVATE",
    ):
        assert secret not in shown
    assert REDACTED in shown
    attributes = dict(clean.attributes or {})
    assert attributes["note"] == "fine"
    assert attributes["gen_ai.usage.output_tokens"] == 12  # a token COUNT is not a token
    assert redact_span(clean) is clean


def test_a_clean_span_is_passed_through_untouched() -> None:
    p, memory = provider()
    with p.get_tracer("t").start_as_current_span("plain", attributes={"a": "b"}):
        pass
    span = memory.get_finished_spans()[0]
    assert redact_span(span) is span


def test_the_exporter_redacts_even_when_content_capture_is_on() -> None:
    platform = AgentPlatform()
    sent: list[ReadableSpan] = []

    class Inner:
        def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
            sent.extend(spans)
            return SpanExportResult.SUCCESS

    with authfi.AuthFI(platform.issuer, http_client=platform.client()) as af:
        m2m = af.client_credentials("app", authfi.Secret("s"), resource=platform.issuer + "/mcp")
        real_token = "eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiJhZ3RfMSJ9.c2lnbmF0dXJlLWJ5dGVz"
        for capture in (True, False):
            sent.clear()
            exporter = authfi_otel.exporter(m2m, capture_content=capture)
            exporter._inner = Inner()  # type: ignore[assignment]  # pyright: ignore[reportPrivateUsage]
            p, memory = provider()
            with p.get_tracer("t").start_as_current_span(
                "s", attributes={"gen_ai.tool.call.result": real_token, "other": f"Bearer {real_token}"}
            ):
                pass
            exporter.export(memory.get_finished_spans())
            assert sent
            assert real_token not in repr([str(s.attributes) for s in sent])


def test_a_real_agent_token_never_reaches_the_collector() -> None:
    from tests.unit.test_otel import Collector

    collector = Collector()
    try:
        platform = AgentPlatform(issuer=collector.origin)
        with authfi.AuthFI(platform.issuer, http_client=platform.client()) as af:
            m2m = af.client_credentials("app", authfi.Secret("hunter2-secret"), resource=platform.issuer + "/mcp")
            platform.route(
                "POST",
                "/oauth/token",
                lambda _r: authfi_json(
                    {"access_token": platform.agent_token(), "token_type": "Bearer", "expires_in": 900}
                ),
            )
            token = m2m.token()
            p = TracerProvider()
            authfi_otel.register(m2m, capture_content=True, tracer_provider=p)
            with p.get_tracer("t").start_as_current_span(
                "s", attributes={"gen_ai.tool.call.arguments": f"token={token}", "client_secret": "hunter2-secret"}
            ) as span:
                span.add_event("e", {"x": f"Authorization: Bearer {token}"})
            p.force_flush()
            raw = b"".join(request.SerializeToString() for _, request in collector.exports)
            assert collector.exports
            assert token.encode() not in raw
            assert b"hunter2-secret" not in raw
    finally:
        collector.close()


def authfi_json(body: dict[str, Any]) -> Any:
    from tests.support import json_response

    return json_response(body)
