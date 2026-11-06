"""Google ADK, the way the SDK documents it (python-sdk main, docs/frameworks.md#google-adk).

    np:       AUTHFI_ISSUER=https://probe-c1.np.authfi.io AUTHFI_AGENT_ID=agt_... AUTHFI_AGENT_KEY_FILE=... python agent.py
    offline:  FW_OFFLINE=1 python agent.py            (a stub gateway; not np)

Checks: the LiteLLM model sends the token as the integration's own header per call (a placeholder key is the only key),
the toolset uses the gateway's HTTP client, ADK's OpenTelemetry spans go to the capture server with content off, and a
refusal is the tool's answer. ADK needs a Google credential for nothing here.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
import fwcommon as fw  # noqa: E402
import harness  # noqa: E402

PROMPT = "Read the wiki structure of pallets/itsdangerous and tell me the section names."
MODEL = os.environ.get("AUTHFI_MODEL", "claude-haiku-5-5")


async def main() -> int:
    ctx = harness.prepare("google-adk")
    import authfi
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    result: dict[str, object] = {"framework": "google-adk"}
    credential = authfi.connect_async()
    tokens: list[str] = []
    try:
        # ADK exports through the global OpenTelemetry provider; a customer sets one up. Here it exports to the capture server.
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=ctx.capture.url + "/v1/traces")))
        trace.set_tracer_provider(provider)
        adk = credential.adk()
        if os.environ.get("FW_EXPORT_STEPS") == "1":
            adk.export_steps()
        else:
            adk.configure_tracing()  # before the runner is built: ADK content capture off
        agent = adk.agent(name="support", instruction="Use the tool, then answer briefly.", model=MODEL)
        runner = InMemoryRunner(agent=agent, app_name="support")
        session = await runner.session_service.create_session(app_name="support", user_id="u1")
        events = [e async for e in runner.run_async(user_id="u1", session_id=session.id, new_message=types.Content(role="user", parts=[types.Part(text=PROMPT)]))]
        calls = [p.function_call.name for e in events for p in (e.content.parts if e.content else []) if p.function_call]
        text = " ".join(p.text for e in events for p in (e.content.parts if e.content else []) if p.text)
        result["tool_calls"] = calls
        result["final"] = fw.scrub(text)[:300]
        result["held"] = len(adk.held_approvals(events))
        ctx.note(f"tool calls: {calls}")
        ctx.note(f"final: {result['final']}")
        # ADK's spans are exported by its own OpenTelemetry provider; flush it so the capture server sees them
        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            await asyncio.to_thread(getattr(provider, "force_flush", lambda *a, **k: None))
        except Exception as error:  # a flush that cannot happen is a note, not a failure
            ctx.note(f"otel flush: {type(error).__name__}")
        await asyncio.sleep(1.0)
        tokens.append(await credential.token())
    finally:
        await credential.project.aclose()
    tokens += ctx.stub_tokens()
    return harness.finish(ctx, "google-adk", result, tokens)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
