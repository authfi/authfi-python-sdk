"""LangGraph + LangSmith, the way the SDK documents it (python-sdk main, docs/frameworks.md#langgraph).

    np:       AUTHFI_ISSUER=https://probe-c1.np.authfi.io AUTHFI_AGENT_ID=agt_... AUTHFI_AGENT_KEY_FILE=... python agent.py
    offline:  FW_OFFLINE=1 python agent.py            (a stub gateway; not np)

What changed since 2026-10-20 (what this checks):
  - identity is one line, authfi.connect_async(); a bound pod signs in by attestation with no key (see report.md)
  - the model is LangGraphIntegration.chat_model(): the token is sent by the gateway's HTTP client, never default_headers
  - tools come from the gateway under its names (<server>__<tool>): no dotted names, so no rename workaround
  - a refused call is a ToolMessage(status="error"); the run carries on (it used to crash with an ExceptionGroup)
  - LangSmith is on, pointed at the capture server; configure_tracing() hides content from it
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


async def instrument_control(ctx: "harness.Ctx") -> dict[str, object]:
    """Prove the leak check can see a leak. A made-up token-shaped value goes in default_headers (the old documented
    pattern) of a model that points at the capture server; LangSmith must record it and the check must find it.
    The AuthFI token is never used for this."""
    import base64
    import secrets

    from langchain_anthropic import ChatAnthropic

    fake = "eyJ" + base64.urlsafe_b64encode(secrets.token_bytes(30)).decode().rstrip("=") + ".fakepayload" + secrets.token_hex(8) + ".fakesig" + secrets.token_hex(6)
    model = ChatAnthropic(model_name=MODEL, base_url=ctx.capture.url, api_key="dummy-capture-key-not-a-real-key", default_headers={"X-Fake-Credential": fake}, max_retries=0)  # type: ignore[call-arg]
    try:
        await model.ainvoke("control")
    except Exception:
        pass  # the capture server answers {}; only the trace matters
    import langsmith

    langsmith.Client().flush() if hasattr(langsmith.Client(), "flush") else None
    await asyncio.sleep(1.5)
    found = ctx.capture.find(fake)
    ctx.capture.requests[:] = [r for r in ctx.capture.requests if r["path"] != "/v1/messages"]  # the control's own model call
    ctx.note(f"instrument control: a made-up token in default_headers {'WAS' if found else 'was NOT'} seen in the tracer payloads")
    return {"fake_token_in_default_headers_seen_by_tracer": bool(found), "paths": sorted(set(found))}


async def main() -> int:
    ctx = harness.prepare("langgraph")
    import authfi
    from langgraph.checkpoint.memory import InMemorySaver

    from authfi.langgraph import ask

    result: dict[str, object] = {"framework": "langgraph", "model": MODEL}
    credential = authfi.connect_async()
    token_seen: list[str] = []
    try:
        langgraph = credential.langgraph()
        langgraph.configure_tracing()  # LangSmith records metadata, not content
        steps = langgraph.export_steps() if os.environ.get("FW_EXPORT_STEPS") == "1" else None
        tools = await langgraph.tools()
        result["tools"] = [t.name for t in tools]
        ctx.note(f"tools from the gateway: {result['tools']}")
        graph = langgraph.agent_graph(langgraph.chat_model(MODEL, max_tokens=512), tools, checkpointer=InMemorySaver())
        state = await graph.ainvoke(ask(PROMPT), {"configurable": {"thread_id": "fw-lg-1"}})
        kinds = [(m.type, getattr(m, "status", None)) for m in state["messages"]]
        result["message_kinds"] = kinds
        result["final"] = fw.scrub(str(state["messages"][-1].content))[:300]
        result["interrupted"] = "__interrupt__" in state
        ctx.note(f"messages: {kinds}")
        ctx.note(f"final: {result['final']}")
        # the agent's own token is read here only to search the tracer payloads for it
        token_seen.append(await credential.token())
        result["instrument_control"] = await instrument_control(ctx)
        if steps is not None:
            result["steps_flushed"] = await asyncio.to_thread(steps.force_flush, 20000)
    finally:
        await credential.project.aclose()
    token_seen += ctx.stub_tokens()
    return harness.finish(ctx, "langgraph", result, token_seen)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
