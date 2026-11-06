"""OpenAI Agents SDK, the way the SDK documents it (python-sdk main, docs/frameworks.md#openai-agents-sdk).

    np:       AUTHFI_ISSUER=https://probe-c1.np.authfi.io AUTHFI_AGENT_ID=agt_... AUTHFI_AGENT_KEY_FILE=... python agent.py
    offline:  FW_OFFLINE=1 python agent.py            (a stub gateway; not np)

What this checks that 2026-10-20 found broken:
  - the model is LiteLLM-on-the-gateway with the token set per call by the integration: no OpenAI key, so the trace
    exporter has nothing of ours to send (it used to carry the agent token to api.openai.com)
  - tool names are the gateway's (no dots), a refusal is the tool's answer
  - the SDK's own trace exporter is REPLACED by configure_tracing(); the processor handed to it exports to the capture
    server, never to OpenAI
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


def instrument_control(ctx: "harness.Ctx") -> dict[str, object]:
    """Prove the leak check sees what 2026-10-20 found: a token used as the OpenAI key rides the exporter's
    Authorization header. A made-up token-shaped key stands in; the AuthFI token is never used for this."""
    import base64
    import secrets

    from agents.tracing.processors import BackendSpanExporter

    fake = "eyJ" + base64.urlsafe_b64encode(secrets.token_bytes(30)).decode().rstrip("=") + ".fakepayload" + secrets.token_hex(8) + ".fakesig" + secrets.token_hex(6)

    class Item:
        tracing_api_key = None

        def export(self) -> dict[str, object]:
            return {"object": "trace", "id": "trace_control"}

    BackendSpanExporter(api_key=fake, endpoint=ctx.capture.url + "/control/ingest").export([Item()])  # type: ignore[list-item]
    found = ctx.capture.find(fake)
    ctx.note(f"instrument control: a made-up token as the exporter key {'WAS' if found else 'was NOT'} seen")
    # take the control's request out so it does not count as a leak of the real run
    ctx.capture.requests[:] = [r for r in ctx.capture.requests if not str(r["path"]).startswith("/control/")]
    return {"fake_token_as_exporter_key_seen": bool(found)}


async def main() -> int:
    ctx = harness.prepare("openai-agents")
    import authfi
    from agents import Agent, Runner
    from agents.tracing.processors import BackendSpanExporter, BatchTraceProcessor

    result: dict[str, object] = {"framework": "openai-agents"}
    credential = authfi.connect_async()
    tokens: list[str] = []
    try:
        oa = credential.openai_agents()
        # the SDK's default exporter is removed; the one processor named here exports to the capture server
        exporter = BackendSpanExporter(api_key="dummy-capture-key-not-a-real-key", endpoint=ctx.capture.url + "/v1/traces/ingest")
        processor = BatchTraceProcessor(exporter, schedule_delay=0.5)
        steps = None
        if os.environ.get("FW_EXPORT_STEPS") == "1":
            steps = oa.export_steps(processors=(processor,))
        else:
            oa.configure_tracing(processors=(processor,))
        async with oa.mcp_server() as tools:
            agent = Agent(name="support", instructions="Use the tool, then answer briefly.", model=oa.model(MODEL), mcp_servers=[tools])
            run = await Runner.run(agent, PROMPT, run_config=oa.run_config())
            result["interruptions"] = len(run.interruptions or [])
            result["final"] = fw.scrub(str(run.final_output))[:300]
            result["new_items"] = [type(i).__name__ for i in run.new_items]
            ctx.note(f"items: {result['new_items']}")
            ctx.note(f"final: {result['final']}")
        processor.force_flush()
        if steps is not None:
            result["steps_flushed"] = await asyncio.to_thread(steps.force_flush, 20000)
        await asyncio.sleep(1.5)
        tokens.append(await credential.token())
        # nothing of the SDK's exporter may point at OpenAI
        result["instrument_control"] = instrument_control(ctx)
        result["exporter_endpoint_is_capture"] = exporter.endpoint.startswith(ctx.capture.url)
    finally:
        await credential.project.aclose()
    tokens += ctx.stub_tokens()
    return harness.finish(ctx, "openai-agents", result, tokens)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
