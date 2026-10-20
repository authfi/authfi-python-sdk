"""Tests for authfi.openai_agents, against a local stand-in of the AuthFI gateway in both its forms.

The model is the SDK's LiteLLM model talking to the stand-in's Messages API,
which plays back a script; the tools are the SDK's own streamable HTTP MCP
client; traces go through the SDK's own exporter to a local capture server.
"""

import asyncio
import json
import os
import time
import unittest
import uuid
from unittest.mock import patch

try:
    from agents import Agent, Runner, set_default_openai_key
    from agents.tracing import set_trace_processors
    from agents.tracing.processors import BackendSpanExporter, BatchTraceProcessor, default_exporter, default_processor

    from authfi.openai_agents import AuthFIMCPServer, AuthFIModel, configure_tracing, run_config
except ImportError:  # pragma: no cover
    Agent = None

from fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, CaptureServer, FakeGateway

PROMPT_SECRET = "my phone is +1-555-0100 and email priya@example.com"


def _use(tool, repo="pallets/flask"):
    return {"type": "tool_use", "name": tool.replace(".", "__"), "input": {"repoName": repo}}


def _tool_results(model_request):
    out = []
    for message in model_request["json"]["messages"]:
        for block in message.get("content") if isinstance(message.get("content"), list) else ():
            if block.get("type") == "tool_result":
                content = block.get("content")
                text = content if isinstance(content, str) else " ".join(c.get("text", "") for c in content or ())
                out.append(text)
    return out


@unittest.skipIf(Agent is None, "openai-agents is not installed")
class TestOpenAIAgents(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "OPENAI_API_KEY": ""})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: set_trace_processors([default_processor()]))

    def _agent(self, gw, token):
        server = AuthFIMCPServer(lambda: token, url=gw.url)
        model = AuthFIModel(lambda: token, base_url=gw.url)
        return server, Agent(name="research", instructions="Use the tools.", model=model, mcp_servers=[server])

    def _run(self, gw, script, token="tok-agent", prompt="go", after=None):
        gw.model_script.extend(script)

        async def main():
            server, agent = self._agent(gw, token)
            async with server:
                result = await Runner.run(agent, prompt, run_config=run_config(lambda: token))
                if after:
                    result = await after(server, agent, result)
            return result
        return asyncio.run(main())

    def _exporter_to(self, cap):
        return BatchTraceProcessor(BackendSpanExporter(api_key="capture-only", endpoint=cap.url + "/v1/traces/ingest"))

    def test_the_default_exporter_sends_the_token_when_it_is_the_openai_key(self):
        """The control leg: what configure_tracing exists to prevent."""
        token = "tok-%s" % uuid.uuid4().hex
        with FakeGateway() as gw, CaptureServer() as cap:
            exporter, processor = default_exporter(), default_processor()
            old = exporter.endpoint
            exporter.endpoint = cap.url + "/v1/traces/ingest"
            self.addCleanup(setattr, exporter, "endpoint", old)
            set_trace_processors([processor])
            set_default_openai_key(token)
            self.addCleanup(set_default_openai_key, "unused-in-tests", True)
            self._run(gw, [[{"type": "text", "text": "hi"}]], token=token)
            processor.force_flush()
            self.assertIn(token.encode(), cap.blob())

    def test_configured_tracing_exports_no_token_and_no_content(self):
        token = "tok-%s" % uuid.uuid4().hex
        with FakeGateway() as gw, CaptureServer() as cap:
            processors = configure_tracing([self._exporter_to(cap)])
            self.addCleanup(lambda: os.environ.pop("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA", None))
            self._run(gw, [[_use(ALLOWED)], [{"type": "text", "text": "answer " + PROMPT_SECRET}]],
                      token=token, prompt="Look it up. " + PROMPT_SECRET)
            processors[0].force_flush()
            blob = cap.blob()
            self.assertTrue(cap.bodies, "nothing was exported: the capture proves nothing")
            self.assertIn(b"research", blob)
            self.assertNotIn(token.encode(), blob)
            self.assertNotIn(b"priya@example.com", blob)
            self.assertNotIn(("%s answered" % ALLOWED).encode(), blob, "a tool result was exported")

    def test_with_no_processor_tracing_is_off(self):
        with FakeGateway() as gw, CaptureServer() as cap:
            exporter = default_exporter()
            old = exporter.endpoint
            exporter.endpoint = cap.url + "/v1/traces/ingest"
            self.addCleanup(setattr, exporter, "endpoint", old)
            configure_tracing()
            self.addCleanup(lambda: os.environ.pop("OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA", None))
            from agents.tracing import set_tracing_disabled
            self.addCleanup(set_tracing_disabled, False)
            self._run(gw, [[{"type": "text", "text": "hi"}]])
            default_processor().force_flush()
            self.assertEqual(cap.bodies, [])

    def test_model_and_allowed_tool_with_the_token_and_ids(self):
        with FakeGateway() as gw:
            result = self._run(gw, [[_use(ALLOWED)], [{"type": "text", "text": "Done."}]])
            first = gw.model_requests()[0]
            self.assertEqual(first["headers"]["authorization"], "Bearer tok-agent")
            self.assertTrue(first["headers"]["x-request-id"].startswith("trace_"))
            names = [t["name"] for t in first["json"]["tools"]]
            self.assertIn("deepwiki__read_wiki_structure", names)
            self.assertNotIn(NEVER.replace(".", "__"), names)
            (headers, params), = gw.tool_requests()
            self.assertEqual(params["name"], ALLOWED)
            self.assertEqual(headers["authorization"], "Bearer tok-agent")
            self.assertTrue(headers["x-request-id"].startswith("span_"), headers.get("x-request-id"))
            self.assertEqual(result.final_output, "Done.")

    def test_a_refusal_reaches_the_model_in_both_gateway_forms(self):
        for form in ("http", "tool_error"):
            with self.subTest(form=form), FakeGateway(refusals=form) as gw:
                result = self._run(gw, [[_use(UNGRANTED)], [{"type": "text", "text": "Refused."}]])
                text, = _tool_results(gw.model_requests()[1])
                self.assertIn("Refused (insufficient_scope", text)
                self.assertIn("mcp:" + UNGRANTED, text)
                self.assertNotIn("An error occurred", text)
                self.assertEqual(result.final_output, "Refused.")

    def test_a_held_call_pauses_the_run_and_the_approval_is_used_once(self):
        for form in ("http", "tool_error"):
            with self.subTest(form=form), FakeGateway(refusals=form) as gw:
                seen = {}

                async def approve(server, agent, result):
                    item, = result.interruptions
                    held = server.approval_for(item)
                    seen.update(held)
                    self.assertEqual(gw.pending(), [held["approval_id"]])
                    self.assertEqual(gw.tool_calls, [])
                    gw.decide(held["approval_id"], "approved")
                    state = result.to_state()
                    state.approve(item)
                    return await Runner.run(agent, state, run_config=run_config(lambda: "tok-agent"))

                result = self._run(gw, [[_use(STEP_UP)], [_use(STEP_UP)], [{"type": "text", "text": "Read."}]],
                                   after=approve)
                self.assertEqual(seen["tool"], STEP_UP)
                self.assertEqual(len(gw.approvals), 1, "a second approval was opened")
                self.assertEqual(gw.tool_calls, [(STEP_UP, {"repoName": "pallets/flask"})])
                self.assertIn("waiting for a person's approval", _tool_results(gw.model_requests()[1])[0])
                self.assertEqual(result.final_output, "Read.")
                self.assertNotIn("_authfi_approval_id", json.dumps([r["json"] for r in gw.model_requests()]))

    def test_a_denied_call_reaches_the_model_as_refused(self):
        with FakeGateway() as gw:
            async def deny(server, agent, result):
                item, = result.interruptions
                gw.decide(server.approval_for(item)["approval_id"], "denied")
                state = result.to_state()
                state.approve(item)    # the person decided; the gateway says how
                return await Runner.run(agent, state)
            self._run(gw, [[_use(STEP_UP)], [_use(STEP_UP)], [{"type": "text", "text": "ok"}]], after=deny)
            self.assertIn("approval denied", _tool_results(gw.model_requests()[-1])[-1])
            self.assertEqual(gw.tool_calls, [])

    def test_names_from_a_gateway_that_publishes_provider_legal_ones(self):
        with FakeGateway(names="wire") as gw:
            self._run(gw, [[_use(ALLOWED)], [{"type": "text", "text": "Done."}]])
            self.assertEqual([p["name"] for _, p in gw.tool_requests()], ["deepwiki__read_wiki_structure"])
            self.assertEqual(len(gw.tool_calls), 1)

    def test_one_model_refreshes_its_token(self):
        tokens = iter(["tok-1", "tok-2"])
        with FakeGateway() as gw:
            gw.model_script.extend([[{"type": "text", "text": "a"}], [{"type": "text", "text": "b"}]])
            model = AuthFIModel(lambda: next(tokens), base_url=gw.url)
            agent = Agent(name="a", model=model)

            async def main():
                await Runner.run(agent, "one")
                await Runner.run(agent, "two")
            asyncio.run(main())
            self.assertEqual([r["headers"]["authorization"] for r in gw.model_requests()],
                             ["Bearer tok-1", "Bearer tok-2"])

    def test_run_config_names_the_agent_and_keeps_content_out(self):
        from authfi import AgentAuth, AgentKey
        agent = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url="https://acme.authfi.io")
        cfg = run_config(agent, workflow_name="w")
        self.assertEqual(cfg.trace_metadata, {"authfi_agent_id": "agt_1"})
        self.assertFalse(cfg.trace_include_sensitive_data)
        self.assertTrue(run_config(agent, capture_content=True).trace_include_sensitive_data)


if __name__ == "__main__":
    unittest.main()
