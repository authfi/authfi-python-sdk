"""Tests for authfi.adk, against a local stand-in of the AuthFI gateway in both its forms.

The agent is ADK's LlmAgent with LiteLlm talking to the stand-in's Messages
API, which plays back a script, and McpToolset over streamable HTTP. Nothing
inside ADK is mocked.
"""

import asyncio
import json
import os
import unittest
import uuid
from unittest.mock import patch

try:
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from authfi.adk import AuthFIADK, configure_telemetry
except ImportError:  # pragma: no cover
    AuthFIADK = None

from fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway

MODEL = "claude-haiku-5-5"


def _use(tool, repo="pallets/flask"):
    return {"type": "tool_use", "name": tool, "input": {"repoName": repo}}


def _tool_results(model_request):
    out = []
    for message in model_request["json"]["messages"]:
        for block in message.get("content") if isinstance(message.get("content"), list) else ():
            if block.get("type") == "tool_result":
                content = block.get("content")
                out.append(content if isinstance(content, str) else json.dumps(content))
    return out


@unittest.skipIf(AuthFIADK is None, "google-adk is not installed")
class TestADK(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"LITELLM_LOCAL_MODEL_COST_MAP": "True"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _session(self, gw, token="tok-agent"):
        authfi = AuthFIADK(lambda: token, url=gw.url)
        agent = authfi.agent(name="research", instruction="Use the tools.", model=MODEL)
        return authfi, agent

    async def _turn(self, runner, session, message):
        if isinstance(message, str):
            message = types.Content(role="user", parts=[types.Part(text=message)])
        events = []
        async for event in runner.run_async(user_id=session.user_id, session_id=session.id, new_message=message):
            events.append(event)
        return events

    def _converse(self, gw, script, steps):
        """Run the first message, then each step(authfi, events) -> the next message (or None to stop)."""
        gw.model_script.extend(script)
        authfi, agent = self._session(gw)

        async def main():
            service = InMemorySessionService()
            runner = Runner(agent=agent, app_name="t", session_service=service)
            session = await service.create_session(app_name="t", user_id="u")
            try:
                events = await self._turn(runner, session, "go")
                for step in steps:
                    message = step(authfi, events)
                    if message is None:
                        break
                    events = await self._turn(runner, session, message)
                return events
            finally:
                await runner.close()
        return asyncio.run(main())

    def _texts(self, events):
        return [p.text for e in events for p in (e.content.parts if e.content and e.content.parts else ())
                if p.text]

    def test_model_and_allowed_tool(self):
        with FakeGateway() as gw:
            events = self._converse(gw, [[_use(ALLOWED)], [{"type": "text", "text": "Done."}]], [])
            first = gw.model_requests()[0]
            self.assertEqual(first["headers"]["authorization"], "Bearer tok-agent")
            self.assertTrue(first["headers"].get("x-request-id"), "the model call carried no X-Request-Id")
            names = [t["name"] for t in first["json"].get("tools", [])]
            self.assertIn(ALLOWED, names)
            self.assertNotIn(NEVER, names, "a never_allowed tool was offered to the model")
            (headers, params), = gw.tool_requests()
            self.assertEqual(params["name"], ALLOWED)
            self.assertEqual(headers["authorization"], "Bearer tok-agent")
            self.assertEqual(headers.get("x-request-id"), first["headers"]["x-request-id"])
            self.assertIn("Done.", self._texts(events))

    def test_listing_tools_outside_a_run_is_authenticated(self):
        with FakeGateway() as gw:
            toolset = AuthFIADK(lambda: "tok-agent", url=gw.url).toolset()

            async def main():
                try:
                    return await toolset.get_tools()
                finally:
                    await toolset.close()
            tools = asyncio.run(main())
            self.assertIn(ALLOWED, [t.name for t in tools])
            listed = [r for r in gw.requests if isinstance(r["json"], dict) and r["json"].get("method") == "tools/list"]
            self.assertTrue(listed)
            self.assertTrue(all(r["headers"].get("authorization") == "Bearer tok-agent" for r in listed))

    def test_the_gateway_connection_never_takes_googles_mtls_client(self):
        """ADK would swap in a Google-authenticated client: no AuthFI token, a Google one instead."""
        import google.auth
        from google.auth.exceptions import DefaultCredentialsError
        from unittest.mock import MagicMock
        probe = MagicMock(side_effect=DefaultCredentialsError("no Google credentials here"))
        with FakeGateway() as gw, patch.object(google.auth, "default", probe):
            toolset = AuthFIADK(lambda: "tok-agent", url=gw.url).toolset()

            async def main():
                try:
                    return await toolset.get_tools()
                finally:
                    await toolset.close()
            with patch.dict(os.environ, {"GOOGLE_API_USE_CLIENT_CERTIFICATE": "true"}):
                self.assertTrue(asyncio.run(main()))
            self.assertFalse(probe.called, "ADK looked for Google credentials for the AuthFI connection")
            self.assertTrue(all(r["headers"].get("authorization") == "Bearer tok-agent"
                                for r in gw.requests if r["method"] == "POST"))

    def test_a_refusal_names_its_reason_in_both_gateway_forms(self):
        for form in ("http", "tool_error"):
            with self.subTest(form=form), FakeGateway(refusals=form) as gw:
                self._converse(gw, [[_use(UNGRANTED)], [{"type": "text", "text": "Refused."}]], [])
                text, = _tool_results(gw.model_requests()[1])
                self.assertIn("Refused (insufficient_scope", text)
                self.assertIn("mcp:" + UNGRANTED, text)
                self.assertNotIn("MCP tool execution failed", text)

    def test_a_held_call_is_confirmed_once_decided_and_runs_once(self):
        for form in ("http", "tool_error"):
            with self.subTest(form=form), FakeGateway(refusals=form) as gw:
                def decide_then_confirm(authfi, events):
                    held, = authfi.held_approvals(events)
                    self.assertEqual(held["tool"], STEP_UP)
                    self.assertEqual(gw.pending(), [held["approval_id"]])
                    self.assertEqual(gw.tool_calls, [])
                    gw.decide(held["approval_id"], "approved")
                    return authfi.answer(held, confirmed=True)
                events = self._converse(gw, [[_use(STEP_UP)], [{"type": "text", "text": "Read."}]],
                                        [decide_then_confirm])
                self.assertEqual(len(gw.approvals), 1)
                self.assertEqual(gw.tool_calls, [(STEP_UP, {"repoName": "pallets/flask"})])
                self.assertIn("Read.", self._texts(events))

    def test_a_call_held_again_after_confirming_asks_again(self):
        with FakeGateway(refusals="tool_error") as gw:
            seen = []

            def too_early(authfi, events):
                held, = authfi.held_approvals(events)
                seen.append(held["approval_id"])
                return authfi.answer(held, confirmed=True)       # nobody has decided yet

            def decide_then_confirm(authfi, events):
                held = authfi.held_approvals(events)
                self.assertEqual(len(held), 1, "the re-held call ended the run instead of asking again")
                self.assertIn("still waiting", json.dumps(_tool_results(gw.model_requests()[-1])))
                seen.append(held[0]["approval_id"])
                gw.decide(held[0]["approval_id"], "approved")
                return authfi.answer(held[0], confirmed=True)
            # The model calls the tool again when told the call is still waiting.
            events = self._converse(gw, [[_use(STEP_UP)], [_use(STEP_UP)], [{"type": "text", "text": "Read."}]],
                                    [too_early, decide_then_confirm])
            self.assertEqual(seen[0], seen[1])
            self.assertEqual(len(gw.approvals), 1)
            self.assertEqual(len(gw.tool_calls), 1)
            self.assertIn("Read.", self._texts(events))

    def test_a_declined_call_does_not_run(self):
        with FakeGateway() as gw:
            def decline(authfi, events):
                held, = authfi.held_approvals(events)
                gw.decide(held["approval_id"], "denied")
                return authfi.answer(held, confirmed=False)
            self._converse(gw, [[_use(STEP_UP)], [{"type": "text", "text": "Declined."}]], [decline])
            self.assertEqual(gw.tool_calls, [])
            self.assertEqual(len(gw.tool_requests()), 1)
            self.assertIn("declined this call", _tool_results(gw.model_requests()[-1])[-1])

    def test_the_token_is_read_for_every_call(self):
        tokens = iter("tok-%d" % i for i in range(1, 100))
        with FakeGateway() as gw:
            gw.model_script.extend([[_use(ALLOWED)], [{"type": "text", "text": "Done."}]])
            authfi = AuthFIADK(lambda: next(tokens), url=gw.url)
            agent = authfi.agent(name="r", instruction="x", model=MODEL)

            async def main():
                service = InMemorySessionService()
                runner = Runner(agent=agent, app_name="t", session_service=service)
                session = await service.create_session(app_name="t", user_id="u")
                await self._turn(runner, session, "go")
                await runner.close()
            asyncio.run(main())
            sent = [r["headers"]["authorization"] for r in gw.model_requests()]
            self.assertEqual(len(set(sent)), len(sent))

    def test_configure_telemetry(self):
        from authfi import AgentAuth, AgentKey
        agent = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url="https://acme.authfi.io")
        with patch.dict(os.environ, {"OTEL_RESOURCE_ATTRIBUTES": "service.name=x,authfi.agent_id=old"}):
            configure_telemetry(agent)
            self.assertEqual(os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"], "false")
            self.assertEqual(os.environ["OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"], "NO_CONTENT")
            self.assertEqual(os.environ["OTEL_RESOURCE_ATTRIBUTES"], "service.name=x,authfi.agent_id=agt_1")
            configure_telemetry(agent, capture_content=True)
            self.assertEqual(os.environ["ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS"], "true")

    def test_spans_carry_no_content_by_default(self):
        """ADK's own spans, exported in memory, hold no prompt or tool result once configured."""
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        except ImportError:
            self.skipTest("opentelemetry-sdk is not installed")
        secret = "priya@example.com-%s" % uuid.uuid4().hex[:6]
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        trace._TRACER_PROVIDER = None
        trace._TRACER_PROVIDER_SET_ONCE._done = False
        trace.set_tracer_provider(provider)
        with patch.dict(os.environ, {}):
            configure_telemetry(lambda: "tok-agent")
            with FakeGateway() as gw:
                gw.model_script.extend([[_use(ALLOWED, repo=secret)], [{"type": "text", "text": "ok " + secret}]])
                authfi = AuthFIADK(lambda: "tok-agent", url=gw.url)
                agent = authfi.agent(name="r", instruction="x", model=MODEL)

                async def main():
                    service = InMemorySessionService()
                    runner = Runner(agent=agent, app_name="t", session_service=service)
                    session = await service.create_session(app_name="t", user_id="u")
                    await self._turn(runner, session, "look up " + secret)
                    await runner.close()
                asyncio.run(main())
        spans = exporter.get_finished_spans()
        self.assertTrue(spans, "ADK exported no spans: the check proves nothing")
        dump = json.dumps([dict(s.attributes or {}) for s in spans] +
                          [[dict(e.attributes or {}) for e in s.events] for s in spans], default=str)
        self.assertNotIn(secret, dump)
        self.assertNotIn("tok-agent", dump)


if __name__ == "__main__":
    unittest.main()
