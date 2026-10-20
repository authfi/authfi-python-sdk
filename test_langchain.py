"""Tests for authfi.langchain: the token reaches the gateway and never the trace.

Each test runs a real ChatAnthropic call against a local stand-in of the
AuthFI AI gateway, with LangSmith tracing on and pointed at a local capture
server, and then searches every byte LangSmith sent for the token.
"""

import asyncio
import json
import os
import time
import unittest
import uuid
from unittest.mock import patch

from authfi import AgentAuth, AgentKey, DelegatedToken, _b64encode

try:
    import langsmith
    from langchain_anthropic import ChatAnthropic
    from langchain_core.tracers.langchain import LangChainTracer

    from authfi.langchain import AuthFIChatAnthropic, metadata_only_tracing
except ImportError:  # pragma: no cover
    langsmith = None

from fake_gateway import CaptureServer, FakeGateway

MODEL = "claude-haiku-5-5"


def _marker(kind):
    return "tok-%s-%s" % (kind, uuid.uuid4().hex)


@unittest.skipIf(langsmith is None, "langchain-anthropic and langsmith are not installed")
class TestTokenStaysOutOfTheTrace(unittest.TestCase):
    def setUp(self):
        self.gw = FakeGateway().__enter__()
        self.cap = CaptureServer().__enter__()
        self.addCleanup(self.gw.__exit__)
        self.addCleanup(self.cap.__exit__)
        env = {"LANGSMITH_TRACING": "true", "LANGSMITH_ENDPOINT": self.cap.url,
               "LANGSMITH_API_KEY": "capture-only", "ANTHROPIC_API_KEY": "", "ANTHROPIC_AUTH_TOKEN": ""}
        patcher = patch.dict(os.environ, env)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _traced(self, model, prompt="Say OK", stream=False, run_async=False, client=None):
        client = client or langsmith.Client(api_url=self.cap.url, api_key="capture-only")
        tracer = LangChainTracer(client=client, project_name="authfi-test")
        config = {"callbacks": [tracer]}
        if run_async:
            if stream:
                async def go():
                    return [c async for c in model.astream(prompt, config=config)]
            else:
                async def go():
                    return await model.ainvoke(prompt, config=config)
            out = asyncio.run(go())
        elif stream:
            out = list(model.stream(prompt, config=config))
        else:
            out = model.invoke(prompt, config=config)
        tracer.wait_for_futures()
        client.flush()
        time.sleep(0.2)
        return out

    def assertTraced(self):
        self.assertTrue(self.cap.bodies, "LangSmith sent nothing: the capture proves nothing")
        self.assertIn(b"ChatAnthropic", self.cap.blob())

    def test_the_documented_default_headers_way_leaks_the_token(self):
        """The control leg: the instrument sees a token when one is recorded."""
        token = _marker("leaky")
        model = ChatAnthropic(model=MODEL, base_url=self.gw.url, max_retries=0,
                              default_headers={"Authorization": "Bearer " + token})
        self._traced(model)
        self.assertTraced()
        self.assertIn(token.encode(), self.cap.blob())

    def test_an_agent_token_reaches_the_gateway_and_not_the_trace(self):
        token = _marker("agent")
        model = AuthFIChatAnthropic(agent=lambda: token, base_url=self.gw.url, model=MODEL, max_retries=0)
        self._traced(model)
        self.assertTraced()
        self.assertNotIn(token.encode(), self.cap.blob())
        sent = self.gw.model_requests()[0]["headers"]
        self.assertEqual(sent["authorization"], "Bearer " + token)
        self.assertNotIn("x-api-key", sent)

    def test_a_token_acting_for_a_person_stays_out_too(self):
        claims = {"sub": "priya@example.com", "client_id": "agt_1", "act": {"sub": "agt_1"}}
        token = "tok-" + "%s.%s.sig" % (_b64encode(b'{"alg":"EdDSA"}'), _b64encode(json.dumps(claims).encode()))
        delegated = DelegatedToken({"access_token": token, "expires_in": 600})
        model = AuthFIChatAnthropic(agent=delegated, base_url=self.gw.url, model=MODEL, max_retries=0)
        self._traced(model)
        self.assertTraced()
        self.assertNotIn(token.encode(), self.cap.blob())
        self.assertEqual(self.gw.model_requests()[0]["headers"]["authorization"], "Bearer " + token)

    def test_async_and_streaming_calls_too(self):
        for run_async in (False, True):
            for stream in (False, True):
                with self.subTest(run_async=run_async, stream=stream):
                    token = _marker("mode")
                    before = len(self.gw.model_requests())
                    model = AuthFIChatAnthropic(agent=lambda: token, base_url=self.gw.url, model=MODEL, max_retries=0)
                    self._traced(model, stream=stream, run_async=run_async)
                    self.assertNotIn(token.encode(), self.cap.blob())
                    self.assertEqual(self.gw.model_requests()[before]["headers"]["authorization"], "Bearer " + token)

    def test_the_serialized_model_has_no_authorization(self):
        token = _marker("ser")
        model = AuthFIChatAnthropic(agent=lambda: token, base_url=self.gw.url, model=MODEL)
        serialized = json.dumps(model.to_json(), default=str)
        self.assertNotIn(token, serialized)
        self.assertNotIn("Authorization", serialized)
        self.assertNotIn("agent", model.to_json().get("kwargs", {}))

    def test_one_model_object_refreshes_its_token(self):
        agent = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=self.gw.origin)
        minted = []

        def mint(requested):
            token = _marker("minted")
            minted.append(token)
            agent._access_token, agent._expires_at = token, time.time() + 900
            return token
        agent._mint = mint
        model = AuthFIChatAnthropic(agent=agent, model=MODEL, max_retries=0)
        model.invoke("one")
        agent._expires_at = time.time() + 30   # inside the refresh window
        model.invoke("two")
        sent = [r["headers"]["authorization"] for r in self.gw.model_requests()]
        self.assertEqual(sent, ["Bearer " + minted[0], "Bearer " + minted[1]])
        self.assertNotEqual(minted[0], minted[1])

    def test_base_url_defaults_to_the_agents_project_gateway(self):
        agent = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=self.gw.origin)
        agent._access_token, agent._expires_at = _marker("a"), time.time() + 900
        model = AuthFIChatAnthropic(agent=agent, model=MODEL, max_retries=0)
        model.invoke("hi")
        self.assertEqual(self.gw.model_requests()[0]["path"], "/mcp/v1/messages")
        self.assertEqual(model.metadata["authfi_agent_id"], "agt_1")

    def test_the_run_id_is_sent_as_the_request_id(self):
        model = AuthFIChatAnthropic(agent=lambda: "tok-x", base_url=self.gw.url, model=MODEL, max_retries=0)
        run_id = uuid.uuid4()
        model.invoke("hi", config={"run_id": run_id})
        self.assertEqual(self.gw.model_requests()[0]["headers"]["x-request-id"], str(run_id))

    def _metadata_only(self, capture_content):
        from langsmith import run_trees
        self.addCleanup(run_trees.configure, client=None)
        return metadata_only_tracing(capture_content=capture_content, api_url=self.cap.url, api_key="capture-only")

    def test_metadata_only_tracing_keeps_the_prompt_and_answer_out(self):
        from langchain_core.tracers.langchain import get_client
        prompt = "my card number is 4111-%s" % uuid.uuid4().hex[:8]
        client = self._metadata_only(capture_content=False)
        self.assertIs(get_client(), client)
        self.gw.model_script.append([{"type": "text", "text": "answer-" + prompt}])
        model = AuthFIChatAnthropic(agent=lambda: "tok-x", base_url=self.gw.url, model=MODEL, max_retries=0)
        self._traced(model, prompt=prompt, client=get_client())
        self.assertTraced()
        self.assertNotIn(prompt.encode(), self.cap.blob())

    def test_content_capture_is_there_when_asked_for(self):
        from langchain_core.tracers.langchain import get_client
        prompt = "capture me %s" % uuid.uuid4().hex[:8]
        self._metadata_only(capture_content=True)
        model = AuthFIChatAnthropic(agent=lambda: "tok-x", base_url=self.gw.url, model=MODEL, max_retries=0)
        self._traced(model, prompt=prompt, client=get_client())
        self.assertIn(prompt.encode(), self.cap.blob())


if __name__ == "__main__":
    unittest.main()
