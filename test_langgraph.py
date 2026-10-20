"""Tests for authfi.langgraph, against a local stand-in of the AuthFI gateway in both its forms.

The model is AuthFIChatAnthropic talking to the stand-in's Messages API, which
plays back a script of tool calls; the tools are langchain-mcp-adapters over
streamable HTTP. Nothing is mocked inside LangGraph or the MCP client.
"""

import asyncio
import json
import unittest

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.types import Command, interrupt

    from authfi.gateway import GatewayRefusedToken
    from authfi.langgraph import APPROVAL_NODE, AuthFIGateway, ask
except ImportError:  # pragma: no cover
    AuthFIGateway = None

from fake_gateway import ALLOWED, NEVER, STEP_UP, UNGRANTED, FakeGateway

MODEL = "claude-haiku-5-5"


def _use(tool, repo="pallets/flask", tid=None):
    return {"type": "tool_use", "name": tool.replace(".", "__"), "input": {"repoName": repo},
            **({"id": tid} if tid else {})}


def _tool_results(model_request):
    """The tool results the model was sent, from one Messages API request."""
    out = []
    for message in model_request["json"]["messages"]:
        for block in message.get("content") if isinstance(message.get("content"), list) else ():
            if block.get("type") == "tool_result":
                content = block.get("content")
                text = content if isinstance(content, str) else " ".join(c.get("text", "") for c in content or ())
                out.append((block.get("is_error", False), text))
    return out


@unittest.skipIf(AuthFIGateway is None, "langgraph and langchain-mcp-adapters are not installed")
class TestLangGraph(unittest.TestCase):
    def _run(self, gw, script, token="tok-agent", thread="t1", resume_values=()):
        gw.model_script.extend(script)
        gateway = AuthFIGateway(lambda: token, url=gw.url)

        async def main():
            tools = await gateway.tools()
            graph = gateway.agent_graph(gateway.chat_model(MODEL, max_retries=0), tools, checkpointer=InMemorySaver())
            config = {"configurable": {"thread_id": thread}}
            states = [await graph.ainvoke(ask("go"), config)]
            for value in resume_values:
                if callable(value):
                    value = value(states[-1])
                states.append(await graph.ainvoke(Command(resume=value), config))
            return tools, states
        return asyncio.run(main())

    def test_model_and_allowed_tool(self):
        with FakeGateway() as gw:
            tools, states = self._run(gw, [[_use(ALLOWED)], [{"type": "text", "text": "Done."}]])
            names = [t["name"] for t in gw.model_requests()[0]["json"]["tools"]]
            self.assertIn("deepwiki__read_wiki_structure", names)
            self.assertNotIn(NEVER.replace(".", "__"), names, "a never_allowed tool was offered to the model")
            calls = gw.tool_requests()
            self.assertEqual([p["name"] for _, p in calls], [ALLOWED])
            self.assertTrue(calls[0][0].get("x-request-id"), "the tool call carried no X-Request-Id")
            self.assertTrue(gw.model_requests()[0]["headers"].get("x-request-id"))
            self.assertEqual(calls[0][0]["authorization"], "Bearer tok-agent")
            self.assertEqual(_tool_results(gw.model_requests()[1]),
                             [(False, "%s answered for pallets/flask" % ALLOWED)])
        self.assertEqual(states[-1]["messages"][-1].content, "Done.")

    def test_a_refusal_reaches_the_model_in_both_gateway_forms(self):
        for form in ("http", "tool_error"):
            with self.subTest(form=form), FakeGateway(refusals=form) as gw:
                _, states = self._run(gw, [[_use(UNGRANTED)], [{"type": "text", "text": "It was refused."}]])
                results = _tool_results(gw.model_requests()[1])
                self.assertEqual(len(results), 1)
                is_error, text = results[0]
                self.assertTrue(is_error)
                self.assertIn("Refused (insufficient_scope", text)
                self.assertIn("mcp:" + UNGRANTED, text)
                self.assertEqual(states[-1]["messages"][-1].content, "It was refused.")

    def test_an_approved_call_runs_once_and_opens_one_approval(self):
        for form in ("http", "tool_error"):
            with self.subTest(form=form), FakeGateway(refusals=form) as gw:
                def approve(state):
                    held = state["__interrupt__"][0].value["authfi_approvals"]
                    self.assertEqual(len(held), 1)
                    self.assertEqual(held[0]["tool"], STEP_UP)
                    self.assertEqual(held[0]["args"], {"repoName": "pallets/flask"})
                    self.assertEqual(gw.pending(), [held[0]["approval_id"]])
                    self.assertEqual(gw.tool_calls, [], "the held call ran before anyone approved it")
                    gw.decide(held[0]["approval_id"], "approved")
                    return True
                _, states = self._run(gw, [[_use(STEP_UP)], [{"type": "text", "text": "Read it."}]],
                                      resume_values=[approve])
                self.assertIn("__interrupt__", states[0])
                self.assertEqual(len(gw.approvals), 1, "resuming opened a second approval")
                self.assertEqual(gw.tool_calls, [(STEP_UP, {"repoName": "pallets/flask"})])
                self.assertEqual(_tool_results(gw.model_requests()[-1]),
                                 [(False, "%s answered for pallets/flask" % STEP_UP)])
                # The model never saw the approval id as something to copy.
                self.assertNotIn("_authfi_approval_id", json.dumps(gw.model_requests()[-1]["json"]))
                self.assertEqual(states[-1]["messages"][-1].content, "Read it.")

    def test_resuming_before_the_decision_waits_again_on_the_same_approval(self):
        with FakeGateway(refusals="tool_error") as gw:
            seen = []

            def too_early(state):
                seen.append(state["__interrupt__"][0].value["authfi_approvals"][0]["approval_id"])
                return True

            def approve(state):
                seen.append(state["__interrupt__"][0].value["authfi_approvals"][0]["approval_id"])
                gw.decide(seen[-1], "approved")
                return True
            _, states = self._run(gw, [[_use(STEP_UP)], [{"type": "text", "text": "ok"}]],
                                  resume_values=[too_early, approve])
            self.assertEqual(seen[0], seen[1])
            self.assertEqual(len(gw.approvals), 1)
            self.assertEqual(len(gw.tool_calls), 1)
            self.assertEqual(states[-1]["messages"][-1].content, "ok")

    def test_a_denied_call_reaches_the_model_as_refused(self):
        with FakeGateway() as gw:
            def deny(state):
                gw.decide(state["__interrupt__"][0].value["authfi_approvals"][0]["approval_id"], "denied")
                return True
            self._run(gw, [[_use(STEP_UP)], [{"type": "text", "text": "ok"}]], resume_values=[deny])
            (is_error, text), = _tool_results(gw.model_requests()[-1])
            self.assertTrue(is_error)
            self.assertIn("approval denied", text)
            self.assertEqual(gw.tool_calls, [])

    def test_stopping_tells_the_model_the_call_did_not_run(self):
        with FakeGateway() as gw:
            self._run(gw, [[_use(STEP_UP)], [{"type": "text", "text": "ok"}]], resume_values=[{"stop": True}])
            (is_error, text), = _tool_results(gw.model_requests()[-1])
            self.assertTrue(is_error)
            self.assertIn("Not run", text)
            self.assertEqual(len(gw.tool_requests()), 1)

    def test_the_interrupt_inside_the_tool_call_opens_an_orphan(self):
        """The control leg: the design the approval node replaces opens a second approval on resume."""
        from authfi.gateway import read_result, with_approval

        async def naive(request, handler):
            result = await handler(request)
            answer = read_result(result)
            if not answer.held:
                return result
            interrupt({"approval_id": answer.approval_id})
            return await handler(request.override(args=with_approval(request.args, answer.approval_id)))

        with FakeGateway() as gw:
            gw.model_script.extend([[_use(STEP_UP)], [{"type": "text", "text": "ok"}]])
            gateway = AuthFIGateway(lambda: "tok-agent", url=gw.url)
            gateway.interceptor = naive

            async def main():
                tools = await gateway.tools()
                graph = gateway.agent_graph(gateway.chat_model(MODEL, max_retries=0), tools,
                                            checkpointer=InMemorySaver())
                config = {"configurable": {"thread_id": "naive"}}
                state = await graph.ainvoke(ask("go"), config)
                gw.decide(state["__interrupt__"][0].value["approval_id"], "approved")
                await graph.ainvoke(Command(resume=True), config)
            asyncio.run(main())
            self.assertEqual(len(gw.approvals), 2)
            self.assertEqual(len(gw.pending()), 1)

    def test_a_token_the_gateway_refuses_ends_the_run_with_a_clear_error(self):
        with FakeGateway() as gw:
            gw.model_script.append([_use(ALLOWED)])
            current = {"token": "tok-agent"}
            # The agent is revoked while the model answers: its next call, the tool call, gets 401.
            gw.after_model = lambda: current.update(token="revoked")
            gateway = AuthFIGateway(lambda: current["token"], url=gw.url)

            async def main():
                tools = await gateway.tools()
                graph = gateway.agent_graph(gateway.chat_model(MODEL, max_retries=0), tools,
                                            checkpointer=InMemorySaver())
                await graph.ainvoke(ask("go"), {"configurable": {"thread_id": "r"}})
            with self.assertRaises(GatewayRefusedToken) as ctx:
                asyncio.run(main())
            self.assertIn("invalid_token", str(ctx.exception))

    def test_a_gateway_that_publishes_provider_legal_names(self):
        with FakeGateway(names="wire") as gw:
            tools, _ = self._run(gw, [[_use(ALLOWED)], [{"type": "text", "text": "Done."}]])
            self.assertIn("deepwiki__read_wiki_structure", [t.name for t in tools])
            self.assertEqual([p["name"] for _, p in gw.tool_requests()], ["deepwiki__read_wiki_structure"])
            self.assertEqual(len(gw.tool_calls), 1)

    def test_the_approval_node_is_named(self):
        self.assertEqual(APPROVAL_NODE, "authfi_approval")


if __name__ == "__main__":
    unittest.main()
