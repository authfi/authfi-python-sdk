"""Tests for authfi.claude_agent and the ``authfi`` command, against a local stand-in of the gateway.

The end-to-end tests run the real Claude Code CLI that the Claude Agent SDK
ships, pointed at the stand-in for the model, the MCP tools and the token
endpoint. The model plays back a script.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
import uuid

from authfi import AgentAuth, AgentKey, DelegatedToken
from fake_gateway import ALLOWED, STEP_UP, CaptureServer, FakeGateway

try:
    from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock

    from authfi.claude_agent import AuthFIApprovals, AuthFIClaudeClient, claude_options, trace_with_langsmith
except ImportError:  # pragma: no cover
    claude_options = None

MODEL = "claude-haiku-5-5"


def _key_file(testcase):
    directory = tempfile.mkdtemp()
    path = os.path.join(directory, "agent.key")
    with open(path, "w") as fh:
        fh.write(AgentKey.generate("k1").private_pem())
    os.chmod(path, 0o600)
    testcase.addCleanup(lambda: (os.remove(path), os.rmdir(directory)))
    return path


def _agent(testcase, gw):
    return AgentAuth(agent_id="agt_1", key=AgentKey.from_file(_key_file(testcase), "k1"), platform_url=gw.origin)


class TestCommand(unittest.TestCase):
    """``authfi token`` and ``authfi headers``: what Claude Code runs for a credential."""

    def _run(self, gw, command, extra=None):
        env = {k: v for k, v in os.environ.items() if not k.startswith("AUTHFI_")}
        env.update({"AUTHFI_PLATFORM_URL": gw.origin, "AUTHFI_AGENT_ID": "agt_1",
                    "AUTHFI_AGENT_KEY_FILE": _key_file(self), "AUTHFI_AGENT_KID": "k1"})
        env.update(extra or {})
        return subprocess.run([sys.executable, "-m", "authfi", command], env=env, capture_output=True, text=True,
                              cwd=os.path.dirname(os.path.abspath(__file__)), timeout=60)

    def test_token_prints_a_fresh_token(self):
        with FakeGateway() as gw:
            out = self._run(gw, "token")
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout, "tok-minted-1\n")

    def test_headers_prints_json_with_the_run_id(self):
        with FakeGateway() as gw:
            out = self._run(gw, "headers", {"AUTHFI_RUN_ID": "run-1"})
        self.assertEqual(json.loads(out.stdout), {"Authorization": "Bearer tok-minted-1", "X-Request-Id": "run-1"})

    def test_a_failure_prints_nothing_on_stdout(self):
        with FakeGateway() as gw:
            out = self._run(gw, "token", {"AUTHFI_PLATFORM_URL": ""})
        self.assertEqual(out.returncode, 1)
        self.assertEqual(out.stdout, "")
        self.assertIn("AUTHFI_PLATFORM_URL", out.stderr)

    def test_usage(self):
        with FakeGateway() as gw:
            self.assertEqual(self._run(gw, "nonsense").returncode, 2)


@unittest.skipIf(claude_options is None, "claude-agent-sdk is not installed")
class TestOptions(unittest.TestCase):
    def test_options_for_an_agent(self):
        with FakeGateway() as gw:
            agent = _agent(self, gw)
            opts = claude_options(agent, model=MODEL, wait_for_decision=lambda a: True, run_id="run-1")
        env = opts.env
        self.assertEqual(env["ANTHROPIC_BASE_URL"], gw.url)
        self.assertNotIn("ANTHROPIC_AUTH_TOKEN", env)
        self.assertEqual(env["AUTHFI_AGENT_KEY_FILE"], agent.key.path)
        self.assertEqual(env["AUTHFI_RUN_ID"], "run-1")
        self.assertEqual(env["ANTHROPIC_CUSTOM_HEADERS"], "X-Request-Id: run-1")
        self.assertEqual(env["OTEL_RESOURCE_ATTRIBUTES"], "authfi.agent_id=agt_1,authfi.run_id=run-1")
        for switch in ("OTEL_LOG_USER_PROMPTS", "OTEL_LOG_TOOL_CONTENT", "OTEL_LOG_TOOL_DETAILS",
                       "OTEL_LOG_ASSISTANT_RESPONSES", "OTEL_LOG_RAW_API_BODIES"):
            self.assertEqual(env[switch], "0")
        self.assertIn("-m authfi token", json.loads(opts.settings)["apiKeyHelper"])
        server = opts.mcp_servers["authfi"]
        self.assertEqual(server["url"], gw.url)
        self.assertIn("-m authfi headers", server["headersHelper"])
        self.assertNotIn("headers", server)
        self.assertEqual(opts.permission_mode, "default")
        self.assertIsNotNone(opts.can_use_tool)
        self.assertEqual(set(opts.hooks), {"PreToolUse", "PostToolUse", "PostToolUseFailure"})
        self.assertNotIn("PRIVATE KEY", json.dumps(env))

    def test_content_capture_when_asked_and_env_you_pass_wins(self):
        with FakeGateway() as gw:
            opts = claude_options(_agent(self, gw), wait_for_decision=lambda a: True, capture_content=True,
                                  env={"OTEL_LOG_USER_PROMPTS": "1"})
        self.assertNotIn("OTEL_LOG_TOOL_CONTENT", opts.env)
        self.assertEqual(opts.env["OTEL_LOG_USER_PROMPTS"], "1")

    def test_a_key_not_from_a_file_is_refused(self):
        with FakeGateway() as gw:
            agent = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=gw.origin)
            env = dict(os.environ)
            env.pop("AUTHFI_AGENT_PRIVATE_KEY", None)
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                with self.assertRaises(Exception) as ctx:
                    claude_options(agent, wait_for_decision=lambda a: True)
        self.assertIn("AgentKey.from_file", str(ctx.exception))

    def test_a_delegated_token_is_passed_as_it_is(self):
        token = DelegatedToken({"access_token": "tok-person", "expires_in": 300})
        opts = claude_options(token, url="https://acme.authfi.io/mcp", wait_for_decision=lambda a: True,
                              run_id="r1")
        self.assertEqual(opts.env["ANTHROPIC_AUTH_TOKEN"], "tok-person")
        self.assertEqual(opts.mcp_servers["authfi"]["headers"]["Authorization"], "Bearer tok-person")
        self.assertNotIn("settings", json.dumps(opts.settings or {}))

    def test_other_credentials_are_refused(self):
        with self.assertRaises(Exception):
            claude_options(lambda: "tok", wait_for_decision=lambda a: True)

    def test_other_tools_are_denied_unless_said(self):
        approvals = AuthFIApprovals(wait_for_decision=lambda a: True)
        denied = asyncio.run(approvals.can_use_tool("Bash", {"command": "ls"}, None))
        self.assertEqual(denied.behavior, "deny")
        approvals = AuthFIApprovals(wait_for_decision=lambda a: True, other_tools="allow")
        self.assertEqual(asyncio.run(approvals.can_use_tool("Bash", {}, None)).behavior, "allow")


@unittest.skipIf(claude_options is None, "claude-agent-sdk is not installed")
class TestClaudeCode(unittest.TestCase):
    """The real Claude Code CLI against the stand-in gateway."""

    def _converse(self, gw, script, prompts, wait_for_decision, refresh_every=None, pause=0):
        gw.model_script.extend(script)
        agent = _agent(self, gw)
        run_id = "run-" + uuid.uuid4().hex[:8]
        waits = []

        async def wait(approval):
            waits.append(approval)
            return await wait_for_decision(approval)
        opts = claude_options(agent, model=MODEL, wait_for_decision=wait, run_id=run_id, tools=[],
                              strict_mcp_config=True, setting_sources=[], max_turns=8,
                              cwd=tempfile.gettempdir())
        texts = []

        async def main():
            async with AuthFIClaudeClient(opts, agent=agent, refresh_every=refresh_every) as client:
                for prompt in prompts:
                    await client.query(prompt)
                    async for message in client.receive_response():
                        if isinstance(message, AssistantMessage):
                            texts.extend(b.text for b in message.content if isinstance(b, TextBlock))
                        if isinstance(message, ResultMessage):
                            texts.append("RESULT %s" % message.subtype)
                    if pause:
                        await asyncio.sleep(pause)
        asyncio.run(asyncio.wait_for(main(), 120))
        return run_id, waits, texts

    def _use(self, tool, repo="pallets/flask"):
        return {"type": "tool_use", "name": "mcp__authfi__" + tool.replace(".", "_"), "input": {"repoName": repo}}

    def test_model_tool_and_credentials(self):
        with FakeGateway() as gw:
            async def never(approval):
                raise AssertionError("nothing was held")
            run_id, waits, texts = self._converse(gw, [[self._use(ALLOWED)], [{"type": "text", "text": "Done."}]],
                                                  ["read it"], never)
            self.assertEqual(gw.tool_calls, [(ALLOWED, {"repoName": "pallets/flask"})])
            model = gw.model_requests()[0]["headers"]
            self.assertTrue(model["authorization"].startswith("Bearer tok-minted-"))
            self.assertEqual(model["x-request-id"], run_id)
            tool_headers, _ = gw.tool_requests()[0]
            self.assertTrue(tool_headers["authorization"].startswith("Bearer tok-minted-"))
            self.assertEqual(tool_headers["x-request-id"], run_id)
            self.assertIn("Done.", texts)

    def test_a_held_call_is_approved_through_can_use_tool(self):
        for form in ("http", "tool_error"):
            with self.subTest(form=form), FakeGateway(refusals=form) as gw:
                async def approve(approval):
                    self.assertEqual(gw.pending(), [approval["approval_id"]])
                    self.assertEqual(gw.tool_calls, [])
                    gw.decide(approval["approval_id"], "approved")
                    return True
                _, waits, texts = self._converse(
                    gw, [[self._use(STEP_UP)], [self._use(STEP_UP)], [{"type": "text", "text": "Read."}]],
                    ["read the contents"], approve)
                self.assertEqual(len(waits), 1)
                self.assertEqual(len(gw.approvals), 1, "a second approval was opened")
                self.assertEqual(gw.tool_calls, [(STEP_UP, {"repoName": "pallets/flask"})])
                sent = [p["arguments"] for _, p in gw.tool_requests()]
                self.assertEqual(sent[-1]["_authfi_approval_id"], waits[0]["approval_id"])
                self.assertIn("Read.", texts)

    def test_a_denied_call_is_refused_before_it_is_sent(self):
        with FakeGateway() as gw:
            async def deny(approval):
                gw.decide(approval["approval_id"], "denied")
                return {"status": "denied"}
            _, waits, texts = self._converse(
                gw, [[self._use(STEP_UP)], [self._use(STEP_UP)], [{"type": "text", "text": "Declined."}]],
                ["read the contents"], deny)
            self.assertEqual(len(waits), 1)
            self.assertEqual(gw.tool_calls, [])
            self.assertEqual(len(gw.tool_requests()), 1, "the denied call was sent anyway")
            self.assertIn("declined this call in AuthFI", json.dumps(gw.model_requests()[-1]["json"]["messages"]))

    def test_the_mcp_credential_is_refreshed_while_the_session_is_open(self):
        with FakeGateway() as gw:
            async def never(approval):
                raise AssertionError("nothing was held")
            self._converse(gw, [[{"type": "text", "text": "one"}], [self._use(ALLOWED)], [{"type": "text", "text": "two"}]],
                           ["first", "second"], never, refresh_every=1, pause=2.5)
            initializes = [r["headers"]["authorization"] for r in gw.requests
                           if isinstance(r["json"], dict) and r["json"].get("method") == "initialize"]
            self.assertGreaterEqual(len(initializes), 2, "the MCP server was never reconnected")
            self.assertGreater(len(set(initializes)), 1, "the reconnect reused the old token")
            tool_headers, _ = gw.tool_requests()[-1]
            self.assertIn(tool_headers["authorization"], initializes[1:], "the tool call used the first token")


@unittest.skipIf(claude_options is None, "claude-agent-sdk is not installed")
class TestLangSmith(unittest.TestCase):
    def test_runs_carry_the_agent_and_not_the_prompt(self):
        try:
            import langsmith  # noqa: F401
        except ImportError:
            self.skipTest("langsmith is not installed")
        from langsmith import run_trees
        secret = "card 4111-%s" % uuid.uuid4().hex[:8]
        with FakeGateway() as gw, CaptureServer() as cap:
            gw.model_script.append([{"type": "text", "text": "ok " + secret}])
            agent = _agent(self, gw)
            self.addCleanup(run_trees.configure, client=None)
            trace_with_langsmith(agent, run_id="run-9", client_kwargs={"api_url": cap.url, "api_key": "capture-only"},
                                 project_name="authfi-test")
            opts = claude_options(agent, model=MODEL, wait_for_decision=lambda a: True, tools=[],
                                  strict_mcp_config=True, setting_sources=[], cwd=tempfile.gettempdir(),
                                  env={"LANGSMITH_TRACING": "true"})

            async def main():
                async with AuthFIClaudeClient(opts, agent=agent) as client:
                    await client.query("Say ok. " + secret)
                    async for _ in client.receive_response():
                        pass
            with unittest.mock.patch.dict(os.environ, {"LANGSMITH_TRACING": "true"}):
                asyncio.run(asyncio.wait_for(main(), 120))
                run_trees.get_cached_client().flush()
            time.sleep(0.5)
            blob = cap.blob()
            self.assertTrue(cap.bodies, "LangSmith sent nothing: the capture proves nothing")
            self.assertIn(b"agt_1", blob)
            self.assertIn(b"run-9", blob)
            self.assertNotIn(secret.encode(), blob)


if __name__ == "__main__":
    unittest.main()
