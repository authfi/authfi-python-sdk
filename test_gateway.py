"""Tests for authfi.gateway: credentials, tool names, reading the gateway's answers, and the MCP HTTP client."""

import asyncio
import json
import time
import unittest
from unittest.mock import MagicMock

from authfi import AgentAuth, AgentKey, AsyncAgentAuth, AuthFIError, DelegatedToken, _b64encode
from authfi import gateway
from authfi.gateway import (
    APPROVAL_ARGUMENT,
    HELD,
    REFUSED,
    RESULT,
    Credential,
    ToolNames,
    call_digest,
    never_allowed,
    provider_tool_name,
    read_result,
    request_id,
    with_approval,
)

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None


def _jwt(claims):
    return "%s.%s.sig" % (_b64encode(b'{"alg":"EdDSA"}'), _b64encode(json.dumps(claims).encode()))


def _agent():
    agent = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url="https://acme.authfi.io")
    agent._access_token, agent._expires_at = "tok-agent", time.time() + 900
    return agent


class TestCredential(unittest.TestCase):
    def test_agent_auth_is_read_per_call(self):
        agent = _agent()
        cred = Credential(agent)
        self.assertEqual(cred.token(), "tok-agent")
        self.assertEqual(cred.agent_id, "agt_1")
        self.assertEqual(cred.platform_url, "https://acme.authfi.io")
        agent._access_token = "tok-agent-2"
        self.assertEqual(cred.token(), "tok-agent-2")
        self.assertEqual(asyncio.run(cred.atoken()), "tok-agent-2")

    def test_async_agent_auth_unwraps(self):
        cred = Credential(AsyncAgentAuth(_agent()))
        self.assertEqual(cred.agent_id, "agt_1")
        self.assertEqual(cred.headers(), {"Authorization": "Bearer tok-agent"})

    def test_delegated_token_names_the_agent_and_its_project_and_renews(self):
        tok = _jwt({"sub": "priya", "client_id": "agt_1", "act": {"sub": "agt_1"}, "iss": "https://acme.authfi.io"})
        renewed = _jwt({"sub": "priya", "client_id": "agt_1", "n": 2})
        delegated = DelegatedToken({"access_token": tok, "expires_in": 1},
                                   renew=lambda: {"access_token": renewed, "expires_in": 900})
        cred = Credential(delegated)
        self.assertEqual(cred.agent_id, "agt_1")
        self.assertEqual(gateway.model_base_url(cred), "https://acme.authfi.io/mcp")
        self.assertEqual(cred.token(), renewed)

    def test_functions_sync_and_async(self):
        self.assertEqual(Credential(lambda: "tok-a").token(), "tok-a")

        async def fetch():
            return "tok-b"
        cred = Credential(fetch)
        self.assertEqual(asyncio.run(cred.atoken()), "tok-b")
        with self.assertRaises(AuthFIError):
            cred.token()

    def test_a_fixed_token_is_refused(self):
        with self.assertRaises(AuthFIError) as ctx:
            Credential("eyJhbGciOi.fixed.token")
        self.assertIn("cannot be refreshed", str(ctx.exception))

    def test_an_empty_token_is_an_error_not_a_header(self):
        with self.assertRaises(AuthFIError):
            Credential(lambda: "").token()

    def test_repr_never_shows_the_token(self):
        self.assertNotIn("tok-agent", repr(Credential(_agent())))

    def test_model_base_url(self):
        self.assertEqual(gateway.model_base_url(Credential(_agent())), "https://acme.authfi.io/mcp")
        self.assertEqual(gateway.model_base_url(Credential(lambda: "t"), "https://x/mcp/"), "https://x/mcp")
        with self.assertRaises(AuthFIError):
            gateway.model_base_url(Credential(lambda: "t"))


class TestToolNames(unittest.TestCase):
    def test_dotted_names_become_provider_legal(self):
        self.assertEqual(provider_tool_name("deepwiki.read_wiki_structure"), "deepwiki__read_wiki_structure")

    def test_legal_names_pass_through(self):
        for name in ("deepwiki__read_wiki_structure", "search", "a-b_c"):
            self.assertEqual(provider_tool_name(name), name)

    def test_other_characters_and_first_letter(self):
        self.assertEqual(provider_tool_name("jira/list issues"), "jira_list_issues")
        self.assertEqual(provider_tool_name("9lives.tool"), "t_9lives__tool")

    def test_long_names_stay_distinct_and_within_64(self):
        a = provider_tool_name("server." + "x" * 80 + "a")
        b = provider_tool_name("server." + "x" * 80 + "b")
        self.assertLessEqual(len(a), 64)
        self.assertNotEqual(a, b)

    def test_mapping_goes_both_ways(self):
        names = ToolNames()
        wire = names.provider("deepwiki.read_wiki_structure")
        self.assertEqual(names.gateway(wire), "deepwiki.read_wiki_structure")
        self.assertEqual(names.gateway("unknown"), "unknown")

    def test_two_names_that_collide_stay_distinct(self):
        names = ToolNames()
        a = names.provider("a.b")
        b = names.provider("a__b")
        self.assertNotEqual(a, b)
        self.assertEqual(names.gateway(a), "a.b")
        self.assertEqual(names.gateway(b), "a__b")


class TestGovernance(unittest.TestCase):
    def test_dict_tool(self):
        self.assertTrue(never_allowed({"name": "x", "annotations": {"io.authfi/governance": "never_allowed"}}))
        self.assertFalse(never_allowed({"name": "x", "annotations": {"io.authfi/governance": "may_require_approval"}}))
        self.assertFalse(never_allowed({"name": "x"}))

    def test_pydantic_annotations_keep_unknown_keys_as_extra(self):
        annotations = MagicMock(spec=["model_extra"])
        annotations.model_extra = {"io.authfi/governance": "never_allowed"}
        tool = MagicMock(spec=["annotations"])
        tool.annotations = annotations
        self.assertTrue(never_allowed(tool))

    def test_langchain_tool_metadata(self):
        tool = MagicMock(spec=["metadata"])
        tool.metadata = {"io.authfi/governance": "never_allowed", "readOnlyHint": None}
        self.assertTrue(never_allowed(tool))


def _result(text, is_error=True, meta=None):
    out = {"content": [{"type": "text", "text": text}], "isError": is_error}
    if meta:
        out["_meta"] = meta
    return out


class TestReadResult(unittest.TestCase):
    def test_tool_answer(self):
        answer = read_result(_result("the wiki has 4 pages", is_error=False))
        self.assertEqual(answer.kind, RESULT)
        self.assertEqual(answer.text, "the wiki has 4 pages")

    def test_hold_before_refusals_became_tool_errors(self):
        text = ("Approval required. A human must approve this action in the AuthFI console "
                "(approval 6f1c2a4e-0b7d-4d6e-9a51-3c2b1d0e9f87). Retry this call with an argument "
                "\"_authfi_approval_id\": \"6f1c2a4e-0b7d-4d6e-9a51-3c2b1d0e9f87\" once it is approved.")
        answer = read_result(_result(text))
        self.assertEqual(answer.kind, HELD)
        self.assertEqual(answer.approval_id, "6f1c2a4e-0b7d-4d6e-9a51-3c2b1d0e9f87")

    def test_hold_after(self):
        text = ("Waiting for approval apr_9 (approval_pending). A person must approve this action in the "
                "AuthFI console. Retry this call with an argument \"_authfi_approval_id\": \"apr_9\" once it is approved.")
        answer = read_result(_result(text))
        self.assertEqual((answer.kind, answer.approval_id, answer.code), (HELD, "apr_9", "approval_pending"))

    def test_structured_hold(self):
        answer = read_result(_result("held", meta={"io.authfi/approval": {"id": "apr_2", "status": "pending"}}))
        self.assertEqual((answer.kind, answer.approval_id), (HELD, "apr_2"))
        answer = read_result(_result("held", meta={"io.authfi/refusal": {"code": "approval_required",
                                                                          "approval_id": "apr_3"}}))
        self.assertEqual((answer.kind, answer.approval_id), (HELD, "apr_3"))

    def test_refusal_with_meta(self):
        answer = read_result(_result(
            "Refused (insufficient_scope, no_match): no grant lets this agent call jira.list_issues. "
            "The call needs the scope mcp:jira.list_issues.",
            meta={"io.authfi/refusal": {"error": "insufficient_scope", "code": "no_match",
                                        "reason": "no grant", "required_scope": "mcp:jira.list_issues"}}))
        self.assertEqual(answer.kind, REFUSED)
        self.assertEqual(answer.code, "no_match")
        self.assertEqual(answer.required_scope, "mcp:jira.list_issues")

    def test_denied_approval_is_a_refusal(self):
        self.assertEqual(read_result(_result("Denied by policy: approval denied")).kind, REFUSED)
        meta = {"io.authfi/refusal": {"code": "approval_denied", "approval_id": "apr_9"}}
        answer = read_result(_result("Refused (approval_denied): approval denied (approval apr_9).", meta=meta))
        self.assertEqual((answer.kind, answer.code, answer.approval_id), (REFUSED, "approval_denied", "apr_9"))

    def test_a_tool_that_quotes_the_gateway_is_still_the_tool(self):
        self.assertEqual(read_result(_result("Approval required. (approval x)", is_error=False)).kind, RESULT)
        self.assertEqual(read_result(_result("Refused by the upstream", is_error=False)).kind, RESULT)

    def test_an_error_from_the_tool_itself_is_the_tool_answer(self):
        self.assertEqual(read_result(_result("repository not found")).kind, RESULT)

    def test_text_and_flag(self):
        self.assertEqual(read_result("Denied by policy: no_match", is_error=True).kind, REFUSED)

    def test_object_form(self):
        result = MagicMock(spec=["content", "isError", "meta"])
        block = MagicMock(spec=["text"])
        block.text = "Waiting for approval apr_1 (approval_required). ..."
        result.content, result.isError, result.meta = [block], True, None
        self.assertEqual(read_result(result).approval_id, "apr_1")


class TestRequestIdsAndApprovals(unittest.TestCase):
    def test_request_id_alphabet(self):
        self.assertEqual(request_id("0b7d4d6e-9a51-3c2b"), "0b7d4d6e-9a51-3c2b")
        self.assertEqual(request_id("toolu_01ABC:x.y"), "toolu_01ABC:x.y")
        for bad in ("a b", "a;b", "x" * 129, "é", "", "a\nb"):
            self.assertIsNone(request_id(bad))
        self.assertIsNone(request_id(None))

    def test_digest_ignores_the_approval_and_key_order(self):
        a = call_digest("t", {"x": 1, "y": 2})
        self.assertEqual(a, call_digest("t", {"y": 2, "x": 1, APPROVAL_ARGUMENT: "apr_1"}))
        self.assertNotEqual(a, call_digest("t", {"x": 1, "y": 3}))
        self.assertNotEqual(a, call_digest("u", {"x": 1, "y": 2}))

    def test_with_approval_does_not_touch_the_original(self):
        args = {"x": 1}
        self.assertEqual(with_approval(args, "apr_1"), {"x": 1, APPROVAL_ARGUMENT: "apr_1"})
        self.assertEqual(args, {"x": 1})


@unittest.skipIf(httpx is None, "httpx is not installed")
class TestMcpHttpClient(unittest.TestCase):
    """The client an MCP client uses, against a stand-in of the gateway in each of its forms."""

    def _post(self, gw, cred, body):
        async def main():
            factory = gateway.mcp_http_client_factory(cred)
            async with factory(headers={"Accept": "application/json, text/event-stream"}) as client:
                return await client.post(gw.url, json=body)
        return asyncio.run(main())

    def test_a_403_refusal_of_a_tool_call_becomes_the_tool_error(self):
        from fake_gateway import UNGRANTED, FakeGateway
        with FakeGateway(refusals="http") as gw:
            res = self._post(gw, lambda: "tok-1", {"jsonrpc": "2.0", "id": 7, "method": "tools/call",
                                                   "params": {"name": UNGRANTED, "arguments": {"repoName": "a/b"}}})
        self.assertEqual(res.status_code, 200)
        self.assertNotIn("www-authenticate", res.headers)
        body = res.json()
        self.assertEqual(body["id"], 7)
        answer = read_result(body["result"])
        self.assertEqual(answer.kind, REFUSED)
        self.assertEqual(answer.required_scope, "mcp:" + UNGRANTED)
        self.assertIn("no_match", answer.text)
        self.assertIn("The call needs the scope mcp:" + UNGRANTED, answer.text)

    def test_the_newer_gateway_answer_is_left_as_it_is(self):
        from fake_gateway import UNGRANTED, FakeGateway
        with FakeGateway(refusals="tool_error") as gw:
            res = self._post(gw, lambda: "tok-1", {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                   "params": {"name": UNGRANTED, "arguments": {}}})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(read_result(res.json()["result"]).code, "no_match")

    def test_a_401_is_not_turned_into_a_tool_error(self):
        from fake_gateway import ALLOWED, FakeGateway
        with FakeGateway() as gw:
            res = self._post(gw, lambda: "revoked", {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                     "params": {"name": ALLOWED, "arguments": {}}})
        self.assertEqual(res.status_code, 401)
        self.assertIn("invalid_token", res.headers["www-authenticate"])

    def test_only_a_tool_call_is_rewritten(self):
        async def handler(request):
            return httpx.Response(403, headers={"WWW-Authenticate": 'Bearer error="insufficient_scope"'},
                                  json={"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "no"}})

        async def main():
            transport = gateway._gateway_transport(httpx, httpx.MockTransport(handler))
            async with httpx.AsyncClient(transport=transport) as client:
                listed = await client.post("http://gw/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
                fetched = await client.get("http://gw/mcp")
            return listed.status_code, fetched.status_code
        self.assertEqual(asyncio.run(main()), (403, 403))

    def test_a_403_without_the_scope_challenge_is_left_alone(self):
        async def handler(request):
            return httpx.Response(403, json={"error": "forbidden"})

        async def main():
            transport = gateway._gateway_transport(httpx, httpx.MockTransport(handler))
            async with httpx.AsyncClient(transport=transport) as client:
                res = await client.post("http://gw/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                                                "params": {"name": "t"}})
            return res.status_code
        self.assertEqual(asyncio.run(main()), 403)

    def test_each_request_carries_the_current_token(self):
        from fake_gateway import ALLOWED, FakeGateway
        tokens = iter(["tok-1", "tok-2"])
        cred = Credential(lambda: next(tokens))
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": ALLOWED, "arguments": {}}}

        async def main(gw):
            factory = gateway.mcp_http_client_factory(cred)
            async with factory() as client:
                await client.post(gw.url, json=body)
                await client.post(gw.url, json=body)
        with FakeGateway() as gw:
            asyncio.run(main(gw))
            seen = [h["authorization"] for h, _ in gw.tool_requests()]
        self.assertEqual(seen, ["Bearer tok-1", "Bearer tok-2"])

    def test_a_request_id_in_the_calls_meta_becomes_the_header(self):
        from fake_gateway import ALLOWED, FakeGateway

        def call(meta):
            return {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                    "params": {"name": ALLOWED, "arguments": {}, "_meta": meta}}

        async def main(gw):
            factory = gateway.mcp_http_client_factory(lambda: "tok-1")
            async with factory() as client:
                await client.post(gw.url, json=call({"io.authfi/request_id": "span_0123abc"}))
                await client.post(gw.url, json=call({"io.authfi/request_id": "not valid; id"}))
                await client.post(gw.url, json=call({"io.authfi/request_id": "span_meta"}),
                                  headers={"X-Request-Id": "set-by-caller"})
        with FakeGateway() as gw:
            asyncio.run(main(gw))
            seen = [h.get("x-request-id") for h, _ in gw.tool_requests()]
        self.assertEqual(seen, ["span_0123abc", None, "set-by-caller"])

    def test_an_auth_passed_by_the_caller_wins(self):
        from fake_gateway import ALLOWED, FakeGateway
        body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": ALLOWED, "arguments": {}}}

        async def main(gw):
            factory = gateway.mcp_http_client_factory(lambda: "tok-mine")
            async with factory(auth=Credential(lambda: "tok-theirs").httpx_auth()) as client:
                await client.post(gw.url, json=body)
        with FakeGateway() as gw:
            asyncio.run(main(gw))
            self.assertEqual(gw.tool_requests()[0][0]["authorization"], "Bearer tok-theirs")


class TestTokenRefusal(unittest.TestCase):
    def test_finds_a_401_inside_an_exception_group(self):
        response = MagicMock()
        response.status_code = 401
        response.headers = {"www-authenticate": 'Bearer error="invalid_token"'}
        inner = Exception("401")
        inner.response = response
        inner.request = MagicMock(url="https://acme.authfi.io/mcp")
        outer = RuntimeError("group")
        outer.exceptions = [ValueError("other"), inner]
        refused = gateway.token_refusal(outer)
        self.assertIsInstance(refused, gateway.GatewayRefusedToken)
        self.assertEqual(refused.status, 401)
        self.assertIn("invalid_token", str(refused))
        self.assertIsNone(gateway.token_refusal(ValueError("x")))


if __name__ == "__main__":
    unittest.main()
