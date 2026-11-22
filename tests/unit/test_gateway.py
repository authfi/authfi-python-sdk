from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import httpx
import jwt
import pytest

from authfi import (
    APPROVAL_ARGUMENT,
    HOLD_MESSAGE,
    AgentKey,
    APIError,
    AsyncAuthFI,
    AsyncGateway,
    AuthFI,
    ConfigurationError,
    Gateway,
    GatewayTokenRefused,
    Secret,
)
from authfi.agents import AgentAuth
from authfi.gateway import token_refusal
from tests.agent_platform import AGENT, AgentPlatform
from tests.support import FakeClock, RpcError, fixture_json, json_response

HOLD_ID = "7f1c2a9e-5d1b-4f3a-9a77-2b0c8e1d4f00"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def platform(key: AgentKey) -> AgentPlatform:
    return AgentPlatform(agent_jwk=key.public_jwk())


@pytest.fixture
def agent(platform: AgentPlatform, key: AgentKey, clock: FakeClock) -> Iterator[AgentAuth]:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as af:
        yield af.agent(AGENT, key)


# --- reading the gateway's answers (contract: the platform's own shapes) ---------------------------


def test_a_refusal_is_read_from_its_meta(agent: AgentAuth) -> None:
    outcome = agent.gateway().read_result(fixture_json("mcp/refusal-result.json"))
    assert outcome.kind == "refused"
    assert outcome.code == "ETOOL003"
    assert outcome.error == "insufficient_scope"
    assert outcome.required_scope == "mcp:deepwiki.ask_question"
    assert outcome.reason == "no grant lets this agent call deepwiki__ask_question"
    assert "insufficient_scope" in outcome.model_text or "ETOOL003" in outcome.model_text


def test_a_hold_is_read_from_the_gateways_exact_message(agent: AgentAuth) -> None:
    outcome = agent.gateway().read_result(fixture_json("mcp/hold-result.json"))
    assert outcome.kind == "approval_required"
    assert outcome.approval_id == HOLD_ID
    assert outcome.code == "approval_required"
    # What a model may read: never the instruction to retry, never the id.
    assert outcome.model_text == HOLD_MESSAGE
    assert HOLD_ID not in outcome.model_text
    assert APPROVAL_ARGUMENT not in outcome.model_text


def test_results_that_are_the_tools_own(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    ok = {"content": [{"type": "text", "text": "Topics: intro"}], "isError": False}
    assert gateway.read_result(ok).kind == "allowed"
    assert gateway.read_result(ok).model_text == "Topics: intro"
    tool_error = {"content": [{"type": "text", "text": "repo not found"}], "isError": True}
    assert gateway.read_result(tool_error).kind == "allowed"
    quoting = {
        "content": [
            {"type": "text", "text": "Waiting for approval x (approval_required). A person must approve this action"}
        ]
    }
    assert gateway.read_result(quoting).kind == "allowed"
    as_object = SimpleNamespace(content=[SimpleNamespace(text="hi")], isError=False, meta=None)
    assert gateway.read_result(as_object).model_text == "hi"
    snake = SimpleNamespace(content="text", is_error=True, meta=fixture_json("mcp/refusal-result.json")["_meta"])
    assert gateway.read_result(snake).kind == "refused"


def test_the_scope_a_tool_needs(agent: AgentAuth) -> None:
    tool = fixture_json("mcp/tools-list.json")["tools"][0]
    assert agent.gateway().scope_of(tool) == "mcp:deepwiki.read_wiki_structure"
    assert agent.gateway().scope_of({"name": "x"}) is None


def test_the_agents_outlook_for_a_tool(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    listed = fixture_json("mcp/tools-list.json")["tools"][0]
    assert gateway.governance_of(listed) == "no_policy_restriction"
    # an MCP Tool keeps unknown annotations as pydantic extras
    as_mcp = SimpleNamespace(
        annotations=SimpleNamespace(model_extra={"io.authfi/governance": "never_allowed"}), meta=listed["_meta"]
    )
    assert gateway.governance_of(as_mcp) == "never_allowed"
    assert gateway.scope_of(as_mcp) == "mcp:deepwiki.read_wiki_structure"
    # a framework's tool keeps the MCP tool's annotations and _meta in its metadata
    framework_tool = SimpleNamespace(
        metadata={"io.authfi/governance": "may_require_approval", "_meta": {"io.authfi/scope": "mcp:a.b"}}
    )
    assert gateway.governance_of(framework_tool) == "may_require_approval"
    assert gateway.scope_of(framework_tool) == "mcp:a.b"
    assert gateway.governance_of({"name": "x"}) is None
    assert gateway.governance_of(SimpleNamespace(annotations=None, metadata=None)) is None


def test_a_model_written_approval_id_is_removed_and_logged(agent: AgentAuth, caplog: pytest.LogCaptureFixture) -> None:
    gateway = agent.gateway()
    with caplog.at_level(logging.WARNING, logger="authfi"):
        clean = gateway.strip_approval_argument({"q": 1, APPROVAL_ARGUMENT: "stolen"})
    assert clean == {"q": 1}
    assert any(APPROVAL_ARGUMENT in r.getMessage() for r in caplog.records)
    assert gateway.with_approval({"q": 1}, "apr-1") == {"q": 1, APPROVAL_ARGUMENT: "apr-1"}


def test_model_base_urls(agent: AgentAuth, platform: AgentPlatform) -> None:
    gateway = agent.gateway()
    assert gateway.mcp_url == platform.issuer + "/mcp"
    assert gateway.model_base_url("anthropic") == platform.issuer + "/llm"
    assert gateway.model_base_url("openai") == platform.issuer + "/llm/v1"
    with pytest.raises(ConfigurationError):
        gateway.model_base_url("gemini")  # type: ignore[arg-type]


# --- the HTTP client -------------------------------------------------------------------------------


class GatewayServer:
    def __init__(self, platform: AgentPlatform, status: int = 200) -> None:
        self.platform = platform
        self.status = status
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/mcp":
            self.seen.append(request)
            if self.status == 401:
                return httpx.Response(401, headers={"WWW-Authenticate": 'Bearer error="invalid_token"'})
            return json_response({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}})
        if request.url.path == "/elsewhere":
            return httpx.Response(302, headers={"Location": "https://evil.example/"})
        return self.platform(request)


def gateway_client(platform: AgentPlatform, key: AgentKey, clock: FakeClock, server: GatewayServer) -> Gateway:
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    gateway = af.agent(AGENT, key).gateway()
    client = gateway.http_client()
    client._transport = httpx.MockTransport(server)
    return gateway


def test_the_client_sends_the_credential_ids_and_no_redirects(
    platform: AgentPlatform, key: AgentKey, clock: FakeClock
) -> None:
    server = GatewayServer(platform)
    gateway = gateway_client(platform, key, clock, server)
    client = gateway.http_client()
    assert gateway.http_client() is client
    client.post(gateway.mcp_url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    sent = server.seen[0]
    assert sent.headers["authorization"].startswith("Bearer ")
    assert sent.headers["x-request-id"].startswith("run-")  # the agent's run, then the call's number
    assert "traceparent" in sent.headers
    assert sent.headers["user-agent"].startswith("authfi-python/")
    assert client.get(platform.issuer + "/elsewhere").status_code == 302  # not followed
    with pytest.raises(ConfigurationError):
        client.get("https://evil.example/steal")


def test_a_refused_token_is_found_in_whatever_the_caller_raised(
    platform: AgentPlatform, key: AgentKey, clock: FakeClock
) -> None:
    gateway = gateway_client(platform, key, clock, GatewayServer(platform, status=401))
    response = gateway.http_client().post(gateway.mcp_url, json={})
    assert response.status_code == 401
    with pytest.raises(httpx.HTTPStatusError) as caught:
        response.raise_for_status()
    wrapped = RuntimeError("tool call failed")
    wrapped.__cause__ = caught.value
    refused = token_refusal(wrapped)
    assert refused is not None
    assert refused.challenge == 'Bearer error="invalid_token"'
    assert token_refusal(refused) is refused


def test_token_refusal_finds_a_401_in_a_group() -> None:
    request = httpx.Request("POST", "https://acme.authfi.io/mcp")
    response = httpx.Response(401, request=request, headers={"www-authenticate": "Bearer"})
    error = httpx.HTTPStatusError("401", request=request, response=response)
    found = token_refusal(ExceptionGroup("tasks", [ValueError("x"), error]))
    assert found is not None
    assert found.challenge == "Bearer"
    assert token_refusal(ValueError("x")) is None


def test_token_refusal_reads_a_provider_sdks_httpx2_answer() -> None:
    import anthropic
    import httpx2

    request = httpx2.Request("POST", "https://acme.authfi.io/llm/v1/messages")
    response = httpx2.Response(
        401,
        request=request,
        headers={"www-authenticate": 'Bearer error="invalid_token"', "x-request-id": "afr-answer"},
        json={"error": "invalid_token"},
    )
    error = anthropic.AuthenticationError("401", response=response, body=None)
    found = token_refusal(error)
    assert found is not None
    assert found.challenge == 'Bearer error="invalid_token"'
    assert found.request_id == "afr-answer"
    assert "acme.authfi.io/llm/v1/messages" in str(found)


def test_token_refusal_ignores_other_statuses_and_bare_responses() -> None:
    assert (
        token_refusal(
            httpx.HTTPStatusError("403", request=httpx.Request("GET", "https://a.io"), response=httpx.Response(403))
        )
        is None
    )
    bare = RuntimeError("x")
    bare.response = httpx.Response(401)  # type: ignore[attr-defined]
    found = token_refusal(bare)
    assert found is not None
    assert found.challenge is None


async def test_an_mcp_client_gets_a_refused_token_as_a_json_rpc_error() -> None:
    from authfi.gateway import TOKEN_REFUSED_RPC_CODE, _RefusalAsJsonRpc

    seen: list[GatewayTokenRefused | None] = []

    def answer(request: httpx.Request) -> httpx.Response:
        if request.headers.get("x-case") == "ok":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {}})
        return httpx.Response(401, headers={"www-authenticate": 'Bearer error="invalid_token"'})

    async with httpx.AsyncClient(
        transport=cast(httpx.AsyncBaseTransport, _RefusalAsJsonRpc(httpx, httpx.MockTransport(answer), seen.append))
    ) as client:
        refused = await client.post(
            "https://acme.authfi.io/mcp", json={"jsonrpc": "2.0", "id": 7, "method": "tools/call"}
        )
        notification = await client.post("https://acme.authfi.io/mcp", json={"jsonrpc": "2.0", "method": "x"})
        not_json = await client.post("https://acme.authfi.io/mcp", content=b"{")
        stream = await client.get("https://acme.authfi.io/mcp")
        fine = await client.post("https://acme.authfi.io/mcp", json={"id": 1}, headers={"x-case": "ok"})

    assert refused.status_code == 200
    error = refused.json()["error"]
    assert refused.json()["id"] == 7
    assert error["code"] == TOKEN_REFUSED_RPC_CODE
    assert [notification.status_code, not_json.status_code, stream.status_code] == [401, 401, 401]
    assert fine.status_code == 200
    assert [type(found).__name__ for found in seen] == ["GatewayTokenRefused"] * 4 + ["NoneType"]

    # what an MCP client raises from that answer
    raised = RuntimeError("tool call failed")
    raised.error = SimpleNamespace(code=error["code"], message=error["message"], data=error["data"])  # type: ignore[attr-defined]
    found = token_refusal(raised)
    assert found is not None
    assert found.challenge == 'Bearer error="invalid_token"'
    assert "acme.authfi.io/mcp" in str(found)
    other = RuntimeError("x")
    other.error = SimpleNamespace(data={"authfi_refusal": "something else"})  # type: ignore[attr-defined]
    assert token_refusal(other) is None


async def test_the_mcp_client_factory(platform: AgentPlatform, key: AgentKey, clock: FakeClock) -> None:
    from authfi.gateway import _RefusalAsJsonRpc

    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as af:
        gateway = af.agent(AGENT, key).gateway()
        factory = gateway.mcp_client_factory()
        gateway._saw(GatewayTokenRefused("https://acme.authfi.io/mcp", None))
        assert gateway.token_refused is not None
        client = factory({"X-Extra": "1"}, httpx.Timeout(5.0))
        async with client:
            assert isinstance(client._transport, _RefusalAsJsonRpc)
            assert client.headers["x-extra"] == "1"
            assert client.timeout.connect == 5.0
            assert client.follow_redirects is False
        with pytest.raises(ConfigurationError, match="no auth"):
            factory(None, None, httpx.BasicAuth("a", "b"))


def test_close_closes_the_client(platform: AgentPlatform, key: AgentKey, clock: FakeClock) -> None:
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    gateway = af.agent(AGENT, key).gateway()
    client = gateway.http_client()
    own = gateway.new_http_client()
    af.close()
    assert client.is_closed
    assert not own.is_closed
    own.close()


def test_a_client_for_another_httpx_package(agent: AgentAuth) -> None:
    import types

    fake = types.ModuleType("httpx_fake")
    fake.Auth = httpx.Auth  # type: ignore[attr-defined]
    fake.Client = httpx.Client  # type: ignore[attr-defined]
    fake.Timeout = httpx.Timeout  # type: ignore[attr-defined]
    client = agent.gateway().http_client(fake)
    assert type(client._auth).__name__.endswith("httpx_fake")
    assert agent.gateway().http_client(fake) is client  # one client per package, kept by the agent's one gateway


# --- cloud credentials -----------------------------------------------------------------------------

REDEEM = "/mcp/cloud/credentials/{}/secret"
HANDLE: dict[str, Any] = {
    "credential_id": "c1",
    "account": "prod",
    "role": "deployer",
    "expires_at": "2026-10-20T11:00:00Z",
    "scope": "cloud_role:deployer",
}
SECRET: dict[str, Any] = {
    "credential_id": "c1",
    "provider": "aws",
    "access_key_id": "AKIA1",
    "secret_access_key": "MARKER-SECRET",
    "session_token": "MARKER-SESSION",
    "expiration": "2026-10-20T11:00:00Z",
    "expires_in": 900,
}


def tool_answer(result: dict[str, Any], status: int = 200) -> httpx.Response:
    return json_response({"jsonrpc": "2.0", "id": 1, "result": result}, status=status)


def handle_result(handle: dict[str, Any] = HANDLE) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(handle)}], "structuredContent": handle, "isError": False}


def issue_route(platform: AgentPlatform, answer: httpx.Response) -> list[dict[str, Any]]:
    """``cloud.credentials`` is a ``tools/call`` on ``/mcp``; the bodies it was called with are kept."""
    seen: list[dict[str, Any]] = []

    def door(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(answer.status_code, headers=answer.headers, content=answer.content)

    platform.route("POST", "/mcp", door)
    return seen


def redeem_route(platform: AgentPlatform, credential_id: str, answer: httpx.Response) -> None:
    platform.route("POST", REDEEM.format(credential_id), answer)


def test_cloud_credentials_issued_by_a_tool_call_and_redeemed_once(agent: AgentAuth, platform: AgentPlatform) -> None:
    seen = issue_route(platform, tool_answer(handle_result()))
    redeem_route(platform, "c1", json_response({"data": SECRET}))
    outcome = agent.gateway().cloud_credentials("prod", "deployer", session_policy={"Version": "2012-10-17"})
    assert outcome.kind == "issued"
    credential = outcome.credential
    assert credential is not None
    assert credential.access_key_id == "AKIA1"
    assert credential.secret_access_key == Secret("MARKER-SECRET")
    assert "MARKER" not in repr(credential)
    assert seen[0]["method"] == "tools/call"
    assert seen[0]["params"] == {
        "name": "cloud.credentials",
        "arguments": {
            "account": "prod",
            "role": "deployer",
            "duration_seconds": 900,
            "session_policy": {"Version": "2012-10-17"},
        },
    }
    assert len(platform.calls("POST", "/mcp")) == 1
    assert len(platform.calls("POST", REDEEM.format("c1"))) == 1
    assert platform.calls("POST", "/mcp/v1/cloud/credentials") == []  # the issuing door is gone


def test_the_issue_and_the_redeem_carry_the_tool_token_never_the_model_token(
    agent: AgentAuth, platform: AgentPlatform
) -> None:
    issue_route(platform, tool_answer(handle_result()))
    redeem_route(platform, "c1", json_response({"data": SECRET}))
    agent.gateway().cloud_credentials("prod", "deployer")
    for request in (platform.calls("POST", "/mcp")[0], platform.calls("POST", REDEEM.format("c1"))[0]):
        token = request.headers["authorization"].split(" ", 1)[1]
        assert jwt.decode(token, options={"verify_signature": False})["aud"] == platform.issuer + "/mcp"


def test_cloud_credentials_held_then_retried_with_the_approval(agent: AgentAuth, platform: AgentPlatform) -> None:
    issue_route(platform, tool_answer(fixture_json("mcp/hold-result.json")))
    outcome = agent.gateway().cloud_credentials("prod", "admin")
    assert outcome.kind == "approval_required"
    assert outcome.approval_id == HOLD_ID
    assert platform.calls("POST", REDEEM.format("c1")) == []  # nothing was issued, so nothing is redeemed
    seen = issue_route(platform, tool_answer(handle_result({**HANDLE, "credential_id": "c2", "role": "admin"})))
    redeem_route(platform, "c2", json_response({"data": {"provider": "gcp", "access_token": "T", "expires_in": 600}}))
    issued = agent.gateway().cloud_credentials("prod", "admin", approval_id=HOLD_ID)
    assert seen[0]["params"]["arguments"][APPROVAL_ARGUMENT] == HOLD_ID
    assert issued.credential is not None
    assert issued.credential.provider == "gcp"
    assert issued.credential.credential_id == "c2"
    assert issued.credential.expires_at is not None


def refusal_result(code: str) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": f"Refused ({code}): no grant."}],
        "isError": True,
        "_meta": {"io.authfi/refusal": {"code": code, "reason": "no grant"}},
    }


@pytest.mark.parametrize(
    ("answer", "error", "code"),
    [
        (tool_answer(refusal_result("ECLOUDCRED003")), APIError, "ECLOUDCRED003"),
        (tool_answer(refusal_result("ECLOUDCRED011")), APIError, "ECLOUDCRED011"),
        (tool_answer({"content": [{"type": "text", "text": "boom"}], "isError": True}), APIError, "invalid_response"),
        (tool_answer({"content": [], "isError": False}), APIError, "invalid_response"),
        (
            json_response({"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "tool call failed"}}),
            APIError,
            "rpc_error",
        ),
        (json_response({"error": "unauthorized"}, status=401), GatewayTokenRefused, None),
        (
            json_response({"error": "ECLOUDCRED010", "error_description": "not configured"}, status=501),
            APIError,
            "ECLOUDCRED010",
        ),
    ],
)
def test_cloud_credentials_refusals(
    agent: AgentAuth, platform: AgentPlatform, answer: httpx.Response, error: type[Exception], code: str | None
) -> None:
    issue_route(platform, answer)
    with pytest.raises(error) as caught:
        agent.gateway().cloud_credentials("prod", "admin")
    if code is not None:
        assert cast(APIError, caught.value).code == code
    assert platform.calls("POST", REDEEM.format("c1")) == []


@pytest.mark.parametrize(("status", "code"), [(404, "ECLOUDCRED013"), (410, "ECLOUDCRED014"), (410, "ECLOUDCRED015")])
def test_a_redeem_the_gateway_refuses_is_the_platforms_code(
    agent: AgentAuth, platform: AgentPlatform, status: int, code: str
) -> None:
    issue_route(platform, tool_answer(handle_result()))
    redeem_route(platform, "c1", json_response({"error": code, "error_description": "refused"}, status=status))
    with pytest.raises(APIError) as caught:
        agent.gateway().cloud_credentials("prod", "deployer")
    assert (caught.value.code, caught.value.status) == (code, status)
    assert len(platform.calls("POST", REDEEM.format("c1"))) == 1  # a secret is given once: never asked for again


def test_the_issue_call_is_never_sent_again_after_it_went(agent: AgentAuth, platform: AgentPlatform) -> None:
    issue_route(platform, json_response({"error": "unavailable"}, status=503))
    with pytest.raises(APIError):
        agent.gateway().cloud_credentials("prod", "deployer")
    assert len(platform.calls("POST", "/mcp")) == 1


def test_a_credential_id_that_is_not_an_id_is_not_put_in_a_url(agent: AgentAuth, platform: AgentPlatform) -> None:
    issue_route(platform, tool_answer(handle_result({**HANDLE, "credential_id": "../../v1/x"})))
    with pytest.raises(APIError) as caught:
        agent.gateway().cloud_credentials("prod", "deployer")
    assert caught.value.code == "invalid_response"
    assert [r.url.path for r in platform.requests if r.method == "POST" and r.url.path != "/oauth/token"] == ["/mcp"]


def test_cloud_credentials_arguments(agent: AgentAuth) -> None:
    with pytest.raises(ConfigurationError):
        agent.gateway().cloud_credentials("", "admin")
    with pytest.raises(ConfigurationError):
        agent.gateway().cloud_credentials("prod", "admin", expires_in=0)


def test_a_gateway_takes_the_kind_of_source_its_client_needs(agent: AgentAuth, platform: AgentPlatform) -> None:
    with pytest.raises(ConfigurationError):
        Gateway(agent.project, "token")  # type: ignore[arg-type]
    assert "Gateway(" in repr(agent.gateway())


async def test_the_async_gateway(platform: AgentPlatform, key: AgentKey, clock: FakeClock) -> None:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as aaf:
        agent = aaf.agent(AGENT, key)
        gateway = agent.gateway()
        assert isinstance(gateway, AsyncGateway)
        server = GatewayServer(platform)
        client = gateway.http_client()
        client._transport = httpx.MockTransport(server)
        await client.post(gateway.mcp_url, json={})
        assert server.seen[0].headers["authorization"].startswith("Bearer ")
        server.status = 401
        assert (await client.post(gateway.mcp_url, json={})).status_code == 401
        own = gateway.new_http_client({"X-Extra": "1"})
        assert isinstance(own, httpx.AsyncClient)
        await own.aclose()
        issue_route(platform, tool_answer(fixture_json("mcp/hold-result.json")))
        assert (await gateway.cloud_credentials("a", "r")).approval_id == HOLD_ID
        issue_route(platform, tool_answer(handle_result()))
        redeem_route(platform, "c1", json_response({"data": SECRET}))
        assert (await gateway.cloud_credentials("a", "r")).kind == "issued"
        issue_route(platform, json_response({}, status=401))
        with pytest.raises(GatewayTokenRefused):
            await gateway.cloud_credentials("a", "r")
        with pytest.raises(ConfigurationError):
            AsyncGateway(aaf, "token")  # type: ignore[arg-type]
        assert "AsyncGateway(" in repr(gateway)
    assert client.is_closed


def test_the_outlook_is_read_from_meta_first_and_survives_mcp_2(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    # mcp 2.x drops the annotation key from tools/list; _meta still carries the outlook
    as_mcp_2 = SimpleNamespace(annotations=None, meta={"io.authfi/governance": "never_allowed"})
    assert gateway.governance_of(as_mcp_2) == "never_allowed"
    # _meta wins over a stale annotation, and an old gateway's annotation still counts
    both = SimpleNamespace(
        annotations={"io.authfi/governance": "no_policy_restriction"}, meta={"io.authfi/governance": "never_allowed"}
    )
    assert gateway.governance_of(both) == "never_allowed"
    assert gateway.governance_of({"annotations": {"io.authfi/governance": "never_allowed"}}) == "never_allowed"


# --- the refusal, whichever MCP SDK raised it ----------------------------------------------------------


def test_a_refusal_is_found_in_the_error_a_newer_sdk_raises_and_in_what_wraps_it() -> None:
    refused = token_refusal(RpcError({"authfi_refusal": "gateway_token_refused", "url": "https://x/mcp"}))
    assert isinstance(refused, GatewayTokenRefused)
    wrapped = ExceptionGroup("tg", [RuntimeError("other"), RpcError({"authfi_refusal": "gateway_token_refused"})])
    assert isinstance(token_refusal(wrapped), GatewayTokenRefused)
    chained = RuntimeError("outer")
    chained.__cause__ = RpcError({"authfi_refusal": "gateway_token_refused", "www_authenticate": "Bearer"})
    found = token_refusal(chained)
    assert found is not None
    assert found.challenge == "Bearer"


def test_an_error_that_is_not_a_refusal_is_not_found() -> None:
    assert token_refusal(RuntimeError("x")) is None
    assert token_refusal(RpcError({"other": "thing"})) is None
    assert token_refusal(RpcError("not a mapping")) is None


def test_a_refusal_is_found_in_both_mcp_error_shapes() -> None:
    data = {"authfi_refusal": "gateway_token_refused", "url": "https://x/mcp", "www_authenticate": "Bearer"}
    one = RuntimeError("mcp 1.x")  # the JSON-RPC error is on .error
    one.error = SimpleNamespace(data=data)  # type: ignore[attr-defined]
    two = RpcError(data)  # mcp 2.x keeps data on the error itself
    for error in (one, two):
        found = token_refusal(error)
        assert found is not None
        assert found.challenge == "Bearer"
    other = RuntimeError("mcp 1.x")
    other.error = SimpleNamespace(data={"authfi_refusal": "something else"})  # type: ignore[attr-defined]
    assert token_refusal(other) is None


def test_learn_nonce_from_keeps_the_nonce_of_a_use_dpop_nonce_answer_and_nothing_else() -> None:
    from authfi import _dpop
    from authfi.gateway import learn_nonce_from

    url = "https://nonce-learner.authfi.io/llm/v1/messages"
    asked = httpx.Response(401, headers={"www-authenticate": 'DPoP error="use_dpop_nonce"', "dpop-nonce": "n-1"})
    assert learn_nonce_from(asked, url)
    assert _dpop.nonces.get(url) == "n-1"

    other = "https://other-learner.authfi.io/llm/v1/messages"
    for answer in (
        httpx.Response(401, headers={"www-authenticate": 'Bearer error="invalid_token"', "dpop-nonce": "n-2"}),
        httpx.Response(403, headers={"www-authenticate": 'DPoP error="use_dpop_nonce"', "dpop-nonce": "n-2"}),
        httpx.Response(401, headers={"www-authenticate": 'DPoP error="use_dpop_nonce"'}),
        httpx.Response(200),
    ):
        assert not learn_nonce_from(answer, other)
    assert _dpop.nonces.get(other) is None
    assert not learn_nonce_from(object(), other)
