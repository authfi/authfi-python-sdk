"""One ``Idempotency-Key`` per logical tool call: fresh for each call, the same on every attempt of it.

The gateway answers a retry of a call it already ran with the first answer, so a key shared by two calls drops the
second, and a key that changes between attempts runs the call twice. Each test reads the keys from the requests the
gateway received.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any, cast

import httpx
import jwt
import pytest

from authfi import AgentKey, AsyncAuthFI, AuthFI, Secret, Task, _dpop
from authfi._protocol.ids import IdempotencyLedger
from authfi.agents import AgentAuth
from tests.fake_agent_gateway import AgentGateway
from tests.fake_gateway import ALLOWED

AGENT = "agt_idem1"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture(autouse=True)
def _fresh_nonces() -> Iterator[None]:
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]


def person() -> Secret:
    return Secret(jwt.encode({"sub": "usr_1", "exp": int(time.time()) + 3600}, "k" * 32, algorithm="HS256"))


def rpc(rid: int, tool: str, **arguments: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "method": "tools/call", "params": {"name": tool, "arguments": arguments}}


def message(request: dict[str, Any]) -> dict[str, Any]:
    """The JSON-RPC message a request the gateway received carried, or ``{}``."""
    body: Any = request["json"]
    return cast("dict[str, Any]", body) if isinstance(body, dict) else {}


def sent(gw: AgentGateway) -> list[tuple[int, str | None]]:
    """``(rpc id, Idempotency-Key)`` of every tool call the gateway received, in order, retried attempts included."""
    return [
        (message(r)["id"], cast("str | None", r["headers"].get("idempotency-key")))
        for r in gw.requests
        if r["path"] == "/mcp" and message(r).get("method") == "tools/call"
    ]


def start(af: AuthFI, key: AgentKey) -> Task:
    agent: AgentAuth = af.agent(AGENT, key)
    return agent.task("support.refund", subject_token=person(), params={"order": "4411"})


def test_two_calls_carry_two_keys_and_a_call_answered_leaves_none_behind() -> None:
    ledger = IdempotencyLedger()
    first, second = ledger.key(("s", "1", "tool", b"a")), ledger.key(("s", "2", "tool", b"a"))
    assert first != second and ledger.key(("s", "1", "tool", b"a")) == first
    ledger.done(("s", "1", "tool", b"a"))
    assert ledger.key(("s", "1", "tool", b"a")) not in (first, second)


def test_the_ledger_forgets_the_oldest_call_when_it_is_full() -> None:
    ledger = IdempotencyLedger(limit=2)
    first = ledger.key("a")
    ledger.key("b")
    ledger.key("c")
    assert ledger.key("a") != first  # a was the oldest: it is a new call now, and the ledger did not grow


def test_each_call_has_its_own_key_in_the_form_of_a_uuid_v7(key: AgentKey) -> None:
    with AgentGateway() as gw, AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        for rid in (1, 2):
            assert client.post(gw.mcp_url, json=rpc(rid, ALLOWED, repoName="a/b")).status_code == 200
    (one, a), (two, b) = sent(gw)
    assert (one, two) == (1, 2) and a and b and a != b  # the same tool with the same arguments, twice: two calls
    assert all(k[14] == "7" and len(k) == 36 for k in (a, b))


def test_the_same_call_sent_again_after_a_nonce_challenge_carries_the_same_key(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gw, AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        assert client.post(gw.mcp_url, json=rpc(7, ALLOWED, repoName="a/b")).status_code == 200
        before = gw.nonce_challenges
        gw._nonce = "rotated"  # pyright: ignore[reportPrivateUsage]  # the server's nonce moves on between calls
        assert client.post(gw.mcp_url, json=rpc(8, ALLOWED, repoName="a/b")).status_code == 200
    assert gw.nonce_challenges == before + 1
    attempts = sent(gw)
    assert [rid for rid, _ in attempts] == [7, 8, 8]  # call 8: the challenged attempt and its retry
    keys = {rid: {k for r, k in attempts if r == rid} for rid in (7, 8)}
    assert len(keys[7]) == 1 and len(keys[8]) == 1 and keys[7] != keys[8]


def test_a_call_refused_for_its_token_keeps_its_key_when_it_is_sent_again(key: AgentKey) -> None:
    """The task ended (401), the call is sent again under a credential that works: still one call, one key."""
    with AgentGateway() as gw, AuthFI(gw.issuer) as af:
        task = start(af, key)
        client = task.gateway().http_client()
        gw.end_task(task.mandate_id)
        assert client.post(gw.mcp_url, json=rpc(3, ALLOWED, repoName="a/b")).status_code == 401
        gw.ended.discard(task.mandate_id)
        assert client.post(gw.mcp_url, json=rpc(3, ALLOWED, repoName="a/b")).status_code == 200
        # Answered now: a call that looks the same is a new one, and gets its own key.
        assert client.post(gw.mcp_url, json=rpc(3, ALLOWED, repoName="a/b")).status_code == 200
    (_, refused), (_, retried), (_, later) = sent(gw)
    assert refused and refused == retried and later != retried


def test_a_server_error_leaves_the_call_open_for_its_retry() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("idempotency-key"))
        return httpx.Response(503 if len(seen) == 1 else 200, json={"jsonrpc": "2.0", "id": 5, "result": {}})

    # The gateway's own client, with the platform's transport swapped for the script above.
    with AgentGateway() as gw, AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, AgentKey.generate(kid="k2")).gateway().new_http_client()
        client._transport = httpx.MockTransport(handler)  # pyright: ignore[reportPrivateUsage]
        for _ in range(3):
            client.post(gw.mcp_url, json=rpc(5, ALLOWED, repoName="a/b"), headers={"authorization": "Bearer x"})
    assert seen[0] and seen[0] == seen[1] != seen[2]  # 503, then the retry; the third is a new call


def test_a_held_call_sent_again_with_its_approval_is_a_call_of_its_own(key: AgentKey) -> None:
    with AgentGateway() as gw, AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        client.post(gw.mcp_url, json=rpc(4, ALLOWED, repoName="a/b"))
        # The gateway answered the first (a hold or a decision); the adapter sends it again with the approval.
        client.post(gw.mcp_url, json=rpc(4, ALLOWED, repoName="a/b", _authfi_approval_id="apr_1"))
    (_, first), (_, second) = sent(gw)
    assert first and second and first != second


def test_a_key_the_caller_chose_is_kept_and_other_methods_carry_none(key: AgentKey) -> None:
    with AgentGateway() as gw, AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        client.post(gw.mcp_url, json=rpc(1, ALLOWED, repoName="a/b"), headers={"Idempotency-Key": "mine-1"})
        client.post(gw.mcp_url, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert sent(gw) == [(1, "mine-1")]
    listing = next(r for r in gw.requests if message(r).get("method") == "tools/list")
    assert "idempotency-key" not in listing["headers"]


async def test_the_asyncio_gateway_keeps_the_key_of_a_refused_call(key: AgentKey) -> None:
    """The MCP client's transport turns a refused token into a JSON-RPC error; the call is still one call."""
    with AgentGateway() as gw:
        async with AsyncAuthFI(gw.issuer) as af:
            agent = af.agent(AGENT, key)
            task: Any = await agent.task("support.refund", subject_token=person(), params={"order": "4411"})
            client = task.gateway().mcp_client_factory()()
            gw.end_task(task.mandate_id)
            refused = await client.post(gw.mcp_url, json=rpc(9, ALLOWED, repoName="a/b"))
            assert refused.json()["error"]["data"]["authfi_refusal"] == "gateway_token_refused"
            gw.ended.discard(task.mandate_id)
            assert (await client.post(gw.mcp_url, json=rpc(9, ALLOWED, repoName="a/b"))).status_code == 200
            await client.aclose()
    (_, first), (_, second) = sent(gw)
    assert first and first == second
