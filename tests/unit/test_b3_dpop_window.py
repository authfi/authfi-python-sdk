"""B3: what the SDK does when the platform turns the DPoP rules on (identity's nonce, the Bearer window closing).

The platform side is in ``plans/ga-o/B3.md``. Each test is a rule the SDK must keep once ``DPOP_NONCE_KEY`` is set at
the token endpoint and an unbound agent token is dead:

- a token renewed after the server's nonce rotated gets one challenge at the token endpoint, answered once;
- no request that follows a refusal goes out as Bearer, so a refusal never downgrades the credential;
- every attempt, a nonce retry included, carries a proof of its own (a proof is good once) and the URL it is for;
- a query string is not part of what a proof names (``htu``).
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import httpx
import pytest

from authfi import AgentKey, AuthFI, _dpop
from tests.fake_agent_gateway import AgentGateway

AGENT = "agt_b3window"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture(autouse=True)
def _fresh_nonces() -> Iterator[None]:
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]


def _tools_list() -> dict[str, object]:
    return {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def test_a_token_renewed_after_the_nonce_rotated_is_challenged_once_and_bound_again(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, key)
        first = agent.token()
        assert gateway.nonce_challenges == 1  # the first mint
        gateway._nonce = "nonce-" + uuid.uuid4().hex  # pyright: ignore[reportPrivateUsage]
        second = agent.token(force=True)  # the renewal after the first token was spent

    assert second != first
    assert gateway.nonce_challenges == 2  # one more challenge, not a loop
    token_proofs = [p for p in gateway.proofs if p["path"] == "/oauth/token"]
    assert [p["claims"]["nonce"] for p in token_proofs][-1] == gateway._nonce  # pyright: ignore[reportPrivateUsage]
    assert len({p["claims"]["jti"] for p in token_proofs}) == len(token_proofs)  # the retry is a new proof
    assert gateway.issued[second]["jkt"] == agent.bound_key().thumbprint  # type: ignore[union-attr]
    assert gateway.refused_proofs == []


def test_a_refusal_never_downgrades_the_next_request_to_bearer(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        assert client.post(gateway.mcp_url, json=_tools_list()).status_code == 200
        gateway.revoked = True  # the platform now refuses this token whatever the proof
        refused = client.post(gateway.mcp_url, json=_tools_list())
        assert refused.status_code == 401
        client.post(gateway.mcp_url, json=_tools_list())

    schemes = {r["headers"]["authorization"].partition(" ")[0] for r in gateway.requests if r["path"] == "/mcp"}
    assert schemes == {"DPoP"}  # no attempt went out as Bearer
    assert "token_dpop_proof_required" not in gateway.refused_proofs


def test_a_nonce_retry_is_a_new_proof_for_the_same_request(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, key)
        agent.token()
        gateway._nonce = "nonce-" + uuid.uuid4().hex  # pyright: ignore[reportPrivateUsage]
        client = agent.gateway().http_client()
        assert client.post(gateway.mcp_url, json=_tools_list()).status_code == 200

    sent = [r for r in gateway.requests if r["path"] == "/mcp"]
    assert len(sent) == 2  # asked, then answered
    proofs = [httpx.Headers(r["headers"])["dpop"] for r in sent]
    assert proofs[0] != proofs[1]
    (accepted,) = [p for p in gateway.proofs if p["path"] == "/mcp"]
    assert accepted["claims"]["nonce"] == gateway._nonce  # pyright: ignore[reportPrivateUsage]


def test_a_proof_names_the_url_without_its_query(key: AgentKey) -> None:
    with AgentGateway() as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, key)
        headers = agent.headers("POST", gateway.mcp_url + "?session=abc#frag")
    claims = __import__("jwt").decode(headers["DPoP"], options={"verify_signature": False})
    assert claims["htu"] == gateway.mcp_url
    assert claims["htm"] == "POST"
