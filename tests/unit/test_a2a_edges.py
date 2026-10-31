"""A2A: the wire and the certificate chain, edge by edge."""

from __future__ import annotations

import base64
import datetime as dt
import json
from typing import Any

import httpx
import pytest

import authfi
from authfi import a2a
from authfi._protocol.a2a import A2AError, CardUnverified, canonical_json, parse_card, status_for
from authfi._protocol.http import HttpResponse
from tests import a2a_pki
from tests.a2a_pki import card, make_chain, sign
from tests.agent_platform import AGENT, AgentPlatform
from tests.support import json_response
from tests.unit.test_a2a import REMOTE, Remote, project_for, reheader, remote_agent

KeyAgent = tuple[AgentPlatform, authfi.AgentKey]


@pytest.fixture
def key_agent() -> KeyAgent:
    key = authfi.AgentKey.generate(kid="k1")
    return AgentPlatform(agent_jwk=key.public_jwk()), key


@pytest.fixture
def chain() -> a2a_pki.Chain:
    return make_chain()


def refuse(served: dict[str, Any], roots: list[bytes], key_agent: KeyAgent) -> A2AError:
    platform, key = key_agent
    with project_for(platform, Remote(served)) as project, pytest.raises(A2AError) as caught:
        a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=roots)
    return caught.value


def signed(**options: Any) -> tuple[dict[str, Any], list[bytes]]:
    chain = make_chain(**options)
    alg = a2a_pki.JWS_ALG[options.get("kind", "ed25519")]
    return sign(card(), chain.agent_key, x5c=chain.x5c(), alg=alg), [chain.root_pem]


# --- certificate chains -----------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["ec", "rsa"])
def test_chains_of_other_key_types_verify(kind: str, key_agent: KeyAgent) -> None:
    served, roots = signed(kind=kind)
    platform, key = key_agent
    with project_for(platform, Remote(served)) as project:
        assert a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=roots).card.verified


def test_a_key_type_the_sdk_does_not_trust_in_certificates_is_refused(key_agent: KeyAgent) -> None:
    chain = make_chain(kind="ed448")
    served = sign(card(), a2a_pki.new_key("ed25519"), x5c=chain.x5c())
    assert isinstance(refuse(served, [chain.root_pem], key_agent), CardUnverified)


def test_a_leaf_that_is_a_certificate_authority_and_an_expired_root_are_refused(key_agent: KeyAgent) -> None:
    served, roots = signed(agent_is_ca=True)
    assert "is a certificate authority" in str(refuse(served, roots, key_agent))
    served, roots = signed(root_start=a2a_pki.NOW - dt.timedelta(days=4000))
    assert "root" in str(refuse(served, roots, key_agent))


def test_unreadable_certificates_are_named(chain: a2a_pki.Chain, key_agent: KeyAgent) -> None:
    served = sign(card(), chain.agent_key, x5c=["!!!not base64!!!"])
    assert "cannot be read" in str(refuse(served, [chain.root_pem], key_agent))
    served = sign(card(), chain.agent_key, x5c=chain.x5c())
    junk = b"-----BEGIN CERTIFICATE-----\nAAAA\n-----END CERTIFICATE-----"
    assert "cannot be read" in str(refuse(served, [junk], key_agent))


# --- signatures ---------------------------------------------------------------------------------------------------


def test_a_signature_entry_that_is_malformed_is_named(chain: a2a_pki.Chain, key_agent: KeyAgent) -> None:
    served = sign(card(), chain.agent_key, x5c=chain.x5c())
    good = served["signatures"][0]["signature"]
    for bad, says in (
        ({"protected": 1, "signature": "x"}, "lacks protected"),
        ({"protected": "!!", "signature": "x"}, "not JSON"),
        ({"protected": base64.urlsafe_b64encode(b"[1]").decode(), "signature": "x"}, "not an object"),
        ({"protected": reheader(served, kid=None), "signature": good}, "names no kid"),
    ):
        assert says in str(refuse(dict(served, signatures=[bad]), [chain.root_pem], key_agent))


def test_a_second_good_signature_rescues_a_first_bad_one(chain: a2a_pki.Chain, key_agent: KeyAgent) -> None:
    served = sign(card(), chain.agent_key, x5c=chain.x5c())
    served["signatures"].insert(0, {"protected": "!!", "signature": "x"})
    platform, key = key_agent
    with project_for(platform, Remote(served)) as project:
        assert a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=[chain.root_pem]).card.verified


def test_a_card_that_cannot_be_canonicalised_is_refused(chain: a2a_pki.Chain, key_agent: KeyAgent) -> None:
    served = sign(card(), chain.agent_key, x5c=chain.x5c())
    served["ratio"] = 1.5
    assert "canonicalised" in str(refuse(served, [chain.root_pem], key_agent))


def test_canonical_json_edges() -> None:
    assert canonical_json({"n": 2.0}) == b'{"n":2}'
    with pytest.raises(ValueError, match="key"):
        canonical_json({1: "x"})
    with pytest.raises(ValueError, match="not JSON"):
        canonical_json({"s": {1, 2}})


# --- cards and answers --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("reply", "says"),
    [
        (httpx.Response(404), "HTTP 404"),
        (json_response([1]), "not a JSON object"),
        (json_response({"name": "x", "supportedInterfaces": []}), "no name"),
    ],
)
def test_cards_that_are_not_cards_are_refused(reply: httpx.Response, says: str, key_agent: KeyAgent) -> None:
    platform, key = key_agent

    def route(request: httpx.Request) -> httpx.Response:
        return reply if request.url.host == "research.partner.example" else platform(request)

    transport = httpx.MockTransport(route)
    with (
        authfi.AuthFI(platform.issuer, http_client=httpx.Client(transport=transport)) as project,
        pytest.raises(A2AError, match=says),
    ):
        a2a.remote(project.agent(AGENT, key), REMOTE, allow_unsigned=True)


def test_a_card_skips_interfaces_and_schemes_it_wrote_badly() -> None:
    body = card(
        supportedInterfaces=[
            "not an interface",
            {"url": "", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
            {"url": REMOTE, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"},
        ],
        securitySchemes={"a": "not a scheme", "b": {"oauth2SecurityScheme": {}}},
    )
    parsed = parse_card(HttpResponse(200, (), json.dumps(body).encode(), REMOTE, "r"))
    assert [i.url for i in parsed.interfaces] == [REMOTE]
    assert parsed.schemes == ("oauth2SecurityScheme",)


def test_a_message_can_continue_a_task(chain: a2a_pki.Chain, key_agent: KeyAgent) -> None:
    platform, key = key_agent
    remote = remote_agent(chain)
    with project_for(platform, remote) as project:
        peer = a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=[chain.root_pem])
        peer.send("go on", task_id="task-1")
    assert json.loads(remote.requests[-1].content)["message"]["taskId"] == "task-1"


@pytest.mark.parametrize(
    ("reply", "says"),
    [
        ({"error": {"code": -32000, "message": "boom"}}, "boom"),
        ({"result": "nope"}, "neither a task nor a message"),
        ({"result": {"nothing": 1}}, "neither a task nor a message"),
        ({"result": {"id": "t", "status": {"state": "TASK_STATE_WORKING"}}}, None),
    ],
)
def test_json_rpc_answers(chain: a2a_pki.Chain, key_agent: KeyAgent, reply: dict[str, Any], says: str | None) -> None:
    platform, key = key_agent
    interfaces = [{"url": REMOTE + "/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]
    remote = remote_agent(chain, supportedInterfaces=interfaces)
    remote.answer = lambda _r: json_response(reply)
    with project_for(platform, remote) as project:
        peer = a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=[chain.root_pem])
        if says is None:
            assert peer.send("x").state == "TASK_STATE_WORKING"
        else:
            with pytest.raises(A2AError, match=says):
                peer.send("x")


def test_a_status_without_words_has_no_message() -> None:
    assert status_for("allow").to_status() == {"state": "TASK_STATE_WORKING"}
