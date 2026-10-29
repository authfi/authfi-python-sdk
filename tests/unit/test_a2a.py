"""A2A 1.0 client helpers: the card is verified or refused, the token goes only where the card says."""

from __future__ import annotations

import base64
import copy
import datetime as dt
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519
from jwt.algorithms import OKPAlgorithm

import authfi
from authfi import a2a
from authfi._protocol.a2a import (
    AUTH_REQUIRED,
    A2AError,
    CardUnverified,
    canonical_json,
    status_for,
    status_for_outcome,
)
from authfi._protocol.gateway import ToolAllowed, ToolApprovalRequired, ToolRefused
from tests import a2a_pki
from tests.a2a_pki import card, make_chain, sign
from tests.agent_platform import AGENT, AgentPlatform
from tests.instance_platform import InstancePlatform
from tests.support import json_response

REMOTE = "https://research.partner.example"


class Remote:
    """A remote A2A agent: serves a card, takes messages, answers as told."""

    def __init__(self, served: dict[str, Any]) -> None:
        self.served = served
        self.requests: list[httpx.Request] = []
        self.answer: Callable[[httpx.Request], httpx.Response] = lambda _r: json_response(
            {"task": {"id": "task-1", "contextId": "ctx-1", "status": {"state": "TASK_STATE_COMPLETED"}}}
        )

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.path == "/.well-known/agent-card.json":
            return json_response(self.served)
        return self.answer(request)


def project_for(platform: AgentPlatform, remote: Remote) -> authfi.AuthFI:
    def route(request: httpx.Request) -> httpx.Response:
        return remote(request) if request.url.host == "research.partner.example" else platform(request)

    return authfi.AuthFI(platform.issuer, http_client=httpx.Client(transport=httpx.MockTransport(route)))


@pytest.fixture
def chain() -> a2a_pki.Chain:
    return make_chain()


@pytest.fixture
def key_agent(tmp_path: Path) -> tuple[AgentPlatform, authfi.AgentKey]:
    key = authfi.AgentKey.generate(kid="k1")
    return AgentPlatform(agent_jwk=key.public_jwk()), key


def reheader(served: dict[str, Any], **changes: Any) -> str:
    """The first signature's protected header with ``changes`` (``None`` removes a member), re-encoded."""
    protected = served["signatures"][0]["protected"]
    header = json.loads(base64.urlsafe_b64decode(protected + "=" * (-len(protected) % 4)))
    for name, value in changes.items():
        if value is None:
            header.pop(name, None)
        else:
            header[name] = value
    return base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()


def remote_agent(chain: a2a_pki.Chain, **changes: Any) -> Remote:
    return Remote(sign(card(**changes), chain.agent_key, x5c=chain.x5c()))


# --- a verified card, called with the agent's token ---------------------------------------------------------


def test_a_card_signed_to_a_trusted_root_is_called_with_a_bearer_token(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    platform, key = key_agent
    remote = remote_agent(chain)
    with project_for(platform, remote) as project:
        agent = project.agent(AGENT, key)
        peer = a2a.remote(agent, REMOTE, trust_roots=[chain.root_pem])
        assert peer.card.verified
        assert peer.card.signer is not None and "agt_remote" in peer.card.signer
        status = peer.send("Summarise Q3", context_id="ctx-1")
    assert status.state == "TASK_STATE_COMPLETED" and status.task_id == "task-1"
    sent = remote.requests[-1]
    assert sent.url.path == "/a2a/message:send"
    assert sent.headers["a2a-version"] == "1.0"
    assert sent.headers["authorization"].startswith("Bearer ")
    body = json.loads(sent.content)
    assert body["message"]["role"] == "ROLE_USER"
    assert body["message"]["parts"] == [{"text": "Summarise Q3"}]
    assert body["message"]["contextId"] == "ctx-1"
    token = jwt.decode(sent.headers["authorization"][7:], options={"verify_signature": False})
    assert token["aud"] == REMOTE  # the token is for the remote agent, not for AuthFI's gateway


def test_an_instance_presents_its_dpop_bound_token_to_the_remote_agent(chain: a2a_pki.Chain, tmp_path: Path) -> None:
    platform = InstancePlatform()
    path = tmp_path / "t"
    path.write_text(platform.k8s_token())
    remote = remote_agent(chain)

    def answer(request: httpx.Request) -> httpx.Response:
        scheme, _, raw = request.headers["authorization"].partition(" ")
        claims = jwt.decode(raw, options={"verify_signature": False})
        assert scheme == "DPoP"
        refusal = platform.check_dpop(request, claims["cnf"]["jkt"], access_token=raw)
        return refusal or json_response({"task": {"id": "t", "status": {"state": "TASK_STATE_WORKING"}}})

    remote.answer = answer
    env = {
        "AUTHFI_ISSUER": platform.issuer,
        "AUTHFI_AGENT_ID": AGENT,
        "AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE": str(path),
        "AUTHFI_CONFIG_VERSION": "7",
        "AUTHFI_CONFIG_DIGEST": "sha256:" + "ab" * 32,
    }

    def route(request: httpx.Request) -> httpx.Response:
        return remote(request) if request.url.host == "research.partner.example" else platform(request)

    with authfi.AuthFI(platform.issuer, http_client=httpx.Client(transport=httpx.MockTransport(route))) as project:
        agent = authfi.connect(env, http_client=project._transport.client)  # pyright: ignore[reportPrivateUsage]
        peer = a2a.remote(agent, REMOTE, trust_roots=[chain.root_pem])  # type: ignore[arg-type]
        assert peer.send("go").state == "TASK_STATE_WORKING"


async def test_the_asyncio_twin(chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]) -> None:
    platform, key = key_agent
    remote = remote_agent(chain)

    def route(request: httpx.Request) -> httpx.Response:
        return remote(request) if request.url.host == "research.partner.example" else platform(request)

    async with authfi.AsyncAuthFI(
        platform.issuer, http_client=httpx.AsyncClient(transport=httpx.MockTransport(route))
    ) as project:
        peer = await a2a.async_remote(project.agent(AGENT, key), REMOTE, trust_roots=[chain.root_pem])
        assert (await peer.send("hi")).state == "TASK_STATE_COMPLETED"


def test_json_rpc_is_spoken_when_that_is_all_the_card_offers(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    platform, key = key_agent
    interfaces = [{"url": REMOTE + "/rpc", "protocolBinding": "JSONRPC", "protocolVersion": "1.0"}]
    remote = remote_agent(chain, supportedInterfaces=interfaces)
    remote.answer = lambda _r: json_response(
        {
            "jsonrpc": "2.0",
            "id": "1",
            "result": {"message": {"messageId": "m", "role": "ROLE_AGENT", "parts": [{"text": "done"}]}},
        }
    )
    with project_for(platform, remote) as project:
        status = a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=[chain.root_pem]).send("x")
    assert status.state == "TASK_STATE_COMPLETED" and status.text == "done"
    rpc = json.loads(remote.requests[-1].content)
    assert rpc["method"] == "SendMessage" and rpc["params"]["message"]["role"] == "ROLE_USER"


def test_a_plain_key_card_verifies_with_the_key_you_name(
    key_agent: tuple[AgentPlatform, authfi.AgentKey],
) -> None:
    platform, key = key_agent
    signing = ed25519.Ed25519PrivateKey.generate()
    jwk = json.loads(OKPAlgorithm.to_jwk(signing.public_key()))
    remote = Remote(sign(card(), signing))
    with project_for(platform, remote) as project:
        peer = a2a.remote(project.agent(AGENT, key), REMOTE, keys={"agent-key-1": jwk})
        assert peer.card.verified and peer.card.signer == "agent-key-1"
        other = json.loads(OKPAlgorithm.to_jwk(ed25519.Ed25519PrivateKey.generate().public_key()))
        with pytest.raises(CardUnverified, match="does not match"):
            a2a.remote(project.agent(AGENT, key), REMOTE, keys={"agent-key-1": other})


# --- a card that cannot be trusted is refused ---------------------------------------------------------------------


def refuse(
    served: dict[str, Any], roots: list[bytes], key_agent: tuple[AgentPlatform, authfi.AgentKey], **options: Any
) -> pytest.ExceptionInfo[A2AError]:
    platform, key = key_agent
    remote = Remote(served)
    with project_for(platform, remote) as project, pytest.raises(A2AError) as caught:
        a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=roots, **options)
    assert not [r for r in remote.requests if r.method == "POST"]  # nothing was sent to it
    return caught


def test_an_unsigned_card_is_refused_unless_you_say_otherwise(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    caught = refuse(card(), [chain.root_pem], key_agent)
    assert isinstance(caught.value, CardUnverified) and "no signature" in str(caught.value)
    platform, key = key_agent
    with project_for(platform, Remote(card())) as project:
        peer = a2a.remote(project.agent(AGENT, key), REMOTE, allow_unsigned=True)
        assert not peer.card.verified


def test_a_card_changed_after_signing_is_refused(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    served = sign(card(), chain.agent_key, x5c=chain.x5c())
    served["supportedInterfaces"][0]["url"] = "https://evil.example/a2a"
    assert "does not match" in str(refuse(served, [chain.root_pem], key_agent).value)
    served = sign(card(), chain.agent_key, x5c=chain.x5c())
    served["skills"].append({"id": "x", "name": "exfiltrate", "description": "d", "tags": []})
    assert isinstance(refuse(served, [chain.root_pem], key_agent).value, CardUnverified)


@pytest.mark.parametrize(
    ("make", "says"),
    [
        (lambda: make_chain(agent_policies=("1",)), "agent-custody policy"),
        (lambda: make_chain(agent_policies=()), "agent-custody policy"),
        (lambda: make_chain(project_ca_is_ca=False), "each lead to the next"),
        (lambda: make_chain(agent_start=a2a_pki.NOW - dt.timedelta(days=200)), "not valid now"),
    ],
)
def test_a_chain_that_breaks_a_rule_is_refused(
    make: Callable[[], a2a_pki.Chain], says: str, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    broken = make()
    served = sign(card(), broken.agent_key, x5c=broken.x5c())
    assert says in str(refuse(served, [broken.root_pem], key_agent).value)


def test_a_chain_to_a_root_you_do_not_trust_is_refused(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    stranger = make_chain()
    served = sign(card(), chain.agent_key, x5c=chain.x5c())
    assert "root you trust" in str(refuse(served, [stranger.root_pem], key_agent).value)
    assert "trusted root are both" in str(refuse(served, [], key_agent).value)


def test_a_signature_by_another_key_than_the_certificate_is_refused(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    served = sign(card(), ed25519.Ed25519PrivateKey.generate(), x5c=chain.x5c())
    assert "does not match" in str(refuse(served, [chain.root_pem], key_agent).value)


def test_an_unknown_algorithm_or_kid_is_refused(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    served = sign(card(), chain.agent_key, x5c=chain.x5c())
    served["signatures"][0]["protected"] = reheader(served, alg="HS256")
    assert "algorithm" in str(refuse(served, [chain.root_pem], key_agent).value)
    served = sign(card(), chain.agent_key)  # no chain and no key named
    assert "no certificate chain" in str(refuse(served, [chain.root_pem], key_agent).value)


def test_a_card_that_points_elsewhere_gets_no_token(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    other = [{"url": "https://other.example/a2a", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}]
    served = sign(card(supportedInterfaces=other), chain.agent_key, x5c=chain.x5c())
    assert "allow_origins" in str(refuse(served, [chain.root_pem], key_agent).value)
    platform, key = key_agent
    with project_for(platform, Remote(served)) as project:
        peer = a2a.remote(
            project.agent(AGENT, key), REMOTE, trust_roots=[chain.root_pem], allow_origins=["https://other.example"]
        )
        assert peer.interface.url == "https://other.example/a2a"


def test_a_scheme_this_sdk_does_not_speak_is_named(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    served = sign(
        card(securitySchemes={"k": {"apiKeySecurityScheme": {"location": "header", "name": "X-Key"}}}),
        chain.agent_key,
        x5c=chain.x5c(),
    )
    assert "apiKeySecurityScheme" in str(refuse(served, [chain.root_pem], key_agent).value)


def test_a_card_with_no_interface_this_sdk_speaks_is_refused(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    grpc = [{"url": REMOTE, "protocolBinding": "GRPC", "protocolVersion": "1.0"}]
    served = sign(card(supportedInterfaces=grpc), chain.agent_key, x5c=chain.x5c())
    assert "GRPC" in str(refuse(served, [chain.root_pem], key_agent).value)


def test_a_missing_or_malformed_card_is_refused(key_agent: tuple[AgentPlatform, authfi.AgentKey]) -> None:
    assert "no name" in str(refuse({"name": "x"}, [], key_agent, allow_unsigned=True).value)


# --- answers --------------------------------------------------------------------------------------------------------


def call(
    chain: a2a_pki.Chain,
    key_agent: tuple[AgentPlatform, authfi.AgentKey],
    answer: Callable[[httpx.Request], httpx.Response],
) -> Any:
    platform, key = key_agent
    remote = remote_agent(chain)
    remote.answer = answer
    with project_for(platform, remote) as project:
        peer = a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=[chain.root_pem])
        return peer.send("x")


@pytest.mark.parametrize(
    ("state", "flag"),
    [
        ("TASK_STATE_INPUT_REQUIRED", "needs_input"),
        ("TASK_STATE_AUTH_REQUIRED", "needs_auth"),
        ("TASK_STATE_REJECTED", "rejected"),
    ],
)
def test_the_remote_agents_states_are_readable(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey], state: str, flag: str
) -> None:
    body = {"task": {"id": "t", "status": {"state": state, "message": {"parts": [{"text": "why"}]}}}}
    status = call(chain, key_agent, lambda _r: json_response(body))
    assert getattr(status, flag) and status.text == "why"


def test_a_refusal_unknown_state_and_garbage_are_errors_that_name_the_fix(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    with pytest.raises(A2AError, match="check the remote agent accepts"):
        call(
            chain, key_agent, lambda _r: httpx.Response(401, headers={"WWW-Authenticate": 'DPoP error="invalid_token"'})
        )
    with pytest.raises(A2AError, match="unknown task state"):
        call(chain, key_agent, lambda _r: json_response({"task": {"id": "t", "status": {"state": "DONE"}}}))
    with pytest.raises(A2AError, match="neither a task nor a message"):
        call(chain, key_agent, lambda _r: json_response({"nothing": 1}))
    with pytest.raises(A2AError, match="HTTP 500"):
        call(chain, key_agent, lambda _r: httpx.Response(500))


def test_a_server_nonce_is_retried_once(chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]) -> None:
    calls: list[int] = []

    def answer(_r: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(401, headers={"DPoP-Nonce": "n-1"})
        return json_response({"task": {"id": "t", "status": {"state": "TASK_STATE_WORKING"}}})

    assert call(chain, key_agent, answer).state == "TASK_STATE_WORKING"
    assert len(calls) == 2


def test_the_token_is_not_in_a_repr_or_an_error(
    chain: a2a_pki.Chain, key_agent: tuple[AgentPlatform, authfi.AgentKey]
) -> None:
    platform, key = key_agent
    remote = remote_agent(chain)
    remote.answer = lambda _r: httpx.Response(403)
    with project_for(platform, remote) as project:
        peer = a2a.remote(project.agent(AGENT, key), REMOTE, trust_roots=[chain.root_pem])
        with pytest.raises(A2AError) as caught:
            peer.send("x")
        token = remote.requests[-1].headers["authorization"].split(" ")[1]
    assert token not in repr(peer) + str(caught.value) + repr(peer.card)


# --- RFC 8785 ---------------------------------------------------------------------------------------------------------


def test_canonical_json_sorts_by_utf16_and_escapes_minimally() -> None:
    value = {
        "€": "Euro Sign",
        "\r": "Carriage Return",
        "דּ": "Hebrew Letter Dalet With Dagesh",
        "1": "One",
        "\U0001f600": "Emoji: Grinning Face",
        "\x80": "Control",
        "ö": "Latin Small Letter O With Diaeresis",
    }
    keys = list(json.loads(canonical_json(value).decode()))
    assert keys == ["\r", "1", "\x80", "ö", "€", "\U0001f600", "דּ"]
    assert canonical_json({"b": [True, None, 1], "a": "x\ny\u001f"}) == b'{"a":"x\\ny\\u001f","b":[true,null,1]}'
    assert canonical_json({"é": "ü"}) == '{"é":"ü"}'.encode()
    with pytest.raises(ValueError, match="integer"):
        canonical_json({"n": 1.5})


def test_canonical_json_is_the_same_for_any_member_order() -> None:
    one = card()
    two = dict(reversed(list(copy.deepcopy(one).items())))
    assert canonical_json(one) == canonical_json(two)


# --- decisions as task states ------------------------------------------------------------------------------


def test_each_decision_has_its_state_and_a_reason() -> None:
    assert status_for("allow").state == "TASK_STATE_WORKING"
    held = status_for("needs_approval", "Refund over 500")
    assert held.state == "TASK_STATE_INPUT_REQUIRED" and held.text == "Refund over 500"
    denied = status_for("deny", "no grant for send_email")
    assert denied.state == "TASK_STATE_REJECTED" and denied.text == "no grant for send_email"
    assert status_for("deny", code="consent_required").state == AUTH_REQUIRED
    assert status_for("needs_credentials").state == AUTH_REQUIRED
    unknown = status_for("maybe")
    assert unknown.state == "TASK_STATE_FAILED"  # an answer nobody knows is no yes


def test_gateway_outcomes_map_and_never_carry_an_approval_id() -> None:
    assert status_for_outcome(ToolAllowed("x")).state == "TASK_STATE_WORKING"
    held = status_for_outcome(ToolApprovalRequired("apr_secret", "approval_required"))
    assert held.state == "TASK_STATE_INPUT_REQUIRED"
    assert "apr_secret" not in json.dumps(held.to_status())
    refused = status_for_outcome(ToolRefused(code="no_grant", reason="no grant"))
    assert refused.state == "TASK_STATE_REJECTED" and refused.text == "no grant"
    consent = status_for_outcome(ToolRefused(code="credentials_required"))
    assert consent.state == AUTH_REQUIRED
    shape = refused.to_status()
    assert shape["state"] == "TASK_STATE_REJECTED"
    assert shape["message"]["role"] == "ROLE_AGENT" and shape["message"]["parts"] == [{"text": "no grant"}]
