"""DPoP on every gateway call, acting for a person, tasks and cloud credentials, against a real-HTTP gateway.

The gateway (:mod:`tests.fake_agent_gateway`) checks proofs, bindings, templates and roles from the bytes it receives.
Each capability has its refusal: a stale or missing proof, a nonce challenge and its single retry, a denied or expired
approval, a wrong audience, an action outside the task, and a cloud role that is not granted.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import uuid
from collections.abc import Iterator
from typing import Any

import httpx
import jwt
import pytest

from authfi import (
    AgentKey,
    APIError,
    AsyncAuthFI,
    AsyncGateway,
    AsyncTokenSource,
    AuthFI,
    ConfigurationError,
    ConsentDenied,
    ConsentExpired,
    Gateway,
    GatewayTokenRefused,
    OAuthError,
    Secret,
    Task,
    TaskPending,
    ToolRefused,
    WaitTimeout,
    _dpop,
)
from authfi._protocol.oauth import b64url
from authfi.agents import AgentAuth, AsyncAgentAuth
from tests.fake_agent_gateway import CLOUD_PATH, OUTSIDE, AgentGateway, Template, thumbprint
from tests.fake_gateway import ALLOWED, STEP_UP
from tests.support import FakeClock

AGENT = "agt_caps1"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture(autouse=True)
def _fresh_nonces() -> Iterator[None]:
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]
    yield
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def gw() -> Iterator[AgentGateway]:
    with AgentGateway() as gateway:
        yield gateway


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(wall=time.time())


def person() -> Secret:
    """The signed-in person's access token, as an app holds it."""
    return Secret(jwt.encode({"sub": "usr_1", "exp": int(time.time()) + 3600}, "k" * 32, algorithm="HS256"))


def rpc(method: str, **params: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


def call(client: httpx.Client, gateway: AgentGateway, tool: str, **args: Any) -> Any:
    response = client.post(gateway.mcp_url, json=rpc("tools/call", name=tool, arguments=args))
    response.raise_for_status()
    return response.json()["result"]


def mcp_proofs(gateway: AgentGateway) -> list[dict[str, Any]]:
    return [p for p in gateway.proofs if p["path"] == "/mcp"]


def ath(token: str) -> str:
    return b64url(hashlib.sha256(token.encode()).digest())


# --- DPoP: bound at mint, proved on every call --------------------------------------------------------


def test_the_token_is_bound_at_mint_and_every_gateway_call_proves_it(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        token = agent.token()
        client = agent.gateway().http_client()
        for _ in range(2):
            assert client.post(gw.mcp_url, json=rpc("tools/list")).status_code == 200

    mint = gw.proofs[0]
    assert (mint["path"], mint["method"]) == ("/oauth/token", "POST")
    assert "ath" not in mint["claims"]  # no token yet
    bound = jwt.decode(token, options={"verify_signature": False})["cnf"]["jkt"]
    assert bound == thumbprint(mint["jwk"]) == _dpop.run_key().thumbprint
    sent = mcp_proofs(gw)
    assert len(sent) == 2
    for proof in sent:
        assert proof["claims"]["htm"] == "POST" and proof["claims"]["ath"] == ath(token)
        assert thumbprint(proof["jwk"]) == bound
    assert sent[0]["claims"]["jti"] != sent[1]["claims"]["jti"]  # a new proof per request
    assert all(r["headers"]["authorization"].startswith("DPoP ") for r in gw.requests if r["path"] == "/mcp")


def test_the_run_key_is_ed25519_unless_the_project_lists_only_p256(key: AgentKey) -> None:
    with AgentGateway(dpop_algs=("ES256",)) as gateway, AuthFI(gateway.issuer) as af:
        af.agent(AGENT, key).token()
    assert gateway.proofs[0]["alg"] == "ES256"
    with AgentGateway() as gateway, AuthFI(gateway.issuer) as af:
        af.agent(AGENT, key).token()
    assert gateway.proofs[0]["alg"] == "EdDSA"


def test_a_token_the_platform_left_unbound_is_sent_as_bearer(key: AgentKey) -> None:
    with AgentGateway(bind=False) as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, key)
        client = agent.gateway().http_client()
        assert client.post(gateway.mcp_url, json=rpc("tools/list")).status_code == 200
        assert agent.bound_key() is None
    assert all(r["headers"]["authorization"].startswith("Bearer ") for r in gateway.requests if r["path"] == "/mcp")
    assert "dpop" not in gateway.requests[-1]["headers"]


def test_a_project_that_accepts_no_algorithm_we_hold_gets_no_proof(key: AgentKey) -> None:
    with AgentGateway(dpop_algs=("PS256",)) as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, key)
        assert agent.gateway().http_client().post(gateway.mcp_url, json=rpc("tools/list")).status_code == 200
    assert gateway.proofs == []


def test_a_token_bound_to_a_key_this_process_does_not_hold_is_refused() -> None:
    from authfi._protocol.oauth import TokenResponse

    token = jwt.encode({"cnf": {"jkt": "someone-elses-key"}}, "k" * 32, algorithm="HS256")
    response = TokenResponse(Secret(token), "DPoP", 900)
    for key in (_dpop.run_key(), None):  # a key that is not the named one, and no proof sent at all
        with pytest.raises(OAuthError) as caught:
            _dpop.bound_key(response, key)
        assert caught.value.code == "dpop_binding_mismatch"
    assert _dpop.bound_key(TokenResponse(Secret(token), "Bearer", 900), _dpop.run_key()) is None


def test_a_stale_proof_is_refused_and_says_so(gw: AgentGateway, key: AgentKey) -> None:
    behind = FakeClock(wall=time.time() - 3600)
    with AuthFI(gw.issuer, clock=behind) as af, pytest.raises(OAuthError) as caught:
        af.agent(AGENT, key).token()
    assert caught.value.code == "invalid_dpop_proof"
    assert gw.refused_proofs == ["dpop_proof_iat_invalid"]


def test_a_bound_token_without_its_proof_is_refused_by_the_gateway(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        token = agent.token()
        denied = httpx.post(gw.mcp_url, json=rpc("tools/list"), headers={"Authorization": f"Bearer {token}"})
    assert denied.status_code == 401
    assert "token_dpop_proof_required" in gw.refused_proofs
    assert "DPoP" in denied.headers["www-authenticate"]


def test_a_replayed_proof_is_refused(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        headers = agent.headers("POST", gw.mcp_url)
        assert httpx.post(gw.mcp_url, json=rpc("tools/list"), headers=headers).status_code == 200
        assert httpx.post(gw.mcp_url, json=rpc("tools/list"), headers=headers).status_code == 401
    assert gw.refused_proofs[-1] == "dpop_proof_replayed"


def test_a_proof_for_another_url_is_refused(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        headers = agent.headers("POST", gw.issuer + "/mcp/v1/messages")
        assert httpx.post(gw.mcp_url, json=rpc("tools/list"), headers=headers).status_code == 401
    assert gw.refused_proofs[-1] == "dpop_proof_htu_mismatch"


def test_a_token_for_a_customers_api_is_not_bound(gw: AgentGateway, key: AgentKey) -> None:
    """The customer's API verifies it, and a verifier built on this SDK takes no proof: only AuthFI's own is bound."""
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        orders = agent.for_resource("https://orders.example/api", send_to=["https://orders.example"])
        headers = orders.headers("GET", "https://orders.example/api/orders")
        assert (
            orders.retry_headers("GET", "https://orders.example/api/orders", 401, 'DPoP error="use_dpop_nonce"', "n")
            is None
        )
    assert headers["Authorization"].startswith("Bearer ") and "DPoP" not in headers
    assert [p["path"] for p in gw.proofs] == []  # not even a proof was sent to ask for a binding


def test_the_credential_goes_only_to_authfi() -> None:
    with AgentGateway() as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, AgentKey.generate(kid="k"))
        with pytest.raises(ConfigurationError) as caught:
            agent.headers("POST", "https://evil.example/mcp")
    assert caught.value.code == "destination_not_allowed"
    assert gateway.minted == []  # refused before a token was asked for


# --- DPoP: the server's nonce ---------------------------------------------------------------------------------


def test_a_nonce_challenge_at_the_token_endpoint_is_answered_once(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer) as af:
        assert af.agent(AGENT, key).token()
    assert gateway.nonce_challenges == 1
    assert [p["claims"].get("nonce") for p in gateway.proofs] == ["nonce-1"]
    asked = [r for r in gateway.requests if r["path"] == "/oauth/token"]
    assert len(asked) == 2  # the challenge, then the retry with the nonce


def test_a_nonce_challenge_at_the_gateway_is_answered_once_and_remembered(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        assert client.post(gateway.mcp_url, json=rpc("tools/list")).status_code == 200  # nonce learned at mint
        assert gateway.nonce_challenges == 1
        gateway._nonce = "nonce-rotated"
        before = len([r for r in gateway.requests if r["path"] == "/mcp"])
        assert client.post(gateway.mcp_url, json=rpc("tools/list")).status_code == 200
        assert gateway.nonce_challenges == 2
        assert len([r for r in gateway.requests if r["path"] == "/mcp"]) - before == 2  # asked, then answered
        assert mcp_proofs(gateway)[-1]["claims"]["nonce"] == "nonce-rotated"
        assert client.post(gateway.mcp_url, json=rpc("tools/list")).status_code == 200
        assert gateway.nonce_challenges == 2  # the new nonce is the one it sends now


def test_a_server_that_keeps_challenging_is_asked_once_not_forever(key: AgentKey) -> None:
    with (
        AgentGateway(always_challenge=True) as gateway,
        AuthFI(gateway.issuer) as af,
        pytest.raises(OAuthError) as caught,
    ):
        af.agent(AGENT, key).token()
    assert caught.value.code == "use_dpop_nonce"
    assert len([r for r in gateway.requests if r["path"] == "/oauth/token"]) == 2


async def test_an_mcp_client_gets_its_nonce_answered_not_read_as_a_refusal(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway:
        async with AsyncAuthFI(gateway.issuer) as af:
            agent = af.agent(AGENT, key)
            await agent.token()
            gateway._nonce = "nonce-rotated"
            client = agent.gateway().mcp_client_factory(httpx)()
            async with client:
                response = await client.post(gateway.mcp_url, json=rpc("tools/list"))
            assert response.status_code == 200 and "result" in response.json()
            # Asked again for good: the second 401 is the token being refused, as a JSON-RPC error.
            gateway.always_challenge = True
            async with agent.gateway().mcp_client_factory(httpx)() as refused:
                answer = await refused.post(gateway.mcp_url, json=rpc("tools/list"))
            assert answer.json()["error"]["code"] == -32001


# --- cloud credentials -------------------------------------------------------------------------------------------


def test_cloud_credentials_are_short_lived_and_proved(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        outcome = af.agent(AGENT, key).cloud.credentials("prod", "deployer", 300)
    assert outcome.kind == "issued" and outcome.credential is not None
    assert outcome.credential.access_key_id == "ASIAFAKE"
    assert "fake-secret" not in repr(outcome.credential)
    assert gw.cloud_calls[-1]["request"] == {"account": "prod", "role": "deployer", "duration_seconds": 300}
    proof = [p for p in gw.proofs if p["path"] == CLOUD_PATH][-1]
    assert proof["claims"]["htm"] == "POST" and "ath" in proof["claims"]


def test_a_cloud_role_that_is_not_granted_is_refused_with_the_platforms_code(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af, pytest.raises(APIError) as caught:
        af.agent(AGENT, key).cloud.credentials("prod", "root")
    assert caught.value.code == "ECLOUDCRED003" and caught.value.status == 403


def test_cloud_credentials_that_need_a_person_wait_and_are_sent_again(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        held = agent.cloud.credentials("prod", "admin")
        assert held.kind == "approval_required" and held.approval_id
        again = agent.cloud.credentials("prod", "admin", approval_id=held.approval_id)
        assert again.kind == "approval_required"  # still pending
        gw.decide_cloud(held.approval_id, "approved")
        done = agent.cloud.credentials("prod", "admin", approval_id=held.approval_id)
        assert done.kind == "issued"
        gw.decide_cloud(held.approval_id, "denied")
        with pytest.raises(APIError) as caught:
            agent.cloud.credentials("prod", "admin", approval_id=held.approval_id)
        assert caught.value.code == "ECLOUDCRED007"


def test_the_cloud_door_answers_a_nonce_challenge_once(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, key)
        agent.token()
        gateway._nonce = "nonce-rotated"
        assert agent.cloud.credentials("prod", "deployer").kind == "issued"
    assert gateway.nonce_challenges == 2
    assert len(gateway.cloud_calls) == 1


def test_a_hand_made_credential_for_the_cloud_door_without_its_proof_is_refused(
    gw: AgentGateway, key: AgentKey
) -> None:
    from authfi import TokenSource

    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)

        class BearerOnly(TokenSource):
            @property
            def send_to(self) -> tuple[str, ...]:
                return agent.send_to

            def token(self, *, force: bool = False) -> str:
                return agent.token()

        with pytest.raises(GatewayTokenRefused) as caught:
            Gateway(af, BearerOnly()).cloud_credentials("prod", "deployer")
    assert "DPoP" in (caught.value.challenge or "")


def test_acting_for_a_person_is_not_their_approval_of_the_cloud(
    gw: AgentGateway, key: AgentKey, clock: FakeClock
) -> None:
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        delegated = agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
        with pytest.raises(APIError) as caught:
            Gateway(af, delegated).cloud_credentials("prod", "deployer")
        assert caught.value.code == "ECLOUDCRED011"
        gw.person_scope = "send_email mcp:cloud.credentials"
        approved = agent.token_as_user(
            "priya@acme.com", binding_message="Deploy the fix", scope=["mcp:cloud.credentials"]
        )
        assert Gateway(af, approved).cloud_credentials("prod", "deployer").kind == "issued"
    assert [c["kind"] for c in gw.cloud_calls] == ["person", "person"]


async def test_cloud_credentials_in_asyncio_code(gw: AgentGateway, key: AgentKey) -> None:
    async with AsyncAuthFI(gw.issuer) as af:
        outcome = await af.agent(AGENT, key).cloud.credentials("prod", "deployer")
    assert outcome.kind == "issued"


# --- acting for a person ----------------------------------------------------------------------------------------------


def test_a_person_approves_and_the_agent_acts_for_them_with_a_bound_token(
    gw: AgentGateway, key: AgentKey, clock: FakeClock
) -> None:
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        request = agent.request_user_token("priya@acme.com", binding_message="Refund order 4411")
        assert gw.requests_for_person[-1]["binding_message"] == "Refund order 4411"
        delegated = agent.await_user_token(request)
        assert delegated.subject.id == "usr_person" and delegated.actor.id == AGENT
        assert delegated.bound_key() is not None
        client = Gateway(af, delegated).http_client()
        assert client.post(gw.mcp_url, json=rpc("tools/list")).status_code == 200
    assert gw.requests[-1]["headers"]["authorization"].startswith("DPoP ")
    assert thumbprint(mcp_proofs(gw)[-1]["jwk"]) == _dpop.run_key().thumbprint


@pytest.mark.parametrize(("answer", "error"), [("denied", ConsentDenied), ("expired", ConsentExpired)])
def test_a_person_who_refuses_or_lets_it_expire_ends_the_wait(
    gw: AgentGateway, key: AgentKey, clock: FakeClock, answer: str, error: type[Exception]
) -> None:
    gw.ciba_answers = ["pending", answer]
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        request = agent.request_user_token("priya@acme.com", binding_message="Refund order 4411")
        with pytest.raises(error):
            agent.await_user_token(request)


def test_a_person_who_has_not_decided_is_a_timeout_not_a_refusal(
    gw: AgentGateway, key: AgentKey, clock: FakeClock
) -> None:
    gw.ciba_answers = ["pending"] * 10
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        request = agent.request_user_token("priya@acme.com", binding_message="Refund order 4411")
        with pytest.raises(WaitTimeout):
            agent.await_user_token(request, timeout=2.0)


def test_exchanging_a_persons_token_for_the_gateway(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        delegated = agent.exchange(person())
        claims = jwt.decode(delegated.token(), options={"verify_signature": False})
        assert claims["aud"] == gw.issuer + "/mcp" and claims["act"]["sub"] == AGENT
        assert delegated.bound_key() is not None
        assert Gateway(af, delegated).http_client().post(gw.mcp_url, json=rpc("tools/list")).status_code == 200
    form = gw.exchanges[-1]
    assert "client_assertion" not in form and "client_id" not in form  # the actor token is the agent's credential


def test_an_exchange_for_the_wrong_audience_is_refused(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af, pytest.raises(OAuthError) as caught:
        af.agent(AGENT, key).exchange(
            person(), resource="https://orders.example/api", send_to=["https://orders.example"]
        )
    assert caught.value.code == "invalid_target"


async def test_acting_for_a_person_in_asyncio_code(gw: AgentGateway, key: AgentKey, clock: FakeClock) -> None:
    async with AsyncAuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        delegated = await agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
        headers = await delegated.headers("POST", gw.mcp_url)
        assert headers["Authorization"].startswith("DPoP ") and "DPoP" in headers
        other = await agent.exchange(person())
        assert (await other.headers("POST", gw.mcp_url))["Authorization"].startswith("DPoP ")


# --- tasks ----------------------------------------------------------------------------------------------------


def start(agent: AgentAuth, template: str = "support.refund", **kwargs: Any) -> Task:
    return agent.task(template, subject_token=person(), params={"order": "4411"}, **kwargs)


def test_a_task_is_a_credential_that_holds_calls_to_its_tools(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        with start(agent) as task:
            assert task.tools == (ALLOWED, STEP_UP) and task.template == "support.refund"
            held = task.gateway().http_client()
            assert call(held, gw, ALLOWED, repoName="a/b")["isError"] is False
            token = gw.requests[-1]["headers"]["authorization"].split(" ", 1)[1]
            assert gw.issued[token]["kind"] == "task" and gw.issued[token]["mandate_id"] == task.mandate_id
            assert thumbprint(mcp_proofs(gw)[-1]["jwk"]) == _dpop.run_key().thumbprint == task.bound_key().thumbprint
            # Nothing swaps the agent's own token for the task's: the agent's client is still the agent's.
            call(agent.gateway().http_client(), gw, ALLOWED, repoName="a/b")
            own = gw.requests[-1]["headers"]["authorization"].split(" ", 1)[1]
            assert gw.issued[own]["kind"] == "agent"
    assert gw.task_requests == [{"type": "authfi_mandate", "template": "support.refund", "params": {"order": "4411"}}]


def test_leaving_the_with_forgets_the_credential(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        with start(agent) as task:
            assert task.token()
        for use in (lambda: task.token(), lambda: task.headers("POST", gw.mcp_url)):
            with pytest.raises(ConfigurationError, match="closed"):
                use()


def test_a_call_outside_the_task_is_an_answer_the_agent_can_plan_around(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        with start(agent) as task:
            client = task.gateway().http_client()
            outcome = task.gateway().read_result(call(client, gw, OUTSIDE, repoName="a/b"))
    assert isinstance(outcome, ToolRefused)
    assert (outcome.code, outcome.why) == ("EMDT005", "outside_task")
    answer = outcome.as_dict()
    assert answer["refused"] is True and answer["next_step"] == outcome.next_step
    assert outcome.next_step is not None
    assert "outside the task" in outcome.model_text and outcome.next_step in outcome.model_text
    for text in (outcome.model_text, json.dumps(answer)):
        assert "new task" not in text.lower() and "another task" not in text.lower()


def test_a_call_over_the_tasks_budget_is_refused_as_over_budget(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        with start(agent) as task:
            client = task.gateway().http_client()
            ok = task.gateway().read_result(call(client, gw, ALLOWED, repoName="a/b", amount=60))
            over = task.gateway().read_result(call(client, gw, ALLOWED, repoName="a/b", amount=60))
    assert ok.kind == "allowed"
    assert isinstance(over, ToolRefused) and (over.code, over.why) == ("EMDT006", "over_budget")


def test_an_ended_task_refuses_its_credential_outright(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        with start(agent) as task:
            gw.end_task(task.mandate_id)
            answer = task.gateway().http_client().post(gw.mcp_url, json=rpc("tools/list"))
            assert answer.status_code == 401 and "EMDT004" in answer.headers["www-authenticate"]
            with pytest.raises(GatewayTokenRefused):
                task.cloud.credentials("prod", "deployer")


def test_a_task_is_one_level_deep(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        with start(agent):
            with pytest.raises(ConfigurationError) as caught:
                start(agent)
            assert caught.value.code == "task_nested"
            with pytest.raises(ConfigurationError) as caught:
                agent.request_task("support.refund", subject_token=person())
            assert caught.value.code == "task_nested"
        assert start(agent).mandate_id  # one at a time is fine
    assert len(gw.task_requests) == 2  # the refused ones never reached the platform


def test_a_task_holds_cloud_credentials_too(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        with start(agent) as task:
            assert task.cloud.credentials("prod", "deployer").kind == "issued"
        agent.cloud.credentials("prod", "deployer")
    inside, outside = gw.cloud_calls
    assert (inside["kind"], inside["mandate_id"]) == ("task", task.mandate_id)
    assert outside["kind"] == "agent"


def test_a_task_a_person_must_approve_waits_for_them(gw: AgentGateway, key: AgentKey, clock: FakeClock) -> None:
    gw.templates["support.big"] = Template([ALLOWED], approval="person")
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        pending = agent.request_task("support.big", subject_token=person(), params={})
        assert isinstance(pending, TaskPending)
        assert pending.approval_url == f"{gw.issuer}/approve/task-req-1" and "task-req-1" not in repr(pending)
        gw.decide_task("task-req-1", "approved")
        task = agent.await_task(pending)
        assert task.mandate_id == pending.mandate_id and task.tools == (ALLOWED,)
        assert thumbprint(gw.proofs[-1]["jwk"]) == task.bound_key().thumbprint  # the poll proved the same key


@pytest.mark.parametrize(("decision", "error"), [("denied", ConsentDenied), ("expired", ConsentExpired)])
def test_a_person_who_refuses_a_task_or_lets_it_expire_ends_the_wait(
    gw: AgentGateway, key: AgentKey, clock: FakeClock, decision: str, error: type[Exception]
) -> None:
    gw.templates["support.big"] = Template([ALLOWED], approval="person")
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        pending = agent.request_task("support.big", subject_token=person(), params={})
        assert isinstance(pending, TaskPending)
        gw.decide_task("task-req-1", decision)
        with pytest.raises(error):
            agent.await_task(pending)


def test_a_task_nobody_approved_in_time_is_a_timeout_and_stays_open(
    gw: AgentGateway, key: AgentKey, clock: FakeClock
) -> None:
    gw.templates["support.big"] = Template([ALLOWED], approval="person")
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        pending = agent.request_task("support.big", subject_token=person(), params={})
        assert isinstance(pending, TaskPending)
        with pytest.raises(WaitTimeout):
            agent.await_task(pending, timeout=0.5)
        gw.decide_task("task-req-1", "approved")
        assert agent.await_task(pending).tools == (ALLOWED,)  # waiting again continues it


def test_task_with_a_person_waits_inside_one_call(gw: AgentGateway, key: AgentKey, clock: FakeClock) -> None:
    gw.templates["support.big"] = Template([ALLOWED], approval="person")
    with AuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        original = clock.sleep

        def approve_while_sleeping(seconds: float) -> None:
            gw.decide_task("task-req-1", "approved")
            original(seconds)

        clock.sleep = approve_while_sleeping  # type: ignore[method-assign]
        with agent.task("support.big", subject_token=person(), params={}) as task:
            assert task.tools == (ALLOWED,)


def test_a_template_the_agent_may_not_use_is_refused(gw: AgentGateway, key: AgentKey) -> None:
    gw.templates["support.other"] = Template([ALLOWED], agents=["agt_somebody_else"])
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        for template in ("nope", "support.other"):
            with pytest.raises(OAuthError) as caught:
                start(agent, template)
            assert caught.value.code == "invalid_authorization_details"


def test_a_task_the_platform_would_not_bind_to_our_key_is_not_usable(key: AgentKey) -> None:
    with AgentGateway(bind=False) as gateway, AuthFI(gateway.issuer) as af, pytest.raises(OAuthError) as caught:
        start(af.agent(AGENT, key))
    assert caught.value.code == "invalid_response"


def test_the_parameters_of_a_task_are_json_or_refused_before_anything_is_sent(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        with pytest.raises(ConfigurationError):
            agent.task("support.refund", subject_token=person(), params={"order": object()})
        with pytest.raises(ConfigurationError):
            agent.task("support.refund", subject_token=person(), ttl_seconds=0)
        with pytest.raises(ConfigurationError):
            agent.task(" ", subject_token=person())
    assert gw.task_requests == []


def test_the_actor_assertion_names_the_key_that_holds_the_task(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        start(af.agent(AGENT, key))
    actor = jwt.decode(gw.exchanges[-1]["actor_token"], options={"verify_signature": False, "verify_aud": False})
    assert actor["cnf"] == {"jkt": _dpop.run_key().thumbprint}
    assert actor["aud"] == gw.issuer + "/oauth/token" and actor["iss"] == actor["sub"] == AGENT


def test_a_tasks_credential_is_sent_only_to_authfi_and_never_pickled(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        task = start(af.agent(AGENT, key))
        with pytest.raises(ConfigurationError) as caught:
            task.headers("POST", "https://evil.example/mcp")
        assert caught.value.code == "destination_not_allowed"
        with pytest.raises(TypeError):
            pickle.dumps(task)
        assert task.token() not in repr(task)


def test_a_task_that_has_run_out_says_only_a_person_can_give_another(
    gw: AgentGateway, key: AgentKey, clock: FakeClock
) -> None:
    with AuthFI(gw.issuer, clock=clock) as af:
        task = start(af.agent(AGENT, key))
        clock.advance(601)
        with pytest.raises(ConsentExpired):
            task.headers("POST", gw.mcp_url)


async def test_tasks_in_asyncio_code(gw: AgentGateway, key: AgentKey) -> None:
    async with AsyncAuthFI(gw.issuer) as af:
        agent: AsyncAgentAuth = af.agent(AGENT, key)
        async with await agent.task("support.refund", subject_token=person(), params={"order": "4411"}) as task:
            inside = (await task.headers("POST", gw.mcp_url))["Authorization"].split(" ", 1)[1]
            assert inside == await task.token() and gw.issued[inside]["kind"] == "task"
            with pytest.raises(ConfigurationError):
                await agent.task("support.refund", subject_token=person())
            outcome = await task.cloud.credentials("prod", "deployer")
            assert outcome.kind == "issued" and gw.cloud_calls[-1]["kind"] == "task"
        outside = (await agent.headers("POST", gw.mcp_url))["Authorization"].split(" ", 1)[1]
        assert outside != inside and gw.issued[outside]["kind"] == "agent"
        with pytest.raises(ConfigurationError, match="closed"):
            await task.token()


async def test_an_async_task_waits_for_a_person(gw: AgentGateway, key: AgentKey, clock: FakeClock) -> None:
    gw.templates["support.big"] = Template([ALLOWED], approval="person")
    async with AsyncAuthFI(gw.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        pending = await agent.request_task("support.big", subject_token=person(), params={})
        assert isinstance(pending, TaskPending)
        gw.decide_task("task-req-1", "denied")
        with pytest.raises(ConsentDenied):
            await agent.await_task(pending)


# --- every credential answers the nonce the same way ----------------------------------------------------------------


def rotated(gateway: AgentGateway) -> None:
    gateway._nonce = "nonce-" + uuid.uuid4().hex  # pyright: ignore[reportPrivateUsage]


def test_a_task_answers_a_nonce_challenge_once(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, key)
        with start(agent) as task:
            client = task.gateway().http_client()
            rotated(gateway)
            challenges = gateway.nonce_challenges
            assert client.post(gateway.mcp_url, json=rpc("tools/list")).status_code == 200
            assert gateway.nonce_challenges == challenges + 1
            assert task.retry_headers("POST", gateway.mcp_url, 200, None, None) is None  # not a challenge
            assert task.expires_at.timestamp() > time.time()
            assert task.gateway() is task.gateway()
        assert "mdt_" in repr(task) and task.token.__self__ is task  # type: ignore[attr-defined]


def test_a_person_acting_token_answers_a_nonce_challenge_once(key: AgentKey, clock: FakeClock) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer, clock=clock) as af:
        agent = af.agent(AGENT, key)
        delegated = agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
        client = Gateway(af, delegated).http_client()
        rotated(gateway)
        assert client.post(gateway.mcp_url, json=rpc("tools/list")).status_code == 200
        assert delegated.retry_headers("POST", gateway.mcp_url, 200, None, None) is None


def test_a_token_for_aufis_own_resource_is_bound_and_answers_a_nonce(key: AgentKey) -> None:
    with AgentGateway(nonce=True) as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent(AGENT, key)
        own = agent.for_resource(gateway.issuer + "/mcp")
        assert own.headers("POST", gateway.mcp_url)["Authorization"].startswith("DPoP ")
        rotated(gateway)
        client = Gateway(af, own).http_client()
        assert client.post(gateway.mcp_url, json=rpc("tools/list")).status_code == 200
        assert own.retry_headers("POST", gateway.mcp_url, 200, None, None) is None


async def test_the_asyncio_credentials_answer_a_nonce_challenge_once(key: AgentKey, clock: FakeClock) -> None:
    with AgentGateway(nonce=True) as gateway:
        async with AsyncAuthFI(gateway.issuer, clock=clock) as af:
            agent = af.agent(AGENT, key)
            own = agent.for_resource(gateway.issuer + "/mcp")
            delegated = await agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
            async with await agent.task("support.refund", subject_token=person(), params={}) as task:
                credentials: list[tuple[AsyncTokenSource, httpx.AsyncClient]] = [
                    (agent, AsyncGateway(af, agent).http_client()),
                    (own, AsyncGateway(af, own).http_client()),
                    (delegated, AsyncGateway(af, delegated).http_client()),
                    (task, task.gateway().http_client()),
                ]
                for credential, client in credentials:
                    rotated(gateway)
                    challenges = gateway.nonce_challenges
                    response = await client.post(gateway.mcp_url, json=rpc("tools/list"))
                    assert response.status_code == 200, credential
                    assert gateway.nonce_challenges == challenges + 1, credential
                    assert await credential.retry_headers("POST", gateway.mcp_url, 200, None, None) is None
                assert delegated.bound_key() is not None and task.expires_at.timestamp() > time.time()
                assert task.gateway() is task.gateway() and "mdt_" in repr(task)


async def test_a_task_can_be_asked_for_with_an_asyncio_credential_and_waits_for_the_person(
    key: AgentKey, clock: FakeClock
) -> None:
    class Session(AsyncTokenSource):
        @property
        def send_to(self) -> tuple[str, ...]:
            return ()

        async def token(self, *, force: bool = False) -> str:
            return person().reveal()

    with AgentGateway() as gateway:
        gateway.templates["support.big"] = Template([ALLOWED], approval="person")
        async with AsyncAuthFI(gateway.issuer, clock=clock) as af:
            agent = af.agent(AGENT, key)
            original = clock.asleep

            async def approve_while_sleeping(seconds: float) -> None:
                gateway.decide_task("task-req-1", "approved")
                await original(seconds)

            clock.asleep = approve_while_sleeping  # type: ignore[method-assign]
            task = await agent.task("support.big", subject_token=Session(), params={})
            assert task.tools == (ALLOWED,)
            with pytest.raises(ConfigurationError):
                await agent.task("support.big", subject_token="a plain string")  # type: ignore[arg-type]


def test_a_project_that_accepts_no_key_we_can_make_cannot_hold_a_task(key: AgentKey) -> None:
    from authfi import NotSupportedError

    with AgentGateway(dpop_algs=("PS256",)) as gateway, AuthFI(gateway.issuer) as af:
        with pytest.raises(NotSupportedError) as caught:
            start(af.agent(AGENT, key))
    assert caught.value.feature == "dpop" and gateway.task_requests == []
