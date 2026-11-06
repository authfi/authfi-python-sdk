"""The AI gateway is its own door, ``<issuer>/llm``, and takes only a token requested for it.

The tool door ``<issuer>/mcp`` refuses a token minted for ``<issuer>/llm`` and the model door refuses the reverse, so an
agent that calls models holds two tokens on one key (one run). The gateway stand-in refuses a token at the wrong door as
the platform does, so each test below fails if the SDK sends the wrong token, and a control proves the stand-in refuses.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import httpx
import jwt
import pytest

from authfi import (
    AgentKey,
    AsyncAuthFI,
    AuthFI,
    ConfigurationError,
    Secret,
    TokenSource,
)
from authfi.gateway import MODEL_PATH_PREFIX, Gateway
from authfi.redteam import _PATHS  # pyright: ignore[reportPrivateUsage]
from tests.fake_agent_gateway import AgentGateway
from tests.support import FakeClock, Platform, form_of, json_response
from tests.unit.test_agent_capabilities import start

AGENT = "agt_doors"
MODEL = {"model": "claude-haiku-5-5", "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def agent_clock() -> FakeClock:
    return FakeClock(wall=time.time())


@pytest.fixture
def gw() -> Iterator[AgentGateway]:
    with AgentGateway() as gateway:
        yield gateway


def rpc(method: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": 1, "method": method}


def audience_of(request: dict[str, Any]) -> str:
    token = request["headers"]["authorization"].split(" ", 1)[1]
    return str(jwt.decode(token, options={"verify_signature": False})["aud"])


def sent(gw: AgentGateway, door: str) -> list[dict[str, Any]]:
    return [r for r in gw.requests if r["path"] == door or r["path"].startswith(door + "/")]


# --- where the doors are -----------------------------------------------------------------------------------


def test_the_model_routes_are_under_llm_and_the_tools_stay_under_mcp(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        gateway = af.agent(AGENT, key).gateway()
        assert gateway.mcp_url == gw.issuer + "/mcp"
        assert gateway.llm_url == gw.issuer + "/llm"
        assert gateway.model_base_url("anthropic") == gw.issuer + "/llm"
        assert gateway.model_base_url("openai") == gw.issuer + "/llm/v1"
    assert MODEL_PATH_PREFIX == "/llm/v1/"


def test_the_red_team_sends_to_the_model_door() -> None:
    assert _PATHS == {
        "chat": "/llm/v1/chat/completions",
        "responses": "/llm/v1/responses",
        "messages": "/llm/v1/messages",
    }


def test_the_old_model_route_is_not_served(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        gateway = af.agent(AGENT, key).gateway()
        assert gateway.http_client().post(gw.issuer + "/mcp/v1/messages", json=MODEL).status_code == 404


# --- a second token, requested for the model door, on the same key ----------------------------------------


def test_the_model_token_is_requested_with_the_llm_resource_and_the_tool_token_is_not(
    gw: AgentGateway, key: AgentKey
) -> None:
    with AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        assert client.post(gw.issuer + "/mcp", json=rpc("tools/list")).status_code == 200
        assert client.post(gw.issuer + "/llm/v1/messages", json=MODEL).status_code == 200
    assert gw.mint_resources == [None, gw.issuer + "/llm"]  # one resource per token, the tool token as before
    assert {audience_of(r) for r in sent(gw, "/mcp")} == {gw.issuer + "/mcp"}
    assert {audience_of(r) for r in sent(gw, "/llm")} == {gw.issuer + "/llm"}


def test_each_door_gets_only_its_own_token_even_when_calls_alternate(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        for _ in range(2):
            assert client.post(gw.issuer + "/llm/v1/messages", json=MODEL).status_code == 200
            assert client.post(gw.issuer + "/mcp", json=rpc("tools/list")).status_code == 200
    assert gw.wrong_door == []
    assert len(gw.mint_resources) == 2  # each token cached apart: two mints for four calls
    assert len({r["headers"]["authorization"] for r in sent(gw, "/llm")}) == 1
    assert len({r["headers"]["authorization"] for r in sent(gw, "/mcp")}) == 1


def test_the_two_tokens_are_bound_to_one_key_and_make_one_run(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        client.post(gw.issuer + "/mcp", json=rpc("tools/list"))
        client.post(gw.issuer + "/llm/v1/messages", json=MODEL)
    tool, model = (gw.issued[t] for t in gw.minted)
    assert tool["jkt"] and tool["jkt"] == model["jkt"]
    assert {run for _, run, _ in gw.run_calls} == {gw._run_of(tool)}  # pyright: ignore[reportPrivateUsage]
    assert {p["jwk"]["x"] for p in gw.proofs if p["path"] != "/oauth/token"} == {
        p["jwk"]["x"] for p in gw.proofs if p["path"] == "/oauth/token"
    }  # every proof, both doors, was signed by the one key


def test_the_stand_in_does_refuse_the_wrong_token_at_a_door(gw: AgentGateway, key: AgentKey) -> None:
    """The control: sent by hand, the tool token at the model door and the model token at the tool door are refused."""
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)
        on_model = httpx.post(
            gw.issuer + "/llm/v1/messages", json=MODEL, headers=agent.headers("POST", gw.issuer + "/llm/v1/messages")
        )
        llm = agent.for_resource(gw.issuer + "/llm")
        on_tools = httpx.post(gw.mcp_url, json=rpc("tools/list"), headers=llm.headers("POST", gw.mcp_url))
    assert (on_model.status_code, on_tools.status_code) == (401, 401)
    assert [path for path, _ in gw.wrong_door] == ["/llm/v1/messages", "/mcp"]


async def test_the_asyncio_gateway_sends_each_door_its_own_token(gw: AgentGateway, key: AgentKey) -> None:
    async with AsyncAuthFI(gw.issuer) as af:
        gateway = af.agent(AGENT, key).gateway()
        client = gateway.http_client()
        assert (await client.post(gw.issuer + "/llm/v1/messages", json=MODEL)).status_code == 200
        assert (await client.post(gw.issuer + "/mcp", json=rpc("tools/list"))).status_code == 200
        assert (await gateway.headers("POST", gw.issuer + "/llm/v1/messages"))["Authorization"].startswith("DPoP ")
    assert gw.mint_resources == [gw.issuer + "/llm", None]
    assert gw.wrong_door == []
    assert {audience_of(r) for r in sent(gw, "/llm")} == {gw.issuer + "/llm"}
    assert {audience_of(r) for r in sent(gw, "/mcp")} == {gw.issuer + "/mcp"}


def test_a_provider_sdk_over_the_gateways_client_reaches_the_model_door(gw: AgentGateway, key: AgentKey) -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx2 = pytest.importorskip("httpx2")
    with AuthFI(gw.issuer) as af:
        gateway = af.agent(AGENT, key).gateway()
        sdk = anthropic.Anthropic(
            api_key="unused", base_url=gateway.model_base_url("anthropic"), http_client=gateway.http_client(httpx2)
        )
        sdk.messages.create(model="claude-haiku-5-5", max_tokens=8, messages=[{"role": "user", "content": "hi"}])
    assert [r["path"] for r in sent(gw, "/llm")] == ["/llm/v1/messages"]
    assert gw.wrong_door == [] and gw.mint_resources == [gw.issuer + "/llm"]


# --- never to a customer's API, never a token it was not asked for ----------------------------------------


@pytest.mark.parametrize("path", ["/llm/v1/messages", "/mcp"])
def test_neither_token_goes_to_a_customers_api(gw: AgentGateway, key: AgentKey, path: str) -> None:
    with AuthFI(gw.issuer) as af:
        client = af.agent(AGENT, key).gateway().http_client()
        with pytest.raises(ConfigurationError) as caught:
            client.post("https://evil.example" + path, json=MODEL)
    assert caught.value.code == "destination_not_allowed"
    assert gw.requests == [] or all(r["path"] == "/oauth/token" for r in gw.requests)


def test_the_model_resource_is_authfis_own(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        assert af.names_authfi(gw.issuer + "/llm") and af.names_authfi(gw.issuer + "/mcp")
        assert not af.names_authfi("https://orders.example/llm")
        model = af.agent(AGENT, key).for_resource(gw.issuer + "/llm")
        assert set(model.send_to) <= set(af.authfi_origins)
        with pytest.raises(ConfigurationError):
            model.headers("POST", "https://orders.example/llm/v1/messages")


def test_a_credential_that_cannot_make_a_model_token_is_not_given_the_tool_token_at_the_model_door(
    gw: AgentGateway, key: AgentKey
) -> None:
    with AuthFI(gw.issuer) as af:
        agent = af.agent(AGENT, key)

        class OneToken(TokenSource):
            @property
            def send_to(self) -> tuple[str, ...]:
                return agent.send_to

            def token(self, *, force: bool = False) -> str:
                return agent.token()

        client = Gateway(af, OneToken()).http_client()
        with pytest.raises(ConfigurationError, match="cannot make a token for the AI gateway"):
            client.post(gw.issuer + "/llm/v1/messages", json=MODEL)
    assert sent(gw, "/llm") == []


# --- a task's root is the one token both doors take --------------------------------------------------------


def test_a_task_root_is_accepted_at_both_doors(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af, start(af.agent(AGENT, key)) as task:
        client = task.gateway().http_client()
        assert client.post(gw.issuer + "/mcp", json=rpc("tools/list")).status_code == 200
        assert client.post(gw.issuer + "/llm/v1/messages", json=MODEL).status_code == 200
    roots = {r["headers"]["authorization"].split(" ", 1)[1] for r in sent(gw, "/mcp") + sent(gw, "/llm")}
    assert len(roots) == 1 and gw.issued[roots.pop()]["kind"] == "task"  # one token, both doors, no second mint
    assert gw.mint_resources == []
    assert gw.wrong_door == []


# --- a machine client asks for its model token the same way -----------------------------------------------


def test_a_machine_client_requests_its_model_token_with_the_llm_resource(
    platform: Platform, af: AuthFI, clock: FakeClock
) -> None:
    count = 0

    def mint(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return json_response({"access_token": f"token-{count}", "token_type": "Bearer", "expires_in": 900})

    platform.route("POST", "/oauth/token", mint)
    m2m = af.client_credentials("app_reports", Secret("s3cret"), resource=af.issuer + "/mcp")
    gateway = m2m.gateway()
    assert gateway.headers("POST", af.issuer + "/mcp")["Authorization"] == "Bearer token-1"
    assert gateway.headers("POST", af.issuer + "/llm/v1/messages")["Authorization"] == "Bearer token-2"
    assert gateway.headers("POST", af.issuer + "/mcp")["Authorization"] == "Bearer token-1"
    resources = [form_of(r).get("resource") for r in platform.calls("POST", "/oauth/token")]
    assert resources == [af.issuer + "/mcp", af.issuer + "/llm"]
    assert gateway.llm_source is gateway.llm_source


def test_a_delegated_token_cannot_make_a_model_token_and_is_not_sent_to_the_model_door(
    gw: AgentGateway, key: AgentKey, agent_clock: FakeClock
) -> None:
    """It is refused here, not sent to the model door to be refused there."""
    with AuthFI(gw.issuer, clock=agent_clock) as af:
        agent = af.agent(AGENT, key)
        delegated = agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
        client = Gateway(af, delegated).http_client()
        assert client.post(gw.issuer + "/mcp", json=rpc("tools/list")).status_code == 200
        with pytest.raises(ConfigurationError, match="cannot make a token for the AI gateway"):
            client.post(gw.issuer + "/llm/v1/messages", json=MODEL)
