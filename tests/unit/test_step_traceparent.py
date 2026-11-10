"""A gateway request carries the traceparent of the framework step that is executing when it is made.

The frameworks' own tests (OpenAI Agents, LangGraph, ADK, Claude Agent SDK) check the ids against the spans the
framework exports; this file checks the resolver and the places the SDK sets the header.
"""

from __future__ import annotations

import gc
import json
import weakref
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from authfi import AgentKey, AuthFI
from authfi._protocol.ids import (
    _parents,
    _sources,
    inject_traceparent,
    link_step_owner,
    register_step_source,
    step_traceparent,
    trace_id_of,
    valid_traceparent,
)
from authfi.agents import AgentAuth
from authfi.gateway import Gateway
from tests.agent_platform import AGENT, AgentPlatform
from tests.support import FakeClock, json_response

TRACE = "0af7651916cd43dd8448eb211c80319c"
SPAN = "b7ad6b7169203331"
STEP = f"00-{TRACE}-{SPAN}-01"
OTHER = "00-" + "c" * 32 + "-" + "d" * 16 + "-01"


class Owner:
    """Stands for a credential: anything that can be weakly referenced."""


class Slotted:
    """A slotted credential: it cannot be weakly referenced."""

    __slots__ = ()


def test_a_credential_that_cannot_be_weakly_referenced_has_a_step_too() -> None:
    owner = Slotted()
    register_step_source(owner, lambda: (TRACE, SPAN))
    headers: dict[str, str] = {}
    inject_traceparent(headers, owner)
    assert headers["traceparent"] == STEP
    assert step_traceparent(Slotted()) is None  # another credential of the class has none


def test_the_registered_step_is_the_traceparent_and_is_sampled() -> None:
    owner = Owner()
    register_step_source(owner, lambda: (TRACE, SPAN))
    headers: dict[str, str] = {}
    inject_traceparent(headers, owner)
    assert headers["traceparent"] == STEP
    assert step_traceparent(owner) == STEP


def test_a_step_is_for_the_credential_it_was_registered_for() -> None:
    mine, theirs = Owner(), Owner()
    register_step_source(mine, lambda: (TRACE, SPAN))
    headers: dict[str, str] = {}
    inject_traceparent(headers, theirs)
    assert headers["traceparent"] != STEP and headers["traceparent"].endswith("-00")
    assert step_traceparent(theirs) is None
    assert step_traceparent(None) is None


def test_with_no_step_executing_the_value_is_fresh_and_unsampled() -> None:
    owner = Owner()
    register_step_source(owner, lambda: None)
    seen: set[str] = set()
    for _ in range(3):
        headers: dict[str, str] = {}
        inject_traceparent(headers, owner)
        assert headers["traceparent"].endswith("-00") and trace_id_of(headers["traceparent"])
        seen.add(headers["traceparent"])
    assert len(seen) == 3


def test_a_header_the_caller_set_is_not_overwritten() -> None:
    owner = Owner()
    register_step_source(owner, lambda: (TRACE, SPAN))
    headers = {"Traceparent": OTHER}
    inject_traceparent(headers, owner)
    assert headers == {"Traceparent": OTHER}


def test_the_first_source_that_has_a_step_answers() -> None:
    owner = Owner()
    register_step_source(owner, lambda: None)
    register_step_source(owner, lambda: (TRACE, SPAN))
    assert step_traceparent(owner) == STEP


def test_registering_a_source_twice_adds_it_once() -> None:
    owner, calls = Owner(), list[int]()

    def source() -> None:
        calls.append(1)

    register_step_source(owner, source)
    register_step_source(owner, source)
    step_traceparent(owner)
    assert len(calls) == 1


def test_a_registration_does_not_keep_its_credential_alive() -> None:
    owner = Owner()
    register_step_source(owner, lambda: (TRACE, SPAN))
    alive = weakref.ref(owner)
    del owner
    gc.collect()
    assert alive() is None


def test_the_step_is_the_one_executing_not_the_active_opentelemetry_span() -> None:
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider

    owner = Owner()
    register_step_source(owner, lambda: (TRACE, SPAN))
    with TracerProvider().get_tracer("t").start_as_current_span("ambient"):
        headers: dict[str, str] = {}
        inject_traceparent(headers, owner)
        without: dict[str, str] = {}
        inject_traceparent(without)
    assert headers["traceparent"] == STEP
    assert trace_id_of(without["traceparent"]) != TRACE  # the ambient span is the step when no source names one


def test_only_a_well_formed_traceparent_is_read_from_a_message() -> None:
    assert valid_traceparent(" " + STEP.upper() + " ") == STEP
    for bad in (None, 7, "", "nonsense", STEP + "-x", f"00-{TRACE}-{SPAN}", "00-" + "g" * 32 + "-" + SPAN + "-01"):
        assert valid_traceparent(bad) is None


# --- the gateway's HTTP client ----------------------------------------------------------------------------


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


class Server:
    def __init__(self) -> None:
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return json_response({"jsonrpc": "2.0", "id": 1, "result": {}})


def _client(gateway: Gateway) -> tuple[httpx.Client, Server]:
    client = gateway.http_client()
    server = Server()
    client._transport = httpx.MockTransport(server)
    return client, server


def _call(meta: Any = None, *, drop_meta: bool = False) -> dict[str, Any]:
    params: dict[str, Any] = {"name": "t", "arguments": {}}
    if not drop_meta:
        params["_meta"] = meta
    return {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": params}


def test_an_mcp_call_carries_the_step_its_own_meta_names(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    client, server = _client(gateway)
    register_step_source(agent, lambda: ("e" * 32, "f" * 16))  # the step the session's task began in
    client.post(gateway.mcp_url, json=_call({"traceparent": STEP}))
    assert server.seen[0].headers["traceparent"] == STEP


def test_a_malformed_meta_is_not_a_step_and_the_call_is_still_sent(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    client, server = _client(gateway)
    for meta in ({"traceparent": "nonsense"}, {"traceparent": 7}, {}, None, "x"):
        client.post(gateway.mcp_url, json=_call(meta))
    client.post(gateway.mcp_url, json=_call(drop_meta=True))
    client.post(gateway.mcp_url, content=b"not json")
    client.post(gateway.mcp_url, json=[1])
    assert len(server.seen) == 8
    for request in server.seen:
        assert request.headers["traceparent"].endswith("-00")


def test_a_meta_does_not_replace_a_header_the_caller_set(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    client, server = _client(gateway)
    client.post(gateway.mcp_url, json=_call({"traceparent": STEP}), headers={"traceparent": OTHER})
    assert server.seen[0].headers["traceparent"] == OTHER


def test_the_meta_is_read_from_the_mcp_route_only(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    client, server = _client(gateway)
    client.post(gateway.model_base_url("anthropic") + "/v1/messages", json=_call({"traceparent": STEP}))
    client.get(gateway.mcp_url, params={"_meta": json.dumps({"traceparent": STEP})})
    assert [r.headers["traceparent"] == STEP for r in server.seen] == [False, False]


def test_a_streamed_body_is_not_read(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    client, server = _client(gateway)

    def body() -> Iterator[bytes]:
        yield json.dumps(_call({"traceparent": STEP})).encode()

    client.post(gateway.mcp_url, content=body())
    assert server.seen[0].headers["traceparent"].endswith("-00")


def test_a_call_with_no_meta_carries_the_step_executing_when_it_is_sent(agent: AgentAuth) -> None:
    gateway = agent.gateway()
    client, server = _client(gateway)
    register_step_source(agent, lambda: (TRACE, SPAN))
    client.post(gateway.mcp_url, json=_call(drop_meta=True))
    assert server.seen[0].headers["traceparent"] == STEP


def test_cloud_credentials_are_asked_for_by_the_step_executing(agent: AgentAuth, platform: AgentPlatform) -> None:
    seen: list[httpx.Request] = []

    def door(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return json_response({"data": {"approval_id": "7f1c2a9e-5d1b-4f3a-9a77-2b0c8e1d4f00"}}, status=202)

    platform.route("POST", "/mcp/v1/cloud/credentials", door)
    gateway = agent.gateway()
    gateway.cloud_credentials("prod", "deployer")
    assert seen[0].headers["traceparent"].endswith("-00")  # no step: the transport's own fresh value

    register_step_source(agent, lambda: (TRACE, SPAN))
    gateway.cloud_credentials("prod", "deployer")
    assert seen[1].headers["traceparent"] == STEP


def test_a_source_that_fails_does_not_stop_the_call(caplog: pytest.LogCaptureFixture) -> None:
    owner = Owner()

    def broken() -> None:
        raise RuntimeError("framework moved")

    register_step_source(owner, broken)
    register_step_source(owner, lambda: (TRACE, SPAN))
    with caplog.at_level("WARNING", logger="authfi"):
        assert step_traceparent(owner) == STEP  # the next source still answers
    assert any("step source failed" in r.getMessage() for r in caplog.records)


# --- a credential made from another one names the step through it -----------------------------------------------


def test_a_linked_credential_names_the_step_its_parent_has() -> None:
    parent, child = Owner(), Owner()
    register_step_source(parent, lambda: (TRACE, SPAN))
    assert step_traceparent(child) is None  # unlinked: nothing to say
    link_step_owner(child, parent)
    assert step_traceparent(child) == STEP
    assert step_traceparent(parent) == STEP
    assert step_traceparent(Owner()) is None


def test_a_credentials_own_step_comes_before_its_parents() -> None:
    parent, child = Owner(), Owner()
    register_step_source(parent, lambda: (TRACE, SPAN))
    register_step_source(child, lambda: ("a" * 32, "b" * 16))
    link_step_owner(child, parent)
    assert step_traceparent(child) == "00-" + "a" * 32 + "-" + "b" * 16 + "-01"


def test_a_link_never_leaks_down_or_loops() -> None:
    a, b = Owner(), Owner()
    register_step_source(b, lambda: (TRACE, SPAN))
    link_step_owner(a, b)
    link_step_owner(b, a)  # a loop: the walk ends
    link_step_owner(a, a)
    assert step_traceparent(a) == STEP
    lone = Owner()
    register_step_source(lone, lambda: None)
    link_step_owner(lone, lone)
    assert step_traceparent(lone) is None


def test_a_link_does_not_keep_its_credential_alive() -> None:
    parent, child = Owner(), Owner()
    link_step_owner(child, parent)
    ref = weakref.ref(child)
    key = id(child)
    assert key in _parents
    del child
    gc.collect()
    assert ref() is None
    assert key not in _parents  # the entry goes with the credential: an id reused later inherits nothing


def test_a_dead_credentials_entries_do_not_name_a_step_for_a_new_one() -> None:
    parent = Owner()
    register_step_source(parent, lambda: (TRACE, SPAN))
    seen: set[int] = set()
    for _ in range(200):  # ids are reused once an object is gone
        child = Owner()
        assert id(child) not in _parents and id(child) not in _sources
        assert step_traceparent(child) is None
        link_step_owner(child, parent)
        register_step_source(child, lambda: None)
        seen.add(id(child))
        del child
        gc.collect()
    assert not (seen & set(_parents)) and not (seen & set(_sources))


def test_a_task_names_the_step_its_agent_has() -> None:
    """The framework exports its steps for the agent; a call made under the agent's task is still that step's."""
    import jwt

    from authfi import Secret
    from tests.fake_agent_gateway import AgentGateway

    person = Secret(jwt.encode({"sub": "usr_1", "exp": 4_000_000_000}, "k" * 32, algorithm="HS256"))
    with AgentGateway() as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent("agt_steps1", AgentKey.generate(kid="k1"))
        register_step_source(agent, lambda: (TRACE, SPAN))
        with agent.task("support.refund", subject_token=person, params={"order": "4411"}) as task:
            task.gateway().http_client().post(
                gateway.mcp_url, json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            )
        other = af.agent("agt_steps2", AgentKey.generate(kid="k2"))
        with other.task("support.refund", subject_token=person, params={"order": "4411"}) as task2:
            task2.gateway().http_client().post(
                gateway.mcp_url, json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
            )
    sent = [r["headers"]["traceparent"] for r in gateway.requests if r["path"] == "/mcp"]
    assert sent[0] == STEP
    assert sent[1] != STEP and sent[1].endswith("-00")  # another agent's task has no step of this one's


def _person() -> Any:
    import jwt

    from authfi import Secret

    return Secret(jwt.encode({"sub": "usr_1", "exp": 4_000_000_000}, "k" * 32, algorithm="HS256"))


def test_a_person_acting_token_and_a_resource_token_name_the_step_their_agent_has() -> None:
    from tests.fake_agent_gateway import AgentGateway

    with AgentGateway() as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent("agt_steps3", AgentKey.generate(kid="k3"))
        register_step_source(agent, lambda: (TRACE, SPAN))
        delegated = agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
        own = agent.for_resource(gateway.issuer + "/mcp")
        assert step_traceparent(delegated) == STEP
        assert step_traceparent(own) == STEP
        exchanged = agent.exchange(_person())
        assert step_traceparent(exchanged) == STEP
        assert (
            step_traceparent(af.agent("agt_steps4", AgentKey.generate(kid="k4")).for_resource(gateway.issuer + "/mcp"))
            is None
        )


async def test_an_async_task_names_the_step_its_agent_has() -> None:
    import jwt

    from authfi import AsyncAuthFI, Secret
    from tests.fake_agent_gateway import AgentGateway

    person = Secret(jwt.encode({"sub": "usr_1", "exp": 4_000_000_000}, "k" * 32, algorithm="HS256"))
    body: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    with AgentGateway() as gateway:
        async with AsyncAuthFI(gateway.issuer) as af:
            agent = af.agent("agt_asteps1", AgentKey.generate(kid="k1"))
            register_step_source(agent, lambda: (TRACE, SPAN))
            async with await agent.task("support.refund", subject_token=person, params={"order": "4411"}) as task:
                await task.gateway().http_client().post(gateway.mcp_url, json=body)
            other = af.agent("agt_asteps2", AgentKey.generate(kid="k2"))
            async with await other.task("support.refund", subject_token=person, params={"order": "4411"}) as task2:
                await task2.gateway().http_client().post(gateway.mcp_url, json=body)
    sent = [r["headers"]["traceparent"] for r in gateway.requests if r["path"] == "/mcp"]
    assert sent[0] == STEP
    assert sent[1] != STEP and sent[1].endswith("-00")


async def test_an_async_person_acting_token_and_resource_token_name_the_step_their_agent_has() -> None:
    from authfi import AsyncAuthFI
    from tests.fake_agent_gateway import AgentGateway

    with AgentGateway() as gateway:
        async with AsyncAuthFI(gateway.issuer) as af:
            agent = af.agent("agt_asteps3", AgentKey.generate(kid="k3"))
            register_step_source(agent, lambda: (TRACE, SPAN))
            delegated = await agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
            own = agent.for_resource(gateway.issuer + "/mcp")
            assert step_traceparent(delegated) == STEP
            assert step_traceparent(own) == STEP
            assert step_traceparent(await agent.exchange(_person())) == STEP
            stranger = af.agent("agt_asteps4", AgentKey.generate(kid="k4"))
            assert step_traceparent(stranger.for_resource(gateway.issuer + "/mcp")) is None


def test_a_walk_through_a_loop_with_no_step_ends_with_none() -> None:
    """Two credentials linked both ways and no source on either: the walk must stop, not spin."""
    a, b = Owner(), Owner()
    link_step_owner(a, b)
    link_step_owner(b, a)
    assert step_traceparent(a) is None
    assert step_traceparent(b) is None
    c = Owner()  # a longer cycle
    link_step_owner(b, c)
    link_step_owner(c, a)
    assert step_traceparent(a) is None


def _gone(make: Any) -> None:
    """Build a credential, check it is linked, drop it, and check nothing of it stays in the tables."""
    credential = make()
    key = id(credential)
    ref = weakref.ref(credential)  # raises TypeError when the class has no __weakref__ slot
    assert key in _parents
    del credential, make
    gc.collect()
    assert ref() is None
    assert key not in _parents and key not in _sources


def test_the_credentials_made_from_an_agent_are_collected_with_their_last_reference() -> None:
    from tests.fake_agent_gateway import AgentGateway

    with AgentGateway() as gateway, AuthFI(gateway.issuer) as af:
        agent = af.agent("agt_gc1", AgentKey.generate(kid="k1"))
        register_step_source(agent, lambda: (TRACE, SPAN))
        _gone(lambda: agent.for_resource(gateway.issuer + "/mcp"))
        _gone(lambda: agent.token_as_user("priya@acme.com", binding_message="Refund order 4411"))
        _gone(lambda: agent.exchange(_person()))
        tasks: list[Any] = []
        with agent.task("support.refund", subject_token=_person(), params={"order": "4411"}) as task:
            tasks.append(task)
            assert id(task) in _parents
        key, ref = id(task), weakref.ref(task)
        del task
        tasks.clear()
        gc.collect()
        assert ref() is None and key not in _parents


async def test_the_async_credentials_made_from_an_agent_are_collected_with_their_last_reference() -> None:
    from authfi import AsyncAuthFI
    from tests.fake_agent_gateway import AgentGateway

    with AgentGateway() as gateway:
        async with AsyncAuthFI(gateway.issuer) as af:
            agent = af.agent("agt_gc2", AgentKey.generate(kid="k2"))
            register_step_source(agent, lambda: (TRACE, SPAN))
            _gone(lambda: agent.for_resource(gateway.issuer + "/mcp"))
            delegated = await agent.token_as_user("priya@acme.com", binding_message="Refund order 4411")
            key, ref = id(delegated), weakref.ref(delegated)
            assert key in _parents
            del delegated
            gc.collect()
            assert ref() is None and key not in _parents
            async with await agent.task("support.refund", subject_token=_person(), params={"order": "4411"}) as task:
                tkey, tref = id(task), weakref.ref(task)
            del task
            gc.collect()
            assert tref() is None and tkey not in _parents
