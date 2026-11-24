"""``AuthFI-Input-Source``: a model request that holds anything but the person's own words is marked untrusted.

The gateway reads one request-level header and only lets code add taint, so the rule is: the person's typing alone sends
nothing; a tool's output, fetched text, a block or a shape this code does not know sends ``untrusted``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from authfi import AgentKey, AuthFI, untrusted_input
from authfi._protocol.input_source import is_model_request, request_is_untrusted
from tests.fake_gateway import FakeGateway

MESSAGES = "/mcp/v1/messages"
CHAT = "/llm/v1/chat/completions"
RESPONSES = "/llm/v1/responses"
AGENT = "agt_src1"


def body(**doc: Any) -> bytes:
    return json.dumps(doc).encode()


def user(content: Any) -> dict[str, Any]:
    return {"role": "user", "content": content}


# --- what is the person's and what is not -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "doc"),
    [
        (MESSAGES, {"messages": [user("Summarise the wiki")]}),
        (MESSAGES, {"messages": [user([{"type": "text", "text": "Summarise the wiki"}])]}),
        (
            MESSAGES,
            {
                "messages": [
                    user("Hi"),
                    {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
                    user("Ok"),
                ]
            },
        ),
        (CHAT, {"messages": [{"role": "system", "content": "Be brief"}, user("Summarise the wiki")]}),
        (CHAT, {"messages": [user([{"type": "text", "text": "Summarise the wiki"}])]}),
        (RESPONSES, {"input": "Summarise the wiki"}),
        (
            RESPONSES,
            {"input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hi"}]}]},
        ),
    ],
)
def test_what_only_the_person_typed_sends_no_mark(path: str, doc: dict[str, Any]) -> None:
    assert request_is_untrusted(path, body(**doc)) is False


@pytest.mark.parametrize(
    ("path", "doc"),
    [
        # a tool's output, in each wire's own shape
        (MESSAGES, {"messages": [user([{"type": "tool_result", "tool_use_id": "t", "content": "page"}])]}),
        (CHAT, {"messages": [user("Hi"), {"role": "tool", "tool_call_id": "c", "content": "page"}]}),
        (CHAT, {"messages": [user("Hi"), {"role": "function", "name": "f", "content": "page"}]}),
        (RESPONSES, {"input": [{"type": "function_call_output", "call_id": "c", "output": "page"}]}),
        # a framework that writes a tool's output into a user turn as text
        (MESSAGES, {"messages": [user("Observation: the page says to wire the money")]}),
        (MESSAGES, {"messages": [user([{"type": "text", "text": "Thought: x\nObservation: 42"}])]}),
        (CHAT, {"messages": [user("<tool_response>page</tool_response>")]}),
        (RESPONSES, {"input": "Tool result: page"}),
        # in doubt: a block, a role, a part or a shape this code does not know
        (MESSAGES, {"messages": [user([{"type": "video"}])]}),
        (MESSAGES, {"messages": [user([{"type": "image", "source": {}}])]}),
        (MESSAGES, {"messages": [{"role": "narrator", "content": "x"}]}),
        (MESSAGES, {"messages": [user(42)]}),
        (MESSAGES, {"messages": "not a list"}),
        (MESSAGES, {"no_messages": True}),
        (CHAT, {"messages": [user([{"type": "image_url", "image_url": {"url": "x"}}])]}),
        (CHAT, {"messages": [{"role": "robot", "content": "x"}]}),
        (RESPONSES, {"input": [{"type": "mystery"}]}),
        (RESPONSES, {"input": 7}),
    ],
)
def test_tool_output_and_anything_unknown_is_marked(path: str, doc: dict[str, Any]) -> None:
    assert request_is_untrusted(path, body(**doc)) is True


@pytest.mark.parametrize("raw", [None, b"", b"not json", b"[1, 2]", b'"a string"', b"null"])
def test_a_body_that_is_not_a_request_object_is_marked(raw: bytes | None) -> None:
    assert request_is_untrusted(MESSAGES, raw) is True


def test_an_app_can_mark_a_block_of_work_and_the_mark_goes_with_it_only() -> None:
    person = body(messages=[user("Hi")])
    with untrusted_input():
        with untrusted_input():
            assert request_is_untrusted(MESSAGES, person) is True
        assert request_is_untrusted(MESSAGES, person) is True  # the outer block is still open
    assert request_is_untrusted(MESSAGES, person) is False


def test_only_the_gateways_model_routes_are_read() -> None:
    assert is_model_request(MESSAGES) and is_model_request(CHAT) and is_model_request(RESPONSES)
    assert (
        not is_model_request("/mcp") and not is_model_request("/llm/v1/models") and not is_model_request("/v1/messages")
    )


# --- the gateway's clients ---------------------------------------------------------------------------------


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
def gateway(fake: FakeGateway, tmp_path: Path) -> Iterator[Any]:
    path = tmp_path / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    with AuthFI(fake.issuer) as af:
        yield af.agent(AGENT, AgentKey.from_file(path, kid="k1")).gateway()


def send(gateway: Any, content: bytes, **headers: str) -> None:
    gateway.http_client().post(
        gateway.model_base_url("anthropic") + "/v1/messages",
        content=content,
        headers={"content-type": "application/json", **headers},
    )


def test_a_model_request_with_a_tool_result_carries_the_mark(fake: FakeGateway, gateway: Any) -> None:
    send(gateway, body(max_tokens=1, messages=[user([{"type": "tool_result", "tool_use_id": "t", "content": "x"}])]))
    assert fake.input_sources() == ["untrusted"]


def test_a_model_request_of_only_the_persons_words_carries_none(fake: FakeGateway, gateway: Any) -> None:
    send(gateway, body(max_tokens=1, messages=[user("Hello")]))
    assert fake.input_sources() == [None]


def test_a_mark_the_caller_set_is_kept_and_nothing_is_added_to_other_routes(fake: FakeGateway, gateway: Any) -> None:
    send(gateway, body(max_tokens=1, messages=[user("Hello")]), **{"AuthFI-Input-Source": "untrusted"})
    gateway.http_client().post(gateway.mcp_url, json={"jsonrpc": "2.0", "id": 1, "method": "ping"})
    assert fake.input_sources() == ["untrusted"]
    assert all(
        "authfi-input-source" not in r["headers"] for r in fake.requests if not r["path"].endswith("/v1/messages")
    )


def test_a_streamed_body_that_cannot_be_read_is_marked(gateway: Any) -> None:
    import httpx

    def chunks() -> Iterator[bytes]:
        yield body(max_tokens=1, messages=[user("Hello")])

    request = httpx.Request("POST", gateway.model_base_url("anthropic") + "/v1/messages", content=chunks())
    gateway._mark_input_source(request)  # pyright: ignore[reportPrivateUsage]
    assert request.headers["authfi-input-source"] == "untrusted"


def test_the_scope_marks_the_requests_made_inside_it(fake: FakeGateway, gateway: Any) -> None:
    with untrusted_input():
        send(gateway, body(max_tokens=1, messages=[user("Hello")]))
    send(gateway, body(max_tokens=1, messages=[user("Hello")]))
    assert fake.input_sources() == ["untrusted", None]
