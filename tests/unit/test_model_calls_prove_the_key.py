"""A model call through the AI gateway proves the key its token is bound to, whichever provider SDK makes it.

The gateway stand-in refuses a model call whose token is bound to nothing (the gateway's MCP_MODEL_PROOF_CHECK=enforce,
EMODEL020) and challenges for a DPoP nonce. A provider SDK given ``gateway.http_client()`` is what the OpenAI Agents
SDK, pydantic-ai, LangChain and LlamaIndex adapters are built on, so the provider SDKs themselves are the subject.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import pytest

pytest.importorskip("httpx2")
pytest.importorskip("anthropic")
pytest.importorskip("openai")

import anthropic
import httpx2
import openai

from authfi import AgentKey, AsyncAuthFI, AuthFI
from tests.fake_agent_gateway import AgentGateway

AGENT = "agt_modelproof"
MESSAGES: Any = [{"role": "user", "content": "hi"}]


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def gw() -> Iterator[AgentGateway]:
    with AgentGateway(nonce=True, model_proof=True) as gateway:
        yield gateway


def model_calls(gw: AgentGateway) -> list[dict[str, Any]]:
    return [r for r in gw.requests if r["path"].startswith("/mcp/v1/")]


def assert_every_model_call_proved_the_key(gw: AgentGateway, expected: int) -> None:
    calls = model_calls(gw)
    assert calls and len({c["headers"]["authorization"] for c in calls}) == 1
    assert all(c["headers"]["authorization"].startswith("DPoP ") and "dpop" in c["headers"] for c in calls)
    assert gw.refused_proofs == [] and gw.unproven_model_calls == []
    assert gw.nonce_challenges >= 1  # the nonce was asked for and answered, not skipped
    assert len([p for p in gw.proofs if p["path"].startswith("/mcp/v1/")]) == expected


def test_the_anthropic_sdk_proves_the_key_on_the_models_route(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        gateway = af.agent(AGENT, key).gateway()
        client = anthropic.Anthropic(
            api_key="unused", base_url=gateway.model_base_url("anthropic"), http_client=gateway.http_client(httpx2)
        )
        answer = client.messages.create(model="claude-haiku-5-5", max_tokens=16, messages=MESSAGES)
        again = client.messages.create(model="claude-haiku-5-5", max_tokens=16, messages=MESSAGES)

    assert answer.content and again.content
    assert_every_model_call_proved_the_key(gw, expected=2)


def test_the_openai_sdk_proves_the_key_on_the_chat_route(gw: AgentGateway, key: AgentKey) -> None:
    with AuthFI(gw.issuer) as af:
        gateway = af.agent(AGENT, key).gateway()
        client = openai.OpenAI(
            api_key="unused", base_url=gateway.model_base_url("openai"), http_client=gateway.http_client(httpx2)
        )
        answer = client.chat.completions.create(model="claude-haiku-5-5", messages=MESSAGES)

    assert answer.choices
    assert_every_model_call_proved_the_key(gw, expected=1)


async def test_the_async_anthropic_sdk_proves_the_key_on_the_models_route(gw: AgentGateway, key: AgentKey) -> None:
    async with AsyncAuthFI(gw.issuer) as af:
        gateway = af.agent(AGENT, key).gateway()
        client = anthropic.AsyncAnthropic(
            api_key="unused", base_url=gateway.model_base_url("anthropic"), http_client=gateway.http_client(httpx2)
        )
        answer = await client.messages.create(model="claude-haiku-5-5", max_tokens=16, messages=MESSAGES)

    assert answer.content
    assert_every_model_call_proved_the_key(gw, expected=1)


async def test_the_async_openai_sdk_proves_the_key_on_the_chat_route(gw: AgentGateway, key: AgentKey) -> None:
    async with AsyncAuthFI(gw.issuer) as af:
        gateway = af.agent(AGENT, key).gateway()
        client = openai.AsyncOpenAI(
            api_key="unused", base_url=gateway.model_base_url("openai"), http_client=gateway.http_client(httpx2)
        )
        answer = await client.chat.completions.create(model="claude-haiku-5-5", messages=MESSAGES)

    assert answer.choices
    assert_every_model_call_proved_the_key(gw, expected=1)


def test_a_bearer_credential_is_refused_by_the_gateway_and_the_sdk_says_so_once(
    gw: AgentGateway, key: AgentKey, caplog: pytest.LogCaptureFixture
) -> None:
    """The control leg: the same call with a token bound to nothing is the call the gateway refuses."""
    with AuthFI(gw.issuer) as af, caplog.at_level(logging.WARNING):
        gateway = af.agent(AGENT, key, dpop=False).gateway()
        client = anthropic.Anthropic(
            api_key="unused", base_url=gateway.model_base_url("anthropic"), http_client=gateway.http_client(httpx2)
        )
        for _ in range(2):
            with pytest.raises(anthropic.PermissionDeniedError):
                client.messages.create(model="claude-haiku-5-5", max_tokens=16, messages=MESSAGES)

    assert gw.unproven_model_calls == ["/mcp/v1/messages"] * 2
    warned = [r for r in caplog.records if "MCP_MODEL_PROOF_CHECK" in r.getMessage()]
    assert len(warned) == 1 and "Bearer" in warned[0].getMessage()


def test_a_proven_call_makes_no_warning(gw: AgentGateway, key: AgentKey, caplog: pytest.LogCaptureFixture) -> None:
    with AuthFI(gw.issuer) as af, caplog.at_level(logging.WARNING):
        gateway = af.agent(AGENT, key).gateway()
        client = anthropic.Anthropic(
            api_key="unused", base_url=gateway.model_base_url("anthropic"), http_client=gateway.http_client(httpx2)
        )
        client.messages.create(model="claude-haiku-5-5", max_tokens=16, messages=MESSAGES)

    assert [r for r in caplog.records if "MCP_MODEL_PROOF_CHECK" in r.getMessage()] == []
