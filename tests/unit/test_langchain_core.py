"""authfi.langchain with only langchain-core installed: no Claude package, and LangSmith is optional."""

from __future__ import annotations

import sys
from collections.abc import Iterator

import pytest

pytest.importorskip("langchain_core")

from authfi import AgentKey, AuthFI
from authfi.langchain import LangChainIntegration
from tests.fake_gateway import FakeGateway


@pytest.fixture
def integration() -> Iterator[LangChainIntegration]:
    with FakeGateway() as fake, AuthFI(fake.issuer) as af:
        yield LangChainIntegration(af.agent("agt_core1", AgentKey.generate(kid="k1")))


def test_the_integration_holds_the_gateway_for_any_chat_model(integration: LangChainIntegration) -> None:
    assert integration.agent_id == "agt_core1"
    assert integration.gateway.model_base_url("anthropic").endswith("/mcp")


def test_a_claude_model_without_langchain_anthropic_says_which_extra_to_install(
    integration: LangChainIntegration, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    monkeypatch.delitem(sys.modules, "authfi.langchain_anthropic", raising=False)
    with pytest.raises(ImportError, match=r"authfi\[langchain-anthropic\]"):
        integration.chat_model("claude-haiku-5-5")


def test_tracing_without_langsmith_has_nothing_to_hide_and_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "langsmith", None)
    assert LangChainIntegration.configure_tracing() is None
    assert LangChainIntegration.configure_tracing(capture_content=True) is None


def test_the_credential_checks_still_come_before_the_missing_package(
    integration: LangChainIntegration, monkeypatch: pytest.MonkeyPatch
) -> None:
    from authfi import ConfigurationError

    monkeypatch.setitem(sys.modules, "langchain_anthropic", None)
    with pytest.raises(ConfigurationError, match="api_key is set by the integration"):
        integration.chat_model("claude-haiku-5-5", api_key="sk-x")
