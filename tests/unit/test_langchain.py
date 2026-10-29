"""authfi.langchain against a real-HTTP stand-in for the gateway, and a stand-in tracing backend."""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import cached_property

import pytest

pytest.importorskip("langchain_anthropic")

import langsmith
from langchain_anthropic import ChatAnthropic
from langchain_core.load import dumps

from authfi import AgentKey, AsyncAuthFI, AuthFI, ConfigurationError, GatewayTokenRefused, IntegrationError
from authfi._adapters import async_gateway_for
from authfi.langchain import LangChainIntegration
from authfi.langchain_anthropic import AuthFIChatAnthropic, _check_seam
from tests.fake_gateway import CaptureServer, FakeGateway

AGENT = "agt_langchain1"
PROMPT = "the caller's account number is 99417"


@pytest.fixture(scope="module")
def key() -> AgentKey:
    return AgentKey.generate(kid="k1")


@pytest.fixture
def fake() -> Iterator[FakeGateway]:
    with FakeGateway() as gateway:
        yield gateway


@pytest.fixture
def integration(fake: FakeGateway, key: AgentKey) -> Iterator[LangChainIntegration]:
    with AuthFI(fake.issuer) as af:
        yield LangChainIntegration(af.agent(AGENT, key))


def test_every_integration_on_an_agent_shares_the_agents_one_gateway(fake: FakeGateway, key: AgentKey) -> None:
    """A gateway owns an HTTP client: one per agent, not one per adapter built on it."""
    with AuthFI(fake.issuer) as af:
        agent = af.agent(AGENT, key)
        assert LangChainIntegration(agent).gateway is agent.gateway()
        before = len(af._closables)  # pyright: ignore[reportPrivateUsage]
        for _ in range(100):
            assert LangChainIntegration(agent).gateway is agent.gateway()
        assert len(af._closables) == before  # pyright: ignore[reportPrivateUsage]


async def test_the_asyncio_integration_shares_the_agents_one_gateway(fake: FakeGateway, key: AgentKey) -> None:
    async with AsyncAuthFI(fake.issuer) as af:
        agent = af.agent(AGENT, key)
        assert LangChainIntegration(agent).gateway is agent.gateway()
        assert LangChainIntegration(agent).gateway is LangChainIntegration(agent).gateway
        assert async_gateway_for(agent) is agent.gateway()  # what the asyncio-only frameworks build on


def test_the_model_calls_the_gateway_with_the_agents_token(
    fake: FakeGateway, integration: LangChainIntegration
) -> None:
    model = integration.chat_model("claude-haiku-5-5", max_tokens=64)
    fake.model_script.append([{"type": "text", "text": "Refund sent."}])

    answer = model.invoke(PROMPT)

    assert answer.content == "Refund sent."
    (request,) = fake.model_requests()
    assert request["path"] == "/mcp/v1/messages"
    assert request["headers"]["authorization"] == "Bearer tok-minted-1"
    assert request["headers"]["x-request-id"].startswith(f"{integration.run_id}.")
    assert request["json"]["model"] == "claude-haiku-5-5"


def test_streaming_uses_the_same_client(fake: FakeGateway, integration: LangChainIntegration) -> None:
    model = integration.chat_model("claude-haiku-5-5", max_tokens=64)
    fake.model_script.append([{"type": "text", "text": "Streamed."}])

    text = "".join(str(chunk.content) for chunk in model.stream(PROMPT))

    assert text == "Streamed."
    assert fake.model_requests()[0]["headers"]["authorization"] == "Bearer tok-minted-1"


def test_the_token_is_not_part_of_the_model(fake: FakeGateway, integration: LangChainIntegration) -> None:
    model = integration.chat_model("claude-haiku-5-5", max_tokens=64)
    model.invoke(PROMPT)
    token = fake.minted[0]

    for view in (dumps(model), repr(model), json.dumps(model._get_ls_params(), default=str)):
        assert token not in view
    assert (model.metadata or {})["authfi_agent_id"] == AGENT


def test_a_refused_token_ends_the_run(fake: FakeGateway, integration: LangChainIntegration) -> None:
    model = integration.chat_model("claude-haiku-5-5", max_tokens=64, max_retries=0)
    fake.revoked = True

    with pytest.raises(GatewayTokenRefused) as caught:
        model.invoke(PROMPT)

    assert caught.value.challenge is not None
    assert "revoked" in caught.value.challenge


async def test_an_asyncio_agent_gives_an_asyncio_model(fake: FakeGateway, key: AgentKey) -> None:
    async with AsyncAuthFI(fake.issuer) as af:
        model = LangChainIntegration(af.agent(AGENT, key)).chat_model("claude-haiku-5-5", max_tokens=64)
        fake.model_script.append([{"type": "text", "text": "Async."}])

        answer = await model.ainvoke(PROMPT)

        assert answer.content == "Async."
        assert fake.model_requests()[0]["headers"]["authorization"] == "Bearer tok-minted-1"
        with pytest.raises(ConfigurationError, match="ainvoke"):
            model.invoke(PROMPT)


async def test_a_refused_token_ends_an_asyncio_run(fake: FakeGateway, key: AgentKey) -> None:
    async with AsyncAuthFI(fake.issuer) as af:
        model = LangChainIntegration(af.agent(AGENT, key)).chat_model("claude-haiku-5-5", max_retries=0)
        fake.revoked = True
        with pytest.raises(GatewayTokenRefused):
            await model.ainvoke(PROMPT)


def test_a_blocking_model_refuses_asyncio_calls(integration: LangChainIntegration) -> None:
    model = integration.chat_model("claude-haiku-5-5")
    with pytest.raises(ConfigurationError, match="invoke or stream"):
        _ = model._async_client


@pytest.mark.parametrize("argument", ["api_key", "anthropic_api_key", "base_url", "anthropic_api_url"])
def test_the_credential_arguments_are_the_integrations(integration: LangChainIntegration, argument: str) -> None:
    with pytest.raises(ConfigurationError, match=argument):
        integration.chat_model("claude-haiku-5-5", **{argument: "x"})


@pytest.mark.parametrize("header", ["Authorization", "X-Api-Key"])
def test_a_credential_header_is_refused_because_tracers_record_it(
    integration: LangChainIntegration, header: str
) -> None:
    with pytest.raises(ConfigurationError, match="tracers"):
        integration.chat_model("claude-haiku-5-5", default_headers={header: "Bearer x"})


def test_a_credential_without_its_project_needs_one(fake: FakeGateway, key: AgentKey) -> None:
    from authfi.tokens import TokenSource

    class Bare(TokenSource):
        @property
        def send_to(self) -> tuple[str, ...]:
            return (fake.issuer,)

        def token(self, *, force: bool = False) -> str:
            return "tok-bare"

    with pytest.raises(ConfigurationError, match="project=af"):
        LangChainIntegration(Bare())
    with AuthFI(fake.issuer) as af:
        integration = LangChainIntegration(Bare(), project=af)
        assert integration.agent_id is None
        assert "authfi_agent_id" not in (integration.chat_model("claude-haiku-5-5").metadata or {})


def test_something_that_is_not_a_credential_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="not an AuthFI credential"):
        LangChainIntegration("tok-raw")  # type: ignore[arg-type]


@pytest.mark.parametrize("capture_content", [False, True])
def test_tracing_never_carries_the_token_and_content_only_when_asked(
    fake: FakeGateway, integration: LangChainIntegration, capture_content: bool
) -> None:
    with CaptureServer() as capture:
        client = LangChainIntegration.configure_tracing(
            capture_content=capture_content, api_url=capture.url, api_key="lsv2-capture"
        )
        langsmith.configure(enabled=True)
        try:
            model = integration.chat_model("claude-haiku-5-5", max_tokens=64)
            fake.model_script.append([{"type": "text", "text": "Refund sent to 99417."}])
            model.invoke(PROMPT)
            client.flush()
        finally:
            langsmith.configure(client=None, enabled=None)
        blob = capture.blob()

    assert b"claude-haiku-5-5" in blob  # the run was recorded: the instrument works
    assert fake.minted[0].encode() not in blob
    assert (b"99417" in blob) is capture_content


def test_a_moved_seam_is_refused_at_import() -> None:
    class Moved:
        _client = property(lambda self: None)

    with pytest.raises(IntegrationError, match="tested with"):
        _check_seam(Moved)
    _check_seam(ChatAnthropic)
    assert isinstance(AuthFIChatAnthropic.__dict__["_client"], cached_property)


def test_importing_the_adapter_checks_the_seam_where_it_runs() -> None:
    """The import itself must refuse a langchain-anthropic that moved ``_client``, not only the function."""
    import os
    import subprocess
    import sys

    code = os.linesep.join(
        [
            "import langchain_anthropic",
            "del langchain_anthropic.ChatAnthropic._client",
            "try:",
            "    import authfi.langchain_anthropic",
            "except Exception as error:",
            "    print(type(error).__name__, error)",
            "else:",
            "    print('IMPORTED')",
        ]
    )
    ran = subprocess.run(  # noqa: S603 - this interpreter, on code written above
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False
    )
    assert ran.stdout.startswith("IntegrationError"), ran.stdout + ran.stderr
    assert "ChatAnthropic._client" in ran.stdout


@pytest.mark.parametrize("capture_content", [False, True])
def test_export_steps_sends_the_runs_to_authfi_and_not_to_langsmith(
    fake: FakeGateway, key: AgentKey, capture_content: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    from opentelemetry.sdk.trace import TracerProvider

    with CaptureServer() as langsmith_backend, AuthFI(fake.issuer) as af:
        monkeypatch.setenv("LANGSMITH_ENDPOINT", langsmith_backend.url)  # a stand-in that must see nothing
        monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2-capture")
        agent = af.agent(AGENT, key)
        integration = LangChainIntegration(agent)
        provider = TracerProvider()
        try:
            integration.export_steps(capture_content=capture_content, tracer_provider=provider)
            model = integration.chat_model("claude-haiku-5-5", max_tokens=64)
            fake.model_script.append([{"type": "text", "text": "Refund sent to 99417."}])
            model.invoke(PROMPT)
            langsmith.run_trees.get_cached_client().flush()
            provider.force_flush()
            provider.shutdown()
        finally:
            langsmith.configure(client=None, enabled=None)
        assert [body for _, _, body in langsmith_backend.bodies if body] == []

    steps = fake.exported_spans()
    assert steps, "LangChain's run reached AuthFI as spans"
    assert all(s["attributes"]["authfi.agent_id"] == AGENT for s in steps)
    assert all(s["attributes"]["authfi.run_id"] == agent.run_id for s in steps)
    assert all(s["headers"]["authorization"] == "Bearer tok-minted-1" for s in steps)
    posted = b"".join(r["body"] for r in fake.requests if r["path"] == "/otlp/v1/traces")
    assert fake.minted[0].encode() not in posted
    assert (b"99417" in posted) is capture_content


def test_the_run_is_in_the_models_metadata(integration: LangChainIntegration) -> None:
    model = integration.chat_model("claude-haiku-5-5")
    assert (model.metadata or {})["authfi_run_id"] == integration.run_id
