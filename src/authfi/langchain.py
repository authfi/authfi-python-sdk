"""LangChain: a Claude model on the AuthFI AI gateway, whose token tracers never see.

``pip install "authfi[langchain] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from authfi import AuthFI
    from authfi.langchain import LangChainIntegration

    agent = AuthFI.from_env().agent_from_env()
    langchain = LangChainIntegration(agent)
    langchain.configure_tracing()                 # LangSmith records metadata, not content
    model = langchain.chat_model("claude-haiku-5-5")

LangChain's documented way to give ``ChatAnthropic`` a bearer token is
``default_headers={"Authorization": ...}``, and LangSmith records a model's constructor arguments
in every run: the token went to the tracing backend with every call. Here the token is never a
model argument. The model's HTTP client is the gateway's, which reads the credential for each
request below the tracer.

``ChatAnthropic`` has no public setting for its HTTP client, so this overrides its ``_client`` and
``_async_client`` (checked when this module is imported; the extra pins the tested version range).
"""

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Any, Self, cast

try:
    import anthropic
    import langsmith
    from langchain_anthropic import ChatAnthropic
    from pydantic import PrivateAttr, SecretStr
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.langchain needs langchain-anthropic: "
        "pip install 'authfi[langchain] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import PLACEHOLDER_KEY, agent_id_of, configure_langsmith, gateway_for, httpx_of
from authfi.errors import ConfigurationError, IntegrationError
from authfi.gateway import AsyncGateway, Gateway, token_refusal
from authfi.tokens import AsyncTokenSource, TokenSource

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import TracerProvider

    from authfi.client import AsyncAuthFI, AuthFI

__all__ = ["AuthFIChatAnthropic", "LangChainIntegration"]

#: The langchain-anthropic versions the private seam below is tested against.
TESTED = "langchain-anthropic 1.7"


def _check_seam(cls: type) -> None:
    """Refuse a langchain-anthropic whose private seam moved, rather than send a model call without the token."""
    members = {"_client": True, "_async_client": True, "_client_params": True, "_create": False, "_acreate": False}
    for member, cached in members.items():
        found = cls.__dict__.get(member)
        if isinstance(found, cached_property) if cached else callable(found):
            continue
        raise IntegrationError(
            f"authfi.langchain attaches the AuthFI credential through ChatAnthropic.{member}, which this "
            f"langchain-anthropic does not have. It is tested with {TESTED}."
        )


_check_seam(ChatAnthropic)


def _token_refused(error: anthropic.AuthenticationError) -> BaseException:
    return token_refusal(error) or error


class AuthFIChatAnthropic(ChatAnthropic):
    """``ChatAnthropic`` on the AuthFI AI gateway. Make it with :meth:`LangChainIntegration.chat_model`.

    The credential is attached by the gateway's HTTP client to each request; it is not a field of
    the model, so it is absent from ``serialized`` and from ``invocation_params``. The gateway
    refusing the token itself raises :class:`~authfi.GatewayTokenRefused`.
    """

    _authfi_gateway: Gateway | AsyncGateway | None = PrivateAttr(default=None)

    @classmethod
    def on_gateway(cls, gateway: Gateway | AsyncGateway, **kwargs: Any) -> Self:
        """A model on ``gateway``, with ``ChatAnthropic``'s ``kwargs``."""
        params: dict[str, Any] = {
            **kwargs,
            "api_key": SecretStr(PLACEHOLDER_KEY),
            "base_url": gateway.model_base_url("anthropic"),
        }
        chat = cls(**params)
        chat._authfi_gateway = gateway
        return chat

    def _create(self, payload: dict[str, Any]) -> Any:
        try:
            return super()._create(payload)  # pyright: ignore[reportUnknownMemberType]
        except anthropic.AuthenticationError as error:
            raise _token_refused(error) from error

    async def _acreate(self, payload: dict[str, Any]) -> Any:
        try:
            return await super()._acreate(payload)  # pyright: ignore[reportUnknownMemberType]
        except anthropic.AuthenticationError as error:
            raise _token_refused(error) from error

    def _authfi_params(self) -> dict[str, Any]:
        params = dict(self._client_params)
        params["api_key"] = PLACEHOLDER_KEY
        return params

    @cached_property
    def _client(self) -> anthropic.Client:
        gateway = self._authfi_gateway
        if not isinstance(gateway, Gateway):
            raise ConfigurationError(
                "this model was made from an asyncio credential: call it with ainvoke or astream, or make it from "
                "the blocking AuthFI"
            )
        return anthropic.Client(
            **self._authfi_params(), http_client=gateway.http_client(httpx_of("anthropic._base_client"))
        )

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        gateway = self._authfi_gateway
        if not isinstance(gateway, AsyncGateway):
            raise ConfigurationError(
                "this model was made from a blocking credential: call it with invoke or stream, or make it from "
                "AsyncAuthFI"
            )
        return anthropic.AsyncClient(
            **self._authfi_params(), http_client=gateway.http_client(httpx_of("anthropic._base_client"))
        )


class LangChainIntegration:
    """The AuthFI gateway for LangChain, from any credential.

    Args:
        source: the agent (``af.agent(...)``), or a token it holds for a person. A blocking
            credential gives models for ``invoke``/``stream``; an asyncio one for
            ``ainvoke``/``astream``.
        project: the project, for a credential of your own that does not name it.

    """

    def __init__(
        self,
        source: TokenSource | AsyncTokenSource,
        *,
        project: AuthFI | AsyncAuthFI | None = None,
    ) -> None:
        self.source = source
        self.gateway = gateway_for(source, project)
        self.agent_id = agent_id_of(source)

    @property
    def run_id(self) -> str:
        """The credential's run: on every gateway call (``X-Request-Id``) and every exported step."""
        return cast(str, getattr(self.source, "run_id", None) or "")

    def chat_model(self, model: str, **kwargs: Any) -> AuthFIChatAnthropic:
        """A Claude model on the gateway. ``kwargs`` are ``ChatAnthropic``'s, except the credential ones."""
        for refused in ("api_key", "anthropic_api_key", "base_url", "anthropic_api_url"):
            if refused in kwargs:
                raise ConfigurationError(f"{refused} is set by the integration; the credential is the agent's")
        headers = {k.lower(): v for k, v in dict(kwargs.get("default_headers") or {}).items()}
        if "authorization" in headers or "x-api-key" in headers:
            raise ConfigurationError(
                "a credential in default_headers is recorded by tracers; the integration attaches it per request"
            )
        metadata = dict(kwargs.pop("metadata", None) or {})
        if self.agent_id:
            metadata["authfi_agent_id"] = self.agent_id
        metadata["authfi_run_id"] = self.run_id
        return AuthFIChatAnthropic.on_gateway(self.gateway, model=model, metadata=metadata or None, **kwargs)

    def export_steps(self, *, capture_content: bool = False, **options: Any) -> TracerProvider:
        """Send this agent's steps to AuthFI: LangChain's own tracing, switched to OpenTelemetry (extra ``otel``).

        LangSmith's client turns each run into spans on the OpenTelemetry provider, where the AuthFI exporter
        (see :func:`authfi.otel.register`) sends them with the agent's current token. Nothing goes to
        LangSmith, and runs carry no content unless ``capture_content``. Call it once, before the first run.
        """
        from authfi import otel

        provider = otel.register(self.source, capture_content=capture_content, **options)
        configure_langsmith(capture_content=capture_content, tracing_mode="otel", otel_tracer_provider=provider)
        langsmith.configure(enabled=True)
        return provider

    @staticmethod
    def configure_tracing(*, capture_content: bool = False, **client_kwargs: Any) -> langsmith.Client:
        """Have LangSmith record runs without their content, unless ``capture_content``.

        Prompts, answers and tool results hold people's data; AuthFI's default is that a trace
        carries names, timings and ids only. This installs the LangSmith client LangChain traces
        with, ``hide_inputs`` and ``hide_outputs`` set, and returns it. Turning content on is the
        tenant's policy decision. ``client_kwargs`` go to ``langsmith.Client``.
        """
        client: langsmith.Client = configure_langsmith(capture_content=capture_content, **client_kwargs)
        return client
