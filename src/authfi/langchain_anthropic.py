"""A Claude model for LangChain on the AuthFI AI gateway, whose token tracers never see.

``pip install "authfi[langchain-anthropic] @ git+https://github.com/authfi/authfi-python-sdk@main"``

Make the model with :meth:`authfi.langchain.LangChainIntegration.chat_model`. LangChain's documented way to give
``ChatAnthropic`` a bearer token is ``default_headers={"Authorization": ...}``, and LangSmith records a model's
constructor arguments in every run: the token went to the tracing backend with every call. Here the token is never
a model argument. The model's HTTP client is the gateway's, which reads the credential for each request below the
tracer.

``ChatAnthropic`` has no public setting for its HTTP client, so this overrides its ``_client`` and
``_async_client`` (checked when this module is imported; the extra pins the tested version range).
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, Self, TypeVar, cast

try:
    import anthropic
    from langchain_anthropic import ChatAnthropic
    from pydantic import PrivateAttr, SecretStr
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.langchain_anthropic needs langchain-anthropic: "
        "pip install 'authfi[langchain-anthropic] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

from authfi._adapters import PLACEHOLDER_KEY, httpx_of
from authfi.errors import ConfigurationError, IntegrationError
from authfi.gateway import AsyncGateway, Gateway, token_refusal

__all__ = ["AuthFIChatAnthropic"]

ClientT = TypeVar("ClientT")

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


def _without_propagation(client: ClientT) -> ClientT:
    """``client`` without trace propagation of its own, where the Anthropic SDK has it.

    Its newer versions put the ``traceparent`` of their own ``anthropic.messages.create`` span on every request, in
    place of any set earlier, and that span is parentless: the gateway's record would name it, not the run that made
    the call. The gateway's HTTP client sets the ``traceparent`` of the run instead (see ``_langsmith_steps``). The
    SDK's spans are kept.
    """
    options = getattr(client, "open_telemetry", None)
    if not isinstance(options, dict):
        return client
    return cast(
        "ClientT",
        cast("Any", client).with_options(open_telemetry={**cast("dict[str, Any]", options), "propagation": False}),
    )


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
        return _without_propagation(
            anthropic.Client(
                **self._authfi_params(), http_client=gateway.http_client(httpx_of("anthropic._base_client"))
            )
        )

    @cached_property
    def _async_client(self) -> anthropic.AsyncClient:
        gateway = self._authfi_gateway
        if not isinstance(gateway, AsyncGateway):
            raise ConfigurationError(
                "this model was made from a blocking credential: call it with invoke or stream, or make it from "
                "AsyncAuthFI"
            )
        return _without_propagation(
            anthropic.AsyncClient(
                **self._authfi_params(), http_client=gateway.http_client(httpx_of("anthropic._base_client"))
            )
        )
