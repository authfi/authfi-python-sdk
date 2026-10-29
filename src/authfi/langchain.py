"""LangChain: the AuthFI AI gateway for any chat model, with a Claude model ready made.

``pip install "authfi[langchain] @ git+https://github.com/authfi/authfi-python-sdk@main"``

::

    from authfi import AuthFI
    from authfi.langchain import LangChainIntegration

    agent = AuthFI.from_env().agent_from_env()
    langchain = LangChainIntegration(agent)
    langchain.configure_tracing()                 # LangSmith records metadata, not content
    model = langchain.chat_model("claude-haiku-5-5")   # needs the langchain-anthropic extra

``LangChainIntegration.gateway`` is the credential for any chat model that takes an HTTP client; ``chat_model``
makes a Claude one (see :mod:`authfi.langchain_anthropic`, which needs ``authfi[langchain-anthropic]``). Only
``langchain-core`` is needed for the rest, and LangSmith is used only when it is installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from authfi._adapters import agent_id_of, configure_langsmith, gateway_for
from authfi.errors import ConfigurationError
from authfi.tokens import AsyncTokenSource, TokenSource

try:
    import langchain_core  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "authfi.langchain needs langchain-core: "
        "pip install 'authfi[langchain] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI, AuthFI
    from authfi.langchain_anthropic import AuthFIChatAnthropic

__all__ = ["AuthFIChatAnthropic", "LangChainIntegration"]


def __getattr__(name: str) -> Any:
    """Keep ``from authfi.langchain import AuthFIChatAnthropic`` working, loading langchain-anthropic only then."""
    if name == "AuthFIChatAnthropic":
        from authfi.langchain_anthropic import AuthFIChatAnthropic

        return AuthFIChatAnthropic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
        from authfi.langchain_anthropic import AuthFIChatAnthropic

        return AuthFIChatAnthropic.on_gateway(self.gateway, model=model, metadata=metadata or None, **kwargs)

    @staticmethod
    def configure_tracing(*, capture_content: bool = False, **client_kwargs: Any) -> Any:
        """Have LangSmith record runs without their content, unless ``capture_content``.

        Prompts, answers and tool results hold people's data; AuthFI's default is that a trace
        carries names, timings and ids only. This installs the LangSmith client LangChain traces
        with, ``hide_inputs`` and ``hide_outputs`` set, and returns it. Turning content on is the
        tenant's policy decision. ``client_kwargs`` go to ``langsmith.Client``. Without LangSmith
        installed nothing is traced to it, so there is nothing to hide: this returns ``None``.
        """
        return configure_langsmith(capture_content=capture_content, **client_kwargs)
