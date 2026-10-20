"""What the framework adapters share: the project behind a credential, and which httpx a client needs."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import TYPE_CHECKING, Any

from authfi.errors import ConfigurationError
from authfi.gateway import AsyncGateway, Gateway
from authfi.tokens import AsyncTokenSource, TokenSource

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = ["PLACEHOLDER_KEY", "agent_id_of", "async_gateway_for", "configure_langsmith", "gateway_for", "httpx_of"]

#: Stands in where a provider SDK insists on an API key. The gateway reads the agent's token from
#: the Authorization header, which the gateway's HTTP client sets on every request; this value is
#: never a credential anywhere.
PLACEHOLDER_KEY = "authfi-sets-the-token-per-request"


def _project(source: object, project: object) -> Any:
    chosen = project if project is not None else getattr(source, "project", None)
    if chosen is None:
        raise ConfigurationError("this credential does not name its AuthFI project: pass project=af")
    return chosen


def gateway_for(
    source: TokenSource | AsyncTokenSource, project: AuthFI | AsyncAuthFI | None = None
) -> Gateway | AsyncGateway:
    """The gateway for a credential: a :class:`Gateway` for a blocking one, an :class:`AsyncGateway` for asyncio."""
    if isinstance(source, TokenSource):
        return Gateway(_project(source, project), source)
    if isinstance(source, AsyncTokenSource):  # pyright: ignore[reportUnnecessaryIsInstance]
        return AsyncGateway(_project(source, project), source)
    raise ConfigurationError(
        f"{type(source).__name__} is not an AuthFI credential: pass the agent (af.agent(...)) or a token it holds"
    )


def async_gateway_for(source: AsyncTokenSource, project: AsyncAuthFI | None = None) -> AsyncGateway:
    """The gateway for a framework that runs on asyncio only."""
    if not isinstance(source, AsyncTokenSource):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ConfigurationError(
            "this framework runs on asyncio: build the credential from AsyncAuthFI (an AsyncAgentAuth, or the "
            "AsyncDelegatedToken it returns)"
        )
    return AsyncGateway(_project(source, project), source)


def agent_id_of(source: object) -> str | None:
    """The agent behind a credential: its own id, or the actor of a token it holds for a person."""
    agent_id = getattr(source, "agent_id", None)
    if isinstance(agent_id, str):
        return agent_id
    actor = getattr(source, "actor", None)
    actor_id = getattr(actor, "id", None)
    return actor_id if isinstance(actor_id, str) and actor_id else None


def httpx_of(module_name: str) -> ModuleType:
    """The httpx package a library's HTTP layer is built on: ``httpx2`` for the newer provider SDKs."""
    module = importlib.import_module(module_name)
    for name in ("httpx2", "httpx"):
        found = getattr(module, name, None)
        if isinstance(found, ModuleType):
            return found
    return importlib.import_module("httpx")


def configure_langsmith(*, capture_content: bool, **client_kwargs: Any) -> Any:
    """Install the LangSmith client traces go to, hiding inputs and outputs unless ``capture_content``."""
    import langsmith

    hide = not capture_content
    client = langsmith.Client(hide_inputs=hide, hide_outputs=hide, **client_kwargs)
    langsmith.configure(client=client)
    return client
