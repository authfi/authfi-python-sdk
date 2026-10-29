"""``authfi.connect()``: the credential the environment names, ready to build framework integrations on."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Unpack

import httpx

from authfi.agents.agent import AgentAuth, AsyncAgentAuth
from authfi.client import AsyncAuthFI, AuthFI, AuthFIOptions
from authfi.errors import ConfigurationError
from authfi.machine import AsyncClientCredentials, ClientCredentials

if TYPE_CHECKING:
    from authfi._frameworks import FrameworkCredential

__all__ = ["connect", "connect_async"]


def _names_agent(env: Mapping[str, str]) -> bool:
    return bool(env.get("AUTHFI_AGENT_ID", "").strip())


def _names_client(env: Mapping[str, str]) -> bool:
    return bool(env.get("AUTHFI_CLIENT_ID", "").strip())


def _nothing_named() -> ConfigurationError:
    return ConfigurationError(
        "no credential in the environment: set AUTHFI_AGENT_ID with AUTHFI_AGENT_KEY_FILE (an agent), or "
        "AUTHFI_CLIENT_ID with AUTHFI_CLIENT_SECRET (a machine client)"
    )


def _started(credential: FrameworkCredential, run_id: str | None) -> None:
    if run_id is not None:
        credential.start_run(run_id)


def connect(
    env: Mapping[str, str] | None = None,
    *,
    resource: str | None = None,
    run_id: str | None = None,
    http_client: httpx.Client | None = None,
    **options: Unpack[AuthFIOptions],
) -> AgentAuth | ClientCredentials:
    """The credential the environment names, for blocking code.

    ``AUTHFI_ISSUER`` names the project. Then ``AUTHFI_AGENT_ID`` with ``AUTHFI_AGENT_KEY_FILE`` (or
    ``AUTHFI_AGENT_PRIVATE_KEY``) is an agent, which wins when both are set; otherwise ``AUTHFI_CLIENT_ID``
    with ``AUTHFI_CLIENT_SECRET`` is a machine client. A missing variable is a
    :class:`~authfi.ConfigurationError` that names it. ``run_id`` starts the run under that id.

    The result carries ``langchain()``, ``langgraph()``, ``openai_agents()``, ``claude_agent()`` and
    ``adk()``, and ``.project`` to close when you are done. The frameworks that run on asyncio need
    :func:`connect_async`.
    """
    source = os.environ if env is None else env
    project = AuthFI.from_env(source, http_client=http_client, **options)
    try:
        credential: AgentAuth | ClientCredentials
        if _names_agent(source):
            credential = project.agent_from_env(source, resource=resource)
        elif _names_client(source):
            credential = project.client_credentials_from_env(source, resource=resource)
        else:
            raise _nothing_named()
        _started(credential, run_id)
    except BaseException:
        project.close()
        raise
    return credential


def connect_async(
    env: Mapping[str, str] | None = None,
    *,
    resource: str | None = None,
    run_id: str | None = None,
    http_client: httpx.AsyncClient | None = None,
    **options: Unpack[AuthFIOptions],
) -> AsyncAgentAuth | AsyncClientCredentials:
    """:func:`connect` for asyncio code: the same environment, the asyncio credential.

    It returns at once (nothing is awaited until the credential is used), so a plain ``def`` can call it;
    close the project with ``await credential.project.aclose()``.
    """
    source = os.environ if env is None else env
    project = AsyncAuthFI.from_env(source, http_client=http_client, **options)
    credential: AsyncAgentAuth | AsyncClientCredentials
    if _names_agent(source):
        credential = project.agent_from_env(source, resource=resource)
    elif _names_client(source):
        credential = project.client_credentials_from_env(source, resource=resource)
    else:
        raise _nothing_named()
    _started(credential, run_id)
    return credential
