"""What every launcher shares: the agent the environment names, the loopback model proxy, and the program run behind it.

``authfi claude`` (:mod:`authfi.claude_code`) and ``authfi run`` (:mod:`authfi.launcher`) differ in the environment
they give the program, not in how it leaves: each is one process, so each launch signs in as the agent and makes its
own DPoP run key, and the gateway names a launch's calls as a run of their own.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence

from authfi._adapters import async_gateway_for
from authfi._connect import connect_async
from authfi._model_proxy import ModelProxy
from authfi.agents.agent import AsyncAgentAuth
from authfi.errors import ConfigurationError, GatewayTokenRefused

__all__ = ["Spawn", "serve", "spawn", "to_stderr"]

#: Runs ``argv`` with the environment ``env``, the terminal inherited, and returns its exit code.
Spawn = Callable[[Sequence[str], Mapping[str, str]], Awaitable[int]]


def to_stderr(text: str) -> None:
    print(text, file=sys.stderr)


async def spawn(argv: Sequence[str], env: Mapping[str, str]) -> int:
    """Run ``argv`` on this terminal; Ctrl-C is the program's, because the proxy lives in this process."""
    process = await asyncio.create_subprocess_exec(*argv, env=dict(env))
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return await process.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


async def serve(
    base: Mapping[str, str], api: str, say: Callable[[str], None], launch: Callable[[str], Awaitable[int]]
) -> int:
    """Sign in as the agent ``base`` names, serve its model calls on loopback, and run ``launch(proxy_url)``.

    ``api`` is the model API format the program speaks (``anthropic`` or ``openai``). The proxy is closed when
    ``launch`` returns; a refusal of the agent's token during the run is reported on ``say``.
    """
    credential = connect_async(base)
    try:
        if not isinstance(credential, AsyncAgentAuth):
            raise ConfigurationError(
                "the environment names a machine client, not an agent: the program runs as the person's own "
                "agent (AUTHFI_AGENT_ID and its device's evidence)"
            )
        refused: list[GatewayTokenRefused] = []
        proxy = ModelProxy(async_gateway_for(credential), on_refused=refused.append, on_run=lambda _run: None, api=api)
        try:
            code = await launch(await proxy.start())
        finally:
            await proxy.aclose()
        if refused:
            say(f"authfi: the gateway refused the agent's token: {refused[0].message}")
        return code
    finally:
        await credential.project.aclose()
