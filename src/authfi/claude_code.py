"""Claude Code, as a person runs it in a terminal, through the AuthFI gateways and never Anthropic's sign-in.

::

    authfi claude [--model claude-sonnet-5-5] [--small-model claude-haiku-5-5] [-- <claude arguments>]

It signs in as the agent the environment names (:func:`authfi.connect_async`: a person's personal-assistant agent
bound to their device, ``AUTHFI_ISSUER``, ``AUTHFI_AGENT_ID`` and the device's evidence), starts the loopback model
proxy that :mod:`authfi.claude_agent` uses, and runs the interactive ``claude`` with its model calls pointed at the
proxy. Every call leaves this process with the agent's credential, bound to its key and proved on each request, for
the project's AI gateway (``<issuer>/llm``). Claude Code is given a placeholder key that the proxy drops; it never
holds the agent's key or token, and no Anthropic account, subscription or API key is used.

- The person's own ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or ``ANTHROPIC_BASE_URL`` is overridden, and named
  on stderr; the ``AUTHFI_*`` variables stay in this process and are not given to Claude Code or the tools it runs.
- Non-essential traffic is off, so Claude Code sends nothing to Anthropic: no telemetry, error reports or update
  checks.
- The model is whatever the gateway serves the agent: ``--model`` is the session's starting one, and ``/model`` in
  Claude Code switches among the rest. ``--small-model`` is the one Claude Code uses for its own background work.
- This process stays up while Claude Code runs, because the proxy lives here; Ctrl-C reaches Claude Code, not it.

The gateway's tools are not given to Claude Code by this command yet; Claude Code's own tools run as usual.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence

from authfi._adapters import async_gateway_for
from authfi._claude_env import ATTRIBUTION_HEADER_SWITCH, FIXED_CREDENTIALS, PLACEHOLDER_KEY_COMMAND, model_env
from authfi._connect import connect_async
from authfi._model_proxy import ModelProxy
from authfi.agents.agent import AsyncAgentAuth
from authfi.errors import AuthFIError, ConfigurationError, GatewayTokenRefused

__all__ = ["DEFAULT_MODEL", "Spawn", "child_environment", "command", "run"]

#: The model a session starts on, and the one Claude Code uses for its own background work: the cheapest Claude.
DEFAULT_MODEL = "claude-haiku-5-5"

#: Runs ``argv`` with the environment ``env``, the terminal inherited, and returns its exit code.
Spawn = Callable[[Sequence[str], Mapping[str, str]], Awaitable[int]]

#: Set by the person and overridden here, so named when they had a value.
_OVERRIDDEN = (*FIXED_CREDENTIALS, "ANTHROPIC_BASE_URL")


def child_environment(base: Mapping[str, str], proxy_url: str, small_model: str) -> tuple[dict[str, str], list[str]]:
    """Claude Code's environment, and the names of the person's variables it overrides.

    ``base`` is this process's environment. The ``AUTHFI_*`` variables are left out: they name the agent and its
    evidence, which Claude Code and the tools it runs have no use for. Claude Code's attribution header is off
    unless the person set it.
    """
    overridden = sorted(name for name in base if name.upper() in _OVERRIDDEN and base[name])
    env = {name: value for name, value in base.items() if not name.upper().startswith("AUTHFI_")}
    env.setdefault(ATTRIBUTION_HEADER_SWITCH, "0")
    env.update(model_env(proxy_url, small_model))
    return env, overridden


def _to_stderr(text: str) -> None:
    print(text, file=sys.stderr)


async def _spawn(argv: Sequence[str], env: Mapping[str, str]) -> int:
    process = await asyncio.create_subprocess_exec(*argv, env=dict(env))
    # Ctrl-C is Claude Code's, which shares the terminal: this process must outlive it, as the proxy lives here.
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        return await process.wait()
    finally:
        signal.signal(signal.SIGINT, previous)


async def run(
    args: Sequence[str] = (),
    *,
    model: str = DEFAULT_MODEL,
    small_model: str = DEFAULT_MODEL,
    env: Mapping[str, str] | None = None,
    claude: str | None = None,
    spawn: Spawn | None = None,
    warn: Callable[[str], None] | None = None,
) -> int:
    """Run Claude Code with ``args`` through the AuthFI gateways, as the agent ``env`` names; returns its exit code."""
    base = dict(os.environ if env is None else env)
    say: Callable[[str], None] = warn or _to_stderr
    if not model.strip() or not small_model.strip():
        raise ConfigurationError("the model names cannot be empty")
    binary = claude or shutil.which("claude", path=base.get("PATH"))
    if not binary:
        raise ConfigurationError("Claude Code (`claude`) is not on PATH: install it, then run this again")
    credential = connect_async(base)
    try:
        if not isinstance(credential, AsyncAgentAuth):
            raise ConfigurationError(
                "the environment names a machine client, not an agent: Claude Code runs as the person's own "
                "agent (AUTHFI_AGENT_ID and its device's evidence)"
            )
        refused: list[GatewayTokenRefused] = []
        proxy = ModelProxy(async_gateway_for(credential), on_refused=refused.append, on_run=lambda _run: None)
        try:
            proxy_url = await proxy.start()
            child, overridden = child_environment(base, proxy_url, small_model)
            if overridden:
                say(
                    f"authfi: {', '.join(overridden)} is set; Claude Code is not given it: its model calls go "
                    "through the AuthFI gateway"
                )
            settings = json.dumps({"apiKeyHelper": PLACEHOLDER_KEY_COMMAND})
            code = await (spawn or _spawn)([binary, "--settings", settings, "--model", model, *args], child)
        finally:
            await proxy.aclose()
        if refused:
            say(f"authfi: the gateway refused the agent's token: {refused[0].message}")
        return code
    finally:
        await credential.project.aclose()


def command(args: Sequence[str], *, model: str, small_model: str) -> int:
    """``authfi claude``: :func:`run`, with a failure printed on stderr and exit code 1."""
    try:
        return asyncio.run(run(args, model=model, small_model=small_model))
    except AuthFIError as error:
        print(f"authfi: {error.message}", file=sys.stderr)
        return 1
