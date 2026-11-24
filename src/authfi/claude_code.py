"""Claude Code, as a person runs it in a terminal, through the AuthFI gateways and never Anthropic's sign-in.

::

    authfi claude [--model claude-sonnet-5-5] [--small-model claude-haiku-5-5] [-- <claude arguments>]

It signs in as the agent the environment names (:func:`authfi.connect_async`: a person's personal-assistant agent
bound to their device, ``AUTHFI_ISSUER``, ``AUTHFI_AGENT_ID`` and the device's evidence), starts the loopback model
proxy that :mod:`authfi.claude_agent` uses, and runs the interactive ``claude`` with its model calls pointed at the
proxy. Every call leaves this process with the agent's credential, bound to its key and proved on each request, for
the project's AI gateway (its model routes under ``<issuer>/mcp/v1``). Claude Code is given a placeholder key that the
proxy drops; it never holds the agent's key or token, and no Anthropic account, subscription or API key is used.

Claude Code also asks for ``POST /v1/messages/count_tokens``, ``GET /v1/models`` (only with gateway model discovery on)
and ``HEAD /api/hello``. The gateway's model door serves none of them, so the proxy answers each 404 itself, unsent:
Claude Code then estimates token counts and keeps its own model list.

- The person's own ``ANTHROPIC_API_KEY``, ``ANTHROPIC_AUTH_TOKEN`` or ``ANTHROPIC_BASE_URL`` is overridden, and named
  on stderr; the ``AUTHFI_*`` variables stay in this process and are not given to Claude Code or the tools it runs.
- Non-essential traffic is off, so Claude Code sends nothing to Anthropic: no telemetry, error reports or update
  checks.
- The model is whatever the gateway serves the agent: ``--model`` is the session's starting one, and ``/model`` in
  Claude Code switches among the rest. ``--small-model`` is the one Claude Code uses for its own background work.
- This process stays up while Claude Code runs, because the proxy lives here; Ctrl-C reaches Claude Code, not it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence

from authfi import _launch
from authfi._claude_env import ATTRIBUTION_HEADER_SWITCH, FIXED_CREDENTIALS, PLACEHOLDER_KEY_COMMAND, model_env
from authfi._launch import Spawn
from authfi.errors import AuthFIError, ConfigurationError

__all__ = ["DEFAULT_MODEL", "Spawn", "child_environment", "command", "run"]

#: The model a session starts on, and the one Claude Code uses for its own background work: the cheapest Claude.
DEFAULT_MODEL = "claude-haiku-5-5"

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
    say: Callable[[str], None] = warn or _launch.to_stderr
    if not model.strip() or not small_model.strip():
        raise ConfigurationError("the model names cannot be empty")
    binary = claude or shutil.which("claude", path=base.get("PATH"))
    if not binary:
        raise ConfigurationError("Claude Code (`claude`) is not on PATH: install it, then run this again")

    async def launch(proxy_url: str) -> int:
        child, overridden = child_environment(base, proxy_url, small_model)
        if overridden:
            say(
                f"authfi: {', '.join(overridden)} is set; Claude Code is not given it: its model calls go "
                "through the AuthFI gateway"
            )
        settings = json.dumps({"apiKeyHelper": PLACEHOLDER_KEY_COMMAND})
        return await (spawn or _launch.spawn)([binary, "--settings", settings, "--model", model, *args], child)

    return await _launch.serve(base, "anthropic", say, launch)


def command(args: Sequence[str], *, model: str, small_model: str) -> int:
    """``authfi claude``: :func:`run`, with a failure printed on stderr and exit code 1."""
    try:
        return asyncio.run(run(args, model=model, small_model=small_model))
    except AuthFIError as error:
        print(f"authfi: {error.message}", file=sys.stderr)
        return 1
