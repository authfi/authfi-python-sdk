"""Any coding tool, through the AuthFI gateway's model routes, as the agent the environment names.

::

    authfi run [--format openai|anthropic] -- <tool> [arguments]

The same launch as :mod:`authfi.claude_code` for a tool that is not Claude Code: it signs in as the agent
(:func:`authfi.connect_async`), serves the model calls of one API format on loopback, and runs the tool with the
variables that point its provider SDK there. Every call leaves this process with the agent's credential, bound to its
key and proved on each request; the tool holds only a placeholder key that the proxy drops.

====================  =============================================  ==============================
format                the variables set                              the calls forwarded
====================  =============================================  ==============================
``openai`` (default)  ``OPENAI_BASE_URL``, ``OPENAI_API_BASE``,     ``POST /v1/chat/completions``,
                      ``OPENAI_API_KEY``                             ``POST /v1/responses``
``anthropic``         ``ANTHROPIC_BASE_URL``, ``ANTHROPIC_API_KEY``, ``POST /v1/messages``
                      ``ANTHROPIC_AUTH_TOKEN`` (empty)
====================  =============================================  ==============================

- The person's own value of any of those variables is overridden, and named on stderr. The ``AUTHFI_*`` variables stay
  in this process and are not given to the tool or the programs it runs.
- A tool that reads its base URL from a configuration file rather than the environment can be given it on the command
  line: ``{base_url}`` in an argument is replaced by the value ``OPENAI_BASE_URL`` (or ``ANTHROPIC_BASE_URL``) has.
- This process stays up while the tool runs, because the proxy lives here; Ctrl-C reaches the tool, not it.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence

from authfi import _launch
from authfi._launch import Spawn
from authfi.errors import AuthFIError, ConfigurationError

__all__ = ["FORMATS", "PLACEHOLDER_KEY", "Spawn", "base_url_for", "child_environment", "command", "run"]

FORMATS = ("openai", "anthropic")

#: The key the tool is given. It is not a credential: the proxy drops it and sends the call with this process's own.
PLACEHOLDER_KEY = "authfi-loopback"

#: The variables each format sets, and the placeholder each key variable holds. The base URL is the first.
_BASE_URL = {"openai": "OPENAI_BASE_URL", "anthropic": "ANTHROPIC_BASE_URL"}
_VALUES = {
    "openai": ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_API_KEY"),
    "anthropic": ("ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
}
_TEMPLATE = "{base_url}"


def base_url_for(api: str, proxy_url: str) -> str:
    """What the tool's base-URL variable holds: OpenAI SDKs expect the ``/v1`` in it, Anthropic's add their own."""
    return proxy_url + "/v1" if api == "openai" else proxy_url


def child_environment(base: Mapping[str, str], proxy_url: str, api: str) -> tuple[dict[str, str], list[str]]:
    """The tool's environment, and the names of the person's variables it overrides.

    ``base`` is this process's environment. The ``AUTHFI_*`` variables are left out: they name the agent and its
    evidence, which the tool and the programs it runs have no use for.
    """
    if api not in FORMATS:
        raise ConfigurationError(f"the format is one of {', '.join(FORMATS)}, not {api!r}")
    url = base_url_for(api, proxy_url)
    given = {
        "openai": {"OPENAI_BASE_URL": url, "OPENAI_API_BASE": url, "OPENAI_API_KEY": PLACEHOLDER_KEY},
        "anthropic": {"ANTHROPIC_BASE_URL": url, "ANTHROPIC_API_KEY": PLACEHOLDER_KEY, "ANTHROPIC_AUTH_TOKEN": ""},
    }[api]
    ours = {name.upper() for name in _VALUES[api]}
    overridden = sorted(name for name in base if name.upper() in ours and base[name])
    # A name differing only in case (Windows keeps the person's spelling) must not survive beside ours.
    env = {
        name: value
        for name, value in base.items()
        if not name.upper().startswith("AUTHFI_") and name.upper() not in ours
    }
    env.update(given)
    return env, overridden


async def run(
    argv: Sequence[str],
    *,
    api: str = "openai",
    env: Mapping[str, str] | None = None,
    spawn: Spawn | None = None,
    warn: Callable[[str], None] | None = None,
) -> int:
    """Run ``argv`` (the tool and its arguments) through the AuthFI gateway; returns its exit code."""
    if api not in FORMATS:
        raise ConfigurationError(f"the format is one of {', '.join(FORMATS)}, not {api!r}")
    if not argv:
        raise ConfigurationError("name the tool to run: authfi run [--format openai|anthropic] -- <tool> [arguments]")
    base = dict(os.environ if env is None else env)
    say: Callable[[str], None] = warn or _launch.to_stderr
    tool = shutil.which(argv[0], path=base.get("PATH"))
    if not tool:
        raise ConfigurationError(f"{argv[0]!r} is not on PATH: install it, then run this again")

    async def launch(proxy_url: str) -> int:
        child, overridden = child_environment(base, proxy_url, api)
        if overridden:
            say(
                f"authfi: {', '.join(overridden)} is set; {argv[0]} is not given it: its model calls go "
                "through the AuthFI gateway"
            )
        url = base_url_for(api, proxy_url)
        arguments = [argument.replace(_TEMPLATE, url) for argument in argv[1:]]
        return await (spawn or _launch.spawn)([tool, *arguments], child)

    return await _launch.serve(base, api, say, launch)


def command(argv: Sequence[str], *, api: str) -> int:
    """``authfi run``: :func:`run`, with a failure printed on stderr and exit code 1."""
    try:
        return asyncio.run(run(argv, api=api))
    except AuthFIError as error:
        print(f"authfi: {error.message}", file=sys.stderr)
        return 1
