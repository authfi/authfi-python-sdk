"""The environment that points Claude Code's model calls at a loopback proxy, shared by every way the SDK runs it.

Claude Code is run in two ways: by the Claude Agent SDK (:mod:`authfi.claude_agent`) and interactively by a person
(:mod:`authfi.claude_code`). Both give it the same few settings, so they live here once.
"""

from __future__ import annotations

#: The key Claude Code is given for the AI gateway. It is not a credential: the loopback proxy drops it and sends
#: the call with this process's own. Claude Code needs some key to start, and ``echo`` prints one in every shell.
PLACEHOLDER_KEY_COMMAND = "echo authfi-loopback"

#: Claude Code's first system part is ``x-anthropic-billing-header: cc_version=...; cc_entrypoint=...``, and its
#: three-hex suffix changes with the first user message, so no two runs of one agent send the same system text. "0"
#: has Claude Code leave the part out.
ATTRIBUTION_HEADER_SWITCH = "CLAUDE_CODE_ATTRIBUTION_HEADER"

#: Environment that would hand Claude Code a fixed credential. Claude Code inherits its parent's environment, so
#: each is blanked in the environment it is given.
FIXED_CREDENTIALS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AUTHFI_AGENT_PRIVATE_KEY")


def model_env(proxy_url: str, small_model: str) -> dict[str, str]:
    """The settings that send every model call Claude Code makes to the proxy at ``proxy_url``, and nowhere else.

    ``small_model`` is the model Claude Code uses for its own background work (titles, summaries). The fixed
    credentials are blanked, non-essential traffic (telemetry, error reports, update checks) is off, so nothing
    goes to Anthropic, and Anthropic's experimental beta headers are not sent: the gateway serves the models the
    project configures.
    """
    return {
        **dict.fromkeys(FIXED_CREDENTIALS, ""),
        "ANTHROPIC_BASE_URL": proxy_url,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": small_model,
        "ANTHROPIC_SMALL_FAST_MODEL": small_model,
        "CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
