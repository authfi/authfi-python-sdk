"""Start the framework script as an instance of the bound agent, or say what is missing and stop.

The pod is given, by the Job: AUTHFI_ISSUER, AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE (a projected token whose audience is
the issuer), AUTHFI_AGENT_ID, AUTHFI_CONFIG_VERSION and AUTHFI_CONFIG_DIGEST (the last three from the ConfigMap a person
fills after registering the agent). This refuses to start when one is empty, or when a key file is named, so a pod can
never fall back to the older key-file sign-in and a run that proves the instance path really did use it.

Exit 3 is "something to supply", as everywhere in this folder. No value is printed, only the names that are missing.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FRAMEWORK = os.environ.get("FW_FRAMEWORK", "")
SCRIPT = {
    "langgraph": "langgraph/agent.py",
    "openai-agents": "openai-agents/agent.py",
    "google-adk": "google-adk/agent.py",
}
#: Other scripts a framework image may run, chosen by FW_SCRIPT (a name, never a path). The default is SCRIPT.
ALTERNATE = {"langgraph": {"demo": "langgraph/demo_agent.py"}}
REQUIRED = (
    "AUTHFI_ISSUER",
    "AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE",
    "AUTHFI_AGENT_ID",
    "AUTHFI_CONFIG_VERSION",
    "AUTHFI_CONFIG_DIGEST",
)
CONFIG_NAMES = ("AUTHFI_CONFIG_VERSION", "AUTHFI_CONFIG_DIGEST")


def observing(env: dict[str, str]) -> bool:
    """True for the first, observed run of a new agent: it names no approved version yet."""
    return env.get("AUTHFI_CONFIG_OBSERVE", "").strip() == "1"


def script_for(env: dict[str, str]) -> str | None:
    """The script FW_SCRIPT selects (empty selects the framework's own), or None when the name is not allowed."""
    name = env.get("FW_SCRIPT", "").strip()
    if not name:
        return SCRIPT.get(FRAMEWORK)
    return ALTERNATE.get(FRAMEWORK, {}).get(name)


def problems(env: dict[str, str]) -> list[str]:
    """What stops this pod from signing in as an instance. Names only, never values."""
    out: list[str] = []
    if FRAMEWORK not in SCRIPT:
        out.append("FW_FRAMEWORK is %r, not one of %s" % (FRAMEWORK, ", ".join(sorted(SCRIPT))))
    if env.get("FW_SCRIPT", "").strip() and script_for(env) is None:
        out.append("FW_SCRIPT is not one of %s" % ", ".join(sorted(ALTERNATE.get(FRAMEWORK, {}))))
    if observing(env):
        out += [
            "%s is set while AUTHFI_CONFIG_OBSERVE=1 (an observed run names no version)" % n
            for n in CONFIG_NAMES
            if env.get(n, "").strip()
        ]
        required = tuple(n for n in REQUIRED if n not in CONFIG_NAMES)
    else:
        required = REQUIRED
    out += ["%s is empty or not set" % n for n in required if not env.get(n, "").strip()]
    if env.get("AUTHFI_AGENT_KEY_FILE", "").strip() or env.get("AUTHFI_AGENT_PRIVATE_KEY", "").strip():
        out.append("an agent key is named; a pod signs in as an instance and holds no key")
    token = env.get("AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE", "").strip()
    if token and not Path(token).is_file():
        out.append("the projected service-account token is not at the path named")
    if not observing(env) and not env.get("AUTHFI_CONFIG_VERSION", "0").strip().isdigit():
        out.append("AUTHFI_CONFIG_VERSION is not a version number")
    digest = env.get("AUTHFI_CONFIG_DIGEST", "").strip()
    if digest and not observing(env) and not digest.startswith("sha256:"):
        out.append("AUTHFI_CONFIG_DIGEST is not sha256:<hex>")
    return out


def main() -> int:
    found = problems(dict(os.environ))
    if found:
        print("BLOCKED: this pod cannot sign in as an instance of the agent:")
        for line in found:
            print("  - " + line)
        print("Fill agent.id, agent.configVersion and agent.configDigest (from registering the agent) and run again.")
        return 3
    script = Path("/app") / (script_for(dict(os.environ)) or SCRIPT[FRAMEWORK])
    os.execv(sys.executable, [sys.executable, "-B", "-I", str(script)])
    return 1  # execv does not return


if __name__ == "__main__":
    sys.exit(main())
