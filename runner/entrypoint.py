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
REQUIRED = (
    "AUTHFI_ISSUER",
    "AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE",
    "AUTHFI_AGENT_ID",
    "AUTHFI_CONFIG_VERSION",
    "AUTHFI_CONFIG_DIGEST",
)


def problems(env: dict[str, str]) -> list[str]:
    """What stops this pod from signing in as an instance. Names only, never values."""
    out: list[str] = []
    if FRAMEWORK not in SCRIPT:
        out.append("FW_FRAMEWORK is %r, not one of %s" % (FRAMEWORK, ", ".join(sorted(SCRIPT))))
    out += ["%s is empty or not set" % n for n in REQUIRED if not env.get(n, "").strip()]
    if env.get("AUTHFI_AGENT_KEY_FILE", "").strip() or env.get("AUTHFI_AGENT_PRIVATE_KEY", "").strip():
        out.append("an agent key is named; a pod signs in as an instance and holds no key")
    token = env.get("AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE", "").strip()
    if token and not Path(token).is_file():
        out.append("the projected service-account token is not at the path named")
    if not env.get("AUTHFI_CONFIG_VERSION", "0").strip().isdigit():
        out.append("AUTHFI_CONFIG_VERSION is not a version number")
    digest = env.get("AUTHFI_CONFIG_DIGEST", "").strip()
    if digest and not digest.startswith("sha256:"):
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
    script = Path("/app") / SCRIPT[FRAMEWORK]
    os.execv(sys.executable, [sys.executable, "-B", "-I", str(script)])
    return 1  # execv does not return


if __name__ == "__main__":
    sys.exit(main())
