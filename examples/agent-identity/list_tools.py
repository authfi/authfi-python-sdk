"""An agent with its own identity lists the tools the AuthFI gateway lets it call.

Run it::

    pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
    authfi keygen --out agent.key            # register the printed public JWK in the console
    cp .env.example .env                     # then fill it in
    python list_tools.py
"""

from __future__ import annotations

import httpx

from authfi import AuthFI


def main() -> None:
    """Print the tools this agent may call."""
    with AuthFI.from_env() as af:  # AUTHFI_ISSUER
        agent = af.agent_from_env()  # AUTHFI_AGENT_ID, AUTHFI_AGENT_KEY_FILE, AUTHFI_AGENT_KID
        print("capabilities:", sorted(agent.capabilities()))
        # The token is attached to each request, and only to AuthFI's own origin.
        with httpx.Client(auth=agent.httpx_auth(), timeout=30) as client:
            answer = client.post(
                af.issuer + "/mcp",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={"Accept": "application/json, text/event-stream"},
            )
            answer.raise_for_status()
            for tool in answer.json()["result"]["tools"]:
                print(tool["name"])


if __name__ == "__main__":
    main()
