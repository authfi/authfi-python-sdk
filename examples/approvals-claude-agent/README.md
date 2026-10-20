# Approvals in a Claude Agent SDK agent

Claude Code, driven by the Claude Agent SDK, runs as an AuthFI agent: its model calls and its tool
calls go through the AuthFI gateways, and a call that needs a person's approval waits in
`can_use_tool` until they decide.

## What it shows

- **No token in Claude Code's environment.** Claude Code gets the agent's token by running
  `python -m authfi token`, which signs with the agent's key file, and runs it again before the token
  expires.
- **Governed tools.** The gateway's tools reach Claude Code through an MCP server in this process,
  with the current token on every call.
- **Approval through Claude Code's own permission callback.** The held call waits in `can_use_tool`;
  once the person has decided, the gateway decides the call.

## Run it

```bash
pip install "authfi[claude-agent] @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in; the agent's key must be a file
set -a; . ./.env; set +a
python support_agent.py "Read the wiki contents of pallets/flask"
```
