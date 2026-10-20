# Approvals in a Google ADK agent

A Google ADK agent calls Claude and the gateway's governed tools. A call that needs a person's
approval becomes ADK's own tool confirmation; it is answered once the person has decided in AuthFI.

## What it shows

- **The token on every call, outside the traces.** The model and the tools carry the agent's
  current token; ADK's spans hold neither the token nor, by default, any content.
- **No Google credential sent to AuthFI.** ADK's Google mTLS client is kept off the gateway's
  connection.
- **Approval through ADK's tool confirmation.** The run pauses on a held call and resumes with the
  answer; the gateway decides the call.

## Run it

```bash
pip install "authfi[adk] @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python refund_agent.py "Read the wiki contents of pallets/flask"
```
