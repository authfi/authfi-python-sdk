# Acting for a caller with no browser

A voice agent refunds an order for a caller, after the caller approves on their own phone.

## What it shows

- **The binding message is the safety line.** It is what the caller reads ("Refund order 4411")
  on their own device, so a cloned voice on the line cannot approve something else.
- **No code is ever read back.** The agent never sees or relays one.
- **A decision is an answer.** A refusal raises `ConsentDenied`, a request that runs out
  `ConsentExpired`; a wait that times out raises `WaitTimeout` and leaves the request open.
- **The token acts for the caller.** Its subject is the caller and its actor the agent, so the
  orders API sees both, and it is never renewed: approving again is the caller's choice.

## Run it

```bash
pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python voice_refund.py priya@acme.com 4411
```

The caller must be a person in your project, and the agent must be allowed to act for people.
