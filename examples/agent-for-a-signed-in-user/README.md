# Acting for a person signed in to your app

An assistant inside your app refunds an order for the person who asked it, with a token that says
both who the person is and that the assistant acted.

## What it shows

- **Token exchange.** The person's own access token goes in; a token whose `sub` is the person and
  `act.sub` the assistant comes out. The agent proves who it is with its own token; no client
  secret is sent.
- **The project decides.** Whether this assistant may act for this person is your project's
  policy; until it allows it, the exchange is refused `access_denied`.
- **Renewal follows the person.** The exchanged token is exchanged again near its expiry, and
  stops when the person's own token has expired.
- **The raw token is wrapped.** A token the SDK did not make is passed as `Secret(...)`, so it
  never prints, and the wrap shows in review.

## Run it

```bash
pip install "authfi[fastapi] @ git+https://github.com/authfi/authfi-python-sdk@main" uvicorn
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
uvicorn assistant_api:app
```
