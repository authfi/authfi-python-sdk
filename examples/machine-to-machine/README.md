# Machine to machine

A reporting job that calls the orders API (see `../protect-an-api`) with its own token.

## What it shows

- **`resource` is the API's client id.** The token's audience is then the API, and the API's
  verifier accepts it. A token minted without `resource` names the client itself, which no API
  should accept; the SDK logs a warning when you ask for one.
- **`send_to` is where the token may go.** Any request to another origin is refused before the
  token leaves the process.
- **Reuse.** The token is cached and renewed at two thirds of its lifetime. Threads that need it
  at the same moment share one request to AuthFI.

## Run it

```bash
pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python nightly_job.py
```

In the AuthFI console, register the job as a machine-to-machine application, allow it the orders
API as an audience, and bind it a role that holds `orders:read`.
