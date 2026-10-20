# Agent identity

An AI agent with its own key, its own token, and the AuthFI gateway.

## What it shows

- **The key is the credential.** `authfi keygen` writes a private key only you can read and prints
  its public half. You register the public half; the private half never leaves the machine.
- **No token handling.** The agent signs a short assertion to get its token, reuses it, and
  renews it before it expires. `agent.httpx_auth()` puts the current token on every request.
- **The token goes only to AuthFI.** A request to any other origin is refused before the token
  leaves the process.

## Run it

```bash
pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
authfi keygen --out agent.key
```

In the AuthFI console, register an agent with the printed public JWK, and grant it a tool. Then:

```bash
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python list_tools.py
```
