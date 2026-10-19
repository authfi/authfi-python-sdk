# AuthFI Python SDK

Official Python SDK for [AuthFI](https://authfi.io) — the identity control plane.

## Install

```bash
pip install authfi
```

For RS256 signature verification (recommended for production):
```bash
pip install authfi PyJWT[crypto]
```

## Quick Start (Flask)

```python
from authfi import AuthFI

auth = AuthFI(tenant='acme', api_key='sk_live_...')

@app.route('/api/users')
@auth.require('read:users')
def get_users():
    user = auth.current_user()
    return jsonify(users)

# On startup
auth.sync()
```

## Quick Start (FastAPI)

```python
from authfi import AuthFI
from fastapi import Depends

auth = AuthFI(tenant='acme', api_key='sk_live_...')

@app.get('/api/users')
async def get_users(user=Depends(auth.require('read:users'))):
    return users
```

## Agent authentication (AuthFI Agent Profile)

An AuthFI agent holds its **own** Ed25519 keypair. It authenticates by signing a short
assertion (`private_key_jwt`, RFC 7523 §2.2) — the private half never leaves the process
and is never sent to AuthFI. There is no client secret to leak, because an agent
registered with a public JWK is issued none.

### 1. Make a key, register the public half

```python
from authfi import AgentKey

key = AgentKey.generate("k1")

open("/run/secrets/agent.key", "w").write(key.private_pem())   # chmod 600, never sent
print(key.public_jwk())   # {"kty":"OKP","crv":"Ed25519","x":"...","kid":"k1"}
```

Register the agent with that **public** JWK and its `kid`. If you build a JWK by hand,
run it through `assert_public_jwk(jwk)` first: a `d` where a public key belongs means the
private key has been sent somewhere, and the key is burned rather than fixable.

### 2. Authenticate

```python
from authfi import AgentAuth, AgentKey

agent = AgentAuth(
    agent_id="agt_...",
    key=AgentKey.from_file("/run/secrets/agent.key", key_id="k1"),
    platform_url="https://acme.authfi.io",   # your project's authfi.io address
    audience="https://api.acme.com",   # optional — RFC 8707 resource
)

if agent.can("send_email_external"):
    requests.post(url, headers=agent.auth_header(), json=payload)
```

Or from the environment — `AgentAuth.from_env()`:

| Variable | |
|---|---|
| `AUTHFI_AGENT_ID` | the agent id (`agt_…`) |
| `AUTHFI_AGENT_KEY_FILE` | path to the private key — **preferred** |
| `AUTHFI_AGENT_KID` | the registered `kid` |
| `AUTHFI_AGENT_PRIVATE_KEY` | PEM inline — fallback, warns |
| `AUTHFI_PLATFORM_URL` | your project's authfi.io address, for example `https://acme.authfi.io` |
| `AUTHFI_AGENT_AUDIENCE` | optional resource the token is for |

Prefer the file. A private key in an environment variable is visible in `ps`, inherited by
every child process and captured in crash dumps.

Endpoints are read from `/.well-known/agent-configuration`, not hard-coded — the token
endpoint moves with the project's origin, and the assertion's `aud` is built from the same
document the server checks it against.

`agent.token()` caches and refreshes 60s before expiry. A refusal raises `AgentAuthError`
carrying the registered OAuth code (`err.error` — `invalid_client`, `invalid_target`, …)
and is **not** retried.

`agent.capabilities()` is a set of capability **names**. Any constraint on a capability
lives on the server-side grant record and is checked when the capability executes.

### 3. Acting on a person's behalf (CIBA)

The agent asks; the person approves out of band on a channel the agent does not hold.

```python
token = agent.token_as_user(
    "priya@acme.com",
    binding_message="Send the Q3 summary to the board?",
    scope="send_email_external",
)
token["subject"]        # "user:priya@acme.com" — the PERSON
token["access_token"]   # sub = the person, act.sub = this agent (RFC 8693 §4.1)
```

Split the two halves when the agent should not block:

```python
approval = agent.request_user_approval("priya@acme.com", binding_message="Send it?")
# ... approval["auth_req_id"] is a bearer handle — store it like a credential ...
token = agent.await_user_token(approval, timeout=300)
```

The poll honours the server's `interval` and widens it on `slow_down`. `access_denied`,
`expired_token` and `invalid_grant` are decisions and stop it immediately.

## Features

- JWKS + RS256 verification (via PyJWT when installed)
- `@auth.require("read:users")` — decorator for Flask/FastAPI
- `@auth.require_role("admin")` — role-based access
- Permission auto-sync on `sync()`
- Agent authentication — `private_key_jwt` with Ed25519, discovery-driven
- CIBA — an agent acting as a person who approved out of band
- Cloud credentials (GCP/AWS/Azure/OCI)
- Cloud token exchange
- Works with Flask, FastAPI, Django, any WSGI/ASGI framework

## Cloud Credentials (AuthFI Connect)

```python
creds = auth.cloud_credentials(user_token, 'aws', role_arn='arn:aws:iam::123:role/deploy')
```

## Running Tests

```bash
python -m unittest test_authfi -v
```

65 unit tests — all passing.

## License

MIT
