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

An AuthFI agent holds its **own** keypair: Ed25519 (EdDSA), P-256 (ES256) or RSA 2048 to
4096 bits (PS256). It authenticates by signing a short assertion (`private_key_jwt`, RFC 7523 §2.2) — the private half never leaves the process
and is never sent to AuthFI. There is no client secret to leak, because an agent
registered with a public JWK is issued none.

### 1. Make a key, register the public half

```python
from authfi import AgentKey

key = AgentKey.generate("k1")            # Ed25519; or AgentKey.generate("k1", "ES256") / "PS256"

open("/run/secrets/agent.key", "w").write(key.private_pem())   # chmod 600, never sent
print(key.public_jwk())   # {"kty":"OKP","crv":"Ed25519","x":"...","alg":"EdDSA","kid":"k1"}
```

A key you already hold loads with `AgentKey.from_file(path, key_id=...)`: a PEM private key
(PKCS#8, or the SEC1 and PKCS#1 forms) or a private JWK. The key type decides the algorithm:
Ed25519 signs EdDSA, P-256 signs ES256 and RSA signs PS256 (never RS256). Any other curve or
size, and a JWK labelled with an algorithm its type does not sign, is refused before anything
is sent.

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
| `AUTHFI_PLATFORM_URL` | your project's authfi.io address, for example `https://acme.authfi.io` — **required** |
| `AUTHFI_AGENT_AUDIENCE` | optional resource the token is for |

Prefer the file. A private key in an environment variable is visible in `ps`, inherited by
every child process and captured in crash dumps.

The project address is required, as `platform_url` or `AUTHFI_PLATFORM_URL`, and has no
default: an agent is registered in one project, and only that project can authenticate it.

Endpoints are read from the project, not hard-coded: the token endpoint from its
authorization server metadata (`/.well-known/oauth-authorization-server`, RFC 8414), falling
back to `/.well-known/agent-configuration`, and the CIBA endpoint from agent-configuration.
They move with the project's origin, and the assertion's `aud` is the token endpoint the
server checks it against.

When a response carries a `Deprecation` header (RFC 9745), the SDK logs one warning per
address on the `authfi` logger, naming the `successor-version` link, so a route on its way
out shows up in your logs before it is removed.

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

The token you get back is a `DelegatedToken`: the token response itself (so `token["subject"]`
works as above) that also knows when it expires. Use `token.token()` (or `token.auth_header()`)
each time you send it rather than keeping `token["access_token"]`. Once it nears expiry,
`token.token()` raises `AgentAuthError` `expired_token`: the approval was the person's consent
for that token, and acting for them again means asking them again.

### 4. Acting for a person whose token you hold (token exchange)

When the person is signed in to the application the agent runs in, the agent can exchange their
token for one that acts for them (RFC 8693). The agent's own token goes as the actor, and the
project decides whether this agent may act for this person.

```python
as_priya = agent.exchange(priya_access_token, audience="https://api.acme.com")
requests.get(url, headers=as_priya.auth_header())   # sub = Priya, act.sub = this agent
```

`as_priya.token()` exchanges again from the same subject token when it nears expiry, so it lives
as long as the person's own token does. A refusal raises `AgentAuthError` (`access_denied`,
`invalid_grant`).

### 5. Approvals the agent raised

```python
record = agent.approval_status(approval_id)           # {"status": "pending", ...}
record = agent.wait_for_approval(approval_id, timeout=300)
record["status"]                                      # "approved", "denied" or "expired"
```

Both read the approval at the address discovery advertises (`endpoints.approval`), with the
agent's own token. A project that does not let the agent read it answers 403, raised as
`AuthFIError` with `err.status == 403`.

### 6. Async frameworks

`AsyncAgentAuth` is the same client for asyncio code (LangGraph, the OpenAI Agents SDK, Google
ADK, the Claude Agent SDK). It shares the wrapped `AgentAuth`'s key and token cache, runs each
request in a worker thread and waits with `asyncio.sleep`, so a CIBA wait of minutes does not
block the event loop.

```python
from authfi import AgentAuth, AsyncAgentAuth

agent = AsyncAgentAuth(AgentAuth.from_env())

token = await agent.token()
approval = await agent.request_user_approval("priya@acme.com", binding_message="Send it?")
as_priya = await agent.await_user_token(approval)
headers = {"Authorization": "Bearer " + await as_priya.atoken()}

as_priya = await agent.exchange(priya_access_token)
record = await agent.wait_for_approval(approval_id)
```

### 7. HTTP clients

`httpx_auth()` sets `Authorization: Bearer` on every request, with a token refreshed before it
expires. One object works with `httpx.Client` and `httpx.AsyncClient`.

```bash
pip install 'authfi[httpx]'
```

```python
import httpx

client = httpx.AsyncClient(auth=agent.httpx_auth())     # the agent itself
client = httpx.Client(auth=as_priya.httpx_auth())       # acting for Priya
```

## Features

- JWKS + RS256 verification (via PyJWT when installed)
- `@auth.require("read:users")` — decorator for Flask/FastAPI
- `@auth.require_role("admin")` — role-based access
- Permission auto-sync on `sync()`
- Agent authentication — `private_key_jwt` with Ed25519, P-256 or RSA keys, discovery-driven
- CIBA — an agent acting as a person who approved out of band
- Token exchange (RFC 8693) — an agent acting for a person whose token it holds
- Async client for asyncio agent frameworks, and `httpx` auth that refreshes the token
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

## License

MIT
