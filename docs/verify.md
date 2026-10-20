# Protecting an API

An API built with the SDK verifies every token that reaches it with one verifier: a person's, a
machine client's, an agent's, and a token an agent holds for a person.

```python
from authfi import AuthFI

af = AuthFI.from_env()
verifier = af.verifier(audience=["api_orders"], expected_tenant_id="4e46e0c4-...")

token = verifier.verify_request(request.headers.get("Authorization"))
token.require_permission("orders:read")
```

In asyncio code, `AsyncAuthFI(...).verifier(...)` gives an `AsyncTokenVerifier` whose methods are
coroutines.

## Choosing the audience

`audience` is required, and there is no way to skip it. It is the list of client ids a token may
name in `aud`:

- **Your API's own client id.** Machine clients and agents name it when they mint with
  `resource=<your API's client id>`.
- **Your front ends' client ids.** A person who signs in to your web app gets a token for that
  app. If the app calls your API with it, accept the app's client id here.

AuthFI's own audiences are refused when the verifier is built: the issuer, any URL under it (such
as `<issuer>/mcp`, the audience of tokens meant for the AuthFI gateway), and AuthFI's own
applications. Accepting one would let a token meant for AuthFI be replayed at your API.

`expected_tenant_id` pins the tenant tokens must name. Set it. An empty value is a `ConfigurationError`, never "no pin": leave the argument out for that.

## What is checked

| Check | Refusal code |
|---|---|
| A compact JWS of three parts, at most 16 KiB | `token_malformed` |
| `alg` is RS256, ES256 or EdDSA (`none` and HMAC are refused before any key lookup) | `token_signature_invalid` |
| `typ` is `at+jwt`: an ID token or another JWT the project signs is not an access token | `token_not_access_token` |
| `kid` names a published key of that algorithm | `token_key_unknown` |
| The signature | `token_signature_invalid` |
| `iss` is the project's issuer exactly | `token_issuer_not_accepted` |
| `exp` is present and not reached (a token is expired from `exp` plus the leeway) | `token_claims_invalid`, `token_expired` |
| `nbf` and `iat` are not in the future | `token_claims_invalid` |
| `aud` holds one of your audiences | `token_audience_mismatch` |
| `sub` and `tenant_id` are present; `tenant_id` is the pinned tenant | `token_claims_invalid`, `token_no_tenant`, `token_tenant_not_issuers` |
| A sender-constrained token is not presented as Bearer | `token_dpop_proof_required` |

The codes are the AuthFI platform's own, so a refusal reads the same in your log and in the
project's audit records. They are on `InvalidToken.code` and in the log, never in the answer to
the caller (see below).

## The verified token

`AccessToken` has one shape for every kind of token:

| Field | What it is |
|---|---|
| `principal` | Who the token is for: `Principal(kind, id)`, where `kind` is `user`, `app` or `agent` |
| `actor` | Who is acting for them, on a token an agent holds for a person; else `None` |
| `token_kind` | `person`, `machine`, `agent` or `delegated` |
| `permissions`, `roles`, `groups` | What the project's policy granted, when the token was minted |
| `scopes` | The token's scope |
| `org_id` | The organisation the person signed in to, when there is one |
| `acr`, `amr`, `auth_time` | How and when the person authenticated |
| `client_id`, `tenant_id`, `issuer`, `audience`, `expires_at`, `jti` | As named |
| `claims` | Every claim, read-only |

Compare `principal`, never `sub` strings: tokens of different kinds write the same party
differently. A typed `agent:` or `app:` subject keeps its own kind (`agent` or `machine`) even
when the token names an actor; only a person's subject with an actor is `delegated`. `scopes`
reads the `scope` claim and `client_id` the `client_id` claim; `scp` and `azp` are not read.

## Authorization

```python
token.require_permission("orders:read", "orders:export")    # all of them
token.require_any_permission("orders:read", "orders:admin")  # at least one
token.require_role("support", "admin")                       # at least one
token.require_scope("orders:read")                           # all of them
token.require_acr("urn:authfi:acr:mfa", max_age=600)         # a recent strong sign-in
token.require_agent()                                        # an agent, or an agent for a person
```

`permissions` and `roles` are the default check. On a person's or a machine client's token,
`scope` is what the client asked for, so check it only for scopes your API defines itself; on an
agent's token, it is what the agent was granted.

For a decision about one object, or a fresh answer, ask the project:

```python
decision = af.check(credential, subject="user:u_123", relation="editor", object="doc:42")
if decision.allowed: ...
```

## FastAPI

```python
from fastapi import Depends, FastAPI
from authfi import AccessToken, AsyncAuthFI
from authfi.fastapi import AuthFIBearer

auth = AuthFIBearer(AsyncAuthFI.from_env().verifier(audience=["api_orders"]))
app = FastAPI()
auth.install(app)

@app.get("/orders")
async def orders(token: AccessToken = Depends(auth.require(permissions=["orders:read"]))):
    ...
```

`auth.optional()` gives the token or `None`. `AuthFIBearer` takes the asyncio verifier, so
fetching the project's keys never blocks the event loop.

## Flask

```python
from flask import Flask
from authfi import AuthFI
from authfi.flask import AuthFIFlask

app = Flask(__name__)
guard = AuthFIFlask(app, AuthFI.from_env().verifier(audience=["api_orders"]))

@app.get("/orders")
@guard.protect(permissions=["orders:read"])
def orders():
    return {"for": str(guard.token.principal)}
```

`guard.token` is the current request's token, held on `flask.g` for that request only.

## Requirements in a route

`permissions` (all), `any_permission` (at least one), `roles` (at least one) and `scopes` (all)
combine: each kind given must pass. An empty list is a configuration error, never "no
requirement".

## What the caller is told

| Refusal | Answer |
|---|---|
| No credential | 401, `WWW-Authenticate: Bearer` (with `resource_metadata` when you set it), `{"error":"unauthorized"}` |
| Any other token refusal | 401, `Bearer error="invalid_token", error_description="The access token is not valid."` |
| A missing scope | 403, `Bearer error="insufficient_scope", scope="..."` |
| A missing permission or role | 403, `{"error":"forbidden"}` |
| Authentication too weak or old | 401, `Bearer error="insufficient_user_authentication", acr_values="...", max_age=...` |
| The project's keys or discovery cannot be reached | 503, `Retry-After: 5`, `{"error":"service_unavailable"}`; the cause is logged at error |

Every answer carries `Cache-Control: no-store`. The precise code is logged on the `authfi` logger (a warning, or info when no token was sent) and is not sent: a caller cannot tell a wrong
key from a wrong audience. For an internal API whose callers may see why,
`expose_refusal_code=True` adds `"code"` to the body. A refusal raised inside a route
(`token.require_acr(...)`) is answered the same way.

## Several projects

A server that accepts tokens from several projects keeps one verifier per issuer, each pinned to
its tenant:

```python
from authfi import MultiIssuerVerifier

verifier = MultiIssuerVerifier({
    "https://acme.authfi.io": (acme.verifier(audience=["api_orders"]), "4e46e0c4-..."),
    "https://globex.authfi.io": (globex.verifier(audience=["api_orders"]), "9a1d..."),
})
```

The verifier is chosen by the token's issuer, from this list only. A token whose tenant is not
the one held for its issuer is refused.
