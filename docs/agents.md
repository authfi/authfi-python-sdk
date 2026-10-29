# AI agents

An AuthFI agent is an OAuth client that holds its own private key. It authenticates by signing a
short assertion; the key never leaves the process, and there is no secret to leak. Its token
comes from the same token endpoint as everyone else's, and an API verifies it with the same
verifier.

## The key

```bash
authfi keygen --out /run/secrets/agent.key        # EdDSA; or --alg ES256 / PS256
```

This writes the private key to a new file that only its owner can read, and prints the public
JWK and its thumbprint. Register the public JWK on the agent in the AuthFI console; its `kid` is
the key id the agent signs with.

In code:

```python
from authfi import AgentKey

key = AgentKey.generate()                 # Ed25519; kid = its RFC 7638 thumbprint
key.write("/run/secrets/agent.key")       # exclusive create, owner-only
print(key.public_jwk())                   # what you register

key = AgentKey.from_file("/run/secrets/agent.key", kid="...")
```

| Key | Algorithm |
|---|---|
| Ed25519 | EdDSA (the default) |
| P-256 | ES256 |
| RSA 2048 to 4096 bits | PS256 (3072 bits when generated) |

`from_file` refuses a key file that others can read: a POSIX mode wider than `0600`, or on
Windows a permission that lets Everyone, Users or Authenticated Users read it. No method returns
the private key as a value; `write(path)` is the one way out.

## The agent

```python
from authfi import AuthFI

af = AuthFI.from_env()                    # AUTHFI_ISSUER
agent = af.agent("agt_...", key)          # or af.agent_from_env()

agent.token()                             # its own token, for the AuthFI gateway (<issuer>/mcp)
agent.capabilities()                      # the capability names it was granted
httpx.Client(auth=agent.httpx_auth())     # the current token on every request to AuthFI
```

The token is cached and renewed `refresh_skew` (60 s) before it expires; concurrent callers share
one request. The agent's own token is for AuthFI: `af.agent(..., resource=)` takes only AuthFI's own
resources, because the agent sends this token to AuthFI's endpoints (the action gate, approvals,
token exchange) and must never hold a token minted for another API by default. For a customer's API,
make a token source for it, naming the origins it may go to:

```python
orders = agent.for_resource("api_orders", send_to=["https://api.example.com"])
```

### From the environment

| Variable | What it is |
|---|---|
| `AUTHFI_ISSUER` | Your project's address (read by `AuthFI.from_env`) |
| `AUTHFI_AGENT_ID` | The agent's id, `agt_...` |
| `AUTHFI_AGENT_KEY_FILE` | The path of its private key file. The default and the documented way. |
| `AUTHFI_AGENT_KID` | The key id, when the file is PEM |
| `AUTHFI_AGENT_PRIVATE_KEY` | The PEM itself, for a runtime with no filesystem only. Every child process inherits an environment variable, so this warns. |
| `AUTHFI_AGENT_RESOURCE` | What its own token is for, when not the AuthFI gateway |

## Acting for a person with no browser (CIBA)

A voice or chat agent asks the person on their own device, and receives a token that acts for
them:

```python
request = agent.request_user_token("priya@acme.com", binding_message="Refund order 4411")
as_priya = agent.await_user_token(request)          # waits for their decision
httpx.Client(auth=as_priya.httpx_auth())
```

- `binding_message` is required: it is what the person reads before they approve.
- The agent never sees or relays a code.
- The wait polls at the interval the project asks for. `poll_interval=` may ask for a longer wait; a
  smaller value is raised to the project's interval, and it must be finite and above zero
  (`ConfigurationError`). It raises `ConsentDenied` if they say no and
  `ConsentExpired` if the request runs out. At your own `timeout` it raises `WaitTimeout`
  (`code == "wait_timeout"`); the request stays open, and waiting on it again continues it.
- `agent.token_as_user(...)` does both steps.

The token's `sub` is the person and `act.sub` the agent. It is never renewed: the person's
approval was for this token. Shortly before it expires, `token()` raises `ConsentExpired`, so no
call is made with a token about to die.

## Acting for a person signed in to your app (token exchange)

```python
as_her = agent.exchange(Secret(persons_access_token))     # or her session, a TokenSource
```

The project's policy decides whether this agent may act for this person; a refusal is
`OAuthError` `access_denied`. The token is exchanged again near its expiry, while the person's own
token lives.

A `DelegatedToken` is a `TokenSource`: it goes wherever the agent's own token goes. It does not
print its token and does not pickle: a checkpoint stores the person's id, never their token.

## The action gate

For a high-risk action, freeze it, then ask the gate:

```python
txn = agent.transaction_token("refund", resource="order:4411", context={"amount": 120})
decision = agent.authorize_action(txn)     # allow, deny or approval_required
```

The transaction token lives 60 seconds, and the gate refuses it for any other action, resource or
context. Constraints are checked before it is made, so an action over a limit never gets one.

## Approvals

A person approves one action; no token results:

```python
approval = agent.request_approval("refund order 4411", context={"amount": 120})
approval = agent.wait_for_approval(approval.approval_id)
if approval.status == "approved": ...
```

`wait_for_approval(approval_id, timeout=300, poll_interval=3)` returns the approval whatever the
person decided (`approved`, `denied` or `expired`). At your `timeout` it raises `WaitTimeout`, whose
`pending_id` is the approval's id; the approval stays open, and waiting on it again continues.

## In asyncio code

`AsyncAuthFI(...).agent(...)` gives an `AsyncAgentAuth` with the same methods as coroutines; its
delegated tokens are `AsyncDelegatedToken`. Cancel the task to stop a wait.

## The command line

| Command | What it prints |
|---|---|
| `authfi keygen --out PATH [--alg ...] [--kid ...]` | the public JWK and its thumbprint |
| `authfi token` | the agent's current token |
| `authfi headers` | `{"Authorization": "Bearer ..."}` as JSON, with `X-Request-Id` from `AUTHFI_RUN_ID` |

`token` and `headers` read the environment above. A failure prints nothing on standard output and
exits 1, so a tool that runs them never mistakes an error for a credential.
