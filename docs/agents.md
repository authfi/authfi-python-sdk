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

### The token is bound to a key (DPoP)

Nothing to write. The first time the agent asks for its token, it proves it holds a key that lives
only in this process (Ed25519, or P-256 where the runtime or the project has no Ed25519), and the
platform binds the token to that key (`cnf.jkt`, `token_type` `DPoP`). From then on every call the
agent makes to the gateway, `/mcp`, `/otlp/v1/traces` and the cloud credential door, is
`Authorization: DPoP <token>` with a new proof for that request. A stolen token is of no use without
the key, and a replayed proof is refused.

- A project may answer with a nonce (`use_dpop_nonce`). The SDK learns it, sends the request again
  once, and keeps the nonce for the next. A server that keeps asking is not asked a third time.
- A token the platform leaves unbound (a project without DPoP) is sent as Bearer, as before.
- Only a token for AuthFI itself is bound. `agent.for_resource(...)`, the token for a customer's API, is Bearer: that
  API verifies it, and a verifier built on this SDK takes no proof.
- A machine client's token (`af.client_credentials(...)`) is **not** bound: the platform cannot bind a
  `client_credentials` token for an app, so it stays Bearer.
- `af.agent(..., dpop=False)` asks for a Bearer token. It is for a credential you hand to a process
  that cannot hold the key; `authfi token` and `authfi headers` do it for Claude Code's
  `apiKeyHelper`. `agent.headers(method, url)` carries a proof good for one request, so make one per request.
- The approval status read and the action gate take the agent's token as Bearer, as those routes do.

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

The token is bound to the same key as the agent's own, and `as_priya.headers(method, url)` carries
its proof. Approving an action (below) and lending an identity (this) are different things: the
person's yes to acting for them says nothing about any one action, and `scope=["mcp:cloud.credentials"]`
on the request is what lets the agent ask for cloud credentials as them.

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

## Tasks: authority for one piece of work

A task is what a person gives an agent for one job: the tools it may use, a budget, a deadline. The
project defines the templates (`support.refund`); the agent asks for one with the person's token from an
app linked to it.

```python
with agent.task("support.refund", subject_token=persons_token, params={"order": "4411"}) as task:
    langgraph = LangGraphIntegration(task)      # the task is the credential; so is `task.gateway()`
    run_the_agent(langgraph)
```

That is the whole change to your agent: the task is a credential, like the agent's own token, and an adapter
takes it in the agent's place. The gateway holds each tool call, model call and cloud request to the task:

- a tool the task does not allow is refused `EMDT005`, a call over the budget `EMDT006`, an ended or
  expired task `EMDT004` / `EMDT003`;
- the refusal reaches the model as a tool result that says what it can still do ("Use a tool the task
  allows, or stop and tell the person what you could not do."), and `ToolRefused.why`,
  `ToolRefused.next_step` and `ToolRefused.as_dict()` give code the same answer to plan around;
- an ended task makes the gateway refuse the credential itself (HTTP 401, `GatewayTokenRefused`).

If the template needs the person's approval, `agent.task` waits for it. To show them the link yourself
and not block:

```python
started = agent.request_task("support.refund", subject_token=persons_token, params={"order": "4411"})
if isinstance(started, TaskPending):
    notify(started.approval_url)                    # the person approves on their own device
    task = agent.await_task(started, timeout=600)   # ConsentDenied, ConsentExpired or WaitTimeout
else:
    task = started
```

Nothing swaps the agent's own token for the task's behind your back, because a framework may keep a session
open from before the task (ADK and the OpenAI Agents SDK do), and a call through it would go with the agent's
wider token. So the task is passed in, and the adapter's sessions are made with it. Leaving the `with` forgets
the credential.

A task is one level deep, and nothing raises it. Inside a task `agent.task(...)` is refused
(`ConfigurationError`, `code == "task_nested"`), and no refusal offers a way to ask for more: only a
person gives a task. Its credential goes only to AuthFI's origins, does not print and does not
pickle. `ClaudeAgentIntegration` does not take a task: Claude Code's own model calls go through its
`apiKeyHelper`, a separate process that cannot hold the key.

## Cloud credentials, just in time

```python
outcome = agent.cloud.credentials("prod", "deployer", 300)     # account, role, seconds (1 to 900)
if outcome.kind == "approval_required":
    ...                                                        # once a person has decided:
    outcome = agent.cloud.credentials("prod", "deployer", 300, approval_id=outcome.approval_id)
credential = outcome.credential        # access_key_id, secret_access_key (a Secret), expires_at
```

The credentials are short-lived. `task.cloud.credentials(...)` holds the request to a task. A role the agent is not granted raises `APIError` (`ECLOUDCRED003`). An agent that speaks MCP also
sees this as the gateway's `cloud__credentials` tool, and a held request surfaces as any held tool call
does. A tool result goes to the model: when it should not read the secret, take it from `agent.cloud` in
your own code.

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
