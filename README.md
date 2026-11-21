# AuthFI Python SDK

The AuthFI SDK for Python: one OAuth core for every way your code meets AuthFI.

- **People** sign in to your app.
- **APIs** verify the tokens that reach them.
- **Machine clients** call APIs with their own token.
- **AI agents** hold their own key, act for a person with consent, and go through the AuthFI
  gateways.

Every area is made from one project client, so they share one discovery document, one key set
and one HTTP connection pool.

## Install

The package is installed from this repository until it is published:

```bash
pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
```

Python 3.11 or later. The dependencies are httpx, PyJWT and cryptography.

## The project client

```python
from authfi import AuthFI

af = AuthFI("https://acme.authfi.io")   # your project's address; or AuthFI.from_env() reads AUTHFI_ISSUER

metadata = af.metadata()                 # the discovery document, cached for its lifetime
print(metadata.token_endpoint)
```

The issuer is the one required setting. Everything else is read from the project's discovery
document. Create one client per project and share it: it owns the caches. In asyncio code, use
`AsyncAuthFI`, which has the same methods as coroutines; create one per event loop.

## Sign people in

```python
web = af.web_app(
    client_id="app_web",
    client_secret=Secret(os.environ["AUTHFI_CLIENT_SECRET"]),
    redirect_uri="https://app.example.com/callback",
    state_key=Secret(os.environ["AUTHFI_STATE_KEY"]),   # 32 random bytes, base64
)

pending = web.start_sign_in()                      # /login: redirect to pending.url,
cookie = pending.dump()                            # and keep this encrypted blob in a cookie

signed_in = web.finish_sign_in(callback_url, web.load_pending(cookie))   # /callback
signed_in.user.email, signed_in.tokens
```

The code flow with PKCE, `state`, `nonce` and the callback's `iss` (RFC 9207), with the ID token
verified. Refresh is one request per refresh token, shared; sign-out revokes the refresh token. See
[docs/signin.md](docs/signin.md).

## Protect an API

```python
from fastapi import Depends, FastAPI
from authfi import AccessToken, AsyncAuthFI
from authfi.fastapi import AuthFIBearer

auth = AuthFIBearer(AsyncAuthFI.from_env().verifier(audience=["api_orders"]))
app = FastAPI()
auth.install(app)

@app.get("/orders")
async def orders(token: AccessToken = Depends(auth.require(permissions=["orders:read"]))):
    return {"for": str(token.principal)}
```

`audience` is your API's client id (and your front ends' client ids, if your people call the API
with their own tokens). Every token is checked for its type, issuer, audience, expiry and tenant;
a refused one gets an opaque 401, and the precise reason goes to your log. Flask is supported as
`authfi.flask.AuthFIFlask`. See [docs/verify.md](docs/verify.md).

## Call an API as a machine client

```python
from authfi import AuthFI, Secret

af = AuthFI.from_env()
credential = af.client_credentials(
    "app_reports", Secret(os.environ["AUTHFI_CLIENT_SECRET"]),
    resource="api_orders", send_to=["https://api.example.com"],
)
httpx.Client(auth=credential.httpx_auth()).get("https://api.example.com/orders")
```

The token is reused and renewed before it expires, and is sent only to `send_to`. See
[docs/machine.md](docs/machine.md).

## Give an AI agent its own identity

```bash
authfi keygen --out agent.key      # register the printed public key on the agent in the console
```

```python
from authfi import AuthFI

agent = AuthFI.from_env().agent_from_env()     # AUTHFI_AGENT_ID, AUTHFI_AGENT_KEY_FILE, AUTHFI_AGENT_KID
httpx.Client(auth=agent.httpx_auth())          # its own token, for the AuthFI gateway
```

The agent signs a short assertion with a key that never leaves the process; its token is reused
and renewed before it expires. See [docs/agents.md](docs/agents.md).

## Hand a task to another process

A task is held by the run key of one process, a key that lives in that process's memory and is never written,
logged, put in an environment variable or sent. To give the whole task to another process, no key travels: the
receiver says which key it holds, the sender signs a move to it, and the platform moves the task.

```python
# In the process that will take the task (a worker, a sub-process of your orchestrator):
to = agent.run_key_thumbprint()                  # the public thumbprint only; send it to the sender

# In the process that holds it:
wire = task.hand_over(to=to).to_wire()           # the root and a 30-second signed move; send them privately

# Back in the receiver, within 30 seconds:
with agent.take_over(wire) as task:              # holds the task now; the sender's root stops at once
    ...
```

The sender's task is closed from the moment it hands over. `take_over` refuses with a `TakeOverRefused` that
says why: `KeyMoveRefused` (the move is not the holder's, is for another task or key, or has run out),
`KeyMoveSpent` (a move is good once), `TaskMoved` (another process took the task first), `TaskEnded`,
`TaskNotYours` and `TakeOverSlowDown` (too many asks for one task). A platform that cannot decide now raises
`TemporarilyUnavailable`. Details in [docs/agents.md](docs/agents.md#handing-a-task-to-another-process).

## Act for a person

```python
# A caller with no browser approves on their own phone (CIBA):
request = agent.request_user_token("priya@acme.com", binding_message="Refund order 4411")
as_priya = agent.await_user_token(request)

# A person signed in to your app (token exchange):
as_her = agent.exchange(Secret(her_access_token))
```

Each is a token whose subject is the person and whose actor is the agent. It never prints and
never pickles.

## Claude Code as yourself

`pip install "authfi[login] @ git+https://github.com/authfi/authfi-python-sdk@main"`, then `authfi login --issuer https://acme.authfi.io` signs you in once, and Claude Code
takes its credential for the project's `/llm` door from an `apiKeyHelper` that starts the helper isolated, `"<absolute python> -I -m authfi token --resource llm"` (`authfi login` prints the line). The refresh
token stays in your operating system's credential store. See [docs/claude-code.md](docs/claude-code.md).

## Errors

Every error is an `authfi.AuthFIError`. Branch on its `code`, never on the message:

```python
from authfi import AuthFIError, OAuthError

try:
    metadata = af.metadata()
except OAuthError as e:
    match e.code:
        case "invalid_client":
            ...
        case _:
            raise
```

A code is an open set: keep a default arm, because the platform adds codes between releases.

## Secrets

Client secrets, tokens and keys are held in `authfi.Secret`. It prints as `Secret('…')`, refuses
to be pickled or serialised, and gives its value only through `reveal()`.

## Documentation

- [The core: the project client, timeouts and retries, errors, secrets, credentials](docs/core.md)
- [Signing people in to a web app](docs/signin.md)
- [Protecting an API: the verifier, FastAPI, Flask](docs/verify.md)
- [Machine clients](docs/machine.md)
- [Claude Code as yourself: `authfi login` and the `/llm` door](docs/claude-code.md)
- [AI agents: keys, tokens, acting for a person, the action gate, approvals](docs/agents.md)
- [The AuthFI gateways: tools, models, cloud credentials](docs/gateway.md)
- [Agent frameworks: LangChain, LangGraph, OpenAI Agents SDK, Claude Agent SDK, Google ADK](docs/frameworks.md)
- Examples: [sign people in](examples/sign-in-web), [protect an API](examples/protect-an-api), [machine to machine](examples/machine-to-machine),
  [agent identity](examples/agent-identity), [a caller with no browser](examples/agent-for-a-person-voice),
  [a person signed in to your app](examples/agent-for-a-signed-in-user), [the action gate](examples/action-gate),
  [models through the AI gateway](examples/agent-model-gateway),
  [a LangChain model](examples/langchain-model), [approvals in LangGraph](examples/approvals-langgraph),
  [approvals in the OpenAI Agents SDK](examples/approvals-openai-agents),
  [approvals in the Claude Agent SDK](examples/approvals-claude-agent),
  [approvals in Google ADK](examples/approvals-adk)
- [Moving from 0.1](docs/migration-0.2.md)
- [Changelog](CHANGELOG.md)

## Licence

Apache-2.0. See [LICENSE](LICENSE).
