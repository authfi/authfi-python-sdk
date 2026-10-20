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
- [Protecting an API: the verifier, FastAPI, Flask](docs/verify.md)
- [Machine clients](docs/machine.md)
- [AI agents: keys, tokens, acting for a person, the action gate, approvals](docs/agents.md)
- [The AuthFI gateways: tools, models, cloud credentials](docs/gateway.md)
- [Agent frameworks: LangChain, LangGraph, OpenAI Agents SDK, Claude Agent SDK, Google ADK](docs/frameworks.md)
- Examples: [protect an API](examples/protect-an-api), [machine to machine](examples/machine-to-machine),
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
