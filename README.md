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

```bash
pip install authfi
```

Python 3.11 or later. The dependencies are httpx, PyJWT and cryptography. A framework you use is an extra, and
brings only that framework:

```bash
pip install "authfi[langgraph]"     # or langchain-anthropic, openai-agents, claude-agent, adk, pydantic-ai, crewai,
                                    # llama-index; fastapi and flask for APIs; otel to send steps to AuthFI
```

## A governed agent in 10 minutes

```python
import authfi

af = authfi.connect_async()           # keyless: the agent proves where it runs; nothing to paste
graph = af.langgraph()                # models through /llm, tools through /mcp, approvals that pause and resume
```

Start with the [quickstart](https://github.com/authfi/authfi-python-sdk/blob/main/docs/quickstart.md): the one-time console setup, a model call, a tool call, an approval
and where to see the run. There is [one short page per framework](https://github.com/authfi/authfi-python-sdk/blob/main/docs/frameworks/README.md): LangChain, LangGraph,
OpenAI Agents SDK, Claude Agent SDK, Google ADK, Pydantic AI, CrewAI and LlamaIndex.

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
[docs/signin.md](https://github.com/authfi/authfi-python-sdk/blob/main/docs/signin.md).

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
`authfi.flask.AuthFIFlask`. See [docs/verify.md](https://github.com/authfi/authfi-python-sdk/blob/main/docs/verify.md).

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
[docs/machine.md](https://github.com/authfi/authfi-python-sdk/blob/main/docs/machine.md).

## Give an AI agent its own identity

Register the agent once in the console and bind it to where it runs (a Kubernetes service account, a cloud workload or
a device). The code holds no key and no secret:

```python
import authfi

agent = authfi.connect()               # AUTHFI_ISSUER and AUTHFI_AGENT_ID; it proves where it runs
chain = agent.langchain()              # or .langgraph(), .openai_agents(), .claude_agent(), .adk()
```

The agent's token is bound to a key only this process holds, and every call carries a proof of it (DPoP), so a
stolen token is of no use. See [docs/agents.md](https://github.com/authfi/authfi-python-sdk/blob/main/docs/agents.md).

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

`pip install "authfi[login]"`, then `authfi login --issuer https://acme.authfi.io` signs you in once, and Claude Code
takes its credential for the project's `/llm` door from an `apiKeyHelper` that starts the helper isolated, `"<absolute python> -I -m authfi token --resource llm"` (`authfi login` prints the line). The refresh
token stays in your operating system's credential store. See [docs/claude-code.md](https://github.com/authfi/authfi-python-sdk/blob/main/docs/claude-code.md).

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

- [The core: the project client, timeouts and retries, errors, secrets, credentials](https://github.com/authfi/authfi-python-sdk/blob/main/docs/core.md)
- [Signing people in to a web app](https://github.com/authfi/authfi-python-sdk/blob/main/docs/signin.md)
- [Protecting an API: the verifier, FastAPI, Flask](https://github.com/authfi/authfi-python-sdk/blob/main/docs/verify.md)
- [Machine clients](https://github.com/authfi/authfi-python-sdk/blob/main/docs/machine.md)
- [Claude Code as yourself: `authfi login` and the `/llm` door](https://github.com/authfi/authfi-python-sdk/blob/main/docs/claude-code.md)
- [AI agents: keys, tokens, acting for a person, the action gate, approvals](https://github.com/authfi/authfi-python-sdk/blob/main/docs/agents.md)
- [The AuthFI gateways: tools, models, cloud credentials](https://github.com/authfi/authfi-python-sdk/blob/main/docs/gateway.md)
- [Agent frameworks: LangChain, LangGraph, OpenAI Agents SDK, Claude Agent SDK, Google ADK](https://github.com/authfi/authfi-python-sdk/blob/main/docs/frameworks.md)
- Examples: [sign people in](https://github.com/authfi/authfi-python-sdk/tree/main/examples/sign-in-web), [protect an API](https://github.com/authfi/authfi-python-sdk/tree/main/examples/protect-an-api), [machine to machine](https://github.com/authfi/authfi-python-sdk/tree/main/examples/machine-to-machine),
  [agent identity](https://github.com/authfi/authfi-python-sdk/tree/main/examples/agent-identity), [a caller with no browser](https://github.com/authfi/authfi-python-sdk/tree/main/examples/agent-for-a-person-voice),
  [a person signed in to your app](https://github.com/authfi/authfi-python-sdk/tree/main/examples/agent-for-a-signed-in-user), [the action gate](https://github.com/authfi/authfi-python-sdk/tree/main/examples/action-gate),
  [models through the AI gateway](https://github.com/authfi/authfi-python-sdk/tree/main/examples/agent-model-gateway),
  [a LangChain model](https://github.com/authfi/authfi-python-sdk/tree/main/examples/langchain-model), [approvals in LangGraph](https://github.com/authfi/authfi-python-sdk/tree/main/examples/approvals-langgraph),
  [approvals in the OpenAI Agents SDK](https://github.com/authfi/authfi-python-sdk/tree/main/examples/approvals-openai-agents),
  [approvals in the Claude Agent SDK](https://github.com/authfi/authfi-python-sdk/tree/main/examples/approvals-claude-agent),
  [approvals in Google ADK](https://github.com/authfi/authfi-python-sdk/tree/main/examples/approvals-adk)
- [Moving from 0.1](https://github.com/authfi/authfi-python-sdk/blob/main/docs/migration-0.2.md)
- [Changelog](https://github.com/authfi/authfi-python-sdk/blob/main/CHANGELOG.md)

## Licence

Apache-2.0. See [LICENSE](https://github.com/authfi/authfi-python-sdk/blob/main/LICENSE).
