# The AuthFI gateways

`<issuer>/mcp` is the MCP gateway, which governs an agent's tool calls, and the AI gateway, which
governs its model calls. A `Gateway` connects any credential (an agent's own token, or a token it
holds for a person) to them.

```python
gateway = agent.gateway()                   # or Gateway(af, delegated_token)
```

## An HTTP client for the gateways

```python
client = gateway.http_client()
```

The client puts the current token on every request, sends it to AuthFI's origin only, follows no
redirect, and sends an `X-Request-Id` and a `traceparent`, so a trace joins the gateway's record of
the call. It is made once and closed with the project; do not close it yourself.
`gateway.new_http_client()` makes one you own and close, as an MCP client factory does.

Give it to an MCP client or a provider SDK. The newer `anthropic` and `openai` packages are built
on `httpx2`; ask for a client of that package:

```python
import httpx2

anthropic.Anthropic(base_url=gateway.model_base_url("anthropic"), api_key="unused",
                    http_client=gateway.http_client(httpx2))
openai.OpenAI(base_url=gateway.model_base_url("openai"), api_key="unused",
              http_client=gateway.http_client(httpx2))
```

The token is never the SDK's `api_key` or a default header, which tracers record.

## What the gateway answered

```python
outcome = gateway.read_result(call_tool_result)
match outcome:
    case ToolAllowed(result=result):                   use(result)
    case ToolRefused():                                tell_the_model(outcome.model_text)   # code, reason, required_scope, result
    case ToolApprovalRequired(approval_id=approval):   pause_until_a_person_decides(approval)
```

`ToolOutcome` is the union of those three frozen dataclasses (from `authfi` or `authfi.gateway`), so a
type checker narrows each arm: `approval_id` is a `str` where a call is held, and exists nowhere
else. A refusal keeps the gateway's own answer in `result`.

A refused or held call is a tool result, not an exception, so a run carries on and the model reads
why. Only the token itself being refused (HTTP 401) ends a run: no call can succeed with that token.
The framework adapters raise `GatewayTokenRefused` for it; with your own client,
`authfi.gateway.token_refusal(error)` finds it in whatever the client raised.

**A held call's approval never goes through the model.** Give the model `outcome.model_text` (for
a hold, "This action is waiting for a person's approval in AuthFI."), keep `approval_id` yourself, and send
the call again with `gateway.with_approval(arguments, approval_id)` once the person has decided.
Remove any approval id a model wrote with `gateway.strip_approval_argument(arguments)`.

`gateway.scope_of(tool)` reads the grant a tool from `tools/list` needs, and
`gateway.governance_of(tool)` the agent's outlook for it (`no_policy_restriction`,
`may_require_approval` or `never_allowed`; advisory, as the gateway decides each call). Tool names are
`<server>__<tool>`, as the gateway lists them.

## Cloud credentials

```python
outcome = gateway.cloud_credentials("prod", "deployer", expires_in=900)
if outcome.kind == "approval_required":
    ...   # once a person has decided:
    outcome = gateway.cloud_credentials("prod", "deployer", approval_id=outcome.approval_id)
credential = outcome.credential       # access_key_id, secret_access_key (a Secret), expires_at
```

A refusal raises `APIError` with the gateway's code. `expires_in` is 1 to 900 seconds, the most the platform issues.

## In asyncio code

`AsyncAgentAuth.gateway()` gives an `AsyncGateway`, whose `http_client()` is an
`httpx.AsyncClient` and whose `cloud_credentials` is a coroutine.
