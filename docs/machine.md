# Machine clients

A backend that calls an API on its own behalf uses the client-credentials grant.

```python
from authfi import AuthFI, Secret

af = AuthFI.from_env()
credential = af.client_credentials(
    "app_reports",
    Secret(os.environ["AUTHFI_CLIENT_SECRET"]),
    resource="api_orders",                  # the API's client id: the token's audience
    send_to=["https://api.example.com"],    # the only origin the token may be sent to
)

httpx.Client(auth=credential.httpx_auth()).get("https://api.example.com/orders")
```

| Argument | What it is |
|---|---|
| `client_id`, `client_secret` | The machine-to-machine application registered in the console. The secret is a `Secret`. |
| `resource` | The client id of the API the token is for. |
| `send_to` | The API's origins. Required with `resource`; the token is refused to any other origin. |
| `scope` | Scope names, when the API defines its own. |
| `auth_method` | `client_secret_basic` (the default) or `client_secret_post`. |

- **The token is reused.** It is cached and renewed at two thirds of its lifetime, and concurrent
  callers share one request. `credential.token(force=True)` mints a new one.
- **Name the API.** Without `resource`, the token's audience is the client itself, which no API
  should accept; the SDK logs a warning the first time.
- **Authorization comes from roles.** Bind the client a role in the console; the token carries
  the role's permissions.

`AsyncAuthFI(...).client_credentials(...)` gives the same credential for asyncio code, with an
`httpx.AsyncClient` auth.
