# Moving from 0.1 to 0.2

0.2 rebuilds the SDK on one core shared by every area. Nothing was published to PyPI, and the
changes below are the ones you meet coming from `main`'s single `authfi.py` module.

## Python 3.11 or later

0.1 declared Python 3.8. 0.2 needs 3.11: 3.8 and 3.9 are past end of life and 3.10 reaches it
this month. pip refuses the install on an older Python and names the version.

## Configure by issuer

The project is configured by its address, its issuer, and nothing else:

```python
# 0.1
auth = AuthFI(tenant="acme", api_key="sk_live_...")

# 0.2
af = AuthFI("https://acme.authfi.io")       # or AuthFI.from_env(): AUTHFI_ISSUER
```

`api_key` is no longer used: a project is named by its issuer, and each credential comes from
the client or agent you register in the console. The base domain, the API host and every endpoint
are read from the project's discovery document.

## Package layout

`authfi` is now a package. The names `main` exported are kept at the top level through the 0.x
series. A name that changed still works, is marked deprecated for type checkers and IDEs, and
warns `AuthFIDeprecationWarning` (a `DeprecationWarning`) naming what replaces it. They are
removed at 1.0.

## Protecting an API

| 0.1 | 0.2 |
|---|---|
| `AuthFI(tenant="acme", api_key=...)` | `AuthFI("https://acme.authfi.io")`. The old form still builds the project, from `https://<tenant>.<base_domain>` or `issuer=`, and warns. `api_key` and `api_url` are not used. |
| `auth.verify_token(token)` | `af.verifier(audience=["<your API's client id>"]).verify(token)`. The old method still works when the project was built with `audience=[...]`, and returns the claims read-only. |
| `@auth.require("read:users")`, `@auth.require_role("admin")`, `auth.current_user()` | `authfi.flask.AuthFIFlask`: `@guard.protect(permissions=[...])`, `guard.token`. The old decorators still work in Flask, and now answer 401 and 403 instead of an error page. For FastAPI, use `authfi.fastapi.AuthFIBearer`. |
| `auth.register_permission(...)`, `auth.sync()`, `auto_sync=` | Retired: create permissions and roles in the AuthFI console. The calls raise `NotSupportedError`. |
| `auth.cloud_credentials(...)`, `auth.cloud_token(...)` | Retired: an agent gets cloud credentials through the AuthFI gateway (`agent.gateway().cloud_credentials()`). The calls raise `NotSupportedError`. |
| `AgentAuthError` | `OAuthError`; `.error` is `.code`. |
| `AuthFIError(message, status)` | Unchanged, with `.code` added. |

## Agents

| 0.1 | 0.2 |
|---|---|
| `AgentAuth(agent_id, key, platform_url=...)` | `AuthFI(issuer).agent(agent_id, key)`. The old form still works when `platform_url` is given, and warns. |
| `AgentAuth.from_env()` with `AUTHFI_PLATFORM_URL`, `AUTHFI_AGENT_AUDIENCE` | `AuthFI.from_env().agent_from_env()` with `AUTHFI_ISSUER`, `AUTHFI_AGENT_RESOURCE`. The old names are read, with a warning. The key variables are unchanged. |
| `agent.token(audience)`, `audience=` | `agent.token(resource=...)`; `resource=` names the API. |
| `agent.auth_header()` | `agent.headers(method, url)`, or `httpx.Client(auth=agent.httpx_auth())` |
| `agent.can(name)` | `name in agent.capabilities()` |
| `agent.request_user_approval(...)` | `agent.request_user_token(..., binding_message=...)`; the message is required. |
| `agent.await_user_token(...)`, `token_as_user(...)` returning a `dict` | They return a `DelegatedToken`: call `.token()` for the value, and read `.subject`, `.actor`. |
| `AgentKey.generate("k1")`, `from_pem(pem, "k1")`, `key_id=`, `.key_id` | `kid=` everywhere: `AgentKey.generate(kid="k1")`. |
| `key.private_pem()` | `key.write(path)`: a new file only its owner can read. |
| `token_endpoint=`, `backchannel_endpoint=`, `discovery_url=` | Not used: endpoints come from the project's discovery document. |

## What no longer works, on purpose

- **A token is never verified without an audience.** `verify_token` on a project built without
  `audience=` raises `ConfigurationError`. Without an audience, a token minted for any API of
  the project would be accepted by yours.
- **A token must name its key.** A token with no `kid` is refused, even when the project
  publishes one key.
- **Only access tokens are accepted.** A person's ID token, or any other JWT the project signs,
  is refused by the verifier (`typ` must be `at+jwt`).
- **An agent's project has no default.** 0.1 sent an agent with no address to
  `platform.authfi.io`, where it does not exist.
- **A key file others can read is refused.** Make it readable by its owner only
  (`chmod 600`), or write it with `AgentKey.write`.
- **A person's token never comes back as a dict.** A `DelegatedToken` does not print or pickle its
  token; a checkpoint stores the person's id.
- **The first CIBA poll waits the interval** the project asks for, as the CIBA standard says.
- **The answer to a refused request does not say why.** A 401 carries one fixed description,
  and a 403 no list of missing permissions; the reason is in your log.
