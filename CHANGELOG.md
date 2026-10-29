# Changelog

All notable changes to this package. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased] — 0.2.0

The SDK is rebuilt on one core that every area shares: people signing in, APIs verifying tokens,
machine clients and AI agents. See [docs/migration-0.2.md](docs/migration-0.2.md).

### Added

- An agent's token is bound to a key it holds (DPoP, RFC 9449), and every gateway call proves it: the
  MCP tools, the models, `/otlp/v1/traces` and the cloud credential door. The server's nonce is
  answered once. `af.agent(..., dpop=False)` keeps a Bearer token for a process that cannot hold the key
  (`authfi token` does). Tokens that a person's approval or an exchange yields are bound the same way.
- `agent.task(...)`, `request_task` and `await_task`: authority for one piece of work (a mandate), asked for
  with a token exchange and `authorization_details`. The task is a credential: an adapter takes it in the
  agent's place (`LangGraphIntegration(task)`), and the gateway holds each call to it. A call the task refuses
  reaches the model with what it can still do (`ToolRefused.why`, `next_step`, `as_dict()`). A task is one
  level deep.
- `agent.cloud.credentials(account, role, duration)` for short-lived cloud credentials.

- Extras cover each framework's major version (`google-adk` 2.x, `mcp` 2.x where the framework allows it,
  `litellm` 1.x). `authfi[langchain]` installs only `langchain-core`; the Claude model is
  `authfi[langchain-anthropic]`. LangSmith is never required, and `configure_tracing()` does nothing without it.
- Instance sign-in: on a bound workload `connect()` returns an `InstanceAuth` that signs in by attestation with no key
  to keep (a projected service-account token or a device key, then challenge, attest and a DPoP-bound token with
  the `OAuth-Client-Attestation` headers), renews the attestation before it expires, and sends `Authorization: DPoP`
  with a fresh proof on every call. The agent key file stays as the labelled legacy fallback. See `authfi.instance`.
- LangSmith tracing: `authfi.otel.register()` also sets `langsmith.span.kind` and `langsmith.trace.name`, every gateway
  decision is a `tool` child span (`authfi.decision.*`, `authfi.mandate.id`), and a secret-shaped value or a
  credential-named attribute is replaced before any export, with content capture on or off (`authfi.tracing`).
- `authfi.langgraph_auth`: an `Auth()` for `langgraph.json` that verifies AuthFI tokens (JWKS, DPoP proof, delegation
  chain) and asks the project's policy engine about every thread, assistant, run, cron and store call (extra
  `langgraph-auth`).
- `authfi.a2a`: fetch and verify an A2A 1.0 agent card (JWS over RFC 8785, certificate chain to a root you trust),
  call the remote agent with this agent's DPoP-bound token, and map AuthFI decisions to task states.
- `authfi.connect()` and `connect_async()`: the credential the environment names (`AUTHFI_AGENT_ID` with a
  key, or `AUTHFI_CLIENT_ID` with `AUTHFI_CLIENT_SECRET`), with `langchain()`, `langgraph()`,
  `openai_agents()`, `claude_agent()` and `adk()` building the integrations. `client_credentials_from_env()`
  on both project clients. A machine client has a gateway of its own (`m2m.gateway()`) that every
  integration built on it shares.
- A run id on every credential (`run_id`, `start_run()`): `X-Request-Id: <run id>.<n>` on every gateway call,
  `authfi_run_id` in the metadata of LangChain and OpenAI Agents runs, `authfi.run_id` in the trace resource
  attributes of ADK and Claude Code. The `User-Agent` of a machine client's gateway calls says
  `authfi-credential/client-credentials`.
- `authfi.otel` (extra `otel`): an OTLP/HTTP exporter to `<issuer origin>/otlp/v1/traces` that authenticates
  with the credential's current token on every export, removes GenAI content unless allowed, and tags spans
  with `authfi.agent_id` and `authfi.run_id`; `export_steps()` on the LangChain, LangGraph, OpenAI Agents and
  ADK integrations, and `export_steps=True` on the Claude Agent SDK's options, wire it into each framework's
  own tracing.
- `AuthFI` and `AsyncAuthFI`: the project client, configured by its issuer alone, with discovery
  (RFC 8414 / OpenID), the key set, and one HTTP transport shared by every area.
- Timeouts on every request (connect, read, and a deadline per call), and retries only where
  repeating a request is safe; a refusal is never retried.
- `X-Request-Id`, `traceparent` and a named `User-Agent` on every request; the platform's
  request id on every error.
- `resolve_api_origin()` on both clients, for building a machine-API URL on a project with a custom domain.
- Typed errors with stable codes (`AuthFIError.code`), including the platform's token refusal
  codes, generated from the AuthFI kernel.
- `Secret`, which does not print, pickle or serialise a credential.
- `TokenSource` and `AsyncTokenSource`: one interface for every credential, each refusing to be
  sent to an origin it was not made for.
- `TokenSet`, `TokenStore` and `InMemoryTokenStore` for a person's tokens, with refreshes made
  once per refresh token across threads and tasks.
- Type information (`py.typed`); the package passes `mypy --strict` and pyright in strict mode.
- `af.verifier(audience=[...])`: a strict access-token verifier for APIs. It requires `typ`
  `at+jwt`, the issuer, an audience (never AuthFI's own), `exp`, `sub` and `tenant_id`; verifies
  RS256, ES256 and EdDSA by the issuer's key; pins the tenant; and refuses a sender-constrained
  token presented as Bearer. `AccessToken` gives every token the same shape, with `Principal`.
- `MultiIssuerVerifier`, pinning each accepted issuer to its tenant.
- `af.check(...)`: a live authorization decision from the project's policy.
- `af.client_credentials(...)`: a machine client's token for one API, cached and renewed, and
  sent only to that API's origins.
- `authfi.fastapi.AuthFIBearer` and `authfi.flask.AuthFIFlask`, answering refusals as RFC 6750
  and RFC 9470 say, with the precise reason in the log only.
- `af.agent(agent_id, key)` and `af.agent_from_env()`: an AI agent on the shared core. Its own
  token (client credentials with `private_key_jwt`, at the project's token endpoint), CIBA with a
  required `binding_message`, token exchange, the action gate (`transaction_token`,
  `authorize_action`) and approvals. `AsyncAgentAuth` is its asyncio twin, on
  `httpx.AsyncClient`, never threads.
- `DelegatedToken`: a token an agent holds for a person, as a `TokenSource`. A CIBA token is
  never renewed; an exchanged one is exchanged again while the person's token lives.
- `AgentKey` for Ed25519, P-256 and RSA, signing through PyJWT; `write()` creates an owner-only
  file (POSIX mode or Windows DACL), and `from_file()` refuses one others can read.
- The `authfi` command: `keygen`, `token` and `headers`.
- `agent.gateway()`: the AuthFI gateways for any credential. `http_client()` (the token per
  request, origin-pinned, no redirects, `X-Request-Id` and `traceparent`), `read_result()` into a
  `ToolOutcome` (a union of `ToolAllowed`, `ToolRefused` and `ToolApprovalRequired`, which a `match` narrows), model base URLs, the grant a tool
  needs, and `cloud_credentials()`. A held call's text never reaches a model; a model-written
  approval id is removed.
- `authfi.langchain.LangChainIntegration` (extra `langchain`): a `ChatAnthropic` on the AI gateway
  whose token is attached per request, never a model setting LangSmith records; tracing without
  content by default; `GatewayTokenRefused` when the gateway refuses the token.
- `authfi.langgraph.LangGraphIntegration` (extra `langgraph`): the gateway's MCP tools under the
  gateway's names, through one MCP client; a refusal as the tool's error so the run carries on; a
  held call recorded in graph state and resumed through an approval node that calls `interrupt()`
  first; `agent_graph()` for a ready agent. The approval id never passes through the model.
- `Gateway.governance_of(tool)`: the agent's advisory outlook for a tool from `tools/list`.
- `authfi.openai_agents.OpenAIAgentsIntegration` (extra `openai-agents`): `configure_tracing()`
  removes the SDK's default exporter and keeps content out of spans; `model()` reaches Claude
  through the AI gateway with the agent's token and no OpenAI key; `mcp_server()` gives the
  gateway's tools, refusals the model reads, and approvals through the SDK's own approval items,
  approved only once the gateway has decided (`resolve()`).
- `AsyncGateway.mcp_client_factory()`: HTTP clients for MCP libraries, which get a refused token
  as a JSON-RPC error at once instead of waiting out their session timeout.
- `authfi.claude_agent.ClaudeAgentIntegration` (extra `claude-agent`): `options()` gives Claude Code
  the agent's token through `python -m authfi token` as its `apiKeyHelper` (renewed, never in its
  environment), the gateway's tools through an MCP server in your process, approvals through
  `can_use_tool` with a permission mode that makes Claude Code ask it, and OpenTelemetry content off.
- `authfi.adk.ADKIntegration` (extra `adk`): a LiteLLM model that puts the agent's token on each
  call outside ADK's traces; ADK's `McpToolset` on the gateway with the token on every request and
  Google's mTLS client kept off it (checked against the tested ADK, loudly); refusals the model
  reads; held calls as ADK's tool confirmation; ADK span content off by default.
- `AsyncGateway.token_refused`: the gateway's refusal of the token as an MCP client last saw it.
- `af.web_app(...)`: people signing in to a server-rendered app. The code flow with PKCE S256,
  `state` and `nonce`; the callback's `iss` required (RFC 9207); the pending sign-in encrypted
  (AES-256-GCM under `state_key`, bound to the client and redirect URI, ten minutes); the ID token
  verified (discovery's algorithm, `typ`, audience and `azp`, `nonce`, `at_hash`, `max_age`, `acr`);
  `StepUpRequired` for a weaker sign-in than asked; single-flight `refresh` (callers that overlap share one request; a later caller holding the spent refresh token is refused by the server, never answered from memory); `userinfo` checked
  against the person who signed in; `sign_out` revoking the refresh token before ending the
  session; `session()`, the person's token as a `TokenSource` shared through a `TokenStore`.
  `AsyncWebApp` is its asyncio twin.

### Changed

- Python 3.11 or later is required (was 3.8).
- The package is configured by issuer. `tenant`, `api_key`, `api_url` and `base_domain` are no
  longer how a project is named.
- HTTP goes through httpx (was `urllib`, with no timeout).

### Deprecated

- 0.1's `AuthFI(tenant, api_key, ...)`, `verify_token`, `require`, `require_role`,
  `current_user`, the 0.1 attributes, and `AgentAuthError`; 0.1's `AgentAuth(agent_id, key,
  platform_url=...)`, `AgentAuth.from_env()`, `token(audience)`, `auth_header`, `can`,
  `discovery`, `request_user_approval`, and `AgentKey`'s `key_id` and `private_pem`: each still
  works and warns
  `AuthFIDeprecationWarning`; removed at 1.0. See [docs/migration-0.2.md](docs/migration-0.2.md).

### Removed

- `register_permission`, `sync` and `auto_sync`, and `cloud_credentials` / `cloud_token` on the
  project: they called routes that do not exist. They raise `NotSupportedError` naming what
  replaces them.
- Verifying a token without an audience, or without a `kid`, and accepting a JWT that is not an
  access token. These are intended breaks; see the migration guide.
- `AgentAuth`'s default project (`https://platform.authfi.io`): an agent's project must be named.
- CIBA and token exchange return a `DelegatedToken`, not the token response `dict`, which put a
  person's token in a printable value.
- The single-module `authfi.py` layout. The names it exported are kept at the top level.
- A wait that runs out its own `timeout` no longer raises `OAuthError` `authorization_pending`, the
  server's word for "not decided yet": it raises `WaitTimeout` (`code == "wait_timeout"`), and a
  CIBA request that really expired raises `ConsentExpired`. An intended break.
- `send_to` on `af.agent()`, and a customer API's `resource` there: the agent sends its own token to
  AuthFI's endpoints, so it holds only a token for AuthFI. Use `agent.for_resource(resource,
  send_to=[...])` for a customer API.
