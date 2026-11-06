# Changelog

All notable changes to this package. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [0.2.0]

The SDK is rebuilt on one core that every area shares: people signing in, APIs verifying tokens,
machine clients and AI agents. See [docs/migration-0.2.md](docs/migration-0.2.md).

### Added

- A bound agent (`connect()` returning an `InstanceAuth` or `AsyncInstanceAuth`) asks a person and opens approvals with
  its own credentials and no agent key: `request_user_token` and `request_approval` present its access token with
  a DPoP proof of the instance key (method, URL and `ath`), `await_user_token` presents its attestation as the token
  request does and returns a token bound to the instance key, and the approval read works as before. An agent with a
  key signs in as it did. An instance admitted only to be observed is refused by the platform (403
  `unauthorized_client` or `observe_only_instance`); the error now says to approve a configuration version and run
  with `AUTHFI_CONFIG_VERSION` and `AUTHFI_CONFIG_DIGEST`. A wait that outlives the attestation renews it before the
  next poll.
- Observe mode: `AUTHFI_CONFIG_OBSERVE=1`, `connect(observe=True)` or `Configuration.observe()` signs an instance in
  naming no version (`version` 0, empty `digest`), for a new agent while a person has an observation window open
  for it. It is never the default: an instance that names no version and does not ask for this is still refused
  before anything is sent, and naming a version with it is a `ConfigurationError`. When the platform refuses it, the
  error says to ask a person to open an observation window for the agent in the console, or to set the version and
  digest.
- Claude Agent SDK: `options()` sets `CLAUDE_CODE_ATTRIBUTION_HEADER=0`, so Claude Code sends no `x-anthropic-billing-header` system part (its suffix changes with the first user message), and `tools=["Agent"]`, so the model is offered the agent's AuthFI tools and the sub-agent tool and none of Claude Code's other built-ins. The owner's `env` and `tools` win; see docs/frameworks.md.
- The steps name their agents in the OpenTelemetry GenAI standard: the agent behind a step or a gateway call is the
  nearest `gen_ai.agent.name` above it. LangGraph: the run of each compiled graph is an `invoke_agent` step named for
  the graph (`agent_graph(..., name=...)`; `LangGraph` when LangGraph gave it none), so a sub-agent's graph run names
  the sub-agent. Claude Agent SDK: with `export_steps`, Claude Code's trace goes through the loopback proxy, which
  says `gen_ai.agent.name` on each span: a sub-agent's type (from the `SubagentStart` hook) and the main loop's name
  (`options(agent_name=...)`, else `claude_code`).
- A gateway call names the step that made it. With `export_steps()`, every tool call, model call and cloud
  credential request carries the `traceparent` (trace id and span id) of the OpenAI Agents, LangChain, LangGraph,
  ADK or Claude Agent SDK step executing when it is made, as that step is exported to `/otlp/v1/traces`; model
  calls LiteLLM makes with its own HTTP client carry it too, and an MCP call is sent with the step in its
  `_meta.traceparent`. With no step executing the value is fresh with the sampled flag off, as before.
  LangSmith's OpenTelemetry bridge now makes a run's span from the run's UUID, so the id is known when the run
  starts.
- A model call that goes to the AI gateway with no DPoP proof is said once in the log: the gateway refuses such a
  call once it enforces proofs on model calls, unless the workload is caged.
  `authfi.gateway.learn_nonce_from` keeps the nonce a gateway asked for in an answer the caller cannot resend.
- CrewAI's model calls answer the gateway's DPoP nonce: the nonce is kept and the provider SDK is told to try
  again, where a nonce challenge was reported as a refused token and never learned.
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
  codes.
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
  model calls through a loopback proxy in your process that sends them with the agent's credential (never in
  Claude Code's environment), the gateway's tools through an MCP server in your process, approvals through
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

### Fixed

- `approval()`, `wait_for_approval()` and `transaction_token()` prove the token when it is bound to a key. They sent it as
  Bearer, which AuthFI refuses, so an agent could not learn that a person had decided.
- Claude Code's model calls and its tools are one run of the agent's key. The model calls went with a Bearer
  token from a helper process, so the gateway counted them on the agent's one coarse run while the tools
  (which prove the key) were on another, and the steps Claude Code exported (also sent with that Bearer token) were
  on the coarse run too. The options now point Claude Code at a loopback proxy that sends its model calls with the
  process's credential and name the gateway's run in the exported trace.
- `ClaudeAgentIntegration.options(export_steps=True)` could export nothing. Claude Code runs `otelHeadersHelper`
  at its first export, which for a short run is while it shuts down, when it gives its exporter under two
  seconds; the helper takes about a second, and a slower start (a busy machine) ran out of time with no request
  sent. Claude Code now exports every second and keeps the helper's headers for four minutes.

### Changed

- The AI gateway is its own door, `<issuer>/llm`. `model_base_url("anthropic")` is `<issuer>/llm` and `("openai")`
  is `<issuer>/llm/v1` (were `<issuer>/mcp` and `<issuer>/mcp/v1`), the red team sends to `/llm/v1/...`, and
  `Gateway.llm_url` names the door. Each door takes only a token requested for it, so a gateway now holds a second
  token, requested with `resource=<issuer>/llm` on the same key (one run) and cached apart from the tool token: model
  calls carry it, tool calls and the cloud credential calls carry the other, and neither goes to the other door or to
  a customer's API. A credential that cannot make a model token (a token an agent holds for a person) is refused at
  `/llm` instead of being sent there. A task's root is the one token both doors take.
- Cloud credentials are issued by the `cloud.credentials` tool, a `tools/call` on `<issuer>/mcp`, and the secret is
  redeemed once at `POST <issuer>/mcp/cloud/credentials/{credential_id}/secret`. The `POST /mcp/v1/cloud/credentials`
  door is gone; `Gateway.cloud_credentials` and `agent.cloud.credentials` are called as before.
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
- `AgentAuth` no longer has a default project: an agent's project must be named.
- CIBA and token exchange return a `DelegatedToken`, not the token response `dict`, which put a
  person's token in a printable value.
- The single-module `authfi.py` layout. The names it exported are kept at the top level.
- A wait that runs out its own `timeout` no longer raises `OAuthError` `authorization_pending`, the
  server's word for "not decided yet": it raises `WaitTimeout` (`code == "wait_timeout"`), and a
  CIBA request that really expired raises `ConsentExpired`. An intended break.
- `send_to` on `af.agent()`, and a customer API's `resource` there: the agent sends its own token to
  AuthFI's endpoints, so it holds only a token for AuthFI. Use `agent.for_resource(resource,
  send_to=[...])` for a customer API.
