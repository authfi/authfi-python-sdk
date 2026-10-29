# Pydantic AI, CrewAI and LlamaIndex

Three more frameworks, each through its own tool, model and tracing seam, with the capabilities the other
adapters have: the gateway's governed tools over MCP, approvals, the run id, and steps over OTLP
(`authfi.otel`). Each takes an asyncio credential (`AsyncAuthFI.agent(...)`), like the OpenAI Agents, Claude
Agent and ADK adapters.

| Framework | Extra | Module | Range proven (lowest and newest in CI) |
| --- | --- | --- | --- |
| Pydantic AI | `pydantic-ai` | `authfi.pydantic_ai` | `pydantic-ai-slim` 2.60 to 2.63 |
| CrewAI | `crewai` | `authfi.crewai` | `crewai` 1.15.29 to 1.15.33 |
| LlamaIndex | `llama-index` | `authfi.llama_index` | `llama-index-core` 0.14.5 to 0.14.25, `llama-index-tools-mcp` 0.4.6 to 0.4.8 |

Install one: `pip install "authfi[pydantic-ai]"`. CrewAI pins `anthropic` and `openai` below what the other
extras allow, so install it alone in its environment.

## What each does

- **Model.** The framework's own Anthropic model over the AI gateway; the agent's current token goes on every
  request through the framework's own seam (Pydantic AI's `AnthropicModel` with the gateway's HTTP client,
  CrewAI's transport interceptor, LlamaIndex's `Anthropic` LLM with the gateway's asyncio client). No
  Anthropic key is needed.
- **Tools.** The gateway's MCP tools through the framework's own MCP seam (`MCPToolset`, `BaseTool`,
  `McpToolSpec`), under the gateway's names. Tools the gateway says are `never_allowed` are left out. A
  refusal reaches the model naming why, and the run carries on.
- **Approvals.** Pydantic AI defers the call as `ApprovalRequired` and `AuthFIToolset.resolve()` sends it to
  the gateway; CrewAI and LlamaIndex cannot pause a run, so the tool waits for the person, tells
  `on_approval` once, and gives up after `approval_timeout` without treating silence as a yes. In all three
  the approval id never reaches the model, and one the model writes is removed.
- **A token the gateway refuses (401)** ends the run with `GatewayTokenRefused`. CrewAI and LlamaIndex hand a
  failed call to the model as text, so wrap the run in `async with <integration>.guard():`.
- **Steps.** `export_steps()` sends the run's steps to AuthFI tagged `authfi.agent_id` and `authfi.run_id`,
  with no words: Pydantic AI's own instrumentation with content off; CrewAI's event bus and LlamaIndex's
  instrumentation spans as OpenTelemetry GenAI spans.

## Limits

- The adapters stay on `mcp` 1.x. The `mcp` 2 SDK drops the gateway's governance annotation from
  `tools/list`, so `never_allowed` tools would be offered and approvals would be found only when the gateway
  holds the call. The weekly canary installs `mcp` 2 to show when that changes.
- CrewAI's blocking `kickoff()` is refused with a message that says to use `akickoff()`.
- LlamaIndex needs `anthropic` below 2 (its Anthropic package builds messages from the 1.x types).
