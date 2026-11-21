# Frameworks

One credential, one line, and each framework's own tool, model, approval and tracing seams go through AuthFI:

```python
import authfi

af = authfi.connect_async()      # authfi.connect() for blocking code (LangChain)
```

| Framework | Install | Page |
|---|---|---|
| LangChain | `pip install "authfi[langchain-anthropic]"` | [langchain](langchain.md) |
| LangGraph | `pip install "authfi[langgraph]"` | [langgraph](langgraph.md) |
| OpenAI Agents SDK | `pip install "authfi[openai-agents]"` | [openai-agents](openai-agents.md) |
| Claude Agent SDK | `pip install "authfi[claude-agent]"` | [claude-agent](claude-agent.md) |
| Google ADK | `pip install "authfi[adk]"` | [adk](adk.md) |
| Pydantic AI | `pip install "authfi[pydantic-ai]"` | [pydantic-ai](pydantic-ai.md) |
| CrewAI | `pip install "authfi[crewai]"` (in its own environment) | [crewai](crewai.md) |
| LlamaIndex | `pip install "authfi[llama-index]"` | [llama-index](llama-index.md) |

Start with the [quickstart](../quickstart.md). Every page has the same four parts: models go through
`<issuer>/llm`, tools through `<issuer>/mcp`, a call a person must approve pauses and resumes on the person's
decision, and `export_steps()` (extra `otel`) sends the framework's steps to AuthFI. The token is attached to each
request by the integration and never appears in a framework setting or a trace.
