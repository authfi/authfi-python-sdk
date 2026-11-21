# OpenAI Agents SDK

```bash
pip install "authfi[openai-agents]"
```

```python
import authfi
from agents import Agent, Runner

af = authfi.connect_async()
oa = af.openai_agents()
oa.configure_tracing()                                 # first, before any run

async with oa.mcp_server() as tools:
    agent = Agent(name="support", model=oa.model("claude-haiku-5-5"), mcp_servers=[tools])
    config = oa.run_config()
    result = await Runner.run(agent, "Refund order 4411", run_config=config)
    while result.interruptions:
        state = result.to_state()
        if waiting := await tools.resolve(state, result.interruptions):
            print(waiting)                             # approval_id, tool, args; the person decides in AuthFI
            continue
        result = await Runner.run(agent, state, run_config=config)
```

- **Model** through `<issuer>/llm`, **tools** through `<issuer>/mcp`, with the agent's token on each request.
- **Approval** is the SDK's own interruption: `resolve()` approves a paused call only once AuthFI has decided it.
- **Tracing:** `configure_tracing()` keeps content out of traces; `export_steps()` sends the steps to AuthFI.

More: [docs/frameworks.md](../frameworks.md#openai-agents-sdk).
