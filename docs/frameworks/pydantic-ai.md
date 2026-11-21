# Pydantic AI

```bash
pip install "authfi[pydantic-ai]"
```

```python
import authfi
from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults

af = authfi.connect_async()
pai = af.pydantic_ai()

async with pai.toolset() as tools:
    agent = Agent(pai.model("claude-haiku-5-5"), toolsets=[tools], output_type=[str, DeferredToolRequests])
    result = await agent.run("Refund order 4411")
    while isinstance(result.output, DeferredToolRequests):     # a call a person must approve
        decisions = DeferredToolResults()
        if waiting := await tools.resolve(result.output, decisions):
            print(waiting)                                     # approval_id, tool, args
            continue                                           # resolve again once they decided in AuthFI
        result = await agent.run(message_history=result.all_messages(), deferred_tool_results=decisions)
```

- **Model:** Pydantic AI's `AnthropicModel` on `<issuer>/llm`, no provider key. **Tools:** `MCPToolset` on `<issuer>/mcp`.
- **Approval** is Pydantic AI's deferred tool call, approved only once AuthFI has decided.
- **Steps:** `pai.export_steps()`, with content off.
