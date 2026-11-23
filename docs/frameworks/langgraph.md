# LangGraph

```bash
pip install "authfi[langgraph]"
```

```python
import authfi
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from authfi.langgraph import ask

af = authfi.connect_async()
lg = af.langgraph()
graph = lg.agent_graph(lg.chat_model("claude-haiku-5-5"), await lg.tools(), checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "run-1"}}
state = await graph.ainvoke(ask("Refund order 4411"), config)
while "__interrupt__" in state:                       # a call a person must approve
    print(state["__interrupt__"][0].value["authfi_approvals"])
    state = await graph.ainvoke(Command(resume=True), config)   # once they decided in AuthFI
```

- **Model** through `<issuer>/llm`; **tools** from `<issuer>/mcp`, under the gateway's names. A refused call goes back
  to the model as the tool's error.
- **Approval** is LangGraph's own `interrupt()`. The model never reads the approval id.
- **Steps:** `lg.export_steps()`. Name each graph (`agent_graph(..., name="support")`) so a sub-agent's calls read
  as its own.

More: [docs/frameworks.md](../frameworks.md#langgraph).
