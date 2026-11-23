# LlamaIndex

```bash
pip install "authfi[llama-index]"
```

```python
import authfi
from llama_index.core.agent.workflow import FunctionAgent

af = authfi.connect_async()
llama = af.llama_index()

agent = FunctionAgent(tools=await llama.tools(), llm=llama.llm("claude-haiku-5-5"))
async with llama.guard():                                      # a token the gateway refuses ends the run
    answer = await agent.run("Describe pallets/flask's wiki")
```

- **Model:** LlamaIndex's `Anthropic` LLM on `<issuer>/llm`. **Tools:** `McpToolSpec` on `<issuer>/mcp`, under the gateway's names.
- **Approval:** a held tool waits for the person (`on_approval`, `approval_timeout`) and returns the answer or the refusal.
- **Steps:** `llama.export_steps()`.
