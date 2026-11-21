# LangChain

```bash
pip install "authfi[langchain-anthropic]"
```

```python
import authfi

af = authfi.connect()                                  # blocking code; connect_async() for ainvoke
model = af.langchain().chat_model("claude-haiku-5-5", max_tokens=512)
print(model.invoke("Summarise the open refunds").content)
```

- **Model:** a `ChatAnthropic` whose calls go to `<issuer>/llm` with the agent's token and no provider key. It takes
  `ChatAnthropic`'s arguments except credentials and addresses.
- **Tracing:** `af.langchain().configure_tracing()` keeps prompts and answers out of LangSmith.
  `export_steps()` sends the steps to AuthFI.
- **Tools and approvals:** use [LangGraph](langgraph.md), which gives a graph the gateway's tools and pauses it for
  approvals.

More: [docs/frameworks.md](../frameworks.md#langchain).
