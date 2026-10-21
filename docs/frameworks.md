# Agent frameworks

Each adapter gives a framework the agent's credential through the AuthFI gateways, the same way:

- **The token never becomes a framework setting.** A model's or tool client's HTTP client is the
  gateway's, which puts the current token on each request. Frameworks record their settings in
  traces; a token there is a token anyone with the trace can spend.
- **Traces carry no content by default.** `configure_tracing()` records names, timings and ids.
  Prompts, answers and tool results hold people's data; `capture_content=True` is your decision.
- **A refused token ends the run** with `GatewayTokenRefused`: no call can succeed with it.

Any credential works: the agent itself (`af.agent(...)`), or a token it holds for a person (a
`DelegatedToken`). A blocking credential gives blocking framework objects, an asyncio one gives
asyncio objects. Every adapter built on the same agent shares the agent's one gateway
(`agent.gateway()`) and its HTTP client, so making adapters does not make connections.

Each extra pins the framework versions the adapter is tested with. Where a framework has no public
setting for its HTTP client, the adapter checks the one it uses when it is imported, and raises
`IntegrationError` naming the tested range rather than send a call without the token.

## LangChain

```bash
pip install "authfi[langchain] @ git+https://github.com/authfi/authfi-python-sdk@main"
```

```python
from authfi import AuthFI
from authfi.langchain import LangChainIntegration

af = AuthFI.from_env()
langchain = LangChainIntegration(af.agent_from_env())
langchain.configure_tracing()                       # LangSmith: no inputs or outputs

model = langchain.chat_model("claude-haiku-5-5", max_tokens=512)
print(model.invoke("Summarise the open refunds").content)
```

`chat_model()` returns a `ChatAnthropic` whose calls go to the AI gateway with the agent's token.
It takes `ChatAnthropic`'s arguments, except those that carry a credential or an address
(`api_key`, `base_url`, an `Authorization` header), which the integration sets. The model's
metadata names the agent (`authfi_agent_id`), so a trace shows which agent made each call.

`configure_tracing(capture_content=False, **client_kwargs)` installs the LangSmith client that
LangChain traces with and returns it; `client_kwargs` go to `langsmith.Client`. It is process-wide
(`langsmith.configure`): an app that has its own LangSmith client should not call it, or should pass
that client's settings in `client_kwargs`.
