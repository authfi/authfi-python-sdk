# Google ADK

```bash
pip install "authfi[adk]"
```

```python
import authfi
from google.adk.runners import InMemoryRunner
from google.genai import types

af = authfi.connect_async()
adk = af.adk()
adk.configure_tracing()                                # before the runner is built
agent = adk.agent(name="support", instruction="Help with refunds.", model="claude-haiku-5-5")
runner = InMemoryRunner(agent=agent, app_name="support")
session = await runner.session_service.create_session(app_name="support", user_id="u1")

async def send(message):
    return [e async for e in runner.run_async(user_id="u1", session_id=session.id, new_message=message)]

events = await send(types.Content(role="user", parts=[types.Part(text="Refund order 4411")]))
for pending in adk.held_approvals(events):             # approval_id, tool, args
    events = await send(adk.answer(pending, confirmed=True))   # once the person decided in AuthFI
```

- **Model:** Claude through LiteLLM on `<issuer>/llm`. **Tools:** the gateway's, as an ADK MCP toolset on `<issuer>/mcp`.
- **Approval** is ADK's own confirmation request, answered on AuthFI's decision.
- **Steps:** `adk.export_steps()`.

More: [docs/frameworks.md](../frameworks.md#google-adk).
