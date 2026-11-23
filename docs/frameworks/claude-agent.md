# Claude Agent SDK

```bash
pip install "authfi[claude-agent]"
```

```python
import authfi
from claude_agent_sdk import ClaudeSDKClient
from authfi.claude_agent import PendingApproval

async def until_decided(pending: PendingApproval) -> None:
    print("waiting for", pending.approval_id, pending.tool)   # return once the person decided in AuthFI

af = authfi.connect_async()
claude = af.claude_agent(wait_for_decision=until_decided)
options = await claude.options(model="claude-haiku-5-5")
async with ClaudeSDKClient(options) as client:
    await client.query("Refund order 4411")
    async for message in client.receive_response():
        print(message)
```

- **Model:** Claude Code's model calls leave your process through a loopback proxy that sends them to
  `<issuer>/llm` with the agent's token. Claude Code holds no key or token.
- **Tools:** the gateway's tools through `<issuer>/mcp`. A held call waits in `wait_for_decision` and is sent again
  with the approval.
- **Steps:** `options(export_steps=True)`.

More: [docs/frameworks.md](../frameworks.md#claude-agent-sdk), and [Claude Code as yourself](../claude-code.md).
