# Quickstart: a governed agent in 10 minutes

From `pip install` to an agent that signs in with no secret, calls a model through AuthFI, calls a tool through AuthFI,
pauses for a person's approval and resumes, and shows up in the console. The example uses LangGraph; every other
framework is the same five steps (see [the framework pages](frameworks/README.md)).

Your project's address is its issuer, for example `https://acme.authfi.io`. Every address below is on it.

| What | Address |
|---|---|
| Models (the AI gateway) | `https://acme.authfi.io/llm` |
| Tools (the MCP gateway) | `https://acme.authfi.io/mcp` |
| Sign-in and tokens | `https://acme.authfi.io/oauth/token` |

## 1. Install (1 minute)

```bash
pip install "authfi[langgraph]"
```

Python 3.11 or later. The extra brings the framework adapter and nothing else; `pip install authfi` alone is the core.

## 2. Register the agent once in the console (3 minutes)

An agent here has no password, API key or key file. It proves where it runs, and AuthFI trusts that. You need a
group in **Identity > Groups** whose members answer for the agent, and a Kubernetes cluster
connected to your project (or a device, below).

1. Open the console (`console.authfi.io`), **Identity > AI Agents**, **Register agent**. Registration has the steps
   **Name it**, **Where it runs**, **What it may do**, **Review** and **Done**.
2. **Name it**: a name, **Works on its own**, and the group that answers for the agent.
3. **Where it runs**: **In a connected Kubernetes cluster (recommended)**, then the **Cluster**, **Namespace** and
   **Service account** the agent's pod runs as. Only pods running as that service account can sign in as this agent.
4. **What it may do**: under **Through the MCP gateway**, open **May call** and choose the tools the agent may use.
   Everything you do not choose is denied. Grant the model under **Integrations > MCP Gateway**, **Tools**,
   **Say who may call a tool**: caller type **An AI agent**, your agent, and a model such as
   `anthropic/claude-haiku-5-5` (connect the provider first under **Integrations > Connections**).
5. **Review**, then **Register**. The **Done** step shows the **Agent id** and the two values your code needs:

| Value | Looks like | Goes in |
|---|---|---|
| Project issuer | `https://acme.authfi.io` | `AUTHFI_ISSUER` |
| Agent id | `agt_...` | `AUTHFI_AGENT_ID` |
| Approved configuration (version and digest), after a person approves the agent's first run | `1`, `sha256:...` | `AUTHFI_CONFIG_VERSION`, `AUTHFI_CONFIG_DIGEST` |

A new agent has no approved configuration yet. On the **Done** step select **Observe its first run**, choose how long
(`1 hour`) and **Start observing**; nothing is denied while you watch. Run the agent once with
`AUTHFI_CONFIG_OBSERVE=1`. Then open the agent's **Approved configuration**, tick what it was seen doing, **Approve**,
and copy the version and digest. From then on the agent names that version.

On Kubernetes, project the service-account token for the project's origin (`audience: https://acme.authfi.io`,
`expirationSeconds: 600`). On a laptop, set `AUTHFI_DEVICE_KEY_FILE` to the device key the console gave the device.

## 3. Sign in (1 line)

```python
import authfi

af = authfi.connect_async()        # reads AUTHFI_ISSUER and AUTHFI_AGENT_ID, finds where it runs, signs in
print(af.run_id)                   # run-..., the id you will find in the console
```

There is no secret to paste. The instance makes its own key in memory, the platform checks the attestation, and
the token is bound to that key: every call proves it holds it.

## 4. One model call through `/llm`, one tool call through `/mcp` (2 minutes)

```python
import asyncio

from langgraph.checkpoint.memory import InMemorySaver

import authfi
from authfi.langgraph import ask


async def main() -> None:
    af = authfi.connect_async()
    langgraph = af.langgraph()
    tools = await langgraph.tools()                                  # the gateway's tools, from <issuer>/mcp
    model = langgraph.chat_model("claude-haiku-5-5", max_tokens=1024)   # calls go to <issuer>/llm
    graph = langgraph.agent_graph(model, tools, checkpointer=InMemorySaver())

    config = {"configurable": {"thread_id": "quickstart"}}
    state = await graph.ainvoke(ask("List the tools you have, then use one."), config)
    print(state["messages"][-1].content)


asyncio.run(main())
```

The model is called through AuthFI's AI gateway with the agent's token and no model-provider key in your code. Each tool is
the gateway's, under the gateway's name (`<server>__<tool>`). A tool the agent may not use is not offered; a call that
policy refuses comes back to the model as the tool's answer, with the reason, and the run carries on.

## 5. An approval pause and resume (2 minutes)

When policy says a person must approve a call, the call is held. The model reads only that the action is waiting, and
the graph pauses. Resume it when the person has decided:

```python
from langgraph.types import Command

state = await graph.ainvoke(ask("Refund order 4411"), config)
while "__interrupt__" in state:
    for held in state["__interrupt__"][0].value["authfi_approvals"]:
        print("waiting for approval", held["approval_id"], held["tool"], held["args"])
    input("Decide in the AuthFI console, then press Enter ")
    state = await graph.ainvoke(Command(resume=True), config)    # AuthFI's decision resumes the call
print(state["messages"][-1].content)
```

Approved, the tool runs and the model reads its answer. Denied or expired, the model reads the refusal. Nothing the
app or the model says can approve the call: only the person's decision in AuthFI does.

## 6. See the run in the console

- **Identity > AI Agents**, then your agent: its Agent id, its approved configuration and what it has done.
- **Approvals**: the held call, who it is waiting on and the decision, linked to the run.
- **Security > Audit**: the tape of every sign-in, tool call, model call, refusal and approval, in order. Enter `agent:agt_...` in **Who**.

To see each framework step (which model call, which tool, how long) as well, send the steps to AuthFI:

```bash
pip install "authfi[otel]"
```

```python
langgraph.export_steps()
```

## Next

- [One page per framework](frameworks/README.md)
- [Agents](agents.md): acting for a person, tasks, cloud credentials
- [The gateways](gateway.md)
- [Sign people in](signin.md) and [protect an API](verify.md)
