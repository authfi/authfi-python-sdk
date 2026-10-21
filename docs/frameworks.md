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

## LangGraph

```bash
pip install "authfi[langgraph] @ git+https://github.com/authfi/authfi-python-sdk@main"
```

MCP tools run on asyncio, so the integration takes an asyncio credential.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from authfi import AsyncAuthFI
from authfi.langgraph import LangGraphIntegration, ask

af = AsyncAuthFI.from_env()
langgraph = LangGraphIntegration(af.agent_from_env())
tools = await langgraph.tools()
graph = langgraph.agent_graph(langgraph.chat_model("claude-haiku-5-5"), tools, checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "refund-4411"}}
state = await graph.ainvoke(ask("Refund order 4411"), config)
while "__interrupt__" in state:
    show(state["__interrupt__"][0].value["authfi_approvals"])   # approval_id, tool, args
    # ... the person decides in AuthFI, then:
    state = await graph.ainvoke(Command(resume=True), config)
```

**Tools.** `tools()` lists the gateway's MCP tools under the gateway's names (`<server>__<tool>`).
Every call carries the agent's current token, an `X-Request-Id` and a `traceparent`. A tool the
gateway marks `never_allowed` for this agent is left out (`include_never_allowed=True` keeps it);
the gateway still decides each call. There is one MCP client per integration.

**A refused call** reaches the model as the tool's error (`ToolMessage(status="error")`), naming the
reason and the scope the call needs, and the run carries on. **A refused token** (the agent was
revoked) raises `GatewayTokenRefused` and ends the run.

**A held call** is recorded in the graph's state (`authfi_holds`) and the model reads only "This
action is waiting for a person's approval in AuthFI." `approval_node` then calls `interrupt()` before
anything else, so LangGraph re-running it on resume repeats nothing. On resume it sends each held
call again with its approval, and the gateway decides:

| The person | The model reads |
|---|---|
| approved | the tool's answer, in place of the waiting message |
| denied, or the approval expired | the refusal |
| has not decided yet | nothing new: the call is held again and the graph interrupts again |

Resume with `Command(resume={"stop": True})` to give up; each held call then reaches the model as not
run. The approval id never passes through the model, and one the model writes into a call's
arguments is removed.

For your own graph, the pieces are public: `AuthFIState` (the messages and `authfi_holds`),
`langgraph.approval_node`, `route_after_tools`, and the MCP tool interceptor
`langgraph.interceptor`, which `langgraph.client()` installs. Use LangGraph's own `ToolNode` with
the tools; the guarding is in the tools themselves.

**What a checkpoint stores.** The graph state carries each held call's `args` (`Hold.args`), and a
checkpointer persists them. Tool arguments can hold personal data, so treat the checkpoint store as
holding it. It never holds a token or an approval secret: the approval id is in the state, and the
token is renewed from the agent's key.

## OpenAI Agents SDK

```bash
pip install "authfi[openai-agents] @ git+https://github.com/authfi/authfi-python-sdk@main"
```

```python
from agents import Agent, Runner
from authfi import AsyncAuthFI
from authfi.openai_agents import OpenAIAgentsIntegration

af = AsyncAuthFI.from_env()
openai_agents = OpenAIAgentsIntegration(af.agent_from_env())
openai_agents.configure_tracing()                  # first, before any run

async with openai_agents.mcp_server() as tools:
    agent = Agent(name="support", model=openai_agents.model("claude-haiku-5-5"), mcp_servers=[tools])
    config = openai_agents.run_config()
    result = await Runner.run(agent, "Refund order 4411", run_config=config)
    while result.interruptions:
        state = result.to_state()
        waiting = await tools.resolve(state, result.interruptions)
        if waiting:
            show(waiting)                          # approval_id, tool, args; the person decides in AuthFI
            continue                               # resolve again once they have
        result = await Runner.run(agent, state, run_config=config)
```

**Tracing.** The SDK's default exporter sends traces to OpenAI with the OpenAI key, and records
prompts, answers and tool results unless told otherwise. `configure_tracing(processors=(),
capture_content=False)` removes that exporter, sends spans only to the processors you name (each
wrapped in `AuthFITraceProcessor`, which removes content before your processor sees a span), and
turns tracing off when you name none. **It replaces every trace processor the application registered
before it**, not only the SDK's: pass the ones you want to keep in `processors=`. Call it before the
first run. The wrapper removes prompts, answers and tool inputs and outputs from generation,
response, function, custom, speech and transcription spans, and the `extra_headers`, `extra_args`
and `extra_query` settings from a span's model settings, so a run started without `run_config()`
still reaches your processors without content. Names, timings, ids, usage and the other settings
stay. `run_config()` names the agent in the trace's metadata (`authfi_agent_id`). The adapter does
not write the process environment.

**Models.** `model(name)` is Claude on the AI gateway through LiteLLM; `model(name, api="openai")` is
the SDK's Chat Completions model, for a project with an OpenAI-compatible provider. The agent's
token goes on each call; no OpenAI key is set, so nothing can export it.

**Tools.** `mcp_server()` is the SDK's streamable HTTP MCP server on the gateway: the current token
on every request, the gateway's tool names, `never_allowed` tools left out (`include_never_allowed=True`
keeps them), and a refusal handed to the model as the tool's error, naming why. A refused token
raises `GatewayTokenRefused`.

**Approvals.** The SDK asks for approval before a call; the gateway decides when it arrives. A tool
the gateway marks `may_require_approval` pauses the run as the SDK's own `ToolApprovalItem`.
`resolve(state, interruptions)` sends each paused call to the gateway:

- the gateway opens an approval: the call is returned as waiting (a `PendingApproval` with the
  `approval_id`), and `state` is left alone;
- the gateway has decided (the person approved or denied, or no approval was needed): the item is
  approved in `state`, and the run receives the gateway's answer, the tool's result or the
  refusal, without the call being sent again.

Call `resolve` again once the person has decided, then run again with `state`. Only the gateway's
decision approves an item, one call at a time. A hold the gateway did not announce reaches the
model as "This action is waiting for a person's approval in AuthFI.", and the next call to that tool pauses
the run. The approval id never reaches the model.
