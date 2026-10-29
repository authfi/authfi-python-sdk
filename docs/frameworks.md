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

Each extra covers the framework's major version and is tested at its lowest and newest release. Where a
framework has no public setting for its HTTP client, the adapter checks the one it uses when it is imported,
and raises `IntegrationError` naming the tested version rather than send a call without the token. LangSmith is
never installed by an extra: when it is present, `configure_tracing()` hides content from it; when it is not,
there is nothing to hide and `configure_tracing()` returns nothing.

## One line: `connect()`

`authfi.connect()` reads the environment and returns the credential, ready to build any of the integrations below:

```python
import authfi

af = authfi.connect()                    # blocking code: LangChain
chain = af.langchain()

af = authfi.connect_async()              # asyncio: LangGraph, OpenAI Agents, Claude Agent SDK, ADK
graph = af.langgraph()
```

It reads `AUTHFI_ISSUER`, then `AUTHFI_AGENT_ID` with `AUTHFI_AGENT_KEY_FILE` (an agent, and the default
when both are set) or `AUTHFI_CLIENT_ID` with `AUTHFI_CLIENT_SECRET` (a machine client; its token is for
`<issuer>/mcp`, or `AUTHFI_CLIENT_RESOURCE`). A missing variable is a `ConfigurationError` that names it.
`connect()` builds the same objects as `AuthFI.from_env().agent_from_env()` and
`.client_credentials_from_env()`; they are the way to pass your own options. Close the project when you are
done: `af.project.close()` (`await af.project.aclose()` for asyncio).

`af.langchain()`, `.langgraph()`, `.openai_agents()`, `.claude_agent()` and `.adk()` build the integration
for this credential and take its options (`project=`, `wait_for_decision=` and so on). A machine client's
gateway calls are labelled `client-credentials` in the `User-Agent`, so the console tells them from an
agent's.

### The run

Each credential has a run id, `af.run_id` (`run-` and 24 hex digits, or the one you give:
`connect(run_id="nightly-2026-10-29")`; `af.start_run()` begins another). Every gateway call carries
`X-Request-Id: <run id>.<n>`, every exported step carries `authfi.run_id`, and the integrations name it in the
metadata of the framework's own runs (`authfi_run_id`). The gateway also decides a run of its own from the
token and answers it in `AuthFI-Run-Id`.

### Steps to AuthFI

```bash
pip install "authfi[otel] @ git+https://github.com/authfi/authfi-python-sdk@main"
```

One call per integration sends the agent's steps (which agent ran, which tool, which model call, how long, how
it ended) to `<issuer origin>/otlp/v1/traces`, over OTLP/HTTP protobuf:

```python
af.langgraph().export_steps()            # also .langchain(), .openai_agents(), .adk()
options = await af.claude_agent().options(model="claude-haiku-5-5", export_steps=True)
```

- The token is the credential's current one, read on every export: a token that is renewed is never stale.
- Content is off by default: prompts, answers and tool arguments and results are removed before sending
  (`capture_content=True` is the tenant's decision), and AuthFI drops them again on receipt.
- Spans carry `authfi.agent_id` (`authfi.client_id` for a machine client) and `authfi.run_id`. AuthFI keeps
  each span as the token's agent's, whatever the payload says. A gateway call made inside a step carries that
  step's `traceparent`, so the gateway's record and the steps share a trace id.
- LangChain and LangGraph: LangChain's tracing (LangSmith's client) is switched to OpenTelemetry, so runs go to
  AuthFI and not to LangSmith. OpenAI Agents: the SDK's trace becomes GenAI spans (`invoke_agent`,
  `execute_tool`, `chat`), replacing the SDK's default exporter as `configure_tracing()` does. ADK: its own
  spans, on the global OpenTelemetry provider. Claude Agent SDK: Claude Code's own trace export (a Claude Code
  beta feature) pointed at AuthFI, with the headers from `python -m authfi headers` on each refresh.
- For any other OpenTelemetry-instrumented code, `authfi.otel.register(af)` adds the exporter to the SDK
  provider (or makes one); `authfi.otel.exporter(af)` is the exporter alone, for your own processor.

## LangChain

```bash
pip install "authfi[langchain-anthropic] @ git+https://github.com/authfi/authfi-python-sdk@main"
```

`authfi[langchain]` installs only `langchain-core`: `LangChainIntegration.gateway` is the credential for any chat
model that takes an HTTP client. `chat_model()` makes a Claude model, which needs the `langchain-anthropic` extra.

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

## Claude Agent SDK

```bash
pip install "authfi[claude-agent] @ git+https://github.com/authfi/authfi-python-sdk@main"
```

```python
from claude_agent_sdk import ClaudeSDKClient
from authfi import AsyncAuthFI
from authfi.claude_agent import ClaudeAgentIntegration, PendingApproval

async def until_decided(pending: PendingApproval) -> None:
    ...   # tell someone, and return once the person has decided in AuthFI

af = AsyncAuthFI.from_env()                        # AUTHFI_AGENT_KEY_FILE: the agent's key is a file
claude = ClaudeAgentIntegration(af.agent_from_env(), wait_for_decision=until_decided)
options = await claude.options(model="claude-haiku-5-5")
async with ClaudeSDKClient(options) as client:
    await client.query("Refund order 4411")
    async for message in client.receive_response():
        ...
```

**The model.** Claude Code runs `python -m authfi token` as its `apiKeyHelper` and calls the AI
gateway with that token, running it again every four minutes. The helper signs as the agent with
the agent's key file, so the agent's key must be loaded from a file (`AgentKey.from_file` or
`AUTHFI_AGENT_KEY_FILE`). Claude Code's options and environment hold the file's path, never the key
or a token. The options refuse an `env` (or `settings["env"]`) that would turn the token elsewhere or
switch content on: `ANTHROPIC_BASE_URL`, every `AUTHFI_*` and `OTEL_LOG_*` name, the other variables
the integration sets, and `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN`. Claude Code starts with this
process's environment as well, so the options blank those two (and `AUTHFI_AGENT_PRIVATE_KEY`) in
the environment they give it, and log that they were set here.

**The tools.** The gateway's tools are served to Claude Code by an MCP server in your process
(`mcp_server()`), which calls the gateway with the current token on every request. In Claude Code
they are named `mcp__authfi__<server>__<tool>`. A tool the gateway marks `never_allowed` is left out.

**Approvals.** Claude Code asks `can_use_tool` before a tool runs only when a permission mode asks
it to. The options set `permission_mode="default"` and a `PreToolUse` hook that sends each of the
gateway's tools to the permission flow, so `can_use_tool` always runs for them;
`permission_mode="bypassPermissions"` is refused. `can_use_tool` sends the call to the gateway:

| The gateway | What happens |
|---|---|
| allows it | Claude Code runs the tool and receives the gateway's answer; the call is not sent twice |
| refuses it | the model reads the refusal as the tool's error, naming why |
| holds it for approval | `wait_for_decision(pending)` is awaited, then the call is sent again with the approval, and the gateway decides |

Without `wait_for_decision`, a held call is refused for now with "This action is waiting for a
person's approval in AuthFI."; the same call made after the person decides goes with the approval. The
approval id never reaches the model, and one the model writes is removed. `can_use_tool` never
grants a lasting permission. Tools that are not AuthFI's are refused, unless you pass your own
`can_use_tool` to `options()`, which then decides them.

The integration remembers the answer to a call for up to two minutes (`MEMORY_SECONDS`), from
`can_use_tool` until Claude Code runs the tool. A call Claude Code never ran is then forgotten, and
`claude.reset()` forgets every one at once: call it when a run ends, so the next run, with the same
tool and arguments, is sent afresh.

**A refused token** stops the run (`can_use_tool` interrupts it), and
`claude.token_refused` holds the `GatewayTokenRefused`.

**Content.** Claude Code's OpenTelemetry content switches (`OTEL_LOG_USER_PROMPTS`,
`OTEL_LOG_ASSISTANT_RESPONSES`, `OTEL_LOG_TOOL_CONTENT`, `OTEL_LOG_TOOL_DETAILS`,
`OTEL_LOG_RAW_API_BODIES`) are set off unless `capture_content=True`, and the agent's id is a
resource attribute (`authfi.agent_id`). `configure_tracing()` turns on LangSmith's Claude Agent SDK
tracing with inputs and outputs hidden and `authfi_agent_id` in each run's metadata.

The integration runs Claude Code as the agent itself; a token the agent holds for a person is not
accepted (`NotSupportedError`), as Claude Code could only receive it as a fixed value.

## Google ADK

```bash
pip install "authfi[adk] @ git+https://github.com/authfi/authfi-python-sdk@main"
```

```python
from google.adk.runners import InMemoryRunner
from google.genai import types
from authfi import AsyncAuthFI
from authfi.adk import ADKIntegration

af = AsyncAuthFI.from_env()
adk = ADKIntegration(af.agent_from_env())
adk.configure_tracing()                            # before the runner is built
agent = adk.agent(name="support", instruction="Help with refunds.", model="claude-haiku-5-5")
runner = InMemoryRunner(agent=agent, app_name="support")
session = await runner.session_service.create_session(app_name="support", user_id="u1")

async def send(message):
    return [e async for e in runner.run_async(user_id="u1", session_id=session.id, new_message=message)]

events = await send(types.Content(role="user", parts=[types.Part(text="Refund order 4411")]))
for pending in adk.held_approvals(events):         # approval_id, tool, args
    ...                                            # once the person has decided in AuthFI:
    events = await send(adk.answer(pending, confirmed=True))
```

**The model.** `model(name)` is Claude on the AI gateway through LiteLLM. It puts the agent's
current token on each call itself, in the request's HTTP headers, which ADK leaves out of its
traces; LiteLLM's required API key is a placeholder. A refused token raises `GatewayTokenRefused`.

**The tools.** `toolset()` is ADK's `McpToolset` on the gateway with the gateway's HTTP client, so
every request carries the current token, including a session ADK opens outside a run. A tool the
gateway marks `never_allowed` is left out (`include_never_allowed=True` keeps it), and the tool list
is cached for a minute. ADK replaces an MCP connection's HTTP client with a Google mTLS client when
it finds Google credentials, which would send no AuthFI token; that is turned off for this toolset
only. The integration checks, when the toolset is made, that the ADK it reaches into is the one it
is tested with (`google-adk` 2.12 and 2.13), and raises `IntegrationError` otherwise.

**Refusals** reach the model as `{"error": "<why>"}`. ADK turns every MCP failure into text for the
model; when the gateway refused the token itself, the tool callback raises `GatewayTokenRefused`
instead, ending the run. The callback reads the refusal from that call's own response, so tool
calls ADK runs in parallel cannot clear or inherit one another's.

**Approvals.** A held call becomes ADK's tool confirmation: `held_approvals(events)` lists them, and
`answer(pending, confirmed=...)` is the message that answers one. A pending approval is
`authfi.gateway.PendingApproval` (`approval_id`, `tool`, `args`) with ADK's `call_id`. Send it once the person has
decided in AuthFI: the call is then sent with the approval, and the gateway decides. ADK confirms a
call once, so a call confirmed before the person decided reaches the model as still waiting, and
the same call made again goes under the same approval and asks again; no second approval is
opened. A call answered `confirmed=False` is not sent. The approval id travels in ADK's
confirmation, which ADK keeps out of what the model reads, and one the model writes is removed.

**Content.** `configure_tracing(capture_content=False)` sets ADK's
`ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS` off and the OpenTelemetry GenAI content mode to `NO_CONTENT`,
and adds `authfi.agent_id` to `OTEL_RESOURCE_ATTRIBUTES`. Call it before the runner is built.

`agent(name=..., instruction=..., model=..., tools=...)` builds an `LlmAgent` with all of it;
`before_tool_callback` and `after_tool_callback` you pass run after the integration's.
