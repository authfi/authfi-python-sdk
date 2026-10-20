"""A LangGraph agent on AuthFI: Claude through the AI gateway, tools through the MCP gateway.

    pip install 'authfi[langgraph]'
    export AUTHFI_PLATFORM_URL=https://<project>.authfi.io
    export AUTHFI_AGENT_ID=agt_... AUTHFI_AGENT_KEY_FILE=/run/secrets/agent.key AUTHFI_AGENT_KID=k1
    python agent.py "Use the wiki tools to describe the structure of pallets/flask"

There is no glue here: identity, MCP auth, tool names, refusals, approvals
and run linking come from ``authfi.langgraph``. When a tool call is held for
a person's approval the run pauses; this script then waits until a person
decides in the AuthFI console, and resumes.

Tracing: set ``LANGSMITH_TRACING=true`` (and LangSmith's key) to trace the
run. ``metadata_only_tracing()`` keeps prompts and tool results out of it.

Acting for a person: set ``AUTHFI_ACT_FOR`` to their sign-in name. They
approve on their own device, and the run then carries a token for them.
"""

import asyncio
import os
import sys
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from authfi import AgentAuth, AsyncAgentAuth, AuthFIError
from authfi.langchain import metadata_only_tracing
from authfi.langgraph import AuthFIGateway, ask

MODEL = os.environ.get("AUTHFI_MODEL", "claude-haiku-5-5")
PROMPT = ("You are a research helper. Use the tools you are given to answer questions about public "
          "GitHub repositories. When a tool refuses, say what it said. Keep answers short.")


async def build(agent):
    """The graph, ready to run: the model and the project's governed tools."""
    gateway = AuthFIGateway(agent)
    tools = await gateway.tools()
    return gateway.agent_graph(gateway.chat_model(MODEL, max_tokens=600), tools,
                               checkpointer=InMemorySaver(), prompt=PROMPT)


async def run(graph, question, wait_for_decision, thread_id=None):
    """Ask ``question``; whenever a call is held, await ``wait_for_decision(approval)`` and resume."""
    config = {"configurable": {"thread_id": thread_id or uuid.uuid4().hex}}
    state = await graph.ainvoke(ask(question), config)
    while "__interrupt__" in state:
        for approval in state["__interrupt__"][0].value["authfi_approvals"]:
            await wait_for_decision(approval)
        state = await graph.ainvoke(Command(resume=True), config)
    return state


def console_decision(agent):
    """Wait for a person's decision: read it when the project lets the agent, else ask here."""
    async def wait(approval):
        print("Waiting for a person to approve %s %s (approval %s) in the AuthFI console."
              % (approval["tool"], approval["args"], approval["approval_id"]))
        try:
            record = await agent.wait_for_approval(approval["approval_id"], timeout=600)
            print("Approval %s: %s" % (approval["approval_id"], record.get("status")))
        except AuthFIError as e:
            if e.status != 403:
                raise
            await asyncio.get_running_loop().run_in_executor(
                None, input, "This agent may not read its approvals here. Press Enter once they have decided. ")
    return wait


async def main(question):
    if os.environ.get("LANGSMITH_TRACING") == "true":
        metadata_only_tracing()
    agent = AsyncAgentAuth(AgentAuth.from_env())
    credential = agent
    person = os.environ.get("AUTHFI_ACT_FOR")
    if person:
        # Act for a person: they approve on their own device (CIBA), and every call
        # after that carries a token that names them and this agent.
        print("Asking %s to let this agent act for them..." % person)
        credential = await agent.token_as_user(person, binding_message="Let the research agent act for you")
    graph = await build(credential)
    state = await run(graph, question, console_decision(agent))
    print(state["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Describe the wiki structure of pallets/flask."))
