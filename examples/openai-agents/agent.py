"""An OpenAI Agents SDK agent on AuthFI: Claude through the AI gateway, tools through the MCP gateway.

    pip install 'authfi[openai-agents]'
    export AUTHFI_PLATFORM_URL=https://<project>.authfi.io
    export AUTHFI_AGENT_ID=agt_... AUTHFI_AGENT_KEY_FILE=/run/secrets/agent.key AUTHFI_AGENT_KID=k1
    python agent.py "Use the wiki tools to describe the structure of pallets/flask"

There is no glue here: identity, tracing defaults, the model credential, MCP
auth, tool names, refusals and approvals come from ``authfi.openai_agents``.
Do not set the AuthFI token as ``OPENAI_API_KEY``: nothing here needs it, and
an exporter that sends to OpenAI would send it.

When a tool call is held for a person's approval, the run pauses with an
interruption; this script waits until a person decides in the AuthFI console,
approves the interruption and runs on. The gateway decides whether the call
goes through.
"""

import asyncio
import os
import sys

from agents import Agent, Runner

from authfi import AgentAuth, AsyncAgentAuth, AuthFIError
from authfi.openai_agents import AuthFIMCPServer, AuthFIModel, configure_tracing, run_config

MODEL = os.environ.get("AUTHFI_MODEL", "claude-haiku-5-5")
INSTRUCTIONS = ("You are a research helper. Use the tools you are given to answer questions about public "
                "GitHub repositories. When a tool refuses, say what it said. Keep answers short.")


async def run(agent_auth, credential, question, wait_for_decision):
    """Ask ``question``; for each held call await ``wait_for_decision(approval)``, approve it and go on."""
    async with AuthFIMCPServer(credential) as tools:
        agent = Agent(name="research", instructions=INSTRUCTIONS, model=AuthFIModel(credential, MODEL),
                      mcp_servers=[tools])
        config = run_config(agent_auth)
        result = await Runner.run(agent, question, run_config=config)
        while result.interruptions:
            state = result.to_state()
            for item in result.interruptions:
                approval = tools.approval_for(item)
                if approval is None:
                    state.reject(item)       # not an AuthFI hold: this example approves nothing else
                    continue
                await wait_for_decision(approval)
                state.approve(item)
            result = await Runner.run(agent, state, run_config=config)
        return result


def console_decision(agent_auth):
    """Wait for a person's decision: read it when the project lets the agent, else ask here."""
    async def wait(approval):
        print("Waiting for a person to approve %s %s (approval %s) in the AuthFI console."
              % (approval["tool"], approval["args"], approval["approval_id"]))
        try:
            record = await agent_auth.wait_for_approval(approval["approval_id"], timeout=600)
            print("Approval %s: %s" % (approval["approval_id"], record.get("status")))
        except AuthFIError as e:
            if e.status != 403:
                raise
            await asyncio.get_running_loop().run_in_executor(
                None, input, "This agent may not read its approvals here. Press Enter once they have decided. ")
    return wait


async def main(question):
    # First: no default exporter (it would send to OpenAI), no content in spans. Pass your own
    # processors to trace somewhere; each one gets spans without prompts or tool results.
    configure_tracing()
    agent_auth = AsyncAgentAuth(AgentAuth.from_env())
    credential = agent_auth
    person = os.environ.get("AUTHFI_ACT_FOR")
    if person:
        print("Asking %s to let this agent act for them..." % person)
        credential = await agent_auth.token_as_user(person, binding_message="Let the research agent act for you")
    result = await run(agent_auth, credential, question, console_decision(agent_auth))
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Describe the wiki structure of pallets/flask."))
