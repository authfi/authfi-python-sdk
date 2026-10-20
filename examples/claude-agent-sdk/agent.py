"""A Claude Agent SDK agent on AuthFI: Claude Code with the AI gateway as its model and the MCP gateway as its tools.

    pip install 'authfi[claude-agent]'
    export AUTHFI_PLATFORM_URL=https://<project>.authfi.io
    export AUTHFI_AGENT_ID=agt_... AUTHFI_AGENT_KEY_FILE=/run/secrets/agent.key AUTHFI_AGENT_KID=k1
    python agent.py "Use the wiki tools to describe the structure of pallets/flask"

There is no glue here. Claude Code gets the agent's token from ``authfi token``
(its apiKeyHelper) and the MCP header from ``authfi headers``, both signed
with the agent's key file and refreshed before they expire. A call the
gateway holds for a person's approval is noted by a hook; when the model calls
it again, ``can_use_tool`` waits for the decision and sends the call with the
approval id. Content logging is off.

Acting for a person: set ``AUTHFI_ACT_FOR`` to their sign-in name. They
approve on their own device, and the session carries a token for them.
"""

import asyncio
import os
import sys
import tempfile
import uuid

from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ToolResultBlock, UserMessage

from authfi import AgentAuth, AsyncAgentAuth, AuthFIError
from authfi.claude_agent import AuthFIClaudeClient, claude_options

MODEL = os.environ.get("AUTHFI_MODEL", "claude-haiku-5-5")
SYSTEM = ("You are a research helper. Use the tools you are given to answer questions about public GitHub "
          "repositories. When a tool refuses, say what it said. Keep answers short.")


def console_decision(agent):
    """Wait for a person's decision: read it when the project lets the agent, else ask here."""
    reader = AsyncAgentAuth(agent)

    async def wait(approval):
        print("Waiting for a person to approve %s %s (approval %s) in the AuthFI console."
              % (approval["tool"], approval["args"], approval["approval_id"]))
        try:
            return await reader.wait_for_approval(approval["approval_id"], timeout=600)
        except AuthFIError as e:
            if e.status != 403:
                raise
            await asyncio.get_running_loop().run_in_executor(
                None, input, "This agent may not read its approvals here. Press Enter once they have decided. ")
            return True     # the gateway decides when the call is sent again
    return wait


async def run(agent, credential, prompts, wait_for_decision, run_id=None, show=print):
    """Send ``prompts`` in one Claude Code session; show each message; return the final texts."""
    options = claude_options(credential, model=MODEL, url=agent.platform_url + "/mcp",
                             wait_for_decision=wait_for_decision, run_id=run_id or str(uuid.uuid4()),
                             system_prompt=SYSTEM, tools=[], strict_mcp_config=True, setting_sources=[],
                             max_turns=10, cwd=tempfile.gettempdir())
    finals = []
    async with AuthFIClaudeClient(options, agent=agent) as client:
        for prompt in prompts:
            await client.query(prompt)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            show("assistant: " + block.text)
                elif isinstance(message, UserMessage) and isinstance(message.content, list):
                    for block in message.content:
                        if isinstance(block, ToolResultBlock):
                            show("tool%s: %s" % (" error" if block.is_error else "", str(block.content)[:300]))
                elif isinstance(message, ResultMessage):
                    finals.append(message.result)
    return finals


async def main(question):
    agent = AgentAuth.from_env()
    credential = agent
    person = os.environ.get("AUTHFI_ACT_FOR")
    if person:
        print("Asking %s to let this agent act for them..." % person)
        credential = await AsyncAgentAuth(agent).token_as_user(
            person, binding_message="Let the research agent act for you")
    await run(agent, credential, [question], console_decision(agent))


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Describe the wiki structure of pallets/flask."))
