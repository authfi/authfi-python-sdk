"""An OpenAI Agents SDK agent on the AuthFI gateways that waits for a person's approval."""

from __future__ import annotations

import asyncio
import os
import sys

from agents import Agent, Runner

from authfi import AsyncAuthFI, GatewayTokenRefused
from authfi.openai_agents import OpenAIAgentsIntegration


async def run(question: str) -> int:
    """Run the agent on ``question``, pausing while a call waits for approval."""
    async with AsyncAuthFI.from_env() as af:
        openai_agents = OpenAIAgentsIntegration(af.agent_from_env())
        openai_agents.configure_tracing()
        config = openai_agents.run_config()
        model = openai_agents.model(os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"))
        try:
            async with openai_agents.mcp_server() as tools:
                agent = Agent(name="research", model=model, mcp_servers=[tools])
                result = await Runner.run(agent, question, run_config=config)
                while result.interruptions:
                    state = result.to_state()
                    waiting = await tools.resolve(state, result.interruptions)
                    if waiting:
                        for pending in waiting:
                            print(f"waiting for approval {pending.approval_id}: {pending.tool} {pending.args}")
                        await asyncio.to_thread(input, "Decide in the AuthFI console, then press Enter: ")
                        continue
                    result = await Runner.run(agent, state, run_config=config)
        except GatewayTokenRefused as refused:
            print(f"the gateway refused this agent's token: {refused}", file=sys.stderr)
            return 1
        print(result.final_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(" ".join(sys.argv[1:]) or "Summarise the wiki of pallets/flask")))
