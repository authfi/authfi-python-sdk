"""A Claude Agent SDK agent on the AuthFI gateways that waits for a person's approval."""

from __future__ import annotations

import asyncio
import os
import sys

from claude_agent_sdk import AssistantMessage, ClaudeSDKClient, TextBlock

from authfi import AsyncAuthFI
from authfi.claude_agent import ClaudeAgentIntegration, PendingApproval


async def until_decided(pending: PendingApproval) -> None:
    """Show the held call, and return once the person has decided in the AuthFI console."""
    print(f"waiting for approval {pending.approval_id}: {pending.tool} {pending.args}")
    await asyncio.to_thread(input, "Decide in the AuthFI console, then press Enter: ")


async def run(question: str) -> int:
    """Ask Claude Code ``question`` as the agent."""
    async with AsyncAuthFI.from_env() as af:
        claude = ClaudeAgentIntegration(af.agent_from_env(), wait_for_decision=until_decided)
        options = await claude.options(model=os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"), max_turns=8)
        async with ClaudeSDKClient(options) as client:
            await client.query(question)
            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            print(block.text)
        if claude.token_refused is not None:
            print(f"the gateway refused this agent's token: {claude.token_refused}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(" ".join(sys.argv[1:]) or "Summarise the wiki of pallets/flask")))
