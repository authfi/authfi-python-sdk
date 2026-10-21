"""A LangGraph agent on the AuthFI gateway that pauses for a person's approval and resumes."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from authfi import AsyncAuthFI, GatewayTokenRefused
from authfi.langgraph import LangGraphIntegration, ask


async def run(question: str) -> int:
    """Run the agent on ``question`` until it answers, waiting for each approval it needs."""
    async with AsyncAuthFI.from_env() as af:
        langgraph = LangGraphIntegration(af.agent_from_env())
        tools = await langgraph.tools()
        model = langgraph.chat_model(os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"), max_tokens=1024)
        graph = langgraph.agent_graph(model, tools, checkpointer=InMemorySaver())
        config: Any = {"configurable": {"thread_id": "example"}}
        try:
            state = await graph.ainvoke(ask(question), config)
            while "__interrupt__" in state:
                for approval in state["__interrupt__"][0].value["authfi_approvals"]:
                    print(f"waiting for approval {approval['approval_id']}: {approval['tool']} {approval['args']}")
                answer = await asyncio.to_thread(
                    input, "Decide in the AuthFI console, then press Enter (or type stop): "
                )
                resume: Any = {"stop": True} if answer.strip() == "stop" else True
                state = await graph.ainvoke(Command(resume=resume), config)
        except GatewayTokenRefused as refused:
            print(f"the gateway refused this agent's token: {refused}", file=sys.stderr)
            return 1
        print(state["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(" ".join(sys.argv[1:]) or "Summarise the wiki of pallets/flask")))
