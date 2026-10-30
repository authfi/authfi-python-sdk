"""A LangGraph agent works on one order inside a task: only the tools and budget a person gave it.

Run it::

    pip install "authfi[langgraph] @ git+https://github.com/authfi/authfi-python-sdk@main"
    cp .env.example .env    # then fill it in
    python refund_in_a_task.py 4411 "$PERSONS_ACCESS_TOKEN"
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from authfi import AsyncAuthFI, ConsentDenied, ConsentExpired, Secret, TaskPending
from authfi.langgraph import LangGraphIntegration, ask


async def run(order: str, persons_token: str) -> int:
    """Work on ``order`` inside a task, waiting for the person to approve it if the template needs that."""
    async with AsyncAuthFI.from_env() as af:
        agent = af.agent_from_env()
        started = await agent.request_task(
            "support.refund", subject_token=Secret(persons_token), params={"order": order}
        )
        if isinstance(started, TaskPending):
            print(f"Waiting for the person to approve: {started.approval_url}")
            try:
                task = await agent.await_task(started, timeout=600)
            except (ConsentDenied, ConsentExpired) as refused:
                print(f"No task: {refused.code}", file=sys.stderr)
                return 1
        else:
            task = started
        async with task:
            langgraph = LangGraphIntegration(task)  # the task is the credential: its limits are the gateway's
            tools = await langgraph.tools()
            model = langgraph.chat_model(os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"), max_tokens=1024)
            graph = langgraph.agent_graph(model, tools, checkpointer=InMemorySaver())
            config: Any = {"configurable": {"thread_id": "example"}}
            state = await graph.ainvoke(ask(f"Look at order {order} and refund it if the policy allows"), config)
            print(state["messages"][-1].content)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(sys.argv[1], sys.argv[2])))
