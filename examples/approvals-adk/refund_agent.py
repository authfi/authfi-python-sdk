"""A Google ADK agent on the AuthFI gateways that waits for a person's approval."""

from __future__ import annotations

import asyncio
import os
import sys

from google.adk.events import Event
from google.adk.runners import InMemoryRunner
from google.genai import types

from authfi import AsyncAuthFI, GatewayTokenRefused
from authfi.adk import ADKIntegration


def _say(events: list[Event]) -> None:
    for event in events:
        for part in event.content.parts if event.content and event.content.parts else ():
            if part.text:
                print(part.text)


async def run(question: str) -> int:
    """Run the agent on ``question``, answering each approval once the person has decided."""
    async with AsyncAuthFI.from_env() as af:
        adk = ADKIntegration(af.agent_from_env())
        adk.configure_tracing()
        agent = adk.agent(
            name="research",
            instruction="Answer briefly, using the tools.",
            model=os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"),
        )
        runner = InMemoryRunner(agent=agent, app_name="authfi-example")
        session = await runner.session_service.create_session(app_name="authfi-example", user_id="example")

        async def send(message: types.Content) -> list[Event]:
            return [e async for e in runner.run_async(user_id="example", session_id=session.id, new_message=message)]

        try:
            events = await send(types.Content(role="user", parts=[types.Part(text=question)]))
            while held := adk.held_approvals(events):
                _say(events)
                pending = held[0]
                print(f"waiting for approval {pending.approval_id}: {pending.tool} {pending.args}")
                answer = await asyncio.to_thread(input, "Decide in the AuthFI console, then type yes or no: ")
                events = await send(adk.answer(pending, confirmed=answer.strip().lower() == "yes"))
        except GatewayTokenRefused as refused:
            print(f"the gateway refused this agent's token: {refused}", file=sys.stderr)
            return 1
        _say(events)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run(" ".join(sys.argv[1:]) or "Summarise the wiki of pallets/flask")))
