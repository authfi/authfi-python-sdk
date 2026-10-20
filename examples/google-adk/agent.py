"""A Google ADK agent on AuthFI: Claude through the AI gateway, tools through the MCP gateway.

    pip install 'authfi[adk]'
    export AUTHFI_PLATFORM_URL=https://<project>.authfi.io
    export AUTHFI_AGENT_ID=agt_... AUTHFI_AGENT_KEY_FILE=/run/secrets/agent.key AUTHFI_AGENT_KID=k1
    python agent.py "Use the wiki tools to describe the structure of pallets/flask"

There is no glue here: the model credential, MCP auth, refusals, approvals and
telemetry defaults come from ``authfi.adk``. When a tool call is held for a
person's approval, ADK pauses with a tool confirmation; this script waits until
the person decides in the AuthFI console and only then answers it, because ADK
asks for a confirmation once per call.

Acting for a person: set ``AUTHFI_ACT_FOR`` to their sign-in name. They
approve on their own device, and the run carries a token for them.
"""

import asyncio
import os
import sys

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from authfi import AgentAuth, AsyncAgentAuth, AuthFIError
from authfi.adk import AuthFIADK, configure_telemetry

MODEL = os.environ.get("AUTHFI_MODEL", "claude-haiku-5-5")
INSTRUCTION = ("You are a research helper. Use the tools you are given to answer questions about public GitHub "
               "repositories. When a tool refuses, say what it said. Keep answers short.")


async def run(credential, question, wait_for_decision, show=print):
    """Ask ``question``; for each held call await ``wait_for_decision(held)`` -> bool, then answer ADK."""
    authfi = AuthFIADK(credential)
    agent = authfi.agent(name="research", instruction=INSTRUCTION, model=MODEL)
    sessions = InMemorySessionService()
    runner = Runner(agent=agent, app_name="authfi-example", session_service=sessions)
    session = await sessions.create_session(app_name="authfi-example", user_id="example")
    message = types.Content(role="user", parts=[types.Part(text=question)])
    try:
        while message is not None:
            events = []
            async for event in runner.run_async(user_id=session.user_id, session_id=session.id, new_message=message):
                events.append(event)
                for part in (event.content.parts if event.content and event.content.parts else ()):
                    if part.text:
                        show("%s: %s" % (event.author, part.text))
            held = authfi.held_approvals(events)
            message = None
            if held:
                approved = await wait_for_decision(held[0])
                message = authfi.answer(held[0], confirmed=approved)
    finally:
        await runner.close()


def console_decision(agent_auth):
    """Wait for a person's decision: read it when the project lets the agent, else ask here."""
    async def wait(held):
        print("Waiting for a person to approve %s %s (approval %s) in the AuthFI console."
              % (held["tool"], held["args"], held["approval_id"]))
        try:
            record = await agent_auth.wait_for_approval(held["approval_id"], timeout=600)
            return record.get("status") == "approved"
        except AuthFIError as e:
            if e.status != 403:
                raise
            answer = await asyncio.get_running_loop().run_in_executor(
                None, input, "This agent may not read its approvals here. Once they have decided, "
                             "type y if they approved: ")
            return answer.strip().lower().startswith("y")
    return wait


async def main(question):
    agent_auth = AsyncAgentAuth(AgentAuth.from_env())
    configure_telemetry(agent_auth)       # before anything builds ADK's telemetry
    credential = agent_auth
    person = os.environ.get("AUTHFI_ACT_FOR")
    if person:
        print("Asking %s to let this agent act for them..." % person)
        credential = await agent_auth.token_as_user(person, binding_message="Let the research agent act for you")
    await run(credential, question, console_decision(agent_auth))


if __name__ == "__main__":
    asyncio.run(main(" ".join(sys.argv[1:]) or "Describe the wiki structure of pallets/flask."))
