"""One Claude call through your project's AI gateway, traced to LangSmith without the token.

    pip install 'authfi[langchain]'
    export AUTHFI_PLATFORM_URL=https://<project>.authfi.io
    export AUTHFI_AGENT_ID=agt_... AUTHFI_AGENT_KEY_FILE=/run/secrets/agent.key AUTHFI_AGENT_KID=k1
    export LANGSMITH_TRACING=true LANGSMITH_API_KEY=...       # optional
    python chat.py

The agent needs a grant for the model (``mcp:anthropic/<model>``) in the project.
"""

import os

from authfi import AgentAuth
from authfi.langchain import AuthFIChatAnthropic, metadata_only_tracing

MODEL = os.environ.get("AUTHFI_MODEL", "claude-haiku-5-5")


def main():
    if os.environ.get("LANGSMITH_TRACING") == "true":
        # Runs are recorded with their names, timings, ids and the agent id; the prompt
        # and the answer are not. Pass capture_content=True to record them.
        metadata_only_tracing()
    agent = AgentAuth.from_env()
    model = AuthFIChatAnthropic(agent=agent, model=MODEL, max_tokens=200)
    print(model.invoke("In one sentence, what does an identity provider do?").content)


if __name__ == "__main__":
    main()
