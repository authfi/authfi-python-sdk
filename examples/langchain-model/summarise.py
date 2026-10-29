"""A LangChain model on the AuthFI AI gateway, traced without content."""

from __future__ import annotations

import os
import sys

from authfi import AuthFI, GatewayTokenRefused
from authfi.langchain import LangChainIntegration


def main() -> int:
    """Ask the model one question as the agent, and report a refused token plainly."""
    with AuthFI.from_env() as af:
        langchain = LangChainIntegration(af.agent_from_env())
        if os.environ.get("LANGSMITH_TRACING") == "true":
            langchain.configure_tracing()
        model = langchain.chat_model(os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"), max_tokens=256)
        try:
            answer = model.invoke("In one sentence: why should an AI agent hold its own credential?")
        except GatewayTokenRefused as refused:
            print(f"the gateway refused this agent's token: {refused}", file=sys.stderr)
            return 1
        print(answer.content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
