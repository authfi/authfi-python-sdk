"""An agent calls Claude and an OpenAI-compatible model through the AuthFI AI gateway.

The agent's token goes on each request through the gateway's HTTP client, never into the SDK's
api_key, default headers or any setting a tracer records.

Run it::

    pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main" anthropic openai
    cp .env.example .env    # then fill it in
    python models.py
"""

from __future__ import annotations

import os

import anthropic
import httpx2
import openai

from authfi import AuthFI

# Stands in where the provider SDKs require a key. The gateway reads the agent's token from the
# Authorization header, which the gateway's HTTP client sets on every request.
NOT_A_KEY = "authfi-sets-the-token-per-request"


def main() -> None:
    """Ask each model one question."""
    with AuthFI.from_env() as af:
        gateway = af.agent_from_env().gateway()
        claude = anthropic.Anthropic(
            base_url=gateway.model_base_url("anthropic"), api_key=NOT_A_KEY, http_client=gateway.http_client(httpx2)
        )
        answer = claude.messages.create(
            model=os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"),
            max_tokens=64,
            messages=[{"role": "user", "content": "Reply with one word: ready?"}],
        )
        print("claude:", answer.content[0].text if answer.content and answer.content[0].type == "text" else answer)
        if os.environ.get("OPENAI_MODEL"):
            gpt = openai.OpenAI(
                base_url=gateway.model_base_url("openai"), api_key=NOT_A_KEY, http_client=gateway.http_client(httpx2)
            )
            reply = gpt.chat.completions.create(
                model=os.environ["OPENAI_MODEL"], messages=[{"role": "user", "content": "Reply with one word: ready?"}]
            )
            print("openai:", reply.choices[0].message.content)


if __name__ == "__main__":
    main()
