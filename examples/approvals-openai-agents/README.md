# Approvals in an OpenAI Agents SDK agent

An agent built with the OpenAI Agents SDK calls Claude and the gateway's governed tools. A call
that needs a person's approval pauses the run as the SDK's own approval item; it resumes once the
person has decided in AuthFI.

## What it shows

- **Traces without the token or content.** `configure_tracing()` runs first: the default exporter
  is gone and spans carry no prompts, answers or tool results.
- **No OpenAI key.** The model goes through the AI gateway with the agent's own token.
- **Approval by AuthFI's decision.** `resolve()` approves a paused call in the SDK only once the
  gateway has decided it.

## Run it

```bash
pip install "authfi[openai-agents] @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python research_agent.py "Read the wiki contents of pallets/flask"
```
