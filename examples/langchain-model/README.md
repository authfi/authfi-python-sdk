# A LangChain model through the AI gateway

A LangChain agent's Claude model, called through the AuthFI AI gateway with the agent's own
credential.

## What it shows

- **The token stays out of LangSmith.** The model's HTTP client puts the agent's token on each
  request; it is not a model argument, so a trace never records it.
- **Traces without content.** `configure_tracing()` hides prompts and answers; the run, its
  timings and the agent's id are still recorded.
- **A refused token ends the run.** When the agent is revoked, the next call raises
  `GatewayTokenRefused` instead of retrying.

## Run it

```bash
pip install "authfi[langchain-anthropic] @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python summarise.py
```
