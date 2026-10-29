# Models through the AI gateway

An agent calls Claude (and, when the project has an OpenAI-compatible provider, an OpenAI model)
through the AuthFI AI gateway.

## What it shows

- **The token stays out of traces.** The provider SDK gets the gateway's HTTP client, which puts
  the agent's current token on each request. The token is never the SDK's `api_key`, a default
  header, or any other setting a tracer records.
- **One base URL per API.** `gateway.model_base_url("anthropic")` and `("openai")`.
- **The gateway governs the call.** The model must be granted to the agent; an ungranted one is
  refused with a 403 the provider SDK reports.

## Run it

```bash
pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main" anthropic openai
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python models.py
```
