# Approvals in a LangGraph agent

A LangGraph agent that calls the gateway's governed tools. When a call needs a person's approval,
the graph pauses; once the person has decided in AuthFI, it resumes and the gateway decides the call.

## What it shows

- **Governed tools with no glue.** `tools()` gives the gateway's tools with the agent's token on
  every call; a tool never allowed for the agent is not offered to the model.
- **A refusal is an answer.** The model reads why a call was refused and carries on.
- **The pause is LangGraph's own.** The approval node calls `interrupt()`; the run waits in the
  checkpointer until you resume it.
- **The approval stays out of the model.** The model reads only that the action waits for a
  person; the approval id travels in the graph's state.
- **A checkpoint holds the held call's arguments.** The checkpointer stores each held call's
  `args` until you resume it; they can contain personal data, so protect the checkpoint store as you
  would that data. It never holds a token or an approval secret.

## Run it

```bash
pip install "authfi[langgraph] @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python agent.py "Read the wiki contents of pallets/flask"
```

The agent prints each approval it waits for. Approve or deny it in the AuthFI console, then press
Enter to resume; type `stop` to give up instead.
