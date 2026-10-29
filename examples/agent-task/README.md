# A task: authority for one piece of work

A person gives an agent a task: the tools it may use, a budget, a deadline. The agent works inside
it, and the gateway holds every call to it.

## What it shows

- **One credential.** `agent.request_task(...)` asks for the task with the person's token. The task is
  a credential, so `LangGraphIntegration(task)` is the only line that changes; every tool and model
  call then carries the task's limits.
- **A person may have to approve.** `TaskPending.approval_url` is the link for them;
  `agent.await_task(...)` waits (`ConsentDenied`, `ConsentExpired`, or `WaitTimeout` at your own
  deadline).
- **A refusal is an answer.** A tool outside the task, or a spend over its budget, reaches the model
  as a tool result that says what it can still do. It never offers to start another task: a task is
  one level deep, and only a person gives one.
- **The credential is held by a key.** The task is bound to the DPoP key of this process and goes only
  to AuthFI. Leaving the `async with` forgets it.

## Run it

```bash
pip install "authfi[langgraph] @ git+https://github.com/authfi/authfi-python-sdk@main"
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
python refund_in_a_task.py 4411 "$PERSONS_ACCESS_TOKEN"
```
