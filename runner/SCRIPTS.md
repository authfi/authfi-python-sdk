# Where runner/scripts came from

Copied unchanged (line endings normalised to LF) on 2026-11-05 from the framework-test folder
`plans/agent-framework-tests/2026-10-31/` (an internal planning tree that is not a git repository, so there is no commit):
`common/fwcommon.py`, `common/harness.py`, `langgraph/agent.py`, `openai-agents/agent.py`, `google-adk/agent.py`.
Only what these three scripts import is here. `scripts.sha256` holds their hashes; `test_entrypoint.py` fails when a
copy changes without it, so a change is deliberate.
