# CrewAI

```bash
pip install "authfi[crewai]"          # in its own environment
```

```python
import authfi
from crewai import Agent, Crew, Task

af = authfi.connect_async()
crewai = af.crewai()

researcher = Agent(role="Researcher", goal="Describe a wiki", backstory="You read documentation.",
                   llm=crewai.llm("claude-haiku-5-5"), tools=await crewai.tools())
task = Task(description="Describe pallets/flask", expected_output="A summary", agent=researcher)
async with crewai.guard():                                     # a token the gateway refuses ends the run
    result = await Crew(agents=[researcher], tasks=[task]).akickoff()
```

- **Model:** CrewAI's `LLM` on `<issuer>/llm`. **Tools:** the gateway's, as CrewAI tools on `<issuer>/mcp`.
- **Approval:** a held tool waits for the person (`on_approval` is told once, `approval_timeout` bounds the wait) and
  returns the tool's answer when approved, the refusal when denied.
- **Steps:** `crewai.export_steps()`. CrewAI runs on asyncio here: use `akickoff()`.
