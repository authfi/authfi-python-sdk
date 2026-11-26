"""The Saturday demo agent: a long-running LangGraph agent that signs in keylessly as a bound pod.

Each run lists orders through the AuthFI MCP gateway, then has the model write a streamed summary through the AI
gateway (``<issuer>/llm``). It backs off while the agent is Stopped and resumes by itself after Release.

Settings (all environment variables; none is a credential):
  AUTHFI_ISSUER, AUTHFI_AGENT_ID, AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE, AUTHFI_CONFIG_VERSION, AUTHFI_CONFIG_DIGEST
      the bound-instance sign-in (checked by entrypoint.py); AUTHFI_CONFIG_OBSERVE=1 replaces the last two for the
      first, observed run.
  DEMO_INTERVAL_SECONDS (60), DEMO_STOPPED_BACKOFF_SECONDS (30), DEMO_MODEL (claude-haiku-5-5),
  DEMO_MAX_TOKENS (1100), DEMO_DIRECT_PROVIDER_PROBE (1 = one credential-less connect to api.anthropic.com per run),
  DEMO_TASK_TEMPLATE (see demo-sat/agent.md: a task needs a person's token, so this only logs why none is started).
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_loop as loop  # noqa: E402

PROMPT = (
    "Here are the current orders from our order system:\n\n{orders}\n\n"
    "Write a thorough briefing for the operations lead, at least 400 words: one paragraph per order, then a section "
    "on risks and follow-ups, then a closing recommendation. Do not skip any order."
)


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", ".", name.lower()).strip(".")


def find_tool(tools: list, wanted: str):  # type: ignore[type-arg]
    """The gateway tool whose name ends in ``wanted`` (e.g. ``orders.list``), however the gateway spells the separator."""
    for tool in tools:
        if _norm(tool.name).endswith(_norm(wanted)):
            return tool
    raise LookupError(f"the gateway lists no tool named like {wanted!r}")


def _text(result: object) -> str:
    if isinstance(result, (list, tuple)):
        return "\n".join(_text(r) for r in result)
    if isinstance(result, dict):
        return str(result.get("text", result))
    return str(result)


class SdkSession:
    """One signed-in agent on the AuthFI SDK."""

    def __init__(self) -> None:
        import authfi

        self.credential = authfi.connect_async()
        self.langgraph = self.credential.langgraph()
        self.model_name = os.environ.get("DEMO_MODEL", "claude-haiku-5-5")
        self.max_tokens = int(os.environ.get("DEMO_MAX_TOKENS", "1100"))
        self.tools: list = []  # type: ignore[type-arg]

    async def sign_in(self) -> None:
        """Mint the first token (proves the sign-in) and discover the gateway's tools."""
        await self.credential.token()
        self.tools = await self.langgraph.tools()
        loop.log("discovery", tools=len(self.tools), model_door=self.langgraph.gateway.model_base_url("anthropic").rsplit("/", 1)[-1])

    async def list_orders(self) -> str:
        return _text(await find_tool(self.tools, "orders.list").ainvoke({}))

    async def read_order(self, order_text: str) -> str | None:
        match = re.search(r"\b(?:ord|order)[-_ ]?[A-Za-z0-9-]+\b|\b\d{3,}\b", order_text)
        if match is None:
            return None
        tool = find_tool(self.tools, "orders.read")
        argument = next(iter(tool.args), "id")  # the tool's own first parameter, whatever it is called
        return _text(await tool.ainvoke({argument: match.group(0)}))

    async def summarize(self, orders_text: str, on_chunk: Callable[[], None]) -> dict[str, int]:
        model = self.langgraph.chat_model(self.model_name, max_tokens=self.max_tokens)
        total = None
        async for chunk in model.astream(PROMPT.format(orders=orders_text[:4000])):
            on_chunk()
            total = chunk if total is None else total + chunk
        stop = (getattr(total, "response_metadata", None) or {}).get("stop_reason")
        if total is None or not stop:
            raise loop.StreamCut("the model stream ended without a stop reason")
        usage = getattr(total, "usage_metadata", None) or {}
        return {"input_tokens": int(usage.get("input_tokens", 0)), "output_tokens": int(usage.get("output_tokens", 0))}

    async def aclose(self) -> None:
        await self.credential.project.aclose()


async def open_session() -> SdkSession:
    """Sign in afresh: a new credential, a new token, new tool discovery."""
    session = SdkSession()
    try:
        await session.sign_in()
    except BaseException:
        await session.aclose()
        raise
    return session


async def main() -> int:
    cfg = loop.Config.from_env(os.environ)
    loop.main_log_start(os.environ, cfg)
    if cfg.task_template:
        loop.log(
            "task",
            started=False,
            reason="a task needs a person's token (request_task) and the SDK has no group-approved start; running without one",
        )
    await loop.DemoLoop(open_session, cfg).forever()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
