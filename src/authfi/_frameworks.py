"""What a connected credential offers: its run, and the framework integrations built on it.

:func:`authfi.connect` returns an agent or a machine client. Both carry this mixin, so
``af.langgraph()`` and the others build the integration for the credential that is already there.
The framework packages are imported when the method is called, so a missing extra fails at that
line with the install command, and never at ``import authfi``.
"""

from __future__ import annotations

import importlib
import itertools
import os
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

from authfi.errors import ConfigurationError

if TYPE_CHECKING:
    from authfi.adk import ADKIntegration
    from authfi.claude_agent import ClaudeAgentIntegration
    from authfi.crewai import CrewAIIntegration
    from authfi.langchain import LangChainIntegration
    from authfi.langgraph import LangGraphIntegration
    from authfi.llama_index import LlamaIndexIntegration
    from authfi.openai_agents import OpenAIAgentsIntegration
    from authfi.pydantic_ai import PydanticAIIntegration

__all__ = ["FrameworkCredential", "new_run_id", "valid_run_id"]

#: The run id prefixes ``X-Request-Id`` (``<run>.<n>``), which the gateway keeps as 1-128 of these characters.
_RUN_ID = re.compile(r"[A-Za-z0-9._:-]{1,96}")

CredentialKind = Literal["agent", "client_credentials"]


def new_run_id() -> str:
    """A fresh run id: ``run-`` and 24 hex digits."""
    return "run-" + os.urandom(12).hex()


def valid_run_id(value: str) -> bool:
    """``True`` when ``value`` can prefix a request id the gateway keeps."""
    return _RUN_ID.fullmatch(value) is not None


def _adapter(module: str, extra: str) -> Any:
    """The adapter module, or a configuration error that names the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise ConfigurationError(
            f"{module} needs its framework installed: pip install 'authfi[{extra}] @ "
            f"git+https://github.com/authfi/authfi-python-sdk@main' ({error})"
        ) from error


class FrameworkCredential:
    """A credential's run, and the framework integrations it builds. Mixed into every connected credential."""

    #: What the credential is, for the gateway's client label: the console tells a machine client from an agent.
    credential_kind: ClassVar[CredentialKind] = "agent"

    _run_id: str | None = None
    _calls: itertools.count[int] | None = None

    @property
    def run_id(self) -> str:
        """The id of this credential's run: one per process unless you start another with :meth:`start_run`.

        Every gateway call carries ``X-Request-Id: <run id>.<n>``, every exported step carries
        ``authfi.run_id``, and every framework integration names it in the run's metadata.
        """
        if self._run_id is None:
            self.start_run()
        assert self._run_id is not None  # noqa: S101 - set by start_run above
        return self._run_id

    def start_run(self, run_id: str | None = None) -> str:
        """Begin a new run (a fresh id unless ``run_id`` is given) and return its id.

        Call it once per task, before the task's first gateway call. Calls already in flight keep the
        id they were sent with.
        """
        chosen = run_id if run_id is not None else new_run_id()
        if not valid_run_id(chosen):
            raise ConfigurationError("a run id is 1-96 of A-Z a-z 0-9 . _ : - (it prefixes X-Request-Id)")
        self._run_id = chosen
        self._calls = itertools.count(1)
        return chosen

    def next_request_id(self) -> str:
        """The id for the next gateway call of this run: ``<run id>.<n>``."""
        run_id = self.run_id
        calls = self._calls
        assert calls is not None  # noqa: S101 - set by start_run
        return f"{run_id}.{next(calls)}"

    # --- frameworks -----------------------------------------------------------------------------------

    def langchain(self, **options: Any) -> LangChainIntegration:
        """The LangChain integration for this credential (extra ``langchain``)."""
        return cast(
            "LangChainIntegration", _adapter("authfi.langchain", "langchain").LangChainIntegration(self, **options)
        )

    def langgraph(self, **options: Any) -> LangGraphIntegration:
        """The LangGraph integration (extra ``langgraph``); needs the asyncio credential (``connect_async``)."""
        return cast(
            "LangGraphIntegration", _adapter("authfi.langgraph", "langgraph").LangGraphIntegration(self, **options)
        )

    def openai_agents(self, **options: Any) -> OpenAIAgentsIntegration:
        """The OpenAI Agents SDK integration (extra ``openai-agents``); needs the asyncio credential."""
        return cast(
            "OpenAIAgentsIntegration",
            _adapter("authfi.openai_agents", "openai-agents").OpenAIAgentsIntegration(self, **options),
        )

    def claude_agent(self, **options: Any) -> ClaudeAgentIntegration:
        """The Claude Agent SDK integration (extra ``claude-agent``); needs an asyncio agent with a key file."""
        return cast(
            "ClaudeAgentIntegration",
            _adapter("authfi.claude_agent", "claude-agent").ClaudeAgentIntegration(self, **options),
        )

    def adk(self, **options: Any) -> ADKIntegration:
        """The Google ADK integration (extra ``adk``); needs the asyncio credential."""
        return cast("ADKIntegration", _adapter("authfi.adk", "adk").ADKIntegration(self, **options))

    def pydantic_ai(self, **options: Any) -> PydanticAIIntegration:
        """The Pydantic AI integration (extra ``pydantic-ai``); needs the asyncio credential."""
        return cast(
            "PydanticAIIntegration",
            _adapter("authfi.pydantic_ai", "pydantic-ai").PydanticAIIntegration(self, **options),
        )

    def crewai(self, **options: Any) -> CrewAIIntegration:
        """The CrewAI integration (extra ``crewai``); needs the asyncio credential."""
        return cast("CrewAIIntegration", _adapter("authfi.crewai", "crewai").CrewAIIntegration(self, **options))

    def llama_index(self, **options: Any) -> LlamaIndexIntegration:
        """The LlamaIndex integration (extra ``llama-index``); needs the asyncio credential."""
        return cast(
            "LlamaIndexIntegration",
            _adapter("authfi.llama_index", "llama-index").LlamaIndexIntegration(self, **options),
        )
