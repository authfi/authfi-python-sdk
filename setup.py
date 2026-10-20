import io
import re

from setuptools import setup

# ONE spelling of the version. The package needs it at runtime (it goes in the
# User-Agent, which AuthFI's Cloudflare edge requires), and a copy here would be
# a second place to forget.
with io.open("authfi/__init__.py", encoding="utf-8") as fh:
    VERSION = re.search(r'^__version__ = "([^"]+)"', fh.read(), re.M).group(1)

setup(
    name="authfi",
    version=VERSION,
    description="AuthFI Python SDK — JWT validation, RBAC decorators, permission auto-sync, agent authentication",
    packages=["authfi"],
    python_requires=">=3.8",
    install_requires=[
        "PyJWT[crypto]>=2.4.0",
        # Ed25519 signing for agent client assertions (private_key_jwt). This is
        # NOT a new install: PyJWT[crypto] above already pulls cryptography in for
        # RS256 verification. It is declared because authfi imports it directly,
        # and a direct import that relies on somebody else's extra breaks silently
        # the day that extra changes.
        "cryptography>=3.4.0",
    ],
    extras_require={
        # AgentAuth.httpx_auth() and DelegatedToken.httpx_auth(). Optional: the
        # rest of the SDK, the async client included, does not use it.
        "httpx": ["httpx>=0.23"],
        # The framework adapters, one extra each. They need Python 3.10 or later,
        # as the frameworks do; the core SDK keeps Python 3.8.
        # authfi.langchain: a Claude client whose token LangSmith never records.
        "langchain": ["langchain-anthropic>=1.0", "langsmith>=0.4"],
        # authfi.langgraph: governed MCP tools, refusals as tool results,
        # approvals through interrupt().
        "langgraph": ["langchain-anthropic>=1.0", "langsmith>=0.4", "langgraph>=1.0",
                      "langchain-mcp-adapters>=0.2", "httpx>=0.23"],
        # authfi.openai_agents: tracing without the token or content, LiteLLM on
        # the AI gateway, governed MCP tools and approvals.
        "openai-agents": ["openai-agents[litellm]>=0.24", "httpx>=0.23"],
        # authfi.claude_agent: approvals through can_use_tool, the `authfi token`
        # and `authfi headers` helpers, content logging off.
        "claude-agent": ["claude-agent-sdk>=0.2"],
        # authfi.adk: the model credential, the MCP toolset, approvals through
        # tool confirmation, telemetry without content.
        "adk": ["google-adk>=2.0", "litellm>=1.70", "httpx>=0.23"],
    },
    entry_points={"console_scripts": ["authfi = authfi.cli:main"]},
    author="AuthFI",
    url="https://github.com/authfi/authfi-python-sdk",
    license="Apache-2.0",
    classifiers=[
        "Framework :: Flask",
        "Framework :: FastAPI",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
    ],
)
