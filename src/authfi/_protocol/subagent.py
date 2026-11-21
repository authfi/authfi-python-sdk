"""Sub-agents: the requests that make one and the answers that describe it.

A sub-agent is an agent of its own that a running agent makes by presenting its own access token. It holds no key, so
only the agent that made it can get a token for it, and it holds only what that agent holds: a request for more is
refused whole (``ESUBAGENT002``). The answer names where it sits in its tree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from authfi._protocol.errors import api_error
from authfi._protocol.http import HttpRequest, HttpResponse, Retry
from authfi.errors import APIError, ConfigurationError
from authfi.secret import Secret

__all__ = [
    "MAX_SUB_AGENT_PERMISSIONS",
    "SUB_AGENT_PATH",
    "SubAgentGrant",
    "parse_sub_agent",
    "sub_agent_request",
    "sub_agent_token_request",
]

#: Below the plane and version prefix, next to the transaction-token route.
SUB_AGENT_PATH = "/apis/regional/v1/ai/agents/subagents"
#: The most permissions one sub-agent holds (the platform refuses more).
MAX_SUB_AGENT_PERMISSIONS = 64


@dataclass(frozen=True, slots=True)
class SubAgentGrant:
    """A sub-agent as the platform recorded it, and its first token."""

    agent_id: str
    wimse_id: str
    name: str
    parent_agent_id: str
    root_agent_id: str
    depth: int
    #: The agents above it, its parent first and the registered root last.
    chain: tuple[str, ...]
    capabilities: tuple[str, ...]
    access_token: Secret
    token_type: str
    expires_in: int


def _permissions(values: Sequence[str]) -> list[str]:
    if isinstance(values, str):
        raise ConfigurationError("permissions is a list of the permissions the sub-agent needs, not a string")
    out: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip() or any(c.isspace() for c in value.strip()):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigurationError(
                "a permission is a name with no spaces, such as mcp:crm.read", code="sub_agent_request_invalid"
            )
        if value.strip() not in out:
            out.append(value.strip())
    if not out:
        raise ConfigurationError(
            "a sub-agent needs at least one permission, and only ones its parent holds",
            code="sub_agent_request_invalid",
        )
    if len(out) > MAX_SUB_AGENT_PERMISSIONS:
        raise ConfigurationError(f"a sub-agent holds at most {MAX_SUB_AGENT_PERMISSIONS} permissions")
    return out


def sub_agent_request(
    url: str,
    credential: Mapping[str, str],
    *,
    name: str,
    permissions: Sequence[str],
    audience: str | None,
    run_id: str | None,
    framework: str | None,
) -> HttpRequest:
    """Make a sub-agent with the parent's own access token and its DPoP proof. Never retried after send."""
    if not name or not name.strip():
        raise ConfigurationError("name is required: it is what people read for this sub-agent")
    body: dict[str, Any] = {"name": name.strip(), "permissions": _permissions(permissions)}
    if audience:
        body["audience"] = audience
    if run_id:
        body["run_id"] = run_id
    if framework:
        body["framework"] = framework
    return _json("POST", url, "the sub-agent endpoint", credential, body)


def sub_agent_token_request(url: str, credential: Mapping[str, str], *, audience: str | None) -> HttpRequest:
    """A fresh token for a sub-agent the caller made."""
    return _json("POST", url, "the sub-agent token endpoint", credential, {"audience": audience} if audience else {})


def _json(method: str, url: str, what: str, headers: Mapping[str, str], body: Mapping[str, Any]) -> HttpRequest:
    return HttpRequest(
        "POST",
        url,
        what=what,
        headers={"Accept": "application/json", **headers},
        json_body=dict(body),
        retry=Retry.NEVER_AFTER_SEND,
    )


def _text(data: Mapping[str, Any], key: str, what: str, status: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise APIError("invalid_response", f"{what} answered no {key}", status)
    return value


def parse_sub_agent(response: HttpResponse, what: str = "the sub-agent endpoint") -> SubAgentGrant:
    """The sub-agent and its token, or the platform's refusal (``ESUBAGENT002``: more than its parent holds)."""
    if response.status not in (200, 201):
        raise api_error(response, what)
    body = response.json_object() or {}
    data = body.get("data")
    if not isinstance(data, dict):
        raise APIError("invalid_response", f"{what} answered no data", response.status)
    found = cast(dict[str, Any], data)
    caps = found.get("capabilities")
    chain = found.get("chain")
    depth = found.get("depth")
    expires = found.get("expires_in")
    return SubAgentGrant(
        agent_id=_text(found, "agent_id", what, response.status),
        wimse_id=str(found.get("wimse_id") or ""),
        name=str(found.get("name") or ""),
        parent_agent_id=str(found.get("parent_agent_id") or ""),
        root_agent_id=str(found.get("root_agent_id") or ""),
        depth=depth if isinstance(depth, int) and not isinstance(depth, bool) else 0,
        chain=tuple(c for c in cast(list[object], chain) if isinstance(c, str)) if isinstance(chain, list) else (),
        capabilities=tuple(c for c in cast(list[object], caps) if isinstance(c, str)) if isinstance(caps, list) else (),
        access_token=Secret(_text(found, "access_token", what, response.status)),
        token_type=str(found.get("token_type") or "Bearer"),
        expires_in=expires if isinstance(expires, int) and not isinstance(expires, bool) and expires > 0 else 300,
    )
