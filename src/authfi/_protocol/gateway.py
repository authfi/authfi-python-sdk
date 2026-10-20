"""What the AuthFI gateway answers, read one way. No I/O.

- A tool call the gateway allowed comes back as the tool's own result (which may itself be the
  tool's error).
- A refused one comes back as a result with ``isError`` and ``_meta["io.authfi/refusal"]``.
- A held one (a person must approve) comes back as a result with ``isError`` whose text is the
  gateway's hold message. It carries no structured ``_meta`` yet, so one parser, pinned to that
  exact message, reads it; it goes when the gateway ships a structured hold.

The hold's text tells a model to retry with ``_authfi_approval_id``. If a model saw it and retried,
the approval would be spent before the adapter's own resume. So the text is never handed to a
model: :data:`HOLD_MESSAGE` is, and the approval id travels outside the model.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from authfi._protocol.errors import api_error
from authfi._protocol.http import HttpRequest, HttpResponse, Retry
from authfi.errors import APIError, ConfigurationError
from authfi.secret import Secret

__all__ = [
    "APPROVAL_ARGUMENT",
    "CLOUD_PATH",
    "HOLD_MESSAGE",
    "REFUSAL_META",
    "SCOPE_META",
    "CloudCredential",
    "ToolOutcome",
    "cloud_request",
    "parse_cloud",
    "read_result",
    "scope_of",
    "strip_approval_argument",
    "with_approval",
]

#: The argument a held call is sent again with, naming the approval a person gave.
APPROVAL_ARGUMENT = "_authfi_approval_id"
#: ``_meta`` key of a refused call's detail.
REFUSAL_META = "io.authfi/refusal"
#: ``_meta`` key, in ``tools/list``, of the grant object a tool needs.
SCOPE_META = "io.authfi/scope"
#: The tool annotation with the calling agent's advisory outlook for a tool.
GOVERNANCE_ANNOTATION = "io.authfi/governance"
#: What a model is told about a held call: the wait, never an instruction or an id.
HOLD_MESSAGE = "This action is waiting for a person's approval."
#: The gateway's cloud-credential door, under the model routes.
CLOUD_PATH = "/mcp/v1/cloud/credentials"

# The gateway's hold message (authfi-regional-mcp internal/http/mcp/dispatch.go, at 3617565):
# "Waiting for approval %s (%s). A person must approve this action in the AuthFI console. ..."
_HOLD = re.compile(r"^Waiting for approval ([A-Za-z0-9._:-]+) \(([a-z_]+)\)\. A person must approve this action")

OutcomeKind = Literal["allowed", "refused", "approval_required"]


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """What the gateway did with one tool call.

    ``kind`` is ``allowed`` (``result`` is the tool's own answer), ``refused`` (``code``,
    ``error``, ``reason`` and ``required_scope`` say why) or ``approval_required`` (``approval_id``
    names the approval a person must give).
    """

    kind: OutcomeKind
    result: Any = None
    code: str | None = None
    error: str | None = None
    reason: str | None = None
    required_scope: str | None = None
    approval_id: str | None = None

    @property
    def model_text(self) -> str:
        """What a model may read about this outcome when the framework needs a tool answer."""
        if self.kind == "approval_required":
            return HOLD_MESSAGE
        if self.kind == "refused":
            parts = [f"Refused by AuthFI ({self.code or self.error or 'refused'})"]
            if self.reason:
                parts.append(self.reason)
            text = ": ".join(parts) + "."
            if self.required_scope:
                text += f" The call needs the scope {self.required_scope}."
            return text
        return _text_of(self.result)


def _field(value: object, name: str) -> Any:
    if isinstance(value, Mapping):
        return cast(Mapping[str, Any], value).get(name)
    return getattr(value, name, None)


def _mapping(value: object) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], value) if isinstance(value, Mapping) else {}


def _meta(result: object) -> Mapping[str, Any]:
    meta = _field(result, "_meta")
    if meta is None:
        meta = _field(result, "meta")
    if meta is None:  # a framework's tool keeps the MCP tool's _meta in its metadata
        meta = _mapping(_field(result, "metadata")).get("_meta")
    return _mapping(meta)


def _annotation(tool: object, name: str) -> object:
    annotations = cast(object, _field(tool, "annotations"))
    if annotations is not None and not isinstance(annotations, Mapping):
        annotations = getattr(annotations, "model_extra", None)  # an MCP ToolAnnotations keeps ours as extras
    found = _mapping(cast(object, annotations)).get(name)
    if found is None:  # a framework's tool keeps the MCP tool's annotations in its metadata
        found = _mapping(_field(tool, "metadata")).get(name)
    return found


def _text_of(result: object) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    content = _field(result, "content")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for block in cast(list[object], content):
            text = _field(block, "text")
            if isinstance(text, str) and text:
                parts.append(text)
    return "\n".join(parts)


def _is_error(result: object) -> bool:
    flag = _field(result, "isError")
    if flag is None:
        flag = _field(result, "is_error")
    return flag is True


def _string(detail: Mapping[str, Any], name: str) -> str | None:
    value = detail.get(name)
    return value if isinstance(value, str) and value else None


def read_result(result: object, *, is_error: bool | None = None) -> ToolOutcome:
    """Read one ``tools/call`` result: an MCP ``CallToolResult``, its dict form, or its text.

    ``is_error`` overrides the result's own flag (pass it with text). Only an error result can be
    a refusal or a hold, so a tool whose output quotes the gateway's words is still its answer.
    """
    error = _is_error(result) if is_error is None else is_error
    if not error:
        return ToolOutcome("allowed", result=result)
    refusal = _meta(result).get(REFUSAL_META)
    if isinstance(refusal, Mapping):
        detail = cast(Mapping[str, Any], refusal)
        return ToolOutcome(
            "refused",
            code=_string(detail, "code"),
            error=_string(detail, "error"),
            reason=_string(detail, "reason"),
            required_scope=_string(detail, "required_scope"),
            approval_id=_string(detail, "approval_id"),
        )
    hold = _HOLD.match(_text_of(result).strip())
    if hold:
        return ToolOutcome("approval_required", approval_id=hold.group(1), code=hold.group(2))
    return ToolOutcome("allowed", result=result)


def scope_of(tool: object) -> str | None:
    """The grant object a tool needs, from ``tools/list`` ``_meta["io.authfi/scope"]``; ``None`` if absent."""
    value = _meta(tool).get(SCOPE_META)
    return value if isinstance(value, str) and value else None


def governance_of(tool: object) -> str | None:
    """The calling agent's advisory outlook for a tool from ``tools/list``.

    ``no_policy_restriction``, ``may_require_approval`` or ``never_allowed``; ``None`` when the
    gateway has no opinion. Advisory only: the gateway decides each call.
    """
    value = _annotation(tool, GOVERNANCE_ANNOTATION)
    return value if isinstance(value, str) and value else None


def with_approval(arguments: Mapping[str, Any] | None, approval_id: str) -> dict[str, Any]:
    """``arguments`` plus the approval a person gave, to send a held call again."""
    out = dict(arguments or {})
    out[APPROVAL_ARGUMENT] = approval_id
    return out


def strip_approval_argument(arguments: Mapping[str, Any] | None) -> tuple[dict[str, Any], bool]:
    """``arguments`` without an approval id, and whether one was there.

    Only the adapter adds the approval id, on its own resume. One the model wrote is removed, so a
    model that read a hold message elsewhere cannot spend a person's approval.
    """
    out = dict(arguments or {})
    stripped = out.pop(APPROVAL_ARGUMENT, None) is not None
    return out, stripped


# --- cloud credentials ----------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, repr=False)
class CloudCredential:
    """A short-lived cloud credential. The secret parts are :class:`~authfi.Secret`."""

    provider: str
    credential_id: str
    expires_at: datetime | None
    access_key_id: str | None
    secret_access_key: Secret | None
    session_token: Secret | None
    access_token: Secret | None

    def __repr__(self) -> str:
        expires = self.expires_at.isoformat() if self.expires_at else None
        return (
            f"CloudCredential(provider={self.provider!r}, credential_id={self.credential_id!r}, expires_at={expires!r})"
        )


def cloud_request(
    api_origin: str,
    authorization: Mapping[str, str],
    *,
    account: str,
    role: str,
    expires_in: int,
    session_policy: Mapping[str, Any] | None,
    approval_id: str | None,
) -> HttpRequest:
    """``POST <api origin>/mcp/v1/cloud/credentials``. Never retried after send: it may open an approval."""
    for name, value in (("account", account), ("role", role)):
        if not isinstance(cast(object, value), str) or not value.strip():
            raise ConfigurationError(f"cloud_credentials needs the {name}")
    if not 0 < expires_in <= 43_200:
        raise ConfigurationError("expires_in must be a positive number of seconds")
    body: dict[str, Any] = {"account": account, "role": role, "duration_seconds": int(expires_in)}
    if session_policy:
        body["session_policy"] = dict(session_policy)
    if approval_id:
        body["approval_id"] = approval_id
    return HttpRequest(
        "POST",
        api_origin + CLOUD_PATH,
        what="the cloud credential door",
        headers={"Accept": "application/json", **authorization},
        json_body=body,
        retry=Retry.NEVER_AFTER_SEND,
    )


def _secret(data: Mapping[str, Any], name: str) -> Secret | None:
    value = data.get(name)
    return Secret(value) if isinstance(value, str) and value else None


def _expiry(data: Mapping[str, Any], now: float) -> datetime | None:
    stamp = data.get("expiration")
    if isinstance(stamp, str) and stamp:
        try:
            return datetime.fromisoformat(stamp.replace("Z", "+00:00"))
        except ValueError:
            pass
    expires_in = data.get("expires_in")
    if isinstance(expires_in, int) and expires_in > 0:
        return datetime.fromtimestamp(now + expires_in, UTC)
    return None


def parse_cloud(response: HttpResponse, now: float) -> tuple[CloudCredential | None, str | None]:
    """``(credential, None)`` when issued, ``(None, approval_id)`` when a person must approve."""
    if response.status not in (201, 202):
        raise api_error(response, "the cloud credential door")
    body = response.json_object() or {}
    data_obj = body.get("data")
    if not isinstance(data_obj, dict):
        raise APIError("invalid_response", "the cloud credential door answered no data", response.status)
    data = cast(dict[str, Any], data_obj)
    if response.status == 202:
        approval_id = data.get("approval_id")
        if not isinstance(approval_id, str) or not approval_id:
            raise APIError("invalid_response", "the cloud credential door held the call without an approval id", 202)
        return None, approval_id
    access_key = data.get("access_key_id")
    return (
        CloudCredential(
            provider=str(data.get("provider") or ""),
            credential_id=str(data.get("credential_id") or ""),
            expires_at=_expiry(data, now),
            access_key_id=access_key if isinstance(access_key, str) and access_key else None,
            secret_access_key=_secret(data, "secret_access_key"),
            session_token=_secret(data, "session_token"),
            access_token=_secret(data, "access_token"),
        ),
        None,
    )
