"""A gateway decision as a child span of the agent's step, in LangSmith's words.

Each decision the gateway makes about a tool call becomes one finished child span of the span that made the
call: ``langsmith.span.kind`` ``tool``, and the decision id, the outcome (``allow``, ``deny`` or
``needs_approval``), the rule or grant that decided, and the mandate id, where the gateway named them. Nothing
the tool was given or answered is in it. Without OpenTelemetry installed, or outside a span, it does nothing.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from authfi._protocol.gateway import REFUSAL_META, ToolAllowed, ToolApprovalRequired, ToolOutcome

__all__ = ["DECISION_META", "decision_attributes", "record_decision"]

#: ``_meta`` key of a result the gateway decided, with the ids it names.
DECISION_META = "io.authfi/decision"
_NAMED = {
    "decision_id": "authfi.decision.id",
    "rule": "authfi.decision.rule",
    "grant": "authfi.decision.grant",
    "mandate_id": "authfi.mandate.id",
}


def _meta_of(result: object) -> Mapping[str, Any]:
    for holder in (result, getattr(result, "model_extra", None)):
        meta = (
            cast("Mapping[str, Any]", holder).get("_meta")
            if isinstance(holder, Mapping)
            else getattr(holder, "meta", None)
        )
        if isinstance(meta, Mapping):
            return cast(Mapping[str, Any], meta)
    return {}


def _named(result: object) -> dict[str, str]:
    meta = _meta_of(result)
    found: dict[str, str] = {}
    for source in (meta.get(DECISION_META), meta.get(REFUSAL_META)):
        if isinstance(source, Mapping):
            for key, value in cast(Mapping[str, Any], source).items():
                if key in _NAMED and isinstance(value, str) and value:
                    found.setdefault(key, value)
    return found


def decision_attributes(
    outcome: ToolOutcome,
    *,
    tool: str | None = None,
    decision_id: str | None = None,
    rule: str | None = None,
    grant: str | None = None,
    mandate_id: str | None = None,
) -> dict[str, str]:
    """The span attributes of ``outcome``. Arguments name what the gateway's answer did not."""
    named = _named(getattr(outcome, "result", None))
    given = {"decision_id": decision_id, "rule": rule, "grant": grant, "mandate_id": mandate_id}
    found = {**named, **{k: v for k, v in given.items() if v}}
    attributes = {"langsmith.span.kind": "tool"}
    if tool:
        attributes["gen_ai.tool.name"] = tool
    if isinstance(outcome, ToolAllowed):
        attributes["authfi.decision.outcome"] = "allow"
    elif isinstance(outcome, ToolApprovalRequired):
        attributes["authfi.decision.outcome"] = "needs_approval"
        attributes["authfi.decision.approval_id"] = outcome.approval_id
        attributes["authfi.decision.code"] = outcome.code
    else:  # a refusal: the one outcome left
        attributes["authfi.decision.outcome"] = "deny"
        for key, value in (("code", outcome.code or outcome.error), ("reason", outcome.reason)):
            if value:
                attributes[f"authfi.decision.{key}"] = value
        if outcome.required_scope:
            attributes["authfi.decision.required_scope"] = outcome.required_scope
    for key, attribute in _NAMED.items():
        if key in found:
            attributes[attribute] = found[key]
    return attributes


def record_decision(
    outcome: ToolOutcome,
    *,
    tool: str | None = None,
    decision_id: str | None = None,
    rule: str | None = None,
    grant: str | None = None,
    mandate_id: str | None = None,
) -> None:
    """Add the decision as a finished child span of the current span; a no-op when there is none to attach to."""
    try:
        from opentelemetry import trace
    except ImportError:
        return
    if not trace.get_current_span().get_span_context().is_valid:
        return
    attributes = decision_attributes(
        outcome, tool=tool, decision_id=decision_id, rule=rule, grant=grant, mandate_id=mandate_id
    )
    name = f"authfi.decision {tool}" if tool else "authfi.decision"
    with trace.get_tracer("authfi").start_as_current_span(name, attributes=attributes):
        pass
