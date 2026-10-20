"""The live authorization check: ``POST <api origin>/apis/regional/v1/check``. No I/O."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from authfi._protocol.errors import api_error
from authfi._protocol.http import HttpRequest, HttpResponse, Retry
from authfi.errors import APIError, ConfigurationError

__all__ = ["CHECK_PATH", "check_request", "parse_check"]

CHECK_PATH = "/apis/regional/v1/check"


def check_request(
    api_origin: str,
    authorization: Mapping[str, str],
    *,
    subject: str,
    relation: str,
    obj: str,
    context: Mapping[str, Any] | None,
) -> HttpRequest:
    """The check request. It decides and changes nothing, so it is safe to repeat."""
    for name, given in (("subject", subject), ("relation", relation), ("object", obj)):
        value = cast(object, given)
        if not isinstance(value, str) or not value.strip():
            raise ConfigurationError(f"check needs a {name}")
    body: dict[str, Any] = {"subject": subject, "relation": relation, "object": obj}
    if context:
        body["context"] = dict(context)
    return HttpRequest(
        "POST",
        api_origin + CHECK_PATH,
        what="the authorization check",
        headers={"Accept": "application/json", **authorization},
        json_body=body,
        retry=Retry.SAFE,
    )


def parse_check(response: HttpResponse) -> tuple[bool, bool, str | None]:
    """``(allowed, approval_required, reason)`` from the answer, or the typed refusal."""
    if response.status != 200:
        raise api_error(response, "the authorization check")
    body = response.json_object() or {}
    data = body.get("data")
    if not isinstance(data, dict):
        raise APIError("invalid_response", "the authorization check answered no data", status=response.status)
    answer = cast(dict[str, Any], data)
    allowed = answer.get("allowed") is True
    held = answer.get("requires_approval") is True
    reason = answer.get("reason")
    return allowed, held, reason if isinstance(reason, str) and reason else None
