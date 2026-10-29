"""Reading the platform's error envelopes into typed errors. Pure functions.

- OAuth endpoints answer RFC 6749 §5.2: ``{"error", "error_description"}``.
- AuthFI APIs answer ``{"error": "<stable code>", "error_description": "..."}``.

A body with no ``error`` is not given an invented code: it becomes a :class:`TransportError`
naming the status.
"""

from __future__ import annotations

from collections.abc import Sequence

from authfi._protocol.http import HttpResponse
from authfi.errors import (
    APIError,
    AuthFIError,
    ConsentDenied,
    ConsentExpired,
    OAuthError,
    StepUpRequired,
    TemporarilyUnavailable,
    TransportError,
)

__all__ = ["CIBA_GRANT", "api_error", "describe", "oauth_error", "with_answer"]

CIBA_GRANT = "urn:openid:params:grant-type:ciba"

_MAX_DESCRIPTION = 300


def _envelope(response: HttpResponse) -> tuple[str | None, str | None]:
    body = response.json_object()
    if body is None:
        return None, None
    code = body.get("error")
    if isinstance(code, dict):  # a provider envelope relayed by the AI gateway: {"error": {"code", "message"}}
        inner: dict[str, object] = code  # pyright: ignore[reportUnknownVariableType]
        nested_code = inner.get("code") or inner.get("type")
        message = inner.get("message")
        return (
            nested_code if isinstance(nested_code, str) else None,
            message[:_MAX_DESCRIPTION] if isinstance(message, str) else None,
        )
    description = body.get("error_description")
    return (
        code if isinstance(code, str) and code else None,
        description[:_MAX_DESCRIPTION] if isinstance(description, str) and description else None,
    )


def with_answer(error: AuthFIError, response: HttpResponse) -> AuthFIError:
    """``error`` with the answer's request and run ids attached."""
    error.request_id = response.answer_request_id
    error.run_id = response.header("AuthFI-Run-Id")
    error.dropped_header = response.dropped_header
    if error.status is None and not 200 <= response.status < 300:
        error.status = response.status
    return error


def describe(response: HttpResponse, what: str) -> TransportError:
    """A response with no error envelope, as a :class:`TransportError` naming what failed."""
    retryable = response.status in (429, 502, 503, 504)
    error = TransportError(
        f"{what} answered HTTP {response.status} with no error code",
        code="unexpected_response",
        retryable=retryable,
        status=response.status,
        request_id=response.answer_request_id,
    )
    return error


def oauth_error(
    response: HttpResponse,
    what: str,
    *,
    grant: str | None = None,
    step_up_acr: Sequence[str] = (),
) -> AuthFIError:
    """The typed error for an OAuth endpoint's refusal.

    ``grant`` decides what ``access_denied`` and ``expired_token`` mean: on the CIBA grant they
    are a person's decision (:class:`ConsentDenied`, :class:`ConsentExpired`). ``step_up_acr`` is
    what a step-up should ask for; it is never parsed out of the description.
    """
    code, description = _envelope(response)
    status = response.status
    error: AuthFIError
    if code is None:
        return describe(response, what)
    if code == "insufficient_user_authentication":
        error = StepUpRequired(description, acr_values=step_up_acr, status=status)
    elif code == "temporarily_unavailable" or status == 503:
        error = TemporarilyUnavailable(code, description, status=status)
    elif grant == CIBA_GRANT and code == "access_denied":
        error = ConsentDenied(code, description, status=status)
    elif grant == CIBA_GRANT and code == "expired_token":
        error = ConsentExpired(code, description, status=status)
    else:
        error = OAuthError(code, description, status=status)
    return with_answer(error, response)


def api_error(response: HttpResponse, what: str) -> AuthFIError:
    """The typed error for an AuthFI API's refusal."""
    code, description = _envelope(response)
    if code is None:
        return describe(response, what)
    return with_answer(APIError(code, description, status=response.status), response)
