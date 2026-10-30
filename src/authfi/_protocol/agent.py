"""The agent's wire: CIBA, token exchange, the action gate and approvals. No I/O.

- **CIBA** (OpenID CIBA, poll mode): the agent asks a person, who approves on their own device;
  the agent never sees or relays a code. The first poll waits ``interval`` (CIBA §7.3),
  ``slow_down`` adds five seconds, and a fresh assertion signs every poll.
- **Token exchange** (RFC 8693): the person's token in, a token whose ``sub`` is the person and
  ``act.sub`` the agent out. It carries no client authentication: the agent authenticates through
  its own access token as ``actor_token``.
- **The action gate** is AuthFI's own (named after, not following, the IETF Transaction Tokens
  draft): a 60-second token freezes an action, and the gate decides it.
- **Approvals**: a person approves one action; no token results.

A request that may create state (a CIBA request, an exchange, an approval, a frozen action) is
never sent again after it may have arrived.
"""

from __future__ import annotations

import dataclasses
import json
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast
from urllib.parse import quote

from authfi._protocol.errors import CIBA_GRANT, api_error, oauth_error
from authfi._protocol.http import Attempt, Flow, HttpRequest, HttpResponse, Retry, call, pause
from authfi._protocol.oauth import (
    ACCESS_TOKEN_TYPE,
    ClientAuth,
    TokenResponse,
    parse_token_response,
    token_request,
)
from authfi._protocol.urls import require_endpoint
from authfi._protocol.verify import unverified_claims
from authfi.errors import (
    APIError,
    ConfigurationError,
    ConsentExpired,
    DiscoveryError,
    InvalidToken,
    NotSupportedError,
    OAuthError,
    WaitTimeout,
)
from authfi.secret import Secret

__all__ = [
    "CIBA_GRANT",
    "TASK_DETAIL_TYPE",
    "TOKEN_EXCHANGE_GRANT",
    "TRANSACTION_TOKEN_PATH",
    "AgentEndpoints",
    "ApprovalRecord",
    "CibaStart",
    "TaskDetails",
    "TaskPending",
    "approval_request",
    "authorize_request",
    "ciba_poll_flow",
    "ciba_start_request",
    "exchange_request",
    "label_claims",
    "parse_action_decision",
    "parse_agent_configuration",
    "parse_approval",
    "parse_ciba_start",
    "parse_task_start",
    "parse_transaction_token",
    "read_approval_request",
    "task_details",
    "task_request",
    "transaction_token_request",
]

TOKEN_EXCHANGE_GRANT = "urn:ietf:params:oauth:grant-type:token-exchange"
#: RFC 9396 ``type`` of the one ``authorization_details`` entry that asks for a task.
TASK_DETAIL_TYPE = "authfi_mandate"
_JWT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:jwt"
_MAX_TASK_SECONDS = 30 * 24 * 3600
TRANSACTION_TOKEN_PATH = "/apis/regional/v1/ai/auth/transaction-token"
_AUTHORIZE_PATH = "/apis/regional/v1/ai/auth/authorize"
_SLOW_DOWN_STEP = 5.0
_DEFAULT_INTERVAL = 5.0


@dataclass(frozen=True, slots=True)
class AgentEndpoints:
    """The two agent endpoints with no standard home, from ``/.well-known/agent-configuration``."""

    request_approval: str | None
    approval: str | None
    authorize: str | None


def parse_agent_configuration(issuer: str, doc: object) -> AgentEndpoints:
    """The approval endpoints of the agent configuration document. Its token endpoint is never used."""
    if not isinstance(doc, dict):
        raise DiscoveryError(f"The agent configuration of {issuer} is not a JSON object.", code="metadata_invalid")
    body = cast(dict[str, Any], doc)
    named = body.get("issuer")
    if named is not None and named != issuer:
        raise DiscoveryError(f"The agent configuration at {issuer} names issuer {named!r}.", code="issuer_mismatch")
    endpoints_obj = body.get("endpoints")
    endpoints = cast(dict[str, Any], endpoints_obj) if isinstance(endpoints_obj, dict) else {}

    def endpoint(name: str) -> str | None:
        value = endpoints.get(name)
        return require_endpoint(issuer, f"endpoints.{name}", value) if value is not None else None

    return AgentEndpoints(endpoint("request_approval"), endpoint("approval"), endpoint("authorize"))


def _given(value: object) -> bool:
    """``True`` for a string with something in it (callers without a type checker pass anything)."""
    return isinstance(value, str) and bool(value.strip())


def _space(values: Sequence[str] | None) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise ConfigurationError("scope is a sequence of names, not one string")
    joined = " ".join(v.strip() for v in values if v.strip())
    return joined or None


# --- CIBA ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, repr=False)
class CibaStart:
    """A CIBA request's answer. ``auth_req_id`` is a bearer handle, held as a :class:`Secret`."""

    auth_req_id: Secret
    expires_in: int
    interval: float
    approval_url: str | None

    def __repr__(self) -> str:
        return f"CibaStart(expires_in={self.expires_in}, interval={self.interval})"


def ciba_start_request(
    endpoint: str,
    auth: ClientAuth,
    audience: str,
    *,
    login_hint: str,
    binding_message: str,
    scope: Sequence[str] | None,
    expires_in: int | None,
) -> HttpRequest:
    """Ask a person to approve the agent acting for them (CIBA §7.1). Never retried after send."""
    if not _given(login_hint):
        raise ConfigurationError("login_hint is required: the platform must know which person to ask")
    if not _given(binding_message):
        raise ConfigurationError(
            "binding_message is required: it is what the person reads on their own device "
            '("Refund order 4411"), and the line that stops them approving something else'
        )
    form = {"login_hint": login_hint.strip(), "binding_message": binding_message.strip()}
    scope_text = _space(scope)
    if scope_text:
        form["scope"] = scope_text
    if expires_in is not None:
        if expires_in <= 0:
            raise ConfigurationError("expires_in must be a positive number of seconds")
        form["requested_expiry"] = str(int(expires_in))
    return HttpRequest(
        "POST",
        endpoint,
        what="the CIBA endpoint",
        headers={"Accept": "application/json"},
        form=form,
        retry=Retry.NEVER_AFTER_SEND,
        per_attempt=lambda: auth.attempt(audience),
    )


def parse_ciba_start(response: HttpResponse) -> CibaStart:
    """The CIBA answer, or the refusal."""
    if response.status != 200:
        raise oauth_error(response, "the CIBA endpoint", grant=CIBA_GRANT)
    body = response.json_object() or {}
    auth_req_id = body.get("auth_req_id")
    if not isinstance(auth_req_id, str) or not auth_req_id:
        raise OAuthError("invalid_response", "the CIBA endpoint answered no auth_req_id", response.status)
    expires_in = body.get("expires_in")
    interval = body.get("interval")
    url = body.get("approval_url")
    return CibaStart(
        auth_req_id=Secret(auth_req_id),
        expires_in=expires_in if isinstance(expires_in, int) and expires_in > 0 else 300,
        interval=float(interval) if isinstance(interval, int | float) and interval > 0 else _DEFAULT_INTERVAL,
        approval_url=url if isinstance(url, str) and url else None,
    )


def ciba_poll_flow(
    token_endpoint: str,
    auth: ClientAuth,
    *,
    auth_req_id: Secret,
    interval: float,
    wait_deadline: float,
    request_deadline: float,
    now: Callable[[], float],
    resource: str | None,
    dpop: Callable[[str, str, str | None], str] | None = None,
) -> Flow[TokenResponse]:
    """Poll until the person decides (CIBA §7.3), waiting ``interval`` before the first poll.

    ``authorization_pending`` keeps it waiting and ``slow_down`` widens the interval by five
    seconds. A decision is raised at once: :class:`ConsentDenied`, :class:`ConsentExpired`. The next
    poll would fall after ``request_deadline`` (when the request expires, monotonic): the request has
    run out, :class:`ConsentExpired`. After ``wait_deadline`` (the caller's timeout): :class:`WaitTimeout`,
    and the request may still be open.
    """
    wait = interval
    while True:
        if now() + wait > min(wait_deadline, request_deadline):
            if request_deadline <= wait_deadline:
                raise ConsentExpired("expired_token", "the request ran out before the person decided")
            raise WaitTimeout(
                "the person had not decided in time; the request is still open, and waiting on it again continues it"
            )
        yield from pause(wait)
        form = {"grant_type": CIBA_GRANT, "auth_req_id": auth_req_id.reveal()}
        if resource:
            form["resource"] = resource
        response = yield from call(token_request(token_endpoint, form, auth, retry=Retry.NEVER_AFTER_SEND, dpop=dpop))
        if response.status == 200:
            return parse_token_response(response, "the token endpoint", grant=CIBA_GRANT)
        error = oauth_error(response, "the token endpoint", grant=CIBA_GRANT)
        if error.code == "slow_down":
            wait += _SLOW_DOWN_STEP
            continue
        if error.code == "authorization_pending":
            continue
        raise error


# --- token exchange --------------------------------------------------------------------------------


def exchange_request(
    token_endpoint: str,
    *,
    subject_token: str,
    actor_token: str,
    resource: str | None,
    scope: Sequence[str] | None,
    dpop: Callable[[str, str, str | None], str] | None = None,
) -> HttpRequest:
    """RFC 8693 token exchange with the agent's own token as the actor, and no client authentication.

    A ``client_id`` or ``client_assertion`` would route the request to the agent service, which
    serves no exchange; the agent authenticates through ``actor_token``.
    """
    form = {
        "grant_type": TOKEN_EXCHANGE_GRANT,
        "subject_token": subject_token,
        "subject_token_type": ACCESS_TOKEN_TYPE,
        "actor_token": actor_token,
        "actor_token_type": ACCESS_TOKEN_TYPE,
    }
    if resource:
        form["resource"] = resource
    scope_text = _space(scope)
    if scope_text:
        form["scope"] = scope_text
    return token_request(token_endpoint, form, None, retry=Retry.NEVER_AFTER_SEND, dpop=dpop)


# --- tasks ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TaskDetails:
    """What the platform granted for a task: the ``authfi_mandate`` entry of the token answer."""

    mandate_id: str
    template: str
    tools: tuple[str, ...]
    budgets: Mapping[str, int]
    expires_at: str | None


@dataclass(frozen=True, slots=True, repr=False)
class TaskPending:
    """A task a person must approve first. ``auth_req_id`` is a bearer handle, held as a :class:`Secret`."""

    auth_req_id: Secret
    mandate_id: str
    interval: float
    approval_url: str | None

    def __repr__(self) -> str:
        return f"TaskPending(mandate_id={self.mandate_id!r}, interval={self.interval})"


def task_request(
    token_endpoint: str,
    *,
    subject_token: str,
    actor_assertion: Callable[[str], str],
    template: str,
    params: Mapping[str, Any] | None,
    ttl_seconds: int | None,
    resource: str | None,
    dpop: Callable[[str, str, str | None], str],
) -> HttpRequest:
    """RFC 8693 exchange with RFC 9396 ``authorization_details`` that asks for a task. Never retried after send.

    The person's token is the ``subject_token``. The ``actor_token`` is the agent's own signed assertion,
    new for every attempt, that names the DPoP key that will hold the task; the request carries a proof of
    that key. There is no client authentication: the actor assertion is the agent's.
    """
    if not _given(template):
        raise ConfigurationError("template is required: it is the task's name, such as support.refund")
    if params is not None and not isinstance(cast(object, params), Mapping):
        raise ConfigurationError("params is a mapping of the template's parameters")
    detail: dict[str, Any] = {"type": TASK_DETAIL_TYPE, "template": template.strip(), "params": dict(params or {})}
    if ttl_seconds is not None:
        if not 0 < ttl_seconds <= _MAX_TASK_SECONDS:
            raise ConfigurationError("ttl_seconds must be from 1 second to 30 days")
        detail["ttl_seconds"] = int(ttl_seconds)
    try:
        details = json.dumps([detail], separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError("params must be JSON: strings, numbers, booleans, lists and mappings") from exc
    form = {
        "grant_type": TOKEN_EXCHANGE_GRANT,
        "subject_token": subject_token,
        "subject_token_type": ACCESS_TOKEN_TYPE,
        "actor_token_type": _JWT_TOKEN_TYPE,
        "authorization_details": details,
    }
    if resource:
        form["resource"] = resource
    return dataclasses.replace(
        token_request(token_endpoint, form, None, retry=Retry.NEVER_AFTER_SEND, dpop=dpop),
        per_attempt=lambda: Attempt(form={"actor_token": actor_assertion(token_endpoint)}),
    )


def parse_task_start(response: HttpResponse) -> TokenResponse | TaskPending:
    """The task's first token, or the wait for the person's approval (``authorization_pending``), or the refusal."""
    body = response.json_object() or {}
    if response.status == 400 and body.get("error") == "authorization_pending":
        auth_req_id = body.get("auth_req_id")
        mandate_id = body.get("authfi_mandate_id")
        if not isinstance(auth_req_id, str) or not auth_req_id or not isinstance(mandate_id, str) or not mandate_id:
            raise OAuthError("invalid_response", "the token endpoint held the task without a request to wait on", 400)
        interval = body.get("interval")
        url = body.get("approval_url")
        return TaskPending(
            auth_req_id=Secret(auth_req_id),
            mandate_id=mandate_id,
            interval=float(interval) if isinstance(interval, int | float) and interval > 0 else _DEFAULT_INTERVAL,
            approval_url=url if isinstance(url, str) and url else None,
        )
    return parse_token_response(response, "the token endpoint", grant=TOKEN_EXCHANGE_GRANT)


def task_details(response: TokenResponse) -> TaskDetails:
    """The ``authfi_mandate`` entry of a task's token answer (RFC 9396 section 7)."""
    entries = response.extra.get("authorization_details")
    if isinstance(entries, list):
        for entry in cast(list[object], entries):
            if isinstance(entry, dict) and cast(dict[str, Any], entry).get("type") == TASK_DETAIL_TYPE:
                found = cast(dict[str, Any], entry)
                tools = found.get("tools")
                budgets = found.get("budgets")
                expires = found.get("expires_at")
                return TaskDetails(
                    mandate_id=str(found.get("mandate_id") or ""),
                    template=str(found.get("template") or ""),
                    tools=tuple(t for t in cast(list[object], tools) if isinstance(t, str))
                    if isinstance(tools, list)
                    else (),
                    budgets=types.MappingProxyType(
                        {
                            k: v
                            for k, v in cast(dict[str, Any], budgets).items()
                            if isinstance(v, int) and not isinstance(v, bool)
                        }
                        if isinstance(budgets, dict)
                        else {}
                    ),
                    expires_at=expires if isinstance(expires, str) and expires else None,
                )
    raise OAuthError("invalid_response", "the token endpoint answered a task without its authorization_details", 200)


# --- the action gate -------------------------------------------------------------------------------


def _json_request(
    method: Literal["GET", "POST"], url: str, what: str, headers: Mapping[str, str], body: Any
) -> HttpRequest:
    return HttpRequest(
        method,
        url,
        what=what,
        headers={"Accept": "application/json", **headers},
        json_body=body,
        retry=Retry.NEVER_AFTER_SEND if method == "POST" else Retry.SAFE,
    )


def _data(response: HttpResponse, what: str, ok: tuple[int, ...] = (200,)) -> dict[str, Any]:
    if response.status not in ok:
        raise api_error(response, what)
    body = response.json_object() or {}
    data = body.get("data")
    if not isinstance(data, dict):
        raise APIError("invalid_response", f"{what} answered no data", response.status)
    return cast(dict[str, Any], data)


def transaction_token_request(
    api_origin: str,
    bearer: str,
    *,
    action: str,
    resource: str | None,
    context: Mapping[str, Any] | None,
) -> HttpRequest:
    """Freeze ``action`` (with ``resource`` and ``context``) into a 60-second transaction token."""
    if not _given(action):
        raise ConfigurationError("action is required")
    body: dict[str, Any] = {"action": action}
    if resource:
        body["resource"] = resource
    if context:
        body["context"] = dict(context)
    return _json_request(
        "POST",
        api_origin + TRANSACTION_TOKEN_PATH,
        "the transaction-token endpoint",
        {"Authorization": "Bearer " + bearer},
        body,
    )


def parse_transaction_token(response: HttpResponse) -> tuple[Secret, str, int]:
    """``(token, txn, expires_in)``, or the refusal."""
    data = _data(response, "the transaction-token endpoint")
    token = data.get("transaction_token")
    if not isinstance(token, str) or not token:
        raise APIError("invalid_response", "the transaction-token endpoint answered no token", response.status)
    txn = data.get("txn")
    expires_in = data.get("expires_in")
    return Secret(token), str(txn or ""), expires_in if isinstance(expires_in, int) and expires_in > 0 else 60


def authorize_request(
    endpoint: str | None,
    api_origin: str,
    *,
    agent_id: str,
    transaction_token: Secret,
    action: str,
    resource: str | None,
    context: Mapping[str, Any] | None,
) -> HttpRequest:
    """Ask the gate to decide a frozen action. The transaction token is the credential."""
    body: dict[str, Any] = {"agent_id": agent_id, "transaction_token": transaction_token.reveal(), "action": action}
    if resource:
        body["resource"] = resource
    if context:
        body["context"] = dict(context)
    return _json_request("POST", endpoint or api_origin + _AUTHORIZE_PATH, "the action gate", {}, body)


Decision = Literal["allow", "deny", "approval_required"]


def parse_action_decision(response: HttpResponse) -> tuple[Decision, str | None, str | None]:
    """``(decision, approval_id, reason)``; the decision is ``allow``, ``deny`` or ``approval_required``."""
    data = _data(response, "the action gate")
    decision = data.get("decision")
    if decision == "allow":
        verdict: Decision = "allow"
    elif decision == "deny":
        verdict = "deny"
    elif decision == "approval_required":
        verdict = "approval_required"
    else:
        raise APIError("invalid_response", f"the action gate answered decision {decision!r}", response.status)
    approval_id = data.get("approval_id")
    reason = data.get("reason")
    return (
        verdict,
        approval_id if isinstance(approval_id, str) and approval_id else None,
        reason if isinstance(reason, str) and reason else None,
    )


# --- approvals -------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    """An approval as the platform answers it."""

    approval_id: str
    status: str
    action: str
    created_at: str | None
    expires_at: str | None
    resolved_at: str | None
    context: Mapping[str, Any] = field(default_factory=lambda: types.MappingProxyType({}))


def approval_request(
    endpoint: str,
    *,
    agent_id: str,
    sign: Callable[[str], str],
    audience: str,
    action: str,
    context: Mapping[str, Any] | None,
    expires_in: int,
) -> HttpRequest:
    """Open an approval for ``action``, signing a new assertion for ``audience`` on every attempt.

    Never retried after send: a retry would open a second one. When it is sent again, because the
    request provably never left or the platform answered 429 or 503 with ``Retry-After``, the
    assertion and its ``jti`` are new.
    """
    if not _given(action):
        raise ConfigurationError("action is required: it is what the person approves")
    if expires_in <= 0:
        raise ConfigurationError("expires_in must be a positive number of seconds")
    body: dict[str, Any] = {"agent_id": agent_id, "requesting_action": action, "timeout_seconds": int(expires_in)}
    if context:
        body["context"] = dict(context)
    request = _json_request("POST", endpoint, "the approval endpoint", {}, body)
    return dataclasses.replace(
        request,
        per_attempt=lambda: Attempt(
            json_body={
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": sign(audience),
            }
        ),
    )


def read_approval_request(template: str, approval_id: str, bearer: str) -> HttpRequest:
    """Read approval ``approval_id`` with the agent's own token."""
    if "{id}" not in template:
        raise DiscoveryError("the approval endpoint has no {id} in it", code="metadata_invalid")
    if not _given(approval_id):
        raise ConfigurationError("approval_id is required")
    url = template.replace("{id}", quote(approval_id.strip(), safe=""))
    return _json_request("GET", url, "the approval endpoint", {"Authorization": "Bearer " + bearer}, None)


def parse_approval(response: HttpResponse, *, reading: bool) -> ApprovalRecord:
    """An approval record; a 403 on a read is :class:`NotSupportedError` (``agent_approval_status``)."""
    if reading and response.status == 403:
        raise NotSupportedError(
            "agent_approval_status",
            "This AuthFI project does not offer approval status to agents.",
            403,
        )
    data = _data(response, "the approval endpoint", ok=(200, 201, 202))
    approval_id = data.get("approval_id")
    if not isinstance(approval_id, str) or not approval_id:
        raise APIError("invalid_response", "the approval endpoint answered no approval_id", response.status)
    context = data.get("context")

    def text(name: str) -> str | None:
        value = data.get(name)
        return value if isinstance(value, str) and value else None

    return ApprovalRecord(
        approval_id=approval_id,
        status=text("status") or "pending",
        action=text("requesting_action") or "",
        created_at=text("created_at"),
        expires_at=text("expires_at"),
        resolved_at=text("resolved_at"),
        context=types.MappingProxyType(cast(dict[str, Any], context) if isinstance(context, dict) else {}),
    )


def label_claims(raw: str) -> dict[str, Any]:
    """A token's claims WITHOUT checking its signature: only to label a token this SDK just received.

    Never for a decision: deciding is the resource server's job, with the issuer's key.
    """
    try:
        return unverified_claims(raw)
    except InvalidToken:
        return {}
