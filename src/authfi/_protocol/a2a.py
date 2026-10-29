"""A2A 1.0 on the wire: the agent card and its signature, messages, task states, and the AuthFI mapping. No I/O.

The field names are the specification's (a2a-protocol.org/latest/specification): ``supportedInterfaces`` with
``url``, ``protocolBinding`` and ``protocolVersion``; ``securitySchemes``; ``signatures`` of ``protected``,
``signature`` and ``header``; ``TASK_STATE_*``; ``ROLE_USER`` / ``ROLE_AGENT``; ``parts`` of ``{"text": ...}``.

**A card is verified, or it is not trusted.** The signature is a JWS (RFC 7515) over the card without its
``signatures`` member, canonicalised with RFC 8785, exactly as the card was served. The key is either the leaf of
the ``x5c`` chain in the protected header, which must lead to a root you trust through certificates that issue CAs
and a leaf with the agent-custody policy (an AuthFI agent: the chain ends at Root R1), or a key you name by ``kid``.
"""

from __future__ import annotations

import base64
import datetime as dt
import itertools
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import jwt
from cryptography import x509
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

from authfi._protocol.gateway import ToolAllowed, ToolApprovalRequired, ToolOutcome
from authfi._protocol.http import DOCUMENT_LIMIT, HttpRequest, HttpResponse, Retry
from authfi._protocol.oauth import b64url
from authfi.errors import AuthFIError

__all__ = [
    "AGENT_CUSTODY_POLICY",
    "AUTH_REQUIRED",
    "A2AError",
    "AgentCard",
    "CardUnverified",
    "Interface",
    "TaskStatus",
    "canonical_json",
    "card_request",
    "message_body",
    "message_request",
    "message_url",
    "parse_card",
    "parse_task",
    "status_for",
    "status_for_outcome",
    "verify_card",
]

WELL_KNOWN = "/.well-known/agent-card.json"
VERSION = "1.0"
AGENT_CUSTODY_POLICY = "1.3.6.1.4.1.67322.1.4"
_ALGORITHMS = ("EdDSA", "ES256", "PS256")

INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"
REJECTED = "TASK_STATE_REJECTED"
FAILED = "TASK_STATE_FAILED"
COMPLETED = "TASK_STATE_COMPLETED"
WORKING = "TASK_STATE_WORKING"
STATES = frozenset(
    {
        "TASK_STATE_SUBMITTED",
        WORKING,
        COMPLETED,
        FAILED,
        "TASK_STATE_CANCELED",
        INPUT_REQUIRED,
        REJECTED,
        AUTH_REQUIRED,
    }
)

#: Refusal codes that mean "a person's consent or credentials are missing", not "no".
_AUTH_CODES = frozenset(
    {
        "consent_required",
        "credentials_required",
        "connection_required",
        "authorization_required",
        "interaction_required",
        "login_required",
    }
)


class A2AError(AuthFIError):
    """An A2A call or card the SDK refused. ``code`` says which, the message says what to do."""

    def __init__(self, code: str, message: str, status: int | None = None) -> None:
        super().__init__(message, status, code=code)


class CardUnverified(A2AError):
    """The agent card's signature did not verify, or the card has none."""

    def __init__(self, message: str) -> None:
        super().__init__("card_unverified", message)


# --- RFC 8785 --------------------------------------------------------------------------------------


def canonical_json(value: Any) -> bytes:
    """RFC 8785 (JCS): members sorted by UTF-16 code units, no space, the shortest string escapes, UTF-8.

    Numbers must be integers (a card has none that are not); a float, NaN or a non-string key is refused.
    """

    def emit(v: Any) -> str:
        if v is None:
            return "null"
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            if v.is_integer() and abs(v) < 2**53:
                return str(int(v))
            raise ValueError("a card number must be an integer")
        if isinstance(v, str):
            return json.dumps(v, ensure_ascii=False)
        if isinstance(v, list):
            return "[" + ",".join(emit(i) for i in cast("list[object]", v)) + "]"
        if isinstance(v, dict):
            items = cast("dict[object, object]", v)
            if any(not isinstance(k, str) for k in items):
                raise ValueError("a JSON object key must be a string")
            ordered = sorted(items, key=lambda k: cast(str, k).encode("utf-16-be"))
            return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + emit(items[k]) for k in ordered) + "}"
        raise ValueError(f"{type(v).__name__} is not JSON")

    return emit(value).encode("utf-8")


# --- the card ----------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Interface:
    """One ``AgentInterface``: where and how the agent is called."""

    url: str
    binding: str
    version: str


@dataclass(frozen=True, slots=True)
class AgentCard:
    """A fetched agent card. ``verified`` says whether its signature was checked; ``signer`` names the key."""

    raw: Mapping[str, Any]
    name: str
    interfaces: tuple[Interface, ...]
    schemes: tuple[str, ...]
    verified: bool = False
    signer: str | None = None

    def interface(self) -> Interface:
        """The interface to call: HTTP+JSON first, then JSON-RPC. Others (gRPC) this SDK does not speak."""
        for wanted in ("HTTP+JSON", "JSONRPC"):
            for found in self.interfaces:
                if found.binding == wanted:
                    return found
        raise A2AError(
            "no_interface",
            f"{self.name} offers only {sorted({i.binding for i in self.interfaces}) or 'no interface'}; "
            "this SDK speaks HTTP+JSON and JSONRPC",
        )


def card_request(origin: str) -> HttpRequest:
    """``GET <origin>/.well-known/agent-card.json``."""
    return HttpRequest(
        "GET",
        origin.rstrip("/") + WELL_KNOWN,
        what="the agent card",
        headers={"Accept": "application/json"},
        retry=Retry.SAFE,
        max_bytes=DOCUMENT_LIMIT,
    )


def parse_card(response: HttpResponse) -> AgentCard:
    """The card, unverified."""
    if response.status != 200:
        raise A2AError("card_unavailable", f"the agent card answered HTTP {response.status}", response.status)
    body = response.json_object()
    if body is None:
        raise A2AError("card_invalid", "the agent card is not a JSON object")
    name = body.get("name")
    raw_interfaces = body.get("supportedInterfaces")
    if not isinstance(name, str) or not isinstance(raw_interfaces, list) or not raw_interfaces:
        raise A2AError("card_invalid", "the agent card has no name or supportedInterfaces")
    interfaces: list[Interface] = []
    for item in cast("list[object]", raw_interfaces):
        if isinstance(item, dict):
            entry = cast("dict[str, Any]", item)
            url, binding, version = entry.get("url"), entry.get("protocolBinding"), entry.get("protocolVersion")
            if all(isinstance(x, str) and x for x in (url, binding, version)):
                interfaces.append(Interface(cast(str, url), cast(str, binding), cast(str, version)))
    schemes_raw = body.get("securitySchemes")
    schemes: list[str] = []
    if isinstance(schemes_raw, dict):
        for scheme in cast("dict[str, Any]", schemes_raw).values():
            if isinstance(scheme, dict):
                schemes.extend(str(k) for k in cast("dict[str, Any]", scheme))
    return AgentCard(body, name, tuple(interfaces), tuple(schemes))


# --- verification -------------------------------------------------------------------------------------


def _load_roots(roots: Sequence[bytes | str]) -> list[x509.Certificate]:
    loaded: list[x509.Certificate] = []
    for item in roots:
        data = item.encode() if isinstance(item, str) else item
        loaded.append(
            x509.load_pem_x509_certificate(data)
            if b"BEGIN CERTIFICATE" in data
            else x509.load_der_x509_certificate(data)
        )
    return loaded


def _signed_by(cert: x509.Certificate, issuer: x509.Certificate) -> bool:
    key = issuer.public_key()
    try:
        if isinstance(key, ed25519.Ed25519PublicKey):
            key.verify(cert.signature, cert.tbs_certificate_bytes)
        elif isinstance(key, ec.EllipticCurvePublicKey):
            key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                ec.ECDSA(cast(hashes.HashAlgorithm, cert.signature_hash_algorithm)),
            )
        elif isinstance(key, rsa.RSAPublicKey):
            key.verify(
                cert.signature,
                cert.tbs_certificate_bytes,
                padding.PKCS1v15(),
                cast(hashes.HashAlgorithm, cert.signature_hash_algorithm),
            )
        else:
            return False
    except (InvalidSignature, ValueError, TypeError):
        return False
    return cert.issuer == issuer.subject


def _is_ca(cert: x509.Certificate) -> bool:
    try:
        return cert.extensions.get_extension_for_class(x509.BasicConstraints).value.ca
    except x509.ExtensionNotFound:
        return False


def verify_chain(
    x5c: Sequence[str], roots: Sequence[bytes | str], *, now: dt.datetime, policy: str = AGENT_CUSTODY_POLICY
) -> x509.Certificate:
    """The leaf of ``x5c`` (leaf first, base64 DER), after checking it leads to a trusted root.

    Every link is signed by the next; every issuer is a CA; every certificate is valid now; the last one is
    issued by one of ``roots``; the leaf is no CA and carries certificate policy ``policy``.
    """
    if not x5c or not roots:
        raise CardUnverified("a chain and a trusted root are both needed to verify a card by certificate")
    try:
        chain = [x509.load_der_x509_certificate(base64.b64decode(c, validate=True)) for c in x5c]
        trusted = _load_roots(roots)
    except (ValueError, TypeError) as error:
        raise CardUnverified("a certificate in the chain or the trusted roots cannot be read") from error
    for cert in chain:
        if not cert.not_valid_before_utc <= now <= cert.not_valid_after_utc:
            raise CardUnverified(f"the certificate {cert.subject.rfc4514_string()} is not valid now")
    for cert, issuer in itertools.pairwise(chain):
        if not _signed_by(cert, issuer) or not _is_ca(issuer):
            raise CardUnverified("the chain's links do not each lead to the next certificate authority")
    top = chain[-1]
    anchor = next(
        (
            r
            for r in trusted
            if r.fingerprint(hashes.SHA256()) == top.fingerprint(hashes.SHA256()) or _signed_by(top, r)
        ),
        None,
    )
    if anchor is None:
        raise CardUnverified("the chain does not lead to a root you trust")
    if anchor is not top and (
        not _is_ca(anchor) or not anchor.not_valid_before_utc <= now <= anchor.not_valid_after_utc
    ):
        raise CardUnverified("the trusted root is not a valid certificate authority now")
    leaf = chain[0]
    if _is_ca(leaf) and len(chain) > 1:
        raise CardUnverified("the signing certificate is a certificate authority")
    held: set[str] = set()
    try:
        held = {
            p.policy_identifier.dotted_string
            for p in leaf.extensions.get_extension_for_class(x509.CertificatePolicies).value
        }
    except x509.ExtensionNotFound:
        pass
    if policy not in held:
        raise CardUnverified(f"the signing certificate does not carry the agent-custody policy {policy}")
    return leaf


def verify_card(
    card: AgentCard,
    *,
    trust_roots: Sequence[bytes | str] = (),
    keys: Mapping[str, Mapping[str, Any]] | None = None,
    now: dt.datetime | None = None,
) -> AgentCard:
    """``card`` with ``verified`` set, after at least one of its signatures checks; else :class:`CardUnverified`.

    A signature whose protected header has an ``x5c`` chain is checked with the chain's leaf, which must lead to
    one of ``trust_roots``. Otherwise the key is ``keys[kid]`` (a public JWK you obtained and trust).
    """
    moment = now or dt.datetime.now(dt.UTC)
    raw = dict(card.raw)
    signatures = raw.pop("signatures", None)
    if not isinstance(signatures, list) or not signatures:
        raise CardUnverified(f"{card.name}'s agent card carries no signature, so it cannot be trusted")
    try:
        payload = b64url(canonical_json(raw))
    except ValueError as error:
        raise CardUnverified(f"the card cannot be canonicalised ({error})") from error
    problems: list[str] = []
    for entry in cast("list[object]", signatures):
        try:
            signer = _check_signature(cast("Mapping[str, Any]", entry), payload, trust_roots, keys or {}, moment)
        except CardUnverified as error:
            problems.append(str(error))
            continue
        return AgentCard(card.raw, card.name, card.interfaces, card.schemes, verified=True, signer=signer)
    raise CardUnverified(f"{card.name}'s agent card did not verify: " + "; ".join(problems))


def _check_signature(
    entry: Mapping[str, Any],
    payload: str,
    roots: Sequence[bytes | str],
    keys: Mapping[str, Mapping[str, Any]],
    now: dt.datetime,
) -> str:
    protected, signature = entry.get("protected"), entry.get("signature")
    if not isinstance(protected, str) or not isinstance(signature, str):
        raise CardUnverified("a signature lacks protected or signature")
    try:
        header = json.loads(base64.urlsafe_b64decode(protected + "=" * (-len(protected) % 4)))
    except (ValueError, TypeError) as error:
        raise CardUnverified("a signature's protected header is not JSON") from error
    if not isinstance(header, dict):
        raise CardUnverified("a signature's protected header is not an object")
    head = cast("dict[str, Any]", header)
    alg, kid = head.get("alg"), head.get("kid")
    if alg not in _ALGORITHMS:
        raise CardUnverified(f"signature algorithm {alg!r} is not one of {', '.join(_ALGORITHMS)}")
    if not isinstance(kid, str) or not kid:
        raise CardUnverified("a signature names no kid")
    x5c = head.get("x5c")
    key: Any
    if isinstance(x5c, list) and x5c:
        leaf = verify_chain(cast("list[str]", x5c), roots, now=now)
        key, name = leaf.public_key(), f"{kid} ({leaf.subject.rfc4514_string() or 'agent certificate'})"
    elif kid in keys:
        key, name = jwt.PyJWK(dict(keys[kid])).key, kid
    else:
        raise CardUnverified(f"no certificate chain in the signature and no key given for kid {kid!r}")
    try:
        jwt.api_jws.PyJWS().decode(f"{protected}.{payload}.{signature}", key, algorithms=[alg])
    except jwt.PyJWTError as error:
        raise CardUnverified("the signature does not match the card") from error
    return name


# --- messages and tasks ---------------------------------------------------------------------------------


def message_body(text: str, *, context_id: str | None = None, task_id: str | None = None) -> dict[str, Any]:
    """A ``SendMessageRequest`` body for one user text part."""
    message: dict[str, Any] = {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": [{"text": text}]}
    if context_id:
        message["contextId"] = context_id
    if task_id:
        message["taskId"] = task_id
    return {"message": message}


def message_url(interface: Interface) -> str:
    """The URL a ``SendMessage`` goes to, which the token's DPoP proof must name."""
    return interface.url.rstrip("/") + "/message:send" if interface.binding == "HTTP+JSON" else interface.url


def message_request(interface: Interface, body: Mapping[str, Any], headers: Mapping[str, str]) -> HttpRequest:
    """``SendMessage`` over the interface's binding. Never sent again once it may have arrived."""
    common = {"A2A-Version": VERSION, "Accept": "application/a2a+json, application/json", **headers}
    if interface.binding == "HTTP+JSON":
        return HttpRequest(
            "POST",
            message_url(interface),
            what="the A2A agent",
            headers={**common, "Content-Type": "application/a2a+json"},
            json_body=dict(body),
            retry=Retry.NEVER_AFTER_SEND,
        )
    rpc = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": "SendMessage", "params": dict(body)}
    return HttpRequest(
        "POST",
        interface.url,
        what="the A2A agent",
        headers={**common, "Content-Type": "application/json"},
        json_body=rpc,
        retry=Retry.NEVER_AFTER_SEND,
    )


@dataclass(frozen=True, slots=True)
class TaskStatus:
    """A task's state (``TASK_STATE_*``) and the words the agent gave with it."""

    state: str
    text: str | None = None
    task_id: str | None = None
    context_id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict[str, Any], repr=False)

    @property
    def needs_input(self) -> bool:
        """A person's answer or approval is awaited."""
        return self.state == INPUT_REQUIRED

    @property
    def needs_auth(self) -> bool:
        """Consent or credentials are awaited."""
        return self.state == AUTH_REQUIRED

    @property
    def rejected(self) -> bool:
        """The agent declined the task."""
        return self.state == REJECTED

    def to_status(self) -> dict[str, Any]:
        """A ``TaskStatus`` object for a response: ``state`` and, if there are words, an agent ``message``."""
        status: dict[str, Any] = {"state": self.state}
        if self.text:
            status["message"] = {"messageId": str(uuid.uuid4()), "role": "ROLE_AGENT", "parts": [{"text": self.text}]}
        return status


def _text_of(message: object) -> str | None:
    if not isinstance(message, dict):
        return None
    parts = cast("dict[str, Any]", message).get("parts")
    texts: list[str] = []
    if isinstance(parts, list):
        for part in cast("list[object]", parts):
            text = cast("dict[str, Any]", part).get("text") if isinstance(part, dict) else None
            if isinstance(text, str):
                texts.append(text)
    return "\n".join(texts) or None


def parse_task(response: HttpResponse) -> TaskStatus:
    """The task (or the reply message) a ``SendMessage`` answered; a refusal is the typed error."""
    if response.status in (401, 403):
        challenge = response.header("WWW-Authenticate")
        raise A2AError(
            "remote_refused",
            f"the A2A agent refused the credential (HTTP {response.status}"
            + (f", {challenge}" if challenge else "")
            + "); check the remote agent accepts this project's tokens for the resource you named",
            response.status,
        )
    if not 200 <= response.status < 300:
        raise A2AError("remote_error", f"the A2A agent answered HTTP {response.status}", response.status)
    body = response.json_object() or {}
    if "error" in body and isinstance(body["error"], dict):
        err = cast("dict[str, Any]", body["error"])
        raise A2AError("remote_error", f"the A2A agent answered an error: {err.get('message', err.get('code'))}")
    result = body.get("result", body)
    if not isinstance(result, dict):
        raise A2AError("response_invalid", "the A2A agent answered neither a task nor a message")
    answer = cast("dict[str, Any]", result)
    task = answer.get("task") if isinstance(answer.get("task"), dict) else None
    if task is None and isinstance(answer.get("status"), dict):
        task = answer
    if task is not None:
        status = cast("dict[str, Any]", task.get("status") or {})
        state = status.get("state")
        if state not in STATES:
            raise A2AError("response_invalid", f"the A2A agent answered an unknown task state {state!r}")
        return TaskStatus(
            state, _text_of(status.get("message")), _str(task.get("id")), _str(task.get("contextId")), answer
        )
    message = answer.get("message") if isinstance(answer.get("message"), dict) else None
    if message is not None:
        return TaskStatus(
            COMPLETED, _text_of(message), _str(message.get("taskId")), _str(message.get("contextId")), answer
        )
    raise A2AError("response_invalid", "the A2A agent answered neither a task nor a message")


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


# --- AuthFI decisions as task states ----------------------------------------------------------------------


def status_for(decision: str, reason: str | None = None, *, code: str | None = None) -> TaskStatus:
    """The task state for a policy decision.

    ``needs_approval`` is ``INPUT_REQUIRED`` (a person must decide); ``needs_consent`` or ``needs_credentials``, or a
    deny whose ``code`` says consent or credentials are missing, is ``AUTH_REQUIRED``; ``deny`` is ``REJECTED`` with
    the reason; ``allow`` is ``WORKING``. Anything else is ``FAILED``: an answer this SDK does not know is no yes.
    """
    if decision == "allow":
        return TaskStatus(WORKING)
    if decision == "needs_approval":
        return TaskStatus(INPUT_REQUIRED, reason or "A person must approve this action.")
    if decision in ("needs_consent", "needs_credentials") or (decision == "deny" and code in _AUTH_CODES):
        return TaskStatus(AUTH_REQUIRED, reason or "Consent or credentials are needed to continue.")
    if decision == "deny":
        return TaskStatus(REJECTED, reason or "Denied by policy.")
    return TaskStatus(FAILED, reason or "The decision could not be read.")


def status_for_outcome(outcome: ToolOutcome) -> TaskStatus:
    """:func:`status_for` for what the gateway did with a tool call."""
    if isinstance(outcome, ToolAllowed):
        return status_for("allow")
    if isinstance(outcome, ToolApprovalRequired):
        return status_for("needs_approval")
    return status_for("deny", outcome.reason or outcome.code or outcome.error, code=outcome.code or outcome.error)
