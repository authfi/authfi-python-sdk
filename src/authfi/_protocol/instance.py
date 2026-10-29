"""Instance sign-in: the wire of challenge, attest and the attested token request. No I/O.

An instance is one execution of an agent. It makes a key in memory, shows the platform where it runs
(evidence), receives an *instance attestation* (``typ`` ``oauth-client-attestation+jwt``, 15 minutes at
most) and presents it at the token endpoint with a proof of the same key
(OAuth Attestation-Based Client Authentication) and a DPoP proof. The attestation is signed by the
platform; this module checks only what the client can check: that it names this agent and this key.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import jwt

from authfi._protocol.errors import api_error
from authfi._protocol.http import Attempt, HttpRequest, HttpResponse, Retry
from authfi._protocol.oauth import b64url
from authfi._protocol.verify import unverified_claims
from authfi.errors import APIError, AuthFIError, OAuthError
from authfi.secret import Secret

__all__ = [
    "ATTESTATION_HEADER",
    "ATTESTATION_TYP",
    "ATTEST_PATH",
    "CHALLENGE_PATH",
    "MAX_ATTESTATION_LIFETIME",
    "POP_HEADER",
    "POP_TYP",
    "PROOF_TYP",
    "Attestation",
    "Configuration",
    "attest_request",
    "attested_token_request",
    "challenge_request",
    "jwk_thumbprint",
    "parse_attestation",
    "parse_challenge",
    "pop_claims",
    "refusal_with_fix",
]

ATTEST_PATH = "/apis/regional/v1/ai/agents/attest"
CHALLENGE_PATH = ATTEST_PATH + "/challenge"
ATTESTATION_TYP = "oauth-client-attestation+jwt"
POP_TYP = "oauth-client-attestation-pop+jwt"
#: The proof an instance signs over the challenge when it asks to be attested.
PROOF_TYP = "authfi-instance-proof+jwt"
ATTESTATION_HEADER = "OAuth-Client-Attestation"
POP_HEADER = "OAuth-Client-Attestation-PoP"
#: The platform signs nothing that lives longer (C2).
MAX_ATTESTATION_LIFETIME = 900
POP_LIFETIME = 60
_INSTANCE_ID = re.compile(r"inst_[0-9a-f]{32}")

#: What to do about each refusal the attest route gives, named for the person reading the error.
_FIXES = {
    "config_not_approved": (
        "Approve this configuration version in the console (Agents > the agent > Configuration), or deploy the "
        "approved version, then restart the instance."
    ),
    "proof_kind_not_built": (
        "Sign-in from this kind of workload (gcp, aws) is not built yet. Run the agent on Kubernetes or on a "
        "device bound to it."
    ),
    "evidence_replayed": (
        "A service-account token works once. Project a fresh one (expirationSeconds: 600) and let the pod "
        "rotate it; the SDK never sends the same token twice."
    ),
    "evidence_kind_mismatch": (
        "The agent is bound to a different kind of workload than the evidence found here. Check the binding in "
        "the console (Agents > the agent > Workloads)."
    ),
    "binding_not_found": (
        "No workload is bound to this agent for this evidence. Bind the namespace and service account (or the "
        "device) in the console (Agents > the agent > Workloads)."
    ),
    "pod_not_live": "The cluster does not report this pod running. Check the operator is connected for the cluster.",
    "challenge_invalid": "The challenge was used or expired (60 s, single use). Retry; it was not the SDK's last word.",
}


@dataclass(frozen=True, slots=True)
class Configuration:
    """The agent's signed configuration this instance runs: its version and digest."""

    version: int
    digest: str


@dataclass(frozen=True, slots=True, repr=False)
class Attestation:
    """An instance attestation, held as a :class:`Secret`."""

    token: Secret
    instance_id: str
    expires_at: float
    key_thumbprint: str
    claims: Mapping[str, Any] = field(default_factory=dict[str, Any])

    def __repr__(self) -> str:
        return f"Attestation(instance_id={self.instance_id!r})"


def jwk_thumbprint(jwk: Mapping[str, Any]) -> str | None:
    """The RFC 7638 thumbprint of an EC or OKP public JWK; ``None`` for a key shape this SDK does not make."""
    kty = jwk.get("kty")
    members = {"EC": ("crv", "kty", "x", "y"), "OKP": ("crv", "kty", "x")}.get(str(kty))
    if members is None or any(not isinstance(jwk.get(m), str) for m in members):
        return None
    canonical = json.dumps({m: jwk[m] for m in members}, sort_keys=True, separators=(",", ":"))
    return b64url(hashlib.sha256(canonical.encode("ascii")).digest())


def refusal_with_fix(response: HttpResponse, what: str) -> AuthFIError:
    """The platform's refusal, with the fix appended where this SDK knows one."""
    error = api_error(response, what)
    fix = _FIXES.get(error.code or "")
    if fix is None or not isinstance(error, APIError):
        return error
    named = APIError(error.code, f"{error.description + '. ' if error.description else ''}{fix}", response.status)
    named.request_id = error.request_id
    named.run_id = error.run_id
    return named


def _data(response: HttpResponse, what: str) -> dict[str, Any]:
    body = response.json_object()
    if body is None:
        raise OAuthError("invalid_response", f"{what} did not answer a JSON object", status=response.status)
    inner = body.get("data")
    return inner if isinstance(inner, dict) else body  # pyright: ignore[reportUnknownVariableType]


def challenge_request(api_origin: str) -> HttpRequest:
    """``POST .../attest/challenge``: a single-use nonce, valid 60 seconds. Public."""
    return HttpRequest(
        "POST",
        api_origin + CHALLENGE_PATH,
        what="the attestation challenge",
        headers={"Accept": "application/json"},
        json_body={},
        retry=Retry.SAFE,
    )


def parse_challenge(response: HttpResponse) -> str:
    """The nonce, or the platform's refusal."""
    if response.status != 200:
        raise refusal_with_fix(response, "the attestation challenge")
    nonce = _data(response, "the attestation challenge").get("nonce")
    if not isinstance(nonce, str) or not nonce:
        raise OAuthError("invalid_response", "the attestation challenge answered no nonce", status=response.status)
    return nonce


def attest_request(api_origin: str, body: Mapping[str, Any]) -> HttpRequest:
    """``POST .../attest``. Never sent again once it may have arrived: the evidence works once."""
    return HttpRequest(
        "POST",
        api_origin + ATTEST_PATH,
        what="the instance attestation",
        headers={"Accept": "application/json"},
        json_body=dict(body),
        retry=Retry.NEVER_AFTER_SEND,
    )


def parse_attestation(response: HttpResponse, *, agent_id: str, key_thumbprint: str, now: float) -> Attestation:
    """The attestation, after the checks a client can make: it names this agent and this instance key."""
    if response.status != 200:
        raise refusal_with_fix(response, "the instance attestation")
    raw = _data(response, "the instance attestation").get("attestation")
    if not isinstance(raw, str) or not raw:
        raise OAuthError("invalid_response", "the attest route answered no attestation", status=response.status)
    try:
        header = jwt.get_unverified_header(raw)
        claims = unverified_claims(raw)
    except Exception as error:
        raise OAuthError("invalid_response", "the attestation is not a JWT", status=response.status) from error
    problem = _attestation_problem(header, claims, agent_id, key_thumbprint, now)
    if problem:
        raise OAuthError("invalid_response", f"the attestation {problem}", status=response.status)
    instance = claims["authfi_instance"]
    return Attestation(Secret(raw), instance["instance_id"], float(claims["exp"]), key_thumbprint, claims)


def _attestation_problem(
    header: Mapping[str, Any], claims: Mapping[str, Any], agent_id: str, thumbprint: str, now: float
) -> str | None:
    if header.get("typ") != ATTESTATION_TYP:
        return f"has typ {header.get('typ')!r}, not {ATTESTATION_TYP}"
    if claims.get("sub") != agent_id:
        return f"is for {claims.get('sub')!r}, not {agent_id}"
    cnf = claims.get("cnf")
    jwk: object = cast("Mapping[str, Any]", cnf).get("jwk") if isinstance(cnf, dict) else None
    if not isinstance(jwk, dict) or jwk_thumbprint(cast("Mapping[str, Any]", jwk)) != thumbprint:
        return "is bound to a different key than this instance's"
    instance: object = claims.get("authfi_instance")
    if not isinstance(instance, dict) or not _INSTANCE_ID.fullmatch(
        str(cast("Mapping[str, Any]", instance).get("instance_id"))
    ):
        return "names no instance_id (inst_ and 32 hex digits)"
    iat, exp = claims.get("iat"), claims.get("exp")
    if not isinstance(iat, int | float) or not isinstance(exp, int | float) or isinstance(exp, bool):
        return "has no lifetime"
    if exp - iat > MAX_ATTESTATION_LIFETIME:
        return f"lives longer than {MAX_ATTESTATION_LIFETIME} seconds"
    if exp <= now:
        return "has already expired (is the clock right?)"
    return None


def attested_token_request(
    token_endpoint: str,
    agent_id: str,
    attestation: Attestation,
    *,
    pop: Callable[[], str],
    dpop: Callable[[], str],
    resource: str | None,
) -> HttpRequest:
    """``client_credentials`` for ``agent_id`` with the attestation, a proof of its key, and a DPoP proof.

    The proofs are made fresh for each attempt. The request is not sent again after it may have arrived:
    its proof carries a single-use challenge.
    """
    form = {"grant_type": "client_credentials", "client_id": agent_id}
    if resource:
        form["resource"] = resource

    def attempt() -> Attempt:
        return Attempt(headers={ATTESTATION_HEADER: attestation.token.reveal(), POP_HEADER: pop(), "DPoP": dpop()})

    return HttpRequest(
        "POST",
        token_endpoint,
        what="the token endpoint",
        headers={"Accept": "application/json"},
        form=form,
        retry=Retry.NEVER_AFTER_SEND,
        per_attempt=attempt,
    )


def pop_claims(agent_id: str, audience: str, challenge: str, now: float) -> dict[str, Any]:
    """The claims of the proof of possession (``iss`` the agent, ``aud`` the issuer)."""
    issued = int(now)
    return {
        "iss": agent_id,
        "aud": audience,
        "jti": str(uuid.uuid4()),
        "iat": issued,
        "exp": issued + POP_LIFETIME,
        "challenge": challenge,
    }
