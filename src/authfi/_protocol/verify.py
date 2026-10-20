"""Access-token verification rules. No I/O: the caller fetches the key the token names.

Each rule refuses with the kernel's code, so a refusal reads the same in this SDK's log, a
platform log and the tenant's audit tape:

1. A compact JWS of three parts, at most 16 KiB; a JWE is refused (``token_malformed``).
2. ``alg`` is one of RS256, ES256, EdDSA; ``none`` and ``HS*`` are refused before any key lookup
   (``token_signature_invalid``). The key named by ``kid`` must be a key for that alg
   (``token_key_unknown``, as the kernel decides it).
3. ``typ`` is ``at+jwt`` or ``application/at+jwt`` (``token_not_access_token``): an ID token, a
   transaction token, a security event or an Actions JWT is not an access token.
4. ``kid`` is required; ``jku``, ``x5u`` and an embedded ``jwk`` are ignored; an unknown
   ``crit`` is refused.
5. After the signature: ``iss`` exactly, an ``aud`` this API accepts, ``exp`` required, ``nbf``
   and ``iat`` within the leeway, ``sub`` and ``tenant_id`` required, the tenant pin, a bound
   token never accepted as Bearer, and no nested ``act``.
"""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import jwt

from authfi._protocol.jwks import ACCEPTED_ALGS, PublicKey
from authfi.errors import InvalidToken

__all__ = [
    "MAX_TOKEN_BYTES",
    "Unverified",
    "VerifyConfig",
    "check_claims",
    "inspect_token",
    "unverified_claims",
    "verify_signature",
]

#: The largest token this SDK reads.
MAX_TOKEN_BYTES = 16 * 1024

_ACCESS_TYPES = frozenset({"at+jwt", "application/at+jwt"})
_UNDERSTOOD_CRIT: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class VerifyConfig:
    """What a verifier accepts."""

    issuer: str
    audiences: frozenset[str]
    leeway: float
    expected_tenant_id: str | None = None


@dataclass(frozen=True, slots=True, repr=False)
class Unverified:
    """A token whose shape and header passed, before its signature is checked."""

    raw: str
    header: Mapping[str, Any]
    kid: str
    alg: str

    def __repr__(self) -> str:
        return f"Unverified(kid={self.kid!r}, alg={self.alg!r})"


def _b64json(segment: str) -> object:
    padded = segment + "=" * (-len(segment) % 4)
    return json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))


def _segments(raw: str) -> list[str]:
    if not raw:
        raise InvalidToken("token_missing", "no credential")
    if len(raw.encode("utf-8")) > MAX_TOKEN_BYTES:
        raise InvalidToken("token_malformed", f"longer than {MAX_TOKEN_BYTES} bytes")
    parts = raw.split(".")
    if len(parts) == 5:
        raise InvalidToken("token_malformed", "an encrypted token (JWE) is not an access token")
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise InvalidToken("token_malformed", "not a compact JWS of three parts")
    return parts


def _object(segment: str, what: str) -> dict[str, Any]:
    try:
        value = _b64json(segment)
    except (ValueError, UnicodeDecodeError, binascii.Error) as exc:
        raise InvalidToken("token_malformed", f"its {what} is not base64url JSON") from exc
    if not isinstance(value, dict):
        raise InvalidToken("token_malformed", f"its {what} is not a JSON object")
    return cast(dict[str, Any], value)


def inspect_token(raw: str) -> Unverified:
    """Check the token's shape and header; returns what the key lookup needs."""
    header = _object(_segments(raw)[0], "header")
    alg = header.get("alg")
    if not isinstance(alg, str) or alg not in ACCEPTED_ALGS:
        raise InvalidToken("token_signature_invalid", f"alg {alg!r} is refused")
    typ = header.get("typ")
    if not isinstance(typ, str) or typ.lower() not in _ACCESS_TYPES:
        raise InvalidToken("token_not_access_token", f"typ {typ!r} is not at+jwt")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise InvalidToken("token_key_unknown", "the token names no kid")
    crit = header.get("crit")
    if crit is not None:
        names = cast(list[object], crit) if isinstance(crit, list) else None
        if not names or any(not isinstance(n, str) or n not in _UNDERSTOOD_CRIT for n in names):
            raise InvalidToken("token_claims_invalid", f"crit {crit!r} names an extension this verifier lacks")
    return Unverified(raw=raw, header=header, kid=kid, alg=alg)


def unverified_claims(raw: str) -> dict[str, Any]:
    """The payload of ``raw`` WITHOUT checking its signature. Only to choose an issuer's verifier."""
    return _object(_segments(raw)[1], "payload")


def verify_signature(token: Unverified, key: PublicKey | None) -> dict[str, Any]:
    """The verified payload, or the refusal."""
    if key is None:
        raise InvalidToken("token_key_unknown", f"the issuer publishes no key {token.kid!r}")
    if key.alg != token.alg:
        raise InvalidToken("token_key_unknown", f"key {token.kid!r} is not a key for {token.alg}")
    try:
        decoded = jwt.api_jws.PyJWS().decode_complete(token.raw, key=key.key, algorithms=[key.alg])
    except jwt.InvalidSignatureError as exc:
        raise InvalidToken("token_signature_invalid", "the signature does not verify") from exc
    except jwt.PyJWTError as exc:
        raise InvalidToken("token_malformed", "the token could not be decoded") from exc
    payload = decoded["payload"]
    try:
        value = json.loads(payload)
    except (ValueError, UnicodeDecodeError) as exc:
        raise InvalidToken("token_malformed", "its payload is not JSON") from exc
    if not isinstance(value, dict):
        raise InvalidToken("token_malformed", "its payload is not a JSON object")
    return cast(dict[str, Any], value)


def _number(claims: Mapping[str, Any], name: str) -> float | None:
    value = claims.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InvalidToken("token_claims_invalid", f"{name} is not a number")
    return float(value)


def audiences_of(claims: Mapping[str, Any]) -> tuple[str, ...]:
    """``aud`` as a tuple, whether the token carries a string or an array."""
    aud = claims.get("aud")
    if isinstance(aud, str):
        return (aud,)
    if isinstance(aud, list):
        return tuple(a for a in cast(list[object], aud) if isinstance(a, str))
    return ()


def check_claims(claims: Mapping[str, Any], config: VerifyConfig, now: float, *, bearer: bool = True) -> None:
    """Check the verified claims against ``config`` at wall-clock ``now``.

    ``bearer`` is ``True`` for a token presented with the Bearer scheme, where a sender-constrained
    token (``cnf`` or ``authfi_mandate``) is refused.
    """
    if claims.get("iss") != config.issuer:
        raise InvalidToken("token_issuer_not_accepted", f"issuer {claims.get('iss')!r} is not {config.issuer!r}")
    exp = _number(claims, "exp")
    if exp is None:
        raise InvalidToken("token_claims_invalid", "exp is missing")
    if now > exp + config.leeway:
        raise InvalidToken("token_expired", "the token has expired")
    nbf = _number(claims, "nbf")
    if nbf is not None and nbf > now + config.leeway:
        raise InvalidToken("token_claims_invalid", "the token is not valid yet (nbf)")
    iat = _number(claims, "iat")
    if iat is not None and iat > now + config.leeway:
        raise InvalidToken("token_claims_invalid", "the token was issued in the future (iat)")
    if not config.audiences.intersection(audiences_of(claims)):
        raise InvalidToken("token_audience_mismatch", "no audience this API accepts")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise InvalidToken("token_claims_invalid", "sub is missing")
    tenant = claims.get("tenant_id")
    if not isinstance(tenant, str) or not tenant:
        raise InvalidToken("token_no_tenant", "the token names no tenant")
    if config.expected_tenant_id is not None and tenant != config.expected_tenant_id:
        raise InvalidToken("token_tenant_not_issuers", "tenant_id is not the tenant this issuer is pinned to")
    act = claims.get("act")
    if act is not None:
        if not isinstance(act, dict) or not isinstance(cast(dict[str, Any], act).get("sub"), str):
            raise InvalidToken("token_claims_invalid", "act is not an actor claim")
        if "act" in cast(dict[str, Any], act):
            raise InvalidToken("token_act_chain_unsupported", "a chain of actors is not accepted")
    if bearer and ("cnf" in claims or "authfi_mandate" in claims):
        raise InvalidToken("token_dpop_proof_required", "a sender-constrained token presented as Bearer")
