"""Checking a DPoP proof (RFC 9449 §4.3) that arrives with a bound token. For servers that take AuthFI tokens.

The proof is a JWT whose header names its own public key. It must be ``typ`` ``dpop+jwt``, signed by that key
(ES256 or EdDSA), for this method and this URL, issued within a minute, never seen before, and carry the hash of
the access token it travels with; and the key must be the one the token is bound to (``cnf.jkt``). Each refusal
has the kernel's code, so it reads the same in this SDK's log and a platform log.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Any, cast

import jwt
from jwt.algorithms import ECAlgorithm, OKPAlgorithm

from authfi._dpop import htu
from authfi._protocol.instance import jwk_thumbprint
from authfi._protocol.oauth import b64url
from authfi.errors import InvalidToken

__all__ = ["ProofReplayCache", "check_proof"]

_ALGORITHMS = ("ES256", "EdDSA")
_PRIVATE = ("d", "p", "q", "dp", "dq", "qi", "k")


class ProofReplayCache:
    """The ``jti`` of every proof seen in the last ``window`` seconds, bounded."""

    def __init__(self, window: float = 120.0, max_entries: int = 100_000) -> None:
        self._window = window
        self._max = max_entries
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def first_use(self, jti: str, now: float) -> bool:
        """``True`` the first time ``jti`` is offered inside the window."""
        with self._lock:
            if len(self._seen) >= self._max:
                self._seen = {k: t for k, t in self._seen.items() if now - t < self._window}
                if len(self._seen) >= self._max:
                    return False  # full of live entries: refuse rather than forget one
            known = self._seen.get(jti)
            if known is not None and now - known < self._window:
                return False
            self._seen[jti] = now
            return True


def _public_key(jwk: object) -> Any:
    if not isinstance(jwk, dict):
        raise InvalidToken("dpop_proof_key_invalid", "the proof header carries no jwk")
    members = cast("dict[str, Any]", jwk)
    if any(member in members for member in _PRIVATE):
        raise InvalidToken("dpop_proof_key_invalid", "the proof's jwk carries private key material")
    try:
        if members.get("kty") == "EC" and members.get("crv") == "P-256":
            return ECAlgorithm(ECAlgorithm.SHA256).from_jwk(members)
        if members.get("kty") == "OKP" and members.get("crv") == "Ed25519":
            return OKPAlgorithm().from_jwk(members)
    except Exception as error:
        raise InvalidToken("dpop_proof_key_invalid", "the proof's jwk is not a valid key") from error
    raise InvalidToken("dpop_proof_key_invalid", "the proof's key is not P-256 or Ed25519")


def check_proof(
    proof: str | None,
    *,
    method: str,
    url: str,
    access_token: str,
    jkt: str | None,
    now: float,
    replay: ProofReplayCache,
    leeway: float = 60.0,
) -> str:
    """Check ``proof`` for ``method url`` with ``access_token``; return the thumbprint of its key.

    ``jkt`` is the token's ``cnf.jkt``: a token bound to no key is refused (``token_dpop_unbound``).
    """
    if not jkt:
        raise InvalidToken("token_dpop_unbound", "the DPoP scheme with a token bound to no key")
    if not proof:
        raise InvalidToken("dpop_proof_missing", "no DPoP header")
    try:
        header = jwt.get_unverified_header(proof)
    except jwt.PyJWTError as error:
        raise InvalidToken("dpop_proof_malformed", "the proof is not a JWT") from error
    if header.get("typ") != "dpop+jwt":
        raise InvalidToken("dpop_proof_type_invalid", f"typ {header.get('typ')!r} is not dpop+jwt")
    alg = header.get("alg")
    if alg not in _ALGORITHMS:
        raise InvalidToken("dpop_proof_alg_mismatch", f"alg {alg!r} is not ES256 or EdDSA")
    jwk = header.get("jwk")
    key = _public_key(jwk)
    try:
        claims = jwt.decode(
            proof, key, algorithms=[alg], options={"verify_aud": False, "verify_exp": False, "verify_iat": False}
        )
    except jwt.InvalidSignatureError as error:
        raise InvalidToken("dpop_proof_signature_invalid", "the signature does not verify") from error
    except jwt.PyJWTError as error:
        raise InvalidToken("dpop_proof_malformed", "the proof could not be read") from error
    if claims.get("htm") != method.upper():
        raise InvalidToken("dpop_proof_htm_mismatch", "the proof is for another method")
    if claims.get("htu") != htu(url):
        raise InvalidToken("dpop_proof_htu_mismatch", "the proof is for another URL")
    iat = claims.get("iat")
    if isinstance(iat, bool) or not isinstance(iat, int | float) or abs(now - iat) > leeway:
        raise InvalidToken("dpop_proof_iat_invalid", "the proof was not issued just now")
    jti = claims.get("jti")
    if not isinstance(jti, str) or not jti:
        raise InvalidToken("dpop_proof_jti_missing", "the proof has no jti")
    expected = b64url(hashlib.sha256(access_token.encode("ascii", "replace")).digest())
    if claims.get("ath") != expected:
        raise InvalidToken("dpop_proof_ath_mismatch", "the proof does not bind this access token")
    thumbprint = jwk_thumbprint(cast("dict[str, Any]", jwk))
    if thumbprint != jkt:
        raise InvalidToken("dpop_proof_key_mismatch", "the proof's key is not the one the token is bound to")
    if not replay.first_use(jti, now):
        raise InvalidToken("dpop_proof_replayed", "this proof was already used")
    return jkt
