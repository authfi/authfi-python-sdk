"""DPoP (RFC 9449): the proof of the key an agent's token is bound to.

- One run key per process and algorithm, held in memory and never serialised; a forked child makes its
  own (a key two processes share is not bound to either). It is Ed25519 (``EdDSA``), the platform's
  first choice, and P-256 (``ES256``) where the runtime has no Ed25519 or the project lists only ES256.
- A fresh proof for every attempt: a new ``jti`` and ``iat``, and the origin's current nonce.
- ``htu`` is the request URL without its query or fragment; ``ath`` binds the access token.
- A token bound at mint (``token_type`` ``DPoP``, ``cnf.jkt`` this key's thumbprint) is sent as
  ``Authorization: DPoP <token>`` with a proof. One the platform left unbound stays Bearer.
- A server nonce is asked for with ``use_dpop_nonce`` and a ``DPoP-Nonce`` header; the request is sent
  again once with it.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, cast

import jwt
from cryptography.exceptions import UnsupportedAlgorithm
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from authfi._protocol.http import HttpResponse
from authfi._protocol.oauth import TokenResponse, b64url
from authfi._protocol.urls import origin_of
from authfi._protocol.verify import unverified_claims
from authfi.errors import InvalidToken, OAuthError

__all__ = [
    "AuthHeaders",
    "DPoPKey",
    "NonceCache",
    "authorization_headers",
    "bound_key",
    "htu",
    "is_nonce_challenge",
    "key_for",
    "learn_nonce",
    "run_key",
    "signer",
]

DPoPAlg = Literal["EdDSA", "ES256"]
#: The ``DPoP-Nonce`` response header.
NONCE_HEADER = "DPoP-Nonce"
#: The error code of a server that wants its nonce in the proof (RFC 9449 section 8).
USE_NONCE = "use_dpop_nonce"


def htu(url: str) -> str:
    """The ``htu`` of ``url``: no query, no fragment."""
    return url.split("#", 1)[0].split("?", 1)[0]


def _int_b64(value: int, length: int) -> str:
    return b64url(value.to_bytes(length, "big"))


class DPoPKey:
    """A key that signs DPoP proofs. The private half has no accessor and does not pickle."""

    __slots__ = ("_key", "_public_jwk", "alg", "thumbprint")

    def __init__(self, alg: DPoPAlg | None = None) -> None:
        if alg not in (None, "EdDSA", "ES256"):
            raise ValueError("a DPoP key is EdDSA (Ed25519) or ES256 (P-256)")
        self.alg: DPoPAlg
        self._key: ed25519.Ed25519PrivateKey | ec.EllipticCurvePrivateKey
        if alg in (None, "EdDSA"):
            try:
                self._key = ed25519.Ed25519PrivateKey.generate()
                self.alg = "EdDSA"
            except UnsupportedAlgorithm:
                if alg == "EdDSA":
                    raise
                self._key = ec.generate_private_key(ec.SECP256R1())
                self.alg = "ES256"
        else:
            self._key = ec.generate_private_key(ec.SECP256R1())
            self.alg = "ES256"
        if isinstance(self._key, ed25519.Ed25519PrivateKey):
            raw = self._key.public_key().public_bytes_raw()
            self._public_jwk = {"kty": "OKP", "crv": "Ed25519", "x": b64url(raw)}
        else:
            numbers = self._key.public_key().public_numbers()
            self._public_jwk = {"kty": "EC", "crv": "P-256", "x": _int_b64(numbers.x, 32), "y": _int_b64(numbers.y, 32)}
        canonical = json.dumps(self._public_jwk, sort_keys=True, separators=(",", ":"))
        #: The RFC 7638 thumbprint, which a bound token names in ``cnf.jkt``.
        self.thumbprint = b64url(hashlib.sha256(canonical.encode("ascii")).digest())

    def public_jwk(self) -> dict[str, str]:
        """The public key, as a proof's header carries it."""
        return dict(self._public_jwk)

    def proof(
        self,
        method: str,
        url: str,
        *,
        now: float,
        access_token: str | None = None,
        nonce: str | None = None,
    ) -> str:
        """A fresh proof for one attempt of ``method url``."""
        claims: dict[str, Any] = {"jti": str(uuid.uuid4()), "htm": method.upper(), "htu": htu(url), "iat": int(now)}
        if access_token is not None:
            claims["ath"] = b64url(hashlib.sha256(access_token.encode("ascii")).digest())
        if nonce is not None:
            claims["nonce"] = nonce
        return jwt.encode(claims, self._key, algorithm=self.alg, headers={"typ": "dpop+jwt", "jwk": self.public_jwk()})

    def sign_jwt(self, claims: dict[str, Any], *, typ: str) -> str:
        """A compact JWS of ``claims`` with this key, in its own algorithm, for the proofs an instance signs.

        The platform refuses a proof whose algorithm is not the one its key implies, so this never signs
        with another.
        """
        return jwt.encode(claims, self._key, algorithm=self.alg, headers={"typ": typ})

    def __repr__(self) -> str:
        return f"DPoPKey(alg={self.alg!r}, thumbprint={self.thumbprint!r})"

    def __reduce__(self) -> Any:
        raise TypeError("A DPoP key cannot be pickled: it belongs to the process that made it")


class NonceCache:
    """The latest ``DPoP-Nonce`` each origin answered."""

    def __init__(self) -> None:
        self._nonces: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, url: str) -> str | None:
        """The nonce to send to ``url``'s origin."""
        with self._lock:
            return self._nonces.get(origin_of(url))

    def remember(self, url: str, nonce: str | None) -> None:
        """Keep the nonce ``url``'s origin answered."""
        if nonce:
            with self._lock:
                self._nonces[origin_of(url)] = nonce


_keys: dict[str, DPoPKey] = {}
_key_lock = threading.Lock()
nonces = NonceCache()


def run_key(alg: DPoPAlg | None = None) -> DPoPKey:
    """This process's DPoP key for ``alg`` (Ed25519 where the runtime has it), made on first use."""
    with _key_lock:
        if alg is None:
            if "EdDSA" in _keys:
                return _keys["EdDSA"]
            try:
                return _keys.setdefault("EdDSA", DPoPKey("EdDSA"))
            except UnsupportedAlgorithm:
                alg = "ES256"  # no Ed25519 in this runtime
        if alg not in _keys:
            _keys[alg] = DPoPKey(alg)
        return _keys[alg]


def key_for(advertised: Sequence[str]) -> DPoPKey | None:
    """The run key a project accepts: ours when it lists no algorithms or lists ours, else the other, else ``None``."""
    first = run_key()
    if not advertised or first.alg in advertised:
        return first
    if first.alg != "ES256" and "ES256" in advertised:
        return run_key("ES256")
    return None


def _after_fork_in_child() -> None:
    global _key_lock, nonces
    _keys.clear()
    _key_lock = threading.Lock()
    nonces = NonceCache()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


#: ``(method, url, nonce) -> proof``: what a request signs with, once per attempt.
ProofSigner = Callable[[str, str, str | None], str]


def signer(key: DPoPKey, now: Callable[[], float], access_token: str | None = None) -> ProofSigner:
    """A function that signs a proof of ``key`` for ``access_token`` (none at the token endpoint)."""

    def sign(method: str, url: str, nonce: str | None) -> str:
        return key.proof(method, url, now=now(), access_token=access_token, nonce=nonce)

    return sign


class AuthHeaders(dict[str, str]):
    """``Authorization`` (and ``DPoP``) headers, and the signer that makes the next proof for the same token.

    A caller that sends the request itself (the gateway's cloud door) takes ``signer`` to sign a proof for each
    attempt, because a proof is good once. It is ``None`` for a Bearer token.
    """

    signer: ProofSigner | None = None


def authorization_headers(
    token: str, key: DPoPKey | None, method: str, url: str, *, clock: Callable[[], float], nonce: str | None = None
) -> AuthHeaders:
    """``Authorization`` (and ``DPoP``) for ``token``: a proof of ``key`` when it is bound, else Bearer."""
    if key is None:
        return AuthHeaders({"Authorization": "Bearer " + token})
    sent = nonce if nonce is not None else nonces.get(url)
    out = AuthHeaders(
        {
            "Authorization": "DPoP " + token,
            "DPoP": key.proof(method, url, now=clock(), access_token=token, nonce=sent),
        }
    )
    out.signer = signer(key, clock, token)
    return out


def bound_key(response: TokenResponse, key: DPoPKey | None) -> DPoPKey | None:
    """``key`` when ``response`` is a token bound to it, ``None`` when the platform left it a Bearer token.

    A token bound to a key this process does not hold can be used by no one here, so it is refused.
    """
    if response.token_type.lower() != "dpop":
        return None
    named: object = None
    try:
        cnf = unverified_claims(response.access_token.reveal()).get("cnf")
        named = cast(Mapping[str, Any], cnf).get("jkt") if isinstance(cnf, dict) else None
    except InvalidToken:
        named = None
    if key is None or named != key.thumbprint:
        raise OAuthError(
            "dpop_binding_mismatch",
            "the token endpoint bound the token to a key this process does not hold",
            200,
        )
    return key


def is_nonce_challenge(response: HttpResponse) -> bool:
    """``True`` when the server refused for want of its nonce and sent one (RFC 9449 sections 8 and 9)."""
    if response.status not in (400, 401) or not response.header(NONCE_HEADER):
        return False
    if USE_NONCE in (response.header("WWW-Authenticate") or "").lower():
        return True
    body = response.json_object() or {}
    return body.get("error") == USE_NONCE


def is_nonce_challenge_headers(status: int, www_authenticate: str | None, nonce: str | None) -> bool:
    """:func:`is_nonce_challenge` for a response whose body is not read (an HTTP client's auth flow)."""
    return status == 401 and bool(nonce) and USE_NONCE in (www_authenticate or "").lower()


def learn_nonce(url: str, status: int, www_authenticate: str | None, nonce: str | None) -> bool:
    """Keep the nonce a resource server sent; ``True`` when it refused the request for want of one."""
    if is_nonce_challenge_headers(status, www_authenticate, nonce):
        nonces.remember(url, nonce)
        return True
    return False
