"""DPoP (RFC 9449) proofs, for the sender-constrained tokens mandates bring. Internal until then.

- One ES256 run key per process, held in memory and never serialised; a forked child makes its
  own (a key two processes share is not bound to either).
- A fresh proof for every attempt: a new ``jti`` and ``iat``, and the origin's current nonce.
- ``htu`` is the request URL without its query or fragment; ``ath`` binds the access token.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from typing import Any

import jwt
from cryptography.hazmat.primitives.asymmetric import ec

from authfi._protocol.oauth import b64url
from authfi._protocol.urls import origin_of

__all__ = ["DPoPKey", "NonceCache", "htu", "run_key"]


def htu(url: str) -> str:
    """The ``htu`` of ``url``: no query, no fragment."""
    return url.split("#", 1)[0].split("?", 1)[0]


def _int_b64(value: int, length: int) -> str:
    return b64url(value.to_bytes(length, "big"))


class DPoPKey:
    """An ES256 key that signs DPoP proofs. The private half has no accessor and does not pickle."""

    __slots__ = ("_key", "_public_jwk", "thumbprint")

    def __init__(self) -> None:
        self._key = ec.generate_private_key(ec.SECP256R1())
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
        return jwt.encode(claims, self._key, algorithm="ES256", headers={"typ": "dpop+jwt", "jwk": self.public_jwk()})

    def __repr__(self) -> str:
        return f"DPoPKey(thumbprint={self.thumbprint!r})"

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


_key: DPoPKey | None = None
_key_lock = threading.Lock()
nonces = NonceCache()


def run_key() -> DPoPKey:
    """This process's DPoP key, made on first use."""
    global _key
    with _key_lock:
        if _key is None:
            _key = DPoPKey()
        return _key


def _after_fork_in_child() -> None:
    global _key, _key_lock, nonces
    _key = None
    _key_lock = threading.Lock()
    nonces = NonceCache()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
