"""The project's signing keys (its JWKS), filtered when loaded. No I/O.

A key is used only when:

- ``use`` is ``sig`` or absent;
- it carries no private member (``d``, ``p``, ``q``, ``dp``, ``dq``, ``qi``, ``k``): a published
  private key is burned, and no symmetric key verifies anything here;
- its ``alg`` agrees with its ``kty`` and ``crv`` (or is derived from them when absent), and is one
  of ``RS256``, ``ES256``, ``EdDSA``;
- an RSA key is at least 2048 bits.

A key that fails is left out and logged; the rest of the set still works.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

from authfi._protocol.http import DOCUMENT_LIMIT, Flow, HttpRequest, Retry, call
from authfi.errors import DiscoveryError

__all__ = ["ACCEPTED_ALGS", "KeySet", "PublicKey", "alg_for_jwk", "jwks_flow", "parse_jwks"]

#: The algorithms an AuthFI access token may be signed with.
ACCEPTED_ALGS = frozenset({"RS256", "ES256", "EdDSA"})

_PRIVATE_MEMBERS = ("d", "p", "q", "dp", "dq", "qi", "k")
_MIN_RSA_BITS = 2048

_log = logging.getLogger("authfi")


def alg_for_jwk(jwk: Mapping[str, Any]) -> str | None:
    """The one algorithm a public JWK verifies, from its type and curve; ``None`` if none is accepted."""
    kty, crv = jwk.get("kty"), jwk.get("crv")
    if kty == "RSA":
        return "RS256"
    if kty == "EC" and crv == "P-256":
        return "ES256"
    if kty == "OKP" and crv == "Ed25519":
        return "EdDSA"
    return None


@dataclass(frozen=True, slots=True)
class PublicKey:
    """One usable verification key."""

    kid: str
    alg: str
    key: Any = field(repr=False)


@dataclass(frozen=True, slots=True)
class KeySet:
    """The usable keys of one issuer, by ``kid``."""

    keys: Mapping[str, PublicKey] = field(default_factory=lambda: types.MappingProxyType({}))

    def get(self, kid: str) -> PublicKey | None:
        """The key named ``kid``, or ``None``."""
        return self.keys.get(kid)

    def __len__(self) -> int:
        return len(self.keys)


def _load(jwk: Mapping[str, Any], where: str) -> PublicKey | None:
    kid = jwk.get("kid")
    if not isinstance(kid, str) or not kid:
        _log.warning("ignoring a key with no kid in %s", where)
        return None
    if jwk.get("use") not in (None, "sig"):
        return None
    leaked = [m for m in _PRIVATE_MEMBERS if m in jwk]
    if leaked:
        _log.error("refusing key %s in %s: it carries private member(s) %s", kid, where, ", ".join(leaked))
        return None
    implied = alg_for_jwk(jwk)
    label = jwk.get("alg")
    if implied is None or (label is not None and label != implied) or implied not in ACCEPTED_ALGS:
        _log.warning(
            "ignoring key %s in %s: %s/%s with alg %r is not accepted",
            kid,
            where,
            jwk.get("kty"),
            jwk.get("crv"),
            label,
        )
        return None
    try:
        loaded = jwt.PyJWK(dict(jwk), algorithm=implied)
    except (jwt.PyJWKError, jwt.InvalidKeyError, ValueError, TypeError) as exc:
        _log.warning("ignoring key %s in %s: %s", kid, where, exc)
        return None
    key = loaded.key
    if isinstance(key, rsa.RSAPublicKey) and key.key_size < _MIN_RSA_BITS:
        _log.warning("ignoring key %s in %s: RSA %d bits is below %d", kid, where, key.key_size, _MIN_RSA_BITS)
        return None
    return PublicKey(kid=kid, alg=implied, key=key)


def parse_jwks(doc: object, where: str) -> KeySet:
    """The usable keys in a JWKS document."""
    if not isinstance(doc, dict) or not isinstance(doc.get("keys"), list):  # pyright: ignore[reportUnknownMemberType]
        raise DiscoveryError(f"The key set at {where} is not a JWKS document.", code="jwks_invalid")
    entries = cast(list[object], doc["keys"])
    keys: dict[str, PublicKey] = {}
    for entry in entries:
        if isinstance(entry, dict):
            jwk: dict[str, Any] = entry  # pyright: ignore[reportUnknownVariableType]
            loaded = _load(jwk, where)
            if loaded is not None:
                keys[loaded.kid] = loaded
    return KeySet(types.MappingProxyType(keys))


def jwks_flow(jwks_uri: str) -> Flow[tuple[KeySet, str | None]]:
    """Fetch the key set; returns it with the answer's ``Cache-Control``."""
    response = yield from call(
        HttpRequest(
            "GET",
            jwks_uri,
            what="the key set",
            headers={"Accept": "application/json"},
            retry=Retry.SAFE,
            max_bytes=DOCUMENT_LIMIT,
        )
    )
    if response.status != 200:
        raise DiscoveryError(f"The key set at {jwks_uri} answered HTTP {response.status}.", status=response.status)
    try:
        doc = response.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise DiscoveryError(f"The key set at {jwks_uri} is not JSON.", code="jwks_invalid") from exc
    return parse_jwks(doc, jwks_uri), response.header("Cache-Control")
