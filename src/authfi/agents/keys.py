"""An agent's key: it authenticates by signing a short assertion, and the key never leaves the process.

The algorithm follows the key: Ed25519 signs EdDSA (the default), P-256 signs ES256, RSA (2048 to
4096 bits, exponent 65537) signs PS256. Signing goes through PyJWT, the same JOSE library that
verifies, so the package has one JOSE implementation.

::

    key = AgentKey.generate()                 # EdDSA; kid = its RFC 7638 thumbprint
    key.write("/run/secrets/agent.key")       # exclusive create, readable by its owner only
    print(key.public_jwk())                   # what you register in the console

There is no method that returns the private key as a value: :meth:`AgentKey.write` is the one
way out.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
import uuid
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, NoReturn, Protocol, Self, TypeVar, cast, overload, runtime_checkable

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from jwt.algorithms import ECAlgorithm, OKPAlgorithm, RSAPSSAlgorithm, get_default_algorithms
from jwt.exceptions import InvalidKeyError
from typing_extensions import deprecated

from authfi._protocol.oauth import b64url
from authfi.agents._files import check_private, write_private
from authfi.errors import AuthFIDeprecationWarning, ConfigurationError
from authfi.secret import Secret

__all__ = ["ASSERTION_LIFETIME", "AgentKey", "KeyAlgorithm", "Signer", "assert_public_jwk", "client_assertion"]

KeyAlgorithm = Literal["EdDSA", "ES256", "PS256"]
_BY_NAME: dict[object, KeyAlgorithm] = {"EdDSA": "EdDSA", "ES256": "ES256", "PS256": "PS256"}

#: How long an assertion lives, in seconds. The platform refuses one that claims longer.
ASSERTION_LIFETIME = 300

_RSA_MIN_BITS, _RSA_MAX_BITS, _RSA_EXPONENT, _RSA_GENERATED_BITS = 2048, 4096, 65537, 3072
_PRIVATE_MEMBERS = ("d", "p", "q", "dp", "dq", "qi", "k")
_KEY_TYPES = "Ed25519 (EdDSA), P-256 (ES256) or RSA 2048-4096 bits (PS256)"

PrivateKey = ed25519.Ed25519PrivateKey | ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey


@runtime_checkable
class Signer(Protocol):
    """What signs an agent's assertions: a key file today, a cloud KMS or hardware key later."""

    @property
    def kid(self) -> str:
        """The registered key id."""
        ...

    @property
    def alg(self) -> str:
        """The JWS algorithm it signs with."""
        ...

    def sign(self, data: bytes) -> bytes:
        """The JWS signature over ``data`` (RFC 7518 §3)."""
        ...

    def public_jwk(self) -> dict[str, str]:
        """The public key, as registered."""
        ...


def _implied_alg(jwk: Mapping[str, Any]) -> KeyAlgorithm:
    kty, crv = jwk.get("kty"), jwk.get("crv")
    if kty == "OKP" and crv == "Ed25519":
        return "EdDSA"
    if kty == "EC" and crv == "P-256":
        return "ES256"
    if kty == "RSA":
        return "PS256"
    raise ConfigurationError(f"an agent key must be {_KEY_TYPES}, but this JWK is {kty}/{crv or '-'}")


def _check_label(jwk: Mapping[str, Any], implied: KeyAlgorithm) -> None:
    label = jwk.get("alg")
    if label is not None and label != implied:
        raise ConfigurationError(
            f"this {jwk.get('kty')} key is labelled alg={label}, but such an agent key signs {implied}; relabel it"
        )
    if jwk.get("use") not in (None, "sig"):
        raise ConfigurationError(f"this JWK is for use={jwk.get('use')}; an agent key must be a signing key")


def _b64int(value: str) -> int:
    return int.from_bytes(_b64decode(value), "big")


def _b64decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _check_rsa(bits: int, exponent: int) -> None:
    if not _RSA_MIN_BITS <= bits <= _RSA_MAX_BITS:
        raise ConfigurationError(
            f"this RSA key is {bits} bits; an agent key must be {_RSA_MIN_BITS} to {_RSA_MAX_BITS}"
        )
    if exponent != _RSA_EXPONENT:
        raise ConfigurationError(f"this RSA key's exponent is {exponent}; an agent key must use 65537")


JWK = TypeVar("JWK", bound=Mapping[str, Any])


def assert_public_jwk(jwk: JWK) -> JWK:
    """``jwk`` when it is a public agent key the platform accepts; else :class:`ConfigurationError`.

    Call it before sending a JWK anywhere. A JWK carrying a private member (``d``, ``p``...) is not
    a formatting slip: the key has left where it belongs and must be replaced.
    """
    if not isinstance(jwk, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ConfigurationError("a JWK is a JSON object (RFC 7517)")
    leaked = [m for m in _PRIVATE_MEMBERS if m in jwk]
    if leaked:
        raise ConfigurationError(
            f"this JWK carries private member(s) {', '.join(repr(m) for m in leaked)} where a public key belongs. "
            "Treat the key as burned: generate a new one (AgentKey.generate) and register its public half."
        )
    implied = _implied_alg(jwk)
    _check_label(jwk, implied)
    required = {"EdDSA": ("x",), "ES256": ("x", "y"), "PS256": ("n", "e")}[implied]
    for member in required:
        if not jwk.get(member):
            raise ConfigurationError(f"this JWK has no {member!r}: there is no public key in it")
    if implied == "PS256":
        _check_rsa(_b64int(jwk["n"]).bit_length(), _b64int(jwk["e"]))
    return jwk


def _alg_of(key: object) -> KeyAlgorithm:
    if isinstance(key, ed25519.Ed25519PrivateKey):
        return "EdDSA"
    if isinstance(key, ec.EllipticCurvePrivateKey):
        if not isinstance(key.curve, ec.SECP256R1):
            raise ConfigurationError(f"the agent key is on curve {key.curve.name}; agent keys are {_KEY_TYPES}")
        return "ES256"
    if isinstance(key, rsa.RSAPrivateKey):
        _check_rsa(key.key_size, key.public_key().public_numbers().e)
        return "PS256"
    raise ConfigurationError(f"the agent key is a {type(key).__name__}; agent keys are {_KEY_TYPES}")


def _public_jwk(key: PrivateKey, alg: KeyAlgorithm) -> dict[str, str]:
    algorithm: Any
    algorithm = {"EdDSA": OKPAlgorithm, "ES256": ECAlgorithm, "PS256": RSAPSSAlgorithm}[alg]
    raw = json.loads(algorithm.to_jwk(key.public_key()))
    out = {k: str(v) for k, v in cast(dict[str, Any], raw).items() if k in ("kty", "crv", "x", "y", "n", "e")}
    return out


def _thumbprint(jwk: Mapping[str, str]) -> str:
    members = {"OKP": ("crv", "kty", "x"), "EC": ("crv", "kty", "x", "y"), "RSA": ("e", "kty", "n")}[jwk["kty"]]
    canonical = json.dumps({m: jwk[m] for m in members}, separators=(",", ":"), sort_keys=True)
    return b64url(hashlib.sha256(canonical.encode("utf-8")).digest())


def client_assertion(
    signer: Signer, client_id: str, audience: str, *, now: float, lifetime: int = ASSERTION_LIFETIME
) -> str:
    """An RFC 7523 §2.2 client assertion: ``iss`` = ``sub`` = the client, a fresh ``jti``, at most 300 s.

    ``audience`` is the endpoint that checks it, so an assertion signed for one endpoint is refused
    at another.
    """
    if not client_id or not audience:
        raise ConfigurationError("a client assertion needs the client id and the audience")
    if not 0 < lifetime <= ASSERTION_LIFETIME:
        raise ConfigurationError(f"an assertion lives at most {ASSERTION_LIFETIME} seconds")
    issued = int(now)
    header = {"alg": signer.alg, "typ": "JWT", "kid": signer.kid}
    claims = {
        "iss": client_id,
        "sub": client_id,
        "aud": audience,
        "jti": str(uuid.uuid4()),
        "iat": issued,
        "exp": issued + lifetime,
    }

    def segment(value: Mapping[str, Any]) -> str:
        return b64url(json.dumps(value, separators=(",", ":")).encode("utf-8"))

    signing_input = f"{segment(header)}.{segment(claims)}"
    return f"{signing_input}.{b64url(signer.sign(signing_input.encode('ascii')))}"


_PEM_MARKERS = (b"BEGIN PRIVATE KEY", b"BEGIN EC PRIVATE KEY", b"BEGIN RSA PRIVATE KEY")


class AgentKey:
    """An agent's private key, held in this process and never sent anywhere.

    Make one with :meth:`generate`, :meth:`from_file`, :meth:`from_pem` or :meth:`from_jwk`. It
    prints as ``AgentKey(kid=..., alg=...)`` and does not pickle.
    """

    __slots__ = ("_alg", "_key", "_kid", "path")

    def __init__(self, private_key: PrivateKey, key_id: str) -> None:
        kid = str(key_id or "").strip()
        if not kid:
            raise ConfigurationError(
                "kid is required: it is the key id registered with the agent's public key, and the platform "
                "checks the assertion header against it"
            )
        self._alg: KeyAlgorithm = _alg_of(private_key)
        self._key: PrivateKey = private_key
        self._kid = kid
        #: The file the key was read from, when it was: a helper process is pointed at the file,
        #: never handed the key.
        self.path: Path | None = None

    # --- making one ------------------------------------------------------------------------------

    @overload
    @classmethod
    def generate(cls, alg: KeyAlgorithm = "EdDSA", *, kid: str | None = None) -> Self: ...

    @overload
    @classmethod
    @deprecated("AgentKey.generate(key_id) is AgentKey.generate(kid=key_id)", category=AuthFIDeprecationWarning)
    def generate(cls, alg: str, *, kid: None = None) -> Self: ...

    @classmethod
    def generate(cls, alg: str = "EdDSA", *, kid: str | None = None) -> Self:
        """A new key pair. ``kid`` defaults to the public key's RFC 7638 thumbprint; RSA is 3072 bits."""
        chosen = _BY_NAME.get(cast(object, alg))
        if chosen is None:  # 0.1 passed the key id first
            warnings.warn(
                "AgentKey.generate(key_id) is AgentKey.generate(kid=key_id)", AuthFIDeprecationWarning, stacklevel=2
            )
            if not alg:
                raise ConfigurationError("key_id is required: it is the kid the agent's public key is registered with")
            chosen, kid = "EdDSA", str(alg)
        key: PrivateKey
        if chosen == "EdDSA":
            key = ed25519.Ed25519PrivateKey.generate()
        elif chosen == "ES256":
            key = ec.generate_private_key(ec.SECP256R1())
        else:
            key = rsa.generate_private_key(public_exponent=_RSA_EXPONENT, key_size=_RSA_GENERATED_BITS)
        return cls(key, kid or _thumbprint(_public_jwk(key, chosen)))

    @classmethod
    def from_file(cls, path: str | os.PathLike[str], *, kid: str | None = None, key_id: str | None = None) -> Self:
        """Load a PEM or JWK private key from a file only its owner can read.

        A PEM file needs ``kid``; a JWK file supplies its own. ``key_id`` is 0.1's name for ``kid``.
        """
        kid = _kid_alias(kid, key_id)
        file = Path(path)
        check_private(file)
        data = file.read_bytes()
        if not data.strip():
            raise ConfigurationError(f"the key file {file} is empty")
        key = (
            cls.from_jwk(json.loads(data.decode("utf-8")), kid=kid)
            if data.lstrip()[:1] == b"{"
            else cls.from_pem(data, kid=kid or "")
        )
        key.path = file.resolve()
        return key

    @overload
    @classmethod
    def from_pem(cls, pem: str | bytes | Secret, *, kid: str) -> Self: ...

    @overload
    @classmethod
    @deprecated(
        "AgentKey.from_pem(pem, key_id) is AgentKey.from_pem(pem, kid=key_id)", category=AuthFIDeprecationWarning
    )
    def from_pem(cls, pem: str | bytes | Secret, key_id: str) -> Self: ...

    @classmethod
    def from_pem(cls, pem: str | bytes | Secret, key_id: str | None = None, *, kid: str | None = None) -> Self:
        """Load an unencrypted private key from PEM (PKCS#8, SEC1 or PKCS#1)."""
        if key_id is not None:
            warnings.warn(
                "AgentKey.from_pem(pem, key_id) is AgentKey.from_pem(pem, kid=key_id)",
                AuthFIDeprecationWarning,
                stacklevel=2,
            )
        kid = kid or key_id
        raw = (
            pem.reveal().encode("utf-8")
            if isinstance(pem, Secret)
            else pem.encode("utf-8")
            if isinstance(pem, str)
            else pem
        )
        if not any(marker in raw for marker in _PEM_MARKERS):
            first = raw.strip().splitlines()[0].decode("utf-8", "replace") if raw.strip() else ""
            what = f"it is a {first.strip('-').strip()}" if first.startswith("-----BEGIN") else "it is not PEM"
            raise ConfigurationError(
                f"the agent key must be a private key in PEM form, but {what}. Make one with: authfi keygen --out "
                "agent.key (or openssl genpkey -algorithm ed25519)"
            )
        try:
            key = serialization.load_pem_private_key(raw, password=None)
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(
                "the agent key is not a usable private key; an encrypted key must be decrypted before it is mounted"
            ) from exc
        return cls(cast(PrivateKey, key), kid or "")

    @classmethod
    def from_jwk(
        cls, jwk: Mapping[str, Any] | str | bytes, *, kid: str | None = None, key_id: str | None = None
    ) -> Self:
        """Load a private key from a JWK: the one shape where ``d`` belongs."""
        kid = _kid_alias(kid, key_id)
        loaded: Any = json.loads(jwk) if isinstance(jwk, str | bytes) else jwk
        if not isinstance(loaded, Mapping):
            raise ConfigurationError("a JWK is a JSON object (RFC 7517)")
        data = cast(Mapping[str, Any], loaded)
        implied = _implied_alg(data)
        _check_label(data, implied)
        if not data.get("d"):
            raise ConfigurationError("this JWK has no 'd': it is the public half, and signing needs the private key")
        algorithm: Any = {"EdDSA": OKPAlgorithm, "ES256": ECAlgorithm, "PS256": RSAPSSAlgorithm}[implied]
        try:
            # With "d" present, PyJWT loads the private key of the type _implied_alg allowed.
            key = cast(PrivateKey, algorithm.from_jwk(json.dumps(dict(data))))
        except (InvalidKeyError, ValueError, KeyError, TypeError) as exc:
            raise ConfigurationError(f"this {data.get('kty')} JWK is not a usable private key") from exc
        return cls(key, kid or str(data.get("kid") or ""))

    # --- using it ----------------------------------------------------------------------------------

    @property
    def kid(self) -> str:
        """The registered key id."""
        return self._kid

    @property
    @deprecated("AgentKey.key_id is AgentKey.kid", category=AuthFIDeprecationWarning)
    def key_id(self) -> str:
        """0.1's name for :attr:`kid`."""
        return self._kid

    @property
    def alg(self) -> str:
        """The algorithm this key signs with: ``EdDSA``, ``ES256`` or ``PS256``."""
        return self._alg

    def sign(self, data: bytes) -> bytes:
        """The JWS signature over ``data``: ES256 as R||S, PS256 with a 32-byte salt (RFC 7518 §3)."""
        algorithm = get_default_algorithms()[self._alg]
        return algorithm.sign(data, self._key)  # type: ignore[arg-type,unused-ignore]

    def public_jwk(self) -> dict[str, str]:
        """The public JWK to register: the only half that ever leaves here."""
        jwk = {**_public_jwk(self._key, self._alg), "alg": self._alg, "kid": self._kid, "use": "sig"}
        assert_public_jwk(jwk)
        return jwk

    def thumbprint(self) -> str:
        """The RFC 7638 thumbprint of the public key."""
        return _thumbprint(_public_jwk(self._key, self._alg))

    def write(self, path: str | os.PathLike[str]) -> None:
        """Write the private key to a new file (PKCS#8 PEM) that only its owner can read."""
        pem = self._key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        write_private(Path(path), pem)
        self.path = Path(path).resolve()

    @deprecated(
        "AgentKey.private_pem() returns the private key as a value; write it with AgentKey.write(path)",
        category=AuthFIDeprecationWarning,
    )
    def private_pem(self) -> str:
        """0.1's way to get the private key as PEM. Use :meth:`write`."""
        return self._key.private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        ).decode("ascii")

    @deprecated(
        "AgentKey.client_assertion is internal to the agent client, and returns a Secret from 1.0",
        category=AuthFIDeprecationWarning,
    )
    def client_assertion(self, agent_id: str, audience: str, lifetime: int = ASSERTION_LIFETIME) -> str:
        """A fresh client assertion signed by this key (see :func:`client_assertion`)."""
        return client_assertion(self, agent_id, audience, now=time.time(), lifetime=lifetime)

    def __repr__(self) -> str:
        return f"AgentKey(kid={self._kid!r}, alg={self._alg!r})"

    def __reduce__(self) -> NoReturn:
        raise TypeError("An AgentKey cannot be pickled: the private key stays in the process that loaded it")


def _kid_alias(kid: str | None, key_id: str | None) -> str | None:
    if key_id is not None:
        warnings.warn("key_id= is kid=", AuthFIDeprecationWarning, stacklevel=3)
        return kid or key_id
    return kid
