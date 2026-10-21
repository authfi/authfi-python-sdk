"""People signing in: the authorize URL, the sealed pending sign-in, the callback and the ID token.

Pure functions with no I/O; :mod:`authfi.signin` drives them for blocking and asyncio code.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import parse_qsl, urlencode, urlsplit

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from authfi._protocol.jwks import ACCEPTED_ALGS, PublicKey
from authfi._protocol.oauth import b64url, random_value
from authfi._protocol.verify import Unverified, audiences_of, verify_signature
from authfi.errors import (
    ConfigurationError,
    ConsentDenied,
    InvalidCallback,
    InvalidIdToken,
    InvalidPendingSignIn,
    InvalidToken,
    NotSupportedError,
    OAuthError,
    StepUpRequired,
)
from authfi.secret import Secret

__all__ = [
    "PENDING_LIFETIME",
    "IdTokenClaims",
    "PendingState",
    "authorize_url",
    "check_redirect_uri",
    "end_session_url",
    "inspect_id_token",
    "read_callback",
    "seal",
    "state_key_bytes",
    "unseal",
    "verify_id_token",
]

#: Seconds a pending sign-in may wait for its callback.
PENDING_LIFETIME = 600

_BLOB_PREFIX = "v1."
_PROMPTS = ("login", "consent")
#: The hash at_hash is made with, per signing algorithm (OpenID Connect Core 3.1.3.6; EdDSA: SHA-512).
_AT_HASH = {"RS256": hashlib.sha256, "ES256": hashlib.sha256, "EdDSA": hashlib.sha512}

STATE_KEY_HELP = (
    "make one with: python -c "
    '"import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"'
)


def state_key_bytes(state_key: Secret) -> bytes:
    """The 32 bytes ``state_key`` encodes (base64, either alphabet), or :class:`ConfigurationError`."""
    if not isinstance(cast(object, state_key), Secret):
        raise ConfigurationError("state_key must be a Secret holding 32 random bytes in base64; " + STATE_KEY_HELP)
    text = state_key.reveal().strip()
    try:
        raw = base64.urlsafe_b64decode(text.replace("+", "-").replace("/", "_") + "=" * (-len(text) % 4))
    except (ValueError, binascii.Error) as exc:
        raise ConfigurationError("state_key is not base64; " + STATE_KEY_HELP) from exc
    if len(raw) != 32:
        raise ConfigurationError(f"state_key holds {len(raw)} bytes, not 32; " + STATE_KEY_HELP)
    return raw


def check_redirect_uri(redirect_uri: str) -> str:
    """``redirect_uri`` if it is an absolute https URL (http only on loopback) with no fragment."""
    parts = urlsplit(redirect_uri)
    loopback = parts.hostname in ("localhost", "127.0.0.1", "::1")
    if parts.scheme != "https" and not (parts.scheme == "http" and loopback):
        raise ConfigurationError(f"redirect_uri must be https (http only on loopback): {redirect_uri!r}")
    if not parts.netloc or parts.fragment:
        raise ConfigurationError(f"redirect_uri must be an absolute URL with no fragment: {redirect_uri!r}")
    return redirect_uri


@dataclass(frozen=True, slots=True, repr=False)
class PendingState:
    """What a sign-in must remember until its callback: never readable by the browser."""

    state: str
    nonce: Secret
    verifier: Secret
    created_at: float
    acr_values: tuple[str, ...] = ()
    max_age: int | None = None

    def __repr__(self) -> str:
        return f"PendingState(acr_values={self.acr_values!r}, max_age={self.max_age!r})"


def new_pending(now: float, acr_values: Sequence[str], max_age: int | None) -> PendingState:
    """A fresh state, nonce and PKCE verifier (32 random bytes each)."""
    return PendingState(
        state=random_value(),
        nonce=Secret(random_value()),
        verifier=Secret(random_value()),
        created_at=now,
        acr_values=tuple(acr_values),
        max_age=max_age,
    )


def _aad(client_id: str, redirect_uri: str) -> bytes:
    return f"{client_id}\n{redirect_uri}".encode()


def seal(pending: PendingState, key: bytes, client_id: str, redirect_uri: str) -> str:
    """``pending`` as an opaque string: AES-256-GCM, bound to the client and its redirect URI."""
    body = json.dumps(
        {
            "s": pending.state,
            "n": pending.nonce.reveal(),
            "v": pending.verifier.reveal(),
            "t": pending.created_at,
            "a": list(pending.acr_values),
            "m": pending.max_age,
        },
        separators=(",", ":"),
    ).encode()
    iv = os.urandom(12)
    sealed = AESGCM(key).encrypt(iv, body, _aad(client_id, redirect_uri))
    return _BLOB_PREFIX + b64url(iv + sealed)


def unseal(blob: str, key: bytes, client_id: str, redirect_uri: str, now: float) -> PendingState:
    """The pending sign-in in ``blob``, or :class:`InvalidPendingSignIn` (tampered, another client's, expired)."""
    if not isinstance(cast(object, blob), str) or not blob.startswith(_BLOB_PREFIX):
        raise InvalidPendingSignIn("pending_sign_in_malformed", "not a pending sign-in from this SDK")
    text = blob[len(_BLOB_PREFIX) :]
    try:
        raw = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
        body = AESGCM(key).decrypt(raw[:12], raw[12:], _aad(client_id, redirect_uri))
        data: object = json.loads(body)
    except (ValueError, binascii.Error, InvalidTag) as exc:
        raise InvalidPendingSignIn(
            "pending_sign_in_tampered", "it does not decrypt for this client, redirect URI and state_key"
        ) from exc
    if not isinstance(data, dict):
        raise InvalidPendingSignIn("pending_sign_in_malformed", "its content is not an object")
    fields = cast(dict[str, Any], data)
    created, state, nonce, verifier = fields.get("t"), fields.get("s"), fields.get("n"), fields.get("v")
    if not isinstance(created, int | float) or not all(isinstance(v, str) and v for v in (state, nonce, verifier)):
        raise InvalidPendingSignIn("pending_sign_in_malformed", "a field is missing")
    if now - created > PENDING_LIFETIME:
        raise InvalidPendingSignIn("pending_sign_in_expired", f"it is older than {PENDING_LIFETIME} seconds")
    acr = fields.get("a")
    max_age = fields.get("m")
    return PendingState(
        state=cast(str, state),
        nonce=Secret(cast(str, nonce)),
        verifier=Secret(cast(str, verifier)),
        created_at=float(created),
        acr_values=tuple(a for a in cast(list[object], acr) if isinstance(a, str)) if isinstance(acr, list) else (),
        max_age=max_age if isinstance(max_age, int) else None,
    )


def authorize_url(
    authorization_endpoint: str,
    *,
    client_id: str,
    redirect_uri: str,
    pending: PendingState,
    scope: Sequence[str],
    login_hint: str | None = None,
    prompt: str | None = None,
) -> str:
    """The URL that starts the sign-in: code flow with PKCE S256, ``state`` and ``nonce``."""
    if isinstance(cast(object, scope), str):
        raise ConfigurationError("scope is a sequence of names, such as ('openid', 'profile'), not one string")
    names = [s.strip() for s in scope if s.strip()]
    if "openid" not in names:
        names.insert(0, "openid")
    if prompt is not None and prompt not in _PROMPTS:
        if prompt == "none":
            raise NotSupportedError(
                "prompt_none", "AuthFI does not offer prompt=none; sign the person in with a redirect."
            )
        raise ConfigurationError(f"prompt must be one of {_PROMPTS}, not {prompt!r}")
    if pending.max_age is not None and pending.max_age < 0:
        raise ConfigurationError("max_age is a number of seconds, 0 or more")
    challenge = b64url(hashlib.sha256(pending.verifier.reveal().encode("ascii")).digest())
    query: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(names),
        "state": pending.state,
        "nonce": pending.nonce.reveal(),
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if pending.acr_values:
        query["acr_values"] = " ".join(pending.acr_values)
    if pending.max_age is not None:
        query["max_age"] = str(pending.max_age)
    if login_hint:
        query["login_hint"] = login_hint
    if prompt:
        query["prompt"] = prompt
    separator = "&" if urlsplit(authorization_endpoint).query else "?"
    return authorization_endpoint + separator + urlencode(query)


#: Parameters of the callback that must appear at most once (RFC 6749 §3.1, RFC 9207 §2.4).
_SINGLE = ("iss", "state", "code", "error", "error_description")


def _pairs(callback: Mapping[str, str] | str) -> list[tuple[str, str]]:
    """Every name and value the callback carries, repeats included where its type keeps them."""
    if isinstance(callback, str):
        return parse_qsl(urlsplit(callback).query, keep_blank_values=True)
    multi_items = getattr(callback, "multi_items", None)  # Starlette's and FastAPI's query parameters
    if callable(multi_items):
        return [(str(k), str(v)) for k, v in cast("Iterable[tuple[object, object]]", multi_items())]
    lists = getattr(callback, "lists", None)  # Werkzeug's and Flask's MultiDict
    if callable(lists):
        return [
            (str(k), str(v)) for k, values in cast("Iterable[tuple[object, Iterable[object]]]", lists()) for v in values
        ]
    return [(str(k), str(v)) for k, v in callback.items()]


def _callback_params(callback: Mapping[str, str] | str) -> dict[str, str]:
    pairs = _pairs(callback)
    seen: set[str] = set()
    for name, _ in pairs:
        if name in _SINGLE and name in seen:
            raise InvalidCallback(
                "callback_parameter_repeated",
                f"the callback carries {name} more than once; which one is meant is unknown",
            )
        seen.add(name)
    return dict(pairs)


def read_callback(callback: Mapping[str, str] | str, *, issuer: str, state: str) -> str:
    """The authorization code from the callback, after its ``iss`` (RFC 9207) and ``state`` are checked."""
    params = _callback_params(callback)
    iss = params.get("iss")
    if not iss:
        raise InvalidCallback("callback_iss_missing", "the callback names no iss (RFC 9207); AuthFI always sends it")
    if iss != issuer:
        raise InvalidCallback("callback_iss_mismatch", f"the callback comes from {iss!r}, not {issuer!r}")
    if not hmac.compare_digest(params.get("state", "").encode(), state.encode()):
        raise InvalidCallback("callback_state_mismatch", "the callback's state is not this sign-in's")
    error = params.get("error")
    if error:
        description = params.get("error_description") or None
        if error == "access_denied":
            raise ConsentDenied(error, description)
        raise OAuthError(error, description)
    code = params.get("code")
    if not code:
        raise InvalidCallback("callback_code_missing", "the callback carries no code")
    return code


@dataclass(frozen=True, slots=True)
class IdTokenClaims:
    """Who signed in, from the verified ID token. ``raw`` holds every claim."""

    sub: str
    auth_time: int | None
    acr: str | None
    amr: tuple[str, ...]
    email: str | None
    email_verified: bool | None
    name: str | None
    groups: tuple[str, ...]
    roles: tuple[str, ...]
    sid: str | None
    raw: Mapping[str, Any]

    @classmethod
    def of(cls, claims: Mapping[str, Any]) -> IdTokenClaims:
        """The typed view of a verified ID token's claims."""

        def text(name: str) -> str | None:
            value = claims.get(name)
            return value if isinstance(value, str) and value else None

        def texts(name: str) -> tuple[str, ...]:
            value = claims.get(name)
            if isinstance(value, list):
                return tuple(v for v in cast(list[object], value) if isinstance(v, str))
            return ()

        auth_time = claims.get("auth_time")
        verified = claims.get("email_verified")
        return cls(
            sub=str(claims["sub"]),
            auth_time=int(auth_time) if isinstance(auth_time, int | float) else None,
            acr=text("acr"),
            amr=texts("amr"),
            email=text("email"),
            email_verified=verified if isinstance(verified, bool) else None,
            name=text("name"),
            groups=texts("groups"),
            roles=texts("roles"),
            sid=text("sid"),
            raw=dict(claims),
        )


def inspect_id_token(raw: str, allowed_algs: Sequence[str]) -> Unverified:
    """Check the ID token's header: an alg discovery names and this SDK verifies, ``typ`` JWT, a ``kid``."""
    parts = raw.split(".") if isinstance(cast(object, raw), str) else []
    if len(parts) != 3 or not parts[0] or not parts[1]:
        raise InvalidIdToken("id_token_malformed", "not a compact JWS of three parts")
    try:
        padded = parts[0] + "=" * (-len(parts[0]) % 4)
        header_value: object = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
    except (ValueError, binascii.Error, UnicodeDecodeError) as exc:
        raise InvalidIdToken("id_token_malformed", "its header is not base64url JSON") from exc
    if not isinstance(header_value, dict):
        raise InvalidIdToken("id_token_malformed", "its header is not a JSON object")
    header = cast(dict[str, Any], header_value)
    alg = header.get("alg")
    allowed = [a for a in (allowed_algs or ("RS256",)) if a in ACCEPTED_ALGS]
    if not isinstance(alg, str) or alg not in allowed:
        raise InvalidIdToken("id_token_alg_refused", f"alg {alg!r} is not one of {allowed}")
    if header.get("typ") != "JWT":
        raise InvalidIdToken("id_token_not_id_token", f"typ {header.get('typ')!r} is not JWT")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise InvalidIdToken("id_token_key_unknown", "the token names no kid")
    return Unverified(raw=raw, header=header, kid=kid, alg=alg)


def _at_hash(alg: str, access_token: Secret) -> str:
    digest = _AT_HASH[alg](access_token.reveal().encode("ascii")).digest()
    return b64url(digest[: len(digest) // 2])


def verify_id_token(
    token: Unverified,
    key: PublicKey | None,
    *,
    issuer: str,
    client_id: str,
    now: float,
    leeway: float,
    nonce: Secret | None,
    access_token: Secret | None = None,
    max_age: int | None = None,
    acr_values: Sequence[str] = (),
    subject: str | None = None,
) -> IdTokenClaims:
    """The verified ID token (OpenID Connect Core 3.1.3.7), or :class:`InvalidIdToken`.

    ``nonce`` is the sign-in's (``None`` on a refresh, which carries none). ``subject``, on a refresh,
    is who signed in: a refreshed ID token must name the same person. An ``acr`` that does not meet
    ``acr_values`` raises :class:`StepUpRequired`.
    """
    try:
        claims = verify_signature(token, key)
    except InvalidToken as exc:
        raise InvalidIdToken(str(exc.code).replace("token_", "id_token_", 1), "its signature does not verify") from exc
    if claims.get("iss") != issuer:
        raise InvalidIdToken("id_token_issuer_mismatch", f"iss {claims.get('iss')!r} is not {issuer!r}")
    audiences = audiences_of(claims)
    if client_id not in audiences:
        raise InvalidIdToken("id_token_audience_mismatch", f"it is not for client {client_id!r}")
    azp = claims.get("azp")
    if (len(audiences) > 1 or azp is not None) and azp != client_id:
        raise InvalidIdToken("id_token_audience_mismatch", f"azp {azp!r} is not {client_id!r}")
    sub = claims.get("sub")
    if not isinstance(sub, str) or not sub:
        raise InvalidIdToken("id_token_claims_invalid", "it names no sub")
    if subject is not None and sub != subject:
        raise InvalidIdToken("id_token_subject_changed", "a refreshed ID token names another person")
    exp, iat = claims.get("exp"), claims.get("iat")
    if not isinstance(exp, int | float) or not isinstance(iat, int | float):
        raise InvalidIdToken("id_token_claims_invalid", "exp and iat are required")
    if exp + leeway <= now:
        raise InvalidIdToken("id_token_expired", "it has expired")
    if iat - leeway > now:
        raise InvalidIdToken("id_token_claims_invalid", "it was issued in the future")
    if nonce is not None:
        given = claims.get("nonce")
        if not isinstance(given, str) or not hmac.compare_digest(given.encode(), nonce.reveal().encode()):
            raise InvalidIdToken("id_token_nonce_mismatch", "its nonce is not this sign-in's")
    at_hash = claims.get("at_hash")
    if at_hash is not None and access_token is not None:
        if not isinstance(at_hash, str) or not hmac.compare_digest(at_hash, _at_hash(token.alg, access_token)):
            raise InvalidIdToken("id_token_at_hash_mismatch", "it was not issued with this access token")
    signed_in = IdTokenClaims.of(claims)
    if max_age is not None:
        if signed_in.auth_time is None:
            raise InvalidIdToken("id_token_claims_invalid", "max_age was asked, and it names no auth_time")
        if signed_in.auth_time + max_age + leeway < now:
            raise StepUpRequired(
                "the person last authenticated longer ago than max_age",
                acr_values=tuple(acr_values),
                max_age=max_age,
            )
    if acr_values and signed_in.acr not in acr_values:
        raise StepUpRequired(
            f"the sign-in met acr {signed_in.acr!r}, not one of {list(acr_values)}", acr_values=acr_values
        )
    return signed_in


def end_session_url(
    end_session_endpoint: str,
    *,
    client_id: str,
    post_logout_redirect_uri: str | None,
    state: str | None,
) -> str:
    """The end-session URL to send the browser to, naming the client."""
    query = {"client_id": client_id}
    if post_logout_redirect_uri:
        query["post_logout_redirect_uri"] = post_logout_redirect_uri
    if state:
        query["state"] = state
    separator = "&" if urlsplit(end_session_endpoint).query else "?"
    return end_session_endpoint + separator + urlencode(query)


Prompt = Literal["login", "consent"]
