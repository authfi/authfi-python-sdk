"""Checking an incoming DPoP proof, one refusal at a time."""

from __future__ import annotations

import json
import time
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec, ed25519
from jwt.algorithms import ECAlgorithm, OKPAlgorithm

from authfi._dpop import DPoPKey
from authfi._dpop_check import ProofReplayCache, check_proof
from authfi.errors import InvalidToken

URL = "https://graphs.example.com/threads"
TOKEN = "an-access-token"


def check(proof: str | None, key: DPoPKey, *, jkt: str | None = "same", now: float | None = None) -> str:
    return check_proof(
        proof,
        method="POST",
        url=URL,
        access_token=TOKEN,
        jkt=key.thumbprint if jkt == "same" else jkt,
        now=now or time.time(),
        replay=ProofReplayCache(),
    )


def good(key: DPoPKey) -> str:
    return key.proof("POST", URL, now=time.time(), access_token=TOKEN)


def code_of(call: Any) -> str:
    with pytest.raises(InvalidToken) as caught:
        call()
    return str(caught.value.code)


def test_a_good_proof_returns_the_thumbprint_of_its_key() -> None:
    key = DPoPKey()
    assert check(good(key), key) == key.thumbprint


def test_each_wrong_proof_has_its_own_code() -> None:
    key = DPoPKey()
    assert code_of(lambda: check(None, key)) == "dpop_proof_missing"
    assert code_of(lambda: check("garbage", key)) == "dpop_proof_malformed"
    assert code_of(lambda: check(good(key), key, jkt=None)) == "token_dpop_unbound"
    assert code_of(lambda: check(good(key), key, jkt="another")) == "dpop_proof_key_mismatch"
    assert code_of(lambda: check(good(key), key, now=time.time() + 3600)) == "dpop_proof_iat_invalid"


def test_the_header_decides_whether_the_proof_is_a_proof() -> None:
    key = DPoPKey()
    jwk = key.public_jwk()
    claims = {"htm": "POST", "htu": URL, "iat": int(time.time()), "jti": "j"}
    ec_key = ec.generate_private_key(ec.SECP256R1())
    wrong_type = jwt.encode(claims, ec_key, algorithm="ES256", headers={"typ": "JWT", "jwk": jwk})
    assert code_of(lambda: check(wrong_type, key)) == "dpop_proof_type_invalid"
    hs = jwt.encode(claims, "k" * 32, algorithm="HS256", headers={"typ": "dpop+jwt", "jwk": jwk})
    assert code_of(lambda: check(hs, key)) == "dpop_proof_alg_mismatch"
    no_jwk = jwt.encode(claims, ec_key, algorithm="ES256", headers={"typ": "dpop+jwt"})
    assert code_of(lambda: check(no_jwk, key)) == "dpop_proof_key_invalid"
    private = {**jwk, "d": "AAAA"}
    assert code_of(lambda: check(jwt.encode(claims, ec_key, "ES256", {"typ": "dpop+jwt", "jwk": private}), key)) == (
        "dpop_proof_key_invalid"
    )
    rsa_like = {"kty": "RSA", "n": "AQAB", "e": "AQAB"}
    assert code_of(lambda: check(jwt.encode(claims, ec_key, "ES256", {"typ": "dpop+jwt", "jwk": rsa_like}), key)) == (
        "dpop_proof_key_invalid"
    )
    broken = {"kty": "EC", "crv": "P-256", "x": "AA", "y": "AA"}
    assert code_of(lambda: check(jwt.encode(claims, ec_key, "ES256", {"typ": "dpop+jwt", "jwk": broken}), key)) == (
        "dpop_proof_key_invalid"
    )


def test_a_signature_by_another_key_and_a_missing_jti_are_refused() -> None:
    key = DPoPKey("ES256")
    claims = {"htm": "POST", "htu": URL, "iat": int(time.time()), "jti": "j"}
    other = ec.generate_private_key(ec.SECP256R1())
    forged = jwt.encode(claims, other, algorithm="ES256", headers={"typ": "dpop+jwt", "jwk": key.public_jwk()})
    assert code_of(lambda: check(forged, key)) == "dpop_proof_signature_invalid"
    without = {k: v for k, v in claims.items() if k != "jti"}
    own = jwt.encode(without, other, algorithm="ES256", headers={"typ": "dpop+jwt", "jwk": _jwk_of(other)})
    thumb = DPoPKey()  # the token's key is irrelevant: the jti check comes first
    assert code_of(lambda: check(own, thumb)) == "dpop_proof_jti_missing"


def test_an_algorithm_that_does_not_fit_the_proofs_key_is_refused_not_a_server_error() -> None:
    ed_key = DPoPKey("EdDSA")
    claims = {"htm": "POST", "htu": URL, "iat": int(time.time()), "jti": "j"}
    other = ec.generate_private_key(ec.SECP256R1())
    mislabelled = jwt.encode(claims, other, algorithm="ES256", headers={"typ": "dpop+jwt", "jwk": ed_key.public_jwk()})
    assert code_of(lambda: check(mislabelled, ed_key)) == "dpop_proof_alg_mismatch"
    ec_key = DPoPKey("ES256")
    ed_signed = jwt.encode(
        claims,
        ed25519.Ed25519PrivateKey.generate(),
        algorithm="EdDSA",
        headers={"typ": "dpop+jwt", "jwk": ec_key.public_jwk()},
    )
    assert code_of(lambda: check(ed_signed, ec_key)) == "dpop_proof_alg_mismatch"


def _jwk_of(key: ec.EllipticCurvePrivateKey) -> dict[str, Any]:
    jwk: dict[str, Any] = json.loads(ECAlgorithm.to_jwk(key.public_key()))
    return jwk


def test_an_ed25519_proof_is_checked_too() -> None:
    key = ed25519.Ed25519PrivateKey.generate()
    jwk = json.loads(OKPAlgorithm.to_jwk(key.public_key()))
    claims = {"htm": "POST", "htu": URL, "iat": int(time.time()), "jti": "j-ed", "ath": _ath()}
    proof = jwt.encode(claims, key, algorithm="EdDSA", headers={"typ": "dpop+jwt", "jwk": jwk})
    from authfi._protocol.instance import jwk_thumbprint

    thumbprint = jwk_thumbprint(jwk)
    assert thumbprint is not None
    result = check_proof(
        proof,
        method="POST",
        url=URL,
        access_token=TOKEN,
        jkt=thumbprint,
        now=time.time(),
        replay=ProofReplayCache(),
    )
    assert result == thumbprint


def _ath() -> str:
    import hashlib

    from authfi._protocol.oauth import b64url

    return b64url(hashlib.sha256(TOKEN.encode()).digest())


def test_the_replay_cache_forgets_after_its_window_and_refuses_when_full_of_live_entries() -> None:
    cache = ProofReplayCache(window=10, max_entries=2)
    assert cache.first_use("a", 0) and not cache.first_use("a", 5)
    assert cache.first_use("a", 11)  # outside the window: a different proof by now
    assert cache.first_use("b", 11)
    assert not cache.first_use("c", 12)  # two live entries fill it: refuse, never forget a live one
    assert cache.first_use("c", 30)  # the old ones aged out
