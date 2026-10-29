from __future__ import annotations

import base64
import hashlib
import os
import pickle

import jwt
import pytest

from authfi import _dpop


@pytest.mark.parametrize("alg", ["EdDSA", "ES256"])
def test_a_proof_is_fresh_per_attempt_and_binds_method_url_and_token(alg: str) -> None:
    key = _dpop.DPoPKey(alg)  # type: ignore[arg-type]
    one = key.proof("post", "https://acme.authfi.io/mcp?x=1#f", now=1000.0, access_token="AT", nonce="n1")
    two = key.proof("post", "https://acme.authfi.io/mcp?x=1#f", now=1001.0, access_token="AT")
    header = jwt.get_unverified_header(one)
    assert header["typ"] == "dpop+jwt" and header["alg"] == alg
    assert "d" not in header["jwk"]
    public = jwt.PyJWK(header["jwk"], algorithm=alg)
    claims = jwt.decode(one, public.key, algorithms=[alg])
    assert claims["htm"] == "POST"
    assert claims["htu"] == "https://acme.authfi.io/mcp"
    assert claims["iat"] == 1000
    assert claims["nonce"] == "n1"
    expected_ath = base64.urlsafe_b64encode(hashlib.sha256(b"AT").digest()).rstrip(b"=").decode()
    assert claims["ath"] == expected_ath
    other = jwt.decode(two, public.key, algorithms=[alg])
    assert other["jti"] != claims["jti"]
    assert "nonce" not in other


def test_the_run_key_is_one_per_process_and_never_pickles() -> None:
    assert _dpop.run_key() is _dpop.run_key()
    with pytest.raises(TypeError):
        pickle.dumps(_dpop.run_key())
    assert len(_dpop.run_key().thumbprint) == 43
    assert _dpop.run_key().alg == "EdDSA"  # Ed25519 is the first choice
    assert _dpop.run_key("ES256") is _dpop.run_key("ES256")
    assert _dpop.run_key("ES256").thumbprint != _dpop.run_key().thumbprint


def test_the_key_follows_what_the_project_accepts() -> None:
    assert _dpop.key_for(()) is _dpop.run_key()  # no list: ours
    assert _dpop.key_for(("EdDSA", "ES256")) is _dpop.run_key()
    assert _dpop.key_for(("ES256",)) is _dpop.run_key("ES256")
    assert _dpop.key_for(("PS256",)) is None  # nothing it can sign: the token stays Bearer


def test_ed25519_missing_from_the_runtime_falls_back_to_p256(monkeypatch: pytest.MonkeyPatch) -> None:
    from cryptography.exceptions import UnsupportedAlgorithm
    from cryptography.hazmat.primitives.asymmetric import ed25519

    def refuse() -> None:
        raise UnsupportedAlgorithm("no Ed25519 here")

    monkeypatch.setattr(ed25519.Ed25519PrivateKey, "generate", staticmethod(refuse))
    assert _dpop.DPoPKey().alg == "ES256"
    with pytest.raises(UnsupportedAlgorithm):
        _dpop.DPoPKey("EdDSA")


def test_nonces_are_kept_per_origin() -> None:
    cache = _dpop.NonceCache()
    cache.remember("https://acme.authfi.io/mcp", "n1")
    cache.remember("https://acme.authfi.io/x", None)
    assert cache.get("https://acme.authfi.io/other") == "n1"
    assert cache.get("https://other.example/") is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX only")
def test_a_forked_child_makes_its_own_run_key() -> None:
    parent = _dpop.run_key().thumbprint
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        os.close(read)
        os.write(write, _dpop.run_key().thumbprint.encode())
        os._exit(0)
    os.close(write)
    child = os.read(read, 100).decode()
    os.waitpid(pid, 0)
    assert child and child != parent
