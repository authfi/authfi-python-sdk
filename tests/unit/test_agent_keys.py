from __future__ import annotations

import json
import os
import pickle
import stat
import sys
import time
from pathlib import Path
from typing import Any, cast

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from authfi import AgentKey, ConfigurationError, Secret, assert_public_jwk
from authfi.agents import Signer
from authfi.agents._files import broad_readers
from authfi.agents.keys import KeyAlgorithm, client_assertion

ALGS: list[KeyAlgorithm] = ["EdDSA", "ES256", "PS256"]


@pytest.mark.parametrize("alg", ALGS)
def test_each_key_type_signs_an_assertion_its_public_key_verifies(alg: KeyAlgorithm) -> None:
    key = AgentKey.generate(alg, kid="k1")
    assert isinstance(key, Signer)
    assert key.alg == alg
    jwk = key.public_jwk()
    assert not {"d", "p", "q", "dp", "dq", "qi"} & set(jwk)
    assertion = client_assertion(key, "agt_1", "https://acme.authfi.io/oauth/token", now=time.time())
    header = jwt.get_unverified_header(assertion)
    assert header == {"alg": alg, "typ": "JWT", "kid": "k1"}
    public = jwt.PyJWK(jwk)
    claims = jwt.decode(assertion, public.key, algorithms=[alg], audience="https://acme.authfi.io/oauth/token")
    assert claims["iss"] == claims["sub"] == "agt_1"
    assert claims["exp"] - claims["iat"] == 300
    assert claims["jti"]


def test_every_assertion_has_a_fresh_jti_and_lives_at_most_300_seconds() -> None:
    key = AgentKey.generate(kid="k1")
    jtis = {
        jwt.decode(client_assertion(key, "a", "b", now=0), options={"verify_signature": False})["jti"]
        for _ in range(20)
    }
    assert len(jtis) == 20
    with pytest.raises(ConfigurationError):
        client_assertion(key, "a", "b", now=0, lifetime=301)
    with pytest.raises(ConfigurationError):
        client_assertion(key, "", "b", now=0)


def test_the_kid_defaults_to_the_rfc_7638_thumbprint() -> None:
    key = AgentKey.generate()
    assert key.kid == key.thumbprint()
    assert len(key.kid) == 43


def test_the_rfc_7638_example_thumbprint() -> None:
    # RFC 7638 §3.1: the example RSA key and its thumbprint.
    n = (
        "0vx7agoebGcQSuuPiLJXZptN9nndrQmbXEps2aiAFbWhM78LhWx4cbbfAAtVT86zwu1RK7aPFFxuhDR1L6tSoc_BJECPebWKRXjBZCiFV4n3oknjhMs"
        "tn64tZ_2W-5JsGY4Hc5n9yBXArwl93lqt7_RN5w6Cf0h4QyQ5v-65YGjQR0_FDW2QvzqY368QQMicAtaSqzs8KJZgnYb9c7d0zgdAZHzu6qMQvRL5hajr"
        "n1n91CbOpbISD08qNLyrdkt-bFTWhAI4vMQFh6WeZu0fM4lFd2NcRwr3XPksINHaQ-G_xBniIqbw0Ls1jF44-csFCur-kEgU8awapJzKnqDKgw"
    )
    from authfi.agents.keys import _thumbprint

    assert _thumbprint({"kty": "RSA", "n": n, "e": "AQAB"}) == "NzbLsXh8uDCcd-6MNwXF4W_7noWXFZAfHkxZsRGC9Xs"


def test_rsa_keys_are_3072_bits_and_out_of_range_ones_refused() -> None:
    assert AgentKey.generate("PS256", kid="k").public_jwk()["kty"] == "RSA"
    small = rsa.generate_private_key(public_exponent=65537, key_size=1024)  # noqa: S505 - the weak key under test
    with pytest.raises(ConfigurationError, match="2048 to 4096"):
        AgentKey(small, "k")
    with pytest.raises(ConfigurationError, match="P-256"):
        AgentKey(ec.generate_private_key(ec.SECP384R1()), "k")


def pem_of(key: ec.EllipticCurvePrivateKey) -> bytes:
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


def test_pem_round_trip_and_refusals() -> None:
    raw = ec.generate_private_key(ec.SECP256R1())
    key = AgentKey.from_pem(pem_of(raw), kid="k1")
    assert key.alg == "ES256"
    assert AgentKey.from_pem(Secret(pem_of(raw)), kid="k1").public_jwk() == key.public_jwk()
    with pytest.raises(ConfigurationError, match="OPENSSH PRIVATE KEY"):
        AgentKey.from_pem("-----BEGIN OPENSSH PRIVATE KEY-----\nxx\n", kid="k1")
    with pytest.raises(ConfigurationError, match="not PEM"):
        AgentKey.from_pem("hello", kid="k1")
    with pytest.raises(ConfigurationError, match="not a usable"):
        AgentKey.from_pem("-----BEGIN PRIVATE KEY-----\nnot base64\n-----END PRIVATE KEY-----\n", kid="k1")
    with pytest.raises(ConfigurationError, match="kid is required"):
        AgentKey.from_pem(pem_of(raw), kid="")


@pytest.mark.parametrize("alg", ALGS)
def test_jwk_round_trip(alg: KeyAlgorithm) -> None:
    from jwt.algorithms import ECAlgorithm, OKPAlgorithm, RSAAlgorithm

    key = AgentKey.generate(alg, kid="k9")
    algorithm: Any = {"EdDSA": OKPAlgorithm, "ES256": ECAlgorithm, "PS256": RSAAlgorithm}[alg]
    private = json.loads(algorithm.to_jwk(cast(Any, key)._key))
    private["kid"] = "k9"
    reloaded = AgentKey.from_jwk(private)
    assert reloaded.kid == "k9"
    assert reloaded.public_jwk() == key.public_jwk()
    with pytest.raises(ConfigurationError, match="public half"):
        AgentKey.from_jwk(key.public_jwk())


def test_a_mislabelled_or_wrong_use_jwk_is_refused() -> None:
    jwk = AgentKey.generate("PS256", kid="k").public_jwk()
    with pytest.raises(ConfigurationError, match="relabel"):
        assert_public_jwk({**jwk, "alg": "RS256"})
    with pytest.raises(ConfigurationError, match="signing key"):
        assert_public_jwk({**jwk, "use": "enc"})
    with pytest.raises(ConfigurationError, match="no 'y'"):
        assert_public_jwk({"kty": "EC", "crv": "P-256", "x": "AA"})
    with pytest.raises(ConfigurationError):
        assert_public_jwk({"kty": "EC", "crv": "P-384", "x": "AA", "y": "AA"})


def test_a_public_jwk_carrying_a_private_member_is_burned() -> None:
    with pytest.raises(ConfigurationError, match="burned") as caught:
        assert_public_jwk({"kty": "OKP", "crv": "Ed25519", "x": "AA", "d": "secret"})
    assert "'d'" in str(caught.value)
    assert "secret" not in str(caught.value)
    jwk = AgentKey.generate(kid="k").public_jwk()
    assert assert_public_jwk(jwk) is jwk


def test_the_key_never_prints_or_pickles() -> None:
    key = AgentKey.generate("ES256", kid="k1")
    assert repr(key) == "AgentKey(kid='k1', alg='ES256')"
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(key)
    assert not hasattr(key, "__dict__")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_write_creates_an_owner_only_file_and_never_overwrites(tmp_path: Path) -> None:
    key = AgentKey.generate(kid="k1")
    path = tmp_path / "agent.key"
    key.write(path)
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert key.path == path.resolve()
    with pytest.raises(ConfigurationError, match="exists"):
        key.write(path)
    loaded = AgentKey.from_file(path, kid="k1")
    assert loaded.public_jwk() == key.public_jwk()
    assert loaded.path == path.resolve()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_a_key_file_others_can_read_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    os.chmod(path, 0o644)
    with pytest.raises(ConfigurationError, match="chmod 600"):
        AgentKey.from_file(path, kid="k1")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_a_jwk_file_supplies_its_kid_and_an_empty_file_is_refused(tmp_path: Path) -> None:
    from jwt.algorithms import OKPAlgorithm

    key = AgentKey.generate(kid="from-jwk")
    private = json.loads(OKPAlgorithm.to_jwk(cast(Any, key)._key))
    private["kid"] = "from-jwk"
    path = tmp_path / "agent.jwk"
    path.write_text(json.dumps(private))
    os.chmod(path, 0o600)
    assert AgentKey.from_file(path).kid == "from-jwk"
    empty = tmp_path / "empty.key"
    empty.write_text("")
    os.chmod(empty, 0o600)
    with pytest.raises(ConfigurationError, match="empty"):
        AgentKey.from_file(empty, kid="k")


@pytest.mark.parametrize(
    ("sddl", "readers"),
    [
        ("O:S-1-5-21-1G:S-1-5-21-1D:P(A;;FA;;;OW)", []),
        ("D:(A;;FA;;;SY)(A;;FA;;;BA)(A;;FA;;;S-1-5-21-1-2-3-1001)", []),
        ("D:AI(A;ID;FA;;;SY)(A;ID;0x1200a9;;;BU)", ["BU"]),
        ("D:(A;;FR;;;WD)", ["WD"]),
        ("D:(A;;GR;;;AU)(D;;FA;;;BU)", ["AU"]),
        ("D:(A;;0x100000;;;BU)", []),
    ],
)
def test_windows_permission_entries_that_let_others_read(sddl: str, readers: list[str]) -> None:
    assert broad_readers(sddl) == readers


@pytest.mark.skipif(sys.platform != "win32", reason="Windows permissions")
def test_windows_key_files_are_owner_only_and_a_shared_one_is_refused(tmp_path: Path) -> None:  # pragma: no cover
    import subprocess

    key = AgentKey.generate(kid="k1")
    path = tmp_path / "agent.key"
    key.write(path)
    assert AgentKey.from_file(path, kid="k1").public_jwk() == key.public_jwk()
    with pytest.raises(ConfigurationError, match="exists"):
        key.write(path)
    # Let BUILTIN\Users read it, as an inherited default often does.
    subprocess.run(["icacls", str(path), "/grant", "*S-1-5-32-545:R"], check=True, capture_output=True)  # noqa: S603, S607
    with pytest.raises(ConfigurationError, match="BU"):
        AgentKey.from_file(path, kid="k1")
