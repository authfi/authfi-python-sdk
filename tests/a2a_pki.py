"""A ceremony-shaped chain for agent cards: Root R1, a region intermediate (policies .1.1 and .1.2), a Project CA, an
agent certificate (policy .1.4, no SAN), and a signed agent card."""

from __future__ import annotations

import base64
import datetime as dt
from dataclasses import dataclass
from typing import Any, cast

import jwt
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519, rsa
from cryptography.x509.oid import NameOID

from authfi._protocol.a2a import canonical_json

POLICY = "1.3.6.1.4.1.67322.1."
NOW = dt.datetime.now(dt.UTC)
JWS_ALG = {"ed25519": "EdDSA", "ec": "ES256", "rsa": "PS256"}

Key = ed25519.Ed25519PrivateKey | ec.EllipticCurvePrivateKey | rsa.RSAPrivateKey | ed448.Ed448PrivateKey


def new_key(kind: str) -> Key:
    """A private key of ``kind``: ``ed25519``, ``ec`` (P-256), ``rsa`` (2048) or ``ed448`` (which cards do not use)."""
    if kind == "ec":
        return ec.generate_private_key(ec.SECP256R1())
    if kind == "rsa":
        return rsa.generate_private_key(65537, 2048)
    if kind == "ed448":
        return ed448.Ed448PrivateKey.generate()
    return ed25519.Ed25519PrivateKey.generate()


def _name(cn: str) -> x509.Name:
    return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, cn)])


def issue(
    subject: str,
    key: Key,
    issuer_name: x509.Name,
    issuer_key: Key,
    *,
    ca: bool,
    policies: tuple[str, ...] = (),
    days: int = 90,
    start: dt.datetime | None = None,
) -> x509.Certificate:
    begin = start or NOW - dt.timedelta(minutes=5)
    builder = (
        x509.CertificateBuilder()
        .subject_name(_name(subject))
        .issuer_name(issuer_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(begin)
        .not_valid_after(begin + dt.timedelta(days=days))
        .add_extension(x509.BasicConstraints(ca=ca, path_length=None), critical=True)
    )
    if policies:
        builder = builder.add_extension(
            x509.CertificatePolicies(
                [x509.PolicyInformation(x509.ObjectIdentifier(POLICY + p), None) for p in policies]
            ),
            critical=False,
        )
    keyless_digest = isinstance(issuer_key, ed25519.Ed25519PrivateKey | ed448.Ed448PrivateKey)
    return builder.sign(issuer_key, None if keyless_digest else hashes.SHA256())


def der(cert: x509.Certificate) -> str:
    return base64.b64encode(cert.public_bytes(serialization.Encoding.DER)).decode()


@dataclass
class Chain:
    root: x509.Certificate
    intermediate: x509.Certificate
    project_ca: x509.Certificate
    agent_cert: x509.Certificate
    agent_key: Key

    @property
    def root_pem(self) -> bytes:
        return self.root.public_bytes(serialization.Encoding.PEM)

    def x5c(self) -> list[str]:
        return [der(self.agent_cert), der(self.project_ca), der(self.intermediate)]


def make_chain(
    *,
    agent_policies: tuple[str, ...] = ("4",),
    agent_start: dt.datetime | None = None,
    project_ca_is_ca: bool = True,
    agent_is_ca: bool = False,
    root_start: dt.datetime | None = None,
    kind: str = "ed25519",
) -> Chain:
    root_key, int_key, pca_key, agent_key = (new_key(kind) for _ in range(4))
    root = issue("AuthFI Root R1", root_key, _name("AuthFI Root R1"), root_key, ca=True, days=3650, start=root_start)
    intermediate = issue("ine1 intermediate", int_key, root.subject, root_key, ca=True, policies=("1", "2"), days=730)
    project_ca = issue("project CA", pca_key, intermediate.subject, int_key, ca=project_ca_is_ca, days=365)
    agent_cert = issue(
        "agt_remote",
        agent_key,
        project_ca.subject,
        pca_key,
        ca=agent_is_ca,
        policies=agent_policies,
        days=90,
        start=agent_start,
    )
    return Chain(root, intermediate, project_ca, agent_cert, agent_key)


def card(url: str = "https://research.partner.example", **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "name": "research",
        "description": "Finds things out",
        "supportedInterfaces": [{"url": url + "/a2a", "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "version": "1.0.0",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [{"id": "s1", "name": "search", "description": "search", "tags": ["search"]}],
        "securitySchemes": {"oauth": {"oauth2SecurityScheme": {"flows": {}}}},
    }
    body.update(extra)
    return body


def sign(
    body: dict[str, Any],
    key: Key,
    *,
    x5c: list[str] | None = None,
    kid: str = "agent-key-1",
    alg: str = "EdDSA",
) -> dict[str, Any]:
    """``body`` with a ``signatures`` member: a JWS over its RFC 8785 form, by ``key``."""
    headers: dict[str, Any] = {"kid": kid, "typ": "JOSE"}
    if x5c:
        headers["x5c"] = x5c
    compact = jwt.api_jws.PyJWS().encode(canonical_json(body), cast("Any", key), algorithm=alg, headers=headers)
    protected, _, signature = compact.split(".")
    signed = dict(body)
    signed["signatures"] = [{"protected": protected, "signature": signature}]
    return signed
