"""A fake AuthFI SAML identity provider: it signs the Assertion only (RSA-SHA256, exclusive c14n), as the
platform does, and can be told to break each rule a service provider must enforce."""

# pyright: basic
from __future__ import annotations

import base64
import datetime as dt
import uuid
from dataclasses import dataclass, field
from typing import Any

import xmlsec
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID
from lxml import etree
from xmlsec import constants as C

BASE = "https://acme.authfi.io"
ENTITY = BASE + "/saml/metadata"
SP_ENTITY = "https://app.example.com/saml"
ACS = "https://app.example.com/saml/acs"
EMAIL_FORMAT = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"
SAML = "urn:oasis:names:tc:SAML:2.0:assertion"
SAMLP = "urn:oasis:names:tc:SAML:2.0:protocol"
DS = "http://www.w3.org/2000/09/xmldsig#"


def iso(epoch: float) -> str:
    return dt.datetime.fromtimestamp(epoch, dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class Signer:
    """A signing key and its self-signed certificate."""

    key_pem: bytes
    cert_pem: bytes
    cert_b64: str

    @classmethod
    def make(cls, name: str = "AuthFI test") -> Signer:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, name)])
        now = dt.datetime.now(dt.UTC)
        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(subject)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1))
            .not_valid_after(now + dt.timedelta(days=30))
            .sign(key, hashes.SHA256())
        )
        pem = cert.public_bytes(serialization.Encoding.PEM)
        der = cert.public_bytes(serialization.Encoding.DER)
        return cls(
            key_pem=key.private_bytes(
                serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
            ),
            cert_pem=pem,
            cert_b64=base64.b64encode(der).decode("ascii"),
        )


@dataclass
class FakeIdp:
    """The project's IdP. ``response`` builds the base64 ``SAMLResponse`` form field."""

    signer: Signer = field(default_factory=Signer.make)
    entity_id: str = ENTITY
    sso_url: str = BASE + "/saml/sso"

    def metadata(
        self, *, extra_certs: tuple[str, ...] = (), entity_id: str | None = None, doctype: bool = False
    ) -> bytes:
        certs = "".join(f"<ds:X509Certificate>{c}</ds:X509Certificate>" for c in (self.signer.cert_b64, *extra_certs))
        prefix = '<!DOCTYPE x [<!ENTITY e "x">]>' if doctype else ""
        return (
            f'{prefix}<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata" xmlns:ds="{DS}" '
            f'entityID="{entity_id or self.entity_id}"><md:IDPSSODescriptor '
            'protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">'
            f'<md:KeyDescriptor use="signing"><ds:KeyInfo><ds:X509Data>{certs}</ds:X509Data></ds:KeyInfo>'
            "</md:KeyDescriptor>"
            f'<md:SingleSignOnService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect" '
            f'Location="{self.sso_url}"/>'
            "</md:IDPSSODescriptor></md:EntityDescriptor>"
        ).encode()

    def response(
        self,
        *,
        now: float,
        in_response_to: str | None,
        audience: str = SP_ENTITY,
        recipient: str = ACS,
        destination: str | None = None,
        issuer: str | None = None,
        name_id: str = "ada@example.com",
        name_format: str = EMAIL_FORMAT,
        attributes: dict[str, list[str]] | None = None,
        not_before: float | None = None,
        not_on_or_after: float | None = None,
        confirm_not_on_or_after: float | None = None,
        assertion_id: str | None = None,
        status: str = "urn:oasis:names:tc:SAML:2.0:status:Success",
        sign: str | None = "assertion",
        sha1: bool = False,
        signer: Signer | None = None,
        tamper: bool = False,
        wrap: bool = False,
        doctype: bool = False,
    ) -> str:
        aid = assertion_id or "_a" + uuid.uuid4().hex
        rid = "_r" + uuid.uuid4().hex
        issuer = issuer or self.entity_id
        attrs = attributes if attributes is not None else {"groups": ["staff"], "roles": ["admin"], "name": ["Ada L"]}
        attr_xml = "".join(
            f'<saml:Attribute Name="{k}">'
            + "".join(f"<saml:AttributeValue>{v}</saml:AttributeValue>" for v in vs)
            + "</saml:Attribute>"
            for k, vs in attrs.items()
        )
        irt = f' InResponseTo="{in_response_to}"' if in_response_to else ""
        nb = iso(not_before if not_before is not None else now - 30)
        na = iso(not_on_or_after if not_on_or_after is not None else now + 300)
        cna = iso(confirm_not_on_or_after) if confirm_not_on_or_after is not None else na
        statement = f"<saml:AttributeStatement>{attr_xml}</saml:AttributeStatement>" if attr_xml else ""
        xml = (
            f'<samlp:Response xmlns:samlp="{SAMLP}" xmlns:saml="{SAML}" ID="{rid}" Version="2.0" '
            f'IssueInstant="{iso(now)}" Destination="{destination or recipient}"{irt}>'
            f"<saml:Issuer>{issuer}</saml:Issuer>"
            f'<samlp:Status><samlp:StatusCode Value="{status}"/></samlp:Status>'
            f'<saml:Assertion ID="{aid}" Version="2.0" IssueInstant="{iso(now)}">'
            f"<saml:Issuer>{issuer}</saml:Issuer>"
            f'<saml:Subject><saml:NameID Format="{name_format}">{name_id}</saml:NameID>'
            '<saml:SubjectConfirmation Method="urn:oasis:names:tc:SAML:2.0:cm:bearer">'
            f'<saml:SubjectConfirmationData NotOnOrAfter="{cna}" Recipient="{recipient}"{irt}/>'
            "</saml:SubjectConfirmation></saml:Subject>"
            f'<saml:Conditions NotBefore="{nb}" NotOnOrAfter="{na}"><saml:AudienceRestriction>'
            f"<saml:Audience>{audience}</saml:Audience></saml:AudienceRestriction></saml:Conditions>"
            f'<saml:AuthnStatement AuthnInstant="{iso(now - 5)}" SessionIndex="_s1">'
            "<saml:AuthnContext><saml:AuthnContextClassRef>urn:authfi:acr:pwd</saml:AuthnContextClassRef>"
            "</saml:AuthnContext></saml:AuthnStatement>"
            f"{statement}</saml:Assertion>"
            "</samlp:Response>"
        )
        root = etree.fromstring(xml.encode())
        use = signer or self.signer
        assertion = root.find(f"{{{SAML}}}Assertion")
        assert assertion is not None
        if sign == "assertion":
            self._sign(assertion, use, sha1)
        elif sign == "response":
            self._sign(root, use, sha1)
        if tamper:
            nid = root.find(f".//{{{SAML}}}NameID")
            assert nid is not None
            nid.text = "mallory@example.com"
        if wrap:
            evil = etree.fromstring(etree.tostring(assertion))
            for sig in evil.findall(f"{{{DS}}}Signature"):
                evil.remove(sig)
            evil.set("ID", "_evil")
            evil_name = evil.find(f".//{{{SAML}}}NameID")
            assert evil_name is not None
            evil_name.text = "admin@example.com"
            root.insert(1, evil)
        raw = etree.tostring(root)
        if doctype:
            raw = b'<!DOCTYPE r [<!ENTITY e "x">]>' + raw
        return base64.b64encode(raw).decode("ascii")

    @staticmethod
    def _sign(node: Any, signer: Signer, sha1: bool = False) -> None:
        sig = xmlsec.template.create(
            node, C.TransformExclC14N, (C.TransformRsaSha1 if sha1 else C.TransformRsaSha256), ns="ds"
        )
        node.find(f"{{{SAML}}}Issuer").addnext(sig)
        ref = xmlsec.template.add_reference(
            sig, C.TransformSha1 if sha1 else C.TransformSha256, uri="#" + node.get("ID")
        )
        xmlsec.template.add_transform(ref, C.TransformEnveloped)
        xmlsec.template.add_transform(ref, C.TransformExclC14N)
        xmlsec.template.add_x509_data(xmlsec.template.ensure_key_info(sig))
        ctx = xmlsec.SignatureContext()
        key = xmlsec.Key.from_memory(signer.key_pem, C.KeyDataFormatPem, None)
        key.load_cert_from_memory(signer.cert_pem, C.KeyDataFormatCertPem)
        ctx.key = key
        ctx.register_id(node, "ID")
        ctx.sign(sig)
