"""SAML 2.0 sign-in for a standard app: AuthFI is the identity provider, your app the service provider.

Install the extra::

    pip install "authfi[saml] @ git+https://github.com/authfi/authfi-python-sdk@main"

Register the app in the project with its entity id and assertion consumer URL, and read the project's metadata::

    from authfi import AuthFI
    from authfi.saml import SamlServiceProvider

    af = AuthFI.from_env()
    sp = SamlServiceProvider.from_project(
        af, sp_entity_id="https://app.example.com/saml", acs_url="https://app.example.com/saml/acs"
    )

    # /login
    pending = sp.start_sign_in(relay_state="/dashboard")
    session["saml"] = pending.dump()
    return redirect(pending.url)

    # POST /saml/acs
    signed_in = sp.finish_sign_in(request.form["SAMLResponse"], PendingSamlSignIn.load(session.pop("saml")))
    signed_in.user.sub, signed_in.user.email

Only a **signed assertion** is accepted, with the signature made by a key in the project's metadata. The
audience, the recipient, ``InResponseTo``, the issuer and the validity window are checked against what you
configured, never against the request that arrived. An assertion is accepted once.
"""

from __future__ import annotations

# python3-saml and lxml ship no type information: what they return is checked at the boundary below.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false
import base64
import binascii
import json
import threading
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, cast
from urllib.parse import urlsplit

from authfi._protocol.http import DOCUMENT_LIMIT, Effect, HttpRequest, HttpResponse, Retry, call
from authfi._protocol.signin import IdTokenClaims
from authfi._protocol.urls import origin_of
from authfi.errors import AuthFIError, ConfigurationError, DiscoveryError, IntegrationError

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI, AuthFI

try:
    import xmlsec as _xmlsec  # noqa: F401  # pyright: ignore[reportMissingImports,reportUnusedImport]
    from onelogin.saml2.auth import OneLogin_Saml2_Auth  # pyright: ignore[reportMissingImports]
    from onelogin.saml2.errors import OneLogin_Saml2_ValidationError  # pyright: ignore[reportMissingImports]
    from onelogin.saml2.response import OneLogin_Saml2_Response  # pyright: ignore[reportMissingImports]
    from onelogin.saml2.settings import OneLogin_Saml2_Settings  # pyright: ignore[reportMissingImports]
    from onelogin.saml2.xml_utils import OneLogin_Saml2_XML  # pyright: ignore[reportMissingImports]
except ImportError as exc:  # pragma: no cover - exercised by the missing-extra test through a stub
    raise IntegrationError(
        "authfi.saml needs python3-saml: pip install 'authfi[saml] @ git+https://github.com/authfi/authfi-python-sdk@main'"
    ) from exc

__all__ = [
    "InMemoryReplayStore",
    "PendingSamlSignIn",
    "ReplayStore",
    "SamlAssertionRefused",
    "SamlIdp",
    "SamlServiceProvider",
    "SamlSignIn",
    "parse_idp_metadata",
]

#: The project's IdP metadata, relative to its origin.
METADATA_PATH: Final = "/saml/metadata"
#: The largest SAMLResponse read (1 MiB of base64 text).
RESPONSE_LIMIT: Final = 1 << 20
#: How long a started sign-in may be finished, as the platform keeps its request ids (10 minutes).
PENDING_LIFETIME: Final = 600.0
_MD: Final = "urn:oasis:names:tc:SAML:2.0:metadata"
_DS: Final = "http://www.w3.org/2000/09/xmldsig#"
_REDIRECT: Final = "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-Redirect"
_POST_BINDING: Final = "urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
_EMAIL_FORMAT: Final = "urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress"

#: Attribute names read for each field of the user, first match wins. Pass ``attributes=`` to change them.
DEFAULT_ATTRIBUTES: Final[Mapping[str, tuple[str, ...]]] = {
    "email": ("email", "mail", "emailAddress"),
    "name": ("name", "full_name", "displayName"),
    "groups": ("groups",),
    "roles": ("roles",),
}

# python3-saml's validation errors by constant name, as stable codes a caller can branch on. Names a
# release does not define are skipped, so the table holds across the supported range.
_BY_NAME: Final[Mapping[str, str]] = {
    "UNSUPPORTED_SAML_VERSION": "saml_version_unsupported",
    "MISSING_ID": "saml_id_missing",
    "WRONG_NUMBER_OF_ASSERTIONS": "saml_assertion_count",
    "MISSING_STATUS": "saml_status_missing",
    "MISSING_STATUS_CODE": "saml_status_missing",
    "STATUS_CODE_IS_NOT_SUCCESS": "saml_status_not_success",
    "WRONG_SIGNED_ELEMENT": "saml_signed_element_wrong",
    "ID_NOT_FOUND_IN_SIGNED_ELEMENT": "saml_signed_element_wrong",
    "DUPLICATED_ID_IN_SIGNED_ELEMENTS": "saml_signed_element_wrong",
    "INVALID_SIGNED_ELEMENT": "saml_signed_element_wrong",
    "DUPLICATED_REFERENCE_IN_SIGNED_ELEMENTS": "saml_signed_element_wrong",
    "UNEXPECTED_SIGNED_ELEMENTS": "saml_signed_element_wrong",
    "WRONG_NUMBER_OF_SIGNATURES_IN_RESPONSE": "saml_signature_missing",
    "WRONG_NUMBER_OF_SIGNATURES_IN_ASSERTION": "saml_signature_missing",
    "WRONG_NUMBER_OF_SIGNATURES": "saml_signature_missing",
    "NO_SIGNED_MESSAGE": "saml_signature_missing",
    "NO_SIGNED_ASSERTION": "saml_signature_missing",
    "NO_SIGNATURE_FOUND": "saml_signature_missing",
    "KEYINFO_NOT_FOUND_IN_ENCRYPTED_DATA": "saml_encrypted_unsupported",
    "CHILDREN_NODE_NOT_FOUND_IN_KEYINFO": "saml_encrypted_unsupported",
    "UNSUPPORTED_RETRIEVAL_METHOD": "saml_encrypted_unsupported",
    "NO_ENCRYPTED_ASSERTION": "saml_encrypted_unsupported",
    "NO_ENCRYPTED_NAMEID": "saml_encrypted_unsupported",
    "NO_NAMEID": "saml_name_id_missing",
    "EMPTY_NAMEID": "saml_name_id_missing",
    "SP_NAME_QUALIFIER_NAME_MISMATCH": "saml_name_id_invalid",
    "DUPLICATED_ATTRIBUTE_NAME_FOUND": "saml_attribute_duplicated",
    "INVALID_SIGNATURE": "saml_signature_invalid",
    "RESPONSE_EXPIRED": "saml_assertion_expired",
    "ASSERTION_EXPIRED": "saml_assertion_expired",
    "ASSERTION_TOO_EARLY": "saml_assertion_not_yet_valid",
    "AUTHN_CONTEXT_MISMATCH": "saml_authn_context_mismatch",
    "WRONG_AUDIENCE": "saml_audience_mismatch",
    "ISSUER_NOT_FOUND_IN_ASSERTION": "saml_issuer_mismatch",
    "ISSUER_MULTIPLE_IN_RESPONSE": "saml_issuer_mismatch",
    "WRONG_ISSUER": "saml_issuer_mismatch",
    "WRONG_INRESPONSETO": "saml_in_response_to_mismatch",
    "WRONG_NUMBER_OF_AUTHSTATEMENTS": "saml_authn_statement_missing",
    "SESSION_EXPIRED": "saml_session_expired",
    "WRONG_SUBJECTCONFIRMATION": "saml_subject_confirmation_invalid",
    "WRONG_DESTINATION": "saml_destination_mismatch",
    "EMPTY_DESTINATION": "saml_destination_mismatch",
    "INVALID_XML_FORMAT": "saml_xml_invalid",
    "DEPRECATED_SIGNATURE_METHOD": "saml_algorithm_rejected",
    "DEPRECATED_DIGEST_METHOD": "saml_algorithm_rejected",
    "MISSING_CONDITIONS": "saml_conditions_missing",
}
_CODES: Final[Mapping[int, str]] = {
    getattr(OneLogin_Saml2_ValidationError, name): code
    for name, code in _BY_NAME.items()
    if hasattr(OneLogin_Saml2_ValidationError, name)
}


class SamlAssertionRefused(AuthFIError):
    """A SAML response was refused, or a SAML sign-in could not be started or restored.

    ``code`` names why: ``saml_signature_missing``, ``saml_signature_invalid``, ``saml_audience_mismatch``,
    ``saml_issuer_mismatch``, ``saml_in_response_to_mismatch``, ``saml_destination_mismatch``,
    ``saml_subject_confirmation_invalid``, ``saml_assertion_expired``, ``saml_assertion_replayed``,
    ``saml_unsolicited``, ``saml_xml_invalid``, ``saml_pending_expired``, ``saml_pending_malformed`` and
    the other ``saml_`` codes. The message never carries the assertion.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"The SAML response was refused ({code}): {detail}", 401, code=code)


@dataclass(frozen=True, slots=True)
class SamlIdp:
    """The project as a SAML identity provider, read from its metadata."""

    entity_id: str
    sso_url: str
    #: The signing certificates (base64 DER, no PEM armour). Only these may sign an assertion.
    signing_certs: tuple[str, ...]


def _text(node: Any) -> str:
    return "".join(str(node.text or "").split())


def parse_idp_metadata(xml: bytes, *, expected_entity_id: str) -> SamlIdp:
    """The :class:`SamlIdp` described by ``xml``, which must name ``expected_entity_id``.

    The first certificate of each signing ``KeyDescriptor`` is the signing key; the custody and issuing
    certificates that follow it in the same ``X509Data`` are not trusted to sign.
    """
    if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
        raise DiscoveryError("The SAML metadata declares a DTD.", code="metadata_invalid")
    try:
        root = OneLogin_Saml2_XML.to_etree(xml)
    except Exception as exc:
        raise DiscoveryError("The SAML metadata is not valid XML.", code="metadata_invalid") from exc
    ns = {"md": _MD, "ds": _DS}
    descriptors = [root] if root.tag == f"{{{_MD}}}EntityDescriptor" else []
    if len(descriptors) != 1:
        raise DiscoveryError("The SAML metadata is not one EntityDescriptor.", code="metadata_invalid")
    entity_id = descriptors[0].get("entityID")
    if entity_id != expected_entity_id:
        raise DiscoveryError(
            f"The SAML metadata names {entity_id!r}, not {expected_entity_id!r}: it is not this project's.",
            code="metadata_invalid",
        )
    idps = descriptors[0].findall("md:IDPSSODescriptor", ns)
    if len(idps) != 1:
        raise DiscoveryError("The SAML metadata has no identity provider role.", code="metadata_invalid")
    idp = idps[0]
    sso = {s.get("Binding"): s.get("Location") for s in idp.findall("md:SingleSignOnService", ns)}
    sso_url = sso.get(_REDIRECT) or sso.get(_POST_BINDING)
    if not sso_url or not sso_url.startswith(("https://", "http://")):
        raise DiscoveryError("The SAML metadata has no sign-on URL.", code="metadata_invalid")
    certs: list[str] = []
    for key in idp.findall("md:KeyDescriptor", ns):
        if key.get("use") not in (None, "signing"):
            continue
        first = key.find("ds:KeyInfo/ds:X509Data/ds:X509Certificate", ns)
        if first is not None and _text(first):
            certs.append(_text(first))
    if not certs:
        raise DiscoveryError("The SAML metadata has no signing certificate.", code="metadata_invalid")
    return SamlIdp(entity_id=entity_id, sso_url=sso_url, signing_certs=tuple(certs))


def _metadata_flow(url: str) -> Generator[Effect, HttpResponse | None, bytes]:
    response = yield from call(
        HttpRequest(
            "GET",
            url,
            what="the SAML metadata",
            headers={"Accept": "application/samlmetadata+xml, application/xml"},
            retry=Retry.SAFE,
            max_bytes=DOCUMENT_LIMIT,
        )
    )
    if response.status != 200:
        raise DiscoveryError(f"The SAML metadata at {url} answered HTTP {response.status}.", status=response.status)
    return response.body


class ReplayStore(Protocol):
    """Remembers what was used, so it is accepted once. Share one across processes (Redis, a database)."""

    def remember(self, key: str, expires_at: float, now: float) -> bool:
        """Record ``key`` until ``expires_at`` and return ``True``; return ``False`` when it is already held."""
        ...


class InMemoryReplayStore:
    """A replay store for one process. Full of live entries, it refuses rather than forget one."""

    def __init__(self, max_entries: int = 100_000) -> None:
        self._max = max_entries
        self._seen: dict[str, float] = {}
        self._lock = threading.Lock()

    def remember(self, key: str, expires_at: float, now: float) -> bool:
        """See :class:`ReplayStore`."""
        with self._lock:
            held = self._seen.get(key)
            if held is not None and held > now:
                return False
            if len(self._seen) >= self._max:
                self._seen = {k: v for k, v in self._seen.items() if v > now}
                if len(self._seen) >= self._max:
                    raise SamlAssertionRefused("saml_replay_store_full", "the replay store holds only live entries")
            self._seen[key] = expires_at
            return True


@dataclass(frozen=True, slots=True)
class PendingSamlSignIn:
    """A started sign-in: where to send the browser, and what the response must answer."""

    url: str
    request_id: str
    relay_state: str | None
    expires_at: float

    def dump(self) -> str:
        """A string to keep in the server-side session between the redirect and the response."""
        return json.dumps(
            {"id": self.request_id, "rs": self.relay_state, "exp": self.expires_at, "url": self.url},
            separators=(",", ":"),
        )

    @classmethod
    def load(cls, text: str) -> PendingSamlSignIn:
        """The pending sign-in :meth:`dump` made, or :class:`SamlAssertionRefused` for anything else."""
        try:
            data: Any = json.loads(text)
            request_id, relay, expires, url = data["id"], data["rs"], data["exp"], data["url"]
        except (ValueError, KeyError, TypeError) as exc:
            raise SamlAssertionRefused("saml_pending_malformed", "the pending sign-in is not one") from exc
        if (
            not isinstance(request_id, str)
            or not request_id
            or not isinstance(url, str)
            or not (relay is None or isinstance(relay, str))
            or isinstance(expires, bool)
            or not isinstance(expires, int | float)
        ):
            raise SamlAssertionRefused("saml_pending_malformed", "the pending sign-in is not one")
        return cls(url=url, request_id=request_id, relay_state=relay, expires_at=float(expires))


@dataclass(frozen=True, slots=True)
class SamlSignIn:
    """A finished SAML sign-in."""

    #: Who signed in. ``sub`` is the assertion's NameID; ``email_verified`` is never claimed.
    user: IdTokenClaims
    assertion_id: str
    session_index: str | None
    #: The ``RelayState`` you passed to :meth:`SamlServiceProvider.start_sign_in`; ``None`` for an IdP-initiated one.
    relay_state: str | None
    #: When the IdP session ends, if it said.
    session_not_on_or_after: float | None


def _refuse(exc: Exception) -> SamlAssertionRefused:
    code = _CODES.get(getattr(exc, "code", -1), "saml_assertion_invalid")
    return SamlAssertionRefused(code, str(exc).split("\n", 1)[0][:200])


def _first(attributes: Mapping[str, Sequence[Any]], names: Sequence[str]) -> list[str]:
    for name in names:
        values = [v for v in attributes.get(name, ()) if isinstance(v, str) and v]
        if values:
            return values
    return []


class SamlServiceProvider:
    """Your app as a service provider of one AuthFI project.

    Args:
        idp: the project's :class:`SamlIdp` (see :meth:`from_project`).
        sp_entity_id: the entity id the app is registered under in the project; the audience of its assertions.
        acs_url: the app's assertion consumer URL, registered in the project. The recipient is checked against
            this value, not against the request that arrived.
        leeway: seconds of clock skew allowed on the validity window (0 to 300).
        clock: a callable returning the epoch seconds; the system clock unless given (for tests).
        replay_store: where used assertions are remembered. One per process by default.
        allow_idp_initiated: accept a response that answers no request of yours. Off by default; turn it on
            only for an app launched from the AuthFI portal, and never rely on ``RelayState`` then.
        attributes: attribute names per user field, replacing :data:`DEFAULT_ATTRIBUTES`.
    """

    def __init__(
        self,
        idp: SamlIdp,
        *,
        sp_entity_id: str,
        acs_url: str,
        leeway: float = 60.0,
        clock: Callable[[], float] | None = None,
        replay_store: ReplayStore | None = None,
        allow_idp_initiated: bool = False,
        attributes: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        parts = urlsplit(acs_url)
        local = parts.hostname in ("localhost", "127.0.0.1", "::1")
        if parts.scheme != "https" and not (parts.scheme == "http" and local):
            raise ConfigurationError("acs_url must be an https URL (plain http only for loopback)")
        if not sp_entity_id.strip():
            raise ConfigurationError("sp_entity_id is required")
        if isinstance(leeway, bool) or not 0 <= leeway <= 300:
            raise ConfigurationError("leeway must be from 0 to 300 seconds")
        import time

        self._idp = idp
        self._sp_entity_id = sp_entity_id
        self._acs_url = acs_url
        self._leeway = float(leeway)
        self._now = clock or time.time
        self._replay: ReplayStore = replay_store or InMemoryReplayStore()
        self._allow_idp_initiated = allow_idp_initiated
        self._attributes = {**DEFAULT_ATTRIBUTES, **(attributes or {})}
        self._request_data: dict[str, Any] = {
            "https": "on" if parts.scheme == "https" else "off",
            "http_host": parts.netloc,
            "script_name": parts.path or "/",
            "get_data": {},
            "post_data": {},
        }
        self._settings = OneLogin_Saml2_Settings(
            {
                "strict": True,
                "debug": False,
                "sp": {
                    "entityId": sp_entity_id,
                    "assertionConsumerService": {"url": acs_url, "binding": _POST_BINDING},
                },
                "idp": {
                    "entityId": idp.entity_id,
                    "singleSignOnService": {"url": idp.sso_url, "binding": _REDIRECT},
                    "x509certMulti": {"signing": list(idp.signing_certs)},
                },
                "security": {
                    "wantAssertionsSigned": True,
                    "wantMessagesSigned": False,
                    "wantAssertionsEncrypted": False,
                    "wantNameIdEncrypted": False,
                    "wantNameId": True,
                    "authnRequestsSigned": False,
                    "rejectDeprecatedAlgorithm": True,
                    "rejectUnsolicitedResponsesWithInResponseTo": False,
                    "allowRepeatAttributeName": False,
                    "wantAttributeStatement": False,
                    "requestedAuthnContext": False,
                },
            }
        )

    @classmethod
    def from_project(cls, project: AuthFI, *, sp_entity_id: str, acs_url: str, **options: Any) -> SamlServiceProvider:
        """Read the project's metadata from ``<origin>/saml/metadata`` and build the provider."""
        url = origin_of(project.issuer) + METADATA_PATH
        body = project._transport.run(_metadata_flow(url))  # pyright: ignore[reportPrivateUsage]
        idp = parse_idp_metadata(body, expected_entity_id=url)
        options.setdefault("leeway", project.leeway)
        options.setdefault("clock", project.clock.time)
        return cls(idp, sp_entity_id=sp_entity_id, acs_url=acs_url, **options)

    @classmethod
    async def afrom_project(
        cls, project: AsyncAuthFI, *, sp_entity_id: str, acs_url: str, **options: Any
    ) -> SamlServiceProvider:
        """:meth:`from_project` for an :class:`AsyncAuthFI`."""
        url = origin_of(project.issuer) + METADATA_PATH
        body = await project._transport.run(_metadata_flow(url))  # pyright: ignore[reportPrivateUsage]
        idp = parse_idp_metadata(body, expected_entity_id=url)
        options.setdefault("leeway", project.leeway)
        options.setdefault("clock", project.clock.time)
        return cls(idp, sp_entity_id=sp_entity_id, acs_url=acs_url, **options)

    @property
    def idp(self) -> SamlIdp:
        """The identity provider this provider trusts."""
        return self._idp

    def start_sign_in(self, relay_state: str | None = None) -> PendingSamlSignIn:
        """The redirect that sends the browser to AuthFI, and the request id the response must answer."""
        auth = OneLogin_Saml2_Auth(dict(self._request_data), self._settings)
        url = cast(str, auth.login(return_to=relay_state))
        request_id = cast("str | None", auth.get_last_request_id())
        if not request_id:  # pragma: no cover - the library always sets it
            raise SamlAssertionRefused("saml_request_failed", "no request id was made")
        return PendingSamlSignIn(
            url=url, request_id=request_id, relay_state=relay_state, expires_at=self._now() + PENDING_LIFETIME
        )

    def finish_sign_in(self, saml_response: str, pending: PendingSamlSignIn | None) -> SamlSignIn:
        """Validate the ``SAMLResponse`` form field of the POST to your ACS and return who signed in.

        ``pending`` is what :meth:`start_sign_in` returned for this browser. ``None`` accepts an unsolicited
        (IdP-initiated) response, and only when the provider was built with ``allow_idp_initiated=True``.
        Raises :class:`SamlAssertionRefused` with the reason, whatever is wrong.
        """
        now = self._now()
        if pending is None:
            if not self._allow_idp_initiated:
                raise SamlAssertionRefused("saml_unsolicited", "no sign-in was started for this response")
        elif now > pending.expires_at:
            raise SamlAssertionRefused("saml_pending_expired", "the sign-in was started too long ago")
        raw = self._decode(saml_response)
        data = dict(self._request_data)
        data["post_data"] = {"SAMLResponse": saml_response}
        try:
            response = OneLogin_Saml2_Response(self._settings, base64.b64encode(raw).decode("ascii"))
            response.is_valid(data, request_id=pending.request_id if pending else None, raise_exceptions=True)
        except SamlAssertionRefused:
            raise
        except OneLogin_Saml2_ValidationError as exc:
            raise _refuse(exc) from exc
        except Exception as exc:
            raise SamlAssertionRefused("saml_assertion_invalid", "the response could not be validated") from exc
        if pending is None and response.get_in_response_to() is not None:
            raise SamlAssertionRefused("saml_in_response_to_mismatch", "an unsolicited response answers a request")
        try:
            return self._accept(response, pending, now)
        except OneLogin_Saml2_ValidationError as exc:
            raise _refuse(exc) from exc

    def _decode(self, saml_response: str) -> bytes:
        given = cast(object, saml_response)
        if not isinstance(given, str) or not given or len(given) > RESPONSE_LIMIT:
            raise SamlAssertionRefused("saml_xml_invalid", "the SAMLResponse is empty or too large")
        try:
            raw = base64.b64decode(saml_response, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise SamlAssertionRefused("saml_xml_invalid", "the SAMLResponse is not base64") from exc
        if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
            raise SamlAssertionRefused("saml_xml_invalid", "the SAMLResponse declares a DTD")
        return raw

    def _accept(self, response: Any, pending: PendingSamlSignIn | None, now: float) -> SamlSignIn:
        not_on_or_after = self._check_window(response, now)
        attributes = cast("Mapping[str, Sequence[str]]", response.get_attributes())
        assertion_id = cast(str, response.get_assertion_id())
        name_id = cast("str | None", response.get_nameid())
        if not name_id:
            raise SamlAssertionRefused("saml_name_id_missing", "the assertion names nobody")
        expires = float(not_on_or_after) + self._leeway
        keys = [f"assertion:{assertion_id}"] + ([f"request:{pending.request_id}"] if pending else [])
        for key in keys:
            if not self._replay.remember(key, expires, now):
                raise SamlAssertionRefused("saml_assertion_replayed", "this response was already used")
        user = self._user(name_id, cast("str | None", response.get_nameid_format()), attributes, response)
        session_not_on_or_after = response.get_session_not_on_or_after()
        return SamlSignIn(
            user=user,
            assertion_id=assertion_id,
            session_index=cast("str | None", response.get_session_index()),
            relay_state=pending.relay_state if pending else None,
            session_not_on_or_after=float(session_not_on_or_after) if session_not_on_or_after else None,
        )

    def _check_window(self, response: Any, now: float) -> float:
        """Refuse an assertion outside its validity window, with this provider's leeway, and return its end."""
        from onelogin.saml2.utils import OneLogin_Saml2_Utils  # pyright: ignore[reportMissingImports]

        nodes = OneLogin_Saml2_XML.query(response.document, "/samlp:Response/saml:Assertion/saml:Conditions")
        try:
            if len(nodes) != 1 or not nodes[0].get("NotOnOrAfter"):
                raise ValueError("no validity window")
            ends = float(OneLogin_Saml2_Utils.parse_SAML_to_time(nodes[0].get("NotOnOrAfter")))
            begins = nodes[0].get("NotBefore")
            started = float(OneLogin_Saml2_Utils.parse_SAML_to_time(begins)) if begins else None
        except Exception as exc:
            raise SamlAssertionRefused("saml_assertion_expired", "the assertion has no usable validity window") from exc
        if now - self._leeway >= ends:
            raise SamlAssertionRefused("saml_assertion_expired", "the assertion is outside its validity window")
        if started is not None and now + self._leeway < started:
            raise SamlAssertionRefused("saml_assertion_not_yet_valid", "the assertion is not valid yet")
        return ends

    def _user(
        self, name_id: str, name_id_format: str | None, attributes: Mapping[str, Sequence[str]], response: Any
    ) -> IdTokenClaims:
        emails = _first(attributes, self._attributes["email"])
        email = emails[0] if emails else (name_id if name_id_format == _EMAIL_FORMAT and "@" in name_id else None)
        names = _first(attributes, self._attributes["name"])
        contexts = cast("list[str]", response.get_authn_contexts())
        claims: dict[str, Any] = {
            "sub": name_id,
            "groups": _first(attributes, self._attributes["groups"]),
            "roles": _first(attributes, self._attributes["roles"]),
            "amr": ["saml"],
            "saml_attributes": {k: list(v) for k, v in attributes.items()},
        }
        if email:
            claims["email"] = email
        if names:
            claims["name"] = names[0]
        if contexts:
            claims["acr"] = contexts[0]
        instant = self._authn_instant(response)
        if instant is not None:
            claims["auth_time"] = instant
        return IdTokenClaims.of(claims)

    @staticmethod
    def _authn_instant(response: Any) -> int | None:
        from onelogin.saml2.utils import OneLogin_Saml2_Utils  # pyright: ignore[reportMissingImports]

        nodes = OneLogin_Saml2_XML.query(response.document, "/samlp:Response/saml:Assertion/saml:AuthnStatement")
        value = nodes[0].get("AuthnInstant") if len(nodes) == 1 else None
        if not value:
            return None
        try:
            return int(OneLogin_Saml2_Utils.parse_SAML_to_time(value))
        except Exception:
            return None
