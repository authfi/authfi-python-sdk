"""The project's discovery document (RFC 8414 / OpenID Connect Discovery). No I/O.

The issuer is the one required setting; everything else is read from
``<issuer>/.well-known/openid-configuration``, falling back to
``/.well-known/oauth-authorization-server``. The document must name the issuer exactly
(RFC 8414 §3.3), and every endpoint it names must be https.
"""

from __future__ import annotations

import re
import types
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from authfi._protocol.errors import describe
from authfi._protocol.http import DOCUMENT_LIMIT, Flow, HttpRequest, Retry, call
from authfi._protocol.urls import require_endpoint
from authfi.errors import DiscoveryError

__all__ = [
    "DEFAULT_TTL",
    "ServerMetadata",
    "metadata_flow",
    "parse_metadata",
    "ttl_from_headers",
]

#: Seconds a document is kept when its answer says nothing.
DEFAULT_TTL = 300.0
_MIN_TTL, _MAX_TTL = 60.0, 86_400.0
_MAX_AGE = re.compile(r"(?:^|,)\s*max-age\s*=\s*\"?(\d+)\"?", re.IGNORECASE)

_PATHS = ("/.well-known/openid-configuration", "/.well-known/oauth-authorization-server")


def ttl_from_headers(cache_control: str | None) -> float:
    """Seconds to keep a document: ``Cache-Control: max-age`` clamped to 60 s - 24 h, 300 s if absent."""
    if not cache_control:
        return DEFAULT_TTL
    lowered = cache_control.lower()
    if "no-store" in lowered or "no-cache" in lowered:
        return _MIN_TTL
    match = _MAX_AGE.search(cache_control)
    if not match:
        return DEFAULT_TTL
    return min(_MAX_TTL, max(_MIN_TTL, float(match.group(1))))


def _strings(doc: Mapping[str, Any], name: str) -> tuple[str, ...]:
    value = doc.get(name)
    if not isinstance(value, list):
        return ()
    items: list[object] = value  # pyright: ignore[reportUnknownVariableType]
    return tuple(v for v in items if isinstance(v, str))


def _optional_endpoint(issuer: str, doc: Mapping[str, Any], name: str) -> str | None:
    value = doc.get(name)
    if value is None:
        return None
    return require_endpoint(issuer, name, value)


@dataclass(frozen=True, slots=True)
class ServerMetadata:
    """The fields of the project's discovery document this SDK uses. ``raw`` holds the rest, read-only."""

    issuer: str
    token_endpoint: str
    jwks_uri: str
    authorization_endpoint: str | None = None
    userinfo_endpoint: str | None = None
    revocation_endpoint: str | None = None
    introspection_endpoint: str | None = None
    end_session_endpoint: str | None = None
    backchannel_authentication_endpoint: str | None = None
    grant_types_supported: tuple[str, ...] = ()
    token_endpoint_auth_methods_supported: tuple[str, ...] = ()
    token_endpoint_auth_signing_alg_values_supported: tuple[str, ...] = ()
    id_token_signing_alg_values_supported: tuple[str, ...] = ()
    code_challenge_methods_supported: tuple[str, ...] = ()
    acr_values_supported: tuple[str, ...] = ()
    scopes_supported: tuple[str, ...] = ()
    dpop_signing_alg_values_supported: tuple[str, ...] = ()
    raw: Mapping[str, Any] = field(default_factory=lambda: types.MappingProxyType({}), repr=False)

    def supports_grant(self, grant: str) -> bool:
        """``True`` when the document lists ``grant`` (an absent list means the RFC 8414 default)."""
        if not self.grant_types_supported:
            return grant in ("authorization_code", "implicit")
        return grant in self.grant_types_supported


def parse_metadata(issuer: str, doc: object) -> ServerMetadata:
    """``doc`` as :class:`ServerMetadata` for ``issuer``, or :class:`DiscoveryError` naming the fault."""
    if not isinstance(doc, dict):
        raise DiscoveryError(f"The discovery document of {issuer} is not a JSON object.", code="metadata_invalid")
    body: dict[str, Any] = doc  # pyright: ignore[reportUnknownVariableType]
    named = body.get("issuer")
    if named != issuer:
        raise DiscoveryError(
            f"The discovery document at {issuer} names issuer {named!r}, not {issuer!r} (RFC 8414 §3.3), "
            "so its endpoints are not this project's.",
            code="issuer_mismatch",
        )
    return ServerMetadata(
        issuer=issuer,
        token_endpoint=require_endpoint(issuer, "token_endpoint", body.get("token_endpoint")),
        jwks_uri=require_endpoint(issuer, "jwks_uri", body.get("jwks_uri")),
        authorization_endpoint=_optional_endpoint(issuer, body, "authorization_endpoint"),
        userinfo_endpoint=_optional_endpoint(issuer, body, "userinfo_endpoint"),
        revocation_endpoint=_optional_endpoint(issuer, body, "revocation_endpoint"),
        introspection_endpoint=_optional_endpoint(issuer, body, "introspection_endpoint"),
        end_session_endpoint=_optional_endpoint(issuer, body, "end_session_endpoint"),
        backchannel_authentication_endpoint=_optional_endpoint(issuer, body, "backchannel_authentication_endpoint"),
        grant_types_supported=_strings(body, "grant_types_supported"),
        token_endpoint_auth_methods_supported=_strings(body, "token_endpoint_auth_methods_supported"),
        token_endpoint_auth_signing_alg_values_supported=_strings(
            body, "token_endpoint_auth_signing_alg_values_supported"
        ),
        id_token_signing_alg_values_supported=_strings(body, "id_token_signing_alg_values_supported"),
        code_challenge_methods_supported=_strings(body, "code_challenge_methods_supported"),
        acr_values_supported=_strings(body, "acr_values_supported"),
        scopes_supported=_strings(body, "scopes_supported"),
        dpop_signing_alg_values_supported=_strings(body, "dpop_signing_alg_values_supported"),
        raw=types.MappingProxyType(dict(body)),
    )


def metadata_flow(issuer: str) -> Flow[tuple[ServerMetadata, float]]:
    """Fetch and check the discovery document; returns it with its cache lifetime in seconds."""
    for index, path in enumerate(_PATHS):
        response = yield from call(
            HttpRequest(
                "GET",
                issuer + path,
                what="the discovery document",
                headers={"Accept": "application/json"},
                retry=Retry.SAFE,
                max_bytes=DOCUMENT_LIMIT,
            )
        )
        if response.status == 404 and index + 1 < len(_PATHS):
            continue
        if response.status != 200:
            error = describe(response, f"The discovery document at {issuer}{path}")
            raise DiscoveryError(str(error), status=response.status)
        try:
            doc = response.json()
        except (ValueError, UnicodeDecodeError) as exc:
            raise DiscoveryError(
                f"The discovery document at {issuer}{path} is not JSON.", code="metadata_invalid"
            ) from exc
        return parse_metadata(issuer, doc), ttl_from_headers(response.header("Cache-Control"))
    raise DiscoveryError(f"{issuer} publishes no discovery document.")  # pragma: no cover - loop returns
