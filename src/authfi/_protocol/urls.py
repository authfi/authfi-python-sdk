"""Issuer, origin and endpoint rules. Pure functions.

- An issuer is ``https://host[:port][/path]`` with no query, fragment or trailing slash. Plain
  ``http`` is accepted for loopback only, so a typo cannot send a credential in clear.
- An origin is a scheme, host and port, and nothing else.
- Every endpoint a discovery document names must itself be ``https`` (or loopback ``http``).
"""

from __future__ import annotations

import ipaddress
from urllib.parse import SplitResult, urlsplit

from authfi.errors import ConfigurationError, DiscoveryError

__all__ = [
    "is_loopback",
    "is_under",
    "origin_of",
    "require_endpoint",
    "validate_issuer",
    "validate_origin",
]

_LOOPBACK_NAMES = frozenset({"localhost"})


def is_loopback(host: str | None) -> bool:
    """``True`` for ``localhost``, ``127.0.0.0/8`` and ``::1``."""
    if not host:
        return False
    name = host.strip("[]").lower()
    if name in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def _scheme_allowed(parts: SplitResult) -> bool:
    if parts.scheme == "https":
        return True
    return parts.scheme == "http" and is_loopback(parts.hostname)


def _default_port(scheme: str) -> int:
    return 443 if scheme == "https" else 80


def _netloc(parts: SplitResult) -> str:
    host = (parts.hostname or "").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parts.port
    if port is not None and port != _default_port(parts.scheme):
        return f"{host}:{port}"
    return host


def origin_of(url: str) -> str:
    """The origin of ``url`` (``scheme://host[:port]``), lower-cased, default port dropped."""
    parts = urlsplit(url)
    return f"{parts.scheme.lower()}://{_netloc(parts)}"


def validate_issuer(value: str) -> str:
    """``value`` as an issuer, or :class:`ConfigurationError` saying what to pass."""
    text = (value or "").strip()
    if not text:
        raise ConfigurationError(
            "issuer is required: pass your project's own address, for example "
            'issuer="https://acme.authfi.io", or set AUTHFI_ISSUER.'
        )
    if "://" not in text:
        raise ConfigurationError(
            f"issuer must be your project's address with its scheme, such as https://{text}.authfi.io, not {text!r}."
        )
    parts = urlsplit(text)
    if not parts.hostname:
        raise ConfigurationError(f"issuer {text!r} names no host.")
    if not _scheme_allowed(parts):
        raise ConfigurationError(
            f"issuer {text!r} must use https. Plain http is accepted only for localhost, 127.0.0.1 "
            "and ::1, so a credential is never sent in clear."
        )
    if parts.query or parts.fragment or text.endswith("?") or text.endswith("#"):
        raise ConfigurationError(f"issuer {text!r} must have no query or fragment (RFC 8414 §2).")
    if text.endswith("/"):
        raise ConfigurationError(
            f"issuer {text!r} must have no trailing slash: tokens carry it exactly, as {text.rstrip('/')!r}."
        )
    if parts.username or parts.password:
        raise ConfigurationError(f"issuer {text!r} must not carry credentials.")
    return f"{parts.scheme.lower()}://{_netloc(parts)}{parts.path}"


def validate_origin(value: str, what: str = "send_to") -> str:
    """``value`` as an exact origin (scheme, host, port; no path), or :class:`ConfigurationError`."""
    text = (value or "").strip()
    parts = urlsplit(text)
    if not parts.hostname or parts.scheme not in ("https", "http"):
        raise ConfigurationError(f"{what} value {text!r} is not an origin like https://api.example.com.")
    if not _scheme_allowed(parts):
        raise ConfigurationError(f"{what} value {text!r} must use https; plain http is accepted only for loopback.")
    if parts.path not in ("", "/") or parts.query or parts.fragment or parts.username or parts.password:
        raise ConfigurationError(
            f"{what} value {text!r} must be an origin only (scheme, host and port), with no path, query or fragment."
        )
    return origin_of(text)


def require_endpoint(doc_issuer: str, name: str, value: object) -> str:
    """A discovery document's endpoint ``name``, checked to be an absolute https URL."""
    if not isinstance(value, str) or not value:
        raise DiscoveryError(f"The discovery document of {doc_issuer} has no {name}.", code="metadata_invalid")
    parts = urlsplit(value)
    if not parts.hostname or not _scheme_allowed(parts) or parts.fragment:
        raise DiscoveryError(
            f"The discovery document of {doc_issuer} names {name} {value!r}, which is not an https URL.",
            code="metadata_invalid",
        )
    return value


def is_under(url: str, base: str) -> bool:
    """``True`` when ``url`` is ``base`` or a URL under it (same origin, path below base's)."""
    try:
        u, b = urlsplit(url), urlsplit(base)
    except ValueError:
        return False
    if origin_of(url) != origin_of(base):
        return False
    base_path = b.path.rstrip("/")
    return u.path.rstrip("/") == base_path or u.path.startswith(base_path + "/")
