"""Dynamic client registration (RFC 7591) for a public client that signs a person in from a command line.

The request and its answer only; the caller sends it. A registration the server answers with a redirect
that is not on this machine is refused, so a client is never left with a code sent somewhere else.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from authfi._protocol.errors import oauth_error
from authfi._protocol.http import HttpRequest, HttpResponse, Retry
from authfi.errors import OAuthError

__all__ = ["is_loopback_redirect", "parse_registration", "registration_request"]

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})


def is_loopback_redirect(uri: str) -> bool:
    """``True`` for ``http`` on the loopback address itself (RFC 8252 section 7.3); ``localhost`` is not accepted."""
    try:
        parts = urlsplit(uri)
        return parts.scheme == "http" and parts.hostname in _LOOPBACK_HOSTS and not parts.fragment
    except ValueError:
        return False


def registration_request(
    endpoint: str, *, client_name: str, redirect_uris: Sequence[str], scope: Sequence[str]
) -> HttpRequest:
    """The registration of a public client (``token_endpoint_auth_method: none``) using the code and refresh grants.

    Never retried after send: a registration that arrived made a client, and a second one would make another.
    """
    return HttpRequest(
        "POST",
        endpoint,
        what="the client registration endpoint",
        headers={"Accept": "application/json"},
        json_body={
            "client_name": client_name,
            "redirect_uris": list(redirect_uris),
            "grant_types": ["authorization_code", "refresh_token"],
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
            "scope": " ".join(scope),
        },
        retry=Retry.NEVER_AFTER_SEND,
    )


def parse_registration(response: HttpResponse, *, redirect_uris: Sequence[str]) -> str:
    """The new ``client_id``, or the typed refusal.

    The answer must keep the client public and every redirect it names must be one the client asked for.
    """
    if response.status not in (200, 201):
        raise oauth_error(response, "the client registration endpoint")
    body: dict[str, Any] | None = response.json_object()
    if body is None:
        raise OAuthError(
            "invalid_response", "the client registration endpoint did not answer a JSON object", status=response.status
        )
    client_id = body.get("client_id")
    if not isinstance(client_id, str) or not client_id:
        raise OAuthError(
            "invalid_response", "the client registration endpoint answered no client_id", status=response.status
        )
    if body.get("client_secret"):
        raise OAuthError(
            "invalid_response",
            "the client registration endpoint issued a secret to a public client",
            status=response.status,
        )
    bound = body.get("redirect_uris")
    if bound is not None:
        asked = set(redirect_uris)
        if not isinstance(bound, list) or any(
            not isinstance(uri, str) or uri not in asked or not is_loopback_redirect(uri)
            for uri in bound  # pyright: ignore[reportUnknownVariableType]
        ):
            raise OAuthError(
                "invalid_response",
                "the client registration endpoint bound redirect URIs this client did not ask for",
                status=response.status,
            )
    return client_id
