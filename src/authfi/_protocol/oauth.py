"""The OAuth wire at ``<issuer>/oauth/token``: client authentication, grants, the token answer. No I/O.

People, machine clients and agents all get tokens from the one token endpoint, so one module
builds every grant:

- ``authorization_code`` and ``refresh_token`` for a person (never retried after send: a code is
  single-use, and a reused refresh token revokes its whole family);
- ``client_credentials`` for a machine client or an agent (safe to retry, because the client
  authentication is made fresh for every attempt);
- revocation.

The agent module adds CIBA and token exchange on the same builders.
"""

from __future__ import annotations

import base64
import hashlib
import os
import types
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote_plus

from authfi._protocol.errors import oauth_error
from authfi._protocol.http import Attempt, HttpRequest, HttpResponse, Retry
from authfi.errors import ConfigurationError, OAuthError
from authfi.secret import Secret

__all__ = [
    "ACCESS_TOKEN_TYPE",
    "CLIENT_ASSERTION_TYPE",
    "ClientAuth",
    "ClientAuthMethod",
    "TokenResponse",
    "authorization_code_request",
    "b64url",
    "client_credentials_request",
    "parse_token_response",
    "pkce_pair",
    "random_value",
    "refresh_request",
    "revocation_request",
    "token_request",
]

#: RFC 7523 §2.2: the client assertion type of ``private_key_jwt``.
CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
#: RFC 8693 §3: the token type of an OAuth access token.
ACCESS_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"  # noqa: S105 - a URN, not a secret

ClientAuthMethod = Literal["none", "client_secret_basic", "client_secret_post", "private_key_jwt"]

#: The token response members held as :class:`Secret` and never kept in ``extra``.
_SECRET_MEMBERS = frozenset({"access_token", "refresh_token", "id_token"})


def b64url(raw: bytes) -> str:
    """Base64url without padding (RFC 7515 §2)."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def random_value(nbytes: int = 32) -> str:
    """``nbytes`` random bytes, base64url: a ``state``, ``nonce`` or PKCE verifier."""
    return b64url(os.urandom(nbytes))


def pkce_pair() -> tuple[Secret, str]:
    """A PKCE verifier (32 random bytes, held as a :class:`Secret`) and its S256 challenge."""
    verifier = random_value(32)
    challenge = b64url(hashlib.sha256(verifier.encode("ascii")).digest())
    return Secret(verifier), challenge


@dataclass(frozen=True, slots=True, repr=False)
class ClientAuth:
    """How a client authenticates at the token endpoint.

    Build it with :meth:`none`, :meth:`secret_basic`, :meth:`secret_post` or
    :meth:`private_key_jwt`. ``private_key_jwt`` takes a function that signs a fresh assertion
    for an audience; it is called once per attempt, so a retry never replays an assertion.
    """

    method: ClientAuthMethod
    client_id: str
    secret: Secret | None = None
    assertion: Callable[[str], str] | None = None

    @classmethod
    def none(cls, client_id: str) -> ClientAuth:
        """A public client: ``client_id`` only, as PKCE requires."""
        return cls("none", _client_id(client_id))

    @classmethod
    def secret_basic(cls, client_id: str, secret: Secret) -> ClientAuth:
        """``client_secret_basic``: the id and secret in a Basic header (RFC 6749 §2.3.1)."""
        return cls("client_secret_basic", _client_id(client_id), _secret(secret))

    @classmethod
    def secret_post(cls, client_id: str, secret: Secret) -> ClientAuth:
        """``client_secret_post``: the id and secret in the form."""
        return cls("client_secret_post", _client_id(client_id), _secret(secret))

    @classmethod
    def private_key_jwt(cls, client_id: str, assertion: Callable[[str], str]) -> ClientAuth:
        """``private_key_jwt`` (RFC 7523 §2.2): ``assertion(audience)`` signs a fresh assertion."""
        return cls("private_key_jwt", _client_id(client_id), assertion=assertion)

    def attempt(self, audience: str) -> Attempt:
        """The form fields and headers that authenticate one attempt. ``audience`` is the endpoint."""
        if self.method == "none":
            return Attempt(form={"client_id": self.client_id})
        if self.method == "private_key_jwt":
            if self.assertion is None:
                raise ConfigurationError("private_key_jwt needs a function that signs the assertion")
            return Attempt(
                form={
                    "client_id": self.client_id,
                    "client_assertion_type": CLIENT_ASSERTION_TYPE,
                    "client_assertion": self.assertion(audience),
                }
            )
        if self.secret is None:
            raise ConfigurationError(f"{self.method} needs the client secret")
        if self.method == "client_secret_post":
            return Attempt(form={"client_id": self.client_id, "client_secret": self.secret.reveal()})
        pair = f"{quote_plus(self.client_id)}:{quote_plus(self.secret.reveal())}"
        return Attempt(headers={"Authorization": "Basic " + base64.b64encode(pair.encode()).decode("ascii")})

    def __repr__(self) -> str:
        return f"ClientAuth({self.method}, client_id={self.client_id!r})"


def _client_id(value: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ConfigurationError("client_id is required")
    return text


def _secret(value: Secret) -> Secret:
    if not isinstance(value, Secret):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ConfigurationError(
            "the client secret must be a Secret: wrap it, Secret(os.environ['AUTHFI_CLIENT_SECRET']), so "
            "it never prints"
        )
    if not value:
        raise ConfigurationError("the client secret is empty")
    return value


@dataclass(frozen=True, slots=True, repr=False)
class TokenResponse:
    """A token endpoint's answer (RFC 6749 §5.1). Tokens are :class:`Secret`; ``extra`` holds the rest."""

    access_token: Secret
    token_type: str
    expires_in: int | None
    refresh_token: Secret | None = None
    id_token: Secret | None = None
    scope: frozenset[str] | None = None
    issued_token_type: str | None = None
    extra: Mapping[str, Any] = field(default_factory=lambda: types.MappingProxyType({}))

    def __repr__(self) -> str:
        return f"TokenResponse(token_type={self.token_type!r}, expires_in={self.expires_in!r})"


def _opt_secret(body: Mapping[str, Any], name: str) -> Secret | None:
    value = body.get(name)
    return Secret(value) if isinstance(value, str) and value else None


def parse_token_response(
    response: HttpResponse,
    what: str,
    *,
    grant: str | None = None,
    step_up_acr: Sequence[str] = (),
) -> TokenResponse:
    """The answer as a :class:`TokenResponse`, or the typed refusal."""
    if response.status != 200:
        raise oauth_error(response, what, grant=grant, step_up_acr=step_up_acr)
    body = response.json_object()
    if body is None:
        raise OAuthError("invalid_response", f"{what} did not answer a JSON object", status=response.status)
    access = body.get("access_token")
    if not isinstance(access, str) or not access:
        raise OAuthError("invalid_response", f"{what} answered no access_token", status=response.status)
    token_type = body.get("token_type")
    expires_in = body.get("expires_in")
    if isinstance(expires_in, str) and expires_in.isdigit():
        expires_in = int(expires_in)
    scope = body.get("scope")
    issued = body.get("issued_token_type")
    return TokenResponse(
        access_token=Secret(access),
        token_type=token_type if isinstance(token_type, str) else "Bearer",
        expires_in=expires_in if isinstance(expires_in, int) and expires_in >= 0 else None,
        refresh_token=_opt_secret(body, "refresh_token"),
        id_token=_opt_secret(body, "id_token"),
        scope=frozenset(scope.split()) if isinstance(scope, str) else None,
        issued_token_type=issued if isinstance(issued, str) else None,
        extra=types.MappingProxyType({k: v for k, v in body.items() if k not in _SECRET_MEMBERS}),
    )


def token_request(
    endpoint: str,
    form: Mapping[str, str],
    auth: ClientAuth | None,
    *,
    retry: Retry,
    what: str = "the token endpoint",
    headers: Mapping[str, str] | None = None,
) -> HttpRequest:
    """A form POST to ``endpoint`` with ``auth`` made fresh per attempt (``None``: no client authentication)."""
    return HttpRequest(
        "POST",
        endpoint,
        what=what,
        headers={"Accept": "application/json", **(headers or {})},
        form=dict(form),
        retry=retry,
        per_attempt=(lambda: auth.attempt(endpoint)) if auth is not None else None,
    )


def _space(values: Sequence[str] | None) -> str | None:
    if values is None:
        return None
    if isinstance(values, str):
        raise ConfigurationError("scope is a sequence of names, such as ('openid', 'profile'), not one string")
    joined = " ".join(v.strip() for v in values if v.strip())
    return joined or None


def client_credentials_request(
    token_endpoint: str,
    auth: ClientAuth,
    *,
    resource: str | None = None,
    scope: Sequence[str] | None = None,
) -> HttpRequest:
    """``grant_type=client_credentials`` (RFC 6749 §4.4), optionally for one ``resource`` (RFC 8707)."""
    form = {"grant_type": "client_credentials"}
    if resource:
        form["resource"] = resource
    scope_text = _space(scope)
    if scope_text:
        form["scope"] = scope_text
    return token_request(token_endpoint, form, auth, retry=Retry.SAFE)


def authorization_code_request(
    token_endpoint: str,
    auth: ClientAuth,
    *,
    code: str,
    redirect_uri: str,
    code_verifier: Secret,
) -> HttpRequest:
    """``grant_type=authorization_code`` with PKCE. Never retried after send: a code is single-use."""
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": code_verifier.reveal(),
    }
    return token_request(token_endpoint, form, auth, retry=Retry.NEVER_AFTER_SEND)


def refresh_request(
    token_endpoint: str,
    auth: ClientAuth,
    *,
    refresh_token: Secret,
    scope: Sequence[str] | None = None,
) -> HttpRequest:
    """``grant_type=refresh_token``. Never retried after send: reusing a refresh token revokes its family."""
    form = {"grant_type": "refresh_token", "refresh_token": refresh_token.reveal()}
    scope_text = _space(scope)
    if scope_text:
        form["scope"] = scope_text
    return token_request(token_endpoint, form, auth, retry=Retry.NEVER_AFTER_SEND)


def revocation_request(
    revocation_endpoint: str,
    auth: ClientAuth,
    *,
    token: Secret,
    token_type_hint: Literal["refresh_token", "access_token"] = "refresh_token",  # noqa: S107 - a hint name
) -> HttpRequest:
    """RFC 7009 revocation. Safe to repeat: revoking twice changes nothing."""
    form = {"token": token.reveal(), "token_type_hint": token_type_hint}
    return token_request(revocation_endpoint, form, auth, retry=Retry.SAFE, what="the revocation endpoint")
