"""The SDK's errors: one base class, read from the platform's three envelopes.

Branch on :attr:`AuthFIError.code`, never on the message. A code is an open set
(``KnownCode | str``): a ``match`` on it keeps a ``case _:`` arm, because the platform's code
families grow between SDK releases.

A refused tool call and a held one are not exceptions: the gateway answers them as a tool result,
read with :meth:`authfi.Gateway.read_result`.
"""

from __future__ import annotations

from collections.abc import Sequence

from typing_extensions import deprecated

from authfi._codes import KnownCode

__all__ = [
    "APIError",
    "AuthFIDeprecationWarning",
    "AuthFIError",
    "AuthFISecurityWarning",
    "ConfigurationError",
    "ConsentDenied",
    "ConsentExpired",
    "DiscoveryError",
    "GatewayTokenRefused",
    "InsufficientAuthentication",
    "InsufficientPermission",
    "InsufficientScope",
    "IntegrationError",
    "InvalidCallback",
    "InvalidIdToken",
    "InvalidPendingSignIn",
    "InvalidToken",
    "KnownCode",
    "NotSupportedError",
    "OAuthError",
    "StepUpRequired",
    "TemporarilyUnavailable",
    "TransportError",
    "WebhookVerificationError",
]


class AuthFIDeprecationWarning(DeprecationWarning):
    """A name or form this SDK keeps for one more minor version. The message names its successor."""


class AuthFISecurityWarning(UserWarning):
    """A configuration that works but weakens a credential's protection, such as a key in the environment."""


class AuthFIError(Exception):
    """The base of every error this SDK raises.

    Attributes:
        code: the stable code to branch on (``KnownCode | str``).
        status: the HTTP status the platform answered, or ``None`` when no request was answered.
        request_id: the ``X-Request-Id`` (or ``AuthFI-Request-Id``) of the call, for support.
        run_id: the ``AuthFI-Run-Id`` the platform answered with, when it sent one.
        dropped_header: a header this call sent that an edge on the path removed, when the answer
            shows it (``DPoP``, ``Idempotency-Key``, ``traceparent``...).

    """

    code: KnownCode | str
    status: int | None
    request_id: str | None
    run_id: str | None
    dropped_header: str | None

    def __init__(
        self,
        message: str,
        status: int | None = None,
        *,
        code: KnownCode | str = "authfi_error",
        request_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_id = request_id
        self.run_id = run_id
        self.dropped_header = None

    @property
    def message(self) -> str:
        """The human-readable message. Never carries a credential."""
        return str(self.args[0]) if self.args else ""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(code={self.code!r}, status={self.status!r}, message={self.message!r})"


class ConfigurationError(AuthFIError):
    """A setting or an argument is wrong. No request was made, so there is no status."""

    def __init__(self, message: str, *, code: str = "configuration_error") -> None:
        super().__init__(message, None, code=code)


class IntegrationError(ConfigurationError):
    """A framework adapter cannot attach to the installed framework version.

    Raised at construction, naming the tested version range, rather than letting the adapter run in
    a way that would send the wrong credential.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message, code="integration_unsupported")


class NotSupportedError(AuthFIError):
    """The AuthFI project does not offer this to this kind of client.

    Attributes:
        feature: what is not offered (``"private_key_jwt"``, ``"agent_approval_status"``...).

    """

    feature: str

    def __init__(self, feature: str, message: str, status: int | None = None) -> None:
        super().__init__(message, status, code="not_supported")
        self.feature = feature


class TransportError(AuthFIError):
    """The request did not get a usable answer.

    A timeout, a connection or TLS failure, a redirect, an oversized body, or a server error that
    outlasted the retries.

    Attributes:
        retryable: ``True`` when calling again is safe and may succeed: the request never left, or
            it is safe to repeat.

    """

    retryable: bool

    def __init__(
        self,
        message: str,
        *,
        code: str = "transport_error",
        retryable: bool = False,
        status: int | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message, status, code=code, request_id=request_id)
        self.retryable = retryable


class DiscoveryError(AuthFIError):
    """The project's discovery document or key set is missing, malformed or names another issuer."""

    def __init__(self, message: str, *, code: str = "discovery_failed", status: int | None = None) -> None:
        super().__init__(message, status, code=code)


class OAuthError(AuthFIError):
    """An OAuth endpoint refused the request (RFC 6749 §5.2).

    ``code`` is the response's ``error``; ``description`` its ``error_description``, which is prose
    for people and never something to parse.
    """

    description: str | None

    def __init__(
        self,
        code: str,
        description: str | None = None,
        status: int | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        message = code if not description else f"{code}: {description}"
        super().__init__(message, status, code=code, request_id=request_id)
        self.description = description

    @property
    @deprecated("OAuthError.error is OAuthError.code", category=AuthFIDeprecationWarning)
    def error(self) -> str:
        """The OAuth error code. Deprecated: read :attr:`code`."""
        return self.code


class StepUpRequired(OAuthError):
    """The person must authenticate more strongly (``insufficient_user_authentication``, RFC 9470).

    Attributes:
        acr_values: the authentication context classes to ask for when signing them in again.
        max_age: the most seconds since their last authentication, when one is required.

    """

    acr_values: tuple[str, ...]
    max_age: int | None

    def __init__(
        self,
        description: str | None = None,
        *,
        acr_values: Sequence[str] = (),
        max_age: int | None = None,
        status: int | None = 401,
        request_id: str | None = None,
    ) -> None:
        super().__init__("insufficient_user_authentication", description, status=status, request_id=request_id)
        self.acr_values = tuple(acr_values)
        self.max_age = max_age


class ConsentDenied(OAuthError):
    """The person said no to an agent acting for them (CIBA ``access_denied``)."""


class ConsentExpired(OAuthError):
    """The person's consent ran out: the CIBA request expired, or a delegated token outlived it."""


class TemporarilyUnavailable(OAuthError):
    """The platform could not decide now and failed closed (HTTP 503). Calling again later may work."""

    @property
    def retryable(self) -> bool:
        """Always ``True``: the refusal is about now, not about the request."""
        return True


class APIError(AuthFIError):
    """An AuthFI API refused the request: ``{"error": code, "error_description": ...}``."""

    description: str | None

    def __init__(
        self,
        code: str,
        description: str | None = None,
        status: int | None = None,
        *,
        request_id: str | None = None,
    ) -> None:
        message = code if not description else f"{code}: {description}"
        super().__init__(message, status, code=code, request_id=request_id)
        self.description = description


class InvalidToken(AuthFIError):
    """A token was refused by this SDK's verifier.

    ``code`` is the kernel's refusal code (``token_expired``, ``token_audience_mismatch``...). It
    goes to the log, never to the caller: an API answers every refusal with the one opaque
    ``invalid_token``, so a caller cannot tell a wrong key from a wrong audience.
    """

    def __init__(self, code: KnownCode | str, detail: str = "") -> None:
        message = f"The access token was refused ({code})" + (f": {detail}" if detail else "")
        super().__init__(message, 401, code=code)


class InsufficientPermission(InvalidToken):
    """The token is valid but lacks the permissions or roles the call requires.

    Attributes:
        required: what was required, for the log. An API never sends it to the caller.

    """

    required: tuple[str, ...]

    def __init__(self, required: Sequence[str], detail: str = "") -> None:
        super().__init__("forbidden", detail)
        self.status = 403
        self.required = tuple(required)


class InsufficientScope(InvalidToken):
    """The token is valid but lacks the scopes the call requires.

    Attributes:
        required: the scopes to ask for. An API names them in its challenge (RFC 6750 §3.1).

    """

    required: tuple[str, ...]

    def __init__(self, required: Sequence[str], detail: str = "") -> None:
        super().__init__("insufficient_scope", detail)
        self.status = 403
        self.required = tuple(required)


class InsufficientAuthentication(InvalidToken):
    """The person behind the token did not authenticate strongly or recently enough (RFC 9470).

    Attributes:
        acr_values: the authentication context classes that would satisfy the call.
        max_age: the most seconds since their authentication, when one is required.

    """

    acr_values: tuple[str, ...]
    max_age: int | None

    def __init__(self, acr_values: Sequence[str], max_age: int | None = None, detail: str = "") -> None:
        super().__init__("insufficient_user_authentication", detail)
        self.acr_values = tuple(acr_values)
        self.max_age = max_age


class GatewayTokenRefused(AuthFIError):
    """The gateway refused the agent's token itself (HTTP 401).

    The token is missing, expired, revoked or for another audience. No call can succeed with it,
    so the run ends here and nothing retries.

    Attributes:
        challenge: the gateway's ``WWW-Authenticate`` value.

    """

    challenge: str | None

    def __init__(self, url: str, challenge: str | None, *, request_id: str | None = None) -> None:
        super().__init__(
            f"The AuthFI gateway refused this credential at {url}. No call can succeed with it; "
            "the agent may have been revoked.",
            401,
            code="gateway_token_refused",
            request_id=request_id,
        )
        self.challenge = challenge


class InvalidPendingSignIn(AuthFIError):
    """A pending sign-in could not be restored: tampered with, made for another client, or expired.

    ``code`` is ``pending_sign_in_tampered``, ``pending_sign_in_expired`` or ``pending_sign_in_malformed``.
    Start the sign-in again.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(
            f"The pending sign-in was refused ({code}): {detail}. Start the sign-in again.", 400, code=code
        )


class InvalidCallback(AuthFIError):
    """A sign-in callback was refused before its code was used.

    ``code`` is ``callback_iss_missing``, ``callback_iss_mismatch`` (RFC 9207, a mix-up),
    ``callback_state_mismatch`` or ``callback_code_missing``.
    """

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"The sign-in callback was refused ({code}): {detail}", 400, code=code)


class InvalidIdToken(AuthFIError):
    """An ID token was refused (``id_token_nonce_mismatch``, ``id_token_audience_mismatch``...)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"The ID token was refused ({code}): {detail}", 401, code=code)


class WebhookVerificationError(AuthFIError):
    """A webhook's signature, timestamp or body did not verify."""

    def __init__(self, message: str, *, code: str = "webhook_invalid") -> None:
        super().__init__(message, None, code=code)
