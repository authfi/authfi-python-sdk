"""AuthFI SDK: sign people in, protect APIs, machine clients and AI agents, on one OAuth core.

Start from the project::

    from authfi import AuthFI

    af = AuthFI("https://acme.authfi.io")       # or AuthFI.from_env(): AUTHFI_ISSUER

Every area is made from it and shares its discovery document, key set and HTTP client.
"""

import warnings
from typing import TYPE_CHECKING, Final

from authfi._protocol.agent import CIBA_GRANT as CIBA_GRANT_TYPE
from authfi._protocol.discovery import ServerMetadata
from authfi._protocol.ids import idempotency_key, request_id
from authfi._protocol.oauth import CLIENT_ASSERTION_TYPE
from authfi._transport import user_agent
from authfi._version import __version__
from authfi.admin import Admin, AsyncAdmin
from authfi.agents import (
    ActionDecision,
    AgentAuth,
    AgentKey,
    Approval,
    AsyncAgentAuth,
    AsyncDelegatedToken,
    CibaRequest,
    DelegatedToken,
    Signer,
    TransactionToken,
    assert_public_jwk,
)
from authfi.client import AsyncAuthFI, AuthFI, AuthFIOptions, Timeout
from authfi.errors import (
    APIError,
    AuthFIDeprecationWarning,
    AuthFIError,
    AuthFISecurityWarning,
    ConfigurationError,
    ConsentDenied,
    ConsentExpired,
    DiscoveryError,
    GatewayTokenRefused,
    InsufficientAuthentication,
    InsufficientPermission,
    InsufficientScope,
    IntegrationError,
    InvalidCallback,
    InvalidIdToken,
    InvalidPendingSignIn,
    InvalidToken,
    KnownCode,
    NotSupportedError,
    OAuthError,
    StepUpRequired,
    TemporarilyUnavailable,
    TransportError,
    WaitTimeout,
    WebhookVerificationError,
)
from authfi.gateway import (
    APPROVAL_ARGUMENT,
    HOLD_MESSAGE,
    AsyncGateway,
    CloudCredential,
    CloudCredentialOutcome,
    Gateway,
    ToolAllowed,
    ToolApprovalRequired,
    ToolOutcome,
    ToolRefused,
)
from authfi.machine import AsyncClientCredentials, ClientCredentials
from authfi.secret import Secret
from authfi.signin import (
    AsyncPersonSession,
    AsyncWebApp,
    IdTokenClaims,
    PendingSignIn,
    PersonSession,
    SignedIn,
    UserInfo,
    WebApp,
)
from authfi.tokens import (
    AsyncInMemoryTokenStore,
    AsyncTokenSource,
    AsyncTokenStore,
    InMemoryTokenStore,
    TokenSet,
    TokenSource,
    TokenStore,
)
from authfi.verify import (
    AccessToken,
    AsyncMultiIssuerVerifier,
    AsyncTokenVerifier,
    CheckResult,
    MultiIssuerVerifier,
    Principal,
    TokenVerifier,
)

#: The ``User-Agent`` this SDK sends (the AuthFI edge refuses an anonymous client).
USER_AGENT: Final = user_agent()

if TYPE_CHECKING:
    #: Deprecated: 0.1's name for :class:`OAuthError`.
    AgentAuthError = OAuthError

_DEPRECATED_ALIASES: dict[str, tuple[object, str]] = {
    "AgentAuthError": (OAuthError, "authfi.AgentAuthError is authfi.OAuthError"),
}


def __getattr__(name: str) -> object:
    """0.1's names that moved, each with a deprecation warning."""
    if name in _DEPRECATED_ALIASES:
        value, message = _DEPRECATED_ALIASES[name]
        warnings.warn(message, AuthFIDeprecationWarning, stacklevel=2)
        return value
    raise AttributeError(f"module 'authfi' has no attribute {name!r}")


__all__ = [
    "APPROVAL_ARGUMENT",
    "CIBA_GRANT_TYPE",
    "CLIENT_ASSERTION_TYPE",
    "HOLD_MESSAGE",
    "USER_AGENT",
    "APIError",
    "AccessToken",
    "ActionDecision",
    "Admin",
    "AgentAuth",
    "AgentKey",
    "Approval",
    "AsyncAdmin",
    "AsyncAgentAuth",
    "AsyncAuthFI",
    "AsyncClientCredentials",
    "AsyncDelegatedToken",
    "AsyncGateway",
    "AsyncInMemoryTokenStore",
    "AsyncMultiIssuerVerifier",
    "AsyncPersonSession",
    "AsyncTokenSource",
    "AsyncTokenStore",
    "AsyncTokenVerifier",
    "AsyncWebApp",
    "AuthFI",
    "AuthFIDeprecationWarning",
    "AuthFIError",
    "AuthFIOptions",
    "AuthFISecurityWarning",
    "CheckResult",
    "CibaRequest",
    "ClientCredentials",
    "CloudCredential",
    "CloudCredentialOutcome",
    "ConfigurationError",
    "ConsentDenied",
    "ConsentExpired",
    "DelegatedToken",
    "DiscoveryError",
    "Gateway",
    "GatewayTokenRefused",
    "IdTokenClaims",
    "InMemoryTokenStore",
    "InsufficientAuthentication",
    "InsufficientPermission",
    "InsufficientScope",
    "IntegrationError",
    "InvalidCallback",
    "InvalidIdToken",
    "InvalidPendingSignIn",
    "InvalidToken",
    "KnownCode",
    "MultiIssuerVerifier",
    "NotSupportedError",
    "OAuthError",
    "PendingSignIn",
    "PersonSession",
    "Principal",
    "Secret",
    "ServerMetadata",
    "SignedIn",
    "Signer",
    "StepUpRequired",
    "TemporarilyUnavailable",
    "Timeout",
    "TokenSet",
    "TokenSource",
    "TokenStore",
    "TokenVerifier",
    "ToolAllowed",
    "ToolApprovalRequired",
    "ToolOutcome",
    "ToolRefused",
    "TransactionToken",
    "TransportError",
    "UserInfo",
    "WaitTimeout",
    "WebApp",
    "WebhookVerificationError",
    "__version__",
    "assert_public_jwk",
    "idempotency_key",
    "request_id",
]
