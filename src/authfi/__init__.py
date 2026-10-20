"""AuthFI SDK: sign people in, protect APIs, machine clients and AI agents, on one OAuth core.

Start from the project::

    from authfi import AuthFI

    af = AuthFI("https://acme.authfi.io")       # or AuthFI.from_env(): AUTHFI_ISSUER

Every area is made from it and shares its discovery document, key set and HTTP client.
"""

from authfi._protocol.discovery import ServerMetadata
from authfi._protocol.ids import idempotency_key, request_id
from authfi._version import __version__
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
    InvalidToken,
    KnownCode,
    NotSupportedError,
    OAuthError,
    StepUpRequired,
    TemporarilyUnavailable,
    TransportError,
    WebhookVerificationError,
)
from authfi.machine import AsyncClientCredentials, ClientCredentials
from authfi.secret import Secret
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

__all__ = [
    "APIError",
    "AccessToken",
    "AsyncAuthFI",
    "AsyncClientCredentials",
    "AsyncInMemoryTokenStore",
    "AsyncMultiIssuerVerifier",
    "AsyncTokenSource",
    "AsyncTokenStore",
    "AsyncTokenVerifier",
    "AuthFI",
    "AuthFIDeprecationWarning",
    "AuthFIError",
    "AuthFIOptions",
    "AuthFISecurityWarning",
    "CheckResult",
    "ClientCredentials",
    "ConfigurationError",
    "ConsentDenied",
    "ConsentExpired",
    "DiscoveryError",
    "GatewayTokenRefused",
    "InMemoryTokenStore",
    "InsufficientAuthentication",
    "InsufficientPermission",
    "InsufficientScope",
    "IntegrationError",
    "InvalidToken",
    "KnownCode",
    "MultiIssuerVerifier",
    "NotSupportedError",
    "OAuthError",
    "Principal",
    "Secret",
    "ServerMetadata",
    "StepUpRequired",
    "TemporarilyUnavailable",
    "Timeout",
    "TokenSet",
    "TokenSource",
    "TokenStore",
    "TokenVerifier",
    "TransportError",
    "WebhookVerificationError",
    "__version__",
    "idempotency_key",
    "request_id",
]
