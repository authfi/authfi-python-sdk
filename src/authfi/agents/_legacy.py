"""0.1's ``AgentAuth``: its constructor, ``from_env`` and the methods 0.2 renamed, each deprecated.

The one default 0.1 had that 0.2 refuses is the project: ``platform_url`` defaulted to
``https://platform.authfi.io``, which sent an agent's signed assertion to a project where the agent
does not exist. Every other form keeps working, with a warning naming its replacement.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from typing_extensions import deprecated

from authfi._protocol.urls import origin_of
from authfi.agents.keys import AgentKey, Signer
from authfi.errors import AuthFIDeprecationWarning, AuthFISecurityWarning, ConfigurationError
from authfi.secret import Secret

if TYPE_CHECKING:
    from authfi._protocol.discovery import ServerMetadata
    from authfi.agents.agent import AgentAuth, CibaRequest
    from authfi.client import AuthFI

__all__ = ["LegacyAgentMixin", "is_legacy_agent", "key_from_env", "legacy_agent_arguments"]

_LEGACY_ORDER = (
    "agent_id",
    "key",
    "platform_url",
    "audience",
    "token_endpoint",
    "backchannel_endpoint",
    "discovery_url",
    "refresh_skew",
)


def _warn(message: str, stacklevel: int = 4) -> None:
    warnings.warn(message, AuthFIDeprecationWarning, stacklevel=stacklevel)


def is_legacy_agent(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> bool:
    """0.1's form: the agent id first, or ``agent_id=`` with no project."""
    if args:
        return isinstance(args[0], str)
    return "agent_id" in kwargs and "project" not in kwargs


def key_from_env(env: Mapping[str, str]) -> AgentKey:
    """The agent key the environment names: the file by default; a PEM only where there is no filesystem."""
    key_file = env.get("AUTHFI_AGENT_KEY_FILE", "").strip()
    pem = env.get("AUTHFI_AGENT_PRIVATE_KEY", "").strip()
    kid = env.get("AUTHFI_AGENT_KID", "").strip() or None
    if key_file:
        if pem:
            warnings.warn(
                "both AUTHFI_AGENT_KEY_FILE and AUTHFI_AGENT_PRIVATE_KEY are set; using the file",
                AuthFISecurityWarning,
                stacklevel=3,
            )
        return AgentKey.from_file(key_file, kid=kid)
    if pem:
        warnings.warn(
            "AUTHFI_AGENT_PRIVATE_KEY holds a private key in the environment, where every child process inherits it "
            "and crash dumps capture it. Mount it as a file and set AUTHFI_AGENT_KEY_FILE.",
            AuthFISecurityWarning,
            stacklevel=3,
        )
        if not kid:
            raise ConfigurationError("AUTHFI_AGENT_KID is required with AUTHFI_AGENT_PRIVATE_KEY")
        return AgentKey.from_pem(Secret(pem), kid=kid)
    raise ConfigurationError("no agent key: set AUTHFI_AGENT_KEY_FILE to the path of the agent's private key file")


def legacy_agent_arguments(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[AuthFI, str, Signer, dict[str, Any]]:
    """The project, agent id, key and new-style options of 0.1's ``AgentAuth(agent_id, key, ...)``."""
    from authfi.client import AuthFI

    given: dict[str, Any] = dict(zip(_LEGACY_ORDER, args, strict=False))
    for name in _LEGACY_ORDER:
        if name in kwargs:
            given[name] = kwargs.pop(name)
    agent_id = str(given.get("agent_id") or "").strip()
    if not agent_id:
        raise ConfigurationError("agent_id is required (agt_...)")
    key = given.get("key")
    if not isinstance(key, Signer):
        raise ConfigurationError(
            "key must be an AgentKey: build one with AgentKey.from_file(...). An agent authenticates with a key it "
            "holds, and there is no secret fallback."
        )
    platform_url = given.get("platform_url")
    if not platform_url:
        raise ConfigurationError(
            "the agent's project is required: af = AuthFI('https://<project>.authfi.io'); af.agent(agent_id, key). "
            "(0.1's platform_url had a default that sent the agent's signed assertion to another project.)"
        )
    _warn("AgentAuth(agent_id, key, platform_url=...) is AuthFI(platform_url).agent(agent_id, key)")
    for ignored in ("token_endpoint", "backchannel_endpoint", "discovery_url"):
        if given.get(ignored):
            _warn(f"{ignored} is not used: endpoints are read from the project's discovery document")
    audience = given.get("audience")
    options: dict[str, Any] = {"resource": audience}
    if audience and str(audience).startswith(("https://", "http://")):
        # 0.1 had no send_to. A URL audience names where its token goes; a client id does not.
        options["send_to"] = (origin_of(str(audience)),)
    if "refresh_skew" in given:
        options["refresh_skew"] = float(given["refresh_skew"])
    return AuthFI(str(platform_url).rstrip("/")), agent_id, key, options


class LegacyAgentMixin:
    """0.1's ``AgentAuth`` names on the 0.2 agent. Each is deprecated."""

    if TYPE_CHECKING:  # what the mixin reads from AgentAuth
        project: AuthFI
        agent_id: str
        key: Signer
        resource: str | None

        def token(self, *, resource: str | None = None, force: bool = False) -> str: ...
        def capabilities(self, *, resource: str | None = None) -> frozenset[str]: ...
        def request_user_token(
            self,
            login_hint: str,
            *,
            binding_message: str,
            scope: Sequence[str] | None = None,
            expires_in: int | None = None,
        ) -> CibaRequest: ...

    @classmethod
    @deprecated("AgentAuth.from_env() is AuthFI.from_env().agent_from_env()", category=AuthFIDeprecationWarning)
    def from_env(cls, **overrides: Any) -> AgentAuth:
        """0.1's environment form. ``AUTHFI_PLATFORM_URL`` and ``AUTHFI_AGENT_AUDIENCE`` are read as fallbacks."""
        from authfi.client import AuthFI

        env = os.environ
        agent_id = env.get("AUTHFI_AGENT_ID", "").strip()
        if not agent_id:
            raise ConfigurationError("AUTHFI_AGENT_ID is not set")
        key = key_from_env(env)
        issuer = overrides.pop("platform_url", None) or env.get("AUTHFI_ISSUER") or env.get("AUTHFI_PLATFORM_URL")
        if not issuer:
            raise ConfigurationError("AUTHFI_ISSUER is not set: set it to your project's address")
        if not env.get("AUTHFI_ISSUER") and env.get("AUTHFI_PLATFORM_URL"):
            _warn("AUTHFI_PLATFORM_URL is AUTHFI_ISSUER", stacklevel=3)
        resource = (
            overrides.pop("audience", None) or env.get("AUTHFI_AGENT_RESOURCE") or env.get("AUTHFI_AGENT_AUDIENCE")
        )
        if not env.get("AUTHFI_AGENT_RESOURCE") and env.get("AUTHFI_AGENT_AUDIENCE"):
            _warn("AUTHFI_AGENT_AUDIENCE is AUTHFI_AGENT_RESOURCE", stacklevel=3)
        return AuthFI(str(issuer).rstrip("/")).agent(agent_id, key, resource=resource, **overrides)

    @property
    @deprecated("AgentAuth.audience is AgentAuth.resource", category=AuthFIDeprecationWarning)
    def audience(self) -> str | None:
        """0.1's name for :attr:`resource`."""
        return self.resource

    @property
    @deprecated("AgentAuth.platform_url is AgentAuth.project.issuer", category=AuthFIDeprecationWarning)
    def platform_url(self) -> str:
        """0.1's name for the project's issuer."""
        return self.project.issuer

    def _metadata(self) -> ServerMetadata:
        return self.project.metadata()

    @property
    @deprecated("AgentAuth.token_endpoint is AuthFI.metadata().token_endpoint", category=AuthFIDeprecationWarning)
    def token_endpoint(self) -> str:
        """The project's token endpoint."""
        return self._metadata().token_endpoint

    @property
    @deprecated(
        "AgentAuth.backchannel_endpoint is AuthFI.metadata().backchannel_authentication_endpoint",
        category=AuthFIDeprecationWarning,
    )
    def backchannel_endpoint(self) -> str | None:
        """The project's CIBA endpoint."""
        return self._metadata().backchannel_authentication_endpoint

    @deprecated("AgentAuth.discovery() is AuthFI.metadata()", category=AuthFIDeprecationWarning)
    def discovery(self) -> dict[str, Any]:
        """The project's discovery document, as a dict."""
        return dict(self._metadata().raw)

    @deprecated("AgentAuth.auth_header() is AgentAuth.headers(method, url)", category=AuthFIDeprecationWarning)
    def auth_header(self, audience: str | None = None) -> dict[str, str]:
        """``{"Authorization": "Bearer ..."}`` with the current token."""
        return {"Authorization": "Bearer " + self.token(resource=audience)}

    @deprecated("AgentAuth.can(name) is name in AgentAuth.capabilities()", category=AuthFIDeprecationWarning)
    def can(self, capability: str) -> bool:
        """``True`` when the agent holds ``capability``."""
        return capability in self.capabilities()

    @deprecated("AgentAuth.request_user_approval is AgentAuth.request_user_token", category=AuthFIDeprecationWarning)
    def request_user_approval(
        self,
        login_hint: str,
        binding_message: str | None = None,
        scope: str | Sequence[str] | None = None,
        requested_expiry: int | None = None,
    ) -> CibaRequest:
        """0.1's CIBA request. ``binding_message`` is now required."""
        if not binding_message:
            raise ConfigurationError(
                "binding_message is required: it is what the person reads on their own device before approving"
            )
        scopes = scope.split() if isinstance(scope, str) else scope
        return self.request_user_token(
            login_hint, binding_message=binding_message, scope=scopes, expires_in=requested_expiry
        )
