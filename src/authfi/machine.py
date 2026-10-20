"""Machine clients: a backend calling an API with its own token (client credentials, RFC 6749 §4.4).

::

    m2m = af.client_credentials(
        "app_reports", Secret(os.environ["AUTHFI_CLIENT_SECRET"]),
        resource="api_orders",                     # the API's client id: the token's audience
        send_to=["https://api.example.com"],       # the only origin the token may go to
    )
    httpx.Client(auth=m2m.httpx_auth()).get("https://api.example.com/orders")

The token is cached and renewed at two thirds of its lifetime; concurrent callers share one mint.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from authfi._cache import AsyncCache, SyncCache
from authfi._log import warn_once
from authfi._protocol.discovery import ServerMetadata
from authfi._protocol.oauth import ClientAuth, client_credentials_request, parse_token_response
from authfi.errors import ConfigurationError, NotSupportedError
from authfi.secret import Secret
from authfi.tokens import AsyncTokenSource, Destinations, TokenSource

if TYPE_CHECKING:
    from authfi._protocol.oauth import TokenResponse
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = ["AsyncClientCredentials", "ClientCredentials"]

SecretAuthMethod = Literal["client_secret_basic", "client_secret_post"]

#: A token answer with no lifetime is renewed after this many seconds.
_UNKNOWN_LIFETIME = 60.0
_GRANT = "client_credentials"


def lifetime_for(response: TokenResponse) -> float:
    """How long to keep a token: two thirds of its lifetime."""
    if response.expires_in is None:
        return _UNKNOWN_LIFETIME
    return max(1.0, response.expires_in * 2 / 3)


def check_grant(metadata: ServerMetadata, auth: ClientAuth) -> None:
    """Refuse up front a grant or client authentication the project does not offer."""
    if not metadata.supports_grant(_GRANT):
        raise NotSupportedError(_GRANT, "This AuthFI project does not offer the client credentials grant.")
    methods = metadata.token_endpoint_auth_methods_supported
    if methods and auth.method not in methods:
        raise NotSupportedError(auth.method, f"This AuthFI project does not accept {auth.method}.")


class _Config:
    def __init__(
        self,
        project: AuthFI | AsyncAuthFI,
        client_id: str,
        client_secret: Secret,
        *,
        resource: str | None,
        send_to: Sequence[str],
        scope: Sequence[str] | None,
        auth_method: str,
        warn_without_resource: bool,
    ) -> None:
        if auth_method == "private_key_jwt":
            raise NotSupportedError(
                "private_key_jwt",
                "AuthFI accepts a signed client assertion from agents only; a machine client authenticates "
                "with its secret (client_secret_basic or client_secret_post).",
            )
        if auth_method not in ("client_secret_basic", "client_secret_post"):
            raise ConfigurationError(
                f"auth_method must be client_secret_basic or client_secret_post, not {auth_method!r}"
            )
        self.auth = (
            ClientAuth.secret_basic(client_id, client_secret)
            if auth_method == "client_secret_basic"
            else ClientAuth.secret_post(client_id, client_secret)
        )
        self.project = project
        self.resource = resource.strip() if resource else None
        if isinstance(scope, str):
            raise ConfigurationError("scope is a sequence of names, such as ('orders:read',), not one string")
        self.scope = tuple(scope) if scope is not None else None
        self.destinations = Destinations(project, self.resource, send_to)
        if self.resource is None and warn_without_resource:
            warn_once(
                project.logger,
                ("no-resource", client_id),
                "client %s asks for a token with no resource: its audience is the client itself, which no API "
                "should accept. Pass resource=<the API's client id>.",
                client_id,
            )
        self.key = f"{self.resource or ''}|{' '.join(self.scope or ())}"

    @property
    def send_to(self) -> tuple[str, ...]:
        return self.destinations.origins


class ClientCredentials(TokenSource):
    """A machine client's own token, for blocking code. Make it with :meth:`AuthFI.client_credentials`."""

    def __init__(
        self,
        project: AuthFI,
        client_id: str,
        client_secret: Secret,
        *,
        resource: str | None = None,
        send_to: Sequence[str] = (),
        scope: Sequence[str] | None = None,
        auth_method: SecretAuthMethod = "client_secret_basic",
        warn_without_resource: bool = True,
    ) -> None:
        self._config = _Config(
            project,
            client_id,
            client_secret,
            resource=resource,
            send_to=send_to,
            scope=scope,
            auth_method=auth_method,
            warn_without_resource=warn_without_resource,
        )
        self.project = project
        self._cache: SyncCache[Secret] = SyncCache(project.clock, max_entries=1)

    @property
    def client_id(self) -> str:
        """The client's id."""
        return self._config.auth.client_id

    @property
    def resource(self) -> str | None:
        """The API the token is for (its audience)."""
        return self._config.resource

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`TokenSource.send_to`."""
        return self._config.send_to

    def token(self, *, force: bool = False) -> str:
        """The current token: cached, renewed at two thirds of its lifetime, one mint at a time."""

        def mint() -> tuple[Secret, float]:
            metadata = self.project.metadata()
            check_grant(metadata, self._config.auth)
            request = client_credentials_request(
                metadata.token_endpoint, self._config.auth, resource=self._config.resource, scope=self._config.scope
            )
            transport = self.project._transport  # pyright: ignore[reportPrivateUsage]
            response = parse_token_response(transport.send(request), "the token endpoint")
            self.project.logger.debug("minted a client-credentials token for %s", self.client_id)
            return response.access_token, lifetime_for(response)

        return self._cache.get(self._config.key, mint, refresh=force).reveal()

    def __repr__(self) -> str:
        return f"ClientCredentials(client_id={self.client_id!r}, resource={self.resource!r})"


class AsyncClientCredentials(AsyncTokenSource):
    """A machine client's own token, for asyncio code. Make it with :meth:`AsyncAuthFI.client_credentials`."""

    def __init__(
        self,
        project: AsyncAuthFI,
        client_id: str,
        client_secret: Secret,
        *,
        resource: str | None = None,
        send_to: Sequence[str] = (),
        scope: Sequence[str] | None = None,
        auth_method: SecretAuthMethod = "client_secret_basic",
        warn_without_resource: bool = True,
    ) -> None:
        self._config = _Config(
            project,
            client_id,
            client_secret,
            resource=resource,
            send_to=send_to,
            scope=scope,
            auth_method=auth_method,
            warn_without_resource=warn_without_resource,
        )
        self.project = project
        self._cache: AsyncCache[Secret] = AsyncCache(project.clock, max_entries=1)

    @property
    def client_id(self) -> str:
        """The client's id."""
        return self._config.auth.client_id

    @property
    def resource(self) -> str | None:
        """The API the token is for (its audience)."""
        return self._config.resource

    @property
    def send_to(self) -> tuple[str, ...]:
        """See :attr:`TokenSource.send_to`."""
        return self._config.send_to

    async def token(self, *, force: bool = False) -> str:
        """See :meth:`ClientCredentials.token`."""

        async def mint() -> tuple[Secret, float]:
            metadata = await self.project.metadata()
            check_grant(metadata, self._config.auth)
            request = client_credentials_request(
                metadata.token_endpoint, self._config.auth, resource=self._config.resource, scope=self._config.scope
            )
            transport = self.project._transport  # pyright: ignore[reportPrivateUsage]
            response = parse_token_response(await transport.send(request), "the token endpoint")
            self.project.logger.debug("minted a client-credentials token for %s", self.client_id)
            return response.access_token, lifetime_for(response)

        return (await self._cache.get(self._config.key, mint, refresh=force)).reveal()

    def __repr__(self) -> str:
        return f"AsyncClientCredentials(client_id={self.client_id!r}, resource={self.resource!r})"
