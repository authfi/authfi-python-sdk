"""The project client: one per issuer and process, shared by every area of the SDK.

``AuthFI`` (blocking, on ``httpx.Client``) and ``AsyncAuthFI`` (asyncio, on ``httpx.AsyncClient``)
hold the project's issuer, its discovery document, its key set, one HTTP transport and the logger.
Every area is made from it (``af.verifier()``, ``af.client_credentials()``, ``af.agent()``,
``af.web_app()``), so they all share one cache and one connection pool. A second instance for
the same issuer doubles the fetches: create one and share it.
"""

from __future__ import annotations

import logging
import os
import threading
import weakref
from collections.abc import Mapping
from types import TracebackType
from typing import Any, ClassVar, Self, TypedDict, Unpack, cast

import httpx

from authfi._cache import AsyncCache, SyncCache
from authfi._clock import Clock, SystemClock
from authfi._log import get_logger
from authfi._protocol.discovery import ServerMetadata, metadata_flow, ttl_from_headers
from authfi._protocol.jwks import KeySet, PublicKey, jwks_flow
from authfi._protocol.retry import RetryPolicy
from authfi._protocol.urls import origin_of, validate_issuer, validate_origin
from authfi._transport import DEFAULT_TIMEOUT, AsyncTransport, SyncTransport, Timeout, user_agent
from authfi.errors import ConfigurationError

__all__ = ["AsyncAuthFI", "AuthFI", "AuthFIOptions", "Timeout"]

#: An unknown ``kid`` refetches the key set at most once per this many seconds, per issuer.
KID_REFETCH_INTERVAL = 30.0
_MAX_LEEWAY = 300.0


class AuthFIOptions(TypedDict, total=False):
    """The keyword options of :class:`AuthFI` and :meth:`AuthFI.from_env` (PEP 692)."""

    api_origin: str
    timeout: Timeout | float
    retries: int
    leeway: float
    logger: logging.Logger
    user_agent_suffix: str
    clock: Clock


class _Project:
    """Configuration shared by the sync and async clients."""

    _ENV_ISSUER: ClassVar[str] = "AUTHFI_ISSUER"
    _ENV_API_ORIGIN: ClassVar[str] = "AUTHFI_API_ORIGIN"

    issuer: str
    leeway: float
    _api_origin: str | None
    _derived_api_origin: str | None
    _last_kid_refetch: float | None
    _log: logging.Logger
    _clock: Clock

    def _configure(
        self,
        issuer: str,
        *,
        api_origin: str | None,
        timeout: Timeout | float,
        retries: int,
        leeway: float,
        logger: logging.Logger | None,
        user_agent_suffix: str | None,
        clock: Clock | None,
    ) -> tuple[Timeout, RetryPolicy, str]:
        self.issuer = validate_issuer(issuer)
        self._api_origin = validate_origin(api_origin, "api_origin") if api_origin else None
        self._derived_api_origin = None
        self._last_kid_refetch = None
        given_retries = cast(object, retries)
        if not isinstance(given_retries, int) or isinstance(given_retries, bool) or not 0 <= given_retries <= 5:
            raise ConfigurationError(f"retries must be an integer from 0 to 5, not {given_retries!r}")
        given_leeway = cast(object, leeway)
        if not isinstance(given_leeway, int | float) or not 0 <= given_leeway <= _MAX_LEEWAY:
            raise ConfigurationError(f"leeway must be from 0 to {_MAX_LEEWAY:.0f} seconds, not {given_leeway!r}")
        self.leeway = float(given_leeway)
        self._log = get_logger(logger)
        self._clock = clock or SystemClock()
        return Timeout.of(timeout), RetryPolicy(retries=retries), user_agent(user_agent_suffix)

    @property
    def api_origin(self) -> str:
        """Where the project's machine APIs are served.

        The configured ``api_origin``; else, once discovery is read, the origin of its CIBA endpoint
        (the slug host, for a project on a custom domain); else the issuer's origin. Before discovery
        is read this is the issuer's origin: use :meth:`AuthFI.resolve_api_origin` to build a URL.
        """
        return self._api_origin or self._derived_api_origin or origin_of(self.issuer)

    @property
    def authfi_origins(self) -> tuple[str, ...]:
        """The origins a credential for AuthFI itself may be sent to: the issuer and the API origin."""
        issuer = origin_of(self.issuer)
        api = self.api_origin
        return (issuer,) if api == issuer else (issuer, api)

    @property
    def clock(self) -> Clock:
        """The clock this project reads (the system clock unless one was injected)."""
        return self._clock

    @property
    def logger(self) -> logging.Logger:
        """The logger every area of this project logs to."""
        return self._log

    def _refetch_due(self, fetched: float | None) -> bool:
        """Whether an unknown ``kid`` may refetch the key set now, and note that it did.

        At most one refetch per :data:`KID_REFETCH_INTERVAL`, counted from the last attempt, successful
        or not (``fetched`` is when the cached set was last stored): a key set that answers 500 must
        not be hit once per forged token.
        """
        now = self._clock.monotonic()
        last = [t for t in (fetched, self._last_kid_refetch) if t is not None]
        if last and now - max(last) < KID_REFETCH_INTERVAL:
            return False
        self._last_kid_refetch = now
        return True

    def _learn(self, metadata: ServerMetadata) -> None:
        endpoint = metadata.backchannel_authentication_endpoint
        if endpoint and self._api_origin is None:
            self._derived_api_origin = origin_of(endpoint)

    @classmethod
    def _env_options(cls, env: Mapping[str, str] | None) -> tuple[str, dict[str, Any]]:
        source = os.environ if env is None else env
        issuer = source.get(cls._ENV_ISSUER, "").strip()
        if not issuer:
            raise ConfigurationError(
                f"{cls._ENV_ISSUER} is not set: set it to your project's own address, for example "
                "https://acme.authfi.io."
            )
        options: dict[str, Any] = {}
        api_origin = source.get(cls._ENV_API_ORIGIN, "").strip()
        if api_origin:
            options["api_origin"] = api_origin
        return issuer, options

    def __repr__(self) -> str:
        return f"{type(self).__name__}(issuer={self.issuer!r})"


# Every live client, so a forked child can reset each one's per-process state.
_instances: weakref.WeakSet[AuthFI | AsyncAuthFI] = weakref.WeakSet()
_instances_lock = threading.Lock()


def _after_fork_in_child() -> None:
    global _instances_lock  # the parent's lock may be held at fork time
    _instances_lock = threading.Lock()
    for instance in list(_instances):
        instance._after_fork()  # pyright: ignore[reportPrivateUsage]


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)


def _register(instance: AuthFI | AsyncAuthFI) -> None:
    with _instances_lock:
        _instances.add(instance)


class AuthFI(_Project):
    """An AuthFI project, for blocking code.

    Args:
        issuer: the project's address, ``https://<slug>.authfi.io`` or its custom domain. The one
            required setting: everything else is read from its discovery document.
        api_origin: where the project's machine APIs are served, for a project on a custom domain
            whose APIs stay on the slug host. Read from discovery when absent.
        timeout: a :class:`Timeout`, or the total deadline of one call in seconds.
        retries: how many times a failed request may be sent again (0 to 5).
        leeway: seconds of clock skew allowed on ``exp``, ``nbf`` and ``iat`` (0 to 300).
        http_client: an ``httpx.Client`` for proxies or mTLS. The SDK then never closes it, and a
            forked child must make its own.
        logger: where to log; the ``authfi`` logger by default.
        user_agent_suffix: appended to the ``User-Agent``.
        clock: for tests, a clock that advances on demand.

    """

    def __init__(
        self,
        issuer: str,
        *,
        api_origin: str | None = None,
        timeout: Timeout | float = DEFAULT_TIMEOUT,
        retries: int = 2,
        leeway: float = 60.0,
        http_client: httpx.Client | None = None,
        logger: logging.Logger | None = None,
        user_agent_suffix: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        timeout_, policy, agent = self._configure(
            issuer,
            api_origin=api_origin,
            timeout=timeout,
            retries=retries,
            leeway=leeway,
            logger=logger,
            user_agent_suffix=user_agent_suffix,
            clock=clock,
        )
        self._transport = SyncTransport(
            client=http_client, timeout=timeout_, policy=policy, agent=agent, logger=self._log, clock=self._clock
        )
        self._metadata: SyncCache[ServerMetadata] = SyncCache(self._clock, max_entries=1)
        self._jwks: SyncCache[KeySet] = SyncCache(self._clock, max_entries=1)
        self._closables: list[weakref.ref[Any]] = []
        _register(self)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        http_client: httpx.Client | None = None,
        **overrides: Unpack[AuthFIOptions],
    ) -> Self:
        """Build from ``AUTHFI_ISSUER`` (required) and ``AUTHFI_API_ORIGIN``, then ``overrides``."""
        issuer, options = cls._env_options(env)
        options.update(overrides)
        return cls(issuer, http_client=http_client, **options)

    def metadata(self, *, refresh: bool = False) -> ServerMetadata:
        """The project's discovery document, cached for its ``Cache-Control`` lifetime."""

        def fetch() -> tuple[ServerMetadata, float]:
            return self._transport.run(metadata_flow(self.issuer))

        metadata = self._metadata.get("metadata", fetch, refresh=refresh)
        self._learn(metadata)
        return metadata

    def resolve_api_origin(self) -> str:
        """Where the project's machine APIs are served, after reading discovery if it is needed.

        :attr:`api_origin` alone is the issuer's origin until discovery has been read, which is
        the wrong host for a project on a custom domain. Call this before building the URL of any
        machine API. A configured ``api_origin`` needs no discovery.
        """
        if self._api_origin is None:
            self.metadata()
        return self.api_origin

    def _key(self, kid: str) -> PublicKey | None:
        """The published key named ``kid``. An unknown kid refetches the set, at most once per 30 s."""
        keys = self._key_set()
        key = keys.get(kid)
        if key is not None:
            return key
        if not self._refetch_due(self._jwks.fetched_at("jwks")):
            return None
        return self._key_set(refresh=True).get(kid)

    def _key_set(self, *, refresh: bool = False) -> KeySet:
        def fetch() -> tuple[KeySet, float]:
            keys, cache_control = self._transport.run(jwks_flow(self.metadata().jwks_uri))
            return keys, ttl_from_headers(cache_control)

        return self._jwks.get("jwks", fetch, refresh=refresh)

    def _track(self, closable: Any) -> None:
        self._closables.append(weakref.ref(closable))

    def close(self) -> None:
        """Close the SDK-owned HTTP client and every HTTP client this project handed out."""
        for ref in self._closables:
            target = ref()
            if target is not None:
                target.close()
        self._closables.clear()
        self._transport.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def _after_fork(self) -> None:
        self._transport.after_fork()
        self._metadata.after_fork()
        self._jwks.after_fork()


class AsyncAuthFI(_Project):
    """An AuthFI project, for asyncio code. The same as :class:`AuthFI`, with coroutines.

    One instance belongs to one event loop: its locks and HTTP client bind to the loop that first
    uses them, and a call from another loop raises :class:`~authfi.errors.ConfigurationError`.
    """

    def __init__(
        self,
        issuer: str,
        *,
        api_origin: str | None = None,
        timeout: Timeout | float = DEFAULT_TIMEOUT,
        retries: int = 2,
        leeway: float = 60.0,
        http_client: httpx.AsyncClient | None = None,
        logger: logging.Logger | None = None,
        user_agent_suffix: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        timeout_, policy, agent = self._configure(
            issuer,
            api_origin=api_origin,
            timeout=timeout,
            retries=retries,
            leeway=leeway,
            logger=logger,
            user_agent_suffix=user_agent_suffix,
            clock=clock,
        )
        self._transport = AsyncTransport(
            client=http_client, timeout=timeout_, policy=policy, agent=agent, logger=self._log, clock=self._clock
        )
        self._metadata: AsyncCache[ServerMetadata] = AsyncCache(self._clock, max_entries=1)
        self._jwks: AsyncCache[KeySet] = AsyncCache(self._clock, max_entries=1)
        self._closables: list[weakref.ref[Any]] = []
        _register(self)

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        **overrides: Unpack[AuthFIOptions],
    ) -> Self:
        """See :meth:`AuthFI.from_env`."""
        issuer, options = cls._env_options(env)
        options.update(overrides)
        return cls(issuer, http_client=http_client, **options)

    async def metadata(self, *, refresh: bool = False) -> ServerMetadata:
        """See :meth:`AuthFI.metadata`."""
        self._transport.check_loop()

        async def fetch() -> tuple[ServerMetadata, float]:
            return await self._transport.run(metadata_flow(self.issuer))

        metadata = await self._metadata.get("metadata", fetch, refresh=refresh)
        self._learn(metadata)
        return metadata

    async def resolve_api_origin(self) -> str:
        """See :meth:`AuthFI.resolve_api_origin`."""
        if self._api_origin is None:
            await self.metadata()
        return self.api_origin

    async def _key(self, kid: str) -> PublicKey | None:
        keys = await self._key_set()
        key = keys.get(kid)
        if key is not None:
            return key
        if not self._refetch_due(self._jwks.fetched_at("jwks")):
            return None
        return (await self._key_set(refresh=True)).get(kid)

    async def _key_set(self, *, refresh: bool = False) -> KeySet:
        self._transport.check_loop()

        async def fetch() -> tuple[KeySet, float]:
            keys, cache_control = await self._transport.run(jwks_flow((await self.metadata()).jwks_uri))
            return keys, ttl_from_headers(cache_control)

        return await self._jwks.get("jwks", fetch, refresh=refresh)

    def _track(self, closable: Any) -> None:
        self._closables.append(weakref.ref(closable))

    async def aclose(self) -> None:
        """Close the SDK-owned HTTP client and every HTTP client this project handed out."""
        for ref in self._closables:
            target = ref()
            if target is not None:
                await target.aclose()
        self._closables.clear()
        await self._transport.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        await self.aclose()

    def _after_fork(self) -> None:
        self._transport.after_fork()
        self._metadata.after_fork()
        self._jwks.after_fork()
