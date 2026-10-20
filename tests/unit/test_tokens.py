from __future__ import annotations

import asyncio
import copy
import pickle
import threading
import time
from datetime import UTC, datetime

import httpx
import pytest

from authfi import (
    AsyncInMemoryTokenStore,
    AsyncTokenSource,
    AsyncTokenStore,
    ConfigurationError,
    InMemoryTokenStore,
    Secret,
    TokenSet,
    TokenSource,
    TokenStore,
)
from authfi._protocol.oauth import TokenResponse
from authfi.tokens import AsyncRefreshGate, RefreshGate, normalise_send_to
from tests.support import FakeClock

MARKER = "MARKER-access-token"


class StaticSource(TokenSource):
    def __init__(self, value: str, send_to: tuple[str, ...]) -> None:
        self._value = Secret(value)
        self._send_to = send_to

    @property
    def send_to(self) -> tuple[str, ...]:
        return self._send_to

    def token(self, *, force: bool = False) -> str:
        return self._value.reveal()


class AsyncStaticSource(AsyncTokenSource):
    def __init__(self, value: str, send_to: tuple[str, ...]) -> None:
        self._value = Secret(value)
        self._send_to = send_to

    @property
    def send_to(self) -> tuple[str, ...]:
        return self._send_to

    async def token(self, *, force: bool = False) -> str:
        return self._value.reveal()


def test_sources_are_nominal_so_an_async_source_is_never_a_sync_one() -> None:
    def kinds(source: object) -> tuple[bool, bool]:
        return isinstance(source, TokenSource), isinstance(source, AsyncTokenSource)

    assert kinds(StaticSource("t", ())) == (True, False)
    assert kinds(AsyncStaticSource("t", ())) == (False, True)
    assert StaticSource.is_async is False
    assert AsyncStaticSource.is_async is True


def test_headers_go_only_to_the_named_origins() -> None:
    source = StaticSource(MARKER, ("https://api.example.com",))
    assert source.headers("GET", "https://api.example.com/orders") == {"Authorization": "Bearer " + MARKER}
    with pytest.raises(ConfigurationError) as caught:
        source.headers("GET", "https://evil.example/steal")
    assert caught.value.code == "destination_not_allowed"
    assert MARKER not in str(caught.value)


async def test_async_headers_go_only_to_the_named_origins() -> None:
    source = AsyncStaticSource(MARKER, ("https://api.example.com",))
    assert await source.headers("GET", "https://api.example.com/orders") == {"Authorization": "Bearer " + MARKER}
    with pytest.raises(ConfigurationError) as caught:
        await source.headers("GET", "https://evil.example/steal")
    assert caught.value.code == "destination_not_allowed"
    assert MARKER not in str(caught.value)


def test_httpx_auth_attaches_per_request_and_refuses_other_origins() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200)

    source = StaticSource(MARKER, ("https://api.example.com",))
    with httpx.Client(transport=httpx.MockTransport(handler), auth=source.httpx_auth()) as client:
        client.get("https://api.example.com/x")
        with pytest.raises(ConfigurationError):
            client.get("https://other.example/x")
    assert seen == ["Bearer " + MARKER]


async def test_async_httpx_auth_and_the_wrong_client_kind() -> None:
    seen: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        return httpx.Response(200)

    asyn = AsyncStaticSource(MARKER, ("https://api.example.com",))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=asyn.httpx_auth()) as client:
        await client.get("https://api.example.com/x")
    assert seen == ["Bearer " + MARKER]
    sync = StaticSource(MARKER, ("https://api.example.com",))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), auth=sync.httpx_auth()) as client:
        with pytest.raises(ConfigurationError, match="blocking code"):
            await client.get("https://api.example.com/x")
    assert len(seen) == 1


def test_an_async_credential_refuses_a_blocking_client() -> None:
    asyn = AsyncStaticSource(MARKER, ("https://api.example.com",))
    with (
        httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(200)), auth=asyn.httpx_auth()) as client,
        pytest.raises(ConfigurationError, match="asyncio code"),
    ):
        client.get("https://api.example.com/x")


def test_send_to_values_are_checked() -> None:
    assert normalise_send_to(["https://API.example.com", "https://api.example.com"]) == ("https://api.example.com",)
    with pytest.raises(ConfigurationError):
        normalise_send_to("https://api.example.com")
    with pytest.raises(ConfigurationError):
        normalise_send_to(["https://api.example.com/path"])


def token_set(access: str = MARKER, refresh: str | None = "RT", expires: float = 0.0) -> TokenSet:
    return TokenSet(
        access_token=Secret(access),
        expires_at=datetime.fromtimestamp(expires, UTC),
        refresh_token=Secret(refresh) if refresh else None,
    )


def test_a_token_set_does_not_print_or_pickle() -> None:
    ts = token_set()
    assert MARKER not in repr(ts) and "RT" not in repr(ts)
    with pytest.raises(TypeError, match="session id"):
        pickle.dumps(ts)
    with pytest.raises(TypeError):
        hash(ts)


def test_a_framework_that_copies_its_config_can_copy_a_token_set() -> None:
    ts = token_set()
    assert copy.copy(ts) is ts
    assert copy.deepcopy(ts) is ts
    assert copy.deepcopy({"tokens": ts})["tokens"] is ts


def test_a_token_set_from_a_refresh_keeps_what_the_answer_left_out() -> None:
    first = TokenSet.from_response(
        TokenResponse(Secret("A1"), "Bearer", 900, Secret("R1"), Secret("ID"), frozenset({"openid"})), now=1000.0
    )
    assert first.expires_at == datetime.fromtimestamp(1900, UTC)
    second = TokenSet.from_response(TokenResponse(Secret("A2"), "Bearer", 900), now=2000.0, previous=first)
    assert second.refresh_token == Secret("R1")
    assert second.id_token == Secret("ID")
    assert second.scope == frozenset({"openid"})
    assert second.expires_within(60, 2850) and not second.expires_within(60, 2000)
    assert first == TokenSet.from_response(TokenResponse(Secret("A1"), "Bearer", 1, Secret("R1")), now=0)
    assert first != second


def test_the_in_memory_store_compares_and_sets() -> None:
    store = InMemoryTokenStore()
    assert isinstance(store, TokenStore)
    a, b, c = token_set("a"), token_set("b"), token_set("c")
    assert store.compare_and_set("k", None, a)
    assert not store.compare_and_set("k", None, b)
    assert store.compare_and_set("k", a, b)
    assert not store.compare_and_set("k", a, c)
    assert store.load("k") == b
    assert store.load("other") is None


async def test_the_async_in_memory_store_compares_and_sets() -> None:
    store = AsyncInMemoryTokenStore()
    assert isinstance(store, AsyncTokenStore)
    a, b = token_set("a"), token_set("b")
    assert await store.compare_and_set("k", None, a)
    assert not await store.compare_and_set("k", None, b)
    assert await store.compare_and_set("k", a, b)
    assert await store.load("k") == b


def test_one_refresh_per_refresh_token_across_threads(clock: FakeClock) -> None:
    gate = RefreshGate(clock)
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def refresh() -> TokenSet:
        calls.append(1)
        started.set()
        release.wait(5)
        return token_set("new")

    results: list[TokenSet] = []
    threads = [threading.Thread(target=lambda: results.append(gate.run(Secret("R1"), refresh))) for _ in range(6)]
    for t in threads:
        t.start()
    started.wait(5)
    time.sleep(0.05)
    release.set()
    for t in threads:
        t.join()
    assert len(calls) == 1
    assert results == [token_set("new")] * 6
    # A caller still holding the spent refresh token gets the result, and never spends it again.
    assert gate.run(Secret("R1"), refresh) == token_set("new")
    assert len(calls) == 1


async def test_one_refresh_per_refresh_token_across_tasks(clock: FakeClock) -> None:
    gate = AsyncRefreshGate(clock)
    calls: list[int] = []

    async def refresh() -> TokenSet:
        calls.append(1)
        await asyncio.sleep(0.01)
        return token_set("new")

    results = await asyncio.gather(*(gate.run(Secret("R1"), refresh) for _ in range(6)))
    assert len(calls) == 1
    assert all(r == token_set("new") for r in results)


def test_a_failed_refresh_is_not_remembered(clock: FakeClock) -> None:
    gate = RefreshGate(clock)

    def boom() -> TokenSet:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        gate.run(Secret("R1"), boom)
    assert gate.run(Secret("R1"), lambda: token_set("ok")) == token_set("ok")
