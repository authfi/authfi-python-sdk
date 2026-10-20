from __future__ import annotations

import asyncio
import os
import signal
import sys
import threading
from typing import Any

import httpx
import pytest

from authfi import AsyncAuthFI, AuthFI, ConfigurationError, Timeout
from tests.support import FakeClock, Platform


def test_the_issuer_is_the_one_required_setting() -> None:
    af = AuthFI("https://acme.authfi.io")
    assert af.issuer == "https://acme.authfi.io"
    assert af.api_origin == "https://acme.authfi.io"
    assert repr(af) == "AuthFI(issuer='https://acme.authfi.io')"
    with pytest.raises(ConfigurationError, match=r"https://acme\.authfi\.io"):
        AuthFI("acme")


@pytest.mark.parametrize(
    "kwargs",
    [{"retries": -1}, {"retries": 6}, {"retries": True}, {"leeway": 301}, {"leeway": -1}, {"timeout": 0}],
)
def test_settings_are_checked(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        AuthFI("https://acme.authfi.io", **kwargs)


def test_from_env_reads_the_issuer_and_api_origin() -> None:
    af = AuthFI.from_env({"AUTHFI_ISSUER": "https://acme.authfi.io", "AUTHFI_API_ORIGIN": "https://acme-api.authfi.io"})
    assert af.api_origin == "https://acme-api.authfi.io"
    af2 = AuthFI.from_env({"AUTHFI_ISSUER": "https://acme.authfi.io"}, leeway=10, timeout=Timeout(total=5))
    assert af2.leeway == 10.0
    with pytest.raises(ConfigurationError, match="AUTHFI_ISSUER"):
        AuthFI.from_env({})


async def test_async_from_env() -> None:
    aaf = AsyncAuthFI.from_env({"AUTHFI_ISSUER": "https://acme.authfi.io"})
    assert aaf.issuer == "https://acme.authfi.io"
    await aaf.aclose()


def test_close_closes_an_owned_client_and_leaves_an_injected_one() -> None:
    injected = httpx.Client()
    with AuthFI("https://acme.authfi.io", http_client=injected) as af:
        af._transport.client  # noqa: B018 - touch it
    assert not injected.is_closed
    owned = AuthFI("https://acme.authfi.io")
    client = owned._transport.client
    owned.close()
    assert client.is_closed


def test_close_also_closes_what_the_project_handed_out() -> None:
    class Thing:
        closed = False

        def close(self) -> None:
            self.closed = True

    thing = Thing()
    af = AuthFI("https://acme.authfi.io")
    af._track(thing)
    af.close()
    assert thing.closed


async def test_one_async_project_per_event_loop(platform: Platform, clock: FakeClock) -> None:
    aaf = AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock)
    await aaf.metadata()
    failures: list[BaseException] = []

    def other() -> None:
        try:
            asyncio.run(aaf.metadata(refresh=True))
        except BaseException as exc:
            failures.append(exc)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert len(failures) == 1 and isinstance(failures[0], ConfigurationError)
    assert "one AsyncAuthFI per event loop" in str(failures[0])


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX only")
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_a_forked_child_completes_a_call_with_a_lock_held_and_a_fetch_in_flight(platform: Platform) -> None:
    clock = FakeClock()
    entered = threading.Event()
    release = threading.Event()
    inner = platform.routes[("GET", "/.well-known/openid-configuration")]

    def slow(request: httpx.Request) -> httpx.Response:
        if threading.current_thread() is not threading.main_thread():
            entered.set()
            release.wait(10)
        return inner(request)

    platform.route("GET", "/.well-known/openid-configuration", slow)
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    background = threading.Thread(target=af.metadata)
    background.start()
    assert entered.wait(5)
    # The parent has a fetch in flight. The child must not wait on it, or on the lock.
    af._metadata._lock.acquire()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - runs in the child; its exit code is the assertion
        signal.alarm(10)  # a child that deadlocks dies by SIGALRM, so the parent's waitpid fails the test
        try:
            ok = af.metadata().issuer == platform.issuer
        except BaseException:
            ok = False
        os._exit(0 if ok else 1)
    af._metadata._lock.release()
    release.set()
    background.join()
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0


@pytest.mark.skipif(not hasattr(os, "fork"), reason="fork is POSIX only")
def test_a_forked_child_drops_an_owned_client_without_closing_the_parents() -> None:
    af = AuthFI("https://acme.authfi.io")
    parent_client = af._transport.client
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        os._exit(0 if af._transport._client is None else 1)
    _, status = os.waitpid(pid, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    assert not parent_client.is_closed
    af.close()


def test_the_package_exports_what_it_documents() -> None:
    import authfi

    for name in authfi.__all__:
        assert hasattr(authfi, name), name
    assert sys.modules["authfi"].__version__
