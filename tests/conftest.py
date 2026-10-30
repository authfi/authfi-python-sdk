from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from authfi import AsyncAuthFI, AuthFI
from authfi._log import reset_warn_once
from tests.support import FakeClock, Platform


@pytest.fixture(autouse=True)
def _fresh_warnings() -> Iterator[None]:
    reset_warn_once()
    yield
    reset_warn_once()


@pytest.fixture(autouse=True)
def _no_pod_token(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Whatever runs the tests may be a pod with a service-account token: no test finds it unless it names one."""
    monkeypatch.setattr("authfi.instance.K8S_TOKEN_PATH", str(tmp_path / "no-pod-token"))


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def platform() -> Platform:
    return Platform()


@pytest.fixture
def af(platform: Platform, clock: FakeClock) -> Iterator[AuthFI]:
    with AuthFI(platform.issuer, http_client=platform.client(), clock=clock) as project:
        yield project


@pytest.fixture
async def aaf(platform: Platform, clock: FakeClock) -> AsyncIterator[AsyncAuthFI]:
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as project:
        yield project


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip the live suite unless it is pointed at a project."""
    if os.environ.get("AUTHFI_LIVE_ISSUER"):
        return
    skip = pytest.mark.skip(reason="live: set AUTHFI_LIVE_ISSUER (and the client settings) to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
