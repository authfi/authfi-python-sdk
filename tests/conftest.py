from __future__ import annotations

from collections.abc import AsyncIterator, Iterator

import pytest

from authfi import AsyncAuthFI, AuthFI
from authfi._log import reset_warn_once
from tests.support import FakeClock, Platform


@pytest.fixture(autouse=True)
def _fresh_warnings() -> Iterator[None]:
    reset_warn_once()
    yield
    reset_warn_once()


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
