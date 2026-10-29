"""The core against a real AuthFI project (``pytest -m live``).

Reads only what the project publishes, plus, when ``AUTHFI_LIVE_CLIENT_ID`` and
``AUTHFI_LIVE_CLIENT_SECRET`` are set, a machine token for AuthFI itself.
"""

from __future__ import annotations

import os

import pytest

from authfi import AsyncAuthFI, AuthFI, Secret

pytestmark = pytest.mark.live


@pytest.fixture
def issuer() -> str:
    return os.environ["AUTHFI_LIVE_ISSUER"]


def test_discovery_and_the_key_set(issuer: str) -> None:
    with AuthFI(issuer) as af:
        metadata = af.metadata()
        assert metadata.issuer == issuer
        assert metadata.supports_grant("client_credentials")
        assert metadata.supports_grant("urn:openid:params:grant-type:ciba")
        keys = af._key_set()
        assert {k.alg for k in keys.keys.values()} & {"EdDSA", "RS256"}


async def test_async_discovery(issuer: str) -> None:
    async with AsyncAuthFI(issuer) as aaf:
        assert (await aaf.metadata()).issuer == issuer


def test_a_machine_token_for_authfi_itself(issuer: str) -> None:
    client_id = os.environ.get("AUTHFI_LIVE_CLIENT_ID")
    secret = os.environ.get("AUTHFI_LIVE_CLIENT_SECRET")
    if not client_id or not secret:
        pytest.skip("AUTHFI_LIVE_CLIENT_ID and AUTHFI_LIVE_CLIENT_SECRET are not set")
    with AuthFI(issuer) as af:
        credential = af.client_credentials(client_id, Secret(secret), resource="authfi-console")
        token = credential.token()
        assert token.count(".") == 2
