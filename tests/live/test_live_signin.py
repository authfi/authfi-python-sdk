"""A person's sign-in to a web app on a real project (see ``console.py``).

Registers one web app and removes it. The sign-in page's part, a person typing their password, is
played by a harness for a probe person through ``AUTHFI_LIVE_PERSON_DIR``: the suite writes the
authorize URL there, and the harness writes back the URL the browser would come back to.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from authfi import AuthFI, OAuthError, Secret
from tests.live.console import ISSUER, PERSON_DIR, RUN, Cleanups, gql, needs_console

pytestmark = [
    pytest.mark.live,
    needs_console,
    pytest.mark.skipif(not PERSON_DIR, reason="AUTHFI_LIVE_PERSON_DIR is not set"),
]


@pytest.fixture
def cleanups() -> Iterator[Cleanups]:
    steps = Cleanups()
    yield steps
    steps.run()


def _browser(url: str) -> str:
    """Hand the authorize URL to the harness and wait for the callback it answers."""
    request, answer = Path(PERSON_DIR) / "authorize-request", Path(PERSON_DIR) / "sign-in-callback"
    fd = os.open(request, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as out:
        out.write(json.dumps({"url": url}))
    for _ in range(60):
        time.sleep(1)
        if answer.exists():
            callback = answer.read_text(encoding="utf-8").strip()
            answer.unlink()
            return callback
    raise AssertionError("the harness answered no callback")


def test_a_person_signs_in_refreshes_and_signs_out(cleanups: Cleanups) -> None:
    redirect_uri = f"https://sdk-live-{RUN}.example.com/callback"
    created = gql(
        "mutation($i:CreateApplicationInput!)"
        "{ createApplication(input:$i){ application { id clientId } clientSecret } }",
        {"i": {"name": f"sdk-live-{RUN}", "type": "web", "redirectUris": [redirect_uri]}},
    )["createApplication"]
    app_id = created["application"]["id"]
    cleanups.add("web app", lambda: gql("mutation($id:ID!){ deleteApplication(id:$id) }", {"id": app_id}))

    with AuthFI(ISSUER, user_agent_suffix=f"sdk-live/{RUN}") as af:
        web = af.web_app(
            client_id=created["application"]["clientId"],
            client_secret=Secret(created["clientSecret"]),
            redirect_uri=redirect_uri,
            state_key=Secret(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode()),
        )
        started = web.start_sign_in()
        pending = web.load_pending(started.dump())  # as the callback request would, from a cookie
        signed_in = web.finish_sign_in(_browser(started.url), pending)

        assert signed_in.user.sub
        assert signed_in.tokens.refresh_token is not None
        refreshed = web.refresh(signed_in.tokens)
        assert refreshed.refresh_token != signed_in.tokens.refresh_token  # rotated
        assert web.refresh(signed_in.tokens) == refreshed  # a second caller gets the same result
        assert web.userinfo(refreshed).sub == signed_in.user.sub
        end = web.sign_out(refreshed)
        assert "/oauth/logout" in end
        with pytest.raises(OAuthError) as caught:
            web.refresh(refreshed)
        assert caught.value.code == "invalid_grant"
    print(f"  sign-in ok: sub={signed_in.user.sub[:6]}... acr={signed_in.user.acr}; refresh rotated; sign-out revoked")
