"""A Flask app that signs people in with AuthFI.

Run it with ``flask --app flask_app run``. Flask's own session cookie is signed, not encrypted, so
the pending sign-in goes in it only as the encrypted ``pending.dump()``, and the tokens stay on the
server, keyed by a random session id.
"""

from __future__ import annotations

import os
import secrets

from flask import Flask, redirect, request, session
from werkzeug.wrappers import Response

from authfi import AuthFI, AuthFIError, InMemoryTokenStore, Secret

af = AuthFI.from_env()
web = af.web_app(
    client_id=os.environ["AUTHFI_CLIENT_ID"],
    client_secret=Secret(os.environ["AUTHFI_CLIENT_SECRET"]),
    redirect_uri=os.environ["APP_URL"] + "/callback",
    state_key=Secret(os.environ["AUTHFI_STATE_KEY"]),
)
# One process here. With several workers, use a TokenStore on your database or cache.
store = InMemoryTokenStore()
app = Flask(__name__)
app.secret_key = os.environ["FLASK_SECRET_KEY"]


@app.get("/login")
def login() -> Response:
    """Send the browser to AuthFI."""
    pending = web.start_sign_in(scope=("openid", "profile", "email", "offline_access"))
    session["authfi_pending"] = pending.dump()  # encrypted: never the verifier itself
    return redirect(pending.url)


@app.get("/callback")
def callback() -> Response | tuple[str, int]:
    """Finish the sign-in and keep the person's tokens on the server."""
    try:
        signed_in = web.finish_sign_in(request.url, web.load_pending(session.pop("authfi_pending", "")))
    except AuthFIError as error:
        return f"Sign-in failed ({error.code}). <a href='/login'>Try again</a>.", 400
    session_id = secrets.token_urlsafe(32)
    store.compare_and_set(session_id, None, signed_in.tokens)
    session["sid"] = session_id
    return redirect("/")


@app.get("/")
def home() -> str:
    """Who is signed in."""
    tokens = store.load(session.get("sid", ""))
    if tokens is None:
        return "<a href='/login'>Sign in</a>"
    info = web.userinfo(tokens)
    return f"Signed in as {info.email or info.sub}. <a href='/logout'>Sign out</a>"


@app.get("/logout")
def logout() -> Response:
    """Revoke the refresh token and end the AuthFI session."""
    tokens = store.load(session.pop("sid", ""))
    if tokens is None:
        return redirect("/")
    return redirect(web.sign_out(tokens, post_logout_redirect_uri=os.environ["APP_URL"] + "/"))
