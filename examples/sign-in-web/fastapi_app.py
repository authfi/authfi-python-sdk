"""A FastAPI app that signs people in with AuthFI and calls an API as them.

Run it with ``uvicorn fastapi_app:app``. The tokens stay on the server, keyed by a random session id
in a cookie; the pending sign-in travels in a cookie only as ``pending.dump()`` encrypts it.
"""

from __future__ import annotations

import os
import secrets

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from authfi import AsyncAuthFI, AsyncInMemoryTokenStore, AuthFIError, Secret

af = AsyncAuthFI.from_env()
web = af.web_app(
    client_id=os.environ["AUTHFI_CLIENT_ID"],
    client_secret=Secret(os.environ["AUTHFI_CLIENT_SECRET"]),
    redirect_uri=os.environ["APP_URL"] + "/callback",
    state_key=Secret(os.environ["AUTHFI_STATE_KEY"]),
)
# One process here. With several workers, use a TokenStore on your database or cache.
store = AsyncInMemoryTokenStore()
app = FastAPI()
SECURE = os.environ["APP_URL"].startswith("https://")


@app.get("/login")
async def login() -> Response:
    """Send the browser to AuthFI."""
    pending = await web.start_sign_in(scope=("openid", "profile", "email", "offline_access"))
    response = RedirectResponse(pending.url)
    response.set_cookie("authfi_pending", pending.dump(), httponly=True, secure=SECURE, samesite="lax", max_age=600)
    return response


@app.get("/callback")
async def callback(request: Request) -> Response:
    """Finish the sign-in and keep the person's tokens on the server."""
    blob = request.cookies.get("authfi_pending", "")
    try:
        signed_in = await web.finish_sign_in(str(request.url), web.load_pending(blob))
    except AuthFIError as error:
        return HTMLResponse(f"Sign-in failed ({error.code}). <a href='/login'>Try again</a>.", status_code=400)
    session_id = secrets.token_urlsafe(32)
    await store.compare_and_set(session_id, None, signed_in.tokens)
    response = RedirectResponse("/")
    response.delete_cookie("authfi_pending")
    response.set_cookie("session", session_id, httponly=True, secure=SECURE, samesite="lax")
    return response


@app.get("/")
async def home(request: Request) -> Response:
    """Who is signed in, from the userinfo endpoint."""
    tokens = await store.load(request.cookies.get("session", ""))
    if tokens is None:
        return HTMLResponse("<a href='/login'>Sign in</a>")
    info = await web.userinfo(tokens)
    return HTMLResponse(f"Signed in as {info.email or info.sub}. <a href='/logout'>Sign out</a>")


@app.get("/logout")
async def logout(request: Request) -> Response:
    """Revoke the refresh token and end the AuthFI session."""
    tokens = await store.load(request.cookies.get("session", ""))
    if tokens is None:
        return RedirectResponse("/")
    url = await web.sign_out(tokens, post_logout_redirect_uri=os.environ["APP_URL"] + "/")
    response = RedirectResponse(url)
    response.delete_cookie("session")
    return response
