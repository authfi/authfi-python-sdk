# Signing people in to a web app

A server-rendered web app signs people in with the OpenID Connect code flow: the browser is sent to
AuthFI, the person signs in, and AuthFI sends the browser back to your callback with a code your
server exchanges for the person's tokens.

```python
import os
from authfi import AuthFI, Secret

af = AuthFI.from_env()
web = af.web_app(
    client_id="app_web",
    client_secret=Secret(os.environ["AUTHFI_CLIENT_SECRET"]),   # None for a public client
    redirect_uri="https://app.example.com/callback",
    state_key=Secret(os.environ["AUTHFI_STATE_KEY"]),
)
```

`state_key` is 32 random bytes in base64. Make one with:

```bash
python -c "import base64, secrets; print(base64.urlsafe_b64encode(secrets.token_bytes(32)).decode())"
```

## Sign in

```python
# /login
pending = web.start_sign_in()                       # scope=("openid", "profile", "email")
response.set_cookie("authfi_pending", pending.dump(), httponly=True, secure=True, samesite="lax", max_age=600)
return redirect(pending.url)

# /callback
pending = web.load_pending(request.cookies["authfi_pending"])
signed_in = web.finish_sign_in(str(request.url), pending)
signed_in.user.sub, signed_in.user.email            # who signed in
signed_in.tokens                                    # access, refresh and ID tokens, as Secrets
```

Every sign-in uses PKCE (S256), a fresh `state` and a fresh `nonce`. `finish_sign_in`:

- requires the callback's `iss` and checks it is this project exactly (RFC 9207), so a response from
  another issuer cannot be mixed in;
- refuses a callback that repeats `iss`, `state`, `code`, `error` or `error_description`
  (`InvalidCallback`, code `callback_parameter_repeated`): which value is meant is unknown. Pass the
  full URL, or the framework's own query parameters (`request.query_params` in FastAPI and
  Starlette, `request.args` in Flask), which keep every value, so a repeat is seen;
- checks `state` in constant time;
- exchanges the code with the PKCE verifier;
- verifies the ID token: an algorithm the project's discovery names, `typ` `JWT`, the issuer, the
  audience (your client id, and `azp`), the `nonce`, `at_hash`, and `max_age` and `acr` when you
  asked for them.

**Keep the pending sign-in only as `dump()` gives it.** It holds the PKCE verifier and the nonce.
`dump()` encrypts it (AES-256-GCM under `state_key`), binds it to your client and redirect URI, and
it expires after ten minutes, so it is safe in a cookie or a session. A framework's default session
cookie is signed, not encrypted: anything put there as a dict can be read by the browser.
`load_pending` raises `InvalidPendingSignIn` for a blob that was tampered with, made for another
client or redirect URI, or has expired; start the sign-in again.

Options: `start_sign_in(scope=..., acr_values=[...], max_age=seconds, login_hint=..., prompt="login" | "consent")`.

## Step-up

```python
pending = web.start_sign_in(acr_values=["urn:authfi:acr:mfa"])
```

When the sign-in does not meet the `acr` you asked for, or the person last authenticated longer ago
than `max_age`, `finish_sign_in` raises `StepUpRequired`; its `acr_values` and `max_age` say what to
ask for in a new sign-in. A refresh that the project refuses with `insufficient_user_authentication`
raises `StepUpRequired` too, carrying the `step_up_acr` you gave `web_app`.

## Keep the person signed in

```python
tokens = web.refresh(signed_in.tokens)              # one refresh per refresh token, shared
info = web.userinfo(tokens)                         # must name the same person
```

A refresh token can be used once: a second use revokes it and every token made from it, which signs
the person out. `refresh` makes one request per refresh token across the threads (and tasks, in
`AsyncWebApp`) that call it at the same time, and each of them gets its result. The result is not
kept: a caller that arrives later with the refresh token that was already spent is sent to the
server, which refuses it (`OAuthError`, `invalid_grant`), and is never handed the earlier answer. Store
the new tokens that `refresh` returns and refresh those.

Across workers, keep the tokens in a shared `TokenStore` (a protocol with `load(key)` and
`compare_and_set(key, old, new)`, which your database or cache implements) and use the person's
session:

```python
session = web.session(signed_in.tokens, store=store, key=session_id, send_to=["https://api.example.com"])
httpx.Client(auth=session.httpx_auth()).get("https://api.example.com/orders")
```

The session refreshes the access token a minute before it expires; when two workers refresh at
once, the one whose `compare_and_set` lands wins and the other uses its tokens. `InMemoryTokenStore`
is for a single process.

## Sign out

```python
url = web.sign_out(tokens, post_logout_redirect_uri="https://app.example.com/")
return redirect(url)
```

`sign_out` revokes the refresh token first (ending the session at AuthFI does not; with no refresh
token, it revokes the access token), then returns the end-session URL, naming your client, to send the
browser to. A project whose discovery names no revocation endpoint raises `NotSupportedError`
(`feature == "revocation"`) instead of signing the person out with their tokens still alive.

## In asyncio code

`AsyncAuthFI.web_app(...)` gives an `AsyncWebApp` with the same methods as coroutines, and
`session()` takes an `AsyncTokenStore`.
