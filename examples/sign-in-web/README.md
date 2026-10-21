# Sign people in to a web app

Two small apps, one on FastAPI and one on Flask, that sign people in with AuthFI, show who signed in,
and sign them out.

## What it shows

- **The code flow with PKCE, state, nonce and `iss`.** `start_sign_in` and `finish_sign_in` do all of
  it, and verify the ID token.
- **The pending sign-in stays encrypted.** It travels in a cookie only as `pending.dump()` gives it;
  Flask's session cookie is signed, not encrypted, so nothing readable goes there.
- **Tokens stay on the server**, keyed by a random session id. With several workers, put them in a
  `TokenStore` on your database or cache.
- **Sign-out revokes the refresh token** before ending the AuthFI session.

## Set up

In the AuthFI console, add a web application with the redirect URI `http://localhost:8000/callback`
(or your app's address) and the sign-out redirect `http://localhost:8000/`.

```bash
pip install "authfi[fastapi] @ git+https://github.com/authfi/authfi-python-sdk@main" uvicorn
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
uvicorn fastapi_app:app --port 8000
# or: pip install flask && flask --app flask_app run --port 8000
```
