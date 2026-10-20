# Protect an API

An orders API, in FastAPI, that accepts AuthFI access tokens.

## What it shows

- **Choosing the audience.** A token names who it is for in `aud`. A machine client's or an
  agent's token names your API when it is minted with `resource=<your API's client id>`. A person
  signing in to one of your front ends gets a token for that front end, so add its client id to
  `audience` if your people call this API directly.
- **Permissions, not scope.** `permissions` and `roles` are resolved by your project's policy
  when the token is minted. On a person's or a machine client's token, `scope` is what the client
  asked for, so check it only for scopes your API defines.
- **The answers.** A missing or bad token is answered 401 with one opaque description; a token
  without the permission, 403. The precise reason is logged on the `authfi` logger.
- **Step-up.** The refund route asks a person for a recent strong sign-in; the 401 it answers
  tells the client which `acr_values` to sign in with.

## Run it

```bash
pip install "authfi[fastapi] @ git+https://github.com/authfi/authfi-python-sdk@main" uvicorn
cp .env.example .env    # fill it in
set -a; . ./.env; set +a
uvicorn orders_api:app
```

In the AuthFI console, register the API as an application and copy its client id into
`ORDERS_API_CLIENT_ID`. Grant `orders:read` to the people and clients that may call it.
