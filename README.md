# AuthFI Python SDK

The AuthFI SDK for Python: one OAuth core for every way your code meets AuthFI.

- **People** sign in to your app.
- **APIs** verify the tokens that reach them.
- **Machine clients** call APIs with their own token.
- **AI agents** hold their own key, act for a person with consent, and go through the AuthFI
  gateways.

Every area is made from one project client, so they share one discovery document, one key set
and one HTTP connection pool.

## Install

The package is installed from this repository until it is published:

```bash
pip install "authfi @ git+https://github.com/authfi/authfi-python-sdk@main"
```

Python 3.11 or later. The dependencies are httpx, PyJWT and cryptography.

## The project client

```python
from authfi import AuthFI

af = AuthFI("https://acme.authfi.io")   # your project's address; or AuthFI.from_env() reads AUTHFI_ISSUER

metadata = af.metadata()                 # the discovery document, cached for its lifetime
print(metadata.token_endpoint)
```

The issuer is the one required setting. Everything else is read from the project's discovery
document. Create one client per project and share it: it owns the caches. In asyncio code, use
`AsyncAuthFI`, which has the same methods as coroutines; create one per event loop.

## Errors

Every error is an `authfi.AuthFIError`. Branch on its `code`, never on the message:

```python
from authfi import AuthFIError, OAuthError

try:
    metadata = af.metadata()
except OAuthError as e:
    match e.code:
        case "invalid_client":
            ...
        case _:
            raise
```

A code is an open set: keep a default arm, because the platform adds codes between releases.

## Secrets

Client secrets, tokens and keys are held in `authfi.Secret`. It prints as `Secret('…')`, refuses
to be pickled or serialised, and gives its value only through `reveal()`.

## Documentation

- [The core: the project client, timeouts and retries, errors, secrets, credentials](docs/core.md)
- [Moving from 0.1](docs/migration-0.2.md)
- [Changelog](CHANGELOG.md)

## Licence

Apache-2.0. See [LICENSE](LICENSE).
