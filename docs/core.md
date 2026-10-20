# The core

Everything in the SDK sits on one core: the project's issuer and discovery document, its key set,
one HTTP transport, client authentication, typed errors and secrets.

## The project client

```python
from authfi import AuthFI, Timeout

af = AuthFI(
    "https://acme.authfi.io",
    timeout=Timeout(connect=5, read=10, total=30),   # the defaults
    retries=2,                                       # the default
    leeway=60,                                       # seconds of clock skew on exp, nbf, iat
)
```

| Argument | What it is |
|---|---|
| `issuer` | Your project's address: `https://<slug>.authfi.io`, or its custom domain. Required. `http` is accepted for `localhost`, `127.0.0.1` and `::1` only. |
| `api_origin` | Where the project's machine APIs are served. Set it when the issuer is a custom domain; otherwise it is read from discovery. |
| `timeout` | A `Timeout`, or a number of seconds for the whole call. |
| `retries` | How many times a failed request may be sent again, 0 to 5. |
| `leeway` | Clock skew allowed when checking token times, 0 to 300 seconds. |
| `http_client` | Your own `httpx.Client` (`httpx.AsyncClient` for `AsyncAuthFI`), for a proxy or mutual TLS. The SDK never closes it. |
| `logger` | A `logging.Logger`; the `authfi` logger by default. |
| `user_agent_suffix` | Appended to the `User-Agent`. |

`AuthFI.from_env()` reads `AUTHFI_ISSUER` and `AUTHFI_API_ORIGIN`, and takes the same keyword
arguments.

`AsyncAuthFI` is the same client for asyncio code: every method that makes a request is a
coroutine, and `async with AsyncAuthFI(...)` closes it. One `AsyncAuthFI` belongs to one event
loop; a call from another loop raises `ConfigurationError`.

Create one client per project and process, and share it. In a server that forks workers
(gunicorn `--preload`, Celery prefork), each child resets the client's locks and connections on
its own; an `http_client` you passed is yours to recreate in the child.

## Discovery

`af.metadata()` returns the project's discovery document as a `ServerMetadata`. It is read from
`<issuer>/.well-known/openid-configuration` (falling back to
`/.well-known/oauth-authorization-server`), checked to name your issuer exactly, and kept for as
long as its `Cache-Control` says, between one minute and one day. Concurrent callers share one
fetch, and a failed fetch is not kept.

The project's signing keys are read from the document's `jwks_uri`. A key the token names but
the set does not hold causes one fetch of the set, at most once every 30 seconds.

## Requests: timeouts, retries and ids

Every request the SDK makes:

- has a connect timeout, a read timeout, and a deadline for the whole call, retries included;
- follows no redirect, and reads at most 1 MiB of an answer (256 KiB of a discovery document);
- sends a `User-Agent` naming the SDK, an `X-Request-Id` (the same on every retry of one call)
  and a W3C `traceparent` (from OpenTelemetry when `opentelemetry-api` is installed).

A request is sent again only when that is safe:

| Request | Sent again after |
|---|---|
| Reads, client-credentials and agent token requests, and any request carrying an `Idempotency-Key` | a connection failure, 429, 502, 503 or 504 |
| Anything that may have created state: a code or refresh-token exchange, a CIBA request, an approval request | only a failure that never sent it, 429, and 503 with `Retry-After` |

A refusal (any other 4xx) is never sent again. `Retry-After` is honoured within the deadline.

Each error carries the request id the platform answered with, as `error.request_id`; quote it
to support.

## Errors

| Class | Raised when |
|---|---|
| `ConfigurationError` | an argument or setting is wrong; no request was made |
| `TransportError` | no usable answer: timeout, connection, redirect, size; `.retryable` says whether calling again is safe |
| `DiscoveryError` | the discovery document or key set is missing, malformed, or names another issuer |
| `OAuthError` | an OAuth endpoint refused; `.code` is its `error` |
| `StepUpRequired` | the person must authenticate more strongly; `.acr_values` |
| `ConsentDenied`, `ConsentExpired` | a person declined, or their consent ran out |
| `TemporarilyUnavailable` | the platform could not decide now and failed closed |
| `APIError` | an AuthFI API refused; `.code` is its stable code |
| `NotSupportedError` | the project does not offer this to this kind of client; `.feature` |

Every error has `.code`, `.status` and `.request_id`. Branch on `.code`.

## Secrets

`authfi.Secret` holds a client secret, a token or a key:

```python
from authfi import Secret

secret = Secret(os.environ["AUTHFI_CLIENT_SECRET"])
print(secret)          # Secret('…')
secret.reveal()        # the value
```

It refuses `pickle` and `json.dumps`, and compares in constant time. Anything that takes a
client secret takes a `Secret`, so the place a raw value is wrapped shows in review.

## Credentials

Every credential the SDK makes is a `TokenSource` (`AsyncTokenSource` in asyncio code):

```python
source.token()                       # the current token, renewed when due
source.headers("GET", url)           # {"Authorization": "Bearer ..."}
httpx.Client(auth=source.httpx_auth())
```

Each knows where it may be sent (`source.send_to`) and refuses any other origin, so a token for
one API cannot be attached to a request for another. Read `token()` and `headers()` inside the
request that needs them, and keep no reference to what they return.

To bring a credential of your own, subclass `TokenSource` or `AsyncTokenSource`.

## Logging

The SDK logs to the `authfi` logger and never formats a credential into a message. A filter on
that logger also rewrites anything shaped like a JWT or a `Bearer` credential, as a second net.
A response that carries a `Deprecation` header is logged once per address, with its successor.
