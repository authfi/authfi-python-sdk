# Verifying webhooks

AuthFI signs every webhook delivery. `authfi.webhooks.verify` checks the signature and the age and returns the event.

```python
from authfi import Secret, WebhookVerificationError
from authfi.webhooks import verify

SECRET = Secret(os.environ["AUTHFI_WEBHOOK_SECRET"])      # the subscription's signing secret

@app.post("/authfi/webhooks")
async def receive(request: Request) -> Response:
    body = await request.body()                           # the bytes as received, before any JSON parsing
    try:
        event = verify(body, request.headers, SECRET)
    except WebhookVerificationError as error:
        return Response(status_code=400)                  # error.code says which check failed
    if event.type == "user.created":
        ...
    return Response(status_code=204)
```

| Header | |
|---|---|
| `X-AuthFI-Signature-256` | `sha256=` and the hex HMAC-SHA256 of `timestamp + "." + body`, keyed by the secret |
| `X-AuthFI-Timestamp` | Unix seconds when the delivery was signed |
| `X-AuthFI-Delivery` | the delivery's id, the same on a redelivery |
| `X-AuthFI-Event` | the event name (unsigned: `verify` reads it from the signed body) |

`verify(body, headers, secret, tolerance=300)` compares in constant time, refuses a timestamp more than
`tolerance` seconds from now in either direction, and raises `WebhookVerificationError` with a `code`:
`webhook_signature_missing`, `webhook_signature_malformed`, `webhook_timestamp_missing`,
`webhook_timestamp_invalid`, `webhook_signature_mismatch`, `webhook_timestamp_outside_tolerance` or
`webhook_body_invalid`. Pass a list of secrets while you rotate one.

The event has `id`, `type`, `timestamp`, `actor`, `subject`, `resource_type`, `resource_id`, `delivery_id` and
the whole decoded `body`. `type` is one of `authfi.webhooks.EVENT_TYPES` today (`user.created`,
`user.updated`, `user.deleted`, `user.blocked`, `user.mfa_enrolled`, `group.member_added`,
`group.member_removed`, `organization.created`, `role.assigned`, `session.revoked`, `login.success`,
`login.failed`); keep a default branch for a name this release does not list.

A delivery replayed inside the tolerance window verifies again: record `delivery_id` and ignore one you have
handled. `authfi.webhooks.sign(secret, body)` makes the headers, to test your endpoint.
