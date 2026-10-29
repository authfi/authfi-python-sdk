# SAML sign-in for a standard app

An app that speaks SAML 2.0 and no OpenID Connect uses AuthFI as its identity provider. `authfi.saml` is
the service-provider side: it reads the project's metadata, sends the browser to AuthFI, and turns the
returned assertion into the same user model the OpenID Connect sign-in gives.

```bash
pip install "authfi[saml]"
```

Register the app in your project first: its entity id and its assertion consumer URL (ACS). Then:

```python
from authfi import AuthFI
from authfi.saml import PendingSamlSignIn, SamlServiceProvider

af = AuthFI.from_env()
sp = SamlServiceProvider.from_project(
    af,
    sp_entity_id="https://app.example.com/saml",     # as registered in the project
    acs_url="https://app.example.com/saml/acs",      # as registered in the project
)

# GET /login
pending = sp.start_sign_in(relay_state="/dashboard")
session["saml"] = pending.dump()                      # keep it server-side
return redirect(pending.url)

# POST /saml/acs
pending = PendingSamlSignIn.load(session.pop("saml"))
signed_in = sp.finish_sign_in(request.form["SAMLResponse"], pending)
signed_in.user.sub          # the assertion's NameID
signed_in.user.email, signed_in.user.groups, signed_in.user.roles
signed_in.relay_state       # "/dashboard"
```

`from_project` reads `<project origin>/saml/metadata` (use `await SamlServiceProvider.afrom_project(...)` with
an `AsyncAuthFI`) and refuses metadata that does not name that address as its entity id. Only the first
certificate of the signing key descriptor may sign an assertion.

## What is checked

`finish_sign_in` accepts an assertion only when all of this holds, and raises `SamlAssertionRefused` with a
stable `code` otherwise:

| check | code on failure |
|---|---|
| the Assertion is signed with a key in the project's metadata (a SHA-1 signature is refused) | `saml_signature_missing`, `saml_signature_invalid` |
| exactly one assertion, and the signed element is the one read | `saml_assertion_count`, `saml_signed_element_wrong` |
| the audience is your `sp_entity_id` | `saml_audience_mismatch` |
| the recipient and destination are your `acs_url` (not the host of the request that arrived) | `saml_subject_confirmation_invalid`, `saml_destination_mismatch` |
| the issuer is the project | `saml_issuer_mismatch` |
| `InResponseTo` is the request this browser started | `saml_in_response_to_mismatch` |
| it is inside its validity window, with `leeway` seconds of skew | `saml_assertion_expired`, `saml_assertion_not_yet_valid` |
| the status is Success | `saml_status_not_success` |
| it was not used before | `saml_assertion_replayed` |
| the sign-in was started in the last ten minutes | `saml_pending_expired` |

A response with a DTD, a non-base64 body or an encrypted assertion is refused too.

An assertion is accepted once. The default replay store is per process: pass `replay_store=` (anything with
`remember(key, expires_at, now) -> bool`) backed by Redis or your database when several processes serve the
ACS.

## The user

`signed_in.user` is an `IdTokenClaims`: `sub` is the NameID, `email` comes from the `email` attribute (or an
email-format NameID), `groups` and `roles` from the attributes of those names, `acr` from the authentication
context, `auth_time` from `AuthnInstant`, and `raw["saml_attributes"]` holds every attribute. `email_verified`
is never claimed. Pass `attributes={"email": ("work_email",)}` to read other attribute names.

## Launching from the AuthFI portal

A tile in the portal starts at the identity provider, so the response answers no request of yours. Build the
provider with `allow_idp_initiated=True` and call `finish_sign_in(response, None)`. It is still signed,
checked and single-use; do not trust a `RelayState` that arrives that way.
