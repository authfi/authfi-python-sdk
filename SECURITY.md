# Security

## Reporting a vulnerability

Please report a vulnerability privately, through GitHub's **Report a vulnerability** button on
this repository's Security tab, or by email to security@authfi.io. Do not open a public issue.

Include what you found, how to reproduce it, and the version or commit. We acknowledge a report
within three working days and keep you informed until it is fixed.

## Supported versions

The latest released minor version receives security fixes.

## How this SDK protects credentials

- Every credential is held in `authfi.Secret`, which does not print, pickle or serialise.
- A credential is sent only to the origins its source was made for.
- Every request has a timeout, follows no redirect, and is refused over plain `http` except to
  loopback.
- An API built with the SDK verifies the token type, issuer, audience, expiry and tenant of every
  token, and tells a caller only that its token was refused.
