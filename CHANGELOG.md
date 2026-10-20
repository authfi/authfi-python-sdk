# Changelog

All notable changes to this package. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased] — 0.2.0

The SDK is rebuilt on one core that every area shares: people signing in, APIs verifying tokens,
machine clients and AI agents. See [docs/migration-0.2.md](docs/migration-0.2.md).

### Added

- `AuthFI` and `AsyncAuthFI`: the project client, configured by its issuer alone, with discovery
  (RFC 8414 / OpenID), the key set, and one HTTP transport shared by every area.
- Timeouts on every request (connect, read, and a deadline per call), and retries only where
  repeating a request is safe; a refusal is never retried.
- `X-Request-Id`, `traceparent` and a named `User-Agent` on every request; the platform's
  request id on every error.
- `resolve_api_origin()` on both clients, for building a machine-API URL on a project with a custom domain.
- Typed errors with stable codes (`AuthFIError.code`), including the platform's token refusal
  codes, generated from the AuthFI kernel.
- `Secret`, which does not print, pickle or serialise a credential.
- `TokenSource` and `AsyncTokenSource`: one interface for every credential, each refusing to be
  sent to an origin it was not made for.
- `TokenSet`, `TokenStore` and `InMemoryTokenStore` for a person's tokens, with refreshes made
  once per refresh token across threads and tasks.
- Type information (`py.typed`); the package passes `mypy --strict` and pyright in strict mode.

### Changed

- Python 3.11 or later is required (was 3.8).
- The package is configured by issuer. `tenant`, `api_key`, `api_url` and `base_domain` are no
  longer how a project is named.
- HTTP goes through httpx (was `urllib`, with no timeout).

### Removed

- The single-module `authfi.py` layout. The names it exported are kept, with warnings, as the
  areas that replace them land in this release.
