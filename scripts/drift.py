"""Check that a live AuthFI project still publishes what this SDK depends on.

Reads only public documents (no credential)::

    python scripts/drift.py https://platform-beta.authfi.io

It fails when a field the SDK reads is missing or changed in a way the SDK does not handle, so a
platform change is seen before a customer meets it.
"""

from __future__ import annotations

import sys
from typing import Any, cast

import httpx

REQUIRED_DISCOVERY = ("issuer", "token_endpoint", "jwks_uri", "authorization_endpoint", "revocation_endpoint")
REQUIRED_GRANTS = (
    "authorization_code",
    "client_credentials",
    "refresh_token",
    "urn:ietf:params:oauth:grant-type:token-exchange",
    "urn:openid:params:grant-type:ciba",
)
REQUIRED_AUTH_METHODS = ("client_secret_basic", "client_secret_post", "none", "private_key_jwt")
ACCEPTED_KEY_ALGS = {"RS256", "ES256", "EdDSA"}


def _get(client: httpx.Client, url: str) -> dict[str, Any]:
    response = client.get(url)
    response.raise_for_status()
    body: object = response.json()
    if not isinstance(body, dict):
        raise SystemExit(f"{url} is not a JSON object")
    return cast(dict[str, Any], body)


def check(issuer: str) -> list[str]:
    """The problems found, empty when the project publishes what the SDK needs."""
    problems: list[str] = []
    with httpx.Client(headers={"User-Agent": "authfi-python-sdk-drift/1"}, timeout=15) as client:
        doc = _get(client, issuer + "/.well-known/openid-configuration")
        if doc.get("issuer") != issuer:
            problems.append(f"discovery names issuer {doc.get('issuer')!r}")
        problems += [f"discovery has no {name}" for name in REQUIRED_DISCOVERY if not doc.get(name)]
        grants = cast(list[str], doc.get("grant_types_supported") or [])
        problems += [f"grant {g} is no longer advertised" for g in REQUIRED_GRANTS if g not in grants]
        methods = cast(list[str], doc.get("token_endpoint_auth_methods_supported") or [])
        problems += [
            f"client authentication {m} is no longer advertised" for m in REQUIRED_AUTH_METHODS if m not in methods
        ]
        if "S256" not in cast(list[str], doc.get("code_challenge_methods_supported") or []):
            problems.append("PKCE S256 is no longer advertised")
        keys = cast(list[object], _get(client, str(doc.get("jwks_uri"))).get("keys") or [])
        algs = {cast(dict[str, Any], k).get("alg") for k in keys if isinstance(k, dict)}
        if not algs & ACCEPTED_KEY_ALGS:
            problems.append(f"the key set has no key the SDK verifies: {sorted(map(str, algs))}")
        agent = _get(client, issuer + "/.well-known/agent-configuration")
        endpoints = cast(dict[str, Any], agent.get("endpoints") or {})
        for name in ("request_approval", "approval"):
            if not endpoints.get(name):
                problems.append(f"agent-configuration has no endpoints.{name}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Print each problem; exit 1 when there is any."""
    args = sys.argv[1:] if argv is None else argv
    issuer = args[0] if args else "https://platform-beta.authfi.io"
    problems = check(issuer)
    for problem in problems:
        print(problem)
    print(f"{issuer}: {'ok' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
