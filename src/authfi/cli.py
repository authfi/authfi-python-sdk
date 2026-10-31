"""``authfi`` on the command line: make an agent key, and print the agent's current credential.

::

    authfi keygen --out agent.key [--alg EdDSA|ES256|PS256] [--kid KID]
        writes a new private key (readable by its owner only) and prints the public JWK and its
        thumbprint: what you register in the console
    authfi token
        the agent's current access token, on one line (Claude Code's apiKeyHelper)
    authfi headers
        {"Authorization": "Bearer ..."} as JSON (an MCP server's headersHelper); with
        AUTHFI_RUN_ID set, X-Request-Id too
    authfi redteam --origin URL [--baseline FILE] [--send]
        the platform's red-team corpus against your agent's configuration, read back from your call
        tape; without --send it prints what it would send (see :mod:`authfi.redteam`)

``token`` and ``headers`` read the agent as ``AuthFI.from_env().agent_from_env()`` does:
``AUTHFI_ISSUER``, ``AUTHFI_AGENT_ID``, ``AUTHFI_AGENT_KEY_FILE``, ``AUTHFI_AGENT_KID`` and
``AUTHFI_AGENT_RESOURCE``. A failure prints the reason on stderr and exits 1 with nothing on
stdout, so a calling tool never mistakes an error for a credential.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence

from authfi import redteam
from authfi._protocol.ids import is_valid_request_id
from authfi.agents.keys import AgentKey, KeyAlgorithm
from authfi.client import AuthFI
from authfi.errors import AuthFIError

__all__ = ["main"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="authfi", description="AuthFI agent credentials")
    commands = parser.add_subparsers(dest="command", required=True)
    keygen = commands.add_parser("keygen", help="make a new agent key and print its public half")
    keygen.add_argument("--out", required=True, help="the file to write the private key to (must not exist)")
    keygen.add_argument("--alg", choices=["EdDSA", "ES256", "PS256"], default="EdDSA")
    keygen.add_argument("--kid", help="the key id; the public key's thumbprint by default")
    commands.add_parser("token", help="print the agent's current access token")
    commands.add_parser("headers", help="print the agent's request headers as JSON")
    redteam.add_arguments(
        commands.add_parser("redteam", help="run the red-team corpus against your agent (a dry run unless --send)")
    )
    return parser


def _keygen(out: str, alg: str, kid: str | None) -> int:
    algorithm: KeyAlgorithm = "ES256" if alg == "ES256" else "PS256" if alg == "PS256" else "EdDSA"
    key = AgentKey.generate(algorithm, kid=kid)
    key.write(out)
    print(json.dumps({"public_jwk": key.public_jwk(), "thumbprint": key.thumbprint()}, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; returns the exit code."""
    args = _parser().parse_args(argv)
    if args.command == "redteam":
        return redteam.command(args, os.environ, print, lambda text: print(text, file=sys.stderr))
    try:
        if args.command == "keygen":
            return _keygen(args.out, args.alg, args.kid)
        with AuthFI.from_env() as af:
            # A Bearer token: the process that asked for it (Claude Code, an MCP server) cannot hold this key.
            token = af.agent_from_env(dpop=False).token()
    except AuthFIError as error:
        print(f"authfi: {error.message}", file=sys.stderr)
        return 1
    if args.command == "token":
        sys.stdout.write(token + "\n")
        return 0
    headers = {"Authorization": "Bearer " + token}
    run_id = os.environ.get("AUTHFI_RUN_ID", "")
    if run_id and is_valid_request_id(run_id):
        headers["X-Request-Id"] = run_id
    sys.stdout.write(json.dumps(headers) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
