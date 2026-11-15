"""``authfi`` on the command line: make an agent key, and print the agent's current credential.

::

    authfi keygen --out agent.key [--alg EdDSA|ES256|PS256] [--kid KID]
        writes a new private key (readable by its owner only) and prints the public JWK and its
        thumbprint: what you register in the console
    authfi token
        the agent's current access token, on one line (a tool's apiKeyHelper)
    authfi login --issuer URL [--no-browser] [--reregister]
        signs a person in once, in a browser, for the project's /llm door (needs authfi[login])
    authfi token --resource llm [--issuer URL]
        a fresh person access token for /llm, one line (Claude Code's apiKeyHelper, run as ``python -I -m authfi``;
        refuses to start otherwise: ``authfi login`` prints the exact command); bare ``token`` is the agent's
    authfi logout [--issuer URL] [--forget-allow]
        revokes the person's refresh token and deletes what is stored
    authfi headers
        {"Authorization": "Bearer ..."} as JSON (an MCP server's headersHelper); with
        AUTHFI_RUN_ID set, X-Request-Id too
    authfi claude [--model M] [--small-model M] [-- ARGS]
        Claude Code, through the AuthFI gateways as the agent the environment names, with no Anthropic
        sign-in (see :mod:`authfi.claude_code`)
    authfi redteam --origin URL [--baseline FILE] [--send]
        the platform's red-team corpus against your agent's configuration, read back from your call
        tape; without --send it prints what it would send (see :mod:`authfi.redteam`)

``token`` (without ``--resource``) and ``headers`` read the agent as ``AuthFI.from_env().agent_from_env()`` does:
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

import httpx

from authfi import _login, claude_code, redteam
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
    token = commands.add_parser("token", help="print the agent's current access token")
    token.add_argument("--resource", choices=["llm"], help="print the signed-in person's token for this door instead")
    token.add_argument("--issuer", help="the project origin; default AUTHFI_ISSUER, else the one signed in to")
    token.add_argument(
        "--min-validity",
        type=float,
        default=_login.DEFAULT_MIN_VALIDITY,
        help="seconds a reused token must still have left (--resource only)",
    )
    login = commands.add_parser("login", help="sign a person in, for Claude Code's use of the /llm door")
    login.add_argument("--issuer", help="the project origin, such as https://acme.authfi.io")
    login.add_argument("--no-browser", action="store_true", help="print the address instead of opening a browser")
    login.add_argument("--reregister", action="store_true", help="register this machine's client again")
    login.add_argument(
        "--allow-proxy",
        action="append",
        default=[],
        metavar="HOST",
        help="let `authfi token` run behind this proxy host",
    )
    login.add_argument(
        "--allow-env",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="let `authfi token` run with this exact CA or TLS variable value",
    )
    login.add_argument("--timeout", type=float, default=_login.CALLBACK_TIMEOUT, help="seconds to wait for the browser")
    logout = commands.add_parser("logout", help="revoke the person's refresh token and delete what is stored")
    logout.add_argument("--issuer", help="the project origin; default the one signed in to")
    logout.add_argument(
        "--forget-allow", action="store_true", help="also drop every proxy host and value allowed with `authfi login`"
    )
    commands.add_parser("headers", help="print the agent's request headers as JSON")
    claude = commands.add_parser("claude", help="run Claude Code through the AuthFI gateways")
    claude.add_argument("--model", default=claude_code.DEFAULT_MODEL, help="the model the session starts on")
    claude.add_argument(
        "--small-model", default=claude_code.DEFAULT_MODEL, help="the model Claude Code uses for its background work"
    )
    claude.add_argument("args", nargs=argparse.REMAINDER, help="arguments for claude, after --")
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


def _person_command(args: argparse.Namespace) -> int:
    """``login``, ``logout`` and ``token --resource``: a person's own credential, kept in the credential store."""
    try:
        runtime = _login.Runtime(env=os.environ, vault=_login.open_vault(), http_client=_http_client())
        if args.command == "login":
            return _login.login(
                runtime,
                args.issuer,
                browser=not args.no_browser,
                reregister=args.reregister,
                timeout=args.timeout,
                allow_proxies=args.allow_proxy,
                allow_env=args.allow_env,
            )
        if args.command == "logout":
            return _login.logout(runtime, args.issuer, forget_allow=args.forget_allow)
        token = _login.access_token(runtime, args.issuer, min_validity=args.min_validity)
    except AuthFIError as error:
        print(f"authfi: {error.message}", file=sys.stderr)
        return 1
    sys.stdout.write(token + "\n")
    return 0


def _isolated() -> bool:
    """``True`` when Python was started with ``-I``: no ``PYTHON*`` variables, no user site, no working directory."""
    return bool(sys.flags.isolated)


def _http_client() -> httpx.Client | None:
    """The HTTP client the person commands use; the SDK's own by default."""
    return None


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command; returns the exit code."""
    args = _parser().parse_args(argv)
    if args.command == "token" and args.resource and not _isolated():
        # Python has already read PYTHONPATH and the user site by now: the environment guard cannot undo that.
        print(
            "authfi: this helper was not started in isolated mode, so Python may already have run code chosen by "
            "the environment (PYTHONPATH, the user site or the working directory); set Claude Code's apiKeyHelper "
            f"to: {_login.helper_command()}",
            file=sys.stderr,
        )
        return 1
    if args.command in ("login", "logout") or (args.command == "token" and args.resource):
        return _person_command(args)
    if args.command == "claude":
        passed = list(args.args)
        return claude_code.command(
            passed[1:] if passed[:1] == ["--"] else passed, model=args.model, small_model=args.small_model
        )
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
