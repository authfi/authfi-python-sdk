"""``authfi`` on the command line: the agent's current token, for tools that run a command to get one.

    authfi token      the access token, on one line (Claude Code's apiKeyHelper)
    authfi headers    {"Authorization": "Bearer ..."} as JSON (an MCP server's headersHelper)

The agent is configured the way :meth:`authfi.AgentAuth.from_env` reads it:
``AUTHFI_PLATFORM_URL``, ``AUTHFI_AGENT_ID``, ``AUTHFI_AGENT_KEY_FILE``,
``AUTHFI_AGENT_KID`` and, optionally, ``AUTHFI_AGENT_AUDIENCE``. ``headers``
adds ``X-Request-Id`` from ``AUTHFI_RUN_ID`` when it is set. A failure prints
the reason on stderr and exits 1, with nothing on stdout, so the calling tool
never mistakes an error for a token.
"""

import json
import os
import sys

from . import AgentAuth, AuthFIError
from .gateway import request_id_header

USAGE = "usage: authfi token | authfi headers"


def main(argv=None):
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1 or args[0] not in ("token", "headers"):
        print(USAGE, file=sys.stderr)
        return 2
    try:
        token = AgentAuth.from_env().token()
    except AuthFIError as e:
        print("authfi: %s" % e, file=sys.stderr)
        return 1
    if args[0] == "token":
        sys.stdout.write(token + "\n")
    else:
        headers = {"Authorization": "Bearer " + token}
        headers.update(request_id_header(os.environ.get("AUTHFI_RUN_ID")))
        sys.stdout.write(json.dumps(headers) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
