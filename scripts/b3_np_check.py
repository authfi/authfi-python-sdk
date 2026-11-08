"""B3 on a live np project, from the SDK: what a registered agent gets right and what the platform refuses.

    AUTHFI_B3_ISSUER=https://probe-c1.np.authfi.io AUTHFI_B3_AGENT_ID=agt_... AUTHFI_B3_KEY_FILE=agent.key \
        AUTHFI_B3_KID=<the key id registered with the agent> \
        python scripts/b3_np_check.py [--out result.json]

Each refusal is paired with a control that is admitted, so a refusal is the rule and not a dead endpoint. The legs:

  mint      the agent gets a token bound to its run key (``cnf.jkt``)
  proved    ``tools/list`` with the proof is admitted (control)
  bearer    the same bound token sent as Bearer is refused
  noproof   ``Authorization: DPoP`` with no proof is refused
  replay    a proof sent twice: the first is admitted (200), the second refused
  htu       a proof for another URL is refused
  htm       a proof for another method is refused
  ath       a proof made for another token is refused
  nonce     whether the token endpoint asked for a nonce and the SDK answered it. Informational until identity is
            set to ask (``DPOP_NONCE_KEY``, B3 by 14 Nov); with ``AUTHFI_B3_EXPECT_NONCE=1`` it fails unless the
            token endpoint asked and the SDK answered, so use that on the rerun

Every refusal leg records why it was refused (the ``error`` of the response). The proof of htu, htm and ath carries
the nonce ``/mcp`` asked for and is made again after a challenge, so only the broken property can refuse it: a leg
fails when its refusal is a nonce challenge.

It prints no token, key or proof. It refuses a host that is not an https ``probe-*.np.authfi.io`` project and never
touches production. Exit 0 when every leg behaved, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from functools import partial
from typing import Any, cast
from urllib.parse import urlsplit

import httpx
import jwt

from authfi import AgentKey, AuthFI, _dpop

JSON_RPC: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def _np_only(issuer: str) -> str:
    """Live proofs run in the probe projects on np only: ``https://probe-<name>.np.authfi.io``."""
    parts = urlsplit(issuer)
    host = parts.hostname or ""
    if parts.scheme != "https" or not host.startswith("probe-") or not host.endswith(".np.authfi.io"):
        raise SystemExit(f"refused: {issuer!r} is not an https probe-*.np.authfi.io project")
    return issuer.rstrip("/")


def _reason(response: httpx.Response) -> str:
    """Why a request was refused: the ``error`` of ``WWW-Authenticate`` and of the body, as the platform sent them."""
    found = re.findall(r'error="([^"]+)"', response.headers.get("www-authenticate", ""))
    try:
        body: object = response.json()
    except ValueError:
        body = None
    if isinstance(body, dict):
        fields = cast(dict[str, Any], body)
        found += [str(fields[k]) for k in ("error", "code") if isinstance(fields.get(k), str)]
    return ",".join(dict.fromkeys(found)) or "none given"


def main() -> int:
    """Run the legs against the project in the environment; 0 when every one behaved."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    result = check(
        _np_only(os.environ["AUTHFI_B3_ISSUER"]),
        os.environ["AUTHFI_B3_AGENT_ID"],
        os.environ["AUTHFI_B3_KEY_FILE"],
        os.environ["AUTHFI_B3_KID"],
        expect_nonce=os.environ.get("AUTHFI_B3_EXPECT_NONCE") == "1",
    )
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
    return 0 if result["ok"] else 1


def check(issuer: str, agent_id: str, key_file: str, kid: str, *, expect_nonce: bool = False) -> dict[str, Any]:
    """Run the legs against ``issuer``; ``{"issuer", "agent", "ok", "legs"}``."""
    mcp = issuer + "/mcp"
    token_requests: list[bool] = []  # per request to the token endpoint: did the proof carry a nonce
    token_challenges: list[int] = []  # the status of each token-endpoint answer that carried a ``DPoP-Nonce``

    def on_request(request: httpx.Request) -> None:
        if request.url.path == "/oauth/token":
            token_requests.append(_proof_claims(request.headers.get("dpop", "")).get("nonce") is not None)

    def on_response(response: httpx.Response) -> None:
        if response.request.url.path == "/oauth/token" and response.headers.get("dpop-nonce"):
            token_challenges.append(response.status_code)

    legs: dict[str, dict[str, Any]] = {}

    def leg(name: str, want: str, got: str, ok: bool, reason: str | None = None) -> None:
        legs[name] = {"want": want, "got": got, "ok": ok, **({"reason": reason} if reason is not None else {})}

    # One client for the agent's mints and for these requests, so the hooks see both.
    with httpx.Client(timeout=30, event_hooks={"request": [on_request], "response": [on_response]}) as http:
        with AuthFI(issuer, http_client=http) as af:
            agent = af.agent(agent_id, AgentKey.from_file(key_file, kid=kid))
            token = agent.token()
            key = agent.bound_key()
            leg("mint", "a token bound to the run key", "bound" if key is not None else "bearer", key is not None)
            if key is None:
                raise SystemExit("the token is not bound: nothing to check")

            def post(headers: dict[str, str]) -> httpx.Response:
                return http.post(mcp, json=JSON_RPC, headers={"Content-Type": "application/json", **headers})

            def proof(method: str = "POST", url: str = mcp, tok: str = token) -> str:
                # A proof for the wrong URL still carries the nonce /mcp asked for: only the broken property refuses it.
                return key.proof(method, url, now=time.time(), access_token=tok, nonce=_dpop.nonces.get(mcp))

            def dpop(proof_value: str | None) -> dict[str, str]:
                return {"Authorization": f"DPoP {token}", **({"DPoP": proof_value} if proof_value else {})}

            def settle(make: Callable[[], dict[str, str]]) -> httpx.Response:
                """Send once; a nonce challenge is answered once with a new proof (``make`` is called again)."""
                response = post(make())
                if _dpop.is_nonce_challenge_headers(
                    response.status_code, response.headers.get("www-authenticate"), response.headers.get("dpop-nonce")
                ):
                    _dpop.nonces.remember(mcp, response.headers.get("dpop-nonce"))
                    response = post(make())
                return response

            control = settle(lambda: dpop(proof()))
            leg("proved", "200", str(control.status_code), control.status_code == 200)
            for name, headers in {
                "bearer": {"Authorization": f"Bearer {token}"},
                "noproof": dpop(None),
            }.items():
                r = post(headers)
                leg(name, "401", str(r.status_code), r.status_code == 401, _reason(r))
            replayed = proof()
            first, second = post(dpop(replayed)), post(dpop(replayed))
            leg(
                "replay",
                "200 then 401",
                f"{first.status_code} then {second.status_code}",
                first.status_code == 200 and second.status_code == 401,
                _reason(second),
            )
            wrong_token = token[:-2] + ("BB" if token.endswith("AA") else "AA")
            wrong: dict[str, Callable[[], str]] = {
                "htu": partial(proof, url=issuer + "/mcp/other"),
                "htm": partial(proof, method="GET"),
                "ath": partial(proof, tok=wrong_token),
            }
            for name, make_proof in wrong.items():
                r = settle(lambda make_proof=make_proof: dpop(make_proof()))
                why = _reason(r)
                leg(
                    name,
                    "401, not for a nonce",
                    str(r.status_code),
                    r.status_code == 401 and "use_dpop_nonce" not in why,
                    why,
                )
            asked = len(token_requests) > 1 and any(token_requests) and bool(token_challenges)
            legs["nonce"] = {
                "want": "asked and answered"
                if expect_nonce
                else "informational: asked once DPOP_NONCE_KEY is set (B3, 14 Nov)",
                "got": "asked and answered" if asked else "not asked",
                "ok": asked or not expect_nonce,
            }

    return {"issuer": issuer, "agent": agent_id, "ok": all(item["ok"] for item in legs.values()), "legs": legs}


def _proof_claims(proof: str) -> dict[str, Any]:
    try:
        return jwt.decode(proof, options={"verify_signature": False})
    except jwt.PyJWTError:
        return {}


if __name__ == "__main__":
    sys.exit(main())
