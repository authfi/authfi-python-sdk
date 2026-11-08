"""B3 on a live np project, from the SDK: what a registered agent gets right and what the platform refuses.

    AUTHFI_B3_ISSUER=https://probe-c1.np.authfi.io AUTHFI_B3_AGENT_ID=agt_... AUTHFI_B3_KEY_FILE=agent.key \
        AUTHFI_B3_KID=<the key id registered with the agent> \
        python scripts/b3_np_check.py [--out result.json]

Each refusal is paired with a control that is admitted, so a refusal is the rule and not a dead endpoint. The legs:

  mint      the agent gets a token bound to its run key (``cnf.jkt``), and the nonce identity asks for, if it asks
  proved    ``tools/list`` with the proof is admitted (control)
  bearer    the same bound token sent as Bearer is refused
  noproof   ``Authorization: DPoP`` with no proof is refused
  replay    a proof sent twice is admitted once
  htu       a proof for another URL is refused
  htm       a proof for another method is refused
  ath       a proof made for another token is refused
  nonce     identity: whether the token endpoint asked for a nonce (``DPoP-Nonce`` on the first attempt)

It prints no token, key or proof. It refuses a host that is not ``*.np.authfi.io`` and never touches production.
Exit 0 when every leg behaved, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from functools import partial
from typing import Any
from urllib.parse import urlsplit

import httpx

from authfi import AgentKey, AuthFI, _dpop

JSON_RPC: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}


def _np_only(issuer: str) -> str:
    host = urlsplit(issuer).hostname or ""
    if not host.endswith(".np.authfi.io") or urlsplit(issuer).scheme != "https":
        raise SystemExit(f"refused: {issuer!r} is not an https *.np.authfi.io project")
    return issuer.rstrip("/")


def main() -> int:
    """Run the legs against the project in the environment; 0 when every one behaved."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    issuer = _np_only(os.environ["AUTHFI_B3_ISSUER"])
    agent_id, key_file = os.environ["AUTHFI_B3_AGENT_ID"], os.environ["AUTHFI_B3_KEY_FILE"]
    mcp = issuer + "/mcp"
    token_requests: list[dict[str, Any]] = []

    def hook(request: httpx.Request) -> None:
        if request.url.path == "/oauth/token":
            token_requests.append(
                {"proof_nonce": _proof_claims(request.headers.get("dpop", "")).get("nonce") is not None}
            )

    legs: dict[str, dict[str, Any]] = {}

    def leg(name: str, want: str, got: str, ok: bool) -> None:
        legs[name] = {"want": want, "got": got, "ok": ok}

    with AuthFI(issuer) as af, httpx.Client(timeout=30, event_hooks={"request": [hook]}) as http:
        agent = af.agent(agent_id, AgentKey.from_file(key_file, kid=os.environ["AUTHFI_B3_KID"]))
        token = agent.token()
        key = agent.bound_key()
        leg("mint", "a token bound to the run key", "bound" if key is not None else "bearer", key is not None)
        if key is None:
            raise SystemExit("the token is not bound: nothing to check")

        def post(headers: dict[str, str]) -> httpx.Response:
            return http.post(mcp, json=JSON_RPC, headers={"Content-Type": "application/json", **headers})

        def proof(method: str = "POST", url: str = mcp, tok: str = token) -> str:
            return key.proof(method, url, now=time.time(), access_token=tok, nonce=_dpop.nonces.get(url))

        def dpop(proof_value: str | None, tok: str = token) -> dict[str, str]:
            return {"Authorization": f"DPoP {tok}", **({"DPoP": proof_value} if proof_value else {})}

        def settle(make: Callable[[], dict[str, str]]) -> httpx.Response:
            """Send once; a nonce challenge is answered once, as an SDK does."""
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
            leg(name, "401", str(r.status_code), r.status_code == 401)
        replayed = proof()
        first, second = post(dpop(replayed)), post(dpop(replayed))
        leg("replay", "admitted once", f"{first.status_code} then {second.status_code}", second.status_code == 401)
        other = {
            "htu": proof(url=issuer + "/mcp/other"),
            "htm": proof(method="GET"),
            "ath": proof(tok=token[:-2] + ("AA" if not token.endswith("AA") else "BB")),
        }
        for name, bad in other.items():
            r = settle(partial(dpop, bad))
            leg(name, "401", str(r.status_code), r.status_code == 401)
        asked = bool(token_requests) and len(token_requests) > 1
        legs["nonce"] = {
            "want": "informational: identity asks for a nonce once DPOP_NONCE_KEY is set (B3 by 14 Nov)",
            "got": "asked and answered" if asked else "not asked",
            "ok": True,
        }

    ok = all(item["ok"] for item in legs.values())
    result = {"issuer": issuer, "agent": agent_id, "ok": ok, "legs": legs}
    print(json.dumps(result, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as out:
            json.dump(result, out, indent=2)
    return 0 if ok else 1


def _proof_claims(proof: str) -> dict[str, Any]:
    import jwt

    try:
        return jwt.decode(proof, options={"verify_signature": False})
    except jwt.PyJWTError:
        return {}


if __name__ == "__main__":
    sys.exit(main())
