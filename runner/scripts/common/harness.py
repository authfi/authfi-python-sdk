"""Run set-up shared by the framework scripts.

Two modes, one script per framework:

  np (default)   AUTHFI_ISSUER is an np project; the agent is the registered one the person supplied
                 (AUTHFI_AGENT_ID, AUTHFI_AGENT_KEY_FILE). Model and tool calls go through that project's gateways.
  offline        FW_OFFLINE=1: a stub gateway on 127.0.0.1 stands in (common/stub_gateway.py) with a fresh key.
                 It proves the SDK and the framework version work together and that no tracer sees the token. It proves
                 nothing about np. Every result file says which mode produced it.

Either way every tracer exports to a local capture server, and the check at the end looks for the agent's token (and
key material) in everything the tracers sent.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fwcommon as fw  # noqa: E402


@dataclass
class Ctx:
    mode: str
    issuer: str
    capture: fw.Capture
    stub: Any = None
    tmp: Path | None = None
    notes: list[str] = field(default_factory=list)

    def stub_tokens(self) -> list[str]:
        """Every access token the stub minted, so the leak check can look for them. Empty against np."""
        return [] if self.stub is None else list(self.stub.tokens) + list(self.stub.bearer_tokens)

    def note(self, text: str) -> None:
        self.notes.append(text)
        fw.log("  " + text)


def prepare(name: str) -> Ctx:
    cap = fw.Capture()
    for k, v in fw.tracer_env(cap).items():
        os.environ[k] = v
    if os.environ.get("FW_OFFLINE") == "1":
        from authfi import AgentKey
        from stub_gateway import StubGateway

        stub = StubGateway(nonce_on=os.environ.get("FW_NONCE", "1") == "1")
        tmp = Path(tempfile.mkdtemp(prefix=f"fw-{name}-"))
        key = AgentKey.generate()
        key_path = tmp / "agent.key"
        key.write(str(key_path))
        os.environ["AUTHFI_ISSUER"] = stub.url
        os.environ["AUTHFI_AGENT_ID"] = "agt_" + "a1" * 16
        os.environ["AUTHFI_AGENT_KEY_FILE"] = str(key_path)
        os.environ["AUTHFI_AGENT_KID"] = key.kid
        fw.log(f"[{name}] OFFLINE: stub gateway {stub.url} (nonce {'on' if stub.nonce_on else 'off'}); this is not np")
        return Ctx("offline", stub.url, cap, stub, tmp)
    iss = fw.issuer()
    fw.need_env("AUTHFI_AGENT_ID")
    if not os.environ.get("AUTHFI_AGENT_KEY_FILE") and not instance_evidence():
        gap = instance_gap()
        if gap:
            # A pod (it shows a service-account token) with part of its configuration missing: asking for a key file
            # would send the person to the older path. Say what the pod lacks.
            print("BLOCKED: this process runs in a pod but cannot sign in as an instance: " + "; ".join(gap))
            sys.exit(3)
        fw.need_env("AUTHFI_AGENT_KEY_FILE")
    fw.log(f"[{name}] np: {iss}")
    return Ctx("np", iss, cap)


def _token_file() -> str:
    """The service-account token this process would show, or "" when it runs outside a pod."""
    named = os.environ.get("AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE", "").strip()
    if named:
        return named
    default = Path("/var/run/secrets/kubernetes.io/serviceaccount/token")
    return str(default) if default.is_file() else ""


def instance_gap() -> list[str]:
    """What a pod lacks to sign in as an instance: [] when it is complete, or when this is not a pod at all.

    A pod needs a service-account token (projected, audience = the project origin) and the approved configuration
    it runs: AUTHFI_CONFIG_VERSION (a number) and AUTHFI_CONFIG_DIGEST (sha256:<hex>). The SDK takes both or neither, and
    the platform admits an instance only for an approved version, so neither alone is enough. There is no key file here.
    """
    token = _token_file()
    if not token:
        return []
    out: list[str] = []
    if not Path(token).is_file():
        out.append("the service-account token file %s does not exist" % token)
    version, digest = os.environ.get("AUTHFI_CONFIG_VERSION", "").strip(), os.environ.get("AUTHFI_CONFIG_DIGEST", "").strip()
    if not version.isdigit():
        out.append("AUTHFI_CONFIG_VERSION is not set to the approved version number")
    if not digest.startswith("sha256:"):
        out.append("AUTHFI_CONFIG_DIGEST is not set to sha256:<hex>")
    return out


def instance_evidence() -> bool:
    """True when this process can sign in as a bound instance: a pod with its token and a complete configuration."""
    return bool(_token_file()) and not instance_gap()


def sign_in_kind() -> str:
    return "instance (bound workload, no key)" if instance_evidence() and not os.environ.get("AUTHFI_AGENT_KEY_FILE") else "agent key file (legacy)"


def leak_check(ctx: Ctx, secrets_: list[str]) -> dict[str, Any]:
    """Search everything the tracers sent for the values in ``secrets_`` (tokens). Never prints a value."""
    found: dict[str, list[str]] = {}
    for i, s in enumerate(x for x in secrets_ if x):
        hits = ctx.capture.find(s)
        if hits:
            found[f"secret#{i}"] = sorted(set(hits))
    bodies = " ".join(str(r["body"]) for r in ctx.capture.requests) + " ".join(json.dumps(r["headers"]) for r in ctx.capture.requests)
    markers = {m: (m in bodies) for m in ("BEGIN PRIVATE KEY", "BEGIN OPENSSH PRIVATE KEY", "client_assertion=", "Authorization")}
    auth = [str(r["headers"].get("Authorization", r["headers"].get("authorization", ""))) for r in ctx.capture.requests]  # type: ignore[union-attr]
    other = [a[:7] + "..." for a in auth if a and "dummy-capture-key-not-a-real-key" not in a]
    markers.pop("Authorization")
    return {
        "token_in_tracer_payload": found,
        "markers_present": {k: v for k, v in markers.items() if v},
        "authorization_headers_other_than_the_dummy_key": other,
        "tracer_requests": ctx.capture.summary(),
        # the prompt asks about this repository: if the name is in a tracer payload, content is being exported
        "prompt_or_tool_content_in_tracer_payloads": len(ctx.capture.find("itsdangerous")),
    }


def finish(ctx: Ctx, name: str, result: dict[str, Any], secrets_: list[str]) -> int:
    result["mode"] = ctx.mode
    result["issuer"] = ctx.issuer
    result["sign_in"] = sign_in_kind() if ctx.mode == "np" else "agent key file (stub)"
    result["leak_check"] = leak_check(ctx, secrets_)
    if ctx.stub is not None:
        ev = ctx.stub.events
        result["gateway_events"] = {
            "count": len(ev),
            "nonce_challenges": sum(1 for e in ev if e.get("why") == "use_dpop_nonce"),
            "ok_with_dpop": sum(1 for e in ev if e.get("status") == 200 and e.get("dpop")),
            "refused_not_dpop": [e for e in ev if e.get("why") in ("not_a_bound_token", "no_proof")][:5],
            "paths": sorted({e["path"] for e in ev}),
            "tools_called": [e["tool"] for e in ev if e.get("tool")],
        }
    result["notes"] = ctx.notes
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = fw.save(f"{name}_{ctx.mode}_{stamp}.json", result)
    leaked = bool(result["leak_check"]["token_in_tracer_payload"]) or bool(result["leak_check"]["authorization_headers_other_than_the_dummy_key"])
    fw.log(f"[{name}] {'LEAK: a token reached a tracer' if leaked else 'no token in any tracer payload'}; saved {path.name}")
    if ctx.stub is not None:
        ctx.stub.stop()
    ctx.capture.stop()
    return 1 if leaked else 0
