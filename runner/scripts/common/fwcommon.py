"""Shared helpers for the 2026-10-31 agent framework tests: np only, the AuthFI SDK as a customer uses it.

What this module is, and is not:
  - It names the np hosts from ONE place (the environment), refuses every other host, and refuses tenant 4e46e0c4.
  - It scrubs anything token-shaped from what a test prints.
  - It starts the local capture server every tracer exports to. No tracer ever gets a real endpoint.
  - It reads credentials only from the environment or a file the person names. It never prints or asks for one.
"""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

#: The only suffix a test may talk to. np is ine1; as1 and prod are *.authfi.io and are refused.
NP_SUFFIX = ".np.authfi.io"
#: The tenant that was retired on 2026-10-25. It must never be used.
RETIRED_TENANT = "4e46e0c4"
#: Staff only. A test never signs in as this account.
NEVER_ACCOUNT = "admin@authfi.io"

OUT = Path(__file__).resolve().parent.parent / "out"


def issuer() -> str:
    """The np project the test runs against: AUTHFI_ISSUER, e.g. https://probe-c1.np.authfi.io."""
    value = os.environ.get("AUTHFI_ISSUER", "").strip().rstrip("/")
    if not value:
        raise SystemExit("AUTHFI_ISSUER is not set. Use the np project origin, https://<slug>.np.authfi.io")
    return require_np(value)


def require_np(url: str) -> str:
    host = urlparse(url).hostname or ""
    if not host.endswith(NP_SUFFIX) or host.count(".") != 3:
        raise SystemExit(f"refusing {url}: a test talks to https://<slug>{NP_SUFFIX} and nothing else")
    if RETIRED_TENANT in url or host.startswith("platform-beta"):
        raise SystemExit("refusing the retired tenant 4e46e0c4 / platform-beta")
    return url


def mcp_url(iss: str | None = None) -> str:
    return (iss or issuer()) + "/mcp"


_SECRET = re.compile(
    r"(eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}"  # a JWT
    r"|sk-[A-Za-z0-9_-]{12,}"  # a provider key
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    r"|(?i:bearer|dpop)\s+[A-Za-z0-9._~+/=-]{16,})"
)


def scrub(text: object) -> str:
    return _SECRET.sub("<redacted>", str(text))


def log(msg: str = "") -> None:
    print(scrub(msg), flush=True)


def save(name: str, payload: object) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(scrub(payload if isinstance(payload, str) else json.dumps(payload, indent=2, default=str)), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------ capture server
class Capture:
    """A local HTTP server that stores every request a tracer sends and answers 200 {}.

    LangSmith, the OpenAI Agents trace exporter, OTLP/HTTP (ADK, Vercel AI, Claude Code) all point here. Nothing
    is forwarded. ``find_secrets`` is the leak check: did a token, a key or a prompt reach a tracer?
    """

    def __init__(self, port: int = 0) -> None:
        self.requests: list[dict[str, object]] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def _take(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append(
                    {"method": self.command, "path": self.path, "headers": dict(self.headers), "body": body.decode("utf-8", "replace")}
                )
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.end_headers()
                self.wfile.write(b"{}")

            do_POST = do_PUT = do_PATCH = do_GET = _take

            def log_message(self, *args: object) -> None:  # quiet
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.port = self._server.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def find(self, needle: str) -> list[str]:
        """Paths of captured requests whose headers or body contain ``needle`` (a token's value, say)."""
        if not needle:
            return []
        return [str(r["path"]) for r in self.requests if needle in json.dumps(r["headers"]) or needle in str(r["body"])]

    def summary(self) -> dict[str, object]:
        paths: dict[str, int] = {}
        for r in self.requests:
            paths[str(r["path"]).split("?")[0]] = paths.get(str(r["path"]).split("?")[0], 0) + 1
        return {"requests": len(self.requests), "paths": paths}

    def stop(self) -> None:
        self._server.shutdown()


def tracer_env(cap: Capture) -> dict[str, str]:
    """Environment that points every tracer at the capture server and gives it a dummy key that is no one's."""
    return {
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_ENDPOINT": cap.url,
        "LANGSMITH_API_KEY": "dummy-capture-key-not-a-real-key",
        "LANGCHAIN_TRACING_V2": "true",
        "LANGCHAIN_ENDPOINT": cap.url,
        "LANGCHAIN_API_KEY": "dummy-capture-key-not-a-real-key",
        "OTEL_EXPORTER_OTLP_ENDPOINT": cap.url,
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": cap.url + "/v1/traces",
        "OPENAI_API_KEY": "dummy-capture-key-not-a-real-key",
        "OPENAI_BASE_URL": cap.url + "/openai",  # a stray OpenAI call lands here, not at api.openai.com
        "OPENAI_AGENTS_DISABLE_TRACING": "0",
    }


# ------------------------------------------------------------------------------------ what a test needs
NEEDS = {
    "project": "an np project (not platform.np, not 4e46e0c4): https://<slug>.np.authfi.io, e.g. probe-c1",
    "agent": "a registered agent agt_... : legacy path = its Ed25519 key file (AUTHFI_AGENT_KEY_FILE, mode 0600); "
    "bound path = a k8s binding, a signed configuration, and the pod (see report.md)",
    "grants": "the agent granted the tool(s) and the model (mcp:<upstream>/<tool>, mcp:anthropic/<model>)",
    "tool_upstream": "an MCP upstream on the project, e.g. deepwiki https://mcp.deepwiki.com/mcp",
    "model_provider": "an Anthropic provider upstream on the project (transport anthropic) holding the key, sealed by KMS",
}


def need_env(*names: str) -> dict[str, str]:
    """The named environment variables, or exit 3 naming what is missing (never their values)."""
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        print("BLOCKED: set " + ", ".join(missing) + " (see report.md, 'What a person must supply')")
        sys.exit(3)
    return {n: os.environ[n] for n in names}


def now() -> float:
    return time.time()
