"""A local stand-in for an AuthFI project and its gateways, over real HTTP, for the adapter tests.

Frameworks build their own HTTP clients, so this is a server on 127.0.0.1, not a mock transport.
It speaks the wire the platform speaks today:

- discovery, the agent configuration, and a token endpoint that mints ``tok-minted-<n>`` for a
  client assertion (the signature is the platform's business, tested elsewhere);
- ``/mcp``: MCP over streamable HTTP with JSON answers, tool names ``<server>__<tool>``, a refusal
  as an ``isError`` result with ``_meta["io.authfi/refusal"]``, a hold as the gateway's exact
  "Waiting for approval ..." text;
- ``/mcp/v1/messages``: the Anthropic Messages API, playing back ``model_script``;
- the approval address, which answers 403 unless ``approvals_readable``.

A bearer starting with ``tok-`` is accepted; anything else gets 401. Every request is kept.
"""

from __future__ import annotations

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, cast
from urllib.parse import parse_qs

ALLOWED = "deepwiki__read_wiki_structure"
UNGRANTED = "deepwiki__ask_question"
STEP_UP = "deepwiki__read_wiki_contents"
NEVER = "deepwiki__delete_wiki"

TOOLS: dict[str, str | None] = {
    ALLOWED: "no_policy_restriction",
    UNGRANTED: None,
    STEP_UP: "may_require_approval",
    NEVER: "never_allowed",
}


def _scope(name: str) -> str:
    server, _, tool = name.partition("__")
    return f"mcp:{server}.{tool}"


class FakeGateway:
    """Use as a context manager; ``issuer`` is its address."""

    def __init__(self, *, approvals_readable: bool = False, token_lifetime: int = 900) -> None:
        self.approvals_readable = approvals_readable
        self.token_lifetime = token_lifetime
        self.minted: list[str] = []
        self.requests: list[dict[str, Any]] = []
        self.approvals: dict[str, dict[str, Any]] = {}
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.model_script: list[list[dict[str, Any]]] = []
        self.revoked = False
        self._lock = threading.Lock()
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def do_GET(self) -> None:
                gateway._record(self, b"")
                status, out = gateway._get(self.path.split("?")[0], {k.lower(): v for k, v in self.headers.items()})
                self._send(status, out)

            def do_DELETE(self) -> None:
                gateway._record(self, b"")
                self._send(200, b"")

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                gateway._record(self, body)
                status, headers, out = gateway._post(
                    self.path.split("?")[0], {k.lower(): v for k, v in self.headers.items()}, body
                )
                self._send(status, out, headers)

            def _send(self, status: int, body: bytes, headers: dict[str, str] | None = None) -> None:
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                if body and not any(k.lower() == "content-type" for k in (headers or {})):
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.issuer = f"http://127.0.0.1:{self._server.server_address[1]}"
        self.mcp_url = self.issuer + "/mcp"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeGateway:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    # --- what tests drive and read -----------------------------------------------------------------

    def decide(self, approval_id: str, status: str) -> None:
        self.approvals[approval_id]["status"] = status

    def pending(self) -> list[str]:
        return [a for a, v in self.approvals.items() if v["status"] == "pending"]

    def tool_requests(self) -> list[tuple[dict[str, str], dict[str, Any]]]:
        out: list[tuple[dict[str, str], dict[str, Any]]] = []
        for r in self.requests:
            body = cast(dict[str, Any], r["json"]) if isinstance(r["json"], dict) else {}
            if r["path"] == "/mcp" and body.get("method") == "tools/call":
                out.append((r["headers"], cast(dict[str, Any], body.get("params") or {})))
        return out

    def model_requests(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["path"].endswith("/v1/messages")]

    # --- internals -----------------------------------------------------------------------------------

    def _record(self, handler: BaseHTTPRequestHandler, body: bytes) -> None:
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            parsed = None
        with self._lock:
            self.requests.append(
                {
                    "method": handler.command,
                    "path": handler.path.split("?")[0],
                    "headers": {k.lower(): v for k, v in handler.headers.items()},
                    "body": body,
                    "json": parsed,
                }
            )

    def _get(self, path: str, headers: dict[str, str]) -> tuple[int, bytes]:
        o = self.issuer
        if path == "/.well-known/openid-configuration":
            return 200, json.dumps(
                {
                    "issuer": o,
                    "token_endpoint": o + "/oauth/token",
                    "jwks_uri": o + "/.well-known/jwks.json",
                    "backchannel_authentication_endpoint": o + "/apis/regional/v1/ai/auth/backchannel-authorize",
                    "grant_types_supported": ["client_credentials", "urn:openid:params:grant-type:ciba"],
                    "token_endpoint_auth_methods_supported": ["private_key_jwt"],
                    "token_endpoint_auth_signing_alg_values_supported": ["EdDSA", "ES256", "PS256"],
                }
            ).encode()
        if path == "/.well-known/agent-configuration":
            return 200, json.dumps(
                {
                    "issuer": o,
                    "endpoints": {
                        "approval": o + "/apis/regional/v1/ai/auth/approvals/{id}",
                        "request_approval": o + "/apis/regional/v1/ai/auth/request-approval",
                    },
                }
            ).encode()
        prefix = "/apis/regional/v1/ai/auth/approvals/"
        if path.startswith(prefix):
            if not self._authorized(headers):
                return 401, b'{"error":"invalid_token"}'
            if not self.approvals_readable:
                return 403, b'{"error":"forbidden"}'
            record = self.approvals.get(path[len(prefix) :])
            if record is None:
                return 404, b'{"error":"not_found"}'
            return 200, json.dumps({"data": {"approval_id": path[len(prefix) :], "status": record["status"]}}).encode()
        return 405, b""

    def _authorized(self, headers: dict[str, str]) -> bool:
        return not self.revoked and headers.get("authorization", "").startswith("Bearer tok-")

    def _post(self, path: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, str], bytes]:
        if path == "/oauth/token":
            form = {k: v[0] for k, v in parse_qs(body.decode()).items()}
            if form.get("grant_type") != "client_credentials" or not form.get("client_assertion"):
                return 400, {}, b'{"error":"invalid_request"}'
            with self._lock:
                token = f"tok-minted-{len(self.minted) + 1}"
                self.minted.append(token)
            return (
                200,
                {},
                json.dumps({"access_token": token, "token_type": "Bearer", "expires_in": self.token_lifetime}).encode(),
            )
        if not self._authorized(headers):
            return (
                401,
                {"WWW-Authenticate": 'Bearer error="invalid_token", error_description="revoked"'},
                (b'{"error":"invalid_token"}'),
            )
        if path.endswith("/v1/messages"):
            return self._model(json.loads(body))
        if path.endswith("/v1/chat/completions"):
            return self._chat(json.loads(body))
        message = cast(dict[str, Any], json.loads(body))
        method, rid = message.get("method"), message.get("id")
        if rid is None:
            return 202, {}, b""
        if method == "initialize":
            return (
                200,
                {},
                _rpc(
                    rid,
                    {
                        "protocolVersion": cast(dict[str, Any], message.get("params") or {}).get("protocolVersion")
                        or "2025-06-18",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "authfi-mcp-gateway", "version": "1.0"},
                    },
                ),
            )
        if method == "tools/list":
            tools: list[dict[str, Any]] = []
            for name, outlook in TOOLS.items():
                tool: dict[str, Any] = {
                    "name": name,
                    "description": f"The {name} tool",
                    "inputSchema": {
                        "type": "object",
                        "properties": {"repoName": {"type": "string"}},
                        "required": ["repoName"],
                    },
                    "_meta": {"io.authfi/scope": _scope(name)},
                }
                if outlook:
                    tool["annotations"] = {"io.authfi/governance": outlook}
                tools.append(tool)
            return 200, {}, _rpc(rid, {"tools": tools})
        if method == "tools/call":
            return self._call(rid, message.get("params") or {})
        if method == "ping":
            return 200, {}, _rpc(rid, {})
        return 200, {}, json.dumps({"jsonrpc": "2.0", "id": rid, "error": {"code": -32601, "message": "no"}}).encode()

    def _call(self, rid: Any, params: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        name = params.get("name") or ""
        args = dict(params.get("arguments") or {})
        approval = args.pop("_authfi_approval_id", None)
        if name not in TOOLS:
            return 200, {}, _rpc(rid, _text(f"unknown tool {name}", is_error=True))
        if name in (UNGRANTED, NEVER):
            return self._refuse(rid, name)
        if name == STEP_UP:
            if approval is None:
                approval_id = str(uuid.uuid4())
                with self._lock:
                    self.approvals[approval_id] = {"status": "pending", "tool": name, "args": args}
                return self._hold(rid, approval_id, "approval_required")
            record = self.approvals.get(approval)
            if record is None or record["status"] in ("expired", "used", "denied"):
                code = "approval_" + (record["status"] if record else "expired")
                return self._refuse(rid, name, code=code, approval_id=approval)
            if record["status"] == "pending":
                return self._hold(rid, approval, "approval_pending")
            record["status"] = "used"
        with self._lock:
            self.tool_calls.append((name, args))
        return 200, {}, _rpc(rid, _text(f"{name} answered for {args.get('repoName')}", is_error=False))

    def _hold(self, rid: Any, approval_id: str, code: str) -> tuple[int, dict[str, str], bytes]:
        text = (
            f"Waiting for approval {approval_id} ({code}). A person must approve this action in the AuthFI console. "
            f'Retry this call with an argument "_authfi_approval_id": "{approval_id}" once it is approved.'
        )
        return 200, {}, _rpc(rid, _text(text, is_error=True))

    def _refuse(
        self, rid: Any, name: str, *, code: str | None = None, approval_id: str | None = None
    ) -> tuple[int, dict[str, str], bytes]:
        if code:
            reason = code.replace("_", " ")
            result = _text(f"Refused ({code}): {reason} (approval {approval_id}).", is_error=True)
            result["_meta"] = {"io.authfi/refusal": {"code": code, "reason": reason, "approval_id": approval_id}}
            return 200, {}, _rpc(rid, result)
        scope = _scope(name)
        reason = f"no grant lets this agent call {name}"
        result = _text(
            f"Refused (insufficient_scope, ETOOL003): {reason}. The call needs the scope {scope}.", is_error=True
        )
        result["_meta"] = {
            "io.authfi/refusal": {
                "error": "insufficient_scope",
                "code": "ETOOL003",
                "reason": reason,
                "required_scope": scope,
            }
        }
        return 200, {}, _rpc(rid, result)

    def _chat(self, request: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        with self._lock:
            blocks = self.model_script.pop(0) if self.model_script else [{"type": "text", "text": "Done."}]
        text = "".join(str(b.get("text", "")) for b in blocks)
        return 200, {}, _chat_answer(request.get("model"), text)

    def _model(self, request: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        with self._lock:
            blocks = self.model_script.pop(0) if self.model_script else [{"type": "text", "text": "Done."}]
        content: list[dict[str, Any]] = []
        stop = "end_turn"
        for block in blocks:
            if block["type"] == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.get("id") or "toolu_" + uuid.uuid4().hex[:20],
                        "name": block["name"],
                        "input": block.get("input") or {},
                    }
                )
                stop = "tool_use"
            else:
                content.append(block)
        message = {
            "id": "msg_" + uuid.uuid4().hex[:12],
            "type": "message",
            "role": "assistant",
            "model": request.get("model"),
            "content": content,
            "stop_reason": stop,
            "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
        if request.get("stream"):
            return 200, {"Content-Type": "text/event-stream"}, _sse(message)
        return 200, {}, json.dumps(message).encode()


def _chat_answer(model: object, text: str) -> bytes:
    return json.dumps(
        {
            "id": "chatcmpl-" + uuid.uuid4().hex[:12],
            "object": "chat.completion",
            "created": 0,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        }
    ).encode()


def _rpc(rid: Any, result: dict[str, Any]) -> bytes:
    return json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode()


def _text(text: str, *, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _sse(message: dict[str, Any]) -> bytes:
    events: list[tuple[str, dict[str, Any]]] = [
        ("message_start", {"type": "message_start", "message": dict(message, content=[], stop_reason=None)})
    ]
    for i, block in enumerate(message["content"]):
        if block["type"] == "text":
            events.append(
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}},
                )
            )
        else:
            events.append(
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": i, "content_block": dict(block, input={})},
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])},
                    },
                )
            )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": message["stop_reason"], "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {e}\ndata: {json.dumps(d)}\n\n" for e, d in events).encode()


class CaptureServer:
    """A local stand-in for a tracing backend: accepts every request and keeps its bytes."""

    def __init__(self) -> None:
        self.bodies: list[tuple[str, dict[str, str], bytes]] = []
        capture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _read(self) -> None:
                if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                    body = b""
                    while True:
                        size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                        if size == 0:
                            self.rfile.readline()
                            break
                        body += self.rfile.read(size)
                        self.rfile.readline()
                else:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = self.rfile.read(length) if length else b""
                if self.headers.get("Content-Encoding") == "zstd":
                    try:
                        import zstandard

                        body = zstandard.ZstdDecompressor().decompressobj().decompress(body)
                    except ImportError:
                        pass
                capture.bodies.append((self.path, dict(self.headers), body))

            def do_GET(self) -> None:
                self._read()
                body = json.dumps(
                    {
                        "version": "0.0.0-capture",
                        "batch_ingest_config": {
                            "size_limit": 100,
                            "size_limit_bytes": 20971520,
                            "scale_up_nthreads_limit": 1,
                            "scale_up_qsize_trigger": 1000,
                            "scale_down_nempty_trigger": 4,
                        },
                        "instance_flags": {"zstd_compression_enabled": False},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:
                self._read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            do_PATCH = do_POST
            do_PUT = do_POST

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> CaptureServer:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()

    def blob(self) -> bytes:
        bodies = b"\n".join(b for _, _, b in self.bodies)
        headers = b"\n".join(json.dumps(h).encode() for _, h, _ in self.bodies)
        return bodies + b"\n" + headers
