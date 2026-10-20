"""A local stand-in for the AuthFI gateway, for the adapter tests.

It speaks the wire the real gateway speaks at ``<project>/mcp``: MCP over
streamable HTTP (JSON responses) and the Anthropic Messages API at
``/mcp/v1/messages``. It answers each tool call the way the gateway does, in
either of its two published forms:

- ``refusals="http"``: a refused tool call is HTTP 403 with an
  ``insufficient_scope`` challenge and the JSON-RPC answer in the body, and a
  hold is the "Approval required ... (approval <id>)" text (gateways before
  authfi-regional-mcp#65);
- ``refusals="tool_error"``: a refusal is HTTP 200 with an ``isError`` result
  and ``_meta["io.authfi/refusal"]``, and a hold is "Waiting for approval <id>
  (approval_required)" (after #65).

``names="dotted"`` publishes ``<server>.<tool>``; ``names="wire"`` publishes
``<server>__<tool>`` and accepts both (after authfi-regional-mcp#64).

A token is accepted when it starts with ``accept_prefix``; anything else gets
401. The model endpoint plays back ``model_script``: each entry is a list of
content blocks (``{"type": "text", ...}`` or ``{"type": "tool_use", ...}``).
Everything received is kept in ``requests`` for the tests to inspect.
"""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ALLOWED = "deepwiki.read_wiki_structure"
UNGRANTED = "deepwiki.ask_wiki_question"
STEP_UP = "deepwiki.read_wiki_contents"
NEVER = "deepwiki.delete_wiki"

TOOLS = {
    ALLOWED: "no_policy_restriction",
    UNGRANTED: None,
    STEP_UP: "may_require_approval",
    NEVER: "never_allowed",
}


class FakeGateway:
    def __init__(self, refusals="http", names="dotted", accept_prefix="tok-"):
        self.refusals = refusals
        self.names = names
        self.accept_prefix = accept_prefix
        self.requests = []
        self.approvals = {}       # id -> {"status", "tool", "args"}
        self.tool_calls = []      # (gateway name, arguments) that reached the tool
        self.model_script = []
        self._lock = threading.Lock()
        gateway = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                gateway._record(self, None)
                self._send(405, b"")

            def do_DELETE(self):
                gateway._record(self, None)
                self._send(200, b"")

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                gateway._record(self, body)
                status, headers, out = gateway._answer(self.path, dict(self.headers), body)
                self._send(status, out, headers)

            def _send(self, status, body, headers=None):
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                if body and not any(k.lower() == "content-type" for k in (headers or {})):
                    self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.origin = "http://127.0.0.1:%d" % self._server.server_address[1]
        self.url = self.origin + "/mcp"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    # --- what the tests drive

    def decide(self, approval_id, status):
        self.approvals[approval_id]["status"] = status

    def pending(self):
        return [a for a, v in self.approvals.items() if v["status"] == "pending"]

    def tool_requests(self):
        """Every tools/call that reached the gateway: (headers, params)."""
        out = []
        for r in self.requests:
            body = r["json"] or {}
            if r["path"].startswith("/mcp") and body.get("method") == "tools/call":
                out.append((r["headers"], body.get("params") or {}))
        return out

    def model_requests(self):
        return [r for r in self.requests if r["path"].endswith("/v1/messages")]

    # --- internals

    def _record(self, handler, body):
        try:
            parsed = json.loads(body) if body else None
        except ValueError:
            parsed = None
        with self._lock:
            self.requests.append({"method": handler.command, "path": handler.path,
                                  "headers": {k.lower(): v for k, v in handler.headers.items()},
                                  "body": body, "json": parsed})

    def _authorized(self, headers):
        auth = {k.lower(): v for k, v in headers.items()}.get("authorization") or ""
        return auth.startswith("Bearer " + self.accept_prefix)

    def _answer(self, path, headers, body):
        if not self._authorized(headers):
            return 401, {"WWW-Authenticate": 'Bearer error="invalid_token", error_description="token revoked"'}, \
                b'{"error":"invalid_token"}'
        if path.endswith("/v1/messages"):
            return self._model(json.loads(body))
        msg = json.loads(body)
        method, rid = msg.get("method"), msg.get("id")
        if rid is None:
            return 202, {}, b""
        if method == "initialize":
            return 200, {"Mcp-Session-Id": "s-" + uuid.uuid4().hex[:8]}, _rpc(rid, {
                "protocolVersion": (msg.get("params") or {}).get("protocolVersion") or "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "authfi-mcp-gateway", "version": "1.0"}})
        if method == "tools/list":
            tools = []
            for name, outlook in TOOLS.items():
                tool = {"name": self._wire(name), "description": "The %s tool" % name,
                        "inputSchema": {"type": "object", "properties": {"repoName": {"type": "string"}},
                                        "required": ["repoName"]}}
                if outlook:
                    tool["annotations"] = {"io.authfi/governance": outlook}
                tools.append(tool)
            return 200, {}, _rpc(rid, {"tools": tools})
        if method == "tools/call":
            return self._call(rid, msg.get("params") or {})
        if method == "ping":
            return 200, {}, _rpc(rid, {})
        return 200, {}, json.dumps({"jsonrpc": "2.0", "id": rid,
                                    "error": {"code": -32601, "message": "method not found"}}).encode()

    def _wire(self, name):
        return name.replace(".", "__") if self.names == "wire" else name

    def _call(self, rid, params):
        name = params.get("name") or ""
        if self.names == "wire" and "__" in name and "." not in name:
            name = name.replace("__", ".", 1)
        args = dict(params.get("arguments") or {})
        approval = args.pop("_authfi_approval_id", None)
        if name not in TOOLS:
            return 200, {}, _rpc(rid, _text("unknown tool %s" % name, True))
        if name in (UNGRANTED, NEVER):
            return self._refuse(rid, name)
        if name == STEP_UP:
            if approval is None:
                aid = str(uuid.uuid4())
                with self._lock:
                    self.approvals[aid] = {"status": "pending", "tool": name, "args": args}
                return self._hold(rid, aid, "approval_required")
            record = self.approvals.get(approval)
            if record is None or record["status"] == "expired":
                return self._refuse(rid, name, code="approval_expired", approval_id=approval)
            if record["status"] == "pending":
                return self._hold(rid, approval, "approval_pending")
            if record["status"] == "denied":
                return self._refuse(rid, name, code="approval_denied", approval_id=approval)
            if record["status"] == "used":
                return self._refuse(rid, name, code="approval_used", approval_id=approval)
            record["status"] = "used"
        with self._lock:
            self.tool_calls.append((name, args))
        return 200, {}, _rpc(rid, _text("%s answered for %s" % (name, args.get("repoName")), False))

    def _hold(self, rid, aid, code):
        if self.refusals == "http":
            text = ("Approval required. A human must approve this action in the AuthFI console (approval %s). "
                    "Retry this call with an argument \"_authfi_approval_id\": \"%s\" once it is approved." % (aid, aid))
        else:
            text = ("Waiting for approval %s (%s). A person must approve this action in the AuthFI console. "
                    "Retry this call with an argument \"_authfi_approval_id\": \"%s\" once it is approved."
                    % (aid, code, aid))
        return 200, {}, _rpc(rid, _text(text, True))

    def _refuse(self, rid, name, code=None, approval_id=None):
        scope = "mcp:" + name
        if code:
            reason = code.replace("_", " ")
            if self.refusals == "http":
                return 200, {}, _rpc(rid, _text("Denied by policy: %s" % reason, True))
            result = _text("Refused (%s): %s (approval %s)." % (code, reason, approval_id), True)
            result["_meta"] = {"io.authfi/refusal": {"code": code, "reason": reason, "approval_id": approval_id}}
            return 200, {}, _rpc(rid, result)
        if self.refusals == "http":
            challenge = 'Bearer error="insufficient_scope", scope="%s", resource_metadata="%s"' % (
                scope, self.origin + "/.well-known/oauth-protected-resource")
            return 403, {"WWW-Authenticate": challenge}, _rpc(rid, _text("Denied by policy: no_match", True))
        reason = "no grant lets this agent call %s" % name
        result = _text("Refused (insufficient_scope, no_match): %s. The call needs the scope %s." % (reason, scope), True)
        result["_meta"] = {"io.authfi/refusal": {"error": "insufficient_scope", "code": "no_match",
                                                 "reason": reason, "required_scope": scope}}
        return 200, {}, _rpc(rid, result)

    def _model(self, req):
        with self._lock:
            blocks = self.model_script.pop(0) if self.model_script else [{"type": "text", "text": "Done."}]
        content, stop = [], "end_turn"
        for b in blocks:
            if b["type"] == "tool_use":
                content.append({"type": "tool_use", "id": b.get("id") or "toolu_" + uuid.uuid4().hex[:20],
                                "name": b["name"], "input": b.get("input") or {}})
                stop = "tool_use"
            else:
                content.append(b)
        msg = {"id": "msg_" + uuid.uuid4().hex[:12], "type": "message", "role": "assistant",
               "model": req.get("model"), "content": content, "stop_reason": stop, "stop_sequence": None,
               "usage": {"input_tokens": 3, "output_tokens": 2}}
        if req.get("stream"):
            return 200, {"Content-Type": "text/event-stream"}, _sse(msg)
        return 200, {}, json.dumps(msg).encode()


def _rpc(rid, result):
    return json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode()


def _text(text, is_error):
    return {"content": [{"type": "text", "text": text}], "isError": is_error}


def _sse(msg):
    events = []
    start = dict(msg, content=[], stop_reason=None)
    events.append(("message_start", {"type": "message_start", "message": start}))
    for i, block in enumerate(msg["content"]):
        if block["type"] == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                   "content_block": {"type": "text", "text": ""}}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "text_delta", "text": block["text"]}}))
        else:
            events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                   "content_block": dict(block, input={})}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                   "delta": {"type": "input_json_delta",
                                                             "partial_json": json.dumps(block["input"])}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {"type": "message_delta",
                                     "delta": {"stop_reason": msg["stop_reason"], "stop_sequence": None},
                                     "usage": {"output_tokens": 2}}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join("event: %s\ndata: %s\n\n" % (e, json.dumps(d)) for e, d in events).encode()


class CaptureServer:
    """A local stand-in for a tracing backend: accepts every request and keeps its bytes."""

    def __init__(self):
        self.bodies = []
        cap = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _read(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                if self.headers.get("Content-Encoding") == "zstd":
                    try:
                        import zstandard
                        body = zstandard.ZstdDecompressor().decompressobj().decompress(body)
                    except ImportError:
                        pass
                cap.bodies.append((self.path, dict(self.headers), body))

            def do_GET(self):
                self._read()
                body = json.dumps({"version": "0.0.0-capture", "batch_ingest_config": {
                    "size_limit": 100, "size_limit_bytes": 20971520, "scale_up_nthreads_limit": 1,
                    "scale_up_qsize_trigger": 1000, "scale_down_nempty_trigger": 4},
                    "instance_flags": {"zstd_compression_enabled": False}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self._read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            do_PATCH = do_POST
            do_PUT = do_POST

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:%d" % self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()

    def blob(self):
        return b"\n".join(b for _, _, b in self.bodies) + b"\n" + b"\n".join(
            json.dumps(h).encode() for _, h, _ in self.bodies)
