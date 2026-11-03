"""A loopback proxy that gives a process which cannot hold the agent's key the key's proof on its model calls.

Claude Code makes its own model calls and can neither hold the agent's DPoP key nor sign a proof. Left to itself it
sends a Bearer token that a helper process minted, and the gateway counts such a call on the agent's one coarse run,
apart from the tool calls this process makes with the key's proof. Pointed at this proxy instead, Claude Code sends
its model calls here, and they leave with this process's credential: bound to the same key, proved per request, on
the same run as the tools.

The proxy forwards one request: ``POST /v1/messages`` under a secret this process made for the run. The secret is
in the environment of every tool Claude Code runs, so it does not stand for the right to use the credential: any
other method or path, including the rest of the gateway's model and cloud routes, is refused unsent. Whatever
credential the caller sends is dropped; the gateway sees only this process's.

When the application exports Claude Code's steps (``export_steps``), a second route relays Claude Code's trace export to
AuthFI (:meth:`ModelProxy.relay_traces`): the same listener and secret, one fixed target, with the caller's own headers
(the helper's token, which is what Claude Code sends AuthFI), and a rewrite of the body that names each span's agent.
The export is untrusted input: its body has a cap of its own, the rewrite runs on a worker thread, one at a time, and
an export that is over a cap, cannot be read or cannot be named is refused, never sent on unnamed.

The caller's ``traceparent`` is kept when it is one: Claude Code sends the one of the model request span it exports
(``CLAUDE_CODE_PROPAGATE_TRACEPARENT``), so the gateway's record of the call names that step. The proxy also reads the
tool calls the answer asks for, so that the call made for a tool can name the model request that asked for it.
"""

from __future__ import annotations

import asyncio
import hmac
import re
import secrets
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from authfi._protocol.ids import valid_traceparent
from authfi.gateway import _refused_response  # pyright: ignore[reportPrivateUsage]

if TYPE_CHECKING:
    from authfi.errors import GatewayTokenRefused
    from authfi.gateway import AsyncGateway

__all__ = ["RUN_HEADER", "ModelProxy"]

#: The header the gateway names the call's run in.
RUN_HEADER = "AuthFI-Run-Id"

#: The most a request head or body may be; a model request carries the whole conversation, images included.
_MAX_HEAD = 64 * 1024
_MAX_BODY = 64 * 1024 * 1024

#: The only requests forwarded, by method and path: the Anthropic API calls Claude Code makes. Nothing else of the
#: gateway is reachable through the proxy, whoever holds its secret (the secret is in the environment of every tool
#: Claude Code runs): not the cloud credential door, not the other model routes, not the tools.
ROUTES = frozenset({("POST", "/v1/messages")})
#: The route of Claude Code's trace export, served only after :meth:`ModelProxy.relay_traces`.
TRACE_ROUTE = ("POST", "/otlp/v1/traces")
#: The most of a trace export's body that is read (a model request carries a conversation; an export carries spans).
_MAX_TRACE_BODY = 4 * 1024 * 1024
#: The headers of a trace export that are sent on. The body is sent decoded, so its Content-Encoding is not one.
_TRACE_HEADERS = frozenset({"content-type", "authorization", "dpop", "user-agent", "traceparent"})
#: How often one kind of refusal is logged at warning level, in seconds; the rest are debug.
_LOG_EVERY = 60.0
_QUERY = re.compile(r"[A-Za-z0-9_=&.-]{0,128}")

#: How long a connection may take to send its head, and then its body. A caller without the secret never gets past
#: the head, so these and the caps below are what it can hold: at most _MAX_UNKNOWN sockets for _HEAD_SECONDS each, and
#: a new connection takes the place of the oldest of them, so holding them all does not keep Claude Code out.
#: Claude Code sends its head at once.
_HEAD_SECONDS = 2.0
_BODY_SECONDS = 120.0
_MAX_UNKNOWN = 16
_MAX_CONNECTIONS = 128
_CONTENT_LENGTH = re.compile(r"[0-9]{1,10}")
_CHUNK_SIZE = re.compile(r"[0-9A-Fa-f]{1,8}")
_MAX_TRAILERS = 32

#: Sent on by neither side of a proxy (RFC 9110 section 7.6.1), and the headers this proxy sets itself.
_NOT_FORWARDED = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
        "accept-encoding",
        "expect",
    }
)
#: What the caller may not choose: the credential, the proof, and the request id. Its User-Agent is kept: the
#: gateway records it as the client of the call, and the console should say Claude Code. Its ``traceparent`` is kept
#: only when it is one (see ``_forward``).
_CALLERS_OWN = frozenset({"authorization", "x-api-key", "dpop", "cookie", "x-request-id", "traceparent"})

#: A tool call in a model answer, whether streamed (``content_block_start``) or whole, in either key order.
_TOOL_USE = re.compile(
    rb'"type"\s*:\s*"tool_use"\s*,\s*"id"\s*:\s*"([A-Za-z0-9_-]{1,128})"'
    rb'|"id"\s*:\s*"([A-Za-z0-9_-]{1,128})"\s*,\s*"type"\s*:\s*"tool_use"'
)
#: How much of a chunk's end is kept to find a tool call that straddles two chunks.
_SCAN_TAIL = 160
#: How many tool calls it remembers the model request of.
_MAX_ASKED = 2048


class _ToolUses:
    """Finds the tool calls a model answer asks for, in chunks of any size."""

    def __init__(self) -> None:
        self._tail = b""

    def feed(self, chunk: bytes) -> set[str]:
        """The ids of the tool calls in ``chunk``, counting the end of the chunks before it."""
        window = self._tail + chunk
        self._tail = window[-_SCAN_TAIL:]
        return {(a or b).decode() for a, b in _TOOL_USE.findall(window)}


@dataclass(frozen=True, slots=True)
class TraceRelay:
    """Where Claude Code's trace export goes, and how its body is rewritten on the way.

    ``rewrite`` takes the body and its ``Content-Encoding`` and returns the new body (not compressed), or raises
    :class:`TraceRefused`. It is blocking: it is run on a worker thread.
    """

    url: str
    rewrite: Callable[[bytes, str], bytes]


class TraceRefused(Exception):
    """A trace export that is not sent on: ``status`` is what Claude Code is told, ``kind`` what the log says."""

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.kind = kind
        self.message = message


class _BadRequest(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class ModelProxy:
    """The model routes of one :class:`~authfi.gateway.AsyncGateway`, on the loopback interface.

    Args:
        gateway: the gateway whose HTTP client (the credential's proof on every request) carries the calls.
        on_refused: called with the gateway's refusal of the token, when a model call is answered 401.
        on_run: called with the run the gateway names on an answer.
    """

    def __init__(
        self,
        gateway: AsyncGateway,
        *,
        on_refused: Callable[[GatewayTokenRefused], None],
        on_run: Callable[[str], None],
    ) -> None:
        self._gateway = gateway
        self._on_refused = on_refused
        self._on_run = on_run
        self._secret = secrets.token_urlsafe(24)
        self._server: asyncio.Server | None = None
        self._base_url: str | None = None
        self._lock = asyncio.Lock()
        self._connections: set[asyncio.Task[None]] = set()
        #: The connections that have not yet shown the secret and a route, oldest first.
        self._preauth: dict[asyncio.Task[None], asyncio.StreamWriter] = {}
        #: Tool call id -> the ``traceparent`` of the model request whose answer asked for it, oldest first.
        self._asked_by: OrderedDict[str, str] = OrderedDict()
        self._traces: TraceRelay | None = None
        self._trace_client: httpx.AsyncClient | None = None
        #: One export is rewritten at a time, so a flood of them cannot use every core or all the memory.
        self._rewriting = asyncio.Semaphore(1)
        self._logged: dict[str, float] = {}

    def relay_traces(self, relay: TraceRelay) -> str:
        """Serve Claude Code's trace export at the returned URL, sent on to ``relay.url``. Needs :meth:`start` first."""
        if self._base_url is None:
            raise RuntimeError("relay_traces needs the proxy started")
        self._traces = relay
        return self._base_url + TRACE_ROUTE[1]

    def stop_relaying_traces(self) -> None:
        """Stop serving the trace export route, as before :meth:`relay_traces`."""
        self._traces = None

    def asked_by(self, tool_use_id: str | None) -> str | None:
        """The ``traceparent`` of the model request that asked for the tool call ``tool_use_id``, if it was seen."""
        return self._asked_by.get(tool_use_id) if tool_use_id else None

    def _note_asked(self, traceparent: str, found: set[str]) -> None:
        for tool_use_id in found:
            self._asked_by[tool_use_id] = traceparent
        while len(self._asked_by) > _MAX_ASKED:
            self._asked_by.popitem(last=False)

    async def start(self) -> str:
        """Listen, once; returns the base URL to give the model client (``ANTHROPIC_BASE_URL``)."""
        async with self._lock:
            if self._base_url is None:
                server = await asyncio.start_server(self._accept, "127.0.0.1", 0, limit=_MAX_HEAD)
                port = server.sockets[0].getsockname()[1]
                self._server = server
                self._base_url = f"http://127.0.0.1:{port}/{self._secret}"
                self._gateway.project.logger.debug("model calls proxied on 127.0.0.1:%d", port)
            return self._base_url

    async def aclose(self) -> None:
        """Stop listening and end the calls in flight."""
        server, self._server, self._base_url = self._server, None, None
        if server is not None:
            server.close()
        tasks = list(self._connections)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if server is not None:
            await server.wait_closed()
        client, self._trace_client, self._traces = self._trace_client, None, None
        if client is not None:
            await client.aclose()

    def __repr__(self) -> str:
        return f"ModelProxy({'listening' if self._base_url else 'stopped'})"

    # --- one connection --------------------------------------------------------------------------------

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if len(self._preauth) >= _MAX_UNKNOWN:
            # A process that holds sockets without a request must not keep the real caller out: its oldest goes.
            oldest = next(iter(self._preauth))
            self._preauth.pop(oldest).transport.abort()
        elif len(self._connections) >= _MAX_CONNECTIONS:
            writer.transport.abort()
            return
        task = asyncio.get_running_loop().create_task(self._connection(reader, writer))
        self._connections.add(task)
        self._preauth[task] = writer
        task.add_done_callback(self._forget)  # runs for a task cancelled before its first step, which has no finally

    @staticmethod
    def _self() -> asyncio.Task[Any]:
        task = asyncio.current_task()
        assert task is not None  # noqa: S101 - only called from a task
        return task

    def _forget(self, task: asyncio.Task[None]) -> None:
        self._connections.discard(task)
        self._preauth.pop(task, None)

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        logger = self._gateway.project.logger
        try:
            try:
                method, target, headers = await asyncio.wait_for(_read_head(reader), _HEAD_SECONDS)
                url = self._route(method, target)  # the secret and the route, before a byte of the body is read
                self._preauth.pop(self._self(), None)
                limit = _MAX_BODY if url is not None else _MAX_TRACE_BODY
                body = await asyncio.wait_for(_read_body(reader, writer, headers, limit), _BODY_SECONDS)
            except _BadRequest as bad:
                logger.info("a request to the model proxy was refused: %d %s", bad.status, bad.message)
                await _answer(writer, bad.status, bad.message)
                return
            if url is None:
                await self._forward_traces(writer, headers, body)
            else:
                await self._forward(writer, method, url, headers, body)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            logger.debug("a model call ended before it was answered")
        except TimeoutError:
            logger.info("a request to the model proxy was not sent in time and was closed")
            await _answer(writer, 408, "the request was not sent in time")
        except Exception:
            logger.exception("the model proxy failed on a call")
            await _answer(writer, 502, "the model call could not be forwarded")
        finally:
            writer.close()

    def _route(self, method: str, target: str) -> str | None:
        """The gateway URL a request names (``None`` for the trace export), or ``_BadRequest`` unless it is listed."""
        path, _, query = target.partition("?")
        given, _, rest = path[1:].partition("/")
        if not hmac.compare_digest(given.encode(), self._secret.encode()):
            raise _BadRequest(404, "not found")
        route = "/" + rest
        if (method, route) == TRACE_ROUTE and self._traces is not None and not query:
            return None
        if (method, route) not in ROUTES or not _QUERY.fullmatch(query):
            self._gateway.project.logger.info("the model proxy does not forward %s %r", method, route[:64])
            raise _BadRequest(404, "not found")
        return self._gateway.mcp_url + route + ("?" + query if query else "")

    async def _forward(
        self, writer: asyncio.StreamWriter, method: str, url: str, headers: list[tuple[str, str]], body: bytes
    ) -> None:
        sent = [
            (name, value)
            for name, value in headers
            if name.lower() not in _NOT_FORWARDED and name.lower() not in _CALLERS_OWN
        ]
        step = next((v for n, v in headers if n.lower() == "traceparent" and valid_traceparent(v)), None)
        if step is not None:
            sent.append(("traceparent", step.strip().lower()))
        client: Any = self._gateway.http_client(httpx)
        request = client.build_request(method, url, headers=sent, content=body)
        request.headers["Accept-Encoding"] = "identity"
        try:
            response: httpx.Response = await client.send(request, stream=True)
        except httpx.HTTPError as error:
            self._gateway.project.logger.warning("a model call could not reach the gateway: %s", type(error).__name__)
            await _answer(writer, 502, "the AuthFI gateway could not be reached")
            return
        try:
            run = response.headers.get(RUN_HEADER)
            if run:
                self._on_run(run)
            refused = _refused_response(response)
            if refused is not None:
                self._on_refused(refused)
            await _relay(
                writer, method, response, None if step is None else lambda found: self._note_asked(step, found)
            )
        finally:
            await response.aclose()

    def _said(self, kind: str, message: str) -> None:
        """Log a refusal at warning level once a minute for each kind, and at debug level between."""
        now = time.monotonic()
        loud = now - self._logged.get(kind, -_LOG_EVERY) >= _LOG_EVERY
        if loud:
            self._logged[kind] = now
        self._gateway.project.logger.log(
            30 if loud else 10, "a Claude Code trace export was refused (%s): %s", kind, message
        )

    async def _forward_traces(self, writer: asyncio.StreamWriter, headers: list[tuple[str, str]], body: bytes) -> None:
        """Send Claude Code's trace export to AuthFI with its own credential, the body naming each span's agent.

        An export that is over a cap, cannot be read or cannot be named is answered 4xx or 5xx and not sent: unnamed,
        a sub-agent's spans would be read as its parent's. Claude Code sees the failure and can try again.
        """
        relay = self._traces
        if relay is None:  # not served, or stopped while the body was read
            await _answer(writer, 404, "not found")
            return
        encoding = next((v for n, v in headers if n.lower() == "content-encoding"), "")
        try:
            async with self._rewriting:
                rewritten = await asyncio.to_thread(relay.rewrite, body, encoding)
        except TraceRefused as refused:
            self._said(refused.kind, refused.message)
            await _answer(writer, refused.status, refused.message)
            return
        except Exception as error:  # a bug here is the proxy's, not the export's: a 5xx, which is retried
            self._said("internal", type(error).__name__)
            await _answer(writer, 500, "the export could not be processed")
            return
        sent = [(name, value) for name, value in headers if name.lower() in _TRACE_HEADERS]
        if self._trace_client is None:
            self._trace_client = httpx.AsyncClient(
                follow_redirects=False, timeout=httpx.Timeout(15.0), limits=httpx.Limits(max_connections=8)
            )
        request = self._trace_client.build_request("POST", relay.url, headers=sent, content=rewritten)
        request.headers["Accept-Encoding"] = "identity"
        try:
            response = await self._trace_client.send(request, stream=True)
        except httpx.HTTPError as error:
            self._gateway.project.logger.warning(
                "a Claude Code trace export could not reach AuthFI: %s", type(error).__name__
            )
            await _answer(writer, 502, "AuthFI could not be reached")
            return
        try:
            await _relay(writer, "POST", response)
        finally:
            await response.aclose()


async def _read_head(reader: asyncio.StreamReader) -> tuple[str, str, list[tuple[str, str]]]:
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except asyncio.LimitOverrunError as error:
        raise _BadRequest(431, "the request head is too large") from error
    lines = head.decode("latin-1").split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) != 3 or not parts[1].startswith("/") or not parts[2].startswith("HTTP/1."):
        raise _BadRequest(400, "not an HTTP/1.1 request")
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        name, colon, value = line.partition(":")
        if not colon or not name or name != name.strip():
            raise _BadRequest(400, "a malformed header")
        headers.append((name, value.strip()))
    return parts[0], parts[1], headers


async def _read_body(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: list[tuple[str, str]], limit: int = _MAX_BODY
) -> bytes:
    names = [name.lower() for name, _ in headers]
    fields = {name.lower(): value for name, value in headers}
    if any(names.count(n) > 1 for n in ("content-length", "transfer-encoding")):
        raise _BadRequest(400, "a repeated length header")
    if "transfer-encoding" in fields and "content-length" in fields:
        raise _BadRequest(400, "both a Content-Length and a Transfer-Encoding")
    if fields.get("expect", "").lower() == "100-continue":
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
    if "chunked" in fields.get("transfer-encoding", "").lower():
        return await _read_chunked(reader, limit)
    declared = fields.get("content-length", "0")
    if not _CONTENT_LENGTH.fullmatch(declared):
        raise _BadRequest(400, "a malformed Content-Length")
    length = int(declared)
    if length > limit:
        raise _BadRequest(413, "the request is too large")
    return await reader.readexactly(length) if length else b""


async def _line(reader: asyncio.StreamReader) -> bytes:
    try:
        return await reader.readline()
    except ValueError as error:  # a line longer than the reader's limit
        raise _BadRequest(400, "a malformed chunk") from error


async def _read_chunked(reader: asyncio.StreamReader, limit: int = _MAX_BODY) -> bytes:
    body = bytearray()
    while True:
        size_text = (await _line(reader)).split(b";")[0].strip().decode("latin-1")
        if not _CHUNK_SIZE.fullmatch(size_text):
            raise _BadRequest(400, "a malformed chunk")
        size = int(size_text, 16)
        if size == 0:
            for _ in range(_MAX_TRAILERS):  # trailers
                if not (await _line(reader)).strip():
                    return bytes(body)
            raise _BadRequest(400, "too many trailers")
        if len(body) + size > limit:
            raise _BadRequest(413, "the request is too large")
        body += await reader.readexactly(size)
        if await reader.readexactly(2) != b"\r\n":
            raise _BadRequest(400, "a malformed chunk")


async def _relay(
    writer: asyncio.StreamWriter,
    method: str,
    response: httpx.Response,
    on_tool_uses: Callable[[set[str]], None] | None = None,
) -> None:
    """Send the gateway's answer on as it arrives, so a streamed answer (server-sent events) is not held back.

    ``on_tool_uses`` is told the ids of the tool calls the answer asks for, as they go by.
    """
    bodiless = method == "HEAD" or response.status_code in (204, 304) or 100 <= response.status_code < 200
    head = [f"HTTP/1.1 {response.status_code} {response.reason_phrase}"]
    for name, value in response.headers.multi_items():
        if name.lower() not in _NOT_FORWARDED:
            head.append(f"{name}: {value}")
    head.append("Connection: close")
    if not bodiless:
        head.append("Transfer-Encoding: chunked")
    writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
    if not bodiless:
        asked = _ToolUses()
        try:
            async for chunk in response.aiter_raw():
                if chunk:
                    found = asked.feed(chunk) if on_tool_uses is not None else None
                    if found and on_tool_uses is not None:
                        on_tool_uses(found)
                    writer.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    await writer.drain()
        except httpx.HTTPError:
            writer.transport.abort()  # the answer was cut short: the caller must not read it as complete
            return
        writer.write(b"0\r\n\r\n")
    await writer.drain()


async def _answer(writer: asyncio.StreamWriter, status: int, message: str) -> None:
    """A refusal this proxy makes itself, in the shape the Anthropic API answers errors in."""
    reason = {
        400: "Bad Request",
        404: "Not Found",
        408: "Request Timeout",
        413: "Payload Too Large",
        415: "Unsupported Media Type",
        500: "Internal Server Error",
        431: "Header Fields Too Large",
    }.get(status, "Bad Gateway")
    body = ('{"type":"error","error":{"type":"api_error","message":"' + message + '"}}').encode()
    head = f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
    try:
        writer.write(head.encode() + b"Connection: close\r\n\r\n" + body)
        await writer.drain()
    except ConnectionError:
        return
