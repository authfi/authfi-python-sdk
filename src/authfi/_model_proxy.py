"""A loopback proxy that gives a process which cannot hold the agent's key the key's proof on its model calls.

Claude Code makes its own model calls and can neither hold the agent's DPoP key nor sign a proof. Left to itself it
sends a Bearer token that a helper process minted, and the gateway counts such a call on the agent's one coarse run,
apart from the tool calls this process makes with the key's proof. Pointed at this proxy instead, Claude Code sends
its model calls here, and they leave with this process's credential: bound to the same key, proved per request, on
the same run as the tools.

The proxy accepts one thing: a request whose path starts with a secret this process made for the run, naming a model
route (``/v1/...``). Whatever credential the caller sends is dropped; the gateway sees only this process's.
"""

from __future__ import annotations

import asyncio
import hmac
import re
import secrets
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import httpx

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

_MODEL_ROUTE = re.compile(r"/v1/[A-Za-z0-9._~/-]*")

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
#: What the caller may not choose: the credential, the proof, and the labels the gateway's client sets.
_CALLERS_OWN = frozenset({"authorization", "x-api-key", "dpop", "cookie", "x-request-id", "user-agent", "traceparent"})


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
        for task in list(self._connections):
            task.cancel()
        if self._connections:
            await asyncio.gather(*self._connections, return_exceptions=True)
        if server is not None:
            await server.wait_closed()

    def __repr__(self) -> str:
        return f"ModelProxy({'listening' if self._base_url else 'stopped'})"

    # --- one connection --------------------------------------------------------------------------------

    def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.get_running_loop().create_task(self._connection(reader, writer))
        self._connections.add(task)
        task.add_done_callback(self._connections.discard)

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        logger = self._gateway.project.logger
        try:
            try:
                method, target, headers = await _read_head(reader)
                route = self._route(target)
                body = await _read_body(reader, writer, headers)
            except _BadRequest as bad:
                await _answer(writer, bad.status, bad.message)
                return
            await self._forward(writer, method, route, headers, body)
        except (ConnectionError, asyncio.IncompleteReadError, asyncio.LimitOverrunError, TimeoutError):
            logger.debug("a model call ended before it was answered")
        except Exception:
            logger.exception("the model proxy failed on a call")
            await _answer(writer, 502, "the model call could not be forwarded")
        finally:
            writer.close()

    def _route(self, target: str) -> str:
        """The gateway URL a request names, or ``_BadRequest`` for anything but a model route under the secret."""
        path, _, query = target.partition("?")
        given, _, rest = path[1:].partition("/")
        route = "/" + rest
        if not hmac.compare_digest(given.encode(), self._secret.encode()):
            raise _BadRequest(404, "not found")
        if not _MODEL_ROUTE.fullmatch(route) or any(part in ("", ".", "..") for part in route[1:].split("/")):
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
            await _relay(writer, method, response)
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
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, headers: list[tuple[str, str]]
) -> bytes:
    fields = {name.lower(): value for name, value in headers}
    if fields.get("expect", "").lower() == "100-continue":
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
    if "chunked" in fields.get("transfer-encoding", "").lower():
        return await _read_chunked(reader)
    declared = fields.get("content-length", "0")
    if not declared.isdigit():
        raise _BadRequest(400, "a malformed Content-Length")
    length = int(declared)
    if length > _MAX_BODY:
        raise _BadRequest(413, "the request is too large")
    return await reader.readexactly(length) if length else b""


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    body = bytearray()
    while True:
        size_line = (await reader.readline()).split(b";")[0].strip()
        try:
            size = int(size_line, 16)
        except ValueError as error:
            raise _BadRequest(400, "a malformed chunk") from error
        if size == 0:
            while (await reader.readline()).strip():  # trailers
                pass
            return bytes(body)
        if len(body) + size > _MAX_BODY:
            raise _BadRequest(413, "the request is too large")
        body += await reader.readexactly(size)
        await reader.readexactly(2)


async def _relay(writer: asyncio.StreamWriter, method: str, response: httpx.Response) -> None:
    """Send the gateway's answer on as it arrives, so a streamed answer (server-sent events) is not held back."""
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
        try:
            async for chunk in response.aiter_raw():
                if chunk:
                    writer.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    await writer.drain()
        except httpx.HTTPError:
            writer.transport.abort()  # the answer was cut short: the caller must not read it as complete
            return
        writer.write(b"0\r\n\r\n")
    await writer.drain()


async def _answer(writer: asyncio.StreamWriter, status: int, message: str) -> None:
    """A refusal this proxy makes itself, in the shape the Anthropic API answers errors in."""
    reason = {400: "Bad Request", 404: "Not Found", 413: "Payload Too Large", 431: "Header Fields Too Large"}.get(
        status, "Bad Gateway"
    )
    body = ('{"type":"error","error":{"type":"api_error","message":"' + message + '"}}').encode()
    head = f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {len(body)}\r\n"
    try:
        writer.write(head.encode() + b"Connection: close\r\n\r\n" + body)
        await writer.drain()
    except ConnectionError:
        return
