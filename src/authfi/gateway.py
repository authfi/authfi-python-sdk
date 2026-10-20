"""The AuthFI gateways, read the same way by every framework adapter.

``<issuer>/mcp`` is the MCP gateway (governed tools) and the AI gateway (model calls). A
:class:`Gateway` gives an agent's credential to any HTTP client that talks to them, reads what the
gateway answered (:class:`ToolOutcome`), and asks for cloud credentials::

    gateway = agent.gateway()
    client = gateway.http_client()                 # the current token on every request; no redirects
    outcome = gateway.read_result(call_result)     # allowed, refused or approval_required

A refused tool call and a held one are tool results, not exceptions: the framework hands them to
the model as the tool's answer. Only the token itself being refused (HTTP 401) is an exception,
:class:`~authfi.GatewayTokenRefused`, and it ends the run.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import httpx

from authfi._protocol.gateway import (
    APPROVAL_ARGUMENT,
    HOLD_MESSAGE,
    CloudCredential,
    ToolOutcome,
    cloud_request,
    governance_of,
    parse_cloud,
    read_result,
    scope_of,
    strip_approval_argument,
    with_approval,
)
from authfi._protocol.ids import inject_traceparent, request_id
from authfi._transport import user_agent
from authfi.errors import ConfigurationError, GatewayTokenRefused
from authfi.tokens import AsyncTokenSource, TokenSource

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = [
    "APPROVAL_ARGUMENT",
    "HOLD_MESSAGE",
    "AsyncGateway",
    "CloudCredential",
    "CloudCredentialOutcome",
    "Gateway",
    "ToolOutcome",
    "token_refusal",
]

ModelApi = Literal["anthropic", "openai"]

#: The JSON-RPC error an MCP client gets when the gateway refuses the token (see ``mcp_client_factory``).
TOKEN_REFUSED_RPC_CODE = -32001
_REFUSAL_DATA = "authfi_refusal"


@dataclass(frozen=True, slots=True)
class CloudCredentialOutcome:
    """``issued`` with the ``credential``, or ``approval_required`` with the ``approval_id`` to retry with."""

    kind: Literal["issued", "approval_required"]
    credential: CloudCredential | None = None
    approval_id: str | None = None


def _refused_response(response: object) -> GatewayTokenRefused | None:
    # httpx and httpx2 responses alike: provider SDKs are built on either.
    if getattr(response, "status_code", None) != 401:
        return None
    try:
        url = str(getattr(getattr(response, "request", None), "url", "") or "")
    except RuntimeError:  # a response built without its request
        url = ""
    headers = getattr(response, "headers", None)
    get = getattr(headers, "get", None)
    challenge = get("www-authenticate") if callable(get) else None
    answer_id = get("x-request-id") if callable(get) else None
    return GatewayTokenRefused(
        url,
        challenge if isinstance(challenge, str) else None,
        request_id=answer_id if isinstance(answer_id, str) else None,
    )


def _refused_rpc_error(error: BaseException) -> GatewayTokenRefused | None:
    data = getattr(getattr(error, "error", None), "data", None)
    if not isinstance(data, Mapping):
        return None
    detail = cast(Mapping[str, Any], data)
    if detail.get(_REFUSAL_DATA) != "gateway_token_refused":
        return None
    challenge, url = detail.get("www_authenticate"), detail.get("url")
    return GatewayTokenRefused(url if isinstance(url, str) else "", challenge if isinstance(challenge, str) else None)


def token_refusal(error: BaseException) -> GatewayTokenRefused | None:
    """The gateway's refusal of the token itself, found anywhere in ``error``'s chain or group."""
    seen: set[int] = set()
    stack: list[BaseException | None] = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, GatewayTokenRefused):
            return current
        refused = _refused_response(getattr(current, "response", None)) or _refused_rpc_error(current)
        if refused is not None:
            return refused
        stack.extend(getattr(current, "exceptions", None) or ())
        stack.append(current.__cause__)
        stack.append(current.__context__)
    return None


def _ignore(refused: GatewayTokenRefused | None) -> None:
    return None


class _RefusalAsJsonRpc:
    """An async transport that answers a 401 to a JSON-RPC request as that request's JSON-RPC error."""

    def __init__(
        self, package: ModuleType, inner: Any, seen: Callable[[GatewayTokenRefused | None], None] | None = None
    ) -> None:
        self._package = package
        self._inner = inner
        self._seen = seen or _ignore

    async def handle_async_request(self, request: Any) -> Any:
        response = await self._inner.handle_async_request(request)
        if response.status_code != 401:
            if response.status_code < 400:
                self._seen(None)
            return response
        challenge = response.headers.get("www-authenticate")
        self._seen(GatewayTokenRefused(str(request.url), challenge))
        if request.method != "POST":
            return response
        try:
            message = json.loads(request.content or b"null")
        except ValueError:
            message = None
        rid = cast(Mapping[str, Any], message).get("id") if isinstance(message, Mapping) else None
        if rid is None:
            return response
        await response.aclose()
        error = {
            "code": TOKEN_REFUSED_RPC_CODE,
            "message": "The AuthFI gateway refused this credential",
            "data": {_REFUSAL_DATA: "gateway_token_refused", "url": str(request.url), "www_authenticate": challenge},
        }
        return self._package.Response(
            200,
            json={"jsonrpc": "2.0", "id": rid, "error": error},
            request=request,
        )

    async def __aenter__(self) -> _RefusalAsJsonRpc:
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._inner.__aexit__(*exc)

    async def aclose(self) -> None:
        await self._inner.aclose()


class _GatewayBase:
    def __init__(self, project: AuthFI | AsyncAuthFI) -> None:
        self._project = project
        #: ``<issuer>/mcp``: the MCP gateway, and the base URL of the Anthropic API on the AI gateway.
        self.mcp_url = project.issuer + "/mcp"

    def model_base_url(self, api: ModelApi) -> str:
        """The base URL to give a provider SDK: ``<issuer>/mcp`` for Anthropic, ``<issuer>/mcp/v1`` for OpenAI."""
        if api == "anthropic":
            return self.mcp_url
        if api == "openai":
            return self.mcp_url + "/v1"
        raise ConfigurationError(f"the AI gateway speaks the anthropic and openai APIs, not {api!r}")

    @staticmethod
    def scope_of(tool: object) -> str | None:
        """The grant object a tool from ``tools/list`` needs, or ``None`` for a tool with no AuthFI ``_meta``."""
        return scope_of(tool)

    @staticmethod
    def governance_of(tool: object) -> str | None:
        """The agent's advisory outlook for a tool from ``tools/list``; the gateway still decides each call.

        ``no_policy_restriction``, ``may_require_approval``, ``never_allowed``, or ``None``.
        """
        return governance_of(tool)

    @staticmethod
    def read_result(result: object) -> ToolOutcome:
        """What the gateway did with one tool call (see :class:`ToolOutcome`)."""
        return read_result(result)

    @staticmethod
    def with_approval(arguments: Mapping[str, Any] | None, approval_id: str) -> dict[str, Any]:
        """``arguments`` with the approval a person gave, to send a held call again."""
        return with_approval(arguments, approval_id)

    def strip_approval_argument(self, arguments: Mapping[str, Any] | None) -> dict[str, Any]:
        """``arguments`` without a model-written approval id; logs when one was removed."""
        clean, stripped = strip_approval_argument(arguments)
        if stripped:
            self._project.logger.warning(
                "removed %s from a tool call's arguments: only the adapter adds an approval, on its resume",
                APPROVAL_ARGUMENT,
            )
        return clean

    def _stamp(self, request: httpx.Request) -> None:
        if "x-request-id" not in request.headers:
            request.headers["X-Request-Id"] = request_id()
        headers = dict(request.headers)
        inject_traceparent(headers)
        if "traceparent" in headers and "traceparent" not in request.headers:
            request.headers["traceparent"] = headers["traceparent"]

    def _now(self) -> float:
        return self._project.clock.time()


class Gateway(_GatewayBase):
    """The gateways for one credential, in blocking code. Make it with ``agent.gateway()``."""

    def __init__(self, project: AuthFI, source: TokenSource) -> None:
        if not isinstance(source, TokenSource):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigurationError("Gateway takes a TokenSource; for asyncio code use AsyncGateway")
        super().__init__(project)
        self.project = project
        self.source = source
        self._clients: dict[str, Any] = {}
        project._track(self)  # pyright: ignore[reportPrivateUsage]

    @overload
    def http_client(self) -> httpx.Client: ...

    @overload
    def http_client(self, module: ModuleType) -> Any: ...

    def http_client(self, module: ModuleType | None = None) -> Any:
        """An HTTP client for the gateways: the credential per request, sent nowhere else, no redirects.

        ``module`` is the httpx package the caller needs: ``httpx`` by default, or ``httpx2`` for the
        newer provider SDKs. The client is made once per package and closed by ``AuthFI.close()``;
        do not close it yourself. A refused token is a 401 the caller's SDK reports; adapters turn it
        into :class:`~authfi.GatewayTokenRefused` (see :func:`token_refusal`).
        """
        package = module or httpx
        client = self._clients.get(package.__name__)
        if client is None:
            client = self.new_http_client(module=package)
            self._clients[package.__name__] = client
        return client

    def new_http_client(
        self,
        headers: Mapping[str, str] | None = None,
        timeout: Any = None,
        *,
        module: ModuleType | None = None,
    ) -> Any:
        """A new client the caller owns and closes, set up as :meth:`http_client`'s."""
        package = module or httpx

        def on_request(request: Any) -> None:
            self._stamp(request)

        return package.Client(
            auth=self.source.httpx_auth(package),
            follow_redirects=False,
            timeout=timeout
            if timeout is not None
            else package.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            headers={"User-Agent": user_agent(), **(headers or {})},
            event_hooks={"request": [on_request]},
        )

    def cloud_credentials(
        self,
        account: str,
        role: str,
        *,
        expires_in: int = 900,
        session_policy: Mapping[str, Any] | None = None,
        approval_id: str | None = None,
    ) -> CloudCredentialOutcome:
        """Short-lived credentials for ``role`` in cloud ``account``, through the gateway.

        ``issued`` carries the credential; ``approval_required`` the approval to call again with,
        once a person has decided. A refusal raises :class:`~authfi.APIError` with the gateway's
        code. The platform caps ``expires_in``.
        """
        url = self.project.api_origin + "/mcp/v1/cloud/credentials"
        request = cloud_request(
            self.project.api_origin,
            self.source.headers("POST", url),
            account=account,
            role=role,
            expires_in=expires_in,
            session_policy=session_policy,
            approval_id=approval_id,
        )
        response = self.project._transport.send(request)  # pyright: ignore[reportPrivateUsage]
        if response.status == 401:
            raise GatewayTokenRefused(url, response.header("WWW-Authenticate"), request_id=response.answer_request_id)
        credential, held = parse_cloud(response, self._now())
        if held is not None:
            self.project.logger.info("cloud credentials for %s wait for approval %s", role, held)
            return CloudCredentialOutcome("approval_required", approval_id=held)
        return CloudCredentialOutcome("issued", credential=credential)

    def close(self) -> None:
        """Close the HTTP clients this gateway made."""
        for client in self._clients.values():
            client.close()
        self._clients.clear()

    def __repr__(self) -> str:
        return f"Gateway({self.mcp_url!r}, {self.source!r})"


class AsyncGateway(_GatewayBase):
    """The gateways for one credential, in asyncio code. Make it with ``agent.gateway()``."""

    def __init__(self, project: AsyncAuthFI, source: AsyncTokenSource) -> None:
        if not isinstance(source, AsyncTokenSource):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigurationError("AsyncGateway takes an AsyncTokenSource; for blocking code use Gateway")
        super().__init__(project)
        self.project = project
        self.source = source
        #: The gateway's refusal of this credential, as an MCP client last saw it; ``None`` once it is
        #: accepted again. For frameworks that turn every tool failure into text for the model.
        self.token_refused: GatewayTokenRefused | None = None
        self._clients: dict[str, Any] = {}
        project._track(self)  # pyright: ignore[reportPrivateUsage]

    @overload
    def http_client(self) -> httpx.AsyncClient: ...

    @overload
    def http_client(self, module: ModuleType) -> Any: ...

    def http_client(self, module: ModuleType | None = None) -> Any:
        """See :meth:`Gateway.http_client`; closed by ``AsyncAuthFI.aclose()``."""
        package = module or httpx
        client = self._clients.get(package.__name__)
        if client is None:
            client = self.new_http_client(module=package)
            self._clients[package.__name__] = client
        return client

    def new_http_client(
        self,
        headers: Mapping[str, str] | None = None,
        timeout: Any = None,
        *,
        module: ModuleType | None = None,
    ) -> Any:
        """A new client the caller owns and closes, set up as :meth:`http_client`'s.

        It has the signature an MCP client factory takes (``headers``, ``timeout``), so
        ``functools.partial(gateway.new_http_client, module=httpx)`` is one.
        """
        return self._client(module or httpx, headers, timeout, None)

    def _client(self, package: ModuleType, headers: Mapping[str, str] | None, timeout: Any, transport: Any) -> Any:
        async def on_request(request: Any) -> None:
            self._stamp(request)

        return package.AsyncClient(
            auth=self.source.httpx_auth(package),
            follow_redirects=False,
            timeout=timeout
            if timeout is not None
            else package.Timeout(connect=10.0, read=300.0, write=30.0, pool=10.0),
            headers={"User-Agent": user_agent(), **(headers or {})},
            event_hooks={"request": [on_request]},
            transport=transport,
        )

    def _saw(self, refused: GatewayTokenRefused | None) -> None:
        self.token_refused = refused

    def mcp_client_factory(self, module: ModuleType | None = None) -> Callable[..., Any]:
        """An MCP client's HTTP client factory (``headers``, ``timeout``, ``auth``) for the MCP gateway.

        Each client is set up as :meth:`new_http_client`'s. MCP clients keep a session open and
        answer each request from a background task, where an HTTP 401 is logged and the request then
        waits for its timeout. So a 401 to a JSON-RPC request reaches the client as a JSON-RPC error
        instead, which :func:`token_refusal` reads as :class:`~authfi.GatewayTokenRefused`.
        ``auth`` is refused: the credential is this gateway's.
        """
        package = module or httpx

        def factory(headers: Mapping[str, str] | None = None, timeout: Any = None, auth: Any = None) -> Any:
            if auth is not None:
                raise ConfigurationError("the MCP gateway client carries the agent's credential; pass no auth")
            transport = _RefusalAsJsonRpc(package, package.AsyncHTTPTransport(), self._saw)
            return self._client(package, headers, timeout, transport)

        return factory

    async def cloud_credentials(
        self,
        account: str,
        role: str,
        *,
        expires_in: int = 900,
        session_policy: Mapping[str, Any] | None = None,
        approval_id: str | None = None,
    ) -> CloudCredentialOutcome:
        """See :meth:`Gateway.cloud_credentials`."""
        url = self.project.api_origin + "/mcp/v1/cloud/credentials"
        request = cloud_request(
            self.project.api_origin,
            await self.source.headers("POST", url),
            account=account,
            role=role,
            expires_in=expires_in,
            session_policy=session_policy,
            approval_id=approval_id,
        )
        response = await self.project._transport.send(request)  # pyright: ignore[reportPrivateUsage]
        if response.status == 401:
            raise GatewayTokenRefused(url, response.header("WWW-Authenticate"), request_id=response.answer_request_id)
        credential, held = parse_cloud(response, self._now())
        if held is not None:
            self.project.logger.info("cloud credentials for %s wait for approval %s", role, held)
            return CloudCredentialOutcome("approval_required", approval_id=held)
        return CloudCredentialOutcome("issued", credential=credential)

    async def aclose(self) -> None:
        """Close the HTTP clients this gateway made."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    def __repr__(self) -> str:
        return f"AsyncGateway({self.mcp_url!r}, {self.source!r})"
