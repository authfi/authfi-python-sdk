"""What the AuthFI gateway says, read the same way by every framework adapter.

The AuthFI MCP gateway governs each tool call. It answers with the tool's own
result, a refusal, or a hold for a person's approval. This module reads those
answers, in the forms the gateway uses today and the forms it is moving to, so
each adapter (``authfi.langgraph``, ``authfi.openai_agents``,
``authfi.claude_agent``, ``authfi.adk``) only has to turn them into its
framework's shapes. It imports no framework.

- :class:`Credential`: where the bearer token comes from, refreshed per call.
- :func:`mcp_http_client_factory`: the HTTP client an MCP client should use.
  It sends a current token on every request and turns the gateway's older
  HTTP 403 refusal of one tool call into the tool error newer gateways send.
- :func:`read_result`: is a tool result the tool's answer, a refusal or a hold?
- :class:`ToolNames`: tool names every model provider accepts, and back.
- :func:`never_allowed`: the tools the gateway says no call can succeed for.
"""

import hashlib
import inspect
import json
import logging
import re

from . import (
    AgentAuth,
    AsyncAgentAuth,
    AuthFIError,
    DelegatedToken,
    _bearer_auth,
    _in_thread,
    _unverified_claims,
)

_log = logging.getLogger("authfi")

#: The argument a held call is sent again with, naming the approval a person gave.
APPROVAL_ARGUMENT = "_authfi_approval_id"

#: Where the gateway puts its outlook for a tool in ``tools/list`` annotations.
GOVERNANCE_ANNOTATION = "io.authfi/governance"

#: The outlook for a tool no call can succeed for, whatever its arguments.
NEVER_ALLOWED = "never_allowed"

#: ``_meta`` key of a refused or held tool call's structured detail.
REFUSAL_META = "io.authfi/refusal"

#: ``_meta`` key of a structured hold (``{id, status, expires_at, status_url}``).
APPROVAL_META = "io.authfi/approval"

#: The header a framework's run or call id is sent in, so a trace joins the gateway's record.
REQUEST_ID_HEADER = "X-Request-Id"

#: Where an adapter puts a tool call's request id in the call's ``_meta`` when it
#: cannot set a header per call; the MCP HTTP client sends it as ``X-Request-Id``.
REQUEST_ID_META = "io.authfi/request_id"

# The gateway keeps a caller's request id only when it is 1-128 of these characters.
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")

# Tool names every provider accepts: OpenAI, Bedrock and Mistral cap at 64
# characters of [A-Za-z0-9_-]; Gemini also wants a letter first.
_PROVIDER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
_NOT_PROVIDER_CHAR = re.compile(r"[^A-Za-z0-9_-]")

# A hold as text. The first form is the gateway's before refusals became tool
# errors; the second is the form after.
_HOLD_TEXTS = (
    re.compile(r"\bApproval required\..*?\(approval ([A-Za-z0-9._:-]+)\)", re.S),
    re.compile(r"\bWaiting for approval ([A-Za-z0-9._:-]+) \((approval_[a-z_]+)\)"),
)
_HOLD_CODES = ("approval_required", "approval_pending")
_REFUSAL_PREFIXES = ("Refused", "Denied by policy")

RESULT, REFUSED, HELD = "result", "refused", "held"


# --------------------------------------------------------------------- credentials


class Credential:
    """The bearer token an agent presents, fetched for every call so it never goes stale.

    Built from what the agent already holds:

    - an :class:`authfi.AgentAuth` or :class:`authfi.AsyncAgentAuth`: the
      agent's own token, cached and refreshed before it expires;
    - a :class:`authfi.DelegatedToken`: a token acting for a person, renewed
      or refused as it allows;
    - a zero-argument callable returning a token (``async`` or not).

    A plain string is refused: a fixed token stops working when it expires,
    mid-run, and is the value tracers copy.
    """

    def __init__(self, source):
        if isinstance(source, Credential):
            self._get, self._aget, self.agent_id = source._get, source._aget, source.agent_id
            self.platform_url = source.platform_url
            return
        self.agent_id = None
        self.platform_url = None
        if isinstance(source, AsyncAgentAuth):
            source = source.agent
        if isinstance(source, AgentAuth):
            self.agent_id = source.agent_id
            self.platform_url = source.platform_url
            self._get = source.token
            self._aget = lambda: _in_thread(source.token)
        elif isinstance(source, DelegatedToken):
            claims = _unverified_claims(source["access_token"])
            self.agent_id = claims.get("client_id") or (claims.get("act") or {}).get("sub")
            # The project that issued the token is the one whose gateway accepts it.
            self.platform_url = str(claims.get("iss") or "").rstrip("/") or None
            self._get = source.token
            self._aget = source.atoken
        elif isinstance(source, (str, bytes)):
            raise AuthFIError(
                "A fixed token cannot be refreshed, and is the value tracers record. "
                "Pass the AgentAuth (or AsyncAgentAuth, or the DelegatedToken), or a "
                "function that returns the current token.",
                status=400,
            )
        elif inspect.iscoroutinefunction(source) or inspect.iscoroutinefunction(
            getattr(source, "__call__", None)
        ):
            self._aget = source
            self._get = _no_sync(source)
        elif callable(source):
            self._get = source
            self._aget = lambda: _in_thread(source)
        else:
            raise AuthFIError(
                "Cannot take a token from %s: pass an AgentAuth, AsyncAgentAuth, "
                "DelegatedToken or a function returning the token" % type(source).__name__,
                status=400,
            )

    def token(self):
        """The current token. May sign and mint a new one: in asyncio code use :meth:`atoken`."""
        return _checked(self._get())

    async def atoken(self):
        """The current token, without blocking the event loop."""
        return _checked(await self._aget())

    def headers(self):
        """``{"Authorization": "Bearer ..."}`` with the current token."""
        return {"Authorization": "Bearer " + self.token()}

    async def aheaders(self):
        """:meth:`headers`, without blocking the event loop."""
        return {"Authorization": "Bearer " + await self.atoken()}

    def httpx_auth(self, httpx_module=None):
        """An ``httpx.Auth`` (or ``httpx2.Auth``) sending the current token on every request."""
        return _bearer_auth(self.token, self.atoken, httpx_module)

    def __repr__(self):
        # Never the token: tracers and crash reporters print reprs.
        return "Credential(agent_id=%r)" % (self.agent_id,)


def _no_sync(afn):
    def get():
        raise AuthFIError(
            "This credential is an async function; this caller needs the token "
            "synchronously. Pass a plain function or the AgentAuth.",
            status=500,
        )
    return get


def _checked(token):
    if not token or not isinstance(token, str):
        raise AuthFIError("The credential returned no token", status=500)
    return token


def credential(source):
    """:class:`Credential` from ``source``, or ``source`` itself when it already is one."""
    return source if isinstance(source, Credential) else Credential(source)


def model_base_url(cred, base_url=None):
    """The AI gateway's address: ``base_url`` when given, else ``<project>/mcp``.

    The project is the agent's ``platform_url``, or the issuer of a token
    acting for a person.
    """
    if base_url:
        return base_url.rstrip("/")
    if cred.platform_url:
        return cred.platform_url + "/mcp"
    raise AuthFIError(
        "base_url is required: set it to your project's gateway, "
        "https://<project>.authfi.io/mcp. It is taken from the AgentAuth when you pass one.",
        status=400,
    )


# --------------------------------------------------------------------- request ids


def request_id(value):
    """``value`` as a request id the gateway keeps, or None when it would be dropped."""
    if value is None:
        return None
    text = str(value)
    if _REQUEST_ID.match(text):
        return text
    _log.debug("not sending %r as %s: outside [A-Za-z0-9._:-]{1,128}", text[:40], REQUEST_ID_HEADER)
    return None


def request_id_header(value):
    """``{"X-Request-Id": value}``, or ``{}`` when ``value`` is not a usable request id."""
    rid = request_id(value)
    return {REQUEST_ID_HEADER: rid} if rid else {}


# --------------------------------------------------------------------- tool names


def provider_tool_name(name):
    """``name`` in the form every model provider accepts.

    A name that already qualifies is returned unchanged, so gateways that
    publish provider-legal names (``deepwiki__read_wiki_structure``) pass
    through. A dotted gateway name becomes ``<server>__<tool>``; any other
    character becomes ``_``; a name still too long is shortened and given a
    hash suffix so two long names stay distinct.
    """
    if _PROVIDER_NAME.match(name):
        return name
    out = _NOT_PROVIDER_CHAR.sub("_", name.replace(".", "__"))
    if not out[:1].isalpha():
        out = "t_" + out
    if len(out) > 64:
        out = out[:55] + "_" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:8]
    return out


class ToolNames:
    """Gateway tool names to provider-legal names and back, one to one.

    The gateway names a tool the way it was registered; the model sees the
    provider-legal form; the call goes back under the gateway's name.
    """

    def __init__(self):
        self._provider = {}
        self._gateway = {}

    def provider(self, gateway_name):
        """The name to show the model for ``gateway_name``."""
        if gateway_name in self._provider:
            return self._provider[gateway_name]
        name = provider_tool_name(gateway_name)
        if name in self._gateway and self._gateway[name] != gateway_name:
            # Two gateway names that map to one: keep both callable.
            name = name[:55] + "_" + hashlib.sha256(gateway_name.encode("utf-8")).hexdigest()[:8]
        self._provider[gateway_name] = name
        self._gateway[name] = gateway_name
        return name

    def gateway(self, provider_name):
        """The gateway's name for ``provider_name`` (unchanged when it was never mapped)."""
        return self._gateway.get(provider_name, provider_name)


# --------------------------------------------------------------------- governance


def governance(tool):
    """The gateway's outlook for ``tool`` from ``tools/list``, or None.

    ``tool`` is an MCP tool (``annotations``), a dict of one, or a LangChain
    tool (``metadata``). The outlook is advisory: the call is still decided
    when it is made.
    """
    for holder in (_attr(tool, "annotations"), _attr(tool, "metadata")):
        value = _attr(holder, GOVERNANCE_ANNOTATION)
        if value is None and holder is not None and not isinstance(holder, dict):
            extra = getattr(holder, "model_extra", None) or getattr(holder, "__dict__", {})
            value = extra.get(GOVERNANCE_ANNOTATION)
        if value:
            return value
    return None


def never_allowed(tool):
    """True when the gateway says no call to ``tool`` can be allowed, so it is best left out."""
    return governance(tool) == NEVER_ALLOWED


def _attr(obj, name):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


# --------------------------------------------------------------------- answers


class Answer:
    """What the gateway said about one tool call.

    ``kind`` is :data:`RESULT` (the tool's own answer, which may itself be an
    error from the tool), :data:`REFUSED` or :data:`HELD`. ``text`` is what
    the model should read. For a refusal ``code``, ``reason`` and
    ``required_scope`` are set when the gateway gave them; for a hold,
    ``approval_id``.
    """

    __slots__ = ("kind", "text", "code", "reason", "required_scope", "approval_id", "status")

    def __init__(self, kind, text, code=None, reason=None, required_scope=None, approval_id=None, status=None):
        self.kind = kind
        self.text = text
        self.code = code
        self.reason = reason
        self.required_scope = required_scope
        self.approval_id = approval_id
        self.status = status

    @property
    def refused(self):
        return self.kind == REFUSED

    @property
    def held(self):
        return self.kind == HELD

    def __repr__(self):
        return "Answer(%s, %r)" % (self.kind, self.text[:80])


def result_text(result):
    """The text of an MCP tool result: a ``CallToolResult``, its dict form, or a string."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    content = _attr(result, "content")
    if isinstance(content, str):
        return content
    parts = []
    for block in content or ():
        text = _attr(block, "text")
        if text:
            parts.append(text)
    if not parts and isinstance(result, dict) and result.get("error"):
        parts.append(str(result["error"]))
    return "\n".join(parts)


def _meta(result):
    if isinstance(result, dict):
        meta = result.get("_meta") or result.get("meta")
    else:
        meta = getattr(result, "meta", None) or getattr(result, "_meta", None)
    return meta if isinstance(meta, dict) else {}


def _is_error(result):
    if isinstance(result, dict):
        return bool(result.get("isError") or result.get("is_error"))
    return bool(getattr(result, "isError", False) or getattr(result, "is_error", False))


def read_result(result, is_error=None):
    """Read one tool call's result: the tool's answer, a refusal, or a hold.

    ``result`` is an MCP ``CallToolResult``, its dict form, or the result's
    text. ``is_error`` overrides the result's own flag (pass it with text).
    Only an error result can be a refusal or a hold, so a tool whose output
    happens to quote the gateway's words is still the tool's answer.
    """
    text = result_text(result)
    error = _is_error(result) if is_error is None else bool(is_error)
    meta = _meta(result)

    approval = meta.get(APPROVAL_META)
    if isinstance(approval, dict) and approval.get("id"):
        status = approval.get("status") or "pending"
        code = status if status.startswith("approval_") else "approval_" + status
        if code in _HOLD_CODES:
            return Answer(HELD, text, code=code, approval_id=str(approval["id"]), status=status)

    refusal = meta.get(REFUSAL_META)
    if isinstance(refusal, dict):
        code = refusal.get("code")
        if code in _HOLD_CODES and refusal.get("approval_id"):
            return Answer(HELD, text, code=code, approval_id=str(refusal["approval_id"]), status="pending")
        return Answer(REFUSED, text, code=code or refusal.get("error"), reason=refusal.get("reason"),
                      required_scope=refusal.get("required_scope"), approval_id=refusal.get("approval_id"))

    if not error:
        return Answer(RESULT, text)
    for pattern in _HOLD_TEXTS:
        m = pattern.search(text)
        if m:
            code = m.group(2) if m.lastindex and m.lastindex >= 2 else "approval_required"
            return Answer(HELD, text, code=code, approval_id=m.group(1), status="pending")
    if text.lstrip().startswith(_REFUSAL_PREFIXES):
        return Answer(REFUSED, text)
    return Answer(RESULT, text)


def held_text(answer):
    """What the model reads while a call waits for a person: the wait, not an instruction to retry."""
    return ("This call is waiting for a person's approval in AuthFI (approval %s). "
            "It has not run. Do not call it again yourself; it is sent again once they decide."
            % answer.approval_id)


# --------------------------------------------------------------------- approvals


def with_approval(arguments, approval_id):
    """``arguments`` plus the approval a person gave, for sending a held call again."""
    out = dict(arguments or {})
    out[APPROVAL_ARGUMENT] = approval_id
    return out


def without_approval(arguments):
    """``arguments`` without an approval id, the way the model wrote them."""
    return {k: v for k, v in (arguments or {}).items() if k != APPROVAL_ARGUMENT}


def call_digest(name, arguments):
    """A stable key for one call: its tool and its arguments, without an approval id."""
    body = json.dumps(without_approval(arguments), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256((name + "\n" + body).encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------- HTTP


def mcp_http_client_factory(cred=None, httpx_module=None):
    """A factory for the async HTTP client an MCP client uses to reach the gateway.

    It has the signature MCP clients take, ``(headers=None, timeout=None,
    auth=None)``. The client it builds:

    - sends ``Authorization: Bearer`` with a current token on every request
      when ``cred`` is given and the caller passes no ``auth`` of its own;
    - turns a gateway's HTTP 403 refusal of one tool call (gateways before
      refusals became tool errors) into the tool error later gateways send,
      so the model reads why instead of the client failing the call as an
      outage. A 401 (an unusable token) is left alone and still ends the run;
    - sends a tool call's ``_meta["io.authfi/request_id"]`` as ``X-Request-Id``,
      for MCP clients that keep one session open and cannot set a header per
      call.

    ``httpx_module`` is the package the MCP client uses: ``httpx`` for MCP SDK
    v1, ``httpx2`` for v2.
    """
    module = httpx_module
    if module is None:
        import httpx as module
    cred = credential(cred) if cred is not None else None

    def factory(headers=None, timeout=None, auth=None):
        if auth is None and cred is not None:
            auth = cred.httpx_auth(module)
        return module.AsyncClient(
            headers=headers,
            timeout=timeout if timeout is not None else module.Timeout(30.0, read=300.0),
            auth=auth,
            follow_redirects=True,
            transport=_gateway_transport(module, module.AsyncHTTPTransport()),
        )

    return factory


def _gateway_transport(module, inner):
    class AuthFIGatewayTransport(module.AsyncBaseTransport):
        async def handle_async_request(self, request):
            call = _tools_call(request) if request.method == "POST" else None
            if call is not None and REQUEST_ID_HEADER not in request.headers:
                meta = (call.get("params") or {}).get("_meta") or {}
                rid = request_id(meta.get(REQUEST_ID_META)) if isinstance(meta, dict) else None
                if rid:
                    request.headers[REQUEST_ID_HEADER] = rid
            response = await inner.handle_async_request(request)
            if call is None or response.status_code != 403:
                return response
            challenge = response.headers.get("www-authenticate") or ""
            if "insufficient_scope" not in challenge:
                return response
            body = await response.aread()
            await response.aclose()
            rewritten = refusal_from_403(call, body, challenge)
            headers = [(k, v) for k, v in response.headers.items()
                       if k.lower() not in ("content-length", "content-encoding", "transfer-encoding",
                                            "content-type", "www-authenticate")]
            headers.append(("content-type", "application/json"))
            _log.info("gateway refused tool call %s with HTTP 403; handing it to the model as the tool's error",
                      call.get("params", {}).get("name"))
            return module.Response(200, headers=headers, content=rewritten, request=request)

        async def aclose(self):
            await inner.aclose()

    return AuthFIGatewayTransport()


def _tools_call(request):
    try:
        body = json.loads(request.content or b"")
    except (ValueError, TypeError):
        return None
    if isinstance(body, dict) and body.get("method") == "tools/call":
        return body
    return None


_CHALLENGE_PARAM = re.compile(r'(\w+)="([^"]*)"')


def refusal_from_403(call, body, challenge):
    """The JSON-RPC answer a gateway now gives for a 403 refusal of a ``tools/call``."""
    params = dict(_CHALLENGE_PARAM.findall(challenge or ""))
    scope = params.get("scope")
    reason = ""
    try:
        old = json.loads(body or b"")
        reason = result_text((old or {}).get("result"))
    except (ValueError, TypeError, AttributeError):
        pass
    if reason.startswith("Denied by policy:"):
        reason = reason[len("Denied by policy:"):].strip()
    name = (call.get("params") or {}).get("name") or "this tool"
    text = "Refused (insufficient_scope): %s." % (reason or "no grant lets this agent call %s" % name)
    detail = {"error": "insufficient_scope", "reason": reason or "no_grant"}
    if scope:
        text += " The call needs the scope %s." % scope
        detail["required_scope"] = scope
    result = {"content": [{"type": "text", "text": text}], "isError": True, "_meta": {REFUSAL_META: detail}}
    return json.dumps({"jsonrpc": "2.0", "id": call.get("id"), "result": result}).encode("utf-8")


def status_errors(exc):
    """Every HTTP status error inside ``exc``: its causes, and the members of an exception group."""
    seen, out, stack = set(), [], [exc]
    while stack:
        e = stack.pop()
        if e is None or id(e) in seen:
            continue
        seen.add(id(e))
        response = getattr(e, "response", None)
        if response is not None and isinstance(getattr(response, "status_code", None), int):
            out.append(e)
        stack.extend(getattr(e, "exceptions", None) or ())
        stack.append(e.__cause__)
        stack.append(e.__context__)
    return out


class GatewayRefusedToken(AuthFIError):
    """The gateway would not accept the agent's token at all (HTTP 401).

    Not a decision about one tool: the token is missing, expired, revoked or
    for another audience, so no call will succeed. ``challenge`` is the
    gateway's ``WWW-Authenticate`` value.
    """

    def __init__(self, url, challenge):
        super().__init__(
            "The AuthFI gateway refused this agent's token at %s (%s). No call can succeed "
            "with it: the agent may have been revoked." % (url, challenge or "HTTP 401"),
            status=401,
        )
        self.challenge = challenge


def token_refusal(exc):
    """:class:`GatewayRefusedToken` when ``exc`` holds a 401 from the gateway, else None."""
    for e in status_errors(exc):
        if e.response.status_code == 401:
            request = getattr(e, "request", None) or getattr(e.response, "request", None)
            return GatewayRefusedToken(getattr(request, "url", "the gateway"),
                                       e.response.headers.get("www-authenticate"))
    return None
