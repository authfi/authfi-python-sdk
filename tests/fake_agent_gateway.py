"""The fake gateway with the platform's agent security on: DPoP, tasks (mandates), acting for a person, cloud.

It extends :class:`tests.fake_gateway.FakeGateway` (the same ``/mcp`` tools, holds and refusals) with what the
platform does on the routes in ``plans/sdk-platform-routes.md``. It re-derives everything it checks from the bytes
it receives, so it is not the SDK grading itself:

- ``/oauth/token``: an agent's ``client_credentials`` with a DPoP proof is bound to the proof's key (``cnf.jkt``,
  ``token_type`` DPoP); without one it is a Bearer token. A token exchange is a person's delegation, or with
  ``authorization_details`` of type ``authfi_mandate`` a task: the actor assertion must name the key of the request's
  proof (``cnf.jkt``), a template decides auto-approval or a person (``authorization_pending`` and a CIBA poll).
  The audience of a delegation is the gateway only (``invalid_target``).
- Proofs are checked as the kernel does: ``typ``, the embedded key and its algorithm, ``htm``, ``htu``, ``iat``
  within 60 seconds, a ``jti`` seen once, ``ath`` for a token, and a server nonce when ``nonce`` is set
  (``use_dpop_nonce`` with ``DPoP-Nonce``: 400 at the token endpoint, 401 at a resource).
- ``/mcp`` and ``/mcp/v1/cloud/credentials``: a bound token is refused as Bearer (``token_dpop_proof_required``);
  a task's root is a bound token whose tools, budget and life decide each call (EMDT005, EMDT006, EMDT004).
"""

from __future__ import annotations

import base64
import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, cast
from urllib.parse import parse_qs

import jwt

from tests.fake_gateway import ALLOWED, STEP_UP, FakeGateway

CIBA = "urn:openid:params:grant-type:ciba"
EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
CLOUD_PATH = "/mcp/v1/cloud/credentials"
#: The gateway's own tool, as ``tools/list`` advertises it.
CLOUD_TOOL: dict[str, Any] = {
    "name": "cloud__credentials",
    "description": "Get short-lived credentials for a cloud role you are granted.",
    "inputSchema": {
        "type": "object",
        "required": ["account", "role"],
        "properties": {"account": {"type": "string"}, "role": {"type": "string"}},
    },
    "_meta": {"io.authfi/scope": "mcp:cloud.credentials"},
}
#: A tool the task templates leave out.
OUTSIDE = "deepwiki__ask_question"


@dataclass
class Template:
    """A task template: the tools it allows, its budget, and who approves it."""

    tools: list[str]
    budget: int = 100
    approval: str = "auto"  # "auto" or "person"
    agents: list[str] = field(default_factory=list[str])


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def thumbprint(jwk: dict[str, Any]) -> str:
    """RFC 7638, derived here and not imported from the SDK."""
    members = {"OKP": ("crv", "kty", "x"), "EC": ("crv", "kty", "x", "y")}[jwk["kty"]]
    canonical = json.dumps({m: jwk[m] for m in members}, separators=(",", ":"), sort_keys=True)
    return _b64(hashlib.sha256(canonical.encode()).digest())


class AgentGateway(FakeGateway):
    """Use as a context manager, as :class:`FakeGateway`."""

    def __init__(
        self,
        *,
        nonce: bool = False,
        always_challenge: bool = False,
        bind: bool = True,
        dpop_algs: tuple[str, ...] = ("EdDSA", "ES256"),
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.nonce = nonce
        self.always_challenge = always_challenge
        self.bind = bind
        self.dpop_algs = dpop_algs
        #: token -> what the platform issued it as.
        self.issued: dict[str, dict[str, Any]] = {}
        self.proofs: list[dict[str, Any]] = []  # every proof accepted: its claims, alg and path
        self.refused_proofs: list[str] = []  # the kernel reason of each proof refused
        self.nonce_challenges = 0
        self.templates: dict[str, Template] = {"support.refund": Template([ALLOWED, STEP_UP], budget=100)}
        self.ciba_answers: list[str] = []
        self.requests_for_person: list[dict[str, str]] = []
        self.exchanges: list[dict[str, str]] = []
        self.task_requests: list[dict[str, Any]] = []
        self.pending_tasks: dict[str, dict[str, Any]] = {}
        #: (account, role) -> "allow", "approval", or "forbidden"
        self.cloud_roles: dict[tuple[str, str], str] = {("prod", "deployer"): "allow", ("prod", "admin"): "approval"}
        self.cloud_calls: list[dict[str, Any]] = []
        self.cloud_approvals: dict[str, str] = {}
        self.ended: set[str] = set()
        self._cloud_tool = False
        #: The scope a person's CIBA approval grants the agent.
        self.person_scope = "send_email"
        self._seen_jti: set[str] = set()
        self._nonce = "nonce-1"
        self._jwt_lock = threading.Lock()

    # --- discovery ---------------------------------------------------------------------------------

    def __enter__(self) -> AgentGateway:
        super().__enter__()
        return self

    def _get(self, path: str, headers: dict[str, str]) -> tuple[int, bytes]:
        status, body = super()._get(path, headers)
        if path == "/.well-known/openid-configuration":
            doc = json.loads(body)
            doc["dpop_signing_alg_values_supported"] = list(self.dpop_algs)
            doc["grant_types_supported"] = ["client_credentials", CIBA, EXCHANGE]
            return status, json.dumps(doc).encode()
        return status, body

    def _authorized(self, headers: dict[str, str]) -> bool:
        # Approval reads are Bearer on the agent service, whatever the token is bound to.
        auth = headers.get("authorization", "")
        return not self.revoked and auth.startswith(("Bearer ", "DPoP "))

    # --- test controls ----------------------------------------------------------------------------

    def decide_task(self, auth_req_id: str, status: str) -> None:
        self.pending_tasks[auth_req_id]["status"] = status

    def end_task(self, mandate_id: str) -> None:
        self.ended.add(mandate_id)

    def decide_cloud(self, approval_id: str, status: str) -> None:
        self.cloud_approvals[approval_id] = status

    # --- tokens ----------------------------------------------------------------------------------

    def _mint(
        self, kind: str, client: str, jkt: str | None, subject: str | None = None, **extra: Any
    ) -> tuple[str, dict[str, Any]]:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "sub": subject or client,
            "aud": self.issuer + "/mcp",
            "iat": now,
            "exp": now + self.token_lifetime,
            "jti": uuid.uuid4().hex,
            "client_id": client,
        }
        if jkt and self.bind:
            claims["cnf"] = {"jkt": jkt}
        claims.update(extra)
        token = jwt.encode(claims, "x" * 32, algorithm="HS256")
        with self._jwt_lock:
            self.issued[token] = {"kind": kind, "jkt": claims.get("cnf", {}).get("jkt"), "claims": claims, **extra}
            self.minted.append(token)
        return token, claims

    # --- proofs ---------------------------------------------------------------------------------

    def _challenge(self, token_endpoint: bool, reason: str = "use_dpop_nonce") -> tuple[int, dict[str, str], bytes]:
        self.nonce_challenges += 1
        if self.always_challenge:
            self._nonce = f"nonce-{self.nonce_challenges + 1}"
        headers = {"DPoP-Nonce": self._nonce}
        if token_endpoint:
            return 400, headers, json.dumps({"error": reason}).encode()
        headers["WWW-Authenticate"] = f'DPoP algs="EdDSA ES256 PS256", error="{reason}"'
        return 401, headers, json.dumps({"error": reason}).encode()

    def _verify(
        self, method: str, path: str, headers: dict[str, str], token: str | None
    ) -> tuple[str | None, tuple[int, dict[str, str], bytes] | None]:
        """``(jkt, None)`` for a good proof, ``(None, None)`` for none, ``(None, refusal)`` for a bad one."""
        proof = headers.get("dpop")
        if not proof:
            return None, None
        token_endpoint = path == "/oauth/token"

        def refuse(reason: str) -> tuple[None, tuple[int, dict[str, str], bytes]]:
            self.refused_proofs.append(reason)
            headers_out = {"WWW-Authenticate": 'DPoP algs="EdDSA ES256 PS256", error="invalid_dpop_proof"'}
            status = 400 if token_endpoint else 401
            body = json.dumps({"error": "invalid_dpop_proof", "code": reason}).encode()
            return None, (status, headers_out, body)

        try:
            header = jwt.get_unverified_header(proof)
        except jwt.PyJWTError:
            return refuse("dpop_proof_malformed")
        if header.get("typ") != "dpop+jwt":
            return refuse("dpop_proof_type_invalid")
        jwk = header.get("jwk")
        alg = header.get("alg")
        if not isinstance(jwk, dict) or "d" in jwk:
            return refuse("dpop_proof_key_invalid")
        if alg not in self.dpop_algs:
            return refuse("dpop_proof_alg_mismatch")
        try:
            claims = jwt.decode(
                proof,
                jwt.PyJWK(cast(dict[str, Any], jwk), algorithm=alg).key,
                algorithms=[alg],
                options={"verify_exp": False, "verify_iat": False, "verify_aud": False},
            )
        except jwt.PyJWTError:
            return refuse("dpop_proof_signature_invalid")
        if claims.get("htm") != method:
            return refuse("dpop_proof_htm_mismatch")
        if claims.get("htu") != self.issuer + path:
            return refuse("dpop_proof_htu_mismatch")
        iat = claims.get("iat")
        if not isinstance(iat, int) or abs(time.time() - iat) > 60:
            return refuse("dpop_proof_iat_invalid")
        jti = claims.get("jti")
        if not isinstance(jti, str) or not jti:
            return refuse("dpop_proof_jti_missing")
        if token is not None:
            expected = _b64(hashlib.sha256(token.encode()).digest())
            if claims.get("ath") != expected:
                return refuse("dpop_proof_ath_mismatch")
        if self.nonce and claims.get("nonce") != self._nonce:
            return None, self._challenge(token_endpoint)
        if self.always_challenge:
            return None, self._challenge(token_endpoint)
        with self._lock:
            if jti in self._seen_jti:
                return refuse("dpop_proof_replayed")
            self._seen_jti.add(jti)
            self.proofs.append({"claims": claims, "alg": alg, "path": path, "method": method, "jwk": jwk})
        return thumbprint(cast(dict[str, Any], jwk)), None

    def _credential(
        self, method: str, path: str, headers: dict[str, str]
    ) -> tuple[dict[str, Any] | None, tuple[int, dict[str, str], bytes] | None]:
        """The issued token a request proves, or the refusal."""
        scheme, _, token = headers.get("authorization", "").partition(" ")
        info = self.issued.get(token)
        unauth = (401, {"WWW-Authenticate": 'Bearer error="invalid_token"'}, b'{"error":"invalid_token"}')
        if info is None or self.revoked or not token:
            return None, unauth
        bound = info.get("jkt")
        if bound and scheme.lower() != "dpop":
            self.refused_proofs.append("token_dpop_proof_required")
            return None, (
                401,
                {"WWW-Authenticate": 'DPoP algs="EdDSA ES256 PS256", error="invalid_token"'},
                json.dumps({"error": "invalid_token", "code": "token_dpop_proof_required"}).encode(),
            )
        if scheme.lower() == "dpop" and not bound:
            self.refused_proofs.append("token_dpop_unbound")
            return None, unauth
        if bound:
            jkt, refusal = self._verify(method, path, headers, token)
            if refusal is not None:
                return None, refusal
            if jkt is None:
                self.refused_proofs.append("dpop_proof_missing")
                return None, (
                    401,
                    {"WWW-Authenticate": 'DPoP algs="EdDSA ES256 PS256", error="invalid_token"'},
                    json.dumps({"error": "invalid_token", "code": "dpop_proof_missing"}).encode(),
                )
            if jkt != bound:
                self.refused_proofs.append("dpop_proof_key_mismatch")
                return None, unauth
        if info["claims"]["exp"] < time.time():
            return None, unauth
        return info, None

    # --- POST ------------------------------------------------------------------------------------

    def _post(self, path: str, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, str], bytes]:
        if path == "/oauth/token":
            return self._token(headers, {k: v[0] for k, v in parse_qs(body.decode()).items()})
        if path == "/apis/regional/v1/ai/auth/backchannel-authorize":
            form = {k: v[0] for k, v in parse_qs(body.decode()).items()}
            self.requests_for_person.append(form)
            if not form.get("client_assertion"):
                return 401, {}, b'{"error":"invalid_client"}'
            return 200, {}, json.dumps({"auth_req_id": "ciba-1", "expires_in": 300, "interval": 1}).encode()
        if path == CLOUD_PATH:
            return self._cloud(headers, body)
        if path == "/mcp" or path.startswith("/mcp/"):
            info, refusal = self._credential("POST", path, headers)
            if refusal is not None:
                return refusal
            assert info is not None
            if info["kind"] == "task" and info["mandate_id"] in self.ended:
                return (
                    401,
                    {"WWW-Authenticate": 'DPoP error="invalid_token", error_description="EMDT004: task ended"'},
                    json.dumps({"error": "EMDT004", "message": "the task has ended"}).encode(),
                )
            message: dict[str, Any] = {}
            if path == "/mcp" and body:
                parsed: Any = json.loads(body)
                message = cast(dict[str, Any], parsed) if isinstance(parsed, dict) else {}
            if info["kind"] == "task" and message.get("method") == "tools/call":
                refused = self._task_call(info, message)
                if refused is not None:
                    return refused
            if self._cloud_tool and message.get("method") == "tools/list":
                status, out_headers, out = super()._post(path, headers, body)
                listing: dict[str, Any] = json.loads(out)
                listing["result"]["tools"].append(CLOUD_TOOL)
                return status, out_headers, json.dumps(listing).encode()
            if self._cloud_tool and message.get("method") == "tools/call":
                params: dict[str, Any] = message.get("params") or {}
                if params.get("name") == "cloud__credentials":
                    return self._cloud_tool_call(message)
        return super()._post(path, headers, body)

    def add_cloud_tool(self) -> None:
        """List the gateway's own ``cloud__credentials`` tool, which holds a privileged role like any held tool."""
        self._cloud_tool = True

    def _cloud_tool_call(self, message: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
        rid: Any = message.get("id")
        call: dict[str, Any] = message.get("params") or {}
        args: dict[str, Any] = dict(call.get("arguments") or {})
        approval: str | None = args.pop("_authfi_approval_id", None)
        decision = self.cloud_roles.get((str(args.get("account")), str(args.get("role"))), "forbidden")
        if decision == "approval":
            if approval is None:
                approval = str(uuid.uuid4())
                self.approvals[approval] = {"status": "pending", "tool": "cloud__credentials", "args": args}
                return self._hold(rid, approval, "approval_required")
            record = self.approvals.get(approval)
            if record is None or record["status"] == "pending":
                return self._hold(rid, approval, "approval_pending")
            if record["status"] != "approved":
                return self._refuse(
                    rid, "cloud__credentials", code="approval_" + record["status"], approval_id=approval
                )
        elif decision == "forbidden":
            return self._refuse(rid, "cloud__credentials", code="ECLOUDCRED003")
        credential = {"provider": "aws", "credential_id": "cred-1", "access_key_id": "ASIAFAKE"}
        result = {
            "content": [{"type": "text", "text": json.dumps(credential)}],
            "structuredContent": credential,
            "isError": False,
        }
        return 200, {}, json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode()

    def _task_call(self, info: dict[str, Any], message: dict[str, Any]) -> tuple[int, dict[str, str], bytes] | None:
        """The answer to a call the task refuses, or ``None`` for a call it allows."""
        params: dict[str, Any] = message.get("params") or {}
        name: str | None = params.get("name")
        args: dict[str, Any] = params.get("arguments") or {}
        rid: Any = message.get("id")

        def refuse(code: str, reason: str) -> tuple[int, dict[str, str], bytes]:
            result = {
                "content": [{"type": "text", "text": f"Refused ({code}): {reason}."}],
                "isError": True,
                "_meta": {"io.authfi/refusal": {"code": code, "reason": reason}},
            }
            return 200, {}, json.dumps({"jsonrpc": "2.0", "id": rid, "result": result}).encode()

        if name not in info["tools"]:
            return refuse("EMDT005", f"tool {name} is not in the task")
        amount: Any = args.get("amount")
        if isinstance(amount, int):
            left = info["budget_left"]
            if amount > left:
                return refuse("EMDT006", f"amount {amount} is above what is left of the task's budget")
            info["budget_left"] = left - amount
        return None

    def _cloud(self, headers: dict[str, str], body: bytes) -> tuple[int, dict[str, str], bytes]:
        info, refusal = self._credential("POST", CLOUD_PATH, headers)
        if refusal is not None:
            return refusal
        assert info is not None
        request: dict[str, Any] = json.loads(body) if body else {}
        self.cloud_calls.append({"kind": info["kind"], "request": request, "mandate_id": info.get("mandate_id")})
        if info["kind"] == "task" and info["mandate_id"] in self.ended:
            return 401, {}, b'{"error":"EMDT004"}'
        decision = self.cloud_roles.get((str(request.get("account")), str(request.get("role"))), "forbidden")
        envelope = {"Content-Type": "application/json"}
        if info["kind"] == "person" and "mcp:cloud.credentials" not in info["claims"].get("scope", "").split():
            # A delegated token's scope is what the person approved: lending an identity is not approving the cloud.
            envelope["WWW-Authenticate"] = 'Bearer error="insufficient_scope", scope="mcp:cloud.credentials"'
            denied = {
                "error": "ECLOUDCRED011",
                "error_description": "the person this agent acts for did not approve cloud.credentials",
            }
            return 403, envelope, json.dumps(denied).encode()
        approval = request.get("approval_id")
        if decision == "forbidden":
            return (
                403,
                envelope,
                json.dumps(
                    {"error": "ECLOUDCRED003", "error_description": "no grant lets this agent assume that role"}
                ).encode(),
            )
        if decision == "approval":
            if approval is None:
                approval_id = str(uuid.uuid4())
                self.cloud_approvals[approval_id] = "pending"
                data = {
                    "status": "approval_required",
                    "approval_id": approval_id,
                    "message": "a person must approve this role",
                }
                return 202, envelope, json.dumps({"data": data}).encode()
            state = self.cloud_approvals.get(approval)
            if state == "pending":
                return (
                    202,
                    envelope,
                    json.dumps({"data": {"status": "approval_required", "approval_id": approval}}).encode(),
                )
            if state != "approved":
                return (
                    403,
                    envelope,
                    json.dumps(
                        {"error": "ECLOUDCRED007", "error_description": "the approval was not granted"}
                    ).encode(),
                )
            self.cloud_approvals[approval] = "used"
        data = {
            "provider": "aws",
            "credential_id": "cred-" + uuid.uuid4().hex[:8],
            "access_key_id": "ASIAFAKE",
            "secret_access_key": "fake-secret",
            "session_token": "fake-session",
            "expiration": "2099-01-01T00:00:00Z",
        }
        return 201, envelope, json.dumps({"data": data}).encode()

    # --- the token endpoint -----------------------------------------------------------------------

    def _token(self, headers: dict[str, str], form: dict[str, str]) -> tuple[int, dict[str, str], bytes]:
        jkt, refusal = self._verify("POST", "/oauth/token", headers, None)
        if refusal is not None:
            return refusal
        grant = form.get("grant_type")
        if grant == EXCHANGE:
            self.exchanges.append(form)
            return self._exchange(form, jkt)
        if not form.get("client_assertion"):
            return 400, {}, b'{"error":"invalid_request"}'
        agent = form.get("client_id", "")
        if grant == "client_credentials":
            token, _ = self._mint("agent", agent, jkt)
            kind = "DPoP" if jkt and self.bind else "Bearer"
            return 200, {}, self._token_body(token, kind, capabilities=["mcp:deepwiki"])
        if grant == CIBA:
            return self._ciba(form, jkt, agent)
        return 400, {}, b'{"error":"unsupported_grant_type"}'

    def _token_body(self, token: str, kind: str, **extra: Any) -> bytes:
        return json.dumps(
            {"access_token": token, "token_type": kind, "expires_in": self.token_lifetime, **extra}
        ).encode()

    def _ciba(self, form: dict[str, str], jkt: str | None, agent: str) -> tuple[int, dict[str, str], bytes]:
        req = form.get("auth_req_id", "")
        task = self.pending_tasks.get(req)
        if task is not None:
            if task["status"] == "pending":
                return 400, {}, b'{"error":"authorization_pending"}'
            if task["status"] == "denied":
                return 400, {}, b'{"error":"access_denied"}'
            if task["status"] == "expired":
                return 400, {}, b'{"error":"expired_token"}'
            if jkt is None or jkt != task["jkt"]:
                return 400, {}, b'{"error":"invalid_grant"}'
            return 200, {}, self._root(task["mandate"], task["agent"], jkt)
        answer = self.ciba_answers.pop(0) if self.ciba_answers else "ok"
        if answer == "ok":
            token, _ = self._mint(
                "person", agent, jkt, subject="user:usr_person", act={"sub": agent}, scope=self.person_scope
            )
            kind = "DPoP" if jkt and self.bind else "Bearer"
            return 200, {}, self._token_body(token, kind, scope=self.person_scope)
        code = {"pending": "authorization_pending", "denied": "access_denied", "expired": "expired_token"}.get(
            answer, answer
        )
        return 400, {}, json.dumps({"error": code}).encode()

    def _exchange(self, form: dict[str, str], jkt: str | None) -> tuple[int, dict[str, str], bytes]:
        if form.get("client_id") or form.get("client_assertion"):
            return 401, {}, b'{"error":"invalid_client"}'
        if "authorization_details" in form:
            return self._task(form, jkt)
        resource = form.get("resource")
        if resource and resource.rstrip("/") != self.issuer + "/mcp":
            reason = {"error": "invalid_target", "error_description": "audience_not_gateway"}
            return 400, {}, json.dumps(reason).encode()
        actor = self.issued.get(form.get("actor_token", ""))
        if actor is None:
            return 400, {}, b'{"error":"invalid_grant"}'
        if (
            actor["jkt"] and actor["jkt"] != jkt
        ):  # checkActorBinding: a bound actor is the agent's only when its key signs
            return 400, {}, b'{"error":"invalid_grant"}'
        agent = actor["claims"]["sub"]
        token, _ = self._mint("person", agent, jkt, subject="user:usr_person", act={"sub": agent})
        kind = "DPoP" if jkt and self.bind else "Bearer"
        return 200, {}, self._token_body(token, kind, issued_token_type="urn:ietf:params:oauth:token-type:access_token")

    def _task(self, form: dict[str, str], jkt: str | None) -> tuple[int, dict[str, str], bytes]:
        def refuse(code: str, status: int = 400) -> tuple[int, dict[str, str], bytes]:
            return status, {}, json.dumps({"error": code}).encode()

        if jkt is None:
            return refuse("invalid_grant")
        try:
            details = json.loads(form["authorization_details"])
            (entry,) = details
        except (ValueError, KeyError):
            return refuse("invalid_authorization_details")
        if entry.get("type") != "authfi_mandate" or set(entry) - {"type", "template", "params", "ttl_seconds"}:
            return refuse("invalid_authorization_details")
        actor = form.get("actor_token", "")
        try:
            claims = jwt.decode(actor, options={"verify_signature": False, "verify_aud": False, "verify_exp": False})
        except jwt.PyJWTError:
            return refuse("invalid_client", 401)
        if claims.get("aud") != self.issuer + "/oauth/token" or claims.get("iss") != claims.get("sub"):
            return refuse("invalid_client", 401)
        if claims["jti"] in self._seen_jti:
            return refuse("invalid_client", 401)
        self._seen_jti.add(claims["jti"])
        if claims.get("cnf", {}).get("jkt") != jkt:
            return refuse("invalid_grant")
        agent = claims["sub"]
        template = self.templates.get(str(entry.get("template")))
        if template is None or (template.agents and agent not in template.agents):
            return refuse("invalid_authorization_details")
        if not form.get("subject_token"):
            return refuse("invalid_request")
        self.task_requests.append(entry)
        mandate = {"id": "mdt_" + uuid.uuid4().hex[:8], "template": entry["template"], "template_def": template}
        if template.approval == "person":
            req = "task-req-" + str(len(self.pending_tasks) + 1)
            self.pending_tasks[req] = {"status": "pending", "mandate": mandate, "jkt": jkt, "agent": agent}
            body = {
                "error": "authorization_pending",
                "auth_req_id": req,
                "interval": 1,
                "authfi_mandate_id": mandate["id"],
                "approval_url": f"{self.issuer}/approve/{req}",
            }
            return 400, {}, json.dumps(body).encode()
        return 200, {}, self._root(mandate, agent, jkt)

    def _root(self, mandate: dict[str, Any], agent: str, jkt: str) -> bytes:
        template: Template = mandate["template_def"]
        token, _ = self._mint(
            "task",
            agent,
            jkt,
            mandate_id=mandate["id"],
            tools=list(template.tools),
            budget_left=template.budget,
            authfi_mandate=True,
        )
        detail = {
            "type": "authfi_mandate",
            "mandate_id": mandate["id"],
            "template": mandate["template"],
            "tools": template.tools,
            "budgets": {"spend": template.budget},
            "expires_at": "2099-01-01T00:00:00Z",
        }
        return json.dumps(
            {
                "access_token": token,
                "token_type": "DPoP" if self.bind else "Bearer",
                "expires_in": 600,
                "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "authorization_details": [detail],
            }
        ).encode()
