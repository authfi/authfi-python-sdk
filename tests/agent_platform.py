"""A fake AuthFI project that serves agents: the token grants, CIBA, the action gate and approvals.

It checks what the real platform checks of an agent's request: the assertion's signature against
the registered key, ``iss`` = ``sub`` = the agent, ``aud`` = the endpoint, at most 300 seconds,
and a ``jti`` never seen before. A token exchange must carry no client authentication.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import httpx
import jwt

from tests.support import TENANT, Platform, fixture_json, form_of, json_response

AGENT = "agt_test1"
CIBA = "urn:openid:params:grant-type:ciba"
EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"


@dataclass
class AgentPlatform(Platform):
    """A project with one registered agent."""

    agent_id: str = AGENT
    agent_jwk: dict[str, Any] | None = None
    capabilities: list[str] = field(default_factory=lambda: ["mcp:deepwiki.read_wiki", "send_email"])
    expires_in: int = 900
    seen_jti: set[str] = field(default_factory=set[str])
    minted: int = 0
    ciba_answers: list[str] = field(default_factory=list[str])  # "pending", "slow_down", "ok", "denied", ...
    ciba_polls: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    exchanges: list[dict[str, str]] = field(default_factory=list[dict[str, str]])
    exchange_refusal: str | None = None
    approval_status: str | int = "pending"  # a status, or an HTTP status to answer
    approval_reads: int = 0
    approvals_opened: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    decision: str = "allow"

    def __post_init__(self) -> None:
        super().__post_init__()
        config = json.loads(
            json.dumps(fixture_json("platform-beta/agent-configuration.json")).replace(
                "https://platform-beta.authfi.io", self.issuer
            )
        )
        self.agent_configuration = cast(dict[str, Any], config)
        self.route("GET", "/.well-known/agent-configuration", lambda _r: json_response(self.agent_configuration))
        self.route("POST", "/oauth/token", self._token)
        self.route("POST", "/apis/regional/v1/ai/auth/backchannel-authorize", self._ciba_start)
        self.route("POST", "/apis/regional/v1/ai/auth/transaction-token", self._transaction_token)
        self.route("POST", "/apis/regional/v1/ai/auth/authorize", self._authorize)
        self.route("POST", "/apis/regional/v1/ai/auth/request-approval", self._request_approval)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path.startswith("/apis/regional/v1/ai/auth/approvals/"):
            self.requests.append(request)
            return self._read_approval(request)
        return super().__call__(request)

    # --- helpers -------------------------------------------------------------------------------

    def refuse(self, code: str, status: int = 400) -> httpx.Response:
        return json_response({"error": code, "error_description": f"refused: {code}"}, status=status)

    def check_assertion(self, assertion: str | None, audience: str) -> str | None:
        """None when the assertion is good, else the OAuth error."""
        if not assertion or self.agent_jwk is None:
            return "invalid_client"
        header = jwt.get_unverified_header(assertion)
        if header.get("kid") != self.agent_jwk["kid"]:
            return "invalid_client"
        key = jwt.PyJWK(self.agent_jwk)
        try:
            # The test clock moves on its own, so times are checked against the assertion itself.
            claims = jwt.decode(
                assertion,
                key.key,
                algorithms=[self.agent_jwk["alg"]],
                audience=audience,
                options={"verify_exp": False, "verify_iat": False, "verify_nbf": False},
            )
        except jwt.PyJWTError:
            return "invalid_client"
        if claims.get("iss") != self.agent_id or claims.get("sub") != self.agent_id:
            return "invalid_client"
        if claims["exp"] - claims["iat"] > 300:
            return "invalid_client"
        if claims["jti"] in self.seen_jti:
            return "invalid_grant"
        self.seen_jti.add(claims["jti"])
        return None

    def agent_token(self, resource: str | None = None, **extra: Any) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": self.issuer,
            "sub": self.agent_id,
            "client_id": self.agent_id,
            "aud": resource or self.issuer + "/mcp",
            "exp": now + self.expires_in,
            "iat": now,
            "tenant_id": TENANT,
            "scope": " ".join(self.capabilities),
            "jti": f"t-{self.minted}",
        }
        claims.update(extra)
        return self.keys.sign(claims, alg="EdDSA")

    # --- endpoints ------------------------------------------------------------------------------

    def _token(self, request: httpx.Request) -> httpx.Response:
        form = form_of(request)
        grant = form.get("grant_type")
        if grant == EXCHANGE:
            return self._exchange(form)
        error = self.check_assertion(form.get("client_assertion"), self.issuer + "/oauth/token")
        if error or form.get("client_id") != self.agent_id:
            return self.refuse(error or "invalid_client", 401)
        if grant == "client_credentials":
            self.minted += 1
            return json_response(
                {
                    "access_token": self.agent_token(form.get("resource")),
                    "token_type": "Bearer",
                    "expires_in": self.expires_in,
                    "capabilities": self.capabilities,
                    "scope": " ".join(self.capabilities),
                }
            )
        if grant == CIBA:
            self.ciba_polls.append(form)
            answer = self.ciba_answers.pop(0) if self.ciba_answers else "ok"
            if answer == "ok":
                token = self.agent_token(
                    form.get("resource"),
                    sub="user:usr_person",
                    act={"sub": f"agent:{self.agent_id}"},
                    scope="send_email",
                )
                return json_response(
                    {"access_token": token, "token_type": "Bearer", "expires_in": 600, "scope": "send_email"}
                )
            return self.refuse(
                {"pending": "authorization_pending", "denied": "access_denied", "expired": "expired_token"}.get(
                    answer, answer
                )
            )
        return self.refuse("unsupported_grant_type")

    def _exchange(self, form: dict[str, str]) -> httpx.Response:
        self.exchanges.append(form)
        if "client_id" in form or "client_assertion" in form:
            return self.refuse("invalid_client", 401)
        if self.exchange_refusal:
            return self.refuse(self.exchange_refusal, 403)
        subject = jwt.decode(form["subject_token"], options={"verify_signature": False})
        actor = jwt.decode(form["actor_token"], options={"verify_signature": False})
        now = int(time.time())
        claims = {
            "iss": self.issuer,
            "sub": subject["sub"],
            "act": {"sub": actor["sub"]},
            "client_id": actor["sub"],
            "aud": [form.get("resource") or self.issuer + "/mcp"],
            "exp": now + 300,
            "iat": now,
            "tenant_id": TENANT,
            "jti": f"x-{len(self.exchanges)}",
        }
        return json_response(
            {
                "access_token": self.keys.sign(claims),
                "issued_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "token_type": "Bearer",
                "expires_in": 300,
            }
        )

    def _ciba_start(self, request: httpx.Request) -> httpx.Response:
        form = form_of(request)
        error = self.check_assertion(form.get("client_assertion"), self.issuer + "/oauth/token")
        if error:
            return self.refuse(error, 401)
        return json_response({"auth_req_id": "req-123", "expires_in": 300, "interval": 5})

    def _bearer_agent(self, request: httpx.Request) -> bool:
        raw = request.headers.get("authorization", "").removeprefix("Bearer ")
        try:
            claims = jwt.decode(raw, options={"verify_signature": False})
        except jwt.PyJWTError:
            return False
        return bool(claims.get("sub") == self.agent_id)

    def _transaction_token(self, request: httpx.Request) -> httpx.Response:
        if not self._bearer_agent(request):
            return self.refuse("unauthorized", 401)
        body = json.loads(request.content)
        return json_response(
            {
                "data": {
                    "transaction_token": "txn-token." + body["action"],
                    "txn": "txn-1",
                    "token_type": "N_A",
                    "expires_in": 60,
                }
            }
        )

    def _authorize(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("agent_id") != self.agent_id or not body.get("transaction_token"):
            return self.refuse("unauthorized", 401)
        data: dict[str, Any] = {"decision": self.decision}
        if self.decision == "approval_required":
            data["approval_id"] = "apr-1"
        return json_response({"data": data})

    def _request_approval(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        error = self.check_assertion(body.get("client_assertion"), self.issuer)
        if error or body.get("agent_id") != self.agent_id:
            return self.refuse("unauthorized", 401)
        self.approvals_opened.append(body)
        return json_response(
            {
                "data": {
                    "approval_id": "apr-1",
                    "status": "pending",
                    "requesting_action": body["requesting_action"],
                    "created_at": "2026-10-20T10:00:00Z",
                    "expires_at": "2026-10-20T10:05:00Z",
                }
            },
            status=202,
        )

    def _read_approval(self, request: httpx.Request) -> httpx.Response:
        self.approval_reads += 1
        if isinstance(self.approval_status, int):
            return self.refuse("forbidden", self.approval_status)
        if not self._bearer_agent(request):
            return self.refuse("unauthorized", 401)
        status: str = self.approval_status
        return json_response(
            {
                "data": {
                    "approval_id": request.url.path.rsplit("/", 1)[1],
                    "status": status,
                    "requesting_action": "refund",
                }
            }
        )


ResponseFactory = Callable[[httpx.Request], httpx.Response]
