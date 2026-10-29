"""A stand-in for the directory and authorization public routes, faithful to their wire contract.

It answers what the platform's controllers answer (read from ``authfi-regional-directory`` and
``authfi-regional-authorization``): the ``{"data": ...}`` envelope, ``{"data": {"data": [...], "next_cursor",
"has_more"}}`` pages, 201 on create, 204 on delete, ``{"error", "error_description"}`` refusals, UUID path ids, the
per-route ``read:``/``manage:`` permission, the refusal to accept ``email_verified``, and join requests that only a
signed-in person (not a machine client) may decide. What it is NOT: proof that a machine token is granted these
permissions on a live project.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from typing import Any

import httpx

from tests.support import Platform

ROOT = "/apis/regional/v1"
NOW = "2026-10-29T10:00:00Z"
STATUSES = ("active", "invited", "suspended")


def envelope(data: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json={"data": data})


def page(items: list[Any], next_cursor: str = "") -> httpx.Response:
    body: dict[str, Any] = {"data": items, "has_more": next_cursor != ""}
    if next_cursor:
        body["next_cursor"] = next_cursor
    return httpx.Response(200, json={"data": body})


def refusal(status: int, code: str, description: str = "") -> httpx.Response:
    body = {"error": code}
    if description:
        body["error_description"] = description
    return httpx.Response(status, json=body)


def new_id() -> str:
    return str(uuid.uuid4())


class FakeAdmin:
    """The platform plus the admin routes, with state. ``grants`` maps a bearer token to the permissions it holds."""

    def __init__(self, platform: Platform) -> None:
        self.platform = platform
        self.grants: dict[str, set[str]] = {}
        self.default_grants: set[str] = {
            "read:organizations",
            "manage:organizations",
            "read:users",
            "manage:users",
            "read:policy",
            "manage:roles",
        }
        self.callers: dict[str, str] = {}
        self.requests: list[httpx.Request] = []
        self.orgs: dict[str, dict[str, Any]] = {}
        self.members: dict[str, dict[str, dict[str, Any]]] = {}
        self.invitations: dict[str, dict[str, Any]] = {}
        self.join_requests: dict[str, dict[str, Any]] = {}
        self.users: dict[str, dict[str, Any]] = {}
        self.roles: dict[str, dict[str, Any]] = {}
        self.role_permissions: dict[str, list[str]] = {}
        self.permissions: dict[str, dict[str, Any]] = {}
        self.bindings: dict[str, dict[str, Any]] = {}
        self.plan_blocks: set[str] = set()
        self.fail_next: list[httpx.Response] = []
        self.routes: list[tuple[str, re.Pattern[str], Callable[[httpx.Request, re.Match[str]], httpx.Response]]] = []
        self._mount()

    # --- plumbing -------------------------------------------------------------------------------------------

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    def seen(self, method: str, suffix: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == ROOT + suffix]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.startswith(ROOT + "/"):
            return self.platform(request)
        self.requests.append(request)
        if self.fail_next:
            return self.fail_next.pop(0)
        header = request.headers.get("Authorization", "")
        token = header[7:] if header.startswith("Bearer ") else ""
        if not token or (token not in self.grants and token not in self.callers and not token.startswith("token-")):
            return refusal(401, "unauthorized")
        path = request.url.path[len(ROOT) :]
        for method, pattern, handler in self.routes:
            match = pattern.fullmatch(path)
            if match and method == request.method:
                feature = path.split("/")[1]
                if feature in self.plan_blocks:
                    return refusal(402, "feature_not_in_plan", f"{feature} is not in this plan")
                needed = self._permission(request, path)
                held = self.grants.get(token, self.default_grants)
                if needed and needed not in held:
                    return refusal(403, "forbidden", f"{needed} is required")
                return handler(request, match)
        return refusal(404, "not_found")

    @staticmethod
    def _permission(request: httpx.Request, path: str) -> str:
        if (
            path.startswith("/organizations/invitations/accept")
            or "/join-requests" in path
            or path.endswith("/domains")
        ):
            return ""
        resource = {"organizations": "organizations", "users": "users"}.get(path.split("/")[1], "")
        write = request.method != "GET"
        if resource:
            return ("manage:" if write else "read:") + resource
        return "manage:roles" if write else "read:policy"

    @staticmethod
    def body(request: httpx.Request) -> dict[str, Any]:
        return json.loads(request.content) if request.content else {}

    def route(
        self, method: str, pattern: str, handler: Callable[[httpx.Request, re.Match[str]], httpx.Response]
    ) -> None:
        self.routes.append((method, re.compile(pattern), handler))

    @staticmethod
    def paged(request: httpx.Request, items: list[Any]) -> httpx.Response:
        limit = int(request.url.params.get("limit", "2"))
        start = int(request.url.params.get("cursor", "0") or 0)
        chunk = items[start : start + limit]
        return page(chunk, str(start + limit) if start + limit < len(items) else "")

    # --- the routes -----------------------------------------------------------------------------------------

    def _mount(self) -> None:
        uid = r"([0-9a-f-]{36})"
        self.route("POST", "/organizations", self._create_org)
        self.route("GET", "/organizations", lambda r, m: self.paged(r, list(self.orgs.values())))
        self.route("GET", f"/organizations/{uid}", self._get_org)
        self.route("PATCH", f"/organizations/{uid}", self._update_org)
        self.route("DELETE", f"/organizations/{uid}", self._delete_org)
        self.route("GET", f"/organizations/{uid}/members", self._members)
        self.route("POST", f"/organizations/{uid}/members", self._add_member)
        self.route("DELETE", f"/organizations/{uid}/members/{uid}", self._remove_member)
        self.route("GET", f"/organizations/{uid}/invitations", self._invitations)
        self.route("POST", f"/organizations/{uid}/invitations", self._invite)
        self.route("DELETE", f"/organizations/{uid}/invitations/{uid}", self._revoke_invitation)
        self.route("POST", "/organizations/invitations/accept", self._accept)
        self.route("GET", f"/organizations/{uid}/join-requests", self._join_requests)
        self.route("POST", f"/organizations/{uid}/join-requests/{uid}/approve", self._decide("approved"))
        self.route("POST", f"/organizations/{uid}/join-requests/{uid}/decline", self._decide("declined"))
        self.route("GET", "/organizations/join-requests/mine", lambda r, m: envelope(list(self.join_requests.values())))
        self.route("POST", f"/organizations/join-requests/{uid}/withdraw", self._withdraw)
        self.route("GET", f"/organizations/{uid}/domains", self._domains)
        self.route("POST", "/users", self._create_user)
        self.route("GET", "/users", self._list_users)
        self.route("GET", f"/users/{uid}", self._get_user)
        self.route("PATCH", f"/users/{uid}", self._update_user)
        self.route("DELETE", f"/users/{uid}", self._delete_user)
        self.route("GET", "/roles", lambda r, m: self.paged(r, list(self.roles.values())))
        self.route("POST", "/roles", self._create_role)
        self.route("GET", f"/roles/{uid}/permissions", self._role_permissions)
        self.route("PUT", f"/roles/{uid}/permissions", self._set_role_permissions)
        self.route("GET", "/permissions", lambda r, m: self.paged(r, list(self.permissions.values())))
        self.route("POST", "/permissions", self._create_permission)
        self.route("GET", "/bindings", self._list_bindings)
        self.route("POST", "/bindings", self._assign)
        self.route("DELETE", r"/bindings/([^/]+)/([0-9a-f-]{36})", self._revoke)

    def _create_org(self, request: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        body = self.body(request)
        if not body.get("slug") or not body.get("display_name"):
            return refusal(400, "invalid_body", "slug is required")
        if any(o["slug"] == body["slug"] for o in self.orgs.values()):
            return refusal(409, "conflict", "slug taken")
        org: dict[str, Any] = {
            "id": new_id(),
            "slug": body["slug"],
            "display_name": body["display_name"],
            "attributes": {},
            "created_at": NOW,
            "updated_at": NOW,
        }
        self.orgs[str(org["id"])] = org
        self.members[str(org["id"])] = {}
        return envelope(org, 201)

    def _get_org(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        org = self.orgs.get(m.group(1))
        return envelope(org) if org else refusal(404, "not_found")

    def _update_org(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        org = self.orgs.get(m.group(1))
        if not org:
            return refusal(404, "not_found")
        body = self.body(request)
        if not body:
            return refusal(400, "invalid_body", "display_name is required")
        org.update(body)
        return envelope(org)

    def _delete_org(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        if self.orgs.pop(m.group(1), None) is None:
            return refusal(404, "not_found")
        return httpx.Response(204)

    def _members(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        return page(list(self.members.get(m.group(1), {}).values()))

    def _add_member(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        body = self.body(request)
        if m.group(1) not in self.orgs:
            return refusal(404, "not_found")
        self.members[m.group(1)][body["user_id"]] = {
            "user_id": body["user_id"],
            "role": body.get("role") or "member",
            "created_at": NOW,
        }
        return envelope({"ok": True})

    def _remove_member(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        self.members.get(m.group(1), {}).pop(m.group(2), None)
        return envelope({"ok": True})

    def _invitations(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        return page([i for i in self.invitations.values() if i["org_id"] == m.group(1)])

    def _invite(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        body = self.body(request)
        if not body.get("email"):
            return refusal(400, "invalid_body", "email is required")
        inv = {
            "id": new_id(),
            "org_id": m.group(1),
            "email": body["email"],
            "role": body.get("role") or "member",
            "token": "inv_" + new_id(),
            "status": "pending",
            "expires_at": NOW,
            "created_at": NOW,
        }
        self.invitations[inv["id"]] = inv
        return envelope(inv, 201)

    def _revoke_invitation(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        if self.invitations.pop(m.group(2), None) is None:
            return refusal(404, "not_found")
        return httpx.Response(204)

    def _accept(self, request: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        body = self.body(request)
        token = request.headers["Authorization"][7:]
        if self.callers.get(token) != body.get("user_id"):
            return refusal(403, "forbidden", "the user joining must be the signed-in caller")
        for inv in self.invitations.values():
            if inv["token"] == body.get("token"):
                inv["status"] = "accepted"
                return envelope(inv)
        return refusal(404, "not_found")

    def _decider_is_a_person(self, request: httpx.Request) -> bool:
        return request.headers["Authorization"][7:] in self.callers

    def _join_requests(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        if not self._decider_is_a_person(request):
            return refusal(403, "forbidden", "only an owner or admin of this organization can do this")
        status = request.url.params.get("status")
        return envelope(
            [
                j
                for j in self.join_requests.values()
                if j["org_id"] == m.group(1) and (not status or j["status"] == status)
            ]
        )

    def _decide(self, outcome: str) -> Callable[[httpx.Request, re.Match[str]], httpx.Response]:
        def decide(request: httpx.Request, m: re.Match[str]) -> httpx.Response:
            if not self._decider_is_a_person(request):
                return refusal(403, "forbidden", "only an owner or admin of this organization can do this")
            found = self.join_requests.get(m.group(2))
            if not found:
                return refusal(404, "not_found")
            found["status"] = outcome
            return envelope(found)

        return decide

    def _withdraw(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        self.join_requests.pop(m.group(1), None)
        return httpx.Response(204)

    def _domains(self, _r: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        return envelope(
            [{"domain": "acme.example", "claimed_via": "signup", "claimed_by": "user:1", "created_at": NOW}]
        )

    def _create_user(self, request: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        body = self.body(request)
        if body.get("email_verified") or body.get("phone_verified"):
            return refusal(403, "forbidden", "email_verified and phone_verified are set only by the identity service")
        if "@" not in body.get("email", "") or not body.get("display_name"):
            return refusal(400, "invalid_body", "email and display_name are required")
        user: dict[str, Any] = {
            "id": new_id(),
            "email": body["email"],
            "email_verified": False,
            "display_name": body["display_name"],
            "phone_verified": False,
            "status": "active",
            "created_at": NOW,
            "updated_at": NOW,
        }
        user.update({k: v for k, v in body.items() if k not in ("email", "display_name")})
        self.users[str(user["id"])] = user
        return envelope(user, 201)

    def _list_users(self, request: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        q = request.url.params.get("q", "").lower()
        return self.paged(request, [u for u in self.users.values() if q in u["email"].lower()])

    def _get_user(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        user = self.users.get(m.group(1))
        return envelope(user) if user else refusal(404, "not_found")

    def _update_user(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        user = self.users.get(m.group(1))
        body = self.body(request)
        if not user:
            return refusal(404, "not_found")
        if not body:
            return refusal(400, "invalid_body", "at least one field is required")
        if body.get("email_verified") or body.get("phone_verified"):
            return refusal(403, "forbidden", "email_verified and phone_verified are set only by the identity service")
        if "status" in body and body["status"] not in STATUSES:
            return refusal(400, "invalid_body", "status must be active, invited, or suspended")
        user.update(body)
        return envelope(user)

    def _delete_user(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        if self.users.pop(m.group(1), None) is None:
            return refusal(404, "not_found")
        return httpx.Response(204)

    def _create_role(self, request: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        body = self.body(request)
        if not body.get("name"):
            return refusal(400, "invalid_body", "name is required")
        if any(r["name"] == body["name"] for r in self.roles.values()):
            return refusal(409, "conflict", "role exists")
        role: dict[str, Any] = {
            "id": new_id(),
            "name": body["name"],
            "description": body.get("description", ""),
            "system": False,
            "created_at": NOW,
        }
        self.roles[str(role["id"])] = role
        self.role_permissions[str(role["id"])] = []
        return envelope(role, 201)

    def _role_permissions(self, _r: httpx.Request, m: re.Match[str]) -> httpx.Response:
        if m.group(1) not in self.roles:
            return refusal(404, "not_found")
        return envelope([self.permissions[p] for p in self.role_permissions[m.group(1)]])

    def _set_role_permissions(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        if m.group(1) not in self.roles:
            return refusal(404, "not_found")
        ids = self.body(request).get("permission_ids", [])
        if any(i not in self.permissions for i in ids):
            return refusal(400, "invalid_body", "a permission is outside the tenant")
        self.role_permissions[m.group(1)] = ids
        return httpx.Response(204)

    def _create_permission(self, request: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        body = self.body(request)
        if not body.get("action") or not body.get("resource"):
            return refusal(400, "invalid_body", "action and resource are required")
        perm = {
            "id": new_id(),
            "action": body["action"],
            "resource": body["resource"],
            "description": body.get("description", ""),
            "created_at": NOW,
        }
        self.permissions[perm["id"]] = perm
        return envelope(perm, 201)

    def _list_bindings(self, request: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        subject, role = request.url.params.get("subject"), request.url.params.get("role")
        if not subject and not role:
            return refusal(400, "invalid_request", "a subject or role query parameter is required")
        found = [
            b
            for b in self.bindings.values()
            if (subject and b["subject"] == subject) or (role and b["role_id"] == role)
        ]
        return self.paged(request, found)

    def _assign(self, request: httpx.Request, _m: re.Match[str]) -> httpx.Response:
        body = self.body(request)
        if body.get("role_id") not in self.roles:
            return refusal(404, "not_found", "role")
        binding = {
            "id": new_id(),
            "subject": body["subject"],
            "role_id": body["role_id"],
            "role_name": self.roles[body["role_id"]]["name"],
            "created_at": NOW,
        }
        if body.get("expires_at"):
            binding["expires_at"] = body["expires_at"]
        self.bindings[binding["id"]] = binding
        return envelope(binding, 201)

    def _revoke(self, request: httpx.Request, m: re.Match[str]) -> httpx.Response:
        from urllib.parse import unquote

        subject = unquote(m.group(1))
        for key, b in list(self.bindings.items()):
            if b["subject"] == subject and b["role_id"] == m.group(2):
                del self.bindings[key]
                return httpx.Response(204)
        return refusal(404, "not_found")
