"""The admin requests, one builder per route. No I/O; the clients in ``authfi.admin`` send them.

Routes and bodies are the platform's (directory: ``/v1/organizations`` and ``/v1/users``; authorization:
``/v1/roles``, ``/v1/permissions`` and ``/v1/bindings``), served on the project origin as
``/apis/regional/v1/...``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, TypedDict

from authfi._protocol.admin import (
    ROOT,
    Call,
    JoinRequest,
    Organization,
    OrgDomain,
    OrgInvitation,
    OrgMember,
    Page,
    Permission,
    Role,
    RoleBinding,
    User,
    compact,
    day,
    listing,
    many,
    nothing,
    paged,
    required_text,
    segment,
    sequence_of_ids,
    valid_id,
    valid_subject,
)
from authfi.errors import ConfigurationError
from authfi.secret import Secret

__all__ = [
    "BindingCalls",
    "NewUserFields",
    "OrganizationCalls",
    "PermissionCalls",
    "RoleCalls",
    "UserCalls",
    "UserChanges",
    "UserProfile",
]


class UserProfile(TypedDict, total=False):
    """The profile fields a user can be created with or changed to. Dates are ``YYYY-MM-DD`` (or a ``date``)."""

    given_name: str
    family_name: str
    middle_name: str
    nickname: str
    honorific_prefix: str
    honorific_suffix: str
    picture_url: str
    phone_number: str
    profile_url: str
    title: str
    user_type: str
    preferred_language: str
    locale: str
    timezone: str
    street_address: str
    locality: str
    region: str
    postal_code: str
    country: str
    employee_id: str
    cost_center: str
    organization: str
    division: str
    department: str
    hire_date: Any
    manager_id: str
    birthdate: Any
    gender: str
    attributes: Mapping[str, Any]


class NewUserFields(UserProfile, total=False):
    """What a user can be created with, beyond the email and display name."""

    subject_id: str


class UserChanges(UserProfile, total=False):
    """What a user can be changed to. ``status`` is ``active``, ``invited`` or ``suspended``."""

    display_name: str
    status: str
    clear_manager: bool


_PROFILE = frozenset(UserProfile.__annotations__)
_CREATE = _PROFILE | {"subject_id"}
_CHANGE = _PROFILE | {"display_name", "status", "clear_manager"}
_PROVEN_BY_IDENTITY = ("email_verified", "phone_verified")
_STATUSES = ("active", "invited", "suspended")
_DATES = ("hire_date", "birthdate")


def _fields(given: Mapping[str, Any], allowed: frozenset[str], what: str) -> dict[str, Any]:
    for name in given:
        if name in _PROVEN_BY_IDENTITY:
            raise ConfigurationError(
                f"{name} is set by AuthFI when the person proves it by signing in; an admin call cannot claim it"
            )
        if name not in allowed:
            raise ConfigurationError(f"{name} is not a field a user can be {what} with: {', '.join(sorted(allowed))}")
    out = dict(given)
    for name in _DATES:
        if name in out and out[name] is not None:
            out[name] = day(out[name])
    if "status" in out and out["status"] not in _STATUSES:
        raise ConfigurationError(f"status is one of {', '.join(_STATUSES)}")
    if "attributes" in out and not isinstance(out["attributes"], Mapping):
        raise ConfigurationError("attributes is an object (a dict)")
    if "manager_id" in out:
        out["manager_id"] = valid_id(out["manager_id"], "manager_id")
    return out


class OrganizationCalls:
    """Organizations, members, invitations and join requests (directory)."""

    base = f"{ROOT}/organizations"

    @classmethod
    def create(cls, slug: str, display_name: str) -> Call[Organization]:
        """Create an organization. The slug is its URL-safe handle, unique in the project."""
        body = {"slug": required_text(slug, "slug"), "display_name": required_text(display_name, "display_name")}
        return Call("POST", cls.base, "creating the organization", Organization.from_wire, body=body, ok=(201,))

    @classmethod
    def get(cls, org_id: str) -> Call[Organization]:
        """Read one organization."""
        return Call(
            "GET", f"{cls.base}/{valid_id(org_id, 'org_id')}", "reading the organization", Organization.from_wire
        )

    @classmethod
    def list(cls, limit: int | None, cursor: str | None) -> Call[Page[Organization]]:
        """One page of organizations."""
        return listing(cls.base, "listing organizations", paged(Organization.from_wire), limit, cursor)

    @classmethod
    def update(cls, org_id: str, display_name: str | None, attributes: Mapping[str, Any] | None) -> Call[Organization]:
        """Change the display name and/or replace the business details."""
        if display_name is None and attributes is None:
            raise ConfigurationError("give a display_name or attributes to change")
        if display_name is not None:
            required_text(display_name, "display_name")
        if attributes is not None and not isinstance(attributes, Mapping):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigurationError("attributes is an object (a dict)")
        body = compact({"display_name": display_name, "attributes": None if attributes is None else dict(attributes)})
        return Call(
            "PATCH",
            f"{cls.base}/{valid_id(org_id, 'org_id')}",
            "updating the organization",
            Organization.from_wire,
            body=body,
        )

    @classmethod
    def delete(cls, org_id: str) -> Call[None]:
        """Delete an organization."""
        return Call(
            "DELETE", f"{cls.base}/{valid_id(org_id, 'org_id')}", "deleting the organization", nothing, ok=(204,)
        )

    @classmethod
    def members(cls, org_id: str) -> Call[tuple[OrgMember, ...]]:
        """The organization's members."""
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/members"
        return Call("GET", path, "listing the organization's members", many(OrgMember.from_wire))

    @classmethod
    def add_member(cls, org_id: str, user_id: str, role: str | None) -> Call[None]:
        """Add a user, with the organization role ``role`` (``member`` when not given)."""
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/members"
        body = compact({"user_id": valid_id(user_id, "user_id"), "role": role})
        return Call("POST", path, "adding the member", nothing, body=body)

    @classmethod
    def remove_member(cls, org_id: str, user_id: str) -> Call[None]:
        """Remove a user from the organization."""
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/members/{valid_id(user_id, 'user_id')}"
        return Call("DELETE", path, "removing the member", nothing)  # answers 200 {"ok": true}

    @classmethod
    def invitations(cls, org_id: str) -> Call[tuple[OrgInvitation, ...]]:
        """The organization's invitations."""
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/invitations"
        return Call("GET", path, "listing the invitations", many(OrgInvitation.from_wire))

    @classmethod
    def invite(cls, org_id: str, email: str, role: str | None, invited_by: str | None) -> Call[OrgInvitation]:
        """Invite an email address, with the organization role ``role``."""
        if "@" not in required_text(email, "email"):
            raise ConfigurationError("email is not an email address")
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/invitations"
        body = compact(
            {
                "email": email,
                "role": role,
                "invited_by": None if invited_by is None else valid_id(invited_by, "invited_by"),
            }
        )
        return Call("POST", path, "creating the invitation", OrgInvitation.from_wire, body=body, ok=(201,))

    @classmethod
    def revoke_invitation(cls, org_id: str, invitation_id: str) -> Call[None]:
        """Revoke a pending invitation."""
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/invitations/{valid_id(invitation_id, 'invitation_id')}"
        return Call("DELETE", path, "revoking the invitation", nothing, ok=(204,))

    @classmethod
    def accept_invitation(cls, token: Secret | str, user_id: str) -> Call[OrgInvitation]:
        """Accept an invitation. The platform requires the caller to be the invited user."""
        value = token.reveal() if isinstance(token, Secret) else token
        body = {"token": required_text(value, "token"), "user_id": valid_id(user_id, "user_id")}
        return Call(
            "POST", f"{cls.base}/invitations/accept", "accepting the invitation", OrgInvitation.from_wire, body=body
        )

    @classmethod
    def join_requests(cls, org_id: str, status: str | None) -> Call[tuple[JoinRequest, ...]]:
        """The organization's join requests, optionally only those with ``status`` (``pending``)."""
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/join-requests"
        query = {} if status is None else {"status": required_text(status, "status")}
        return Call("GET", path, "listing the join requests", many(JoinRequest.from_wire), query=query)

    @classmethod
    def decide_join_request(cls, org_id: str, request_id: str, *, approve: bool) -> Call[JoinRequest]:
        """Approve or decline a join request."""
        verb = "approve" if approve else "decline"
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/join-requests/{valid_id(request_id, 'request_id')}/{verb}"
        return Call("POST", path, f"{verb} the join request", JoinRequest.from_wire)

    @classmethod
    def my_join_requests(cls) -> Call[tuple[JoinRequest, ...]]:
        """The signed-in person's own join requests."""
        return Call("GET", f"{cls.base}/join-requests/mine", "listing your join requests", many(JoinRequest.from_wire))

    @classmethod
    def withdraw_join_request(cls, request_id: str) -> Call[None]:
        """Withdraw the signed-in person's own join request."""
        path = f"{cls.base}/join-requests/{valid_id(request_id, 'request_id')}/withdraw"
        return Call("POST", path, "withdrawing the join request", nothing, ok=(204,))

    @classmethod
    def domains(cls, org_id: str) -> Call[tuple[OrgDomain, ...]]:
        """The email domains the organization owns."""
        path = f"{cls.base}/{valid_id(org_id, 'org_id')}/domains"
        return Call("GET", path, "listing the organization's domains", many(OrgDomain.from_wire))


class UserCalls:
    """Users (directory)."""

    base = f"{ROOT}/users"

    @classmethod
    def create(cls, email: str, display_name: str, fields: Mapping[str, Any]) -> Call[User]:
        """Create a user."""
        if "@" not in required_text(email, "email"):
            raise ConfigurationError("email is not an email address")
        body = {"email": email, "display_name": required_text(display_name, "display_name")}
        body.update(compact(_fields(fields, _CREATE, "created")))
        return Call("POST", cls.base, "creating the user", User.from_wire, body=body, ok=(201,))

    @classmethod
    def get(cls, user_id: str) -> Call[User]:
        """Read one user."""
        return Call("GET", f"{cls.base}/{valid_id(user_id, 'user_id')}", "reading the user", User.from_wire)

    @classmethod
    def list(cls, limit: int | None, cursor: str | None, search: str | None) -> Call[Page[User]]:
        """One page of users, optionally matching ``search``."""
        extra = {} if search is None else {"q": required_text(search, "search")}
        return listing(cls.base, "listing users", paged(User.from_wire), limit, cursor, **extra)

    @classmethod
    def update(cls, user_id: str, changes: Mapping[str, Any]) -> Call[User]:
        """Change a user. Only the fields given change."""
        body = _fields(changes, _CHANGE, "changed")
        if not body:
            raise ConfigurationError("give at least one field to change")
        if body.get("clear_manager") and "manager_id" in body:
            raise ConfigurationError("manager_id and clear_manager contradict each other")
        return Call(
            "PATCH", f"{cls.base}/{valid_id(user_id, 'user_id')}", "updating the user", User.from_wire, body=body
        )

    @classmethod
    def delete(cls, user_id: str) -> Call[None]:
        """Delete a user."""
        return Call("DELETE", f"{cls.base}/{valid_id(user_id, 'user_id')}", "deleting the user", nothing, ok=(204,))


class RoleCalls:
    """Policy roles and what they grant (authorization)."""

    base = f"{ROOT}/roles"

    @classmethod
    def list(cls, limit: int | None, cursor: str | None) -> Call[Page[Role]]:
        """One page of roles."""
        return listing(cls.base, "listing roles", paged(Role.from_wire), limit, cursor)

    @classmethod
    def create(cls, name: str, description: str | None) -> Call[Role]:
        """Create a role."""
        body = compact({"name": required_text(name, "name"), "description": description})
        return Call("POST", cls.base, "creating the role", Role.from_wire, body=body, ok=(201,))

    @classmethod
    def permissions(cls, role_id: str) -> Call[tuple[Permission, ...]]:
        """The permissions a role grants."""
        path = f"{cls.base}/{valid_id(role_id, 'role_id')}/permissions"
        return Call("GET", path, "reading the role's permissions", many(Permission.from_wire))

    @classmethod
    def set_permissions(cls, role_id: str, permission_ids: Sequence[str]) -> Call[None]:
        """Replace the role's permissions with exactly ``permission_ids`` (an empty list clears them)."""
        path = f"{cls.base}/{valid_id(role_id, 'role_id')}/permissions"
        body = {"permission_ids": sequence_of_ids(permission_ids, "permission_ids")}
        return Call("PUT", path, "setting the role's permissions", nothing, body=body, ok=(204,))


class PermissionCalls:
    """The permission catalogue (authorization)."""

    base = f"{ROOT}/permissions"

    @classmethod
    def list(cls, limit: int | None, cursor: str | None) -> Call[Page[Permission]]:
        """One page of permissions."""
        return listing(cls.base, "listing permissions", paged(Permission.from_wire), limit, cursor)

    @classmethod
    def create(cls, action: str, resource: str, description: str | None) -> Call[Permission]:
        """Create a permission ``action:resource``."""
        body = compact(
            {
                "action": required_text(action, "action"),
                "resource": required_text(resource, "resource"),
                "description": description,
            }
        )
        return Call("POST", cls.base, "creating the permission", Permission.from_wire, body=body, ok=(201,))


class BindingCalls:
    """Who holds which role (authorization)."""

    base = f"{ROOT}/bindings"

    @classmethod
    def list(
        cls, subject: str | None, role_id: str | None, limit: int | None, cursor: str | None
    ) -> Call[Page[RoleBinding]]:
        """One page of the roles a subject holds, or of the subjects that hold a role. Give exactly one."""
        if (subject is None) == (role_id is None):
            raise ConfigurationError("give a subject (its roles) or a role_id (who holds it), not both or neither")
        extra = {"subject": valid_subject(subject)} if subject is not None else {"role": valid_id(role_id, "role_id")}
        return listing(cls.base, "listing role bindings", paged(RoleBinding.from_wire), limit, cursor, **extra)

    @classmethod
    def assign(cls, subject: str, role_id: str, expires_at: datetime | None) -> Call[RoleBinding]:
        """Give ``subject`` the role, until ``expires_at`` when given (it needs a time zone)."""
        body: dict[str, Any] = {"subject": valid_subject(subject), "role_id": valid_id(role_id, "role_id")}
        if expires_at is not None:
            if expires_at.tzinfo is None:
                raise ConfigurationError("expires_at needs a time zone, such as datetime.now(UTC) + timedelta(days=1)")
            body["expires_at"] = expires_at.isoformat()
        return Call("POST", cls.base, "assigning the role", RoleBinding.from_wire, body=body, ok=(201,))

    @classmethod
    def revoke(cls, subject: str, role_id: str) -> Call[None]:
        """Take the role from the subject."""
        path = f"{cls.base}/{segment(valid_subject(subject))}/{valid_id(role_id, 'role_id')}"
        return Call("DELETE", path, "revoking the role", nothing, ok=(204,))
