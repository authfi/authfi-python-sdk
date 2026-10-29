"""The admin client: organizations, users, roles and role bindings, over the project's public API.

::

    m2m = af.client_credentials("app_backend", Secret(os.environ["AUTHFI_CLIENT_SECRET"]))
    admin = m2m.admin

    org = admin.organizations.create("acme", "Acme Inc.")
    admin.organizations.invite(org.id, "ada@acme.example", role="admin")
    for user in admin.users.iterate():
        ...

Every call goes to ``<api origin>/apis/regional/v1/...`` with the credential's token. The credential must be
one the project's API accepts: a client-credentials token of an OAuth application that holds the roles with
``read:`` and ``manage:`` permissions for what it calls (``manage:organizations``, ``manage:users``,
``read:policy``, ``manage:roles``), or a signed-in person's token that does. Without the permission the call
raises :class:`~authfi.APIError` with status 403; a plan that does not include the feature answers 402 or 403
with its own code. Read ``error.code`` and ``error.status``.

Ids are the platform's UUID strings. A list returns a :class:`Page`; ``iterate()`` follows the cursors.
The organization ``role`` on a member or an invitation is the organization's own; policy roles and their
bindings (``roles``, ``bindings``) are tenant-wide.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, TypeVar, Unpack

from authfi._protocol.admin import (
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
)
from authfi._protocol.admin_calls import (
    BindingCalls,
    NewUserFields,
    OrganizationCalls,
    PermissionCalls,
    RoleCalls,
    UserCalls,
    UserChanges,
)
from authfi.errors import APIError, ConfigurationError
from authfi.secret import Secret
from authfi.tokens import AsyncTokenSource, TokenSource

if TYPE_CHECKING:
    from authfi.client import AsyncAuthFI, AuthFI

__all__ = [
    "Admin",
    "AsyncAdmin",
    "JoinRequest",
    "NewUserFields",
    "OrgDomain",
    "OrgInvitation",
    "OrgMember",
    "Organization",
    "Page",
    "Permission",
    "Role",
    "RoleBinding",
    "User",
    "UserChanges",
]

T = TypeVar("T")


def _next(cursor: str | None, previous: str | None) -> str:
    """The cursor for the next page, refusing one that did not move (a loop that never ends)."""
    if cursor is None or cursor == previous:
        raise APIError("invalid_response", "the platform said there is another page and gave no new cursor")
    return cursor


class _Runner:
    """Sends one :class:`Call` with the credential's token, on the project's transport."""

    def __init__(self, project: AuthFI, source: TokenSource) -> None:
        self.project = project
        self.source = source

    def run(self, call: Call[T]) -> T:
        origin = self.project.resolve_api_origin()
        url = call.url(origin)
        authorization = self.source.headers(call.method, url)
        transport = self.project._transport  # pyright: ignore[reportPrivateUsage]
        return call.read(transport.send(call.request(origin, authorization)))


class _AsyncRunner:
    def __init__(self, project: AsyncAuthFI, source: AsyncTokenSource) -> None:
        self.project = project
        self.source = source

    async def run(self, call: Call[T]) -> T:
        origin = await self.project.resolve_api_origin()
        url = call.url(origin)
        authorization = await self.source.headers(call.method, url)
        transport = self.project._transport  # pyright: ignore[reportPrivateUsage]
        return call.read(await transport.send(call.request(origin, authorization)))


# --- blocking -----------------------------------------------------------------------------------------------


class Organizations:
    """Organizations, their members, invitations and join requests (``manage:organizations`` to change)."""

    def __init__(self, runner: _Runner) -> None:
        self._r = runner

    def create(self, slug: str, display_name: str) -> Organization:
        """Create an organization. ``slug`` is its URL-safe handle, unique in the project."""
        return self._r.run(OrganizationCalls.create(slug, display_name))

    def get(self, org_id: str) -> Organization:
        """Read one organization."""
        return self._r.run(OrganizationCalls.get(org_id))

    def list(self, *, limit: int | None = None, cursor: str | None = None) -> Page[Organization]:
        """One page of organizations."""
        return self._r.run(OrganizationCalls.list(limit, cursor))

    def iterate(self, *, limit: int | None = None) -> Iterator[Organization]:
        """Every organization, page after page."""
        cursor: str | None = None
        while True:
            page = self.list(limit=limit, cursor=cursor)
            yield from page.items
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    def update(
        self, org_id: str, *, display_name: str | None = None, attributes: Mapping[str, Any] | None = None
    ) -> Organization:
        """Change the display name and/or replace the business details (``attributes``)."""
        return self._r.run(OrganizationCalls.update(org_id, display_name, attributes))

    def delete(self, org_id: str) -> None:
        """Delete an organization."""
        self._r.run(OrganizationCalls.delete(org_id))

    def members(self, org_id: str) -> tuple[OrgMember, ...]:
        """The organization's members."""
        return self._r.run(OrganizationCalls.members(org_id))

    def add_member(self, org_id: str, user_id: str, *, role: str | None = None) -> None:
        """Add a user with the organization role ``role`` (``member`` when not given)."""
        self._r.run(OrganizationCalls.add_member(org_id, user_id, role))

    def remove_member(self, org_id: str, user_id: str) -> None:
        """Remove a user from the organization."""
        self._r.run(OrganizationCalls.remove_member(org_id, user_id))

    def invitations(self, org_id: str) -> tuple[OrgInvitation, ...]:
        """The organization's invitations."""
        return self._r.run(OrganizationCalls.invitations(org_id))

    def invite(
        self, org_id: str, email: str, *, role: str | None = None, invited_by: str | None = None
    ) -> OrgInvitation:
        """Invite an email address. The returned ``token`` is the invitee's proof to accept: email it, never log it."""
        return self._r.run(OrganizationCalls.invite(org_id, email, role, invited_by))

    def revoke_invitation(self, org_id: str, invitation_id: str) -> None:
        """Revoke a pending invitation."""
        self._r.run(OrganizationCalls.revoke_invitation(org_id, invitation_id))

    def accept_invitation(self, token: Secret | str, user_id: str) -> OrgInvitation:
        """Accept an invitation as the invited user: the credential must be that person's own token."""
        return self._r.run(OrganizationCalls.accept_invitation(token, user_id))

    def join_requests(self, org_id: str, *, status: str | None = None) -> tuple[JoinRequest, ...]:
        """The organization's join requests (``status="pending"`` for those awaiting a decision).

        The platform answers a person who is an owner or admin of the organization; a machine credential is
        refused with 403.
        """
        return self._r.run(OrganizationCalls.join_requests(org_id, status))

    def approve_join_request(self, org_id: str, request_id: str) -> JoinRequest:
        """Approve a join request (an owner or admin of the organization, signed in)."""
        return self._r.run(OrganizationCalls.decide_join_request(org_id, request_id, approve=True))

    def decline_join_request(self, org_id: str, request_id: str) -> JoinRequest:
        """Decline a join request (an owner or admin of the organization, signed in)."""
        return self._r.run(OrganizationCalls.decide_join_request(org_id, request_id, approve=False))

    def my_join_requests(self) -> tuple[JoinRequest, ...]:
        """The signed-in person's own join requests."""
        return self._r.run(OrganizationCalls.my_join_requests())

    def withdraw_join_request(self, request_id: str) -> None:
        """Withdraw the signed-in person's own join request."""
        self._r.run(OrganizationCalls.withdraw_join_request(request_id))

    def domains(self, org_id: str) -> tuple[OrgDomain, ...]:
        """The email domains the organization owns."""
        return self._r.run(OrganizationCalls.domains(org_id))


class Users:
    """Users (``manage:users`` to change, ``read:users`` to read)."""

    def __init__(self, runner: _Runner) -> None:
        self._r = runner

    def create(self, email: str, display_name: str, **fields: Unpack[NewUserFields]) -> User:
        """Create a user. ``fields`` are the profile fields; verification of an email or phone is not one of them."""
        return self._r.run(UserCalls.create(email, display_name, fields))

    def get(self, user_id: str) -> User:
        """Read one user."""
        return self._r.run(UserCalls.get(user_id))

    def list(self, *, limit: int | None = None, cursor: str | None = None, search: str | None = None) -> Page[User]:
        """One page of users, optionally those matching ``search``."""
        return self._r.run(UserCalls.list(limit, cursor, search))

    def iterate(self, *, limit: int | None = None, search: str | None = None) -> Iterator[User]:
        """Every user, page after page."""
        cursor: str | None = None
        while True:
            page = self.list(limit=limit, cursor=cursor, search=search)
            yield from page.items
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    def update(self, user_id: str, **changes: Unpack[UserChanges]) -> User:
        """Change the fields given (``status="suspended"`` blocks sign-in)."""
        return self._r.run(UserCalls.update(user_id, changes))

    def delete(self, user_id: str) -> None:
        """Delete a user."""
        self._r.run(UserCalls.delete(user_id))


class Roles:
    """Policy roles (``read:policy`` to read, ``manage:roles`` to change)."""

    def __init__(self, runner: _Runner) -> None:
        self._r = runner

    def list(self, *, limit: int | None = None, cursor: str | None = None) -> Page[Role]:
        """One page of roles."""
        return self._r.run(RoleCalls.list(limit, cursor))

    def iterate(self, *, limit: int | None = None) -> Iterator[Role]:
        """Every role, page after page."""
        cursor: str | None = None
        while True:
            page = self.list(limit=limit, cursor=cursor)
            yield from page.items
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    def create(self, name: str, *, description: str | None = None) -> Role:
        """Create a role."""
        return self._r.run(RoleCalls.create(name, description))

    def permissions(self, role_id: str) -> tuple[Permission, ...]:
        """The permissions the role grants."""
        return self._r.run(RoleCalls.permissions(role_id))

    def set_permissions(self, role_id: str, permission_ids: Sequence[str]) -> None:
        """Replace what the role grants with exactly these permissions (an empty list clears them)."""
        self._r.run(RoleCalls.set_permissions(role_id, permission_ids))


class Permissions:
    """The permission catalogue."""

    def __init__(self, runner: _Runner) -> None:
        self._r = runner

    def list(self, *, limit: int | None = None, cursor: str | None = None) -> Page[Permission]:
        """One page of permissions."""
        return self._r.run(PermissionCalls.list(limit, cursor))

    def iterate(self, *, limit: int | None = None) -> Iterator[Permission]:
        """Every permission, page after page."""
        cursor: str | None = None
        while True:
            page = self.list(limit=limit, cursor=cursor)
            yield from page.items
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    def create(self, action: str, resource: str, *, description: str | None = None) -> Permission:
        """Create the permission ``action:resource``."""
        return self._r.run(PermissionCalls.create(action, resource, description))


class Bindings:
    """Who holds which role."""

    def __init__(self, runner: _Runner) -> None:
        self._r = runner

    def list(
        self,
        *,
        subject: str | None = None,
        role_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> Page[RoleBinding]:
        """The roles ``subject`` holds, or the subjects that hold ``role_id``. Give exactly one."""
        return self._r.run(BindingCalls.list(subject, role_id, limit, cursor))

    def iterate(
        self, *, subject: str | None = None, role_id: str | None = None, limit: int | None = None
    ) -> Iterator[RoleBinding]:
        """Every binding of a subject or a role, page after page."""
        cursor: str | None = None
        while True:
            page = self.list(subject=subject, role_id=role_id, limit=limit, cursor=cursor)
            yield from page.items
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    def assign(self, subject: str, role_id: str, *, expires_at: datetime | None = None) -> RoleBinding:
        """Give ``subject`` (``user:<id>``, ``app:<id>``, ``agent:<id>``, ``group:<id>``) the role."""
        return self._r.run(BindingCalls.assign(subject, role_id, expires_at))

    def revoke(self, subject: str, role_id: str) -> None:
        """Take the role from the subject."""
        self._r.run(BindingCalls.revoke(subject, role_id))


class Admin:
    """The admin API for blocking code. Get it from a credential: ``m2m.admin``.

    ``organizations``, ``users``, ``roles``, ``permissions`` and ``bindings`` share the credential's token
    and the project's HTTP transport.
    """

    def __init__(self, source: TokenSource, *, project: AuthFI | None = None) -> None:
        chosen = project if project is not None else getattr(source, "project", None)
        if chosen is None or not isinstance(source, TokenSource):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigurationError(
                "Admin needs a blocking AuthFI credential (such as af.client_credentials(...)) and its project"
            )
        runner = _Runner(chosen, source)
        self.organizations = Organizations(runner)
        self.users = Users(runner)
        self.roles = Roles(runner)
        self.permissions = Permissions(runner)
        self.bindings = Bindings(runner)


# --- asyncio ------------------------------------------------------------------------------------------------


class AsyncOrganizations:
    """Organizations, their members, invitations and join requests; see :class:`Organizations`."""

    def __init__(self, runner: _AsyncRunner) -> None:
        self._r = runner

    async def create(self, slug: str, display_name: str) -> Organization:
        """See :meth:`Organizations.create`."""
        return await self._r.run(OrganizationCalls.create(slug, display_name))

    async def get(self, org_id: str) -> Organization:
        """See :meth:`Organizations.get`."""
        return await self._r.run(OrganizationCalls.get(org_id))

    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> Page[Organization]:
        """See :meth:`Organizations.list`."""
        return await self._r.run(OrganizationCalls.list(limit, cursor))

    async def iterate(self, *, limit: int | None = None) -> AsyncIterator[Organization]:
        """See :meth:`Organizations.iterate`."""
        cursor: str | None = None
        while True:
            page = await self.list(limit=limit, cursor=cursor)
            for item in page.items:
                yield item
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    async def update(
        self, org_id: str, *, display_name: str | None = None, attributes: Mapping[str, Any] | None = None
    ) -> Organization:
        """See :meth:`Organizations.update`."""
        return await self._r.run(OrganizationCalls.update(org_id, display_name, attributes))

    async def delete(self, org_id: str) -> None:
        """See :meth:`Organizations.delete`."""
        await self._r.run(OrganizationCalls.delete(org_id))

    async def members(self, org_id: str) -> tuple[OrgMember, ...]:
        """See :meth:`Organizations.members`."""
        return await self._r.run(OrganizationCalls.members(org_id))

    async def add_member(self, org_id: str, user_id: str, *, role: str | None = None) -> None:
        """See :meth:`Organizations.add_member`."""
        await self._r.run(OrganizationCalls.add_member(org_id, user_id, role))

    async def remove_member(self, org_id: str, user_id: str) -> None:
        """See :meth:`Organizations.remove_member`."""
        await self._r.run(OrganizationCalls.remove_member(org_id, user_id))

    async def invitations(self, org_id: str) -> tuple[OrgInvitation, ...]:
        """See :meth:`Organizations.invitations`."""
        return await self._r.run(OrganizationCalls.invitations(org_id))

    async def invite(
        self, org_id: str, email: str, *, role: str | None = None, invited_by: str | None = None
    ) -> OrgInvitation:
        """See :meth:`Organizations.invite`."""
        return await self._r.run(OrganizationCalls.invite(org_id, email, role, invited_by))

    async def revoke_invitation(self, org_id: str, invitation_id: str) -> None:
        """See :meth:`Organizations.revoke_invitation`."""
        await self._r.run(OrganizationCalls.revoke_invitation(org_id, invitation_id))

    async def accept_invitation(self, token: Secret | str, user_id: str) -> OrgInvitation:
        """See :meth:`Organizations.accept_invitation`."""
        return await self._r.run(OrganizationCalls.accept_invitation(token, user_id))

    async def join_requests(self, org_id: str, *, status: str | None = None) -> tuple[JoinRequest, ...]:
        """See :meth:`Organizations.join_requests`."""
        return await self._r.run(OrganizationCalls.join_requests(org_id, status))

    async def approve_join_request(self, org_id: str, request_id: str) -> JoinRequest:
        """See :meth:`Organizations.approve_join_request`."""
        return await self._r.run(OrganizationCalls.decide_join_request(org_id, request_id, approve=True))

    async def decline_join_request(self, org_id: str, request_id: str) -> JoinRequest:
        """See :meth:`Organizations.decline_join_request`."""
        return await self._r.run(OrganizationCalls.decide_join_request(org_id, request_id, approve=False))

    async def my_join_requests(self) -> tuple[JoinRequest, ...]:
        """See :meth:`Organizations.my_join_requests`."""
        return await self._r.run(OrganizationCalls.my_join_requests())

    async def withdraw_join_request(self, request_id: str) -> None:
        """See :meth:`Organizations.withdraw_join_request`."""
        await self._r.run(OrganizationCalls.withdraw_join_request(request_id))

    async def domains(self, org_id: str) -> tuple[OrgDomain, ...]:
        """See :meth:`Organizations.domains`."""
        return await self._r.run(OrganizationCalls.domains(org_id))


class AsyncUsers:
    """Users; see :class:`Users`."""

    def __init__(self, runner: _AsyncRunner) -> None:
        self._r = runner

    async def create(self, email: str, display_name: str, **fields: Unpack[NewUserFields]) -> User:
        """See :meth:`Users.create`."""
        return await self._r.run(UserCalls.create(email, display_name, fields))

    async def get(self, user_id: str) -> User:
        """See :meth:`Users.get`."""
        return await self._r.run(UserCalls.get(user_id))

    async def list(
        self, *, limit: int | None = None, cursor: str | None = None, search: str | None = None
    ) -> Page[User]:
        """See :meth:`Users.list`."""
        return await self._r.run(UserCalls.list(limit, cursor, search))

    async def iterate(self, *, limit: int | None = None, search: str | None = None) -> AsyncIterator[User]:
        """See :meth:`Users.iterate`."""
        cursor: str | None = None
        while True:
            page = await self.list(limit=limit, cursor=cursor, search=search)
            for item in page.items:
                yield item
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    async def update(self, user_id: str, **changes: Unpack[UserChanges]) -> User:
        """See :meth:`Users.update`."""
        return await self._r.run(UserCalls.update(user_id, changes))

    async def delete(self, user_id: str) -> None:
        """See :meth:`Users.delete`."""
        await self._r.run(UserCalls.delete(user_id))


class AsyncRoles:
    """Policy roles; see :class:`Roles`."""

    def __init__(self, runner: _AsyncRunner) -> None:
        self._r = runner

    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> Page[Role]:
        """See :meth:`Roles.list`."""
        return await self._r.run(RoleCalls.list(limit, cursor))

    async def iterate(self, *, limit: int | None = None) -> AsyncIterator[Role]:
        """See :meth:`Roles.iterate`."""
        cursor: str | None = None
        while True:
            page = await self.list(limit=limit, cursor=cursor)
            for item in page.items:
                yield item
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    async def create(self, name: str, *, description: str | None = None) -> Role:
        """See :meth:`Roles.create`."""
        return await self._r.run(RoleCalls.create(name, description))

    async def permissions(self, role_id: str) -> tuple[Permission, ...]:
        """See :meth:`Roles.permissions`."""
        return await self._r.run(RoleCalls.permissions(role_id))

    async def set_permissions(self, role_id: str, permission_ids: Sequence[str]) -> None:
        """See :meth:`Roles.set_permissions`."""
        await self._r.run(RoleCalls.set_permissions(role_id, permission_ids))


class AsyncPermissions:
    """The permission catalogue; see :class:`Permissions`."""

    def __init__(self, runner: _AsyncRunner) -> None:
        self._r = runner

    async def list(self, *, limit: int | None = None, cursor: str | None = None) -> Page[Permission]:
        """See :meth:`Permissions.list`."""
        return await self._r.run(PermissionCalls.list(limit, cursor))

    async def iterate(self, *, limit: int | None = None) -> AsyncIterator[Permission]:
        """See :meth:`Permissions.iterate`."""
        cursor: str | None = None
        while True:
            page = await self.list(limit=limit, cursor=cursor)
            for item in page.items:
                yield item
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    async def create(self, action: str, resource: str, *, description: str | None = None) -> Permission:
        """See :meth:`Permissions.create`."""
        return await self._r.run(PermissionCalls.create(action, resource, description))


class AsyncBindings:
    """Who holds which role; see :class:`Bindings`."""

    def __init__(self, runner: _AsyncRunner) -> None:
        self._r = runner

    async def list(
        self,
        *,
        subject: str | None = None,
        role_id: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
    ) -> Page[RoleBinding]:
        """See :meth:`Bindings.list`."""
        return await self._r.run(BindingCalls.list(subject, role_id, limit, cursor))

    async def iterate(
        self, *, subject: str | None = None, role_id: str | None = None, limit: int | None = None
    ) -> AsyncIterator[RoleBinding]:
        """See :meth:`Bindings.iterate`."""
        cursor: str | None = None
        while True:
            page = await self.list(subject=subject, role_id=role_id, limit=limit, cursor=cursor)
            for item in page.items:
                yield item
            if not page.has_more:
                return
            cursor = _next(page.next_cursor, cursor)

    async def assign(self, subject: str, role_id: str, *, expires_at: datetime | None = None) -> RoleBinding:
        """See :meth:`Bindings.assign`."""
        return await self._r.run(BindingCalls.assign(subject, role_id, expires_at))

    async def revoke(self, subject: str, role_id: str) -> None:
        """See :meth:`Bindings.revoke`."""
        await self._r.run(BindingCalls.revoke(subject, role_id))


class AsyncAdmin:
    """The admin API for asyncio code. Get it from a credential: ``m2m.admin`` on an ``AsyncClientCredentials``."""

    def __init__(self, source: AsyncTokenSource, *, project: AsyncAuthFI | None = None) -> None:
        chosen = project if project is not None else getattr(source, "project", None)
        if chosen is None or not isinstance(source, AsyncTokenSource):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ConfigurationError(
                "AsyncAdmin needs an asyncio AuthFI credential (such as af.client_credentials(...) on AsyncAuthFI) "
                "and its project"
            )
        runner = _AsyncRunner(chosen, source)
        self.organizations = AsyncOrganizations(runner)
        self.users = AsyncUsers(runner)
        self.roles = AsyncRoles(runner)
        self.permissions = AsyncPermissions(runner)
        self.bindings = AsyncBindings(runner)
