"""The admin API: organizations, users, roles and bindings under ``<origin>/apis/regional/v1``. No I/O.

Every call is a :class:`Call`: the request to send and how to read the answer. The blocking and the asyncio
admin clients run the same calls, so they cannot drift apart.

The platform answers ``{"data": <resource>}`` for one resource, ``{"data": {"data": [...], "next_cursor": "...",
"has_more": bool}}`` for a page, ``{"data": [...]}`` for a list that is not paged, and an empty 204 for a delete.
A refusal is ``{"error": "<code>", "error_description": "..."}``.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Generic, TypeVar, cast
from urllib.parse import quote, urlencode

from authfi._protocol.errors import api_error
from authfi._protocol.http import HttpRequest, HttpResponse, Method, Retry
from authfi.errors import APIError, ConfigurationError
from authfi.secret import Secret

__all__ = [
    "ROOT",
    "Call",
    "JoinRequest",
    "OrgDomain",
    "OrgInvitation",
    "OrgMember",
    "Organization",
    "Page",
    "Permission",
    "Role",
    "RoleBinding",
    "User",
    "compact",
    "cursor_of",
    "day",
    "limit_of",
    "listing",
    "many",
    "nothing",
    "paged",
    "required_text",
    "segment",
    "sequence_of_ids",
    "valid_id",
    "valid_subject",
]

#: Where the admin routes are served, on the project's API origin.
ROOT = "/apis/regional/v1"

T = TypeVar("T")


# --- reading the answer ----------------------------------------------------------------------------------


def _invalid(what: str, why: str) -> APIError:
    return APIError("invalid_response", f"{what}: {why}")


def _object(value: object, what: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid(what, "expected an object")
    return cast(dict[str, Any], value)


def _required(data: Mapping[str, Any], key: str, what: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise _invalid(what, f"no {key}")
    return value


def _text(data: Mapping[str, Any], key: str) -> str | None:
    value = data.get(key)
    return value if isinstance(value, str) and value else None


def _flag(data: Mapping[str, Any], key: str) -> bool:
    return data.get(key) is True


def _when(data: Mapping[str, Any], key: str) -> datetime | None:
    value = data.get(key)
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _bag(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = data.get(key)
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


# --- models ----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    """One page of a list. Pass ``next_cursor`` as ``cursor=`` for the next page; ``has_more`` says if there is one."""

    items: tuple[T, ...]
    next_cursor: str | None
    has_more: bool

    def __iter__(self) -> Iterator[T]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)


@dataclass(frozen=True, slots=True)
class Organization:
    """A business customer of the project. ``attributes`` are its business details."""

    id: str
    slug: str
    display_name: str
    attributes: Mapping[str, Any]
    created_at: datetime | None
    updated_at: datetime | None

    @classmethod
    def from_wire(cls, value: object) -> Organization:
        """Read the platform's organization."""
        d = _object(value, "organization")
        return cls(
            id=_required(d, "id", "organization"),
            slug=_required(d, "slug", "organization"),
            display_name=_text(d, "display_name") or "",
            attributes=_bag(d, "attributes"),
            created_at=_when(d, "created_at"),
            updated_at=_when(d, "updated_at"),
        )


@dataclass(frozen=True, slots=True)
class OrgMember:
    """A user in an organization. ``role`` is the role they hold in the organization, not a policy role."""

    user_id: str
    role: str
    created_at: datetime | None

    @classmethod
    def from_wire(cls, value: object) -> OrgMember:
        """Read the platform's member."""
        d = _object(value, "organization member")
        return cls(_required(d, "user_id", "organization member"), _text(d, "role") or "member", _when(d, "created_at"))


@dataclass(frozen=True, slots=True)
class OrgInvitation:
    """A pending invitation. ``token`` is what the invitee presents to accept: a secret, to email and never to log."""

    id: str
    org_id: str
    email: str
    role: str
    token: Secret
    status: str
    user_id: str | None
    expires_at: datetime | None
    created_at: datetime | None

    @classmethod
    def from_wire(cls, value: object) -> OrgInvitation:
        """Read the platform's invitation."""
        d = _object(value, "organization invitation")
        return cls(
            id=_required(d, "id", "organization invitation"),
            org_id=_required(d, "org_id", "organization invitation"),
            email=_required(d, "email", "organization invitation"),
            role=_text(d, "role") or "member",
            token=Secret(_text(d, "token") or ""),
            status=_text(d, "status") or "",
            user_id=_text(d, "user_id"),
            expires_at=_when(d, "expires_at"),
            created_at=_when(d, "created_at"),
        )


@dataclass(frozen=True, slots=True)
class JoinRequest:
    """A person's ask to join an organization."""

    id: str
    org_id: str
    org_name: str
    user_id: str
    email: str
    display_name: str
    status: str
    created_at: datetime | None
    decided_at: datetime | None
    decided_by: str | None

    @classmethod
    def from_wire(cls, value: object) -> JoinRequest:
        """Read the platform's join request."""
        d = _object(value, "join request")
        return cls(
            id=_required(d, "id", "join request"),
            org_id=_required(d, "org_id", "join request"),
            org_name=_text(d, "org_name") or "",
            user_id=_required(d, "user_id", "join request"),
            email=_text(d, "email") or "",
            display_name=_text(d, "display_name") or "",
            status=_text(d, "status") or "",
            created_at=_when(d, "created_at"),
            decided_at=_when(d, "decided_at"),
            decided_by=_text(d, "decided_by"),
        )


@dataclass(frozen=True, slots=True)
class OrgDomain:
    """An email domain an organization owns."""

    domain: str
    claimed_via: str
    claimed_by: str
    created_at: datetime | None

    @classmethod
    def from_wire(cls, value: object) -> OrgDomain:
        """Read the platform's organization domain."""
        d = _object(value, "organization domain")
        return cls(
            _required(d, "domain", "organization domain"),
            _text(d, "claimed_via") or "",
            _text(d, "claimed_by") or "",
            _when(d, "created_at"),
        )


#: The optional text fields of a user, in the order the platform documents them.
USER_TEXT_FIELDS = (
    "given_name",
    "family_name",
    "middle_name",
    "nickname",
    "honorific_prefix",
    "honorific_suffix",
    "picture_url",
    "phone_number",
    "profile_url",
    "title",
    "user_type",
    "preferred_language",
    "locale",
    "timezone",
    "street_address",
    "locality",
    "region",
    "postal_code",
    "country",
    "employee_id",
    "cost_center",
    "organization",
    "division",
    "department",
    "hire_date",
    "manager_id",
    "birthdate",
    "gender",
)


@dataclass(frozen=True, slots=True)
class User:
    """A person in the project's directory. A field the platform has no value for is ``None``."""

    id: str
    email: str
    display_name: str
    status: str
    email_verified: bool
    phone_verified: bool
    subject_id: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    middle_name: str | None = None
    nickname: str | None = None
    honorific_prefix: str | None = None
    honorific_suffix: str | None = None
    picture_url: str | None = None
    phone_number: str | None = None
    profile_url: str | None = None
    title: str | None = None
    user_type: str | None = None
    preferred_language: str | None = None
    locale: str | None = None
    timezone: str | None = None
    street_address: str | None = None
    locality: str | None = None
    region: str | None = None
    postal_code: str | None = None
    country: str | None = None
    employee_id: str | None = None
    cost_center: str | None = None
    organization: str | None = None
    division: str | None = None
    department: str | None = None
    hire_date: str | None = None
    manager_id: str | None = None
    birthdate: str | None = None
    gender: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict[str, Any])
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_login_at: datetime | None = None

    @classmethod
    def from_wire(cls, value: object) -> User:
        """Read the platform's user."""
        d = _object(value, "user")
        optional = {name: _text(d, name) for name in USER_TEXT_FIELDS}
        return cls(
            id=_required(d, "id", "user"),
            email=_required(d, "email", "user"),
            display_name=_text(d, "display_name") or "",
            status=_text(d, "status") or "",
            email_verified=_flag(d, "email_verified"),
            phone_verified=_flag(d, "phone_verified"),
            subject_id=_text(d, "subject_id"),
            attributes=_bag(d, "attributes"),
            created_at=_when(d, "created_at"),
            updated_at=_when(d, "updated_at"),
            last_login_at=_when(d, "last_login_at"),
            **optional,
        )


@dataclass(frozen=True, slots=True)
class Role:
    """A policy role. ``system`` roles are the platform's own."""

    id: str
    name: str
    description: str
    system: bool
    created_at: datetime | None

    @classmethod
    def from_wire(cls, value: object) -> Role:
        """Read the platform's role."""
        d = _object(value, "role")
        return cls(
            _required(d, "id", "role"),
            _required(d, "name", "role"),
            _text(d, "description") or "",
            _flag(d, "system"),
            _when(d, "created_at"),
        )


@dataclass(frozen=True, slots=True)
class Permission:
    """A permission, ``action:resource`` (``manage:users``)."""

    id: str
    action: str
    resource: str
    description: str
    created_at: datetime | None

    @property
    def key(self) -> str:
        """``action:resource``, the form tokens carry."""
        return f"{self.action}:{self.resource}"

    @classmethod
    def from_wire(cls, value: object) -> Permission:
        """Read the platform's permission."""
        d = _object(value, "permission")
        return cls(
            _required(d, "id", "permission"),
            _required(d, "action", "permission"),
            _required(d, "resource", "permission"),
            _text(d, "description") or "",
            _when(d, "created_at"),
        )


@dataclass(frozen=True, slots=True)
class RoleBinding:
    """A role held by a subject (``user:``, ``app:``, ``agent:`` or ``group:``), until ``expires_at`` when set."""

    id: str
    subject: str
    role_id: str
    role_name: str | None
    role_system: bool
    expires_at: datetime | None
    created_at: datetime | None

    @classmethod
    def from_wire(cls, value: object) -> RoleBinding:
        """Read the platform's binding."""
        d = _object(value, "role binding")
        return cls(
            _required(d, "id", "role binding"),
            _required(d, "subject", "role binding"),
            _required(d, "role_id", "role binding"),
            _text(d, "role_name"),
            _flag(d, "role_system"),
            _when(d, "expires_at"),
            _when(d, "created_at"),
        )


# --- checking what goes into a path or a body --------------------------------------------------------------


def valid_id(value: object, what: str) -> str:
    """``value`` as the canonical id the platform uses (a UUID), or a :class:`ConfigurationError`.

    An id goes into a URL path: anything but a UUID (``..``, ``/``, ``?``) is refused here.
    """
    if not isinstance(value, str):
        raise ConfigurationError(f"{what} is the id string the platform gave, not {type(value).__name__}")
    try:
        return str(uuid.UUID(value.strip()))
    except ValueError:
        raise ConfigurationError(f"{what} {value!r} is not an id (a UUID)") from None


_SUBJECT_KINDS = ("user", "app", "agent", "group")


def valid_subject(value: object) -> str:
    """``value`` as a typed subject, ``user:<id>``, ``app:<id>``, ``agent:<id>`` or ``group:<id>``."""
    if not isinstance(value, str):
        raise ConfigurationError("subject is a string such as 'user:<id>'")
    kind, _, rest = value.partition(":")
    if kind not in _SUBJECT_KINDS or not rest.strip() or "/" in rest or "?" in rest or "#" in rest or ".." in rest:
        raise ConfigurationError(f"subject {value!r} must be user:<id>, app:<id>, agent:<id> or group:<id>")
    return value


def limit_of(limit: int | None) -> int | None:
    """``limit`` if it is a positive whole number."""
    if limit is None:
        return None
    value = cast(object, limit)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigurationError("limit is a positive whole number")
    return limit


def cursor_of(cursor: str | None) -> str | None:
    """``cursor`` as given by a previous page's ``next_cursor``."""
    if cursor is None:
        return None
    value = cast(object, cursor)
    if not isinstance(value, str) or not value:
        raise ConfigurationError("cursor is the next_cursor of the previous page")
    return cursor


def required_text(value: object, what: str) -> str:
    """``value`` as a non-blank string."""
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"{what} is required")
    return value


def day(value: date | str) -> str:
    """A calendar date as ``YYYY-MM-DD``."""
    if isinstance(value, datetime):
        raise ConfigurationError("a date, not a date and time: use .date()")
    if isinstance(value, date):
        return value.isoformat()
    return value


# --- calls -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Call(Generic[T]):
    """One admin request and how to read its answer.

    ``parse`` receives the ``data`` of the answer (``None`` for an empty 204).
    """

    method: Method
    path: str
    what: str
    parse: Callable[[Any], T]
    query: Mapping[str, str] = field(default_factory=dict[str, str])
    body: Mapping[str, Any] | None = None
    ok: tuple[int, ...] = (200,)

    def url(self, origin: str) -> str:
        """The full URL on ``origin``."""
        url = origin.rstrip("/") + self.path
        return url + "?" + urlencode(self.query) if self.query else url

    def request(self, origin: str, authorization: Mapping[str, str]) -> HttpRequest:
        """The request. A read may be sent again; a write may not, once it has been sent."""
        return HttpRequest(
            self.method,
            self.url(origin),
            what=self.what,
            headers={"Accept": "application/json", **authorization},
            json_body=None if self.body is None else dict(self.body),
            retry=Retry.SAFE if self.method == "GET" else Retry.NEVER_AFTER_SEND,
        )

    def read(self, response: HttpResponse) -> T:
        """The result, or the typed refusal."""
        if response.status not in self.ok:
            raise api_error(response, self.what)
        if response.status == 204:
            return self.parse(None)
        body = response.json_object()
        if body is None or "data" not in body:
            raise APIError("invalid_response", f"{self.what} answered no data", status=response.status)
        return self.parse(body["data"])


def segment(value: str) -> str:
    """One path segment, encoded."""
    return quote(value, safe=":")


def nothing(_: object) -> None:
    """A parser for an answer with nothing to read."""
    return None


def many(parser: Callable[[object], T]) -> Callable[[Any], tuple[T, ...]]:
    """A parser for ``data`` that is a list (or a page of one, ignoring the cursor)."""

    def parse(data: object) -> tuple[T, ...]:
        items: object = data
        if isinstance(data, dict):
            items = cast(dict[str, Any], data).get("data")
        if not isinstance(items, list):
            raise _invalid("a list", "expected an array")
        return tuple(parser(item) for item in cast(list[object], items))

    return parse


def paged(parser: Callable[[object], T]) -> Callable[[Any], Page[T]]:
    """A parser for ``data`` that is a page."""

    def parse(data: object) -> Page[T]:
        d = _object(data, "a page")
        items = d.get("data")
        if not isinstance(items, list):
            raise _invalid("a page", "no data array")
        cursor = d.get("next_cursor")
        return Page(
            tuple(parser(item) for item in cast(list[object], items)),
            cursor if isinstance(cursor, str) and cursor else None,
            d.get("has_more") is True,
        )

    return parse


def compact(fields: Mapping[str, Any]) -> dict[str, Any]:
    """``fields`` without the ones that are ``None``: an absent key means "leave it" to the platform."""
    return {k: v for k, v in fields.items() if v is not None}


def listing(
    path: str, what: str, parser: Callable[[Any], T], limit: int | None, cursor: str | None, **extra: str
) -> Call[T]:
    """A GET of a page with the common ``limit`` and ``cursor``."""
    query = dict(extra)
    if (n := limit_of(limit)) is not None:
        query["limit"] = str(n)
    if (c := cursor_of(cursor)) is not None:
        query["cursor"] = c
    return Call("GET", path, what, parser, query=query)


def sequence_of_ids(values: Sequence[str], what: str) -> list[str]:
    """Each of ``values`` as an id."""
    if isinstance(values, str):
        raise ConfigurationError(f"{what} is a list of ids, not one string")
    return [valid_id(v, what) for v in values]
