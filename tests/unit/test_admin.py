"""authfi.admin against a stand-in for the directory and authorization public routes (tests/fake_admin.py).

Whether a machine client's token is granted ``manage:organizations`` or ``manage:users`` on a live project is not
proven here: the stand-in grants what a test says it grants, and the platform's own refusal is replayed.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Iterator
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import httpx
import pytest

from authfi import (
    Admin,
    APIError,
    AsyncAdmin,
    AsyncAuthFI,
    AuthFI,
    ConfigurationError,
    Secret,
    TokenSource,
    TransportError,
)
from tests.fake_admin import ROOT, FakeAdmin, envelope, new_id, refusal
from tests.support import FakeClock, Platform, json_response

ORG = new_id()
USER = new_id()
ROLE = new_id()


class Mint:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.count += 1
        return json_response({"access_token": f"token-{self.count}", "token_type": "Bearer", "expires_in": 900})


class Person(TokenSource):
    """A signed-in person's token, as the platform's sign-in gives it."""

    def __init__(self, project: AuthFI, token: str, user_id: str) -> None:
        self.project = project
        self._token = token
        self.user_id = user_id

    @property
    def send_to(self) -> tuple[str, ...]:
        return self.project.authfi_origins

    def token(self, *, force: bool = False) -> str:
        return self._token


@pytest.fixture
def fake(platform: Platform) -> FakeAdmin:
    platform.route("POST", "/oauth/token", Mint())
    return FakeAdmin(platform)


@pytest.fixture
def project(fake: FakeAdmin, clock: FakeClock) -> Iterator[AuthFI]:
    with AuthFI(fake.platform.issuer, http_client=fake.client(), clock=clock) as af:
        yield af


@pytest.fixture
def admin(project: AuthFI) -> Admin:
    return project.client_credentials("app_backend", Secret("s3cret")).admin


@pytest.fixture
async def aadmin(fake: FakeAdmin, clock: FakeClock) -> AsyncIterator[AsyncAdmin]:
    async with AsyncAuthFI(fake.platform.issuer, http_client=fake.async_client(), clock=clock) as af:
        yield af.client_credentials("app_backend", Secret("s3cret")).admin


def sent(fake: FakeAdmin) -> httpx.Request:
    return fake.requests[-1]


def body_of(request: httpx.Request) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(request.content))


# --- the credential --------------------------------------------------------------------------------------------


def test_the_admin_is_the_credentials_and_cached(project: AuthFI) -> None:
    m2m = project.client_credentials("app_backend", Secret("s"))
    assert isinstance(m2m.admin, Admin)
    assert m2m.admin is m2m.admin


def test_calls_carry_the_credentials_token_on_the_canonical_path(admin: Admin, fake: FakeAdmin) -> None:
    created = admin.organizations.create("acme", "Acme Inc.")
    request = sent(fake)
    assert request.method == "POST"
    assert request.url.path == f"{ROOT}/organizations"
    assert request.url.host == "acme.authfi.io"
    assert request.headers["Authorization"] == "Bearer token-1"
    assert request.headers["Accept"] == "application/json"
    assert body_of(request) == {"slug": "acme", "display_name": "Acme Inc."}
    assert created.slug == "acme"


def test_a_credential_for_another_api_is_not_sent_to_the_admin_api(project: AuthFI) -> None:
    other = project.client_credentials("app", Secret("s"), resource="api_orders", send_to=["https://api.example.com"])
    with pytest.raises(ConfigurationError, match="refusing to send this credential"):
        Admin(other).users.get(USER)


def test_admin_needs_a_blocking_credential_that_names_its_project(project: AuthFI) -> None:
    with pytest.raises(ConfigurationError, match="Admin needs"):
        Admin("a-token")  # type: ignore[arg-type]

    class Bare(TokenSource):
        @property
        def send_to(self) -> tuple[str, ...]:
            return ()

        def token(self, *, force: bool = False) -> str:
            return "t"

    with pytest.raises(ConfigurationError, match="Admin needs"):
        Admin(Bare())
    assert Admin(Bare(), project=project).users is not None


def test_a_person_token_works_too(project: AuthFI, fake: FakeAdmin) -> None:
    fake.callers["person-1"] = USER
    person = Admin(Person(project, "person-1", USER))
    assert person.users.list().items == ()
    assert sent(fake).headers["Authorization"] == "Bearer person-1"


# --- organizations -----------------------------------------------------------------------------------------------


def test_an_organization_lifecycle(admin: Admin, fake: FakeAdmin) -> None:
    org = admin.organizations.create("acme", "Acme Inc.")
    assert (org.display_name, org.attributes, org.created_at) == (
        "Acme Inc.",
        {},
        datetime(2026, 10, 29, 10, tzinfo=UTC),
    )
    assert admin.organizations.get(org.id) == org
    changed = admin.organizations.update(org.id, display_name="Acme Corp", attributes={"size": 10})
    assert (changed.display_name, changed.attributes) == ("Acme Corp", {"size": 10})
    assert body_of(sent(fake)) == {"display_name": "Acme Corp", "attributes": {"size": 10}}
    admin.organizations.update(org.id, display_name="Acme")
    assert body_of(sent(fake)) == {"display_name": "Acme"}
    admin.organizations.delete(org.id)
    assert sent(fake).method == "DELETE"
    with pytest.raises(APIError) as caught:
        admin.organizations.get(org.id)
    assert (caught.value.code, caught.value.status) == ("not_found", 404)


def test_a_taken_slug_is_a_conflict(admin: Admin) -> None:
    admin.organizations.create("acme", "A")
    with pytest.raises(APIError) as caught:
        admin.organizations.create("acme", "B")
    assert (caught.value.code, caught.value.status) == ("conflict", 409)


def test_members(admin: Admin, fake: FakeAdmin) -> None:
    org = admin.organizations.create("acme", "A")
    admin.organizations.add_member(org.id, USER)
    assert body_of(sent(fake)) == {"user_id": USER}
    admin.organizations.add_member(org.id, new_id(), role="admin")
    members = admin.organizations.members(org.id)
    assert sorted(m.role for m in members) == ["admin", "member"]
    admin.organizations.remove_member(org.id, USER)
    assert sent(fake).url.path == f"{ROOT}/organizations/{org.id}/members/{USER}"
    assert [m.role for m in admin.organizations.members(org.id)] == ["admin"]


def test_invitations(admin: Admin, fake: FakeAdmin) -> None:
    org = admin.organizations.create("acme", "A")
    invitation = admin.organizations.invite(org.id, "ada@acme.example", role="admin", invited_by=USER)
    assert body_of(sent(fake)) == {"email": "ada@acme.example", "role": "admin", "invited_by": USER}
    assert (invitation.status, invitation.role, invitation.org_id) == ("pending", "admin", org.id)
    assert isinstance(invitation.token, Secret)
    assert "inv_" not in repr(invitation)
    assert [i.id for i in admin.organizations.invitations(org.id)] == [invitation.id]
    admin.organizations.revoke_invitation(org.id, invitation.id)
    assert admin.organizations.invitations(org.id) == ()
    with pytest.raises(APIError) as caught:
        admin.organizations.revoke_invitation(org.id, invitation.id)
    assert caught.value.status == 404


def test_accepting_an_invitation_is_the_invitees_own_call(project: AuthFI, admin: Admin, fake: FakeAdmin) -> None:
    org = admin.organizations.create("acme", "A")
    invitation = admin.organizations.invite(org.id, "ada@acme.example")
    fake.callers["person-ada"] = USER
    ada = Admin(Person(project, "person-ada", USER))
    accepted = ada.organizations.accept_invitation(invitation.token, USER)
    assert accepted.status == "accepted"
    assert body_of(sent(fake)) == {"token": invitation.token.reveal(), "user_id": USER}
    assert ada.organizations.accept_invitation(invitation.token.reveal(), USER).id == invitation.id
    with pytest.raises(APIError) as caught:  # a machine client cannot accept for someone else
        admin.organizations.accept_invitation(invitation.token, USER)
    assert caught.value.status == 403


def test_join_requests_belong_to_people_not_machines(project: AuthFI, admin: Admin, fake: FakeAdmin) -> None:
    org = admin.organizations.create("acme", "A")
    request_id = new_id()
    fake.join_requests[request_id] = {
        "id": request_id,
        "org_id": org.id,
        "org_name": "Acme",
        "user_id": USER,
        "email": "e@x.example",
        "display_name": "E",
        "status": "pending",
        "created_at": "2026-10-29T10:00:00Z",
        "decided_at": None,
        "decided_by": "",
    }
    for call in (
        lambda: admin.organizations.join_requests(org.id),
        lambda: admin.organizations.approve_join_request(org.id, request_id),
        lambda: admin.organizations.decline_join_request(org.id, request_id),
    ):
        with pytest.raises(APIError) as caught:
            call()
        assert (caught.value.code, caught.value.status) == ("forbidden", 403)

    fake.callers["person-owner"] = new_id()
    owner = Admin(Person(project, "person-owner", fake.callers["person-owner"]))
    pending = owner.organizations.join_requests(org.id, status="pending")
    assert [j.id for j in pending] == [request_id]
    assert sent(fake).url.params["status"] == "pending"
    assert owner.organizations.approve_join_request(org.id, request_id).status == "approved"
    assert owner.organizations.decline_join_request(org.id, request_id).status == "declined"
    assert owner.organizations.join_requests(org.id, status="pending") == ()
    assert [j.status for j in owner.organizations.my_join_requests()] == ["declined"]
    owner.organizations.withdraw_join_request(request_id)
    assert owner.organizations.my_join_requests() == ()


def test_domains(admin: Admin) -> None:
    org = admin.organizations.create("acme", "A")
    (domain,) = admin.organizations.domains(org.id)
    assert (domain.domain, domain.claimed_via) == ("acme.example", "signup")


# --- users -------------------------------------------------------------------------------------------------------


def test_a_user_lifecycle(admin: Admin, fake: FakeAdmin) -> None:
    user = admin.users.create(
        "ada@acme.example",
        "Ada",
        given_name="Ada",
        hire_date=date(2026, 1, 5),
        attributes={"team": "x"},
        subject_id="ext-1",
    )
    assert body_of(sent(fake)) == {
        "email": "ada@acme.example",
        "display_name": "Ada",
        "given_name": "Ada",
        "hire_date": "2026-01-05",
        "attributes": {"team": "x"},
        "subject_id": "ext-1",
    }
    assert (user.status, user.email_verified, user.given_name, user.family_name) == ("active", False, "Ada", None)
    assert admin.users.get(user.id).hire_date == "2026-01-05"
    blocked = admin.users.update(user.id, status="suspended")
    assert blocked.status == "suspended"
    assert body_of(sent(fake)) == {"status": "suspended"}
    admin.users.update(user.id, display_name="Ada L.", clear_manager=True)
    assert body_of(sent(fake)) == {"display_name": "Ada L.", "clear_manager": True}
    admin.users.delete(user.id)
    with pytest.raises(APIError) as caught:
        admin.users.get(user.id)
    assert caught.value.status == 404


def test_the_platform_refusing_a_value_is_a_typed_error(admin: Admin, fake: FakeAdmin) -> None:
    fake.fail_next = [refusal(400, "invalid_body", "email is malformed")]
    with pytest.raises(APIError) as caught:
        admin.users.create("a@b", "A")
    assert (caught.value.code, caught.value.status, caught.value.description) == (
        "invalid_body",
        400,
        "email is malformed",
    )


@pytest.mark.parametrize("field", ["email_verified", "phone_verified"])
def test_a_verified_claim_is_refused_before_it_is_sent(admin: Admin, fake: FakeAdmin, field: str) -> None:
    with pytest.raises(ConfigurationError, match="set by AuthFI"):
        admin.users.create("a@x.example", "A", **{field: True})  # type: ignore[arg-type]
    user = admin.users.create("a@x.example", "A")
    with pytest.raises(ConfigurationError, match="set by AuthFI"):
        admin.users.update(user.id, **{field: True})  # type: ignore[arg-type]
    assert len(fake.seen("POST", "/users")) == 1


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({}, "at least one field"),
        ({"shoe_size": 9}, "not a field"),
        ({"status": "gone"}, "status is one of"),
        ({"attributes": "x"}, "attributes is an object"),
        ({"manager_id": "../x"}, "not an id"),
        ({"manager_id": ROLE, "clear_manager": True}, "contradict"),
    ],
)
def test_bad_changes_are_refused_locally(admin: Admin, fake: FakeAdmin, changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ConfigurationError, match=match):
        admin.users.update(USER, **changes)
    assert fake.requests == []


def test_a_create_must_name_an_email_and_a_display_name(admin: Admin) -> None:
    with pytest.raises(ConfigurationError, match="email"):
        admin.users.create("ada", "Ada")
    with pytest.raises(ConfigurationError, match="display_name"):
        admin.users.create("ada@x.example", " ")
    with pytest.raises(ConfigurationError, match="date and time"):
        admin.users.create("ada@x.example", "Ada", birthdate=datetime(2000, 1, 1, tzinfo=UTC))


def test_users_page_and_iterate(admin: Admin, fake: FakeAdmin) -> None:
    for n in range(5):
        admin.users.create(f"u{n}@acme.example", f"U{n}")
    first = admin.users.list(limit=2)
    assert (len(first), first.has_more, first.next_cursor) == (2, True, "2")
    assert [u.email for u in first] == ["u0@acme.example", "u1@acme.example"]
    assert sent(fake).url.params["limit"] == "2"
    last = admin.users.list(limit=2, cursor="4")
    assert (len(last), last.has_more, last.next_cursor) == (1, False, None)
    assert [u.email for u in admin.users.iterate(limit=2)] == [f"u{n}@acme.example" for n in range(5)]
    assert [u.email for u in admin.users.iterate(search="u3")] == ["u3@acme.example"]
    assert sent(fake).url.params["q"] == "u3"


def test_a_cursor_that_does_not_move_stops_the_walk(admin: Admin, fake: FakeAdmin) -> None:
    stuck = envelope({"data": [], "next_cursor": "same", "has_more": True})
    fake.fail_next = [stuck, stuck]
    with pytest.raises(APIError, match="no new cursor"):
        list(admin.users.iterate())
    fake.fail_next = [envelope({"data": [], "has_more": True})]
    with pytest.raises(APIError, match="no new cursor"):
        list(admin.users.iterate())


@pytest.mark.parametrize(("limit", "cursor"), [(0, None), (-1, None), (True, None), ("5", None), (None, "")])
def test_a_bad_page_request_is_refused_locally(admin: Admin, fake: FakeAdmin, limit: Any, cursor: Any) -> None:
    with pytest.raises(ConfigurationError):
        admin.users.list(limit=limit, cursor=cursor)
    assert fake.requests == []


# --- roles, permissions, bindings -----------------------------------------------------------------------------------


def test_roles_permissions_and_bindings(admin: Admin, fake: FakeAdmin) -> None:
    role = admin.roles.create("billing-admin", description="Billing")
    assert (role.name, role.description, role.system) == ("billing-admin", "Billing", False)
    read = admin.permissions.create("read", "invoices", description="Read invoices")
    write = admin.permissions.create("manage", "invoices")
    assert read.key == "read:invoices"
    admin.roles.set_permissions(role.id, [read.id, write.id])
    assert body_of(sent(fake)) == {"permission_ids": [read.id, write.id]}
    assert sorted(p.key for p in admin.roles.permissions(role.id)) == ["manage:invoices", "read:invoices"]
    admin.roles.set_permissions(role.id, [])
    assert admin.roles.permissions(role.id) == ()

    subject = f"user:{USER}"
    until = datetime(2027, 1, 1, tzinfo=UTC)
    binding = admin.bindings.assign(subject, role.id, expires_at=until)
    assert body_of(sent(fake)) == {"subject": subject, "role_id": role.id, "expires_at": until.isoformat()}
    assert (binding.role_name, binding.expires_at) == ("billing-admin", until)
    assert [b.id for b in admin.bindings.list(subject=subject)] == [binding.id]
    assert sent(fake).url.params["subject"] == subject
    assert [b.subject for b in admin.bindings.list(role_id=role.id)] == [subject]
    assert [b.id for b in admin.bindings.iterate(subject=subject, limit=1)] == [binding.id]
    admin.bindings.revoke(subject, role.id)
    assert sent(fake).url.path == f"{ROOT}/bindings/user:{USER}/{role.id}"
    assert list(admin.bindings.iterate(subject=subject)) == []
    with pytest.raises(APIError) as caught:
        admin.bindings.revoke(subject, role.id)
    assert caught.value.status == 404


def test_roles_and_permissions_iterate(admin: Admin) -> None:
    for n in range(3):
        admin.roles.create(f"r{n}")
        admin.permissions.create("read", f"thing{n}")
    assert [r.name for r in admin.roles.iterate(limit=2)] == ["r0", "r1", "r2"]
    assert [p.key for p in admin.permissions.iterate(limit=2)] == [f"read:thing{n}" for n in range(3)]
    assert len(admin.roles.list()) == 2


def test_a_duplicate_role_is_a_conflict(admin: Admin) -> None:
    admin.roles.create("r")
    with pytest.raises(APIError) as caught:
        admin.roles.create("r")
    assert caught.value.status == 409


def test_a_permission_outside_the_tenant_is_refused_by_the_platform(admin: Admin) -> None:
    role = admin.roles.create("r")
    with pytest.raises(APIError) as caught:
        admin.roles.set_permissions(role.id, [new_id()])
    assert (caught.value.code, caught.value.status) == ("invalid_body", 400)


@pytest.mark.parametrize(
    "subject", ["", "alice", "federated:x", "user:", "user:a/b", "user:../x", "user:a?b", "user:a#b", 5, None]
)
def test_a_subject_is_typed_and_cannot_reach_another_path(admin: Admin, fake: FakeAdmin, subject: Any) -> None:
    with pytest.raises(ConfigurationError):
        admin.bindings.revoke(subject, ROLE)
    with pytest.raises(ConfigurationError):
        admin.bindings.assign(subject, ROLE)
    assert fake.requests == []


def test_binding_arguments(admin: Admin, fake: FakeAdmin) -> None:
    with pytest.raises(ConfigurationError, match="not both or neither"):
        admin.bindings.list()
    with pytest.raises(ConfigurationError, match="not both or neither"):
        admin.bindings.list(subject=f"user:{USER}", role_id=ROLE)
    with pytest.raises(ConfigurationError, match="time zone"):
        admin.bindings.assign(f"user:{USER}", ROLE, expires_at=datetime.now() + timedelta(days=1))
    for bad in ("not-a-uuid", "../x", ROLE + "/x", 5):
        with pytest.raises(ConfigurationError):
            admin.bindings.revoke(f"user:{USER}", bad)  # type: ignore[arg-type]
        with pytest.raises(ConfigurationError):
            admin.roles.permissions(bad)  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="list of ids"):
        admin.roles.set_permissions(ROLE, ROLE)
    assert fake.requests == []


def case(call: Callable[[Admin], object]) -> Callable[[Admin], object]:
    return call


MISTAKES = [
    case(lambda a: a.organizations.get("not-a-uuid")),
    case(lambda a: a.organizations.get("../users")),
    case(lambda a: a.organizations.delete(ORG + "/members")),
    case(lambda a: a.organizations.add_member(ORG, "x")),
    case(lambda a: a.organizations.remove_member("x", USER)),
    case(lambda a: a.organizations.revoke_invitation(ORG, "x")),
    case(lambda a: a.organizations.invite(ORG, "no-at-sign")),
    case(lambda a: a.organizations.invite(ORG, "a@x.example", invited_by="x")),
    case(lambda a: a.organizations.update(ORG)),
    case(lambda a: a.organizations.update(ORG, display_name=" ")),
    case(lambda a: a.organizations.update(ORG, attributes=cast(Any, "x"))),
    case(lambda a: a.organizations.create(" ", "A")),
    case(lambda a: a.organizations.create("a", "")),
    case(lambda a: a.organizations.accept_invitation("", USER)),
    case(lambda a: a.organizations.accept_invitation("t", "x")),
    case(lambda a: a.organizations.join_requests(ORG, status="")),
    case(lambda a: a.organizations.withdraw_join_request("x")),
    case(lambda a: a.organizations.approve_join_request(ORG, "x")),
    case(lambda a: a.users.get("x")),
    case(lambda a: a.users.delete("x")),
    case(lambda a: a.users.list(search=" ")),
    case(lambda a: a.roles.create(" ")),
    case(lambda a: a.permissions.create("", "x")),
    case(lambda a: a.permissions.create("read", " ")),
]


@pytest.mark.parametrize("call", MISTAKES)
def test_a_mistake_in_an_argument_never_reaches_the_network(
    admin: Admin, fake: FakeAdmin, call: Callable[[Admin], object]
) -> None:
    with pytest.raises(ConfigurationError):
        call(admin)
    assert fake.requests == []


# --- refusals ---------------------------------------------------------------------------------------------------


def test_a_missing_permission_is_a_403_with_the_platforms_code(admin: Admin, fake: FakeAdmin) -> None:
    fake.default_grants = {"read:users"}
    assert admin.users.list().items == ()
    with pytest.raises(APIError) as caught:
        admin.users.create("a@x.example", "A")
    error = caught.value
    assert (error.code, error.status) == ("forbidden", 403)
    assert error.description == "manage:users is required"
    assert error.request_id is not None


def test_a_plan_without_the_feature_is_a_402_that_names_it(admin: Admin, fake: FakeAdmin) -> None:
    fake.plan_blocks = {"organizations"}
    with pytest.raises(APIError) as caught:
        admin.organizations.list()
    assert (caught.value.code, caught.value.status) == ("feature_not_in_plan", 402)


def test_an_unknown_token_is_a_401(project: AuthFI) -> None:
    with pytest.raises(APIError) as caught:
        Admin(Person(project, "stranger", USER)).users.list()
    assert (caught.value.code, caught.value.status) == ("unauthorized", 401)


def test_an_answer_with_no_error_code_is_a_transport_error(admin: Admin, fake: FakeAdmin) -> None:
    fake.fail_next = [httpx.Response(502, text="bad gateway")]
    with pytest.raises(TransportError) as caught:
        admin.users.delete(USER)
    assert caught.value.status == 502


@pytest.mark.parametrize(
    "answer",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"nothing": 1}),
        httpx.Response(200, json={"data": "x"}),
        httpx.Response(200, json={"data": {"id": ""}}),
    ],
)
def test_an_answer_of_the_wrong_shape_is_an_invalid_response(
    admin: Admin, fake: FakeAdmin, answer: httpx.Response
) -> None:
    fake.fail_next = [answer]
    with pytest.raises(APIError) as caught:
        admin.users.get(USER)
    assert caught.value.code == "invalid_response"


def test_a_list_of_the_wrong_shape(admin: Admin, fake: FakeAdmin) -> None:
    for answer in (
        httpx.Response(200, json={"data": "x"}),
        httpx.Response(200, json={"data": [1]}),
        httpx.Response(200, json={"data": {"data": 5}}),
    ):
        fake.fail_next = [answer]
        with pytest.raises(APIError) as caught:
            admin.organizations.members(ORG)
        assert caught.value.code == "invalid_response"


def test_a_write_is_not_sent_twice_but_a_read_is_retried(admin: Admin, fake: FakeAdmin) -> None:
    fake.fail_next = [refusal(503, "temporarily_unavailable")]
    with pytest.raises(APIError):
        admin.users.create("a@x.example", "A")
    assert len(fake.seen("POST", "/users")) == 1
    fake.requests.clear()
    fake.fail_next = [refusal(503, "temporarily_unavailable")]
    assert admin.users.list().items == ()
    assert len(fake.seen("GET", "/users")) == 2


def test_a_server_that_leaves_out_optional_fields_still_parses(admin: Admin, fake: FakeAdmin) -> None:
    fake.fail_next = [envelope({"id": USER, "email": "a@x.example", "created_at": "garbage", "attributes": 5})]
    user = admin.users.get(USER)
    assert (user.display_name, user.status, user.created_at, user.attributes, user.email_verified) == (
        "",
        "",
        None,
        {},
        False,
    )


# --- asyncio ----------------------------------------------------------------------------------------------------


async def test_the_async_admin_runs_the_same_calls(aadmin: AsyncAdmin) -> None:
    org = await aadmin.organizations.create("acme", "Acme")
    assert (await aadmin.organizations.get(org.id)).slug == "acme"
    assert (await aadmin.organizations.update(org.id, display_name="B")).display_name == "B"
    await aadmin.organizations.add_member(org.id, USER, role="admin")
    assert [m.role for m in await aadmin.organizations.members(org.id)] == ["admin"]
    await aadmin.organizations.remove_member(org.id, USER)
    invitation = await aadmin.organizations.invite(org.id, "ada@acme.example")
    assert [i.id for i in await aadmin.organizations.invitations(org.id)] == [invitation.id]
    await aadmin.organizations.revoke_invitation(org.id, invitation.id)
    assert len(await aadmin.organizations.domains(org.id)) == 1
    assert [o.id async for o in aadmin.organizations.iterate(limit=1)] == [org.id]
    assert len(await aadmin.organizations.list()) == 1
    await aadmin.organizations.delete(org.id)

    user = await aadmin.users.create("a@x.example", "A", locale="en")
    assert (await aadmin.users.get(user.id)).locale == "en"
    assert (await aadmin.users.update(user.id, status="suspended")).status == "suspended"
    assert [u.id async for u in aadmin.users.iterate(search="a@", limit=1)] == [user.id]
    assert len(await aadmin.users.list(search="a@")) == 1
    await aadmin.users.delete(user.id)

    role = await aadmin.roles.create("r")
    perm = await aadmin.permissions.create("read", "x")
    await aadmin.roles.set_permissions(role.id, [perm.id])
    assert [p.key for p in await aadmin.roles.permissions(role.id)] == ["read:x"]
    assert [r.id async for r in aadmin.roles.iterate()] == [role.id]
    assert [p.id async for p in aadmin.permissions.iterate()] == [perm.id]
    assert len(await aadmin.roles.list()) == 1
    assert len(await aadmin.permissions.list()) == 1
    binding = await aadmin.bindings.assign(f"user:{USER}", role.id)
    assert [b.id for b in await aadmin.bindings.list(subject=f"user:{USER}")] == [binding.id]
    assert [b.id async for b in aadmin.bindings.iterate(role_id=role.id)] == [binding.id]
    await aadmin.bindings.revoke(f"user:{USER}", role.id)


async def test_the_async_admin_refuses_like_the_blocking_one(aadmin: AsyncAdmin) -> None:
    with pytest.raises(APIError) as caught:
        await aadmin.users.get(USER)
    assert caught.value.status == 404
    with pytest.raises(ConfigurationError, match="set by AuthFI"):
        await aadmin.users.create("a@x.example", "A", email_verified=True)  # type: ignore[call-arg]
    with pytest.raises(APIError) as forbidden:
        await aadmin.organizations.join_requests(ORG)
    assert forbidden.value.status == 403


async def test_the_async_admin_needs_an_asyncio_credential(aadmin: AsyncAdmin) -> None:
    with pytest.raises(ConfigurationError, match="AsyncAdmin needs"):
        AsyncAdmin("a-token")  # type: ignore[arg-type]


async def test_the_async_admin_pages_and_the_join_calls(aadmin: AsyncAdmin) -> None:
    for n in range(3):
        await aadmin.roles.create(f"r{n}")
    assert [r.name async for r in aadmin.roles.iterate(limit=2)] == ["r0", "r1", "r2"]
    with pytest.raises(APIError) as caught:
        await aadmin.organizations.approve_join_request(ORG, new_id())
    assert caught.value.status == 403
    with pytest.raises(APIError):
        await aadmin.organizations.decline_join_request(ORG, new_id())
    with pytest.raises(APIError):
        await aadmin.organizations.accept_invitation("t", USER)
    assert await aadmin.organizations.my_join_requests() == ()
    await aadmin.organizations.withdraw_join_request(new_id())


def test_every_listing_follows_its_cursors(admin: Admin) -> None:
    for n in range(3):
        admin.organizations.create(f"org{n}", f"Org {n}")
        admin.permissions.create("read", f"p{n}")
        role = admin.roles.create(f"r{n}")
        admin.bindings.assign(f"user:{USER}", role.id)
    assert len(list(admin.organizations.iterate(limit=2))) == 3
    assert len(list(admin.permissions.iterate(limit=2))) == 3
    assert len(list(admin.bindings.iterate(subject=f"user:{USER}", limit=2))) == 3


def test_a_date_may_be_given_as_text_and_a_page_must_be_a_page(admin: Admin, fake: FakeAdmin) -> None:
    admin.users.create("a@x.example", "A", hire_date="2026-01-05")
    assert body_of(sent(fake))["hire_date"] == "2026-01-05"
    fake.fail_next = [envelope({"data": "x"})]
    with pytest.raises(APIError, match="no data array"):
        admin.users.list()


async def test_the_async_admin_is_cached_and_every_listing_follows_its_cursors(
    fake: FakeAdmin, clock: FakeClock
) -> None:
    async with AsyncAuthFI(fake.platform.issuer, http_client=fake.async_client(), clock=clock) as af:
        m2m = af.client_credentials("app_backend", Secret("s3cret"))
        assert m2m.admin is m2m.admin
        admin = m2m.admin
        for n in range(3):
            await admin.organizations.create(f"org{n}", f"Org {n}")
            await admin.users.create(f"u{n}@x.example", f"U{n}")
            await admin.permissions.create("read", f"p{n}")
            role = await admin.roles.create(f"r{n}")
            await admin.bindings.assign(f"user:{USER}", role.id)
        assert len([o async for o in admin.organizations.iterate(limit=2)]) == 3
        assert len([u async for u in admin.users.iterate(limit=2)]) == 3
        assert len([p async for p in admin.permissions.iterate(limit=2)]) == 3
        assert len([r async for r in admin.roles.iterate(limit=2)]) == 3
        assert len([b async for b in admin.bindings.iterate(subject=f"user:{USER}", limit=2)]) == 3
