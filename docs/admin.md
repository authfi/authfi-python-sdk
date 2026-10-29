# Organizations, users and roles from your backend

`credential.admin` calls the project's public admin API with the credential's own token. It covers
organizations (members, invitations, join requests), users, policy roles, permissions and role bindings.

```python
from authfi import AuthFI, Secret

af = AuthFI.from_env()
m2m = af.client_credentials("app_backend", Secret(os.environ["AUTHFI_CLIENT_SECRET"]))
admin = m2m.admin

org = admin.organizations.create("acme", "Acme Inc.")
invitation = admin.organizations.invite(org.id, "ada@acme.example", role="admin")
send_email(to=invitation.email, link=f"https://app.example.com/join?t={invitation.token.reveal()}")

user = admin.users.create("grace@acme.example", "Grace Hopper", given_name="Grace")
admin.organizations.add_member(org.id, user.id, role="member")

role = admin.roles.create("billing-admin")
admin.bindings.assign(f"user:{user.id}", role.id)

for u in admin.users.iterate(search="acme.example"):
    print(u.email, u.status)
```

`AsyncAuthFI(...).client_credentials(...).admin` is the same API with `await` and `async for`. A signed-in
person's token works too: `Admin(person_token_source)`.

## What the token needs

Each call needs the permission of its route, held through a role bound to the application (or the person):

| Calls | Permission |
|---|---|
| read organizations, users | `read:organizations`, `read:users` |
| create, change, delete organizations, members, invitations, users | `manage:organizations`, `manage:users` |
| read roles, permissions, bindings | `read:policy` |
| create roles and permissions, set what a role grants, assign and revoke | `manage:roles` |

Organizations need a plan that includes them. A call the token may not make raises `APIError` with
`status == 403`, one the plan does not include with the platform's own code (for example `402`). Branch on
`error.code` and `error.status`.

## Things to know

- **Ids** are the UUID strings the platform returns. An id that is not a UUID is refused before anything is sent.
- **Pages.** `list()` returns a `Page` (`items`, `next_cursor`, `has_more`); `iterate()` follows the cursors.
- **Organization roles are not policy roles.** The `role` on a member or an invitation belongs to that
  organization. Policy roles and their bindings (`roles`, `bindings`) apply to the whole project.
- **Verified email and phone.** `email_verified` and `phone_verified` are set when a person proves them by
  signing in; `users.create` and `users.update` cannot claim them and refuse the argument.
- **Blocking a user** is `users.update(id, status="suspended")`; `"active"` lifts it.
- **Invitations.** `OrgInvitation.token` is a `Secret`: it is the invitee's proof to accept, so email it and do
  not log it. The invitee accepts with their own token: `Admin(their_source).organizations.accept_invitation(token, their_user_id)`.
- **Join requests.** Listing, approving and declining are for a signed-in owner or admin of the organization;
  `my_join_requests()` and `withdraw_join_request()` are for the person who asked.
- **Writes are not repeated.** A read is retried on a transient failure; a create, change or delete is sent once.
