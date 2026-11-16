# Claude Code as yourself, through the project's `/llm` door

Claude Code can call models through your project's AI gateway (`<project origin>/llm`) as you: your own AuthFI
sign-in, not a shared key and not a personal Anthropic account. You sign in once in a browser; after that Claude Code
asks `authfi token --resource llm` for a fresh access token whenever it needs one.

```bash
pip install "authfi[login] @ git+https://github.com/authfi/authfi-python-sdk@main"
authfi login --issuer https://acme.authfi.io
```

`authfi login` reads the project's discovery document, registers a public client once for this machine (RFC 7591), and
signs you in with the authorization code flow and PKCE (S256). The redirect is `http://127.0.0.1:<a free port>/`
(RFC 8252): a listener that answers one callback, checks `state` and the issuer, and closes. It asks for
`resource=<project origin>/llm` and the scopes `openid llm:invoke offline_access`, so the token is addressed to the
`/llm` door and to nothing else. The address is printed too, for a shell with no browser; `--no-browser` skips opening
one.

## Where the credentials live

The refresh token is kept in the operating system's own credential store, through
[`keyring`](https://pypi.org/project/keyring/): the macOS Keychain, the Windows Credential Locker, or Secret Service or
KWallet on Linux. Nothing else is accepted. A plaintext or encrypted-file backend (`keyrings.alt`), a chain that falls
back to one, and a `PYTHON_KEYRING_BACKEND` or `keyringrc.cfg` that selects one are refused, and `authfi login` stops
before it contacts the project and writes nothing. The error names the fix: uninstall `keyrings.alt` and unset
`PYTHON_KEYRING_BACKEND`.

A value longer than 900 characters is kept in pieces, so a long refresh token fits the Windows Credential Locker's
2560-byte limit. The only file is `login.json` in your config directory (`%APPDATA%uthfi`, or
`$XDG_CONFIG_HOME/authfi`, or `~/.config/authfi`; `AUTHFI_CONFIG_DIR` changes it), which holds the project origin, the
client id and no secret. Each config directory is its own sign-in with its own lock, so two of them never share a
refresh token.

The project origin you type is only where discovery is read. The issuer, and so the `/llm` resource the token is
asked for, is the one the discovery document names, which for a project with a custom domain is the custom domain.

## Claude Code settings

In `~/.claude/settings.json`:

```json
{
  "apiKeyHelper": "authfi token --resource llm",
  "env": {
    "ANTHROPIC_BASE_URL": "https://acme.authfi.io/llm",
    "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "300000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

- `apiKeyHelper` runs the command and uses what it prints. Claude Code sends that value in both `Authorization: Bearer`
  and `x-api-key`. `authfi token --resource llm` prints the access token and a newline and nothing else; on any
  failure it prints the reason on stderr, prints nothing on stdout and exits 1.
- `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` is how long Claude Code keeps the helper's value before running it again (five
  minutes by default). The helper reuses a token that still has at least 400 seconds left (`--min-validity` changes
  that) and otherwise renews it, so a value Claude Code holds for its whole lifetime is still a live token.
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` stops Claude Code's background requests to anywhere but the gateway.

With one project, `--issuer` is optional after the first login: the command uses `AUTHFI_ISSUER`, else the project you
last signed in to.

## A base URL that points elsewhere

The helper runs as a child of the tool and inherits its environment. `authfi token --resource llm` prints a token only
when `ANTHROPIC_BASE_URL` (and `OPENAI_BASE_URL`, `OPENAI_API_BASE`), if set, is `https://` on this project's origin with
the path `/llm` or below it. Anything else (another host, `http`, a port, user info, a query, `..`, `/llmx`) is refused
with the variable's name and nothing on stdout, so settings in a repository cannot steer the token to another server.

What the helper can and cannot see, per tool:

| tool | the helper can see | the helper cannot see |
|---|---|---|
| Claude Code (`apiKeyHelper`) | the process environment, which includes `env` from user and project settings files; so `ANTHROPIC_BASE_URL` | a base URL the tool holds only in memory or takes from a command-line flag |
| a tool that reads its key from a helper and its URL from a config file or flag | the environment only | the URL: the check cannot run, so keep that tool's URL under your own control |

The check is a guard on the environment, not proof of where the tool sends the token.

## Renewing, signing in and out at once

AuthFI rotates the refresh token every time it is used, and presenting one that was already used ends the whole
sign-in. `login`, `logout` and the renewal in `authfi token --resource llm` therefore all take one lock, kept in the
config directory that holds the token: two Claude Code windows asking at the same moment cause one renewal, the second
takes the first's result, and a `logout` waits for a renewal in flight and then revokes the token that renewal stored.

A renewal whose outcome is unknown is never repeated with the old token. If the answer was lost, the server answered
with an error after receiving the request, or the credential store refused the new token, the stored token is dropped
and the command asks you to run `authfi login` again; only a failure that proves the token was not spent (the request
never left, a 503 or a refusal such as `interaction_required`) leaves it for the next try. A client the server no longer
knows (`invalid_client`) is registered again once, and then you are asked to sign in.

## Signing out

```bash
authfi logout
```

revokes the refresh token at the project's revocation endpoint (RFC 7009) and then deletes the stored tokens. If the
project did not confirm the revocation, the tokens are still deleted here and the command exits 1 saying so.
