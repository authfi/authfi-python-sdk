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

The refresh token is kept in the operating system's credential store (Windows Credential Manager, the macOS Keychain,
or a Secret Service such as GNOME Keyring or KWallet) through [`keyring`](https://pypi.org/project/keyring/). With no
usable store, `authfi login` stops before it contacts the project and writes nothing: the SDK never keeps a refresh
token in a file. The only file is `login.json` in your config directory (`%APPDATA%\authfi`, or
`$XDG_CONFIG_HOME/authfi`, or `~/.config/authfi`; `AUTHFI_CONFIG_DIR` changes it), which holds the project origin and
the client id and no secret.

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

## Renewing, and two windows at once

AuthFI rotates the refresh token every time it is used, and presenting one that was already used ends the whole
sign-in. `authfi token --resource llm` therefore renews under a lock shared by every process of your user: two Claude
Code windows asking at the same moment cause one renewal, and the second takes the first's result. The new refresh
token is stored before the lock is released.

## Signing out

```bash
authfi logout
```

revokes the refresh token at the project's revocation endpoint (RFC 7009) and then deletes the stored tokens. If the
project did not confirm the revocation, the tokens are still deleted here and the command exits 1 saying so.
