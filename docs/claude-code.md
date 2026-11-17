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
2560-byte limit. The only file is `login.json` in your config directory (`%APPDATA%\authfi`, or
`$XDG_CONFIG_HOME/authfi`, or `~/.config/authfi`; `AUTHFI_CONFIG_DIR` changes it), which holds the project origin, the
client id and no secret. Each config directory is its own sign-in with its own lock, so two of them never share a
refresh token.

The project origin you type is only where discovery is read. The issuer, and so the `/llm` resource the token is
asked for, is the one the discovery document names, which for a project with a custom domain is the custom domain.

## Claude Code settings

In `~/.claude/settings.json`:

```json
{
  "apiKeyHelper": "/usr/bin/python3 -I -m authfi token --resource llm",
  "env": {
    "ANTHROPIC_BASE_URL": "https://acme.authfi.io/llm",
    "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": "300000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

- The helper must start isolated. Python reads `PYTHONPATH`, `PYTHONUSERBASE` and the user site, and puts the working directory on its path, before any line of this tool runs, so a check made afterwards cannot undo them. `-I` turns those off. It does not stop the dynamic loader, OpenSSL or the shell that starts the helper: `LD_PRELOAD`, `OPENSSL_CONF` and `BASH_ENV` load code before any Python runs (see [The boundary](#the-boundary)). A console script cannot carry the flag, so use the absolute path of the interpreter that has `authfi` installed, then `-I -m authfi`; `authfi login` prints the exact line for this machine. Claude Code runs the line through `/bin/sh` on macOS and Linux and through `cmd` on Windows ([settings reference](https://code.claude.com/docs/en/settings-reference#apikeyhelper)), so on Windows the interpreter is double-quoted for `cmd`: `"C:\\Python\\python.exe" -I -m authfi token --resource llm` (written with doubled backslashes inside the JSON). Run `authfi login` with the interpreter you want Claude Code to use: from a project's virtualenv the line names that virtualenv's Python and its packages, and a Python whose path is empty or relative gets no line (the helper does not search `PATH` for it, which could find a different interpreter). Started any other way, `authfi token --resource llm` prints the fix on stderr and exits 1 without opening the credential store. The credential store is opened by naming the operating system's store (Keychain, Credential Locker, Secret Service or KWallet) directly: `keyringrc.cfg`, its `keyring-path` and `PYTHON_KEYRING_BACKEND` are never read.
- `apiKeyHelper` runs the command and uses what it prints. Claude Code sends that value in both `Authorization: Bearer`
  and `x-api-key`. `authfi token --resource llm` prints the access token and a newline and nothing else; on any
  failure it prints the reason on stderr, prints nothing on stdout and exits 1.
- `CLAUDE_CODE_API_KEY_HELPER_TTL_MS` is how long Claude Code keeps the helper's value before running it again (five
  minutes by default). The helper reuses a token that still has at least 400 seconds left (`--min-validity` changes
  that) and otherwise renews it, so a value Claude Code holds for its whole lifetime is still a live token.
- `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` stops Claude Code's background requests to anywhere but the gateway.

With one project, `--issuer` is optional after the first login: the command uses `AUTHFI_ISSUER`, else the project you
last signed in to.

## What the helper checks before it prints a token

`authfi token --resource llm` runs as a child of the tool and inherits its environment. A repository's
`.claude/settings.json` can put values in that environment through its `env` block, and Claude Code applies them to the
process (and so to its helper) once the folder is trusted ([settings](https://code.claude.com/docs/en/settings),
"Shared project"). So before it prints anything the helper reads its own environment and prints no token, with the
variable's name in the message and exit code 1, when any of these is present:

| class | variables | refused when |
|---|---|---|
| tool base URL | `ANTHROPIC_BASE_URL`, `OPENAI_BASE_URL`, `OPENAI_API_BASE` | not `https://` on this project's canonical origin with the path `/llm` or below (another host, `http`, a port, user info, a query, `..`, `//`, `/llmx`, and the slug host of a project whose issuer is a custom domain) |
| another provider's endpoint | `ANTHROPIC_BEDROCK_BASE_URL`, `ANTHROPIC_BEDROCK_MANTLE_BASE_URL`, `ANTHROPIC_VERTEX_BASE_URL`, `ANTHROPIC_AWS_BASE_URL`, `ANTHROPIC_FOUNDRY_BASE_URL`, `ANTHROPIC_FOUNDRY_RESOURCE`, `ANTHROPIC_GOOGLE_CLOUD_BASE_URL` | set to anything |
| provider switch | `CLAUDE_CODE_USE_BEDROCK`, `_VERTEX`, `_MANTLE`, `_FOUNDRY`, `_ANTHROPIC_AWS`, and `CLAUDE_CODE_SKIP_BEDROCK_AUTH`, `_VERTEX_AUTH`, `_MANTLE_AUTH`, `_FOUNDRY_AUTH` | anything but empty, `0`, `false`, `no`, `off` |
| proxy | `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` and their lowercase forms | the proxy host is not one you allowed |
| certificate trust | `NODE_EXTRA_CA_CERTS`, `SSL_CERT_FILE`, `SSL_CERT_DIR`, `REQUESTS_CA_BUNDLE`, `CURL_CA_BUNDLE` | set, unless you allowed that exact value |
| | `CLAUDE_CODE_CERT_STORE` | anything but `bundled`, `system` or both |
| | `NODE_OPTIONS` | set to anything, unless you allowed that exact value. Node's own parser (quotes, escapes, `_` for `-`) reads options in more spellings than a deny-list can match, so the helper does not try |
| code or trust store in the helper or the tool | `PYTHONPATH`, `PYTHONSTARTUP`, `PYTHONUSERBASE`, `PYTHONHOME`, `PYTHONBREAKPOINT`, `PYTHONINSPECT`, `PYTHONSAFEPATH`, `PYTHONPYCACHEPREFIX`, `PYTHONWARNINGS`, `NODE_PATH`, `JAVA_TOOL_OPTIONS`, `_JAVA_OPTIONS`, `JDK_JAVA_OPTIONS`, `BASH_ENV`, `ENV`, `OPENSSL_CONF`, `OPENSSL_MODULES`, `OPENSSL_ENGINES`, `SHELLOPTS`, `GIT_SSL_CAINFO`, `GIT_PROXY_COMMAND`, `PERL5OPT`, `PERL5LIB`, `RUBYOPT`, `RUBYLIB`, and every `LD_*` and `DYLD_*` variable | set, unless you allowed that exact value |
| session bus (Linux, Secret Service and KWallet) | `DBUS_SESSION_BUS_ADDRESS` | it names a TCP address, or a Unix socket that is missing, is not a socket, or is owned by someone else, checked before the credential store is opened because the store talks to that bus and one you do not own can answer as the Secret Service. The ordinary forms pass: systemd's `/run/user/<uid>/bus`, `$XDG_RUNTIME_DIR/bus`, and the abstract or `/tmp/dbus-*` address `dbus-launch` sets on a desktop without systemd. macOS and Windows ignore the variable. There is nothing to allow: point it at your own bus |
| | `NODE_TLS_REJECT_UNAUTHORIZED`, `PYTHONHTTPSVERIFY` | `0`; this cannot be allowed |
| | `GIT_SSL_NO_VERIFY` | set to anything; this cannot be allowed |
| TLS session keys | `SSLKEYLOGFILE` (any non-empty value, a blank or quoted one too), and `--tls-keylog` in `NODE_OPTIONS` in any spelling (`--tls_keylog`, `"--tls-key\log"`) | set; this cannot be allowed, because whoever can read the key log can read the token |

The names come from Claude Code's own documentation (the [settings](https://code.claude.com/docs/en/settings),
[environment variables](https://code.claude.com/docs/en/env-vars),
[network configuration](https://code.claude.com/docs/en/network-config),
[Amazon Bedrock](https://code.claude.com/docs/en/amazon-bedrock) and
[Google Vertex AI](https://code.claude.com/docs/en/google-vertex-ai) pages, read on 2026-11-16); the OpenAI names are the
ones its SDKs read, and `CLAUDE_CODE_USE_ANTHROPIC_AWS` and the `CLAUDE_CODE_SKIP_*_AUTH` names beyond Bedrock and
Mantle are included by pattern without having been found on a page I read. The env-vars page was read only to its
first 100,000 characters.

A proxy or a certificate you need is allowed once, from your own terminal, and recorded in `login.json`:

```bash
authfi login --allow-proxy proxy.corp.example
authfi login --allow-env NODE_EXTRA_CA_CERTS=/etc/ssl/corp-ca.pem
```

An allowance needs a proxy host and, for a path variable, an absolute path in every member of the list (`/opt/lib:.` and a trailing `:` are refused). The same checks run again when the file is read, so a hand-edited entry that would not have been recorded is ignored. On Linux and macOS the allowances in a `login.json` that another user owns, or that its group or others can write, are ignored. `authfi logout --forget-allow` drops all of them; if the file cannot be rewritten it says so and the allowances stay in force.

The allowance is by proxy host, and by variable name together with its exact value, compared as written: no trimming, so a trailing space or a Unicode space is a different value, so a repository cannot swap the
file behind an allowed name for another path. Only `login.json` grants it: no environment variable does.

### The boundary

This guard catches misconfiguration. It does not defend against a repository you have trusted. The dynamic loader,
OpenSSL and the shell read `LD_PRELOAD`, `OPENSSL_CONF`, `BASH_ENV` and similar variables before any line of this tool
runs, so the helper refuses them, and names the variable, only as a second line of defence: code they name may already
have run. That code runs as you, and code running as you can read your credential store directly. A repository that can
set the helper's environment can already run commands as you (through a hook in the same `.claude/settings.json`, or
`BASH_ENV` in the shell that starts the helper), so no check inside the helper can stop it.

The control is Claude Code's workspace trust: it asks before it applies a repository's settings. Trust only repositories
you would let run code on your machine. This tool makes no stronger claim than that.

### What this does not cover

The guard reads the helper's environment and nothing else. It cannot see:

- a flag on the tool's command line (for example a `--settings` JSON or a model-endpoint flag) or a base URL the tool
  holds only in memory;
- a settings file the tool applies to itself without exporting the value, so any setting that is not an environment
  variable;
- `apiKeyHelper` itself: the same repository settings file can set the helper command, and a different command prints
  no token from this tool at all. Claude Code asks you to trust a folder before it applies a repository's `env`; that
  trust decision is yours and is not something this command can check;
- a proxy or CA that the operating system applies outside the environment (a system proxy, a CA in the OS store);
- a `.npmrc` `node-options` setting. npm, not Node, turns it into `NODE_OPTIONS`, and only for what npm launches (`npm run`, `npx`); Node run directly reads no such file. When Claude Code is launched through npm or npx the guard cannot see that setting (it does refuse the environment variables npm reads for `node-options`, `script-shell`, `userconfig` and `globalconfig`, in any case and with `_` or `-` in the key, such as `npm_config_node-options`). Launch the installed `claude` binary directly;
- an environment variable this list does not name.

What the tests prove is that, for each variable above, the helper refuses with its name and prints nothing on stdout,
that the proxy and certificate allowances need an exact match, and that removing any one of those checks fails a test.
They do not prove the tool's behaviour: I did not run Claude Code, and what a given tool inherits into its helper is
taken from its documentation. The check is a guard on the environment, not proof of where the tool sends the token.

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
