"""A person's own sign-in from a command line: ``authfi login``, ``authfi token --resource llm``, ``authfi logout``.

Claude Code (and any tool that takes its credential from a helper command) reaches the project's ``/llm`` door as
the person, not as an agent. The person signs in once in a browser; after that a helper prints a fresh access token.

* The client is public and registered once per machine (RFC 7591); its redirect is the loopback address with any
  port (RFC 8252 section 7.3), served by a one-shot listener that exits after the first callback.
* The refresh token lives in the operating system's credential store (the ``authfi[login]`` extra, ``keyring``) and
  nowhere else. With no usable store nothing is signed in and nothing is written: there is no plaintext fallback.
  The access token is kept there too, only until shortly before it expires. What goes to a file is not secret:
  the issuer and the client id.
* The server rotates a refresh token on every use and a second use of a spent one ends the whole family, so every
  command that reads or writes the stored token (``login``, ``logout`` and the renewal) takes one cross-process lock
  kept in the config directory that names the token, and the second finds the first one's token in the store instead
  of spending the old one again. A refresh whose outcome is unknown (the answer was lost, the store refused the new
  token) is never repeated with the old token: the old one is dropped and the person signs in again.
* The issuer and the ``/llm`` resource come from the project's discovery document (its canonical issuer), not from
  the host that was typed.

The protocol is the SDK's own: PKCE, ``state``, ``iss`` and the ID token are checked by :class:`~authfi.signin.WebApp`.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import os
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import IO, Any, Protocol, cast
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

import httpx

from authfi._protocol.register import parse_registration, registration_request
from authfi._protocol.verify import unverified_claims
from authfi.client import AuthFI
from authfi.errors import (
    AuthFIError,
    ConfigurationError,
    DiscoveryError,
    NotSupportedError,
    OAuthError,
    TemporarilyUnavailable,
    TransportError,
)
from authfi.secret import Secret
from authfi.signin import SignedIn, WebApp
from authfi.tokens import TokenSet

if sys.platform == "win32":  # pragma: no cover - exercised on Windows only
    import msvcrt
else:  # pragma: no cover - exercised off Windows only
    import fcntl

__all__ = [
    "KeyringVault",
    "LoginError",
    "Vault",
    "access_token",
    "login",
    "logout",
    "open_vault",
    "resolve_issuer",
]

LLM_PATH = "/llm"
SCOPE = ("openid", "llm:invoke", "offline_access")
CLIENT_NAME = "authfi command line"
#: The loopback redirect that is registered: the bare host, which the server matches with any port and path.
REGISTERED_REDIRECT = "http://127.0.0.1"
#: A cached access token is used while it has this many seconds left. Claude Code keeps a helper's value for five
#: minutes by default, so the token must outlive that, not just the moment the helper printed it.
DEFAULT_MIN_VALIDITY = 400.0
LOCK_TIMEOUT = 60.0
LOCK_NAME = "refresh.lock"
CALLBACK_TIMEOUT = 300.0

Clock = Callable[[], float]
Opener = Callable[[str], object]


class LoginError(ConfigurationError):
    """A sign-in step that cannot proceed. The message says what to do; it never carries a credential."""

    def __init__(self, message: str, *, code: str = "login_failed") -> None:
        super().__init__(message, code=code)


def resource_for(issuer: str) -> str:
    """The ``/llm`` door of the project at ``issuer``: the resource the token is asked for."""
    return issuer + LLM_PATH


def normalise_issuer(value: str) -> str:
    """``value`` without a trailing slash, when it is an absolute https origin; otherwise a :class:`LoginError`."""
    issuer = value.strip().rstrip("/")
    parts = urlsplit(issuer)
    if parts.scheme != "https" or not parts.hostname or parts.path or parts.query or parts.fragment:
        raise LoginError(f"the issuer must be the project's https origin, such as https://acme.authfi.io: {value!r}")
    return issuer


# --- where the secrets and the plain state live ---------------------------------------------------------


class Vault(Protocol):
    """Where the secrets live: the operating system's credential store."""

    def get(self, issuer: str, name: str) -> str | None: ...

    def set(self, issuer: str, name: str, value: str) -> None: ...

    def delete(self, issuer: str, name: str) -> None: ...


#: Windows Credential Locker refuses a blob over 2560 bytes (CRED_MAX_CREDENTIAL_BLOB_SIZE) and keyring stores it as
#: UTF-16, so a value over 1280 characters would not fit. A value is kept in pieces of this many characters.
CHUNK = 900
_CHUNKS = "chunks:"


class KeyringVault:
    """The :mod:`keyring` store (Credential Manager, Keychain, Secret Service): one entry per issuer and name.

    A value longer than :data:`CHUNK` is split across entries, so a long refresh token fits the Windows blob limit.
    The head entry names the piece count; a piece that is missing reads as no value.
    """

    def __init__(self, keyring_module: Any) -> None:
        self._keyring = keyring_module

    @staticmethod
    def _user(issuer: str, name: str, piece: int | None = None) -> str:
        return f"{issuer}#{name}" if piece is None else f"{issuer}#{name}#{piece}"

    def _read(self, issuer: str, name: str, piece: int | None = None) -> str | None:
        value = self._keyring.get_password("authfi", self._user(issuer, name, piece))
        return value if isinstance(value, str) and value else None

    def get(self, issuer: str, name: str) -> str | None:
        head = self._read(issuer, name)
        if head is None or not head.startswith(_CHUNKS):
            return head
        try:
            count = int(head[len(_CHUNKS) :])
        except ValueError:
            return None
        pieces = [self._read(issuer, name, i) for i in range(count)]
        return None if count < 1 or any(piece is None for piece in pieces) else "".join(cast("list[str]", pieces))

    def set(self, issuer: str, name: str, value: str) -> None:
        pieces = [value[i : i + CHUNK] for i in range(0, len(value), CHUNK)] or [""]
        previous = self._count(issuer, name)
        for index, piece in enumerate(pieces):
            self._keyring.set_password("authfi", self._user(issuer, name, index), piece)
        self._keyring.set_password("authfi", self._user(issuer, name), f"{_CHUNKS}{len(pieces)}")
        for index in range(len(pieces), previous):
            self._drop(issuer, name, index)

    def _count(self, issuer: str, name: str) -> int:
        head = self._read(issuer, name)
        if head is None or not head.startswith(_CHUNKS):
            return 0
        try:
            return int(head[len(_CHUNKS) :])
        except ValueError:
            return 0

    def _drop(self, issuer: str, name: str, piece: int | None = None) -> None:
        from keyring.errors import PasswordDeleteError

        with contextlib.suppress(PasswordDeleteError):
            self._keyring.delete_password("authfi", self._user(issuer, name, piece))

    def delete(self, issuer: str, name: str) -> None:
        for index in range(self._count(issuer, name)):
            self._drop(issuer, name, index)
        self._drop(issuer, name)


_NO_STORE = (
    "authfi login keeps your refresh token in the operating system's credential store and will not write it to a "
    "file: {why}. Install the keyring package (pip install 'authfi[login] @ "
    "git+https://github.com/authfi/authfi-python-sdk@main'); on Linux it needs a Secret Service such as GNOME "
    "Keyring or KWallet."
)

#: The backends that keep a secret under the operating system's own protection. Anything else (``keyrings.alt``'s
#: plaintext and encrypted files, a chain that falls to one, a ``PYTHON_KEYRING_BACKEND`` naming a file store) is
#: refused, because the refresh token would be readable by any process of the user.
SECURE_BACKENDS = frozenset(
    {
        "keyring.backends.macOS.Keyring",
        "keyring.backends.Windows.WinVaultKeyring",
        "keyring.backends.SecretService.Keyring",
        "keyring.backends.libsecret.Keyring",
        "keyring.backends.kwallet.DBusKeyring",
        "keyring.backends.kwallet.DBusKeyringKWallet4",
    }
)


def _backend_name(backend: object) -> str:
    kind = type(backend)
    return f"{kind.__module__}.{kind.__qualname__}"


def _unsecure_backend(backend: Any) -> str | None:
    """The name of the first backend in ``backend`` (or its chain) that is not an operating-system store."""
    members = getattr(backend, "backends", None)
    if _backend_name(backend) == "keyring.backends.chainer.ChainerBackend":
        chain = list(members or [])
        return next((_backend_name(b) for b in chain if _backend_name(b) not in SECURE_BACKENDS), None) or (
            None if chain else _backend_name(backend)
        )
    return None if _backend_name(backend) in SECURE_BACKENDS else _backend_name(backend)


def open_vault() -> Vault:
    """The credential store, or a :class:`LoginError` saying why there is none. Called before anything is written."""
    try:
        import keyring
    except ImportError as exc:
        raise LoginError(
            _NO_STORE.format(why="the keyring package is not installed"), code="no_credential_store"
        ) from exc
    try:
        backend = keyring.get_keyring()
        refused = _unsecure_backend(backend)
    except Exception as exc:  # a backend that fails while it is being chosen is no backend
        raise LoginError(_NO_STORE.format(why="no keyring backend started"), code="no_credential_store") from exc
    if refused is not None:
        raise LoginError(
            _NO_STORE.format(
                why=f"the keyring backend {refused} is not an operating-system store (Keychain, Credential Locker, "
                "Secret Service or KWallet); uninstall keyrings.alt and unset PYTHON_KEYRING_BACKEND and keyringrc.cfg "
                "if they select it"
            ),
            code="no_credential_store",
        )
    return KeyringVault(keyring)


def config_dir(env: Mapping[str, str]) -> Path:
    """The folder of the non-secret state: ``AUTHFI_CONFIG_DIR``, else the user's config directory."""
    if env.get("AUTHFI_CONFIG_DIR"):
        return Path(env["AUTHFI_CONFIG_DIR"])
    if sys.platform == "win32":
        base = env.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "authfi"


def _object(value: object) -> dict[str, Any]:
    """``value`` as a JSON object, or an empty one."""
    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


class Config:
    """``login.json``: the issuer last signed in to and, per issuer, the client id registered for it. No secrets."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.path = directory / "login.json"

    def _read(self) -> dict[str, Any]:
        try:
            raw = self.path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        except OSError as exc:
            raise LoginError(f"cannot read {self.path}: {exc.strerror or exc}") from exc
        try:
            data = json.loads(raw)
        except ValueError as exc:
            raise LoginError(f"{self.path} is not JSON; delete it and run authfi login again") from exc
        if not isinstance(data, dict):
            raise LoginError(f"{self.path} is not a JSON object; delete it and run authfi login again")
        return cast("dict[str, Any]", data)

    def default_issuer(self) -> str | None:
        value = self._read().get("default_issuer")
        return value if isinstance(value, str) and value else None

    def client_id(self, issuer: str) -> str | None:
        entry = _object(_object(self._read().get("clients")).get(issuer))
        value = entry.get("client_id")
        return value if isinstance(value, str) and value else None

    def canonical(self, typed: str) -> str:
        """The issuer the project names for itself, when ``typed`` was another door onto it (a slug host)."""
        value = _object(self._read().get("aliases")).get(typed)
        return value if isinstance(value, str) and value else typed

    def save(
        self, issuer: str, *, client_id: str | None = None, default: bool = True, alias: str | None = None
    ) -> None:
        data = self._read()
        clients = _object(data.get("clients"))
        if client_id is not None:
            clients[issuer] = {"client_id": client_id}
        data["clients"] = clients
        if alias is not None and alias != issuer:
            aliases = _object(data.get("aliases"))
            aliases[alias] = issuer
            data["aliases"] = aliases
        if default:
            data["default_issuer"] = issuer
        self.directory.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            self.directory.chmod(0o700)
        temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with contextlib.suppress(OSError):
            temporary.chmod(0o600)
        os.replace(temporary, self.path)


def slot(issuer: str, directory: Path) -> str:
    """The credential-store key of the sign-in kept for ``issuer`` by the config directory ``directory``.

    The store is shared by every process of the user, so the secret is keyed by the directory whose lock guards it:
    two config directories hold two sign-ins and never race over one refresh token.
    """
    where = os.path.normcase(os.path.abspath(directory))
    return f"{issuer}#{hashlib.sha256(where.encode()).hexdigest()[:12]}"


def resolve_issuer(explicit: str | None, env: Mapping[str, str], config: Config) -> str:
    """The project to act for: ``--issuer``, else ``AUTHFI_ISSUER``, else the one last signed in to."""
    chosen = explicit or env.get("AUTHFI_ISSUER") or config.default_issuer()
    if not chosen:
        raise LoginError("name the project: authfi login --issuer https://<project origin>", code="issuer_missing")
    return normalise_issuer(chosen)


@contextlib.contextmanager
def file_lock(path: Path, *, timeout: float = LOCK_TIMEOUT) -> Generator[None]:
    """An exclusive lock on ``path`` shared by every process of this user, released when the block ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle: IO[bytes] = open(path, "a+b")
    deadline = time.monotonic() + timeout
    try:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"\0")
            handle.flush()
        while True:
            try:
                _try_lock(handle)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LoginError(
                        "another authfi process is renewing the token and did not finish; try again",
                        code="lock_timeout",
                    ) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            _unlock(handle)
    finally:
        handle.close()


def _try_lock(handle: IO[bytes]) -> None:  # pragma: no cover - one branch per platform
    handle.seek(0)
    if sys.platform == "win32":
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(handle: IO[bytes]) -> None:  # pragma: no cover - one branch per platform
    handle.seek(0)
    if sys.platform == "win32":
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


# --- the loopback listener -----------------------------------------------------------------------------


class _Receiver(HTTPServer):
    """Serves the redirect once: the first request that carries a ``code`` or an ``error``, then it is closed."""

    allow_reuse_address = False

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Callback)
        self.params: dict[str, str] | None = None
        self.timeout = 0.25

    @property
    def redirect_uri(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/"

    def wait(self, seconds: float, *, cancelled: threading.Event | None = None) -> dict[str, str]:
        deadline = time.monotonic() + seconds
        while self.params is None:
            if time.monotonic() >= deadline or (cancelled is not None and cancelled.is_set()):
                raise LoginError("no sign-in came back to this machine in time; run authfi login again", code="timeout")
            self.handle_request()
        return self.params


class _Callback(BaseHTTPRequestHandler):
    server: _Receiver  # pyright: ignore[reportIncompatibleVariableOverride]

    def log_message(self, format: str, *args: Any) -> None:
        """The listener is silent: a callback's query holds a code."""

    def _answer(self, status: int, text: str) -> None:
        body = text.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        port = self.server.server_address[1]
        # A browser sends the address it was given; any other Host is a page reaching this port by name (DNS rebinding).
        if self.headers.get("Host") != f"127.0.0.1:{port}":
            self._answer(403, "forbidden")
            return
        parts = urlsplit(self.path)
        params = dict(parse_qsl(parts.query, keep_blank_values=True))
        if parts.path != "/" or not ("code" in params or "error" in params):
            self._answer(404, "not found")
            return
        self.server.params = params
        self._answer(200, "You are signed in to authfi. You can close this tab.")

    def do_POST(self) -> None:
        self._answer(405, "method not allowed")


# --- commands ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Runtime:
    """What the commands act through; tests replace the pieces."""

    env: Mapping[str, str]
    vault: Vault
    http_client: httpx.Client | None = None
    clock: Clock = time.time
    open_browser: Opener = webbrowser.open
    out: Callable[[str], None] = print
    err: Callable[[str], None] = lambda text: print(text, file=sys.stderr)

    def config(self) -> Config:
        return Config(config_dir(self.env))


def _state_key() -> Secret:
    return Secret(base64.urlsafe_b64encode(os.urandom(32)).decode())


def _held(refresh: str) -> TokenSet:
    """The token set of a machine that holds only a refresh token."""
    return TokenSet(access_token=Secret("-"), expires_at=datetime.fromtimestamp(0, UTC), refresh_token=Secret(refresh))


def _web_app(af: AuthFI, client_id: str, redirect_uri: str) -> WebApp:
    return af.web_app(client_id=client_id, client_secret=None, redirect_uri=redirect_uri, state_key=_state_key())


def _register(af: AuthFI) -> str:
    metadata = af.metadata()
    if metadata.registration_endpoint is None:
        raise LoginError(
            "this project does not advertise client registration, so this machine cannot sign in on its own; "
            "ask your administrator",
            code="registration_unavailable",
        )
    request = registration_request(
        metadata.registration_endpoint,
        client_name=CLIENT_NAME,
        redirect_uris=[REGISTERED_REDIRECT],
        scope=SCOPE,
    )
    response = af._transport.send(request)  # pyright: ignore[reportPrivateUsage]
    return parse_registration(response, redirect_uris=[REGISTERED_REDIRECT])


def canonical_issuer(rt: Runtime, typed: str) -> str:
    """The issuer the project's discovery document names, which is not always the host that was typed.

    A project has one issuer, its canonical origin, however many doors lead to it (RFC 8414); the resource and the
    token audience are built from it. The typed host is the trust anchor: what it publishes is checked as an https
    origin, and the issuer it names must then publish its own discovery document naming itself.
    """
    client = rt.http_client or httpx.Client(timeout=15.0, follow_redirects=False)
    try:
        for path in ("/.well-known/openid-configuration", "/.well-known/oauth-authorization-server"):
            try:
                response = client.get(typed + path, headers={"Accept": "application/json"})
            except httpx.HTTPError as exc:
                raise LoginError(
                    f"cannot reach {typed} to read its discovery document ({type(exc).__name__})",
                    code="discovery_failed",
                ) from exc
            if response.status_code != 200:
                continue
            try:
                doc: Any = response.json()
            except ValueError:
                continue
            named = cast("dict[str, Any]", doc).get("issuer") if isinstance(doc, dict) else None
            if isinstance(named, str):
                return normalise_issuer(named)
        raise LoginError(f"{typed} publishes no discovery document that names its issuer", code="discovery_failed")
    finally:
        if rt.http_client is None:
            client.close()


def _sign_in(
    rt: Runtime, config: Config, issuer: str, *, browser: bool, register: bool, timeout: float
) -> tuple[SignedIn, str]:
    receiver = _Receiver()
    try:
        with AuthFI(issuer, http_client=rt.http_client) as af:
            metadata = af.metadata()
            if metadata.authorization_endpoint is None:
                raise LoginError("this project's discovery names no authorization endpoint", code="not_supported")
            client_id = None if register else config.client_id(issuer)
            if client_id is None:
                client_id = _register(af)
            web = _web_app(af, client_id, receiver.redirect_uri)
            pending = web.start_sign_in(SCOPE, resource=resource_for(issuer))
            rt.err(
                "Sign in at this address (opening your browser):\n  " + pending.url
                if browser
                else "Sign in at this address:\n  " + pending.url
            )
            if browser:
                with contextlib.suppress(Exception):  # a headless shell: the address above is the way
                    rt.open_browser(pending.url)
            params = receiver.wait(timeout)
            receiver.server_close()
            return web.finish_sign_in(params, pending), client_id
    finally:
        receiver.server_close()


def login(
    rt: Runtime,
    issuer_arg: str | None,
    *,
    browser: bool = True,
    reregister: bool = False,
    timeout: float = CALLBACK_TIMEOUT,
) -> int:
    """Sign the person in for the project's ``/llm`` door and keep the refresh token in the credential store.

    A stored client the server no longer knows (``invalid_client``) is registered again once, and the sign-in
    repeated; a second refusal is the answer.
    """
    config = rt.config()
    typed = resolve_issuer(issuer_arg, rt.env, config)
    issuer = canonical_issuer(rt, typed)
    try:
        signed_in, client_id = _sign_in(rt, config, issuer, browser=browser, register=reregister, timeout=timeout)
    except OAuthError as error:
        if error.code != "invalid_client":
            raise
        rt.err("this machine's client is no longer registered; registering it again")
        signed_in, client_id = _sign_in(rt, config, issuer, browser=browser, register=True, timeout=timeout)
    tokens = signed_in.tokens
    if tokens.refresh_token is None:
        raise LoginError(
            "the server issued no refresh token, so the sign-in could not be kept", code="no_refresh_token"
        )
    if tokens.scope and "llm:invoke" not in tokens.scope:
        raise LoginError("the server did not grant llm:invoke; ask your administrator for access to the /llm door")
    _check_audience(tokens.access_token, resource_for(issuer))
    key = slot(issuer, config.directory)
    with file_lock(config.directory / LOCK_NAME):
        try:
            rt.vault.set(key, "refresh", tokens.refresh_token.reveal())
        except Exception as exc:
            raise LoginError(
                "the credential store refused the refresh token, so nothing is signed in "
                f"({type(exc).__name__} from the store)",
                code="store_failed",
            ) from exc
        _store_access(rt, key, tokens)
        config.save(issuer, client_id=client_id, alias=typed)
    rt.out(f"signed in to {issuer} as {signed_in.user.sub}")
    return 0


def _check_audience(access: Secret, resource: str) -> None:
    """A token that names an audience must name ``resource``; an opaque token cannot be read and is left to the door."""
    text = access.reveal()
    if text.count(".") != 2:
        return
    try:
        audience = unverified_claims(text).get("aud")
    except AuthFIError:
        return
    names: list[str] = []
    if isinstance(audience, str):
        names = [audience]
    elif isinstance(audience, list):
        names = [a for a in cast("list[object]", audience) if isinstance(a, str)]
    if names and resource not in names:
        raise LoginError(f"the token is not addressed to {resource}; run authfi login again", code="wrong_audience")


def _store_access(rt: Runtime, issuer: str, tokens: TokenSet) -> None:
    """Keep the access token until it nearly expires. A store that cannot hold it only costs a refresh."""
    try:
        rt.vault.set(
            issuer, "access", json.dumps({"token": tokens.access_token.reveal(), "exp": tokens.expires_at.timestamp()})
        )
    except Exception:
        with contextlib.suppress(Exception):
            rt.vault.delete(issuer, "access")


def _cached(rt: Runtime, issuer: str, min_validity: float) -> str | None:
    try:
        raw = rt.vault.get(issuer, "access")
        entry: Any = json.loads(raw) if raw else None
    except Exception:
        return None
    if not isinstance(entry, dict):
        return None
    token, exp = entry.get("token"), entry.get("exp")  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
    if isinstance(token, str) and token and isinstance(exp, (int, float)) and exp - min_validity > rt.clock():
        return token
    return None


#: Variables a tool reads to learn where its model calls go. The helper inherits the tool's environment, so a repo's
#: settings that point one of these elsewhere would otherwise send the person's token there.
BASE_URL_VARIABLES = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")


def _is_llm_url(value: str, origins: frozenset[str]) -> bool:
    try:
        parts = urlsplit(value.strip())
        port = parts.port
    except ValueError:
        return False
    if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
        return False
    if parts.query or parts.fragment or port not in (None, 443):
        return False
    path = unquote(unquote(parts.path))
    if ".." in path.split("/") or "." in path.split("/") or "\\" in path or "//" in path:
        return False
    origin = urlunsplit(("https", parts.hostname.lower(), "", "", ""))
    return origin in origins and (path.rstrip("/") == LLM_PATH or path.startswith(LLM_PATH + "/"))


def check_base_url(env: Mapping[str, str], origins: set[str]) -> None:
    """Refuse when a tool's base URL names anything but this project's ``/llm`` door."""
    allowed = frozenset(origin.lower() for origin in origins)
    for name in BASE_URL_VARIABLES:
        value = env.get(name, "").strip()
        if value and not _is_llm_url(value, allowed):
            shown = urlsplit(value)
            where = f"{shown.scheme}://{shown.hostname}" if shown.hostname else "a value that is not an address"
            raise LoginError(
                f"{name} points at {where}, not at this project's {LLM_PATH} door, so no token is printed; "
                f"set it to {resource_for(sorted(allowed)[0])} or unset it",
                code="base_url_mismatch",
            )


def _forget(rt: Runtime, key: str) -> None:
    """Drop the stored tokens (best effort): a refresh token whose use is in doubt is never presented again."""
    for name in ("refresh", "access"):
        with contextlib.suppress(Exception):
            rt.vault.delete(key, name)


def _kept_after(error: AuthFIError) -> bool:
    """``True`` when the failure proves the refresh token was not spent, so it may be presented again."""
    if isinstance(error, (TemporarilyUnavailable, DiscoveryError, NotSupportedError)):
        return True
    if isinstance(error, TransportError):
        return error.retryable
    if isinstance(error, OAuthError):
        return error.status is not None and error.status < 500 and error.code not in ("invalid_grant", "invalid_client")
    return False


def _refresh_failed(rt: Runtime, config: Config, issuer: str, key: str, error: AuthFIError) -> AuthFIError:
    """The error to raise after a failed refresh; the stored token is dropped unless it is known to be unspent."""
    if _kept_after(error):
        return error
    _forget(rt, key)
    if isinstance(error, OAuthError) and error.code == "invalid_client":
        with contextlib.suppress(Exception):  # the dead client is replaced once; the person then signs in
            with AuthFI(issuer, http_client=rt.http_client) as af:
                config.save(issuer, client_id=_register(af), default=False)
        return LoginError(
            "the server no longer knows this machine's client; it was registered again: "
            f"run authfi login --issuer {issuer}",
            code="login_required",
        )
    return LoginError(
        f"the stored sign-in cannot be used again ({error.message}); run authfi login --issuer {issuer}",
        code="login_required",
    )


def access_token(rt: Runtime, issuer_arg: str | None, *, min_validity: float = DEFAULT_MIN_VALIDITY) -> str:
    """A person access token for ``/llm``: the cached one while it lasts, otherwise a refresh under the lock.

    Nothing is returned when a tool's base URL variable names another host. A refresh whose outcome is unknown is not
    repeated: the stored refresh token is dropped and the person signs in again.
    """
    config = rt.config()
    typed = resolve_issuer(issuer_arg, rt.env, config)
    issuer = config.canonical(typed)
    check_base_url(rt.env, {typed, issuer})
    key = slot(issuer, config.directory)
    cached = _cached(rt, key, min_validity)
    if cached is not None:
        return cached
    with file_lock(config.directory / LOCK_NAME):
        cached = _cached(rt, key, min_validity)  # another process may have renewed it while this one waited
        if cached is not None:
            return cached
        refresh = rt.vault.get(key, "refresh")
        client_id = config.client_id(issuer)
        if refresh is None or client_id is None:
            raise LoginError(f"not signed in to {issuer}: run authfi login --issuer {issuer}", code="not_signed_in")
        try:
            with AuthFI(issuer, http_client=rt.http_client) as af:
                tokens = _web_app(af, client_id, REGISTERED_REDIRECT).refresh(_held(refresh))
        except AuthFIError as error:
            raise _refresh_failed(rt, config, issuer, key, error) from error
        if tokens.refresh_token is None:  # unreachable: from_response carries the previous one forward
            _forget(rt, key)
            raise LoginError("the server answered no refresh token; run authfi login again", code="login_required")
        rotated = tokens.refresh_token.reveal()
        if rotated != refresh:
            try:
                rt.vault.set(key, "refresh", rotated)
            except Exception as exc:
                _forget(rt, key)  # the old token is spent and the new one is lost: never present the old one
                raise LoginError(
                    "the server renewed the sign-in but the credential store refused it; run authfi login again",
                    code="store_failed",
                ) from exc
        _check_audience(tokens.access_token, resource_for(issuer))
        _store_access(rt, key, tokens)
        return tokens.access_token.reveal()


def logout(rt: Runtime, issuer_arg: str | None) -> int:
    """Revoke the refresh token at the server (RFC 7009), then delete what is stored (also when it fails).

    Under the same lock as the renewal, so a helper in flight cannot store a rotated token after this deleted it.
    """
    config = rt.config()
    issuer = config.canonical(resolve_issuer(issuer_arg, rt.env, config))
    key = slot(issuer, config.directory)
    with file_lock(config.directory / LOCK_NAME):
        refresh = rt.vault.get(key, "refresh")
        failure: str | None = None
        client_id = config.client_id(issuer)
        if refresh is not None and client_id is None:
            failure = "this machine has no registered client to revoke with"
        elif refresh is not None and client_id is not None:
            try:
                with AuthFI(issuer, http_client=rt.http_client) as af:
                    _web_app(af, client_id, REGISTERED_REDIRECT).revoke(_held(refresh))
            except (AuthFIError, NotSupportedError) as error:
                failure = error.message
        rt.vault.delete(key, "refresh")
        rt.vault.delete(key, "access")
    if failure is not None:
        raise LoginError(
            f"the stored tokens were deleted but the server did not confirm the revocation ({failure}); "
            "the refresh token stays valid there until it expires",
            code="revocation_failed",
        )
    rt.out(f"signed out of {issuer}")
    return 0
