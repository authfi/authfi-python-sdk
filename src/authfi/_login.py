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
import importlib
import json
import os
import re
import shlex
import stat
import sys
import threading
import time
import webbrowser
from collections.abc import Callable, Generator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import IO, Any, Protocol, cast
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

import httpx

from authfi import _bus
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
    "helper_command",
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
    "file: {why}. Install the keyring package (pip install 'authfi[login]'); on Linux it needs a Secret Service "
    "such as GNOME Keyring or KWallet."
)

#: The backends that keep a secret under the operating system's own protection. Anything else (``keyrings.alt``'s
#: plaintext and encrypted files, a chain that falls to one, a ``PYTHON_KEYRING_BACKEND`` naming a file store) is
#: refused, because the refresh token would be readable by any process of the user.
SECURE_BACKENDS = frozenset(
    {
        "keyring.backends.macOS.Keyring",
        "keyring.backends.Windows.WinVaultKeyring",
        "authfi._login.VerifiedSecretService",
    }
)


def helper_command(executable: str | None = None) -> str:
    """The ``apiKeyHelper`` value that starts the helper isolated: this interpreter, by absolute path, with ``-I``.

    ``-I`` ignores ``PYTHON*`` variables and the user site and keeps the working directory off ``sys.path``. It does
    not touch the dynamic loader, OpenSSL's configuration or the shell that starts the helper (``LD_PRELOAD``,
    ``OPENSSL_CONF``, ``BASH_ENV``). A console script cannot carry the flag, which is why the helper is started as
    ``python -I -m authfi``. Claude Code runs the line through ``/bin/sh`` on macOS and Linux and through ``cmd`` on
    Windows, so the interpreter is quoted for that shell.
    """
    exe = executable if executable is not None else sys.executable
    if not exe or not os.path.isabs(exe):
        raise LoginError(
            "this Python has no absolute path (sys.executable is empty or relative), so the apiKeyHelper line cannot "
            "name it; run authfi login with the interpreter's full path",
            code="no_interpreter",
        )
    tail = "-I -m authfi token --resource llm"
    if sys.platform == "win32":
        if '"' in exe or "%" in exe:
            raise LoginError(f"the interpreter path {exe!r} has a character cmd cannot quote", code="no_interpreter")
        return f'"{exe}" {tail}'
    return f"{shlex.quote(exe)} {tail}"


def _backend_classes(platform: str) -> list[tuple[str, str]]:
    """The operating system's own stores for ``platform``, as ``(module, class)``, in the order they are tried.

    keyring is never asked which backend to use: its own choice reads ``keyringrc.cfg`` from a folder the environment
    names (``XDG_CONFIG_HOME``, ``LOCALAPPDATA``), imports a module from that file's ``keyring-path``, and honours
    ``PYTHON_KEYRING_BACKEND``, all before this helper could refuse anything. The class is named here and nowhere else.
    """
    if platform == "win32":
        return [("keyring.backends.Windows", "WinVaultKeyring")]
    if platform == "darwin":
        return [("keyring.backends.macOS", "Keyring")]
    if platform.startswith("linux"):  # KWallet answers the Secret Service API too (KDE 5.97 and later)
        return [("authfi._login", "VerifiedSecretService")]
    return []


def _verified_dbus_init(secretstorage: Any, util: Any) -> Any:
    """A jeepney connection to the session bus whose listener runs as this user: ``dbus_init``, but checked."""
    address = os.environ.get(_bus.DBUS, "")
    if not address:
        raise secretstorage.SecretServiceNotAvailableException(f"Environment variable {_bus.DBUS} is unset")
    try:
        connection = _bus.open_verified_bus(address, os.getuid())
    except _bus.BusPeerError as exc:
        raise _refuse(
            _bus.DBUS, f"names a session bus that is not yours ({exc})", "unset it, or point it at your own session bus"
        ) from exc
    except (_bus.BusError, ValueError) as exc:
        raise secretstorage.SecretServiceNotAvailableException(str(exc)) from exc
    util.add_match_rules(connection)
    return connection


def _build_verified_secret_service() -> type:
    """Keyring's Secret Service backend, with every bus connection opened and checked by :mod:`authfi._bus`.

    keyring's own backend calls ``secretstorage.dbus_init()``, which connects to whatever
    ``DBUS_SESSION_BUS_ADDRESS`` names and trusts it. Here the connection is made first, the listener's uid is read
    from the socket, and that checked connection is what ``secretstorage`` is given.
    """
    secretstorage = importlib.import_module("secretstorage")
    util = importlib.import_module("secretstorage.util")
    exceptions = importlib.import_module("secretstorage.exceptions")
    errors = importlib.import_module("keyring.errors")
    base = importlib.import_module("keyring.backends.SecretService").Keyring

    class VerifiedSecretService(base):  # type: ignore[misc, valid-type]
        @property
        def priority(self) -> float:
            with contextlib.closing(_verified_dbus_init(secretstorage, util)) as connection:
                if not secretstorage.check_service_availability(connection):
                    raise RuntimeError("The Secret Service daemon is neither running nor activatable through D-Bus")
            return 5

        def get_preferred_collection(self) -> Any:
            bus = _verified_dbus_init(secretstorage, util)
            try:
                if hasattr(self, "preferred_collection"):
                    collection = secretstorage.Collection(bus, self.preferred_collection)
                else:
                    collection = secretstorage.get_default_collection(bus)
            except exceptions.SecretStorageException as exc:
                bus.close()
                raise errors.InitError(f"Failed to create the collection: {exc}.") from exc
            if collection.is_locked():
                collection.unlock()
                if collection.is_locked():  # the person dismissed the prompt
                    bus.close()
                    raise errors.KeyringLocked("Failed to unlock the collection!")
            return collection

    VerifiedSecretService.__module__ = __name__
    return VerifiedSecretService


def __getattr__(name: str) -> Any:
    if name == "VerifiedSecretService":
        built = _build_verified_secret_service()
        globals()[name] = built
        return built
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _started(module: str, name: str) -> Any:
    """An instance of the named store class when it starts and reports itself usable, else ``None``."""
    try:
        backend = getattr(importlib.import_module(module), name)()
        return backend if backend.priority > 0 else None  # priority raises when the store is not there
    except LoginError:  # a refusal says why; "no store" would hide it
        raise
    except Exception:  # a store that fails to start is no store
        return None


def _os_backend() -> Any:
    """An instance of the first operating-system store that is available here, or ``None``."""
    for module, name in _backend_classes(sys.platform):
        if f"{module}.{name}" not in SECURE_BACKENDS:
            continue  # never instantiate a class that is not an operating-system store
        backend = _started(module, name)
        if backend is not None:
            return backend
    return None


def open_vault() -> Vault:
    """The credential store, or a :class:`LoginError` saying why there is none. Called before anything is written."""
    try:
        importlib.import_module("keyring")  # only its backend classes are used; its backend selection is never called
    except ImportError as exc:
        raise LoginError(
            _NO_STORE.format(why="the keyring package is not installed"), code="no_credential_store"
        ) from exc
    backend = _os_backend()
    if backend is None:
        raise LoginError(
            _NO_STORE.format(
                why="no operating-system store (Keychain, Credential Locker or Secret Service) is available "
                "here; the helper names that store itself and does not read keyringrc.cfg or PYTHON_KEYRING_BACKEND"
            ),
            code="no_credential_store",
        )
    return KeyringVault(backend)


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


@dataclass(frozen=True, slots=True)
class Allowed:
    """The proxy hosts and the exact environment values the person allowed by name."""

    proxies: frozenset[str]
    env: Mapping[str, str]


def private_to(info: os.stat_result, uid: int) -> bool:
    """``True`` when the file belongs to ``uid`` and neither its group nor others can write it."""
    return info.st_uid == uid and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)


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

    def allowed(self) -> Allowed:
        """What the person allowed with ``authfi login --allow-proxy`` and ``--allow-env``; nothing by default."""
        if not self._trusted():
            return Allowed(proxies=frozenset(), env={})
        entry = _object(self._read().get("allow"))
        proxies = entry.get("proxies")
        listed: list[object] = cast("list[object]", proxies) if isinstance(proxies, list) else []
        env = _object(entry.get("env"))
        return Allowed(
            proxies=frozenset(
                p.lower() for p in listed if isinstance(p, str) and p.strip() and _host_of(p) == p.lower()
            ),
            env={k: v for k, v in env.items() if isinstance(v, str) and allowance_problem(k, v) is None},
        )

    def _trusted(self) -> bool:
        """``False`` when another user owns ``login.json`` or can write it: its allowances are then ignored."""
        geteuid = getattr(os, "geteuid", None)
        if geteuid is None:
            return True
        try:
            info = self.path.stat()
        except OSError:
            return True
        return private_to(info, geteuid())

    def forget_allow(self) -> None:
        """Drop every proxy host and environment value the person allowed."""
        data = self._read()
        if data.pop("allow", None) is not None:
            temporary = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
            try:
                temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                with contextlib.suppress(OSError):
                    temporary.chmod(0o600)
                os.replace(temporary, self.path)
            except OSError as exc:
                with contextlib.suppress(OSError):
                    temporary.unlink()
                raise LoginError(
                    f"the stored tokens were deleted but the allowances in {self.path} could not be removed: "
                    f"{exc.strerror or exc}; they are still in force",
                    code="forget_failed",
                ) from exc

    def save(
        self,
        issuer: str,
        *,
        client_id: str | None = None,
        default: bool = True,
        alias: str | None = None,
        allow_proxies: Sequence[str] = (),
        allow_env: Mapping[str, str] | None = None,
    ) -> None:
        data = self._read()
        if allow_proxies or allow_env:
            allow = _object(data.get("allow"))
            known = cast("list[str]", allow.get("proxies") or [])
            allow["proxies"] = sorted({*known, *(p.lower() for p in allow_proxies)})
            allow["env"] = {**_object(allow.get("env")), **(allow_env or {})}
            data["allow"] = allow
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


def _allowed_proxies(hosts: Sequence[str]) -> list[str]:
    """The proxy hosts to allow; a value that names no host would allow every proxy value that does not parse."""
    out: list[str] = []
    for value in hosts:
        host = _host_of(value)
        if not host:
            raise LoginError(f"--allow-proxy takes a proxy host, not {value!r}", code="bad_allow")
        out.append(host)
    return out


def _allowed_env(pairs: Sequence[str]) -> dict[str, str]:
    """``NAME=VALUE`` pairs for the variables the guard can allow; anything else is a :class:`LoginError`."""
    out: dict[str, str] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if why := allowance_problem(name, value):
            raise LoginError(f"--allow-env {name} is refused: {why}", code="bad_allow")
        out[name] = value
    return out


def login(
    rt: Runtime,
    issuer_arg: str | None,
    *,
    browser: bool = True,
    reregister: bool = False,
    timeout: float = CALLBACK_TIMEOUT,
    allow_proxies: Sequence[str] = (),
    allow_env: Sequence[str] = (),
) -> int:
    """Sign the person in for the project's ``/llm`` door and keep the refresh token in the credential store.

    A stored client the server no longer knows (``invalid_client``) is registered again once, and the sign-in
    repeated; a second refusal is the answer.
    """
    config = rt.config()
    allowed_env = _allowed_env(allow_env)
    allowed_proxies = _allowed_proxies(allow_proxies)
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
        config.save(issuer, client_id=client_id, alias=typed, allow_proxies=allowed_proxies, allow_env=allowed_env)
    rt.out(f"signed in to {issuer} as {signed_in.user.sub}")
    rt.out(f'set this in Claude Code\'s settings: "apiKeyHelper": {json.dumps(helper_command())}')
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


#: Variables through which a tool learns where its model calls go. A repository's settings file can set every one
#: of them (its ``env`` block reaches the helper's environment), so no token is printed unless each is absent or right.
#: Sources: Claude Code's settings, env-vars, network-config, amazon-bedrock and google-vertex-ai pages
#: (code.claude.com/docs/en), read on 2026-11-16; the OpenAI names are those its SDKs read.
#:
#: These must name this project's ``/llm`` door.
BASE_URL_VARIABLES = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OPENAI_API_BASE")
#: Another provider's endpoint: the request would go to Bedrock, Vertex, Foundry or a cloud gateway, never to ``/llm``.
PROVIDER_URL_VARIABLES = (
    "ANTHROPIC_BEDROCK_BASE_URL",
    "ANTHROPIC_BEDROCK_MANTLE_BASE_URL",
    "ANTHROPIC_VERTEX_BASE_URL",
    "ANTHROPIC_AWS_BASE_URL",
    "ANTHROPIC_FOUNDRY_BASE_URL",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "ANTHROPIC_GOOGLE_CLOUD_BASE_URL",
)
#: Switches that send the request to another provider; a truthy value is refused.
PROVIDER_SWITCHES = (
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_MANTLE",
    "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_CODE_USE_ANTHROPIC_AWS",
    "CLAUDE_CODE_SKIP_BEDROCK_AUTH",
    "CLAUDE_CODE_SKIP_VERTEX_AUTH",
    "CLAUDE_CODE_SKIP_MANTLE_AUTH",
    "CLAUDE_CODE_SKIP_FOUNDRY_AUTH",
)
#: A proxy sees the request, and the bearer in it, unless the person allowed that proxy host.
PROXY_VARIABLES = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
)
#: Trust-store and TLS variables: a CA the person did not name lets a proxy read the token in transit. ``NODE_OPTIONS``
#: is here because it can load code and change TLS and the guard cannot parse it as node does: any value is refused.
TLS_VARIABLES = (
    "NODE_EXTRA_CA_CERTS",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "CLAUDE_CODE_CERT_STORE",
    "NODE_OPTIONS",
)
#: Variables the dynamic loader, OpenSSL or the shell reads before any line of this tool runs: refusing them here is a
#: second line of defence (the first is that the person only trusts repositories they would let run code).
_READ_BEFORE_THE_GUARD = frozenset(
    {"OPENSSL_CONF", "OPENSSL_ENGINES", "OPENSSL_MODULES", "BASH_ENV", "ENV", "SHELLOPTS"}
)
#: Variables that load code or swap a trust store inside the helper or the tool: refused when set, unless the person
#: allowed that exact value. ``LD_*`` and ``DYLD_*`` are covered by prefix.
CODE_VARIABLES = (
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONUSERBASE",
    "PYTHONHOME",
    "PYTHONBREAKPOINT",
    "PYTHONINSPECT",
    "PYTHONSAFEPATH",
    "PYTHONPYCACHEPREFIX",
    "PYTHONWARNINGS",
    "NODE_PATH",
    "JAVA_TOOL_OPTIONS",
    "_JAVA_OPTIONS",
    "JDK_JAVA_OPTIONS",
    "BASH_ENV",
    "ENV",
    "OPENSSL_CONF",
    "OPENSSL_MODULES",
    "OPENSSL_ENGINES",
    "SHELLOPTS",
    "GIT_SSL_CAINFO",
    "GIT_PROXY_COMMAND",
    "PERL5OPT",
    "PERL5LIB",
    "RUBYOPT",
    "RUBYLIB",
)
CODE_PREFIXES = ("LD_", "DYLD_")
#: Guarded variables whose value is not a path: an allowance for one needs no absolute-path check.
_NOT_PATHS = frozenset(
    {
        "CLAUDE_CODE_CERT_STORE",
        "NODE_OPTIONS",
        "JAVA_TOOL_OPTIONS",
        "_JAVA_OPTIONS",
        "JDK_JAVA_OPTIONS",
        "PERL5OPT",
        "RUBYOPT",
        "GIT_PROXY_COMMAND",
        "PYTHONBREAKPOINT",
        "PYTHONINSPECT",
        "PYTHONSAFEPATH",
        "PYTHONWARNINGS",
    }
)
_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/]")
#: Verification turned off and exported session keys are never allowed.
TLS_OFF = "NODE_TLS_REJECT_UNAUTHORIZED"
_NEVER_NAMES = (TLS_OFF, "PYTHONHTTPSVERIFY", "SSLKEYLOGFILE", "GIT_SSL_NO_VERIFY")


#: The npm config keys that reach what npm launches: ``node-options`` becomes ``NODE_OPTIONS``, ``script-shell`` names
#: the program that runs a script, and the two ``.npmrc`` files can carry either (the guard cannot see inside them).
NPM_KEYS = frozenset({"node-options", "script-shell", "userconfig", "globalconfig"})


def npm_key(name: str) -> str | None:
    """The npm config key an environment variable sets, as npm reads it, or ``None`` when it is not ``npm_config_*``.

    npm takes the prefix in any case and lower-cases the rest, and ``_`` and ``-`` in a key mean the same, so
    ``npm_config_node-options`` and ``NPM_CONFIG_NODE_OPTIONS`` are one key.
    """
    prefix = "npm_config_"
    return name[len(prefix) :].lower().replace("_", "-") if name[: len(prefix)].lower() == prefix else None


def _is_node_options(name: str) -> bool:
    return name == "NODE_OPTIONS" or npm_key(name) == "node-options"


def is_guarded(name: str) -> bool:
    """``True`` for a variable the guard refuses when set, unless the person allowed that exact value."""
    return (
        name in TLS_VARIABLES or name in CODE_VARIABLES or name.startswith(CODE_PREFIXES) or npm_key(name) in NPM_KEYS
    )


def _absolute(member: str) -> bool:
    """An absolute path on this platform: a drive-letter spelling is a relative name anywhere but Windows."""
    return PurePosixPath(member).is_absolute() or (sys.platform == "win32" and PureWindowsPath(member).is_absolute())


def node_options(value: str) -> list[str]:
    """``NODE_OPTIONS`` as node reads it: split on spaces, ``"`` quotes, escapes inside quotes, ``_`` as ``-``."""
    tokens: list[str] = []
    current: list[str] = []
    in_string = False
    index = 0
    while index < len(value):
        char = value[index]
        if char == "\\" and in_string:
            index += 1
            if index >= len(value):
                break
            current.append(value[index])
        elif char == " " and not in_string:
            if current:
                tokens.append("".join(current))
                current = []
        elif char == '"':
            in_string = not in_string
        else:
            current.append(char)
        index += 1
    if current:
        tokens.append("".join(current))
    out: list[str] = []
    for token in tokens:
        head, equals, tail = token.partition("=")
        out.append(head.replace("_", "-") + equals + tail)
    return out


def _carries_keylog(value: str) -> bool:
    """``True`` when node would read ``--tls-keylog`` out of this value, or when any spelling of it could hide in it."""
    if any(token.partition("=")[0] == "--tls-keylog" for token in node_options(value)):
        return True
    return "tls-keylog" in re.sub(r"[\\\"'\s]", "", value)  # a spelling node's own parser would miss


def never_allowed(name: str, value: str) -> str | None:
    """Why this variable and value can never be allowed, whatever the person records; ``None`` when it can be.

    Each rule either turns certificate verification off or exports the TLS session keys, which lets whoever holds
    the key log or sits on the path read the token.
    """
    plain = value.strip().replace('"', "").replace("'", "")
    if name in (TLS_OFF, "PYTHONHTTPSVERIFY") and plain == "0":
        return "turns certificate verification off"
    if name == "GIT_SSL_NO_VERIFY" and value:
        return "turns certificate verification off"
    if name == "SSLKEYLOGFILE" and value:  # any non-empty value, blank or quoted too: Python opens it as written
        return "writes the TLS session keys to a file, so anyone who can read that file can read the token"
    if _is_node_options(name) and _carries_keylog(value):
        return "carries --tls-keylog, which writes the TLS session keys to a file anyone who can read it can use"
    return None


def _never_allowed_name(name: str) -> str | None:
    """The reason an allowance for ``name`` is refused whatever its value, or ``None``."""
    return never_allowed(name, "0") if name in _NEVER_NAMES else None


_FILE_NAME = re.compile(r"[\\/]|\.(js|mjs|cjs|so|dylib|dll|cnf|jar|rb|pl|pm|py)$")
_FLAG_THEN_PATH = re.compile(r"^-[A-Za-z](?=[./~\\])")


def _relative_file_in(value: str) -> str | None:
    """The first token of an option string that names a file by a relative path, or ``None``.

    ``--require=./hook.js``, ``--openssl-config=openssl.cnf``, ``-javaagent:agent.jar`` and ``-r./x`` all name a file
    that the repository the tool runs in can supply. A module name (``-MHook``) is found through a search path the
    guard covers separately (``PERL5LIB``, ``RUBYLIB``, ``NODE_PATH``).
    """
    for piece in re.split(r"[\s=,]+", value):
        for token in [piece] if sys.platform == "win32" and _WINDOWS_PATH.match(piece) else piece.split(":"):
            token = _FLAG_THEN_PATH.sub("", token)
            if token and _FILE_NAME.search(token) and not _absolute(token):
                return token
    return None


def allowance_problem(name: str, value: str) -> str | None:
    """Why ``NAME=VALUE`` cannot be recorded as an allowance, or ``None``; also applied when the file is read."""
    if why := _never_allowed_name(name):
        return f"it {why}; this cannot be allowed"
    if not is_guarded(name):
        names = ", ".join((*TLS_VARIABLES, *CODE_VARIABLES))
        return f"it is not one of the variables the guard can allow ({names}, LD_*, DYLD_*)"
    if not value:
        return "it needs a value"
    if why := never_allowed(name, value):
        return f"it {why}; this cannot be allowed"
    if name.upper() not in _NOT_PATHS and npm_key(name) != "node-options":
        members = [
            member
            for piece in re.split(r"[;\s]+", value)
            for member in ([piece] if sys.platform == "win32" and _WINDOWS_PATH.match(piece) else piece.split(":"))
        ]
        if not all(m and _absolute(m) for m in members):
            return f"every path in it must be absolute, not {value!r}"
    elif bad := _relative_file_in(value):
        return f"it names {bad!r} by a relative path, which the repository the tool runs in can supply"
    return None


_FALSE = frozenset({"", "0", "false", "no", "off"})
_CERT_STORE_SOURCES = frozenset({"bundled", "system"})


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


def _host_of(proxy: str) -> str:
    """The host a proxy value names, without its credentials or port (``host:port`` with no scheme is accepted)."""
    text = proxy.strip()
    try:
        parts = urlsplit(text if "://" in text else "//" + text)
        return (parts.hostname or "").lower()
    except ValueError:
        return ""


def _refuse(name: str, why: str, fix: str) -> LoginError:
    return LoginError(f"{name} {why}, so no token is printed; {fix}", code="environment_refused")


def check_environment(env: Mapping[str, str], issuer: str, config: Config) -> None:
    """Refuse to print a token when the environment would steer it to another server or expose it in transit.

    This reads the helper's own environment only. A flag on the tool's command line, a settings file the tool did not
    load into the environment, or a base URL the tool holds in memory is not visible here.
    """
    allow = config.allowed()
    door = resource_for(issuer)
    origins = frozenset({issuer.lower()})
    for name in BASE_URL_VARIABLES:
        value = env.get(name, "").strip()
        if value and not _is_llm_url(value, origins):
            shown = urlsplit(value)
            where = f"{shown.scheme}://{shown.hostname}" if shown.hostname else "a value that is not an address"
            raise _refuse(
                name, f"points at {where}, not at this project's {LLM_PATH} door", f"set it to {door} or unset it"
            )
    for name in PROVIDER_URL_VARIABLES:
        if env.get(name, "").strip():
            raise _refuse(name, "sends the request to another provider's endpoint, not to the /llm door", "unset it")
    for name in PROVIDER_SWITCHES:
        if env.get(name, "").strip().lower() not in _FALSE:
            raise _refuse(name, "switches the tool to another provider", "unset it")
    for name in PROXY_VARIABLES:
        value = env.get(name, "").strip()
        host = _host_of(value) if value else ""
        if value and host not in allow.proxies:
            raise _refuse(
                name,
                f"names the proxy {host or 'that is not an address'}, which could read the token",
                f"unset it, or allow that host once with: authfi login --allow-proxy {host or 'HOST'}",
            )
    for name in (*_NEVER_NAMES, *sorted(n for n in env if _is_node_options(n))):
        if why := never_allowed(
            name, env.get(name, "")
        ):  # before the allowances: a hand-edited login.json cannot lift it
            raise _refuse(name, why, "unset it; this cannot be allowed")
    for name in sorted(n for n in env if is_guarded(n)):
        value = env[name]  # as written: an allowance matches byte for byte
        if not value or allow.env.get(name) == value:
            continue
        if name == "CLAUDE_CODE_CERT_STORE" and {part.strip() for part in value.split(",")} <= _CERT_STORE_SOURCES:
            continue  # a subset of the built-in sources only narrows what is trusted
        raise _refuse(
            name,
            "is read by the dynamic loader, OpenSSL or the shell before this check runs, so code it names may already "
            "have run (this refusal is a second line of defence)"
            if name in _READ_BEFORE_THE_GUARD or name.startswith(CODE_PREFIXES)
            else "loads code or changes which certificates the tool trusts",
            f"unset it, or allow exactly this value once with: authfi login --allow-env {name}=<value>",
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
    check_environment(rt.env, issuer, config)
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


def logout(rt: Runtime, issuer_arg: str | None, *, forget_allow: bool = False) -> int:
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
        if forget_allow:
            config.forget_allow()
    if failure is not None:
        raise LoginError(
            f"the stored tokens were deleted but the server did not confirm the revocation ({failure}); "
            "the refresh token stays valid there until it expires",
            code="revocation_failed",
        )
    rt.out(f"signed out of {issuer}")
    return 0
