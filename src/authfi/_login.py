"""A person's own sign-in from a command line: ``authfi login``, ``authfi token --resource llm``, ``authfi logout``.

Claude Code (and any tool that takes its credential from a helper command) reaches the project's ``/llm`` door as
the person, not as an agent. The person signs in once in a browser; after that a helper prints a fresh access token.

* The client is public and registered once per machine (RFC 7591); its redirect is the loopback address with any
  port (RFC 8252 section 7.3), served by a one-shot listener that exits after the first callback.
* The refresh token lives in the operating system's credential store (the ``authfi[login]`` extra, ``keyring``) and
  nowhere else. With no usable store nothing is signed in and nothing is written: there is no plaintext fallback.
  The access token is kept there too, only until shortly before it expires. What goes to a file is not secret:
  the issuer and the client id.
* The server rotates a refresh token on every use and a second use of a spent one ends the whole family, so two
  helpers running at once (two editor windows) take a cross-process lock around the refresh, and the second finds
  the first one's token in the store instead of spending the old one again.

The protocol is the SDK's own: PKCE, ``state``, ``iss`` and the ID token are checked by :class:`~authfi.signin.WebApp`.
"""

from __future__ import annotations

import base64
import contextlib
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
from urllib.parse import parse_qsl, urlsplit

import httpx

from authfi._protocol.register import parse_registration, registration_request
from authfi._protocol.verify import unverified_claims
from authfi.client import AuthFI
from authfi.errors import AuthFIError, ConfigurationError, NotSupportedError
from authfi.secret import Secret
from authfi.signin import WebApp
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


class KeyringVault:
    """The :mod:`keyring` store (Credential Manager, Keychain, Secret Service): one entry per issuer and name."""

    def __init__(self, keyring_module: Any) -> None:
        self._keyring = keyring_module

    @staticmethod
    def _user(issuer: str, name: str) -> str:
        return f"{issuer}#{name}"

    def get(self, issuer: str, name: str) -> str | None:
        value = self._keyring.get_password("authfi", self._user(issuer, name))
        return value if isinstance(value, str) and value else None

    def set(self, issuer: str, name: str, value: str) -> None:
        self._keyring.set_password("authfi", self._user(issuer, name), value)

    def delete(self, issuer: str, name: str) -> None:
        from keyring.errors import PasswordDeleteError

        with contextlib.suppress(PasswordDeleteError):
            self._keyring.delete_password("authfi", self._user(issuer, name))


_NO_STORE = (
    "authfi login keeps your refresh token in the operating system's credential store and will not write it to a "
    "file: {why}. Install the keyring package (pip install 'authfi[login] @ "
    "git+https://github.com/authfi/authfi-python-sdk@main'); on Linux it needs a Secret Service such as GNOME "
    "Keyring or KWallet."
)


def open_vault() -> Vault:
    """The credential store, or a :class:`LoginError` saying why there is none. Called before anything is written."""
    try:
        import keyring
        import keyring.backends.fail
        import keyring.backends.null
    except ImportError as exc:
        raise LoginError(
            _NO_STORE.format(why="the keyring package is not installed"), code="no_credential_store"
        ) from exc
    try:
        backend = keyring.get_keyring()
        unusable = isinstance(backend, (keyring.backends.fail.Keyring, keyring.backends.null.Keyring))
    except Exception as exc:  # a backend that fails while it is being chosen is no backend
        raise LoginError(_NO_STORE.format(why="no keyring backend started"), code="no_credential_store") from exc
    if unusable:
        raise LoginError(_NO_STORE.format(why="this machine has no keyring backend"), code="no_credential_store")
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

    def save(self, issuer: str, *, client_id: str | None = None, default: bool = True) -> None:
        data = self._read()
        clients = _object(data.get("clients"))
        if client_id is not None:
            clients[issuer] = {"client_id": client_id}
        data["clients"] = clients
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


def login(
    rt: Runtime,
    issuer_arg: str | None,
    *,
    browser: bool = True,
    reregister: bool = False,
    timeout: float = CALLBACK_TIMEOUT,
) -> int:
    """Sign the person in for the project's ``/llm`` door and keep the refresh token in the credential store."""
    config = rt.config()
    issuer = resolve_issuer(issuer_arg, rt.env, config)
    receiver = _Receiver()
    try:
        with AuthFI(issuer, http_client=rt.http_client) as af:
            metadata = af.metadata()
            if metadata.authorization_endpoint is None:
                raise LoginError("this project's discovery names no authorization endpoint", code="not_supported")
            client_id = None if reregister else config.client_id(issuer)
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
            signed_in = web.finish_sign_in(params, pending)
    finally:
        receiver.server_close()
    tokens = signed_in.tokens
    if tokens.refresh_token is None:
        raise LoginError(
            "the server issued no refresh token, so the sign-in could not be kept", code="no_refresh_token"
        )
    if tokens.scope and "llm:invoke" not in tokens.scope:
        raise LoginError("the server did not grant llm:invoke; ask your administrator for access to the /llm door")
    _check_audience(tokens.access_token, resource_for(issuer))
    rt.vault.set(issuer, "refresh", tokens.refresh_token.reveal())
    _store_access(rt, issuer, tokens)
    config.save(issuer, client_id=client_id)
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


def access_token(rt: Runtime, issuer_arg: str | None, *, min_validity: float = DEFAULT_MIN_VALIDITY) -> str:
    """A person access token for ``/llm``: the cached one while it lasts, otherwise a refresh under the lock."""
    config = rt.config()
    issuer = resolve_issuer(issuer_arg, rt.env, config)
    cached = _cached(rt, issuer, min_validity)
    if cached is not None:
        return cached
    with file_lock(config.directory / "refresh.lock"):
        cached = _cached(rt, issuer, min_validity)  # another process may have renewed it while this one waited
        if cached is not None:
            return cached
        refresh = rt.vault.get(issuer, "refresh")
        client_id = config.client_id(issuer)
        if refresh is None or client_id is None:
            raise LoginError(f"not signed in to {issuer}: run authfi login --issuer {issuer}", code="not_signed_in")
        with AuthFI(issuer, http_client=rt.http_client) as af:
            web = _web_app(af, client_id, REGISTERED_REDIRECT)
            held = _held(refresh)
            tokens = web.refresh(held)
        if tokens.refresh_token is None:  # unreachable: from_response carries the previous one forward
            raise LoginError("the server answered no refresh token", code="no_refresh_token")
        rotated = tokens.refresh_token.reveal()
        if rotated != refresh:
            try:
                rt.vault.set(issuer, "refresh", rotated)
            except Exception as exc:
                raise LoginError(
                    "the server renewed the sign-in but the credential store refused it; run authfi login again",
                    code="store_failed",
                ) from exc
        _check_audience(tokens.access_token, resource_for(issuer))
        _store_access(rt, issuer, tokens)
        return tokens.access_token.reveal()


def logout(rt: Runtime, issuer_arg: str | None) -> int:
    """Revoke the refresh token at the server, then delete what is stored (also when the revocation fails)."""
    config = rt.config()
    issuer = resolve_issuer(issuer_arg, rt.env, config)
    refresh = rt.vault.get(issuer, "refresh")
    failure: AuthFIError | None = None
    client_id = config.client_id(issuer)
    if refresh is not None and client_id is not None:
        try:
            with AuthFI(issuer, http_client=rt.http_client) as af:
                held = _held(refresh)
                _web_app(af, client_id, REGISTERED_REDIRECT).revoke(held)
        except (AuthFIError, NotSupportedError) as error:
            failure = error
    rt.vault.delete(issuer, "refresh")
    rt.vault.delete(issuer, "access")
    if failure is not None:
        raise LoginError(
            f"the stored tokens were deleted but the server did not confirm the revocation ({failure.message}); "
            "the refresh token stays valid there until it expires",
            code="revocation_failed",
        )
    rt.out(f"signed out of {issuer}")
    return 0
