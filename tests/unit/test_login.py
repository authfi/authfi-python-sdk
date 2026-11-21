"""``authfi login``, ``authfi token --resource llm`` and ``authfi logout`` against a faithful fake authorization server.

The fake registers public clients, checks PKCE S256, rotates refresh tokens and revokes the whole family when a
spent one comes back (as identity does), so a missing lock or a lost rotation fails here as it would live.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import http.client
import json
import os
import secrets
import shlex
import subprocess
import sys
import threading
import time
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit

import httpx
import keyring
import keyring.backend
import keyring.errors
import pytest

from authfi import _bus, _login
from authfi import cli as cli_module
from authfi._login import KeyringVault, LoginError, Runtime
from authfi._protocol.http import HttpResponse
from authfi._protocol.register import parse_registration
from authfi.errors import InvalidCallback, OAuthError
from tests.support import Platform, b64url, json_response

ISSUER = "https://acme.authfi.io"
LLM = ISSUER + "/llm"
REGISTER = "/oauth/register"


def slot(config_dir: Path) -> str:
    return _login.slot(ISSUER, config_dir)


class MemoryVault:
    """The credential store of a test: a dict, with a switch for a store that refuses the access token."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.refuse_access = False
        self.refuse_refresh = False

    def get(self, issuer: str, name: str) -> str | None:
        return self.values.get((issuer, name))

    def set(self, issuer: str, name: str, value: str) -> None:
        if self.refuse_access and name == "access":
            raise OSError("value too long for this store")
        if self.refuse_refresh and name == "refresh":
            raise OSError("the store is locked")
        self.values[(issuer, name)] = value

    def delete(self, issuer: str, name: str) -> None:
        self.values.pop((issuer, name), None)


class FakeAS:
    """Registration, authorize, token (code and refresh) and revocation of one project."""

    def __init__(self, platform: Platform) -> None:
        self.platform = platform
        platform.discovery["registration_endpoint"] = platform.issuer + REGISTER
        self.registrations: list[dict[str, Any]] = []
        self.redirects_echoed: list[str] | None = None
        self.codes: dict[str, dict[str, str]] = {}
        self.clients: dict[str, dict[str, Any]] = {}
        self.token_forms: list[dict[str, str]] = []
        self.refreshes = 0
        self.live: set[str] = set()
        self.family_revoked = False
        self.refresh_hold: threading.Event | None = None
        self.refresh_error: dict[str, Any] | None = None
        self.refresh_status = 400
        self.refresh_exception: Exception | None = None
        self.login_refresh: str | None = None
        self.dead_clients: set[str] = set()
        self.all_clients_dead = False
        self.exchange_error: dict[str, Any] | None = None
        self.revoked: list[dict[str, str]] = []
        self.revoke_status = 200
        self.issued = 0
        self.rebinding_status = 0
        self.expires_in = 900
        self.aud: list[str] = [platform.issuer + "/llm"]
        platform.route("POST", REGISTER, self.register)
        platform.route("POST", "/oauth/token", self.token)
        platform.route("POST", "/oauth/token/revoke", self.revoke)

    # -- endpoints
    def register(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.registrations.append(body)
        client_id = "dyn_" + secrets.token_hex(6)
        self.clients[client_id] = body
        answer = {"client_id": client_id, "redirect_uris": self.redirects_echoed or body["redirect_uris"]}
        return json_response(answer, 201)

    def authorize(self, url: str) -> tuple[dict[str, str], str]:
        """The person signs in at ``url``: the fake keeps the request and returns it with a code."""
        query = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        assert query["client_id"] in self.clients, "an unregistered client"
        redirect = query["redirect_uri"]
        registered = self.clients[query["client_id"]]["redirect_uris"]
        assert any(redirect.startswith(r) for r in registered), "a redirect that was not registered"
        code = "code-" + secrets.token_hex(4)
        self.codes[code] = query
        return query, code

    def _access(self, client_id: str) -> str:
        now = int(time.time())
        self.issued += 1
        return self.platform.keys.sign(
            {
                "iss": self.platform.issuer,
                "sub": "usr_ada",
                "aud": self.aud,
                "exp": now + self.expires_in,
                "iat": now,
                "client_id": client_id,
                "scope": "openid llm:invoke offline_access",
                "jti": f"jti-{self.issued}",
            },
            alg="RS256",
        )

    def _remember(self, refresh: str) -> str:
        self.login_refresh = refresh
        return refresh

    def _rotate(self) -> str:
        fresh = "rt-" + secrets.token_hex(6)
        self.live.add(fresh)
        return fresh

    def token(self, request: httpx.Request) -> httpx.Response:
        form = dict(parse_qsl(request.content.decode()))
        self.token_forms.append(form)
        if form["grant_type"] == "authorization_code":
            if self.exchange_error is not None:
                return json_response(self.exchange_error, 400)
            if self.all_clients_dead or form["client_id"] in self.dead_clients:
                return json_response({"error": "invalid_client", "error_description": "unknown client"}, 401)
            asked = self.codes.pop(form["code"], None)
            if asked is None or asked["client_id"] != form["client_id"]:
                return json_response({"error": "invalid_grant"}, 400)
            if b64url(hashlib.sha256(form["code_verifier"].encode()).digest()) != asked["code_challenge"]:
                return json_response({"error": "invalid_grant", "error_description": "PKCE"}, 400)
            if form["redirect_uri"] != asked["redirect_uri"]:
                return json_response({"error": "invalid_grant"}, 400)
            access = self._access(form["client_id"])
            now = int(time.time())
            id_token = self.platform.keys.sign(
                {
                    "iss": self.platform.issuer,
                    "sub": "usr_ada",
                    "aud": form["client_id"],
                    "exp": now + 300,
                    "iat": now,
                    "auth_time": now,
                    "acr": "urn:authfi:acr:pwd",
                    "nonce": asked["nonce"],
                    "at_hash": b64url(hashlib.sha256(access.encode()).digest()[:16]),
                },
                alg="RS256",
                typ="JWT",
            )
            return json_response(
                {
                    "access_token": access,
                    "token_type": "Bearer",
                    "expires_in": self.expires_in,
                    "refresh_token": self._remember(self._rotate()),
                    "id_token": id_token,
                    "scope": "openid llm:invoke offline_access",
                }
            )
        assert form["grant_type"] == "refresh_token"
        self.refreshes += 1
        if self.refresh_hold is not None:
            self.refresh_hold.wait(5)
        if self.refresh_exception is not None:
            raise self.refresh_exception  # the answer is lost after the server has acted
        if self.refresh_error is not None:
            return json_response(self.refresh_error, self.refresh_status)
        presented = form["refresh_token"]
        if self.family_revoked or presented not in self.live:
            self.family_revoked = True  # reuse of a spent token ends the family
            return json_response({"error": "invalid_grant"}, 400)
        self.live.discard(presented)
        return json_response(
            {
                "access_token": self._access(form["client_id"]),
                "token_type": "Bearer",
                "expires_in": self.expires_in,
                "refresh_token": self._rotate(),
                "scope": "openid llm:invoke offline_access",
            }
        )

    def revoke(self, request: httpx.Request) -> httpx.Response:
        form = dict(parse_qsl(request.content.decode()))
        self.revoked.append(form)
        if self.revoke_status == 200:
            self.live.discard(form["token"])
        return httpx.Response(self.revoke_status)


@pytest.fixture
def platform() -> Platform:
    return Platform(issuer=ISSUER)


@pytest.fixture
def server(platform: Platform) -> FakeAS:
    return FakeAS(platform)


@pytest.fixture
def vault() -> MemoryVault:
    return MemoryVault()


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    return tmp_path / "config"


def runtime(
    platform: Platform,
    vault: MemoryVault,
    config_dir: Path,
    *,
    opener: Callable[[str], object] | None = None,
    clock: Callable[[], float] = time.time,
    env: dict[str, str] | None = None,
) -> Runtime:
    out: list[str] = []
    return Runtime(
        env={"AUTHFI_CONFIG_DIR": str(config_dir), **(env or {})},
        vault=vault,
        http_client=platform.client(),
        clock=clock,
        open_browser=opener or (lambda url: False),
        out=out.append,
        err=lambda text: None,
    )


def browser(
    server: FakeAS, *, state: str | None = None, host: str | None = None, extra: dict[str, str] | None = None
) -> Callable[[str], object]:
    """A browser: signs in at the server and returns to the listener, from another thread (the login is waiting)."""

    def open_url(url: str) -> bool:
        query, code = server.authorize(url)
        port = urlsplit(query["redirect_uri"]).port
        callback = {"code": code, "state": state or query["state"], "iss": server.platform.issuer, **(extra or {})}

        def go() -> None:
            if host is not None:  # a page that reached this port under another name
                bad = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                bad.request("GET", "/?" + urlencode(callback), headers={"Host": host})
                server.rebinding_status = bad.getresponse().status
                bad.close()
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            conn.request("GET", "/?" + urlencode(callback))
            conn.getresponse().read()
            conn.close()

        threading.Thread(target=go, daemon=True).start()
        return True

    return open_url


def signed_in(platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path) -> None:
    rt = runtime(platform, vault, config_dir, opener=browser(server))
    assert _login.login(rt, ISSUER, timeout=10) == 0


# --- login -------------------------------------------------------------------------------------------------


def test_login_registers_once_and_asks_for_the_llm_door_exactly(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    seen: list[str] = []
    drive = browser(server)

    def opener(url: str) -> object:
        seen.append(url)
        return drive(url)

    rt = runtime(platform, vault, config_dir, opener=opener)
    assert _login.login(rt, ISSUER, timeout=10) == 0

    (registration,) = server.registrations
    assert registration["redirect_uris"] == ["http://127.0.0.1"]
    assert registration["token_endpoint_auth_method"] == "none"
    assert registration["grant_types"] == ["authorization_code", "refresh_token"]
    assert registration["scope"] == "openid llm:invoke offline_access"

    query = {k: v[0] for k, v in parse_qs(urlsplit(seen[0]).query).items()}
    assert urlsplit(seen[0]).path == "/oauth/authorize"
    assert query["resource"] == LLM
    assert query["scope"] == "openid llm:invoke offline_access"
    assert query["code_challenge_method"] == "S256"
    assert len(query["state"]) >= 43
    redirect = urlsplit(query["redirect_uri"])
    assert (redirect.scheme, redirect.hostname, redirect.path) == ("http", "127.0.0.1", "/")
    assert redirect.port and redirect.port > 0

    code_form = server.token_forms[0]
    assert code_form["client_id"] == query["client_id"] and code_form["redirect_uri"] == query["redirect_uri"]
    assert vault.get(slot(config_dir), "refresh") in server.live

    # the second sign-in reuses the client registered for this machine
    assert _login.login(runtime(platform, vault, config_dir, opener=browser(server)), ISSUER, timeout=10) == 0
    assert len(server.registrations) == 1
    assert server.token_forms[-1]["client_id"] == query["client_id"]
    # a forced re-registration makes another
    assert (
        _login.login(runtime(platform, vault, config_dir, opener=browser(server)), ISSUER, reregister=True, timeout=10)
        == 0
    )
    assert len(server.registrations) == 2


def test_login_keeps_no_secret_in_any_file(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    files = [p for p in config_dir.rglob("*") if p.is_file()]
    assert {p.name for p in files} == {"login.json", "refresh.lock"}
    text = (config_dir / "login.json").read_text()
    data = json.loads(text)
    assert data["default_issuer"] == ISSUER and set(data["clients"][ISSUER]) == {"client_id"}
    assert (vault.get(slot(config_dir), "refresh") or "x") not in text
    assert (json.loads(vault.get(slot(config_dir), "access") or "{}").get("token") or "x") not in text


def test_a_callback_with_another_state_is_refused_and_stores_nothing(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    rt = runtime(platform, vault, config_dir, opener=browser(server, state="not-the-state"))
    with pytest.raises(InvalidCallback) as caught:
        _login.login(rt, ISSUER, timeout=10)
    assert caught.value.code == "callback_state_mismatch"
    assert vault.values == {}
    assert not (config_dir / "login.json").exists()
    assert server.token_forms == []  # the code was never redeemed


def test_a_callback_from_another_issuer_is_refused(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    rt = runtime(platform, vault, config_dir, opener=browser(server, extra={"iss": "https://evil.example"}))
    with pytest.raises(InvalidCallback) as caught:
        _login.login(rt, ISSUER, timeout=10)
    assert caught.value.code == "callback_iss_mismatch"
    assert vault.values == {}


def test_a_request_reaching_the_port_under_another_host_name_is_refused_and_does_not_end_the_login(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    rt = runtime(platform, vault, config_dir, opener=browser(server, host="evil.example"))
    assert _login.login(rt, ISSUER, timeout=10) == 0
    assert server.rebinding_status == 403
    assert vault.get(slot(config_dir), "refresh") is not None


def test_a_registration_that_binds_a_redirect_away_from_this_machine_is_refused(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    server.redirects_echoed = ["https://evil.example/cb"]
    rt = runtime(platform, vault, config_dir, opener=browser(server))
    with pytest.raises(OAuthError):
        _login.login(rt, ISSUER, timeout=10)
    assert vault.values == {} and not config_dir.exists()


def test_a_registration_answer_that_issues_a_secret_is_refused() -> None:
    answer = HttpResponse(201, (), json.dumps({"client_id": "c", "client_secret": "s"}).encode(), "u", "r")
    with pytest.raises(OAuthError):
        parse_registration(answer, redirect_uris=["http://127.0.0.1"])


def test_a_token_for_another_door_is_not_kept(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    server.aud = [ISSUER + "/mcp"]
    with pytest.raises(LoginError) as caught:
        _login.login(runtime(platform, vault, config_dir, opener=browser(server)), ISSUER, timeout=10)
    assert caught.value.code == "wrong_audience"
    assert vault.get(slot(config_dir), "refresh") is None


def test_the_issuer_must_be_an_https_origin(platform: Platform, vault: MemoryVault, config_dir: Path) -> None:
    rt = runtime(platform, vault, config_dir)
    for bad in ("http://acme.authfi.io", "https://acme.authfi.io/path", "acme"):
        with pytest.raises(LoginError):
            _login.login(rt, bad, timeout=1)
    assert platform.requests == []


# --- the credential store ----------------------------------------------------------------------------------


class _Memory(keyring.backend.KeyringBackend):
    priority = 1  # pyright: ignore[reportAssignmentType]

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.items.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.items[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.items:
            raise keyring.errors.PasswordDeleteError("absent")
        del self.items[(service, username)]


@pytest.fixture(autouse=True)
def _started_isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """pytest is not run with ``-I``; the tests that exercise the check say otherwise."""
    monkeypatch.setattr(cli_module, "_isolated", lambda: True)


@pytest.fixture
def use_store(monkeypatch: pytest.MonkeyPatch) -> Callable[[type], None]:
    """Make ``open_vault`` build the given class, as it builds the operating system's own store."""

    def use(kind: type) -> None:
        name = f"{kind.__module__}.{kind.__qualname__}"

        def classes(platform: str) -> list[tuple[str, str]]:
            return [(kind.__module__, kind.__qualname__)]

        monkeypatch.setattr(_login, "_backend_classes", classes)
        monkeypatch.setattr(_login, "SECURE_BACKENDS", _login.SECURE_BACKENDS | {name})

    return use


def test_without_a_keyring_backend_login_refuses_and_writes_nothing(
    platform: Platform,
    server: FakeAS,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def nothing(platform: str) -> list[tuple[str, str]]:
        return []

    monkeypatch.setattr(_login, "_backend_classes", nothing)
    monkeypatch.setenv("AUTHFI_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(cli_module, "_http_client", lambda: platform.client())
    for argv in (["login", "--issuer", ISSUER], ["token", "--resource", "llm", "--issuer", ISSUER]):
        assert cli_module.main(argv) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "credential store" in captured.err and "will not write it to a file" in captured.err
    assert platform.requests == []  # refused before any call
    assert not config_dir.exists()  # and before any file


def test_the_keyring_vault_round_trips_through_the_named_backend(use_store: Callable[[type], None]) -> None:
    use_store(_Memory)
    vault = _login.open_vault()
    assert isinstance(vault, KeyringVault)
    vault.set(ISSUER, "refresh", "rt-1")
    assert vault.get(ISSUER, "refresh") == "rt-1"
    vault.delete(ISSUER, "refresh")
    vault.delete(ISSUER, "refresh")  # deleting what is not there is not an error
    assert vault.get(ISSUER, "refresh") is None


def test_keyring_is_never_asked_which_backend_to_use(
    use_store: Callable[[type], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    import keyring.backend
    import keyring.core

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("keyring chose a backend")

    for owner, name in (
        (keyring, "get_keyring"),
        (keyring.core, "get_keyring"),
        (keyring.core, "init_backend"),
        (keyring.core, "load_config"),
        (keyring.core, "load_env"),
        (keyring.backend, "get_all_keyring"),
    ):
        monkeypatch.setattr(owner, name, forbidden)
    use_store(_Memory)
    assert isinstance(_login.open_vault(), KeyringVault)


@pytest.mark.parametrize("platform_name", ["win32", "darwin", "linux"])
def test_the_classes_named_for_a_platform_are_all_operating_system_stores(platform_name: str) -> None:
    named = _login._backend_classes(platform_name)
    assert named and all(f"{module}.{name}" in _login.SECURE_BACKENDS for module, name in named)


# --- token -------------------------------------------------------------------------------------------------


def test_token_refreshes_stores_the_rotation_and_reuses_the_token_until_it_nears_expiry(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    first_refresh = vault.get(slot(config_dir), "refresh")
    rt = runtime(platform, vault, config_dir)
    # the sign-in left a good access token: no refresh yet
    cached = _login.access_token(rt, None)
    assert server.refreshes == 0

    token = _login.access_token(rt, None, min_validity=1000)  # the 900 s token is too short for this caller
    assert server.refreshes == 1 and token != cached
    assert vault.get(slot(config_dir), "refresh") != first_refresh  # the rotated token replaced the spent one
    assert vault.get(slot(config_dir), "refresh") in server.live
    assert server.token_forms[-1]["grant_type"] == "refresh_token"
    assert "resource" not in server.token_forms[-1]  # a refresh is pinned to its grant's door

    assert _login.access_token(rt, None) == token and server.refreshes == 1  # reused, not refreshed again


def test_token_command_prints_only_the_token(
    platform: Platform,
    server: FakeAS,
    vault: MemoryVault,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    signed_in(platform, server, vault, config_dir)
    monkeypatch.setattr(_login, "open_vault", lambda: vault)
    monkeypatch.setattr(cli_module, "_http_client", lambda: platform.client())
    monkeypatch.setenv("AUTHFI_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("AUTHFI_ISSUER", raising=False)
    assert cli_module.main(["token", "--resource", "llm", "--min-validity", "100000"]) == 0
    out = capsys.readouterr()
    assert out.err == "" and out.out.count("\n") == 1 and out.out.strip().count(".") == 2
    assert server.refreshes == 1  # --min-validity forced a refresh of the cached token


def test_a_failed_refresh_prints_nothing_on_stdout_and_exits_1(
    platform: Platform,
    server: FakeAS,
    vault: MemoryVault,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    signed_in(platform, server, vault, config_dir)
    server.refresh_error = {"error": "invalid_grant", "error_description": "the session ended"}
    monkeypatch.setattr(_login, "open_vault", lambda: vault)
    monkeypatch.setattr(cli_module, "_http_client", lambda: platform.client())
    monkeypatch.setenv("AUTHFI_CONFIG_DIR", str(config_dir))
    assert cli_module.main(["token", "--resource", "llm", "--issuer", ISSUER, "--min-validity", "100000"]) == 1
    out = capsys.readouterr()
    assert out.out == ""
    assert (
        "authfi:" in out.err and "invalid_grant" in out.err.replace("InvalidGrant", "invalid_grant")
    ) or "session ended" in out.err
    assert "rt-" not in out.err


def test_not_signed_in_says_how_to_sign_in(platform: Platform, vault: MemoryVault, config_dir: Path) -> None:
    with pytest.raises(LoginError) as caught:
        _login.access_token(runtime(platform, vault, config_dir), ISSUER)
    assert "authfi login" in caught.value.message and caught.value.code == "not_signed_in"


def test_two_concurrent_helpers_make_one_refresh_call(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    server.expires_in = 100  # the sign-in's access token is already too short to reuse
    signed_in(platform, server, vault, config_dir)
    server.expires_in = 900
    server.refresh_hold = threading.Event()
    results: list[str] = []
    errors: list[BaseException] = []

    def helper() -> None:
        try:
            results.append(_login.access_token(runtime(platform, vault, config_dir), ISSUER))
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=helper) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.4)  # both are in: one holds the lock inside the refresh, the other waits on it
    server.refresh_hold.set()
    for t in threads:
        t.join(10)
    assert errors == []
    assert server.refreshes == 1
    assert len(results) == 2 and results[0] == results[1]
    assert not server.family_revoked


def test_helpers_are_serialised_even_when_the_store_cannot_hold_the_access_token(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    vault.refuse_access = True
    vault.delete(slot(config_dir), "access")
    server.refresh_hold = threading.Event()
    errors: list[BaseException] = []

    def helper() -> None:
        try:
            _login.access_token(runtime(platform, vault, config_dir), ISSUER)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=helper) for _ in range(2)]
    for t in threads:
        t.start()
    time.sleep(0.4)
    server.refresh_hold.set()
    for t in threads:
        t.join(10)
    # each refreshed once, one after the other, the second with the first's rotated token: no reuse, no revoked family
    assert errors == [] and server.refreshes == 2 and not server.family_revoked


# --- logout ------------------------------------------------------------------------------------------------


def test_logout_revokes_the_refresh_token_then_deletes_what_is_stored(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    refresh = vault.get(slot(config_dir), "refresh")
    assert _login.logout(runtime(platform, vault, config_dir), None) == 0
    assert server.revoked == [
        {"token": refresh, "token_type_hint": "refresh_token", "client_id": server.revoked[0]["client_id"]}
    ]
    assert refresh not in server.live
    assert vault.values == {}
    assert _login.Config(config_dir).client_id(ISSUER)  # the machine's client is kept


def test_logout_deletes_locally_even_when_the_server_did_not_confirm(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    server.revoke_status = 503
    with pytest.raises(LoginError) as caught:
        _login.logout(runtime(platform, vault, config_dir), ISSUER)
    assert caught.value.code == "revocation_failed"
    assert vault.values == {}


# --- review fixes: the credential store, the base URL, the lock, a refresh of unknown outcome ---------------


class _PlainFile(_Memory):
    """Stands in for ``keyrings.alt``'s plaintext file store, which is not installed here."""

    created: ClassVar[list[str]] = []

    def __init__(self) -> None:
        type(self).created.append("created")
        super().__init__()


def test_a_class_that_is_not_an_operating_system_store_is_refused_without_being_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def classes(platform: str) -> list[tuple[str, str]]:
        return [(__name__, "_PlainFile")]

    _PlainFile.created.clear()
    monkeypatch.setattr(_login, "_backend_classes", classes)
    with pytest.raises(LoginError) as caught:
        _login.open_vault()
    assert caught.value.code == "no_credential_store" and "will not write it to a file" in caught.value.message
    assert _PlainFile.created == []


def test_a_keyring_backend_named_by_the_environment_is_never_built(monkeypatch: pytest.MonkeyPatch) -> None:
    _PlainFile.created.clear()
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", f"{__name__}._PlainFile")
    with contextlib.suppress(LoginError):  # on a machine with no store the answer is the refusal
        _login.open_vault()
    assert _PlainFile.created == []


def _keyring_config_planted(tmp_path: Path) -> tuple[dict[str, str], Path]:
    """An environment whose config folders hold a ``keyringrc.cfg`` naming a hostile backend that writes a marker."""
    marker = tmp_path / "hostile-ran.txt"
    evil = tmp_path / "evil"
    evil.mkdir()
    (evil / "hostile_backend.py").write_text(
        "import pathlib\n"
        f"pathlib.Path({str(marker)!r}).write_text('ran')\n"
        "from keyring.backend import KeyringBackend\n"
        "class Hostile(KeyringBackend):\n"
        "    priority = 99\n"
        "    def get_password(self, service, username): return 'x'\n"
        "    def set_password(self, service, username, password): pass\n"
        "    def delete_password(self, service, username): pass\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    home.mkdir()
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_CONFIG_HOME": str(home),
        "LOCALAPPDATA": str(home),
        "APPDATA": str(home),
        "ProgramData": str(home),
    }
    env.pop("PYTHONPATH", None)
    env.pop("PYTHON_KEYRING_BACKEND", None)
    root = subprocess.run(
        [sys.executable, "-c", "import keyring.util.platform_ as p; print(p.config_root())"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    Path(root).mkdir(parents=True, exist_ok=True)
    (Path(root) / "keyringrc.cfg").write_text(
        f"[backend]\ndefault-keyring=hostile_backend.Hostile\nkeyring-path={evil}\n", encoding="utf-8"
    )
    return env, marker


def test_a_planted_keyringrc_with_a_keyring_path_is_never_read(tmp_path: Path) -> None:
    env, marker = _keyring_config_planted(tmp_path)
    subprocess.run(
        [sys.executable, "-c", "import keyring; keyring.get_keyring()"], env=env, capture_output=True, check=False
    )
    assert marker.exists(), "the control failed: keyring itself did not read the planted file"
    marker.unlink()
    code = "from authfi import _login\ntry:\n    _login.open_vault()\nexcept Exception:\n    pass\n"
    subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, check=True)  # noqa: S603
    assert not marker.exists()


def _helper(args: list[str], tmp_path: Path, extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "AUTHFI_CONFIG_DIR": str(tmp_path / "cfg"), **extra}
    return subprocess.run(  # noqa: S603 - this interpreter, on code written above
        [sys.executable, *args, "-m", "authfi", "token", "--resource", "llm", "--issuer", ISSUER],
        env=env,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        check=False,
    )


def _planted_pythonpath(tmp_path: Path) -> tuple[dict[str, str], Path]:
    marker = tmp_path / "sitecustomize-ran.txt"
    planted = tmp_path / "planted"
    planted.mkdir()
    (planted / "sitecustomize.py").write_text(
        f"import pathlib\npathlib.Path({str(marker)!r}).write_text('ran')\n", encoding="utf-8"
    )
    return {"PYTHONPATH": str(planted)}, marker


def test_the_helper_started_without_isolation_refuses_and_says_how_to_start_it(tmp_path: Path) -> None:
    extra, _marker = _planted_pythonpath(tmp_path)
    done = _helper([], tmp_path, extra)
    assert done.returncode == 1 and done.stdout == ""
    assert "isolated mode" in done.stderr and "-I -m authfi token --resource llm" in done.stderr
    assert sys.executable in done.stderr


def test_the_helper_started_isolated_runs_no_code_from_pythonpath(tmp_path: Path) -> None:
    extra, marker = _planted_pythonpath(tmp_path)
    done = _helper(["-I"], tmp_path, extra)
    assert not marker.exists()  # -I ignored PYTHONPATH
    assert "isolated mode" not in done.stderr  # it got past the startup check (and then found nobody signed in)
    assert done.stdout == ""


def test_the_startup_check_runs_before_the_credential_store_is_opened(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def opened() -> None:
        raise AssertionError("the store was opened")

    monkeypatch.setattr(cli_module, "_isolated", lambda: False)
    monkeypatch.setattr(_login, "open_vault", opened)
    assert cli_module.main(["token", "--resource", "llm", "--issuer", ISSUER]) == 1
    assert "isolated mode" in capsys.readouterr().err


def test_the_agents_own_token_command_does_not_need_isolation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_module, "_isolated", lambda: False)
    cli_module.main(["token"])
    assert "isolated mode" not in capsys.readouterr().err


def test_the_helper_command_is_this_interpreter_by_absolute_path_with_i() -> None:
    command = _login.helper_command()
    quoted = f'"{sys.executable}"' if sys.platform == "win32" else shlex.quote(sys.executable)
    assert command == f"{quoted} -I -m authfi token --resource llm"
    assert Path(sys.executable).is_absolute()


def test_login_prints_the_helper_line_to_put_in_claude_codes_settings(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    lines: list[str] = []
    rt = dataclasses.replace(runtime(platform, vault, config_dir, opener=browser(server)), out=lines.append)
    assert _login.login(rt, ISSUER, timeout=10) == 0
    helper = next(line for line in lines if "apiKeyHelper" in line)
    assert json.dumps(_login.helper_command()) in helper and "-I -m authfi token --resource llm" in helper


class _SmallBlob(_Memory):
    """Windows Credential Locker: a blob over 2560 bytes is refused, and keyring stores UTF-16."""

    def set_password(self, service: str, username: str, password: str) -> None:
        if len(password.encode("utf-16-le")) > 2560:
            raise OSError("The stub received bad data")
        super().set_password(service, username, password)  # pyright: ignore[reportUnknownMemberType]


def test_a_refresh_token_longer_than_the_windows_blob_limit_is_kept_in_pieces(
    use_store: Callable[[type], None],
) -> None:
    use_store(_SmallBlob)
    vault = cast("KeyringVault", _login.open_vault())
    backend = cast("_SmallBlob", vault._keyring)  # pyright: ignore[reportPrivateUsage]
    long = "rt-" + "x7" * 2500
    vault.set(ISSUER, "refresh", long)
    assert vault.get(ISSUER, "refresh") == long
    assert len(backend.items) > 2  # head plus pieces
    vault.set(ISSUER, "refresh", "short")  # a shorter value leaves no stale piece behind
    assert vault.get(ISSUER, "refresh") == "short" and len(backend.items) == 2
    vault.set(ISSUER, "refresh", long)
    del backend.items[("authfi", f"{ISSUER}#refresh#2")]  # a lost piece reads as no value, never a truncated token
    assert vault.get(ISSUER, "refresh") is None
    vault.delete(ISSUER, "refresh")
    assert backend.items == {}


GOOD_BASE_URLS = [
    LLM,
    LLM + "/",
    LLM + "/v1",
    "HTTPS://ACME.authfi.io/llm",
    "https://acme.authfi.io:443/llm",
]
BAD_BASE_URLS = [
    "https://evil.example/llm",
    "https://acme.authfi.io.evil.example/llm",
    "https://evil.example/https://acme.authfi.io/llm",
    "http://acme.authfi.io/llm",
    "https://user:pw@acme.authfi.io/llm",
    "https://acme.authfi.io@evil.example/llm",
    "https://acme.authfi.io:8443/llm",
    "https://acme.authfi.io",
    "https://acme.authfi.io/mcp",
    "https://acme.authfi.io/llmx",
    "https://acme.authfi.io/llm/../x",
    "https://acme.authfi.io/llm/%2e%2e/x",
    "https://acme.authfi.io/llm?next=https://evil.example",
    "https://acme.authfi.io//llm",
    "not a url",
]


@pytest.mark.parametrize("name", list(_login.BASE_URL_VARIABLES))
@pytest.mark.parametrize("value", BAD_BASE_URLS)
def test_no_token_is_given_when_the_tools_base_url_is_not_this_projects_llm_door(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str, value: str
) -> None:
    signed_in(platform, server, vault, config_dir)  # a valid cached token exists: it must still not be printed
    with pytest.raises(LoginError) as caught:
        _login.access_token(runtime(platform, vault, config_dir, env={name: value}), None)
    assert caught.value.code == "environment_refused" and name in caught.value.message
    assert "pw" not in caught.value.message.replace("pw@", "") and "user:" not in caught.value.message
    assert server.refreshes == 0


@pytest.mark.parametrize("value", GOOD_BASE_URLS)
def test_this_projects_llm_door_as_the_base_url_gets_its_token(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, value: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    assert _login.access_token(runtime(platform, vault, config_dir, env={"ANTHROPIC_BASE_URL": value}), None)


def test_the_token_command_prints_nothing_when_the_base_url_points_elsewhere(
    platform: Platform,
    server: FakeAS,
    vault: MemoryVault,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    signed_in(platform, server, vault, config_dir)
    monkeypatch.setattr(_login, "open_vault", lambda: vault)
    monkeypatch.setattr(cli_module, "_http_client", lambda: platform.client())
    monkeypatch.setenv("AUTHFI_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://evil.example/llm")
    assert cli_module.main(["token", "--resource", "llm", "--issuer", ISSUER]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "ANTHROPIC_BASE_URL" in out.err and "evil.example" in out.err


def test_two_config_directories_hold_two_sign_ins_and_never_race_over_one_token(
    platform: Platform, server: FakeAS, vault: MemoryVault, tmp_path: Path
) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    signed_in(platform, server, vault, first)
    signed_in(platform, server, vault, second)
    assert slot(first) != slot(second)
    assert vault.get(slot(first), "refresh") != vault.get(slot(second), "refresh")
    for directory in (first, second):
        assert _login.access_token(runtime(platform, vault, directory), ISSUER, min_validity=1000)
    assert not server.family_revoked and server.refreshes == 2


def _slow_refresh(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> tuple[threading.Thread, list[BaseException]]:
    """A helper that is inside its refresh (it holds the lock) until ``server.refresh_hold`` is set."""
    server.refresh_hold = threading.Event()
    errors: list[BaseException] = []

    def helper() -> None:
        try:
            _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=helper)
    thread.start()
    time.sleep(0.4)
    return thread, errors


def test_logout_waits_for_a_refresh_in_flight_and_revokes_the_token_that_refresh_stored(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    thread, errors = _slow_refresh(platform, server, vault, config_dir)
    result: list[int] = []
    out = threading.Thread(target=lambda: result.append(_login.logout(runtime(platform, vault, config_dir), ISSUER)))
    out.start()
    time.sleep(0.4)
    assert result == [] and server.revoked == []  # waiting on the lock, not revoking the token being spent
    assert server.refresh_hold is not None
    server.refresh_hold.set()
    thread.join(10)
    out.join(10)
    assert errors == [] and result == [0]
    (revocation,) = server.revoked
    assert server.live == set()  # the rotated token, not the spent one, was revoked
    assert revocation["token"] not in server.live
    assert vault.values == {}  # and nothing was stored again after the logout


def test_login_stores_its_token_after_a_refresh_in_flight_has_stored_its_own(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    thread, errors = _slow_refresh(platform, server, vault, config_dir)
    rt = runtime(platform, vault, config_dir, opener=browser(server))
    result: list[int] = []
    again = threading.Thread(target=lambda: result.append(_login.login(rt, ISSUER, timeout=10)))
    again.start()
    time.sleep(0.6)
    assert result == []  # the new sign-in is waiting to store
    assert server.refresh_hold is not None
    server.refresh_hold.set()
    thread.join(10)
    again.join(10)
    assert errors == [] and result == [0]
    assert vault.get(slot(config_dir), "refresh") == server.login_refresh  # the person's latest sign-in survived


def test_a_response_lost_after_the_server_rotated_is_never_retried_with_the_spent_token(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    server.refresh_exception = httpx.ReadTimeout("the answer never came")
    with pytest.raises(LoginError) as caught:
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert caught.value.code == "login_required" and "authfi login" in caught.value.message
    assert vault.values == {}
    server.refresh_exception = None
    with pytest.raises(LoginError) as again:
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert again.value.code == "not_signed_in"
    assert server.refreshes == 1 and not server.family_revoked  # the spent token was not presented a second time


def test_a_store_that_refuses_the_rotated_token_does_not_leave_the_spent_one_to_be_replayed(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    vault.refuse_refresh = True
    with pytest.raises(LoginError) as caught:
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert caught.value.code == "store_failed"
    vault.refuse_refresh = False
    with pytest.raises(LoginError) as again:
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert again.value.code == "not_signed_in" and server.refreshes == 1 and not server.family_revoked


def test_a_refusal_that_happened_before_the_token_was_spent_leaves_it_usable(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    kept = vault.get(slot(config_dir), "refresh")
    server.refresh_error = {"error": "temporarily_unavailable", "error_description": "try again"}
    server.refresh_status = 503
    with pytest.raises(OAuthError):
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert vault.get(slot(config_dir), "refresh") == kept
    server.refresh_error = {"error": "interaction_required"}
    server.refresh_status = 400
    with pytest.raises(OAuthError):
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert vault.get(slot(config_dir), "refresh") == kept
    server.refresh_error = None
    assert _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert not server.family_revoked


def test_a_server_error_after_the_send_is_treated_as_an_unknown_outcome(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    server.refresh_error = {"error": "server_error"}
    server.refresh_status = 500
    with pytest.raises(LoginError) as caught:
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert caught.value.code == "login_required" and vault.values == {}


def test_an_ended_sign_in_is_dropped_and_asks_for_a_new_one(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    server.refresh_error = {"error": "invalid_grant", "error_description": "the session ended"}
    with pytest.raises(LoginError) as caught:
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert caught.value.code == "login_required" and "authfi login" in caught.value.message
    assert vault.values == {}


def test_a_dead_client_on_refresh_is_registered_again_once_and_the_person_is_asked_to_sign_in(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    old = _login.Config(config_dir).client_id(ISSUER)
    server.refresh_error = {"error": "invalid_client", "error_description": "unknown client"}
    server.refresh_status = 401
    with pytest.raises(LoginError) as caught:
        _login.access_token(runtime(platform, vault, config_dir), ISSUER, min_validity=1000)
    assert caught.value.code == "login_required" and "registered again" in caught.value.message
    assert len(server.registrations) == 2 and _login.Config(config_dir).client_id(ISSUER) != old
    assert vault.values == {}
    server.refresh_error = None
    assert _login.login(runtime(platform, vault, config_dir, opener=browser(server)), ISSUER, timeout=10) == 0
    assert len(server.registrations) == 2  # the new client was used, not a third


def test_login_registers_again_once_when_the_stored_client_is_refused(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    server.dead_clients.add(_login.Config(config_dir).client_id(ISSUER) or "")
    assert _login.login(runtime(platform, vault, config_dir, opener=browser(server)), ISSUER, timeout=10) == 0
    assert len(server.registrations) == 2


def test_login_gives_up_after_one_new_registration(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    server.all_clients_dead = True
    with pytest.raises(OAuthError) as caught:
        _login.login(runtime(platform, vault, config_dir, opener=browser(server)), ISSUER, timeout=10)
    assert caught.value.code == "invalid_client" and len(server.registrations) == 2
    assert vault.values == {}


def test_login_registers_again_only_for_a_refused_client_not_for_any_other_refusal(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    server.exchange_error = {"error": "invalid_grant", "error_description": "the code was used"}
    with pytest.raises(OAuthError) as caught:
        _login.login(runtime(platform, vault, config_dir, opener=browser(server)), ISSUER, timeout=10)
    assert caught.value.code == "invalid_grant" and len(server.registrations) == 1
    assert len(server.token_forms) == 1  # one exchange, not a second sign-in


def test_logout_with_no_client_to_revoke_with_says_so_and_still_deletes(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    (config_dir / "login.json").write_text(json.dumps({"default_issuer": ISSUER}))
    with pytest.raises(LoginError) as caught:
        _login.logout(runtime(platform, vault, config_dir), ISSUER)
    assert caught.value.code == "revocation_failed" and vault.values == {} and server.revoked == []


# --- a custom domain: the issuer is the one the project names ----------------------------------------------

CUSTOM = "https://login.acme.example"


def test_the_issuer_and_the_resource_come_from_discovery_not_from_the_typed_host(
    vault: MemoryVault, config_dir: Path
) -> None:
    platform = Platform(issuer=CUSTOM)  # the slug host's discovery names the project's custom domain
    server = FakeAS(platform)
    seen: list[str] = []
    drive = browser(server)

    def opener(url: str) -> object:
        seen.append(url)
        return drive(url)

    rt = runtime(platform, vault, config_dir, opener=opener)
    assert _login.login(rt, ISSUER, timeout=10) == 0
    query = {k: v[0] for k, v in parse_qs(urlsplit(seen[0]).query).items()}
    assert query["resource"] == CUSTOM + "/llm"
    data = json.loads((config_dir / "login.json").read_text())
    assert data["default_issuer"] == CUSTOM and data["aliases"] == {ISSUER: CUSTOM}
    assert vault.get(_login.slot(CUSTOM, config_dir), "refresh") in server.live

    # the helper typed with the slug host finds the same sign-in without asking discovery again
    platform.requests.clear()
    token = _login.access_token(runtime(platform, vault, config_dir), ISSUER)
    assert token and platform.requests == []
    # the tool's base URL must name the issuer the token is for, not the alias that was typed
    _login.access_token(runtime(platform, vault, config_dir, env={"ANTHROPIC_BASE_URL": CUSTOM + "/llm"}), ISSUER)
    with pytest.raises(LoginError) as refused:
        _login.access_token(runtime(platform, vault, config_dir, env={"ANTHROPIC_BASE_URL": ISSUER + "/llm"}), ISSUER)
    assert refused.value.code == "environment_refused"
    assert _login.logout(runtime(platform, vault, config_dir), ISSUER) == 0
    assert vault.values == {} and server.revoked


def test_a_discovery_document_that_names_no_usable_issuer_stops_the_login(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    for bad in (None, "http://acme.authfi.io", "https://acme.authfi.io/path"):
        platform.discovery["issuer"] = bad
        with pytest.raises(LoginError):
            _login.login(runtime(platform, vault, config_dir, opener=browser(server)), ISSUER, timeout=2)
    assert vault.values == {} and server.registrations == []


# --- the rest of the environment: other providers, proxies, certificates -----------------------------------


def _refused(platform: Platform, vault: MemoryVault, config_dir: Path, env: dict[str, str], name: str) -> LoginError:
    with pytest.raises(LoginError) as caught:
        _login.access_token(runtime(platform, vault, config_dir, env=env), None)
    assert caught.value.code == "environment_refused" and name in caught.value.message
    return caught.value


@pytest.mark.parametrize("name", list(_login.PROVIDER_URL_VARIABLES))
def test_another_providers_endpoint_variable_stops_the_token(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    for value in ("https://bedrock-runtime.us-east-1.amazonaws.com", LLM, "my-foundry-resource"):
        _refused(platform, vault, config_dir, {name: value}, name)
    assert server.refreshes == 0


@pytest.mark.parametrize("name", list(_login.PROVIDER_SWITCHES))
def test_a_provider_switch_stops_the_token_only_when_it_is_on(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    for on in ("1", "true", "TRUE", "yes"):
        _refused(platform, vault, config_dir, {name: on}, name)
    for off in ("", "0", "false", "no"):
        assert _login.access_token(runtime(platform, vault, config_dir, env={name: off}), None)


PROXY_NAMES = ["HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy", "all_proxy"]


@pytest.mark.parametrize("name", PROXY_NAMES)
@pytest.mark.parametrize(
    "value",
    ["http://evil.example:3128", "evil.example:3128", "http://user:s3cret@evil.example:3128", "socks5://10.0.0.9"],
)
def test_a_proxy_nobody_allowed_stops_the_token_and_its_password_is_not_echoed(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str, value: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    error = _refused(platform, vault, config_dir, {name: value}, name)
    assert "s3cret" not in error.message and "--allow-proxy" in error.message
    assert server.refreshes == 0


def test_a_proxy_the_person_allowed_by_name_is_let_through_and_no_other_is(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    rt = runtime(platform, vault, config_dir, opener=browser(server))
    assert _login.login(rt, ISSUER, timeout=10, allow_proxies=["Proxy.Corp.Example"]) == 0
    ok = {"HTTPS_PROXY": "http://u:p@proxy.corp.example:8080"}
    assert _login.access_token(runtime(platform, vault, config_dir, env=ok), None)
    assert _login.access_token(
        runtime(platform, vault, config_dir, env={"https_proxy": "proxy.corp.example:3128"}), None
    )
    _refused(platform, vault, config_dir, {"HTTPS_PROXY": "http://other.corp.example:8080"}, "HTTPS_PROXY")
    _refused(platform, vault, config_dir, {**ok, "ALL_PROXY": "http://evil.example"}, "ALL_PROXY")
    # the allowance is read from the person's own file: no environment variable can grant it
    _refused(
        platform,
        vault,
        config_dir,
        {"HTTPS_PROXY": "http://evil.example", "AUTHFI_ALLOW_PROXY": "evil.example"},
        "HTTPS_PROXY",
    )


@pytest.mark.parametrize("value", ["0", " 0 "])
def test_certificate_verification_turned_off_always_stops_the_token(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, value: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    _refused(platform, vault, config_dir, {"NODE_TLS_REJECT_UNAUTHORIZED": value}, "NODE_TLS_REJECT_UNAUTHORIZED")
    assert _login.access_token(runtime(platform, vault, config_dir, env={"NODE_TLS_REJECT_UNAUTHORIZED": "1"}), None)
    with pytest.raises(LoginError):  # it cannot be allowed by name
        _login.login(runtime(platform, vault, config_dir), ISSUER, allow_env=["NODE_TLS_REJECT_UNAUTHORIZED=0"])


@pytest.mark.parametrize(
    "name",
    ["NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"],
)
def test_a_trust_store_variable_stops_the_token_unless_that_exact_value_was_allowed(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    _refused(platform, vault, config_dir, {name: "/repo/evil-ca.pem"}, name)
    assert (
        _login.login(
            runtime(platform, vault, config_dir, opener=browser(server)),
            ISSUER,
            timeout=10,
            allow_env=[f"{name}=/etc/ssl/corp-ca.pem"],
        )
        == 0
    )
    assert _login.access_token(runtime(platform, vault, config_dir, env={name: "/etc/ssl/corp-ca.pem"}), None)
    _refused(platform, vault, config_dir, {name: "/repo/evil-ca.pem"}, name)  # the name alone allows nothing


@pytest.mark.parametrize(
    ("value", "refused"),
    [
        ("bundled", False),
        ("system", False),
        ("bundled, system", False),
        ("bundled,/repo/ca.pem", True),
        ("/repo/ca.pem", True),
    ],
)
def test_the_cert_store_variable_may_only_narrow_the_built_in_sources(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, value: str, refused: bool
) -> None:
    signed_in(platform, server, vault, config_dir)
    env = {"CLAUDE_CODE_CERT_STORE": value}
    if refused:
        _refused(platform, vault, config_dir, env, "CLAUDE_CODE_CERT_STORE")
    else:
        assert _login.access_token(runtime(platform, vault, config_dir, env=env), None)


#: Every spelling here loads code or changes TLS in node; the three that the round-5 review found come first.
NODE_BYPASSES = [
    '"--requ\\ire=./p.js"',
    "--tls_keylog=/tmp/k",
    "--experimental_loader /repo/l.mjs",
    "--use_openssl_ca",
    "--openssl_config=/repo/o.cnf",
    '"--req\\uire" ./p.js',
    '"--tls-key\\log=/tmp/k"',
    '"--tls_key\\log=/tmp/k"',
    "--require /repo/hook.js",
    "-r /repo/hook.js",
    "--import=/repo/hook.mjs",
    "--experimental-loader /repo/l.mjs",
    "--use-openssl-ca",
    "--tls-min-v1.0",
    '"--require=./probe.js"',
    '"--require" ./probe.js',
    "'--require=./probe.js'",
    '"--tls-keylog=/tmp/k"',
    "--max-old-space-size=4096",
    '"--max-old-space-size=4096"',
    "--inspect",
    "--use-system-ca",
]


@pytest.mark.parametrize("value", NODE_BYPASSES)
def test_node_options_set_to_anything_stop_the_token(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, value: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    _refused(platform, vault, config_dir, {"NODE_OPTIONS": value}, "NODE_OPTIONS")


def test_node_options_that_is_empty_is_not_set_and_a_blank_one_is_a_value(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    assert _login.access_token(runtime(platform, vault, config_dir, env={"NODE_OPTIONS": ""}), None)
    _refused(platform, vault, config_dir, {"NODE_OPTIONS": "  "}, "NODE_OPTIONS")  # exact means exact


def test_an_allowed_node_options_value_passes_only_when_exact(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    runtime(platform, vault, config_dir).config().save(
        ISSUER, allow_env=_login._allowed_env(["NODE_OPTIONS=--max-old-space-size=4096"])
    )
    ok = {"NODE_OPTIONS": "--max-old-space-size=4096"}
    assert _login.access_token(runtime(platform, vault, config_dir, env=ok), None)
    for other in ("--max-old-space-size=8192", "--max-old-space-size=4096 --require=./p.js", '"--requ\\ire=./p.js"'):
        _refused(platform, vault, config_dir, {"NODE_OPTIONS": other}, "NODE_OPTIONS")


KEYLOG_SPELLINGS = [
    "--tls-keylog=/x",
    "--tls_keylog=/x",
    '"--tls-keylog=/x"',
    '"--tls-key\\log=/x"',
    '"--tls_key\\log=/x"',
    "--max-old-space-size=4096 --tls_keylog /x",
    "'--tls-keylog=/x'",
]


@pytest.mark.parametrize("value", KEYLOG_SPELLINGS)
def test_a_key_log_flag_in_any_spelling_cannot_be_allowed_or_run(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, value: str
) -> None:
    with pytest.raises(LoginError) as caught:
        _login.login(runtime(platform, vault, config_dir), ISSUER, allow_env=[f"NODE_OPTIONS={value}"])
    assert caught.value.code == "bad_allow" and "cannot be allowed" in caught.value.message
    assert platform.requests == [] and vault.values == {}
    signed_in(platform, server, vault, config_dir)
    runtime(platform, vault, config_dir).config().save(ISSUER, allow_env={"NODE_OPTIONS": value})  # hand-edited
    error = _refused(platform, vault, config_dir, {"NODE_OPTIONS": value}, "NODE_OPTIONS")
    assert "cannot be allowed" in error.message


@pytest.mark.parametrize(
    ("value", "tokens"),
    [
        ("--a_b=c_d  --e", ["--a-b=c_d", "--e"]),
        ('"--requ\\ire=./p.js"', ["--require=./p.js"]),
        ('--x="a b"', ["--x=a b"]),
        ("", []),
    ],
)
def test_node_options_are_read_as_node_reads_them(value: str, tokens: list[str]) -> None:
    assert _login.node_options(value) == tokens


GUARDED_CODE_VARIABLES = [*_login.CODE_VARIABLES, "LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT", "DYLD_INSERT_LIBRARIES"]


@pytest.mark.parametrize("name", GUARDED_CODE_VARIABLES)
def test_a_variable_that_loads_code_stops_the_token_unless_that_exact_value_is_allowed(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    value = "ok" if name in _login._NOT_PATHS else "/opt/ok"
    _refused(platform, vault, config_dir, {name: value}, name)
    runtime(platform, vault, config_dir).config().save(ISSUER, allow_env=_login._allowed_env([f"{name}={value}"]))
    assert _login.access_token(runtime(platform, vault, config_dir, env={name: value}), None)
    _refused(platform, vault, config_dir, {name: value + "x"}, name)


def test_a_path_allowance_needs_every_member_absolute(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    for bad in ("/opt/lib:", "/opt/lib:.", "/opt/lib:repo/src", "rel.so", "/opt/a.so rel.so"):
        with pytest.raises(LoginError) as caught:
            _login.login(runtime(platform, vault, config_dir), ISSUER, allow_env=[f"PYTHONPATH={bad}"])
        assert caught.value.code == "bad_allow"
    assert platform.requests == []
    assert _login._allowed_env(["PYTHONPATH=/opt/a:/opt/b"]) == {"PYTHONPATH": "/opt/a:/opt/b"}


@pytest.mark.parametrize("value", ["/x", " ", '""', "''", "\t"])
def test_a_blank_or_quoted_key_log_name_counts_as_set(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, value: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    _refused(platform, vault, config_dir, {"SSLKEYLOGFILE": value}, "SSLKEYLOGFILE")


def test_an_allowance_written_by_hand_is_checked_when_it_is_read(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    runtime(platform, vault, config_dir).config().save(
        ISSUER,
        allow_proxies=[""],
        allow_env={"SSL_CERT_FILE": "ca.pem", "PYTHONPATH": ".:/x"},
    )
    allowed = runtime(platform, vault, config_dir).config().allowed()
    assert not allowed.proxies and not allowed.env
    _refused(platform, vault, config_dir, {"SSL_CERT_FILE": "ca.pem"}, "SSL_CERT_FILE")
    _refused(platform, vault, config_dir, {"HTTPS_PROXY": "not an address"}, "HTTPS_PROXY")


def test_a_file_another_user_owns_or_can_write_is_not_private() -> None:
    mine = types.SimpleNamespace(st_uid=1000, st_mode=0o100600)
    assert _login.private_to(cast("os.stat_result", mine), 1000)
    assert not _login.private_to(cast("os.stat_result", mine), 1001)
    for mode in (0o100620, 0o100602, 0o100666):
        shared = types.SimpleNamespace(st_uid=1000, st_mode=mode)
        assert not _login.private_to(cast("os.stat_result", shared), 1000)


def _always(answer: bool) -> Callable[[os.stat_result, int], bool]:
    def private_to(info: os.stat_result, uid: int) -> bool:
        return answer

    return private_to


def test_allowances_in_a_file_that_is_not_private_are_ignored(
    platform: Platform,
    server: FakeAS,
    vault: MemoryVault,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signed_in(platform, server, vault, config_dir)
    config = runtime(platform, vault, config_dir).config()
    config.save(ISSUER, allow_proxies=["proxy.corp.example"])
    monkeypatch.setattr(_login, "private_to", _always(True))
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    assert config.allowed().proxies == {"proxy.corp.example"}
    monkeypatch.setattr(_login, "private_to", _always(False))
    assert not config.allowed().proxies


def test_a_failed_forget_is_a_clear_error_and_leaves_no_temporary_file(
    platform: Platform,
    server: FakeAS,
    vault: MemoryVault,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signed_in(platform, server, vault, config_dir)
    config = runtime(platform, vault, config_dir).config()
    config.save(ISSUER, allow_proxies=["proxy.corp.example"])

    def refuse(*_: object) -> None:
        raise OSError(13, "Permission denied")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(LoginError) as caught:
        config.forget_allow()
    assert caught.value.code == "forget_failed" and "still in force" in caught.value.message
    assert not list(config_dir.glob("*.tmp"))


NEVER_ALLOWED = [
    ("NODE_TLS_REJECT_UNAUTHORIZED", "0", "certificate verification off"),
    ("PYTHONHTTPSVERIFY", "0", "certificate verification off"),
    ("GIT_SSL_NO_VERIFY", "1", "certificate verification off"),
    ("SSLKEYLOGFILE", "/home/me/keys.log", "TLS session keys"),
    ("NODE_OPTIONS", "--tls-keylog=/home/me/keys.log", "TLS session keys"),
    ("NODE_OPTIONS", '"--tls-keylog=/home/me/keys.log"', "TLS session keys"),
]


@pytest.mark.parametrize(("name", "value", "why"), NEVER_ALLOWED)
def test_a_variable_that_disables_verification_or_exports_keys_cannot_be_allowed(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str, value: str, why: str
) -> None:
    with pytest.raises(LoginError) as caught:
        _login.login(runtime(platform, vault, config_dir), ISSUER, allow_env=[f"{name}={value}"])
    assert caught.value.code == "bad_allow" and name in caught.value.message and why in caught.value.message
    assert "cannot be allowed" in caught.value.message
    assert platform.requests == [] and vault.values == {}  # refused before any call


@pytest.mark.parametrize(("name", "value", "why"), NEVER_ALLOWED)
def test_a_hand_edited_allowance_does_not_lift_a_never_allowed_variable(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str, value: str, why: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    runtime(platform, vault, config_dir).config().save(ISSUER, allow_env={name: value})
    error = _refused(platform, vault, config_dir, {name: value}, name)
    assert why in error.message


def test_the_never_allowed_variables_are_fine_when_harmless(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    for env in ({"PYTHONHTTPSVERIFY": "1"}, {"SSLKEYLOGFILE": ""}, {"GIT_SSL_NO_VERIFY": ""}):
        assert _login.access_token(runtime(platform, vault, config_dir, env=env), None)


def test_an_allowance_must_name_a_proxy_host_and_an_absolute_path(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    cases: list[tuple[list[str], list[str]]] = [
        ([""], []),
        (["  "], []),
        ([], ["SSL_CERT_FILE=ca.pem"]),
        ([], ["SSLKEYLOGFILE=keys.log"]),
    ]
    for proxies, pairs in cases:
        with pytest.raises(LoginError) as caught:
            _login.login(runtime(platform, vault, config_dir), ISSUER, allow_proxies=proxies, allow_env=pairs)
        assert caught.value.code == "bad_allow"
    assert platform.requests == [] and vault.values == {}


def test_logout_can_forget_what_login_allowed(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    config = runtime(platform, vault, config_dir).config()
    config.save(ISSUER, allow_proxies=["proxy.corp.example"], allow_env={"SSL_CERT_FILE": "/etc/ssl/ca.pem"})
    assert config.allowed().proxies
    _login.logout(runtime(platform, vault, config_dir), None, forget_allow=True)
    allowed = runtime(platform, vault, config_dir).config().allowed()
    assert not allowed.proxies and not allowed.env


def test_allow_env_takes_only_the_names_the_guard_can_allow(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    for bad in ("HTTPS_PROXY=http://x", "ANTHROPIC_BASE_URL=https://evil.example/llm", "NODE_EXTRA_CA_CERTS=", "x"):
        with pytest.raises(LoginError) as caught:
            _login.login(runtime(platform, vault, config_dir), ISSUER, allow_env=[bad])
        assert caught.value.code == "bad_allow"
    assert platform.requests == [] and vault.values == {}  # refused before any call


def test_a_clean_environment_and_the_allowed_ones_get_their_token_through_the_command(
    platform: Platform,
    server: FakeAS,
    vault: MemoryVault,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    signed_in(platform, server, vault, config_dir)
    monkeypatch.setattr(_login, "open_vault", lambda: vault)
    monkeypatch.setattr(cli_module, "_http_client", lambda: platform.client())
    monkeypatch.setenv("AUTHFI_CONFIG_DIR", str(config_dir))
    for name in (
        *_login.PROXY_VARIABLES,
        *_login.TLS_VARIABLES,
        *_login.CODE_VARIABLES,
        _login.TLS_OFF,
        *_login.PROVIDER_SWITCHES,
    ):
        monkeypatch.delenv(name, raising=False)
    assert cli_module.main(["token", "--resource", "llm", "--issuer", ISSUER]) == 0
    assert capsys.readouterr().out.strip().count(".") == 2
    monkeypatch.setenv("HTTPS_PROXY", "http://evil.example:3128")
    assert cli_module.main(["token", "--resource", "llm", "--issuer", ISSUER]) == 1
    out = capsys.readouterr()
    assert out.out == "" and "HTTPS_PROXY" in out.err and "evil.example" in out.err


UNICODE_SPACES = ["\xa0", "\u2003", "\u3000", "\x85", "\x1c", "\x0b", "\x0c", "\u2028"]


@pytest.mark.parametrize("space", UNICODE_SPACES)
@pytest.mark.parametrize("where", ["after", "before"])
def test_an_allowed_value_with_a_space_added_is_not_that_exact_value(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, space: str, where: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    runtime(platform, vault, config_dir).config().save(
        ISSUER, allow_env=_login._allowed_env(["NODE_EXTRA_CA_CERTS=/opt/corp/ca.pem"])
    )
    exact = {"NODE_EXTRA_CA_CERTS": "/opt/corp/ca.pem"}
    assert _login.access_token(runtime(platform, vault, config_dir, env=exact), None)
    other = "/opt/corp/ca.pem" + space if where == "after" else space + "/opt/corp/ca.pem"
    _refused(platform, vault, config_dir, {"NODE_EXTRA_CA_CERTS": other}, "NODE_EXTRA_CA_CERTS")


def test_an_allowance_is_recorded_as_written(platform: Platform, vault: MemoryVault, config_dir: Path) -> None:
    assert _login._allowed_env(["NODE_OPTIONS=--max-old-space-size=4096 "]) == {
        "NODE_OPTIONS": "--max-old-space-size=4096 "
    }
    assert _login.allowance_problem("NODE_EXTRA_CA_CERTS", " ") is not None  # a lone space is a value, not a path
    assert _login.allowance_problem("NODE_EXTRA_CA_CERTS", "") is not None


def test_a_drive_letter_path_is_only_absolute_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert _login.allowance_problem("SSL_CERT_FILE", "C:\\ca.pem") is not None
    monkeypatch.setattr(sys, "platform", "win32")
    assert _login.allowance_problem("SSL_CERT_FILE", "C:\\ca.pem") is None


NPM_NAMES = [
    "npm_config_node_options",
    "NPM_CONFIG_NODE_OPTIONS",
    "Npm_Config_Node_Options",
    "npm_config_userconfig",
    "NPM_CONFIG_GLOBALCONFIG",
    "npm_config_node-options",
    "NPM_CONFIG_NODE-OPTIONS",
    "npm_config_script_shell",
    "npm_config_script-shell",
    "NPM_CONFIG_SCRIPT-SHELL",
]


@pytest.mark.parametrize("name", NPM_NAMES)
def test_the_options_npm_hands_to_node_stop_the_token_unless_that_exact_value_is_allowed(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    value = "--max-old-space-size=4096" if _login.npm_key(name) == "node-options" else "/opt/npmrc"
    _refused(platform, vault, config_dir, {name: value}, name)
    runtime(platform, vault, config_dir).config().save(ISSUER, allow_env=_login._allowed_env([f"{name}={value}"]))
    assert _login.access_token(runtime(platform, vault, config_dir, env={name: value}), None)
    _refused(platform, vault, config_dir, {name: value + " "}, name)


@pytest.mark.parametrize("name", ["npm_config_node_options", "NPM_CONFIG_NODE_OPTIONS"])
@pytest.mark.parametrize("value", KEYLOG_SPELLINGS)
def test_a_key_log_flag_through_npm_cannot_be_allowed_or_run(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str, value: str
) -> None:
    with pytest.raises(LoginError) as caught:
        _login.login(runtime(platform, vault, config_dir), ISSUER, allow_env=[f"{name}={value}"])
    assert caught.value.code == "bad_allow" and "cannot be allowed" in caught.value.message
    signed_in(platform, server, vault, config_dir)
    runtime(platform, vault, config_dir).config().save(ISSUER, allow_env={name: value})  # hand-edited
    assert "cannot be allowed" in _refused(platform, vault, config_dir, {name: value}, name).message


BEFORE_THE_GUARD = [
    "LD_PRELOAD",
    "LD_AUDIT",
    "LD_LIBRARY_PATH",
    "DYLD_INSERT_LIBRARIES",
    "DYLD_LIBRARY_PATH",
    "OPENSSL_CONF",
    "OPENSSL_ENGINES",
    "BASH_ENV",
    "ENV",
    "SHELLOPTS",
]


@pytest.mark.parametrize("name", BEFORE_THE_GUARD)
def test_a_variable_the_loader_or_shell_reads_first_is_still_refused_and_named(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str
) -> None:
    signed_in(platform, server, vault, config_dir)
    error = _refused(platform, vault, config_dir, {name: "/repo/x"}, name)
    assert "before this check runs" in error.message and "second line of defence" in error.message


BUS = "DBUS_SESSION_BUS_ADDRESS"
ME = 1000


@pytest.mark.parametrize(
    ("platform_name", "classes"),
    [
        ("linux", [("authfi._login", "VerifiedSecretService")]),
        ("darwin", [("keyring.backends.macOS", "Keyring")]),
        ("win32", [("keyring.backends.Windows", "WinVaultKeyring")]),
        ("freebsd14", []),
    ],
)
def test_the_store_named_for_each_platform(platform_name: str, classes: list[tuple[str, str]]) -> None:
    assert _login._backend_classes(platform_name) == classes
    assert all(f"{module}.{name}" in _login.SECURE_BACKENDS for module, name in classes)


def test_the_session_bus_variable_is_not_one_the_guard_allows() -> None:
    assert not _login.is_guarded(BUS)
    with pytest.raises(LoginError):
        _login._allowed_env([f"{BUS}=unix:path=/tmp/x"])


class _Recorder:
    """What the Secret Service backend did with the bus: the connection it was given, and the calls it made."""

    def __init__(self) -> None:
        self.connection = types.SimpleNamespace(closed=False, close=lambda: None, name="the checked connection")
        self.opened: list[tuple[str, int]] = []
        self.matched: list[object] = []
        self.collection_for: list[object] = []
        self.dbus_init_calls = 0


@pytest.fixture
def secret_service(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, _Recorder]:
    """keyring's Secret Service backend, rebuilt on this package's checked connection, with the bus faked."""
    secretstorage = pytest.importorskip("secretstorage")
    util = pytest.importorskip("secretstorage.util")
    pytest.importorskip("keyring.backends.SecretService")
    record = _Recorder()
    monkeypatch.setenv(BUS, "unix:path=/run/user/1000/bus")
    monkeypatch.setattr(os, "getuid", lambda: ME, raising=False)

    def open_verified_bus(address: str, uid: int, **_: object) -> object:
        record.opened.append((address, uid))
        return record.connection

    def dbus_init() -> None:
        record.dbus_init_calls += 1
        raise AssertionError("secretstorage connected to the bus by itself")

    class Collection:
        def __init__(self, connection: object) -> None:
            record.collection_for.append(connection)
            self.connection = connection

        def is_locked(self) -> bool:
            return False

    monkeypatch.setattr(_bus, "open_verified_bus", open_verified_bus)
    monkeypatch.setattr(secretstorage, "dbus_init", dbus_init)

    def get_default_collection(connection: object) -> Collection:
        return Collection(connection)

    def check_service_availability(connection: object) -> bool:
        return True

    def add_match_rules(connection: object) -> None:
        record.matched.append(connection)

    monkeypatch.setattr(secretstorage, "get_default_collection", get_default_collection)
    monkeypatch.setattr(secretstorage, "check_service_availability", check_service_availability)
    monkeypatch.setattr(util, "add_match_rules", add_match_rules)
    monkeypatch.delitem(vars(_login), "VerifiedSecretService", raising=False)
    return _login.VerifiedSecretService(), record


def test_the_secret_service_is_handed_the_connection_that_was_checked(secret_service: tuple[Any, _Recorder]) -> None:
    backend, record = secret_service
    collection = backend.get_preferred_collection()
    assert record.opened == [("unix:path=/run/user/1000/bus", ME)]  # checked as this user
    assert collection.connection is record.connection and record.collection_for == [record.connection]
    assert record.matched == [record.connection]
    assert record.dbus_init_calls == 0


def test_the_availability_probe_uses_the_checked_connection_too(secret_service: tuple[Any, _Recorder]) -> None:
    backend, record = secret_service
    assert backend.priority > 0
    assert record.opened and record.dbus_init_calls == 0


def test_a_bus_run_by_another_user_stops_the_store_and_says_so(
    secret_service: tuple[Any, _Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, record = secret_service

    def refused(address: str, uid: int, **_: object) -> object:
        raise _bus.BusPeerError("the bus socket is served by uid 0, not by you (uid 1000)")

    monkeypatch.setattr(_bus, "open_verified_bus", refused)
    for call in (backend.get_preferred_collection, lambda: backend.priority):
        with pytest.raises(LoginError) as caught:
            call()
        assert caught.value.code == "environment_refused" and BUS in caught.value.message
        assert "uid 0" in caught.value.message
    assert record.dbus_init_calls == 0 and record.collection_for == []


def test_opening_the_vault_reports_a_foreign_bus_not_a_missing_store(
    secret_service: tuple[Any, _Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    def classes(platform: str) -> list[tuple[str, str]]:
        return [("authfi._login", "VerifiedSecretService")]

    def refused(address: str, uid: int, **_: object) -> object:
        raise _bus.BusPeerError("the bus socket is served by uid 0, not by you (uid 1000)")

    monkeypatch.setattr(_login, "_backend_classes", classes)
    monkeypatch.setattr(_bus, "open_verified_bus", refused)
    with pytest.raises(LoginError) as caught:
        _login.open_vault()
    assert caught.value.code == "environment_refused"


def test_with_no_session_bus_address_the_store_is_missing_not_guessed(
    secret_service: tuple[Any, _Recorder], monkeypatch: pytest.MonkeyPatch
) -> None:
    backend, record = secret_service
    monkeypatch.delenv(BUS)
    secretstorage = pytest.importorskip("secretstorage")
    with pytest.raises(secretstorage.SecretServiceNotAvailableException):
        backend.get_preferred_collection()
    assert record.opened == [] and record.dbus_init_calls == 0


@pytest.mark.parametrize(
    ("name", "key"),
    [
        ("npm_config_node_options", "node-options"),
        ("npm_config_node-options", "node-options"),
        ("NPM_CONFIG_NODE-OPTIONS", "node-options"),
        ("Npm_Config_Node_Options", "node-options"),
        ("npm_config_script_shell", "script-shell"),
        ("npm_config_script-shell", "script-shell"),
        ("NPM_CONFIG_USERCONFIG", "userconfig"),
        ("NODE_OPTIONS", None),
        ("npm_config", None),
        ("xnpm_config_node_options", None),
    ],
)
def test_npm_keys_are_read_as_npm_reads_them(name: str, key: str | None) -> None:
    assert _login.npm_key(name) == key
    assert _login.is_guarded(name) is (key is not None or name == "NODE_OPTIONS")


@pytest.mark.parametrize("name", ["npm_config_node-options", "NPM_CONFIG_NODE-OPTIONS"])
@pytest.mark.parametrize("value", KEYLOG_SPELLINGS)
def test_a_key_log_flag_through_npms_dash_spelling_cannot_be_allowed_or_run(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path, name: str, value: str
) -> None:
    with pytest.raises(LoginError) as caught:
        _login._allowed_env([f"{name}={value}"])
    assert caught.value.code == "bad_allow" and "cannot be allowed" in caught.value.message
    signed_in(platform, server, vault, config_dir)
    runtime(platform, vault, config_dir).config().save(ISSUER, allow_env={name: value})  # hand-edited
    assert "cannot be allowed" in _refused(platform, vault, config_dir, {name: value}, name).message


def test_an_allowance_cannot_name_a_file_by_a_relative_path() -> None:
    for pair in (
        "NODE_OPTIONS=--require=./hook.js",
        "NODE_OPTIONS=--openssl-config=openssl.cnf",
        "NODE_OPTIONS=-r ./hook.js",
        "JAVA_TOOL_OPTIONS=-javaagent:agent.jar",
        "RUBYOPT=-r./x",
        "NPM_CONFIG_NODE_OPTIONS=--require=../hook.js",
    ):
        name = pair.partition("=")[0]
        with pytest.raises(LoginError) as caught:
            _login._allowed_env([pair])
        assert caught.value.code == "bad_allow" and name in caught.value.message and "relative" in caught.value.message
    assert _login._allowed_env(["NODE_OPTIONS=--require=/opt/hook.js --max-old-space-size=4096"])
    assert _login._allowed_env(["JAVA_TOOL_OPTIONS=-javaagent:/opt/agent.jar"])
    assert _login._allowed_env(["NODE_OPTIONS=--max-old-space-size=4096"])


def test_a_relative_file_in_an_allowance_written_by_hand_is_ignored(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    runtime(platform, vault, config_dir).config().save(ISSUER, allow_env={"NODE_OPTIONS": "--require=./hook.js"})
    _refused(platform, vault, config_dir, {"NODE_OPTIONS": "--require=./hook.js"}, "NODE_OPTIONS")


@pytest.mark.parametrize("executable", ["", "python", "./venv/bin/python", "venv\\Scripts\\python.exe"])
def test_the_helper_line_needs_an_absolute_interpreter(
    executable: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))  # nothing named python is on it
    with pytest.raises(LoginError) as caught:
        _login.helper_command(executable)
    assert caught.value.code == "no_interpreter"


def test_the_helper_line_runs_in_the_shell_claude_code_uses(tmp_path: Path) -> None:
    """``/bin/sh`` on macOS and Linux, ``cmd`` on Windows: a space and an ampersand in the path keep the args."""
    folder = tmp_path / "dir with space & amp"
    folder.mkdir()
    if sys.platform == "win32":
        program = folder / "echo args.cmd"
        program.write_text("@echo off\r\necho ARGS: %*\r\n", encoding="utf-8")
    else:
        program = folder / "echo args.sh"
        program.write_text('#!/bin/sh\necho "ARGS: $@"\n', encoding="utf-8")
        program.chmod(0o755)
    command = _login.helper_command(str(program))
    done = subprocess.run(command, shell=True, capture_output=True, text=True, check=False)  # noqa: S602
    assert done.returncode == 0 and "ARGS: -I -m authfi token --resource llm" in done.stdout, done.stderr


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe quoting")
def test_the_windows_helper_line_quotes_the_interpreter_for_cmd() -> None:
    assert _login.helper_command("C:\\Program Files\\Python\\python.exe") == (
        '"C:\\Program Files\\Python\\python.exe" -I -m authfi token --resource llm'
    )
    for bad in ("C:\\a%PATH%b\\python.exe", 'C:\\a"b\\python.exe'):
        with pytest.raises(LoginError):
            _login.helper_command(bad)


def test_the_posix_helper_line_quotes_the_interpreter_for_sh(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    program = tmp_path / "a dir; touch x" / "python"
    monkeypatch.setattr(sys, "platform", "linux")
    command = _login.helper_command(str(program))
    assert command == f"{shlex.quote(str(program))} -I -m authfi token --resource llm"
    assert shlex.split(command)[0] == str(program)  # one word for sh, whatever the path holds


def test_a_bare_interpreter_name_is_not_resolved_through_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A Python named ``python`` on ``PATH`` may be another interpreter than the one running: no line is printed."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("python", "python.cmd", "python.exe"):
        (bin_dir / name).write_text("", encoding="utf-8")
        (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.chdir(bin_dir)
    for name in ("python", "./python", "python.exe"):
        with pytest.raises(LoginError) as caught:
            _login.helper_command(name)
        assert caught.value.code == "no_interpreter"
