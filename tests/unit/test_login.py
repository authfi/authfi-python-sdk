"""``authfi login``, ``authfi token --resource llm`` and ``authfi logout`` against a faithful fake authorization server.

The fake registers public clients, checks PKCE S256, rotates refresh tokens and revokes the whole family when a
spent one comes back (as identity does), so a missing lock or a lost rotation fails here as it would live.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import secrets
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, parse_qsl, urlencode, urlsplit

import httpx
import keyring
import keyring.backend
import keyring.backends.fail
import keyring.errors
import pytest

from authfi import _login
from authfi import cli as cli_module
from authfi._login import KeyringVault, LoginError, Runtime
from authfi._protocol.http import HttpResponse
from authfi._protocol.register import parse_registration
from authfi.errors import InvalidCallback, OAuthError
from tests.support import Platform, b64url, json_response

ISSUER = "https://acme.authfi.io"
LLM = ISSUER + "/llm"
REGISTER = "/oauth/register"


class MemoryVault:
    """The credential store of a test: a dict, with a switch for a store that refuses the access token."""

    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}
        self.refuse_access = False

    def get(self, issuer: str, name: str) -> str | None:
        return self.values.get((issuer, name))

    def set(self, issuer: str, name: str, value: str) -> None:
        if self.refuse_access and name == "access":
            raise OSError("value too long for this store")
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

    def _rotate(self) -> str:
        fresh = "rt-" + secrets.token_hex(6)
        self.live.add(fresh)
        return fresh

    def token(self, request: httpx.Request) -> httpx.Response:
        form = dict(parse_qsl(request.content.decode()))
        self.token_forms.append(form)
        if form["grant_type"] == "authorization_code":
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
                    "refresh_token": self._rotate(),
                    "id_token": id_token,
                    "scope": "openid llm:invoke offline_access",
                }
            )
        assert form["grant_type"] == "refresh_token"
        self.refreshes += 1
        if self.refresh_hold is not None:
            self.refresh_hold.wait(5)
        if self.refresh_error is not None:
            return json_response(self.refresh_error, 400)
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
) -> Runtime:
    out: list[str] = []
    return Runtime(
        env={"AUTHFI_CONFIG_DIR": str(config_dir)},
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
        callback = {"code": code, "state": state or query["state"], "iss": ISSUER, **(extra or {})}

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
    assert vault.get(ISSUER, "refresh") in server.live

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
    assert {p.name for p in files} == {"login.json"}
    text = files[0].read_text()
    data = json.loads(text)
    assert data["default_issuer"] == ISSUER and set(data["clients"][ISSUER]) == {"client_id"}
    assert (vault.get(ISSUER, "refresh") or "x") not in text
    assert (json.loads(vault.get(ISSUER, "access") or "{}").get("token") or "x") not in text


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
    assert vault.get(ISSUER, "refresh") is not None


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
    assert vault.get(ISSUER, "refresh") is None


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


@pytest.fixture
def os_keyring() -> Iterator[Any]:
    before = keyring.get_keyring()
    yield keyring
    keyring.set_keyring(before)


def test_without_a_keyring_backend_login_refuses_and_writes_nothing(
    os_keyring: Any,
    platform: Platform,
    server: FakeAS,
    config_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    keyring.set_keyring(keyring.backends.fail.Keyring())  # type: ignore[no-untyped-call]
    monkeypatch.setenv("AUTHFI_CONFIG_DIR", str(config_dir))
    monkeypatch.setattr(cli_module, "_http_client", lambda: platform.client())
    for argv in (["login", "--issuer", ISSUER], ["token", "--resource", "llm", "--issuer", ISSUER]):
        assert cli_module.main(argv) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "credential store" in captured.err and "will not write it to a file" in captured.err
    assert platform.requests == []  # refused before any call
    assert not config_dir.exists()  # and before any file


def test_the_keyring_vault_round_trips_through_a_real_backend(os_keyring: Any) -> None:
    keyring.set_keyring(_Memory())
    vault = _login.open_vault()
    assert isinstance(vault, KeyringVault)
    vault.set(ISSUER, "refresh", "rt-1")
    assert vault.get(ISSUER, "refresh") == "rt-1"
    vault.delete(ISSUER, "refresh")
    vault.delete(ISSUER, "refresh")  # deleting what is not there is not an error
    assert vault.get(ISSUER, "refresh") is None


# --- token -------------------------------------------------------------------------------------------------


def test_token_refreshes_stores_the_rotation_and_reuses_the_token_until_it_nears_expiry(
    platform: Platform, server: FakeAS, vault: MemoryVault, config_dir: Path
) -> None:
    signed_in(platform, server, vault, config_dir)
    first_refresh = vault.get(ISSUER, "refresh")
    rt = runtime(platform, vault, config_dir)
    # the sign-in left a good access token: no refresh yet
    cached = _login.access_token(rt, None)
    assert server.refreshes == 0

    token = _login.access_token(rt, None, min_validity=1000)  # the 900 s token is too short for this caller
    assert server.refreshes == 1 and token != cached
    assert vault.get(ISSUER, "refresh") != first_refresh  # the rotated token replaced the spent one
    assert vault.get(ISSUER, "refresh") in server.live
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
    vault.delete(ISSUER, "access")
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
    refresh = vault.get(ISSUER, "refresh")
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
