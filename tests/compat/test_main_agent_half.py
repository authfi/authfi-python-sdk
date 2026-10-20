# pyright: reportDeprecated=false
"""``main``'s ``test_authfi.py`` (7d5cfeb), agent half, run against 0.2.

As in ``test_main_user_half.py``: each test keeps its name and assertions. Tests that patched
``authfi.urlopen`` run on httpx.MockTransport, and those that built an agent with no project are
given the one argument 0.2 requires (``platform_url``, the fake project's address). An intended
break is a strict ``xfail`` naming why.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import time
import unittest
from collections.abc import Iterator
from typing import Any, ClassVar, cast
from urllib.parse import parse_qsl

import httpx
import pytest

from authfi import (
    CIBA_GRANT_TYPE,
    CLIENT_ASSERTION_TYPE,
    USER_AGENT,
    AgentAuth,
    AgentKey,
    AuthFIError,
    OAuthError,
    assert_public_jwk,
)
from authfi._clock import SystemClock
from authfi._transport import SyncTransport

pytestmark = [
    pytest.mark.filterwarnings("ignore::authfi.errors.AuthFIDeprecationWarning"),
    pytest.mark.usefixtures("network"),
]

AgentAuthError = OAuthError  # main's name, which 0.2 keeps as a deprecated alias of OAuthError

PLATFORM = "https://platform.authfi.io"
TOKEN_ENDPOINT = PLATFORM + "/apis/regional/v1/ai/auth/token"  # main's; 0.2 uses discovery's token_endpoint
DISCOVERY = {
    "issuer": PLATFORM,
    "token_endpoint": PLATFORM + "/oauth/token",
    "jwks_uri": PLATFORM + "/.well-known/jwks.json",
    "grant_types_supported": ["client_credentials", CIBA_GRANT_TYPE],
    "token_endpoint_auth_methods_supported": ["private_key_jwt"],
    "token_endpoint_auth_signing_alg_values_supported": ["EdDSA"],
    "backchannel_authentication_endpoint": PLATFORM + "/apis/regional/v1/ai/auth/backchannel-authorize",
}
NO_DEFAULT = "intended break: there is no default project (0.1 sent the assertion to platform.authfi.io)"
DICT_RETURN = "intended break: CIBA returns a CibaRequest and a DelegatedToken, never a printable dict"
PRIVATE_FILE = "intended break: a key file others can read is refused"


def _no_sleep(_seconds: float) -> None:
    return None


def _b64decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


class Network:
    """main's ``urlopen`` mock: answers queued per call to the token or CIBA endpoint."""

    def __init__(self) -> None:
        self.answers: list[httpx.Response] = []
        self.requests: list[httpx.Request] = []
        self.sleeps: list[float] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(200, json=DISCOVERY)
        self.requests.append(request)
        if not self.answers:
            return httpx.Response(500)
        return self.answers.pop(0)

    def queue(self, *answers: httpx.Response) -> None:
        self.answers = list(answers)

    @property
    def call_count(self) -> int:
        return len(self.requests)

    def forms(self) -> list[dict[str, str]]:
        return [dict(parse_qsl(r.content.decode())) for r in self.requests]


NETWORK: list[Network] = []


@pytest.fixture
def network(monkeypatch: pytest.MonkeyPatch) -> Iterator[Network]:
    net = Network()
    client = httpx.Client(transport=httpx.MockTransport(net))
    monkeypatch.setattr(SyncTransport, "client", property(lambda _self: client))

    def no_sleep(_self: SystemClock, seconds: float) -> None:
        net.sleeps.append(seconds)

    monkeypatch.setattr(SystemClock, "sleep", no_sleep)
    NETWORK.append(net)
    yield net
    NETWORK.remove(net)


def net() -> Network:
    return NETWORK[-1]


def _response(body: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=body)


def _oauth_error(code: str, description: str | None = None, status: int = 400) -> httpx.Response:
    body = {"error": code}
    if description:
        body["error_description"] = description
    return httpx.Response(status, json=body)


class TestAgentKey(unittest.TestCase):
    def test_generate_and_public_jwk(self) -> None:
        key = AgentKey.generate("k1")
        jwk = key.public_jwk()
        self.assertEqual(jwk["kty"], "OKP")
        self.assertEqual(jwk["crv"], "Ed25519")
        self.assertEqual(jwk["kid"], "k1")
        self.assertEqual(len(_b64decode(jwk["x"])), 32)
        self.assertNotIn("d", jwk)

    def test_key_without_kid_is_refused_at_construction(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.generate("")
        self.assertIn("key_id is required", str(ctx.exception))
        with self.assertRaises(AuthFIError):
            AgentKey.generate(cast(str, None))

    def test_pem_round_trip(self) -> None:
        key = AgentKey.generate("k1")
        reloaded = AgentKey.from_pem(key.private_pem(), "k1")
        self.assertEqual(reloaded.public_jwk(), key.public_jwk())

    def test_pem_refuses_a_key_that_is_not_pkcs8(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.from_pem("-----BEGIN OPENSSH PRIVATE KEY-----\nxx\n", "k1")
        msg = str(ctx.exception)
        self.assertIn("OPENSSH PRIVATE KEY", msg)
        self.assertIn("openssl genpkey -algorithm ed25519", msg)

    @pytest.mark.xfail(strict=True, reason="0.2 accepts P-256 and RSA agent keys (python-sdk#5)")
    def test_pem_refuses_a_non_ed25519_key(self) -> None:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        pem = rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
            serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
        )
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.from_pem(pem, "k1")
        self.assertIn("Ed25519", str(ctx.exception))

    def test_jwk_round_trip_and_kid_from_the_jwk(self) -> None:
        from cryptography.hazmat.primitives import serialization

        key = AgentKey.generate("k1")
        priv: dict[str, Any] = dict(key.public_jwk())
        raw = cast(Any, key)._key.private_bytes(
            serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()
        )
        priv["d"] = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
        reloaded = AgentKey.from_jwk(priv)
        self.assertEqual(reloaded.key_id, "k1")
        self.assertEqual(reloaded.public_jwk(), key.public_jwk())

    def test_jwk_without_d_cannot_sign(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.from_jwk(AgentKey.generate("k1").public_jwk())
        self.assertIn("no 'd'", str(ctx.exception))

    # A new file in a temporary directory is private by default on Windows, and readable by others on POSIX.
    @pytest.mark.xfail(sys.platform != "win32", strict=True, reason=PRIVATE_FILE)
    def test_from_file_sniffs_pem_and_jwk(self) -> None:
        key = AgentKey.generate("k1")
        with tempfile.TemporaryDirectory() as d:
            pem_path = os.path.join(d, "agent.key")
            with open(pem_path, "w") as fh:
                fh.write(key.private_pem())
            self.assertEqual(AgentKey.from_file(pem_path, key_id="k1").public_jwk(), key.public_jwk())


class TestPublicJWKRefusal(unittest.TestCase):
    def test_a_jwk_carrying_d_is_refused_as_burned(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            assert_public_jwk({"kty": "OKP", "crv": "Ed25519", "x": "AA", "d": "secret"})
        msg = str(ctx.exception)
        self.assertIn("burned", msg)
        self.assertIn("'d'", msg)

    def test_wrong_curve_is_refused(self) -> None:
        with self.assertRaises(AuthFIError):
            assert_public_jwk({"kty": "EC", "crv": "P-256", "x": "AA"})

    def test_public_jwk_passes(self) -> None:
        jwk = AgentKey.generate("k1").public_jwk()
        self.assertIs(assert_public_jwk(jwk), jwk)


class TestClientAssertion(unittest.TestCase):
    def setUp(self) -> None:
        self.key = AgentKey.generate("k1")
        self.jws = self.key.client_assertion("agt_123", TOKEN_ENDPOINT)
        self.header, self.claims, self.sig = self.jws.split(".")

    def test_header(self) -> None:
        header = json.loads(_b64decode(self.header))
        self.assertEqual(header["alg"], "EdDSA")
        self.assertEqual(header["typ"], "JWT")
        self.assertEqual(header["kid"], "k1")

    def test_claims(self) -> None:
        c = json.loads(_b64decode(self.claims))
        self.assertEqual(c["iss"], "agt_123")
        self.assertEqual(c["sub"], "agt_123")
        self.assertEqual(c["aud"], TOKEN_ENDPOINT)
        self.assertTrue(c["jti"])
        self.assertEqual(c["exp"] - c["iat"], 300)
        self.assertLessEqual(abs(c["iat"] - int(time.time())), 5)

    def test_jti_is_fresh_every_time(self) -> None:
        ids = {
            json.loads(_b64decode(self.key.client_assertion("agt_123", TOKEN_ENDPOINT).split(".")[1]))["jti"]
            for _ in range(5)
        }
        self.assertEqual(len(ids), 5)

    def test_signature_verifies_against_the_public_key(self) -> None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        pub = Ed25519PublicKey.from_public_bytes(_b64decode(self.key.public_jwk()["x"]))
        pub.verify(_b64decode(self.sig), f"{self.header}.{self.claims}".encode("ascii"))

    def test_a_tampered_payload_does_not_verify(self) -> None:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        pub = Ed25519PublicKey.from_public_bytes(_b64decode(self.key.public_jwk()["x"]))
        forged = json.loads(_b64decode(self.claims))
        forged["sub"] = "agt_someone_else"
        payload = base64.urlsafe_b64encode(json.dumps(forged, separators=(",", ":")).encode()).rstrip(b"=").decode()
        with self.assertRaises(InvalidSignature):
            pub.verify(_b64decode(self.sig), f"{self.header}.{payload}".encode("ascii"))

    def test_segments_are_base64url_without_padding(self) -> None:
        for part in (self.header, self.claims, self.sig):
            self.assertNotIn("=", part)
            self.assertNotIn("+", part)
            self.assertNotIn("/", part)

    def test_requires_agent_id_and_audience(self) -> None:
        with self.assertRaises(AuthFIError):
            self.key.client_assertion("", TOKEN_ENDPOINT)
        with self.assertRaises(AuthFIError):
            self.key.client_assertion("agt_123", "")


class TestAgentAuthConstruction(unittest.TestCase):
    def test_requires_an_agent_key_and_offers_no_secret_fallback(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            AgentAuth(agent_id="agt_1", key=cast(Any, "a-client-secret"))
        self.assertIn("AgentKey", str(ctx.exception))
        self.assertNotIn("client_secret", AgentAuth.__init__.__code__.co_varnames)

    def test_requires_an_agent_id(self) -> None:
        with self.assertRaises(AuthFIError):
            AgentAuth(agent_id="", key=AgentKey.generate("k1"))

    @pytest.mark.xfail(strict=True, reason=NO_DEFAULT)
    def test_discovery_url_defaults_to_the_platform(self) -> None:
        a = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"))
        self.assertEqual(cast(Any, a).discovery_url, "https://platform.authfi.io/.well-known/agent-configuration")


class TestAgentDiscovery(unittest.TestCase):
    def _agent(self) -> AgentAuth:
        return AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"))

    @pytest.mark.xfail(strict=True, reason=NO_DEFAULT + "; the token endpoint is discovery's, never the retired route")
    def test_endpoints_come_from_discovery_and_are_fetched_once(self) -> None:
        a = self._agent()
        self.assertEqual(a.token_endpoint, TOKEN_ENDPOINT)

    @pytest.mark.xfail(strict=True, reason=NO_DEFAULT)
    def test_a_deployment_without_private_key_jwt_is_refused_up_front(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            _ = self._agent().token_endpoint
        self.assertIn("private_key_jwt", str(ctx.exception))

    @pytest.mark.xfail(strict=True, reason=NO_DEFAULT)
    def test_a_deployment_without_eddsa_is_refused(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            _ = self._agent().token_endpoint
        self.assertIn("EdDSA", str(ctx.exception))

    @pytest.mark.xfail(strict=True, reason=NO_DEFAULT)
    def test_no_ciba_says_so_rather_than_guessing_a_url(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            _ = self._agent().backchannel_endpoint
        self.assertIn("CIBA", str(ctx.exception))

    @pytest.mark.xfail(strict=True, reason="intended break: endpoints always come from the project's discovery")
    def test_an_explicit_token_endpoint_skips_discovery(self) -> None:
        a = AgentAuth(
            agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=PLATFORM, token_endpoint=TOKEN_ENDPOINT
        )
        self.assertEqual(a.token_endpoint, TOKEN_ENDPOINT)


class TestAgentToken(unittest.TestCase):
    TOKEN: ClassVar[dict[str, Any]] = {
        "access_token": "at_abc",
        "token_type": "Bearer",
        "expires_in": 3600,
        "agent_id": "agt_1",
        "type": "autonomous",
        "capabilities": ["send_email_external", "read_crm"],
        "scope": "send_email_external read_crm",
    }

    def _agent(self, **kw: Any) -> AgentAuth:
        return AgentAuth(
            agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=PLATFORM, token_endpoint=TOKEN_ENDPOINT, **kw
        )

    def test_sends_a_conformant_client_credentials_grant(self) -> None:
        net().queue(_response(self.TOKEN))
        a = self._agent(audience="https://api.acme.com")
        self.assertEqual(a.token(), "at_abc")
        req = net().requests[-1]
        self.assertEqual(req.headers["Content-type"], "application/x-www-form-urlencoded")
        form = net().forms()[-1]
        self.assertEqual(form["grant_type"], "client_credentials")
        self.assertEqual(form["client_id"], "agt_1")
        self.assertEqual(form["client_assertion_type"], CLIENT_ASSERTION_TYPE)
        self.assertEqual(len(form["client_assertion"].split(".")), 3)
        self.assertEqual(form["resource"], "https://api.acme.com")
        self.assertNotIn("audience", form)
        self.assertNotIn("client_secret", form)

    def test_every_request_names_this_client(self) -> None:
        net().queue(_response(self.TOKEN))
        self._agent().token()
        self.assertEqual(net().requests[-1].headers["User-agent"], USER_AGENT)
        self.assertNotIn("urllib", USER_AGENT)

    def test_discovery_names_this_client_too(self) -> None:
        seen: list[str] = []

        def record(request: httpx.Request) -> None:
            seen.append(request.headers["User-Agent"])

        client = httpx.Client(transport=httpx.MockTransport(net()), event_hooks={"request": [record]})
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(SyncTransport, "client", property(lambda _self: client))
            AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=PLATFORM).discovery()
        self.assertEqual(seen[-1], USER_AGENT)

    def test_capabilities_are_names(self) -> None:
        net().queue(_response(self.TOKEN))
        a = self._agent()
        self.assertEqual(a.capabilities(), frozenset(["send_email_external", "read_crm"]))
        self.assertTrue(a.can("send_email_external"))
        self.assertFalse(a.can("wire_transfer"))

    @pytest.mark.xfail(strict=True, reason="a private detail of 0.1 (_expires_at); renewal is tested in test_agents.py")
    def test_token_is_cached_and_refreshed_before_expiry(self) -> None:
        net().queue(_response(self.TOKEN), _response(self.TOKEN))
        a = self._agent()
        a.token()
        a.token()
        cast(Any, a)._expires_at = time.time() + a.refresh_skew - 1
        a.token()
        self.assertEqual(net().call_count, 2)

    def test_a_different_audience_is_not_served_from_cache(self) -> None:
        net().queue(_response(self.TOKEN), _response(self.TOKEN))
        a = self._agent(audience="https://api.acme.com")
        a.token()
        a.token(audience="https://other.acme.com")
        self.assertEqual(net().call_count, 2)

    def test_auth_header(self) -> None:
        net().queue(_response(self.TOKEN))
        self.assertEqual(self._agent().auth_header(), {"Authorization": "Bearer at_abc"})

    def test_a_refusal_is_a_refusal_and_is_not_retried(self) -> None:
        net().queue(_oauth_error("invalid_client", "client authentication failed", status=401))
        a = self._agent()
        with self.assertRaises(AgentAuthError) as ctx:
            a.token()
        self.assertEqual(ctx.exception.error, "invalid_client")
        self.assertEqual(ctx.exception.status, 401)
        self.assertIsInstance(ctx.exception, AuthFIError)
        self.assertEqual(net().call_count, 1)

    def test_a_body_with_no_access_token_is_an_error_not_a_none_token(self) -> None:
        net().queue(_response({"expires_in": 60}))
        with self.assertRaises(AuthFIError):
            self._agent().token()


class TestAgentCIBA(unittest.TestCase):
    DELEGATED: ClassVar[dict[str, Any]] = {
        "access_token": "at_delegated",
        "token_type": "Bearer",
        "expires_in": 900,
        "agent_id": "agt_1",
        "type": "delegated",
        "subject": "user:priya@acme.com",
        "capabilities": ["send_email_external"],
        "scope": "send_email_external",
    }

    def _agent(self) -> AgentAuth:
        return AgentAuth(
            agent_id="agt_1",
            key=AgentKey.generate("k1"),
            platform_url=PLATFORM,
            token_endpoint=TOKEN_ENDPOINT,
            backchannel_endpoint=cast(str, DISCOVERY["backchannel_authentication_endpoint"]),
        )

    @pytest.mark.xfail(strict=True, reason=DICT_RETURN)
    def test_request_carries_the_binding_message_and_client_authentication(self) -> None:
        net().queue(_response({"auth_req_id": "req_1", "expires_in": 300, "interval": 5}))
        out = self._agent().request_user_approval(
            "priya@acme.com", binding_message="Send the Q3 summary to Priya?", scope="send_email_external"
        )
        self.assertEqual(cast(Any, out)["auth_req_id"], "req_1")

    def test_a_request_with_nobody_to_ask_is_refused(self) -> None:
        with self.assertRaises(AuthFIError):
            self._agent().request_user_approval("")

    @pytest.mark.xfail(strict=True, reason=DICT_RETURN)
    def test_poll_waits_on_authorization_pending_then_returns_the_person(self) -> None:
        net().queue(
            _oauth_error("authorization_pending"), _oauth_error("authorization_pending"), _response(self.DELEGATED)
        )
        out = self._agent().await_user_token({"auth_req_id": "req_1", "interval": 5})
        self.assertEqual(cast(Any, out)["subject"], "user:priya@acme.com")

    def test_poll_sends_the_ciba_grant_with_a_fresh_assertion_each_time(self) -> None:
        net().queue(_oauth_error("authorization_pending"), _response(self.DELEGATED))
        self._agent().await_user_token({"auth_req_id": "req_1", "interval": 1}, sleep=_no_sleep)
        forms = net().forms()
        for form in forms:
            self.assertEqual(form["grant_type"], CIBA_GRANT_TYPE)
            self.assertEqual(form["auth_req_id"], "req_1")
        self.assertNotEqual(forms[0]["client_assertion"], forms[1]["client_assertion"])

    @pytest.mark.xfail(strict=True, reason="intended change: the first poll waits the interval (CIBA §7.3)")
    def test_slow_down_widens_the_interval(self) -> None:
        net().queue(_oauth_error("slow_down"), _oauth_error("authorization_pending"), _response(self.DELEGATED))
        self._agent().await_user_token({"auth_req_id": "req_1", "interval": 5})
        self.assertEqual(net().sleeps, [10, 10])

    def test_a_decision_stops_the_poll_immediately(self) -> None:
        for code in ("access_denied", "expired_token", "invalid_grant"):
            before = net().call_count
            net().queue(_oauth_error(code))
            with self.assertRaises(AgentAuthError) as ctx:
                self._agent().await_user_token({"auth_req_id": "req_1", "interval": 1}, sleep=_no_sleep)
            self.assertEqual(ctx.exception.error, code)
            self.assertEqual(net().call_count - before, 1)

    @pytest.mark.xfail(
        strict=True,
        reason="intended break: a wait that times out is WaitTimeout, never the server's authorization_pending",
    )
    def test_the_poll_gives_up_at_the_timeout(self) -> None:
        net().queue(*[_oauth_error("authorization_pending")] * 5)
        with self.assertRaises(AgentAuthError) as ctx:
            self._agent().await_user_token({"auth_req_id": "req_1", "interval": 5}, timeout=1, sleep=_no_sleep)
        self.assertEqual(ctx.exception.error, "authorization_pending")

    @pytest.mark.xfail(strict=True, reason=DICT_RETURN)
    def test_token_as_user_composes_request_and_poll(self) -> None:
        net().queue(_response({"auth_req_id": "req_1", "expires_in": 300, "interval": 1}), _response(self.DELEGATED))
        out = self._agent().token_as_user("priya@acme.com", binding_message="Send it?")
        self.assertEqual(cast(Any, out)["subject"], "user:priya@acme.com")


class TestAgentFromEnv(unittest.TestCase):
    NAMES = (
        "AUTHFI_AGENT_ID",
        "AUTHFI_AGENT_KEY_FILE",
        "AUTHFI_AGENT_KID",
        "AUTHFI_AGENT_PRIVATE_KEY",
        "AUTHFI_PLATFORM_URL",
        "AUTHFI_AGENT_AUDIENCE",
        "AUTHFI_ISSUER",
    )

    def setUp(self) -> None:
        self._saved = {k: os.environ.pop(k, None) for k in self.NAMES}

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    @pytest.mark.xfail(strict=True, reason=NO_DEFAULT + "; " + PRIVATE_FILE)
    def test_reads_the_key_from_a_file(self) -> None:
        key = AgentKey.generate("k1")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "agent.key")
            with open(path, "w") as fh:
                fh.write(key.private_pem())
            os.environ.update(AUTHFI_AGENT_ID="agt_1", AUTHFI_AGENT_KEY_FILE=path, AUTHFI_AGENT_KID="k1")
            a = AgentAuth.from_env()
            self.assertEqual(a.agent_id, "agt_1")

    @pytest.mark.xfail(strict=True, reason=NO_DEFAULT)
    def test_a_key_in_an_environment_variable_warns(self) -> None:
        os.environ.update(
            AUTHFI_AGENT_ID="agt_1",
            AUTHFI_AGENT_PRIVATE_KEY=AgentKey.generate("k1").private_pem(),
            AUTHFI_AGENT_KID="k1",
        )
        with self.assertWarns(UserWarning):
            AgentAuth.from_env()

    @pytest.mark.xfail(strict=True, reason=NO_DEFAULT + "; " + PRIVATE_FILE)
    def test_the_file_wins_when_both_are_set(self) -> None:
        file_key, env_key = AgentKey.generate("k1"), AgentKey.generate("k2")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "agent.key")
            with open(path, "w") as fh:
                fh.write(file_key.private_pem())
            os.environ.update(
                AUTHFI_AGENT_ID="agt_1",
                AUTHFI_AGENT_KEY_FILE=path,
                AUTHFI_AGENT_PRIVATE_KEY=env_key.private_pem(),
                AUTHFI_AGENT_KID="k1",
            )
            with self.assertWarns(UserWarning):
                a = AgentAuth.from_env()
            self.assertEqual(a.key.public_jwk(), file_key.public_jwk())

    def test_missing_id_and_missing_key_are_named(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            AgentAuth.from_env()
        self.assertIn("AUTHFI_AGENT_ID", str(ctx.exception))
        os.environ["AUTHFI_AGENT_ID"] = "agt_1"
        with self.assertRaises(AuthFIError) as ctx:
            AgentAuth.from_env()
        self.assertIn("AUTHFI_AGENT_KEY_FILE", str(ctx.exception))
