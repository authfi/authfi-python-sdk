# pyright: reportDeprecated=false
"""``main``'s ``test_authfi.py`` (7d5cfeb), user half, run against 0.2.

Every test of ``main`` is here under its own name, with ``main``'s assertions:

- a test that patched ``authfi.urlopen`` (a private transport detail) is ported to
  ``httpx.MockTransport`` with the same assertions, and its project is given the audience 0.2
  requires (``audience=``), the one argument it adds;
- an intended break is a strict ``xfail`` naming the changelog row, so it fails the suite the day
  it starts passing.

The agent half is ported with the agent module (``test_main_agent_half.py``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import unittest
from typing import Any, ClassVar, cast

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from jwt.algorithms import OKPAlgorithm, RSAAlgorithm

from authfi import USER_AGENT, AuthFI, AuthFIError

pytestmark = pytest.mark.filterwarnings("ignore::authfi.errors.AuthFIDeprecationWarning")

AUDIENCE_REQUIRED = "intended break: a token is never verified without an audience (CHANGELOG, verify_token row)"
KID_REQUIRED = "intended break: a token with no kid is refused (CHANGELOG, verify_token row)"
SYNC_RETIRED = "intended break: register_permission and sync are retired; no route serves them"
PRIVATE = "a private helper of the 0.1 module that 0.2 does not have"


class TestAuthFIInit(unittest.TestCase):
    def test_defaults(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        self.assertEqual(auth.tenant, "acme")
        self.assertEqual(auth.api_url, "https://api.authfi.io")
        self.assertEqual(auth.jwks_ttl, 300)
        self.assertTrue(auth.auto_sync)

    def test_custom_url(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test", api_url="https://custom.api.com")
        self.assertEqual(auth.api_url, "https://custom.api.com")

    @pytest.mark.xfail(strict=True, reason=PRIVATE + " (_manage_url, for the retired api.authfi.io host)")
    def test_manage_url(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        self.assertEqual(cast(Any, auth)._manage_url, "https://api.authfi.io/acme/v1")

    @pytest.mark.xfail(strict=True, reason=PRIVATE + " (_auth_url and _tenant_base)")
    def test_jwks_is_on_the_project_origin(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        self.assertEqual(cast(Any, auth)._auth_url, "https://api.authfi.io/acme/v1")
        self.assertEqual(cast(Any, auth)._tenant_base, "https://api.authfi.io/acme")
        self.assertEqual(auth.issuer, "https://acme.authfi.io")
        self.assertEqual(auth.jwks_url, "https://acme.authfi.io/.well-known/jwks.json")

    def test_base_domain_and_issuer_override(self) -> None:
        self.assertEqual(AuthFI(tenant="acme", api_key="k", base_domain="authfi.app").issuer, "https://acme.authfi.app")
        self.assertEqual(
            AuthFI(tenant="acme", api_key="k", issuer="https://login.acme.example/").jwks_url,
            "https://login.acme.example/.well-known/jwks.json",
        )

    def test_api_url_does_not_move_the_keys(self) -> None:
        auth = AuthFI(tenant="acme", api_key="k", api_url="https://custom.api.com")
        self.assertEqual(auth.jwks_url, "https://acme.authfi.io/.well-known/jwks.json")


class TestTokenVerification(unittest.TestCase):
    def _make_token(self, payload: dict[str, Any], header: dict[str, Any] | None = None) -> str:
        if header is None:
            header = {"alg": "RS256", "typ": "JWT", "kid": "test-key-1"}
        h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
        p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        s = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
        return f"{h}.{p}.{s}"

    @pytest.mark.xfail(strict=True, reason=AUDIENCE_REQUIRED)
    def test_rejects_invalid_format(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        with self.assertRaises(AuthFIError) as ctx:
            auth.verify_token("not-a-jwt")
        self.assertIn("Invalid token format", str(ctx.exception))

    @pytest.mark.xfail(strict=True, reason=AUDIENCE_REQUIRED + "; PyJWT is a required dependency")
    def test_without_pyjwt_verification_refuses_rather_than_falling_back(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        token = self._make_token({"sub": "usr_123", "exp": int(time.time()) - 3600})
        with self.assertRaises(AuthFIError) as ctx:
            auth.verify_token(token)
        self.assertIn("PyJWT is required", str(ctx.exception))


def _rsa_signer() -> tuple[rsa.RSAPrivateKey, dict[str, Any]]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    return key, cast(dict[str, Any], jwk)


class TestVerifyTokenFetchesTheJWKSItself(unittest.TestCase):
    """Ported from ``patch("authfi.urlopen")`` to httpx.MockTransport, with the same assertions.

    0.2 reads ``jwks_uri`` from discovery, so the stand-in network also serves the discovery
    document; the assertions are about the key-set requests, as ``main``'s were. The project gets
    ``audience=["api"]``, and the tokens carry what the strict verifier requires (``typ``, ``aud``,
    ``tenant_id``).
    """

    _DEFAULT_HEADERS: ClassVar[object] = object()

    def setUp(self) -> None:
        self.key, self.jwk = _rsa_signer()
        self.jwk["kid"] = "k1"
        self.requests: list[httpx.Request] = []
        self.keys: list[dict[str, Any]] = [self.jwk]

        def serve(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            if request.url.path == "/.well-known/openid-configuration":
                return httpx.Response(
                    200,
                    json={
                        "issuer": "https://acme.authfi.io",
                        "token_endpoint": "https://acme.authfi.io/oauth/token",
                        "jwks_uri": "https://acme.authfi.io/.well-known/jwks.json",
                    },
                )
            return httpx.Response(200, json={"keys": self.keys})

        self.auth = AuthFI(
            tenant="acme",
            api_key="sk_test",
            audience=["api"],
            http_client=httpx.Client(transport=httpx.MockTransport(serve)),
        )

    def _serve(self, keys: list[dict[str, Any]]) -> None:
        self.keys = keys

    def _jwks_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests if r.url.path.endswith("jwks.json")]

    def _token(
        self,
        claims: dict[str, Any] | None = None,
        headers: object = _DEFAULT_HEADERS,
        key: Any = None,
        alg: str = "RS256",
    ) -> str:
        body: dict[str, Any] = {
            "sub": "usr_1",
            "iss": self.auth.issuer,
            "exp": int(time.time()) + 300,
            "aud": "api",
            "tenant_id": "t1",
        }
        body.update(claims or {})
        header = {"kid": "k1"} if headers is self._DEFAULT_HEADERS else dict(cast(dict[str, Any], headers))
        header.setdefault("typ", "at+jwt")
        return jwt.encode(body, key or self.key, algorithm=alg, headers=header)

    def test_verifies_through_our_fetcher_with_the_named_user_agent(self) -> None:
        claims = self.auth.verify_token(self._token())
        self.assertEqual(claims["sub"], "usr_1")
        self.assertEqual(len(self._jwks_requests()), 1)
        self.assertEqual(str(self._jwks_requests()[0].url), self.auth.jwks_url)
        self.assertEqual(self._jwks_requests()[0].headers["User-Agent"], USER_AGENT)

    def test_the_jwks_is_cached_across_verifications(self) -> None:
        self.auth.verify_token(self._token())
        self.auth.verify_token(self._token())
        self.assertEqual(len(self._jwks_requests()), 1)

    def test_a_token_naming_an_unpublished_kid_is_refused(self) -> None:
        with self.assertRaises(AuthFIError) as ctx:
            self.auth.verify_token(self._token(headers={"kid": "retired"}))
        self.assertIn("retired", str(ctx.exception))

    def test_a_kidless_token_is_refused_when_several_keys_are_published(self) -> None:
        _, second = _rsa_signer()
        second["kid"] = "k2"
        self._serve([self.jwk, second])
        with self.assertRaises(AuthFIError):
            self.auth.verify_token(self._token(headers={}))

    @pytest.mark.xfail(strict=True, reason=KID_REQUIRED)
    def test_a_kidless_token_verifies_against_a_single_published_key(self) -> None:
        self.assertEqual(self.auth.verify_token(self._token(headers={}))["sub"], "usr_1")

    def test_the_algorithm_comes_from_the_key_not_the_header(self) -> None:
        forged = ".".join(
            [
                base64.urlsafe_b64encode(json.dumps({"alg": "none", "kid": "k1"}).encode()).rstrip(b"=").decode(),
                base64.urlsafe_b64encode(json.dumps({"sub": "attacker"}).encode()).rstrip(b"=").decode(),
                "",
            ]
        )
        with self.assertRaises(AuthFIError):
            self.auth.verify_token(forged)
        pub = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM, format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

        def seg(obj: dict[str, Any]) -> str:
            return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

        signing_input = f"{seg({'alg': 'HS256', 'kid': 'k1'})}.{seg({'sub': 'attacker', 'iss': self.auth.issuer})}"
        sig = (
            base64.urlsafe_b64encode(hmac.new(pub, signing_input.encode(), hashlib.sha256).digest())
            .rstrip(b"=")
            .decode()
        )
        with self.assertRaises(AuthFIError):
            self.auth.verify_token(f"{signing_input}.{sig}")

    def test_an_ed25519_key_verifies_too(self) -> None:
        key = ed25519.Ed25519PrivateKey.generate()
        jwk = cast(dict[str, Any], json.loads(OKPAlgorithm.to_jwk(key.public_key())))
        jwk["kid"] = "ed1"
        self._serve([jwk])
        claims = self.auth.verify_token(self._token(headers={"kid": "ed1"}, key=key, alg="EdDSA"))
        self.assertEqual(claims["sub"], "usr_1")

    def test_a_token_from_another_issuer_is_refused(self) -> None:
        with self.assertRaises(AuthFIError):
            self.auth.verify_token(self._token({"iss": "https://evil.example"}))

    @pytest.mark.xfail(strict=True, reason="0.2 refuses it as token_key_unknown, worded per kid")
    def test_an_issuer_publishing_no_keys_is_named_as_such(self) -> None:
        self._serve([])
        with self.assertRaises(AuthFIError) as ctx:
            self.auth.verify_token(self._token())
        self.assertIn("published no keys", str(ctx.exception))


class TestPermissionRegistration(unittest.TestCase):
    @pytest.mark.xfail(strict=True, reason=SYNC_RETIRED)
    def test_register(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        auth.register_permission("read:users", "Read user data")
        self.assertIn("read:users", cast(Any, auth)._registered_permissions)

    @pytest.mark.xfail(strict=True, reason=SYNC_RETIRED)
    def test_sync_empty(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        auth.sync()

    @pytest.mark.xfail(strict=True, reason=SYNC_RETIRED)
    def test_sync_sends_permissions(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        auth.register_permission("read:users", "Read")
        auth.sync()


class TestRequireDecorator(unittest.TestCase):
    @pytest.mark.xfail(strict=True, reason=SYNC_RETIRED + " (require no longer registers)")
    def test_registers_permissions(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")

        @auth.require("read:users", "write:users")
        def handler() -> None:
            pass

        self.assertIn("read:users", cast(Any, auth)._registered_permissions)

    def test_require_role_decorator_exists(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")

        @auth.require_role("admin")
        def handler() -> None:
            pass

        self.assertTrue(callable(handler))


class TestCurrentUser(unittest.TestCase):
    def test_returns_none_when_not_set(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        self.assertIsNone(auth.current_user())


class TestCloudCredentials(unittest.TestCase):
    def test_requires_app_id_and_secret(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        with self.assertRaises(AuthFIError):
            auth.cloud_credentials("token", "aws")

    def test_cloud_token_requires_credentials(self) -> None:
        auth = AuthFI(tenant="acme", api_key="sk_test")
        with self.assertRaises(AuthFIError):
            auth.cloud_token("token", "audience")


class TestB64Decode(unittest.TestCase):
    @pytest.mark.xfail(strict=True, reason=PRIVATE + " (_b64decode)")
    def test_decodes_padded(self) -> None:
        import authfi

        decode = cast(Any, authfi)._b64decode
        original = b'{"test": true}'
        self.assertEqual(decode(base64.urlsafe_b64encode(original).rstrip(b"=").decode()), original)

    @pytest.mark.xfail(strict=True, reason=PRIVATE + " (_b64decode)")
    def test_decodes_already_padded(self) -> None:
        import authfi

        decode = cast(Any, authfi)._b64decode
        self.assertEqual(decode(base64.urlsafe_b64encode(b"hello").decode()), b"hello")
