"""Tests for AuthFI Python SDK."""

import io
import os
import json
import time
import base64
import unittest
from urllib.error import HTTPError
from unittest.mock import patch, MagicMock
from authfi import (
    AuthFI,
    AuthFIError,
    AgentAuth,
    AgentAuthError,
    AgentKey,
    assert_public_jwk,
    CLIENT_ASSERTION_TYPE,
    CIBA_GRANT_TYPE,
    _b64decode,
)


class TestAuthFIInit(unittest.TestCase):
    def test_defaults(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")
        self.assertEqual(auth.tenant, "acme")
        self.assertEqual(auth.api_url, "https://api.authfi.io")
        self.assertEqual(auth.jwks_ttl, 300)
        self.assertTrue(auth.auto_sync)

    def test_custom_url(self):
        auth = AuthFI(tenant="acme", api_key="sk_test", api_url="https://custom.api.com")
        self.assertEqual(auth.api_url, "https://custom.api.com")

    def test_manage_url(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")
        # The edge dispatch is /<slug>/<path> — slug FIRST — and no /manage/ prefix exists.
        self.assertEqual(auth._manage_url, "https://api.authfi.io/acme/v1")

    def test_jwks_is_on_the_project_origin(self):
        # It used to be {api_url}/v1/{tenant}/.well-known/jwks.json, which 404s:
        # api.authfi.io answers 404 for every path this SDK built, and the tenant-in-path
        # form 404s even against a live project host. No JWKS means no token can be
        # verified, which is the whole job.
        auth = AuthFI(tenant="acme", api_key="sk_test")
        # The api-host shape stays correct for what it is for — API calls hang off /v1, and the
        # tenant base is the slug-first form, NOT /v1/<tenant>.
        self.assertEqual(auth._auth_url, "https://api.authfi.io/acme/v1")
        self.assertEqual(auth._tenant_base, "https://api.authfi.io/acme")
        # But the KEYS come from the issuer, which is a different host on purpose.
        self.assertEqual(auth.issuer, "https://acme.authfi.io")
        self.assertEqual(auth.jwks_url, "https://acme.authfi.io/.well-known/jwks.json")

    def test_base_domain_and_issuer_override(self):
        # A dev deployment lives under authfi.app and a self-hosted one under neither, so a
        # domain must never be hard-coded into an issuer.
        self.assertEqual(
            AuthFI(tenant="acme", api_key="k", base_domain="authfi.app").issuer,
            "https://acme.authfi.app",
        )
        self.assertEqual(
            AuthFI(tenant="acme", api_key="k", issuer="https://login.acme.example/").jwks_url,
            "https://login.acme.example/.well-known/jwks.json",
        )

    def test_api_url_does_not_move_the_keys(self):
        # Conflating "where I call the API" with "where this project publishes its keys" is
        # how the wrong host became the default in the first place.
        auth = AuthFI(tenant="acme", api_key="k", api_url="https://custom.api.com")
        self.assertEqual(auth.jwks_url, "https://acme.authfi.io/.well-known/jwks.json")


class TestTokenVerification(unittest.TestCase):
    def _make_token(self, payload, header=None):
        if header is None:
            header = {"alg": "RS256", "typ": "JWT", "kid": "test-key-1"}
        h = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
        p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        s = base64.urlsafe_b64encode(b"fakesig").rstrip(b"=").decode()
        return f"{h}.{p}.{s}"

    def test_rejects_invalid_format(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")
        with self.assertRaises(AuthFIError) as ctx:
            auth.verify_token("not-a-jwt")
        self.assertIn("Invalid token format", str(ctx.exception))

    def test_without_pyjwt_verification_refuses_rather_than_falling_back(self):
        """Without PyJWT there is NO unverified fallback.

        This test used to assert that an unsigned fallback decoder rejected an
        expired token — i.e. that the SDK still returned claims from tokens
        whose signature nobody checked, as long as the expiry looked fine. That
        fallback was removed because it is an auth bypass, and the test kept
        asserting the old behaviour and failing. The code is right; the
        assertion was stale.
        """
        auth = AuthFI(tenant="acme", api_key="sk_test")
        token = self._make_token({"sub": "usr_123", "exp": int(time.time()) - 3600})

        # Mock PyJWT import to fail.
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "jwt":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            with self.assertRaises(AuthFIError) as ctx:
                auth.verify_token(token)
            self.assertIn("PyJWT is required", str(ctx.exception))


def _rsa_signer():
    """An RSA keypair, as a real signer and as the JWK the issuer would publish."""
    import jwt
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    return key, jwk


class TestVerifyTokenFetchesTheJWKSItself(unittest.TestCase):
    """Verification must go through this SDK's own fetcher, not PyJWT's.

    PyJWT's `PyJWKClient` opens its own urllib connection and sends the default
    `Python-urllib/3.x`, which Cloudflare answers 403 for — so on every production origin
    verification failed at the edge and reported itself as a signature problem. These tests
    fail if anything reaches the network except through `_fetch_jwks`, which sends USER_AGENT.
    """

    def setUp(self):
        try:
            import jwt  # noqa: F401
        except ImportError:
            self.skipTest("PyJWT not installed")
        self.key, self.jwk = _rsa_signer()
        self.jwk["kid"] = "k1"
        self.auth = AuthFI(tenant="acme", api_key="sk_test")
        self.requests = []

    def _serve(self, keys):
        """Stand in for the network, recording every request the SDK actually makes."""
        def fake_urlopen(req, *a, **kw):
            self.requests.append(req)
            body = json.dumps({"keys": keys}).encode()
            res = MagicMock()
            res.read.return_value = body
            res.__enter__.return_value = res
            res.__exit__.return_value = False
            return res
        return patch("authfi.urlopen", side_effect=fake_urlopen)

    # `{}` is a meaningful value here — it means "a token carrying no kid" — so it cannot be
    # spelled with `headers or {...}`, which treats an empty dict as absent.
    _DEFAULT_HEADERS = object()

    def _token(self, claims=None, headers=_DEFAULT_HEADERS, key=None, alg="RS256"):
        import jwt

        body = {"sub": "usr_1", "iss": self.auth.issuer, "exp": int(time.time()) + 300}
        body.update(claims or {})
        if headers is self._DEFAULT_HEADERS:
            headers = {"kid": "k1"}
        return jwt.encode(body, key or self.key, algorithm=alg, headers=headers)

    def test_verifies_through_our_fetcher_with_the_named_user_agent(self):
        with self._serve([self.jwk]):
            claims = self.auth.verify_token(self._token())
        self.assertEqual(claims["sub"], "usr_1")
        # One request, ours, named. A PyJWKClient fetch would not appear here at all — it
        # would bypass this patch entirely and hit the real network.
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].full_url, self.auth.jwks_url)
        self.assertEqual(self.requests[0].get_header("User-agent"), "authfi-python-sdk/0.1.0")

    def test_the_jwks_is_cached_across_verifications(self):
        """PyJWKClient held a fresh client per call, so every verification refetched."""
        with self._serve([self.jwk]):
            self.auth.verify_token(self._token())
            self.auth.verify_token(self._token())
        self.assertEqual(len(self.requests), 1)

    def test_a_token_naming_an_unpublished_kid_is_refused(self):
        with self._serve([self.jwk]):
            with self.assertRaises(AuthFIError) as ctx:
                self.auth.verify_token(self._token(headers={"kid": "retired"}))
        self.assertIn("retired", str(ctx.exception))

    def test_a_kidless_token_is_refused_when_several_keys_are_published(self):
        """Trying keys until one verifies would let a retired key keep validating tokens."""
        _, second = _rsa_signer()
        second["kid"] = "k2"
        with self._serve([self.jwk, second]):
            with self.assertRaises(AuthFIError) as ctx:
                self.auth.verify_token(self._token(headers={}))
        self.assertIn("2 keys", str(ctx.exception))

    def test_a_kidless_token_verifies_against_a_single_published_key(self):
        with self._serve([self.jwk]):
            self.assertEqual(self.auth.verify_token(self._token(headers={}))["sub"], "usr_1")

    def test_the_algorithm_comes_from_the_key_not_the_header(self):
        """alg:none and HS256 confusion both die here: the key decides, the sender does not."""
        import jwt
        from cryptography.hazmat.primitives import serialization

        forged = ".".join([
            base64.urlsafe_b64encode(json.dumps({"alg": "none", "kid": "k1"}).encode())
            .rstrip(b"=").decode(),
            base64.urlsafe_b64encode(json.dumps({"sub": "attacker"}).encode())
            .rstrip(b"=").decode(),
            "",
        ])
        with self._serve([self.jwk]):
            with self.assertRaises(AuthFIError):
                self.auth.verify_token(forged)

        # And the classic HS256 confusion: sign with the issuer's PUBLIC key as an HMAC secret,
        # which succeeds against any verifier that takes its algorithm from the header. Built by
        # hand with hmac, because PyJWT refuses to encode it — and so would an attacker's own
        # tooling never be PyJWT.
        pub = self.key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        import hmac as _hmac
        import hashlib as _hashlib

        def seg(obj):
            return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

        signing_input = f'{seg({"alg": "HS256", "kid": "k1"})}.{seg({"sub": "attacker", "iss": self.auth.issuer})}'
        sig = base64.urlsafe_b64encode(
            _hmac.new(pub, signing_input.encode(), _hashlib.sha256).digest()
        ).rstrip(b"=").decode()
        with self._serve([self.jwk]):
            with self.assertRaises(AuthFIError):
                self.auth.verify_token(f"{signing_input}.{sig}")

    def test_an_ed25519_key_verifies_too(self):
        """Agent tokens under the AuthFI Agent Profile are EdDSA, not RS256."""
        import jwt
        from cryptography.hazmat.primitives.asymmetric import ed25519

        key = ed25519.Ed25519PrivateKey.generate()
        jwk = json.loads(jwt.algorithms.OKPAlgorithm.to_jwk(key.public_key()))
        jwk["kid"] = "ed1"
        with self._serve([jwk]):
            claims = self.auth.verify_token(
                self._token(headers={"kid": "ed1"}, key=key, alg="EdDSA")
            )
        self.assertEqual(claims["sub"], "usr_1")

    def test_a_token_from_another_issuer_is_refused(self):
        with self._serve([self.jwk]):
            with self.assertRaises(AuthFIError):
                self.auth.verify_token(self._token({"iss": "https://evil.example"}))

    def test_an_issuer_publishing_no_keys_is_named_as_such(self):
        with self._serve([]):
            with self.assertRaises(AuthFIError) as ctx:
                self.auth.verify_token(self._token())
        self.assertIn("published no keys", str(ctx.exception))


class TestPermissionRegistration(unittest.TestCase):
    def test_register(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")
        auth.register_permission("read:users", "Read user data")
        auth.register_permission("write:users")

        self.assertIn("read:users", auth._registered_permissions)
        self.assertEqual(auth._registered_permissions["read:users"], "Read user data")
        self.assertIn("write:users", auth._registered_permissions)

    def test_sync_empty(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")
        # Should not throw
        auth.sync()

    @patch("authfi.urlopen")
    def test_sync_sends_permissions(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"synced": 2, "total": 2}).encode()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_response

        auth = AuthFI(tenant="acme", api_key="sk_test")
        auth.register_permission("read:users", "Read")
        auth.register_permission("write:users", "Write")
        auth.sync()

        mock_urlopen.assert_called_once()
        call_args = mock_urlopen.call_args
        sent_data = json.loads(call_args[0][1])
        self.assertEqual(len(sent_data["permissions"]), 2)


class TestRequireDecorator(unittest.TestCase):
    def test_registers_permissions(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")

        @auth.require("read:users", "write:users")
        def handler():
            pass

        self.assertIn("read:users", auth._registered_permissions)
        self.assertIn("write:users", auth._registered_permissions)

    def test_require_role_decorator_exists(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")

        @auth.require_role("admin")
        def handler():
            pass

        # Should create a wrapper without error
        self.assertTrue(callable(handler))


class TestCurrentUser(unittest.TestCase):
    def test_returns_none_when_not_set(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")
        self.assertIsNone(auth.current_user())


class TestCloudCredentials(unittest.TestCase):
    def test_requires_app_id_and_secret(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")
        with self.assertRaises(AuthFIError):
            auth.cloud_credentials("token", "aws")

    def test_cloud_token_requires_credentials(self):
        auth = AuthFI(tenant="acme", api_key="sk_test")
        with self.assertRaises(AuthFIError):
            auth.cloud_token("token", "audience")


class TestB64Decode(unittest.TestCase):
    def test_decodes_padded(self):
        original = b'{"test": true}'
        encoded = base64.urlsafe_b64encode(original).rstrip(b"=").decode()
        decoded = _b64decode(encoded)
        self.assertEqual(decoded, original)

    def test_decodes_already_padded(self):
        original = b"hello"
        encoded = base64.urlsafe_b64encode(original).decode()
        decoded = _b64decode(encoded)
        self.assertEqual(decoded, original)


# =============================================================================
# Agent authentication
# =============================================================================

#: The project an agent is registered in. Every AgentAuth names it: there is no default.
ORIGIN = "https://acme.authfi.io"

DISCOVERY = {
    "issuer": ORIGIN,
    "algorithms": ["EdDSA"],
    "grant_types_supported": ["client_credentials", CIBA_GRANT_TYPE],
    "token_endpoint_auth_methods_supported": ["private_key_jwt"],
    "token_endpoint_auth_signing_alg_values_supported": ["EdDSA"],
    "backchannel_authentication_endpoint":
        ORIGIN + "/apis/regional/v1/ai/auth/backchannel-authorize",
    "endpoints": {
        "token": ORIGIN + "/apis/regional/v1/ai/auth/token",
        "backchannel_authorize":
            ORIGIN + "/apis/regional/v1/ai/auth/backchannel-authorize",
    },
}
TOKEN_ENDPOINT = DISCOVERY["endpoints"]["token"]


def _headers(pairs=()):
    """Response headers as urllib gives them (an email.message.Message), so repeats survive."""
    from email.message import Message
    msg = Message()
    for name, value in pairs:
        msg[name] = value
    return msg


def _response(body, headers=()):
    """A urlopen() context manager returning ``body`` as JSON."""
    res = MagicMock()
    res.read.return_value = json.dumps(body).encode()
    res.headers = _headers(headers)
    res.__enter__ = lambda s: s
    res.__exit__ = MagicMock(return_value=False)
    return res


def _oauth_error(code, description=None, status=400, headers=()):
    body = {"error": code}
    if description:
        body["error_description"] = description
    return HTTPError(
        TOKEN_ENDPOINT, status, code, _headers(headers), io.BytesIO(json.dumps(body).encode())
    )


def _http_error(url, status):
    return HTTPError(url, status, "error", _headers(), io.BytesIO(b"{}"))


#: The project's RFC 8414 metadata, as platform-beta publishes it (trimmed).
AS_METADATA_URL = ORIGIN + "/.well-known/oauth-authorization-server"
AGENT_CONFIGURATION_URL = ORIGIN + "/.well-known/agent-configuration"
OAUTH_TOKEN = ORIGIN + "/oauth/token"
AS_METADATA = {
    "issuer": ORIGIN,
    "token_endpoint": OAUTH_TOKEN,
    "grant_types_supported": ["client_credentials", CIBA_GRANT_TYPE],
    "token_endpoint_auth_methods_supported": ["client_secret_post", "private_key_jwt"],
    "token_endpoint_auth_signing_alg_values_supported": ["EdDSA", "ES256", "PS256"],
}


def _routes(table):
    """A urlopen() side effect answering by URL: a dict body, an exception, or a list of either."""
    def answer(req, data=None, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else req
        if url not in table:
            raise AssertionError("unexpected request to %s" % url)
        out = table[url]
        if isinstance(out, list):
            out = out.pop(0)
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, MagicMock) else _response(out)
    return answer


def _form(call):
    """Decode the form body urlopen() was called with."""
    from urllib.parse import parse_qs
    return {k: v[0] for k, v in parse_qs(call[0][1].decode()).items()}


class TestAgentKey(unittest.TestCase):
    def test_generate_and_public_jwk(self):
        key = AgentKey.generate("k1")
        jwk = key.public_jwk()
        self.assertEqual(jwk["kty"], "OKP")
        self.assertEqual(jwk["crv"], "Ed25519")
        self.assertEqual(jwk["kid"], "k1")
        self.assertEqual(len(_b64decode(jwk["x"])), 32)
        # The half that leaves the process carries no private component.
        self.assertNotIn("d", jwk)

    def test_key_without_kid_is_refused_at_construction(self):
        # The server COMPARES kid against the one key it stored. A key it cannot
        # name is not a credential, and finding that out at the first token call
        # means finding it out in production.
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.generate("")
        self.assertIn("key_id is required", str(ctx.exception))
        with self.assertRaises(AuthFIError):
            AgentKey.generate(None)

    def test_pem_round_trip(self):
        key = AgentKey.generate("k1")
        reloaded = AgentKey.from_pem(key.private_pem(), "k1")
        self.assertEqual(reloaded.public_jwk(), key.public_jwk())

    def test_pem_refuses_a_key_that_is_not_pkcs8(self):
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.from_pem("-----BEGIN OPENSSH PRIVATE KEY-----\nxx\n", "k1")
        msg = str(ctx.exception)
        self.assertIn("OPENSSH PRIVATE KEY", msg)
        self.assertIn("openssl genpkey -algorithm ed25519", msg)

    def test_pem_refuses_a_key_type_the_platform_does_not_take(self):
        from cryptography.hazmat.primitives.asymmetric import ec
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.from_pem(_pem(ec.generate_private_key(ec.SECP384R1())), "k1")
        msg = str(ctx.exception)
        self.assertIn("secp384r1", msg)
        self.assertIn("P-256 (ES256)", msg)

    def test_jwk_round_trip_and_kid_from_the_jwk(self):
        key = AgentKey.generate("k1")
        priv = dict(key.public_jwk())
        # A PRIVATE JWK is the one shape where 'd' belongs.
        from cryptography.hazmat.primitives import serialization
        raw = key._key.private_bytes(
            serialization.Encoding.Raw,
            serialization.PrivateFormat.Raw,
            serialization.NoEncryption(),
        )
        priv["d"] = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
        reloaded = AgentKey.from_jwk(priv)
        self.assertEqual(reloaded.key_id, "k1")
        self.assertEqual(reloaded.public_jwk(), key.public_jwk())

    def test_jwk_without_d_cannot_sign(self):
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.from_jwk(AgentKey.generate("k1").public_jwk())
        self.assertIn("no 'd'", str(ctx.exception))

    def test_from_file_sniffs_pem_and_jwk(self):
        import tempfile
        key = AgentKey.generate("k1")
        with tempfile.TemporaryDirectory() as d:
            pem_path = os.path.join(d, "agent.key")
            with open(pem_path, "w") as fh:
                fh.write(key.private_pem())
            self.assertEqual(
                AgentKey.from_file(pem_path, key_id="k1").public_jwk(), key.public_jwk()
            )

            empty = os.path.join(d, "empty.key")
            open(empty, "w").close()
            with self.assertRaises(AuthFIError):
                AgentKey.from_file(empty, key_id="k1")


def _pem(key, fmt="PKCS8"):
    """A private key as unencrypted PEM: PKCS#8, or the type's own TraditionalOpenSSL form."""
    from cryptography.hazmat.primitives import serialization
    return key.private_bytes(
        serialization.Encoding.PEM,
        getattr(serialization.PrivateFormat, fmt),
        serialization.NoEncryption(),
    ).decode("ascii")


def _private_jwk(key):
    """The private JWK of an AgentKey, built from its numbers the way another tool would write it."""
    from authfi import _int_b64
    jwk = dict(key.public_jwk())
    nums = key._key.private_numbers()
    if key.alg == "ES256":
        jwk["d"] = _int_b64(nums.private_value, 32)
    else:
        jwk.update(d=_int_b64(nums.d), p=_int_b64(nums.p), q=_int_b64(nums.q),
                   dp=_int_b64(nums.dmp1), dq=_int_b64(nums.dmq1), qi=_int_b64(nums.iqmp))
    return jwk


class TestAgentKeyTypes(unittest.TestCase):
    """Ed25519, P-256 and RSA: the three key types the platform advertises and accepts."""

    @classmethod
    def setUpClass(cls):
        cls.keys = {alg: AgentKey.generate("k-" + alg, alg) for alg in ("EdDSA", "ES256", "PS256")}

    def test_the_key_type_decides_the_algorithm(self):
        shapes = {"EdDSA": ("OKP", "Ed25519"), "ES256": ("EC", "P-256"), "PS256": ("RSA", None)}
        for alg, key in self.keys.items():
            self.assertEqual(key.alg, alg)
            jwk = key.public_jwk()
            self.assertEqual((jwk["kty"], jwk.get("crv")), shapes[alg])
            self.assertEqual(jwk["alg"], alg)
            self.assertEqual(jwk["kid"], "k-" + alg)
            for member in ("d", "p", "q", "dp", "dq", "qi"):
                self.assertNotIn(member, jwk)
        self.assertEqual(self.keys["PS256"]._key.key_size, 3072)
        self.assertEqual(self.keys["PS256"].public_jwk()["e"], "AQAB")
        # P-256 coordinates are fixed-width, even when the integer is shorter.
        ec = self.keys["ES256"].public_jwk()
        self.assertEqual((len(_b64decode(ec["x"])), len(_b64decode(ec["y"]))), (32, 32))

    def test_the_assertion_verifies_under_an_independent_jose_implementation(self):
        # PyJWT is a separate JOSE implementation: if it verifies the assertion
        # with only the PUBLIC JWK, the signature encoding is right for that alg
        # (ES256 as raw R||S, not DER; PS256 as RSASSA-PSS, not PKCS#1 v1.5).
        import jwt
        for alg, key in self.keys.items():
            assertion = key.client_assertion("agt_123", TOKEN_ENDPOINT)
            header = json.loads(_b64decode(assertion.split(".")[0]))
            self.assertEqual(header["alg"], alg)
            public = jwt.PyJWK(key.public_jwk(), algorithm=alg).key
            claims = jwt.decode(assertion, public, algorithms=[alg], audience=TOKEN_ENDPOINT)
            self.assertEqual(claims["sub"], "agt_123")
            if alg == "ES256":
                self.assertEqual(len(_b64decode(assertion.split(".")[2])), 64)

    def test_a_tampered_assertion_does_not_verify_for_any_key_type(self):
        import jwt
        for alg, key in self.keys.items():
            h, c, sig = key.client_assertion("agt_123", TOKEN_ENDPOINT).split(".")
            forged = json.loads(_b64decode(c))
            forged["sub"] = "agt_someone_else"
            c2 = base64.urlsafe_b64encode(json.dumps(forged).encode()).rstrip(b"=").decode()
            public = jwt.PyJWK(key.public_jwk(), algorithm=alg).key
            with self.assertRaises(jwt.InvalidSignatureError):
                jwt.decode("%s.%s.%s" % (h, c2, sig), public, algorithms=[alg], audience=TOKEN_ENDPOINT)

    def test_pss_salt_is_the_digest_length(self):
        # RFC 7518 §3.5. A verifier that pins the salt to 32 bytes accepts it.
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        key = self.keys["PS256"]
        sig = key.sign(b"input")
        key._key.public_key().verify(
            sig, b"input", padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=32), hashes.SHA256()
        )

    def test_pem_round_trip_for_every_type(self):
        for alg, key in self.keys.items():
            reloaded = AgentKey.from_pem(key.private_pem(), key.key_id)
            self.assertEqual(reloaded.alg, alg)
            self.assertEqual(reloaded.public_jwk(), key.public_jwk())

    def test_sec1_and_pkcs1_pem_load(self):
        # What `openssl ecparam -genkey` and `openssl genrsa` write.
        for alg, marker in (("ES256", "BEGIN EC PRIVATE KEY"), ("PS256", "BEGIN RSA PRIVATE KEY")):
            pem = _pem(self.keys[alg]._key, "TraditionalOpenSSL")
            self.assertIn(marker, pem)
            self.assertEqual(AgentKey.from_pem(pem, "k1").public_jwk()["kty"],
                             self.keys[alg].public_jwk()["kty"])

    def test_private_jwk_round_trip_for_ec_and_rsa(self):
        for alg in ("ES256", "PS256"):
            key = self.keys[alg]
            reloaded = AgentKey.from_jwk(_private_jwk(key))
            self.assertEqual(reloaded.key_id, key.key_id)
            self.assertEqual(reloaded.public_jwk(), key.public_jwk())

    def test_an_rsa_jwk_with_only_n_e_d_loads(self):
        jwk = _private_jwk(self.keys["PS256"])
        for member in ("p", "q", "dp", "dq", "qi"):
            jwk.pop(member)
        self.assertEqual(AgentKey.from_jwk(jwk).public_jwk(), self.keys["PS256"].public_jwk())

    def test_an_rsa_key_labelled_rs256_is_refused_locally(self):
        jwk = dict(_private_jwk(self.keys["PS256"]), alg="RS256")
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.from_jwk(jwk)
        msg = str(ctx.exception)
        self.assertIn("alg=RS256", msg)
        self.assertIn("signs PS256", msg)

    def test_a_p384_key_is_refused_locally(self):
        from cryptography.hazmat.primitives.asymmetric import ec
        with self.assertRaises(AuthFIError):
            AgentKey(ec.generate_private_key(ec.SECP384R1()), "k1")
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey.from_jwk({"kty": "EC", "crv": "P-384", "x": "AA", "y": "AA", "d": "AA"})
        self.assertIn("EC/P-384", str(ctx.exception))

    def test_an_ec_jwk_whose_d_is_not_its_point_is_refused(self):
        jwk = _private_jwk(self.keys["ES256"])
        jwk["d"] = _private_jwk(AgentKey.generate("k2", "ES256"))["d"]
        with self.assertRaises(AuthFIError):
            AgentKey.from_jwk(jwk)

    def test_rsa_keys_the_platform_refuses_are_refused_locally(self):
        from cryptography.hazmat.primitives.asymmetric import rsa
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey(rsa.generate_private_key(public_exponent=65537, key_size=1024), "k1")
        self.assertIn("1024 bits", str(ctx.exception))
        with self.assertRaises(AuthFIError) as ctx:
            AgentKey(rsa.generate_private_key(public_exponent=3, key_size=2048), "k1")
        self.assertIn("65537", str(ctx.exception))

    def test_generate_refuses_an_algorithm_it_cannot_make(self):
        for alg in ("RS256", "ES384", None):
            with self.assertRaises(AuthFIError):
                AgentKey.generate("k1", alg)


class TestPublicJWKRefusal(unittest.TestCase):
    def test_a_jwk_carrying_d_is_refused_as_burned(self):
        # Not stripped and carried on from: the key was sent, so it is gone.
        with self.assertRaises(AuthFIError) as ctx:
            assert_public_jwk({"kty": "OKP", "crv": "Ed25519", "x": "AA", "d": "secret"})
        msg = str(ctx.exception)
        self.assertIn("burned", msg)
        self.assertIn("'d'", msg)

    def test_wrong_curve_is_refused(self):
        with self.assertRaises(AuthFIError):
            assert_public_jwk({"kty": "EC", "crv": "P-384", "x": "AA", "y": "AA"})

    def test_an_rsa_private_member_is_refused_as_burned_too(self):
        jwk = dict(AgentKey.generate("k1", "PS256").public_jwk(), p="secret")
        with self.assertRaises(AuthFIError) as ctx:
            assert_public_jwk(jwk)
        self.assertIn("'p'", str(ctx.exception))

    def test_an_ec_key_without_y_has_no_public_key(self):
        jwk = AgentKey.generate("k1", "ES256").public_jwk()
        jwk.pop("y")
        with self.assertRaises(AuthFIError) as ctx:
            assert_public_jwk(jwk)
        self.assertIn("'y'", str(ctx.exception))

    def test_a_key_labelled_with_another_algorithm_is_refused(self):
        jwk = dict(AgentKey.generate("k1", "PS256").public_jwk(), alg="RS256")
        with self.assertRaises(AuthFIError) as ctx:
            assert_public_jwk(jwk)
        self.assertIn("RS256", str(ctx.exception))
        self.assertIn("PS256", str(ctx.exception))

    def test_an_rsa_key_outside_the_accepted_sizes_is_refused(self):
        from cryptography.hazmat.primitives.asymmetric import rsa
        from authfi import _int_b64
        small = rsa.generate_private_key(public_exponent=65537, key_size=1024).public_key()
        with self.assertRaises(AuthFIError) as ctx:
            assert_public_jwk({"kty": "RSA", "n": _int_b64(small.public_numbers().n), "e": "AQAB"})
        self.assertIn("1024 bits", str(ctx.exception))

    def test_public_jwk_passes(self):
        for alg in ("EdDSA", "ES256", "PS256"):
            jwk = AgentKey.generate("k1", alg).public_jwk()
            self.assertIs(assert_public_jwk(jwk), jwk)


class TestClientAssertion(unittest.TestCase):
    def setUp(self):
        self.key = AgentKey.generate("k1")
        self.jws = self.key.client_assertion("agt_123", TOKEN_ENDPOINT)
        self.header, self.claims, self.sig = self.jws.split(".")

    def test_header(self):
        header = json.loads(_b64decode(self.header))
        self.assertEqual(header["alg"], "EdDSA")
        self.assertEqual(header["typ"], "JWT")
        self.assertEqual(header["kid"], "k1")

    def test_claims(self):
        c = json.loads(_b64decode(self.claims))
        # RFC 7523 §3: in client authentication iss and sub are both the client id.
        self.assertEqual(c["iss"], "agt_123")
        self.assertEqual(c["sub"], "agt_123")
        # The audience is the token endpoint, so an assertion signed for one
        # endpoint cannot be replayed at another.
        self.assertEqual(c["aud"], TOKEN_ENDPOINT)
        self.assertTrue(c["jti"])
        # The server caps the lifetime at five minutes and refuses longer.
        self.assertEqual(c["exp"] - c["iat"], 300)
        self.assertLessEqual(abs(c["iat"] - int(time.time())), 5)

    def test_jti_is_fresh_every_time(self):
        # The server replay-caches jti; a reused one is refused outright.
        ids = {
            json.loads(_b64decode(self.key.client_assertion("agt_123", TOKEN_ENDPOINT).split(".")[1]))["jti"]
            for _ in range(5)
        }
        self.assertEqual(len(ids), 5)

    def test_signature_verifies_against_the_public_key(self):
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = Ed25519PublicKey.from_public_bytes(_b64decode(self.key.public_jwk()["x"]))
        # Verifies, or raises — this is the whole credential.
        pub.verify(
            _b64decode(self.sig), ("%s.%s" % (self.header, self.claims)).encode("ascii")
        )

    def test_a_tampered_payload_does_not_verify(self):
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        pub = Ed25519PublicKey.from_public_bytes(_b64decode(self.key.public_jwk()["x"]))
        forged = json.loads(_b64decode(self.claims))
        forged["sub"] = "agt_someone_else"
        payload = base64.urlsafe_b64encode(
            json.dumps(forged, separators=(",", ":")).encode()
        ).rstrip(b"=").decode()
        with self.assertRaises(InvalidSignature):
            pub.verify(
                _b64decode(self.sig), ("%s.%s" % (self.header, payload)).encode("ascii")
            )

    def test_segments_are_base64url_without_padding(self):
        for part in (self.header, self.claims, self.sig):
            self.assertNotIn("=", part)
            self.assertNotIn("+", part)
            self.assertNotIn("/", part)

    def test_requires_agent_id_and_audience(self):
        with self.assertRaises(AuthFIError):
            self.key.client_assertion("", TOKEN_ENDPOINT)
        with self.assertRaises(AuthFIError):
            self.key.client_assertion("agt_123", "")


class TestAgentAuthConstruction(unittest.TestCase):
    def test_requires_an_agent_key_and_offers_no_secret_fallback(self):
        # There is no client_secret parameter at all: the server picks the
        # credential from the agent's stored record, so a client that offered a
        # fallback would be offering a downgrade.
        with self.assertRaises(AuthFIError) as ctx:
            AgentAuth(agent_id="agt_1", key="a-client-secret", platform_url=ORIGIN)
        self.assertIn("AgentKey", str(ctx.exception))
        self.assertNotIn("client_secret", AgentAuth.__init__.__code__.co_varnames)

    def test_requires_an_agent_id(self):
        with self.assertRaises(AuthFIError):
            AgentAuth(agent_id="", key=AgentKey.generate("k1"), platform_url=ORIGIN)

    def test_the_project_address_is_required_and_has_no_default(self):
        # A default sent the agent's signed assertion to platform.authfi.io, a
        # project the agent is not registered in, where it was refused.
        for missing in (None, "", "   "):
            with self.assertRaises(AuthFIError) as ctx:
                AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=missing)
            self.assertIn("platform_url is required", str(ctx.exception))
        with self.assertRaises(AuthFIError):
            AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"))
        import authfi
        self.assertFalse(hasattr(authfi, "DEFAULT_PLATFORM_URL"))

    def test_an_address_without_a_scheme_is_refused(self):
        with self.assertRaises(AuthFIError) as ctx:
            AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url="acme.authfi.io")
        self.assertIn("https://acme.authfi.io", str(ctx.exception))

    def test_discovery_is_read_from_the_project(self):
        a = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=ORIGIN + "/")
        self.assertEqual(a.platform_url, ORIGIN)
        self.assertEqual(a.discovery_url, ORIGIN + "/.well-known/agent-configuration")


class TestAgentDiscovery(unittest.TestCase):
    def _agent(self, alg="EdDSA"):
        return AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1", alg), platform_url=ORIGIN)

    @patch("authfi.urlopen")
    def test_endpoints_come_from_discovery_and_are_fetched_once(self, mock_urlopen):
        mock_urlopen.side_effect = _routes({
            AS_METADATA_URL: _http_error(AS_METADATA_URL, 404),
            AGENT_CONFIGURATION_URL: DISCOVERY,
        })
        a = self._agent()
        self.assertEqual(a.token_endpoint, TOKEN_ENDPOINT)
        self.assertEqual(a.backchannel_endpoint, DISCOVERY["backchannel_authentication_endpoint"])
        a.token_endpoint
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("authfi.urlopen")
    def test_a_deployment_without_private_key_jwt_is_refused_up_front(self, mock_urlopen):
        doc = dict(DISCOVERY, token_endpoint_auth_methods_supported=["client_secret_post"])
        mock_urlopen.return_value = _response(doc)
        with self.assertRaises(AuthFIError) as ctx:
            self._agent().token_endpoint
        self.assertIn("private_key_jwt", str(ctx.exception))

    @patch("authfi.urlopen")
    def test_a_deployment_without_eddsa_is_refused(self, mock_urlopen):
        doc = dict(DISCOVERY, token_endpoint_auth_signing_alg_values_supported=["RS256"])
        mock_urlopen.return_value = _response(doc)
        with self.assertRaises(AuthFIError) as ctx:
            self._agent().token_endpoint
        self.assertIn("EdDSA", str(ctx.exception))

    @patch("authfi.urlopen")
    def test_the_advertised_list_is_checked_for_the_keys_own_algorithm(self, mock_urlopen):
        # The check used to be "EdDSA is advertised", whatever the key was.
        doc = dict(DISCOVERY, token_endpoint_auth_signing_alg_values_supported=["EdDSA", "ES256"])
        mock_urlopen.return_value = _response(doc)
        self.assertEqual(self._agent("ES256").token_endpoint, TOKEN_ENDPOINT)
        with self.assertRaises(AuthFIError) as ctx:
            self._agent("PS256").token_endpoint
        msg = str(ctx.exception)
        self.assertIn("PS256", msg)
        self.assertIn("EdDSA, ES256", msg)

    @patch("authfi.urlopen")
    def test_no_ciba_says_so_rather_than_guessing_a_url(self, mock_urlopen):
        doc = dict(DISCOVERY)
        doc.pop("backchannel_authentication_endpoint")
        doc["endpoints"] = {"token": TOKEN_ENDPOINT}
        mock_urlopen.return_value = _response(doc)
        with self.assertRaises(AuthFIError) as ctx:
            self._agent().backchannel_endpoint
        self.assertIn("CIBA", str(ctx.exception))

    def test_an_explicit_token_endpoint_skips_discovery(self):
        a = AgentAuth(
            agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=ORIGIN,
            token_endpoint=TOKEN_ENDPOINT,
        )
        with patch("authfi.urlopen") as mock_urlopen:
            self.assertEqual(a.token_endpoint, TOKEN_ENDPOINT)
            mock_urlopen.assert_not_called()


class TestTokenEndpointFromMetadata(unittest.TestCase):
    """The token endpoint is the project's RFC 8414 token_endpoint, not agent-configuration's."""

    def setUp(self):
        import authfi
        authfi._deprecations_reported.clear()

    def _agent(self, alg="EdDSA"):
        return AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1", alg), platform_url=ORIGIN)

    @staticmethod
    def _urls(mock_urlopen):
        return [c[0][0].full_url for c in mock_urlopen.call_args_list]

    @patch("authfi.urlopen")
    def test_the_metadata_names_the_token_endpoint(self, mock_urlopen):
        mock_urlopen.side_effect = _routes({AS_METADATA_URL: AS_METADATA})
        a = self._agent()
        self.assertEqual(a.token_endpoint, OAUTH_TOKEN)
        a.token_endpoint
        # Read once, and agent-configuration is not needed for the token endpoint.
        self.assertEqual(self._urls(mock_urlopen), [AS_METADATA_URL])

    @patch("authfi.urlopen")
    def test_the_agent_mints_and_signs_for_the_metadata_endpoint(self, mock_urlopen):
        mock_urlopen.side_effect = _routes({
            AS_METADATA_URL: AS_METADATA,
            OAUTH_TOKEN: {"access_token": "at_abc", "expires_in": 900},
        })
        self.assertEqual(self._agent().token(), "at_abc")
        self.assertEqual(self._urls(mock_urlopen)[-1], OAUTH_TOKEN)
        claims = json.loads(_b64decode(_form(mock_urlopen.call_args)["client_assertion"].split(".")[1]))
        self.assertEqual(claims["aud"], OAUTH_TOKEN)

    @patch("authfi.urlopen")
    def test_no_metadata_falls_back_to_agent_configuration(self, mock_urlopen):
        mock_urlopen.side_effect = _routes({
            AS_METADATA_URL: _http_error(AS_METADATA_URL, 404),
            AGENT_CONFIGURATION_URL: DISCOVERY,
        })
        with self.assertLogs("authfi", "INFO") as logs:
            self.assertEqual(self._agent().token_endpoint, TOKEN_ENDPOINT)
        self.assertIn("publishes no RFC 8414 metadata", logs.output[0])

    @patch("authfi.urlopen")
    def test_metadata_without_a_token_endpoint_falls_back(self, mock_urlopen):
        doc = dict(AS_METADATA)
        doc.pop("token_endpoint")
        mock_urlopen.side_effect = _routes({AS_METADATA_URL: doc, AGENT_CONFIGURATION_URL: DISCOVERY})
        self.assertEqual(self._agent().token_endpoint, TOKEN_ENDPOINT)

    @patch("authfi.urlopen")
    def test_metadata_naming_another_issuer_is_refused_not_used(self, mock_urlopen):
        doc = dict(AS_METADATA, issuer="https://evil.example", token_endpoint="https://evil.example/token")
        mock_urlopen.side_effect = _routes({AS_METADATA_URL: doc})
        with self.assertRaises(AuthFIError) as ctx:
            self._agent().token_endpoint
        self.assertIn("RFC 8414", str(ctx.exception))
        self.assertEqual(self._urls(mock_urlopen), [AS_METADATA_URL])

    @patch("authfi.urlopen")
    def test_a_failing_metadata_read_is_an_error_not_a_fallback(self, mock_urlopen):
        # Only "there is no metadata" falls back. A 5xx is reported as what it is.
        mock_urlopen.side_effect = _routes({AS_METADATA_URL: _http_error(AS_METADATA_URL, 503)})
        with self.assertRaises(AuthFIError) as ctx:
            self._agent().token_endpoint
        self.assertEqual(ctx.exception.status, 503)

    @patch("authfi.urlopen")
    def test_the_metadata_is_checked_for_this_keys_algorithm(self, mock_urlopen):
        doc = dict(AS_METADATA, token_endpoint_auth_signing_alg_values_supported=["EdDSA"])
        mock_urlopen.side_effect = _routes({AS_METADATA_URL: doc})
        with self.assertRaises(AuthFIError) as ctx:
            self._agent("ES256").token_endpoint
        self.assertIn("ES256", str(ctx.exception))

    @patch("authfi.urlopen")
    def test_a_deprecation_header_is_logged_once_with_its_successor(self, mock_urlopen):
        deprecated = {
            "Deprecation": "@1792108800",
            "Link": '<%s>; rel="successor-version"' % OAUTH_TOKEN,
        }.items()
        body = {"access_token": "at_abc", "expires_in": 900}
        mock_urlopen.side_effect = lambda req, data=None: _response(body, deprecated)
        a = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=ORIGIN,
                      token_endpoint=TOKEN_ENDPOINT)
        with self.assertLogs("authfi", "WARNING") as logs:
            a.token()
            a.token(force=True)
        self.assertEqual(len(logs.output), 1)
        self.assertIn(TOKEN_ENDPOINT + " is deprecated (Deprecation: @1792108800)", logs.output[0])
        self.assertIn("its successor is " + OAUTH_TOKEN, logs.output[0])

    @patch("authfi.urlopen")
    def test_a_refusal_from_a_deprecated_route_is_reported_too(self, mock_urlopen):
        mock_urlopen.side_effect = _oauth_error(
            "invalid_client", status=401, headers=[("Deprecation", "@1792108800")]
        )
        a = AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=ORIGIN,
                      token_endpoint=TOKEN_ENDPOINT)
        with self.assertLogs("authfi", "WARNING") as logs:
            with self.assertRaises(AgentAuthError):
                a.token()
        self.assertIn("is deprecated", logs.output[0])
        self.assertNotIn("successor", logs.output[0])

    @patch("authfi.urlopen")
    def test_no_deprecation_header_no_warning(self, mock_urlopen):
        import logging
        mock_urlopen.side_effect = _routes({
            AS_METADATA_URL: AS_METADATA,
            OAUTH_TOKEN: {"access_token": "at_abc", "expires_in": 900},
        })
        seen = []
        handler = logging.Handler()
        handler.emit = seen.append
        logging.getLogger("authfi").addHandler(handler)
        try:
            self._agent().token()
        finally:
            logging.getLogger("authfi").removeHandler(handler)
        self.assertEqual([r for r in seen if r.levelno >= logging.WARNING], [])

    def test_the_successor_is_found_among_several_links(self):
        from authfi import _successor
        links = ['<https://a.example/docs>; rel="help", <%s>; rel="successor-version"' % OAUTH_TOKEN]
        self.assertEqual(_successor(links), OAUTH_TOKEN)
        self.assertEqual(_successor(['<https://a.example/x>; rel=successor-version']), "https://a.example/x")
        self.assertIsNone(_successor(['<https://a.example/docs>; rel="help"']))
        self.assertIsNone(_successor([]))


class TestAgentToken(unittest.TestCase):
    TOKEN = {
        "access_token": "at_abc",
        "token_type": "Bearer",
        "expires_in": 3600,
        "agent_id": "agt_1",
        "type": "autonomous",
        "capabilities": ["send_email_external", "read_crm"],
        "scope": "send_email_external read_crm",
    }

    def _agent(self, **kw):
        return AgentAuth(
            agent_id="agt_1",
            key=AgentKey.generate("k1"),
            platform_url=ORIGIN,
            token_endpoint=TOKEN_ENDPOINT,
            **kw
        )

    @patch("authfi.urlopen")
    def test_sends_a_conformant_client_credentials_grant(self, mock_urlopen):
        mock_urlopen.return_value = _response(self.TOKEN)
        a = self._agent(audience="https://api.acme.com")
        self.assertEqual(a.token(), "at_abc")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.get_header("Content-type"), "application/x-www-form-urlencoded")
        form = _form(mock_urlopen.call_args)
        self.assertEqual(form["grant_type"], "client_credentials")
        self.assertEqual(form["client_id"], "agt_1")
        self.assertEqual(form["client_assertion_type"], CLIENT_ASSERTION_TYPE)
        self.assertEqual(len(form["client_assertion"].split(".")), 3)
        # RFC 8707 spelling: the endpoint prefers `resource` over `audience`.
        self.assertEqual(form["resource"], "https://api.acme.com")
        self.assertNotIn("audience", form)
        # No secret, ever.
        self.assertNotIn("client_secret", form)

    @patch("authfi.urlopen")
    def test_every_request_names_this_client(self, mock_urlopen):
        # Cloudflare answers 403 to the default Python-urllib agent on both the
        # discovery document and the token endpoint (measured 2026-09-23), so an
        # unnamed request never reaches the service at all.
        from authfi import USER_AGENT
        mock_urlopen.return_value = _response(self.TOKEN)
        self._agent().token()
        self.assertEqual(
            mock_urlopen.call_args[0][0].get_header("User-agent"), USER_AGENT
        )
        self.assertNotIn("urllib", USER_AGENT)

    @patch("authfi.urlopen")
    def test_discovery_names_this_client_too(self, mock_urlopen):
        from authfi import USER_AGENT
        mock_urlopen.return_value = _response(DISCOVERY)
        AgentAuth(agent_id="agt_1", key=AgentKey.generate("k1"), platform_url=ORIGIN).discovery()
        self.assertEqual(
            mock_urlopen.call_args[0][0].get_header("User-agent"), USER_AGENT
        )

    @patch("authfi.urlopen")
    def test_capabilities_are_names(self, mock_urlopen):
        mock_urlopen.return_value = _response(self.TOKEN)
        a = self._agent()
        self.assertEqual(
            a.capabilities(), frozenset(["send_email_external", "read_crm"])
        )
        self.assertTrue(a.can("send_email_external"))
        self.assertFalse(a.can("wire_transfer"))

    @patch("authfi.urlopen")
    def test_token_is_cached_and_refreshed_before_expiry(self, mock_urlopen):
        mock_urlopen.return_value = _response(self.TOKEN)
        a = self._agent()
        a.token()
        a.token()
        self.assertEqual(mock_urlopen.call_count, 1)

        # Inside the refresh skew, the token is re-minted rather than handed out
        # in the second it dies.
        a._expires_at = time.time() + a.refresh_skew - 1
        a.token()
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("authfi.urlopen")
    def test_a_different_audience_is_not_served_from_cache(self, mock_urlopen):
        mock_urlopen.return_value = _response(self.TOKEN)
        a = self._agent(audience="https://api.acme.com")
        a.token()
        a.token(audience="https://other.acme.com")
        self.assertEqual(mock_urlopen.call_count, 2)

    @patch("authfi.urlopen")
    def test_auth_header(self, mock_urlopen):
        mock_urlopen.return_value = _response(self.TOKEN)
        self.assertEqual(
            self._agent().auth_header(), {"Authorization": "Bearer at_abc"}
        )

    @patch("authfi.urlopen")
    def test_a_refusal_is_a_refusal_and_is_not_retried(self, mock_urlopen):
        mock_urlopen.side_effect = _oauth_error(
            "invalid_client", "client authentication failed", status=401
        )
        a = self._agent()
        with self.assertRaises(AgentAuthError) as ctx:
            a.token()
        self.assertEqual(ctx.exception.error, "invalid_client")
        self.assertEqual(ctx.exception.status, 401)
        self.assertIsInstance(ctx.exception, AuthFIError)
        # One request. Re-signing the same rejected credential is just load.
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("authfi.urlopen")
    def test_a_body_with_no_access_token_is_an_error_not_a_none_token(self, mock_urlopen):
        mock_urlopen.return_value = _response({"expires_in": 60})
        with self.assertRaises(AuthFIError):
            self._agent().token()


class TestAgentCIBA(unittest.TestCase):
    DELEGATED = {
        "access_token": "at_delegated",
        "token_type": "Bearer",
        "expires_in": 900,
        "agent_id": "agt_1",
        "type": "delegated",
        "subject": "user:priya@acme.com",
        "capabilities": ["send_email_external"],
        "scope": "send_email_external",
    }

    def _agent(self):
        return AgentAuth(
            agent_id="agt_1",
            key=AgentKey.generate("k1"),
            platform_url=ORIGIN,
            token_endpoint=TOKEN_ENDPOINT,
            backchannel_endpoint=DISCOVERY["backchannel_authentication_endpoint"],
        )

    @patch("authfi.urlopen")
    def test_request_carries_the_binding_message_and_client_authentication(self, mock_urlopen):
        mock_urlopen.return_value = _response(
            {"auth_req_id": "req_1", "expires_in": 300, "interval": 5}
        )
        out = self._agent().request_user_approval(
            "priya@acme.com",
            binding_message="Send the Q3 summary to Priya?",
            scope="send_email_external",
        )
        self.assertEqual(out["auth_req_id"], "req_1")
        form = _form(mock_urlopen.call_args)
        self.assertEqual(form["login_hint"], "priya@acme.com")
        self.assertEqual(form["binding_message"], "Send the Q3 summary to Priya?")
        self.assertEqual(form["scope"], "send_email_external")
        # CIBA §7.1 — the same credential as the token endpoint.
        self.assertEqual(form["client_assertion_type"], CLIENT_ASSERTION_TYPE)
        self.assertEqual(len(form["client_assertion"].split(".")), 3)

    def test_a_request_with_nobody_to_ask_is_refused(self):
        with self.assertRaises(AuthFIError):
            self._agent().request_user_approval("")

    @patch("authfi.urlopen")
    def test_poll_waits_on_authorization_pending_then_returns_the_person(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _oauth_error("authorization_pending"),
            _oauth_error("authorization_pending"),
            _response(self.DELEGATED),
        ]
        slept = []
        out = self._agent().await_user_token(
            {"auth_req_id": "req_1", "interval": 5}, sleep=slept.append
        )
        # sub is the PERSON; act.sub inside the JWT is the agent.
        self.assertEqual(out["subject"], "user:priya@acme.com")
        self.assertEqual(slept, [5, 5])

    @patch("authfi.urlopen")
    def test_poll_sends_the_ciba_grant_with_a_fresh_assertion_each_time(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _oauth_error("authorization_pending"),
            _response(self.DELEGATED),
        ]
        self._agent().await_user_token({"auth_req_id": "req_1", "interval": 1},
                                       sleep=lambda s: None)
        forms = [_form(c) for c in mock_urlopen.call_args_list]
        for form in forms:
            self.assertEqual(form["grant_type"], CIBA_GRANT_TYPE)
            self.assertEqual(form["auth_req_id"], "req_1")
        # The server replay-caches jti, so a reused assertion would be refused.
        self.assertNotEqual(forms[0]["client_assertion"], forms[1]["client_assertion"])

    @patch("authfi.urlopen")
    def test_slow_down_widens_the_interval(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _oauth_error("slow_down"),
            _oauth_error("authorization_pending"),
            _response(self.DELEGATED),
        ]
        slept = []
        self._agent().await_user_token(
            {"auth_req_id": "req_1", "interval": 5}, sleep=slept.append
        )
        self.assertEqual(slept, [10, 10])

    @patch("authfi.urlopen")
    def test_a_decision_stops_the_poll_immediately(self, mock_urlopen):
        for code in ("access_denied", "expired_token", "invalid_grant"):
            mock_urlopen.reset_mock()
            mock_urlopen.side_effect = _oauth_error(code)
            with self.assertRaises(AgentAuthError) as ctx:
                self._agent().await_user_token(
                    {"auth_req_id": "req_1", "interval": 1}, sleep=lambda s: None
                )
            self.assertEqual(ctx.exception.error, code)
            self.assertEqual(mock_urlopen.call_count, 1)

    @patch("authfi.urlopen")
    def test_the_poll_gives_up_at_the_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = _oauth_error("authorization_pending")
        with self.assertRaises(AgentAuthError) as ctx:
            self._agent().await_user_token(
                {"auth_req_id": "req_1", "interval": 5}, timeout=1, sleep=lambda s: None
            )
        self.assertEqual(ctx.exception.error, "authorization_pending")

    @patch("authfi.urlopen")
    def test_token_as_user_composes_request_and_poll(self, mock_urlopen):
        mock_urlopen.side_effect = [
            _response({"auth_req_id": "req_1", "expires_in": 300, "interval": 1}),
            _response(self.DELEGATED),
        ]
        out = self._agent().token_as_user("priya@acme.com", binding_message="Send it?")
        self.assertEqual(out["subject"], "user:priya@acme.com")


class TestAgentFromEnv(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: os.environ.pop(k, None)
            for k in (
                "AUTHFI_AGENT_ID",
                "AUTHFI_AGENT_KEY_FILE",
                "AUTHFI_AGENT_KID",
                "AUTHFI_AGENT_PRIVATE_KEY",
                "AUTHFI_PLATFORM_URL",
                "AUTHFI_AGENT_AUDIENCE",
            )
        }

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_reads_the_key_from_a_file(self):
        import tempfile
        key = AgentKey.generate("k1")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "agent.key")
            with open(path, "w") as fh:
                fh.write(key.private_pem())
            os.environ["AUTHFI_AGENT_ID"] = "agt_1"
            os.environ["AUTHFI_AGENT_KEY_FILE"] = path
            os.environ["AUTHFI_AGENT_KID"] = "k1"
            os.environ["AUTHFI_PLATFORM_URL"] = ORIGIN
            os.environ["AUTHFI_AGENT_AUDIENCE"] = "https://api.acme.com"
            a = AgentAuth.from_env()
            self.assertEqual(a.agent_id, "agt_1")
            self.assertEqual(a.key.public_jwk(), key.public_jwk())
            self.assertEqual(a.audience, "https://api.acme.com")

    def test_a_key_in_an_environment_variable_warns(self):
        # Supported, because some platforms give you nothing else — but it is
        # visible in ps, inherited by children and captured in crash dumps.
        os.environ["AUTHFI_AGENT_ID"] = "agt_1"
        os.environ["AUTHFI_AGENT_PRIVATE_KEY"] = AgentKey.generate("k1").private_pem()
        os.environ["AUTHFI_AGENT_KID"] = "k1"
        os.environ["AUTHFI_PLATFORM_URL"] = ORIGIN
        with self.assertWarns(UserWarning):
            AgentAuth.from_env()

    def test_the_file_wins_when_both_are_set(self):
        import tempfile
        file_key, env_key = AgentKey.generate("k1"), AgentKey.generate("k2")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "agent.key")
            with open(path, "w") as fh:
                fh.write(file_key.private_pem())
            os.environ["AUTHFI_AGENT_ID"] = "agt_1"
            os.environ["AUTHFI_AGENT_KEY_FILE"] = path
            os.environ["AUTHFI_AGENT_PRIVATE_KEY"] = env_key.private_pem()
            os.environ["AUTHFI_AGENT_KID"] = "k1"
            os.environ["AUTHFI_PLATFORM_URL"] = ORIGIN
            with self.assertWarns(UserWarning):
                a = AgentAuth.from_env()
            self.assertEqual(a.key.public_jwk(), file_key.public_jwk())

    def test_missing_id_and_missing_key_are_named(self):
        with self.assertRaises(AuthFIError) as ctx:
            AgentAuth.from_env()
        self.assertIn("AUTHFI_AGENT_ID", str(ctx.exception))

        os.environ["AUTHFI_AGENT_ID"] = "agt_1"
        with self.assertRaises(AuthFIError) as ctx:
            AgentAuth.from_env()
        self.assertIn("AUTHFI_AGENT_KEY_FILE", str(ctx.exception))

    def test_the_project_address_is_required_from_the_environment_too(self):
        os.environ["AUTHFI_AGENT_ID"] = "agt_1"
        os.environ["AUTHFI_AGENT_PRIVATE_KEY"] = AgentKey.generate("k1").private_pem()
        os.environ["AUTHFI_AGENT_KID"] = "k1"
        with self.assertWarns(UserWarning):
            with self.assertRaises(AuthFIError) as ctx:
                AgentAuth.from_env()
        self.assertIn("AUTHFI_PLATFORM_URL is not set", str(ctx.exception))
        # An override counts as setting it.
        with self.assertWarns(UserWarning):
            a = AgentAuth.from_env(platform_url=ORIGIN)
        self.assertEqual(a.platform_url, ORIGIN)
        os.environ["AUTHFI_PLATFORM_URL"] = "https://other.authfi.io"
        with self.assertWarns(UserWarning):
            self.assertEqual(AgentAuth.from_env(platform_url=ORIGIN).platform_url, ORIGIN)


if __name__ == "__main__":
    unittest.main()
