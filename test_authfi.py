"""Tests for AuthFI Python SDK."""

import json
import time
import base64
import unittest
from unittest.mock import patch, MagicMock
from authfi import AuthFI, AuthFIError, _b64decode


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

    def test_rejects_expired_token_fallback(self):
        """Without PyJWT, fallback decoder should reject expired tokens."""
        auth = AuthFI(tenant="acme", api_key="sk_test")
        token = self._make_token({"sub": "usr_123", "exp": int(time.time()) - 3600})

        # Mock PyJWT import to fail (force fallback)
        import builtins
        original_import = builtins.__import__
        def mock_import(name, *args, **kwargs):
            if name == "jwt":
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", side_effect=mock_import):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with self.assertRaises(AuthFIError) as ctx:
                    auth.verify_token(token)
                self.assertIn("expired", str(ctx.exception))


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


if __name__ == "__main__":
    unittest.main()
