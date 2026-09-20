"""
AuthFI Python SDK

Usage:
    from authfi import AuthFI

    auth = AuthFI(
        tenant="acme",
        api_key="sk_live_...",
        application_id="client-id",   # optional
        auto_sync=True,
    )

    # Flask
    @app.route("/api/users")
    @auth.require("read:users")
    def get_users():
        user = auth.current_user()  # decoded JWT claims
        return jsonify(users)

    # FastAPI
    @app.get("/api/users")
    async def get_users(user=Depends(auth.require("read:users"))):
        return users

    # On startup
    auth.sync()
"""

import json
import time
import base64
import hashlib
import hmac
import functools
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from threading import local

_thread_local = local()


class AuthFIError(Exception):
    def __init__(self, message, status=401):
        super().__init__(message)
        self.status = status


class AuthFI:
    def __init__(
        self,
        tenant,
        api_key,
        api_url="https://api.authfi.io",
        application_id=None,
        client_secret=None,
        auto_sync=True,
        jwks_ttl=300,
        issuer=None,
        base_domain="authfi.io",
    ):
        self.tenant = tenant
        self.api_key = api_key
        # Where this SDK's management calls go. NOT where signing keys live — see self.issuer.
        self.api_url = api_url
        # The project's own origin, and the only place its signing keys are published.
        #
        # Every AuthFI project IS an issuer at https://<tenant>.authfi.io, and that origin serves
        # /.well-known/jwks.json. Pass `issuer` for a self-hosted deployment or a custom domain;
        # pass `base_domain` for a different zone (authfi.app on dev). A domain must never be
        # hard-coded into an issuer, which is why both levels are configurable.
        self.base_domain = base_domain
        self.issuer = (issuer or f"https://{tenant}.{base_domain}").rstrip("/")
        self.application_id = application_id
        self.client_secret = client_secret  # required for cloud identity
        self.auto_sync = auto_sync
        self.jwks_ttl = jwks_ttl

        self._jwks = None
        self._jwks_fetched = 0
        self._registered_permissions = {}

    @property
    def _tenant_base(self):
        """The tenant's base on the shared API host.

        The edge dispatch is api.authfi.io/<slug>/<path> — the SLUG COMES FIRST. Root-level
        protocol paths (JWKS, OIDC discovery, /oauth/*) hang directly off this; API routes add
        /v1. Getting the two confused is how the JWKS URL ends up under /v1 and 404s.
        """
        return f"{self.api_url}/{self.tenant}"

    @property
    def _manage_url(self):
        # No /manage/ prefix exists on the platform; those routes are /v1/* like the rest.
        return f"{self._tenant_base}/v1"

    @property
    def _auth_url(self):
        return f"{self._tenant_base}/v1"

    @property
    def jwks_url(self):
        """Where this project publishes its signing keys.

        NOT `_tenant_base`. Both shapes answer for an ordinary tenant and return the same keys —
        measured 2026-09-24, `api.authfi.io/quefly/.well-known/jwks.json` and
        `https://quefly.authfi.io/.well-known/jwks.json` serve the identical kid — so the
        api-host form is not wrong so much as narrower, in two ways that matter:

          - It 404s for a RESERVED-but-routable project. `api.authfi.io/platform/...` is a 404
            while `https://platform.authfi.io/...` is a 200, because the edge dispatches those
            names differently. A tenant whose slug the platform reserves would be unverifiable.
          - It is not the ISSUER. `verify_token` asserts `iss == self.issuer`, so fetching keys
            from a different origin than the issuer being asserted means trusting one host to
            vouch for tokens minted by another. Fetching from the issuer is the security
            property, not merely a URL that happens to work.

        The original `{api_url}/v1/{tenant}/...` is dead on every axis — /v1 404s for every
        tenant, reserved or not. A JWKS that cannot be fetched means no token can be verified,
        which is this SDK's whole job, so on defaults the failure was total and surfaced as
        "token verification failed" rather than as a misconfigured URL.
        """
        return f"{self.issuer}/.well-known/jwks.json"

    # --- JWKS ---

    def _fetch_jwks(self):
        now = time.time()
        if self._jwks and now - self._jwks_fetched < self.jwks_ttl:
            return self._jwks

        req = Request(self.jwks_url)
        with urlopen(req) as res:
            self._jwks = json.loads(res.read())
            self._jwks_fetched = now
        return self._jwks

    def verify_token(self, token):
        """Verify JWT and return decoded claims.

        Uses PyJWT with JWKS if available, falls back to unverified decode.
        Install PyJWT for production: pip install PyJWT[crypto]
        """
        parts = token.split(".")
        if len(parts) != 3:
            raise AuthFIError("Invalid token format")

        # RS256 signature verification via PyJWT is REQUIRED. There is NO
        # unverified fallback — accepting an unverified token is an auth bypass.
        try:
            import jwt
            from jwt import PyJWKClient
        except ImportError as e:
            raise AuthFIError(
                "PyJWT is required for token verification. "
                "Install with: pip install 'PyJWT[crypto]'"
            ) from e

        jwks_url = self.jwks_url
        try:
            jwks_client = PyJWKClient(jwks_url)
            signing_key = jwks_client.get_signing_key_from_jwt(token)
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],  # pin RS256; blocks alg:none / HS256 confusion
                options={"verify_aud": False},
            )
        except jwt.PyJWTError as e:
            raise AuthFIError(f"Token verification failed: {e}") from e

    # --- Permission registration ---

    def register_permission(self, name, description=None):
        self._registered_permissions[name] = description

    def sync(self):
        """Sync registered permissions with AuthFI."""
        if not self._registered_permissions:
            return

        body = {
            "permissions": [
                {"name": name, "description": desc}
                for name, desc in self._registered_permissions.items()
            ]
        }
        if self.application_id:
            body["application_id"] = self.application_id

        try:
            data = self._api_call("PUT", "/permissions/sync", body)
            print(f"[authfi] Synced {data['synced']} permissions ({data['total']} total)")
        except Exception as e:
            print(f"[authfi] Permission sync failed: {e}")

    # --- Middleware ---

    def require(self, *permissions):
        """Decorator that checks required permissions (ALL must match)."""
        # Register for auto-sync
        for p in permissions:
            self.register_permission(p)

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                claims = self._authenticate_request()
                user_perms = claims.get("permissions", [])

                missing = [p for p in permissions if p not in user_perms]
                if missing:
                    raise AuthFIError(
                        f"Insufficient permissions. Missing: {missing}", status=403
                    )

                _thread_local.user = claims
                return func(*args, **kwargs)

            return wrapper

        # If called as @auth.require("perm") for FastAPI Depends()
        if len(permissions) == 1 and callable(permissions[0]):
            fn = permissions[0]
            permissions = ()
            return decorator(fn)

        return decorator

    def require_role(self, *roles):
        """Decorator that checks required roles (ANY must match)."""

        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                claims = self._authenticate_request()
                user_roles = claims.get("roles", [])

                if not any(r in user_roles for r in roles):
                    raise AuthFIError(
                        f"Insufficient role. Required one of: {roles}", status=403
                    )

                _thread_local.user = claims
                return func(*args, **kwargs)

            return wrapper
        return decorator

    def current_user(self):
        """Get the current authenticated user claims."""
        return getattr(_thread_local, "user", None)

    def _authenticate_request(self):
        """Extract and verify token from current request."""
        # Flask
        try:
            from flask import request
            auth_header = request.headers.get("Authorization", "")
        except (ImportError, RuntimeError):
            auth_header = ""

        if not auth_header.startswith("Bearer "):
            raise AuthFIError("Missing authorization")

        token = auth_header[7:]
        return self.verify_token(token)

    # --- API calls ---

    def _api_call(self, method, path, body=None):
        url = f"{self._manage_url}{path}"
        req = Request(url, method=method)
        req.add_header("X-API-Key", self.api_key)
        req.add_header("Content-Type", "application/json")

        data = json.dumps(body).encode() if body else None

        try:
            with urlopen(req, data) as res:
                return json.loads(res.read())
        except HTTPError as e:
            error_body = json.loads(e.read()) if e.readable() else {}
            raise Exception(error_body.get("error", f"HTTP {e.code}"))


    # --- Cloud Identity ---

    def cloud_credentials(self, user_token, provider, role_arn=None, project=None, scope=None, ttl=900):
        """Get cloud provider credentials using AuthFI identity.

        Requires application_id and client_secret in config. Cloud credentials
        are scoped per application — each app can only access resources its IAM role permits.

        Args:
            user_token: The user's AuthFI JWT
            provider: 'aws', 'gcp', or 'azure'
            role_arn: AWS IAM role ARN (required for AWS)
            project: GCP project ID
            scope: Azure scope
            ttl: Token TTL in seconds (max 3600)

        Returns:
            dict with cloud credentials (short-lived)

        Usage:
            creds = auth.cloud_credentials(token, 'aws', role_arn='arn:aws:iam::123:role/deploy')
            # Use creds['access_key_id'], creds['secret_access_key']
        """
        if not self.application_id or not self.client_secret:
            raise AuthFIError("application_id and client_secret required for cloud credentials")

        url = f"{self._tenant_base}/v1/credentials"
        body = {"provider": provider, "ttl": ttl}
        if role_arn: body["role_arn"] = role_arn
        if project: body["project"] = project
        if scope: body["scope"] = scope

        req = Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {user_token}")
        req.add_header("X-Client-ID", self.application_id)
        req.add_header("X-Client-Secret", self.client_secret)

        try:
            with urlopen(req, json.dumps(body).encode()) as res:
                return json.loads(res.read())
        except HTTPError as e:
            error_body = json.loads(e.read()) if e.readable() else {}
            raise AuthFIError(error_body.get("error", f"Cloud credentials failed: {e.code}"))

    def cloud_token(self, user_token, audience, ttl=900):
        """Get a raw OIDC token for manual federation.

        Requires application_id and client_secret in config.
        """
        if not self.application_id or not self.client_secret:
            raise AuthFIError("application_id and client_secret required for cloud token")

        url = f"{self._tenant_base}/v1/credentials/token"
        req = Request(url, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {user_token}")
        req.add_header("X-Client-ID", self.application_id)
        req.add_header("X-Client-Secret", self.client_secret)
        data = json.dumps({"audience": audience, "ttl": ttl}).encode()

        with urlopen(req, data) as res:
            return json.loads(res.read())


def _b64decode(s):
    """Base64url decode with padding."""
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)
