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

Agent authentication (the AI Agent Auth Protocol):

    from authfi import AgentAuth

    agent = AgentAuth.from_env()

    if agent.can("send_email_external"):
        requests.post(url, headers=agent.auth_header(), json=payload)
"""

import os
import json
import time
import uuid
import base64
import hashlib
import hmac
import functools
import warnings
from urllib.request import Request, urlopen
from urllib.parse import urlencode
from urllib.error import HTTPError
from threading import local

__version__ = "0.1.0"

# Every request this SDK makes identifies itself.
#
# NOT cosmetic. AuthFI's production origins sit behind Cloudflare, whose bot
# rules answer 403 to the default `Python-urllib/3.x` User-Agent — MEASURED
# 2026-09-23 against both /.well-known/agent-configuration and the agent token
# endpoint, where the same requests with any named agent answer 200. urllib sends
# that default unless told otherwise, so an SDK built on it is refused at the
# edge before the service ever sees the request, and the failure reads as an
# outage rather than as a blocked client.
USER_AGENT = "authfi-python-sdk/" + __version__

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
        # Named, or Cloudflare refuses the default Python-urllib agent — see USER_AGENT.
        req.add_header("User-Agent", USER_AGENT)
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
        req.add_header("User-Agent", USER_AGENT)

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
        req.add_header("User-Agent", USER_AGENT)

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
        req.add_header("User-Agent", USER_AGENT)
        data = json.dumps({"audience": audience, "ttl": ttl}).encode()

        with urlopen(req, data) as res:
            return json.loads(res.read())


def _b64decode(s):
    """Base64url decode with padding."""
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def _b64encode(raw):
    """Base64url encode without padding — RFC 7515 §2."""
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


# =============================================================================
# Agent authentication — the AI Agent Auth Protocol (AAP)
# =============================================================================
#
# An AuthFI agent holds its OWN Ed25519 keypair. It authenticates by signing a
# short assertion (RFC 7523 §2.2 private_key_jwt); the private half never leaves
# the process and is never sent to AuthFI. That is the whole point, and the
# difference between an agent credential and a bearer artifact the IETF
# agent-auth draft's security considerations call an antipattern.
#
# There is deliberately no second credential path here. The platform's discovery
# document advertises "token_endpoint_auth_methods_supported": ["private_key_jwt"]
# and nothing else, registration REQUIRES a public JWK and issues no secret, and
# the server picks the credential from the agent's stored record rather than from
# the request — so a client that offered to "fall back to a secret" would be
# offering a downgrade the server will not honour and an attacker would like.

#: The RFC 7523 §2.1 assertion type. The token endpoint requires it alongside the
#: assertion, and refuses any other value rather than ignoring it.
CLIENT_ASSERTION_TYPE = "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"

#: The OpenID CIBA grant — an agent redeeming a person's out-of-band approval.
CIBA_GRANT_TYPE = "urn:openid:params:grant-type:ciba"

#: Where discovery lives when no platform URL is given.
DEFAULT_PLATFORM_URL = "https://platform.authfi.io"

# How long an assertion may claim to live. Short by intent: an assertion is a
# bearer credential for its lifetime, so the window in which a captured one is
# worth anything is the window to minimise. The server caps this at five minutes
# (plus a minute of clock skew) and refuses anything longer.
_ASSERTION_LIFETIME = 300


class AgentAuthError(AuthFIError):
    """A refusal from the agent token endpoint, carrying the OAuth error code.

    ``error`` is one of the codes RFC 6749 §5.2 and OpenID CIBA §11 register —
    ``invalid_client``, ``invalid_grant``, ``authorization_pending``,
    ``slow_down``, ``access_denied``, ``expired_token``, ``invalid_target`` and
    so on. It is a CLOSED set on the server side precisely so a client can
    branch on it, so branch on ``err.error``, never on the message text.
    """

    def __init__(self, error, description=None, status=400):
        message = error if not description else "%s: %s" % (error, description)
        super().__init__(message, status=status)
        self.error = error
        self.description = description


def assert_public_jwk(jwk):
    """Refuse a JWK that carries a private component, and return it otherwise.

    Call this before sending a JWK anywhere — registration, a config file, a
    ticket. A key that has been sent is no longer secret, so a private ``d``
    turning up where a public key belongs is not a formatting slip to strip and
    carry on from: the key is burned and must be replaced. The server refuses it
    for the same reason, and quietly using the public half would leave its owner
    believing a secret they just leaked is still secret.
    """
    if not isinstance(jwk, dict):
        raise AuthFIError("A JWK must be a dict (RFC 8037 OKP)", status=400)
    if jwk.get("d"):
        raise AuthFIError(
            "This JWK contains a private key component ('d') where a PUBLIC key "
            "belongs. Refusing to use or send it. Treat that key as burned — "
            "generate a new one (AgentKey.generate) and register its public half.",
            status=400,
        )
    if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
        raise AuthFIError(
            "An agent key must be an Ed25519 OKP JWK, but this one is %s/%s"
            % (jwk.get("kty"), jwk.get("crv")),
            status=400,
        )
    if not jwk.get("x"):
        raise AuthFIError("This JWK has no 'x' — there is no public key in it", status=400)
    return jwk


def _ed25519():
    """Import the Ed25519 primitives, with an actionable error if absent.

    ``cryptography`` is already in this SDK's dependency closure (PyJWT[crypto]
    pulls it in for RS256 verification), so this adds nothing to install. It is
    imported lazily anyway, matching how PyJWT is treated above: importing
    ``authfi`` must not fail on a machine that only uses part of the SDK.
    """
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
        )
    except ImportError as e:  # pragma: no cover - exercised only on a broken install
        raise AuthFIError(
            "Agent authentication needs Ed25519 signing. "
            "Install with: pip install cryptography",
            status=500,
        ) from e
    return serialization, Ed25519PrivateKey


class AgentKey:
    """An agent's Ed25519 signing key, held in this process and never sent.

    The ``key_id`` is required, and required HERE rather than at the first token
    request: it is the ``kid`` the agent registered, and the server COMPARES it
    to the one key it stored for that agent. A key with no id is not a
    credential — the verifier would have to guess — so a construction that
    cannot produce a usable assertion fails at construction.
    """

    def __init__(self, private_key, key_id):
        if not key_id or not str(key_id).strip():
            raise AuthFIError(
                "key_id is required: it is the 'kid' registered with the agent's "
                "public JWK, and the server compares the assertion header against "
                "it. A key the verifier cannot name is not a credential.",
                status=400,
            )
        self._key = private_key
        self.key_id = str(key_id).strip()

    # --- Loading ---

    @classmethod
    def from_file(cls, path, key_id=None):
        """Load a private key from a FILE — the way to hold an agent key.

        A file can be mode 0600, mounted as a secret, and is not inherited by
        every child process. A private key in an environment variable leaks into
        ``ps``, into crash dumps, into any subprocess the agent spawns and into
        most log-scraping tooling, which is why this is the documented path and
        ``AUTHFI_AGENT_PRIVATE_KEY`` is only the fallback.

        Accepts either a PKCS#8 PEM (``-----BEGIN PRIVATE KEY-----``) or an
        RFC 8037 OKP JWK, sniffed from the content. A JWK's own ``kid`` is used
        when ``key_id`` is not given.
        """
        with open(path, "rb") as fh:
            data = fh.read()
        if not data.strip():
            raise AuthFIError("The agent key file %s is empty" % path, status=400)
        if data.lstrip()[:1] == b"{":
            return cls.from_jwk(json.loads(data.decode("utf-8")), key_id=key_id)
        return cls.from_pem(data, key_id)

    @classmethod
    def from_pem(cls, pem, key_id):
        """Load an Ed25519 private key from a PKCS#8 PEM."""
        serialization, Ed25519PrivateKey = _ed25519()
        raw = pem.encode("utf-8") if isinstance(pem, str) else pem
        head = raw.strip().splitlines()[0].decode("utf-8", "replace") if raw.strip() else ""
        if b"BEGIN PRIVATE KEY" not in raw:
            # Name the thing that was actually handed over. Letting this fail
            # inside the crypto library produces an error that reads like a
            # platform fault rather than "you gave me the wrong file".
            hint = ("it is a %s" % head.replace("-", "").strip()) if head.startswith("-----BEGIN") \
                else "it is not PEM at all"
            raise AuthFIError(
                "The agent key must be an Ed25519 private key in PKCS#8 PEM form "
                "(-----BEGIN PRIVATE KEY-----), but %s. "
                "Generate one with: openssl genpkey -algorithm ed25519" % hint,
                status=400,
            )
        try:
            key = serialization.load_pem_private_key(raw, password=None)
        except Exception as e:
            raise AuthFIError(
                "The agent key is not a usable private key: %s. "
                "An encrypted key is not supported — decrypt it before mounting it." % e,
                status=400,
            ) from e
        if not isinstance(key, Ed25519PrivateKey):
            raise AuthFIError(
                "The agent key is a %s key, but AuthFI agents sign with Ed25519 "
                "(discovery advertises EdDSA and nothing else)." % type(key).__name__,
                status=400,
            )
        return cls(key, key_id)

    @classmethod
    def from_jwk(cls, jwk, key_id=None):
        """Load a private key from an RFC 8037 OKP JWK — the one shape where ``d`` belongs."""
        serialization, Ed25519PrivateKey = _ed25519()
        if isinstance(jwk, (str, bytes)):
            jwk = json.loads(jwk)
        if jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519":
            raise AuthFIError(
                "An agent key JWK must be OKP/Ed25519, but this one is %s/%s"
                % (jwk.get("kty"), jwk.get("crv")),
                status=400,
            )
        if not jwk.get("d"):
            raise AuthFIError(
                "This JWK has no 'd' — it is the PUBLIC half. Signing needs the "
                "private key the agent kept.",
                status=400,
            )
        try:
            key = Ed25519PrivateKey.from_private_bytes(_b64decode(jwk["d"]))
        except Exception as e:
            raise AuthFIError("The JWK's 'd' is not a usable Ed25519 private key: %s" % e,
                              status=400) from e
        return cls(key, key_id or jwk.get("kid"))

    @classmethod
    def generate(cls, key_id):
        """Generate a fresh agent keypair, to register and then hold.

        Pair with :meth:`public_jwk` (what you register) and :meth:`private_pem`
        (what you write to a 0600 file and never send anywhere).
        """
        _serialization, Ed25519PrivateKey = _ed25519()
        return cls(Ed25519PrivateKey.generate(), key_id)

    # --- Using ---

    def public_jwk(self):
        """The public JWK to register — and the ONLY half that ever leaves here."""
        serialization, _ = _ed25519()
        raw = self._key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        # Validated on the way out as well: this is the value a caller is about
        # to POST to /register, and the SDK must never be the thing that hands
        # them a private component to upload.
        return assert_public_jwk(
            {"kty": "OKP", "crv": "Ed25519", "x": _b64encode(raw), "kid": self.key_id}
        )

    def private_pem(self):
        """The PKCS#8 PEM, for writing to a protected file at bootstrap."""
        serialization, _ = _ed25519()
        return self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode("ascii")

    def sign(self, data):
        """Raw Ed25519 signature over ``data``."""
        return self._key.sign(data)

    def client_assertion(self, agent_id, audience, lifetime=_ASSERTION_LIFETIME):
        """Build and sign one RFC 7523 §2.2 client assertion.

        ``agent_id`` becomes both ``iss`` and ``sub``: in client authentication
        the agent is both the issuer of the assertion and the subject it speaks
        about, and the server requires both to name the client being
        authenticated — which stops an assertion minted for one agent being
        presented as another's.

        ``audience`` is the token endpoint. The server checks it, so an
        assertion signed for one endpoint cannot be replayed at another.
        """
        if not agent_id or not str(agent_id).strip():
            raise AuthFIError("agent_id is required to build a client assertion", status=400)
        if not audience or not str(audience).strip():
            raise AuthFIError("audience is required to build a client assertion", status=400)

        now = int(time.time())
        header = {"alg": "EdDSA", "typ": "JWT", "kid": self.key_id}
        claims = {
            "iss": agent_id,
            "sub": agent_id,
            "aud": audience,
            # The replay id. A FRESH one per assertion is what lets the server
            # refuse a second use of the same credential — and it refuses
            # assertion auth outright when it has no replay cache rather than
            # accepting replays and hoping, so a reused jti is a hard failure.
            "jti": str(uuid.uuid4()),
            "iat": now,
            "exp": now + int(lifetime),
        }
        signing_input = "%s.%s" % (_b64json(header), _b64json(claims))
        signature = self.sign(signing_input.encode("ascii"))
        return "%s.%s" % (signing_input, _b64encode(signature))


def _b64json(obj):
    return _b64encode(json.dumps(obj, separators=(",", ":")).encode("utf-8"))


class AgentAuth:
    """Agent authentication against AuthFI, bootstrapped from discovery.

    Usage::

        from authfi import AgentAuth, AgentKey

        agent = AgentAuth(
            agent_id="agt_...",
            key=AgentKey.from_file("/run/secrets/agent.key", key_id="k1"),
            audience="https://api.acme.com",   # optional, RFC 8707 resource
        )

        if agent.can("send_email_external"):
            requests.post(url, headers=agent.auth_header(), json=body)

    Endpoints come from ``/.well-known/agent-configuration`` rather than being
    hard-coded: the token endpoint moves with the project's origin (a tenant
    subdomain, a custom domain), and the assertion's ``aud`` is built from the
    same document the server builds its own check from.
    """

    def __init__(
        self,
        agent_id,
        key,
        platform_url=DEFAULT_PLATFORM_URL,
        audience=None,
        token_endpoint=None,
        backchannel_endpoint=None,
        discovery_url=None,
        refresh_skew=60,
    ):
        if not agent_id or not str(agent_id).strip():
            raise AuthFIError("agent_id is required", status=400)
        if not isinstance(key, AgentKey):
            raise AuthFIError(
                "key must be an AgentKey — build one with AgentKey.from_file(...). "
                "An agent authenticates with a key it holds; there is no secret "
                "fallback, because the platform issues no secret to a key-registered "
                "agent and would refuse one if it did.",
                status=400,
            )
        self.agent_id = str(agent_id).strip()
        self.key = key
        self.platform_url = (platform_url or DEFAULT_PLATFORM_URL).rstrip("/")
        self.audience = audience
        self.discovery_url = discovery_url or (
            self.platform_url + "/.well-known/agent-configuration"
        )
        # Refresh this many seconds BEFORE expiry, so a token is never presented
        # to a resource server in the second it dies.
        self.refresh_skew = refresh_skew

        self._token_endpoint = token_endpoint
        self._backchannel_endpoint = backchannel_endpoint
        self._discovery = None

        self._access_token = None
        self._token_audience = None
        self._expires_at = 0
        self._capabilities = frozenset()

    # --- Construction ---

    @classmethod
    def from_env(cls, **overrides):
        """Build from the environment.

        ==============================  ====================================
        ``AUTHFI_AGENT_ID``             the agent id (``agt_...``)
        ``AUTHFI_AGENT_KEY_FILE``       path to the private key — PREFERRED
        ``AUTHFI_AGENT_KID``            the registered ``kid``
        ``AUTHFI_AGENT_PRIVATE_KEY``    PEM inline — fallback, warns
        ``AUTHFI_PLATFORM_URL``         defaults to https://platform.authfi.io
        ``AUTHFI_AGENT_AUDIENCE``       optional resource the token is for
        ==============================  ====================================

        The file wins when both are set. A private key in an environment
        variable is readable in ``ps``, is copied into every child process and
        lands in crash dumps — so it is supported for the platforms that give
        you nothing else, and it says so out loud.
        """
        agent_id = os.environ.get("AUTHFI_AGENT_ID")
        if not agent_id:
            raise AuthFIError("AUTHFI_AGENT_ID is not set", status=400)

        key_file = os.environ.get("AUTHFI_AGENT_KEY_FILE")
        key_pem = os.environ.get("AUTHFI_AGENT_PRIVATE_KEY")
        key_id = os.environ.get("AUTHFI_AGENT_KID")

        if key_file:
            if key_pem:
                warnings.warn(
                    "Both AUTHFI_AGENT_KEY_FILE and AUTHFI_AGENT_PRIVATE_KEY are set; "
                    "using the file and ignoring the environment variable.",
                    stacklevel=2,
                )
            key = AgentKey.from_file(key_file, key_id=key_id)
        elif key_pem:
            warnings.warn(
                "AUTHFI_AGENT_PRIVATE_KEY holds a private key in an environment "
                "variable, where it is visible in ps, inherited by every child "
                "process and captured in crash dumps. Mount it as a file and set "
                "AUTHFI_AGENT_KEY_FILE instead.",
                stacklevel=2,
            )
            key = AgentKey.from_pem(key_pem, key_id)
        else:
            raise AuthFIError(
                "No agent key: set AUTHFI_AGENT_KEY_FILE to the path of the "
                "agent's Ed25519 private key.",
                status=400,
            )

        kwargs = {
            "agent_id": agent_id,
            "key": key,
            "platform_url": os.environ.get("AUTHFI_PLATFORM_URL", DEFAULT_PLATFORM_URL),
            "audience": os.environ.get("AUTHFI_AGENT_AUDIENCE"),
        }
        kwargs.update(overrides)
        return cls(**kwargs)

    # --- Discovery ---

    def discovery(self):
        """Fetch (once) and check ``/.well-known/agent-configuration``."""
        if self._discovery is not None:
            return self._discovery

        req = Request(self.discovery_url)
        req.add_header("User-Agent", USER_AGENT)
        req.add_header("Accept", "application/json")
        try:
            with urlopen(req) as res:
                doc = json.loads(res.read())
        except HTTPError as e:
            raise AuthFIError(
                "Agent discovery failed at %s: HTTP %s" % (self.discovery_url, e.code),
                status=e.code,
            ) from e
        except Exception as e:
            raise AuthFIError(
                "Agent discovery failed at %s: %s" % (self.discovery_url, e), status=502
            ) from e

        methods = doc.get("token_endpoint_auth_methods_supported") or []
        if methods and "private_key_jwt" not in methods:
            # Said here rather than met as an opaque 401 from the token endpoint.
            # This SDK has one credential on purpose; if the deployment does not
            # accept it, that is the fact to report.
            raise AuthFIError(
                "This deployment does not accept private_key_jwt (it advertises %s). "
                "An AuthFI agent authenticates only by signing with its own key."
                % (", ".join(methods)),
                status=400,
            )
        algs = doc.get("token_endpoint_auth_signing_alg_values_supported") or doc.get("algorithms")
        if algs and "EdDSA" not in algs:
            raise AuthFIError(
                "This deployment does not accept EdDSA assertions (it advertises %s)."
                % (", ".join(algs)),
                status=400,
            )
        self._discovery = doc
        return doc

    @property
    def token_endpoint(self):
        if not self._token_endpoint:
            self._token_endpoint = (self.discovery().get("endpoints") or {}).get("token")
            if not self._token_endpoint:
                raise AuthFIError(
                    "Discovery at %s published no token endpoint" % self.discovery_url,
                    status=502,
                )
        return self._token_endpoint

    @property
    def backchannel_endpoint(self):
        if not self._backchannel_endpoint:
            doc = self.discovery()
            self._backchannel_endpoint = doc.get("backchannel_authentication_endpoint") or (
                doc.get("endpoints") or {}
            ).get("backchannel_authorize")
            if not self._backchannel_endpoint:
                raise AuthFIError(
                    "This deployment does not offer CIBA: discovery at %s advertises "
                    "no backchannel authentication endpoint, so an agent cannot ask a "
                    "person to approve acting on their behalf here."
                    % self.discovery_url,
                    status=400,
                )
        return self._backchannel_endpoint

    # --- The agent's own token (client_credentials) ---

    def token(self, audience=None, force=False):
        """The agent's access token, cached and refreshed before it expires.

        A refusal is a refusal: this raises :class:`AgentAuthError` carrying the
        registered OAuth code and does NOT retry. Retrying a rejected credential
        only hammers the token endpoint, and a fresh assertion for the same bad
        credential is rejected for the same reason.
        """
        requested = audience if audience is not None else self.audience
        if (
            not force
            and self._access_token
            and requested == self._token_audience
            and time.time() < self._expires_at - self.refresh_skew
        ):
            return self._access_token

        params = {
            "grant_type": "client_credentials",
            "client_id": self.agent_id,
            "client_assertion_type": CLIENT_ASSERTION_TYPE,
            "client_assertion": self.key.client_assertion(self.agent_id, self.token_endpoint),
        }
        if requested:
            # RFC 8707. The server reads `resource` in preference to `audience`
            # on a form-encoded request, so send the standard spelling.
            params["resource"] = requested

        body = _post_form(self.token_endpoint, params)
        self._access_token = body.get("access_token")
        if not self._access_token:
            raise AuthFIError(
                "The token endpoint returned no access_token", status=502
            )
        self._expires_at = time.time() + int(body.get("expires_in") or 3600)
        self._token_audience = requested
        # capabilities is an array of NAMES. Any constraint on a capability lives
        # on the server-side grant record and is checked when the capability
        # executes — a client that modelled constraints locally would be
        # enforcing a second, stale copy of someone else's decision.
        self._capabilities = frozenset(body.get("capabilities") or ())
        return self._access_token

    def capabilities(self):
        """The capability NAMES granted to this agent (fetched with the token)."""
        if not self._access_token:
            self.token()
        return self._capabilities

    def can(self, capability):
        """True iff this agent holds ``capability``."""
        return capability in self.capabilities()

    def auth_header(self, audience=None):
        """``{"Authorization": "Bearer ..."}`` — ready to pass to requests/httpx."""
        return {"Authorization": "Bearer " + self.token(audience=audience)}

    # --- Acting as a person (CIBA) ---

    def request_user_approval(
        self, login_hint, binding_message=None, scope=None, requested_expiry=None
    ):
        """Ask a PERSON to approve this agent acting on their behalf.

        Returns the CIBA §7.3 body: ``auth_req_id`` (a bearer handle — treat it
        like a credential), ``expires_in``, ``interval``, and ``approval_url``
        only when tenant policy has the client deliver the link itself.

        ``binding_message`` is what the person reads on their own device. Make it
        name the actual action: it is the only description of the request they
        will see, and the scope they approve is bound to it server-side.
        """
        if not login_hint or not str(login_hint).strip():
            raise AuthFIError(
                "login_hint is required: the platform must know WHICH person to ask",
                status=400,
            )
        params = {
            "client_id": self.agent_id,
            "client_assertion_type": CLIENT_ASSERTION_TYPE,
            # CIBA §7.1 — the same credential as the token endpoint, so this
            # surface cannot be the soft way in to a token that names a person.
            "client_assertion": self.key.client_assertion(self.agent_id, self.token_endpoint),
            "login_hint": login_hint,
        }
        if binding_message:
            params["binding_message"] = binding_message
        if scope:
            params["scope"] = scope
        if requested_expiry:
            params["requested_expiry"] = str(int(requested_expiry))
        return _post_form(self.backchannel_endpoint, params)

    def await_user_token(self, approval, timeout=300, audience=None, sleep=time.sleep):
        """Poll until the person decides, then return the delegated token.

        ``approval`` is what :meth:`request_user_approval` returned (or a bare
        ``auth_req_id``). The returned body's ``subject`` is the PERSON; inside
        the JWT ``sub`` is the person and ``act.sub`` is this agent (RFC 8693
        §4.1), so a resource server sees who is being acted for and by whom.

        Only ``authorization_pending`` and ``slow_down`` keep this waiting.
        ``access_denied``, ``expired_token`` and ``invalid_grant`` are decisions,
        and are raised immediately rather than retried — the person said no, or
        the window closed, and polling past that is just load.
        """
        if isinstance(approval, dict):
            auth_req_id = approval.get("auth_req_id")
            interval = int(approval.get("interval") or 5)
        else:
            auth_req_id, interval = approval, 5
        if not auth_req_id:
            raise AuthFIError("No auth_req_id to poll", status=400)

        requested = audience if audience is not None else self.audience
        deadline = time.time() + timeout
        while True:
            params = {
                "grant_type": CIBA_GRANT_TYPE,
                "client_id": self.agent_id,
                "client_assertion_type": CLIENT_ASSERTION_TYPE,
                # A fresh assertion every poll: the server replay-caches jti, so
                # reusing one would turn a slow approval into a refused poll.
                "client_assertion": self.key.client_assertion(
                    self.agent_id, self.token_endpoint
                ),
                "auth_req_id": auth_req_id,
            }
            if requested:
                params["resource"] = requested
            try:
                return _post_form(self.token_endpoint, params)
            except AgentAuthError as e:
                if e.error == "slow_down":
                    # The server is telling us our interval is too fast, and it
                    # has already widened its own. Honour it rather than keep
                    # the pace that earned the warning.
                    interval += 5
                elif e.error != "authorization_pending":
                    raise
            if time.time() + interval >= deadline:
                raise AgentAuthError(
                    "authorization_pending",
                    "the person had not decided within %ss; the request may still be "
                    "open — poll again with the same auth_req_id" % timeout,
                    status=408,
                )
            sleep(interval)

    def token_as_user(self, login_hint, binding_message=None, scope=None, timeout=300):
        """Request approval and wait for it — :meth:`request_user_approval` then
        :meth:`await_user_token`, for the common case where the agent is willing
        to block."""
        approval = self.request_user_approval(
            login_hint, binding_message=binding_message, scope=scope
        )
        return self.await_user_token(approval, timeout=timeout)


def _post_form(url, params):
    """POST an OAuth form and decode the response, RFC 6749 §4.4.2 / §5.1 / §5.2.

    Form-encoded, because that is what the RFC mandates and what the endpoint
    reads ``resource`` from — its JSON path reads only ``audience``. The body is
    at the TOP LEVEL, not inside this platform's ``{"data": ...}`` envelope, and
    an error is ``{"error": ..., "error_description": ...}``.
    """
    req = Request(url, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    data = urlencode(params).encode("utf-8")
    try:
        with urlopen(req, data) as res:
            return json.loads(res.read())
    except HTTPError as e:
        raw = e.read()
        try:
            body = json.loads(raw)
        except Exception:
            body = {}
        code = body.get("error")
        if not code:
            # No registered code to branch on — do not invent one.
            raise AuthFIError(
                "Agent token request failed: HTTP %s %s"
                % (e.code, raw.decode("utf-8", "replace")[:200]),
                status=e.code,
            ) from e
        raise AgentAuthError(code, body.get("error_description"), status=e.code) from e
    except AuthFIError:
        raise
    except Exception as e:
        raise AuthFIError("Agent token request failed: %s" % e, status=502) from e
