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

Agent authentication (the AuthFI Agent Profile):

    from authfi import AgentAuth

    agent = AgentAuth.from_env()   # AUTHFI_PLATFORM_URL = your project's address

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
from types import SimpleNamespace

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
        measured 2026-09-24, `api.authfi.io/<tenant>/.well-known/jwks.json` and
        `https://<tenant>.authfi.io/.well-known/jwks.json` serve the identical kid — so the
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

    @staticmethod
    def _segment(token, index):
        """Decode one base64url JWT segment as JSON. Never trusted — only read."""
        try:
            return json.loads(_b64decode(token.split(".")[index]))
        except Exception as e:
            raise AuthFIError(f"Invalid token format: {e}") from e

    # The algorithm a key can possibly have signed with, by key type. Taken from the KEY —
    # fetched over TLS from the issuer — and never from the token header, which the sender
    # controls. That is what closes alg:none (no key is ever selected) and the HS256 confusion
    # (a public key is never handed to PyJWT as an HMAC secret).
    _ALG_BY_KTY = {"RSA": "RS256", "EC": "ES256", "OKP": "EdDSA"}

    def _signing_key_for(self, token):
        """Find the published key that this token names, as a PyJWK."""
        import jwt

        kid = self._segment(token, 0).get("kid")
        keys = (self._fetch_jwks() or {}).get("keys") or []
        if not keys:
            raise AuthFIError(f"{self.jwks_url} published no keys")

        # A token without a kid is only unambiguous when the issuer publishes exactly one key.
        # Guessing among several would mean trying keys until one verifies, which turns a
        # rotation into a window where a retired key still validates tokens.
        if kid is None:
            if len(keys) != 1:
                raise AuthFIError(
                    f"token names no kid and {self.jwks_url} publishes {len(keys)} keys"
                )
            jwk = keys[0]
        else:
            jwk = next((k for k in keys if k.get("kid") == kid), None)
            if jwk is None:
                raise AuthFIError(f"no published key with kid {kid!r}")

        alg = jwk.get("alg") or self._ALG_BY_KTY.get(jwk.get("kty"))
        if not alg:
            raise AuthFIError(f"key {kid!r} declares no usable algorithm")
        return jwt.PyJWK(jwk, algorithm=alg), alg

    def verify_token(self, token):
        """Verify the signature against this project's published keys, and return the claims.

        The JWKS comes from `_fetch_jwks` — this SDK's own fetcher, which sends a named
        User-Agent and caches for `jwks_ttl`.

        It used to come from PyJWT's `PyJWKClient`, which fetches the document itself through
        urllib and sends the default `Python-urllib/3.x`. AuthFI's origins sit behind Cloudflare,
        which answers 403 to that agent — see USER_AGENT. So verification failed at the edge for
        every caller on a production origin, and reported itself as "token verification failed":
        a signature problem, for what was a blocked client. The cache was the second cost — every
        verification re-fetched the JWKS, because PyJWKClient held a fresh client per call.

        There is NO unverified fallback. Accepting an unverified token is an auth bypass.
        """
        if len(token.split(".")) != 3:
            raise AuthFIError("Invalid token format")

        try:
            import jwt
        except ImportError as e:
            raise AuthFIError(
                "PyJWT is required for token verification. "
                "Install with: pip install 'PyJWT[crypto]'"
            ) from e

        signing_key, alg = self._signing_key_for(token)
        try:
            return jwt.decode(
                token,
                signing_key.key,
                algorithms=[alg],  # the one algorithm this key can have signed with
                issuer=self.issuer,
                options={"verify_aud": False, "verify_iss": True},
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
# Agent authentication — the AuthFI Agent Profile
# =============================================================================
#
# An AuthFI agent holds its OWN keypair: Ed25519 (EdDSA), P-256 (ES256) or RSA
# 2048-4096 bits (PS256). It authenticates by signing a short assertion
# (RFC 7523 §2.2 private_key_jwt); the private half never leaves the process and
# is never sent to AuthFI. That is the whole point, and the difference between an
# agent credential and a bearer artifact the IETF agent-auth draft's security
# considerations call an antipattern.
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

#: The algorithm each accepted key type signs with. The KEY decides, as it does on
#: the server: a P-256 key signs ES256 and an RSA key signs PS256, and a label that
#: says otherwise is refused. RS256 is never used for an agent assertion.
ALG_EDDSA, ALG_ES256, ALG_PS256 = "EdDSA", "ES256", "PS256"

# The RSA modulus sizes the platform accepts for an agent key (NIST SP 800-57:
# 2048 bits through 2030), and the one public exponent it accepts.
_RSA_MIN_BITS, _RSA_MAX_BITS, _RSA_EXPONENT = 2048, 4096, 65537

# The JWK members that only a PRIVATE key has (RFC 7518 §6.2.2 and §6.3.2).
_PRIVATE_JWK_MEMBERS = ("d", "p", "q", "dp", "dq", "qi")

_KEY_TYPES_TEXT = "Ed25519 (EdDSA), P-256 (ES256) or RSA 2048-4096 bits (PS256)"

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


def _jwk_alg_label(jwk, implied):
    """Refuse a JWK whose ``alg`` or ``use`` contradicts what the key is.

    The server compares the label with the algorithm the key type implies and
    refuses a mismatch (an RSA key labelled RS256), so a mislabelled key is
    refused here, where the message can say what is wrong with it.
    """
    label = jwk.get("alg")
    if label and label != implied:
        raise AuthFIError(
            "This %s key is labelled alg=%s, but a %s agent key signs %s. AuthFI agents "
            "sign with %s; relabel the key or generate one of those."
            % (jwk.get("kty"), label, jwk.get("kty"), implied, _KEY_TYPES_TEXT),
            status=400,
        )
    use = jwk.get("use")
    if use and use != "sig":
        raise AuthFIError("This JWK is for use=%s; an agent key must be a signing key" % use,
                          status=400)


def _jwk_implied_alg(jwk):
    """The algorithm a JWK's key type and curve imply, or a refusal naming what it is."""
    kty, crv = jwk.get("kty"), jwk.get("crv")
    if kty == "OKP" and crv == "Ed25519":
        return ALG_EDDSA
    if kty == "EC" and crv == "P-256":
        return ALG_ES256
    if kty == "RSA":
        return ALG_PS256
    raise AuthFIError(
        "An agent key must be %s, but this JWK is %s/%s"
        % (_KEY_TYPES_TEXT, kty, crv or "-"),
        status=400,
    )


def assert_public_jwk(jwk):
    """Refuse a JWK that carries a private component, and return it otherwise.

    Call this before sending a JWK anywhere — registration, a config file, a
    ticket. A key that has been sent is no longer secret, so a private member
    (``d``, or RSA's ``p``, ``q``, ``dp``, ``dq``, ``qi``) turning up where a public
    key belongs is not a formatting slip to strip and carry on from: the key is
    burned and must be replaced. The server refuses it for the same reason, and
    quietly using the public half would leave its owner believing a secret they
    just leaked is still secret.

    It also refuses a key the platform would refuse: anything but Ed25519, P-256
    or RSA 2048-4096 bits with exponent 65537, and a key labelled with an
    algorithm its type does not sign.
    """
    if not isinstance(jwk, dict):
        raise AuthFIError("A JWK must be a dict (RFC 7517)", status=400)
    leaked = [m for m in _PRIVATE_JWK_MEMBERS if jwk.get(m)]
    if leaked:
        raise AuthFIError(
            "This JWK contains a private key component (%s) where a PUBLIC key "
            "belongs. Refusing to use or send it. Treat that key as burned — "
            "generate a new one (AgentKey.generate) and register its public half."
            % ", ".join("'%s'" % m for m in leaked),
            status=400,
        )
    implied = _jwk_implied_alg(jwk)
    _jwk_alg_label(jwk, implied)
    required = {ALG_EDDSA: ("x",), ALG_ES256: ("x", "y"), ALG_PS256: ("n", "e")}[implied]
    for member in required:
        if not jwk.get(member):
            raise AuthFIError(
                "This JWK has no '%s' — there is no public key in it" % member, status=400
            )
    if implied == ALG_PS256:
        _check_rsa(_b64int(jwk["n"]).bit_length(), _b64int(jwk["e"]))
    return jwk


def _check_rsa(bits, exponent):
    """Refuse an RSA key the platform would refuse: its size, or its exponent."""
    if not _RSA_MIN_BITS <= bits <= _RSA_MAX_BITS:
        raise AuthFIError(
            "This RSA key is %d bits; an agent key must be %d to %d bits"
            % (bits, _RSA_MIN_BITS, _RSA_MAX_BITS),
            status=400,
        )
    if exponent != _RSA_EXPONENT:
        raise AuthFIError(
            "This RSA key's public exponent is %d; an agent key must use 65537" % exponent,
            status=400,
        )


def _crypto():
    """Import the signing primitives, with an actionable error if absent.

    ``cryptography`` is already in this SDK's dependency closure (PyJWT[crypto]
    pulls it in for RS256 verification), so this adds nothing to install. It is
    imported lazily anyway, matching how PyJWT is treated above: importing
    ``authfi`` must not fail on a machine that only uses part of the SDK.
    """
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa
        from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature
    except ImportError as e:  # pragma: no cover - exercised only on a broken install
        raise AuthFIError(
            "Agent authentication needs the cryptography package. "
            "Install with: pip install cryptography",
            status=500,
        ) from e
    return SimpleNamespace(
        hashes=hashes, serialization=serialization, ec=ec, ed25519=ed25519,
        padding=padding, rsa=rsa, decode_dss_signature=decode_dss_signature,
    )


def _alg_for_private_key(key):
    """The algorithm ``key`` signs with, or a refusal naming what the key is."""
    c = _crypto()
    if isinstance(key, c.ed25519.Ed25519PrivateKey):
        return ALG_EDDSA
    if isinstance(key, c.ec.EllipticCurvePrivateKey):
        if key.curve.name != "secp256r1":
            raise AuthFIError(
                "The agent key is on curve %s; an elliptic-curve agent key must be "
                "P-256 (ES256). AuthFI agents sign with %s." % (key.curve.name, _KEY_TYPES_TEXT),
                status=400,
            )
        return ALG_ES256
    if isinstance(key, c.rsa.RSAPrivateKey):
        _check_rsa(key.key_size, key.public_key().public_numbers().e)
        return ALG_PS256
    raise AuthFIError(
        "The agent key is a %s key, but AuthFI agents sign with %s."
        % (type(key).__name__, _KEY_TYPES_TEXT),
        status=400,
    )


# The PEM blocks that hold an unencrypted private key: PKCS#8, and the SEC1 and
# PKCS#1 forms `openssl ecparam -genkey` and older tools write.
_PRIVATE_PEM_MARKERS = (b"BEGIN PRIVATE KEY", b"BEGIN EC PRIVATE KEY", b"BEGIN RSA PRIVATE KEY")


class AgentKey:
    """An agent's signing key, held in this process and never sent.

    Ed25519 signs EdDSA, P-256 signs ES256 and RSA (2048-4096 bits) signs PS256;
    :attr:`alg` is the algorithm this key signs with, decided by the key type.

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
        self.alg = _alg_for_private_key(private_key)
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

        Accepts a PEM private key or a private JWK, sniffed from the content. A
        JWK's own ``kid`` is used when ``key_id`` is not given.
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
        """Load an unencrypted private key from PEM: PKCS#8, or SEC1 / PKCS#1."""
        c = _crypto()
        raw = pem.encode("utf-8") if isinstance(pem, str) else pem
        head = raw.strip().splitlines()[0].decode("utf-8", "replace") if raw.strip() else ""
        if not any(marker in raw for marker in _PRIVATE_PEM_MARKERS):
            # Name the thing that was actually handed over. Letting this fail
            # inside the crypto library produces an error that reads like a
            # platform fault rather than "you gave me the wrong file".
            hint = ("it is a %s" % head.replace("-", "").strip()) if head.startswith("-----BEGIN") \
                else "it is not PEM at all"
            raise AuthFIError(
                "The agent key must be a private key in PEM form "
                "(-----BEGIN PRIVATE KEY-----), but %s. "
                "Generate one with: openssl genpkey -algorithm ed25519" % hint,
                status=400,
            )
        try:
            key = c.serialization.load_pem_private_key(raw, password=None)
        except Exception as e:
            raise AuthFIError(
                "The agent key is not a usable private key: %s. "
                "An encrypted key is not supported — decrypt it before mounting it." % e,
                status=400,
            ) from e
        return cls(key, key_id)

    @classmethod
    def from_jwk(cls, jwk, key_id=None):
        """Load a private key from a JWK — the one shape where ``d`` belongs.

        OKP/Ed25519, EC/P-256 or RSA. An ``alg`` member, when present, must be
        the algorithm the key type signs: an RSA key labelled RS256 is refused.
        """
        c = _crypto()
        if isinstance(jwk, (str, bytes)):
            jwk = json.loads(jwk)
        implied = _jwk_implied_alg(jwk)
        _jwk_alg_label(jwk, implied)
        if not jwk.get("d"):
            raise AuthFIError(
                "This JWK has no 'd' — it is the PUBLIC half. Signing needs the "
                "private key the agent kept.",
                status=400,
            )
        try:
            if implied == ALG_EDDSA:
                key = c.ed25519.Ed25519PrivateKey.from_private_bytes(_b64decode(jwk["d"]))
            elif implied == ALG_ES256:
                key = c.ec.EllipticCurvePrivateNumbers(
                    _b64int(jwk["d"]),
                    c.ec.EllipticCurvePublicNumbers(
                        _b64int(jwk["x"]), _b64int(jwk["y"]), c.ec.SECP256R1()
                    ),
                ).private_key()
            else:
                key = _rsa_private_from_jwk(c, jwk)
        except Exception as e:
            raise AuthFIError(
                "This %s JWK is not a usable private key: %s" % (jwk.get("kty"), e),
                status=400,
            ) from e
        return cls(key, key_id or jwk.get("kid"))

    @classmethod
    def generate(cls, key_id, alg=ALG_EDDSA):
        """Generate a fresh agent keypair, to register and then hold.

        ``alg`` picks the key type: ``"EdDSA"`` (Ed25519, the default),
        ``"ES256"`` (P-256) or ``"PS256"`` (RSA, 3072 bits). Pair with
        :meth:`public_jwk` (what you register) and :meth:`private_pem` (what you
        write to a 0600 file and never send anywhere).
        """
        c = _crypto()
        if alg == ALG_EDDSA:
            key = c.ed25519.Ed25519PrivateKey.generate()
        elif alg == ALG_ES256:
            key = c.ec.generate_private_key(c.ec.SECP256R1())
        elif alg == ALG_PS256:
            key = c.rsa.generate_private_key(public_exponent=_RSA_EXPONENT, key_size=3072)
        else:
            raise AuthFIError(
                "alg must be EdDSA, ES256 or PS256, not %r" % (alg,), status=400
            )
        return cls(key, key_id)

    # --- Using ---

    def public_jwk(self):
        """The public JWK to register — and the ONLY half that ever leaves here."""
        c = _crypto()
        pub = self._key.public_key()
        if self.alg == ALG_EDDSA:
            raw = pub.public_bytes(c.serialization.Encoding.Raw, c.serialization.PublicFormat.Raw)
            jwk = {"kty": "OKP", "crv": "Ed25519", "x": _b64encode(raw)}
        elif self.alg == ALG_ES256:
            nums = pub.public_numbers()
            jwk = {"kty": "EC", "crv": "P-256", "x": _int_b64(nums.x, 32), "y": _int_b64(nums.y, 32)}
        else:
            nums = pub.public_numbers()
            jwk = {"kty": "RSA", "n": _int_b64(nums.n), "e": _int_b64(nums.e)}
        jwk.update({"alg": self.alg, "kid": self.key_id})
        # Validated on the way out as well: this is the value a caller is about
        # to POST to /register, and the SDK must never be the thing that hands
        # them a private component to upload.
        return assert_public_jwk(jwk)

    def private_pem(self):
        """The PKCS#8 PEM, for writing to a protected file at bootstrap."""
        c = _crypto()
        return self._key.private_bytes(
            c.serialization.Encoding.PEM,
            c.serialization.PrivateFormat.PKCS8,
            c.serialization.NoEncryption(),
        ).decode("ascii")

    def sign(self, data):
        """The JWS signature over ``data`` (RFC 7518 §3), in this key's algorithm.

        ES256 is the 64-byte R||S form JWS requires, not the DER the crypto
        library returns. PS256 uses a salt as long as the SHA-256 digest
        (RFC 7518 §3.5).
        """
        c = _crypto()
        if self.alg == ALG_EDDSA:
            return self._key.sign(data)
        if self.alg == ALG_ES256:
            r, s = c.decode_dss_signature(self._key.sign(data, c.ec.ECDSA(c.hashes.SHA256())))
            return r.to_bytes(32, "big") + s.to_bytes(32, "big")
        return self._key.sign(
            data,
            c.padding.PSS(mgf=c.padding.MGF1(c.hashes.SHA256()), salt_length=32),
            c.hashes.SHA256(),
        )

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
        header = {"alg": self.alg, "typ": "JWT", "kid": self.key_id}
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


def _b64int(s):
    """A base64url JWK integer member (RFC 7518 §2 Base64urlUInt) as an int."""
    return int.from_bytes(_b64decode(s), "big")


def _int_b64(n, length=None):
    """An int as a base64url JWK member; ``length`` pads a fixed-size coordinate."""
    size = length or max(1, (n.bit_length() + 7) // 8)
    return _b64encode(n.to_bytes(size, "big"))


def _rsa_private_from_jwk(c, jwk):
    """An RSA private key from its JWK, recovering the primes when only n, e and d are given."""
    n, e, d = _b64int(jwk["n"]), _b64int(jwk["e"]), _b64int(jwk["d"])
    if all(jwk.get(m) for m in ("p", "q", "dp", "dq", "qi")):
        p, q = _b64int(jwk["p"]), _b64int(jwk["q"])
        dmp1, dmq1, iqmp = _b64int(jwk["dp"]), _b64int(jwk["dq"]), _b64int(jwk["qi"])
    else:
        p, q = c.rsa.rsa_recover_prime_factors(n, e, d)
        dmp1, dmq1, iqmp = c.rsa.rsa_crt_dmp1(d, p), c.rsa.rsa_crt_dmq1(d, q), c.rsa.rsa_crt_iqmp(p, q)
    return c.rsa.RSAPrivateNumbers(
        p, q, d, dmp1, dmq1, iqmp, c.rsa.RSAPublicNumbers(e, n)
    ).private_key()


def _project_origin(platform_url):
    """The project's origin from ``platform_url``, or a refusal saying what to pass.

    Required, with no default: see :class:`AgentAuth`.
    """
    url = str(platform_url or "").strip().rstrip("/")
    if not url:
        raise AuthFIError(
            "platform_url is required: pass your project's own address, for example "
            "platform_url=\"https://acme.authfi.io\" (or set AUTHFI_PLATFORM_URL with "
            "AgentAuth.from_env()). The agent is registered in that project, and no "
            "other project can authenticate it.",
            status=400,
        )
    if not url.startswith(("https://", "http://")):
        raise AuthFIError(
            "platform_url must be an address with its scheme, such as "
            "https://acme.authfi.io, not %r" % (platform_url,),
            status=400,
        )
    return url


class AgentAuth:
    """Agent authentication against AuthFI, bootstrapped from discovery.

    Usage::

        from authfi import AgentAuth, AgentKey

        agent = AgentAuth(
            agent_id="agt_...",
            key=AgentKey.from_file("/run/secrets/agent.key", key_id="k1"),
            platform_url="https://acme.authfi.io",   # your project's own address
            audience="https://api.acme.com",   # optional, RFC 8707 resource
        )

        if agent.can("send_email_external"):
            requests.post(url, headers=agent.auth_header(), json=body)

    ``platform_url`` is required and has no default. An agent is registered in
    ONE project, so the only address that can authenticate it is that project's
    own (``https://<project>.authfi.io`` or its custom domain). A default would
    send the agent's signed assertion to a project where it does not exist.

    Endpoints come from ``/.well-known/agent-configuration`` rather than being
    hard-coded: the token endpoint moves with the project's origin (a tenant
    subdomain, a custom domain), and the assertion's ``aud`` is built from the
    same document the server builds its own check from.
    """

    def __init__(
        self,
        agent_id,
        key,
        platform_url=None,
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
        self.platform_url = _project_origin(platform_url)
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
        ``AUTHFI_PLATFORM_URL``         your project's address — REQUIRED
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
                "agent's private key.",
                status=400,
            )

        platform_url = overrides.pop("platform_url", None) or os.environ.get("AUTHFI_PLATFORM_URL")
        if not platform_url:
            raise AuthFIError(
                "AUTHFI_PLATFORM_URL is not set: set it to your project's own address, "
                "for example https://acme.authfi.io. The agent is registered in that "
                "project, and no other project can authenticate it.",
                status=400,
            )

        kwargs = {
            "agent_id": agent_id,
            "key": key,
            "platform_url": platform_url,
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
        if algs and self.key.alg not in algs:
            # Said here, by name, rather than met as an opaque invalid_client.
            raise AuthFIError(
                "This agent's key signs %s, which this deployment does not accept "
                "(it advertises %s). Use a key of an advertised type."
                % (self.key.alg, ", ".join(algs)),
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
