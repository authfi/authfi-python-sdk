"""Test doubles: a fake clock, a fake AuthFI project on httpx.MockTransport, and signing keys."""

from __future__ import annotations

import base64
import copy
import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, rsa
from jwt.algorithms import ECAlgorithm, OKPAlgorithm, RSAAlgorithm

FIXTURES = Path(__file__).parent / "fixtures"
ISSUER = "https://acme.authfi.io"
TENANT = "4e46e0c4-ec26-48e1-b5bb-04a1ec3ff358"


def fixture_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


class FakeClock:
    """A clock that moves only when told to, or when slept on."""

    def __init__(self, wall: float | None = None) -> None:
        self.mono = 10_000.0
        self.wall = time.time() if wall is None else wall
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.mono

    def time(self) -> float:
        return self.wall

    def advance(self, seconds: float) -> None:
        self.mono += seconds
        self.wall += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)

    async def asleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


@dataclass
class Keys:
    """The project's signing keys: RSA (people, machines), Ed25519 (agents), P-256."""

    rsa_key: rsa.RSAPrivateKey
    ed_key: ed25519.Ed25519PrivateKey
    ec_key: ec.EllipticCurvePrivateKey
    rsa_kid: str = "rsa-1"
    ed_kid: str = "ed-1"
    ec_kid: str = "ec-1"

    @classmethod
    def generate(cls) -> Keys:
        return cls(
            rsa_key=rsa.generate_private_key(public_exponent=65537, key_size=2048),
            ed_key=ed25519.Ed25519PrivateKey.generate(),
            ec_key=ec.generate_private_key(ec.SECP256R1()),
        )

    def jwk(self, alg: str) -> dict[str, Any]:
        if alg == "RS256":
            out = json.loads(RSAAlgorithm.to_jwk(self.rsa_key.public_key()))
            out.update(kid=self.rsa_kid, alg="RS256", use="sig")
        elif alg == "EdDSA":
            out = json.loads(OKPAlgorithm.to_jwk(self.ed_key.public_key()))
            out.update(kid=self.ed_kid, alg="EdDSA", use="sig")
        else:
            out = json.loads(ECAlgorithm.to_jwk(self.ec_key.public_key()))
            out.update(kid=self.ec_kid, alg="ES256", use="sig")
        return cast(dict[str, Any], out)

    def jwks(self) -> dict[str, Any]:
        return {"keys": [self.jwk("RS256"), self.jwk("EdDSA"), self.jwk("ES256")]}

    def private(self, alg: str) -> Any:
        return {"RS256": self.rsa_key, "EdDSA": self.ed_key, "ES256": self.ec_key}[alg]

    def kid(self, alg: str) -> str:
        return {"RS256": self.rsa_kid, "EdDSA": self.ed_kid, "ES256": self.ec_kid}[alg]

    def sign(
        self,
        claims: Mapping[str, Any],
        *,
        alg: str = "RS256",
        typ: str | None = "at+jwt",
        kid: str | None = None,
        headers: Mapping[str, Any] | None = None,
    ) -> str:
        header: dict[str, Any] = {"kid": kid if kid is not None else self.kid(alg)}
        if typ is not None:
            header["typ"] = typ
        header.update(headers or {})
        return jwt.encode(dict(claims), self.private(alg), algorithm=alg, headers=header)


_keys: Keys | None = None


def shared_keys() -> Keys:
    """One key set for the whole run: RSA generation is slow."""
    global _keys
    if _keys is None:
        _keys = Keys.generate()
    return _keys


Handler = Callable[[httpx.Request], httpx.Response]


def json_response(body: Any, status: int = 200, headers: Mapping[str, str] | None = None) -> httpx.Response:
    return httpx.Response(status, json=body, headers=dict(headers or {}))


@dataclass
class Platform:
    """A fake AuthFI project behind httpx.MockTransport. Records every request."""

    issuer: str = ISSUER
    keys: Keys = field(default_factory=shared_keys)
    requests: list[httpx.Request] = field(default_factory=list[httpx.Request])
    routes: dict[tuple[str, str], Handler] = field(default_factory=dict[tuple[str, str], Handler])
    discovery_cache_control: str = "public, max-age=300"
    jwks_cache_control: str = "public, max-age=300"

    def __post_init__(self) -> None:
        self.discovery = self._discovery()
        self.route(
            "GET",
            "/.well-known/openid-configuration",
            lambda _r: json_response(self.discovery, headers={"Cache-Control": self.discovery_cache_control}),
        )
        self.route(
            "GET",
            "/.well-known/jwks.json",
            lambda _r: json_response(self.keys.jwks(), headers={"Cache-Control": self.jwks_cache_control}),
        )

    def _discovery(self) -> dict[str, Any]:
        doc = copy.deepcopy(fixture_json("platform-beta/openid-configuration.json"))
        text = json.dumps(doc).replace("https://platform-beta.authfi.io", self.issuer)
        return cast(dict[str, Any], json.loads(text))

    def route(self, method: str, path: str, handler: Handler | httpx.Response) -> None:
        if isinstance(handler, httpx.Response):
            response = handler

            def fixed(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(response.status_code, headers=response.headers, content=response.content)

            self.routes[(method, path)] = fixed
        else:
            self.routes[(method, path)] = handler

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        handler = self.routes.get((request.method, request.url.path))
        if handler is None:
            return httpx.Response(404, json={"error": "not_found"})
        return handler(request)

    def calls(self, method: str, path: str) -> list[httpx.Request]:
        return [r for r in self.requests if r.method == method and r.url.path == path]

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))

    def async_client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    def access_token(self, *, alg: str = "RS256", **claims: Any) -> str:
        now = int(time.time())
        base: dict[str, Any] = {
            "iss": self.issuer,
            "sub": "usr_1",
            "aud": ["api_orders"],
            "exp": now + 900,
            "iat": now,
            "tenant_id": TENANT,
            "client_id": "app_web",
            "jti": "jti-1",
        }
        base.update(claims)
        return self.keys.sign({k: v for k, v in base.items() if v is not None}, alg=alg)


def form_of(request: httpx.Request) -> dict[str, str]:
    from urllib.parse import parse_qsl

    return dict(parse_qsl(request.content.decode("utf-8")))
