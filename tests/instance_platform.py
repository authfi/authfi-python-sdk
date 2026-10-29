"""A faithful fake of contract C3: instance sign-in on the agent surface.

It checks what the platform checks: a challenge that works once, evidence the binding row allows (a
service-account token for the project's origin, ten minutes at most, single use by ``jti``, carrying the
``kubernetes.io`` pod claims, from a pod that is live; or the bound device key), the approved configuration, a
proof signed by the instance key, and then at the token endpoint the attestation chain's signature, the proof of
possession, and a DPoP proof by the key the attestation names. It refuses a bound token presented as Bearer.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import ed25519

from tests.agent_platform import AgentPlatform
from tests.support import json_response

ATTEST = "/apis/regional/v1/ai/agents/attest"
CHALLENGE = ATTEST + "/challenge"
CONFIG = {"version": 7, "digest": "sha256:" + "ab" * 32}


def thumbprint(jwk: dict[str, Any]) -> str:
    members = ("crv", "kty", "x", "y") if jwk["kty"] == "EC" else ("crv", "kty", "x")
    canonical = json.dumps({m: jwk[m] for m in members}, sort_keys=True, separators=(",", ":"))
    return base64.urlsafe_b64encode(hashlib.sha256(canonical.encode()).digest()).rstrip(b"=").decode()


@dataclass
class InstancePlatform(AgentPlatform):
    """A project whose agent is bound to a workload."""

    binding_kind: str = "k8s"
    envelope: bool = False  # answer {"data": {...}} instead of the bare contract shape
    approved_config: dict[str, Any] = field(default_factory=lambda: dict(CONFIG))
    pod_live: bool = True
    device_jwk: dict[str, Any] | None = None
    cluster_key: ed25519.Ed25519PrivateKey = field(default_factory=ed25519.Ed25519PrivateKey.generate)
    nonces: dict[str, float] = field(default_factory=dict[str, float])
    evidence_jtis: set[str] = field(default_factory=set[str])
    pop_jtis: set[str] = field(default_factory=set[str])
    dpop_jtis: set[str] = field(default_factory=set[str])
    attests: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])
    token_requests: list[httpx.Request] = field(default_factory=list[httpx.Request])
    attestation_lifetime: int = 900
    require_dpop_nonce: str | None = None
    token_lifetime: int = 900

    def __post_init__(self) -> None:
        super().__post_init__()
        self.route("POST", CHALLENGE, self._challenge)
        self.route("POST", ATTEST, self._attest)
        self.route("GET", "/instance/echo", self._echo)

    # --- helpers ---------------------------------------------------------------------------------

    def _ok(self, body: dict[str, Any]) -> httpx.Response:
        return json_response({"data": body} if self.envelope else body)

    def k8s_token(self, *, aud: str | None = None, ttl: int = 600, jti: str | None = None, pod: bool = True) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": "https://oidc.eks.ap-south-1.amazonaws.com/id/ABC",
            "sub": "system:serviceaccount:agents:researcher",
            "aud": [aud or self.issuer],
            "iat": now,
            "exp": now + ttl,
            "jti": jti or str(uuid.uuid4()),
        }
        if pod:
            claims["kubernetes.io"] = {
                "namespace": "agents",
                "serviceaccount": {"name": "researcher", "uid": "sa-uid"},
                "pod": {"name": "researcher-0", "uid": "pod-uid"},
                "node": {"name": "node-1"},
            }
        return jwt.encode(claims, self.cluster_key, algorithm="EdDSA", headers={"kid": "cluster-1"})

    def refuse_with(self, code: str, status: int = 403) -> httpx.Response:
        return json_response({"error": code, "error_description": f"refused: {code}"}, status=status)

    # --- challenge and attest ---------------------------------------------------------------------

    def _challenge(self, _request: httpx.Request) -> httpx.Response:
        nonce = uuid.uuid4().hex
        self.nonces[nonce] = time.time()
        return self._ok({"nonce": nonce})

    def _take_nonce(self, nonce: object) -> bool:
        issued = self.nonces.pop(str(nonce), None)
        return issued is not None and time.time() - issued <= 60

    def _attest(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        self.attests.append(body)
        if body.get("agent_id") != self.agent_id:
            return self.refuse_with("binding_not_found", 404)
        run_jwk: dict[str, Any] = body.get("run_key") or {}
        try:
            proof = jwt.decode(
                body["proof"],
                jwt.PyJWK(run_jwk).key,
                algorithms=["ES256"],
                audience=self.issuer,
                options={"verify_exp": False, "verify_iat": False},
            )
            proof_header = jwt.get_unverified_header(body["proof"])
        except (jwt.PyJWTError, KeyError, ValueError):
            return self.refuse_with("proof_invalid", 401)
        if proof_header.get("typ") != "authfi-instance-proof+jwt" or proof.get("agent_id") != self.agent_id:
            return self.refuse_with("proof_invalid", 401)
        if not self._take_nonce(proof.get("nonce")):
            return self.refuse_with("challenge_invalid", 400)
        evidence: dict[str, Any] = body.get("evidence") or {}
        kind: str | None = evidence.get("kind")
        if kind in ("gcp", "aws"):
            return self.refuse_with("proof_kind_not_built", 501)
        if kind != self.binding_kind:
            return self.refuse_with("evidence_kind_mismatch")
        problem = self._k8s(evidence) if kind == "k8s" else self._device(evidence, proof)
        if problem is not None:
            return problem
        if (body.get("config") or {}) != self.approved_config:
            return self.refuse_with("config_not_approved", 409)
        return self._ok({"attestation": self._attestation(body, run_jwk)})

    def _k8s(self, evidence: dict[str, Any]) -> httpx.Response | None:
        try:
            claims = jwt.decode(
                evidence["token"],
                self.cluster_key.public_key(),
                algorithms=["EdDSA"],
                audience=self.issuer,
                options={"verify_iat": False},
            )
        except (jwt.PyJWTError, KeyError):
            return self.refuse_with("evidence_invalid", 401)
        if claims["exp"] - claims["iat"] > 600:
            return self.refuse_with("evidence_invalid", 401)
        pod = claims.get("kubernetes.io", {})
        if not (pod.get("pod", {}).get("name") and pod.get("pod", {}).get("uid") and pod.get("node")):
            return self.refuse_with("evidence_invalid", 401)
        if claims["jti"] in self.evidence_jtis:
            return self.refuse_with("evidence_replayed", 409)
        self.evidence_jtis.add(claims["jti"])
        if not self.pod_live:
            return self.refuse_with("pod_not_live")
        return None

    def _device(self, evidence: dict[str, Any], proof: dict[str, Any]) -> httpx.Response | None:
        if self.device_jwk is None:
            return self.refuse_with("binding_not_found", 404)
        try:
            claims = jwt.decode(
                evidence["token"],
                jwt.PyJWK(self.device_jwk).key,
                algorithms=[self.device_jwk["alg"]],
                audience=self.issuer,
                options={"verify_exp": False, "verify_iat": False},
            )
        except (jwt.PyJWTError, KeyError):
            return self.refuse_with("evidence_invalid", 401)
        if claims.get("nonce") != proof.get("nonce"):
            return self.refuse_with("evidence_invalid", 401)
        return None

    def _attestation(self, body: dict[str, Any], run_jwk: dict[str, Any]) -> str:
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": f"spiffe://acme.authfi.io/ai/agent/{self.agent_id}",
            "sub": self.agent_id,
            "iat": now,
            "exp": now + self.attestation_lifetime,
            "jti": str(uuid.uuid4()),
            "cnf": {"jwk": run_jwk},
            "authfi_instance": {
                "instance_id": "inst_" + uuid.uuid4().hex,
                "workload": {"kind": body["evidence"]["kind"], "namespace": "agents", "service_account": "researcher"},
                "config": body.get("config"),
                "custody": "software",
                "environment": "np",
            },
        }
        if body.get("image_digest"):
            claims["authfi_instance"]["image_digest"] = body["image_digest"]
        headers = {
            "typ": "oauth-client-attestation+jwt",
            "kid": self.keys.ed_kid,
            "x5c": ["agent-cert", "project-ca", "intermediate"],
        }
        return jwt.encode(claims, self.keys.ed_key, algorithm="EdDSA", headers=headers)

    # --- the token endpoint -------------------------------------------------------------------------

    def _token(self, request: httpx.Request) -> httpx.Response:
        if "oauth-client-attestation" not in request.headers:
            return super()._token(request)
        self.token_requests.append(request)
        return self._attested_token(request)

    def _attested_token(self, request: httpx.Request) -> httpx.Response:
        from tests.support import form_of

        form = form_of(request)
        if form.get("grant_type") != "client_credentials" or form.get("client_id") != self.agent_id:
            return self.refuse("invalid_client", 401)
        try:
            attestation = request.headers["oauth-client-attestation"]
            header = jwt.get_unverified_header(attestation)
            claims = jwt.decode(
                attestation, self.keys.ed_key.public_key(), algorithms=["EdDSA"], options={"verify_aud": False}
            )
        except jwt.PyJWTError:
            return self.refuse("invalid_client", 401)
        if header.get("typ") != "oauth-client-attestation+jwt" or claims["sub"] != self.agent_id:
            return self.refuse("invalid_client", 401)
        cnf_jwk = claims["cnf"]["jwk"]
        try:
            pop = jwt.decode(
                request.headers["oauth-client-attestation-pop"],
                jwt.PyJWK(cnf_jwk).key,
                algorithms=["ES256"],
                audience=self.issuer,
                options={"verify_exp": False, "verify_iat": False},
            )
            pop_header = jwt.get_unverified_header(request.headers["oauth-client-attestation-pop"])
        except (jwt.PyJWTError, KeyError):
            return self.refuse("invalid_client", 401)
        if pop_header.get("typ") != "oauth-client-attestation-pop+jwt" or pop["iss"] != self.agent_id:
            return self.refuse("invalid_client", 401)
        if pop["jti"] in self.pop_jtis or not self._take_nonce(pop.get("challenge")):
            return self.refuse("invalid_client", 401)
        self.pop_jtis.add(pop["jti"])
        refusal = self.check_dpop(request, thumbprint(cnf_jwk), access_token=None)
        if refusal is not None:
            return refusal
        self.minted += 1
        instance = claims["authfi_instance"]
        token = self.agent_token(
            form.get("resource"),
            cnf={"jkt": thumbprint(cnf_jwk)},
            run_id=instance["instance_id"],
            authfi_config=instance.get("config"),
            exp=int(time.time()) + self.token_lifetime,
        )
        return json_response(
            {
                "access_token": token,
                "token_type": "DPoP",
                "expires_in": self.token_lifetime,
                "scope": " ".join(self.capabilities),
            }
        )

    def check_dpop(self, request: httpx.Request, jkt: str, *, access_token: str | None) -> httpx.Response | None:
        raw = request.headers.get("dpop")
        if not raw:
            return self.refuse("invalid_dpop_proof", 400)
        try:
            header = jwt.get_unverified_header(raw)
            proof = jwt.decode(
                raw,
                jwt.PyJWK(header["jwk"]).key,
                algorithms=["ES256"],
                options={"verify_exp": False, "verify_iat": False, "verify_aud": False},
            )
        except (jwt.PyJWTError, KeyError):
            return self.refuse("invalid_dpop_proof", 400)
        url = str(request.url).split("?", 1)[0]
        ok = (
            header.get("typ") == "dpop+jwt"
            and thumbprint(header["jwk"]) == jkt
            and proof.get("htm") == request.method
            and proof.get("htu") == url
            and proof["jti"] not in self.dpop_jtis
        )
        if ok and access_token is not None:
            digest = hashlib.sha256(access_token.encode()).digest()
            ok = proof.get("ath") == base64.urlsafe_b64encode(digest).rstrip(b"=").decode()
        if not ok:
            return self.refuse("invalid_dpop_proof", 400)
        if self.require_dpop_nonce and proof.get("nonce") != self.require_dpop_nonce:
            return json_response(
                {"error": "use_dpop_nonce"}, status=400, headers={"DPoP-Nonce": self.require_dpop_nonce}
            )
        self.dpop_jtis.add(proof["jti"])
        return None

    def _echo(self, request: httpx.Request) -> httpx.Response:
        """A resource that takes a DPoP-bound token only (C4)."""
        scheme, _, raw = request.headers.get("authorization", "").partition(" ")
        if not raw:
            return self.refuse("unauthorized", 401)
        claims = jwt.decode(raw, options={"verify_signature": False})
        if "cnf" in claims and scheme.lower() != "dpop":
            return self.refuse("token_dpop_proof_required", 401)
        refusal = self.check_dpop(request, claims.get("cnf", {}).get("jkt", ""), access_token=raw)
        return refusal or json_response({"ok": True, "sub": claims["sub"]})
