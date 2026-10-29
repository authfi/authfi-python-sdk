"""Sign in as an instance of an agent: the credential ``authfi.connect()`` returns on a bound workload.

An agent has no long-lived secret here. Each execution (an *instance*) makes a key in memory, shows where
it runs, and trades that for a 15-minute attestation and a token bound to its key::

    af = authfi.connect()        # AUTHFI_ISSUER and AUTHFI_AGENT_ID; nothing else on a bound pod

It finds its evidence itself: the projected service-account token on Kubernetes
(``/var/run/secrets/kubernetes.io/serviceaccount/token``, or ``AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE``; project it
with ``audience`` = the project's origin and ``expirationSeconds: 600``), or the device key named by
``AUTHFI_DEVICE_KEY_FILE``. ``AUTHFI_CONFIG_VERSION`` and ``AUTHFI_CONFIG_DIGEST`` name the signed
configuration the instance runs, and ``AUTHFI_IMAGE_DIGEST`` its image.

Sign-in is challenge and attest, then the token request (``OAuth-Client-Attestation``, ``-PoP`` and
``DPoP`` headers). The token is bound to the instance key, so every call carries ``Authorization: DPoP``
and a fresh proof; ``headers()`` does that, and the gateway and the exporters call it. The attestation is
renewed before it expires.

Acting for a person (CIBA) and the approval calls still authenticate with an agent key and are refused here
with the fix; token exchange works.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from authfi import _dpop
from authfi._clock import Clock
from authfi._protocol.http import HttpRequest
from authfi._protocol.instance import POP_TYP as _POP_TYP
from authfi._protocol.instance import PROOF_TYP as _PROOF_TYP
from authfi._protocol.instance import (
    Attestation,
    Configuration,
    attest_request,
    attested_token_request,
    challenge_request,
    parse_attestation,
    parse_challenge,
    pop_claims,
)
from authfi._protocol.oauth import b64url, parse_token_response
from authfi._protocol.urls import origin_of
from authfi._protocol.verify import unverified_claims
from authfi.agents.agent import (
    AgentAuth,
    AsyncAgentAuth,
    _AgentResource,  # pyright: ignore[reportPrivateUsage]
    _AsyncAgentResource,  # pyright: ignore[reportPrivateUsage]
    _Minted,  # pyright: ignore[reportPrivateUsage]
)
from authfi.agents.keys import AgentKey, Signer
from authfi.client import AsyncAuthFI, AuthFI
from authfi.errors import ConfigurationError, NotSupportedError
from authfi.tokens import AsyncTokenSource, TokenSource, check_destination

__all__ = ["AsyncInstanceAuth", "Configuration", "Evidence", "InstanceAuth", "find_evidence"]

K8S_TOKEN_PATH = "/var/run/secrets/kubernetes.io/serviceaccount/token"
#: What a device key signs, ahead of the nonce, the agent and the instance key (the platform checks these exact bytes).
_DEVICE_PREFIX = "authfi-agent-attest\n"
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_NONCE_ERROR = "use_dpop_nonce"


@dataclass(frozen=True, slots=True)
class Evidence:
    """Where this instance runs: ``kind`` is ``k8s`` or ``device``; ``path`` is the file that proves it."""

    kind: str
    path: str
    kid: str | None = None


def find_evidence(env: Mapping[str, str] | None = None) -> Evidence | None:
    """The evidence this process can show, or ``None``: the device key if named, else the projected token."""
    source = os.environ if env is None else env
    device = source.get("AUTHFI_DEVICE_KEY_FILE", "").strip()
    if device:
        return Evidence("device", device, source.get("AUTHFI_DEVICE_KEY_KID", "").strip() or None)
    named = source.get("AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE", "").strip()
    path = named or K8S_TOKEN_PATH
    if named or Path(path).is_file():
        return Evidence("k8s", path)
    return None


def _config_from(env: Mapping[str, str]) -> Configuration | None:
    version, digest = env.get("AUTHFI_CONFIG_VERSION", "").strip(), env.get("AUTHFI_CONFIG_DIGEST", "").strip()
    if not version and not digest:
        return None
    if not (version.isdigit() and digest.startswith("sha256:")):
        raise ConfigurationError(
            "AUTHFI_CONFIG_VERSION is the approved version number and AUTHFI_CONFIG_DIGEST is sha256:<hex>; "
            f"got {version!r} and {digest!r}"
        )
    return Configuration(int(version), digest)


class _InstanceSigner:
    """The instance key as an agent signer: it names the key, and signs nothing an agent key would."""

    @property
    def alg(self) -> str:
        return _dpop.run_key().alg

    @property
    def kid(self) -> str:
        return _dpop.run_key().thumbprint

    def sign(self, data: bytes) -> bytes:
        raise NotSupportedError("instance_assertion", _CIBA_REFUSAL)

    def public_jwk(self) -> dict[str, str]:
        return _dpop.run_key().public_jwk()


_CIBA_REFUSAL = (
    "An instance signs in by attestation, which does not yet cover asking a person (CIBA) or opening an "
    "approval. Use token exchange for a person who signed in to your app, or sign in with an agent key for "
    "this call."
)


class _Core:
    """What the blocking and asyncio instance credentials share: evidence, bodies and proofs."""

    def __init__(
        self,
        evidence: Evidence,
        agent_id: str,
        issuer: str,
        config: Configuration | None,
        image_digest: str | None,
    ) -> None:
        if config is None:
            raise ConfigurationError(
                "an instance runs one approved configuration: set AUTHFI_CONFIG_VERSION and AUTHFI_CONFIG_DIGEST "
                "(sha256:<hex>), or pass config=Configuration(version, digest). The platform attests no instance "
                "without one."
            )
        if image_digest is not None and not _IMAGE_DIGEST.fullmatch(image_digest):
            raise ConfigurationError(
                f"AUTHFI_IMAGE_DIGEST is sha256: and 64 lowercase hex digits; got {image_digest!r}"
            )
        self.evidence = evidence
        self.agent_id = agent_id
        self.issuer = issuer
        self.config = config
        self.image_digest = image_digest
        self.attestation: Attestation | None = None
        self._last_evidence: str | None = None

    def key(self) -> _dpop.DPoPKey:
        return _dpop.run_key()

    def reusable(self, now: float, skew: float) -> Attestation | None:
        held = self.attestation
        if held is not None and held.key_thumbprint == self.key().thumbprint and held.expires_at - now > skew:
            return held
        return None

    def _read(self) -> str:
        try:
            return Path(self.evidence.path).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ConfigurationError(
                f"cannot read the evidence file {self.evidence.path} ({error.strerror}). On Kubernetes project a "
                f"service-account token there (audience {origin_of(self.issuer)}, expirationSeconds 600), or set "
                "AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE to where it is."
            ) from error

    def evidence_token(self, nonce: str) -> str:
        if self.evidence.kind == "device":
            try:
                device = AgentKey.from_file(self.evidence.path, kid=self.evidence.kid)
            except (OSError, ConfigurationError) as error:
                raise ConfigurationError(
                    f"cannot use the device key {self.evidence.path}: {error}. Set AUTHFI_DEVICE_KEY_FILE to the "
                    "key file the device was bound with and AUTHFI_DEVICE_KEY_KID to its key id."
                ) from error
            return _device_signature(device, self.agent_id, nonce, self.key().thumbprint)
        token = self._read()
        if not token:
            raise ConfigurationError(f"the evidence file {self.evidence.path} is empty")
        self._check_k8s(token)
        if token == self._last_evidence:
            raise ConfigurationError(
                f"the service-account token in {self.evidence.path} has not rotated since it was used, and it "
                "works once. Project it with expirationSeconds: 600 so the kubelet renews it every 8 minutes."
            )
        self._last_evidence = token
        return token

    def _check_k8s(self, token: str) -> None:
        try:
            aud = unverified_claims(token).get("aud")
        except Exception as error:
            raise ConfigurationError(f"{self.evidence.path} does not hold a service-account JWT") from error
        accepted = {origin_of(self.issuer), self.issuer}
        audiences: set[str] = set()
        if isinstance(aud, str):
            audiences.add(aud)
        elif isinstance(aud, list):
            audiences.update(a for a in cast("list[object]", aud) if isinstance(a, str))
        if not audiences & accepted:
            raise ConfigurationError(
                f"the service-account token in {self.evidence.path} is for {sorted(audiences) or 'no audience'}, "
                f"not {origin_of(self.issuer)}. Project one with audience {origin_of(self.issuer)}."
            )

    def attest_body(self, nonce: str, now: float) -> dict[str, Any]:
        key = self.key()
        proof = key.sign_jwt(
            {"nonce": nonce, "agent_id": self.agent_id, "aud": self.issuer, "iat": int(now), "jti": str(uuid.uuid4())},
            typ=_PROOF_TYP,
        )
        body: dict[str, Any] = {
            "agent_id": self.agent_id,
            "evidence": {"kind": self.evidence.kind, "token": self.evidence_token(nonce)},
            "instance_key": key.public_jwk(),
            "config": {"version": self.config.version, "digest": self.config.digest},
            "proof": proof,
        }
        if self.image_digest:
            body["image_digest"] = self.image_digest
        return body

    def pop(self, now: float) -> str:
        return self.key().sign_jwt(pop_claims(self.agent_id, self.issuer, now), typ=_POP_TYP)

    def token_request(self, endpoint: str, attestation: Attestation, resource: str | None, clock: Clock) -> HttpRequest:
        """The token request: attestation, proof of the key, and a DPoP proof, fresh per attempt."""
        key = self.key()

        def pop() -> str:
            return self.pop(clock.time())

        def dpop() -> str:
            return key.proof("POST", endpoint, now=clock.time(), nonce=_dpop.nonces.get(endpoint))

        return attested_token_request(endpoint, self.agent_id, attestation, pop=pop, dpop=dpop, resource=resource)


def _device_signature(device: Signer, agent_id: str, nonce: str, instance_key_thumbprint: str) -> str:
    """The device key's evidence: its signature over the nonce, the agent and the instance key, base64url.

    Binding the instance key means a captured signature cannot be spent for an attestation naming another key.
    """
    message = f"{_DEVICE_PREFIX}{nonce}\n{agent_id}\n{instance_key_thumbprint}".encode()
    return b64url(device.sign(message))


def _dpop_headers(core: _Core, clock_time: float, method: str, url: str, token: str) -> dict[str, str]:
    proof = core.key().proof(method, url, now=clock_time, access_token=token, nonce=_dpop.nonces.get(url))
    return {"Authorization": "DPoP " + token, "DPoP": proof}


class _BoundResource(_AgentResource):
    """The instance's token for one resource (a customer API, a remote agent): DPoP, sent only to ``send_to``."""

    __slots__ = ()

    def headers(self, method: str, url: str) -> dict[str, str]:
        """``Authorization: DPoP`` and a proof of the instance key for ``method url``."""
        check_destination(url, self.send_to)
        agent = cast("InstanceAuth", self._agent)
        return _dpop_headers(agent._core, agent._clock.time(), method, url, self.token())  # pyright: ignore[reportPrivateUsage]


class _AsyncBoundResource(_AsyncAgentResource):
    """:class:`_BoundResource` for asyncio code."""

    __slots__ = ()

    async def headers(self, method: str, url: str) -> dict[str, str]:
        """See :meth:`_BoundResource.headers`."""
        check_destination(url, self.send_to)
        agent = cast("AsyncInstanceAuth", self._agent)
        return _dpop_headers(agent._core, agent._clock.time(), method, url, await self.token())  # pyright: ignore[reportPrivateUsage]


def _needs_nonce(status: int, body: Mapping[str, Any] | None) -> bool:
    return status in (400, 401) and body is not None and body.get("error") == _NONCE_ERROR


def _agent_id(env: Mapping[str, str]) -> str:
    agent_id = env.get("AUTHFI_AGENT_ID", "").strip()
    if not agent_id:
        raise ConfigurationError("AUTHFI_AGENT_ID is required (agt_...)")
    return agent_id


class InstanceAuth(AgentAuth):
    """An agent instance that signs in by attestation, for blocking code. See the module text."""

    sign_in = "instance"

    def __init__(
        self,
        project: AuthFI,
        agent_id: str,
        evidence: Evidence,
        *,
        config: Configuration | None = None,
        image_digest: str | None = None,
        resource: str | None = None,
        refresh_skew: float = 60.0,
    ) -> None:
        self._init(project, agent_id, cast("Signer", _InstanceSigner()), resource=resource, refresh_skew=refresh_skew)
        self._core = _Core(evidence, self.agent_id, project.issuer, config, image_digest)

    @classmethod
    def from_environment(
        cls, project: AuthFI, env: Mapping[str, str] | None = None, *, resource: str | None = None
    ) -> InstanceAuth:
        """The instance the environment names; :class:`~authfi.ConfigurationError` when it shows no evidence."""
        source = os.environ if env is None else env
        evidence = find_evidence(source)
        if evidence is None:
            raise ConfigurationError(_NO_EVIDENCE)
        chosen = resource or source.get("AUTHFI_AGENT_RESOURCE", "").strip() or None
        return cls(
            project,
            _agent_id(source),
            evidence,
            config=_config_from(source),
            image_digest=source.get("AUTHFI_IMAGE_DIGEST", "").strip() or None,
            resource=chosen,
        )

    @property
    def instance_id(self) -> str | None:
        """The id the platform gave this instance, once it has signed in."""
        held = self._core.attestation
        return held.instance_id if held else None

    def _assertion(self, audience: str) -> str:
        raise NotSupportedError("instance_assertion", _CIBA_REFUSAL)

    def _attest(self) -> Attestation:
        core, now = self._core, self._clock.time()
        held = core.reusable(now, self.refresh_skew)
        if held is not None:
            return held
        origin = self.project.resolve_api_origin()
        nonce = parse_challenge(self._send(challenge_request(origin)))
        body = core.attest_body(nonce, now)
        response = self._send(attest_request(origin, body))
        core.attestation = parse_attestation(
            response, agent_id=self.agent_id, key_thumbprint=core.key().thumbprint, now=self._clock.time()
        )
        self._log.debug("instance %s of agent %s attested", core.attestation.instance_id, self.agent_id)
        return core.attestation

    def _mint(self, resource: str | None) -> tuple[_Minted, float]:
        metadata = self.project.metadata()
        if not metadata.supports_grant("client_credentials"):
            raise NotSupportedError(
                "client_credentials", "This AuthFI project does not offer the client credentials grant."
            )
        attestation = self._attest()
        endpoint = metadata.token_endpoint

        def attested() -> HttpRequest:
            return self._core.token_request(endpoint, attestation, resource, self._clock)

        response = self._send(attested())
        _dpop.nonces.remember(endpoint, response.header("DPoP-Nonce"))
        if _needs_nonce(response.status, response.json_object()):
            response = self._send(attested())
            _dpop.nonces.remember(endpoint, response.header("DPoP-Nonce"))
        parsed = parse_token_response(response, "the token endpoint")
        self._log.debug("minted an instance token for agent %s", self.agent_id)
        return self._minted(parsed, self._core.key())

    def headers(self, method: str, url: str) -> dict[str, str]:
        """``Authorization: DPoP <token>`` and a fresh proof of the instance key for ``method url``."""
        check_destination(url, self.send_to)
        return _dpop_headers(self._core, self._clock.time(), method, url, self.token())

    def for_resource(self, resource: str, *, send_to: Sequence[str] = ()) -> TokenSource:
        """The instance's token for ``resource`` (DPoP-bound), sent only to ``send_to``."""
        if not resource or not resource.strip():
            raise ConfigurationError("resource is required")
        return _BoundResource(self, resource.strip(), self._check_destination_args(resource.strip(), send_to))

    def __repr__(self) -> str:
        return f"InstanceAuth(agent_id={self.agent_id!r}, evidence={self._core.evidence.kind!r})"


class AsyncInstanceAuth(AsyncAgentAuth):
    """:class:`InstanceAuth` for asyncio code."""

    sign_in = "instance"

    def __init__(
        self,
        project: AsyncAuthFI,
        agent_id: str,
        evidence: Evidence,
        *,
        config: Configuration | None = None,
        image_digest: str | None = None,
        resource: str | None = None,
        refresh_skew: float = 60.0,
    ) -> None:
        super().__init__(
            project, agent_id, cast("Signer", _InstanceSigner()), resource=resource, refresh_skew=refresh_skew
        )
        self._core = _Core(evidence, self.agent_id, project.issuer, config, image_digest)

    @classmethod
    def from_environment(
        cls, project: AsyncAuthFI, env: Mapping[str, str] | None = None, *, resource: str | None = None
    ) -> AsyncInstanceAuth:
        """See :meth:`InstanceAuth.from_environment`."""
        source = os.environ if env is None else env
        evidence = find_evidence(source)
        if evidence is None:
            raise ConfigurationError(_NO_EVIDENCE)
        chosen = resource or source.get("AUTHFI_AGENT_RESOURCE", "").strip() or None
        return cls(
            project,
            _agent_id(source),
            evidence,
            config=_config_from(source),
            image_digest=source.get("AUTHFI_IMAGE_DIGEST", "").strip() or None,
            resource=chosen,
        )

    @property
    def instance_id(self) -> str | None:
        """The id the platform gave this instance, once it has signed in."""
        held = self._core.attestation
        return held.instance_id if held else None

    def _assertion(self, audience: str) -> str:
        raise NotSupportedError("instance_assertion", _CIBA_REFUSAL)

    async def _attest(self) -> Attestation:
        core, now = self._core, self._clock.time()
        held = core.reusable(now, self.refresh_skew)
        if held is not None:
            return held
        origin = await self.project.resolve_api_origin()
        nonce = parse_challenge(await self._send(challenge_request(origin)))
        body = core.attest_body(nonce, now)
        response = await self._send(attest_request(origin, body))
        core.attestation = parse_attestation(
            response, agent_id=self.agent_id, key_thumbprint=core.key().thumbprint, now=self._clock.time()
        )
        self._log.debug("instance %s of agent %s attested", core.attestation.instance_id, self.agent_id)
        return core.attestation

    async def _mint(self, resource: str | None) -> tuple[_Minted, float]:
        metadata = await self.project.metadata()
        if not metadata.supports_grant("client_credentials"):
            raise NotSupportedError(
                "client_credentials", "This AuthFI project does not offer the client credentials grant."
            )
        attestation = await self._attest()
        endpoint = metadata.token_endpoint

        async def attested() -> HttpRequest:
            return self._core.token_request(endpoint, attestation, resource, self._clock)

        response = await self._send(await attested())
        _dpop.nonces.remember(endpoint, response.header("DPoP-Nonce"))
        if _needs_nonce(response.status, response.json_object()):
            response = await self._send(await attested())
            _dpop.nonces.remember(endpoint, response.header("DPoP-Nonce"))
        parsed = parse_token_response(response, "the token endpoint")
        self._log.debug("minted an instance token for agent %s", self.agent_id)
        return self._minted(parsed, self._core.key())

    async def headers(self, method: str, url: str) -> dict[str, str]:
        """See :meth:`InstanceAuth.headers`."""
        check_destination(url, self.send_to)
        return _dpop_headers(self._core, self._clock.time(), method, url, await self.token())

    def for_resource(self, resource: str, *, send_to: Sequence[str] = ()) -> AsyncTokenSource:
        """See :meth:`InstanceAuth.for_resource`."""
        if not resource or not resource.strip():
            raise ConfigurationError("resource is required")
        return _AsyncBoundResource(self, resource.strip(), self._check_destination_args(resource.strip(), send_to))

    def __repr__(self) -> str:
        return f"AsyncInstanceAuth(agent_id={self.agent_id!r}, evidence={self._core.evidence.kind!r})"


_NO_EVIDENCE = (
    "this agent shows no evidence of where it runs: on Kubernetes project a service-account token at "
    f"{K8S_TOKEN_PATH} (or name it in AUTHFI_SERVICE_ACCOUNT_TOKEN_FILE); on a device set AUTHFI_DEVICE_KEY_FILE"
)
