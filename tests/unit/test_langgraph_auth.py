"""``authfi.langgraph_auth``: an ``Auth()`` for langgraph.json, built on the real ``langgraph-sdk`` type."""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from langgraph_sdk import Auth

import authfi
from authfi import AsyncAuthFI, Secret
from authfi._dpop import DPoPKey
from authfi._dpop_check import ProofReplayCache, check_proof
from authfi.langgraph_auth import create_auth, resource_of
from tests.agent_platform import AGENT, AgentPlatform
from tests.instance_platform import thumbprint
from tests.support import json_response

AUDIENCE = "api_graphs"
ORIGIN = "https://graphs.example.com"
THREADS = ORIGIN + "/threads"


class Deployment:
    """A project, the deployment's own credential, and the Auth built from them."""

    def __init__(self) -> None:
        self.platform = AgentPlatform()
        self.checks: list[dict[str, Any]] = []
        self.answer: dict[str, Any] = {"allowed": True}
        self.platform.route("POST", "/apis/regional/v1/check", self._check)
        self.platform.route(
            "POST",
            "/oauth/token",
            lambda _r: json_response({"access_token": "svc-token", "token_type": "Bearer", "expires_in": 900}),
        )
        self.project = AsyncAuthFI(self.platform.issuer, http_client=self.platform.async_client())
        self.service = self.project.client_credentials(
            "app_graphs", Secret("s"), resource=self.platform.issuer + "/mcp"
        )
        self.auth = create_auth(self.project, self.service, audience=[AUDIENCE])

    def _check(self, request: httpx.Request) -> httpx.Response:
        import json

        self.checks.append(json.loads(request.content))
        return json_response({"data": self.answer})

    async def authenticate(
        self, authorization: str | None, *, dpop: str | None = None, method: str = "POST", url: str = THREADS
    ) -> Any:
        headers = {b"authorization": (authorization or "").encode()}
        if dpop:
            headers[b"dpop"] = dpop.encode()
        parts = httpx.URL(url)
        request = SimpleNamespace(
            url=SimpleNamespace(scheme=parts.scheme, netloc=parts.netloc.decode(), path=parts.path)
        )
        return await self.auth._authenticate_handler(  # pyright: ignore[reportPrivateUsage]
            request=request, method=method, authorization=authorization, headers=headers
        )

    async def authorize(
        self, resource: str, action: str, value: dict[str, Any], user: dict[str, Any] | None = None
    ) -> Any:
        ctx = Auth.types.AuthContext(
            resource=resource,  # type: ignore[arg-type]
            action=action,  # type: ignore[arg-type]
            user=_User(user or {"identity": "user:usr_1", "permissions": []}),  # type: ignore[arg-type]
            permissions=[],
        )
        return await self.auth._global_handlers[0](ctx=ctx, value=value)  # pyright: ignore[reportPrivateUsage]


class _User(dict[str, Any]):
    @property
    def identity(self) -> str:
        return str(self["identity"])


@pytest.fixture
async def deployment() -> Any:
    d = Deployment()
    yield d
    await d.project.aclose()


def bound_token(d: Deployment, key: DPoPKey, **claims: Any) -> str:
    return d.platform.agent_token(AUDIENCE, cnf={"jkt": key.thumbprint}, client_id=AGENT, **claims)


def proof(key: DPoPKey, token: str, *, method: str = "POST", url: str = THREADS, now: float | None = None) -> str:
    return key.proof(method, url, now=now or time.time(), access_token=token)


# --- the object is the real type --------------------------------------------------------------------


async def test_it_is_a_langgraph_auth_with_one_authenticate_and_one_authorize(deployment: Deployment) -> None:
    assert isinstance(deployment.auth, Auth)
    assert deployment.auth._authenticate_handler is not None  # pyright: ignore[reportPrivateUsage]
    assert len(deployment.auth._global_handlers) == 1  # pyright: ignore[reportPrivateUsage]


# --- authenticate -------------------------------------------------------------------------------------


async def test_a_persons_bearer_token_is_the_user(deployment: Deployment) -> None:
    token = deployment.platform.access_token(sub="user:usr_1", aud=[AUDIENCE], scope="threads:read")
    user = await deployment.authenticate("Bearer " + token)
    assert user["identity"] == "user:usr_1"
    assert "threads:read" in user["permissions"]
    assert user["token_kind"] == "person"


async def test_an_agent_token_with_its_proof_is_the_agent(deployment: Deployment) -> None:
    key = DPoPKey()
    token = bound_token(deployment, key)
    user = await deployment.authenticate("DPoP " + token, dpop=proof(key, token))
    assert user["identity"] == f"agent:{AGENT}"
    assert user["token_kind"] == "agent"


async def test_an_agent_acting_for_a_person_is_the_person_with_the_agent_named(deployment: Deployment) -> None:
    key = DPoPKey()
    token = bound_token(deployment, key, sub="user:usr_7", act={"sub": f"agent:{AGENT}"})
    user = await deployment.authenticate("DPoP " + token, dpop=proof(key, token))
    assert user["identity"] == "user:usr_7"
    assert user["actor"] == f"agent:{AGENT}"


def refused(code: int = 401) -> Any:
    return pytest.raises(Auth.exceptions.HTTPException, match=str(code))


async def test_a_bound_token_as_bearer_is_refused(deployment: Deployment) -> None:
    key = DPoPKey()
    token = bound_token(deployment, key)
    with refused() as caught:
        await deployment.authenticate("Bearer " + token)
    assert caught.value.detail == "invalid_token"  # one opaque answer: no code for a caller to probe with


@pytest.mark.parametrize(
    "tamper",
    [
        "no_proof",
        "other_url",
        "other_method",
        "other_token",
        "other_key",
        "stale",
        "garbage",
        "unbound_token",
    ],
)
async def test_every_wrong_proof_is_refused(deployment: Deployment, tamper: str) -> None:
    key, other = DPoPKey(), DPoPKey()
    token = bound_token(deployment, key)
    other_token = bound_token(deployment, key, jti="another")
    proofs = {
        "no_proof": None,
        "other_url": proof(key, token, url=ORIGIN + "/assistants"),
        "other_method": proof(key, token, method="GET"),
        "other_token": proof(key, other_token),
        "other_key": proof(other, token),
        "stale": proof(key, token, now=time.time() - 600),
        "garbage": "not.a.jwt",
        "unbound_token": proof(key, token),
    }
    if tamper == "unbound_token":
        token = deployment.platform.agent_token(AUDIENCE)
        proofs["unbound_token"] = proof(key, token)
    with refused():
        await deployment.authenticate("DPoP " + token, dpop=proofs[tamper])


async def test_a_proof_works_once(deployment: Deployment) -> None:
    key = DPoPKey()
    token = bound_token(deployment, key)
    p = proof(key, token)
    await deployment.authenticate("DPoP " + token, dpop=p)
    with refused():
        await deployment.authenticate("DPoP " + token, dpop=p)


async def test_a_public_origin_behind_a_proxy_is_what_the_proof_names(deployment: Deployment) -> None:
    deployment.auth = create_auth(
        deployment.project, deployment.service, audience=[AUDIENCE], public_origin="https://public.example.com"
    )
    key = DPoPKey()
    token = bound_token(deployment, key)
    user = await deployment.authenticate(
        "DPoP " + token, dpop=proof(key, token, url="https://public.example.com/threads")
    )
    assert user["identity"] == f"agent:{AGENT}"


@pytest.mark.parametrize("header", [None, "", "Basic abc", "Bearer ", "Bearer not-a-jwt"])
async def test_no_usable_credential_is_refused(deployment: Deployment, header: str | None) -> None:
    with refused():
        await deployment.authenticate(header)


async def test_a_token_for_another_audience_or_tenant_or_time_is_refused(deployment: Deployment) -> None:
    platform = deployment.platform
    for token in (
        platform.access_token(aud=["api_other"]),
        platform.access_token(aud=[AUDIENCE], exp=int(time.time()) - 3600),
        platform.access_token(aud=[AUDIENCE], iss="https://evil.example"),
        platform.access_token(aud=[AUDIENCE], tenant_id=None),
    ):
        with refused():
            await deployment.authenticate("Bearer " + token)


# --- authorize ----------------------------------------------------------------------------------------


async def test_the_policy_engine_is_asked_what_the_caller_wants_to_do(deployment: Deployment) -> None:
    assert await deployment.authorize("threads", "read", {"thread_id": "t-9"}) is None
    sent = deployment.checks[-1]
    assert sent == {"subject": "user:usr_1", "relation": "read", "object": "langgraph:threads:t-9"}


async def test_creating_asks_about_the_collection_and_a_store_call_about_its_namespace(deployment: Deployment) -> None:
    await deployment.authorize("assistants", "create", {})
    assert deployment.checks[-1]["object"] == "langgraph:assistants:*"
    await deployment.authorize("store", "put", {"namespace": ["u", "notes"]})
    assert deployment.checks[-1]["object"] == "langgraph:store:u/notes"
    assert resource_of("runs", {"thread_id": "t-1", "run_id": "r-2"}) == "langgraph:runs:r-2"


async def test_the_agent_acting_for_a_person_is_told_to_the_engine(deployment: Deployment) -> None:
    user: dict[str, Any] = {"identity": "user:usr_7", "permissions": [], "actor": f"agent:{AGENT}"}
    await deployment.authorize("threads", "create", {}, user)
    assert deployment.checks[-1]["context"] == {"actor": f"agent:{AGENT}"}


async def test_a_denied_call_is_a_403_and_a_held_call_says_so(deployment: Deployment) -> None:
    deployment.answer = {"allowed": False, "reason": "no grant"}
    with refused(403) as caught:
        await deployment.authorize("threads", "delete", {"thread_id": "t-1"})
    assert caught.value.detail == "forbidden"
    deployment.answer = {"allowed": False, "requires_approval": True}
    with refused(403) as held:
        await deployment.authorize("runs", "create_run", {"thread_id": "t-1"})
    assert held.value.detail == "needs_approval"


async def test_a_failed_check_denies_rather_than_allows(deployment: Deployment) -> None:
    deployment.platform.route("POST", "/apis/regional/v1/check", json_response({"error": "boom"}, status=500))
    with pytest.raises(authfi.AuthFIError):
        await deployment.authorize("threads", "read", {"thread_id": "t-1"})


async def test_a_studio_user_is_asked_about_like_anyone_else(deployment: Deployment) -> None:
    ctx = Auth.types.AuthContext(
        resource="threads",
        action="read",
        user=Auth.types.StudioUser("dev", is_authenticated=True),  # type: ignore[arg-type]
        permissions=[],
    )
    deployment.answer = {"allowed": False}
    with refused(403):
        await deployment.auth._global_handlers[0](ctx=ctx, value={"thread_id": "t"})  # pyright: ignore[reportPrivateUsage]
    assert deployment.checks[-1]["subject"] == "dev"


def test_thumbprint_helpers_agree() -> None:
    key = DPoPKey()
    assert thumbprint(key.public_jwk()) == key.thumbprint


def test_a_token_bound_to_no_key_is_named_so_before_the_proof_is_read() -> None:
    key = DPoPKey()
    with pytest.raises(authfi.InvalidToken) as caught:
        check_proof(
            proof(key, "t"),
            method="POST",
            url=THREADS,
            access_token="t",
            jkt=None,
            now=time.time(),
            replay=ProofReplayCache(),
        )
    assert caught.value.code == "token_dpop_unbound"
