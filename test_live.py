"""Live tests against a real AuthFI project, through its public address only.

Skipped unless ``AUTHFI_LIVE_ORIGIN`` is set, so ``python -m unittest`` runs
without credentials. Each leg also skips when its own variables are missing.
Nothing secret is printed: a failure names claims and OAuth codes, never a token.

    AUTHFI_LIVE_ORIGIN               the project's address, e.g. https://acme.authfi.io
    AUTHFI_LIVE_<ALG>_AGENT_ID       an agent registered with a key of that type,
    AUTHFI_LIVE_<ALG>_KEY_FILE       its private key file (PEM or private JWK),
    AUTHFI_LIVE_<ALG>_KID            and the kid it was registered with,
                                     for <ALG> in EDDSA, ES256, PS256
    AUTHFI_LIVE_CIBA_LOGIN_HINT      a person in the project the EDDSA agent may ask
    AUTHFI_LIVE_CIBA_SCOPE           a scope within that agent's capabilities
    AUTHFI_LIVE_PERSON_TOKEN         that person's own console access token, used to
                                     approve the request as they would in the console

    AUTHFI_LIVE_ORIGIN=https://acme.authfi.io ... python -m unittest test_live -v
"""

import asyncio
import json
import os
import threading
import time
import unittest
import uuid
from urllib.request import Request, urlopen

from authfi import (
    USER_AGENT,
    AgentAuth,
    AgentAuthError,
    AgentKey,
    AsyncAgentAuth,
    AuthFI,
    DelegatedToken,
    _b64decode,
)

ORIGIN = (os.environ.get("AUTHFI_LIVE_ORIGIN") or "").rstrip("/")
ALGS = ("EDDSA", "ES256", "PS256")


def _env(name):
    return os.environ.get("AUTHFI_LIVE_" + name)


def _agent(alg):
    """The configured agent for ``alg``, or None when that leg is not configured."""
    agent_id, key_file = _env(alg + "_AGENT_ID"), _env(alg + "_KEY_FILE")
    if not (agent_id and key_file):
        return None
    key = AgentKey.from_file(key_file, key_id=_env(alg + "_KID"))
    return AgentAuth(agent_id=agent_id, key=key, platform_url=ORIGIN)


def _claims(token):
    return json.loads(_b64decode(token.split(".")[1]))


def _verified(token):
    """The token's claims, after checking its signature against the project's published keys."""
    return AuthFI(tenant="live", api_key="unused", issuer=ORIGIN).verify_token(token)


def _console(query, variables, token):
    """One console API call made as the person, with their own token."""
    req = Request(ORIGIN + "/graphql", method="POST")
    req.add_header("Authorization", "Bearer " + token)
    req.add_header("Content-Type", "application/json")
    req.add_header("User-Agent", USER_AGENT)
    body = json.dumps({"query": query, "variables": variables}).encode()
    with urlopen(req, body) as res:
        out = json.loads(res.read())
    if out.get("errors"):
        raise AssertionError("console API refused: %s" % [e.get("message") for e in out["errors"]])
    return out["data"]


def approve_as_person(agent_id, person_token, timeout=60):
    """Find the agent's pending request in the person's own list and approve it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        mine = _console("{ myAgentRequests { authReqId agentId status } }", {}, person_token)
        pending = [r for r in mine["myAgentRequests"]
                   if r["agentId"] == agent_id and r["status"] == "pending"]
        if pending:
            decided = _console(
                "mutation($id:ID!){ decideMyAgentRequest(authReqId:$id, decision:APPROVE){ status } }",
                {"id": pending[0]["authReqId"]}, person_token,
            )
            return decided["decideMyAgentRequest"]["status"]
        time.sleep(1)
    raise AssertionError("no pending request from %s reached the person" % agent_id)


@unittest.skipUnless(ORIGIN, "AUTHFI_LIVE_ORIGIN is not set")
class TestLiveAgentToken(unittest.TestCase):
    def test_each_key_type_gets_a_token(self):
        ran = 0
        for alg in ALGS:
            agent = _agent(alg)
            if agent is None:
                continue
            with self.subTest(alg=alg):
                token = agent.token()
                header = json.loads(_b64decode(token.split(".")[0]))
                claims = _verified(token)
                self.assertEqual(claims["iss"], ORIGIN)
                self.assertIn(agent.agent_id, (claims.get("sub"), claims.get("client_id")))
                self.assertEqual(header.get("typ"), "at+jwt")
                # The second call is the cached token, not a second mint.
                self.assertEqual(agent.token(), token)
                ran += 1
        if not ran:
            self.skipTest("no AUTHFI_LIVE_<ALG>_AGENT_ID / _KEY_FILE configured")

    def test_the_agent_mints_at_the_token_endpoint_the_project_publishes(self):
        agent = _agent("EDDSA")
        if agent is None:
            self.skipTest("AUTHFI_LIVE_EDDSA_AGENT_ID / _KEY_FILE not configured")
        published = agent.authorization_server_metadata()["token_endpoint"]
        self.assertEqual(agent.token_endpoint, published)
        self.assertEqual(published, ORIGIN + "/oauth/token")
        self.assertTrue(agent.token())

    def test_a_deprecated_token_route_is_reported(self):
        agent = _agent("EDDSA")
        if agent is None:
            self.skipTest("AUTHFI_LIVE_EDDSA_AGENT_ID / _KEY_FILE not configured")
        legacy = (agent.discovery().get("endpoints") or {}).get("token")
        if not legacy or legacy == agent.token_endpoint:
            self.skipTest("agent-configuration names no separate token route")
        import authfi
        authfi._deprecations_reported.discard(legacy)
        old = AgentAuth(agent_id=agent.agent_id, key=agent.key, platform_url=ORIGIN, token_endpoint=legacy)
        with self.assertLogs("authfi", "WARNING") as logs:
            self.assertTrue(old.token())
        self.assertIn(legacy + " is deprecated", logs.output[0])
        self.assertIn("its successor is " + ORIGIN + "/oauth/token", logs.output[0])

    def test_a_key_the_agent_did_not_register_is_refused(self):
        for alg in ALGS:
            agent_id, kid = _env(alg + "_AGENT_ID"), _env(alg + "_KID")
            if not (agent_id and kid):
                continue
            stranger = AgentKey.generate(kid, {"EDDSA": "EdDSA"}.get(alg, alg))
            agent = AgentAuth(agent_id=agent_id, key=stranger, platform_url=ORIGIN)
            with self.assertRaises(AgentAuthError) as ctx:
                agent.token()
            self.assertEqual(ctx.exception.error, "invalid_client")
            return
        self.skipTest("no AUTHFI_LIVE_<ALG>_AGENT_ID / _KID configured")


@unittest.skipUnless(ORIGIN, "AUTHFI_LIVE_ORIGIN is not set")
class TestLiveCIBA(unittest.TestCase):
    def setUp(self):
        self.agent = _agent("EDDSA")
        self.login_hint = _env("CIBA_LOGIN_HINT")
        self.person_token = _env("PERSON_TOKEN")
        if not (self.agent and self.login_hint and self.person_token):
            self.skipTest("CIBA needs the EDDSA agent, AUTHFI_LIVE_CIBA_LOGIN_HINT and AUTHFI_LIVE_PERSON_TOKEN")
        self.scope = _env("CIBA_SCOPE")

    def test_the_person_approves_and_the_agent_acts_for_them(self):
        approval = self.agent.request_user_approval(
            self.login_hint,
            binding_message="SDK live test %s: let the agent act for you" % uuid.uuid4().hex[:6],
            scope=self.scope,
            requested_expiry=120,
        )
        decided = {}
        person = threading.Thread(
            target=lambda: decided.update(status=approve_as_person(self.agent.agent_id, self.person_token))
        )
        person.start()
        out = self.agent.await_user_token(approval, timeout=90)
        person.join()
        self.assertEqual(decided.get("status"), "approved")
        claims = _verified(out["access_token"])
        # sub is the PERSON; act.sub is this agent (RFC 8693 §4.1).
        self.assertEqual(claims["sub"], out["subject"])
        self.assertIn(self.agent.agent_id, claims["act"]["sub"])
        self.assertEqual(claims["client_id"], self.agent.agent_id)


@unittest.skipUnless(ORIGIN, "AUTHFI_LIVE_ORIGIN is not set")
class TestLiveAsync(unittest.TestCase):
    def setUp(self):
        agent = _agent("EDDSA")
        if agent is None:
            self.skipTest("AUTHFI_LIVE_EDDSA_AGENT_ID / _KEY_FILE not configured")
        self.agent = AsyncAgentAuth(agent)

    def test_the_async_ciba_wait_leaves_the_loop_free(self):
        login_hint, person_token = _env("CIBA_LOGIN_HINT"), _env("PERSON_TOKEN")
        if not (login_hint and person_token):
            self.skipTest("needs AUTHFI_LIVE_CIBA_LOGIN_HINT and AUTHFI_LIVE_PERSON_TOKEN")
        agent_id = self.agent.agent.agent_id

        async def main():
            approval = await self.agent.request_user_approval(
                login_hint,
                binding_message="SDK async live test %s: let the agent act for you" % uuid.uuid4().hex[:6],
                scope=_env("CIBA_SCOPE"),
                requested_expiry=120,
            )
            ticks, stop = [], asyncio.Event()

            async def ticker():
                while not stop.is_set():
                    ticks.append(time.monotonic())
                    await asyncio.sleep(0.05)
            tick_task = asyncio.ensure_future(ticker())
            loop = asyncio.get_running_loop()
            person = loop.run_in_executor(None, approve_as_person, agent_id, person_token)
            started = time.monotonic()
            token = await self.agent.await_user_token(approval, timeout=90)
            waited = time.monotonic() - started
            decided = await person
            stop.set()
            await tick_task
            return token, decided, ticks, waited, await token.atoken()

        token, decided, ticks, waited, current = asyncio.run(main())
        self.assertEqual(decided, "approved")
        self.assertIsInstance(token, DelegatedToken)
        self.assertEqual(current, token["access_token"])
        claims = _verified(current)
        self.assertEqual(claims["sub"], token.subject)
        self.assertIn(agent_id, claims["act"]["sub"])
        # The wait took at least one poll interval; the loop ticked through it.
        self.assertGreater(waited, 1)
        self.assertGreater(len(ticks), waited / 0.05 * 0.5)

    def test_httpx_auth_carries_a_token_the_project_accepts(self):
        try:
            import httpx
        except ImportError:
            self.skipTest("httpx is not installed")
        resource = self.agent.agent.discovery().get("default_location")
        if not resource:
            self.skipTest("discovery names no default_location to call")
        # An MCP initialize: the cheapest call the default resource answers for an agent.
        initialize = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "authfi-python-sdk-live-test", "version": "0"}}}
        headers = {"Accept": "application/json, text/event-stream", "User-Agent": USER_AGENT}

        async def main():
            async with httpx.AsyncClient(timeout=30) as client:
                anonymous = await client.post(resource, json=initialize, headers=headers)
                signed = await client.post(resource, json=initialize, headers=headers,
                                           auth=self.agent.httpx_auth())
            return anonymous.status_code, signed.status_code
        anonymous, signed = asyncio.run(main())
        self.assertEqual(anonymous, 401)
        self.assertEqual(signed, 200)


if __name__ == "__main__":
    unittest.main()
