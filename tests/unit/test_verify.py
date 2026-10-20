from __future__ import annotations

import hashlib
import hmac
import json
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization

from authfi import (
    AccessToken,
    AsyncAuthFI,
    AsyncMultiIssuerVerifier,
    AuthFI,
    ConfigurationError,
    InsufficientAuthentication,
    InsufficientPermission,
    InsufficientScope,
    InvalidToken,
    MultiIssuerVerifier,
    Principal,
    Secret,
)
from tests.support import TENANT, FakeClock, Platform, b64url

AUD = ["api_orders"]


def code_of(fn: Any, *args: Any, **kwargs: Any) -> str:
    with pytest.raises(InvalidToken) as caught:
        fn(*args, **kwargs)
    return caught.value.code


def unsigned(header: dict[str, Any], claims: dict[str, Any], signature: bytes = b"") -> str:
    return ".".join([b64url(json.dumps(header).encode()), b64url(json.dumps(claims).encode()), b64url(signature)])


def claims(platform: Platform, **extra: Any) -> dict[str, Any]:
    now = int(time.time())
    base: dict[str, Any] = {
        "iss": platform.issuer,
        "sub": "usr_1",
        "aud": AUD,
        "exp": now + 600,
        "iat": now,
        "tenant_id": TENANT,
        "client_id": "app_web",
    }
    base.update(extra)
    return {k: v for k, v in base.items() if v is not None}


# --- the seven shapes -----------------------------------------------------------------------


def test_a_persons_token_t1(af: AuthFI, platform: Platform) -> None:
    token = af.verifier(audience=AUD).verify(
        platform.access_token(
            permissions=["orders:read"],
            roles=["admin"],
            groups=["g1"],
            acr="urn:authfi:acr:mfa",
            amr=["pwd", "otp"],
            auth_time=int(time.time()) - 30,
            custom={"org_id": "org_9"},
            scope="openid profile",
        )
    )
    assert token.principal == Principal("user", "usr_1")
    assert token.actor is None
    assert token.token_kind == "person"
    assert token.permissions == frozenset({"orders:read"})
    assert token.roles == frozenset({"admin"})
    assert token.groups == frozenset({"g1"})
    assert token.org_id == "org_9"
    assert token.acr == "urn:authfi:acr:mfa"
    assert token.amr == ("pwd", "otp")
    assert token.scopes == frozenset({"openid", "profile"})
    assert token.tenant_id == TENANT
    assert token.audience == ("api_orders",)
    assert str(token.principal) == "user:usr_1"


def test_a_machine_token_t2(af: AuthFI, platform: Platform) -> None:
    token = af.verifier(audience=AUD).verify(platform.access_token(sub="app_reports", client_id="app_reports"))
    assert token.principal == Principal("app", "app_reports")
    assert token.token_kind == "machine"


def test_an_agent_token_t4_is_eddsa_with_a_string_audience(af: AuthFI, platform: Platform) -> None:
    raw = platform.access_token(alg="EdDSA", sub="agt_1", client_id="agt_1", aud="api_orders", scope="mcp:x.y")
    token = af.verifier(audience=AUD).verify(raw)
    assert token.principal == Principal("agent", "agt_1")
    assert token.token_kind == "agent"
    assert token.scopes == frozenset({"mcp:x.y"})
    token.require_agent()


def test_a_ciba_delegated_token_t5_has_typed_parties(af: AuthFI, platform: Platform) -> None:
    raw = platform.access_token(alg="EdDSA", sub="user:usr_7", act={"sub": "agent:agt_1"}, client_id="agt_1")
    token = af.verifier(audience=AUD).verify(raw)
    assert token.principal == Principal("user", "usr_7")
    assert token.actor == Principal("agent", "agt_1")
    assert token.token_kind == "delegated"
    token.require_agent()


def test_an_exchanged_delegated_token_t6_has_a_bare_subject(af: AuthFI, platform: Platform) -> None:
    raw = platform.access_token(sub="usr_7", act={"sub": "agt_1"}, client_id="agt_1")
    token = af.verifier(audience=AUD).verify(raw)
    assert token.principal == Principal("user", "usr_7")
    assert token.actor == Principal("agent", "agt_1")
    assert token.token_kind == "delegated"


# One table, read the same by every SDK (the conformance vectors carry these cases too).
KINDS: list[tuple[str, dict[str, Any], tuple[str, str, str | None, str | None]]] = [
    (
        "typed agent with an actor",
        {"sub": "agent:agt_1", "act": {"sub": "agt_2"}},
        ("agent", "agent:agt_1", "agent:agt_2", None),
    ),
    (
        "typed app with an actor",
        {"sub": "app:app_1", "act": {"sub": "agt_1"}},
        ("machine", "app:app_1", "agent:agt_1", None),
    ),
    ("typed app alone", {"sub": "app:app_1"}, ("machine", "app:app_1", None, None)),
    (
        "typed user with an actor",
        {"sub": "user:usr_1", "act": {"sub": "agent:agt_1"}},
        ("delegated", "user:usr_1", "agent:agt_1", None),
    ),
    ("typed user alone", {"sub": "user:usr_1"}, ("person", "user:usr_1", None, None)),
    (
        "bare subject with an actor",
        {"sub": "usr_1", "act": {"sub": "agt_1"}},
        ("delegated", "user:usr_1", "agent:agt_1", None),
    ),
    ("agt_ subject", {"sub": "agt_1", "client_id": "agt_1"}, ("agent", "agent:agt_1", None, None)),
    ("subject equals client_id", {"sub": "app_1", "client_id": "app_1"}, ("machine", "app:app_1", None, None)),
    (
        "subject equals azp, no client_id",
        {"sub": "app_1", "client_id": None, "azp": "app_1"},
        ("person", "user:app_1", None, None),
    ),
    ("scp is not the scope claim", {"scp": "a b", "scope": None}, ("person", "user:usr_1", None, "")),
]


@pytest.mark.parametrize(("name", "extra", "expected"), KINDS, ids=[k[0] for k in KINDS])
def test_the_kind_of_token_is_read_the_way_node_reads_it(
    af: AuthFI, platform: Platform, name: str, extra: dict[str, Any], expected: tuple[str, str, str | None, str | None]
) -> None:
    kind, principal, actor, scopes = expected
    token = af.verifier(audience=AUD).verify(platform.access_token(**extra))
    assert token.token_kind == kind
    assert str(token.principal) == principal
    assert (str(token.actor) if token.actor else None) == actor
    if scopes is not None:
        assert token.scopes == frozenset(scopes.split())
    if "azp" in extra:
        assert token.client_id == ""


def test_an_es256_token_verifies(af: AuthFI, platform: Platform) -> None:
    assert af.verifier(audience=AUD).verify(platform.access_token(alg="ES256")).principal.id == "usr_1"


def test_a_person_is_not_an_agent(af: AuthFI, platform: Platform) -> None:
    token = af.verifier(audience=AUD).verify(platform.access_token())
    assert code_of(token.require_agent) == "token_no_agent"


def test_the_token_keeps_no_raw_token(af: AuthFI, platform: Platform) -> None:
    raw = platform.access_token()
    token = af.verifier(audience=AUD).verify(Secret(raw))
    assert raw not in repr(token)
    assert raw.split(".")[2] not in repr(token.claims)


# --- refusals, each with the kernel's code --------------------------------------------------


@pytest.fixture
def verify(af: AuthFI) -> Any:
    return af.verifier(audience=AUD).verify


def test_alg_none_is_refused_before_any_key_lookup(verify: Any, platform: Platform) -> None:
    raw = unsigned({"alg": "none", "typ": "at+jwt", "kid": "rsa-1"}, claims(platform))
    assert code_of(verify, raw) == "token_signature_invalid"
    assert platform.calls("GET", "/.well-known/jwks.json") == []


def test_hs256_signed_with_the_rsa_public_key_is_refused(verify: Any, platform: Platform) -> None:
    public_pem = platform.keys.rsa_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    header = b64url(json.dumps({"alg": "HS256", "typ": "at+jwt", "kid": "rsa-1"}).encode())
    payload = b64url(json.dumps(claims(platform)).encode())
    signature = hmac.new(public_pem, f"{header}.{payload}".encode(), hashlib.sha256).digest()
    assert code_of(verify, f"{header}.{payload}.{b64url(signature)}") == "token_signature_invalid"


@pytest.mark.parametrize("typ", ["JWT", "txn_token+jwt", "secevent+jwt", None])
def test_only_an_access_token_type_is_accepted(verify: Any, platform: Platform, typ: str | None) -> None:
    raw = platform.keys.sign(claims(platform), typ=typ)
    assert code_of(verify, raw) == "token_not_access_token"


def test_the_application_at_jwt_spelling_is_accepted_in_any_case(verify: Any, platform: Platform) -> None:
    assert verify(platform.keys.sign(claims(platform), typ="Application/AT+JWT")).principal.id == "usr_1"


@pytest.mark.parametrize(
    ("aud", "ok"),
    [(None, False), ("api_orders", True), (["x", "api_orders"], True), (["x"], False), ("api_other", False)],
)
def test_the_audience_is_normalised_and_must_match(verify: Any, platform: Platform, aud: Any, ok: bool) -> None:
    raw = platform.keys.sign(claims(platform, aud=aud))
    if ok:
        verify(raw)
    else:
        assert code_of(verify, raw) == "token_audience_mismatch"


def test_exp_is_required(verify: Any, platform: Platform) -> None:
    assert code_of(verify, platform.keys.sign(claims(platform, exp=None))) == "token_claims_invalid"
    assert code_of(verify, platform.keys.sign(claims(platform, exp="soon"))) == "token_claims_invalid"


def test_expiry_and_issue_times_respect_the_leeway(af: AuthFI, platform: Platform) -> None:
    now = int(time.time())
    verifier = af.verifier(audience=AUD, leeway=60)
    verifier.verify(platform.keys.sign(claims(platform, exp=now - 30)))
    assert code_of(verifier.verify, platform.keys.sign(claims(platform, exp=now - 120))) == "token_expired"
    verifier.verify(platform.keys.sign(claims(platform, nbf=now + 30, iat=now + 30)))
    assert code_of(verifier.verify, platform.keys.sign(claims(platform, nbf=now + 120))) == "token_claims_invalid"
    assert code_of(verifier.verify, platform.keys.sign(claims(platform, iat=now + 120))) == "token_claims_invalid"
    strict = af.verifier(audience=AUD, leeway=0)
    assert code_of(strict.verify, platform.keys.sign(claims(platform, exp=now - 1))) == "token_expired"


def test_a_token_is_expired_from_the_second_exp_plus_the_leeway_is_reached(
    af: AuthFI, platform: Platform, clock: FakeClock
) -> None:
    """RFC 7519 §4.1.4: ``exp`` is the first instant the token is not to be accepted. Node does the same."""
    now = int(clock.wall)
    clock.wall = float(now)
    verifier = af.verifier(audience=AUD, leeway=60)
    assert verifier.verify(platform.keys.sign(claims(platform, exp=now - 59))).principal.id == "usr_1"
    assert code_of(verifier.verify, platform.keys.sign(claims(platform, exp=now - 60))) == "token_expired"
    assert code_of(af.verifier(audience=AUD, leeway=0).verify, platform.keys.sign(claims(platform, exp=now))) == (
        "token_expired"
    )


def test_the_issuer_must_match_exactly(verify: Any, platform: Platform) -> None:
    raw = platform.keys.sign(claims(platform, iss=platform.issuer + "/"))
    assert code_of(verify, raw) == "token_issuer_not_accepted"
    assert code_of(verify, platform.keys.sign(claims(platform, iss="https://evil.authfi.io"))) == (
        "token_issuer_not_accepted"
    )


def test_an_unknown_kid_refetches_once_then_is_rate_limited(af: AuthFI, platform: Platform, clock: FakeClock) -> None:
    verifier = af.verifier(audience=AUD)
    verifier.verify(platform.access_token())
    clock.advance(31)
    for n in range(5):
        raw = platform.keys.sign(claims(platform), kid=f"invented-{n}")
        assert code_of(verifier.verify, raw) == "token_key_unknown"
    assert len(platform.calls("GET", "/.well-known/jwks.json")) == 2


def test_a_token_without_a_kid_is_refused(verify: Any, platform: Platform) -> None:
    raw = platform.keys.sign(claims(platform), kid="")
    assert code_of(verify, raw) == "token_key_unknown"


def test_an_alg_that_disagrees_with_the_key_is_refused(verify: Any, platform: Platform) -> None:
    # An ES256 signature presented under the RSA key's kid.
    raw = platform.keys.sign(claims(platform), alg="ES256", kid="rsa-1")
    assert code_of(verify, raw) == "token_key_unknown"


def test_an_unknown_crit_is_refused(verify: Any, platform: Platform) -> None:
    raw = platform.keys.sign(claims(platform), headers={"crit": ["exp"]})
    assert code_of(verify, raw) == "token_claims_invalid"


def test_a_jku_in_the_header_is_ignored(verify: Any, platform: Platform) -> None:
    raw = platform.keys.sign(claims(platform), headers={"jku": "https://evil.example/jwks.json"})
    verify(raw)
    assert not [r for r in platform.requests if "evil" in str(r.url)]


@pytest.mark.parametrize("bound", [{"cnf": {"jkt": "abc"}}, {"authfi_mandate": {"id": "m1"}}])
def test_a_sender_constrained_token_is_refused_as_bearer(
    verify: Any, platform: Platform, bound: dict[str, Any]
) -> None:
    assert code_of(verify, platform.keys.sign(claims(platform, **bound))) == "token_dpop_proof_required"


def test_a_bound_token_is_refused_before_an_actor_chain_is_looked_at(verify: Any, platform: Platform) -> None:
    raw = platform.access_token(cnf={"jkt": "x"}, act={"sub": "agt_1", "act": {"sub": "agt_2"}})
    assert code_of(verify, raw) == "token_dpop_proof_required"


def test_a_nested_actor_chain_is_refused(verify: Any, platform: Platform) -> None:
    raw = platform.keys.sign(claims(platform, act={"sub": "agt_1", "act": {"sub": "agt_2"}}))
    assert code_of(verify, raw) == "token_act_chain_unsupported"


def test_tenant_and_subject_are_required(verify: Any, platform: Platform) -> None:
    assert code_of(verify, platform.keys.sign(claims(platform, tenant_id=None))) == "token_no_tenant"
    assert code_of(verify, platform.keys.sign(claims(platform, sub=None))) == "token_claims_invalid"


def test_the_tenant_pin(af: AuthFI, platform: Platform) -> None:
    pinned = af.verifier(audience=AUD, expected_tenant_id=TENANT)
    pinned.verify(platform.access_token())
    other = platform.keys.sign(claims(platform, tenant_id="00000000-0000-0000-0000-000000000000"))
    assert code_of(pinned.verify, other) == "token_tenant_not_issuers"


@pytest.mark.parametrize(
    "raw",
    ["", "a.b", "a.b.c.d.e", "!!!.???.###", "e30.e30", "W10.e30.sig", "x" * 20000],
)
def test_malformed_tokens(verify: Any, raw: str) -> None:
    assert code_of(verify, raw) in ("token_malformed", "token_missing")


def test_a_token_over_16_kib_is_refused_before_any_key_is_fetched(af: AuthFI, platform: Platform) -> None:
    padding = "x" * (17 * 1024)
    raw = f"{b64url(b'{}')}.{b64url(padding.encode())}.{b64url(b'sig')}"
    assert code_of(af.verifier(audience=AUD).verify, raw) == "token_malformed"
    assert platform.calls("GET", "/.well-known/jwks.json") == []


def test_a_tampered_payload_does_not_verify(verify: Any, platform: Platform) -> None:
    header, _, signature = platform.access_token().split(".")
    forged = b64url(json.dumps(claims(platform, permissions=["admin"])).encode())
    assert code_of(verify, f"{header}.{forged}.{signature}") == "token_signature_invalid"


# --- the audience is never AuthFI's own -----------------------------------------------------


@pytest.mark.parametrize(
    "audience",
    [
        ["https://acme.authfi.io"],
        ["https://acme.authfi.io/mcp"],
        ["https://acme.authfi.io/mcp/v1"],
        ["authfi-console"],
        ["authfi-workspace"],
        ["api_orders", "https://acme.authfi.io/mcp"],
    ],
)
def test_authfis_own_audiences_are_refused(af: AuthFI, audience: list[str]) -> None:
    with pytest.raises(ConfigurationError, match="resource=<your API's client id>"):
        af.verifier(audience=audience)


def test_a_url_under_the_api_origin_is_refused(clock: FakeClock) -> None:
    af = AuthFI("https://login.acme.com", api_origin="https://acme.authfi.io", clock=clock)
    with pytest.raises(ConfigurationError):
        af.verifier(audience=["https://acme.authfi.io/mcp"])


def test_an_audience_under_an_api_origin_learned_from_discovery_is_refused_on_first_use(clock: FakeClock) -> None:
    platform = Platform(issuer="https://login.acme.com")
    platform.discovery["backchannel_authentication_endpoint"] = (
        "https://acme.authfi.io/apis/regional/v1/ai/auth/backchannel-authorize"
    )
    af = AuthFI(platform.issuer, http_client=platform.client(), clock=clock)
    verifier = af.verifier(audience=["https://acme.authfi.io/mcp"])  # the API origin is not known yet
    with pytest.raises(ConfigurationError, match="resource=<your API's client id>"):
        verifier.verify(platform.access_token(aud="https://acme.authfi.io/mcp"))


async def test_the_async_verifier_refuses_an_audience_under_a_learned_api_origin(clock: FakeClock) -> None:
    platform = Platform(issuer="https://login.acme.com")
    platform.discovery["backchannel_authentication_endpoint"] = (
        "https://acme.authfi.io/apis/regional/v1/ai/auth/backchannel-authorize"
    )
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client(), clock=clock) as aaf:
        verifier = aaf.verifier(audience=["https://acme.authfi.io/mcp"])
        with pytest.raises(ConfigurationError, match="resource=<your API's client id>"):
            await verifier.verify(platform.access_token(aud="https://acme.authfi.io/mcp"))


@pytest.mark.parametrize("pin", ["", "   "])
def test_a_blank_tenant_pin_is_an_error_not_no_pin(af: AuthFI, aaf: AsyncAuthFI, pin: str) -> None:
    """``os.environ.get("AUTHFI_TENANT_ID", "")`` must not quietly switch the tenant check off."""
    with pytest.raises(ConfigurationError, match="expected_tenant_id must not be empty"):
        af.verifier(audience=AUD, expected_tenant_id=pin)
    with pytest.raises(ConfigurationError, match="expected_tenant_id must not be empty"):
        aaf.verifier(audience=AUD, expected_tenant_id=pin)
    assert af.verifier(audience=AUD).expected_tenant_id is None


@pytest.mark.parametrize("audience", [[], [""], "api_orders"])
def test_the_audience_is_required(af: AuthFI, audience: Any) -> None:
    with pytest.raises(ConfigurationError):
        af.verifier(audience=audience)


# --- the request form ------------------------------------------------------------------------


def test_verify_request_reads_either_scheme_in_any_case(af: AuthFI, platform: Platform) -> None:
    verifier = af.verifier(audience=AUD)
    raw = platform.access_token()
    assert verifier.verify_request("Bearer " + raw).principal.id == "usr_1"
    assert verifier.verify_request("bearer  " + raw).principal.id == "usr_1"
    for header in (None, "", "Basic dXNlcjpwdw==", "Bearer", "Bearer   "):
        assert code_of(verifier.verify_request, header) == "token_missing"


def test_a_dpop_scheme_is_refused_until_proofs_are_checked(af: AuthFI, platform: Platform) -> None:
    verifier = af.verifier(audience=AUD)
    bound = platform.keys.sign(claims(platform, cnf={"jkt": "x"}))
    unbound = platform.access_token()
    kwargs = {"dpop": "proof", "method": "GET", "url": "https://api.example.com/x"}
    assert code_of(verifier.verify_request, "DPoP " + bound, **kwargs) == "token_dpop_unsupported"
    assert code_of(verifier.verify_request, "DPoP " + unbound, **kwargs) == "token_dpop_unbound"


# --- requirements ----------------------------------------------------------------------------


def test_requirements_both_ways(af: AuthFI, platform: Platform) -> None:
    token = af.verifier(audience=AUD).verify(platform.access_token(permissions=["a", "b"], roles=["r1"], scope="s1 s2"))
    token.require_permission("a", "b")
    with pytest.raises(InsufficientPermission) as caught:
        token.require_permission("a", "c")
    assert caught.value.required == ("a", "c")
    assert caught.value.status == 403
    token.require_any_permission("c", "b")
    with pytest.raises(InsufficientPermission):
        token.require_any_permission("c", "d")
    token.require_role("r0", "r1")
    with pytest.raises(InsufficientPermission):
        token.require_role("r0")
    token.require_scope("s1", "s2")
    with pytest.raises(InsufficientScope):
        token.require_scope("s1", "s3")
    with pytest.raises(ConfigurationError):
        token.require_permission()


def test_require_acr_and_max_age(af: AuthFI, platform: Platform) -> None:
    auth_time = int(time.time()) - 600
    token = af.verifier(audience=AUD).verify(platform.access_token(acr="urn:authfi:acr:mfa", auth_time=auth_time))
    token.require_acr("urn:authfi:acr:mfa", "urn:authfi:acr:passkey")
    with pytest.raises(InsufficientAuthentication) as caught:
        token.require_acr("urn:authfi:acr:passkey")
    assert caught.value.acr_values == ("urn:authfi:acr:passkey",)
    token.require_acr("urn:authfi:acr:mfa", max_age=900)
    with pytest.raises(InsufficientAuthentication):
        token.require_acr("urn:authfi:acr:mfa", max_age=300)
    later = datetime.now(UTC) + timedelta(hours=1)
    with pytest.raises(InsufficientAuthentication):
        token.require_acr("urn:authfi:acr:mfa", max_age=900, now=later)


# --- several projects ------------------------------------------------------------------------


def test_several_issuers_each_pinned_to_its_tenant(clock: FakeClock) -> None:
    acme, globex = Platform("https://acme.authfi.io"), Platform("https://globex.authfi.io")
    acme_af = AuthFI(acme.issuer, http_client=acme.client(), clock=clock)
    globex_af = AuthFI(globex.issuer, http_client=globex.client(), clock=clock)
    other_tenant = "9a1d0000-0000-0000-0000-000000000000"
    verifier = MultiIssuerVerifier(
        {
            acme.issuer: (acme_af.verifier(audience=AUD), TENANT),
            globex.issuer: (globex_af.verifier(audience=AUD), other_tenant),
        }
    )
    assert verifier.verify(acme.access_token()).issuer == acme.issuer
    assert verifier.verify(globex.access_token(tenant_id=other_tenant)).issuer == globex.issuer
    # A token signed under a trusted issuer's key but naming another tenant.
    assert code_of(verifier.verify, globex.access_token()) == "token_tenant_not_issuers"
    stranger = Platform("https://stranger.authfi.io")
    assert code_of(verifier.verify, stranger.access_token()) == "token_issuer_not_accepted"
    assert verifier.verify_request("Bearer " + acme.access_token()).issuer == acme.issuer


def test_multi_issuer_construction_is_checked(af: AuthFI, platform: Platform) -> None:
    v = af.verifier(audience=AUD)
    with pytest.raises(ConfigurationError, match="tenant"):
        MultiIssuerVerifier({platform.issuer: (v, "")})
    with pytest.raises(ConfigurationError, match="verifies"):
        MultiIssuerVerifier({"https://other.authfi.io": (v, TENANT)})
    with pytest.raises(ConfigurationError, match="another tenant"):
        MultiIssuerVerifier({platform.issuer: (af.verifier(audience=AUD, expected_tenant_id="x"), TENANT)})
    with pytest.raises(ConfigurationError):
        MultiIssuerVerifier({})
    with pytest.raises(ConfigurationError):
        MultiIssuerVerifier({platform.issuer: v})  # type: ignore[dict-item]


# --- async parity ----------------------------------------------------------------------------


async def test_the_async_verifier(aaf: AsyncAuthFI, platform: Platform) -> None:
    verifier = aaf.verifier(audience=AUD)
    token = await verifier.verify(platform.access_token(alg="EdDSA", sub="agt_1", client_id="agt_1"))
    assert token.token_kind == "agent"
    assert (await verifier.verify_request("Bearer " + platform.access_token())).token_kind == "person"
    with pytest.raises(InvalidToken) as caught:
        await verifier.verify(platform.keys.sign(claims(platform), typ="JWT"))
    assert caught.value.code == "token_not_access_token"
    with pytest.raises(InvalidToken) as missing:
        await verifier.verify_request(None)
    assert missing.value.code == "token_missing"
    multi = AsyncMultiIssuerVerifier({platform.issuer: (verifier, TENANT)})
    assert (await multi.verify(platform.access_token())).tenant_id == TENANT
    assert (await multi.verify_request("Bearer " + platform.access_token())).tenant_id == TENANT
    with pytest.raises(InvalidToken):
        await verifier.verify_request("DPoP " + platform.access_token())


def test_access_token_from_claims_with_odd_values() -> None:
    token = AccessToken.from_claims({"sub": "app:svc", "exp": "x", "scope": ["a", 1], "custom": "nope"})
    assert token.principal == Principal("app", "svc")
    assert token.scopes == frozenset({"a"})
    assert token.org_id is None


def test_verifier_repr_and_leeway_bounds(af: AuthFI) -> None:
    assert "api_orders" in repr(af.verifier(audience=AUD))
    with pytest.raises(ConfigurationError):
        af.verifier(audience=AUD, leeway=500)
