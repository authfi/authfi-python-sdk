"""SAML sign-in for a standard app: the project's metadata, the redirect, and every refusal the assertion
must meet (unsigned, wrong signer, wrong audience, wrong recipient, wrong request, expired, replayed, wrapped)."""

from __future__ import annotations

import time
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from authfi import AsyncAuthFI, AuthFI, ConfigurationError, DiscoveryError
from authfi.saml import (
    InMemoryReplayStore,
    PendingSamlSignIn,
    SamlAssertionRefused,
    SamlIdp,
    SamlServiceProvider,
    parse_idp_metadata,
)
from tests.saml_idp import ACS, ENTITY, SP_ENTITY, FakeIdp, Signer
from tests.support import Platform


@pytest.fixture(scope="module")
def idp() -> FakeIdp:
    return FakeIdp()


def provider(idp: FakeIdp, **options: object) -> SamlServiceProvider:
    parsed = parse_idp_metadata(idp.metadata(), expected_entity_id=ENTITY)
    return SamlServiceProvider(parsed, sp_entity_id=SP_ENTITY, acs_url=ACS, **options)  # type: ignore[arg-type]


def refused(sp: SamlServiceProvider, response: str, pending: PendingSamlSignIn | None, code: str) -> None:
    with pytest.raises(SamlAssertionRefused) as caught:
        sp.finish_sign_in(response, pending)
    assert caught.value.code == code
    assert caught.value.status == 401


# --- the happy path ---------------------------------------------------------------------------------


def test_start_sign_in_redirects_to_the_project_with_a_request_id(idp: FakeIdp) -> None:
    pending = provider(idp).start_sign_in(relay_state="/dashboard")
    url = urlsplit(pending.url)
    assert f"{url.scheme}://{url.netloc}{url.path}" == idp.sso_url
    query = parse_qs(url.query)
    assert "SAMLRequest" in query
    assert query["RelayState"] == ["/dashboard"]
    assert pending.request_id and pending.relay_state == "/dashboard"


def test_a_signed_assertion_for_this_app_signs_the_person_in(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in(relay_state="/dashboard")
    now = time.time()
    signed_in = sp.finish_sign_in(idp.response(now=now, in_response_to=pending.request_id), pending)
    user = signed_in.user
    assert user.sub == "ada@example.com"
    assert user.email == "ada@example.com"
    assert user.email_verified is None
    assert user.name == "Ada L"
    assert user.groups == ("staff",)
    assert user.roles == ("admin",)
    assert user.acr == "urn:authfi:acr:pwd"
    assert user.amr == ("saml",)
    assert user.auth_time is not None and abs(user.auth_time - (now - 5)) < 2
    assert signed_in.session_index == "_s1"
    assert signed_in.relay_state == "/dashboard"
    assert signed_in.assertion_id.startswith("_a")


def test_the_user_fields_come_from_the_attribute_names_given(idp: FakeIdp) -> None:
    sp = provider(idp, attributes={"email": ("work_email",), "groups": ("teams",)})
    pending = sp.start_sign_in()
    attrs = {"work_email": ["a@corp.example"], "teams": ["x", "y"]}
    response = idp.response(
        now=time.time(),
        in_response_to=pending.request_id,
        attributes=attrs,
        name_id="u1",
        name_format="urn:x:persistent",
    )
    user = sp.finish_sign_in(response, pending).user
    assert (user.sub, user.email, user.groups) == ("u1", "a@corp.example", ("x", "y"))


def test_a_persistent_name_id_is_not_taken_for_an_email(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(
        now=time.time(),
        in_response_to=pending.request_id,
        name_id="user@looks.like",
        name_format="urn:x:persistent",
        attributes={},
    )
    user = sp.finish_sign_in(response, pending).user
    assert user.sub == "user@looks.like" and user.email is None


# --- refusals ---------------------------------------------------------------------------------------


def test_an_unsigned_assertion_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, sign=None)
    refused(sp, response, pending, "saml_signature_missing")


def test_a_signed_response_around_an_unsigned_assertion_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, sign="response")
    refused(sp, response, pending, "saml_signature_missing")


def test_an_assertion_signed_by_a_key_not_in_the_metadata_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, signer=Signer.make("attacker"))
    refused(sp, response, pending, "saml_signature_invalid")


def test_an_assertion_changed_after_signing_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, tamper=True)
    refused(sp, response, pending, "saml_signature_invalid")


def test_a_second_unsigned_assertion_beside_a_signed_one_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, wrap=True)
    with pytest.raises(SamlAssertionRefused) as caught:
        sp.finish_sign_in(response, pending)
    assert caught.value.code == "saml_assertion_count"


def test_an_assertion_for_another_audience_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, audience="https://other.example/saml")
    refused(sp, response, pending, "saml_audience_mismatch")


def test_an_assertion_for_another_recipient_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    other = "https://evil.example/saml/acs"
    response = idp.response(now=time.time(), in_response_to=pending.request_id, recipient=other, destination=ACS)
    refused(sp, response, pending, "saml_subject_confirmation_invalid")


def test_a_response_for_another_destination_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, destination="https://evil.example/x")
    refused(sp, response, pending, "saml_destination_mismatch")


def test_an_assertion_from_another_issuer_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(
        now=time.time(), in_response_to=pending.request_id, issuer="https://evil.authfi.io/saml/metadata"
    )
    refused(sp, response, pending, "saml_issuer_mismatch")


def test_an_assertion_answering_another_request_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to="_someone_elses_request")
    refused(sp, response, pending, "saml_in_response_to_mismatch")


def test_an_expired_assertion_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    now = time.time()
    response = idp.response(
        now=now - 7200, in_response_to=pending.request_id, not_before=now - 7300, not_on_or_after=now - 7000
    )
    refused(sp, response, pending, "saml_assertion_expired")


def test_an_assertion_expired_inside_the_librarys_own_drift_is_refused_by_the_leeway(idp: FakeIdp) -> None:
    sp = provider(idp, leeway=10)
    pending = sp.start_sign_in()
    now = time.time()
    response = idp.response(
        now=now - 400,
        in_response_to=pending.request_id,
        not_before=now - 500,
        not_on_or_after=now - 120,
        confirm_not_on_or_after=now + 300,
    )
    refused(sp, response, pending, "saml_assertion_expired")


def test_an_assertion_not_yet_valid_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    now = time.time()
    response = idp.response(
        now=now, in_response_to=pending.request_id, not_before=now + 3600, not_on_or_after=now + 7200
    )
    refused(sp, response, pending, "saml_assertion_not_yet_valid")


def test_an_assertion_not_yet_valid_inside_the_librarys_own_drift_is_refused_by_the_leeway(idp: FakeIdp) -> None:
    sp = provider(idp, leeway=10)
    pending = sp.start_sign_in()
    now = time.time()
    response = idp.response(now=now, in_response_to=pending.request_id, not_before=now + 120, not_on_or_after=now + 600)
    refused(sp, response, pending, "saml_assertion_not_yet_valid")


def test_an_sha1_signature_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, sha1=True)
    with pytest.raises(SamlAssertionRefused) as caught:
        sp.finish_sign_in(response, pending)
    assert caught.value.code in ("saml_algorithm_rejected", "saml_signature_invalid")


def test_an_assertion_is_accepted_once(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id)
    sp.finish_sign_in(response, pending)
    refused(sp, response, pending, "saml_assertion_replayed")


def test_a_second_assertion_for_one_request_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    now = time.time()
    sp.finish_sign_in(idp.response(now=now, in_response_to=pending.request_id), pending)
    refused(sp, idp.response(now=now, in_response_to=pending.request_id), pending, "saml_assertion_replayed")


def test_a_replay_is_refused_by_a_second_provider_sharing_the_store(idp: FakeIdp) -> None:
    store = InMemoryReplayStore()
    first, second = provider(idp, replay_store=store), provider(idp, replay_store=store)
    pending = first.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id)
    first.finish_sign_in(response, pending)
    refused(second, response, pending, "saml_assertion_replayed")


def test_a_failed_response_is_not_remembered_as_used(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    now = time.time()
    forged = idp.response(now=now, in_response_to=pending.request_id, assertion_id="_same", signer=Signer.make("x"))
    refused(sp, forged, pending, "saml_signature_invalid")
    real = idp.response(now=now, in_response_to=pending.request_id, assertion_id="_same")
    assert sp.finish_sign_in(real, pending).assertion_id == "_same"


def test_a_failed_status_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(
        now=time.time(), in_response_to=pending.request_id, status="urn:oasis:names:tc:SAML:2.0:status:Responder"
    )
    refused(sp, response, pending, "saml_status_not_success")


def test_a_response_with_a_dtd_is_refused(idp: FakeIdp) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    response = idp.response(now=time.time(), in_response_to=pending.request_id, doctype=True)
    refused(sp, response, pending, "saml_xml_invalid")


@pytest.mark.parametrize(
    "junk",
    [
        pytest.param("", id="empty"),
        pytest.param("!!not base64!!", id="not-base64"),
        pytest.param("bm90IHhtbA==", id="not-xml"),
        pytest.param("A" * ((1 << 20) + 4), id="too-large"),
    ],
)
def test_a_response_that_is_not_a_saml_response_is_refused(idp: FakeIdp, junk: str) -> None:
    sp = provider(idp)
    pending = sp.start_sign_in()
    with pytest.raises(SamlAssertionRefused) as caught:
        sp.finish_sign_in(junk, pending)
    assert caught.value.code.startswith("saml_")


def test_a_response_with_no_started_sign_in_is_refused_by_default(idp: FakeIdp) -> None:
    sp = provider(idp)
    refused(sp, idp.response(now=time.time(), in_response_to=None), None, "saml_unsolicited")


def test_a_pending_sign_in_that_is_too_old_is_refused(idp: FakeIdp) -> None:
    now = [time.time()]
    sp = provider(idp, clock=lambda: now[0])
    pending = sp.start_sign_in()
    response = idp.response(now=now[0], in_response_to=pending.request_id)
    now[0] += 601
    refused(sp, response, pending, "saml_pending_expired")


# --- IdP-initiated ----------------------------------------------------------------------------------


def test_an_idp_initiated_response_is_accepted_only_when_allowed(idp: FakeIdp) -> None:
    sp = provider(idp, allow_idp_initiated=True)
    signed_in = sp.finish_sign_in(idp.response(now=time.time(), in_response_to=None), None)
    assert signed_in.user.sub == "ada@example.com" and signed_in.relay_state is None


def test_an_idp_initiated_response_is_still_signed_and_single_use(idp: FakeIdp) -> None:
    sp = provider(idp, allow_idp_initiated=True)
    refused(sp, idp.response(now=time.time(), in_response_to=None, sign=None), None, "saml_signature_missing")
    response = idp.response(now=time.time(), in_response_to=None)
    sp.finish_sign_in(response, None)
    refused(sp, response, None, "saml_assertion_replayed")


def test_a_response_answering_a_request_cannot_be_used_without_that_request(idp: FakeIdp) -> None:
    sp = provider(idp, allow_idp_initiated=True)
    refused(sp, idp.response(now=time.time(), in_response_to="_r1"), None, "saml_in_response_to_mismatch")


# --- the pending sign-in ----------------------------------------------------------------------------


def test_a_pending_sign_in_round_trips(idp: FakeIdp) -> None:
    pending = provider(idp).start_sign_in(relay_state="/x")
    assert PendingSamlSignIn.load(pending.dump()) == pending


@pytest.mark.parametrize(
    "text",
    [
        "",
        "nope",
        "[]",
        "{}",
        '{"id":"","rs":null,"exp":1,"url":"u"}',
        '{"id":"a","rs":3,"exp":1,"url":"u"}',
        '{"id":"a","rs":null,"exp":true,"url":"u"}',
        '{"id":"a","rs":null,"exp":1,"url":5}',
    ],
)
def test_a_pending_sign_in_that_is_not_one_is_refused(text: str) -> None:
    with pytest.raises(SamlAssertionRefused) as caught:
        PendingSamlSignIn.load(text)
    assert caught.value.code == "saml_pending_malformed"


def test_the_replay_store_refuses_rather_than_forget_a_live_entry() -> None:
    store = InMemoryReplayStore(max_entries=2)
    assert store.remember("a", 100, 0) and store.remember("b", 100, 0)
    with pytest.raises(SamlAssertionRefused) as caught:
        store.remember("c", 100, 0)
    assert caught.value.code == "saml_replay_store_full"
    assert store.remember("c", 300, 150)  # the old ones expired
    assert not store.remember("c", 300, 160)
    assert store.remember("a", 400, 200)  # an expired entry may be used again


# --- configuration ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {"acs_url": "http://app.example.com/acs"},
        {"sp_entity_id": " "},
        {"leeway": 301},
        {"leeway": True},
    ],
)
def test_a_bad_configuration_is_refused(idp: FakeIdp, options: dict[str, object]) -> None:
    parsed = parse_idp_metadata(idp.metadata(), expected_entity_id=ENTITY)
    args: dict[str, object] = {"sp_entity_id": SP_ENTITY, "acs_url": ACS, **options}
    with pytest.raises(ConfigurationError):
        SamlServiceProvider(parsed, **args)  # type: ignore[arg-type]


def test_a_loopback_acs_over_http_is_allowed_for_development(idp: FakeIdp) -> None:
    parsed = parse_idp_metadata(idp.metadata(), expected_entity_id=ENTITY)
    sp = SamlServiceProvider(parsed, sp_entity_id=SP_ENTITY, acs_url="http://localhost:8000/saml/acs")
    assert sp.start_sign_in().request_id


# --- metadata ---------------------------------------------------------------------------------------


def test_only_the_first_certificate_of_a_key_descriptor_may_sign(idp: FakeIdp) -> None:
    other = Signer.make("custody")
    parsed = parse_idp_metadata(idp.metadata(extra_certs=(other.cert_b64,)), expected_entity_id=ENTITY)
    assert parsed.signing_certs == (idp.signer.cert_b64,)
    sp = SamlServiceProvider(parsed, sp_entity_id=SP_ENTITY, acs_url=ACS)
    pending = sp.start_sign_in()
    forged = idp.response(now=time.time(), in_response_to=pending.request_id, signer=other)
    refused(sp, forged, pending, "saml_signature_invalid")


@pytest.mark.parametrize(
    "xml",
    [
        b"not xml",
        b"<a/>",
        b'<md:EntityDescriptor xmlns:md="urn:oasis:names:tc:SAML:2.0:metadata" entityID="https://acme.authfi.io/saml/metadata"/>',
    ],
)
def test_metadata_that_is_not_an_idp_description_is_refused(xml: bytes) -> None:
    with pytest.raises(DiscoveryError):
        parse_idp_metadata(xml, expected_entity_id=ENTITY)


def test_metadata_of_another_entity_is_refused(idp: FakeIdp) -> None:
    with pytest.raises(DiscoveryError, match="not this project's"):
        parse_idp_metadata(idp.metadata(entity_id="https://evil.authfi.io/saml/metadata"), expected_entity_id=ENTITY)


def test_metadata_with_a_dtd_is_refused(idp: FakeIdp) -> None:
    with pytest.raises(DiscoveryError, match="DTD"):
        parse_idp_metadata(idp.metadata(doctype=True), expected_entity_id=ENTITY)


def test_metadata_without_a_signing_key_or_sign_on_url_is_refused(idp: FakeIdp) -> None:
    good = idp.metadata().decode()
    no_key = good.replace('use="signing"', 'use="encryption"').encode()
    no_sso = good.replace("HTTP-Redirect", "HTTP-Artifact").replace("/saml/sso", "ftp://x").encode()
    with pytest.raises(DiscoveryError, match="no signing certificate"):
        parse_idp_metadata(no_key, expected_entity_id=ENTITY)
    with pytest.raises(DiscoveryError, match="no sign-on URL"):
        parse_idp_metadata(no_sso, expected_entity_id=ENTITY)


def test_the_idp_is_exposed(idp: FakeIdp) -> None:
    assert isinstance(provider(idp).idp, SamlIdp) and provider(idp).idp.entity_id == ENTITY


# --- reading the project's metadata -----------------------------------------------------------------


def test_from_project_reads_the_metadata_from_the_projects_saml_route(idp: FakeIdp, platform: Platform) -> None:
    platform.route("GET", "/saml/metadata", httpx.Response(200, content=idp.metadata()))
    with AuthFI(platform.issuer, http_client=platform.client()) as af:
        sp = SamlServiceProvider.from_project(af, sp_entity_id=SP_ENTITY, acs_url=ACS)
    assert sp.idp.entity_id == ENTITY and len(platform.calls("GET", "/saml/metadata")) == 1
    pending = sp.start_sign_in()
    assert sp.finish_sign_in(idp.response(now=time.time(), in_response_to=pending.request_id), pending).user.sub


def test_from_project_refuses_metadata_that_is_not_the_projects(idp: FakeIdp, platform: Platform) -> None:
    other = idp.metadata(entity_id="https://evil.authfi.io/saml/metadata")
    platform.route("GET", "/saml/metadata", httpx.Response(200, content=other))
    with AuthFI(platform.issuer, http_client=platform.client()) as af, pytest.raises(DiscoveryError):
        SamlServiceProvider.from_project(af, sp_entity_id=SP_ENTITY, acs_url=ACS)


def test_from_project_reports_a_metadata_route_that_fails(platform: Platform) -> None:
    platform.route("GET", "/saml/metadata", httpx.Response(404))
    with AuthFI(platform.issuer, http_client=platform.client()) as af, pytest.raises(DiscoveryError) as caught:
        SamlServiceProvider.from_project(af, sp_entity_id=SP_ENTITY, acs_url=ACS)
    assert caught.value.status == 404


async def test_afrom_project_reads_the_metadata_too(idp: FakeIdp, platform: Platform) -> None:
    platform.route("GET", "/saml/metadata", httpx.Response(200, content=idp.metadata()))
    async with AsyncAuthFI(platform.issuer, http_client=platform.async_client()) as af:
        sp = await SamlServiceProvider.afrom_project(af, sp_entity_id=SP_ENTITY, acs_url=ACS, allow_idp_initiated=True)
    assert sp.finish_sign_in(idp.response(now=time.time(), in_response_to=None), None).user.sub == "ada@example.com"
