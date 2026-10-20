from __future__ import annotations

import base64
import hashlib

import pytest

from authfi import (
    APIError,
    ConfigurationError,
    ConsentDenied,
    ConsentExpired,
    OAuthError,
    Secret,
    StepUpRequired,
    TemporarilyUnavailable,
    TransportError,
)
from authfi._protocol.errors import CIBA_GRANT, api_error, oauth_error
from authfi._protocol.http import HttpResponse, Retry
from authfi._protocol.oauth import (
    CLIENT_ASSERTION_TYPE,
    ClientAuth,
    authorization_code_request,
    client_credentials_request,
    parse_token_response,
    pkce_pair,
    refresh_request,
    revocation_request,
)

TOKEN = "https://acme.authfi.io/oauth/token"


def answer(status: int, body: bytes, headers: tuple[tuple[str, str], ...] = ()) -> HttpResponse:
    return HttpResponse(status, headers, body, TOKEN, "afr-1")


def test_basic_authentication_form_encodes_the_id_and_secret() -> None:
    auth = ClientAuth.secret_basic("app:1", Secret("p@ss word/+"))
    header = auth.attempt(TOKEN).headers["Authorization"]
    decoded = base64.b64decode(header.removeprefix("Basic ")).decode()
    assert decoded == "app%3A1:p%40ss+word%2F%2B"
    assert auth.attempt(TOKEN).form == {}


def test_post_none_and_private_key_jwt() -> None:
    assert ClientAuth.secret_post("app", Secret("s")).attempt(TOKEN).form == {"client_id": "app", "client_secret": "s"}
    assert ClientAuth.none("app").attempt(TOKEN).form == {"client_id": "app"}
    audiences: list[str] = []

    def sign(audience: str) -> str:
        audiences.append(audience)
        return f"assertion-{len(audiences)}"

    auth = ClientAuth.private_key_jwt("agt_1", sign)
    first, second = auth.attempt(TOKEN), auth.attempt(TOKEN)
    assert first.form["client_assertion_type"] == CLIENT_ASSERTION_TYPE
    assert first.form["client_assertion"] != second.form["client_assertion"]
    assert audiences == [TOKEN, TOKEN]


def test_client_auth_needs_an_id_and_a_secret_object() -> None:
    with pytest.raises(ConfigurationError):
        ClientAuth.none(" ")
    with pytest.raises(ConfigurationError, match="Secret"):
        ClientAuth.secret_basic("app", "plain")  # type: ignore[arg-type]
    with pytest.raises(ConfigurationError, match="empty"):
        ClientAuth.secret_post("app", Secret(""))
    assert "s3cret" not in repr(ClientAuth.secret_post("app", Secret("s3cret")))


def test_grants_and_their_retry_classes() -> None:
    auth = ClientAuth.secret_post("app", Secret("s"))
    cc = client_credentials_request(TOKEN, auth, resource="api_orders", scope=["a", "b"])
    assert cc.form == {"grant_type": "client_credentials", "resource": "api_orders", "scope": "a b"}
    assert cc.retry is Retry.SAFE
    code = authorization_code_request(TOKEN, auth, code="c", redirect_uri="https://app/cb", code_verifier=Secret("v"))
    assert code.retry is Retry.NEVER_AFTER_SEND
    assert code.form is not None and code.form["code_verifier"] == "v"
    refresh = refresh_request(TOKEN, auth, refresh_token=Secret("r"))
    assert refresh.retry is Retry.NEVER_AFTER_SEND
    revoke = revocation_request(TOKEN + "/revoke", auth, token=Secret("r"))
    assert revoke.retry is Retry.SAFE
    assert revoke.form == {"token": "r", "token_type_hint": "refresh_token"}


def test_a_scope_string_is_refused_so_it_is_not_split_into_letters() -> None:
    with pytest.raises(ConfigurationError, match="sequence"):
        client_credentials_request(TOKEN, ClientAuth.none("a"), scope="openid profile")


def test_pkce_is_s256_of_a_32_byte_verifier() -> None:
    verifier, challenge = pkce_pair()
    assert isinstance(verifier, Secret)
    assert len(verifier.reveal()) == 43
    expected = base64.urlsafe_b64encode(hashlib.sha256(verifier.reveal().encode()).digest()).rstrip(b"=").decode()
    assert challenge == expected


def test_a_token_answer_keeps_tokens_secret() -> None:
    body = (
        b'{"access_token":"AT","token_type":"Bearer","expires_in":"900","refresh_token":"RT",'
        b'"id_token":"IT","scope":"openid profile","capabilities":["x"]}'
    )
    parsed = parse_token_response(answer(200, body), "the token endpoint")
    assert parsed.access_token.reveal() == "AT"
    assert parsed.refresh_token is not None and parsed.refresh_token.reveal() == "RT"
    assert parsed.expires_in == 900
    assert parsed.scope == frozenset({"openid", "profile"})
    assert "AT" not in repr(parsed) and "RT" not in repr(parsed)
    assert "access_token" not in parsed.extra
    assert parsed.extra["capabilities"] == ["x"]


def test_an_answer_without_a_token_is_an_error_not_a_none_token() -> None:
    with pytest.raises(OAuthError, match="no access_token"):
        parse_token_response(answer(200, b'{"token_type":"Bearer"}'), "x")
    with pytest.raises(OAuthError):
        parse_token_response(answer(200, b"[1]"), "x")


def test_refusals_become_typed_errors() -> None:
    def err(status: int, body: bytes, grant: str | None = None) -> BaseException:
        return oauth_error(
            answer(status, body, (("X-Request-Id", "rid-9"), ("AuthFI-Run-Id", "run-1"))), "x", grant=grant
        )

    plain = err(400, b'{"error":"invalid_grant","error_description":"code used"}')
    assert type(plain) is OAuthError
    assert plain.code == "invalid_grant" and plain.description == "code used" and plain.status == 400
    assert plain.request_id == "rid-9" and plain.run_id == "run-1"
    step = err(401, b'{"error":"insufficient_user_authentication","error_description":"(acr: urn:authfi:acr:mfa)"}')
    assert isinstance(step, StepUpRequired)
    assert step.acr_values == ()  # never parsed from the prose
    assert isinstance(err(503, b'{"error":"temporarily_unavailable"}'), TemporarilyUnavailable)
    assert isinstance(err(400, b'{"error":"access_denied"}', CIBA_GRANT), ConsentDenied)
    assert isinstance(err(400, b'{"error":"expired_token"}', CIBA_GRANT), ConsentExpired)
    assert type(err(403, b'{"error":"access_denied"}')) is OAuthError
    no_code = err(500, b"<html>")
    assert isinstance(no_code, TransportError) and no_code.status == 500


def test_step_up_carries_the_configured_acr() -> None:
    step = oauth_error(
        answer(401, b'{"error":"insufficient_user_authentication"}'), "x", step_up_acr=["urn:authfi:acr:mfa"]
    )
    assert isinstance(step, StepUpRequired)
    assert step.acr_values == ("urn:authfi:acr:mfa",)


def test_api_errors_and_relayed_provider_envelopes() -> None:
    e = api_error(answer(403, b'{"error":"forbidden","error_description":"no"}'), "x")
    assert isinstance(e, APIError) and e.code == "forbidden"
    nested = api_error(answer(403, b'{"error":{"code":"EMODEL003","message":"not granted"}}'), "x")
    assert isinstance(nested, APIError) and nested.code == "EMODEL003" and nested.description == "not granted"
    assert isinstance(api_error(answer(502, b""), "x"), TransportError)
