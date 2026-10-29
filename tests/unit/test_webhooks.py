"""authfi.webhooks: the verifier against the signature the platform's dispatcher makes."""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import pytest

from authfi import ConfigurationError, Secret, WebhookVerificationError
from authfi.webhooks import EVENT_TYPES, SIGNATURE_HEADER, TIMESTAMP_HEADER, sign, verify

SECRET = "whsec_test_0123456789"
NOW = 1_800_000_000
BODY = json.dumps(
    {
        "id": "req_1",
        "event": "user.created",
        "timestamp": "2026-10-29T10:00:00Z",
        "data": {"actor": "user:a1", "subject": "user:b2", "resource_type": "user", "resource_id": "b2"},
    },
    separators=(",", ":"),
).encode()


def platform_signature(secret: str, timestamp: str, body: bytes) -> str:
    """hex(HMAC-SHA256(secret, timestamp + "." + body)), written out the way webhookdispatch.sign does."""
    return hmac.new(secret.encode(), timestamp.encode() + b"." + body, hashlib.sha256).hexdigest()


def delivery(body: bytes = BODY, *, secret: str = SECRET, at: int = NOW) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-AuthFI-Event": "user.created",
        "X-AuthFI-Delivery": "7d0e5a3e-0000-4000-8000-000000000001",
        "X-AuthFI-Timestamp": str(at),
        "X-AuthFI-Signature-256": "sha256=" + platform_signature(secret, str(at), body),
    }


def check(body: bytes = BODY, headers: Any = None, secret: Any = SECRET, **kwargs: Any) -> Any:
    return verify(body, delivery(body) if headers is None else headers, secret, now=lambda: float(NOW), **kwargs)


def refused(code: str, **kwargs: Any) -> WebhookVerificationError:
    with pytest.raises(WebhookVerificationError) as caught:
        check(**kwargs)
    assert caught.value.code == code
    return caught.value


def test_a_delivery_the_platform_signed_verifies_and_is_read() -> None:
    event = check()
    assert (event.type, event.id) == ("user.created", "req_1")
    assert (event.actor, event.subject) == ("user:a1", "user:b2")
    assert (event.resource_type, event.resource_id) == ("user", "b2")
    assert event.delivery_id == "7d0e5a3e-0000-4000-8000-000000000001"
    assert event.timestamp is not None and event.timestamp.year == 2026
    assert event.signed_at.timestamp() == NOW
    assert event.body["event"] == "user.created"


def test_the_signature_is_the_known_answer() -> None:
    # printf '1700000000.{}' | openssl dgst -sha256 -hmac s3cret
    expected = "97926816e98fbb41ccb1673225ff29a2f35369099990e1b1561651e7bd097ebf"
    assert sign("s3cret", b"{}", timestamp=1_700_000_000)[SIGNATURE_HEADER] == "sha256=" + expected
    headers = {SIGNATURE_HEADER: "sha256=" + expected, TIMESTAMP_HEADER: "1700000000"}
    with pytest.raises(WebhookVerificationError) as caught:  # a well-formed delivery whose body is not an event
        verify(b"{}", headers, "s3cret", now=lambda: 1_700_000_000.0)
    assert caught.value.code == "webhook_body_invalid"  # it got past the signature and the age


def test_sign_makes_what_verify_accepts_and_a_secret_object_works() -> None:
    headers = sign(Secret(SECRET), BODY, timestamp=NOW)
    assert set(headers) == {SIGNATURE_HEADER, TIMESTAMP_HEADER}
    assert verify(BODY, headers, Secret(SECRET), now=lambda: float(NOW)).type == "user.created"
    assert verify(BODY.decode(), headers, SECRET, now=lambda: float(NOW)).type == "user.created"


def test_sign_without_a_time_signs_now() -> None:
    assert verify(BODY, sign(SECRET, BODY), SECRET).type == "user.created"


def test_header_names_are_case_insensitive_and_a_list_of_pairs_works() -> None:
    lowered = {k.lower(): v for k, v in delivery().items()}
    assert check(headers=lowered).type == "user.created"
    assert check(headers=list(delivery().items())).type == "user.created"
    first_is_wrong = [("x-authfi-signature-256", "sha256=" + "0" * 64), *delivery().items()]
    refused("webhook_signature_mismatch", headers=first_is_wrong)  # the first value of a header is the one read


def test_a_changed_body_is_a_mismatch() -> None:
    altered = BODY.replace(b"user.created", b"user.deleted")
    refused("webhook_signature_mismatch", body=altered, headers=delivery())


def test_re_serialised_json_does_not_verify() -> None:
    pretty = json.dumps(json.loads(BODY), indent=1).encode()
    refused("webhook_signature_mismatch", body=pretty, headers=delivery())


def test_the_wrong_secret_is_a_mismatch() -> None:
    refused("webhook_signature_mismatch", headers=delivery(secret="other"))


def test_a_changed_timestamp_is_a_mismatch_because_the_signature_binds_it() -> None:
    headers = delivery()
    headers["X-AuthFI-Timestamp"] = str(NOW + 1)
    refused("webhook_signature_mismatch", headers=headers)


def test_a_rotation_accepts_either_secret() -> None:
    old = delivery(secret="old-secret")
    assert check(headers=old, secret=[SECRET, "old-secret"]).type == "user.created"
    assert check(headers=delivery(), secret=[Secret("old-secret"), Secret(SECRET)]).type == "user.created"
    refused("webhook_signature_mismatch", headers=old, secret=[SECRET, "another"])
    assert (
        check(headers=delivery(), secret=[SECRET, "another"]).type == "user.created"
    )  # the first of the list signed it


@pytest.mark.parametrize("offset", [301, -301, 10_000])
def test_a_delivery_outside_the_window_is_refused_either_way(offset: int) -> None:
    refused("webhook_timestamp_outside_tolerance", headers=delivery(at=NOW + offset))


@pytest.mark.parametrize("offset", [0, 300, -300])
def test_the_window_edges_pass(offset: int) -> None:
    assert check(headers=delivery(at=NOW + offset)).type == "user.created"


def test_the_window_is_the_callers() -> None:
    old = delivery(at=NOW - 1000)
    refused("webhook_timestamp_outside_tolerance", headers=old)
    assert check(headers=old, tolerance=1000).type == "user.created"


def test_a_forged_old_delivery_is_a_mismatch_not_too_old() -> None:
    refused("webhook_signature_mismatch", headers=delivery(secret="forger", at=NOW - 10_000))


def test_the_signature_header_is_required_and_must_be_a_digest() -> None:
    headers = delivery()
    del headers["X-AuthFI-Signature-256"]
    refused("webhook_signature_missing", headers=headers)
    refused("webhook_signature_missing", headers={**delivery(), SIGNATURE_HEADER: ""})
    for bad in ("abcd", "sha1=" + "0" * 64, "sha256=zz", "sha256=" + "0" * 63, "sha256=" + "g" * 64):
        refused("webhook_signature_malformed", headers={**delivery(), SIGNATURE_HEADER: bad})


def test_an_uppercase_hex_digest_is_the_same_digest() -> None:
    headers = delivery()
    headers[SIGNATURE_HEADER] = "sha256=" + platform_signature(SECRET, str(NOW), BODY).upper()
    assert check(headers=headers).type == "user.created"


def test_the_timestamp_header_is_required_and_must_be_seconds() -> None:
    headers = delivery()
    del headers["X-AuthFI-Timestamp"]
    refused("webhook_timestamp_missing", headers=headers)
    for bad in ("soon", "-1", "1.5", "1e9", "123456789012", "١٢٣"):
        refused("webhook_timestamp_invalid", headers={**delivery(), TIMESTAMP_HEADER: bad})


@pytest.mark.parametrize("body", [b"not json", b"[]", b'"x"', b"\xff\xfe", b"{}", b'{"event": ""}', b'{"event": 5}'])
def test_a_signed_body_that_is_not_an_event_is_refused(body: bytes) -> None:
    refused("webhook_body_invalid", body=body, headers=delivery(body))


def test_an_event_with_odd_fields_still_reads() -> None:
    body = json.dumps({"event": "some.future_event", "timestamp": "yesterday", "data": "x"}).encode()
    event = check(body, delivery(body))
    assert (event.type, event.id, event.timestamp, event.actor) == ("some.future_event", "", None, None)
    numeric = json.dumps({"event": "login.success", "timestamp": 1_800_000_000}).encode()
    assert check(numeric, delivery(numeric)).timestamp is None
    naive = json.dumps({"event": "login.failed", "timestamp": "2026-10-29T10:00:00", "data": {"actor": 5}}).encode()
    parsed = check(naive, delivery(naive))
    assert parsed.timestamp is not None and parsed.timestamp.tzinfo is not None
    assert parsed.actor is None
    without_delivery = {k: v for k, v in delivery(naive).items() if k != "X-AuthFI-Delivery"}
    assert check(naive, without_delivery).delivery_id is None


def test_the_body_must_be_bytes_or_text() -> None:
    with pytest.raises(ConfigurationError, match="raw request body"):
        verify({"event": "x"}, delivery(), SECRET)  # type: ignore[arg-type]
    assert verify(bytearray(BODY), delivery(), SECRET, now=lambda: float(NOW)).type == "user.created"
    assert verify(memoryview(BODY), delivery(), SECRET, now=lambda: float(NOW)).type == "user.created"


@pytest.mark.parametrize("secret", ["", Secret(""), [], [""]])
def test_an_empty_secret_is_a_configuration_error_not_a_refusal(secret: Any) -> None:
    with pytest.raises(ConfigurationError):
        verify(BODY, delivery(), secret)


@pytest.mark.parametrize("tolerance", [0, -1, float("nan")])
def test_a_tolerance_that_refuses_everything_is_a_configuration_error(tolerance: float) -> None:
    with pytest.raises(ConfigurationError, match="positive"):
        verify(BODY, delivery(), SECRET, tolerance=tolerance)


def test_the_event_list_is_the_platforms() -> None:
    assert set(EVENT_TYPES) == {
        "user.created",
        "user.updated",
        "user.deleted",
        "user.blocked",
        "user.mfa_enrolled",
        "group.member_added",
        "group.member_removed",
        "organization.created",
        "role.assigned",
        "session.revoked",
        "login.success",
        "login.failed",
    }
