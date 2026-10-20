from __future__ import annotations

import uuid

import pytest
from hypothesis import given
from hypothesis import strategies as st

from authfi import ConfigurationError, idempotency_key, request_id
from authfi._protocol.ids import inject_traceparent, is_valid_request_id, trace_id_of
from authfi._protocol.urls import is_loopback, is_under, origin_of, validate_issuer, validate_origin


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://acme.authfi.io", "https://acme.authfi.io"),
        ("https://ACME.authfi.io", "https://acme.authfi.io"),
        ("https://acme.authfi.io:443", "https://acme.authfi.io"),
        ("https://id.example.com:8443/tenant", "https://id.example.com:8443/tenant"),
        ("http://localhost:8080", "http://localhost:8080"),
        ("http://127.0.0.1:9000", "http://127.0.0.1:9000"),
        ("http://[::1]:9000", "http://[::1]:9000"),
    ],
)
def test_an_issuer_is_accepted_and_normalised(value: str, expected: str) -> None:
    assert validate_issuer(value) == expected


@pytest.mark.parametrize(
    ("value", "says"),
    [
        ("", "required"),
        ("acme", "https://acme.authfi.io"),
        ("http://acme.authfi.io", "https"),
        ("https://acme.authfi.io/", "trailing slash"),
        ("https://acme.authfi.io?x=1", "query"),
        ("https://acme.authfi.io#f", "fragment"),
        ("https://user:pw@acme.authfi.io", "credentials"),
        ("ftp://acme.authfi.io", "https"),
    ],
)
def test_an_issuer_is_refused_naming_the_fix(value: str, says: str) -> None:
    with pytest.raises(ConfigurationError, match=says):
        validate_issuer(value)


def test_loopback() -> None:
    assert is_loopback("localhost")
    assert is_loopback("127.0.0.5")
    assert is_loopback("[::1]")
    assert not is_loopback("10.0.0.1")
    assert not is_loopback("localhost.example.com")
    assert not is_loopback(None)


def test_origins() -> None:
    assert validate_origin("https://API.example.com:443") == "https://api.example.com"
    assert validate_origin("https://api.example.com/") == "https://api.example.com"
    for bad in ("https://api.example.com/v1", "http://api.example.com", "api.example.com", "https://x?y=1"):
        with pytest.raises(ConfigurationError):
            validate_origin(bad)
    assert origin_of("https://a.example.com:8443/x?y") == "https://a.example.com:8443"


def test_is_under() -> None:
    assert is_under("https://acme.authfi.io/mcp", "https://acme.authfi.io")
    assert is_under("https://acme.authfi.io/mcp/", "https://acme.authfi.io/mcp")
    assert not is_under("https://acme.authfi.io/mcpx", "https://acme.authfi.io/mcp")
    assert not is_under("https://evil.example/mcp", "https://acme.authfi.io")


@given(st.integers(min_value=0, max_value=50))
def test_request_ids_are_always_in_the_gateways_alphabet(n: int) -> None:
    rid = request_id() if n >= 0 else ""
    assert rid.startswith("afr-")
    assert len(rid) == 30
    assert is_valid_request_id(rid)
    assert "_" not in rid


@given(st.text(max_size=200))
def test_request_id_validation_matches_the_gateway_rule(value: str) -> None:
    import re

    assert is_valid_request_id(value) == bool(re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", value))


def test_request_ids_sort_by_time() -> None:
    first = request_id()
    later = [request_id() for _ in range(5)]
    assert all(first[:14] <= rid[:14] for rid in later)


def test_idempotency_keys_are_uuid7() -> None:
    keys = {idempotency_key() for _ in range(100)}
    assert len(keys) == 100
    for key in keys:
        parsed = uuid.UUID(key)
        assert parsed.version == 7
        assert parsed.variant == uuid.RFC_4122


def test_traceparent_is_fresh_and_unsampled_without_opentelemetry() -> None:
    headers: dict[str, str] = {}
    inject_traceparent(headers)
    value = headers["traceparent"]
    assert value.startswith("00-") and value.endswith("-00")
    assert trace_id_of(value) is not None


def test_a_callers_traceparent_is_kept() -> None:
    mine = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"
    headers = {"Traceparent": mine}
    inject_traceparent(headers)
    assert headers == {"Traceparent": mine}
    assert trace_id_of(mine) == "0af7651916cd43dd8448eb211c80319c"
    assert trace_id_of("nonsense") is None
    assert trace_id_of(None) is None
