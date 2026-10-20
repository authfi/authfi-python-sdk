from __future__ import annotations

import tomllib
import typing

from authfi import _codes
from tests.support import FIXTURES

# The kernel's refusal codes at the pinned tag (constants/http.go and package dpop), copied here
# by hand: if the generator dropped one, this fails.
KERNEL = {
    "token_missing",
    "token_malformed",
    "token_signature_invalid",
    "token_key_unknown",
    "token_issuer_not_accepted",
    "token_expired",
    "token_audience_mismatch",
    "token_claims_invalid",
    "token_tenant_not_issuers",
    "token_not_access_token",
    "token_no_tenant",
    "token_issuer_not_project",
    "token_foreign_project",
    "token_signer_not_canonical",
    "token_account_mismatch",
    "token_inactive",
    "token_no_agent",
    "token_agent_inactive",
    "token_dpop_proof_required",
    "token_dpop_unbound",
    "token_dpop_unsupported",
    "token_dpop_proof_invalid",
    "dpop_proof_missing",
    "dpop_proof_replayed",
}


def test_the_generated_codes_hold_the_kernels() -> None:
    assert KERNEL <= set(_codes.KERNEL_TOKEN_CODES)
    assert _codes.KERNEL_TAG == "v0.58.0"


def test_known_code_lists_every_generated_code_once() -> None:
    known = typing.get_args(_codes.KnownCode)
    assert len(known) == len(set(known))
    assert set(_codes.KERNEL_TOKEN_CODES) <= set(known)
    assert set(_codes.SDK_TOKEN_CODES) <= set(known)
    assert "token_act_chain_unsupported" in known


def test_every_fixture_names_its_source() -> None:
    sources = tomllib.loads((FIXTURES / "SOURCES.toml").read_text(encoding="utf-8"))
    files = {p.relative_to(FIXTURES).as_posix() for p in FIXTURES.rglob("*.json")}
    assert files, "no fixtures found"
    for name in files:
        assert name in sources, f"{name} has no entry in fixtures/SOURCES.toml"
        entry = sources[name]
        for key in ("observed", "file", "commit"):
            assert entry.get(key), f"{name} has no {key} in fixtures/SOURCES.toml"
