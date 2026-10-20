# pyright: reportDeprecated=false
"""0.1's AgentKey spellings: each still works and warns."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from authfi import AgentKey, AuthFIDeprecationWarning, ConfigurationError


def test_the_0_1_spellings_still_work_and_warn(tmp_path: Path) -> None:
    with pytest.warns(AuthFIDeprecationWarning):
        legacy = AgentKey.generate("k1")
    assert legacy.kid == "k1"
    assert legacy.alg == "EdDSA"
    with pytest.warns(AuthFIDeprecationWarning), pytest.raises(ConfigurationError, match="key_id is required"):
        AgentKey.generate("")
    with pytest.warns(AuthFIDeprecationWarning):
        assert legacy.key_id == "k1"
    with pytest.warns(AuthFIDeprecationWarning):
        pem = legacy.private_pem()
    with pytest.warns(AuthFIDeprecationWarning):
        assert AgentKey.from_pem(pem, "k1").kid == "k1"
    with pytest.warns(AuthFIDeprecationWarning):
        assertion = legacy.client_assertion("agt_1", "https://x/token")
    assert assertion.count(".") == 2
    if sys.platform != "win32":
        path = tmp_path / "k.pem"
        legacy.write(path)
        with pytest.warns(AuthFIDeprecationWarning):
            assert AgentKey.from_file(path, key_id="k1").kid == "k1"
