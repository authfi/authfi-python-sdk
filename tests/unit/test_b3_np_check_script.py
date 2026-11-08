"""``scripts/b3_np_check.py`` against the fake gateway: the legs it reports are the ones that happened.

The script is what is rerun on np once identity asks for a nonce (B3, 14 Nov), so it is tested here with a gateway
that does ask: the nonce leg must read "asked and answered" for the SDK's own mints, and a refusal that is only a
nonce challenge must not pass as the property it names.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from authfi import AgentKey, _dpop
from tests.fake_agent_gateway import AgentGateway

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "b3_np_check.py"


def _script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("b3_np_check", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def _fresh_nonces() -> None:
    _dpop.nonces._nonces.clear()  # pyright: ignore[reportPrivateUsage]


@pytest.fixture
def key_file(tmp_path: Path) -> tuple[Path, str]:
    path = tmp_path / "agent.key"
    AgentKey.generate(kid="k1").write(path)
    return path, "k1"


def test_every_leg_behaves_and_the_nonce_leg_reads_asked_and_answered(key_file: tuple[Path, str]) -> None:
    path, kid = key_file
    with AgentGateway(nonce=True) as gateway:
        result = _script().check(gateway.issuer, "agt_b3script", str(path), kid, expect_nonce=True)
    legs = result["legs"]
    assert result["ok"], legs
    assert legs["nonce"]["got"] == "asked and answered"
    assert legs["replay"]["got"] == "200 then 401"
    for name in ("htu", "htm", "ath"):
        assert "use_dpop_nonce" not in legs[name]["reason"] and legs[name]["got"] == "401"


def test_expecting_a_nonce_fails_when_identity_asks_for_none(key_file: tuple[Path, str]) -> None:
    path, kid = key_file
    with AgentGateway(nonce=False) as gateway:
        informational = _script().check(gateway.issuer, "agt_b3script", str(path), kid)
        strict = _script().check(gateway.issuer, "agt_b3script", str(path), kid, expect_nonce=True)
    assert informational["ok"] and informational["legs"]["nonce"]["got"] == "not asked"
    assert not strict["ok"] and not strict["legs"]["nonce"]["ok"]


@pytest.mark.parametrize(
    "issuer",
    [
        "https://acme.np.authfi.io",  # a project that is not a probe
        "https://probe-c1.authfi.io",  # not np
        "http://probe-c1.np.authfi.io",  # not https
        "https://probe-c1.np.authfi.io.evil.example",
    ],
)
def test_only_a_probe_project_on_np_is_checked(issuer: str) -> None:
    with pytest.raises(SystemExit, match="refused"):
        _script()._np_only(issuer)  # pyright: ignore[reportPrivateUsage]
    assert _script()._np_only("https://probe-c1.np.authfi.io/") == "https://probe-c1.np.authfi.io"  # pyright: ignore[reportPrivateUsage]
