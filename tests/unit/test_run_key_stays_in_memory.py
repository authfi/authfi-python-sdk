"""The run key is never on disk, in the environment, in a log, in an error report or on the wire.

A task is held by this process's run key, and handing it to another process moves the task, not the key. The test
runs the whole path (a task, a handover, a takeover, calls by the receiver, a refused takeover) with the process's
temporary, home and working directories pointed at one folder and logging at DEBUG, then looks for the private key in
every place it could have been left: the environment, every file under that folder, what was logged and printed,
what an error reporter would capture from the refusal, the handover's text, and every byte the gateway received.

The scan is proved first, as an instrument: the same key written to each of those places is found.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed25519

from authfi import AgentKey, AuthFI, Secret, TaskMoved, _dpop
from authfi.agents import AgentAuth
from tests.fake_agent_gateway import AgentGateway
from tests.fake_gateway import ALLOWED
from tests.leaks import captured
from tests.support import as_process

AGENT = "agt_nokey1"


def secrets_of(key: _dpop.DPoPKey) -> list[bytes]:
    """Every spelling of the private half that a careless write would leave: raw, hex, base64, base64url, DER, PEM."""
    private = key._key  # pyright: ignore[reportPrivateUsage]
    if isinstance(private, ed25519.Ed25519PrivateKey):
        raw = private.private_bytes_raw()
    else:
        assert isinstance(private, ec.EllipticCurvePrivateKey)
        raw = private.private_numbers().private_value.to_bytes(32, "big")
    der = private.private_bytes(
        serialization.Encoding.DER, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    pem = private.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )
    body = b"".join(pem.splitlines()[1:-1])
    spellings = [
        raw,
        der,
        binascii.hexlify(raw),
        binascii.hexlify(raw).upper(),
        base64.b64encode(raw),
        base64.urlsafe_b64encode(raw),
        base64.urlsafe_b64encode(raw).rstrip(b"="),
        base64.b64encode(raw).rstrip(b"="),
        body,
        b"PRIVATE KEY",
    ]
    return spellings


def found_in(needles: list[bytes], haystack: bytes) -> list[bytes]:
    return [n for n in needles if n in haystack]


class Places:
    """Everywhere a key could be left behind, read after the run."""

    def __init__(self, root: Path, caplog: pytest.LogCaptureFixture) -> None:
        self.root = root
        self.caplog = caplog
        self.extra: list[bytes] = []

    def environment(self) -> bytes:
        return "\n".join(f"{k}={v}" for k, v in os.environ.items()).encode()

    def files(self) -> bytes:
        return b"\n".join(p.read_bytes() + str(p).encode() for p in self.root.rglob("*") if p.is_file())

    def logs(self) -> bytes:
        return self.caplog.text.encode()

    def everything(self) -> dict[str, bytes]:
        return {
            "environment": self.environment(),
            "files": self.files(),
            "logs": self.logs(),
            "reports and wire": b"\n".join(self.extra),
        }


@pytest.fixture
def places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> Places:
    """A process whose temporary, home and working directories are one folder, logging everything."""
    root = tmp_path / "process"
    for name in ("tmp", "home", "cwd"):
        (root / name).mkdir(parents=True)
    for name in ("TMPDIR", "TMP", "TEMP"):
        monkeypatch.setenv(name, str(root / "tmp"))
    for name in (
        "HOME",
        "USERPROFILE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "APPDATA",
        "LOCALAPPDATA",
    ):
        monkeypatch.setenv(name, str(root / "home"))
    monkeypatch.setattr(tempfile, "tempdir", str(root / "tmp"))
    monkeypatch.chdir(root / "cwd")
    caplog.set_level(logging.DEBUG)
    return Places(root, caplog)


def person() -> Secret:
    return Secret(jwt.encode({"sub": "usr_1", "exp": int(time.time()) + 3600}, "k" * 32, algorithm="HS256"))


def report_of_refused_takeover(agent: AgentAuth, wire: str) -> str:
    """What an error reporter captures from a takeover the platform refuses, frame locals included.

    A frame of its own, so the report holds the SDK's frames and this one, and not the test's (which holds the keys).
    """
    with pytest.raises(TaskMoved) as refused:
        agent.take_over(wire)
    return captured(refused.value)


def test_the_scan_finds_a_key_in_the_environment_a_file_a_log_and_a_report(
    places: Places, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The instrument, proved: a key left in any place is found there, and a run that leaves none finds nothing."""
    key = _dpop.DPoPKey()
    needles = secrets_of(key)
    assert not any(found_in(needles, text) for text in places.everything().values())
    seed = base64.urlsafe_b64encode(needles[0]).rstrip(b"=").decode()

    monkeypatch.setenv("RUN_KEY", seed)
    assert found_in(needles, places.environment())
    monkeypatch.delenv("RUN_KEY")

    (places.root / "tmp" / "key.bin").write_bytes(needles[0])
    assert found_in(needles, places.files())
    (places.root / "tmp" / "key.bin").unlink()
    (places.root / "cwd" / "key.pem").write_text("-----BEGIN PRIVATE KEY-----")
    assert found_in(needles, places.files())

    logging.getLogger("authfi").error("signing with %s", seed)
    assert found_in(needles, places.logs())

    with pytest.raises(ValueError) as leaked:
        raise ValueError(seed)
    assert found_in(needles, captured(leaked.value).encode())


def test_a_run_that_makes_a_task_hands_it_over_and_takes_it_leaves_no_run_key_anywhere(
    places: Places, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sender_key = _dpop.run_key()
    receiver_key = _dpop.DPoPKey()
    needles = secrets_of(sender_key) + secrets_of(receiver_key)
    agent_key = AgentKey.generate(kid="k1")
    with AgentGateway() as gw, AuthFI(gw.issuer) as af:
        agent: AgentAuth = af.agent(AGENT, agent_key)
        with agent.task("support.refund", subject_token=person(), params={"order": "4411"}) as task:
            client = task.gateway().http_client()
            call: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": ALLOWED, "arguments": {}},
            }
            assert client.post(gw.mcp_url, json=call).status_code == 200
            with as_process(receiver_key, monkeypatch):
                to = agent.run_key_thumbprint()
            handover = task.hand_over(to=to)
        wire = handover.to_wire()
        with as_process(receiver_key, monkeypatch):
            taken = agent.take_over(wire)
            assert taken.gateway().http_client().post(gw.mcp_url, json=call).status_code == 200
            # A refused takeover, whose error an error reporter would capture with its frames' locals.
            report = report_of_refused_takeover(agent, wire)
        places.extra += [
            wire.encode(),
            repr(handover).encode(),
            repr(taken).encode(),
            repr(agent).encode(),
            repr(receiver_key).encode(),
            report.encode(),
            b"\n".join(repr(r).encode() for r in gw.requests),  # every byte the gateway received
        ]
    out = capsys.readouterr()
    places.extra += [out.out.encode(), out.err.encode()]

    assert gw.holders[taken.mandate_id] == receiver_key.thumbprint  # the run did what it was for
    assert places.logs(), "the run logged nothing: the log scan proves nothing"
    assert len(gw.requests) > 5
    where = {name: found_in(needles, text) for name, text in places.everything().items()}
    assert where == {"environment": [], "files": [], "logs": [], "reports and wire": []}
    # What does travel is the signed move, and it is not the key: the platform got it and moved the task on it.
    assert [r["how"] for r in gw.renewals] == ["signed_by_previous"]


def test_no_public_way_makes_a_second_key_or_shows_the_private_half() -> None:
    import authfi

    assert not hasattr(authfi, "run_key") and not hasattr(authfi, "DPoPKey")
    key = _dpop.run_key()
    assert key is _dpop.run_key()  # one key for the process
    assert not [a for a in dir(key) if "private" in a.lower() or a in ("d", "export", "to_pem", "to_jwk")]
    assert "d" not in key.public_jwk()
