"""The session bus is opened by this package and its listener's uid is read from the connected socket first."""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from authfi import _bus

jeepney: Any = pytest.importorskip("jeepney")  # untyped, and a dependency of keyring on Linux only
Parser = jeepney.Parser
new_method_return = jeepney.new_method_return

ME = 1000
LINUX = sys.platform == "linux"
pytestmark = pytest.mark.skipif(not hasattr(socket, "AF_UNIX"), reason="needs Unix sockets")


class FakeBus:
    """A Unix-socket listener that speaks just enough D-Bus: ``EXTERNAL`` auth, then answers ``Hello``."""

    def __init__(self, address: str | bytes, *, reject: bool = False) -> None:
        self.received: list[bytes] = []
        self.accepted = 0
        self._reject = reject
        self._listener = socket.socket(socket.AF_UNIX)
        self._listener.bind(address)
        self._listener.listen(256)  # the swap test opens 120 connections faster than the accept loop takes them
        self._listener.settimeout(0.2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self._listener.close()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._listener.accept()
            except TimeoutError:
                continue
            except OSError:
                return
            self.accepted += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _read_until(self, conn: socket.socket, marker: bytes, buffer: bytes) -> bytes | None:
        while marker not in buffer:
            chunk = conn.recv(4096)
            if not chunk:
                return None
            self.received.append(chunk)
            buffer += chunk
        return buffer

    def _serve(self, conn: socket.socket) -> None:
        with contextlib.closing(conn):
            conn.settimeout(3)
            try:
                buffer = self._read_until(conn, b"\r\n", b"")
                if buffer is None:
                    return
                conn.sendall(b"REJECTED EXTERNAL\r\n" if self._reject else b"OK 0123456789abcdef0123456789abcdef\r\n")
                if self._reject:
                    return
                buffer = self._read_until(conn, b"BEGIN\r\n", buffer.split(b"\r\n", 1)[1])
                if buffer is None:
                    return
                parser = Parser()
                parser.add_data(buffer.split(b"BEGIN\r\n", 1)[1])
                message = parser.get_next_message()
                while message is None:
                    chunk = conn.recv(4096)
                    if not chunk:
                        return
                    self.received.append(chunk)
                    parser.add_data(chunk)
                    message = parser.get_next_message()
                conn.sendall(new_method_return(message, "s", (":1.7",)).serialise(serial=1))
                while conn.recv(4096):
                    pass
            except OSError:
                return


@pytest.fixture
def bus_path(tmp_path: Path) -> Iterator[Callable[..., tuple[FakeBus, str]]]:
    made: list[FakeBus] = []

    def make(name: str = "bus", **kwargs: Any) -> tuple[FakeBus, str]:
        path = str(tmp_path / name)
        fake = FakeBus(path, **kwargs)
        made.append(fake)
        return fake, path

    yield make
    for fake in made:
        fake.close()


@pytest.fixture
def peers(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Who the kernel says listens on each socket, by socket name; the default is the person."""
    owners: dict[str, int] = {}

    def peer_uid(sock: socket.socket) -> int:
        name = sock.getpeername()
        return owners.get(name if isinstance(name, str) else name.decode("latin-1"), ME)

    monkeypatch.setattr(_bus, "peer_uid", peer_uid)
    return owners


def test_the_normal_session_bus_connects_and_is_authenticated(
    bus_path: Callable[..., tuple[FakeBus, str]], peers: dict[str, int]
) -> None:
    fake, path = bus_path()
    connection = _bus.open_verified_bus(f"unix:path={path},guid=0123", ME)
    try:
        assert connection.unique_name == ":1.7"
        assert fake.accepted == 1
    finally:
        connection.close()


def test_a_listener_under_another_uid_is_refused_before_a_byte_is_sent(
    bus_path: Callable[..., tuple[FakeBus, str]], peers: dict[str, int]
) -> None:
    fake, path = bus_path()
    peers[path] = 0
    with pytest.raises(_bus.BusPeerError) as caught:
        _bus.open_verified_bus(f"unix:path={path}", ME)
    assert "uid 0" in str(caught.value) and f"uid {ME}" in str(caught.value)
    assert fake.received == []  # not even the NUL byte or the AUTH line


def test_a_bad_address_in_a_list_is_refused_even_after_a_good_one_fails_to_connect(
    bus_path: Callable[..., tuple[FakeBus, str]], peers: dict[str, int], tmp_path: Path
) -> None:
    fake, path = bus_path()
    peers[path] = 0
    missing = tmp_path / "missing"
    with pytest.raises(_bus.BusPeerError):
        _bus.open_verified_bus(f"unix:path={missing};unix:path={path}", ME)
    assert fake.received == []


def test_the_next_address_is_tried_when_one_cannot_be_reached(
    bus_path: Callable[..., tuple[FakeBus, str]], peers: dict[str, int], tmp_path: Path
) -> None:
    _fake, path = bus_path()
    connection = _bus.open_verified_bus(f"unix:path={tmp_path / 'missing'};unix:path={path}", ME)
    connection.close()


@pytest.mark.parametrize(
    "address", ["tcp:host=evil.example,port=1", "nonce-tcp:host=x,port=1,noncefile=/x", "garbage", ""]
)
def test_an_address_that_is_not_a_unix_socket_is_refused(address: str, peers: dict[str, int]) -> None:
    with pytest.raises(_bus.BusError):
        _bus.open_verified_bus(address, ME)


def test_a_bus_that_rejects_the_handshake_is_an_error(
    bus_path: Callable[..., tuple[FakeBus, str]], peers: dict[str, int]
) -> None:
    _fake, path = bus_path(reject=True)
    with pytest.raises(_bus.BusError):
        _bus.open_verified_bus(f"unix:path={path}", ME)


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need a privilege on Windows")
def test_a_symlink_swapped_while_connecting_never_reaches_a_foreign_listener(
    bus_path: Callable[..., tuple[FakeBus, str]], peers: dict[str, int], tmp_path: Path
) -> None:
    """The path check this replaces lost this race (the swap landed between the check and the connect)."""
    _good, good_path = bus_path("good")
    evil, evil_path = bus_path("evil")
    peers[evil_path] = 0
    link = tmp_path / "session-bus"
    stop = threading.Event()

    def swap() -> None:
        flip = False
        while not stop.is_set():
            target = evil_path if flip else good_path
            temporary = tmp_path / "swap.tmp"
            with contextlib.suppress(OSError):
                temporary.unlink()
            temporary.symlink_to(target)
            os.replace(temporary, link)
            flip = not flip

    link.symlink_to(good_path)
    swapper = threading.Thread(target=swap, daemon=True)
    swapper.start()
    connected = refused = 0
    try:
        for _ in range(120):
            try:
                _bus.open_verified_bus(f"unix:path={link}", ME).close()
                connected += 1
            except _bus.BusPeerError:
                refused += 1
    finally:
        stop.set()
        swapper.join(timeout=2)
    assert connected + refused == 120
    assert refused > 0 and connected > 0  # the swap really did land on both sides
    assert evil.received == []  # nothing was ever sent to the foreign listener


@pytest.mark.skipif(not LINUX, reason="abstract sockets and SO_PEERCRED are Linux")
def test_an_abstract_address_is_checked_too(peers: dict[str, int]) -> None:
    name = f"authfi-test-{os.getpid()}"
    fake = FakeBus("\0" + name)
    try:
        connection = _bus.open_verified_bus(f"unix:abstract={name},guid=01", ME)
        connection.close()
        peers["\0" + name] = 0
        before = list(fake.received)
        with pytest.raises(_bus.BusPeerError):
            _bus.open_verified_bus(f"unix:abstract={name}", ME)
        assert fake.received == before
    finally:
        fake.close()


@pytest.mark.skipif(not LINUX, reason="SO_PEERCRED is Linux")
def test_the_real_kernel_credentials_are_read_from_the_connected_socket(
    bus_path: Callable[..., tuple[FakeBus, str]],
) -> None:
    fake, path = bus_path()
    uid = os.getuid()
    _bus.open_verified_bus(f"unix:path={path}", uid).close()  # a listener run by this user passes
    before = list(fake.received)
    with pytest.raises(_bus.BusPeerError):  # the same listener, judged as if we were another user
        _bus.open_verified_bus(f"unix:path={path}", uid + 1)
    assert fake.received == before
