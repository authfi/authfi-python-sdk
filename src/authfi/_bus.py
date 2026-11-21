"""Open the D-Bus session bus ourselves and check who is on the other end of the socket before saying anything to it.

A path check (is the socket mine?) races with a symlink swap, and says nothing about an abstract address. The kernel,
though, records which user called ``listen()`` on a Unix socket, and ``SO_PEERCRED`` hands that back for the connected
socket itself. So the helper connects, reads the peer's uid from the socket it holds, and only then authenticates. A
listener run by another user is refused whatever the address or the path pointed at a moment earlier. Linux only: it is
the one platform whose credential store talks to a bus (the Secret Service).

The check proves the bus is run by the same user. A process of that same user, such as a repository's hook, passes it;
that is the workspace-trust boundary in ``docs/claude-code.md``, not something a check here can move.
"""

from __future__ import annotations

import importlib
import socket
import struct
from typing import Any

#: The variable the session bus address comes from.
DBUS = "DBUS_SESSION_BUS_ADDRESS"


class BusError(ConnectionError):
    """The session bus could not be reached or does not speak D-Bus."""


class BusPeerError(BusError):
    """The socket is served by another user, so nothing was sent to it."""


def peer_uid(sock: socket.socket) -> int:
    """The uid of the process that listens on the other end of a connected Unix socket (Linux ``SO_PEERCRED``)."""
    credentials = sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
    return int(struct.unpack("3i", credentials)[1])


def _authenticate(sock: socket.socket, timeout: float) -> None:
    """The D-Bus ``EXTERNAL`` handshake, as jeepney does it, on a socket that is already connected and checked."""
    authenticator = importlib.import_module("jeepney.auth").Authenticator(inc_null_byte=False)
    sock.settimeout(timeout)
    sock.send(b"\0")  # Linux: the bus identifies us by SO_PEERCRED, no ancillary data
    for request in authenticator:
        sock.sendall(request)
        reply = sock.recv(1024)
        if not reply:
            raise BusError("the bus closed the connection during authentication")
        authenticator.feed(reply)
    sock.sendall(b"BEGIN\r\n")


def open_verified_bus(address: str, uid: int, *, timeout: float = 2.0) -> Any:
    """A jeepney ``DBusConnection`` to the bus at ``address``, once its listener is known to run as ``uid``.

    Raises :class:`BusPeerError` (before any byte is sent) when the listener runs as someone else, and
    :class:`BusError` when no address in the list can be reached or authenticated.
    """
    bus = importlib.import_module("jeepney.bus")
    blocking = importlib.import_module("jeepney.io.blocking")
    try:
        targets = list(bus.get_connectable_addresses(address))
    except (RuntimeError, ValueError) as exc:
        raise BusError(f"the address {address!r} names no Unix socket: {exc}") from exc
    failure: Exception | None = None
    for target in targets:
        sock = socket.socket(socket.AF_UNIX)
        try:
            sock.settimeout(timeout)
            sock.connect(target)
            peer = peer_uid(sock)
            if peer != uid:
                raise BusPeerError(f"the bus socket is served by uid {peer}, not by you (uid {uid})")
            _authenticate(sock, timeout)
            sock.settimeout(None)
            return blocking.DBusConnection(sock)
        except BusPeerError:
            sock.close()
            raise
        except (OSError, ValueError) as exc:  # unreachable, refused, or not D-Bus: try the next address
            sock.close()
            failure = exc
    raise BusError(f"cannot reach the session bus at {address!r}: {failure}")
