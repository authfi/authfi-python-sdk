"""Key files that only their owner can read: the POSIX mode, or the Windows DACL.

- Reading: a key file readable by group or others (POSIX), or whose DACL lets Everyone, Users or
  Authenticated Users read it (Windows), is refused, naming the fix.
- Writing: the file is created exclusively (never over an existing one), mode ``0600`` on POSIX;
  on Windows its DACL is replaced by one entry for the owner, with inheritance cut.
"""

from __future__ import annotations

import os
import re
import stat
import sys
from pathlib import Path
from typing import Any

from authfi.errors import ConfigurationError

__all__ = ["check_private", "write_private"]

# SIDs that make a file readable by people other than its owner, as SDDL writes them.
_BROAD_SIDS = frozenset({"WD", "BU", "AU", "S-1-1-0", "S-1-5-32-545", "S-1-5-11"})
_READ_RIGHTS = frozenset({"FA", "GA", "GR", "FR", "GX", "FX"})
_ACE = re.compile(r"\(([^;()]*);([^;()]*);([^;()]*);([^;()]*);([^;()]*);([^;()]*)\)")
_GENERIC_ALL, _GENERIC_READ, _FILE_READ_DATA = 0x10000000, 0x80000000, 0x1


def _grants_read(rights: str) -> bool:
    if rights.lower().startswith("0x"):
        mask = int(rights, 16)
        return bool(mask & (_GENERIC_ALL | _GENERIC_READ | _FILE_READ_DATA))
    tokens = {rights[i : i + 2] for i in range(0, len(rights), 2)}
    return bool(tokens & _READ_RIGHTS)


def broad_readers(sddl: str) -> list[str]:
    """The broad SIDs an SDDL string's allow entries let read the file."""
    found: list[str] = []
    for ace_type, _flags, rights, _object, _inherit, sid in _ACE.findall(sddl.split("S:", 1)[0]):
        if ace_type in ("A", "OA") and sid in _BROAD_SIDS and _grants_read(rights):
            found.append(sid)
    return found


def _windows_libraries() -> tuple[Any, Any]:  # pragma: no cover - Windows only
    import ctypes

    win_dll: Any = getattr(ctypes, "WinDLL")  # noqa: B009 - absent from the type stubs off Windows
    return win_dll("advapi32", use_last_error=True), win_dll("kernel32", use_last_error=True)


def _windows_sddl(path: Path) -> str:  # pragma: no cover - Windows only; exercised on Windows
    import ctypes
    from ctypes import wintypes

    advapi, kernel = _windows_libraries()
    descriptor = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    status: int = advapi.GetNamedSecurityInfoW(
        ctypes.c_wchar_p(str(path)), 1, 4, None, None, ctypes.byref(dacl), None, ctypes.byref(descriptor)
    )
    if status != 0:
        raise ConfigurationError(f"cannot read the permissions of the key file {path} (error {status})")
    text = wintypes.LPWSTR()
    try:
        if not advapi.ConvertSecurityDescriptorToStringSecurityDescriptorW(descriptor, 1, 4, ctypes.byref(text), None):
            raise ConfigurationError(f"cannot read the permissions of the key file {path}")
        return str(text.value or "")
    finally:
        kernel.LocalFree(text)
        kernel.LocalFree(descriptor)


def check_private(path: Path) -> None:
    """Refuse a key file that others can read."""
    if sys.platform == "win32":  # pragma: no cover - Windows only
        readers = broad_readers(_windows_sddl(path))
        if readers:
            raise ConfigurationError(
                f"the key file {path} can be read by {', '.join(sorted(set(readers)))}; keep it readable by its "
                f'owner only, for example: icacls "{path}" /inheritance:r /grant:r "%USERNAME%:F"'
            )
        return
    mode = os.stat(path).st_mode
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ConfigurationError(
            f"the key file {path} is readable by others (mode {stat.S_IMODE(mode):04o}); keep it readable by its "
            f"owner only: chmod 600 {path}"
        )


def _windows_owner_only(path: Path) -> None:  # pragma: no cover - Windows only; exercised on Windows
    import ctypes

    advapi, kernel = _windows_libraries()
    descriptor = ctypes.c_void_p()
    # D:P: a protected DACL (nothing inherited); (A;;FA;;;OW): full access for the owner only.
    if not advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        ctypes.c_wchar_p("D:P(A;;FA;;;OW)"), 1, ctypes.byref(descriptor), None
    ):
        raise ConfigurationError("cannot build an owner-only permission for the key file")
    try:
        present = ctypes.c_int()
        defaulted = ctypes.c_int()
        dacl = ctypes.c_void_p()
        advapi.GetSecurityDescriptorDacl(descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted))
        protected = 0x80000000  # PROTECTED_DACL_SECURITY_INFORMATION
        status: int = advapi.SetNamedSecurityInfoW(
            ctypes.c_wchar_p(str(path)), 1, 4 | protected, None, None, dacl, None
        )
        if status != 0:
            raise ConfigurationError(f"cannot make the key file {path} private (error {status})")
    finally:
        kernel.LocalFree(descriptor)


def write_private(path: Path, data: bytes) -> None:
    """Create ``path`` (never over an existing file) with ``data``, readable by its owner only."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise ConfigurationError(f"{path} exists; a key is never written over another") from exc
    try:
        if sys.platform == "win32":  # pragma: no cover - Windows only
            _windows_owner_only(path)
        os.write(fd, data)
    finally:
        os.close(fd)
