"""The package is on PyPI: the install text in the source says `pip install authfi`, never a git URL."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GIT_INSTALL = "git+https://github.com/authfi/authfi-python-sdk"


def test_no_source_file_tells_people_to_install_authfi_from_the_repository() -> None:
    found = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in sorted((ROOT / "src").rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if GIT_INSTALL in line
    ]
    assert not found, found
