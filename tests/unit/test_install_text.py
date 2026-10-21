"""Nothing is on PyPI: the install text in the source says to install from the repository."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD_FORM = re.compile(r"pip install ['\"]?authfi(\[[^\]]*\])?['\"]?(?!\s*@)(?=[\s'\"`)]|$)")


def test_no_source_file_tells_people_to_install_authfi_from_the_package_index() -> None:
    found = [
        f"{path.relative_to(ROOT)}:{number}: {line.strip()}"
        for path in sorted((ROOT / "src").rglob("*.py"))
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for match in OLD_FORM.finditer(line)
        if "@" not in line[match.start() :]
    ]
    assert not found, found
