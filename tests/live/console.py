"""What the live suite needs from a real project: its address, and an administrator's console API.

Set, for a project you test against (never production, never a real person's account):

- ``AUTHFI_LIVE_ISSUER``: the project, ``https://<project>.authfi.io``;
- ``AUTHFI_LIVE_CONSOLE_TOKEN_FILE``: a file holding a console token of a probe administrator. It is
  read on every call, so a harness can renew it while the suite runs;
- ``AUTHFI_LIVE_PERSON_DIR`` (sign-in only): a directory shared with a harness that plays the
  sign-in page for the probe person.

Every object the suite registers is removed by the fixture that made it, whatever the outcome. No
token, key or secret is printed.
"""

from __future__ import annotations

import os
import secrets
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

ISSUER = os.environ.get("AUTHFI_LIVE_ISSUER", "")
TOKEN_FILE = os.environ.get("AUTHFI_LIVE_CONSOLE_TOKEN_FILE", "")
PERSON_DIR = os.environ.get("AUTHFI_LIVE_PERSON_DIR", "")
RUN = secrets.token_hex(3)

needs_console = pytest.mark.skipif(not (ISSUER and TOKEN_FILE), reason="AUTHFI_LIVE_CONSOLE_TOKEN_FILE is not set")


def console_token() -> str:
    return Path(TOKEN_FILE).read_text(encoding="utf-8").strip()


def gql(query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
    """One console API call as the probe administrator; raises with the API's errors."""
    response = httpx.post(
        ISSUER + "/graphql",
        json={"query": query, "variables": variables or {}},
        headers={"Authorization": "Bearer " + console_token()},
        timeout=60,
    )
    body = cast(dict[str, Any], response.json())
    if response.status_code != 200 or body.get("errors"):
        raise AssertionError(f"console API {response.status_code}: {str(body.get('errors') or body)[:300]}")
    return cast(dict[str, Any], body["data"])


class Cleanups:
    """Undo steps, run last-in first-out; a failing one is reported and the rest still run."""

    def __init__(self) -> None:
        self._steps: list[tuple[str, Callable[[], object]]] = []

    def add(self, what: str, step: Callable[[], object]) -> None:
        self._steps.append((what, step))

    def run(self) -> None:
        for what, step in reversed(self._steps):
            try:
                step()
            except Exception as error:
                print(f"cleanup {what}: {type(error).__name__}")


def eventually(attempt: Callable[[], Any], *, tries: int = 20, every: float = 1.5) -> Any:
    """``attempt()`` until it returns something other than ``None``."""
    for _ in range(tries):
        result = attempt()
        if result is not None:
            return result
        time.sleep(every)
    return None
