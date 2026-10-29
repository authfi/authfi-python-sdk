"""Fail when a part of the package falls below its coverage gate.

Reads ``coverage json`` output (``coverage.json``) and checks each group of files:

=============================================  ======================
Files                                          Gate
=============================================  ======================
the verifier and the protocol layer            95% of branches
agents and webhooks (when present)             95% of branches
the framework adapters                         85% of lines
everything else in the core                    90% of lines
=============================================  ======================

::

    coverage run -m pytest && coverage json && python scripts/coverage_gates.py
"""

from __future__ import annotations

import fnmatch
import json
import sys
from pathlib import Path
from typing import Any

#: (name, file patterns, measure, minimum percent)
GATES: tuple[tuple[str, tuple[str, ...], str, float], ...] = (
    ("verifier and protocol", ("src/authfi/verify.py", "src/authfi/_protocol/*"), "branches", 95.0),
    ("agents and webhooks", ("src/authfi/agents/*", "src/authfi/webhooks.py"), "branches", 95.0),
    (
        "framework adapters",
        (
            "src/authfi/fastapi.py",
            "src/authfi/flask.py",
            "src/authfi/langchain.py",
            "src/authfi/langgraph.py",
            "src/authfi/openai_agents.py",
            "src/authfi/claude_agent.py",
            "src/authfi/adk.py",
            "src/authfi/pydantic_ai.py",
            "src/authfi/crewai.py",
            "src/authfi/llama_index.py",
            "src/authfi/_holds.py",
        ),
        "lines",
        85.0,
    ),
    ("core", ("src/authfi/*",), "lines", 90.0),
)


def _group(path: str) -> int:
    for index, (_, patterns, _, _) in enumerate(GATES):
        if any(fnmatch.fnmatch(path, p) for p in patterns):
            return index
    return len(GATES) - 1


def main(argv: list[str] | None = None) -> int:
    """Check ``coverage.json`` (or the file named) against the gates."""
    args = sys.argv[1:] if argv is None else argv
    report: dict[str, Any] = json.loads(Path(args[0] if args else "coverage.json").read_text(encoding="utf-8"))
    totals: list[dict[str, int]] = [
        {"lines": 0, "covered_lines": 0, "branches": 0, "covered_branches": 0} for _ in GATES
    ]
    for path, data in report["files"].items():
        summary = data["summary"]
        bucket = totals[_group(Path(path).as_posix())]
        bucket["lines"] += summary["num_statements"]
        bucket["covered_lines"] += summary["covered_lines"]
        bucket["branches"] += summary.get("num_branches", 0)
        bucket["covered_branches"] += summary.get("covered_branches", 0)
    failed = False
    for (name, _, measure, minimum), bucket in zip(GATES, totals, strict=True):
        total = bucket[measure]
        if total == 0:
            continue
        percent = 100.0 * bucket[f"covered_{measure}"] / total
        verdict = "ok" if percent >= minimum else "BELOW"
        print(f"{name:24} {measure:8} {percent:6.2f}%  (gate {minimum:.0f}%)  {verdict}")
        failed |= percent < minimum
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
