# Contributing

The checks a pull request must pass, run locally with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run ruff check .
uv run ruff format --check src tests scripts examples
uv run mypy
uv run pyright
uv run coverage run -m pytest -m "not live" && uv run coverage json -q && uv run python scripts/coverage_gates.py
```

- **Types.** The package passes `mypy --strict` and pyright in strict mode, tests and examples
  included.
- **Tests.** Every behaviour has a test that fails without it, the refusal as well as the success.
  Coverage gates: the verifier and the protocol layer 95% of branches; agents and webhooks 95% of
  branches; adapters 85% of lines; the rest 90% of lines.
- **Fixtures.** A fixture that encodes a platform answer names its source in
  `tests/fixtures/SOURCES.toml`.
- **Refusal codes.** `src/authfi/_codes.py` is generated from the AuthFI kernel by
  `scripts/gen_codes.py`; do not edit it by hand.
- **Live tests** (`pytest -m live`) run against an AuthFI project and are skipped unless the
  `AUTHFI_LIVE_*` settings are present.
