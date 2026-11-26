"""The demo agent's loop: stop, back-off, resume, and the credential-less direct probe (runner/scripts/langgraph)."""

from __future__ import annotations

import asyncio
import errno
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runner" / "scripts" / "langgraph"))
import demo_loop as dl  # noqa: E402

JWT = "eyJhbGciOiJFUzI1NiJ9.eyJzdWIiOiJhZ3RfMSJ9.c2lnbmF0dXJlLWJ5dGVz"


class FakeSession:
    """A session that plays back a script of outcomes: None (ok) or an exception, per call."""

    def __init__(self, fail_at: str | None = None, error: BaseException | None = None) -> None:
        self.fail_at, self.error = fail_at, error
        self.closed = False

    def _maybe(self, where: str) -> None:
        if self.fail_at == where and self.error is not None:
            raise self.error

    async def list_orders(self) -> str:
        self._maybe("list")
        return "order 4411 open"

    async def read_order(self, order_text: str) -> str | None:
        self._maybe("read")
        return "order 4411 detail"

    async def summarize(self, orders_text: str, on_chunk: Any) -> dict[str, int]:
        on_chunk()
        on_chunk()
        self._maybe("model")
        return {"input_tokens": 120, "output_tokens": 640}

    async def aclose(self) -> None:
        self.closed = True


class Opener:
    """Hands out sessions from a queue; an exception in the queue is raised as a refused sign-in."""

    def __init__(self, *items: FakeSession | BaseException) -> None:
        self.items, self.opened = list(items), 0

    async def __call__(self) -> FakeSession:
        self.opened += 1
        item = self.items.pop(0) if self.items else FakeSession()
        if isinstance(item, BaseException):
            raise item
        return item


CFG = dl.Config(interval=60, stopped_backoff=30, error_backoff=10)


def events(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


async def no_probe() -> dict[str, Any]:  # pragma: no cover - only called by a failing test
    raise AssertionError("the probe ran")


async def test_a_run_logs_each_step_and_waits_the_interval(capsys: pytest.CaptureFixture[str]) -> None:
    loop = dl.DemoLoop(Opener(FakeSession()), CFG, probe=no_probe)
    assert await loop.step() == 60
    names = [e["event"] for e in events(capsys)]
    assert names == ["sign_in", "run_start", "tool_call", "tool_call", "model_call_start", "model_call_end"]


async def test_a_stop_on_a_tool_call_backs_off_then_signs_in_afresh_and_resumes(capsys: pytest.CaptureFixture[str]) -> None:
    stopped = FakeSession("list", RuntimeError("refused: EKILL001 the agent is stopped"))
    opener = Opener(stopped, FakeSession())
    loop = dl.DemoLoop(opener, CFG)
    assert await loop.step() == 30
    assert stopped.closed, "the old session is dropped so the next one is a fresh sign-in"
    assert await loop.step() == 60
    assert opener.opened == 2
    names = [e["event"] for e in events(capsys)]
    assert names.count("stopped") == 1 and names[-1] == "model_call_end"
    assert "resume" in names and names.index("resume") > names.index("stopped")


async def test_a_cut_stream_is_a_stop(capsys: pytest.CaptureFixture[str]) -> None:
    loop = dl.DemoLoop(Opener(FakeSession("model", dl.StreamCut("ended early"))), CFG)
    assert await loop.step() == 30
    out = events(capsys)
    end = next(e for e in out if e["event"] == "model_call_end")
    assert end["ok"] is False and end["chunks"] == 2
    assert any(e["event"] == "stopped" for e in out)


async def test_a_refused_token_mint_backs_off_without_crashing_and_never_hammers(capsys: pytest.CaptureFixture[str]) -> None:
    refused = RuntimeError("token endpoint: EKILL001")
    opener = Opener(refused, refused, refused, FakeSession())
    slept: list[float] = []
    runs = 0

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    def done() -> bool:
        nonlocal runs
        runs += 1
        return runs > 4

    await dl.DemoLoop(opener, CFG, sleep=sleep).forever(done)
    assert slept == [30, 30, 30, 60], "every refusal waits the stop back-off; a good run waits the interval"
    assert [e["event"] for e in events(capsys)].count("stopped") == 3


async def test_an_unexpected_error_is_logged_and_does_not_end_the_process(capsys: pytest.CaptureFixture[str]) -> None:
    loop = dl.DemoLoop(Opener(FakeSession("read", ValueError("boom"))), CFG)
    assert await loop.step() == 10
    assert [e["event"] for e in events(capsys)][-1] == "error"


async def test_a_token_in_an_error_is_never_logged(capsys: pytest.CaptureFixture[str]) -> None:
    await dl.DemoLoop(Opener(FakeSession("list", ValueError(f"bad Bearer {JWT} here"))), CFG).step()
    out = capsys.readouterr().out
    assert JWT not in out and "redacted" in out


async def test_the_probe_runs_once_per_run_only_when_asked(capsys: pytest.CaptureFixture[str]) -> None:
    seen: list[int] = []

    async def probe() -> dict[str, Any]:
        seen.append(1)
        return {"outcome": "refused", "errno": errno.ECONNREFUSED}

    await dl.DemoLoop(Opener(FakeSession()), dl.Config(direct_probe=True), probe=probe).step()
    assert seen == [1]
    line = next(e for e in events(capsys) if e["event"] == "direct_probe")
    assert line["outcome"] == "refused" and line["errno"] == errno.ECONNREFUSED


class FakeWriter:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


async def test_the_direct_probe_sends_nothing_and_holds_no_credential() -> None:
    writer = FakeWriter()
    asked: dict[str, Any] = {}

    async def opener(host: str, port: int, **kw: Any) -> tuple[None, FakeWriter]:
        asked.update(host=host, port=port, **kw)
        return None, writer

    result = await dl.direct_probe(opener=opener)
    assert result == {"outcome": "connected", "host": "api.anthropic.com"}
    assert writer.writes == [] and writer.closed
    assert (asked["host"], asked["port"]) == ("api.anthropic.com", 443)
    source = inspect.getsource(dl.direct_probe).lower()
    for word in ("authorization", "x-api-key", "bearer", "environ", "getenv", "prompt", ".write("):
        assert word not in source, word


@pytest.mark.parametrize("code", [errno.ECONNREFUSED, errno.ENETUNREACH])
async def test_a_refused_direct_connect_is_logged_with_its_errno(code: int) -> None:
    async def opener(host: str, port: int, **kw: Any) -> Any:
        raise OSError(code, "refused")

    result = await dl.direct_probe(opener=opener)
    assert result["outcome"] == "refused" and result["errno"] == code and result["errname"]


async def test_a_probe_that_hangs_times_out() -> None:
    async def opener(host: str, port: int, **kw: Any) -> Any:
        await asyncio.sleep(5)

    assert (await dl.direct_probe(timeout=0.01, opener=opener))["outcome"] == "timeout"


def test_the_settings_default_and_read_the_environment() -> None:
    assert dl.Config.from_env({}) == dl.Config(interval=60, stopped_backoff=30)
    cfg = dl.Config.from_env({"DEMO_INTERVAL_SECONDS": "5", "DEMO_STOPPED_BACKOFF_SECONDS": "x", "DEMO_DIRECT_PROVIDER_PROBE": "1"})
    assert (cfg.interval, cfg.stopped_backoff, cfg.direct_probe) == (5, 30, True)
