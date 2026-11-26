"""The demo agent's run loop: pure Python, no SDK import, so the stop / back-off / resume behaviour is unit-testable.

``demo_agent.py`` supplies a :class:`Session` backed by the AuthFI SDK; tests supply a fake one. Every step is one
structured JSON line on stdout. No token, key or prompt text is ever logged.
"""

from __future__ import annotations

import asyncio
import errno as errno_mod
import json
import re
import ssl
import sys
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

#: The model provider the node must refuse a direct connection to.
PROBE_HOST = "api.anthropic.com"
PROBE_PORT = 443

_SECRET = re.compile(r"(eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}|sk-[A-Za-z0-9_-]{12,}|(?i:bearer|dpop)\s+\S{12,})")
_STOP_MARKERS = ("EKILL001", "gateway_token_refused", "refused this credential", "agent_stopped", "agent_suspended")


def scrub(value: object) -> str:
    """The text of ``value`` with anything token-shaped removed, clipped to 300 characters."""
    return _SECRET.sub("<redacted>", str(value))[:300]


def log(event: str, **fields: Any) -> None:
    """One JSON line on stdout."""
    clean = {k: (scrub(v) if isinstance(v, str) else v) for k, v in fields.items()}
    print(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "event": event, **clean}, sort_keys=True), flush=True)


class StreamCut(Exception):
    """A model stream ended before the model finished: the signature of a Stop in flight."""


@dataclass(frozen=True)
class Config:
    """The loop's settings, read from ``DEMO_*`` environment variables."""

    interval: float = 60.0
    stopped_backoff: float = 30.0
    error_backoff: float = 10.0
    direct_probe: bool = False
    task_template: str = ""
    probe_timeout: float = 5.0

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> Config:
        """Read the settings; a value that is not a positive number falls back to its default."""

        def num(name: str, default: float) -> float:
            try:
                value = float(env.get(name, "").strip() or default)
            except ValueError:
                return default
            return value if value > 0 else default

        return cls(
            interval=num("DEMO_INTERVAL_SECONDS", 60.0),
            stopped_backoff=num("DEMO_STOPPED_BACKOFF_SECONDS", 30.0),
            direct_probe=env.get("DEMO_DIRECT_PROVIDER_PROBE", "").strip() == "1",
            task_template=env.get("DEMO_TASK_TEMPLATE", "").strip(),
        )


class Session(Protocol):
    """What one signed-in agent can do. The SDK-backed one lives in ``demo_agent.py``."""

    async def list_orders(self) -> str:
        """Call the gateway's ``orders.list`` tool; returns its text."""

    async def read_order(self, order_text: str) -> str | None:
        """Call ``orders.read`` for the first order in ``order_text``; ``None`` when there is none to read."""

    async def summarize(self, orders_text: str, on_chunk: Callable[[], None]) -> dict[str, int]:
        """Stream the model's summary; returns token counts. Raises :class:`StreamCut` if the stream ends early."""

    async def aclose(self) -> None:
        """Close the session's HTTP clients."""


def is_stop(error: BaseException) -> bool:
    """Whether ``error`` is the platform saying Stop: EKILL001, a refused token, or a cut stream."""
    if isinstance(error, StreamCut):
        return True
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        text = f"{type(current).__name__} {current}"
        if any(m in text for m in _STOP_MARKERS) or getattr(current, "code", None) in ("EKILL001", "gateway_token_refused"):
            return True
        current = current.__cause__ or current.__context__
    return False


async def direct_probe(
    host: str = PROBE_HOST,
    port: int = PROBE_PORT,
    *,
    timeout: float = 5.0,
    opener: Callable[..., Awaitable[tuple[Any, Any]]] = asyncio.open_connection,
) -> dict[str, Any]:
    """A deliberate "skip the gateway" attempt: a TCP and TLS handshake to the provider, sending nothing at all.

    No request, header, key or text of any kind is written: the connection is closed right after the handshake.
    """
    context = ssl.create_default_context()
    try:
        _reader, writer = await asyncio.wait_for(opener(host, port, ssl=context, server_hostname=host), timeout)
    except asyncio.TimeoutError:
        return {"outcome": "timeout", "host": host}
    except ssl.SSLError as error:
        return {"outcome": "tls_failed", "host": host, "reason": type(error).__name__}
    except OSError as error:
        code = error.errno
        return {"outcome": "refused", "host": host, "errno": code, "errname": errno_mod.errorcode.get(code or 0, "")}
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:  # noqa: BLE001 - the probe is over; a close error changes nothing
        pass
    return {"outcome": "connected", "host": host}


class DemoLoop:
    """Runs forever: sign in, do one run per interval, stop and back off on Stop, resume after Release."""

    def __init__(
        self,
        open_session: Callable[[], Awaitable[Session]],
        config: Config,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        probe: Callable[[], Awaitable[dict[str, Any]]] | None = None,
    ) -> None:
        self._open = open_session
        self.cfg = config
        self._sleep = sleep
        self._probe = probe or (lambda: direct_probe(timeout=config.probe_timeout))
        self._session: Session | None = None
        self._stopped = False
        self.runs = 0

    async def _drop(self) -> None:
        session, self._session = self._session, None
        if session is not None:
            try:
                await session.aclose()
            except Exception as error:  # noqa: BLE001
                log("close_failed", error=type(error).__name__)

    async def step(self) -> float:
        """Do one pass (sign in if needed, then one run); returns the seconds to wait before the next."""
        try:
            if self._session is None:
                self._session = await self._open()
                log("sign_in", ok=True, resumed=self._stopped)
            await self._run(self._session)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - nothing in a run may end the process
            await self._drop()
            if is_stop(error):
                self._stopped = True
                log("stopped", reason=type(error).__name__, detail=str(error), backoff_seconds=self.cfg.stopped_backoff)
                return self.cfg.stopped_backoff
            log("error", reason=type(error).__name__, detail=str(error), backoff_seconds=self.cfg.error_backoff)
            return self.cfg.error_backoff
        if self._stopped:
            self._stopped = False
            log("resume", runs=self.runs)
        return self.cfg.interval

    async def _run(self, session: Session) -> None:
        self.runs += 1
        run = self.runs
        log("run_start", run=run)
        orders = await session.list_orders()
        log("tool_call", run=run, tool="orders.list", ok=True)
        if await session.read_order(orders) is not None:
            log("tool_call", run=run, tool="orders.read", ok=True)
        chunks = 0

        def on_chunk() -> None:
            nonlocal chunks
            chunks += 1

        log("model_call_start", run=run, stream=True)
        try:
            usage = await session.summarize(orders, on_chunk)
        except Exception as error:
            log("model_call_end", run=run, ok=False, chunks=chunks, reason=type(error).__name__)
            raise
        log("model_call_end", run=run, ok=True, chunks=chunks, **usage)
        if self.cfg.direct_probe:
            log("direct_probe", run=run, **await self._probe())

    async def forever(self, should_stop: Callable[[], bool] = lambda: False) -> None:
        """Loop until ``should_stop`` says so (never, in the pod)."""
        while not should_stop():
            await self._sleep(await self.step())
        await self._drop()


def main_log_start(env: Mapping[str, str], cfg: Config) -> None:
    """The one start-up line: settings only, never a credential."""
    log(
        "start",
        interval=cfg.interval,
        stopped_backoff=cfg.stopped_backoff,
        direct_probe=cfg.direct_probe,
        task_template=bool(cfg.task_template),
        observe=env.get("AUTHFI_CONFIG_OBSERVE", "").strip() == "1",
        python=sys.version.split()[0],
    )
