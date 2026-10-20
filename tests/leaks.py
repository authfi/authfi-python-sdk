"""What an error reporter would capture from an exception: frame locals, as Sentry and Rich show them."""

from __future__ import annotations

import io
import json
from typing import Any, cast

from rich.console import Console
from rich.traceback import Traceback
from sentry_sdk.utils import event_from_exception

__all__ = ["captured", "rich_traceback", "sdk_frames", "sentry_event"]

_SENTRY_OPTIONS: dict[str, Any] = {
    "include_local_variables": True,
    "include_source_context": False,
    "max_value_length": 4096,
    "custom_repr": None,
    "event_scrubber": None,
    "max_request_body_size": "always",
}


def sentry_event(exc: BaseException) -> str:
    """The event Sentry builds for ``exc`` with local variable capture on (its default), as JSON text."""
    event, _hint = event_from_exception(exc, client_options=_SENTRY_OPTIONS)
    return json.dumps(event, default=repr)


def sdk_frames(exc: BaseException) -> str:
    """The locals Sentry captures from the SDK's own frames only, as JSON text.

    For a failure injected into a library the SDK calls: that library's frames hold what the SDK
    passed it by design, and a test cannot hide that. The SDK's own frames are the SDK's to keep clean.
    """
    event, _hint = event_from_exception(exc, client_options=_SENTRY_OPTIONS)
    document = cast(dict[str, Any], event)
    frames: list[dict[str, Any]] = [
        frame
        for entry in document["exception"]["values"]
        for frame in entry["stacktrace"]["frames"]
        if "/src/authfi/" in frame["abs_path"].replace("\\", "/")
    ]
    return json.dumps([frame.get("vars", {}) for frame in frames], default=repr)


def rich_traceback(exc: BaseException) -> str:
    """The traceback Rich prints for ``exc`` with ``show_locals=True``."""
    console = Console(file=io.StringIO(), width=200, force_terminal=False, record=True)
    console.print(Traceback.from_exception(type(exc), exc, exc.__traceback__, show_locals=True, max_frames=100))
    return console.export_text()


def captured(exc: BaseException) -> str:
    """Everything either reporter would record for ``exc``."""
    return sentry_event(exc) + "\n" + rich_traceback(exc)
