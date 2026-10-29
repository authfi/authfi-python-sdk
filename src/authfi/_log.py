"""The ``authfi`` logger, a redacting filter, and once-only warnings.

A library logs and leaves the configuration to the application. Nothing in this package formats a
credential into a log record; the filter installed on the ``authfi`` logger is the second net, and
rewrites anything shaped like a JWT or a ``Bearer`` credential.
"""

from __future__ import annotations

import logging
import re
import threading
import warnings

__all__ = ["LOGGER_NAME", "RedactingFilter", "get_logger", "warn_once"]

LOGGER_NAME = "authfi"

# Three base64url segments, the first two starting like a JSON object ("eyJ"): a compact JWS.
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]{4,}\.[A-Za-z0-9_-]*")
_BEARER = re.compile(r"\b(Bearer|DPoP)\s+[A-Za-z0-9._~+/-]+=*", re.IGNORECASE)


def redact(text: str) -> str:
    """``text`` with JWT-shaped strings and ``Bearer``/``DPoP`` credentials replaced."""
    return _BEARER.sub(r"\1 [redacted]", _JWT.sub("[redacted-jwt]", text))


class RedactingFilter(logging.Filter):
    """Rewrites a record's message and arguments so no JWT or bearer credential reaches a handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Redact ``record`` in place; never drops it."""
        try:
            message = record.getMessage()
        except (TypeError, ValueError):  # a malformed record: leave it to logging's own handling
            return True
        cleaned = redact(message)
        if cleaned != message:
            record.msg = cleaned
            record.args = None
        return True


_root = logging.getLogger(LOGGER_NAME)
if not any(isinstance(f, RedactingFilter) for f in _root.filters):
    _root.addFilter(RedactingFilter())


def get_logger(logger: logging.Logger | None = None) -> logging.Logger:
    """The logger to use: the caller's own, given the redacting filter, or ``authfi``."""
    if logger is None:
        return _root
    if not any(isinstance(f, RedactingFilter) for f in logger.filters):
        logger.addFilter(RedactingFilter())
    return logger


_seen: set[tuple[str, str]] = set()
_seen_lock = threading.Lock()


def warn_once(
    logger: logging.Logger,
    key: tuple[str, str],
    message: str,
    *args: object,
    category: type[Warning] | None = None,
    stacklevel: int = 3,
) -> None:
    """Log ``message`` at warning level the first time ``key`` is seen in this process.

    With ``category``, also raise it through :mod:`warnings`, so test suites and ``-W error``
    see it.
    """
    with _seen_lock:
        if key in _seen:
            return
        _seen.add(key)
    logger.warning(message, *args)
    if category is not None:
        warnings.warn(message % args if args else message, category, stacklevel=stacklevel)


def reset_warn_once() -> None:
    """Forget which warnings were given. For tests."""
    with _seen_lock:
        _seen.clear()
