"""Where the content of a model request came from, as far as the SDK can tell: the ``AuthFI-Input-Source`` header.

The gateway treats a user turn as the person's own words when a person is behind the call, and a project's
containment decides what outside content may do. A framework (ReAct-style loops, LangGraph and the like) often puts a
tool's output or fetched text into a user turn, and the gateway cannot tell that from the person's typing. The header is
how the agent's code says so: one request-level header, ``AuthFI-Input-Source: untrusted``, which can only add taint and
never remove it (any other value is dropped by the gateway). There is no value for "the person": a request that holds
only what the person typed sends no header.

``untrusted`` is the answer whenever the request holds a tool's output, a block this code does not know, or a shape this
code does not know: when in doubt, mark it. Nothing here reads the model's answer, and nothing changes the request.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, cast

__all__ = [
    "HEADER",
    "UNTRUSTED",
    "add_mark",
    "contents_are_untrusted",
    "is_model_request",
    "request_is_untrusted",
    "responses_input_is_untrusted",
    "untrusted_input",
]

HEADER = "AuthFI-Input-Source"
UNTRUSTED = "untrusted"

_MODEL_SUFFIXES = ("/v1/messages", "/v1/chat/completions", "/v1/responses")
#: How a ReAct-style loop or a framework labels a tool's output when it writes it into a user turn.
_OBSERVATION = re.compile(
    r"(?im)^\s*(observation|tool\s+(result|output)|function\s+(result|output))\s*:"
    r"|<(tool_response|tool_result|function_results)\b"
)
_forced: ContextVar[int] = ContextVar("authfi_untrusted_input", default=0)


@contextmanager
def untrusted_input() -> Generator[None]:
    """Every model request made inside the block is marked untrusted: for an app that fetches text into a user turn."""
    token = _forced.set(_forced.get() + 1)
    try:
        yield
    finally:
        _forced.reset(token)


def is_model_request(path: str) -> bool:
    """Whether ``path`` is a model route of the gateway (``/mcp/v1/...`` or ``/llm/v1/...``)."""
    return path.endswith(_MODEL_SUFFIXES) and path.startswith(("/mcp/v1/", "/llm/v1/"))


def request_is_untrusted(path: str, body: bytes | None) -> bool:
    """Whether a model request must carry ``AuthFI-Input-Source: untrusted``."""
    if _forced.get() or not body:
        return True
    try:
        doc: Any = json.loads(body)
    except ValueError:
        return True
    if not isinstance(doc, dict):
        return True
    fields = cast("dict[str, Any]", doc)
    if path.endswith("/v1/messages"):
        return _anthropic(fields)
    if path.endswith("/v1/chat/completions"):
        return _chat(fields)
    return _responses(fields)


def _text_outside(text: object) -> bool:
    return not isinstance(text, str) or _OBSERVATION.search(text) is not None


def _items(value: object) -> list[Any] | None:
    if isinstance(value, list):
        typed: list[Any] = value  # pyright: ignore[reportUnknownVariableType]
        return typed
    return None


def _fields(value: object) -> dict[str, Any] | None:
    return cast("dict[str, Any]", value) if isinstance(value, dict) else None


def _anthropic(doc: dict[str, Any]) -> bool:
    messages = _items(doc.get("messages"))
    if messages is None:
        return True
    for raw in messages:
        message = _fields(raw)
        if message is None:
            return True
        role, content = message.get("role"), message.get("content")
        if role == "assistant":
            continue
        if role != "user":
            return True
        if isinstance(content, str):
            if _text_outside(content):
                return True
            continue
        blocks = _items(content)
        if blocks is None:
            return True
        for raw_block in blocks:
            block = _fields(raw_block)
            if block is None or block.get("type") != "text" or _text_outside(block.get("text")):
                return True  # a tool_result, an image, a document, a block this code does not know
    return False


def _parts_outside(content: object, kind: str) -> bool:
    if isinstance(content, str):
        return _text_outside(content)
    parts = _items(content)
    if parts is None:
        return True
    for raw in parts:
        part = _fields(raw)
        if part is None or part.get("type") != kind or _text_outside(part.get("text")):
            return True
    return False


def _chat(doc: dict[str, Any]) -> bool:
    messages = _items(doc.get("messages"))
    if messages is None:
        return True
    for raw in messages:
        message = _fields(raw)
        if message is None:
            return True
        role = message.get("role")
        if role in ("system", "developer", "assistant"):
            continue
        if role != "user" or _parts_outside(message.get("content"), "text"):
            return True  # a tool or function message, a part that is not text, a role this code does not know
    return False


def _responses(doc: dict[str, Any]) -> bool:
    given = doc.get("input")
    if isinstance(given, str):
        return _text_outside(given)
    items = _items(given)
    if items is None:
        return True
    for raw in items:
        item = _fields(raw)
        if item is None:
            return True
        kind, role = item.get("type", "message"), item.get("role")
        if kind in ("function_call", "reasoning"):
            continue  # the model's own
        if kind != "message":
            return True  # function_call_output and every item this code does not know
        if role in ("system", "developer", "assistant"):
            continue
        if role != "user" or _parts_outside(item.get("content"), "input_text"):
            return True
    return False


def add_mark(headers: MutableMapping[str, Any], untrusted: bool) -> None:
    """Put the mark in ``headers`` when ``untrusted`` or when an app has marked the scope; never takes one away."""
    if untrusted or _forced.get():
        headers[HEADER] = UNTRUSTED


def _plain(item: object) -> object:
    dump = getattr(item, "model_dump", None)
    return dump() if callable(dump) else item


def responses_input_is_untrusted(value: object) -> bool:
    """For a framework that holds the request as OpenAI Responses ``input`` (a string or a list of items)."""
    if isinstance(value, list):
        value = [_plain(item) for item in cast("list[object]", value)]
    return _responses({"input": value})


#: The fields of a Gemini-style part that make it more than the person's text.
_NOT_TEXT = (
    "function_response",
    "function_call",
    "inline_data",
    "file_data",
    "executable_code",
    "code_execution_result",
    "video_metadata",
    "thought",
)


def contents_are_untrusted(contents: object) -> bool:
    """Whether Google ``Content`` objects (ADK) hold more than the person's words.

    A user turn is the person's only when every part is plain text; a function response, a file or any part this code
    does not know is not.
    """
    items = _items(contents)
    if items is None:
        return True
    for content in items:
        role = getattr(content, "role", None)
        if role in ("model", "assistant"):
            continue
        parts = getattr(content, "parts", None)
        if role not in ("user", None) or not isinstance(parts, list):
            return True
        for part in cast("list[object]", parts):
            if getattr(part, "text", None) is None or any(getattr(part, name, None) for name in _NOT_TEXT):
                return True
            if _text_outside(getattr(part, "text", None)):
                return True
    return False
