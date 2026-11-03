"""Names the agent on the spans Claude Code exports, in the OpenTelemetry GenAI standard.

Claude Code's trace says which sub-agent made a step (``agent_id`` on its spans, and the sub-agent's id as
``gen_ai.conversation.id``) but never what the sub-agent is called, and it names no agent for the main loop. The SDK's
``SubagentStart`` hook is told both, ``agent_id`` and ``agent_type`` (the sub-agent's name in ``agents=``), and the
tool hooks repeat them. Claude Code exports its trace through the loopback proxy of :mod:`authfi._model_proxy`
(``export_steps``), which passes each export through :meth:`AgentNames.rewrite` on its way to AuthFI.

Each span then says ``gen_ai.agent.name``: the sub-agent's type on every span of a sub-agent, and the main loop's name
on every other span. Nothing else in a span is changed or read. A span that already says ``gen_ai.agent.name`` keeps it.
The agent behind a gateway call is then the nearest ``gen_ai.agent.name`` at or above the step that made it.

An export is untrusted input to a process that runs the application, so it is bounded before it is built: a body cap
and a decoded-size cap, a count of its spans and fields taken from the wire format before any message is made, and an
output cap. An export over a cap, malformed, truncated or in an encoding this does not read raises
:class:`~authfi._model_proxy.TraceRefused`; it is never passed on unnamed, which would read a sub-agent's calls as its
parent's.
"""

from __future__ import annotations

import importlib
import threading
import zlib
from collections import OrderedDict

from authfi._model_proxy import TraceRefused

__all__ = ["DEFAULT_MAIN_NAME", "MAX_BODY", "AgentNames", "available"]

#: What Claude Code calls itself in its own trace (``claude_code.interaction``): the name of the main loop when the
#: application gives it none.
DEFAULT_MAIN_NAME = "claude_code"

#: How many sub-agents it remembers the type of.
_MAX_AGENTS = 4096
#: The most an export may be, as it arrives, once decompressed, and once named. (Claude Code exports at most a few
#: hundred spans a second, tens of kilobytes.)
MAX_BODY = 4 * 1024 * 1024
MAX_DECODED = 16 * 1024 * 1024
MAX_OUTPUT = 20 * 1024 * 1024
#: The most spans, and the most fields of resources, scopes and spans together, an export may hold.
MAX_SPANS = 4096
MAX_FIELDS = 262_144


def available() -> bool:
    """``True`` when the OpenTelemetry protocol types the rewrite needs are installed (extra ``otel``)."""
    try:
        importlib.import_module("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    except ImportError:
        return False
    return True


def _decoded(body: bytes, encoding: str) -> bytes:
    """``body`` without its content encoding; raises :class:`TraceRefused` when it is over a cap or cannot be read."""
    if len(body) > MAX_BODY:
        raise TraceRefused(413, "body-too-large", "the export is too large")
    kind = encoding.strip().lower()
    if kind in ("", "identity"):
        return body
    if kind in ("gzip", "x-gzip"):
        reader = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif kind == "deflate":
        reader = zlib.decompressobj()
    else:
        raise TraceRefused(415, "encoding", "the export's content encoding is not one this reads")
    try:
        out = reader.decompress(body, MAX_DECODED + 1)  # never more than the cap and one byte is made
    except zlib.error as error:
        raise TraceRefused(400, "malformed", "the export is not valid compressed data") from error
    if len(out) > MAX_DECODED:
        raise TraceRefused(413, "decoded-too-large", "the export is too large once decompressed")
    if not reader.eof:
        raise TraceRefused(400, "truncated", "the compressed export is cut short")
    return out


def _varint(buf: bytes, at: int) -> tuple[int, int]:
    value = shift = 0
    while at < len(buf) and shift < 64:
        byte = buf[at]
        at += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, at
        shift += 7
    raise TraceRefused(400, "malformed", "the export is not valid")


def _fields(buf: bytes, start: int, end: int, budget: list[int]) -> list[tuple[int, int, int]]:
    """``(field, start, end)`` of each length-delimited field of ``buf[start:end]``, one of ``budget`` per field.

    Reads the wire format only and never makes a message, so the work is bounded by ``budget``, not by what the
    fields contain.
    """
    found: list[tuple[int, int, int]] = []
    at = start
    while at < end:
        tag, at = _varint(buf, at)
        budget[0] -= 1
        if budget[0] < 0:
            raise TraceRefused(413, "too-many-fields", "the export has too many fields")
        kind = tag & 7
        if kind == 0:
            _, at = _varint(buf, at)
        elif kind == 1:
            at += 8
        elif kind == 5:
            at += 4
        elif kind == 2:
            size, at = _varint(buf, at)
            if at + size > end:
                raise TraceRefused(400, "malformed", "the export is not valid")
            found.append((tag >> 3, at, at + size))
            at += size
        else:
            raise TraceRefused(400, "malformed", "the export is not valid")
        if at > end:
            raise TraceRefused(400, "malformed", "the export is not valid")
    return found


def _check_shape(buf: bytes) -> None:
    """Refuse an export with more spans or fields than a real one has, before a message is made of it."""
    budget = [MAX_FIELDS]
    spans = 0
    for number, a, b in _fields(buf, 0, len(buf), budget):
        if number != 1:  # ExportTraceServiceRequest.resource_spans
            continue
        for number2, a2, b2 in _fields(buf, a, b, budget):
            if number2 != 2:  # ResourceSpans.scope_spans
                continue
            for number3, a3, b3 in _fields(buf, a2, b2, budget):
                if number3 != 2:  # ScopeSpans.spans
                    continue
                spans += 1
                if spans > MAX_SPANS:
                    raise TraceRefused(413, "too-many-spans", "the export has too many spans")
                _fields(buf, a3, b3, budget)  # a span's own fields (attributes, events, links)


class AgentNames:
    """What the application's agents are called, learned from Claude Code's hooks, and the rewrite that applies it.

    Args:
        main: the name of the main loop's agent.
    """

    def __init__(self, main: str = DEFAULT_MAIN_NAME) -> None:
        self.main = main
        self._types: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()  # the hooks learn on the event loop while an export is named on a worker thread

    def learn(self, agent_id: object, agent_type: object) -> None:
        """Remember that sub-agent ``agent_id`` is of type ``agent_type``; anything else is ignored."""
        if not isinstance(agent_id, str) or not isinstance(agent_type, str) or not agent_id or not agent_type:
            return
        with self._lock:
            self._types[agent_id] = agent_type
            self._types.move_to_end(agent_id)
            while len(self._types) > _MAX_AGENTS:
                self._types.popitem(last=False)

    def name_of(self, agent_id: str) -> str:
        """The type of sub-agent ``agent_id``; its id when its start was not seen, never its parent's name."""
        with self._lock:
            return self._types.get(agent_id, agent_id)

    def rewrite(self, body: bytes, encoding: str = "") -> bytes:
        """``body``, an OTLP trace export, with ``gen_ai.agent.name`` on each span, not compressed.

        Raises :class:`~authfi._model_proxy.TraceRefused` for an export over a cap or that cannot be read. Blocking and
        CPU bound: run it off the event loop.
        """
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

        plain = _decoded(body, encoding)
        _check_shape(plain)
        request = ExportTraceServiceRequest()
        try:
            request.ParseFromString(plain)
        except Exception as error:  # a protobuf DecodeError
            raise TraceRefused(400, "malformed", "the export is not valid") from error
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    keys = {kv.key for kv in span.attributes}
                    if "gen_ai.agent.name" in keys:  # whatever its type: it is theirs
                        continue
                    seen = {
                        kv.key: kv.value.string_value for kv in span.attributes if kv.value.HasField("string_value")
                    }
                    sub_agent = self._sub_agent_of(seen)
                    if sub_agent is None:
                        span.attributes.append(
                            KeyValue(key="gen_ai.agent.name", value=AnyValue(string_value=self.main))
                        )
                        continue
                    span.attributes.append(
                        KeyValue(key="gen_ai.agent.name", value=AnyValue(string_value=self.name_of(sub_agent)))
                    )
                    if "gen_ai.agent.id" not in keys:
                        span.attributes.append(KeyValue(key="gen_ai.agent.id", value=AnyValue(string_value=sub_agent)))
        out = bytes(request.SerializeToString())
        if len(out) > MAX_OUTPUT:
            raise TraceRefused(413, "output-too-large", "the export is too large once its spans are named")
        return out

    @staticmethod
    def _sub_agent_of(attributes: dict[str, str]) -> str | None:
        """The sub-agent a span belongs to, by ``agent_id``, else by a conversation that is not the session."""
        named = attributes.get("agent_id")
        if named:
            return named
        conversation = attributes.get("gen_ai.conversation.id")
        session = attributes.get("session.id")
        if conversation and session and conversation != session:
            return conversation
        return None
