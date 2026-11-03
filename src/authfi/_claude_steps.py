"""Names the agent on the spans Claude Code exports, in the OpenTelemetry GenAI standard.

Claude Code's trace says which sub-agent made a step (``agent_id`` on its spans, and the sub-agent's id as
``gen_ai.conversation.id``) but never what the sub-agent is called, and it names no agent for the main loop. The SDK's
``SubagentStart`` hook is told both, ``agent_id`` and ``agent_type`` (the sub-agent's name in ``agents=``), and the
tool hooks repeat them. Claude Code exports its trace through the loopback proxy of :mod:`authfi._model_proxy`
(``export_steps``), which passes each export through :meth:`AgentNames.rewrite` on its way to AuthFI.

Each span then says ``gen_ai.agent.name``: the sub-agent's type on every span of a sub-agent, and the main loop's name
on every other span. Nothing else in a span is changed or read. A span that already says ``gen_ai.agent.name`` keeps it.
The agent behind a gateway call is then the nearest ``gen_ai.agent.name`` at or above the step that made it.
"""

from __future__ import annotations

import gzip
import importlib
import zlib
from collections import OrderedDict

__all__ = ["DEFAULT_MAIN_NAME", "AgentNames", "available"]

#: What Claude Code calls itself in its own trace (``claude_code.interaction``): the name of the main loop when the
#: application gives it none.
DEFAULT_MAIN_NAME = "claude_code"

#: How many sub-agents it remembers the type of.
_MAX_AGENTS = 4096
#: The most an export may grow to when decompressed.
_MAX_DECODED = 256 * 1024 * 1024


def available() -> bool:
    """``True`` when the OpenTelemetry protocol types the rewrite needs are installed (extra ``otel``)."""
    try:
        importlib.import_module("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    except ImportError:
        return False
    return True


def _decoded(body: bytes, encoding: str) -> bytes | None:
    """``body`` without its content encoding, or ``None`` when it has one this does not read."""
    kind = encoding.strip().lower()
    try:
        if kind in ("", "identity"):
            return body
        if kind in ("gzip", "x-gzip"):
            reader = zlib.decompressobj(16 + zlib.MAX_WBITS)
        elif kind == "deflate":
            reader = zlib.decompressobj()
        else:
            return None
        out = reader.decompress(body, _MAX_DECODED)
        return None if reader.unconsumed_tail else out  # larger than allowed
    except (zlib.error, EOFError, gzip.BadGzipFile):
        return None


class AgentNames:
    """What the application's agents are called, learned from Claude Code's hooks, and the rewrite that applies it.

    Args:
        main: the name of the main loop's agent.
    """

    def __init__(self, main: str = DEFAULT_MAIN_NAME) -> None:
        self.main = main
        self._types: OrderedDict[str, str] = OrderedDict()

    def learn(self, agent_id: object, agent_type: object) -> None:
        """Remember that sub-agent ``agent_id`` is of type ``agent_type``; anything else is ignored."""
        if not isinstance(agent_id, str) or not isinstance(agent_type, str) or not agent_id or not agent_type:
            return
        self._types[agent_id] = agent_type
        self._types.move_to_end(agent_id)
        while len(self._types) > _MAX_AGENTS:
            self._types.popitem(last=False)

    def name_of(self, agent_id: str) -> str:
        """The type of sub-agent ``agent_id``; its id when its start was not seen, never its parent's name."""
        return self._types.get(agent_id, agent_id)

    def rewrite(self, body: bytes, encoding: str = "") -> bytes | None:
        """``body``, an OTLP trace export, with ``gen_ai.agent.name`` on each span; ``None`` when it cannot be read.

        The result is not compressed. A caller that gets ``None`` sends the export on as it was.
        """
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest
        from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue

        plain = _decoded(body, encoding)
        if plain is None:
            return None
        request = ExportTraceServiceRequest()
        try:
            request.ParseFromString(plain)
        except Exception:  # a protobuf DecodeError; the export is not ours to refuse
            return None
        for resource in request.resource_spans:
            for scope in resource.scope_spans:
                for span in scope.spans:
                    seen: dict[str, str] = {
                        kv.key: kv.value.string_value for kv in span.attributes if kv.value.HasField("string_value")
                    }
                    if "gen_ai.agent.name" in seen:
                        continue
                    sub_agent = self._sub_agent_of(seen)
                    if sub_agent is None:
                        span.attributes.append(
                            KeyValue(key="gen_ai.agent.name", value=AnyValue(string_value=self.main))
                        )
                        continue
                    span.attributes.append(
                        KeyValue(key="gen_ai.agent.name", value=AnyValue(string_value=self.name_of(sub_agent)))
                    )
                    if "gen_ai.agent.id" not in seen:
                        span.attributes.append(KeyValue(key="gen_ai.agent.id", value=AnyValue(string_value=sub_agent)))
        return bytes(request.SerializeToString())

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
