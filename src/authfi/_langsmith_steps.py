"""The LangChain run executing now, as the span LangSmith's OpenTelemetry bridge exports for it.

LangSmith turns a run into an OpenTelemetry span after the run has started, from a background thread, and gives the
span a random span id. A gateway call is made while the run executes, before its span exists, so its ``traceparent``
could not name it. Here the span id is fixed by the run: the last eight bytes of the run's UUID (LangSmith's own
mapping uses the first eight, which it also sets as the parent of a root span, so a span is never its own parent).
The trace id is the run tree's: a root run is made without the OpenTelemetry span that happened to be current when it
started (LangSmith would make it that span's child), so that the trace a call names is the one exported.

A chat model call is not the current run (LangChain leaves the graph node current, and a streamed call has none), so
a callback handler on every LangChain run (:func:`observe_model_runs`) says which model run is executing, in
:data:`MODEL_RUN`.

A LangGraph agent is a compiled graph, and its run is the step that names it: the span of a graph's run carries
``gen_ai.operation.name=invoke_agent`` and ``gen_ai.agent.name=<the graph's name>``, the name LangGraph gave the run
(``create_react_agent(name=...)``, ``StateGraph.compile(name=...)``, else its default, ``LangGraph``). A graph run is
the run its nodes hang under: LangGraph names each node run for the node and gives it a ``langgraph_checkpoint_ns``
that is its graph's, plus ``node:task``. A graph run inside another (a sub-agent) has the same relation to its own
nodes, so each graph is told apart by namespaces, never by the name of a tool. The nearest agent above a step is then
the agent that ran it.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from authfi._protocol.ids import Step
from authfi.errors import IntegrationError

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.id_generator import IdGenerator

__all__ = [
    "MODEL_RUN",
    "TESTED",
    "agent_name_of",
    "current_step",
    "observe_model_runs",
    "span_id_of",
    "use_run_ids",
]

#: The langsmith versions the private seam in :func:`use_run_ids` is tested against.
TESTED = "langsmith 0.14"

#: The id of the LangChain run of the model call being made, for the duration of the call.
MODEL_RUN: ContextVar[uuid.UUID | None] = ContextVar("authfi_model_run", default=None)

_observing = threading.Lock()
_observed = False

#: LangGraph's separator between the segments of a checkpoint namespace (``node:task|node:task``).
_NS_SEP = "|"
#: How many graph runs are remembered until their span is made, and how many open chain runs are tracked.
_MAX_RUNS = 8192

_runs_lock = threading.Lock()
#: Chain runs that have started and not ended: run id -> (checkpoint namespace, name).
_open_chains: OrderedDict[uuid.UUID, tuple[str, str]] = OrderedDict()
#: Runs that are a LangGraph graph (an agent): run id -> the graph's name.
_graph_runs: OrderedDict[uuid.UUID, str] = OrderedDict()


def span_id_of(run_id: uuid.UUID) -> int:
    """The span id of a run's span: a UUIDv7's last eight bytes are random, which a span id needs to be."""
    return int.from_bytes(run_id.bytes[8:], "big")


def agent_name_of(run_id: uuid.UUID) -> str | None:
    """The name of the graph a run is, or ``None`` when it is not a LangGraph graph (or none of its nodes has run)."""
    with _runs_lock:
        return _graph_runs.get(run_id)


def _bound(table: OrderedDict[uuid.UUID, Any]) -> None:
    while len(table) > _MAX_RUNS:
        table.popitem(last=False)


def _chain_started(run_id: uuid.UUID, parent_run_id: uuid.UUID | None, name: str | None, metadata: Any) -> None:
    """Note a chain run; when it is a graph's node, the run it hangs under is that graph.

    A node run is named for its node (``langgraph_node``) and its namespace is its graph's plus ``node:task``. A
    run inside a node, such as a sub-agent's graph run, has the namespace of the node it is in, which is not one
    segment longer than that of the run above it, so it is not taken for one of that run's nodes.
    """
    meta = cast("dict[str, Any]", metadata) if isinstance(metadata, dict) else {}
    ns = str(meta.get("langgraph_checkpoint_ns") or "")
    with _runs_lock:
        _open_chains[run_id] = (ns, name or "")
        _bound(_open_chains)
        if parent_run_id is None or meta.get("langgraph_node") is None or meta.get("langgraph_node") != name:
            return
        parent = _open_chains.get(parent_run_id)
        if parent is not None and ns.rpartition(_NS_SEP)[0] == parent[0] and parent[1]:
            _graph_runs[parent_run_id] = parent[1]
            _bound(_graph_runs)


def _chain_ended(run_id: uuid.UUID) -> None:
    with _runs_lock:
        _open_chains.pop(run_id, None)


def current_step() -> Step | None:
    """The run executing in this context, or ``None`` outside a LangChain run."""
    from langsmith import run_helpers

    tree = run_helpers.get_current_run_tree()
    model_run = MODEL_RUN.get()
    if tree is None and model_run is None:
        return None
    # A chat model called with no run around it is a root run, and a root run's trace is its own id.
    trace_id = tree.trace_id if tree is not None else cast("uuid.UUID", model_run)
    run_id = model_run or cast("Any", tree).id
    return trace_id.hex, format(span_id_of(run_id), "016x")


def observe_model_runs() -> None:
    """Have every LangChain run tell :data:`MODEL_RUN` which model call is executing, and which runs are graphs.

    Once per process.
    """
    global _observed
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tracers.context import register_configure_hook

    class ModelRuns(BaseCallbackHandler):
        # Inline, so that setting the variable happens in the context of the code that makes the model call.
        run_inline = True

        def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            MODEL_RUN.set(run_id)

        def on_llm_start(self, serialized: Any, prompts: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            MODEL_RUN.set(run_id)

        def on_llm_end(self, response: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            MODEL_RUN.set(None)

        def on_llm_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            MODEL_RUN.set(None)

        def on_chain_start(
            self,
            serialized: Any,
            inputs: Any,
            *,
            run_id: uuid.UUID,
            parent_run_id: uuid.UUID | None = None,
            metadata: Any = None,
            name: str | None = None,
            **kwargs: Any,
        ) -> None:
            _chain_started(run_id, parent_run_id, name, metadata)

        def on_chain_end(self, outputs: Any, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            _chain_ended(run_id)

        def on_chain_error(self, error: BaseException, *, run_id: uuid.UUID, **kwargs: Any) -> None:
            _chain_ended(run_id)

    with _observing:
        if _observed:
            return
        # A variable whose default is the handler hands it to every run in every context, which is how LangChain
        # takes a handler that is not passed in the run's config.
        register_configure_hook(ContextVar("authfi_model_runs", default=ModelRuns()), inheritable=True)  # noqa: B039
        _observed = True


class _RunSpanIds:
    """Hands out the span id announced for the span being made, and the wrapped generator's otherwise."""

    def __init__(self, inner: IdGenerator) -> None:
        self._inner = inner
        self._announced = threading.local()

    def announce(self, span_id: int | None) -> None:
        self._announced.span_id = span_id

    def generate_span_id(self) -> int:
        announced = cast("int | None", getattr(self._announced, "span_id", None))
        self._announced.span_id = None  # the announcement is for one span
        return announced if announced is not None else self._inner.generate_span_id()

    def generate_trace_id(self) -> int:
        return self._inner.generate_trace_id()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)  # the rest of the generator's interface, whichever version has it


def use_run_ids(client: Any) -> None:
    """Make the spans ``client``'s OpenTelemetry bridge creates carry :func:`span_id_of` their run."""
    exporter = getattr(client, "otel_exporter", None)
    tracer = getattr(exporter, "_tracer", None)
    create = getattr(exporter, "_create_span_for_run", None)
    if (
        not callable(create)
        or not callable(getattr(exporter, "_set_span_attributes", None))
        or not hasattr(tracer, "id_generator")
    ):
        raise IntegrationError(
            "authfi.langchain names the run executing in a gateway call's traceparent through the LangSmith "
            "client's OpenTelemetry bridge (otel_exporter._create_span_for_run and _set_span_attributes), which this "
            f"langsmith does not have. It is tested with {TESTED}."
        )
    ids = _RunSpanIds(cast("Any", tracer).id_generator)
    cast("Any", tracer).id_generator = ids
    base = cast("type[Any]", type(exporter))

    class RunIdExporter(base):  # type: ignore[misc, valid-type]  # the bridge keeps its state in slots: same layout
        __slots__ = ()

        def _create_span_for_run(self, op: Any, run_info: dict[str, Any], otel_context: Any = None) -> Any:
            # No ambient context: a root run keeps the trace id of its run tree, which is what a call names.
            ids.announce(span_id_of(op.id))
            try:
                return base._create_span_for_run(self, op, run_info, None)
            finally:
                ids.announce(None)

        def _set_span_attributes(self, span: Any, run_info: dict[str, Any], op: Any) -> None:
            base._set_span_attributes(self, span, run_info, op)
            # Called when the run starts and again when it ends, which is after its nodes have started: a graph is
            # known to be one by then.
            graph = agent_name_of(op.id)
            if graph is not None:
                span.set_attribute("gen_ai.operation.name", "invoke_agent")
                span.set_attribute("gen_ai.agent.name", graph)

    cast("Any", exporter).__class__ = RunIdExporter
