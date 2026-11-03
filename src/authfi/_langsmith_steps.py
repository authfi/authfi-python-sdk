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
"""

from __future__ import annotations

import threading
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, cast

from authfi._protocol.ids import Step
from authfi.errors import IntegrationError

if TYPE_CHECKING:
    from opentelemetry.sdk.trace.id_generator import IdGenerator

__all__ = ["MODEL_RUN", "TESTED", "current_step", "observe_model_runs", "span_id_of", "use_run_ids"]

#: The langsmith versions the private seam in :func:`use_run_ids` is tested against.
TESTED = "langsmith 0.14"

#: The id of the LangChain run of the model call being made, for the duration of the call.
MODEL_RUN: ContextVar[uuid.UUID | None] = ContextVar("authfi_model_run", default=None)

_observing = threading.Lock()
_observed = False


def span_id_of(run_id: uuid.UUID) -> int:
    """The span id of a run's span: a UUIDv7's last eight bytes are random, which a span id needs to be."""
    return int.from_bytes(run_id.bytes[8:], "big")


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
    """Have every LangChain run tell :data:`MODEL_RUN` which model call is executing. Once per process."""
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
    if not callable(create) or not hasattr(tracer, "id_generator"):
        raise IntegrationError(
            "authfi.langchain names the run executing in a gateway call's traceparent through the LangSmith "
            f"client's OpenTelemetry bridge (otel_exporter._create_span_for_run), which this langsmith does not "
            f"have. It is tested with {TESTED}."
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

    cast("Any", exporter).__class__ = RunIdExporter
