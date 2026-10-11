import asyncio
import json
import threading
import time
from collections.abc import Callable, Sequence
from typing import cast

from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from rootwalk.observability import (
    BoundedSpanExporter,
    ObservabilityConfig,
    context_span,
    create_tracer_provider,
    model_span,
    record_span_content,
    recovery_run_span,
    run_span,
    tool_span,
    validation_span,
)


def _tracer() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = create_tracer_provider(
        ObservabilityConfig(enabled=True), exporter=exporter
    )
    return provider, exporter


def test_model_span_is_single_child_of_run() -> None:
    provider, exporter = _tracer()
    tracer = provider.get_tracer("test")

    with run_span(tracer, run_id="run-1", execution_attempt_id="execution-1") as root:
        with model_span(
            tracer,
            run_id="run-1",
            execution_attempt_id="execution-1",
            model="recorded",
            prompt="private prompt",
            response="private response",
            usage={"prompt_tokens": 3, "completion_tokens": 5},
        ):
            pass
        with context_span(
            tracer,
            run_id="run-1",
            execution_attempt_id="execution-1",
            attributes={"evidence_count": 2},
        ):
            pass

    assert provider.force_flush(1_000)
    spans = exporter.get_finished_spans()
    models = [span for span in spans if span.name == "rootwalk.model"]
    assert len(models) == 1
    assert models[0].parent is not None
    assert root.get_span_context().span_id == models[0].parent.span_id
    assert (models[0].attributes or {})["mapping_version"] == (
        "openinference-rootwalk-v1"
    )


def test_concurrent_children_and_recovery_link() -> None:
    provider, exporter = _tracer()
    tracer = provider.get_tracer("test")

    async def execute() -> tuple[int, int]:
        with run_span(
            tracer, run_id="run-2", execution_attempt_id="execution-1"
        ) as root:

            async def child(index: int) -> None:
                with tool_span(
                    tracer,
                    run_id="run-2",
                    execution_attempt_id="execution-1",
                    operation_id=f"operation-{index}",
                ):
                    await asyncio.sleep(0)

            await asyncio.gather(child(1), child(2))
            first = root.get_span_context()

        with recovery_run_span(
            tracer,
            run_id="run-2",
            execution_attempt_id="execution-2",
            previous_span_context=first,
        ) as recovered:
            with validation_span(
                tracer,
                run_id="run-2",
                execution_attempt_id="execution-2",
                operation_id="operation-3",
            ):
                pass
            second = recovered.get_span_context()
        return first.trace_id, second.trace_id

    first_trace, second_trace = asyncio.run(execute())
    assert first_trace != second_trace
    assert provider.force_flush(1_000)

    spans = exporter.get_finished_spans()
    first_root = next(
        span
        for span in spans
        if span.name == "rootwalk.run"
        and (span.attributes or {})["execution_attempt_id"] == "execution-1"
    )
    recovered = next(
        span
        for span in spans
        if span.name == "rootwalk.run"
        and (span.attributes or {})["execution_attempt_id"] == "execution-2"
    )
    tools = [span for span in spans if span.name == "rootwalk.tool"]
    assert len(tools) == 2
    assert all(
        span.parent is not None
        and first_root.context is not None
        and span.parent.span_id == first_root.context.span_id
        for span in tools
    )
    assert recovered.parent is None
    assert len(recovered.links) == 1
    assert first_root.context is not None
    assert recovered.links[0].context.span_id == first_root.context.span_id


def test_capture_content_false_filters_content_and_api_keys() -> None:
    provider, exporter = _tracer()
    tracer = provider.get_tracer("test")

    with run_span(
        tracer,
        run_id="run-private",
        execution_attempt_id="execution-private",
        attributes={
            "log": "private log body",
            "evidence": "private evidence body",
            "api_key": "never-export-this",
            "artifact_hash": "a" * 64,
        },
    ):
        with model_span(
            tracer,
            run_id="run-private",
            execution_attempt_id="execution-private",
            model="safe-model",
            prompt="private prompt body",
            response="private response body",
            usage={"prompt_tokens": 7, "completion_tokens": 11},
        ):
            pass

    assert provider.force_flush(1_000)
    serialized = " ".join(
        f"{key}={value}"
        for span in exporter.get_finished_spans()
        for key, value in (span.attributes or {}).items()
    )
    for secret in (
        "private prompt body",
        "private response body",
        "private log body",
        "private evidence body",
        "never-export-this",
    ):
        assert secret not in serialized
    assert "artifact_hash=" in serialized
    assert "llm.model_name=safe-model" in serialized
    assert "llm.token_count.prompt=7" in serialized


def test_capture_content_true_exports_redacted_structured_io() -> None:
    exporter = InMemorySpanExporter()
    config = ObservabilityConfig(enabled=True, capture_content=True)
    provider = create_tracer_provider(config, exporter=exporter)
    tracer = provider.get_tracer("test.content")

    with tool_span(
        tracer,
        config,
        run_id="content-run",
        execution_attempt_id="content-attempt",
        operation_id="tool:1",
        attributes={"tool.name": "query_metric"},
    ) as span:
        record_span_content(
            span,
            config,
            input_value={"metric": "cpu", "api_key": "never-export-this"},
            output_value={"status": "success", "value": 42},
        )

    assert provider.force_flush(1_000)
    attributes = exporter.get_finished_spans()[0].attributes or {}
    assert attributes["input.mime_type"] == "application/json"
    assert attributes["output.mime_type"] == "application/json"
    assert json.loads(cast(str, attributes["input.value"])) == {
        "api_key": "[REDACTED]",
        "metric": "cpu",
    }
    assert json.loads(cast(str, attributes["output.value"])) == {
        "status": "success",
        "value": 42,
    }
    provider.shutdown()


class _FailingExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        del spans
        raise RuntimeError("backend unavailable")

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        del timeout_millis
        time.sleep(0.2)
        return True

    def shutdown(self) -> None:
        raise RuntimeError("backend unavailable")


def test_exporter_failure_is_fail_open_and_flush_is_bounded() -> None:
    exporter = BoundedSpanExporter(_FailingExporter(), timeout_millis=20)

    assert exporter.export([ReadableSpan("test")]) is SpanExportResult.FAILURE
    assert exporter.dropped_count == 1
    started = time.monotonic()
    assert exporter.force_flush(1_000) is False
    assert time.monotonic() - started < 0.15
    exporter.shutdown()

    provider = create_tracer_provider(
        ObservabilityConfig(enabled=True), exporter=_FailingExporter()
    )
    with provider.get_tracer("test").start_as_current_span("failed-export"):
        pass
    assert provider.force_flush(1_000)
    assert provider.telemetry_dropped_count == 1
    provider.shutdown()


class _BlockingExporter(SpanExporter):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        del spans
        self.started.set()
        self.release.wait(2)
        return SpanExportResult.SUCCESS


def test_telemetry_dropped_count_includes_queue_saturation() -> None:
    exporter = _BlockingExporter()
    provider = create_tracer_provider(
        ObservabilityConfig(enabled=True), exporter=exporter
    )
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("block-export"):
        pass
    assert exporter.started.wait(1)
    for index in range(300):
        with tracer.start_as_current_span(f"queued-{index}"):
            pass

    assert provider.telemetry_dropped_count > 0
    exporter.release.set()
    assert provider.force_flush(2_000)
    provider.shutdown()


class _FlushExporter(SpanExporter):
    def __init__(self, result: bool) -> None:
        self.result = result
        self.timeout_millis: int | None = None

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        del spans
        return SpanExportResult.SUCCESS

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        self.timeout_millis = timeout_millis
        return self.result


def test_processor_flushes_exporter_with_remaining_deadline() -> None:
    for downstream_result in (True, False):
        exporter = _FlushExporter(downstream_result)
        provider = create_tracer_provider(
            ObservabilityConfig(enabled=True), exporter=exporter
        )
        with provider.get_tracer("test").start_as_current_span("flush"):
            pass

        assert provider.force_flush(1_000) is downstream_result
        assert exporter.timeout_millis is not None
        assert 0 < exporter.timeout_millis < 1_000
        provider.shutdown()


class _StuckExporter(SpanExporter):
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        self.shutdown_finished = threading.Event()
        self.shutdown_calls = 0
        self.concurrent_shutdown = False

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        del spans
        self.started.set()
        self.release.wait(5)
        self.finished.set()
        return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        self.shutdown_calls += 1
        self.concurrent_shutdown = not self.finished.is_set()
        self.shutdown_finished.set()


def test_shutdown_timeout_counts_drops_without_concurrent_exporter_shutdown() -> None:
    exporter = _StuckExporter()
    provider = create_tracer_provider(
        ObservabilityConfig(enabled=True), exporter=exporter
    )
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("in-flight"):
        pass
    assert exporter.started.wait(1)
    for index in range(10):
        with tracer.start_as_current_span(f"queued-{index}"):
            pass

    started = time.monotonic()
    provider.shutdown()
    elapsed = time.monotonic() - started

    assert elapsed < 2.5
    assert provider.telemetry_dropped_count == 11
    assert exporter.shutdown_calls == 0
    assert not exporter.concurrent_shutdown
    exporter.release.set()
    assert exporter.shutdown_finished.wait(1)
    assert exporter.shutdown_calls == 1
    assert not exporter.concurrent_shutdown
    assert provider.telemetry_dropped_count == 11


def test_otel_on_off_overhead_is_measured_without_threshold(
    record_property: Callable[[str, object], None],
) -> None:
    measurements: dict[bool, int] = {}
    span_count = 128
    for enabled in (False, True):
        provider = create_tracer_provider(
            ObservabilityConfig(enabled=enabled), exporter=InMemorySpanExporter()
        )
        tracer = provider.get_tracer("overhead")
        started = time.perf_counter_ns()
        for index in range(span_count):
            with run_span(
                tracer,
                run_id=f"overhead-{index}",
                execution_attempt_id="one",
            ):
                pass
        assert provider.force_flush(2_000)
        measurements[enabled] = time.perf_counter_ns() - started
        provider.shutdown()

    record_property("otel_disabled_ns_per_span", measurements[False] / span_count)
    record_property("otel_enabled_ns_per_span", measurements[True] / span_count)
    record_property(
        "otel_overhead_ns_per_span",
        (measurements[True] - measurements[False]) / span_count,
    )
    assert all(duration > 0 for duration in measurements.values())
