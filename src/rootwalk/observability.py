"""Optional, fail-open OpenTelemetry tracing for RootWalk."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Generator, TypeVar, cast

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import (
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, Sampler, TraceIdRatioBased
from opentelemetry.trace import Link, Span, SpanContext, SpanKind, Status
from opentelemetry.trace.status import StatusCode
from opentelemetry.util.types import AttributeValue

from .model import redact_sensitive

MAPPING_VERSION = "openinference-rootwalk-v1"
_EXPORT_TIMEOUT_MILLIS = 2_000
_T = TypeVar("_T")


@dataclass(frozen=True)
class ObservabilityConfig:
    enabled: bool = False
    capture_content: bool = False
    endpoint: str | None = None
    sample_ratio: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.sample_ratio <= 1.0:
            raise ValueError("sample_ratio must be between 0 and 1")
        if self.endpoint is not None and not self.endpoint.strip():
            raise ValueError("endpoint must be non-empty when provided")


class _DropCounter:
    def __init__(self) -> None:
        self._count = 0
        self._lock = threading.Lock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def add(self, count: int) -> None:
        with self._lock:
            self._count += count


class RootWalkTracerProvider(TracerProvider):
    def __init__(
        self, counter: _DropCounter, *, sampler: Sampler, resource: Resource
    ) -> None:
        super().__init__(
            sampler=sampler,
            resource=resource,
            shutdown_on_exit=False,
        )
        self._counter = counter

    @property
    def telemetry_dropped_count(self) -> int:
        return self._counter.count


class BoundedSpanExporter(SpanExporter):
    """Keep exporter failures and slow cleanup off the investigation path."""

    def __init__(
        self,
        exporter: SpanExporter,
        *,
        capture_content: bool = False,
        timeout_millis: int = _EXPORT_TIMEOUT_MILLIS,
        counter: _DropCounter | None = None,
    ) -> None:
        if timeout_millis < 1:
            raise ValueError("timeout_millis must be positive")
        self._exporter = exporter
        self._capture_content = capture_content
        self._timeout_millis = timeout_millis
        self._counter = counter or _DropCounter()
        self._state_lock = threading.Lock()
        self._precounted = 0

    @property
    def dropped_count(self) -> int:
        return self._counter.count

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            filtered = tuple(
                _filter_span(span, capture_content=self._capture_content)
                for span in spans
            )
            result = self._exporter.export(filtered)
        except Exception:
            result = SpanExportResult.FAILURE
        with self._state_lock:
            precounted = min(self._precounted, len(spans))
            self._precounted -= precounted
        if result is not SpanExportResult.SUCCESS and precounted < len(spans):
            self._counter.add(len(spans) - precounted)
        return result

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        timeout = min(max(timeout_millis, 0), self._timeout_millis)
        completed, result = _bounded_call(
            lambda: self._exporter.force_flush(timeout), timeout
        )
        return completed and (result is None or bool(result))

    def shutdown(self) -> None:
        _bounded_call(self._exporter.shutdown, self._timeout_millis)

    def record_abandoned(self, count: int) -> None:
        if count < 1:
            return
        with self._state_lock:
            self._precounted += count
        self._counter.add(count)


class _BoundedBatchSpanProcessor(SpanProcessor):
    """Small async processor because the SDK exposes no queue-drop hook."""

    def __init__(
        self,
        exporter: BoundedSpanExporter,
        counter: _DropCounter,
        *,
        max_queue_size: int,
        max_export_batch_size: int,
        schedule_delay_millis: int,
    ) -> None:
        self._exporter = exporter
        self._counter = counter
        self._queue: queue.Queue[ReadableSpan] = queue.Queue(max_queue_size)
        self._max_export_batch_size = max_export_batch_size
        self._schedule_delay_seconds = schedule_delay_millis / 1_000
        self._condition = threading.Condition()
        self._submitted = 0
        self._completed = 0
        self._shutdown = False
        self._abandon = False
        self._abandoned_counted = False
        self._exporter_shutdown = False
        self._worker = threading.Thread(
            target=self._consume,
            name="rootwalk-otel-export",
            daemon=True,
        )
        self._worker.start()

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        del span, parent_context

    def on_end(self, span: ReadableSpan) -> None:
        if span.context is None or not span.context.trace_flags.sampled:
            return
        with self._condition:
            if self._shutdown:
                self._counter.add(1)
                return
            try:
                self._queue.put_nowait(span)
            except queue.Full:
                self._counter.add(1)
                return
            self._submitted += 1

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        deadline = time.monotonic() + max(timeout_millis, 0) / 1_000
        with self._condition:
            target = self._submitted
            while self._completed < target:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._condition.wait(remaining)
        remaining_millis = int(max(deadline - time.monotonic(), 0) * 1_000)
        if remaining_millis < 1:
            return False
        return self._exporter.force_flush(remaining_millis)

    def shutdown(self) -> None:
        with self._condition:
            self._shutdown = True
        self._worker.join(_EXPORT_TIMEOUT_MILLIS / 1_000)
        with self._condition:
            worker_alive = self._worker.is_alive()
            if worker_alive:
                self._abandon = True
                outstanding = (
                    0
                    if self._abandoned_counted
                    else max(self._submitted - self._completed, 0)
                )
                self._abandoned_counted = True
            else:
                outstanding = 0
        if outstanding:
            self._exporter.record_abandoned(outstanding)
        if worker_alive:
            while True:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            return
        self._shutdown_exporter_once()

    def _consume(self) -> None:
        while True:
            with self._condition:
                abandon = self._abandon
            if abandon:
                self._shutdown_exporter_once()
                return
            try:
                first = self._queue.get(timeout=self._schedule_delay_seconds)
            except queue.Empty:
                with self._condition:
                    shutdown = self._shutdown
                    abandon = self._abandon
                if shutdown:
                    if abandon:
                        self._shutdown_exporter_once()
                    return
                continue
            batch = [first]
            while len(batch) < self._max_export_batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break
            self._exporter.export(batch)
            with self._condition:
                self._completed += len(batch)
                self._condition.notify_all()
                abandon = self._abandon
            if abandon:
                self._shutdown_exporter_once()
                return

    def _shutdown_exporter_once(self) -> None:
        with self._condition:
            if self._exporter_shutdown:
                return
            self._exporter_shutdown = True
        self._exporter.shutdown()


def create_tracer_provider(
    config: ObservabilityConfig,
    *,
    exporter: SpanExporter | None = None,
) -> RootWalkTracerProvider:
    """Create an isolated provider; disabled configuration samples nothing."""
    sampler = TraceIdRatioBased(config.sample_ratio) if config.enabled else ALWAYS_OFF
    counter = _DropCounter()
    provider = RootWalkTracerProvider(
        counter,
        sampler=sampler,
        resource=Resource.create({"service.name": "rootwalk"}),
    )
    if not config.enabled:
        return provider

    backend = exporter or OTLPSpanExporter(
        endpoint=config.endpoint,
        timeout=_EXPORT_TIMEOUT_MILLIS / 1_000,
    )
    bounded = BoundedSpanExporter(
        backend,
        capture_content=config.capture_content,
        timeout_millis=_EXPORT_TIMEOUT_MILLIS,
        counter=counter,
    )
    provider.add_span_processor(
        _BoundedBatchSpanProcessor(
            bounded,
            counter,
            max_queue_size=256,
            max_export_batch_size=64,
            schedule_delay_millis=500,
        )
    )
    return provider


def configure(
    config: ObservabilityConfig,
    *,
    exporter: SpanExporter | None = None,
) -> RootWalkTracerProvider:
    """Install and return the RootWalk tracer provider."""
    provider = create_tracer_provider(config, exporter=exporter)
    trace.set_tracer_provider(provider)
    return provider


def run_span(
    tracer: trace.Tracer,
    config: ObservabilityConfig | None = None,
    *,
    run_id: str,
    execution_attempt_id: str,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> AbstractContextManager[Span]:
    """Start one execution as a new root trace."""
    return _span(
        tracer,
        "rootwalk.run",
        config,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        attributes=attributes,
        context=Context(),
    )


def recovery_run_span(
    tracer: trace.Tracer,
    config: ObservabilityConfig | None = None,
    *,
    run_id: str,
    execution_attempt_id: str,
    previous_span_context: SpanContext,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> AbstractContextManager[Span]:
    """Start recovery in a new trace linked to the prior execution."""
    if not previous_span_context.is_valid:
        raise ValueError("previous_span_context must be valid")
    return _span(
        tracer,
        "rootwalk.run",
        config,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        attributes=attributes,
        context=Context(),
        links=(Link(previous_span_context),),
    )


def model_span(
    tracer: trace.Tracer,
    config: ObservabilityConfig | None = None,
    *,
    run_id: str,
    execution_attempt_id: str,
    model: str,
    operation_id: str | None = None,
    prompt: str | bytes | None = None,
    response: str | bytes | None = None,
    usage: Mapping[str, int | float] | None = None,
) -> AbstractContextManager[Span]:
    """Create the sole model span and its fixed OpenInference mapping."""
    resolved = config or ObservabilityConfig()
    attributes: dict[str, AttributeValue] = {
        "mapping_version": MAPPING_VERSION,
        "openinference.span.kind": "LLM",
        "llm.model_name": model,
    }
    _add_content(attributes, "input", prompt, resolved.capture_content)
    _add_content(attributes, "output", response, resolved.capture_content)
    if resolved.capture_content and prompt is not None:
        attributes["input.mime_type"] = "application/json"
    if resolved.capture_content and response is not None:
        attributes["output.mime_type"] = "application/json"
    _add_usage(attributes, usage)
    return _span(
        tracer,
        "rootwalk.model",
        resolved,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        operation_id=operation_id,
        attributes=attributes,
        kind=SpanKind.CLIENT,
    )


def record_model_result(
    span: Span,
    config: ObservabilityConfig,
    *,
    response: str | bytes | None = None,
    usage: Mapping[str, int | float] | None = None,
) -> None:
    """Attach post-call output to the same model span without creating another."""
    attributes: dict[str, AttributeValue] = {}
    _add_content(attributes, "output", response, config.capture_content)
    if config.capture_content and response is not None:
        attributes["output.mime_type"] = "application/json"
    _add_usage(attributes, usage)
    span.set_attributes(_filter_attributes(attributes, config.capture_content))


def record_span_content(
    span: Span,
    config: ObservabilityConfig,
    *,
    input_value: object | None = None,
    output_value: object | None = None,
) -> None:
    """Attach redacted structured IO using OpenInference-compatible fields."""
    attributes: dict[str, AttributeValue] = {}
    _add_structured_content(attributes, "input", input_value, config.capture_content)
    _add_structured_content(attributes, "output", output_value, config.capture_content)
    span.set_attributes(_filter_attributes(attributes, config.capture_content))


def tool_span(
    tracer: trace.Tracer,
    config: ObservabilityConfig | None = None,
    *,
    run_id: str,
    execution_attempt_id: str,
    operation_id: str,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> AbstractContextManager[Span]:
    return _span(
        tracer,
        "rootwalk.tool",
        config,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        operation_id=operation_id,
        attributes=attributes,
    )


def context_span(
    tracer: trace.Tracer,
    config: ObservabilityConfig | None = None,
    *,
    run_id: str,
    execution_attempt_id: str,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> AbstractContextManager[Span]:
    return _span(
        tracer,
        "rootwalk.context",
        config,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        attributes=attributes,
    )


def validation_span(
    tracer: trace.Tracer,
    config: ObservabilityConfig | None = None,
    *,
    run_id: str,
    execution_attempt_id: str,
    operation_id: str | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> AbstractContextManager[Span]:
    return _span(
        tracer,
        "rootwalk.validation",
        config,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        operation_id=operation_id,
        attributes=attributes,
    )


def generic_span(
    tracer: trace.Tracer,
    name: str,
    config: ObservabilityConfig | None = None,
    *,
    run_id: str,
    execution_attempt_id: str,
    operation_id: str | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> AbstractContextManager[Span]:
    """Create a RootWalk stage span using the shared safety filter."""
    return _span(
        tracer,
        name,
        config,
        run_id=run_id,
        execution_attempt_id=execution_attempt_id,
        operation_id=operation_id,
        attributes=attributes,
    )


@contextmanager
def _span(
    tracer: trace.Tracer,
    name: str,
    config: ObservabilityConfig | None,
    *,
    run_id: str,
    execution_attempt_id: str,
    operation_id: str | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
    context: Context | None = None,
    links: Sequence[Link] = (),
    kind: SpanKind = SpanKind.INTERNAL,
) -> Generator[Span]:
    capture_content = (config or ObservabilityConfig()).capture_content
    merged: dict[str, AttributeValue] = {
        "run_id": run_id,
        "execution_attempt_id": execution_attempt_id,
    }
    if operation_id is not None:
        merged["operation_id"] = operation_id
    if attributes:
        merged.update(attributes)
    with tracer.start_as_current_span(
        name,
        context=context,
        kind=kind,
        attributes=_filter_attributes(merged, capture_content),
        links=links,
        record_exception=False,
        set_status_on_exception=False,
    ) as span:
        try:
            yield span
        except BaseException as exc:
            record_span_outcome(span, "error", type(exc).__name__)
            raise


def record_span_outcome(
    span: Span, outcome: str, error_type: str | None = None
) -> None:
    """Export safe failure semantics without exception messages or stack traces."""
    span.set_attribute("outcome", outcome)
    if error_type:
        span.set_attribute("error.type", error_type)
    if error_type or outcome in {"error", "runtime_failure", "cancelled"}:
        span.set_status(Status(StatusCode.ERROR))
    else:
        span.set_status(Status(StatusCode.OK))


def _add_content(
    attributes: dict[str, AttributeValue],
    prefix: str,
    content: str | bytes | None,
    capture_content: bool,
) -> None:
    if content is None:
        return
    content = _redacted_content(content)
    raw = content.encode() if isinstance(content, str) else content
    attributes[f"{prefix}_hash"] = hashlib.sha256(raw).hexdigest()
    attributes[f"{prefix}_count"] = len(raw)
    if capture_content:
        attributes[f"{prefix}.value"] = content


def _add_structured_content(
    attributes: dict[str, AttributeValue],
    prefix: str,
    value: object | None,
    capture_content: bool,
) -> None:
    if value is None:
        return
    if isinstance(value, (str, bytes)):
        content: str | bytes = value
        mime_type = "text/plain"
    else:
        content = json.dumps(
            cast(Any, value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        mime_type = "application/json"
    _add_content(attributes, prefix, content, capture_content)
    if capture_content:
        attributes[f"{prefix}.mime_type"] = mime_type


def _redacted_content(content: str | bytes) -> str | bytes:
    is_bytes = isinstance(content, bytes)
    try:
        text = content.decode() if is_bytes else content
    except UnicodeDecodeError:
        return content
    try:
        parsed: object = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        sanitized = redact_sensitive(text)
        return cast(str, sanitized).encode() if is_bytes else cast(str, sanitized)
    sanitized = json.dumps(
        redact_sensitive(parsed),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sanitized.encode() if is_bytes else sanitized


def _add_usage(
    attributes: dict[str, AttributeValue],
    usage: Mapping[str, int | float] | None,
) -> None:
    if not usage:
        return
    aliases = {
        "input_tokens": "prompt",
        "prompt_tokens": "prompt",
        "output_tokens": "completion",
        "completion_tokens": "completion",
        "total_tokens": "total",
        "cache_read_input_tokens": "cache_read",
        "cached_tokens": "cache_read",
        "reasoning_tokens": "reasoning",
    }
    for key, value in usage.items():
        mapped = aliases.get(key)
        if mapped is not None and not isinstance(value, bool):
            attributes[f"llm.token_count.{mapped}"] = value


def _filter_attributes(
    attributes: Mapping[str, AttributeValue], capture_content: bool
) -> dict[str, AttributeValue]:
    return {
        key: value
        for key, value in attributes.items()
        if not _is_secret_key(key) and (capture_content or _is_safe_metadata(key))
    }


def _is_secret_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    if normalized.startswith("llm.token_count."):
        return False
    return normalized in {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "secret",
        "token",
        "access_token",
        "refresh_token",
        "id_token",
        "cookie",
        "cookies",
        "set_cookie",
    } or normalized.endswith(
        ("_api_key", "_authorization", "_token", "_secret", "_cookie")
    )


def _is_safe_metadata(key: str) -> bool:
    normalized = key.lower()
    return (
        normalized
        in {
            "run_id",
            "execution_attempt_id",
            "operation_id",
            "mapping_version",
            "openinference.span.kind",
            "llm.model_name",
            "tool.name",
            "outcome",
            "error.type",
        }
        or normalized.endswith(("_hash", ".hash", "_count", ".count"))
        or normalized.startswith(("usage.", "llm.token_count."))
    )


def _filter_span(span: ReadableSpan, *, capture_content: bool) -> ReadableSpan:
    status = span.status
    attributes = span.attributes or {}
    return ReadableSpan(
        name=span.name,
        context=span.context,
        parent=span.parent,
        resource=span.resource,
        attributes=_filter_attributes(attributes, capture_content),
        events=()
        if not capture_content
        else tuple(
            Event(
                event.name,
                _filter_attributes(event.attributes or {}, True),
                event.timestamp,
            )
            for event in span.events
        ),
        links=tuple(
            Link(
                link.context,
                _filter_attributes(link.attributes or {}, capture_content),
            )
            for link in span.links
        ),
        kind=span.kind,
        status=Status(status.status_code)
        if status.status_code is not StatusCode.UNSET
        else status,
        start_time=span.start_time,
        end_time=span.end_time,
        instrumentation_scope=span.instrumentation_scope,
    )


def _bounded_call(
    call: Callable[[], _T], timeout_millis: int
) -> tuple[bool, _T | None]:
    completed = threading.Event()
    result: list[_T] = []

    def invoke() -> None:
        try:
            result.append(call())
        except Exception:
            pass
        finally:
            completed.set()

    threading.Thread(target=invoke, daemon=True).start()
    finished = completed.wait(timeout_millis / 1_000)
    return finished and bool(result), result[0] if result else None
