import asyncio
import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pytest import MonkeyPatch

from rootwalk.contracts import Decision
from rootwalk.model import Message, ModelCallResult
from rootwalk.observability import BoundedSpanExporter, ObservabilityConfig
from rootwalk.openrca import BankEnvironment
from rootwalk.runtime import RootWalkRuntime, RuntimeConfig
from rootwalk.store import ArtifactStore, RunStore

PUBLIC_INPUT = (
    Path(__file__).resolve().parents[2]
    / "rootwalk-handoff"
    / "rootwalk-development-handoff"
    / "rootwalk-data-prep"
    / "slices"
    / "bank-row-0001"
    / "public_input"
)


class SequenceModel:
    def __init__(self, responses: list[ModelCallResult]) -> None:
        self.responses = responses
        self.calls = 0

    async def decide(self, messages: Sequence[Message]) -> ModelCallResult:
        del messages
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _response(operation: dict[str, Any]) -> ModelCallResult:
    decision = Decision.model_validate(
        {"decision_summary": "private response body", "operations": [operation]}
    )
    return ModelCallResult(
        decision=decision,
        usage={"prompt_tokens": 7, "completion_tokens": 3},
        provider_model="test-model",
        raw_response={
            "decision": decision.model_dump(mode="json"),
            "private": "private response body",
        },
    )


def _tracer() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _runtime(
    tmp_path: Path,
    model: SequenceModel,
    provider: TracerProvider,
    *,
    enabled: bool = True,
) -> tuple[RootWalkRuntime, RunStore]:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        model,
        store,
        ArtifactStore(tmp_path / "artifacts"),
        tracer=provider.get_tracer("test.runtime"),
        observability_config=ObservabilityConfig(enabled=enabled),
    )
    return runtime, store


def test_runtime_spans_are_children_and_model_content_is_redacted(
    tmp_path: Path,
) -> None:
    provider, exporter = _tracer()
    model = SequenceModel(
        [_response({"type": "finalize", "abstain_reason": "private response body"})]
    )
    runtime, store = _runtime(tmp_path, model, provider)

    result = asyncio.run(
        runtime.run(
            "observed-run",
            tmp_path / "bundle",
            execution_attempt_id="attempt-fixed",
        )
    )

    spans = exporter.get_finished_spans()
    root = next(span for span in spans if span.name == "rootwalk.run")
    graph = next(span for span in spans if span.name == "rootwalk.graph")
    models = [span for span in spans if span.name == "rootwalk.model"]
    assert result.status != "runtime_failure"
    assert model.calls == len(models) == 1
    assert root.context is not None and graph.context is not None
    assert all(
        span.parent is not None and span.parent.span_id == graph.context.span_id
        for span in spans
        if span.name in {"rootwalk.context", "rootwalk.model", "rootwalk.validation"}
    )
    assert Counter(span.name for span in spans) == {
        "rootwalk.run": 1,
        "rootwalk.persistence": 2,
        "rootwalk.graph": 1,
        "rootwalk.context": 1,
        "rootwalk.model": 1,
        "rootwalk.validation": 2,
        "rootwalk.export": 1,
    }
    root_children = [
        span
        for span in spans
        if span.parent is not None and span.parent.span_id == root.context.span_id
    ]
    assert Counter(span.name for span in root_children) == {
        "rootwalk.persistence": 1,
        "rootwalk.graph": 1,
        "rootwalk.export": 1,
    }
    assert all(
        (span.attributes or {})["execution_attempt_id"] == "attempt-fixed"
        for span in spans
    )
    model_attributes = models[0].attributes or {}
    assert model_attributes["llm.model_name"] == "test-model"
    assert model_attributes["llm.token_count.prompt"] == 7
    assert model_attributes["llm.token_count.completion"] == 3
    assert "input_hash" in model_attributes and "output_hash" in model_attributes
    serialized = repr([(span.attributes, span.events) for span in spans])
    assert "private response body" not in serialized
    assert "input.value" not in serialized and "output.value" not in serialized
    store.close()
    provider.shutdown()


def test_tool_attempt_has_operation_id_and_one_model_span_per_call(
    tmp_path: Path,
) -> None:
    provider, exporter = _tracer()
    metric_evidence_id = "tool-run:evidence:decision:tool-run:0001:0"
    log_evidence_id = "tool-run:evidence:decision:tool-run:0002:0"
    hypothesis_id = "tool-run:hypothesis:decision:tool-run:0003:0"
    path_id = "tool-run:path:decision:tool-run:0004:0"
    model = SequenceModel(
        [
            _response(
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                    },
                }
            ),
            _response(
                {
                    "type": "query",
                    "tool": "query_log_samples",
                    "arguments": {"entity_id": "bank:entity:Tomcat01"},
                }
            ),
            _response(
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "source-backed metric result",
                    "status": "supported",
                    "supporting_evidence_refs": [
                        metric_evidence_id,
                        log_evidence_id,
                    ],
                }
            ),
            _response(
                {
                    "type": "submit_path",
                    "hypothesis_id": hypothesis_id,
                    "root_cause_entity_id": "bank:entity:Tomcat01",
                    "symptom_entity_ids": ["bank:entity:Tomcat01"],
                    "steps": [],
                }
            ),
            _response(
                {
                    "type": "finalize",
                    "hypothesis_ids": [hypothesis_id],
                    "path_ids": [path_id],
                    "unresolved_questions": ["collection completeness is unknown"],
                }
            ),
        ]
    )
    runtime, store = _runtime(tmp_path, model, provider)
    runtime.environment.task = runtime.environment.task.model_copy(
        update={"entry_entities": ["bank:entity:Tomcat01"]}
    )

    asyncio.run(runtime.run("tool-run", tmp_path / "bundle"))

    spans = exporter.get_finished_spans()
    tools = [span for span in spans if span.name == "rootwalk.tool"]
    roots = [span for span in spans if span.name == "rootwalk.run"]
    graph = next(span for span in spans if span.name == "rootwalk.graph")
    assert len(tools) == 2
    assert len(roots) == 1
    assert len([span for span in spans if span.name == "rootwalk.model"]) == model.calls
    assert Counter(span.name for span in spans) == {
        "rootwalk.run": 1,
        "rootwalk.persistence": 2,
        "rootwalk.graph": 1,
        "rootwalk.context": 5,
        "rootwalk.model": 5,
        "rootwalk.validation": 6,
        "rootwalk.tool": 2,
        "rootwalk.evidence": 2,
        "rootwalk.investigation_graph": 2,
        "rootwalk.export": 1,
    }
    assert sorted(str((span.attributes or {})["operation_id"]) for span in tools) == [
        "tool-run:decision:tool-run:0001:0",
        "tool-run:decision:tool-run:0002:0",
    ]
    assert graph.context is not None
    assert all(
        span.parent is not None and span.parent.span_id == graph.context.span_id
        for span in tools
    )
    store.close()
    provider.shutdown()


def test_disabled_observability_emits_no_spans(tmp_path: Path) -> None:
    provider, exporter = _tracer()
    model = SequenceModel([_response({"type": "finalize", "abstain_reason": "done"})])
    runtime, store = _runtime(tmp_path, model, provider, enabled=False)

    asyncio.run(runtime.run("disabled-run", tmp_path / "bundle"))

    assert exporter.get_finished_spans() == ()
    store.close()
    provider.shutdown()


def test_full_content_populates_root_context_model_tool_and_evidence(
    tmp_path: Path,
) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(BoundedSpanExporter(exporter, capture_content=True))
    )
    config = ObservabilityConfig(enabled=True, capture_content=True)
    model = SequenceModel(
        [
            _response(
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {},
                }
            ),
            _response({"type": "finalize", "abstain_reason": "content test"}),
        ]
    )
    store = RunStore(tmp_path / "content" / "run.sqlite3")
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            store,
            ArtifactStore(tmp_path / "content" / "artifacts"),
            tracer=provider.get_tracer("test.full-content"),
            observability_config=config,
        ).run("full-content-run", tmp_path / "content" / "bundle")
    )

    spans = exporter.get_finished_spans()
    by_name = {
        name: [span for span in spans if span.name == name]
        for name in {
            "rootwalk.run",
            "rootwalk.context",
            "rootwalk.model",
            "rootwalk.tool",
            "rootwalk.evidence",
            "rootwalk.persistence",
        }
    }
    assert result.status == "completed_partial"
    assert all(by_name.values())
    root_attributes = by_name["rootwalk.run"][0].attributes or {}
    assert json.loads(cast(str, root_attributes["input.value"]))["run_id"] == (
        "full-content-run"
    )
    assert json.loads(cast(str, root_attributes["output.value"]))["status"] == (
        "completed_partial"
    )
    context_attributes = by_name["rootwalk.context"][0].attributes or {}
    assert json.loads(cast(str, context_attributes["output.value"]))["messages"]
    model_attributes = by_name["rootwalk.model"][0].attributes or {}
    assert json.loads(cast(str, model_attributes["input.value"]))
    assert "decision" in json.loads(cast(str, model_attributes["output.value"]))
    tool_attributes = by_name["rootwalk.tool"][0].attributes or {}
    assert json.loads(cast(str, tool_attributes["input.value"]))["tool"] == (
        "summarize_anomalies"
    )
    assert json.loads(cast(str, tool_attributes["output.value"]))["status"] in {
        "success",
        "partial",
    }
    evidence_attributes = by_name["rootwalk.evidence"][0].attributes or {}
    assert "observation_id" in json.loads(cast(str, evidence_attributes["input.value"]))
    graph = next(span for span in spans if span.name == "rootwalk.graph")
    assert "route" in json.loads(cast(str, (graph.attributes or {})["input.value"]))
    export = next(span for span in spans if span.name == "rootwalk.export")
    assert (
        json.loads(cast(str, (export.attributes or {})["output.value"]))["status"]
        == "exported"
    )
    persistence = [span for span in spans if span.name == "rootwalk.persistence"]
    assert all("output.value" in (span.attributes or {}) for span in persistence)
    store.close()
    provider.shutdown()


def test_observability_provenance_is_hashed_reported_and_frozen(
    tmp_path: Path,
) -> None:
    endpoint = "https://collector.invalid/v1/traces?token=never-persist"
    config = ObservabilityConfig(
        enabled=True,
        capture_content=True,
        endpoint=endpoint,
        sample_ratio=0.25,
    )
    provider, _exporter = _tracer()
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        SequenceModel([_response({"type": "finalize", "abstain_reason": "done"})]),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        tracer=provider.get_tracer("test.runtime"),
        observability_config=config,
    )

    result = asyncio.run(runtime.run("otel-frozen", tmp_path / "bundle"))

    expected = {
        "enabled": True,
        "capture_content": True,
        "sample_ratio": 0.25,
        "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
    }
    manifest = json.loads((result.bundle_path / "manifest.json").read_text())
    report = json.loads((result.bundle_path / "report.json").read_text())
    assert manifest["observability"] == report["observability"] == expected
    assert endpoint not in "".join(
        path.read_text(errors="ignore")
        for path in result.bundle_path.rglob("*")
        if path.is_file()
    )
    store.close()

    frozen_store = RunStore(tmp_path / "frozen.sqlite3")
    frozen_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        SequenceModel([]),
        frozen_store,
        ArtifactStore(tmp_path / "frozen-artifacts"),
        observability_config=config,
    )
    frozen_runtime._run_id = "otel-resume"  # pyright: ignore[reportPrivateUsage]
    frozen_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    frozen_store.close()

    resumed_store = RunStore(tmp_path / "frozen.sqlite3")
    with pytest.raises(ValueError, match="different observability"):
        asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                SequenceModel([]),
                resumed_store,
                ArtifactStore(tmp_path / "frozen-artifacts"),
                observability_config=ObservabilityConfig(enabled=False),
            ).run("otel-resume", tmp_path / "bundle-2")
        )
    resumed_store.close()
    provider.shutdown()


def test_resume_starts_new_trace_linked_to_previous_attempt(tmp_path: Path) -> None:
    provider, exporter = _tracer()
    model = SequenceModel([_response({"type": "finalize", "abstain_reason": "done"})])
    runtime, store = _runtime(tmp_path, model, provider)
    asyncio.run(
        runtime.run(
            "resume-run",
            tmp_path / "bundle-1",
            execution_attempt_id="attempt-1",
        )
    )
    first = next(span for span in exporter.get_finished_spans() if span.parent is None)
    assert first.context is not None
    domain_revision = store.domain_revision
    saved_context = store.records("late_audit")[-1].payload
    assert isinstance(saved_context, dict)
    assert saved_context["trace_id"] == f"{first.context.trace_id:032x}"
    store.close()

    store = RunStore(tmp_path / "run.sqlite3")
    resumed = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        model,
        store,
        ArtifactStore(tmp_path / "artifacts"),
        tracer=provider.get_tracer("test.runtime"),
        observability_config=ObservabilityConfig(enabled=True),
    )
    asyncio.run(
        resumed.run(
            "resume-run",
            tmp_path / "bundle-2",
            execution_attempt_id="attempt-2",
        )
    )

    roots = [span for span in exporter.get_finished_spans() if span.parent is None]
    second = next(
        span
        for span in roots
        if (span.attributes or {})["execution_attempt_id"] == "attempt-2"
    )
    assert second.context is not None
    assert second.context.trace_id != first.context.trace_id
    assert len(second.links) == 1
    assert second.links[0].context.span_id == first.context.span_id
    assert model.calls == 1
    assert store.domain_revision == domain_revision
    store.close()
    provider.shutdown()


def test_run_timestamps_survive_recovery_and_sealed_replay(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    now = 1_000.0
    monkeypatch.setattr("rootwalk.runtime.time.time", lambda: now)
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifacts"
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        SequenceModel([]),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(
            wall_clock_seconds=30,
            context_strategy="recent",
            validator_mode="observe_only",
        ),
    )
    first_runtime._run_id = "timed-run"  # pyright: ignore[reportPrivateUsage]
    first_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    initial_manifest = first_store.records("manifest")[0].payload
    assert isinstance(initial_manifest, dict)
    initial_manifest = cast(dict[str, Any], initial_manifest)
    started_at = initial_manifest["started_at"]
    assert initial_manifest["context_strategy"] == "recent"
    assert initial_manifest["validator_mode"] == "observe_only"
    first_store.close()

    now = 1_001.0
    model = SequenceModel([_response({"type": "finalize", "abstain_reason": "done"})])
    second_store = RunStore(database)
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(
                wall_clock_seconds=999,
                context_strategy="recent",
                validator_mode="observe_only",
            ),
        ).run("timed-run", tmp_path / "bundle-1")
    )
    manifest = json.loads((tmp_path / "bundle-1" / "manifest.json").read_text())
    assert result.report["started_at"] == started_at
    assert result.report["ended_at"] >= result.report["started_at"]
    assert result.report["wall_clock_seconds"] >= 0
    assert manifest["started_at"] == result.report["started_at"]
    assert manifest["ended_at"] == result.report["ended_at"]
    assert manifest["wall_clock_seconds"] == result.report["wall_clock_seconds"]
    second_store.close()

    now = 2_000.0
    third_store = RunStore(database)
    replayed = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            SequenceModel([]),
            third_store,
            ArtifactStore(artifact_root),
        ).run("timed-run", tmp_path / "bundle-2")
    )
    assert replayed.report["started_at"] == result.report["started_at"]
    assert replayed.report["ended_at"] == result.report["ended_at"]
    assert replayed.report["wall_clock_seconds"] == result.report["wall_clock_seconds"]
    third_store.close()


@pytest.mark.parametrize(
    ("field", "initial", "resumed"),
    [
        (
            "context_strategy",
            RuntimeConfig(context_strategy="recent"),
            RuntimeConfig(context_strategy="evidence"),
        ),
        (
            "validator_mode",
            RuntimeConfig(validator_mode="observe_only"),
            RuntimeConfig(validator_mode="enforce"),
        ),
    ],
)
def test_resume_rejects_changed_experiment_config(
    tmp_path: Path,
    field: str,
    initial: RuntimeConfig,
    resumed: RuntimeConfig,
) -> None:
    database = tmp_path / field / "run.sqlite3"
    artifact_root = tmp_path / field / "artifacts"
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        SequenceModel([]),
        first_store,
        ArtifactStore(artifact_root),
        initial,
    )
    first_runtime._run_id = "config-run"  # pyright: ignore[reportPrivateUsage]
    first_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    first_store.close()

    model = SequenceModel([])
    second_store = RunStore(database)
    with pytest.raises(ValueError, match=f"different {field}.*frozen=.*current="):
        asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                model,
                second_store,
                ArtifactStore(artifact_root),
                resumed,
            ).run("config-run", tmp_path / field / "bundle")
        )
    assert model.calls == 0
    assert second_store.records("report") == []
    second_store.close()


def test_resume_allows_legacy_manifest_without_experiment_config(
    tmp_path: Path,
) -> None:
    database = tmp_path / "legacy" / "run.sqlite3"
    artifact_root = tmp_path / "legacy" / "artifacts"
    store = RunStore(database)
    store.append(
        "manifest",
        "manifest:legacy-run",
        {
            "run_id": "legacy-run",
            "deadline_at": 9_999_999_999.0,
            "wall_clock_seconds": 300,
        },
    )
    store.close()

    store = RunStore(database)
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            SequenceModel([_response({"type": "finalize", "abstain_reason": "done"})]),
            store,
            ArtifactStore(artifact_root),
            RuntimeConfig(
                context_strategy="recent",
                validator_mode="observe_only",
            ),
        ).run("legacy-run", tmp_path / "legacy" / "bundle")
    )
    assert result.status == "completed_partial"
    store.close()


class FailingExporter(SpanExporter):
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        del spans
        raise RuntimeError("backend unavailable")


def test_exporter_failure_does_not_change_report_status(tmp_path: Path) -> None:
    provider = TracerProvider()
    provider.add_span_processor(
        SimpleSpanProcessor(BoundedSpanExporter(FailingExporter()))
    )
    model = SequenceModel([_response({"type": "finalize", "abstain_reason": "done"})])
    runtime, store = _runtime(tmp_path, model, provider)

    result = asyncio.run(runtime.run("failed-export", tmp_path / "bundle"))

    assert result.status == "completed_partial"
    store.close()
    provider.shutdown()
