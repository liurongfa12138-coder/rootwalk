import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

import pytest
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import (  # pyright: ignore[reportMissingTypeStubs]
    END,
    START,
    StateGraph,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from rootwalk.context import compile_context as real_compile_context
from rootwalk.contracts import (
    AnomalyDisposition,
    ArtifactRef,
    Coverage,
    Decision,
    Evidence,
    EvidenceAssessment,
    Extractor,
    FinalizeOperation,
    Hypothesis,
    MaterialAnomaly,
    Observation,
    QueryOperation,
    TimeRange,
)
from rootwalk.environment import CapabilityManifest
from rootwalk.model import (
    Message,
    ModelCallResult,
    ModelResponseError,
    ModelTimeoutError,
    RecordedModelAdapter,
)
from rootwalk.observability import ObservabilityConfig
from rootwalk.openrca import BankEnvironment as _BankEnvironment
from rootwalk.query_worker import QueryWorker
from rootwalk.runtime import (
    DecisionModel,
    RootWalkRuntime,
    RuntimeConfig,
    RuntimeResult,
    _has_actionable_percent_pressure,  # pyright: ignore[reportPrivateUsage]
    _has_strict_directional_metric_shift,  # pyright: ignore[reportPrivateUsage]
    select_family_compatible_terminal_candidate,
    select_terminal_candidate,
    supported_candidate_binding,
)
from rootwalk.store import ArtifactStore, RunStore
from rootwalk.version import runtime_version_manifest, source_tree_revision

PUBLIC_INPUT = (
    Path(__file__).resolve().parents[2]
    / "rootwalk-handoff"
    / "rootwalk-development-handoff"
    / "rootwalk-data-prep"
    / "slices"
    / "bank-row-0001"
    / "public_input"
)
RUN_ID = "runtime-test"
EVIDENCE_ID = f"{RUN_ID}:evidence:decision:{RUN_ID}:0001:0"
HYPOTHESIS_ID = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0002:0"
PATH_ID = f"{RUN_ID}:path:decision:{RUN_ID}:0003:0"


def material_anomaly(anomaly_id: str = "anomaly-1") -> MaterialAnomaly:
    return MaterialAnomaly(
        anomaly_id=anomaly_id,
        subject_id="bank:entity:Tomcat01",
        signal_family="cpu",
        signal_ids=["cpu"],
        mechanism_candidates=["CPU fault"],
        source_observation_ids=["observation-1"],
        proposal_ids=["proposal-1"],
        materiality_basis="fixture",
    )


def test_runtime_registers_and_extracts_impact_and_trace_operation_profiles(
    tmp_path: Path,
) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(),
    )
    assert "query_impact_profile" in runtime.tools
    assert "query_trace_operation_profile" in runtime.tools
    subject = environment.topology.nodes[0].canonical_id
    start = environment.task.incident_range.start.timestamp()
    end = start + 60.0
    coverage = Coverage(
        requested_range=environment.task.allowed_query_range,
        observed_range=environment.task.incident_range,
        completeness="complete",
        basis="fixture",
    )
    payload_ref = ArtifactRef(content_hash="f" * 64, selector={"fixture": True})
    cases = [
        (
            QueryOperation(type="query", tool="query_impact_profile", arguments={}),
            {
                "status": "success",
                "evidence_role": "impact_only",
                "profiles": [
                    {
                        "entity_id": subject,
                        "evidence_role": "impact_only",
                        "onset_interval": {"start": start, "end": end},
                    }
                ],
            },
            "impact_profile",
            "business_impact",
        ),
        (
            QueryOperation(
                type="query",
                tool="query_trace_operation_profile",
                arguments={"entity_id": subject},
            ),
            {
                "status": "success",
                "entity_id": subject,
                "operations": [
                    {
                        "entity_id": subject,
                        "evidence_role": "component_operation",
                        "onset_interval": {"start": start, "end": end},
                    }
                ],
            },
            "trace_operation_profile",
            "operation_profile",
        ),
    ]
    for index, (operation, result, evidence_kind, dimension) in enumerate(cases):
        observation = Observation(
            observation_id=f"profile-observation-{index}",
            tool=operation.tool,
            arguments=operation.arguments,
            status="success",
            coverage=coverage,
            artifact_refs=[],
            attempt_id=f"profile-attempt-{index}",
        )
        extracted = runtime._extract_evidence(  # pyright: ignore[reportPrivateUsage]
            index,
            operation,
            result,
            observation,
            payload_ref,
            [],
        )
        assert extracted.evidence_kind == evidence_kind
        assert extracted.mechanism_dimension == dimension
        assert extracted.semantic_quality == "typed"
        assert extracted.onset_interval is not None
        assert extracted.subject_ids == [subject]
        assert isinstance(extracted.value, Mapping)
        projected_key = (
            "profiles" if evidence_kind == "impact_profile" else "operations"
        )
        assert projected_key in extracted.value
    store.close()


def test_evidence_onset_prefers_strong_profile_over_earlier_weak_burst() -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    observed = environment.task.incident_range
    weak_onset = observed.start.timestamp() + 10
    strong_onset = observed.start.timestamp() + 120
    result = {
        "metric_profiles": [
            {
                "metric_id": "weak",
                "strong_incident_change": False,
                "finding_sample_ready": True,
                "incident_burst": {"onset_timestamp": weak_onset},
            },
            {
                "metric_id": "strong",
                "strong_incident_change": True,
                "finding_sample_ready": True,
                "incident_burst": {"onset_timestamp": strong_onset},
            },
        ],
        "finding_gate_summary": {
            "fault_type_ready": False,
            "material_event_ready": False,
            "earliest_strong_onset_timestamp": strong_onset,
        },
    }

    onset = RootWalkRuntime._evidence_onset(  # pyright: ignore[reportPrivateUsage]
        result, observed
    )

    assert onset is not None
    assert onset.start.timestamp() == strong_onset
    assert onset.end.timestamp() == strong_onset + 1
    assert (
        RootWalkRuntime._evidence_onset(  # pyright: ignore[reportPrivateUsage]
            {
                "metric_profiles": [
                    {
                        "strong_incident_change": False,
                        "finding_sample_ready": True,
                        "incident_burst": {"onset_timestamp": weak_onset},
                    }
                ]
            },
            observed,
        )
        is None
    )


def test_runtime_optional_profile_tools_fallback_for_legacy_provider(
    tmp_path: Path,
) -> None:
    class LegacyProvider:
        def __init__(self, delegate: BankEnvironment) -> None:
            self.delegate = delegate

        def __getattr__(self, name: str) -> Any:
            if name in {"query_impact_profile", "query_trace_operation_profile"}:
                raise AttributeError(name)
            return getattr(self.delegate, name)

    environment = BankEnvironment(PUBLIC_INPUT)
    legacy = LegacyProvider(environment)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        cast(Any, legacy),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(),
    )
    subject = environment.topology.nodes[0].canonical_id
    operations = [
        QueryOperation(type="query", tool="query_impact_profile", arguments={}),
        QueryOperation(
            type="query",
            tool="query_trace_operation_profile",
            arguments={"entity_id": subject, "limit": 5},
        ),
    ]

    for operation in operations:
        arguments = runtime._validate_query(operation)  # pyright: ignore[reportPrivateUsage]
        result = asyncio.run(
            runtime._execute_tool(  # pyright: ignore[reportPrivateUsage]
                operation.tool, arguments
            )
        )
        assert result["status"] == "unsupported"
        assert result["error_type"] == "unsupported_tool"
        assert result["retryable"] is False
        assert operation.tool in cast(str, result["summary"])
        assert result["artifact_refs"] == []
        coverage = cast_dict(result["coverage"])
        assert coverage["completeness"] == "unknown"
    store.close()


def test_overview_projection_preserves_navigation_materiality() -> None:
    value = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {
            "status": "success",
            "public_fault_types": ["CPU fault"],
            "fault_type_candidates": [
                {
                    "candidate_id": "candidate-1",
                    "candidate_rank": 1,
                    "fault_type": "CPU fault",
                    "entity_id": "bank:entity:Tomcat01",
                    "metric_id": "cpu",
                    "resource_family": "cpu",
                    "score": 1.0,
                    "requires_paired_exact": True,
                    "materiality": "navigation_only",
                    "materiality_basis": "insufficient_event_strength",
                }
            ],
        }
    )

    candidate = cast_dict(cast_dict(value)["fault_type_candidates"][0])
    assert candidate["materiality"] == "navigation_only"
    assert candidate["materiality_basis"] == "insufficient_event_strength"


class BankEnvironment(_BankEnvironment):
    def __init__(self, public_input: Path) -> None:
        super().__init__(public_input)
        if not self.task.entry_entities:
            self.task = self.task.model_copy(
                update={"entry_entities": ["bank:entity:Tomcat01"]}
            )


def overview_signals_from_two_families() -> list[dict[str, Any]]:
    environment = _BankEnvironment(PUBLIC_INPUT)
    result = environment.summarize_anomalies(limit=20)
    selected: dict[str, dict[str, Any]] = {}
    for raw_signal in cast(list[object], result["signals"]):
        signal = cast_dict(raw_signal)
        if not all(
            environment.query_metric(
                cast(str, signal["entity_id"]),
                cast(str, signal["metric_id"]),
                period=period,
            )["status"]
            == "success"
            for period in ("reference", "incident")
        ):
            continue
        selected.setdefault(cast(str, signal["resource_family"]), signal)
    assert len(selected) >= 2
    return list(selected.values())[:2]


def paired_metric_queries(signal: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "type": "query",
            "tool": "query_metric",
            "arguments": {
                "entity_id": signal["entity_id"],
                "metric_id": signal["metric_id"],
                "period": period,
            },
        }
        for period in ("reference", "incident")
    ]


class TopEightCandidateBank(_BankEnvironment):
    def list_metrics(self, entity_id: str | None = None) -> dict[str, Any]:
        result = dict(super().list_metrics(entity_id))
        if entity_id is not None:
            result["metric_ids"] = [
                *cast(list[str], result.get("metric_ids", [])),
                *[f"synthetic_metric_{index}" for index in range(10)],
                *[f"synthetic_source_metric_{index}" for index in range(10)],
            ]
        return result

    def query_metric(
        self,
        entity_id: str,
        metric_id: str,
        *,
        period: str = "incident",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> dict[str, Any]:
        if metric_id.startswith("synthetic_"):
            requested = self._window(period, start, end)[1]
            return {
                "status": "success",
                "summary": {"min": 10.0, "median": 20.0, "max": 30.0},
                "coverage": self._coverage(
                    requested,
                    observed_range=self._range(requested),
                    completeness="complete",
                    basis="synthetic candidate fixture",
                ),
                "artifact_refs": [],
            }
        return super().query_metric(
            entity_id,
            metric_id,
            period=period,
            start=start,
            end=end,
        )


def seed_candidate_overview(
    store: RunStore,
    environment: _BankEnvironment,
    *,
    key_limit: int | None = None,
    source_metric_id: str | None = None,
    source_equals_direct: bool = False,
    exact_series_count: int | None = None,
    source_metric_indices: Sequence[int] = (),
    metric_role: str = "direct",
    has_log_source: bool | None = None,
    alias_same_series_family: bool = False,
    second_series_same_fault: bool = False,
    omit_log_source: bool = False,
    first_alias: str | None = None,
) -> list[dict[str, Any]]:
    first, second = overview_signals_from_two_families()
    overview_result = environment.summarize_anomalies(
        limit=20 if exact_series_count is not None else 1
    )
    fault_types = [
        "high CPU usage",
        "high memory usage",
        "memory pressure qualifier",
    ]
    overview_result["public_fault_types"] = fault_types
    raw_candidates: list[dict[str, Any]] = [
        {
            "candidate_id": f"candidate-{index}",
            "candidate_rank": index,
            "fault_type": fault_types[index],
            "entity_id": signal["entity_id"],
            "metric_id": signal["metric_id"],
            "resource_family": signal["resource_family"],
            "score": signal["overview_score"],
            "overview_score": signal["overview_score"],
            "reference": signal["reference"],
            "incident": signal["incident"],
            "requires_paired_exact": True,
            "metric_role": metric_role,
            **(
                {}
                if omit_log_source
                else {
                    "has_log_source": (
                        has_log_source
                        if has_log_source is not None
                        else bool(
                            environment.list_metrics(
                                cast(str, signal["entity_id"])
                            ).get("log_names")
                        )
                    )
                }
            ),
        }
        for index, signal in enumerate((first, second))
    ]
    if source_metric_id is not None:
        raw_candidates[0]["source_metric_id"] = source_metric_id
    elif source_equals_direct:
        raw_candidates[0]["source_metric_id"] = raw_candidates[0]["metric_id"]
    raw_candidates.append(
        {
            **raw_candidates[1],
            "candidate_id": "candidate-2",
            "candidate_rank": 2,
            "fault_type": fault_types[2],
            "resource_family": first["resource_family"],
        }
    )
    if first_alias is not None:
        raw_candidates.insert(
            1,
            {
                **raw_candidates[0],
                "candidate_id": "candidate-0-alias",
                "candidate_rank": 1,
                "fault_type": first_alias,
            },
        )
        fault_types.append(first_alias)
    if alias_same_series_family:
        raw_candidates[2]["resource_family"] = second["resource_family"]
    if second_series_same_fault:
        raw_candidates[2]["fault_type"] = fault_types[0]
    if exact_series_count is not None:
        entity_id = cast(str, first["entity_id"])
        raw_candidates = [
            {
                "candidate_id": f"candidate-{index}",
                "candidate_rank": index,
                "fault_type": f"fault-{index}",
                "entity_id": entity_id,
                "metric_id": f"synthetic_metric_{index}",
                "resource_family": f"family-{index}",
                "score": 1.0 - index / 100,
                "overview_score": 1.0 - index / 100,
                "reference": {},
                "incident": {},
                "requires_paired_exact": True,
                "metric_role": metric_role,
                **(
                    {}
                    if omit_log_source
                    else {
                        "has_log_source": (
                            has_log_source if has_log_source is not None else False
                        )
                    }
                ),
            }
            for index in range(exact_series_count)
        ]
        fault_types = [cast(str, item["fault_type"]) for item in raw_candidates]
        for index in source_metric_indices:
            raw_candidates[index]["source_metric_id"] = (
                f"synthetic_source_metric_{index}"
            )
    if key_limit is not None:
        selected: list[dict[str, Any]] = []
        keys: set[tuple[object, object]] = set()
        for candidate in raw_candidates:
            key = (candidate["entity_id"], candidate["resource_family"])
            if key not in keys and len(keys) == key_limit:
                continue
            keys.add(key)
            selected.append(candidate)
        raw_candidates = selected
    overview_result["fault_type_candidates"] = raw_candidates
    overview_result["public_fault_types"] = fault_types
    summary = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        overview_result
    )
    observation = Observation.model_validate_json(
        json.dumps(
            {
                "observation_id": "seed:overview",
                "tool": "summarize_anomalies",
                "arguments": {"limit": 1},
                "status": overview_result["status"],
                "coverage": overview_result["coverage"],
                "artifact_refs": [],
                "attempt_id": "attempt:overview",
                "summary": json.dumps(summary, separators=(",", ":"), sort_keys=True),
            }
        )
    )
    store.append("observation", observation.observation_id, observation)
    return [cast_dict(item) for item in summary["fault_type_candidates"]]


def seed_public_fault_types(
    store: RunStore, environment: _BankEnvironment
) -> list[str]:
    fault_types = ["high CPU usage", "high memory usage"]
    overview_result = environment.summarize_anomalies(limit=1)
    observation = Observation.model_validate_json(
        json.dumps(
            {
                "observation_id": "seed:public-vocabulary",
                "tool": "summarize_anomalies",
                "arguments": {"limit": 1},
                "status": overview_result["status"],
                "coverage": overview_result["coverage"],
                "artifact_refs": [],
                "attempt_id": "attempt:public-vocabulary",
                "summary": json.dumps({"public_fault_types": fault_types}),
            }
        )
    )
    store.append("observation", observation.observation_id, observation)
    return fault_types


def strict_paired_metric_summary() -> dict[str, Any]:
    return {
        "counts_complete": True,
        "scale_hint": "0..100",
        "pressure_direction": "higher_is_more_pressure",
        "hint_basis": "heuristic from metric name",
        "reference": {
            "sample_count": 10,
            "min": 10.0,
            "median": 20.0,
            "max": 30.0,
        },
        "incident": {
            "sample_count": 10,
            "min": 40.0,
            "median": 50.0,
            "max": 90.0,
            "above_reference_max_count": 5,
            "below_reference_min_count": 0,
        },
    }


def seed_tool_evidence(
    store: RunStore,
    environment: _BankEnvironment,
    tool: str,
    *,
    status: Literal["success", "partial"] = "success",
    subject: str = "bank:entity:Tomcat01",
    paired_log_summary: bool = False,
    filtered_log: bool | None = None,
    log_name: str = "fixture",
    metric_id: str = "fixture_cpu_usage_percent",
    paired_metric_summary: Mapping[str, Any] | None = None,
) -> str:
    suffix = hashlib.sha256(subject.encode()).hexdigest()[:8]
    observation_id = f"seed:observation:{tool}:{suffix}"
    evidence_id = f"seed:evidence:{tool}:{suffix}"
    coverage = Coverage(
        requested_range=environment.task.allowed_query_range,
        observed_range=environment.task.allowed_query_range,
        completeness="partial" if status == "partial" else "complete",
        collection_health="healthy",
        basis="synthetic corroboration fixture",
    )
    observation = Observation(
        observation_id=observation_id,
        tool=tool,
        arguments=(
            {
                "entity_id": subject,
                "metric_id": metric_id,
                "period": "incident",
            }
            if tool == "query_metric"
            else (
                {"entity_id": subject, "log_name": log_name}
                if tool == "query_log_samples"
                and (paired_log_summary if filtered_log is None else filtered_log)
                else (
                    {"entity_id": subject}
                    if tool in {"query_log_samples", "query_trace_neighbors"}
                    else {}
                )
            )
        ),
        status=status,
        coverage=coverage,
        artifact_refs=[],
        attempt_id=f"attempt:{tool}",
        summary="synthetic evidence",
    )
    value: dict[str, Any] = {"status": status}
    if tool == "query_metric":
        value["paired_metric_summary"] = (
            paired_metric_summary or strict_paired_metric_summary()
        )
    if paired_log_summary:
        value["paired_log_summary"] = {
            "counts_complete": True,
            "samples_truncated": status == "partial",
            "reference": {
                "matched_count": 1,
                "log_name_counts": {log_name: 1},
            },
            "incident": {
                "matched_count": 2,
                "log_name_counts": {log_name: 2},
            },
        }
    evidence = Evidence(
        evidence_id=evidence_id,
        observation_refs=[observation_id],
        subject_ids=[subject],
        predicate=f"{tool}.result",
        value=value,
        observed_interval=environment.task.allowed_query_range,
        time_uncertainty="synthetic fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="test.seed_tool_evidence",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint=f"source:{tool}",
        dataset_snapshot_id=environment.dataset_id,
    )
    store.append("observation", observation_id, observation)
    store.append("evidence", evidence_id, evidence)
    return evidence_id


def support_gate_errors(
    store: RunStore,
    environment: _BankEnvironment,
    *,
    subject: str,
    statement: str,
    evidence_ids: Sequence[str],
    artifact_root: Path,
) -> list[str]:
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(artifact_root),
    )
    runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    return runtime._hypothesis_corroboration_errors(  # pyright: ignore[reportPrivateUsage]
        subject, statement, evidence_ids
    )


def seed_comparison_pair(
    store: RunStore,
    environment: _BankEnvironment,
    candidate: Mapping[str, Any],
    *,
    incident_status: Literal["success", "no_data"] = "success",
) -> None:
    for period, status in (("reference", "success"), ("incident", incident_status)):
        observation = Observation(
            observation_id=f"seed:comparison:{candidate['candidate_id']}:{period}",
            tool="query_metric",
            arguments={
                "entity_id": candidate["entity_id"],
                "metric_id": candidate["metric_id"],
                "period": period,
            },
            status=cast(Any, status),
            coverage=Coverage(
                requested_range=environment.task.allowed_query_range,
                observed_range=environment.task.allowed_query_range,
                completeness="complete",
                collection_health="healthy",
                basis="synthetic comparison fixture",
            ),
            artifact_refs=[],
            attempt_id=f"attempt:comparison:{period}",
        )
        store.append("observation", observation.observation_id, observation)


def run_runtime(
    tmp_path: Path,
    records: list[dict[str, Any]],
    *,
    max_model_calls: int = 12,
    max_context_tokens: int = 12_000,
    context_strategy: str = "evidence",
    validator_mode: str = "enforce",
    max_input_tokens: int = 144_000,
    max_output_tokens: int = 24_000,
    max_tool_attempts: int = 40,
    max_artifact_bytes: int = 64 * 1024 * 1024,
    max_validator_retries: int = 2,
    max_no_progress_rounds: int = 3,
    model: DecisionModel | None = None,
    environment: _BankEnvironment | None = None,
    require_overview: bool = False,
) -> tuple[RunStore, RuntimeResult]:
    store = RunStore(tmp_path / "run.sqlite3")
    selected_environment = environment or (
        _BankEnvironment(PUBLIC_INPUT)
        if require_overview
        else BankEnvironment(PUBLIC_INPUT)
    )
    runtime = RootWalkRuntime(
        selected_environment,
        model or RecordedModelAdapter(records),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(
            max_model_calls=max_model_calls,
            max_context_tokens=max_context_tokens,
            context_strategy=cast(Any, context_strategy),
            validator_mode=cast(Any, validator_mode),
            max_input_tokens=max_input_tokens,
            max_output_tokens=max_output_tokens,
            max_tool_attempts=max_tool_attempts,
            max_artifact_bytes=max_artifact_bytes,
            max_validator_retries=max_validator_retries,
            max_no_progress_rounds=max_no_progress_rounds,
        ),
    )
    result = asyncio.run(runtime.run(RUN_ID, tmp_path / "bundle"))
    return store, result


def test_runtime_config_rejects_nonpositive_model_timeout() -> None:
    with pytest.raises(ValueError, match="model_timeout_seconds"):
        RuntimeConfig(model_timeout_seconds=0)


def test_runtime_config_rejects_inconsistent_sandbox_provenance() -> None:
    with pytest.raises(ValueError, match="required query sandbox"):
        RuntimeConfig(
            query_sandbox_requested="required",
            query_sandbox_effective="spawn",
        )
    with pytest.raises(ValueError, match="record its version"):
        RuntimeConfig(
            query_sandbox_requested="auto",
            query_sandbox_effective="bwrap",
        )


def test_source_tree_revision_is_reproducible_across_install_paths(
    tmp_path: Path,
) -> None:
    left = tmp_path / "checkout/rootwalk"
    right = tmp_path / "wheel/rootwalk"
    for root in (left, right):
        (root / "nested").mkdir(parents=True)
        (root / "__init__.py").write_text("VERSION = 1\n")
        (root / "nested/module.py").write_text("VALUE = 'stable'\n")
    (left / "ignored.txt").write_text("checkout-only metadata")

    assert source_tree_revision(left) == source_tree_revision(right)


def test_runtime_versions_bind_lock_and_installed_direct_dependencies() -> None:
    versions = runtime_version_manifest()
    assert (
        versions["uv_lock_sha256"]
        == hashlib.sha256(
            (Path(__file__).resolve().parents[1] / "uv.lock").read_bytes()
        ).hexdigest()
    )
    dependencies = versions["direct_dependency_versions"]
    assert isinstance(dependencies, dict) and dependencies
    encoded = json.dumps(dependencies, separators=(",", ":"), sort_keys=True).encode()
    assert (
        versions["direct_dependency_fingerprint"] == hashlib.sha256(encoded).hexdigest()
    )


def test_model_topology_summary_omits_duplicate_node_identity_fields(
    tmp_path: Path,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        environment,
        RecordedModelAdapter([]),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )

    summary = runtime._topology_summary()  # pyright: ignore[reportPrivateUsage]

    assert summary["nodes"]
    assert all(
        set(node) == {"canonical_id", "entity_type", "sources"}
        for node in summary["nodes"]
    )
    assert all("fact_kind" in edge for edge in summary["edges"])
    store.close()


@pytest.mark.parametrize("abstain_after_supported_path", [False, True])
def test_state_graph_runs_query_evidence_hypothesis_path_finalize(
    tmp_path: Path, abstain_after_supported_path: bool
) -> None:
    log_evidence_id = f"{RUN_ID}:evidence:decision:{RUN_ID}:0002:0"
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0003:0"
    path_id = f"{RUN_ID}:path:decision:{RUN_ID}:0004:0"
    records = [
        {
            "decision_summary": "inspect one metric",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                    },
                }
            ],
        },
        {
            "decision_summary": "corroborate with one log sample",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_log_samples",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "limit": 1,
                    },
                }
            ],
        },
        {
            "decision_summary": "record the supported hypothesis",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "Tomcat01 showed a source-backed metric result",
                    "status": "supported",
                    "supporting_evidence_refs": [EVIDENCE_ID, log_evidence_id],
                }
            ],
        },
        {
            "decision_summary": "stage a local path",
            "operations": [
                {
                    "type": "submit_path",
                    "hypothesis_id": hypothesis_id,
                    "root_cause_entity_id": "bank:entity:Tomcat01",
                    "symptom_entity_ids": ["bank:entity:Tomcat01"],
                    "steps": [],
                }
            ],
        },
        {
            "decision_summary": "finish with disclosed uncertainty",
            "operations": [
                {"type": "finalize", "abstain_reason": "nothing selected"}
                if abstain_after_supported_path
                else {
                    "type": "finalize",
                    "hypothesis_ids": [hypothesis_id],
                    "path_ids": [path_id],
                    "unresolved_questions": ["Collection completeness is unknown"],
                }
            ],
        },
    ]

    store, result = run_runtime(tmp_path, records)

    assert result.status == "completed_partial"
    assert result.model_calls == 5
    assert result.report["candidates"][0]["hypothesis"]["hypothesis_id"] == (
        hypothesis_id
    )
    assert result.report["findings"][0]["hypothesis_id"] == hypothesis_id
    assert result.report["path_ids"] == [path_id]
    assert result.report["abstain_reason"] is None
    auto_selected = [
        item
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "finalization_auto_selected"
    ]
    assert bool(auto_selected) is abstain_after_supported_path
    assert result.bundle_path == tmp_path / "bundle"
    assert (result.bundle_path / "report.json").is_file()
    assert len(store.records("context_manifest")) == 5
    assert len(store.records("evidence")) == 2
    manifest = cast_dict(store.records("manifest")[0].payload)
    assert manifest["model_id"] == "unknown"
    assert manifest["prompt_version"] == "m124"
    assert manifest["tool_version"] == "openrca-public-input-v82"
    assert manifest["terminal_policy_version"] == 60
    assert manifest["max_artifact_bytes"] == 64 * 1024 * 1024
    assert {
        name: manifest[name] for name in runtime_version_manifest()
    } == runtime_version_manifest()
    decision_revision = store.records("decision")[0].revision
    query_attempt_revision = next(
        item.revision
        for item in store.records("attempt")
        if cast_dict(item.payload).get("tool") == "query_metric"
    )
    assert decision_revision < query_attempt_revision
    model_attempts = [
        cast_dict(item.payload)
        for item in store.records("attempt")
        if item.stable_key.startswith("attempt:model:")
        and cast_dict(item.payload).get("status") == "success"
    ]
    assert {item["provider_model"] for item in model_attempts} == {"recorded"}
    source_hash = cast_dict(json.loads((PUBLIC_INPUT / "manifest.json").read_text()))[
        "artifacts"
    ][0]["sha256"]
    assert not (result.bundle_path / "artifacts" / source_hash).exists()
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
@pytest.mark.parametrize("abstain_after_support", [False, True])
def test_supported_hypothesis_can_finalize_without_a_path(
    tmp_path: Path, validator_mode: str, abstain_after_support: bool
) -> None:
    log_evidence_id = f"{RUN_ID}:evidence:decision:{RUN_ID}:0002:0"
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0003:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "inspect one metric",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                    },
                }
            ],
        },
        {
            "decision_summary": "corroborate with one log sample",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_log_samples",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "limit": 1,
                    },
                }
            ],
        },
        {
            "decision_summary": "record the supported hypothesis",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "Tomcat01 showed a source-backed metric result",
                    "status": "supported",
                    "supporting_evidence_refs": [EVIDENCE_ID, log_evidence_id],
                }
            ],
        },
        {
            "decision_summary": "finish without inventing a causal path",
            "operations": [
                {"type": "finalize", "abstain_reason": "nothing selected"}
                if abstain_after_support
                else {
                    "type": "finalize",
                    "hypothesis_ids": [hypothesis_id],
                    "path_ids": [],
                }
            ],
        },
    ]

    store, result = run_runtime(tmp_path, records, validator_mode=validator_mode)

    assert result.status == "completed_partial"
    assert result.report["hypothesis_ids"] == [hypothesis_id]
    assert len(result.report["findings"]) == 1
    assert result.report["path_ids"] == []
    assert result.report["paths"] == []
    auto_selected = [
        item
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "finalization_auto_selected"
    ]
    assert bool(auto_selected) is abstain_after_support
    store.close()


class CapturingModel(RecordedModelAdapter):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        super().__init__(records)
        self.inputs: list[Sequence[Message]] = []

    async def decide(self, messages: Sequence[Message]) -> ModelCallResult:
        self.inputs.append(messages)
        return await super().decide(messages)


class AuditedCapturingModel:
    def __init__(self, records: Sequence[Mapping[str, Any]]) -> None:
        self._records = [dict(record) for record in records]
        self._cursor = 0
        self._attempt_hook: Any = None
        self.inputs: list[Sequence[Message]] = []

    def set_attempt_hook(self, hook: Any) -> None:
        self._attempt_hook = hook

    async def decide(self, messages: Sequence[Message]) -> Decision:
        if self._cursor >= len(self._records):
            raise AssertionError("no recorded model response remains")
        raw_response = self._records[self._cursor]
        self._cursor += 1
        self.inputs.append(messages)
        attempt = {
            "request_index": 1,
            "status": "success",
            "request": {"messages": [dict(message) for message in messages]},
            "response": raw_response,
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            "started_at": float(self._cursor),
            "ended_at": float(self._cursor) + 0.5,
        }
        if self._attempt_hook is not None:
            self._attempt_hook("started", attempt)
        decision = Decision.model_validate(raw_response)
        if self._attempt_hook is not None:
            self._attempt_hook("finished", attempt)
        return decision


class NoCallModel:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, messages: Sequence[Message]) -> object:
        del messages
        self.calls += 1
        raise AssertionError("model must not be called during recovery")


def test_unknown_tool_rejects_whole_batch_without_execution(tmp_path: Path) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "bad batch",
            "operations": [
                {"type": "query", "tool": "not_a_tool", "arguments": {}},
                {"type": "query", "tool": "list_entities", "arguments": {}},
            ],
        },
        {
            "decision_summary": "abstain",
            "operations": [
                {
                    "type": "finalize",
                    "abstain_reason": "The requested operation was invalid",
                }
            ],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(
        tmp_path, records, model=model, validator_mode="observe_only"
    )

    assert result.status == "completed_partial"
    assert not store.records("observation")
    assert not [
        item for item in store.records("attempt") if cast_dict(item.payload).get("tool")
    ]
    assert "zero_operations_executed" in str(model.inputs[1])
    store.close()


def test_invalid_metric_rejects_whole_batch_before_tool_attempt(tmp_path: Path) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "bad metric",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:MG02",
                        "metric_id": "cpu",
                    },
                },
                {"type": "query", "tool": "list_entities", "arguments": {}},
            ],
        },
        {
            "decision_summary": "abstain",
            "operations": [{"type": "finalize", "abstain_reason": "invalid metric"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model)

    assert result.status == "completed_partial"
    assert not store.records("observation")
    assert "exact value returned by list_metrics" in str(model.inputs[1])
    store.close()


def test_empty_entry_requires_successful_overview_before_other_queries(
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "query too narrowly",
            "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
        },
        {
            "decision_summary": "mix overview with another navigation query",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 1},
                },
                {"type": "query", "tool": "list_entities", "arguments": {}},
            ],
        },
        {
            "decision_summary": "run the required broad overview alone",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 1},
                }
            ],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(
        tmp_path,
        records,
        model=model,
        require_overview=True,
    )

    assert result.status == "completed_partial"
    assert [
        cast_dict(item.payload)["tool"] for item in store.records("observation")
    ] == ["summarize_anomalies"]
    assert "broad overview required" in str(model.inputs[1])
    assert "only one summarize_anomalies" in str(model.inputs[2])
    assert result.report["budget"]["validator_retries_used"] == 2
    assert result.report["budget"]["no_progress_rounds"] == 0
    store.close()


def test_successful_overview_and_existing_entry_each_disable_overview_gate(
    tmp_path: Path,
) -> None:
    after_overview: list[dict[str, Any]] = [
        {
            "decision_summary": "overview",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 1},
                }
            ],
        },
        {
            "decision_summary": "continue with another query",
            "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    first_store, first_result = run_runtime(
        tmp_path / "overview",
        after_overview,
        require_overview=True,
    )
    assert first_result.status == "completed_partial"
    assert [
        cast_dict(item.payload)["tool"] for item in first_store.records("observation")
    ] == ["summarize_anomalies", "list_entities"]
    first_store.close()

    with_entry_records: list[dict[str, Any]] = [
        {
            "decision_summary": "start from the supplied entry",
            "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    second_store, second_result = run_runtime(
        tmp_path / "entry",
        with_entry_records,
    )
    assert second_result.status == "completed_partial"
    assert len(second_store.records("observation")) == 1
    assert second_result.report["budget"]["validator_retries_used"] == 0
    second_store.close()


def test_missing_overview_tool_disables_gate(tmp_path: Path) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    model = CapturingModel(
        [
            {
                "decision_summary": "use the available query",
                "operations": [
                    {"type": "query", "tool": "list_entities", "arguments": {}}
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        environment,
        model,
        store,
        ArtifactStore(tmp_path / "artifacts"),
    )
    runtime.tools.pop("summarize_anomalies")

    result = asyncio.run(runtime.run("missing-overview-tool", tmp_path / "bundle"))

    assert result.status == "completed_partial"
    assert len(store.records("observation")) == 1
    assert result.report["budget"]["validator_retries_used"] == 0
    store.close()


def test_repeated_overview_gate_violation_exhausts_validator_retries(
    tmp_path: Path,
) -> None:
    invalid: dict[str, Any] = {
        "decision_summary": "skip overview",
        "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
    }
    model = CapturingModel([invalid, invalid, invalid])

    store, result = run_runtime(
        tmp_path,
        [],
        model=model,
        require_overview=True,
        max_validator_retries=2,
    )

    assert result.status == "no_progress"
    assert len(model.inputs) == 3
    assert not store.records("observation")
    assert result.report["budget"]["validator_retries_used"] == 2
    assert result.report["budget"]["no_progress_rounds"] == 3
    store.close()


def test_first_exact_metric_batch_rejects_same_family_pairs(tmp_path: Path) -> None:
    signal = overview_signals_from_two_families()[0]
    pair = paired_metric_queries(signal)
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "overview",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 20},
                }
            ],
        },
        {
            "decision_summary": "same family twice",
            "operations": [*pair, *pair],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model, require_overview=True)

    assert result.status == "completed_partial"
    assert len(store.records("observation")) == 1
    assert "different resource families" in str(model.inputs[2])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_first_exact_metric_batch_falls_back_when_fault_candidates_are_empty(
    tmp_path: Path,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    overview = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        environment.summarize_anomalies(limit=20)
    )
    assert overview["fault_type_candidates"] == []
    signal = overview_signals_from_two_families()[0]
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "overview",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 20},
                }
            ],
        },
        {
            "decision_summary": "verify one candidate",
            "operations": paired_metric_queries(signal),
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]

    store, result = run_runtime(tmp_path, records, environment=environment)

    assert result.status == "completed_partial"
    observations = [cast_dict(item.payload) for item in store.records("observation")]
    assert len(observations) == 3
    assert [item["arguments"]["period"] for item in observations[1:]] == [
        "reference",
        "incident",
    ]
    assert result.report["budget"]["validator_retries_used"] == 0
    store.close()


@pytest.mark.parametrize(
    ("query_source", "accepted"),
    [
        pytest.param("fault_candidate", True, id="fault-candidates-pass"),
        pytest.param("signal", False, id="signals-only-rejected"),
    ],
)
def test_first_exact_metric_batch_prefers_fault_candidates_over_top_signals(
    tmp_path: Path,
    query_source: Literal["fault_candidate", "signal"],
    accepted: bool,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    overview_result = environment.summarize_anomalies(limit=1)
    signal_metrics = {
        (signal["entity_id"], metric_id)
        for raw_signal in overview_result["signals"]
        for signal in [cast_dict(raw_signal)]
        for metric_id in [
            signal["metric_id"],
            *cast(list[str], signal["direct_metric_candidates"]),
        ]
    }
    candidate_signal = next(
        signal
        for signal in overview_signals_from_two_families()
        if (signal["entity_id"], signal["metric_id"]) not in signal_metrics
    )
    overview_result["fault_type_candidates"] = [
        {
            "fault_type": cast(list[str], overview_result["public_fault_types"])[0],
            "entity_id": candidate_signal["entity_id"],
            "metric_id": candidate_signal["metric_id"],
            "resource_family": candidate_signal["resource_family"],
            "score": candidate_signal["overview_score"],
            "reference": candidate_signal["reference"],
            "incident": candidate_signal["incident"],
            "requires_paired_exact": True,
        }
    ]
    overview = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        overview_result
    )
    candidate = cast_dict(overview["fault_type_candidates"][0])
    selected = (
        candidate
        if query_source == "fault_candidate"
        else cast_dict(overview["signals"][0])
    )
    observation = Observation.model_validate_json(
        json.dumps(
            {
                "observation_id": "seed:overview",
                "tool": "summarize_anomalies",
                "arguments": {"limit": 1},
                "status": overview_result["status"],
                "coverage": overview_result["coverage"],
                "artifact_refs": [],
                "attempt_id": "attempt:overview",
                "summary": json.dumps(overview, separators=(",", ":"), sort_keys=True),
            }
        )
    )
    store = RunStore(tmp_path / "run.sqlite3")
    store.append("observation", observation.observation_id, observation)
    model = CapturingModel(
        [
            {
                "decision_summary": "verify the selected persisted candidate",
                "operations": paired_metric_queries(selected),
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("fault-candidate", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    observations = [cast_dict(item.payload) for item in store.records("observation")]
    persisted_overview = cast_dict(json.loads(observations[0]["summary"]))
    assert candidate in persisted_overview["fault_type_candidates"]
    assert len(persisted_overview["signals"]) == 1
    assert (candidate["entity_id"], candidate["metric_id"]) not in signal_metrics
    if accepted:
        assert [item["arguments"] for item in observations[1:]] == [
            operation["arguments"] for operation in paired_metric_queries(candidate)
        ]
        assert all(
            item["status"] in {"success", "partial"} for item in observations[1:]
        )
        assert result.report["budget"]["validator_retries_used"] == 0
    else:
        assert len(observations) == 1
        assert "not in the persisted overview mapping" in str(model.inputs[1])
        assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


@pytest.mark.parametrize("candidate_count", [1, 2])
def test_query_candidate_expands_to_exact_metric_pairs(
    tmp_path: Path,
    candidate_count: int,
) -> None:
    environment = TopEightCandidateBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    source_metric_id = "synthetic_source_metric_0"
    available_candidates = seed_candidate_overview(
        store, environment, source_metric_id=source_metric_id
    )
    candidates = available_candidates[:candidate_count]
    expanded_candidates = available_candidates[:2]
    assert expanded_candidates[0]["source_metric_id"] == source_metric_id
    model = CapturingModel(
        [
            {
                "decision_summary": "select persisted candidates",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                    for candidate in candidates
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode="observe_only"),
        ).run("candidate-expand", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    observations = [cast_dict(item.payload) for item in store.records("observation")]
    expected_arguments = [
        {
            "entity_id": candidate["entity_id"],
            "metric_id": metric_id,
            "period": period,
        }
        for candidate in expanded_candidates
        for metric_id in (
            candidate["metric_id"],
            *(
                [cast(str, candidate["source_metric_id"])]
                if isinstance(candidate.get("source_metric_id"), str)
                and candidate["source_metric_id"] != candidate["metric_id"]
                else []
            ),
        )
        for period in ("reference", "incident")
    ]
    assert len(observations) == 1 + len(expected_arguments)
    assert all(item["tool"] == "query_metric" for item in observations[1:])
    assert [item["arguments"] for item in observations[1:]] == expected_arguments
    payloads = [cast_dict(item.payload) for item in store.records("query_payload")]
    effects = [cast_dict(item.payload) for item in store.records("tool_effect")]
    for records in (payloads, effects):
        assert [item["candidate_id"] for item in records] == [
            candidate["candidate_id"]
            for candidate in expanded_candidates
            for _child in range(4 if candidate.get("source_metric_id") else 2)
        ]
        assert [item["logical_operation_index"] for item in records] == [
            index
            for index, candidate in enumerate(expanded_candidates)
            for _child in range(4 if candidate.get("source_metric_id") else 2)
        ]
    decision = cast_dict(store.records("decision")[0].payload)
    assert all(
        set(cast_dict(operation)["arguments"]) == {"candidate_id"}
        for operation in decision["operations"]
    )
    expected_operation_ids = [
        ["candidate-expand", "decision:candidate-expand:0001", index]
        for index in range(len(expected_arguments))
    ]
    assert decision["operation_ids"] == expected_operation_ids
    operation_cursor = 0
    expected_expansions: list[dict[str, Any]] = []
    for logical_index, candidate in enumerate(expanded_candidates):
        child_count = 4 if candidate.get("source_metric_id") else 2
        expected_expansions.append(
            {
                "candidate_id": candidate["candidate_id"],
                "logical_operation_index": logical_index,
                "children": [
                    {
                        "operation_id": expected_operation_ids[
                            operation_cursor + offset
                        ],
                        "period": ("reference", "incident")[offset % 2],
                    }
                    for offset in range(child_count)
                ],
                "auto_filled": True,
            }
        )
        operation_cursor += child_count
    assert decision["candidate_expansions"] == expected_expansions
    evidence_ids = [
        cast_dict(item.payload)["evidence_id"] for item in store.records("evidence")
    ]
    assert len(evidence_ids) == len(set(evidence_ids)) == len(expected_arguments)
    assert [item["operation_id"] for item in effects] == expected_operation_ids
    first_context = cast_dict(json.loads(cast(str, model.inputs[0][1]["content"])))
    assert "candidate_id" in first_context["tools"]["query_candidate"]
    assert result.report["budget"]["tool_attempts_used"] == len(expected_arguments)
    assert all(item["auto_filled"] is True for item in decision["candidate_expansions"])
    replay = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    replay._run_id = "candidate-expand"  # pyright: ignore[reportPrivateUsage]
    replay._restore_from_records(  # pyright: ignore[reportPrivateUsage]
        {"decision_id": "decision:candidate-expand:0001"}
    )
    pending = replay._pending_metric_pair_dispositions()  # pyright: ignore[reportPrivateUsage]
    assert {item["metric_id"] for item in pending} == {
        item["metric_id"] for item in expanded_candidates
    }
    assert source_metric_id not in {item["metric_id"] for item in pending}
    assert len(replay._query_calls) == len(  # pyright: ignore[reportPrivateUsage]
        expected_arguments
    )
    assert replay._candidate_query_origins == {  # pyright: ignore[reportPrivateUsage]
        0: (expanded_candidates[0]["candidate_id"], 0),
        1: (expanded_candidates[0]["candidate_id"], 0),
        2: (expanded_candidates[0]["candidate_id"], 0),
        3: (expanded_candidates[0]["candidate_id"], 0),
        4: (expanded_candidates[1]["candidate_id"], 1),
        5: (expanded_candidates[1]["candidate_id"], 1),
    }
    store.close()


def test_query_candidate_equal_source_keeps_two_direct_children(tmp_path: Path) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(
        store,
        environment,
        key_limit=1,
        source_equals_direct=True,
    )[0]
    model = RecordedModelAdapter(
        [
            {
                "decision_summary": "select persisted candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode="observe_only"),
        ).run("candidate-equal-source", tmp_path / "bundle")
    )

    observations = [cast_dict(item.payload) for item in store.records("observation")][
        1:
    ]
    assert [item["arguments"] for item in observations] == [
        {
            "entity_id": candidate["entity_id"],
            "metric_id": candidate["metric_id"],
            "period": period,
        }
        for period in ("reference", "incident")
    ]
    store.close()


def test_first_query_candidate_auto_expands_top_eight_exact_series(
    tmp_path: Path,
) -> None:
    environment = TopEightCandidateBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidates = seed_candidate_overview(
        store,
        environment,
        exact_series_count=10,
        source_metric_indices=(0, 3, 7),
    )
    model = RecordedModelAdapter(
        [
            {
                "decision_summary": "trigger deterministic schedule",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidates[-1]["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode="observe_only"),
        ).run("candidate-top-eight", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    decision = cast_dict(store.records("decision")[0].payload)
    expansions = cast(list[dict[str, Any]], decision["candidate_expansions"])
    assert [item["candidate_id"] for item in expansions] == [
        item["candidate_id"] for item in candidates[:8]
    ]
    assert [item["logical_operation_index"] for item in expansions] == list(range(8))
    assert all(item["auto_filled"] is True for item in expansions)
    effects = [cast_dict(item.payload) for item in store.records("tool_effect")]
    assert len(effects) == 16 + 2 * 3
    assert result.report["budget"]["tool_attempts_used"] == 16 + 2 * 3
    store.close()


def test_top_eight_expansion_fails_atomically_when_tool_budget_is_short(
    tmp_path: Path,
) -> None:
    environment = TopEightCandidateBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidates = seed_candidate_overview(
        store,
        environment,
        exact_series_count=8,
        source_metric_indices=range(8),
    )
    model = RecordedModelAdapter(
        [
            {
                "decision_summary": "trigger deterministic schedule",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidates[0]["candidate_id"]},
                    }
                ],
            }
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(max_tool_attempts=31),
        ).run("candidate-top-eight-budget", tmp_path / "bundle")
    )

    assert result.status == "budget_exhausted"
    assert not store.records("tool_effect")
    assert not store.records("observation")[1:]
    store.close()


def test_top_eight_pairs_are_disposed_serially_before_abstain(
    tmp_path: Path,
) -> None:
    run_id = "candidate-top-eight-serial"
    environment = TopEightCandidateBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidates = seed_candidate_overview(store, environment, exact_series_count=8)
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "trigger deterministic schedule",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_candidate",
                    "arguments": {"candidate_id": candidates[-1]["candidate_id"]},
                }
            ],
        }
    ]
    records.extend(
        {
            "decision_summary": f"dispose candidate {index}",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": candidate["entity_id"],
                    "statement": candidate["fault_type"],
                    "status": "inconclusive",
                    "supporting_evidence_refs": [
                        f"{run_id}:evidence:decision:{run_id}:0001:{index * 2}",
                        f"{run_id}:evidence:decision:{run_id}:0001:{index * 2 + 1}",
                    ],
                    "open_questions": [],
                }
            ],
        }
        for index, candidate in enumerate(candidates[:8])
    )
    records.append(
        {
            "decision_summary": "all scheduled candidates disposed",
            "operations": [
                {"type": "finalize", "abstain_reason": "no supported candidate"}
            ],
        }
    )
    model = CapturingModel(records)

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert result.model_calls == 10
    assert result.report["abstain_reason"] == "no supported candidate"
    assert len(store.records("hypothesis")) == 8
    assert (
        max(
            cast_dict(item.payload)["estimated_tokens"]
            for item in store.records("context_manifest")
        )
        <= 12_000
    )
    store.close()


def test_enforce_rejects_abstain_while_complete_candidate_pair_is_pending(
    tmp_path: Path,
) -> None:
    environment = TopEightCandidateBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidates = seed_candidate_overview(store, environment, exact_series_count=8)
    model = CapturingModel(
        [
            {
                "decision_summary": "trigger deterministic schedule",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidates[0]["candidate_id"]},
                    }
                ],
            },
            *[
                {
                    "decision_summary": "abstain too early",
                    "operations": [
                        {"type": "finalize", "abstain_reason": "not investigated"}
                    ],
                }
                for _index in range(3)
            ],
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("candidate-pending-abstain", tmp_path / "bundle")
    )

    assert result.status == "no_progress"
    rejected = [
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "finalize_rejected"
    ]
    assert rejected
    assert "complete candidate pair remains undisposed" in str(rejected[0])
    assert str(candidates[0]["entity_id"]) in str(rejected[0])
    store.close()


def test_provider_closeout_defers_model_abstain_until_exact_pair_is_terminal(
    tmp_path: Path,
) -> None:
    run_id = "candidate-deferred-abstain"
    environment = TopEightCandidateBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidates = seed_candidate_overview(
        store, environment, exact_series_count=3, has_log_source=False
    )
    candidate = candidates[-1]
    abstain_operation = {
        "type": "finalize",
        "hypothesis_ids": [],
        "path_ids": [],
        "occurrence_evidence_refs": [],
        "abstain_reason": "no supported candidate after exact comparison",
        "unresolved_questions": ["candidate remains to be dispositioned"],
    }
    model = AuditedCapturingModel(
        [
            {
                "decision_summary": "query exact candidate pair",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidates[0]["candidate_id"]},
                    }
                ],
            },
            *[
                {
                    "decision_summary": f"terminally reject candidate {index}",
                    "operations": [
                        {
                            "type": "update_hypothesis",
                            "action": "create",
                            "subject": "model-supplied-wrong-subject",
                            "statement": "model-supplied prose",
                            "status": "refuted",
                        }
                    ],
                }
                for index in range(2)
            ],
            {
                "decision_summary": "provider closeout abstain",
                "operations": [abstain_operation],
            },
            {
                "decision_summary": "terminally disposition exact pair",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": "model-supplied-wrong-subject",
                        "statement": "model-supplied prose",
                        "status": "inconclusive",
                    }
                ],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(max_model_calls=8),
        ).run(run_id, tmp_path / "bundle")
    )

    events = [
        (record.stable_key, cast_dict(record.payload))
        for record in store.records("investigation_event")
    ]
    rejected = [
        (key, event)
        for key, event in events
        if event.get("event_type") == "finalize_rejected"
    ]
    intents = [
        (key, event)
        for key, event in events
        if event.get("event_type") == "deferred_finalize_intent"
    ]
    dispositions = [
        (key, event)
        for key, event in events
        if event.get("event_type") == "candidate_disposition_applied"
    ]
    applied = [
        (key, event)
        for key, event in events
        if event.get("event_type") == "deferred_finalize_applied"
    ]
    source_decision_id = f"decision:{run_id}:0004"
    expected_pair = {
        "entity_id": candidate["entity_id"],
        "metric_id": candidate["metric_id"],
        "reference_evidence_ids": [f"{run_id}:evidence:decision:{run_id}:0001:4"],
        "incident_evidence_ids": [f"{run_id}:evidence:decision:{run_id}:0001:5"],
    }

    assert result.status == "completed_partial"
    assert result.model_calls == 5
    assert len(model.inputs) == 5
    assert result.report["hypothesis_ids"] == []
    assert result.report["abstain_reason"] == abstain_operation["abstain_reason"]
    assert (
        result.report["unresolved_questions"]
        == abstain_operation["unresolved_questions"]
    )
    assert result.report["budget"]["provider_attempts_used"] == 5
    assert result.report["budget"]["validator_retries_used"] == 1
    assert len(rejected) == len(intents) == len(applied) == 1
    assert len(dispositions) == 3
    assert intents[0][0] == (
        f"event:{run_id}:{source_decision_id}:0:deferred_finalize_intent"
    )
    assert intents[0][1]["policy"] == "deferred-model-abstain-v1"
    assert intents[0][1]["source_decision_id"] == source_decision_id
    assert intents[0][1]["source_decision"] == cast_dict(
        store.get(source_decision_id).payload  # type: ignore[union-attr]
    )
    assert intents[0][1]["original_operation"] == abstain_operation
    assert intents[0][1]["pair_identity"] == expected_pair
    assert applied[0][0] == (
        f"event:{run_id}:{source_decision_id}:0:deferred_finalize_applied"
    )
    assert applied[0][1]["intent_ref"] == intents[0][0]
    assert applied[0][1]["disposition_ref"] == dispositions[-1][0]
    assert applied[0][1]["pair_identity"] == expected_pair
    assert applied[0][1]["terminal_status"] == "inconclusive"
    assert result.report["benchmark_recommendation"]["policy"] == (
        "terminal-candidate-v4"
    )
    recommendation_events = [
        event
        for _key, event in events
        if event.get("event_type") == "benchmark_recommendation_selected"
    ]
    assert recommendation_events == [
        {
            "event_type": "benchmark_recommendation_selected",
            "visibility": "audit_only",
            "recommendation": result.report["benchmark_recommendation"],
        }
    ]
    store.close()


def deferred_finalize_fixture(
    tmp_path: Path,
    *,
    run_id: str,
    terminal_status: Literal["supported", "refuted", "inconclusive", "open"] = (
        "inconclusive"
    ),
    include_rejection: bool = True,
    binding_metric_id: str = "metric-exact",
) -> tuple[RootWalkRuntime, RunStore, dict[str, Any], FinalizeOperation]:
    store = RunStore(tmp_path / f"{run_id}.sqlite3")
    pair = {
        "entity_id": "bank:entity:Tomcat01",
        "metric_id": "metric-exact",
        "reference_evidence_ids": ["evidence-reference"],
        "incident_evidence_ids": ["evidence-incident"],
    }
    operation = FinalizeOperation(
        type="finalize",
        hypothesis_ids=[],
        path_ids=[],
        abstain_reason="model-authored deferred abstain",
        unresolved_questions=["exact pair disposition pending"],
    )
    source_decision_id = f"decision:{run_id}:0004"
    source_decision = Decision(
        decision_summary="provider closeout abstain",
        operations=[operation],
    ).model_dump(mode="json")
    source_decision["decision_id"] = source_decision_id
    store.append("decision", source_decision_id, source_decision)
    expected_rejection = (
        "cannot abstain while a complete candidate pair remains undisposed: "
        + json.dumps(pair, ensure_ascii=False)
    )
    if include_rejection:
        store.append(
            "investigation_event",
            f"event:{run_id}:{source_decision_id}:0:finalize_rejected",
            {
                "event_type": "finalize_rejected",
                "errors": [expected_rejection],
                "feedback": [],
            },
        )
    store.append(
        "investigation_event",
        f"event:{run_id}:{source_decision_id}:0:deferred_finalize_intent",
        {
            "event_type": "deferred_finalize_intent",
            "visibility": "audit_only",
            "policy": "deferred-model-abstain-v1",
            "source_decision_id": source_decision_id,
            "source_decision": source_decision,
            "original_operation": operation.model_dump(mode="json"),
            "pair_identity": pair,
            "pair_identity_hash": RootWalkRuntime._hash_json(pair),  # pyright: ignore[reportPrivateUsage]
            "rejection_error": expected_rejection,
        },
    )
    evidence_refs = ["evidence-reference", "evidence-incident"]
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-exact",
        subject=cast(str, pair["entity_id"]),
        statement="fault-exact",
        status=terminal_status,
        supporting_evidence_refs=(
            evidence_refs if terminal_status in {"open", "supported"} else []
        ),
        opposing_evidence_refs=(
            evidence_refs if terminal_status in {"refuted", "inconclusive"} else []
        ),
    )
    store.append("hypothesis", "hypothesis:exact:terminal", hypothesis)
    applied_operation = {
        "type": "update_hypothesis",
        "action": "create",
        "hypothesis_id": None,
        "subject": pair["entity_id"],
        "statement": hypothesis.statement,
        "status": terminal_status,
        "supporting_evidence_refs": hypothesis.supporting_evidence_refs,
        "opposing_evidence_refs": hypothesis.opposing_evidence_refs,
        "open_questions": [],
    }
    store.append(
        "investigation_event",
        f"event:{run_id}:decision:{run_id}:0005:0:candidate_disposition_applied",
        {
            "event_type": "candidate_disposition_applied",
            "visibility": "audit_only",
            "decision_id": f"decision:{run_id}:0005",
            "candidate_id": "candidate-exact",
            "hypothesis_id": hypothesis.hypothesis_id,
            "status": terminal_status,
            "subject": pair["entity_id"],
            "metric_id": binding_metric_id,
            "selected_fault_type": hypothesis.statement,
            "reference_evidence_id": "evidence-reference",
            "incident_evidence_id": "evidence-incident",
            "original_operation": applied_operation,
            "applied_operation": applied_operation,
        },
    )
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / f"{run_id}-artifacts"),
        RuntimeConfig(max_model_calls=5),
    )
    runtime._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    return runtime, store, pair, operation


@pytest.mark.parametrize(
    "disqualifier",
    (
        "not_provider_closeout",
        "selected_ids",
        "reference_error",
        "comparison_error",
        "extra_quality_error",
        "multiple_pairs",
        "not_source_operation",
        "observe_only",
    ),
)
def test_deferred_finalize_intent_requires_exact_model_authored_closeout_abstain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    disqualifier: str,
) -> None:
    run_id = f"deferred-intent-{disqualifier}"
    config = RuntimeConfig(
        validator_mode=("observe_only" if disqualifier == "observe_only" else "enforce")
    )
    store = RunStore(tmp_path / f"{disqualifier}.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / f"artifacts-{disqualifier}"),
        config,
    )
    runtime._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    runtime._decision_id = f"decision:{run_id}:0001"  # pyright: ignore[reportPrivateUsage]
    runtime._provider_attempts = (  # pyright: ignore[reportPrivateUsage]
        config.max_model_calls - 4
    )
    pair = {
        "entity_id": "bank:entity:Tomcat01",
        "metric_id": "metric-exact",
        "reference_evidence_ids": ["evidence-reference"],
        "incident_evidence_ids": ["evidence-incident"],
    }
    operation = FinalizeOperation(
        type="finalize", abstain_reason="model-authored abstain"
    )
    source_operation = operation
    if disqualifier == "selected_ids":
        operation = operation.model_copy(update={"hypothesis_ids": ["invented"]})
        source_operation = operation
    elif disqualifier == "not_source_operation":
        source_operation = operation.model_copy(update={"abstain_reason": "other"})
    runtime._pending_operation = source_operation  # pyright: ignore[reportPrivateUsage]
    source_decision = (
        {
            "decision_summary": "source",
            "operations": [operation.model_dump(mode="json")],
        }
        if disqualifier == "selected_ids"
        else Decision(decision_summary="source", operations=[operation]).model_dump(
            mode="json"
        )
    )
    store.append(
        "decision",
        runtime._decision_id,  # pyright: ignore[reportPrivateUsage]
        source_decision,
    )
    pairs = [pair, {**pair, "metric_id": "metric-other"}]
    monkeypatch.setattr(
        runtime,
        "_pending_metric_pair_dispositions",
        lambda: pairs if disqualifier == "multiple_pairs" else [pair],
    )
    if disqualifier == "not_provider_closeout":
        runtime._provider_attempts = 0  # pyright: ignore[reportPrivateUsage]
    expected_error = (
        "cannot abstain while a complete candidate pair remains undisposed: "
        + json.dumps(pair, ensure_ascii=False)
    )
    reference_errors = (
        ["unknown reference"] if disqualifier == "reference_error" else []
    )
    comparison_errors = (
        ["comparison incomplete"] if disqualifier == "comparison_error" else []
    )
    quality_errors = [expected_error]
    if disqualifier == "extra_quality_error":
        quality_errors.append("another quality failure")

    intent = runtime._deferred_finalize_intent(  # pyright: ignore[reportPrivateUsage]
        operation,
        reference_errors,
        quality_errors,
        comparison_errors,
    )

    assert intent is None
    store.close()


@pytest.mark.parametrize(
    "guard",
    (
        "missing_rejection",
        "latest_binding_open",
        "wrong_pair_binding",
        "later_mismatched_binding",
        "supported_hypothesis",
        "other_open_hypothesis",
        "other_pending_pair",
        "finalization_error",
        "comparison_error",
    ),
)
def test_deferred_finalize_application_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guard: str,
) -> None:
    runtime, store, pair, _operation = deferred_finalize_fixture(
        tmp_path,
        run_id=f"deferred-guard-{guard}",
        terminal_status=(
            "open"
            if guard == "latest_binding_open"
            else "supported"
            if guard == "supported_hypothesis"
            else "inconclusive"
        ),
        include_rejection=guard != "missing_rejection",
        binding_metric_id=(
            "metric-other" if guard == "wrong_pair_binding" else "metric-exact"
        ),
    )
    if guard == "other_pending_pair":
        monkeypatch.setattr(
            runtime,
            "_pending_metric_pair_dispositions",
            lambda: [{**pair, "metric_id": "metric-other"}],
        )
    elif guard == "later_mismatched_binding":
        prior_binding = cast_dict(store.records("investigation_event")[-1].payload)
        store.append(
            "investigation_event",
            "binding:later-mismatched-pair",
            {
                **prior_binding,
                "reference_evidence_id": "evidence-reference-new",
                "incident_evidence_id": "evidence-incident-new",
            },
        )
    elif guard == "other_open_hypothesis":
        runtime._hypotheses["hypothesis-other-open"] = Hypothesis(  # pyright: ignore[reportPrivateUsage]
            hypothesis_id="hypothesis-other-open",
            subject="bank:entity:Tomcat01",
            statement="unresolved alternate",
            status="open",
        )
    elif guard == "finalization_error":

        def finalization_error(
            _operation: FinalizeOperation,
        ) -> tuple[list[str], list[str]]:
            return ["blocked reference"], []

        monkeypatch.setattr(
            runtime,
            "_finalization_errors",
            finalization_error,
        )
    elif guard == "comparison_error":

        def comparison_error(_hypothesis_ids: Sequence[str]) -> list[str]:
            return ["blocked comparison"]

        monkeypatch.setattr(
            runtime,
            "_comparison_quality_errors",
            comparison_error,
        )

    assert runtime._apply_deferred_finalize() is False  # pyright: ignore[reportPrivateUsage]
    assert runtime._report is None  # pyright: ignore[reportPrivateUsage]
    assert not any(
        cast_dict(record.payload).get("event_type") == "deferred_finalize_applied"
        for record in store.records("investigation_event")
    )
    store.close()


def test_deferred_finalize_reuses_at_provider_cap_without_model_or_request(
    tmp_path: Path,
) -> None:
    run_id = "deferred-resume"
    runtime, store, _pair, operation = deferred_finalize_fixture(
        tmp_path, run_id=run_id
    )
    runtime._model_calls = runtime.config.max_model_calls - 1  # pyright: ignore[reportPrivateUsage]
    runtime._provider_attempts = runtime.config.max_model_calls  # pyright: ignore[reportPrivateUsage]
    no_call = cast(NoCallModel, runtime.model)
    request_count = len(store.records("model_request"))
    attempt_count = len(store.records("attempt"))

    state = asyncio.run(
        runtime._build_context_untraced(cast(Any, {}))  # pyright: ignore[reportPrivateUsage]
    )

    assert state.get("phase") == "build_context:deferred_finalize"
    assert runtime._report is not None  # pyright: ignore[reportPrivateUsage]
    assert runtime._report["status"] == "completed_partial"  # pyright: ignore[reportPrivateUsage]
    assert runtime._report["abstain_reason"] == operation.abstain_reason  # pyright: ignore[reportPrivateUsage]
    assert runtime._failure is None  # pyright: ignore[reportPrivateUsage]
    assert runtime._model_calls == runtime.config.max_model_calls - 1  # pyright: ignore[reportPrivateUsage]
    assert runtime._provider_attempts == runtime.config.max_model_calls  # pyright: ignore[reportPrivateUsage]
    assert no_call.calls == 0
    assert len(store.records("model_request")) == request_count
    assert len(store.records("attempt")) == attempt_count
    assert (
        sum(
            cast_dict(record.payload).get("event_type") == "deferred_finalize_applied"
            for record in store.records("investigation_event")
        )
        == 1
    )
    store.close()

    replay_store = RunStore(tmp_path / f"{run_id}.sqlite3")
    replay_no_call = NoCallModel()
    replay = asyncio.run(
        RootWalkRuntime(
            _BankEnvironment(PUBLIC_INPUT),
            replay_no_call,
            replay_store,
            ArtifactStore(tmp_path / f"{run_id}-replay-artifacts"),
            RuntimeConfig(max_model_calls=5),
        ).run(run_id, tmp_path / "replay-bundle")
    )
    assert replay.status == "completed_partial"
    assert replay.report["abstain_reason"] == operation.abstain_reason
    assert replay_no_call.calls == 0
    assert (
        sum(
            cast_dict(record.payload).get("event_type") == "deferred_finalize_applied"
            for record in replay_store.records("investigation_event")
        )
        == 1
    )
    replay_store.close()


@pytest.mark.parametrize(
    ("hard_failure", "expected_phase", "expected_failure"),
    (
        ("deadline", "build_context:deadline", "deadline"),
        ("artifact", "build_context:artifact_budget", "artifact_bytes"),
    ),
)
def test_hard_failure_prevents_deferred_finalize_reuse(
    tmp_path: Path,
    hard_failure: str,
    expected_phase: str,
    expected_failure: str,
) -> None:
    runtime, store, _pair, _operation = deferred_finalize_fixture(
        tmp_path,
        run_id=f"deferred-hard-failure-{hard_failure}",
    )
    no_call = cast(NoCallModel, runtime.model)
    if hard_failure == "deadline":
        runtime._deadline_at = 1.0  # pyright: ignore[reportPrivateUsage]
    else:
        runtime.artifacts.put_json({"over": "artifact budget"})
        runtime._max_artifact_bytes = 1  # pyright: ignore[reportPrivateUsage]

    state = asyncio.run(
        runtime._build_context_untraced(cast(Any, {}))  # pyright: ignore[reportPrivateUsage]
    )

    assert state.get("phase") == expected_phase
    assert runtime._terminal_status == "budget_exhausted"  # pyright: ignore[reportPrivateUsage]
    assert runtime._failure == expected_failure  # pyright: ignore[reportPrivateUsage]
    assert runtime._report is None  # pyright: ignore[reportPrivateUsage]
    assert no_call.calls == 0
    assert not any(
        cast_dict(record.payload).get("event_type") == "deferred_finalize_applied"
        for record in store.records("investigation_event")
    )
    store.close()


def test_invalid_deferred_finalize_preserves_model_budget_failure(
    tmp_path: Path,
) -> None:
    runtime, store, _pair, _operation = deferred_finalize_fixture(
        tmp_path,
        run_id="deferred-budget-fallback",
        terminal_status="open",
    )
    runtime._model_calls = runtime.config.max_model_calls  # pyright: ignore[reportPrivateUsage]

    state = asyncio.run(
        runtime._build_context_untraced(cast(Any, {}))  # pyright: ignore[reportPrivateUsage]
    )

    assert state.get("phase") == "build_context:budget_exhausted"
    assert runtime._terminal_status == "budget_exhausted"  # pyright: ignore[reportPrivateUsage]
    assert runtime._failure == "model_calls"  # pyright: ignore[reportPrivateUsage]
    assert not any(
        cast_dict(record.payload).get("event_type") == "deferred_finalize_applied"
        for record in store.records("investigation_event")
    )
    store.close()


@pytest.mark.parametrize(
    "case",
    ["unknown", "duplicate", "mixed", "raw-first"],
)
def test_query_candidate_invalid_first_batches_execute_nothing(
    tmp_path: Path,
    case: str,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    first, second, _same_family = seed_candidate_overview(store, environment)

    def candidate_operation(candidate_id: str) -> dict[str, Any]:
        return {
            "type": "query",
            "tool": "query_candidate",
            "arguments": {"candidate_id": candidate_id},
        }

    cases: dict[str, list[dict[str, Any]]] = {
        "unknown": [candidate_operation("unknown-candidate")],
        "duplicate": [
            candidate_operation(cast(str, first["candidate_id"])),
            candidate_operation(cast(str, first["candidate_id"])),
        ],
        "mixed": [
            candidate_operation(cast(str, first["candidate_id"])),
            paired_metric_queries(second)[0],
        ],
        "raw-first": paired_metric_queries(first),
    }
    model = CapturingModel(
        [
            {"decision_summary": f"invalid {case}", "operations": cases[case]},
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("candidate-invalid", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(store.records("observation")) == 1
    assert not store.records("tool_effect")
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_query_candidate_allows_two_distinct_candidates_from_same_family(
    tmp_path: Path,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    first, second, _same_family = seed_candidate_overview(store, environment)
    candidates = [first, second]
    model = RecordedModelAdapter(
        [
            {
                "decision_summary": "compare same-family candidates",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                    for candidate in candidates
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("candidate-same-family", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    effects = [cast_dict(item.payload) for item in store.records("tool_effect")]
    assert len(effects) == 4
    assert [item["candidate_id"] for item in effects] == [
        first["candidate_id"],
        first["candidate_id"],
        second["candidate_id"],
        second["candidate_id"],
    ]
    assert [item["logical_operation_index"] for item in effects] == [0, 0, 1, 1]
    assert result.report["budget"]["tool_attempts_used"] == 4
    store.close()


def test_enforce_blocks_submit_path_until_two_comparison_keys_complete(
    tmp_path: Path,
) -> None:
    run_id = "comparison-submit-enforce"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    first, second = seed_candidate_overview(store, environment)[:2]
    seed_comparison_pair(store, environment, first, incident_status="no_data")
    metric_id = seed_tool_evidence(
        store, environment, "query_metric", subject=cast(str, first["entity_id"])
    )
    log_id = seed_tool_evidence(
        store,
        environment,
        "query_log_samples",
        subject=cast(str, first["entity_id"]),
        paired_log_summary=True,
    )
    hypothesis_id = "comparison-supported"
    store.append(
        "hypothesis",
        hypothesis_id,
        Hypothesis(
            hypothesis_id=hypothesis_id,
            subject=cast(str, first["entity_id"]),
            statement=cast(str, first["fault_type"]),
            status="supported",
            supporting_evidence_refs=[metric_id, log_id],
        ),
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "submit path too early",
                "operations": [
                    {
                        "type": "submit_path",
                        "hypothesis_id": hypothesis_id,
                        "root_cause_entity_id": first["entity_id"],
                        "symptom_entity_ids": [first["entity_id"]],
                        "steps": [],
                    }
                ],
            },
            {
                "decision_summary": "abstain",
                "operations": [
                    {"type": "finalize", "abstain_reason": "comparison incomplete"}
                ],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert not store.records("path")
    assert "comparison coverage is incomplete" in str(model.inputs[1])
    assert str(second["candidate_id"]) in str(model.inputs[1])
    store.close()


def test_observe_only_finalizes_with_idempotent_comparison_audit(
    tmp_path: Path,
) -> None:
    run_id = "comparison-finalize-observe"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    first = seed_candidate_overview(store, environment)[0]
    seed_comparison_pair(store, environment, first)
    metric_id = seed_tool_evidence(
        store, environment, "query_metric", subject=cast(str, first["entity_id"])
    )
    log_id = seed_tool_evidence(
        store,
        environment,
        "query_log_samples",
        subject=cast(str, first["entity_id"]),
        paired_log_summary=True,
    )
    hypothesis_id = "comparison-observe-supported"
    store.append(
        "hypothesis",
        hypothesis_id,
        Hypothesis(
            hypothesis_id=hypothesis_id,
            subject=cast(str, first["entity_id"]),
            statement=cast(str, first["fault_type"]),
            status="supported",
            supporting_evidence_refs=[metric_id, log_id],
        ),
    )
    records = [
        {
            "decision_summary": "finalize observed draft",
            "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
        }
    ]

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            RecordedModelAdapter(records),
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode="observe_only"),
        ).run(run_id, tmp_path / "bundle")
    )

    audits = [
        cast_dict(event.payload)
        for event in store.records("investigation_event")
        if cast_dict(event.payload).get("event_type") == "comparison_quality_audit"
    ]
    assert result.status == "completed_partial"
    assert result.report["findings"][0]["hypothesis_id"] == hypothesis_id
    assert len(audits) == 1
    assert audits[0]["visibility"] == "audit_only"
    assert audits[0]["gate"] == "finalize"
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_query_candidate_requires_complete_pair_disposition_before_next_candidate(
    tmp_path: Path,
    validator_mode: str,
) -> None:
    run_id = f"candidate-disposition-{validator_mode}"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    first, second, third = seed_candidate_overview(store, environment)
    first_evidence = [
        f"{run_id}:evidence:decision:{run_id}:0001:{index}" for index in range(2)
    ]
    model = CapturingModel(
        [
            {
                "decision_summary": "query initial comparison candidates",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                    for candidate in (first, second)
                ],
            },
            {
                "decision_summary": "query next candidate too early",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": third["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "dispose first pair",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": first["entity_id"],
                        "statement": first["fault_type"],
                        "status": "refuted",
                        "opposing_evidence_refs": first_evidence,
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    effects = [cast_dict(item.payload) for item in store.records("tool_effect")]
    assert len(effects) == 4
    assert not any(
        "decision:" + run_id + ":0002" in item.stable_key
        for item in store.records("tool_effect")
    )
    assert "update_hypothesis" in str(model.inputs[2])
    assert str(first["entity_id"]) in str(model.inputs[2])
    assert str(first["metric_id"]) in str(model.inputs[2])
    rejection = next(
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "decision_rejected"
    )
    assert cast(str, rejection["summary"]).count('"metric_id"') == 1
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
@pytest.mark.parametrize("invalid_update", ["unrelated", "single-side"])
def test_pending_pair_binds_hypothesis_update_to_both_sides(
    tmp_path: Path,
    validator_mode: str,
    invalid_update: str,
) -> None:
    run_id = f"pending-update-{validator_mode}-{invalid_update}"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment, key_limit=1)[0]
    evidence_ids = [
        f"{run_id}:evidence:decision:{run_id}:0001:{index}" for index in range(2)
    ]
    unrelated_subject = next(
        node.canonical_id
        for node in environment.topology.nodes
        if node.canonical_id != candidate["entity_id"]
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "query candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "submit invalid hypothesis update",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": (
                            unrelated_subject
                            if invalid_update == "unrelated"
                            else candidate["entity_id"]
                        ),
                        "statement": "unrelated or incomplete candidate",
                        "status": "open",
                        "supporting_evidence_refs": (
                            evidence_ids
                            if invalid_update == "unrelated"
                            else evidence_ids[:1]
                        ),
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    hypotheses = [cast_dict(item.payload) for item in store.records("hypothesis")]
    assert len(hypotheses) == 1
    assert hypotheses[0]["subject"] == candidate["entity_id"]
    assert hypotheses[0]["supporting_evidence_refs"] == evidence_ids
    assert result.report["budget"]["validator_retries_used"] == 0
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
@pytest.mark.parametrize(
    ("invalid_identity", "accepted"),
    [("multiple-pairs", True), ("ambiguous", False), ("wrong-fault", False)],
)
def test_pending_pair_rejects_invalid_pair_to_hypothesis_identity(
    tmp_path: Path,
    validator_mode: str,
    invalid_identity: str,
    accepted: bool,
) -> None:
    run_id = f"pair-identity-{validator_mode}-{invalid_identity}"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    first, second = seed_candidate_overview(store, environment)[:2]
    evidence_ids = [
        f"{run_id}:evidence:decision:{run_id}:0001:{index}" for index in range(4)
    ]
    selected_refs = (
        evidence_ids if invalid_identity == "multiple-pairs" else evidence_ids[:2]
    )
    statement = (
        f"Both {first['fault_type']} and {second['fault_type']} remain possible."
        if invalid_identity == "ambiguous"
        else second["fault_type"]
        if invalid_identity == "wrong-fault"
        else first["fault_type"]
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "query two candidates",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                    for candidate in (first, second)
                ],
            },
            {
                "decision_summary": "submit invalid pair identity",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": first["entity_id"],
                        "statement": statement,
                        "status": "inconclusive",
                        "opposing_evidence_refs": selected_refs,
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(store.records("tool_effect")) == 4
    hypotheses = [cast_dict(item.payload) for item in store.records("hypothesis")]
    assert bool(hypotheses) is accepted
    if accepted:
        assert hypotheses[0]["subject"] == first["entity_id"]
        assert hypotheses[0]["opposing_evidence_refs"] == evidence_ids[:2]
    else:
        assert "outside the current pair" in str(model.inputs[2])
    assert result.report["budget"]["validator_retries_used"] == (0 if accepted else 1)
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
@pytest.mark.parametrize(
    (
        "fault_choice",
        "log_subject",
        "paired_log_summary",
        "filtered_log",
        "accepted",
    ),
    [
        ("primary", None, False, False, True),
        ("alias", None, False, False, False),
        ("alias", "same", False, False, False),
        ("alias", "same", True, False, False),
        ("alias", "same", True, True, True),
        ("alias", "other", True, True, False),
    ],
)
def test_candidate_alias_fault_requires_same_subject_log_evidence(
    tmp_path: Path,
    validator_mode: str,
    fault_choice: str,
    log_subject: str | None,
    paired_log_summary: bool,
    filtered_log: bool,
    accepted: bool,
) -> None:
    run_id = f"alias-gate-{validator_mode}-{fault_choice}-{log_subject}"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    primary, alias = seed_candidate_overview(
        store,
        environment,
        key_limit=1,
        has_log_source=True,
        first_alias="CPU saturation",
    )
    pair_evidence = [
        f"{run_id}:evidence:decision:{run_id}:0001:{index}" for index in (0, 1)
    ]
    update_refs = list(pair_evidence)
    if log_subject is not None:
        subject = (
            cast(str, primary["entity_id"])
            if log_subject == "same"
            else next(
                node.canonical_id
                for node in environment.topology.nodes
                if node.canonical_id != primary["entity_id"]
            )
        )
        update_refs.append(
            seed_tool_evidence(
                store,
                environment,
                "query_log_samples",
                subject=subject,
                paired_log_summary=paired_log_summary,
                filtered_log=filtered_log,
            )
        )
    statement = (
        cast(str, primary["fault_type"])
        if fault_choice == "primary"
        else cast(str, alias["fault_type"])
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "query comparison candidates",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                    for candidate in (primary,)
                ],
            },
            {
                "decision_summary": "record memory hypothesis",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": primary["entity_id"],
                        "statement": statement,
                        "status": "open",
                        "supporting_evidence_refs": update_refs,
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert bool(store.records("hypothesis")) is accepted
    assert result.report["budget"]["validator_retries_used"] == (0 if accepted else 1)
    if not accepted:
        assert "qualifier corroboration" in str(model.inputs[2])
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_open_pair_hypothesis_allows_revision_but_not_next_candidate_until_terminal(
    tmp_path: Path,
    validator_mode: str,
) -> None:
    run_id = f"pending-open-{validator_mode}"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    first, second, third = seed_candidate_overview(store, environment)
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0002:0"
    evidence_ids = [
        f"{run_id}:evidence:decision:{run_id}:0001:{index}" for index in range(2)
    ]
    candidate_query = {
        "type": "query",
        "tool": "query_candidate",
        "arguments": {"candidate_id": third["candidate_id"]},
    }
    model = CapturingModel(
        [
            {
                "decision_summary": "query initial comparison candidates",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                    for candidate in (first, second)
                ],
            },
            {
                "decision_summary": "open hypothesis on both sides",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": first["entity_id"],
                        "statement": f"Investigating {first['fault_type']}.",
                        "status": "open",
                        "supporting_evidence_refs": evidence_ids,
                    }
                ],
            },
            {
                "decision_summary": "try next candidate before terminal status",
                "operations": [candidate_query],
            },
            {
                "decision_summary": "close the first hypothesis",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "revise",
                        "hypothesis_id": hypothesis_id,
                        "subject": first["entity_id"],
                        "statement": first["fault_type"],
                        "status": "refuted",
                        "opposing_evidence_refs": evidence_ids,
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(store.records("hypothesis")) == 2
    assert len(store.records("tool_effect")) == 4
    assert "update_hypothesis" in str(model.inputs[3])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_query_candidate_pending_pair_gate_survives_resume(tmp_path: Path) -> None:
    run_id = "candidate-disposition-resume"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    environment = _BankEnvironment(PUBLIC_INPUT)
    first, second, third = seed_candidate_overview(first_store, environment)
    first_runtime = RootWalkRuntime(
        environment,
        RecordedModelAdapter(
            [
                {
                    "decision_summary": "query initial comparison candidates",
                    "operations": [
                        {
                            "type": "query",
                            "tool": "query_candidate",
                            "arguments": {"candidate_id": candidate["candidate_id"]},
                        }
                        for candidate in (first, second)
                    ],
                }
            ]
        ),
        first_store,
        ArtifactStore(artifact_root),
    )
    first_runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "execute_query"))
    assert len(first_store.records("tool_effect")) == 4
    first_store.close()

    second_store = RunStore(database)
    model = CapturingModel(
        [
            {
                "decision_summary": "query next candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": third["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )
    result = asyncio.run(
        RootWalkRuntime(
            _BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(second_store.records("tool_effect")) == 4
    assert "update_hypothesis" in str(model.inputs[1])
    second_store.close()


def test_pending_pair_hypothesis_update_gate_survives_resume(tmp_path: Path) -> None:
    run_id = "pending-update-resume"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    environment = _BankEnvironment(PUBLIC_INPUT)
    first_candidate, primary_candidate, alias_candidate = seed_candidate_overview(
        first_store, environment
    )
    first_runtime = RootWalkRuntime(
        environment,
        RecordedModelAdapter(
            [
                {
                    "decision_summary": "query candidate",
                    "operations": [
                        {
                            "type": "query",
                            "tool": "query_candidate",
                            "arguments": {"candidate_id": item["candidate_id"]},
                        }
                        for item in (first_candidate, primary_candidate)
                    ],
                }
            ]
        ),
        first_store,
        ArtifactStore(artifact_root),
    )
    first_runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "execute_query"))
    first_store.close()

    evidence_ids = [
        f"{run_id}:evidence:decision:{run_id}:0001:{index}" for index in (2, 3)
    ]
    second_store = RunStore(database)
    model = CapturingModel(
        [
            {
                "decision_summary": "submit wrong fault hypothesis",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": primary_candidate["entity_id"],
                        "statement": alias_candidate["fault_type"],
                        "status": "open",
                        "supporting_evidence_refs": evidence_ids,
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )
    result = asyncio.run(
        RootWalkRuntime(
            _BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert not second_store.records("hypothesis")
    assert "outside the current pair" in str(model.inputs[1])
    second_store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
@pytest.mark.parametrize("terminal_status", ["inconclusive", "refuted"])
def test_candidate_disposition_binding_corrects_identity_refs_and_action(
    tmp_path: Path, validator_mode: str, terminal_status: str
) -> None:
    run_id = f"candidate-binding-{validator_mode}-{terminal_status}"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(
        store, environment, key_limit=1, has_log_source=False
    )[0]
    unrelated_subject = next(
        node.canonical_id
        for node in environment.topology.nodes
        if node.canonical_id != candidate["entity_id"]
    )
    original_operation = {
        "type": "update_hypothesis",
        "action": "revise",
        "hypothesis_id": "model-invented-id",
        "subject": unrelated_subject,
        "statement": "model prose without a public fault type",
        "status": terminal_status,
        "supporting_evidence_refs": ["model-invented-evidence"],
    }
    model = CapturingModel(
        [
            {
                "decision_summary": "query candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "dispose candidate",
                "operations": [original_operation],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    hypothesis = cast_dict(store.records("hypothesis")[-1].payload)
    pair_refs = [
        f"{run_id}:evidence:decision:{run_id}:0001:{index}" for index in range(2)
    ]
    decision = cast_dict(store.get(f"decision:{run_id}:0002").payload)  # type: ignore[union-attr]
    events = [
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "candidate_disposition_applied"
    ]
    assert result.status == "completed_partial"
    assert hypothesis["subject"] == candidate["entity_id"]
    assert hypothesis["statement"] == candidate["fault_type"]
    assert hypothesis["status"] == terminal_status
    assert hypothesis["supporting_evidence_refs"] == []
    assert hypothesis["opposing_evidence_refs"] == pair_refs
    assert decision["operations"][0]["subject"] == unrelated_subject
    assert decision["operations"][0]["hypothesis_id"] == "model-invented-id"
    assert decision["operations"][0]["supporting_evidence_refs"] == [
        "model-invented-evidence"
    ]
    assert events[0]["original_operation"] == decision["operations"][0]
    assert events[0]["applied_operation"]["action"] == "create"
    assert len(events) == 1
    assert len(store.records("tool_effect")) == 2
    assert result.report["budget"]["validator_retries_used"] == 0
    if terminal_status == "refuted":
        assert "benchmark_recommendation" not in result.report
        assert not any(
            cast_dict(item.payload).get("event_type")
            == "benchmark_recommendation_selected"
            for item in store.records("investigation_event")
        )
        store.close()
        return
    recommendation = cast_dict(result.report["benchmark_recommendation"])
    assert recommendation == {
        "schema_version": "1",
        "policy": "terminal-candidate-v4",
        "confidence": "low",
        "usage": "openrca_export_fallback_only",
        "is_finding": False,
        "candidate_id": candidate["candidate_id"],
        "hypothesis": hypothesis,
        "selection": {
            "inconclusive": True,
            "source_strict": False,
            "has_log_source": False,
            "direct_strict": True,
            "ever_open": False,
            "terminal_queue_rank": 0,
        },
        "eligible_terminal_count": 1,
        "audit_ref": (
            f"event:{run_id}:decision:{run_id}:0003:0:benchmark_recommendation_selected"
        ),
    }
    recommendation_events = [
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type")
        == "benchmark_recommendation_selected"
    ]
    assert recommendation_events == [
        {
            "event_type": "benchmark_recommendation_selected",
            "visibility": "audit_only",
            "recommendation": recommendation,
        }
    ]
    store.close()


def test_benchmark_recommendation_prefers_no_log_source_strict_then_terminal_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    runtime._run_id = "recommendation-ranking"  # pyright: ignore[reportPrivateUsage]
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "candidate-source",
            "source_paired_metric_summary": strict_paired_metric_summary(),
            "source_pressure_direction": "higher_is_more_pressure",
            "has_log_source": False,
        },
        {
            "candidate_id": "candidate-plain",
            "has_log_source": False,
            "metric_role": "direct",
            "paired_metric_summary": strict_paired_metric_summary(),
            "candidate_pressure_direction": "higher_is_more_pressure",
        },
        {
            "candidate_id": "candidate-open",
            "source_paired_metric_summary": strict_paired_metric_summary(),
            "source_pressure_direction": "higher_is_more_pressure",
            "has_log_source": False,
        },
    ]

    def candidate_queue(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return candidates

    def candidate_registry() -> tuple[dict[str, tuple[str, str, str, str]], bool]:
        return {"candidate": ("entity", "metric", "family", "fault")}, True

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    monkeypatch.setattr(runtime, "_overview_candidate_registry", candidate_registry)
    for index, candidate in enumerate(candidates):
        hypothesis_id = f"hypothesis-{index}"
        runtime._hypotheses[hypothesis_id] = Hypothesis(  # pyright: ignore[reportPrivateUsage]
            hypothesis_id=hypothesis_id,
            subject=f"entity-{index}",
            statement=f"fault-{index}",
            status="inconclusive",
        )
        store.append(
            "investigation_event",
            f"binding-{index}",
            {
                "event_type": "candidate_disposition_applied",
                "hypothesis_id": hypothesis_id,
                "candidate_id": candidate["candidate_id"],
                "subject": f"entity-{index}",
                "selected_fault_type": f"fault-{index}",
                "status": "inconclusive",
            },
        )
    store.append(
        "hypothesis",
        "open-history",
        Hypothesis(
            hypothesis_id="hypothesis-2",
            subject="entity-2",
            statement="fault-2",
            status="open",
        ),
    )

    recommendation = runtime._benchmark_recommendation(  # pyright: ignore[reportPrivateUsage]
        "completed_partial",
        FinalizeOperation(type="finalize", abstain_reason="insufficient support"),
        [],
    )

    assert recommendation is not None
    assert recommendation["policy"] == "terminal-candidate-v4"
    assert recommendation["hypothesis"]["hypothesis_id"] == "hypothesis-0"
    assert recommendation["selection"]["inconclusive"] is True
    assert recommendation["selection"]["source_strict"] is True
    assert recommendation["selection"]["has_log_source"] is False
    assert recommendation["selection"]["direct_strict"] is False
    assert recommendation["selection"]["ever_open"] is False
    assert recommendation["selection"]["terminal_queue_rank"] == 0

    candidates[0]["has_log_source"] = True
    candidates[2]["has_log_source"] = True
    direct = runtime._benchmark_recommendation(  # pyright: ignore[reportPrivateUsage]
        "completed_partial",
        FinalizeOperation(type="finalize", abstain_reason="insufficient support"),
        [],
    )
    assert direct is not None
    assert direct["hypothesis"]["hypothesis_id"] == "hypothesis-1"
    assert direct["selection"]["direct_strict"] is True
    assert direct["selection"]["terminal_queue_rank"] == 1
    store.close()


def test_terminal_v5_switches_only_for_unique_supported_subject_in_base_reason() -> (
    None
):
    def candidate(
        candidate_id: str, hypothesis_id: str, subject: str, statement: str, rank: int
    ) -> dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "hypothesis": {
                "hypothesis_id": hypothesis_id,
                "subject": subject,
                "statement": statement,
            },
            "selection": {
                "source_strict": False,
                "has_log_source": False,
                "direct_strict": rank == 0,
                "terminal_queue_rank": rank,
            },
        }

    base = candidate("candidate-base", "hypothesis-base", "subject-a", "reason", 0)
    same_reason = candidate(
        "candidate-supported", "hypothesis-supported", "subject-b", "reason", 1
    )
    other_reason = candidate(
        "candidate-other", "hypothesis-other", "subject-c", "other reason", 2
    )
    candidates = [base, same_reason, other_reason]

    winner, switch = select_terminal_candidate(
        candidates,
        {"subject-b": ["supported-b"], "subject-c": ["supported-c"]},
    )
    assert winner == same_reason
    assert switch == {
        "supported_subject_match": True,
        "unique_supported_terminal_subject": "subject-b",
        "base_candidate_id": "candidate-base",
        "base_hypothesis_id": "hypothesis-base",
        "overlap_supported_hypothesis_ids": ["supported-b"],
    }

    fallback_subjects: list[dict[str, list[str]]] = [
        {"subject-a": ["supported-a"], "subject-b": ["supported-b"]},
        {"subject-c": ["supported-c"]},
        {},
    ]
    for supported_subjects in fallback_subjects:
        winner, switch = select_terminal_candidate(candidates, supported_subjects)
        assert winner == base
        assert switch is None

    source_tier = candidate(
        "candidate-source", "hypothesis-source", "subject-d", "reason", 2
    )
    source_tier["selection"]["source_strict"] = True
    winner, switch = select_terminal_candidate(
        [base, same_reason, source_tier], {"subject-b": ["supported-b"]}
    )
    assert winner == source_tier
    assert switch is None


def test_terminal_v6_filters_only_explicit_family_mismatches() -> None:
    def candidate(
        candidate_id: str,
        hypothesis_id: str,
        statement: str,
        family: str | None,
        rank: int,
    ) -> dict[str, Any]:
        return {
            "candidate_id": candidate_id,
            "hypothesis": {
                "hypothesis_id": hypothesis_id,
                "statement": statement,
                "subject": candidate_id,
            },
            "selection": {
                "source_strict": False,
                "has_log_source": False,
                "direct_strict": False,
                "terminal_queue_rank": rank,
            },
            "family": family,
        }

    incompatible = candidate(
        "candidate-memory-on-cpu", "hypothesis-a", "high memory usage", "cpu", 0
    )
    compatible = candidate("candidate-cpu", "hypothesis-b", "high CPU usage", "cpu", 1)
    candidates = [incompatible, compatible]
    families: dict[str, object] = {
        cast(str, item["candidate_id"]): item["family"] for item in candidates
    }

    winner, audit = select_family_compatible_terminal_candidate(
        candidates, incompatible, families, "terminal-candidate-v4"
    )
    assert winner == compatible
    assert audit == {
        "known_mismatch_filter_applied": True,
        "pre_family_policy": "terminal-candidate-v4",
        "base_candidate_id": "candidate-memory-on-cpu",
        "base_hypothesis_id": "hypothesis-a",
        "base_statement": "high memory usage",
        "base_resource_family": "cpu",
        "excluded_candidate_ids": ["candidate-memory-on-cpu"],
    }

    for base in (
        compatible,
        candidate("candidate-unknown", "hypothesis-c", "unknown", "cpu", 0),
        candidate("candidate-none", "hypothesis-d", "high memory usage", None, 0),
        candidate("candidate-other", "hypothesis-e", "high memory usage", "other", 0),
        candidate("candidate-new", "hypothesis-f", "high memory usage", "gpu", 0),
    ):
        winner, audit = select_family_compatible_terminal_candidate(
            [base],
            base,
            {cast(str, base["candidate_id"]): base["family"]},
            "terminal-candidate-v4",
        )
        assert winner == base
        assert audit is None

    winner, audit = select_family_compatible_terminal_candidate(
        [incompatible],
        incompatible,
        {"candidate-memory-on-cpu": "cpu"},
        "terminal-candidate-v4",
    )
    assert winner == incompatible
    assert audit is None

    unknown_survivor = candidate(
        "candidate-unknown-survivor", "hypothesis-g", "future fault", "other", 0
    )
    winner, audit = select_family_compatible_terminal_candidate(
        [incompatible, unknown_survivor],
        incompatible,
        {
            "candidate-memory-on-cpu": "cpu",
            "candidate-unknown-survivor": "other",
        },
        "terminal-candidate-v4",
    )
    assert winner == unknown_survivor
    assert audit is not None
    assert audit["known_mismatch_filter_applied"] is True

    later_stronger = candidate(
        "candidate-later-stronger", "hypothesis-h", "high CPU usage", "cpu", 2
    )
    later_stronger["selection"].update(source_strict=True, direct_strict=True)
    winner, _audit = select_family_compatible_terminal_candidate(
        [incompatible, compatible, later_stronger],
        incompatible,
        {
            "candidate-memory-on-cpu": "cpu",
            "candidate-cpu": "cpu",
            "candidate-later-stronger": "cpu",
        },
        "terminal-candidate-v4",
    )
    assert winner == compatible


def test_weak_rejection_uses_supported_filtered_log_only_without_v4_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    subject = "bank:entity:Tomcat01"
    log_id = seed_tool_evidence(
        store,
        environment,
        "query_log_samples",
        subject=subject,
        paired_log_summary=True,
        filtered_log=True,
        log_name="request_failed",
    )
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    runtime._run_id = "paired-log-fallback"  # pyright: ignore[reportPrivateUsage]
    runtime._decision_id = "decision:paired-log-fallback:0001"  # pyright: ignore[reportPrivateUsage]
    runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    pair_refs = ["evidence-reference", "evidence-incident"]
    for evidence_id, period in zip(pair_refs, ("reference", "incident")):
        observation_id = f"observation-{period}"
        runtime._observations[observation_id] = Observation(  # pyright: ignore[reportPrivateUsage]
            observation_id=observation_id,
            tool="query_metric",
            arguments={
                "entity_id": subject,
                "metric_id": "metric",
                "period": period,
            },
            status="success",
            coverage=Coverage(
                requested_range=environment.task.allowed_query_range,
                observed_range=environment.task.allowed_query_range,
                completeness="complete",
                basis="fixture",
            ),
            artifact_refs=[],
            attempt_id=f"attempt-{period}",
        )
        runtime._evidence[evidence_id] = Evidence(  # pyright: ignore[reportPrivateUsage]
            evidence_id=evidence_id,
            observation_refs=[observation_id],
            subject_ids=[subject],
            predicate="query_metric.result",
            value={"period": period},
            observed_interval=environment.task.allowed_query_range,
            time_uncertainty="fixture",
            artifact_refs=[],
            extractor=Extractor(
                kind="deterministic",
                name="fixture",
                version="1",
                parameters_hash="fixture",
            ),
            source_fingerprint=f"metric-{period}",
            dataset_snapshot_id=environment.dataset_id,
        )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-supported",
        subject=subject,
        statement="high CPU usage",
        status="supported",
        supporting_evidence_refs=[*pair_refs, log_id],
    )
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]
    candidate: dict[str, Any] = {
        "candidate_id": "candidate-supported",
        "candidate_rank": 1,
        "entity_id": subject,
        "metric_id": "metric",
        "fault_type": hypothesis.statement,
        "fault_type_aliases": [],
        "resource_family": "cpu",
        "log_names": ["request_failed"],
        "log_selector_hint": {
            "terms": ["cpu", "load"],
            "exact_log_names": [],
        },
        "has_log_source": True,
        "evidence_ids": {
            "reference": pair_refs[0],
            "incident": pair_refs[1],
        },
    }
    candidates = [candidate]

    def candidate_queue(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return candidates

    def candidate_registry() -> tuple[dict[str, tuple[str, str, str, str]], bool]:
        return {"candidate": (subject, "metric", "cpu", hypothesis.statement)}, True

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    monkeypatch.setattr(runtime, "_overview_candidate_registry", candidate_registry)
    supporting_refs = [*pair_refs, log_id]
    operation = {
        "type": "update_hypothesis",
        "action": "create",
        "hypothesis_id": None,
        "subject": subject,
        "statement": hypothesis.statement,
        "status": "supported",
        "supporting_evidence_refs": supporting_refs,
        "opposing_evidence_refs": [],
        "open_questions": [],
    }
    disposition_key = "event:paired-log:disposition"
    store.append(
        "investigation_event",
        disposition_key,
        {
            "event_type": "candidate_disposition_applied",
            "hypothesis_id": hypothesis.hypothesis_id,
            "candidate_id": candidate["candidate_id"],
            "subject": subject,
            "selected_fault_type": hypothesis.statement,
            "status": "supported",
            "allowed_fault_types": [hypothesis.statement],
            "resource_family": "cpu",
            "log_selector_hint": {
                "terms": ["cpu", "load"],
                "exact_log_names": [],
            },
            "reference_evidence_id": pair_refs[0],
            "incident_evidence_id": pair_refs[1],
            "log_evidence_ids": [log_id],
            "original_operation": operation,
            "applied_operation": operation,
        },
    )
    disposition = cast_dict(store.get(disposition_key).payload)  # type: ignore[union-attr]
    queue_by_id = {
        cast(str, candidate["candidate_id"]): (0, cast(Mapping[str, Any], candidate))
    }
    assert (
        supported_candidate_binding(
            hypothesis,
            [disposition],
            queue_by_id,
            runtime._evidence,  # pyright: ignore[reportPrivateUsage]
            runtime._observations,  # pyright: ignore[reportPrivateUsage]
        )
        == candidate["candidate_id"]
    )
    mismatched = {
        **disposition,
        "reference_evidence_id": pair_refs[1],
        "incident_evidence_id": pair_refs[0],
    }
    assert (
        supported_candidate_binding(
            hypothesis,
            [mismatched],
            queue_by_id,
            runtime._evidence,  # pyright: ignore[reportPrivateUsage]
            runtime._observations,  # pyright: ignore[reportPrivateUsage]
        )
        is None
    )
    runtime._weak_finding_rejection = {  # pyright: ignore[reportPrivateUsage]
        "event_type": "weak_finding_rejected",
        "visibility": "audit_only",
        "policy": "strong-source-no-higher-peer-v1",
        "abstain_reason": (
            "selected Finding did not satisfy the pre-registered strong-Finding policy"
        ),
        "hypothesis_id": hypothesis.hypothesis_id,
        "candidate_id": candidate["candidate_id"],
        "original_finalize_operation": {
            "type": "finalize",
            "hypothesis_ids": [hypothesis.hypothesis_id],
            "path_ids": [],
            "unresolved_questions": [],
            "abstain_reason": None,
        },
    }
    final_request = FinalizeOperation(
        type="finalize",
        abstain_reason=(
            "selected Finding did not satisfy the pre-registered strong-Finding policy"
        ),
    )

    recommendation = runtime._benchmark_recommendation(  # pyright: ignore[reportPrivateUsage]
        "completed_partial", final_request, []
    )

    assert recommendation is not None
    assert recommendation["policy"] == "rejected-supported-filtered-log-v1"
    assert recommendation["hypothesis"] == hypothesis.model_dump(mode="json")
    assert recommendation["candidate_id"] == candidate["candidate_id"]
    assert recommendation["filtered_log_evidence_ids"] == [log_id]
    assert recommendation["disposition_ref"] == disposition_key

    store.append(
        "investigation_event",
        "binding:legacy",
        {
            "event_type": "candidate_disposition_applied",
            "hypothesis_id": hypothesis.hypothesis_id,
            "candidate_id": candidate["candidate_id"],
            "subject": subject,
            "selected_fault_type": hypothesis.statement,
            "status": "supported",
        },
    )
    assert (
        runtime._benchmark_recommendation(  # pyright: ignore[reportPrivateUsage]
            "completed_partial", final_request, []
        )
        is None
    )

    terminal = Hypothesis(
        hypothesis_id="hypothesis-terminal",
        subject="entity-terminal",
        statement="terminal fault",
        status="inconclusive",
    )
    runtime._hypotheses[terminal.hypothesis_id] = terminal  # pyright: ignore[reportPrivateUsage]
    candidates.insert(
        0,
        {
            "candidate_id": "candidate-terminal",
            "entity_id": terminal.subject,
            "fault_type": terminal.statement,
            "has_log_source": False,
        },
    )
    store.append(
        "investigation_event",
        "binding:terminal",
        {
            "event_type": "candidate_disposition_applied",
            "hypothesis_id": terminal.hypothesis_id,
            "candidate_id": "candidate-terminal",
            "subject": terminal.subject,
            "selected_fault_type": terminal.statement,
            "status": terminal.status,
        },
    )
    preferred = runtime._benchmark_recommendation(  # pyright: ignore[reportPrivateUsage]
        "completed_partial", final_request, []
    )
    assert preferred is not None
    assert preferred["policy"] == "terminal-candidate-v4"
    assert preferred["hypothesis"] == terminal.model_dump(mode="json")
    store.close()


def test_observe_only_never_enters_strategy_e_weak_rejection(tmp_path: Path) -> None:
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        RunStore(tmp_path / "run.sqlite3"),
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(validator_mode="observe_only"),
    )

    assert (
        runtime._strategy_e_weak_finding(  # pyright: ignore[reportPrivateUsage]
            FinalizeOperation(type="finalize", hypothesis_ids=["hypothesis"])
        )
        is None
    )
    runtime.store.close()


def test_strategy_e_promotes_strict_candidate_and_rewrites_malformed_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    runtime._run_id = "strategy-e"  # pyright: ignore[reportPrivateUsage]
    runtime._decision_id = "decision:strategy-e:0001"  # pyright: ignore[reportPrivateUsage]
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-selected",
        subject="entity-selected",
        statement="fault",
        status="supported",
        supporting_evidence_refs=["evidence"],
    )
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]
    store.append(
        "investigation_event",
        "binding:selected:open",
        {
            "event_type": "candidate_disposition_applied",
            "hypothesis_id": hypothesis.hypothesis_id,
            "candidate_id": "candidate-selected",
            "subject": hypothesis.subject,
            "selected_fault_type": hypothesis.statement,
            "status": "open",
        },
    )
    store.append(
        "investigation_event",
        "binding:selected:supported",
        {
            "event_type": "candidate_disposition_applied",
            "hypothesis_id": hypothesis.hypothesis_id,
            "candidate_id": "candidate-selected",
            "subject": hypothesis.subject,
            "selected_fault_type": hypothesis.statement,
            "status": "supported",
        },
    )
    selected_candidate: dict[str, Any] = {
        "candidate_id": "candidate-selected",
        "candidate_rank": 0,
        "entity_id": hypothesis.subject,
        "fault_type": hypothesis.statement,
        "fault_type_aliases": [],
        "source_paired_metric_summary": strict_paired_metric_summary(),
        "source_pressure_direction": "higher_is_more_pressure",
    }
    candidates = [selected_candidate]

    def candidate_queue(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return candidates

    def candidate_registry() -> tuple[dict[str, tuple[str, str, str, str]], bool]:
        return {"candidate": ("entity", "metric", "family", "fault")}, True

    def no_finalization_errors(
        _operation: FinalizeOperation,
    ) -> tuple[list[str], list[str]]:
        return [], []

    def no_comparison_errors(_hypothesis_ids: Sequence[str]) -> list[str]:
        return []

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    monkeypatch.setattr(runtime, "_overview_candidate_registry", candidate_registry)
    monkeypatch.setattr(runtime, "_finalization_errors", no_finalization_errors)
    monkeypatch.setattr(runtime, "_comparison_quality_errors", no_comparison_errors)
    operation = FinalizeOperation(
        type="finalize", hypothesis_ids=[hypothesis.hypothesis_id]
    )
    assert runtime._strategy_e_weak_finding(operation) is None  # pyright: ignore[reportPrivateUsage]

    source_summary = selected_candidate.pop("source_paired_metric_summary")
    selected_candidate["paired_metric_summary"] = source_summary
    selected_candidate["candidate_pressure_direction"] = "higher_is_more_pressure"
    assert runtime._strategy_e_weak_finding(operation) is not None  # pyright: ignore[reportPrivateUsage]
    selected_candidate.pop("paired_metric_summary")
    selected_candidate.pop("candidate_pressure_direction")
    selected_candidate["source_paired_metric_summary"] = source_summary

    selected_candidate["candidate_rank"] = 1
    assert runtime._strategy_e_weak_finding(operation) is not None  # pyright: ignore[reportPrivateUsage]
    selected_candidate["candidate_rank"] = 0
    store.append(
        "investigation_event",
        "binding:malformed",
        {
            "event_type": "candidate_disposition_applied",
            "hypothesis_id": hypothesis.hypothesis_id,
            "candidate_id": None,
            "subject": hypothesis.subject,
            "selected_fault_type": hypothesis.statement,
            "status": hypothesis.status,
        },
    )
    assert runtime._strategy_e_weak_finding(operation) is not None  # pyright: ignore[reportPrivateUsage]

    candidates.insert(
        0,
        {
            "candidate_id": "candidate-initial-top1",
            "candidate_rank": 0,
            "entity_id": "entity-top1",
            "fault_type": "other primary",
            "fault_type_aliases": [hypothesis.statement],
            "paired_metric_summary": strict_paired_metric_summary(),
            "candidate_pressure_direction": "higher_is_more_pressure",
        },
    )
    runtime._pending_operation = operation  # pyright: ignore[reportPrivateUsage]
    asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]

    report = cast_dict(store.records("report")[0].payload)
    weak = next(
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "weak_finding_rejected"
    )
    assert report["status"] == "completed_partial"
    assert report["findings"] == []
    assert report["hypothesis_ids"] == []
    assert report["abstain_reason"] == (
        "selected Finding did not satisfy the pre-registered strong-Finding policy"
    )
    assert "benchmark_recommendation" not in report
    assert weak == {
        "event_type": "weak_finding_rejected",
        "visibility": "audit_only",
        "policy": "strong-source-no-higher-peer-v1",
        "abstain_reason": (
            "selected Finding did not satisfy the pre-registered strong-Finding policy"
        ),
        "hypothesis_id": hypothesis.hypothesis_id,
        "candidate_id": "candidate-selected",
        "source_strict": True,
        "candidate_rank": 0,
        "higher_rank_same_fault_candidate_ids": [],
        "initial_top1_candidate_id": "candidate-initial-top1",
        "initial_top1_subject": "entity-top1",
        "initial_top1_fault_type": "other primary",
        "initial_top1_source_strict": False,
        "original_finalize_operation": operation.model_dump(mode="json"),
    }
    assert runtime._validator_retries == 0  # pyright: ignore[reportPrivateUsage]
    store.close()


@pytest.mark.parametrize("source_strict", [True, False])
def test_strategy_e_unique_supported_auto_select_runs_promotion_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_strict: bool
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    runtime._run_id = f"strategy-e-auto-{source_strict}"  # pyright: ignore[reportPrivateUsage]
    runtime._decision_id = (  # pyright: ignore[reportPrivateUsage]
        f"decision:strategy-e-auto-{source_strict}:0001"
    )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-auto",
        subject="entity-auto",
        statement="fault-auto",
        status="supported",
    )
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]
    store.append(
        "investigation_event",
        "binding:auto",
        {
            "event_type": "candidate_disposition_applied",
            "hypothesis_id": hypothesis.hypothesis_id,
            "candidate_id": "candidate-auto",
            "subject": hypothesis.subject,
            "selected_fault_type": hypothesis.statement,
            "status": hypothesis.status,
        },
    )
    candidate: dict[str, Any] = {
        "candidate_id": "candidate-auto",
        "candidate_rank": 0,
        "entity_id": hypothesis.subject,
        "fault_type": hypothesis.statement,
        "fault_type_aliases": [],
        "source_pressure_direction": "higher_is_more_pressure",
    }
    if source_strict:
        candidate["source_paired_metric_summary"] = strict_paired_metric_summary()

    def candidate_queue(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return [candidate]

    def candidate_registry() -> tuple[dict[str, tuple[str, str, str, str]], bool]:
        return {"candidate": ("entity", "metric", "family", "fault")}, True

    def no_finalization_errors(
        _operation: FinalizeOperation,
    ) -> tuple[list[str], list[str]]:
        return [], [] if source_strict else ["retryable quality error"]

    def comparison_errors(_hypothesis_ids: Sequence[str]) -> list[str]:
        return [] if source_strict else ["retryable comparison error"]

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    monkeypatch.setattr(runtime, "_overview_candidate_registry", candidate_registry)
    monkeypatch.setattr(runtime, "_finalization_errors", no_finalization_errors)
    monkeypatch.setattr(runtime, "_comparison_quality_errors", comparison_errors)
    runtime._pending_operation = FinalizeOperation(  # pyright: ignore[reportPrivateUsage]
        type="finalize", abstain_reason="model abstained"
    )

    asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]

    report = cast_dict(store.records("report")[0].payload)
    assert runtime._validator_retries == 0  # pyright: ignore[reportPrivateUsage]
    if source_strict:
        assert report["findings"] == [hypothesis.model_dump(mode="json")]
        assert report["abstain_reason"] is None
        assert "benchmark_recommendation" not in report
        assert any(
            cast_dict(item.payload).get("event_type") == "finalization_auto_selected"
            for item in store.records("investigation_event")
        )
    else:
        assert report["findings"] == []
        assert report["abstain_reason"] == (
            "selected Finding did not satisfy the pre-registered strong-Finding policy"
        )
        assert "benchmark_recommendation" not in report
        assert not any(
            cast_dict(item.payload).get("event_type") == "finalization_auto_selected"
            for item in store.records("investigation_event")
        )
    store.close()


def test_minimum_disposition_coverage_is_family_bounded_latest_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "entity_id": f"entity-{index}",
            "metric_id": f"metric-{index}",
            "resource_family": family,
        }
        for index, family in enumerate(
            ("cpu", "memory", "storage", "network", "cpu", "jvm")
        )
    ]
    queue_calls: list[Sequence[Hypothesis]] = []

    def candidate_queue(
        _observations: object,
        _evidence: object,
        hypotheses: Sequence[Hypothesis] = (),
    ) -> list[dict[str, Any]]:
        queue_calls.append(hypotheses)
        return candidates

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    store = RunStore(tmp_path / "run.sqlite3")

    def append(index: int, status: str, suffix: str, **overrides: object) -> None:
        candidate = candidates[index]
        store.append(
            "investigation_event",
            f"binding:{index}:{suffix}",
            {
                "event_type": "candidate_disposition_applied",
                "candidate_id": candidate["candidate_id"],
                "subject": candidate["entity_id"],
                "metric_id": candidate["metric_id"],
                "resource_family": candidate["resource_family"],
                "status": status,
                **overrides,
            },
        )

    append(0, "open", "open")
    append(0, "supported", "supported")
    append(1, "supported", "supported")
    append(1, "open", "reopened")
    append(2, "inconclusive", "wrong-subject", subject="other")
    append(3, "refuted", "refuted")

    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    coverage = runtime._minimum_candidate_disposition_coverage()  # pyright: ignore[reportPrivateUsage]
    assert coverage == {
        "policy": "finding-min-disposition-v2",
        "required_count": 4,
        "completed_count": 2,
        "target_candidate_ids": [f"candidate-{index}" for index in range(4)],
        "completed_candidate_ids": ["candidate-0", "candidate-3"],
        "remaining_candidate_ids": ["candidate-1", "candidate-2"],
        "remaining_resource_families": ["memory", "storage"],
        "satisfied": False,
    }

    resumed = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store-resumed"),
    )
    assert (
        resumed._minimum_candidate_disposition_coverage()  # pyright: ignore[reportPrivateUsage]
        == coverage
    )
    assert queue_calls and all(not hypotheses for hypotheses in queue_calls)
    store.close()


@pytest.mark.parametrize(
    "exhaust",
    ("logical", "provider", "input", "output", "tool", "deadline", "validator"),
)
def test_minimum_disposition_budget_gate_checks_every_budget(
    tmp_path: Path, exhaust: str
) -> None:
    store = RunStore(tmp_path / f"{exhaust}.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / f"artifact-{exhaust}"),
    )
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    assert runtime._minimum_disposition_budget_available() is True  # pyright: ignore[reportPrivateUsage]
    if exhaust == "logical":
        runtime._model_calls = (  # pyright: ignore[reportPrivateUsage]
            runtime.config.max_model_calls - 4
        )
    elif exhaust == "provider":
        runtime._provider_attempts = (  # pyright: ignore[reportPrivateUsage]
            runtime.config.max_model_calls - 4
        )
    elif exhaust == "input":
        runtime._input_tokens_used = (  # pyright: ignore[reportPrivateUsage]
            runtime.config.max_input_tokens - 4 * runtime.config.max_context_tokens
        )
    elif exhaust == "output":
        runtime._output_tokens_used = (  # pyright: ignore[reportPrivateUsage]
            runtime.config.max_output_tokens
            - 4 * runtime.config.max_output_tokens_per_call
        )
    elif exhaust == "tool":
        runtime._tool_attempts = (  # pyright: ignore[reportPrivateUsage]
            runtime.config.max_tool_attempts
        )
    elif exhaust == "deadline":
        runtime._deadline_at = 1.0  # pyright: ignore[reportPrivateUsage]
    else:
        runtime._validator_retries = (  # pyright: ignore[reportPrivateUsage]
            runtime.config.max_validator_retries
        )
    assert runtime._minimum_disposition_budget_available() is False  # pyright: ignore[reportPrivateUsage]
    store.close()


def test_weak_finalize_rejects_minimum_disposition_once_with_actionable_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "entity_id": f"entity-{index}",
            "metric_id": f"metric-{index}",
            "resource_family": f"family-{index}",
        }
        for index in range(4)
    ]

    def candidate_queue(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return candidates

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    store = RunStore(tmp_path / "run.sqlite3")
    for index, status in ((0, "supported"), (1, "refuted")):
        candidate = candidates[index]
        store.append(
            "investigation_event",
            f"binding:{index}",
            {
                "event_type": "candidate_disposition_applied",
                "candidate_id": candidate["candidate_id"],
                "subject": candidate["entity_id"],
                "metric_id": candidate["metric_id"],
                "resource_family": candidate["resource_family"],
                "status": status,
            },
        )
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(max_validator_retries=1),
    )
    runtime._run_id = "minimum-authority"  # pyright: ignore[reportPrivateUsage]
    runtime._decision_id = "decision:minimum-authority:0001"  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    runtime._model_calls = 1  # pyright: ignore[reportPrivateUsage]
    runtime._provider_attempts = 1  # pyright: ignore[reportPrivateUsage]
    selected = Hypothesis(
        hypothesis_id="hypothesis-selected",
        subject="entity-0",
        statement="fault-0",
        status="supported",
    )
    runtime._hypotheses[selected.hypothesis_id] = selected  # pyright: ignore[reportPrivateUsage]
    strong = selected.model_copy(
        update={"hypothesis_id": "hypothesis-strong", "subject": "entity-strong"}
    )
    runtime._hypotheses[strong.hypothesis_id] = strong  # pyright: ignore[reportPrivateUsage]
    operation = FinalizeOperation(
        type="finalize", hypothesis_ids=[selected.hypothesis_id]
    )
    runtime._pending_operation = operation  # pyright: ignore[reportPrivateUsage]
    weak = {
        "event_type": "weak_finding_rejected",
        "visibility": "audit_only",
        "policy": "strong-source-no-higher-peer-v1",
        "abstain_reason": "weak fixture",
        "hypothesis_id": selected.hypothesis_id,
        "candidate_id": "candidate-0",
        "original_finalize_operation": operation.model_dump(mode="json"),
    }

    def weak_finding(
        candidate_operation: FinalizeOperation,
    ) -> dict[str, Any] | None:
        return (
            None
            if candidate_operation.hypothesis_ids == [strong.hypothesis_id]
            else weak
        )

    def no_finalization_errors(
        _operation: FinalizeOperation,
    ) -> tuple[list[str], list[str]]:
        return [], []

    def no_comparison_errors(_ids: Sequence[str]) -> list[str]:
        return []

    monkeypatch.setattr(runtime, "_strategy_e_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_strategy_e_weak_finding", weak_finding)
    monkeypatch.setattr(runtime, "_finalization_errors", no_finalization_errors)
    monkeypatch.setattr(runtime, "_comparison_quality_errors", no_comparison_errors)

    assert runtime._minimum_candidate_disposition_required() is True  # pyright: ignore[reportPrivateUsage]
    runtime._validator_retries = 1  # pyright: ignore[reportPrivateUsage]
    runtime._record_budget(  # pyright: ignore[reportPrivateUsage]
        "validator_retry_started",
        call_number=runtime._model_calls,  # pyright: ignore[reportPrivateUsage]
        decision_id=runtime._decision_id,  # pyright: ignore[reportPrivateUsage]
    )
    assert runtime._minimum_disposition_rejection() is None  # pyright: ignore[reportPrivateUsage]
    state = asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert state.get("route") == "build_context"
    assert runtime._validator_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert runtime._weak_finding_rejection is None  # pyright: ignore[reportPrivateUsage]
    rejected = [
        cast_dict(record.payload)
        for record in store.records("investigation_event")
        if cast_dict(record.payload).get("event_type") == "finalize_rejected"
    ]
    assert len(rejected) == 1
    feedback = rejected[0]["feedback"][0]
    assert feedback["policy"] == "finding-min-disposition-v2"
    assert feedback["required_count"] == 4
    assert feedback["completed_count"] == 2
    assert feedback["remaining_candidate_ids"] == ["candidate-2", "candidate-3"]
    assert feedback["remaining_resource_families"] == ["family-2", "family-3"]
    assert "update_hypothesis" in feedback["required_correction"]
    assert runtime._minimum_candidate_disposition_feedback() == feedback  # pyright: ignore[reportPrivateUsage]
    assert runtime._validator_retries == 1  # pyright: ignore[reportPrivateUsage]
    replay = asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert replay.get("route") == "build_context"
    assert runtime._validator_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert runtime._report is None  # pyright: ignore[reportPrivateUsage]
    assert runtime._feedback == [feedback]  # pyright: ignore[reportPrivateUsage]
    assert (
        len(
            [
                record
                for record in store.records("investigation_event")
                if cast_dict(record.payload).get("policy")
                == "finding-min-disposition-v2"
            ]
        )
        == 1
    )

    runtime._decision_id = "decision:minimum-authority:0002"  # pyright: ignore[reportPrivateUsage]
    runtime._model_calls = 2  # pyright: ignore[reportPrivateUsage]
    runtime._provider_attempts = 2  # pyright: ignore[reportPrivateUsage]
    asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert runtime._validator_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert runtime._report is not None  # pyright: ignore[reportPrivateUsage]
    assert runtime._report["abstain_reason"] == (  # pyright: ignore[reportPrivateUsage]
        "minimum candidate disposition coverage remained incomplete after one "
        "correction"
    )
    assert "benchmark_recommendation" not in runtime._report  # pyright: ignore[reportPrivateUsage]

    no_retry_store = RunStore(tmp_path / "no-retry.sqlite3")
    no_retry = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        no_retry_store,
        ArtifactStore(tmp_path / "no-retry-artifacts"),
        RuntimeConfig(max_validator_retries=0),
    )
    no_retry._run_id = "minimum-no-retry"  # pyright: ignore[reportPrivateUsage]
    no_retry._decision_id = "decision:minimum-no-retry:0001"  # pyright: ignore[reportPrivateUsage]
    no_retry._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    no_retry._model_calls = 1  # pyright: ignore[reportPrivateUsage]
    no_retry._provider_attempts = 1  # pyright: ignore[reportPrivateUsage]
    no_retry._hypotheses[selected.hypothesis_id] = selected  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(no_retry, "_strategy_e_enabled", lambda: True)
    monkeypatch.setattr(no_retry, "_strategy_e_weak_finding", weak_finding)
    assert no_retry._current_validator_retry_started() is False  # pyright: ignore[reportPrivateUsage]
    assert no_retry._minimum_candidate_disposition_feedback() is None  # pyright: ignore[reportPrivateUsage]
    no_retry_store.close()
    store.close()


def test_minimum_disposition_completion_closes_weak_but_never_blocks_strong(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "entity_id": f"entity-{index}",
            "metric_id": f"metric-{index}",
            "resource_family": f"family-{index}",
        }
        for index in range(4)
    ]

    def candidate_queue(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return candidates

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    runtime._run_id = "minimum-complete"  # pyright: ignore[reportPrivateUsage]
    runtime._decision_id = "decision:minimum-complete:0001"  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    selected = Hypothesis(
        hypothesis_id="hypothesis-selected",
        subject="entity-0",
        statement="fault-0",
        status="supported",
    )
    runtime._hypotheses[selected.hypothesis_id] = selected  # pyright: ignore[reportPrivateUsage]
    operation = FinalizeOperation(
        type="finalize", hypothesis_ids=[selected.hypothesis_id]
    )
    runtime._pending_operation = operation  # pyright: ignore[reportPrivateUsage]

    def no_finalization_errors(
        _operation: FinalizeOperation,
    ) -> tuple[list[str], list[str]]:
        return [], []

    def no_comparison_errors(_ids: Sequence[str]) -> list[str]:
        return []

    monkeypatch.setattr(runtime, "_strategy_e_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_finalization_errors", no_finalization_errors)
    monkeypatch.setattr(runtime, "_comparison_quality_errors", no_comparison_errors)
    weak = {
        "event_type": "weak_finding_rejected",
        "visibility": "audit_only",
        "policy": "strong-source-no-higher-peer-v1",
        "abstain_reason": "weak fixture",
        "hypothesis_id": selected.hypothesis_id,
        "candidate_id": "candidate-0",
        "original_finalize_operation": operation.model_dump(mode="json"),
    }

    def weak_finding(_operation: FinalizeOperation) -> dict[str, Any]:
        return weak

    monkeypatch.setattr(runtime, "_strategy_e_weak_finding", weak_finding)
    assert runtime._minimum_candidate_disposition_required() is True  # pyright: ignore[reportPrivateUsage]
    for index, candidate in enumerate(candidates):
        store.append(
            "investigation_event",
            f"binding:{index}",
            {
                "event_type": "candidate_disposition_applied",
                "candidate_id": candidate["candidate_id"],
                "subject": candidate["entity_id"],
                "metric_id": candidate["metric_id"],
                "resource_family": candidate["resource_family"],
                "status": "supported" if index == 0 else "inconclusive",
            },
        )
    assert runtime._minimum_candidate_disposition_required() is False  # pyright: ignore[reportPrivateUsage]
    asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert runtime._validator_retries == 0  # pyright: ignore[reportPrivateUsage]
    assert runtime._report is not None  # pyright: ignore[reportPrivateUsage]
    assert runtime._report["findings"] == []  # pyright: ignore[reportPrivateUsage]

    strong_store = RunStore(tmp_path / "strong.sqlite3")
    strong = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        strong_store,
        ArtifactStore(tmp_path / "strong-artifact-store"),
    )
    strong._run_id = "minimum-strong"  # pyright: ignore[reportPrivateUsage]
    strong._decision_id = "decision:minimum-strong:0001"  # pyright: ignore[reportPrivateUsage]
    strong._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    strong._hypotheses[selected.hypothesis_id] = selected  # pyright: ignore[reportPrivateUsage]
    strong._pending_operation = operation  # pyright: ignore[reportPrivateUsage]

    def strong_finding(_operation: FinalizeOperation) -> None:
        return None

    monkeypatch.setattr(strong, "_strategy_e_enabled", lambda: True)
    monkeypatch.setattr(strong, "_strategy_e_weak_finding", strong_finding)
    monkeypatch.setattr(strong, "_finalization_errors", no_finalization_errors)
    monkeypatch.setattr(strong, "_comparison_quality_errors", no_comparison_errors)
    assert strong._minimum_candidate_disposition_required() is True  # pyright: ignore[reportPrivateUsage]
    first_strong = asyncio.run(  # pyright: ignore[reportPrivateUsage]
        strong._finalize_untraced(cast(Any, {}))  # pyright: ignore[reportPrivateUsage]
    )
    assert first_strong.get("route") == "build_context"
    assert strong._validator_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert strong._report is None  # pyright: ignore[reportPrivateUsage]
    assert [
        cast_dict(record.payload).get("policy")
        for record in strong_store.records("investigation_event")
        if cast_dict(record.payload).get("event_type") == "finalize_rejected"
    ] == ["finding-min-disposition-v2"]

    strong._decision_id = "decision:minimum-strong:0002"  # pyright: ignore[reportPrivateUsage]
    strong._model_calls = 1  # pyright: ignore[reportPrivateUsage]
    strong._provider_attempts = 1  # pyright: ignore[reportPrivateUsage]
    asyncio.run(strong._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert strong._validator_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert strong._report is not None  # pyright: ignore[reportPrivateUsage]
    assert strong._report["findings"] == [  # pyright: ignore[reportPrivateUsage]
        selected.model_dump(mode="json")
    ]
    assert "benchmark_recommendation" not in strong._report  # pyright: ignore[reportPrivateUsage]

    closeout_store = RunStore(tmp_path / "provider-closeout.sqlite3")
    closeout = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        closeout_store,
        ArtifactStore(tmp_path / "provider-closeout-artifacts"),
    )
    closeout._run_id = "minimum-provider-closeout"  # pyright: ignore[reportPrivateUsage]
    closeout._decision_id = "decision:minimum-provider-closeout:0001"  # pyright: ignore[reportPrivateUsage]
    closeout._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    closeout._provider_attempts = (  # pyright: ignore[reportPrivateUsage]
        closeout.config.max_model_calls - 4
    )
    closeout._hypotheses[selected.hypothesis_id] = selected  # pyright: ignore[reportPrivateUsage]
    closeout._pending_operation = operation  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(closeout, "_strategy_e_enabled", lambda: True)
    monkeypatch.setattr(closeout, "_strategy_e_weak_finding", weak_finding)
    monkeypatch.setattr(closeout, "_finalization_errors", no_finalization_errors)
    monkeypatch.setattr(closeout, "_comparison_quality_errors", no_comparison_errors)
    asyncio.run(closeout._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert closeout._validator_retries == 0  # pyright: ignore[reportPrivateUsage]
    assert closeout._report is not None  # pyright: ignore[reportPrivateUsage]
    assert closeout._report["status"] == "completed_partial"  # pyright: ignore[reportPrivateUsage]
    assert not any(
        cast_dict(record.payload).get("policy") == "finding-min-disposition-v2"
        for record in closeout_store.records("investigation_event")
    )
    store.close()
    strong_store.close()
    closeout_store.close()


@pytest.mark.parametrize("strength", ["strong", "weak"])
@pytest.mark.parametrize(
    "boundary", ["logical", "provider", "input", "output", "tool", "coverage"]
)
def test_minimum_disposition_v2_closeout_and_complete_coverage_do_not_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    strength: str,
    boundary: str,
) -> None:
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "entity_id": f"entity-{index}",
            "metric_id": f"metric-{index}",
            "resource_family": f"family-{index}",
        }
        for index in range(4)
    ]

    def candidate_queue(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return candidates

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    store = RunStore(tmp_path / f"{boundary}-{strength}.sqlite3")
    if boundary == "coverage":
        for index, candidate in enumerate(candidates):
            store.append(
                "investigation_event",
                f"binding:{index}",
                {
                    "event_type": "candidate_disposition_applied",
                    "candidate_id": candidate["candidate_id"],
                    "subject": candidate["entity_id"],
                    "metric_id": candidate["metric_id"],
                    "resource_family": candidate["resource_family"],
                    "status": "supported" if index == 0 else "inconclusive",
                },
            )
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / f"artifacts-{boundary}-{strength}"),
    )
    runtime._run_id = f"minimum-{boundary}-{strength}"  # pyright: ignore[reportPrivateUsage]
    runtime._decision_id = f"decision:minimum-{boundary}-{strength}:0001"  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    selected = Hypothesis(
        hypothesis_id="hypothesis-selected",
        subject="entity-0",
        statement="fault-0",
        status="supported",
    )
    runtime._hypotheses[selected.hypothesis_id] = selected  # pyright: ignore[reportPrivateUsage]
    operation = FinalizeOperation(
        type="finalize", hypothesis_ids=[selected.hypothesis_id]
    )
    runtime._pending_operation = operation  # pyright: ignore[reportPrivateUsage]

    def finding(candidate_operation: FinalizeOperation) -> dict[str, Any] | None:
        if strength == "strong":
            return None
        return {
            "event_type": "weak_finding_rejected",
            "visibility": "audit_only",
            "policy": "strong-source-no-higher-peer-v1",
            "abstain_reason": "weak fixture",
            "hypothesis_id": selected.hypothesis_id,
            "candidate_id": "candidate-0",
            "original_finalize_operation": candidate_operation.model_dump(mode="json"),
        }

    def no_finalization_errors(
        _operation: FinalizeOperation,
    ) -> tuple[list[str], list[str]]:
        return [], []

    def no_comparison_errors(_ids: Sequence[str]) -> list[str]:
        return []

    monkeypatch.setattr(runtime, "_strategy_e_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_strategy_e_weak_finding", finding)
    monkeypatch.setattr(runtime, "_finalization_errors", no_finalization_errors)
    monkeypatch.setattr(runtime, "_comparison_quality_errors", no_comparison_errors)
    if boundary == "logical":
        runtime._model_calls = runtime.config.max_model_calls - 4  # pyright: ignore[reportPrivateUsage]
    elif boundary == "provider":
        runtime._provider_attempts = runtime.config.max_model_calls - 4  # pyright: ignore[reportPrivateUsage]
    elif boundary == "input":
        runtime._input_tokens_used = (  # pyright: ignore[reportPrivateUsage]
            runtime.config.max_input_tokens - 4 * runtime.config.max_context_tokens
        )
    elif boundary == "output":
        runtime._output_tokens_used = (  # pyright: ignore[reportPrivateUsage]
            runtime.config.max_output_tokens
            - 4 * runtime.config.max_output_tokens_per_call
        )
    elif boundary == "tool":
        runtime._tool_attempts = runtime.config.max_tool_attempts  # pyright: ignore[reportPrivateUsage]

    asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert runtime._validator_retries == 0  # pyright: ignore[reportPrivateUsage]
    assert runtime._report is not None  # pyright: ignore[reportPrivateUsage]
    assert not any(
        cast_dict(record.payload).get("policy") == "finding-min-disposition-v2"
        for record in store.records("investigation_event")
    )
    if strength == "strong":
        assert runtime._report["findings"] == [  # pyright: ignore[reportPrivateUsage]
            selected.model_dump(mode="json")
        ]
    else:
        assert runtime._report["findings"] == []  # pyright: ignore[reportPrivateUsage]
        assert runtime._report["abstain_reason"] == (  # pyright: ignore[reportPrivateUsage]
            "selected Finding did not satisfy the pre-registered strong-Finding policy"
        )
    store.close()


def test_minimum_disposition_v2_preserves_v1_same_decision_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "entity_id": f"entity-{index}",
            "metric_id": f"metric-{index}",
            "resource_family": f"family-{index}",
        }
        for index in range(4)
    ]

    def candidate_queue(*_args: object, **_kwargs: object) -> list[dict[str, Any]]:
        return candidates

    monkeypatch.setattr("rootwalk.runtime.candidate_disposition_queue", candidate_queue)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(max_validator_retries=1),
    )
    runtime._run_id = "minimum-v1-replay"  # pyright: ignore[reportPrivateUsage]
    runtime._decision_id = "decision:minimum-v1-replay:0001"  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    runtime._model_calls = 1  # pyright: ignore[reportPrivateUsage]
    runtime._provider_attempts = 1  # pyright: ignore[reportPrivateUsage]
    selected = Hypothesis(
        hypothesis_id="hypothesis-selected",
        subject="entity-0",
        statement="fault-0",
        status="supported",
    )
    runtime._hypotheses[selected.hypothesis_id] = selected  # pyright: ignore[reportPrivateUsage]
    operation = FinalizeOperation(
        type="finalize", hypothesis_ids=[selected.hypothesis_id]
    )
    runtime._pending_operation = operation  # pyright: ignore[reportPrivateUsage]

    def strong_finding(_operation: FinalizeOperation) -> None:
        return None

    def no_finalization_errors(
        _operation: FinalizeOperation,
    ) -> tuple[list[str], list[str]]:
        return [], []

    def no_comparison_errors(_ids: Sequence[str]) -> list[str]:
        return []

    monkeypatch.setattr(runtime, "_strategy_e_enabled", lambda: True)
    monkeypatch.setattr(runtime, "_strategy_e_weak_finding", strong_finding)
    monkeypatch.setattr(runtime, "_finalization_errors", no_finalization_errors)
    monkeypatch.setattr(runtime, "_comparison_quality_errors", no_comparison_errors)
    feedback = {
        "feedback_id": f"minimum_disposition:{runtime._decision_id}",  # pyright: ignore[reportPrivateUsage]
        "type": "exit_gate",
        "category": "quality",
        "policy": "weak-finding-min-disposition-v1",
        "required_count": 4,
        "completed_count": 0,
        "remaining_candidate_ids": [f"candidate-{index}" for index in range(4)],
        "remaining_resource_families": [f"family-{index}" for index in range(4)],
        "required_correction": "legacy correction",
    }
    stable_key = runtime._event_key("finalize_rejected")  # pyright: ignore[reportPrivateUsage]
    store.append(
        "investigation_event",
        stable_key,
        {
            "event_type": "finalize_rejected",
            "decision_id": runtime._decision_id,  # pyright: ignore[reportPrivateUsage]
            "policy": "weak-finding-min-disposition-v1",
            "errors": ["legacy correction"],
            "feedback": [feedback],
        },
    )
    runtime._validator_retries = 1  # pyright: ignore[reportPrivateUsage]
    runtime._record_budget(  # pyright: ignore[reportPrivateUsage]
        "validator_retry_started",
        call_number=runtime._model_calls,  # pyright: ignore[reportPrivateUsage]
        decision_id=runtime._decision_id,  # pyright: ignore[reportPrivateUsage]
    )
    original = cast_dict(store.get(stable_key).payload)  # type: ignore[union-attr]

    replay = asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert replay.get("route") == "build_context"
    assert runtime._validator_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert runtime._report is None  # pyright: ignore[reportPrivateUsage]
    assert cast_dict(store.get(stable_key).payload) == original  # type: ignore[union-attr]
    assert runtime._feedback == [feedback]  # pyright: ignore[reportPrivateUsage]

    runtime._decision_id = "decision:minimum-v1-replay:0002"  # pyright: ignore[reportPrivateUsage]
    runtime._model_calls = 2  # pyright: ignore[reportPrivateUsage]
    runtime._provider_attempts = 2  # pyright: ignore[reportPrivateUsage]
    asyncio.run(runtime._finalize_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert runtime._validator_retries == 1  # pyright: ignore[reportPrivateUsage]
    assert runtime._report is not None  # pyright: ignore[reportPrivateUsage]
    assert runtime._report["findings"] == [  # pyright: ignore[reportPrivateUsage]
        selected.model_dump(mode="json")
    ]
    assert cast_dict(store.get(stable_key).payload) == original  # type: ignore[union-attr]
    store.close()


def test_minimum_disposition_v2_only_selects_known_supported_findings(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    supported = Hypothesis(
        hypothesis_id="hypothesis-supported",
        subject="entity-supported",
        statement="fault",
        status="supported",
    )
    refuted = supported.model_copy(
        update={"hypothesis_id": "hypothesis-refuted", "status": "refuted"}
    )
    runtime._hypotheses = {  # pyright: ignore[reportPrivateUsage]
        supported.hypothesis_id: supported,
        refuted.hypothesis_id: refuted,
    }
    selected = FinalizeOperation(
        type="finalize", hypothesis_ids=[supported.hypothesis_id]
    )
    mixed = FinalizeOperation(
        type="finalize",
        hypothesis_ids=[supported.hypothesis_id, refuted.hypothesis_id],
    )
    terminal = FinalizeOperation(
        type="finalize", hypothesis_ids=[refuted.hypothesis_id]
    )
    unknown = FinalizeOperation(type="finalize", hypothesis_ids=["unknown"])
    abstain = FinalizeOperation(type="finalize", abstain_reason="fixture")

    assert runtime._operation_selects_supported_findings(selected) is True  # pyright: ignore[reportPrivateUsage]
    for operation in (mixed, terminal, unknown, abstain):
        assert runtime._operation_selects_supported_findings(operation) is False  # pyright: ignore[reportPrivateUsage]
    reference_errors, _quality_errors = runtime._finalization_errors(unknown)  # pyright: ignore[reportPrivateUsage]
    assert reference_errors == ["unknown hypotheses: ['unknown']"]
    reference_errors, quality_errors = runtime._finalization_errors(terminal)  # pyright: ignore[reportPrivateUsage]
    assert reference_errors == []
    assert any("only supported hypotheses" in error for error in quality_errors)
    store.close()


def test_build_context_wires_minimum_disposition_requirement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    runtime._run_id = "minimum-context-wiring"  # pyright: ignore[reportPrivateUsage]
    runtime._started_at = 1.0  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    captured: dict[str, Any] = {}

    def required() -> bool:
        return True

    def capturing_compile_context(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_compile_context(**kwargs)

    monkeypatch.setattr(runtime, "_minimum_candidate_disposition_required", required)
    monkeypatch.setattr("rootwalk.runtime.compile_context", capturing_compile_context)

    state = asyncio.run(runtime._build_context_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]
    assert state.get("route") == "call_model"
    assert captured["candidate_disposition_required"] is True
    store.close()


def test_frontier_context_reserves_last_model_timeout_for_closeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_000.0
    monkeypatch.setattr("rootwalk.runtime.time.time", lambda: now)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(
            context_strategy="frontier_reflection",
            model_timeout_seconds=300.0,
            wall_clock_seconds=900,
        ),
    )
    runtime._run_id = "frontier-wall-closeout"  # pyright: ignore[reportPrivateUsage]
    runtime._started_at = 400.0  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 1_329.0  # pyright: ignore[reportPrivateUsage]
    runtime._hypotheses = {  # pyright: ignore[reportPrivateUsage]
        "hypothesis-1": Hypothesis(
            hypothesis_id="hypothesis-1",
            subject="bank:entity:Tomcat01",
            statement="typed CPU pressure",
            status="open",
            supporting_evidence_refs=["evidence-1"],
        )
    }
    captured: dict[str, Any] = {}

    def capturing_compile_context(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_compile_context(**kwargs)

    monkeypatch.setattr("rootwalk.runtime.compile_context", capturing_compile_context)

    state = asyncio.run(runtime._build_context_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]

    assert state.get("route") == "call_model"
    budget = cast_dict(captured["budget"])
    assert budget["deadline_seconds_remaining"] == 329
    assert budget["closeout_required"] is True
    assert budget["closeout_reason"] == "wall_clock_reserve"
    store.close()


def test_frontier_context_reserves_closeout_for_unresolved_registry_without_hypothesis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_000.0
    monkeypatch.setattr("rootwalk.runtime.time.time", lambda: now)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(
            context_strategy="frontier_reflection",
            model_timeout_seconds=300.0,
            wall_clock_seconds=900,
        ),
    )
    runtime._run_id = "frontier-registry-closeout"  # pyright: ignore[reportPrivateUsage]
    runtime._started_at = 400.0  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 1_329.0  # pyright: ignore[reportPrivateUsage]
    anomaly = material_anomaly()
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]
    captured: dict[str, Any] = {}

    def capturing_compile_context(**kwargs: Any) -> Any:
        captured.update(kwargs)
        return real_compile_context(**kwargs)

    monkeypatch.setattr("rootwalk.runtime.compile_context", capturing_compile_context)

    state = asyncio.run(runtime._build_context_untraced(cast(Any, {})))  # pyright: ignore[reportPrivateUsage]

    assert state.get("route") == "call_model"
    budget = cast_dict(captured["budget"])
    assert budget["closeout_required"] is True
    assert budget["closeout_reason"] == "wall_clock_reserve"
    store.close()


def test_frontier_hard_closeout_allows_pure_abstention_without_reflection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_000.0
    monkeypatch.setattr("rootwalk.runtime.time.time", lambda: now)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        _BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(
            context_strategy="frontier_reflection",
            model_timeout_seconds=300.0,
            wall_clock_seconds=900,
        ),
    )
    runtime._deadline_at = 1_100.0  # pyright: ignore[reportPrivateUsage]
    item = Observation(
        observation_id="unreflected-observation",
        tool="query_trace_edges",
        arguments={"entity_id": "bank:entity:Tomcat01", "direction": "incoming"},
        status="success",
        coverage=Coverage(
            requested_range=runtime.environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="unreflected-attempt",
    )
    runtime._observations[item.observation_id] = item  # pyright: ignore[reportPrivateUsage]
    decision = Decision.model_validate(
        {
            "decision_summary": "hard closeout preserves the unresolved evidence",
            "operations": [
                {
                    "type": "finalize",
                    "hypothesis_ids": [],
                    "path_ids": [],
                    "unresolved_questions": ["the trace remains unreflected"],
                    "abstain_reason": "physical closeout prevents another round",
                }
            ],
        }
    )

    assert (
        runtime._validate_frontier_decision(decision)  # pyright: ignore[reportPrivateUsage]
        is None
    )
    runtime._deadline_at = 2_000.0  # pyright: ignore[reportPrivateUsage]
    error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        decision
    )
    assert error is not None
    assert "requires Decision.reflection" in error
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_candidate_disposition_rejects_other_public_fault_without_audit_event(
    tmp_path: Path, validator_mode: str
) -> None:
    run_id = f"candidate-illegal-alias-{validator_mode}"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(
        store, environment, key_limit=1, has_log_source=False
    )[0]
    model = CapturingModel(
        [
            {
                "decision_summary": "query candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "name another public fault",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": candidate["entity_id"],
                        "statement": "high memory usage",
                        "status": "inconclusive",
                    }
                ],
            },
            {
                "decision_summary": "use the candidate default",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": candidate["entity_id"],
                        "statement": "unknown",
                        "status": "inconclusive",
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    events = [
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "candidate_disposition_applied"
    ]
    assert result.status == "completed_partial"
    assert len(events) == 1
    assert events[0]["original_operation"]["statement"] == "unknown"
    assert len(store.records("hypothesis")) == 1
    assert "outside the current pair" in str(model.inputs[2])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_candidate_disposition_revises_open_and_attaches_filtered_log(
    tmp_path: Path, validator_mode: str
) -> None:
    run_id = f"candidate-revise-{validator_mode}"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    alias = "CPU saturation"
    candidate = seed_candidate_overview(
        store,
        environment,
        key_limit=1,
        has_log_source=True,
        first_alias=alias,
    )[0]
    log_evidence_id = seed_tool_evidence(
        store,
        environment,
        "query_log_samples",
        subject=cast(str, candidate["entity_id"]),
        paired_log_summary=True,
        filtered_log=True,
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "query candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "open candidate",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": "wrong",
                        "statement": "unknown",
                        "status": "open",
                    }
                ],
            },
            {
                "decision_summary": "refine to allowed alias",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": "wrong",
                        "statement": alias,
                        "status": "supported",
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    hypotheses = [cast_dict(item.payload) for item in store.records("hypothesis")]
    events = [
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "candidate_disposition_applied"
    ]
    assert result.status == "completed_partial"
    assert len(hypotheses) == 2
    assert hypotheses[0]["hypothesis_id"] == hypotheses[1]["hypothesis_id"]
    assert hypotheses[1]["statement"] == alias
    assert log_evidence_id in hypotheses[1]["supporting_evidence_refs"]
    assert events[1]["applied_operation"]["action"] == "revise"
    assert (
        events[1]["applied_operation"]["hypothesis_id"]
        == hypotheses[0]["hypothesis_id"]
    )
    assert len(store.records("tool_effect")) == 2
    assert result.report["budget"]["validator_retries_used"] == 0
    store.close()


def test_candidate_disposition_fails_closed_for_multiple_matching_open_hypotheses(
    tmp_path: Path,
) -> None:
    run_id = "candidate-ambiguous-open"
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(
        store, environment, key_limit=1, has_log_source=False
    )[0]
    query_runtime = RootWalkRuntime(
        environment,
        RecordedModelAdapter(
            [
                {
                    "decision_summary": "query candidate",
                    "operations": [
                        {
                            "type": "query",
                            "tool": "query_candidate",
                            "arguments": {"candidate_id": candidate["candidate_id"]},
                        }
                    ],
                }
            ]
        ),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    query_runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    asyncio.run(interrupt_runtime_after(query_runtime, run_id, "execute_query"))
    pair_refs = [
        f"{run_id}:evidence:decision:{run_id}:0001:{index}" for index in range(2)
    ]
    for index in range(2):
        hypothesis = Hypothesis(
            hypothesis_id=f"open-{index}",
            subject=cast(str, candidate["entity_id"]),
            statement=cast(str, candidate["fault_type"]),
            status="open",
            supporting_evidence_refs=pair_refs,
        )
        store.append("hypothesis", f"seed:hypothesis:{index}", hypothesis)

    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    runtime._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    operation = Decision.model_validate(
        {
            "decision_summary": "close candidate",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": candidate["entity_id"],
                    "statement": candidate["fault_type"],
                    "status": "inconclusive",
                }
            ],
        }
    ).operations[0]

    _bound, error = runtime._bind_candidate_disposition(  # pyright: ignore[reportPrivateUsage]
        cast(Any, operation)
    )

    assert error == "candidate disposition matches multiple open hypotheses"
    assert not any(
        cast_dict(item.payload).get("event_type") == "candidate_disposition_applied"
        for item in store.records("investigation_event")
    )
    store.close()


def test_candidate_disposition_resume_commits_one_atomic_audit_event(
    tmp_path: Path,
) -> None:
    run_id = "candidate-binding-resume"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    environment = _BankEnvironment(PUBLIC_INPUT)
    first_store = RunStore(database)
    candidate = seed_candidate_overview(
        first_store, environment, key_limit=1, has_log_source=False
    )[0]
    malformed = {
        "type": "update_hypothesis",
        "action": "revise",
        "hypothesis_id": "wrong-id",
        "subject": "wrong-subject",
        "statement": "unknown",
        "status": "inconclusive",
        "supporting_evidence_refs": ["wrong-ref"],
    }
    first_runtime = RootWalkRuntime(
        environment,
        RecordedModelAdapter(
            [
                {
                    "decision_summary": "query candidate",
                    "operations": [
                        {
                            "type": "query",
                            "tool": "query_candidate",
                            "arguments": {"candidate_id": candidate["candidate_id"]},
                        }
                    ],
                },
                {"decision_summary": "dispose candidate", "operations": [malformed]},
            ]
        ),
        first_store,
        ArtifactStore(artifact_root),
    )
    first_runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "execute_query"))
    asyncio.run(resume_runtime_until(first_runtime, run_id, "route"))
    assert not first_store.records("hypothesis")
    assert not any(
        cast_dict(item.payload).get("event_type") == "candidate_disposition_applied"
        for item in first_store.records("investigation_event")
    )
    first_store.close()

    resumed_store = RunStore(database)
    result = asyncio.run(
        RootWalkRuntime(
            _BankEnvironment(PUBLIC_INPUT),
            RecordedModelAdapter(
                [
                    {
                        "decision_summary": "stop",
                        "operations": [
                            {"type": "finalize", "abstain_reason": "fixture"}
                        ],
                    }
                ]
            ),
            resumed_store,
            ArtifactStore(artifact_root),
        ).run(run_id, tmp_path / "bundle")
    )

    events = [
        cast_dict(item.payload)
        for item in resumed_store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "candidate_disposition_applied"
    ]
    assert result.status == "completed_partial"
    assert len(resumed_store.records("hypothesis")) == 1
    assert len(events) == 1
    assert events[0]["original_operation"]["hypothesis_id"] == "wrong-id"
    assert events[0]["applied_operation"]["action"] == "create"
    assert len(resumed_store.records("tool_effect")) == 2
    resumed_store.close()


def test_query_candidate_rejects_whole_batch_when_child_budget_is_insufficient(
    tmp_path: Path,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment)[0]
    model = CapturingModel(
        [
            {
                "decision_summary": "select one candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(max_model_calls=1, max_tool_attempts=1),
        ).run("candidate-budget", tmp_path / "bundle")
    )

    assert result.status == "budget_exhausted"
    assert len(store.records("observation")) == 1
    assert not store.records("tool_effect")
    assert result.report["budget"]["tool_attempts_used"] == 0
    store.close()


def test_query_candidate_budget_counts_only_novel_children(tmp_path: Path) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment, key_limit=1)[0]
    reference = environment.query_metric(
        cast(str, candidate["entity_id"]),
        cast(str, candidate["metric_id"]),
        period="reference",
    )
    observation = Observation.model_validate_json(
        json.dumps(
            {
                "observation_id": "seed:reference",
                "tool": "query_metric",
                "arguments": {
                    "entity_id": candidate["entity_id"],
                    "metric_id": candidate["metric_id"],
                    "period": "reference",
                },
                "status": reference["status"],
                "coverage": reference["coverage"],
                "artifact_refs": [],
                "attempt_id": "attempt:reference",
                "summary": json.dumps(reference["summary"]),
            }
        )
    )
    store.append("observation", observation.observation_id, observation)
    model = CapturingModel(
        [
            {
                "decision_summary": "fill only the missing child",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(max_model_calls=2, max_tool_attempts=1),
        ).run("candidate-novel-budget", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    effects = [cast_dict(item.payload) for item in store.records("tool_effect")]
    assert len(effects) == 1
    assert effects[0]["operation_id"][-1] == 1
    feedback = cast_dict(json.loads(cast(str, model.inputs[1][1]["content"])))[
        "feedback"
    ][0]["existing_results"][0]
    assert feedback["candidate_id"] == candidate["candidate_id"]
    assert feedback["logical_operation_index"] == 0
    assert result.report["budget"]["tool_attempts_used"] == 1
    store.close()


def test_query_candidate_all_duplicate_recall_keeps_candidate_audit(
    tmp_path: Path,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment, key_limit=1)[0]
    selection = {
        "type": "query",
        "tool": "query_candidate",
        "arguments": {"candidate_id": candidate["candidate_id"]},
    }
    model = CapturingModel(
        [
            {"decision_summary": "select candidate", "operations": [selection]},
            {
                "decision_summary": "dispose candidate",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": candidate["entity_id"],
                        "statement": candidate["fault_type"],
                        "status": "refuted",
                        "opposing_evidence_refs": [
                            "candidate-duplicate:evidence:decision:"
                            "candidate-duplicate:0001:0",
                            "candidate-duplicate:evidence:decision:"
                            "candidate-duplicate:0001:1",
                        ],
                    }
                ],
            },
            {"decision_summary": "select it again", "operations": [selection]},
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("candidate-duplicate", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(store.records("tool_effect")) == 2
    feedback = cast_dict(json.loads(cast(str, model.inputs[3][1]["content"])))[
        "feedback"
    ][0]
    assert [item["candidate_id"] for item in feedback["existing_results"]] == [
        candidate["candidate_id"],
        candidate["candidate_id"],
    ]
    assert all(
        item["logical_operation_index"] == 0 for item in feedback["existing_results"]
    )
    store.close()


def test_raw_metric_after_candidate_pair_is_still_rejected(
    tmp_path: Path,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment, key_limit=1)[0]
    model = CapturingModel(
        [
            {
                "decision_summary": "select candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "repeat raw incident",
                "operations": [paired_metric_queries(candidate)[1]],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("candidate-raw-duplicate", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(store.records("observation")) == 3
    assert len(store.records("tool_effect")) == 2
    assert "raw query_metric is disabled" in str(model.inputs[2])
    rejected = [
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "decision_rejected"
    ][0]
    assert "candidate_id" not in rejected
    assert "logical_operation_index" not in rejected
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_query_log_rejects_topology_source_filename_as_log_name(
    tmp_path: Path,
) -> None:
    model = CapturingModel(
        [
            {
                "decision_summary": "mistake a source file for a log name",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_log_samples",
                        "arguments": {
                            "entity_id": "bank:entity:Tomcat01",
                            "log_name": "log_service.csv",
                        },
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    store, result = run_runtime(tmp_path, [], model=model)

    assert result.status == "completed_partial"
    assert not store.records("observation")
    assert not store.records("tool_effect")
    feedback = str(model.inputs[1])
    assert "log_name is not available for entity_id" in feedback
    assert "use one exact log_names value" in feedback
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def candidate_log_queue(*_args: object) -> list[dict[str, object]]:
    return [
        {
            "entity_id": "bank:entity:Tomcat01",
            "log_names": ["apache_access_log", "gc", "localhost_access_log"],
        }
    ]


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [
        (
            {
                "entity_id": "bank:entity:Tomcat01",
                "log_name": "LOCALHOST_ACCESS_LOG.log",
            },
            {
                "entity_id": "bank:entity:Tomcat01",
                "log_name": "localhost_access_log",
            },
        ),
        (
            {
                "entity_id": "bank:entity:Tomcat01",
                "log_name": "invented.log",
                "contains": "OutOfMemory",
            },
            {
                "entity_id": "bank:entity:Tomcat01",
                "contains": "OutOfMemory",
            },
        ),
    ],
)
def test_candidate_log_selector_normalization_is_safe_and_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arguments: dict[str, Any],
    expected: dict[str, Any],
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    monkeypatch.setattr(
        "rootwalk.runtime.candidate_disposition_queue", candidate_log_queue
    )

    operation, validated = runtime._validated_query_call(  # pyright: ignore[reportPrivateUsage]
        QueryOperation(type="query", tool="query_log_samples", arguments=arguments)
    )

    assert validated == expected
    assert operation.arguments == expected
    store.close()


def test_candidate_log_selector_fails_closed_without_unique_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
    )
    monkeypatch.setattr(
        "rootwalk.runtime.candidate_disposition_queue", candidate_log_queue
    )

    with pytest.raises(ValueError, match="log_name is not available"):
        runtime._validated_query_call(  # pyright: ignore[reportPrivateUsage]
            QueryOperation(
                type="query",
                tool="query_log_samples",
                arguments={
                    "entity_id": "bank:entity:Tomcat01",
                    "log_name": "catalina.out",
                },
            )
        )
    store.close()


def test_candidate_log_selector_normalization_rejects_same_batch_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rootwalk.runtime.candidate_disposition_queue", candidate_log_queue
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "duplicate aliases",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_log_samples",
                        "arguments": {
                            "entity_id": "bank:entity:Tomcat01",
                            "log_name": "LOCALHOST_ACCESS_LOG.log",
                        },
                    },
                    {
                        "type": "query",
                        "tool": "query_log_samples",
                        "arguments": {
                            "entity_id": "bank:entity:Tomcat01",
                            "log_name": "localhost_access_log",
                        },
                    },
                ],
            },
            {
                "decision_summary": "stop after rejection",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    store, result = run_runtime(tmp_path, [], model=model)

    assert result.status == "completed_partial"
    assert store.records("observation") == []
    assert store.records("tool_effect") == []
    assert "duplicates operation 0 after normalization" in str(model.inputs[1])
    assert result.report["budget"]["tool_attempts_used"] == 0
    store.close()


def test_non_metric_queries_remain_allowed_after_candidate_pair(tmp_path: Path) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment, key_limit=1)[0]
    model = RecordedModelAdapter(
        [
            {
                "decision_summary": "select candidate",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_candidate",
                        "arguments": {"candidate_id": candidate["candidate_id"]},
                    }
                ],
            },
            {
                "decision_summary": "inspect logs and trace",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_log_samples",
                        "arguments": {
                            "entity_id": "bank:entity:Tomcat01",
                            "period": "incident",
                            "limit": 1,
                        },
                    },
                    {
                        "type": "query",
                        "tool": "query_trace_neighbors",
                        "arguments": {
                            "entity_id": "bank:entity:Tomcat01",
                            "direction": "both",
                            "limit": 1,
                        },
                    },
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("candidate-non-metric", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    observations = [cast_dict(item.payload) for item in store.records("observation")]
    assert [item["tool"] for item in observations[-2:]] == [
        "query_log_samples",
        "query_trace_neighbors",
    ]
    assert result.report["budget"]["validator_retries_used"] == 0
    store.close()


class InterruptingCandidateBank(TopEightCandidateBank):
    def __init__(
        self,
        public_input: Path,
        *,
        interrupt_incident: bool,
        interrupt_metric_id: str | None = None,
    ) -> None:
        super().__init__(public_input)
        self.interrupt_incident = interrupt_incident
        self.interrupt_metric_id = interrupt_metric_id
        self.periods: list[str] = []

    def query_metric(
        self,
        entity_id: str,
        metric_id: str,
        *,
        period: str = "incident",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> dict[str, Any]:
        self.periods.append(period)
        if (
            self.interrupt_incident
            and period == "incident"
            and (
                self.interrupt_metric_id is None
                or metric_id == self.interrupt_metric_id
            )
        ):
            raise asyncio.CancelledError
        return super().query_metric(
            entity_id,
            metric_id,
            period=period,
            start=start,
            end=end,
        )


def test_query_candidate_recovery_keeps_child_indices_and_only_fills_missing_side(
    tmp_path: Path,
) -> None:
    run_id = "candidate-recovery"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    first_environment = InterruptingCandidateBank(
        PUBLIC_INPUT,
        interrupt_incident=True,
        interrupt_metric_id="synthetic_source_metric_0",
    )
    candidates = seed_candidate_overview(
        first_store,
        first_environment,
        exact_series_count=8,
        source_metric_indices=(0,),
    )
    candidate = candidates[0]
    decision = {
        "decision_summary": "select one candidate",
        "operations": [
            {
                "type": "query",
                "tool": "query_candidate",
                "arguments": {"candidate_id": candidate["candidate_id"]},
            }
        ],
    }
    first_runtime = RootWalkRuntime(
        first_environment,
        RecordedModelAdapter([decision]),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=1),
    )

    first_runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    state = asyncio.run(interrupt_runtime_after(first_runtime, run_id, "route"))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            first_runtime._execute_query(  # pyright: ignore[reportPrivateUsage]
                cast(Any, state)
            )
        )
    assert first_environment.periods == [
        "reference",
        "incident",
        "reference",
        "incident",
    ]
    assert [
        item.stable_key.rsplit(":", 1)[-1]
        for item in first_store.records("tool_effect")
    ] == ["0", "1", "2"]
    first_store.close()

    second_store = RunStore(database)
    second_environment = InterruptingCandidateBank(
        PUBLIC_INPUT, interrupt_incident=False
    )
    result = asyncio.run(
        RootWalkRuntime(
            second_environment,
            NoCallModel(),
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=1),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "budget_exhausted"
    assert second_environment.periods == [
        "incident",
        *[
            period
            for _candidate in candidates[1:]
            for period in ("reference", "incident")
        ],
    ]
    effects = [cast_dict(item.payload) for item in second_store.records("tool_effect")]
    assert [item["operation_id"][-1] for item in effects] == list(range(18))
    assert effects[3]["candidate_id"] == candidate["candidate_id"]
    assert effects[3]["logical_operation_index"] == 0
    assert [item["candidate_id"] for item in effects[4::2]] == [
        item["candidate_id"] for item in candidates[1:]
    ]
    second_store.close()


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("unknown", "unknown candidate_id"),
        ("duplicate", "duplicate candidate_id"),
        ("missing", "count does not match"),
        ("reordered", "order is not canonical"),
        ("index", "logical_operation_index is not canonical"),
        ("children", "children do not match canonical"),
        ("auto_filled", "fields do not match canonical"),
        ("extra", "fields do not match canonical"),
        ("whole_field", "missing candidate_expansions"),
        ("null", "must be a list"),
    ],
)
def test_persisted_candidate_expansion_tampering_fails_closed(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    run_id = f"candidate-tamper-{tamper}"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    store = RunStore(database)
    environment = TopEightCandidateBank(PUBLIC_INPUT)
    candidates = seed_candidate_overview(store, environment, exact_series_count=8)
    runtime = RootWalkRuntime(
        environment,
        RecordedModelAdapter(
            [
                {
                    "decision_summary": "trigger schedule",
                    "operations": [
                        {
                            "type": "query",
                            "tool": "query_candidate",
                            "arguments": {
                                "candidate_id": candidates[-1]["candidate_id"]
                            },
                        }
                    ],
                }
            ]
        ),
        store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=1),
    )
    runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    state = asyncio.run(interrupt_runtime_after(runtime, run_id, "route"))
    decision_record = store.records("decision")[0]
    payload = json.loads(json.dumps(decision_record.payload))
    expansions = cast(list[dict[str, Any]], payload["candidate_expansions"])
    if tamper == "whole_field":
        payload.pop("candidate_expansions")
    elif tamper == "null":
        payload["candidate_expansions"] = None
    elif tamper == "unknown":
        expansions[0]["candidate_id"] = "unknown-candidate"
    elif tamper == "duplicate":
        expansions[1]["candidate_id"] = expansions[0]["candidate_id"]
    elif tamper == "missing":
        expansions.pop()
    elif tamper == "reordered":
        expansions[0], expansions[1] = expansions[1], expansions[0]
    elif tamper == "index":
        expansions[0]["logical_operation_index"] = 7
    elif tamper == "children":
        expansions[0]["children"][0]["period"] = "incident"
    elif tamper == "auto_filled":
        expansions[0].pop("auto_filled")
    else:
        expansions[0]["unexpected"] = True
    store.close()

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE records SET payload_json = ? WHERE stable_key = ?",
            (
                json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True
                ),
                decision_record.stable_key,
            ),
        )

    reopened = RunStore(database)
    restored = RootWalkRuntime(
        TopEightCandidateBank(PUBLIC_INPUT),
        NoCallModel(),
        reopened,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=1),
    )
    restored._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    with pytest.raises(ValueError, match=message):
        restored._restore_from_records(  # pyright: ignore[reportPrivateUsage]
            {"decision_id": cast(str, state["decision_id"])}
        )
    reopened.close()


def test_first_exact_metric_batch_rejects_other_operation_counts(
    tmp_path: Path,
) -> None:
    first, second = overview_signals_from_two_families()
    three_operations = [
        *paired_metric_queries(first),
        paired_metric_queries(second)[0],
    ]
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "overview",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 20},
                }
            ],
        },
        {
            "decision_summary": "submit three exact operations",
            "operations": three_operations,
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model, require_overview=True)

    assert result.status == "completed_partial"
    assert len(store.records("observation")) == 1
    assert "two or four query_metric operations" in str(model.inputs[2])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_first_exact_metric_batch_accepts_two_paired_families_then_exempts_later(
    tmp_path: Path,
) -> None:
    first, second = overview_signals_from_two_families()
    exact_batch = [*paired_metric_queries(first), *paired_metric_queries(second)]
    later_query = {
        "type": "query",
        "tool": "query_metric",
        "arguments": {
            "entity_id": first["entity_id"],
            "metric_id": first["metric_id"],
        },
    }
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "overview",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 20},
                }
            ],
        },
        {"decision_summary": "verify two families", "operations": exact_batch},
        {"decision_summary": "one later exact query", "operations": [later_query]},
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]

    store, result = run_runtime(tmp_path, records, require_overview=True)

    assert result.status == "completed_partial"
    observations = [cast_dict(item.payload) for item in store.records("observation")]
    assert len(observations) == 6
    assert [item["arguments"]["period"] for item in observations[1:5]] == [
        "reference",
        "incident",
        "reference",
        "incident",
    ]
    assert all(item["status"] in {"success", "partial"} for item in observations[1:5])
    assert "period" not in observations[5]["arguments"]
    assert result.report["budget"]["validator_retries_used"] == 0
    store.close()


def test_first_exact_metric_batch_rejects_metric_outside_overview_mapping(
    tmp_path: Path,
) -> None:
    first, second = overview_signals_from_two_families()
    exact_batch = [*paired_metric_queries(first), *paired_metric_queries(second)]
    exact_batch[0] = {
        **exact_batch[0],
        "arguments": {
            **cast_dict(exact_batch[0]["arguments"]),
            "metric_id": "not-copied-from-overview",
        },
    }
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "overview",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 20},
                }
            ],
        },
        {"decision_summary": "invent a metric", "operations": exact_batch},
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model, require_overview=True)

    assert result.status == "completed_partial"
    assert len(store.records("observation")) == 1
    assert "not in the persisted overview mapping" in str(model.inputs[2])
    assert "copy entity_id and metric_id exactly" in str(model.inputs[2])
    store.close()


@pytest.mark.parametrize(
    "prior_tool",
    ["query_metric", "query_log_samples", "query_trace_neighbors"],
)
def test_first_exact_gate_is_not_bypassed_without_a_complete_metric_pair(
    tmp_path: Path,
    prior_tool: str,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    signal = overview_signals_from_two_families()[0]
    overview_result = environment.summarize_anomalies(limit=20)
    overview = Observation.model_validate_json(
        json.dumps(
            {
                "observation_id": "seed:overview",
                "tool": "summarize_anomalies",
                "arguments": {"limit": 20},
                "status": overview_result["status"],
                "coverage": overview_result["coverage"],
                "artifact_refs": [],
                "attempt_id": "attempt:overview",
                "summary": json.dumps(
                    RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
                        overview_result
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            }
        )
    )
    prior_arguments: dict[str, Any]
    if prior_tool == "query_metric":
        prior_arguments = {
            "entity_id": signal["entity_id"],
            "metric_id": signal["metric_id"],
            "period": "reference",
        }
    elif prior_tool == "query_log_samples":
        prior_arguments = {"entity_id": signal["entity_id"]}
    else:
        prior_arguments = {"entity_id": signal["entity_id"], "direction": "both"}
    prior = Observation.model_validate_json(
        json.dumps(
            {
                "observation_id": f"seed:{prior_tool}",
                "tool": prior_tool,
                "arguments": prior_arguments,
                "status": "success",
                "coverage": overview_result["coverage"],
                "artifact_refs": [],
                "attempt_id": f"attempt:{prior_tool}",
                "summary": "seeded successful observation",
            }
        )
    )
    store = RunStore(tmp_path / "run.sqlite3")
    store.append("observation", overview.observation_id, overview)
    store.append("observation", prior.observation_id, prior)
    model = CapturingModel(
        [
            {
                "decision_summary": "try one metric without a prior pair",
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_metric",
                        "arguments": {
                            "entity_id": signal["entity_id"],
                            "metric_id": signal["metric_id"],
                            "period": "incident",
                        },
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifacts"),
        ).run(f"bypass-{prior_tool}", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(store.records("observation")) == 2
    assert "two or four query_metric operations" in str(model.inputs[1])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_refuted_hypothesis_cannot_become_a_finding(tmp_path: Path) -> None:
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0001:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "record refuted candidate",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat02",
                    "statement": "routine logs are not the root cause",
                    "status": "refuted",
                }
            ],
        },
        {
            "decision_summary": "incorrectly select it",
            "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
        },
        {
            "decision_summary": "abstain after feedback",
            "operations": [
                {"type": "finalize", "abstain_reason": "no supported hypothesis"}
            ],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model)

    assert result.status == "completed_partial"
    assert result.report["findings"] == []
    assert "only supported hypotheses" in str(model.inputs[2])
    store.close()


@pytest.mark.parametrize(
    ("supporting_tools", "expected_error"),
    [
        (
            ["query_metric"],
            "requires successful or partial query_log_samples corroboration",
        ),
        (
            ["query_log_samples"],
            "requires query_metric evidence",
        ),
    ],
)
def test_enforce_rejects_supported_hypothesis_without_corroboration(
    tmp_path: Path,
    supporting_tools: list[str],
    expected_error: str,
) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    evidence_ids = {
        tool: seed_tool_evidence(store, environment, tool) for tool in supporting_tools
    }
    model = CapturingModel(
        [
            {
                "decision_summary": "claim support too early",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": "bank:entity:Tomcat01",
                        "statement": "candidate under test",
                        "status": "supported",
                        "supporting_evidence_refs": list(evidence_ids.values()),
                    }
                ],
            },
            {
                "decision_summary": "abstain after rejection",
                "operations": [
                    {"type": "finalize", "abstain_reason": "not corroborated"}
                ],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("corroboration-enforce", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert not store.records("hypothesis")
    assert expected_error in str(model.inputs[1])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_enforce_rejects_cross_subject_corroboration(tmp_path: Path) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    metric_id = seed_tool_evidence(store, environment, "query_metric")
    other_trace_id = seed_tool_evidence(
        store,
        environment,
        "query_trace_neighbors",
        subject="bank:entity:Tomcat02",
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "claim support from another subject",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": "bank:entity:Tomcat01",
                        "statement": "candidate under test",
                        "status": "supported",
                        "supporting_evidence_refs": [metric_id, other_trace_id],
                    }
                ],
            },
            {
                "decision_summary": "abstain after rejection",
                "operations": [{"type": "finalize", "abstain_reason": "wrong subject"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run("corroboration-cross-subject", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert not store.records("hypothesis")
    assert "query_log_samples corroboration" in str(model.inputs[1])
    store.close()


def test_enforce_accepts_metric_with_partial_log_corroboration(
    tmp_path: Path,
) -> None:
    run_id = "corroboration-pass"
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    metric_id = seed_tool_evidence(store, environment, "query_metric")
    log_id = seed_tool_evidence(
        store, environment, "query_log_samples", status="partial"
    )
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0001:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "record corroborated support",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "candidate under test",
                    "status": "supported",
                    "supporting_evidence_refs": [metric_id, log_id],
                }
            ],
        },
        {
            "decision_summary": "finish",
            "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
        },
    ]

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            RecordedModelAdapter(records),
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert result.report["findings"][0]["hypothesis_id"] == hypothesis_id
    assert not [
        event
        for event in store.records("investigation_event")
        if cast_dict(event.payload).get("event_type") == "hypothesis_quality_audit"
    ]
    store.close()


@pytest.mark.parametrize(
    ("corroborator", "accepted"),
    [
        pytest.param("none", False, id="metric-only"),
        pytest.param("static-trace", False, id="static-trace"),
        pytest.param("filtered-paired-log", True, id="same-subject-log"),
    ],
)
def test_candidate_with_logs_requires_filtered_paired_log_corroboration(
    tmp_path: Path,
    corroborator: str,
    accepted: bool,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment, has_log_source=True)[0]
    subject = cast(str, candidate["entity_id"])
    evidence_ids = [
        seed_tool_evidence(
            store,
            environment,
            "query_metric",
            subject=subject,
            metric_id=cast(str, candidate["metric_id"]),
        )
    ]
    if corroborator == "static-trace":
        evidence_ids.append(
            seed_tool_evidence(
                store, environment, "query_trace_neighbors", subject=subject
            )
        )
    elif corroborator == "filtered-paired-log":
        evidence_ids.append(
            seed_tool_evidence(
                store,
                environment,
                "query_log_samples",
                subject=subject,
                paired_log_summary=True,
                filtered_log=True,
            )
        )

    errors = support_gate_errors(
        store,
        environment,
        subject=subject,
        statement=cast(str, candidate["fault_type"]),
        evidence_ids=evidence_ids,
        artifact_root=tmp_path / "artifact-store",
    )

    assert bool(errors) is not accepted
    if not accepted:
        assert "query_log_samples corroboration" in str(errors)
    store.close()


def test_candidate_without_logs_accepts_strict_direct_metric_evidence(
    tmp_path: Path,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment, has_log_source=False)[0]
    subject = cast(str, candidate["entity_id"])
    metric_id = seed_tool_evidence(
        store,
        environment,
        "query_metric",
        subject=subject,
        metric_id=cast(str, candidate["metric_id"]),
    )

    assert (
        support_gate_errors(
            store,
            environment,
            subject=subject,
            statement=cast(str, candidate["fault_type"]),
            evidence_ids=[metric_id],
            artifact_root=tmp_path / "artifact-store",
        )
        == []
    )
    store.close()


def test_candidate_source_pair_cannot_support_with_log_evidence(tmp_path: Path) -> None:
    environment = TopEightCandidateBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    source_metric_id = "synthetic_source_metric_0"
    candidate = seed_candidate_overview(
        store,
        environment,
        source_metric_id=source_metric_id,
        has_log_source=True,
    )[0]
    subject = cast(str, candidate["entity_id"])
    metric_evidence_id = seed_tool_evidence(
        store,
        environment,
        "query_metric",
        subject=subject,
        metric_id=source_metric_id,
    )
    log_evidence_id = seed_tool_evidence(
        store,
        environment,
        "query_log_samples",
        subject=subject,
        paired_log_summary=True,
        filtered_log=True,
    )

    errors = support_gate_errors(
        store,
        environment,
        subject=subject,
        statement=cast(str, candidate["fault_type"]),
        evidence_ids=[metric_evidence_id, log_evidence_id],
        artifact_root=tmp_path / "artifact-store",
    )

    assert "persisted direct candidate pair" in str(errors)
    store.close()


@pytest.mark.parametrize(
    "weakness",
    [
        "ambiguous-role",
        "unknown-direction",
        "incomplete-counts",
        "single-crossing",
        "opposite-median",
    ],
)
def test_candidate_without_logs_rejects_non_strict_metric_evidence(
    tmp_path: Path,
    weakness: str,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(
        store,
        environment,
        metric_role="ambiguous" if weakness == "ambiguous-role" else "direct",
        has_log_source=False,
    )[0]
    paired = strict_paired_metric_summary()
    if weakness == "unknown-direction":
        paired["pressure_direction"] = "unknown"
    elif weakness == "incomplete-counts":
        paired["counts_complete"] = False
    elif weakness == "single-crossing":
        cast(dict[str, Any], paired["incident"])["above_reference_max_count"] = 1
    elif weakness == "opposite-median":
        cast(dict[str, Any], paired["incident"]).update(
            {"min": 5.0, "median": 10.0, "below_reference_min_count": 1}
        )
    subject = cast(str, candidate["entity_id"])
    metric_id = seed_tool_evidence(
        store,
        environment,
        "query_metric",
        subject=subject,
        metric_id=cast(str, candidate["metric_id"]),
        paired_metric_summary=paired,
    )

    errors = support_gate_errors(
        store,
        environment,
        subject=subject,
        statement=cast(str, candidate["fault_type"]),
        evidence_ids=[metric_id],
        artifact_root=tmp_path / "artifact-store",
    )

    assert errors
    store.close()


def test_candidate_without_logs_rejects_second_subject_same_fault_family_shift(
    tmp_path: Path,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate, _other_family, competitor = seed_candidate_overview(
        store,
        environment,
        has_log_source=False,
        second_series_same_fault=True,
    )
    subject = cast(str, candidate["entity_id"])
    assert competitor["entity_id"] != subject
    target_metric_id = seed_tool_evidence(
        store,
        environment,
        "query_metric",
        subject=subject,
        metric_id=cast(str, candidate["metric_id"]),
    )
    competitor_evidence_id = seed_tool_evidence(
        store,
        environment,
        "query_metric",
        subject=cast(str, competitor["entity_id"]),
        metric_id=cast(str, competitor["metric_id"]),
    )

    errors = support_gate_errors(
        store,
        environment,
        subject=subject,
        statement=cast(str, candidate["fault_type"]),
        evidence_ids=[target_metric_id],
        artifact_root=tmp_path / "artifact-store",
    )

    assert len(errors) == 1
    prefix, raw_blocker = errors[0].split("; strict_fallback_blocker=", 1)
    assert "no second matching subject" in prefix
    assert json.loads(raw_blocker) == {
        "candidate_id": competitor["candidate_id"],
        "evidence_refs": [competitor_evidence_id],
        "metric_id": competitor["metric_id"],
        "paired_metric_summary": strict_paired_metric_summary(),
        "reason_code": "competing_subject",
        "required_correction": {"status": "inconclusive"},
        "subject": competitor["entity_id"],
    }
    store.close()


@pytest.mark.parametrize("strict", [True, False])
def test_candidate_alias_without_logs_requires_strict_metric_evidence(
    tmp_path: Path,
    strict: bool,
) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    _first, primary, alias = seed_candidate_overview(
        store,
        environment,
        has_log_source=False,
        alias_same_series_family=True,
    )
    paired = strict_paired_metric_summary()
    if not strict:
        cast(dict[str, Any], paired["incident"])["above_reference_max_count"] = 1
    subject = cast(str, primary["entity_id"])
    metric_id = seed_tool_evidence(
        store,
        environment,
        "query_metric",
        subject=subject,
        metric_id=cast(str, primary["metric_id"]),
        paired_metric_summary=paired,
    )

    errors = support_gate_errors(
        store,
        environment,
        subject=subject,
        statement=cast(str, alias["fault_type"]),
        evidence_ids=[metric_id],
        artifact_root=tmp_path / "artifact-store",
    )

    assert bool(errors) is not strict
    store.close()


def test_environment_name_does_not_add_a_trace_requirement(tmp_path: Path) -> None:
    environment = _BankEnvironment(PUBLIC_INPUT)
    environment.task = environment.task.model_copy(
        update={"environment": "openrca_market"}
    )
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(
        store,
        environment,
        has_log_source=True,
    )[0]
    subject = cast(str, candidate["entity_id"])
    metric_id = seed_tool_evidence(
        store,
        environment,
        "query_metric",
        subject=subject,
        metric_id=cast(str, candidate["metric_id"]),
    )
    log_id = seed_tool_evidence(
        store,
        environment,
        "query_log_samples",
        subject=subject,
        paired_log_summary=True,
        filtered_log=True,
    )

    errors = support_gate_errors(
        store,
        environment,
        subject=subject,
        statement=cast(str, candidate["fault_type"]),
        evidence_ids=[metric_id, log_id],
        artifact_root=tmp_path / "artifact-store",
    )
    assert not any("query_trace_neighbors" in error for error in errors)
    store.close()


class UnknownLogStateBank(_BankEnvironment):
    def list_metrics(self, entity_id: str | None = None) -> dict[str, Any]:
        if entity_id is not None:
            return {"status": "error"}
        return super().list_metrics(entity_id)


def test_candidate_unknown_log_state_fails_closed(tmp_path: Path) -> None:
    environment = UnknownLogStateBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    candidate = seed_candidate_overview(store, environment, omit_log_source=True)[0]
    subject = cast(str, candidate["entity_id"])
    metric_id = seed_tool_evidence(
        store,
        environment,
        "query_metric",
        subject=subject,
        metric_id=cast(str, candidate["metric_id"]),
    )

    errors = support_gate_errors(
        store,
        environment,
        subject=subject,
        statement=cast(str, candidate["fault_type"]),
        evidence_ids=[metric_id],
        artifact_root=tmp_path / "artifact-store",
    )

    assert "support fails closed" in str(errors)
    store.close()


def test_observe_only_allows_metric_only_support_and_writes_quality_audit(
    tmp_path: Path,
) -> None:
    run_id = "corroboration-observe"
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    metric_id = seed_tool_evidence(store, environment, "query_metric")
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0001:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "record metric-only support",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "candidate under test",
                    "status": "supported",
                    "supporting_evidence_refs": [metric_id],
                }
            ],
        },
        {
            "decision_summary": "finish",
            "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
        },
    ]

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            RecordedModelAdapter(records),
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode="observe_only"),
        ).run(run_id, tmp_path / "bundle")
    )

    audits = [
        cast_dict(event.payload)
        for event in store.records("investigation_event")
        if cast_dict(event.payload).get("event_type") == "hypothesis_quality_audit"
    ]
    assert result.status == "completed_partial"
    assert result.report["findings"][0]["hypothesis_id"] == hypothesis_id
    assert len(audits) == 1
    assert audits[0]["visibility"] == "audit_only"
    assert "query_log_samples corroboration" in str(audits[0]["errors"])
    store.close()


class MetricOnlyBank(BankEnvironment):
    capabilities = CapabilityManifest(metrics=True, topology=True)


def test_legacy_metric_only_environment_still_requires_metric_evidence(
    tmp_path: Path,
) -> None:
    environment = MetricOnlyBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    trace_id = seed_tool_evidence(
        store,
        environment,
        "query_trace_neighbors",
        subject="bank:entity:Tomcat01",
    )

    errors = support_gate_errors(
        store,
        environment,
        subject="bank:entity:Tomcat01",
        statement="candidate under test",
        evidence_ids=[trace_id],
        artifact_root=tmp_path / "artifact-store",
    )

    assert errors == ["supported hypothesis requires query_metric evidence"]
    store.close()


def test_metric_only_environment_does_not_require_log_or_trace_corroboration(
    tmp_path: Path,
) -> None:
    run_id = "metric-only-capability"
    environment = MetricOnlyBank(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    metric_id = seed_tool_evidence(store, environment, "query_metric")
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0001:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "record metric-only support",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "candidate under test",
                    "status": "supported",
                    "supporting_evidence_refs": [metric_id],
                }
            ],
        },
        {
            "decision_summary": "finish",
            "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
        },
    ]

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            RecordedModelAdapter(records),
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert result.report["findings"][0]["hypothesis_id"] == hypothesis_id
    store.close()


def test_resume_does_not_duplicate_hypothesis_quality_audit(tmp_path: Path) -> None:
    run_id = "resume-hypothesis-audit"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    environment = BankEnvironment(PUBLIC_INPUT)
    first_store = RunStore(database)
    metric_id = seed_tool_evidence(first_store, environment, "query_metric")
    operation = {
        "decision_summary": "record metric-only support",
        "operations": [
            {
                "type": "update_hypothesis",
                "action": "create",
                "subject": "bank:entity:Tomcat01",
                "statement": "candidate under test",
                "status": "supported",
                "supporting_evidence_refs": [metric_id],
            }
        ],
    }
    first_runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=1, validator_mode="observe_only"),
    )
    decision_id = f"decision:{run_id}:0001"
    decision = Decision.model_validate(operation)
    first_store.append("decision", decision_id, decision.model_dump(mode="json"))
    first_runtime._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    first_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    first_runtime._restore_from_records(  # pyright: ignore[reportPrivateUsage]
        {"decision_id": decision_id}
    )
    asyncio.run(
        first_runtime._apply_hypothesis(  # pyright: ignore[reportPrivateUsage]
            cast(Any, {})
        )
    )
    first_store.close()

    second_store = RunStore(database)
    resumed = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        second_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=1, validator_mode="observe_only"),
    )
    resumed._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    resumed._bootstrap()  # pyright: ignore[reportPrivateUsage]
    resumed._restore_from_records(  # pyright: ignore[reportPrivateUsage]
        {"decision_id": decision_id}
    )
    asyncio.run(
        resumed._apply_hypothesis(  # pyright: ignore[reportPrivateUsage]
            cast(Any, {})
        )
    )

    audits = [
        event
        for event in second_store.records("investigation_event")
        if cast_dict(event.payload).get("event_type") == "hypothesis_quality_audit"
    ]
    assert len(audits) == 1
    second_store.close()


def test_budget_exhaustion_never_makes_an_extra_model_call(tmp_path: Path) -> None:
    model = CapturingModel(
        [
            {
                "decision_summary": "inspect entities",
                "operations": [
                    {"type": "query", "tool": "list_entities", "arguments": {}}
                ],
            }
        ]
    )

    store, result = run_runtime(tmp_path, [], max_model_calls=1, model=model)

    assert result.status == "budget_exhausted"
    assert result.model_calls == len(model.inputs) == 1
    assert len(store.records("context_manifest")) == 1
    store.close()


@pytest.mark.parametrize(
    ("max_model_calls", "max_input_tokens", "expected_reason"),
    [
        (6, 144_000, "decision_rounds"),
        (12, 48_000, "input_tokens"),
    ],
)
def test_tight_budget_projects_full_budget_and_requires_closeout(
    tmp_path: Path,
    max_model_calls: int,
    max_input_tokens: int,
    expected_reason: str,
) -> None:
    records = [
        {
            "decision_summary": "verify one exact metric",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "incident",
                    },
                }
            ],
        },
        {
            "decision_summary": "keep one evidence-backed open hypothesis",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "The exact metric result needs causal verification",
                    "status": "open",
                    "supporting_evidence_refs": [EVIDENCE_ID],
                }
            ],
        },
        {
            "decision_summary": "close within budget",
            "operations": [
                {"type": "finalize", "abstain_reason": "causality remains unresolved"}
            ],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(
        tmp_path,
        records,
        max_model_calls=max_model_calls,
        max_input_tokens=max_input_tokens,
        model=model,
    )

    assert result.status == "completed_partial"
    context = cast_dict(json.loads(cast(str, model.inputs[2][1]["content"])))
    budget = cast_dict(context["budget"])
    assert budget["model_calls_used"] == 2
    assert budget["model_calls_remaining"] == max_model_calls - 2
    assert budget["provider_attempts_remaining"] == max_model_calls
    assert budget["input_tokens_remaining"] == (
        max_input_tokens - budget["input_tokens_used"]
    )
    assert budget["output_tokens_remaining"] == (
        budget["max_output_tokens"] - budget["output_tokens_used"]
    )
    assert budget["tool_attempts_used"] == 1
    assert budget["tool_attempts_remaining"] == budget["max_tool_attempts"] - 1
    assert budget["validator_retries_used"] == 0
    assert budget["validator_retries_remaining"] == budget["max_validator_retries"]
    assert budget["no_progress_rounds"] == 0
    assert budget["no_progress_rounds_remaining"] == budget["max_no_progress_rounds"]
    assert 0 <= budget["deadline_seconds_remaining"] <= 300
    assert budget["closeout_required"] is True
    assert budget["closeout_reason"] == expected_reason
    store.close()


class CancellingQueryExecutor:
    sandbox_requested: Literal["auto", "required", "off"] = "off"
    sandbox_effective: Literal["bwrap", "spawn"] = "spawn"
    bwrap_version: str | None = None
    bwrap_executable_sha256: str | None = None

    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any], float]] = []

    def execute(
        self, tool: str, args: Mapping[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        self.calls.append((tool, args, timeout_s))
        return {
            "status": "cancelled",
            "error_type": "query_timeout",
            "summary": "query worker was terminated",
        }


class BwrapProvenanceExecutor(CancellingQueryExecutor):
    sandbox_requested: Literal["auto", "required", "off"] = "required"
    sandbox_effective: Literal["bwrap", "spawn"] = "bwrap"
    bwrap_executable_sha256: str | None = "a" * 64

    def __init__(self, version: str, sha256: str = "a" * 64) -> None:
        super().__init__()
        self.bwrap_version = version
        self.bwrap_executable_sha256 = sha256


def test_runtime_uses_bounded_query_executor(tmp_path: Path) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "list entities",
            "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
        },
        {
            "decision_summary": "abstain",
            "operations": [{"type": "finalize", "abstain_reason": "query timed out"}],
        },
    ]
    executor = CancellingQueryExecutor()
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        CapturingModel(records),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(tool_timeout_seconds=0.5),
        query_executor=executor,
    )

    result = asyncio.run(runtime.run(RUN_ID, tmp_path / "bundle"))

    assert result.status == "completed_partial"
    assert len(executor.calls) == 1
    assert 0 < executor.calls[0][2] <= 0.5
    observation = cast_dict(store.records("observation")[0].payload)
    assert observation["status"] == "cancelled"
    store.close()


def test_runtime_rejects_query_executor_provenance_mismatch(tmp_path: Path) -> None:
    with RunStore(tmp_path / "run.sqlite3") as store:
        with pytest.raises(ValueError, match="does not match RuntimeConfig"):
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                NoCallModel(),
                store,
                ArtifactStore(tmp_path / "artifacts"),
                RuntimeConfig(
                    query_sandbox_requested="required",
                    query_sandbox_effective="bwrap",
                    query_sandbox_bwrap_version="bubblewrap-test",
                    query_sandbox_bwrap_executable_sha256="a" * 64,
                ),
                query_executor=CancellingQueryExecutor(),
            )


def test_spawn_trace_headers_stay_out_of_prompt_bundle_and_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "list entities",
            "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    model = CapturingModel(records)
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    config = ObservabilityConfig(enabled=True)
    store = RunStore(tmp_path / "run.sqlite3")
    worker = QueryWorker(PUBLIC_INPUT)
    runtime_config = RuntimeConfig(
        query_sandbox_requested=worker.sandbox_requested,
        query_sandbox_effective=worker.sandbox_effective,
        query_sandbox_bwrap_version=worker.bwrap_version,
        query_sandbox_bwrap_executable_sha256=worker.bwrap_executable_sha256,
    )
    try:
        result = asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                model,
                store,
                ArtifactStore(tmp_path / "artifact-store"),
                runtime_config,
                query_executor=worker,
                tracer=provider.get_tracer("test.query-worker"),
                observability_config=config,
            ).run(RUN_ID, tmp_path / "bundle")
        )
        assert provider.force_flush(1_000)
    finally:
        worker.close()
        provider.shutdown()

    tool = next(
        span for span in exporter.get_finished_spans() if span.name == "rootwalk.tool"
    )
    assert tool.context is not None
    traceparent = (
        f"00-{tool.context.trace_id:032x}-{tool.context.span_id:016x}-"
        f"{int(tool.context.trace_flags):02x}"
    )
    prompt = json.dumps(model.inputs, default=str)
    bundle = b"".join(
        path.read_bytes() for path in result.bundle_path.rglob("*") if path.is_file()
    )

    assert result.status == "completed_partial"
    for forbidden in ("traceparent", "tracestate", traceparent):
        assert forbidden not in prompt
        assert forbidden.encode() not in bundle
        assert forbidden not in caplog.text
    store.close()


def test_out_of_topology_hypothesis_is_rejected_without_evidence(
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "claim an unauthorized entity",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:not-in-topology",
                    "statement": "unsupported claim",
                    "status": "supported",
                }
            ],
        },
        {
            "decision_summary": "abstain",
            "operations": [
                {
                    "type": "finalize",
                    "abstain_reason": "The entity is outside the allowed topology",
                }
            ],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model)

    assert result.status == "completed_partial"
    assert not store.records("hypothesis")
    assert not store.records("evidence")
    assert "outside topology" in str(model.inputs[1])
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_revise_hypothesis_cannot_change_subject(
    tmp_path: Path,
    validator_mode: str,
) -> None:
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0001:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "create hypothesis",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "initial candidate",
                    "status": "open",
                }
            ],
        },
        {
            "decision_summary": "rewrite another subject",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "revise",
                    "hypothesis_id": hypothesis_id,
                    "subject": "bank:entity:MG02",
                    "statement": "different candidate",
                    "status": "inconclusive",
                }
            ],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(
        tmp_path,
        records,
        model=model,
        validator_mode=validator_mode,
    )

    assert result.status == "completed_partial"
    hypotheses = [cast_dict(item.payload) for item in store.records("hypothesis")]
    assert len(hypotheses) == 1
    assert hypotheses[0]["subject"] == "bank:entity:Tomcat01"
    assert "create a new hypothesis" in str(model.inputs[2])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_revise_hypothesis_allows_same_subject(tmp_path: Path) -> None:
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0001:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "create hypothesis",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "initial candidate",
                    "status": "open",
                }
            ],
        },
        {
            "decision_summary": "revise same subject",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "revise",
                    "hypothesis_id": hypothesis_id,
                    "subject": "bank:entity:Tomcat01",
                    "statement": "same subject is inconclusive",
                    "status": "inconclusive",
                }
            ],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]

    store, result = run_runtime(tmp_path, records)

    assert result.status == "completed_partial"
    hypotheses = [cast_dict(item.payload) for item in store.records("hypothesis")]
    assert len(hypotheses) == 2
    assert hypotheses[-1]["subject"] == "bank:entity:Tomcat01"
    assert hypotheses[-1]["statement"] == "same subject is inconclusive"
    assert result.report["budget"]["validator_retries_used"] == 0
    store.close()


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_revise_hypothesis_cannot_change_resolved_public_fault_type(
    tmp_path: Path,
    validator_mode: str,
) -> None:
    run_id = f"revise-fault-{validator_mode}"
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0001:0"
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    seed_public_fault_types(store, environment)
    model = CapturingModel(
        [
            {
                "decision_summary": "create CPU hypothesis",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": "bank:entity:Tomcat01",
                        "statement": "Evidence suggests high CPU usage.",
                        "status": "open",
                    }
                ],
            },
            {
                "decision_summary": "rewrite as memory fault",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "revise",
                        "hypothesis_id": hypothesis_id,
                        "subject": "bank:entity:Tomcat01",
                        "statement": "Evidence now suggests high memory usage.",
                        "status": "inconclusive",
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode=cast(Any, validator_mode)),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    hypotheses = [cast_dict(item.payload) for item in store.records("hypothesis")]
    assert len(hypotheses) == 1
    assert hypotheses[0]["statement"] == "Evidence suggests high CPU usage."
    assert "create a new hypothesis" in str(model.inputs[2])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


@pytest.mark.parametrize(
    ("existing_statement", "revised_statement"),
    [
        (
            "Evidence suggests HIGH\nCPU USAGE.",
            "High CPU usage remains possible after exact checks.",
        ),
        ("candidate is not uniquely classified", "high memory usage"),
        (
            "high CPU usage",
            "Both high CPU usage and high memory usage remain possible.",
        ),
    ],
)
def test_revise_hypothesis_allows_same_or_unresolved_fault_identity(
    tmp_path: Path,
    existing_statement: str,
    revised_statement: str,
) -> None:
    run_id = "revise-fault-safe-boundary"
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0001:0"
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    seed_public_fault_types(store, environment)
    model = RecordedModelAdapter(
        [
            {
                "decision_summary": "create hypothesis",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "create",
                        "subject": "bank:entity:Tomcat01",
                        "statement": existing_statement,
                        "status": "open",
                    }
                ],
            },
            {
                "decision_summary": "refine hypothesis wording",
                "operations": [
                    {
                        "type": "update_hypothesis",
                        "action": "revise",
                        "hypothesis_id": hypothesis_id,
                        "subject": "bank:entity:Tomcat01",
                        "statement": revised_statement,
                        "status": "inconclusive",
                    }
                ],
            },
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    hypotheses = [cast_dict(item.payload) for item in store.records("hypothesis")]
    assert len(hypotheses) == 2
    assert hypotheses[-1]["statement"] == revised_statement
    assert result.report["budget"]["validator_retries_used"] == 0
    store.close()


def test_supported_hypothesis_requires_supporting_evidence_in_observe_only(
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "claim support without evidence",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "unsupported claim",
                    "status": "supported",
                }
            ],
        },
        {
            "decision_summary": "abstain",
            "operations": [
                {"type": "finalize", "abstain_reason": "no supporting evidence"}
            ],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(
        tmp_path, records, model=model, validator_mode="observe_only"
    )

    assert result.status == "completed_partial"
    assert not store.records("hypothesis")
    assert "supported hypothesis requires supporting evidence" in str(model.inputs[1])
    store.close()


def test_finalize_revalidates_legacy_supported_hypothesis_without_evidence(
    tmp_path: Path,
) -> None:
    hypothesis_id = "legacy-supported-without-evidence"
    store = RunStore(tmp_path / "run.sqlite3")
    store.append(
        "hypothesis",
        "hypothesis:legacy",
        Hypothesis(
            hypothesis_id=hypothesis_id,
            subject="bank:entity:Tomcat01",
            statement="legacy unsupported claim",
            status="supported",
        ),
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "publish legacy claim",
                "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
            },
            {
                "decision_summary": "abstain after rejection",
                "operations": [
                    {"type": "finalize", "abstain_reason": "no supporting evidence"}
                ],
            },
        ]
    )
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode="observe_only"),
        ).run("legacy-supported", tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert not result.report["findings"]
    assert "has no supporting evidence" in str(model.inputs[1])
    store.close()


def test_finalize_rejects_legacy_navigation_evidence_as_support(
    tmp_path: Path,
) -> None:
    run_id = "legacy-navigation-evidence"
    observation_id = "legacy-navigation-observation"
    evidence_id = "legacy-navigation-evidence"
    hypothesis_id = "legacy-navigation-hypothesis"
    environment = BankEnvironment(PUBLIC_INPUT)
    coverage = {
        "requested_range": environment.task.allowed_query_range.model_dump(mode="json"),
        "observed_range": environment.task.allowed_query_range.model_dump(mode="json"),
        "completeness": "complete",
        "collection_health": "healthy",
        "basis": "legacy fixture",
    }
    store = RunStore(tmp_path / "run.sqlite3")
    store.append(
        "observation",
        observation_id,
        {
            "observation_id": observation_id,
            "tool": "summarize_anomalies",
            "arguments": {},
            "status": "success",
            "coverage": coverage,
            "artifact_refs": [],
            "attempt_id": "legacy-attempt",
        },
    )
    store.append(
        "evidence",
        evidence_id,
        {
            "evidence_id": evidence_id,
            "observation_refs": [observation_id],
            "subject_ids": ["bank:entity:Tomcat01"],
            "predicate": "summarize_anomalies.result",
            "value": {"status": "success"},
            "observed_interval": environment.task.allowed_query_range.model_dump(
                mode="json"
            ),
            "time_uncertainty": "legacy",
            "artifact_refs": [],
            "extractor": {
                "kind": "deterministic",
                "name": "legacy",
                "version": "1",
                "parameters_hash": "legacy",
            },
            "source_fingerprint": "legacy",
            "dataset_snapshot_id": environment.dataset_id,
        },
    )
    store.append(
        "hypothesis",
        hypothesis_id,
        Hypothesis(
            hypothesis_id=hypothesis_id,
            subject="bank:entity:Tomcat01",
            statement="navigation-only candidate",
            status="supported",
            supporting_evidence_refs=[evidence_id],
        ),
    )
    model = CapturingModel(
        [
            {
                "decision_summary": "publish navigation-only claim",
                "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
            },
            {
                "decision_summary": "abstain after rejection",
                "operations": [
                    {"type": "finalize", "abstain_reason": "navigation only"}
                ],
            },
        ]
    )

    result = asyncio.run(
        RootWalkRuntime(
            environment,
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            RuntimeConfig(validator_mode="observe_only"),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert not result.report["findings"]
    assert "relies on navigation results" in str(model.inputs[1])
    store.close()


class HealthyNoDataBank(BankEnvironment):
    def query_metric(
        self,
        entity_id: str,
        metric_id: str,
        *,
        period: str = "incident",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> dict[str, Any]:
        del entity_id, metric_id, period, start, end
        return {
            "status": "no_data",
            "summary": "no matching samples",
            "coverage": {
                "requested_range": self.task.allowed_query_range.model_dump(
                    mode="json"
                ),
                "observed_range": self.task.allowed_query_range.model_dump(mode="json"),
                "sample_interval": None,
                "completeness": "complete",
                "sampling": None,
                "truncation": None,
                "collection_health": "healthy",
                "basis": "complete frozen fixture",
            },
            "artifact_refs": [],
        }


class ReferenceNoDataBank(BankEnvironment):
    def query_metric(
        self,
        entity_id: str,
        metric_id: str,
        *,
        period: str = "incident",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> dict[str, Any]:
        del entity_id, metric_id
        _, window = self._window(period, start, end)
        return self._no_data(
            window,
            "no matching samples",
            basis="source collection completeness and health are unknown",
        )


def test_reference_no_data_coverage_survives_runtime_validation(
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "check the reference window",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "reference",
                    },
                }
            ],
        },
        {
            "decision_summary": "stop after the coverage gap",
            "operations": [
                {"type": "finalize", "abstain_reason": "reference data is absent"}
            ],
        },
    ]

    store, result = run_runtime(
        tmp_path,
        records,
        environment=ReferenceNoDataBank(PUBLIC_INPUT),
    )

    observation = cast_dict(store.records("observation")[0].payload)
    requested_range = cast_dict(cast_dict(observation["coverage"])["requested_range"])
    assert result.status == "completed_partial"
    assert set(requested_range) == {"start", "end", "end_exclusive"}
    store.close()


def test_complete_healthy_no_data_creates_qualified_negative_evidence(
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "check an absent series",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                    },
                }
            ],
        }
    ]

    store, result = run_runtime(
        tmp_path,
        records,
        max_model_calls=1,
        environment=HealthyNoDataBank(PUBLIC_INPUT),
    )

    evidence = cast_dict(store.records("evidence")[0].payload)
    assert evidence["predicate"].endswith("no_data_in_complete_healthy_range")
    assert not result.report["gaps"]
    store.close()


class SingleResultModel:
    def __init__(self, result: ModelCallResult) -> None:
        self.result = result
        self.calls = 0

    async def decide(self, messages: Sequence[Message]) -> ModelCallResult:
        del messages
        self.calls += 1
        return self.result


def test_model_usage_is_preserved_and_accumulated(tmp_path: Path) -> None:
    decision = Decision.model_validate(
        {
            "decision_summary": "finish",
            "operations": [
                {"type": "finalize", "abstain_reason": "No investigation needed"}
            ],
        }
    )
    usage: dict[str, Any] = {
        "prompt_tokens": 7,
        "completion_tokens": 3,
        "prompt_tokens_details": {"cached_tokens": 2},
    }
    model = SingleResultModel(
        ModelCallResult(
            decision=decision,
            usage=usage,
            provider_model="recorded-usage",
            raw_response={"decision": decision.model_dump(mode="json")},
        )
    )

    store, result = run_runtime(tmp_path, [], model=model)

    assert cast_dict(store.records("model_response")[0].payload)["usage"] == usage
    success = [
        cast_dict(item.payload)
        for item in store.records("attempt")
        if cast_dict(item.payload)["status"] == "success"
    ][0]
    assert success["usage"] == usage
    completed = [
        cast_dict(item.payload)
        for item in store.records("budget_ledger")
        if cast_dict(item.payload)["event"] == "model_call_completed"
    ][0]
    assert completed["usage"] == usage
    assert completed["cumulative_usage"] == {
        "prompt_tokens": 7,
        "completion_tokens": 3,
    }
    assert result.report["model_usage"] == completed["cumulative_usage"]
    store.close()


def test_recorded_model_usage_is_estimated_and_audited(tmp_path: Path) -> None:
    store, result = run_runtime(
        tmp_path,
        [
            {
                "decision_summary": "finish",
                "operations": [
                    {"type": "finalize", "abstain_reason": "recorded fixture"}
                ],
            }
        ],
    )

    completed = [
        cast_dict(item.payload)
        for item in store.records("budget_ledger")
        if cast_dict(item.payload)["event"] == "model_call_completed"
    ][0]
    requests = [cast_dict(item.payload) for item in store.records("model_request")]

    assert result.status == "completed_partial"
    assert completed["settlement"] == "actual"
    assert 0 < completed["charged_output_tokens"] < 2_000
    assert requests[0]["recorded_cursor"] == 1
    assert requests[0]["recorded_count"] == 1
    store.close()


class CancellingModel:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, messages: Sequence[Message]) -> object:
        del messages
        self.calls += 1
        raise asyncio.CancelledError


def test_cancelled_model_call_seals_bundle_without_another_call(
    tmp_path: Path,
) -> None:
    model = CancellingModel()

    store, result = run_runtime(tmp_path, [], model=model)

    assert result.status == "cancelled"
    assert model.calls == result.model_calls == 1
    assert (result.bundle_path / "report.json").is_file()
    store.close()


def test_context_limit_persists_manifest_without_calling_model(
    tmp_path: Path,
) -> None:
    model = CapturingModel(
        [
            {
                "decision_summary": "must not be called",
                "operations": [{"type": "finalize", "abstain_reason": "not reached"}],
            }
        ]
    )

    store, result = run_runtime(tmp_path, [], max_context_tokens=1, model=model)

    assert result.status == "context_limit"
    assert result.model_calls == len(model.inputs) == 0
    manifest = cast_dict(store.records("context_manifest")[0].payload)
    assert manifest["estimated_tokens"] > manifest["max_context_tokens"]
    assert store.records("model_input")
    store.close()


def test_model_context_uses_shared_context_compiler(tmp_path: Path) -> None:
    model = CapturingModel(
        [
            {
                "decision_summary": "finish",
                "operations": [{"type": "finalize", "abstain_reason": "done"}],
            }
        ]
    )

    store, _result = run_runtime(tmp_path, [], model=model)

    content = model.inputs[0][1]["content"]
    assert isinstance(content, str)
    context = cast_dict(json.loads(content))
    assert context["topology"]["snapshot_id"] == "synthetic-bank-topology-v1"
    assert '"samples"' not in json.dumps(context["topology"])
    assert "filenames are invalid" in context["tools"]["read_artifact_slice"]
    prompt = str(model.inputs[0][0]["content"])
    assert "navigation only, not causal evidence" in prompt
    assert "list_metrics.log_names" in prompt
    assert "successful or explicitly non-retryable" in prompt
    assert "After one broad scan" in prompt
    assert context["context_policy"] == {
        "strategy": "evidence",
        "validator_mode": "enforce",
        "base_revision": 3,
    }
    store.close()


def test_anomaly_payload_is_navigation_only_but_visible_in_next_context(
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "summarize",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 20},
                }
            ],
        },
        {
            "decision_summary": "try to promote the navigation result",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "navigation result is not causal proof",
                    "status": "supported",
                    "supporting_evidence_refs": [EVIDENCE_ID],
                }
            ],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "navigation only"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model)

    assert result.status == "completed_partial"
    second_context = cast_dict(json.loads(cast(str, model.inputs[1][1]["content"])))
    assert not store.records("evidence")
    assert not store.records("hypothesis")
    assert not result.report["findings"]
    assert not result.report["gaps"]
    summary = second_context["recent_interactions"][0]["result"]
    assert summary["signals"]
    assert summary["public_fault_types"] == ["synthetic public fault type"]
    assert "fault_type_candidates" in summary
    assert len(summary["fault_type_candidates"]) <= min(
        len(summary["public_fault_types"]), 16
    )
    for candidate in summary["fault_type_candidates"]:
        assert set(candidate) - {
            "source_metric_id",
            "source_pressure_direction",
            "candidate_id",
            "candidate_rank",
            "fault_type_aliases",
        } == {
            "fault_type",
            "entity_id",
            "metric_id",
            "resource_family",
            "score",
            "overview_score",
            "reference",
            "incident",
            "requires_paired_exact",
            "metric_role",
            "candidate_pressure_direction",
            "has_log_source",
            "log_names",
        }
        assert set(candidate["reference"]) == {"min", "median", "max"}
        assert set(candidate["incident"]) == {"min", "median", "max"}
    assert "navigation only" in summary["method"]
    for signal in summary["signals"]:
        assert set(signal) == {
            "entity_id",
            "metric_id",
            "resource_family",
            "overview_score",
            "relative_median_shift",
            "relative_extreme_shift",
            "stats_line",
            "direct_metric_candidates",
            "reference",
            "incident",
        }
        assert len(signal["direct_metric_candidates"]) <= 2
        assert all(
            isinstance(candidate, str)
            for candidate in signal["direct_metric_candidates"]
        )
        assert set(signal["reference"]) == {"count", "min", "median", "max"}
        assert set(signal["incident"]) == {"count", "min", "median", "max"}
    serialized_summary = json.dumps(summary, sort_keys=True)
    assert "example_source_records" not in serialized_summary
    assert '"mean"' not in serialized_summary
    assert "value_unit" not in serialized_summary
    payload = cast_dict(store.records("query_payload")[0].payload)
    raw_result = cast_dict(
        json.loads(
            (
                tmp_path / "artifact-store" / cast(str, payload["content_hash"])
            ).read_text()
        )
    )
    assert "mean" in raw_result["signals"][0]["reference"]
    assert "example_source_records" in raw_result["signals"][0]["reference"]
    assert "value_unit" in raw_result["signals"][0]
    assert (
        raw_result["signals"][0]["direct_metric_candidates"]
        == summary["signals"][0]["direct_metric_candidates"]
    )
    assert raw_result["fault_type_candidates"] == summary["fault_type_candidates"]
    assert "unknown_series" in raw_result
    assert "unknown evidence references" in str(model.inputs[2])
    store.close()


def test_navigation_signal_summary_drops_unlisted_fields() -> None:
    summary = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {
            "status": "success",
            "public_fault_types": ["public category"],
            "method": "public deterministic method",
            "signals": [
                {
                    "entity_id": "entity-1",
                    "metric_id": "metric-1",
                    "resource_family": "memory",
                    "overview_score": 1.0,
                    "relative_median_shift": 0.5,
                    "relative_extreme_shift": 1.0,
                    "stats_line": 7,
                    "direct_metric_candidates": [
                        "metric-direct-1",
                        "metric-direct-2",
                        "metric-direct-3",
                        4,
                    ],
                    "value_unit": "private-unit-detail",
                    "hidden_label": "must-not-enter-context",
                    "reference": {
                        "count": 2,
                        "min": 1.0,
                        "median": 2.0,
                        "max": 3.0,
                        "mean": 2.0,
                        "example_source_records": [10],
                    },
                    "incident": {
                        "count": 2,
                        "min": 4.0,
                        "median": 5.0,
                        "max": 6.0,
                        "mean": 5.0,
                        "example_source_records": [20],
                    },
                }
            ],
        }
    )

    assert summary["public_fault_types"] == ["public category"]
    assert summary["method"] == "public deterministic method"
    signal = summary["signals"][0]
    assert "hidden_label" not in signal
    assert "value_unit" not in signal
    assert signal["direct_metric_candidates"] == [
        "metric-direct-1",
        "metric-direct-2",
    ]
    assert signal["reference"] == {
        "count": 2,
        "min": 1.0,
        "median": 2.0,
        "max": 3.0,
    }
    assert signal["incident"] == {
        "count": 2,
        "min": 4.0,
        "median": 5.0,
        "max": 6.0,
    }


def test_log_evidence_projects_only_valid_complete_paired_counts() -> None:
    paired = {
        "counts_complete": True,
        "samples_truncated": True,
        "reference": {
            "matched_count": 3,
            "log_name_counts": {"rare": 1, "access": 2},
            "private_rows": [1, 2, 3],
        },
        "incident": {
            "matched_count": 5,
            "log_name_counts": {"rare": 2, "access": 3},
        },
        "private_detail": "drop me",
    }

    projected = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {"status": "partial", "paired_log_summary": paired}
    )

    assert projected["paired_log_summary"] == {
        "counts_complete": True,
        "samples_truncated": True,
        "reference": {
            "matched_count": 3,
            "log_name_counts": {"access": 2, "rare": 1},
        },
        "incident": {
            "matched_count": 5,
            "log_name_counts": {"access": 3, "rare": 2},
        },
    }
    invalid = dict(paired)
    invalid["incident"] = {
        "matched_count": 99,
        "log_name_counts": {"access": 3},
    }
    assert "paired_log_summary" not in RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {"status": "success", "paired_log_summary": invalid}
    )
    samples = [{"log_name": "rare", "text": "failure"}]
    broad = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {"status": "partial", "log_name": None, "contains": None, "samples": samples}
    )
    exact = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {"status": "success", "log_name": "rare", "samples": samples}
    )
    assert broad["sample_count"] == 1
    assert "samples" not in broad
    assert exact["samples"] == samples


def test_metric_evidence_projects_only_compact_valid_paired_summary() -> None:
    paired = {
        "counts_complete": True,
        "scale_hint": "0..100",
        "pressure_direction": "higher_is_more_pressure",
        "hint_basis": "heuristic from metric name",
        "reference": {
            "sample_count": 3,
            "min": 10.0,
            "median": 20.0,
            "max": 30.0,
            "private_points": [10.0, 20.0, 30.0],
        },
        "incident": {
            "sample_count": 4,
            "min": 15.0,
            "median": 20.0,
            "max": 40.0,
            "above_reference_max_count": 2,
            "below_reference_min_count": 0,
        },
        "private_detail": "drop me",
    }

    projected = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {"status": "success", "paired_metric_summary": paired}
    )

    assert projected["paired_metric_summary"] == {
        "counts_complete": True,
        "scale_hint": "0..100",
        "pressure_direction": "higher_is_more_pressure",
        "hint_basis": "heuristic from metric name",
        "reference": {
            "sample_count": 3,
            "min": 10.0,
            "median": 20.0,
            "max": 30.0,
        },
        "incident": {
            "sample_count": 4,
            "min": 15.0,
            "median": 20.0,
            "max": 40.0,
            "above_reference_max_count": 2,
            "below_reference_min_count": 0,
        },
    }
    invalid = dict(paired)
    invalid["incident"] = {
        **cast(dict[str, Any], paired["incident"]),
        "above_reference_max_count": 5,
    }
    assert "paired_metric_summary" not in RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {"status": "success", "paired_metric_summary": invalid}
    )
    inconsistent = dict(paired)
    inconsistent["incident"] = {
        **cast(dict[str, Any], paired["incident"]),
        "max": 20.0,
        "above_reference_max_count": 1,
    }
    assert "paired_metric_summary" not in RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        {"status": "success", "paired_metric_summary": inconsistent}
    )

    actionable = dict(paired)
    actionable["incident"] = {
        **cast(dict[str, Any], paired["incident"]),
        "max": 90.0,
    }
    assert (
        _has_actionable_percent_pressure({"paired_metric_summary": actionable}) is True
    )
    one_crossing = dict(actionable)
    one_crossing["incident"] = {
        **cast(dict[str, Any], actionable["incident"]),
        "above_reference_max_count": 1,
    }
    assert (
        _has_actionable_percent_pressure({"paired_metric_summary": one_crossing})
        is False
    )
    strict = dict(actionable)
    strict["incident"] = {
        **cast(dict[str, Any], actionable["incident"]),
        "median": 50.0,
    }
    assert (
        _has_strict_directional_metric_shift({"paired_metric_summary": strict}) is True
    )
    assert (
        _has_strict_directional_metric_shift({"paired_metric_summary": one_crossing})
        is False
    )
    opposite_median = dict(actionable)
    opposite_median["incident"] = {
        **cast(dict[str, Any], actionable["incident"]),
        "median": 10.0,
    }
    assert (
        _has_strict_directional_metric_shift({"paired_metric_summary": opposite_median})
        is False
    )


def test_fault_type_candidate_summary_is_bounded_and_sanitized() -> None:
    public_fault_types = [f"public-type-{index}" for index in range(20)]
    raw_candidates: list[object] = ["not-an-object"]
    raw_candidates.append(
        {
            "fault_type": "not-public",
            "entity_id": "entity",
            "metric_id": "metric",
            "resource_family": "family",
            "score": 1.0,
            "requires_paired_exact": True,
        }
    )
    raw_candidates.extend(
        {
            "fault_type": fault_type,
            "entity_id": f"entity-{index}",
            "metric_id": f"metric-{index}",
            "resource_family": "family",
            "score": index / 20,
            "requires_paired_exact": True,
            "hidden_label": "must-not-enter-context",
            "reference": {
                "min": 1.0,
                "median": 2.0,
                "max": 3.0,
                "mean": 2.0,
                "example_source_records": [index],
            },
            "incident": {
                "min": 4.0,
                "median": 5.0,
                "max": 6.0,
                "mean": 5.0,
                "example_source_records": [index + 20],
            },
        }
        for index, fault_type in enumerate(public_fault_types)
    )
    raw_result = {
        "status": "success",
        "public_fault_types": public_fault_types,
        "fault_type_candidates": raw_candidates,
    }

    summary = RootWalkRuntime._evidence_value(  # pyright: ignore[reportPrivateUsage]
        raw_result
    )

    candidates = summary["fault_type_candidates"]
    assert len(candidates) == 20
    assert [item["fault_type"] for item in candidates] == public_fault_types
    assert all(
        set(item)
        == {
            "fault_type",
            "entity_id",
            "metric_id",
            "resource_family",
            "score",
            "reference",
            "incident",
            "requires_paired_exact",
        }
        for item in candidates
    )
    serialized = json.dumps(candidates, sort_keys=True)
    assert "hidden_label" not in serialized
    assert "example_source_records" not in serialized
    assert '"mean"' not in serialized
    assert cast(dict[str, Any], raw_candidates[2])["hidden_label"] == (
        "must-not-enter-context"
    )


def test_supported_hypothesis_must_use_persisted_public_fault_type(
    tmp_path: Path,
) -> None:
    evidence_id = f"{RUN_ID}:evidence:decision:{RUN_ID}:0002:0"
    log_evidence_id = f"{RUN_ID}:evidence:decision:{RUN_ID}:0003:0"
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0005:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "load the public vocabulary",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 3},
                }
            ],
        },
        {
            "decision_summary": "collect exact evidence",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "incident",
                    },
                }
            ],
        },
        {
            "decision_summary": "corroborate with logs",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_log_samples",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "limit": 1,
                    },
                }
            ],
        },
        {
            "decision_summary": "use a statement outside the public vocabulary",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "invented private label",
                    "status": "supported",
                    "supporting_evidence_refs": [evidence_id, log_evidence_id],
                }
            ],
        },
        {
            "decision_summary": "use the exact public fault type",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "synthetic public fault type",
                    "status": "supported",
                    "supporting_evidence_refs": [evidence_id, log_evidence_id],
                }
            ],
        },
        {
            "decision_summary": "finish",
            "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model)

    assert result.status == "completed_partial"
    assert "must exactly match one public fault type" in str(model.inputs[4])
    assert "synthetic public fault type" in str(model.inputs[4])
    hypotheses = [cast_dict(item.payload) for item in store.records("hypothesis")]
    assert [item["statement"] for item in hypotheses] == ["synthetic public fault type"]
    store.close()


@pytest.mark.parametrize(
    ("statement", "public_fault_types", "expected_statement", "normalized"),
    [
        pytest.param(
            "synthetic public fault type",
            ["synthetic public fault type"],
            "synthetic public fault type",
            False,
            id="exact",
        ),
        pytest.param(
            "Evidence supports  SYNTHETIC\nPUBLIC FAULT TYPE on this node.",
            ["synthetic public fault type"],
            "synthetic public fault type",
            True,
            id="unique-prose",
        ),
        pytest.param(
            "Both synthetic public fault type and secondary public fault type fit.",
            ["synthetic public fault type", "secondary public fault type"],
            None,
            False,
            id="multiple",
        ),
        pytest.param(
            "synthetic public fault types",
            ["synthetic public fault type"],
            None,
            False,
            id="none",
        ),
    ],
)
def test_supported_hypothesis_normalizes_one_complete_public_fault_type(
    tmp_path: Path,
    statement: str,
    public_fault_types: list[str],
    expected_statement: str | None,
    normalized: bool,
) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    overview_result = environment.summarize_anomalies(limit=1)
    overview = Observation.model_validate_json(
        json.dumps(
            {
                "observation_id": "seed:overview",
                "tool": "summarize_anomalies",
                "arguments": {"limit": 1},
                "status": overview_result["status"],
                "coverage": overview_result["coverage"],
                "artifact_refs": [],
                "attempt_id": "attempt:overview",
                "summary": json.dumps({"public_fault_types": public_fault_types}),
            }
        )
    )
    store = RunStore(tmp_path / "run.sqlite3")
    store.append("observation", overview.observation_id, overview)
    log_evidence_id = seed_tool_evidence(store, environment, "query_log_samples")
    artifacts = ArtifactStore(tmp_path / "artifact-store")
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "collect exact evidence",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "incident",
                    },
                }
            ],
        },
        {
            "decision_summary": "record supported hypothesis",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": statement,
                    "status": "supported",
                    "supporting_evidence_refs": [EVIDENCE_ID, log_evidence_id],
                }
            ],
        },
        {
            "decision_summary": "finish",
            "operations": [
                {"type": "finalize", "hypothesis_ids": [HYPOTHESIS_ID]}
                if expected_statement is not None
                else {"type": "finalize", "abstain_reason": "not uniquely public"}
            ],
        },
    ]
    model = CapturingModel(records)

    result = asyncio.run(
        RootWalkRuntime(environment, model, store, artifacts).run(
            RUN_ID, tmp_path / "bundle"
        )
    )

    assert result.status == "completed_partial"
    decision_record = store.get(f"decision:{RUN_ID}:0002")
    assert decision_record is not None
    decision = cast_dict(decision_record.payload)
    decision_operation = cast_dict(cast(list[object], decision["operations"])[0])
    assert decision_operation["statement"] == statement
    response_record = store.get(f"model_response:decision:{RUN_ID}:0002")
    assert response_record is not None
    response = cast_dict(response_record.payload)
    raw_response = cast_dict(
        json.loads(artifacts.read_bytes(cast(str, response["content_hash"])))
    )
    raw_operation = cast_dict(cast(list[object], raw_response["operations"])[0])
    assert raw_operation["statement"] == statement
    events = [
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if cast_dict(item.payload).get("event_type") == "public_vocabulary_normalized"
    ]
    if expected_statement is None:
        assert not store.records("hypothesis")
        assert not events
        assert "must exactly match one public fault type" in str(model.inputs[2])
        assert result.report["budget"]["validator_retries_used"] == 1
    else:
        hypothesis = cast_dict(store.records("hypothesis")[0].payload)
        assert hypothesis["statement"] == expected_statement
        assert result.report["findings"][0]["statement"] == expected_statement
        assert result.report["budget"]["validator_retries_used"] == 0
        if normalized:
            assert events == [
                {
                    "event_type": "public_vocabulary_normalized",
                    "original_statement_hash": hashlib.sha256(
                        statement.encode()
                    ).hexdigest(),
                    "selected_public_fault_type": expected_statement,
                }
            ]
        else:
            assert not events
    store.close()


def test_finalize_rechecks_public_fault_type_added_after_supported_hypothesis(
    tmp_path: Path,
) -> None:
    evidence_id = f"{RUN_ID}:evidence:decision:{RUN_ID}:0001:0"
    log_evidence_id = f"{RUN_ID}:evidence:decision:{RUN_ID}:0002:0"
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0003:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "collect exact evidence before loading vocabulary",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "incident",
                    },
                }
            ],
        },
        {
            "decision_summary": "corroborate with logs",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_log_samples",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "limit": 1,
                    },
                }
            ],
        },
        {
            "decision_summary": "create supported hypothesis before vocabulary exists",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "invented private label",
                    "status": "supported",
                    "supporting_evidence_refs": [evidence_id, log_evidence_id],
                }
            ],
        },
        {
            "decision_summary": "load public vocabulary later",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 1},
                }
            ],
        },
        {
            "decision_summary": "try to finalize the earlier hypothesis",
            "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
        },
        {
            "decision_summary": "abstain after vocabulary validation",
            "operations": [
                {"type": "finalize", "abstain_reason": "statement is not public"}
            ],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model)

    assert result.status == "completed_partial"
    assert not result.report["findings"]
    assert "must exactly match one public fault type" in str(model.inputs[5])
    assert "synthetic public fault type" in str(model.inputs[5])
    assert result.report["budget"]["validator_retries_used"] == 1
    store.close()


def test_validation_feedback_resets_existing_no_progress_before_retry(
    tmp_path: Path,
) -> None:
    evidence_id = f"{RUN_ID}:evidence:decision:{RUN_ID}:0001:0"
    log_evidence_id = f"{RUN_ID}:evidence:decision:{RUN_ID}:0002:0"
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0007:0"
    overview = {
        "type": "query",
        "tool": "summarize_anomalies",
        "arguments": {"limit": 1},
    }
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "collect exact evidence",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "incident",
                    },
                }
            ],
        },
        {
            "decision_summary": "corroborate with logs",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_log_samples",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "limit": 1,
                    },
                }
            ],
        },
        {"decision_summary": "load public vocabulary", "operations": [overview]},
        {"decision_summary": "repeat overview once", "operations": [overview]},
        {"decision_summary": "repeat overview twice", "operations": [overview]},
        {
            "decision_summary": "submit a non-public supported statement",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "invented private label",
                    "status": "supported",
                    "supporting_evidence_refs": [evidence_id, log_evidence_id],
                }
            ],
        },
        {
            "decision_summary": "correct to the exact public statement",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "synthetic public fault type",
                    "status": "supported",
                    "supporting_evidence_refs": [evidence_id, log_evidence_id],
                }
            ],
        },
        {
            "decision_summary": "finish",
            "operations": [{"type": "finalize", "hypothesis_ids": [hypothesis_id]}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model)

    assert result.status == "completed_partial"
    assert len(model.inputs) == 8
    assert "must exactly match one public fault type" in str(model.inputs[6])
    assert result.report["findings"][0]["statement"] == ("synthetic public fault type")
    assert result.report["budget"]["validator_retries_used"] == 1
    assert result.report["budget"]["no_progress_rounds"] == 0
    store.close()


@pytest.mark.parametrize("status", ["open", "inconclusive", "refuted"])
def test_non_supported_hypothesis_is_not_restricted_by_public_fault_types(
    tmp_path: Path,
    status: str,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "load the public vocabulary",
            "operations": [
                {
                    "type": "query",
                    "tool": "summarize_anomalies",
                    "arguments": {"limit": 1},
                }
            ],
        },
        {
            "decision_summary": "record an uncommitted candidate",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "free-form candidate under investigation",
                    "status": status,
                }
            ],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]

    store, result = run_runtime(tmp_path, records)

    assert result.status == "completed_partial"
    hypothesis = cast_dict(store.records("hypothesis")[0].payload)
    assert hypothesis["statement"] == "free-form candidate under investigation"
    assert hypothesis["status"] == status
    store.close()


def test_runtime_generated_artifact_is_readable_by_hash(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    artifacts = ArtifactStore(tmp_path / "artifact-store")
    content_hash = artifacts.put_text('{"source":"runtime"}\n')
    store.append(
        "artifact",
        f"artifact:{content_hash}",
        {"content_hash": content_hash, "storage": "embedded"},
    )
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "read runtime artifact",
            "operations": [
                {
                    "type": "query",
                    "tool": "read_artifact_slice",
                    "arguments": {"artifact_id": content_hash},
                }
            ],
        },
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    model = CapturingModel(records)
    runtime = RootWalkRuntime(BankEnvironment(PUBLIC_INPUT), model, store, artifacts)

    result = asyncio.run(runtime.run("runtime-artifact", tmp_path / "bundle"))

    assert result.status == "completed_partial"
    second_context = cast_dict(json.loads(cast(str, model.inputs[1][1]["content"])))
    assert not store.records("evidence")
    assert not result.report["gaps"]
    assert second_context["recent_interactions"][0]["tool"] == "read_artifact_slice"
    assert second_context["recent_interactions"][0]["result"]["content"] == (
        '{"source":"runtime"}\n'
    )
    payload = cast_dict(store.records("query_payload")[0].payload)
    query_result = cast_dict(json.loads(artifacts.read_bytes(payload["content_hash"])))
    assert query_result["content"] == '{"source":"runtime"}\n'
    store.close()


class ArtifactReadingModel:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.inputs: list[Sequence[Message]] = []

    async def decide(self, messages: Sequence[Message]) -> Decision:
        self.inputs.append(messages)
        if len(self.inputs) == 1:
            return Decision.model_validate(
                {
                    "decision_summary": "query one metric",
                    "operations": [
                        {
                            "type": "query",
                            "tool": "query_metric",
                            "arguments": {
                                "entity_id": "bank:entity:Tomcat01",
                                "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                            },
                        }
                    ],
                }
            )
        if len(self.inputs) == 2:
            context = cast_dict(json.loads(cast(str, messages[1]["content"])))
            artifact_id = context["evidence"][0]["artifact_refs"][0]["content_hash"]
            return Decision.model_validate(
                {
                    "decision_summary": "read the evidence payload",
                    "operations": [
                        {
                            "type": "query",
                            "tool": "read_artifact_slice",
                            "arguments": {"artifact_id": artifact_id},
                        }
                    ],
                }
            )
        return Decision.model_validate(
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            }
        )


def test_artifact_read_keeps_original_evidence_and_exposes_slice(
    tmp_path: Path,
) -> None:
    run_id = "read-evidence-artifact"
    model = ArtifactReadingModel(run_id)
    store = RunStore(tmp_path / "run.sqlite3")
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run(run_id, tmp_path / "bundle")
    )

    third_context = cast_dict(json.loads(cast(str, model.inputs[2][1]["content"])))
    artifact_read = next(
        item
        for item in third_context["recent_interactions"]
        if item["tool"] == "read_artifact_slice"
    )
    assert result.status == "completed_partial"
    assert len(store.records("evidence")) == 1
    assert third_context["evidence"][0]["evidence_id"].startswith(f"{run_id}:evidence:")
    assert (
        '"metric_id":"OSLinux-CPU_CPU_CPUCpuUtil"' in artifact_read["result"]["content"]
    )
    store.close()


@pytest.mark.parametrize("context_strategy", ["recent", "evidence"])
def test_navigation_queries_create_visible_observations_without_evidence_or_gaps(
    tmp_path: Path, context_strategy: str
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "inspect navigation catalogs",
            "operations": [
                {"type": "query", "tool": "list_entities", "arguments": {}},
                {"type": "query", "tool": "list_metrics", "arguments": {}},
            ],
        },
        {
            "decision_summary": "abstain",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]

    model = CapturingModel(records)
    store, result = run_runtime(
        tmp_path, records, model=model, context_strategy=context_strategy
    )

    assert result.status == "completed_partial"
    assert len(store.records("observation")) == 2
    assert not store.records("evidence")
    assert not result.report["gaps"]
    context = cast_dict(json.loads(cast(str, model.inputs[1][1]["content"])))
    results = {item["tool"]: item["result"] for item in context["recent_interactions"]}
    assert results["list_entities"]["entities"]
    assert "OSLinux-CPU_CPU_CPUCpuUtil" in results["list_metrics"]["metric_ids"]
    store.close()


def test_frontier_reflection_persists_assessments_without_semantic_rewrite(
    tmp_path: Path,
) -> None:
    observation_1 = f"{RUN_ID}:observation:decision:{RUN_ID}:0001:0"
    evidence_1 = f"{RUN_ID}:evidence:decision:{RUN_ID}:0001:0"
    observation_2 = f"{RUN_ID}:observation:decision:{RUN_ID}:0002:0"
    evidence_2 = f"{RUN_ID}:evidence:decision:{RUN_ID}:0002:0"
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0002:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "inspect incident CPU",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "incident",
                    },
                }
            ],
        },
        {
            "decision_summary": "propose bursty pressure and compare the baseline",
            "reflection": {
                "based_on_observation_ids": [observation_1],
                "evidence_update": [
                    {
                        "evidence_id": evidence_1,
                        "hypothesis_id": "H-local",
                        "impact": "supports",
                        "reason_code": "incident_signal",
                        "explanation": "the incident signal seeds the proposal",
                    }
                ],
                "new_hypothesis": {
                    "client_id": "H-local",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "bursty CPU throttling",
                    "derived_from": [evidence_1],
                    "open_questions": ["does the reference show the same burst"],
                },
                "new_hypothesis_is_current_best": False,
                "current_best_hypothesis_id": "H-local",
                "hypothesis_update": [
                    {
                        "hypothesis_id": "H-local",
                        "status": "open",
                        "open_questions": ["does the reference show the same burst"],
                    }
                ],
                "next_check": {
                    "operation_tools": ["query_metric"],
                    "distinguishes_hypothesis_ids": ["H-local"],
                    "purpose": "compare the same signal with the reference window",
                    "expected_outcomes": [
                        "incident-only burst supports the hypothesis",
                        "matched baseline weakens the hypothesis",
                    ],
                },
                "can_finalize": False,
            },
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "reference",
                    },
                }
            ],
        },
        {
            "decision_summary": "support the unchanged semantic hypothesis",
            "reflection": {
                "based_on_observation_ids": [observation_2],
                "evidence_update": [
                    {
                        "evidence_id": evidence_1,
                        "hypothesis_id": hypothesis_id,
                        "impact": "supports",
                        "reason_code": "incident_excursion",
                        "explanation": (
                            "the incident excursion supports bursty pressure"
                        ),
                    },
                    {
                        "evidence_id": evidence_2,
                        "hypothesis_id": hypothesis_id,
                        "impact": "supports",
                        "reason_code": "paired_baseline",
                        "explanation": "the paired baseline preserves the contrast",
                    },
                ],
                "hypothesis_update": [
                    {
                        "hypothesis_id": hypothesis_id,
                        "status": "inconclusive",
                        "open_questions": [],
                    }
                ],
                "current_best_hypothesis_id": None,
                "next_check": None,
                "can_finalize": True,
            },
            "operations": [
                {
                    "type": "finalize",
                    "hypothesis_ids": [],
                    "path_ids": [],
                    "unresolved_questions": ["typed profile evidence is unavailable"],
                    "abstain_reason": "registered typed evidence is unavailable",
                }
            ],
        },
    ]

    store, result = run_runtime(
        tmp_path,
        records,
        context_strategy="frontier_reflection",
        max_model_calls=3,
    )

    assert result.report["outcome_type"] == "abstention"
    assert result.report["current_best_hypothesis_id"] == hypothesis_id
    hypothesis = result.report["candidates"][0]["hypothesis"]
    assert hypothesis["statement"] == "bursty CPU throttling"
    assert hypothesis["supporting_evidence_refs"] == [
        evidence_1,
        evidence_2,
    ]
    assert len(result.report["evidence_assessments"]) == 3
    assert {
        item["hypothesis_id"] for item in result.report["evidence_assessments"]
    } == {hypothesis_id}
    assert len(result.report["reflections"]) == 2
    assert "benchmark_recommendation" not in result.report
    assert (tmp_path / "bundle/evidence_assessments.jsonl").is_file()
    assert (tmp_path / "bundle/reflections.jsonl").is_file()
    store.close()


def test_frontier_atomically_supports_and_finalizes_new_hypothesis(
    tmp_path: Path,
) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    observation_id = "seed:observation:typed-profile"
    evidence_id = "seed:evidence:typed-profile"
    observation = Observation(
        observation_id=observation_id,
        tool="query_metric_profile",
        arguments={
            "entity_id": "bank:entity:Tomcat01",
            "metric_ids": ["cpu", "throttle"],
            "period": "incident",
        },
        status="success",
        coverage=Coverage(
            requested_range=environment.task.incident_range,
            observed_range=environment.task.incident_range,
            completeness="complete",
            collection_health="healthy",
            basis="synthetic typed profile",
        ),
        artifact_refs=[],
        attempt_id="attempt:typed-profile",
        summary="typed CPU profile",
    )
    evidence_item = Evidence(
        evidence_id=evidence_id,
        observation_refs=[observation_id],
        subject_ids=["bank:entity:Tomcat01"],
        predicate="query_metric_profile.result",
        value={
            "metric_profiles": [
                {"metric_id": "cpu", "strong_incident_change": True},
                {"metric_id": "throttle", "strong_incident_change": True},
            ],
            "peer_comparison": [{"peer_count": 3, "robust_z": 4.0, "percentile": 1.0}],
            "finding_gate_summary": {"single_component_ready": True},
        },
        observed_interval=environment.task.incident_range,
        onset_interval=environment.task.incident_range,
        time_uncertainty="synthetic fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="metric-profile",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="typed-profile-source",
        dataset_snapshot_id=environment.dataset_id,
        evidence_kind="metric_profile",
        mechanism_dimension="resource_profile",
        independence_group="typed-profile-source",
        semantic_quality="typed",
    )
    store.append("observation", observation_id, observation)
    store.append("evidence", evidence_id, evidence_item)
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "promote the typed profile finding",
            "reflection": {
                "based_on_observation_ids": [observation_id],
                "evidence_update": [
                    {
                        "evidence_id": evidence_id,
                        "hypothesis_id": "H-local",
                        "impact": "supports",
                        "reason_code": "typed_profile",
                        "explanation": "two aligned metrics isolate the component",
                    }
                ],
                "new_hypothesis": {
                    "client_id": "H-local",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "bursty CPU throttling",
                    "derived_from": [evidence_id],
                    "open_questions": [],
                },
                "new_hypothesis_is_current_best": True,
                "hypothesis_update": [
                    {
                        "hypothesis_id": "H-local",
                        "status": "supported",
                        "open_questions": [],
                    }
                ],
                "next_check": None,
                "can_finalize": True,
            },
            "operations": [
                {
                    "type": "finalize",
                    "hypothesis_ids": ["H-local"],
                    "path_ids": [],
                    "unresolved_questions": [],
                    "abstain_reason": None,
                }
            ],
        }
    ]
    missing_prospective_support = cast(
        dict[str, Any], json.loads(json.dumps(records[0]))
    )
    cast(dict[str, Any], missing_prospective_support["reflection"])[
        "hypothesis_update"
    ] = []
    records.insert(0, missing_prospective_support)
    runtime = RootWalkRuntime(
        environment,
        RecordedModelAdapter(records),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(
            context_strategy="frontier_reflection",
            max_model_calls=2,
            max_validator_retries=1,
        ),
    )

    result = asyncio.run(runtime.run(RUN_ID, tmp_path / "bundle"))

    expected_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0002:0"
    assert result.report["outcome_type"] == "finding"
    assert result.report["hypothesis_ids"] == [expected_id]
    assert result.report["findings"][0]["subject"] == "bank:entity:Tomcat01"
    assert result.report["findings"][0]["statement"] == "bursty CPU throttling"
    assert result.report["findings"][0]["supporting_evidence_refs"] == [evidence_id]
    persisted = cast_dict(store.records("decision")[0].payload)
    assert persisted["operations"][0]["hypothesis_ids"] == ["H-local"]
    store.close()


def test_frontier_finalizes_source_backed_occurrence_only_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    environment.task = environment.task.model_copy(
        update={"requested_outputs": ["root cause occurrence datetime"]}
    )
    store = RunStore(tmp_path / "run.sqlite3")
    impact_observation = Observation(
        observation_id="occurrence-impact-observation",
        tool="query_impact_profile",
        arguments={},
        status="no_data",
        coverage=Coverage(
            requested_range=environment.task.allowed_query_range,
            completeness="unknown",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="occurrence-impact-attempt",
        extraction_gap="fixture has no impact source",
    )
    store.append("observation", impact_observation.observation_id, impact_observation)
    profile_observation_ids = [
        "occurrence-observation-1",
        "occurrence-observation-2",
    ]
    observation_ids = [
        impact_observation.observation_id,
        *profile_observation_ids,
    ]
    evidence_ids = ["occurrence-evidence-1", "occurrence-evidence-2"]
    for index, (observation_id, evidence_id) in enumerate(
        zip(profile_observation_ids, evidence_ids, strict=True)
    ):
        observation = Observation(
            observation_id=observation_id,
            tool="query_metric_profile",
            arguments={"entity_id": "bank:entity:Tomcat01", "metric_ids": ["cpu"]},
            status="success",
            coverage=Coverage(
                requested_range=environment.task.allowed_query_range,
                completeness="complete",
                basis="fixture",
            ),
            artifact_refs=[],
            attempt_id=f"occurrence-attempt-{index}",
        )
        evidence_item = Evidence(
            evidence_id=evidence_id,
            observation_refs=[observation_id],
            subject_ids=["bank:entity:Tomcat01"],
            predicate="source_backed_onset",
            value={"candidate": index},
            observed_interval=environment.task.incident_range,
            onset_interval=environment.task.incident_range,
            time_uncertainty="fixture",
            artifact_refs=[],
            extractor=Extractor(
                kind="deterministic",
                name="occurrence-fixture",
                version="1",
                parameters_hash="fixture",
            ),
            source_fingerprint=f"occurrence-source-{index}",
            dataset_snapshot_id=environment.dataset_id,
            evidence_kind="metric_profile",
            semantic_quality="typed",
        )
        store.append("observation", observation_id, observation)
        store.append("evidence", evidence_id, evidence_item)
    competing_observation = Observation(
        observation_id="occurrence-competing-observation",
        tool="query_metric_profile",
        arguments={"entity_id": "bank:entity:Tomcat02", "metric_ids": ["cpu"]},
        status="success",
        coverage=Coverage(
            requested_range=environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="occurrence-competing-attempt",
    )
    competing_evidence = Evidence(
        evidence_id="occurrence-competing-evidence",
        observation_refs=[competing_observation.observation_id],
        subject_ids=["bank:entity:Tomcat02"],
        predicate="source_backed_onset",
        value={"finding_gate_summary": {"fault_type_ready": True}},
        observed_interval=environment.task.incident_range,
        onset_interval=TimeRange(
            start=environment.task.incident_range.start + timedelta(minutes=1),
            end=environment.task.incident_range.end,
        ),
        time_uncertainty="fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="occurrence-fixture",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="occurrence-competing-source",
        dataset_snapshot_id=environment.dataset_id,
        evidence_kind="metric_profile",
        semantic_quality="typed",
    )
    store.append(
        "observation", competing_observation.observation_id, competing_observation
    )
    store.append("evidence", competing_evidence.evidence_id, competing_evidence)
    observation_ids.append(competing_observation.observation_id)
    observation_ids.sort()
    decision = {
        "decision_summary": "recommend the corroborated source-backed onset only",
        "reflection": {
            "based_on_observation_ids": observation_ids,
            "evidence_update": [],
            "anomaly_disposition_update": [],
            "hypothesis_update": [],
            "current_best_hypothesis_id": None,
            "clear_current_best": False,
            "new_hypothesis": None,
            "new_hypothesis_is_current_best": False,
            "next_check": None,
            "can_finalize": True,
        },
        "operations": [
            {
                "type": "finalize",
                "occurrence_evidence_refs": evidence_ids,
                "unresolved_questions": ["component and reason remain unresolved"],
            }
        ],
    }
    runtime = RootWalkRuntime(
        environment,
        RecordedModelAdapter([decision]),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(context_strategy="frontier_reflection", max_model_calls=1),
    )
    monkeypatch.setattr(
        runtime,
        "_frontier_pending_challenger_page",
        lambda: [
            {
                "subject": "bank:entity:Tomcat02",
                "fault_type": "high CPU usage",
                "metric_ids": ["cpu"],
            }
        ],
    )

    result = asyncio.run(runtime.run(RUN_ID, tmp_path / "bundle"))

    assert result.status == "completed_partial"
    assert result.report["outcome_type"] == "occurrence_recommendation"
    assert result.report["findings"] == []
    assert result.report["occurrence_evidence_refs"] == evidence_ids
    recommendation = cast_dict(result.report["occurrence_recommendation"])
    assert recommendation["policy"] == "source-backed-occurrence-v1"
    assert recommendation["subject_id"] == "bank:entity:Tomcat01"
    assert recommendation["evidence_refs"] == evidence_ids
    events = [
        cast_dict(record.payload)
        for record in store.records("investigation_event")
        if cast_dict(record.payload).get("event_type")
        == "occurrence_recommendation_selected"
    ]
    assert len(events) == 1
    store.close()


def test_frontier_metric_finding_gate_requires_typed_shape_onset_and_peer(
    tmp_path: Path,
) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")
    evidence_item = Evidence(
        evidence_id="typed-profile",
        observation_refs=["profile-observation"],
        subject_ids=["bank:entity:Tomcat01"],
        predicate="query_metric_profile.result",
        value={
            "metric_profiles": [
                {"metric_id": "cpu", "strong_incident_change": True},
                {"metric_id": "throttle", "strong_incident_change": True},
            ],
            "peer_comparison": [
                {
                    "peer_count": 3,
                    "robust_z": 4.0,
                    "percentile": 1.0,
                }
            ],
            "finding_gate_summary": {"single_component_ready": True},
        },
        observed_interval=environment.task.incident_range,
        onset_interval=environment.task.incident_range.model_copy(
            update={
                "end": environment.task.incident_range.start
                + (
                    environment.task.incident_range.end
                    - environment.task.incident_range.start
                )
                / 2
            }
        ),
        time_uncertainty="fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="metric-profile",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="profile-source",
        dataset_snapshot_id=environment.dataset_id,
        evidence_kind="metric_profile",
        mechanism_dimension="resource_profile",
        independence_group="profile-source",
        semantic_quality="typed",
    )
    hypothesis_item = Hypothesis(
        hypothesis_id="hypothesis-profile",
        subject="bank:entity:Tomcat01",
        statement="bursty CPU throttling",
        status="open",
    )
    assessment = EvidenceAssessment(
        assessment_id="assessment-profile",
        evidence_id=evidence_item.evidence_id,
        hypothesis_id=hypothesis_item.hypothesis_id,
        impact="supports",
        reason_code="typed_profile",
        explanation="two aligned metrics and isolated peers",
        observation_digest="a" * 64,
    )
    store.append("evidence", evidence_item.evidence_id, evidence_item)
    store.append(
        "hypothesis", f"hypothesis:{hypothesis_item.hypothesis_id}", hypothesis_item
    )
    store.append("evidence_assessment", assessment.assessment_id, assessment)
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    errors = runtime._frontier_finding_errors(  # pyright: ignore[reportPrivateUsage]
        hypothesis_item.hypothesis_id,
        {(assessment.evidence_id, assessment.hypothesis_id): assessment},
    )

    assert errors == []
    store.close()


def test_frontier_prefers_explicit_state_transition_over_higher_numeric_score(
    tmp_path: Path,
) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    store = RunStore(tmp_path / "run.sqlite3")

    def profile(
        evidence_id: str,
        subject: str,
        signature: str,
        score: float,
    ) -> Evidence:
        return Evidence(
            evidence_id=evidence_id,
            observation_refs=[f"observation:{evidence_id}"],
            subject_ids=[subject],
            predicate="query_metric_profile.result",
            value={
                "metric_profiles": [
                    {"metric_id": "signal", "strong_incident_change": True}
                ],
                "peer_comparison": [],
                "finding_gate_summary": {
                    "requested_fault_type": "synthetic public fault type",
                    "fault_type_ready": True,
                    "single_component_ready": True,
                    "mechanism_signature": signature,
                    "root_cause_score": score,
                },
            },
            observed_interval=environment.task.incident_range,
            onset_interval=environment.task.incident_range,
            time_uncertainty="fixture",
            artifact_refs=[],
            extractor=Extractor(
                kind="deterministic",
                name="metric-profile",
                version="1",
                parameters_hash="fixture",
            ),
            source_fingerprint=f"source:{evidence_id}",
            dataset_snapshot_id=environment.dataset_id,
            evidence_kind="metric_profile",
            mechanism_dimension="resource_profile",
            independence_group=f"source:{evidence_id}",
            semantic_quality="typed",
        )

    saturation = profile(
        "saturation-profile",
        "bank:entity:Tomcat01",
        "sustained_high_saturation",
        0.97,
    )
    transition = profile(
        "transition-profile",
        "bank:entity:Tomcat02",
        "state_transition",
        0.9,
    )
    hypothesis = Hypothesis(
        hypothesis_id="saturation-hypothesis",
        subject="bank:entity:Tomcat01",
        statement="synthetic public fault type",
        status="open",
    )
    assessment = EvidenceAssessment(
        assessment_id="saturation-assessment",
        evidence_id=saturation.evidence_id,
        hypothesis_id=hypothesis.hypothesis_id,
        impact="supports",
        reason_code="typed_profile",
        explanation="higher numeric score",
        observation_digest="a" * 64,
    )
    overview = Observation(
        observation_id="fault-vocabulary-overview",
        tool="summarize_anomalies",
        arguments={"limit": 12},
        status="success",
        coverage=Coverage(
            requested_range=environment.task.allowed_query_range,
            completeness="unknown",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="fault-vocabulary-overview-attempt",
        summary=json.dumps({"public_fault_types": ["synthetic public fault type"]}),
    )
    store.append("observation", overview.observation_id, overview)
    for evidence in (saturation, transition):
        store.append("evidence", evidence.evidence_id, evidence)
    store.append("hypothesis", hypothesis.hypothesis_id, hypothesis)
    store.append("evidence_assessment", assessment.assessment_id, assessment)
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]
    errors = runtime._frontier_finding_errors(  # pyright: ignore[reportPrivateUsage]
        hypothesis.hypothesis_id,
        {(assessment.evidence_id, assessment.hypothesis_id): assessment},
    )

    assert len(errors) == 1
    assert "more causally specific eligible typed profiles" in errors[0]
    assert "Tomcat02" in errors[0]
    store.close()


def test_frontier_next_check_accepts_unique_tool_kind_sequence(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    item = Observation(
        observation_id="observation-1",
        tool="query_metric",
        arguments={"entity_id": "bank:entity:Tomcat01", "metric_id": "cpu"},
        status="success",
        coverage=Coverage(
            requested_range=runtime.environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt-1",
        summary="new evidence",
    )
    runtime._observations[item.observation_id] = item  # pyright: ignore[reportPrivateUsage]
    decision = Decision.model_validate(
        {
            "decision_summary": "batch two checks of one tool kind",
            "reflection": {
                "based_on_observation_ids": [item.observation_id],
                "evidence_update": [],
                "new_hypothesis": None,
                "new_hypothesis_is_current_best": False,
                "current_best_hypothesis_id": None,
                "hypothesis_update": [],
                "anomaly_disposition_update": [],
                "next_check": {
                    "operation_tools": ["query_metric"],
                    "distinguishes_hypothesis_ids": [],
                    "purpose": "compare two independent metrics",
                    "expected_outcomes": ["either metric may distinguish the event"],
                },
                "can_finalize": False,
            },
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": metric_id,
                        "period": "incident",
                    },
                }
                for metric_id in ("cpu", "memory")
            ],
        }
    )

    assert (
        runtime._validate_frontier_decision(decision)  # pyright: ignore[reportPrivateUsage]
        is None
    )
    store.close()


def test_frontier_requires_explicit_leader_selection_for_supported_proposal(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    item = Observation(
        observation_id="observation-1",
        tool="query_metric",
        arguments={"entity_id": "bank:entity:Tomcat01", "metric_id": "cpu"},
        status="success",
        coverage=Coverage(
            requested_range=runtime.environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt-1",
        summary="typed CPU evidence",
    )
    item_evidence = Evidence(
        evidence_id="evidence-1",
        observation_refs=[item.observation_id],
        subject_ids=["bank:entity:Tomcat01"],
        predicate="cpu_profile_ready",
        value={
            "finding_gate_summary": {
                "requested_fault_type": "CPU pressure",
                "fault_type_ready": True,
                "boundary_aligned": False,
                "mechanism_signature": "sustained_high_saturation",
                "root_cause_score": 1.0,
            }
        },
        observed_interval=runtime.environment.task.incident_range,
        time_uncertainty="fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="fixture",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="fixture-source",
        dataset_snapshot_id=runtime.environment.dataset_id,
        evidence_kind="metric_profile",
        mechanism_dimension="resource_profile",
        semantic_quality="typed",
    )
    runtime._observations[item.observation_id] = item  # pyright: ignore[reportPrivateUsage]
    runtime._evidence[item_evidence.evidence_id] = item_evidence  # pyright: ignore[reportPrivateUsage]

    def decision(select_leader: bool) -> Decision:
        return Decision.model_validate(
            {
                "decision_summary": "create one explicitly supported candidate",
                "reflection": {
                    "based_on_observation_ids": [item.observation_id],
                    "evidence_update": [
                        {
                            "evidence_id": item_evidence.evidence_id,
                            "hypothesis_id": "new_hypothesis",
                            "impact": "supports",
                            "reason_code": "typed_profile",
                            "explanation": "typed evidence supports the proposal",
                        }
                    ],
                    "new_hypothesis": {
                        "subject": "bank:entity:Tomcat01",
                        "statement": "CPU pressure",
                        "derived_from": [item_evidence.evidence_id],
                        "open_questions": ["confirm propagation"],
                    },
                    "new_hypothesis_is_current_best": select_leader,
                    "current_best_hypothesis_id": None,
                    "hypothesis_update": [],
                    "anomaly_disposition_update": [],
                    "next_check": {
                        "operation_tools": ["query_metric"],
                        "distinguishes_hypothesis_ids": ["new_hypothesis"],
                        "purpose": "confirm propagation",
                        "expected_outcomes": ["the next metric supports or weakens it"],
                    },
                    "can_finalize": False,
                },
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_metric",
                        "arguments": {
                            "entity_id": "bank:entity:Tomcat01",
                            "metric_id": "memory",
                            "period": "incident",
                        },
                    }
                ],
            }
        )

    error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        decision(False)
    )
    assert error is not None
    assert "leader selection is required" in error
    selected_without_direct_support = decision(True)
    assert selected_without_direct_support.reflection is not None
    selected_without_direct_support = selected_without_direct_support.model_copy(
        update={
            "reflection": selected_without_direct_support.reflection.model_copy(
                update={"evidence_update": []}
            )
        }
    )
    grounding_error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        selected_without_direct_support
    )
    assert grounding_error is not None
    assert "same-subject typed" in grounding_error
    existing = Hypothesis(
        hypothesis_id="existing-leader",
        subject="bank:entity:Tomcat02",
        statement="existing provisional leader",
        status="open",
    )
    runtime._hypotheses[existing.hypothesis_id] = existing  # pyright: ignore[reportPrivateUsage]
    runtime._current_best_hypothesis_id = existing.hypothesis_id  # pyright: ignore[reportPrivateUsage]
    replacement_error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        selected_without_direct_support
    )
    assert replacement_error is not None
    assert "same-subject typed" in replacement_error
    stronger_error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        decision(False)
    )
    assert stronger_error is not None
    assert "strictly stronger direct typed mechanism" in stronger_error
    assert "does not persist the proposal or assessment" in stronger_error
    runtime._hypotheses.clear()  # pyright: ignore[reportPrivateUsage]
    runtime._current_best_hypothesis_id = None  # pyright: ignore[reportPrivateUsage]
    missing_proposal = decision(False)
    assert missing_proposal.reflection is not None
    missing_proposal = missing_proposal.model_copy(
        update={
            "reflection": missing_proposal.reflection.model_copy(
                update={
                    "evidence_update": [],
                    "new_hypothesis": None,
                    "new_hypothesis_is_current_best": False,
                    "next_check": missing_proposal.reflection.next_check.model_copy(
                        update={"distinguishes_hypothesis_ids": []}
                    )
                    if missing_proposal.reflection.next_check is not None
                    else None,
                }
            )
        }
    )
    creation_error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        missing_proposal
    )
    assert creation_error is not None
    assert "hypothesis creation is required" in creation_error
    runtime._hypotheses[existing.hypothesis_id] = existing  # pyright: ignore[reportPrivateUsage]
    runtime._current_best_hypothesis_id = existing.hypothesis_id  # pyright: ignore[reportPrivateUsage]
    upgrade_error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        missing_proposal
    )
    assert upgrade_error is not None
    assert "leader upgrade is required" in upgrade_error
    assert "atomic" in upgrade_error
    runtime._hypotheses.clear()  # pyright: ignore[reportPrivateUsage]
    runtime._current_best_hypothesis_id = None  # pyright: ignore[reportPrivateUsage]
    assert (
        runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
            decision(True)
        )
        is None
    )
    store.close()


def test_frontier_allows_closeout_reflection_without_novel_observation(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    item = Observation(
        observation_id="observation-1",
        tool="query_metric",
        arguments={"entity_id": "bank:entity:Tomcat01", "metric_id": "cpu"},
        status="success",
        coverage=Coverage(
            requested_range=runtime.environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt-1",
        summary="already reflected evidence",
    )
    runtime._observations[item.observation_id] = item  # pyright: ignore[reportPrivateUsage]
    runtime._reflected_observation_ids.add(item.observation_id)  # pyright: ignore[reportPrivateUsage]
    decision = Decision.model_validate(
        {
            "decision_summary": "close out from already reflected state",
            "reflection": {
                "based_on_observation_ids": [item.observation_id],
                "evidence_update": [],
                "new_hypothesis": None,
                "new_hypothesis_is_current_best": False,
                "current_best_hypothesis_id": None,
                "hypothesis_update": [],
                "anomaly_disposition_update": [],
                "next_check": None,
                "can_finalize": True,
            },
            "operations": [
                {
                    "type": "finalize",
                    "hypothesis_ids": [],
                    "path_ids": [],
                    "unresolved_questions": [],
                    "abstain_reason": "fixture closeout",
                }
            ],
        }
    )

    assert (
        runtime._validate_frontier_decision(decision)  # pyright: ignore[reportPrivateUsage]
        is None
    )
    store.close()


def test_frontier_allows_state_changing_advance_from_reflected_evidence(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    item = Observation(
        observation_id="observation-1",
        tool="query_metric",
        arguments={"entity_id": "bank:entity:Tomcat01", "metric_id": "cpu"},
        status="success",
        coverage=Coverage(
            requested_range=runtime.environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt-1",
        summary="already reflected evidence",
    )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-1",
        subject="bank:entity:Tomcat01",
        statement="CPU pressure",
        status="open",
        open_questions=["dispose the current anomaly page"],
    )
    runtime._observations[item.observation_id] = item  # pyright: ignore[reportPrivateUsage]
    runtime._reflected_observation_ids.add(item.observation_id)  # pyright: ignore[reportPrivateUsage]
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]

    def decision(open_questions: list[str]) -> Decision:
        return Decision.model_validate(
            {
                "decision_summary": "advance after a real state update",
                "reflection": {
                    "based_on_observation_ids": [item.observation_id],
                    "evidence_update": [],
                    "new_hypothesis": None,
                    "new_hypothesis_is_current_best": False,
                    "current_best_hypothesis_id": None,
                    "hypothesis_update": [
                        {
                            "hypothesis_id": hypothesis.hypothesis_id,
                            "status": "open",
                            "open_questions": open_questions,
                        }
                    ],
                    "anomaly_disposition_update": [],
                    "next_check": None,
                    "can_finalize": False,
                },
                "operations": [{"type": "advance_frontier"}],
            }
        )

    assert (
        runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
            decision([])
        )
        is None
    )
    error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        decision(hypothesis.open_questions)
    )
    assert error is not None
    assert "actual EvidenceAssessment" in error
    store.close()


def test_frontier_advance_rejects_incomplete_active_anomaly_page(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    item = Observation(
        observation_id="observation-1",
        tool="query_metric",
        arguments={"entity_id": "bank:entity:Tomcat01", "metric_id": "cpu"},
        status="success",
        coverage=Coverage(
            requested_range=runtime.environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt-1",
        summary="novel evidence",
    )
    anomaly = material_anomaly()
    runtime._observations[item.observation_id] = item  # pyright: ignore[reportPrivateUsage]
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]
    decision = Decision.model_validate(
        {
            "decision_summary": "do not page past an unresolved active anomaly",
            "reflection": {
                "based_on_observation_ids": [item.observation_id],
                "evidence_update": [],
                "new_hypothesis": None,
                "new_hypothesis_is_current_best": False,
                "current_best_hypothesis_id": None,
                "hypothesis_update": [],
                "anomaly_disposition_update": [
                    {
                        "anomaly_id": anomaly.anomaly_id,
                        "disposition": "unresolved",
                        "evidence_refs": [],
                        "explanation": "the mechanism is still unknown",
                    }
                ],
                "next_check": None,
                "can_finalize": False,
            },
            "operations": [{"type": "advance_frontier"}],
        }
    )

    error = runtime._validate_frontier_decision(decision)  # pyright: ignore[reportPrivateUsage]

    assert error is not None
    assert anomaly.anomaly_id in error
    assert "current active page" in error
    store.close()


def test_frontier_concurrent_noncausal_requires_audited_winner_support(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    item = Observation(
        observation_id="observation-1",
        tool="query_metric_profile",
        arguments={
            "entity_id": "bank:entity:Tomcat01",
            "metric_ids": ["cpu"],
            "fault_type": "CPU fault",
        },
        status="success",
        coverage=Coverage(
            requested_range=runtime.environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt-1",
        summary="one negative anomaly profile and one provisional winner",
    )
    root_evidence = Evidence(
        evidence_id="root-profile",
        observation_refs=[item.observation_id],
        subject_ids=["bank:entity:Tomcat02"],
        predicate="query_metric_profile.result",
        value={
            "finding_gate_summary": {
                "requested_fault_type": "high CPU usage",
                "fault_type_ready": True,
                "material_event_ready": True,
                "boundary_aligned": False,
                "mechanism_signature": None,
                "signature_free_finding_ready": False,
                "root_cause_score": 0.2,
            }
        },
        observed_interval=runtime.environment.task.incident_range,
        onset_interval=runtime.environment.task.incident_range,
        time_uncertainty="fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="fixture",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="root-source",
        dataset_snapshot_id=runtime.environment.dataset_id,
        evidence_kind="metric_profile",
        semantic_quality="typed",
    )
    negative_evidence = root_evidence.model_copy(
        update={
            "evidence_id": "negative-profile",
            "subject_ids": ["bank:entity:Tomcat01"],
            "predicate": "registered_mechanism_not_ready",
            "value": {"finding_gate_summary": {"fault_type_ready": False}},
            "source_fingerprint": "negative-source",
        }
    )
    hypothesis = Hypothesis(
        hypothesis_id="winner",
        subject="bank:entity:Tomcat02",
        statement="Tomcat02 has high CPU usage",
        status="open",
    )
    assessment = EvidenceAssessment(
        assessment_id="winner-support",
        evidence_id=root_evidence.evidence_id,
        hypothesis_id=hypothesis.hypothesis_id,
        impact="supports",
        reason_code="typed_profile",
        explanation="the typed profile supports the provisional winner",
        observation_digest="a" * 64,
    )
    anomaly = material_anomaly()
    runtime._observations[item.observation_id] = item  # pyright: ignore[reportPrivateUsage]
    runtime._evidence = {  # pyright: ignore[reportPrivateUsage]
        root_evidence.evidence_id: root_evidence,
        negative_evidence.evidence_id: negative_evidence,
    }
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]
    runtime._assessments[assessment.assessment_id] = assessment  # pyright: ignore[reportPrivateUsage]
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]
    runtime._current_best_hypothesis_id = hypothesis.hypothesis_id  # pyright: ignore[reportPrivateUsage]
    decision = Decision.model_validate(
        {
            "decision_summary": "classify one excluded rival relative to the winner",
            "reflection": {
                "based_on_observation_ids": [item.observation_id],
                "evidence_update": [],
                "new_hypothesis": None,
                "new_hypothesis_is_current_best": False,
                "current_best_hypothesis_id": None,
                "hypothesis_update": [],
                "anomaly_disposition_update": [
                    {
                        "anomaly_id": anomaly.anomaly_id,
                        "disposition": "concurrent_noncausal",
                        "hypothesis_id": hypothesis.hypothesis_id,
                        "evidence_refs": [negative_evidence.evidence_id],
                        "explanation": "the registered rival mechanism is excluded",
                    }
                ],
                "next_check": None,
                "can_finalize": True,
            },
            "operations": [
                {
                    "type": "finalize",
                    "abstain_reason": "one root anomaly remains unresolved",
                }
            ],
        }
    )

    error = runtime._validate_frontier_decision(decision)  # pyright: ignore[reportPrivateUsage]
    assert error is not None
    assert "signature_free_finding_ready=true" in error

    runtime._evidence[root_evidence.evidence_id] = root_evidence.model_copy(  # pyright: ignore[reportPrivateUsage]
        update={
            "value": {
                "finding_gate_summary": {
                    "requested_fault_type": "high CPU usage",
                    "fault_type_ready": True,
                    "material_event_ready": True,
                    "boundary_aligned": False,
                    "mechanism_signature": None,
                    "signature_free_finding_ready": True,
                    "material_effect_metric_ids": ["cpu", "load"],
                    "root_cause_score": 0.2,
                }
            }
        }
    )
    assert (
        runtime._validate_frontier_decision(decision)  # pyright: ignore[reportPrivateUsage]
        is None
    )
    store.close()


def test_frontier_partial_page_certificate_is_committed_before_advance_rejection(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    item = Observation(
        observation_id="observation-1",
        tool="query_metric",
        arguments={"entity_id": "bank:entity:Tomcat01", "metric_id": "cpu"},
        status="success",
        coverage=Coverage(
            requested_range=runtime.environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt-1",
        summary="negative profile evidence",
    )
    item_evidence = Evidence(
        evidence_id="evidence-1",
        observation_refs=[item.observation_id],
        subject_ids=["bank:entity:Tomcat01"],
        predicate="profile_not_ready",
        value=False,
        observed_interval=runtime.environment.task.incident_range,
        time_uncertainty="fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="fixture",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="fixture-source",
        dataset_snapshot_id=runtime.environment.dataset_id,
    )
    first = material_anomaly("anomaly-1")
    second = material_anomaly("anomaly-2")
    runtime._observations[item.observation_id] = item  # pyright: ignore[reportPrivateUsage]
    runtime._evidence[item_evidence.evidence_id] = item_evidence  # pyright: ignore[reportPrivateUsage]
    runtime._material_anomalies = {  # pyright: ignore[reportPrivateUsage]
        first.anomaly_id: first,
        second.anomaly_id: second,
    }
    runtime._decision_id = "decision:partial-certificate:0001"  # pyright: ignore[reportPrivateUsage]
    runtime._decision = Decision.model_validate(  # pyright: ignore[reportPrivateUsage]
        {
            "decision_summary": "persist one honest slot and keep the page open",
            "reflection": {
                "based_on_observation_ids": [item.observation_id],
                "evidence_update": [],
                "new_hypothesis": None,
                "new_hypothesis_is_current_best": False,
                "current_best_hypothesis_id": None,
                "hypothesis_update": [],
                "anomaly_disposition_update": [
                    {
                        "anomaly_id": first.anomaly_id,
                        "disposition": "background",
                        "evidence_refs": [item_evidence.evidence_id],
                        "explanation": "the typed profile excludes this event",
                    }
                ],
                "next_check": None,
                "can_finalize": False,
            },
            "operations": [{"type": "advance_frontier"}],
        }
    )

    state = asyncio.run(
        runtime._route_untraced({})  # pyright: ignore[reportPrivateUsage]
    )

    assert state.get("route") == "build_context"
    dispositions = store.records("anomaly_disposition")
    assert len(dispositions) == 1
    assert cast_dict(dispositions[0].payload)["anomaly_id"] == first.anomaly_id
    partial = [
        cast_dict(record.payload)
        for record in store.records("investigation_event")
        if cast_dict(record.payload).get("event_type")
        == "frontier_page_certificate_partial"
    ]
    assert len(partial) == 1
    assert partial[0]["execution"] == ("reflection_committed_advance_not_executed")
    store.close()


def test_frontier_requires_bounded_paginated_challenger_discovery_page(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    environment = BankEnvironment(PUBLIC_INPUT)
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    overview = Observation(
        observation_id="overview",
        tool="summarize_anomalies",
        arguments={"limit": 12},
        status="success",
        coverage=Coverage(
            requested_range=environment.task.allowed_query_range,
            completeness="complete",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="overview-attempt",
        summary=json.dumps(
            {
                "fault_type_candidates": [
                    {
                        "candidate_id": f"challenger-{index}",
                        "candidate_rank": index,
                        "fault_type": f"candidate cpu {index}",
                        "entity_id": "bank:entity:Tomcat02",
                        "metric_id": f"cpu-{index}",
                        "metric_group": [f"cpu-{index}"],
                        "resource_family": "cpu",
                        "score": 1.0 - index / 100,
                        "overview_score": 1.0 - index / 100,
                        "requires_paired_exact": True,
                        "materiality": "navigation_only",
                    }
                    for index in range(10)
                ]
            }
        ),
    )
    profile_observation = Observation(
        observation_id="profile-observation",
        tool="query_metric_profile",
        arguments={
            "entity_id": "bank:entity:Tomcat01",
            "metric_ids": ["material-cpu"],
            "fault_type": "material cpu",
        },
        status="success",
        coverage=overview.coverage,
        artifact_refs=[],
        attempt_id="profile-attempt",
    )
    latest = Observation(
        observation_id="latest-observation",
        tool="query_trace_edges",
        arguments={"entity_id": "bank:entity:Tomcat01", "direction": "incoming"},
        status="success",
        coverage=overview.coverage,
        artifact_refs=[],
        attempt_id="latest-attempt",
    )
    profile_evidence = Evidence(
        evidence_id="material-profile",
        observation_refs=[profile_observation.observation_id],
        subject_ids=["bank:entity:Tomcat01"],
        predicate="query_metric_profile.result",
        value={
            "finding_gate_summary": {
                "requested_fault_type": "material cpu",
                "fault_type_ready": False,
                "root_cause_score": 0.0,
            }
        },
        observed_interval=environment.task.incident_range,
        time_uncertainty="fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="fixture",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="material-profile-source",
        dataset_snapshot_id=environment.dataset_id,
        evidence_kind="metric_profile",
        mechanism_dimension="resource_profile",
        semantic_quality="typed",
    )
    anomaly = MaterialAnomaly(
        anomaly_id="material-anomaly",
        subject_id="bank:entity:Tomcat01",
        signal_family="cpu",
        signal_ids=["material-cpu"],
        mechanism_candidates=["material cpu"],
        source_observation_ids=[overview.observation_id],
        proposal_ids=["material-proposal"],
        materiality_basis="fixture",
    )
    runtime._observations = {  # pyright: ignore[reportPrivateUsage]
        item.observation_id: item for item in (overview, profile_observation, latest)
    }
    runtime._evidence[profile_evidence.evidence_id] = profile_evidence  # pyright: ignore[reportPrivateUsage]
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]
    runtime._reflected_observation_ids.update(  # pyright: ignore[reportPrivateUsage]
        {overview.observation_id, profile_observation.observation_id}
    )
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]

    trace_decision = Decision.model_validate(
        {
            "decision_summary": "deep trace before bounded discovery is invalid",
            "reflection": {
                "based_on_observation_ids": [latest.observation_id],
                "evidence_update": [],
                "new_hypothesis": None,
                "new_hypothesis_is_current_best": False,
                "current_best_hypothesis_id": None,
                "hypothesis_update": [],
                "anomaly_disposition_update": [],
                "next_check": {
                    "operation_tools": ["query_trace_edges"],
                    "distinguishes_hypothesis_ids": [],
                    "purpose": "inspect a deep trace",
                    "expected_outcomes": ["a trace change may identify a cause"],
                },
                "can_finalize": False,
            },
            "operations": [
                {
                    "type": "query",
                    "tool": "query_trace_edges",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "direction": "incoming",
                    },
                }
            ],
        }
    )
    error = runtime._validate_frontier_decision(  # pyright: ignore[reportPrivateUsage]
        trace_decision
    )

    assert error is not None
    assert "bounded challenger discovery" in error
    assert "8 seed(s)" in error
    assert "cpu-0" not in error and "cpu-7" not in error

    weak_material_profile = profile_evidence.model_copy(
        update={
            "value": {
                "finding_gate_summary": {
                    "requested_fault_type": "material cpu",
                    "fault_type_ready": False,
                    "material_event_ready": True,
                    "mechanism_signature": "peer_isolated_queue_congestion_event",
                    "root_cause_score": 0.4,
                }
            }
        }
    )
    runtime._evidence[weak_material_profile.evidence_id] = weak_material_profile  # pyright: ignore[reportPrivateUsage]
    assert runtime._frontier_has_typed_ready_leader() is False  # pyright: ignore[reportPrivateUsage]
    strong_material_profile = weak_material_profile.model_copy(
        update={
            "value": {
                "finding_gate_summary": {
                    "requested_fault_type": "material cpu",
                    "fault_type_ready": False,
                    "material_event_ready": True,
                    "mechanism_signature": "peer_isolated_trace_error_event",
                    "root_cause_score": 0.5,
                }
            }
        }
    )
    runtime._evidence[strong_material_profile.evidence_id] = strong_material_profile  # pyright: ignore[reportPrivateUsage]
    assert runtime._frontier_has_typed_ready_leader() is True  # pyright: ignore[reportPrivateUsage]

    for index in range(8):
        profiled = Observation(
            observation_id=f"challenger-profile-{index}",
            tool="query_metric_profile",
            arguments={
                "entity_id": "bank:entity:Tomcat02",
                "metric_ids": [f"cpu-{index}"],
                "fault_type": f"candidate cpu {index}",
            },
            status="success",
            coverage=overview.coverage,
            artifact_refs=[],
            attempt_id=f"challenger-profile-attempt-{index}",
        )
        runtime._observations[profiled.observation_id] = profiled  # pyright: ignore[reportPrivateUsage]
    second_page = runtime._frontier_pending_challenger_page()  # pyright: ignore[reportPrivateUsage]
    assert [item["fault_type"] for item in second_page] == [
        "candidate cpu 8",
        "candidate cpu 9",
    ]
    store.close()


def test_frontier_advance_commits_reflection_without_external_tool(
    tmp_path: Path,
) -> None:
    observation_id = f"{RUN_ID}:observation:decision:{RUN_ID}:0001:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "inspect one exact metric",
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": "bank:entity:Tomcat01",
                        "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
                        "period": "incident",
                    },
                }
            ],
        },
        {
            "decision_summary": "commit the reflected page without a dummy query",
            "reflection": {
                "based_on_observation_ids": [observation_id],
                "evidence_update": [
                    {
                        "evidence_id": EVIDENCE_ID,
                        "hypothesis_id": "new_hypothesis",
                        "impact": "supports",
                        "reason_code": "typed_observation",
                        "explanation": "the observed CPU signal seeds investigation",
                    }
                ],
                "new_hypothesis": {
                    "subject": "bank:entity:Tomcat01",
                    "statement": "CPU pressure requires causal closeout",
                    "derived_from": [EVIDENCE_ID],
                    "open_questions": ["whether this is causal"],
                },
                "new_hypothesis_is_current_best": True,
                "current_best_hypothesis_id": None,
                "hypothesis_update": [
                    {
                        "hypothesis_id": "new_hypothesis",
                        "status": "open",
                        "open_questions": ["whether this is causal"],
                    }
                ],
                "anomaly_disposition_update": [],
                "next_check": None,
                "can_finalize": False,
            },
            "operations": [{"type": "advance_frontier"}],
        },
        {
            "decision_summary": "finish after the state-only transition",
            "operations": [{"type": "finalize", "abstain_reason": "fixture complete"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(
        tmp_path,
        records,
        model=model,
        context_strategy="frontier_reflection",
    )

    assert result.status == "completed_partial"
    assert result.model_calls == 3
    assert len(store.records("observation")) == 1
    assert len(store.records("reflection")) == 1
    assert cast_dict(result.report["budget"])["tool_attempts_used"] == 1
    store.close()


def test_formal_finding_requires_every_material_anomaly_disposition(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-1",
        subject="bank:entity:Tomcat01",
        statement="CPU fault",
        status="supported",
        supporting_evidence_refs=["evidence-1"],
    )
    anomaly = material_anomaly()
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]

    errors = runtime._anomaly_coverage_errors(  # pyright: ignore[reportPrivateUsage]
        [hypothesis.hypothesis_id]
    )

    assert any("unresolved anomalies" in error for error in errors)
    assert any("root_cause anomaly disposition" in error for error in errors)
    store.close()


def test_root_cause_anomaly_disposition_completes_finding_coverage(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-1",
        subject="bank:entity:Tomcat01",
        statement="CPU fault",
        status="supported",
        supporting_evidence_refs=["evidence-1"],
    )
    anomaly = material_anomaly()
    disposition = AnomalyDisposition(
        disposition_id="disposition-1",
        anomaly_id=anomaly.anomaly_id,
        disposition="root_cause",
        hypothesis_id=hypothesis.hypothesis_id,
        evidence_refs=["evidence-1"],
        explanation="direct CPU fault evidence",
        observation_digest="a" * 64,
    )
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]
    runtime._anomaly_dispositions[anomaly.anomaly_id] = disposition  # pyright: ignore[reportPrivateUsage]

    errors = runtime._anomaly_coverage_errors(  # pyright: ignore[reportPrivateUsage]
        [hypothesis.hypothesis_id]
    )

    assert errors == []
    store.close()


def test_root_cause_coverage_uses_persisted_support_assessments(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-1",
        subject="bank:entity:Tomcat01",
        statement="CPU fault",
        status="supported",
        supporting_evidence_refs=["evidence-profile"],
    )
    anomaly = material_anomaly()
    disposition = AnomalyDisposition(
        disposition_id="disposition-1",
        anomaly_id=anomaly.anomaly_id,
        disposition="root_cause",
        hypothesis_id=hypothesis.hypothesis_id,
        evidence_refs=["evidence-profile", "evidence-trace"],
        explanation="profile proves the mechanism and trace corroborates impact",
        observation_digest="a" * 64,
    )
    assessment = EvidenceAssessment(
        assessment_id="assessment-trace",
        evidence_id="evidence-trace",
        hypothesis_id=hypothesis.hypothesis_id,
        impact="supports",
        reason_code="corroborating_trace",
        explanation="the trace corroborates the explicitly profiled mechanism",
        observation_digest="b" * 64,
    )
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]
    runtime._anomaly_dispositions[anomaly.anomaly_id] = disposition  # pyright: ignore[reportPrivateUsage]
    runtime._assessments[assessment.assessment_id] = assessment  # pyright: ignore[reportPrivateUsage]

    errors = runtime._anomaly_coverage_errors(  # pyright: ignore[reportPrivateUsage]
        [hypothesis.hypothesis_id]
    )

    assert errors == []
    store.close()


def test_same_reflection_support_can_ground_root_cause_disposition(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-1",
        subject="bank:entity:Tomcat01",
        statement="CPU fault",
        status="open",
    )
    anomaly = material_anomaly()
    disposition = AnomalyDisposition(
        disposition_id="disposition-1",
        anomaly_id=anomaly.anomaly_id,
        disposition="root_cause",
        hypothesis_id=hypothesis.hypothesis_id,
        evidence_refs=["evidence-1"],
        explanation="same-reflection direct CPU evidence",
        observation_digest="a" * 64,
    )
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]

    errors = runtime._anomaly_coverage_errors(  # pyright: ignore[reportPrivateUsage]
        [hypothesis.hypothesis_id],
        proposed=[disposition],
        prospective_support={hypothesis.hypothesis_id: {"evidence-1"}},
    )

    assert errors == []
    store.close()


def test_frontier_formal_finding_rejects_unresolved_output_questions(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis-1",
        subject="bank:entity:Tomcat01",
        statement="CPU fault",
        status="supported",
        supporting_evidence_refs=["evidence-1"],
    )
    runtime._hypotheses[hypothesis.hypothesis_id] = hypothesis  # pyright: ignore[reportPrivateUsage]

    _reference, quality = runtime._finalization_errors(  # pyright: ignore[reportPrivateUsage]
        FinalizeOperation(
            type="finalize",
            hypothesis_ids=[hypothesis.hypothesis_id],
            unresolved_questions=["root cause time remains unknown"],
        )
    )

    assert any("cannot retain unresolved_questions" in error for error in quality)
    store.close()


def test_frontier_abstention_allows_unresolved_material_anomalies(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    anomaly = material_anomaly()
    runtime._material_anomalies[anomaly.anomaly_id] = anomaly  # pyright: ignore[reportPrivateUsage]

    reference, quality = runtime._finalization_errors(  # pyright: ignore[reportPrivateUsage]
        FinalizeOperation(
            type="finalize",
            abstain_reason="material anomaly remains observationally unresolved",
        )
    )

    assert reference == []
    assert not any("material anomaly" in error for error in quality)
    store.close()


def test_frontier_market_fault_type_is_bound_by_typed_profile_evidence(
    tmp_path: Path,
) -> None:
    environment = BankEnvironment(PUBLIC_INPUT)
    environment.task = environment.task.model_copy(
        update={"environment": "openrca_market", "expected_fault_count": 1}
    )
    store = RunStore(tmp_path / "run.sqlite3")
    subject = "bank:entity:Tomcat01"
    overview = Observation(
        observation_id="overview",
        tool="summarize_anomalies",
        arguments={"limit": 12},
        status="success",
        coverage=Coverage(
            requested_range=environment.task.allowed_query_range,
            completeness="complete",
            collection_health="healthy",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="overview-attempt",
        summary=json.dumps(
            {
                "public_fault_types": ["node disk space consumption"],
                "service_profile_seeds": [
                    {
                        "service_id": "bank:entity:Tomcat02",
                        "fault_type": "container memory load",
                        "member_entity_ids": ["bank:entity:Tomcat02"],
                        "consensus_member_count": 1,
                        "member_count": 1,
                        "consensus_ratio": 1.0,
                        "overview_score": 0.2,
                    }
                ],
            }
        ),
    )
    evidence_item = Evidence(
        evidence_id="profile",
        observation_refs=[overview.observation_id],
        subject_ids=[subject],
        predicate="query_metric_profile.result",
        value={
            "finding_gate_summary": {
                "requested_fault_type": "node disk space consumption",
                "fault_type_ready": True,
                "root_cause_score": 1.2,
                "single_component_ready": True,
                "qualifying_metric_ids": ["node::system.disk.pct_usage"],
            }
        },
        observed_interval=environment.task.incident_range,
        onset_interval=environment.task.incident_range,
        time_uncertainty="fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="metric-profile",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="profile-source",
        dataset_snapshot_id=environment.dataset_id,
        evidence_kind="metric_profile",
        mechanism_dimension="resource_profile",
        independence_group="profile-source",
        semantic_quality="typed",
    )
    hypothesis = Hypothesis(
        hypothesis_id="hypothesis",
        subject=subject,
        statement="node-1 has sustained storage pressure",
        status="open",
    )
    assessment = EvidenceAssessment(
        assessment_id="assessment",
        evidence_id=evidence_item.evidence_id,
        hypothesis_id=hypothesis.hypothesis_id,
        impact="supports",
        reason_code="typed_profile",
        explanation="the typed profile binds the public fault type",
        observation_digest="a" * 64,
    )
    store.append("observation", overview.observation_id, overview)
    store.append("evidence", evidence_item.evidence_id, evidence_item)
    store.append("hypothesis", hypothesis.hypothesis_id, hypothesis)
    store.append("evidence_assessment", assessment.assessment_id, assessment)
    runtime = RootWalkRuntime(
        environment,
        NoCallModel(),
        store,
        ArtifactStore(tmp_path / "artifacts"),
        RuntimeConfig(context_strategy="frontier_reflection"),
    )
    runtime._restore_from_records()  # pyright: ignore[reportPrivateUsage]
    runtime._deadline_at = 10**12  # pyright: ignore[reportPrivateUsage]

    errors = runtime._frontier_finding_errors(  # pyright: ignore[reportPrivateUsage]
        hypothesis.hypothesis_id,
        {(assessment.evidence_id, assessment.hypothesis_id): assessment},
    )

    assert len(errors) == 1
    assert "signature_free_finding_ready=true" in errors[0]
    assert (
        runtime._frontier_eligible_profile_ids_for_subject(  # pyright: ignore[reportPrivateUsage]
            subject
        )
        == set()
    )

    runtime._observations[overview.observation_id] = overview.model_copy(  # pyright: ignore[reportPrivateUsage]
        update={
            "summary": json.dumps(
                {
                    "public_fault_types": ["node disk space consumption"],
                    "service_profile_seeds": [
                        {
                            "service_id": "bank:entity:Tomcat02",
                            "fault_type": "container network latency",
                            "member_entity_ids": ["bank:entity:Tomcat02"],
                            "consensus_member_count": 1,
                            "member_count": 1,
                            "consensus_ratio": 1.0,
                            "overview_score": 0.2,
                        }
                    ],
                }
            )
        }
    )
    errors = runtime._frontier_finding_errors(  # pyright: ignore[reportPrivateUsage]
        hypothesis.hypothesis_id,
        {(assessment.evidence_id, assessment.hypothesis_id): assessment},
    )
    assert len(errors) == 1
    assert "unresolved higher-rank mechanism-capable service seeds" in errors[0]

    checked_service = Observation(
        observation_id="checked-service-profile",
        tool="query_service_profile",
        arguments={
            "service_id": "bank:entity:Tomcat02",
            "fault_type": "container network latency",
            "member_entity_ids": ["bank:entity:Tomcat02"],
        },
        status="no_data",
        coverage=overview.coverage,
        artifact_refs=[],
        attempt_id="checked-service-attempt",
        summary="no matching trace mechanism",
    )
    runtime._observations[checked_service.observation_id] = checked_service  # pyright: ignore[reportPrivateUsage]
    errors = runtime._frontier_finding_errors(  # pyright: ignore[reportPrivateUsage]
        hypothesis.hypothesis_id,
        {(assessment.evidence_id, assessment.hypothesis_id): assessment},
    )
    assert len(errors) == 1
    assert "signature_free_finding_ready=true" in errors[0]

    runtime._evidence[evidence_item.evidence_id] = evidence_item.model_copy(  # pyright: ignore[reportPrivateUsage]
        update={
            "value": {
                "finding_gate_summary": {
                    "requested_fault_type": "node disk space consumption",
                    "fault_type_ready": True,
                    "signature_free_finding_ready": True,
                    "material_effect_metric_ids": ["node::system.disk.pct_usage"],
                    "root_cause_score": 1.2,
                    "single_component_ready": True,
                    "qualifying_metric_ids": ["node::system.disk.pct_usage"],
                }
            }
        }
    )
    runtime._observations[overview.observation_id] = overview.model_copy(  # pyright: ignore[reportPrivateUsage]
        update={
            "summary": json.dumps(
                {"public_fault_types": ["node disk space consumption"]}
            )
        }
    )
    errors = runtime._frontier_finding_errors(  # pyright: ignore[reportPrivateUsage]
        hypothesis.hypothesis_id,
        {(assessment.evidence_id, assessment.hypothesis_id): assessment},
    )
    assert errors == []
    assert runtime._frontier_eligible_profile_ids_for_subject(  # pyright: ignore[reportPrivateUsage]
        subject
    ) == {evidence_item.evidence_id}
    store.close()


class CountingNavigationBank(BankEnvironment):
    def __init__(self, public_input: Path) -> None:
        super().__init__(public_input)
        self.calls: list[str] = []

    def list_entities(self) -> dict[str, Any]:
        self.calls.append("list_entities")
        return super().list_entities()

    def list_metrics(self, entity_id: str | None = None) -> dict[str, Any]:
        self.calls.append("list_metrics")
        return super().list_metrics(entity_id)


class SequenceQueryExecutor:
    sandbox_requested: Literal["off"] = "off"
    sandbox_effective: Literal["spawn"] = "spawn"
    bwrap_version: None = None
    bwrap_executable_sha256: None = None

    def __init__(self, results: list[dict[str, Any]]) -> None:
        self.results = results
        self.calls = 0

    def execute(
        self, tool: str, args: Mapping[str, Any], timeout_s: float
    ) -> dict[str, Any]:
        del tool, args, timeout_s
        result = self.results[self.calls]
        self.calls += 1
        return result


def test_retryable_error_allows_same_query_to_execute_again(tmp_path: Path) -> None:
    query: dict[str, Any] = {
        "type": "query",
        "tool": "list_entities",
        "arguments": {},
    }
    model = CapturingModel(
        [
            {"decision_summary": "first attempt", "operations": [query]},
            {"decision_summary": "retry", "operations": [query]},
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "done"}],
            },
        ]
    )
    executor = SequenceQueryExecutor(
        [
            {
                "status": "error",
                "error_type": "temporary",
                "retryable": True,
                "summary": "retry me",
            },
            {"status": "success", "summary": "recovered"},
        ]
    )
    store = RunStore(tmp_path / "run.sqlite3")
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            query_executor=executor,
        ).run(RUN_ID, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert executor.calls == 2
    assert len(store.records("observation")) == 2
    assert not [
        record
        for record in store.records("investigation_event")
        if cast_dict(record.payload).get("event_type") == "decision_rejected"
    ]
    store.close()


def test_non_retryable_error_recalls_same_query_without_retrying(
    tmp_path: Path,
) -> None:
    query: dict[str, Any] = {
        "type": "query",
        "tool": "list_entities",
        "arguments": {},
    }
    model = CapturingModel(
        [
            {"decision_summary": "first attempt", "operations": [query]},
            {"decision_summary": "repeat", "operations": [query]},
            {
                "decision_summary": "stop",
                "operations": [{"type": "finalize", "abstain_reason": "done"}],
            },
        ]
    )
    executor = SequenceQueryExecutor(
        [
            {
                "status": "error",
                "error_type": "invalid",
                "retryable": False,
                "summary": "do not retry",
            }
        ]
    )
    store = RunStore(tmp_path / "run.sqlite3")
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
            query_executor=executor,
        ).run(RUN_ID, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert executor.calls == 1
    assert len(store.records("observation")) == 1
    assert "duplicate_query_recall" in str(model.inputs[2])
    assert "zero_operations_executed" in str(model.inputs[2])
    assert "do not retry" in str(model.inputs[2])
    assert result.report["budget"]["validator_retries_used"] == 0
    assert result.report["budget"]["no_progress_rounds"] == 1
    store.close()


def test_prior_duplicate_recalls_old_result_and_executes_novel_operation(
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "list once",
            "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
        },
        {
            "decision_summary": "repeat plus a new query",
            "operations": [
                {"type": "query", "tool": "list_entities", "arguments": {}},
                {
                    "type": "query",
                    "tool": "list_metrics",
                    "arguments": {"entity_id": "bank:entity:Tomcat01"},
                },
            ],
        },
        {
            "decision_summary": "abstain",
            "operations": [{"type": "finalize", "abstain_reason": "duplicate"}],
        },
    ]
    environment = CountingNavigationBank(PUBLIC_INPUT)
    model = CapturingModel(records)

    store, result = run_runtime(
        tmp_path,
        records,
        environment=environment,
        model=model,
        validator_mode="observe_only",
    )

    assert result.status == "completed_partial"
    assert environment.calls == ["list_entities", "list_metrics"]
    assert len(store.records("observation")) == 2
    assert "duplicate_query_recall" in str(model.inputs[2])
    assert "novel_operations_executed" in str(model.inputs[2])
    assert "zero_operations_executed" not in str(model.inputs[2])
    assert not [
        record
        for record in store.records("investigation_event")
        if cast_dict(record.payload).get("event_type") == "decision_rejected"
    ]
    assert [
        record
        for record in store.records("investigation_event")
        if cast_dict(record.payload).get("event_type") == "duplicate_query_recalled"
    ]
    effects = store.records("tool_effect")
    assert len(effects) == 2
    assert effects[1].stable_key.endswith(":1")
    assert result.report["budget"]["tool_attempts_used"] == 2
    assert result.report["budget"]["no_progress_rounds"] == 0
    store.close()


def test_duplicate_navigation_feedback_is_compact_and_keeps_fault_vocabulary(
    tmp_path: Path,
) -> None:
    overview = {
        "type": "query",
        "tool": "summarize_anomalies",
        "arguments": {"limit": 20},
    }
    records: list[dict[str, Any]] = [
        {"decision_summary": "overview", "operations": [overview]},
        {"decision_summary": "repeat overview", "operations": [overview]},
        {
            "decision_summary": "stop",
            "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(tmp_path, records, model=model)

    assert result.status == "completed_partial"
    context = cast_dict(json.loads(cast(str, model.inputs[2][1]["content"])))
    feedback = context["feedback"][0]
    recalled = feedback["existing_results"][0]["observation"]
    assert set(recalled) == {
        "observation_id",
        "tool",
        "arguments",
        "status",
        "artifact_refs",
    }
    assert recalled["tool"] == "summarize_anomalies"
    assert "signals" not in json.dumps(feedback, sort_keys=True)
    assert context["public_output_vocabulary"]["fault_types"] == [
        "synthetic public fault type"
    ]
    manifest = cast_dict(store.records("context_manifest")[2].payload)
    assert manifest["estimated_tokens"] <= manifest["max_context_tokens"]
    store.close()


def test_duplicate_exact_summary_is_bounded() -> None:
    query_range = BankEnvironment(PUBLIC_INPUT).task.allowed_query_range
    observation = Observation(
        observation_id="observation:exact",
        tool="query_metric",
        arguments={"metric_id": "metric"},
        status="success",
        coverage=Coverage(
            requested_range=query_range,
            completeness="unknown",
            basis="test fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt:exact",
        summary="x" * 3_000,
    )

    payload = RootWalkRuntime._duplicate_observation(  # pyright: ignore[reportPrivateUsage]
        observation
    )

    assert payload["summary"] == "x" * 2_000
    assert payload["summary_truncated"] is True


def test_observe_only_audits_path_quality_without_model_leak(tmp_path: Path) -> None:
    hypothesis_id = f"{RUN_ID}:hypothesis:decision:{RUN_ID}:0001:0"
    path_id = f"{RUN_ID}:path:decision:{RUN_ID}:0002:0"
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "create an ungrounded candidate",
            "operations": [
                {
                    "type": "update_hypothesis",
                    "action": "create",
                    "subject": "bank:entity:Tomcat01",
                    "statement": "candidate without evidence",
                    "status": "open",
                }
            ],
        },
        {
            "decision_summary": "stage the candidate",
            "operations": [
                {
                    "type": "submit_path",
                    "hypothesis_id": hypothesis_id,
                    "root_cause_entity_id": "bank:entity:Tomcat01",
                    "symptom_entity_ids": ["bank:entity:Tomcat01"],
                    "steps": [],
                }
            ],
        },
        {
            "decision_summary": "finish experimental output",
            "operations": [
                {
                    "type": "finalize",
                    "hypothesis_ids": [hypothesis_id],
                    "path_ids": [path_id],
                }
            ],
        },
    ]
    model = CapturingModel(records)

    store, result = run_runtime(
        tmp_path, records, model=model, validator_mode="observe_only"
    )

    assert result.status == "completed_partial"
    last_input = json.dumps(model.inputs[2])
    assert "evidence reference grounding failed" not in last_input
    assert '"grounding"' not in last_input
    audit = [
        cast_dict(item.payload)
        for item in store.records("investigation_event")
        if "quality_audit" in str(cast_dict(item.payload).get("event_type"))
    ]
    assert {item["event_type"] for item in audit} == {
        "path_quality_audit",
        "finalize_quality_audit",
    }
    store.close()


class TrackingStore(RunStore):
    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.batches: list[list[str]] = []

    def append_many(self, records: Iterable[tuple[str, str, object]]) -> list[int]:
        batch = list(records)
        self.batches.append([kind for kind, _key, _payload in batch])
        return super().append_many(batch)

    def append_many_and_seal(
        self,
        records: Iterable[tuple[str, str, object]],
        run_id: str,
        status: str,
        report_ref: str,
    ) -> tuple[list[int], Any]:
        batch = list(records)
        self.batches.append([kind for kind, _key, _payload in batch])
        return super().append_many_and_seal(batch, run_id, status, report_ref)


def test_model_query_and_seal_records_use_atomic_batches(tmp_path: Path) -> None:
    model = CapturingModel(
        [
            {
                "decision_summary": "inspect entities",
                "operations": [
                    {"type": "query", "tool": "list_entities", "arguments": {}}
                ],
            }
        ]
    )
    store = TrackingStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        model,
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(max_model_calls=1),
    )

    result = asyncio.run(runtime.run(RUN_ID, tmp_path / "bundle"))

    assert ["model_response", "decision", "attempt"] in store.batches
    assert [
        "query_payload",
        "observation",
        "tool_effect",
        "attempt",
    ] in store.batches
    assert ["report", "investigation_event"] in store.batches
    assert store.read_checkpoint() is None
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0] > 0
    assert store.outbox_jobs("pending") == []
    assert len(store.outbox_jobs("done")) == 1
    assert result.status == "budget_exhausted"
    store.close()


def test_export_failure_leaves_outbox_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        RecordedModelAdapter(
            [
                {
                    "decision_summary": "finish",
                    "operations": [{"type": "finalize", "abstain_reason": "done"}],
                }
            ]
        ),
        store := RunStore(tmp_path / "run.sqlite3"),
        ArtifactStore(tmp_path / "artifact-store"),
    )

    def fail_export(*_args: object, **_kwargs: object) -> Path:
        raise OSError("export failed")

    monkeypatch.setattr("rootwalk.runtime.BundleExporter.export", fail_export)
    with pytest.raises(OSError, match="export failed"):
        asyncio.run(runtime.run(RUN_ID, tmp_path / "bundle"))

    assert len(store.outbox_jobs("pending")) == 1
    assert store.outbox_jobs("done") == []
    store.close()


async def interrupt_runtime_after(
    runtime: RootWalkRuntime, run_id: str, node: str
) -> dict[str, Any]:
    runtime._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    config = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": runtime.config.max_model_calls * 5 + 10,
    }
    async with AsyncSqliteSaver.from_conn_string(str(runtime.store.path)) as saver:
        runtime.graph = runtime._compile_graph(  # pyright: ignore[reportPrivateUsage]
            saver
        )
        result = await runtime.graph.ainvoke(
            {
                "route": "build_context",
                "pending_refs": [],
                "domain_revision": runtime.store.domain_revision,
            },
            config=config,
            durability="sync",
            interrupt_after=[node],
        )
    return cast(dict[str, Any], result)


async def resume_runtime_until(
    runtime: RootWalkRuntime, run_id: str, node: str
) -> dict[str, Any]:
    config = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": runtime.config.max_model_calls * 5 + 10,
    }
    async with AsyncSqliteSaver.from_conn_string(str(runtime.store.path)) as saver:
        runtime.graph = runtime._compile_graph(  # pyright: ignore[reportPrivateUsage]
            saver
        )
        result = await runtime.graph.ainvoke(
            None,
            config=config,
            durability="sync",
            interrupt_after=[node],
        )
    return cast(dict[str, Any], result)


def test_model_timeout_provenance_is_frozen_and_legacy_compatible(
    tmp_path: Path,
) -> None:
    run_id = "timeout-provenance"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(model_timeout_seconds=7.5),
    )
    first_runtime._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    first_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    manifest = cast_dict(first_store.records("manifest")[0].payload)
    assert manifest["model_timeout_seconds"] == 7.5
    first_store.close()

    second_store = RunStore(database)
    with pytest.raises(ValueError, match="different model_timeout_seconds"):
        asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                NoCallModel(),
                second_store,
                ArtifactStore(artifact_root),
                RuntimeConfig(model_timeout_seconds=8.0),
            ).run(run_id, tmp_path / "bundle")
        )
    second_store.close()

    legacy_store = RunStore(tmp_path / "legacy.sqlite3")
    legacy_store.append(
        "manifest",
        "manifest:legacy-timeout",
        {"run_id": "legacy-timeout", "started_at": 1.0, "deadline_at": 2.0},
    )
    legacy_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        legacy_store,
        ArtifactStore(tmp_path / "legacy-artifacts"),
        RuntimeConfig(model_timeout_seconds=9.0),
    )
    legacy_runtime._run_id = (  # pyright: ignore[reportPrivateUsage]
        "legacy-timeout"
    )
    legacy_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    legacy_manifest = cast_dict(legacy_store.records("manifest")[0].payload)
    assert "model_timeout_seconds" not in legacy_manifest
    legacy_store.close()


def test_query_sandbox_provenance_is_frozen_on_resume(tmp_path: Path) -> None:
    run_id = "sandbox-provenance"
    database = tmp_path / "run.sqlite3"
    artifacts = tmp_path / "artifacts"
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        first_store,
        ArtifactStore(artifacts),
        RuntimeConfig(
            query_sandbox_requested="required",
            query_sandbox_effective="bwrap",
            query_sandbox_bwrap_version="bubblewrap-v1",
            query_sandbox_bwrap_executable_sha256="a" * 64,
        ),
        query_executor=BwrapProvenanceExecutor("bubblewrap-v1"),
    )
    first_runtime._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    first_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    manifest = cast_dict(first_store.records("manifest")[0].payload)
    assert manifest["query_sandbox_requested"] == "required"
    assert manifest["query_sandbox_effective"] == "bwrap"
    assert manifest["query_sandbox_bwrap_version"] == "bubblewrap-v1"
    assert manifest["query_sandbox_bwrap_executable_sha256"] == "a" * 64
    first_store.close()

    second_store = RunStore(database)
    with pytest.raises(ValueError, match="different query_sandbox_bwrap_version"):
        asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                NoCallModel(),
                second_store,
                ArtifactStore(artifacts),
                RuntimeConfig(
                    query_sandbox_requested="required",
                    query_sandbox_effective="bwrap",
                    query_sandbox_bwrap_version="bubblewrap-v2",
                    query_sandbox_bwrap_executable_sha256="a" * 64,
                ),
                query_executor=BwrapProvenanceExecutor("bubblewrap-v2"),
            ).run(run_id, tmp_path / "bundle")
        )
    second_store.close()

    third_store = RunStore(database)
    with pytest.raises(
        ValueError, match="different query_sandbox_bwrap_executable_sha256"
    ):
        asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                NoCallModel(),
                third_store,
                ArtifactStore(artifacts),
                RuntimeConfig(
                    query_sandbox_requested="required",
                    query_sandbox_effective="bwrap",
                    query_sandbox_bwrap_version="bubblewrap-v1",
                    query_sandbox_bwrap_executable_sha256="b" * 64,
                ),
                query_executor=BwrapProvenanceExecutor("bubblewrap-v1", "b" * 64),
            ).run(run_id, tmp_path / "bundle")
        )
    third_store.close()


def test_resume_rejects_a_different_code_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "code-revision"
    database = tmp_path / "run.sqlite3"
    artifacts = tmp_path / "artifacts"
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        first_store,
        ArtifactStore(artifacts),
    )
    first_runtime._run_id = run_id  # pyright: ignore[reportPrivateUsage]
    first_runtime._bootstrap()  # pyright: ignore[reportPrivateUsage]
    first_store.close()

    changed = {**runtime_version_manifest(), "code_revision": "0" * 64}
    monkeypatch.setattr("rootwalk.runtime.runtime_version_manifest", lambda: changed)
    second_store = RunStore(database)
    with pytest.raises(ValueError, match="different code_revision"):
        asyncio.run(
            RootWalkRuntime(
                BankEnvironment(PUBLIC_INPUT),
                NoCallModel(),
                second_store,
                ArtifactStore(artifacts),
            ).run(run_id, tmp_path / "bundle")
        )
    second_store.close()


def test_resume_reuses_committed_decision_without_calling_model(
    tmp_path: Path,
) -> None:
    run_id = "resume-decision"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    original_model = CapturingModel(
        [
            {
                "decision_summary": "inspect entities",
                "operations": [
                    {"type": "query", "tool": "list_entities", "arguments": {}}
                ],
            }
        ]
    )
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        original_model,
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=1),
    )

    state = asyncio.run(interrupt_runtime_after(first_runtime, run_id, "build_context"))
    asyncio.run(
        first_runtime._call_model(  # pyright: ignore[reportPrivateUsage]
            cast(Any, state)
        )
    )
    assert len(original_model.inputs) == 1
    first_store.close()

    no_call_model = NoCallModel()
    second_store = RunStore(database)
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            no_call_model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=1),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "budget_exhausted"
    assert no_call_model.calls == 0
    assert len(second_store.records("decision")) == 1
    second_store.close()


def test_resume_routes_persisted_invalid_query_to_normal_rejection(
    tmp_path: Path,
) -> None:
    run_id = "resume-invalid-query"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        CapturingModel(
            [
                {
                    "decision_summary": "invalid tool",
                    "operations": [
                        {"type": "query", "tool": "not_a_tool", "arguments": {}}
                    ],
                }
            ]
        ),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=2),
    )

    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "call_model"))
    first_store.close()

    second_store = RunStore(database)
    model = CapturingModel(
        [
            {
                "decision_summary": "abstain after validation feedback",
                "operations": [{"type": "finalize", "abstain_reason": "invalid query"}],
            }
        ]
    )
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=2),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(model.inputs) == 1
    assert "unknown query tool" in str(model.inputs[0])
    assert result.report["budget"]["validator_retries_used"] == 1
    assert result.report["budget"]["no_progress_rounds"] == 0
    second_store.close()


def test_resume_does_not_duplicate_replayed_validation_feedback(
    tmp_path: Path,
) -> None:
    run_id = "resume-rejected-decision"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        CapturingModel(
            [
                {
                    "decision_summary": "unsupported claim",
                    "operations": [
                        {
                            "type": "update_hypothesis",
                            "action": "create",
                            "subject": "bank:entity:Tomcat01",
                            "statement": "unsupported",
                            "status": "supported",
                        }
                    ],
                }
            ]
        ),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=2),
    )

    state = asyncio.run(interrupt_runtime_after(first_runtime, run_id, "call_model"))
    asyncio.run(
        first_runtime._route(cast(Any, state))  # pyright: ignore[reportPrivateUsage]
    )
    first_store.close()

    second_store = RunStore(database)
    model = CapturingModel(
        [
            {
                "decision_summary": "abstain after validation feedback",
                "operations": [
                    {"type": "finalize", "abstain_reason": "unsupported claim"}
                ],
            }
        ]
    )
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=2),
        ).run(run_id, tmp_path / "bundle")
    )

    context = cast_dict(json.loads(cast(str, model.inputs[0][1]["content"])))
    assert result.status == "completed_partial"
    assert len(context["feedback"]) == 1
    assert result.report["budget"]["validator_retries_used"] == 1
    assert result.report["budget"]["no_progress_rounds"] == 0
    second_store.close()


def test_revise_hypothesis_subject_invariant_survives_resume(tmp_path: Path) -> None:
    run_id = "resume-revise-subject"
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0001:0"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    create = {
        "decision_summary": "create hypothesis",
        "operations": [
            {
                "type": "update_hypothesis",
                "action": "create",
                "subject": "bank:entity:Tomcat01",
                "statement": "initial candidate",
                "status": "open",
            }
        ],
    }
    cross_subject_revise = {
        "decision_summary": "rewrite another subject",
        "operations": [
            {
                "type": "update_hypothesis",
                "action": "revise",
                "hypothesis_id": hypothesis_id,
                "subject": "bank:entity:MG02",
                "statement": "different candidate",
                "status": "inconclusive",
            }
        ],
    }
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        CapturingModel([create, cross_subject_revise]),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=3),
    )

    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "call_model"))
    asyncio.run(resume_runtime_until(first_runtime, run_id, "call_model"))
    assert len(first_store.records("hypothesis")) == 1
    first_store.close()

    second_store = RunStore(database)
    model = CapturingModel(
        [
            {
                "decision_summary": "stop after rejection",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            }
        ]
    )
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=3),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(second_store.records("hypothesis")) == 1
    assert "create a new hypothesis" in str(model.inputs[0])
    assert result.report["budget"]["validator_retries_used"] == 1
    second_store.close()


def test_revise_hypothesis_fault_identity_invariant_survives_resume(
    tmp_path: Path,
) -> None:
    run_id = "resume-revise-fault"
    hypothesis_id = f"{run_id}:hypothesis:decision:{run_id}:0001:0"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    first_environment = BankEnvironment(PUBLIC_INPUT)
    seed_public_fault_types(first_store, first_environment)
    create = {
        "decision_summary": "create CPU hypothesis",
        "operations": [
            {
                "type": "update_hypothesis",
                "action": "create",
                "subject": "bank:entity:Tomcat01",
                "statement": "high CPU usage",
                "status": "open",
            }
        ],
    }
    cross_fault_revise = {
        "decision_summary": "rewrite as memory hypothesis",
        "operations": [
            {
                "type": "update_hypothesis",
                "action": "revise",
                "hypothesis_id": hypothesis_id,
                "subject": "bank:entity:Tomcat01",
                "statement": "high memory usage",
                "status": "inconclusive",
            }
        ],
    }
    first_runtime = RootWalkRuntime(
        first_environment,
        CapturingModel([create, cross_fault_revise]),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=3),
    )

    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "call_model"))
    asyncio.run(resume_runtime_until(first_runtime, run_id, "call_model"))
    assert len(first_store.records("hypothesis")) == 1
    first_store.close()

    second_store = RunStore(database)
    model = CapturingModel(
        [
            {
                "decision_summary": "stop after rejection",
                "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
            }
        ]
    )
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=3),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert len(second_store.records("hypothesis")) == 1
    assert "resolved public fault type" in str(model.inputs[0])
    assert "create a new hypothesis" in str(model.inputs[0])
    assert result.report["budget"]["validator_retries_used"] == 1
    second_store.close()


class CountingBank(BankEnvironment):
    calls = 0

    def list_entities(self) -> dict[str, Any]:
        type(self).calls += 1
        return super().list_entities()


def test_resume_skips_committed_query_effect(tmp_path: Path) -> None:
    run_id = "resume-query"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    CountingBank.calls = 0
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        CountingBank(PUBLIC_INPUT),
        RecordedModelAdapter(
            [
                {
                    "decision_summary": "inspect entities",
                    "operations": [
                        {"type": "query", "tool": "list_entities", "arguments": {}}
                    ],
                }
            ]
        ),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(max_model_calls=1),
    )

    state = asyncio.run(interrupt_runtime_after(first_runtime, run_id, "route"))
    asyncio.run(
        first_runtime._execute_query(  # pyright: ignore[reportPrivateUsage]
            cast(Any, state)
        )
    )
    assert CountingBank.calls == 1
    first_store.close()

    no_call_model = NoCallModel()
    second_store = RunStore(database)
    result = asyncio.run(
        RootWalkRuntime(
            CountingBank(PUBLIC_INPUT),
            no_call_model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=1),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "budget_exhausted"
    assert no_call_model.calls == 0
    assert CountingBank.calls == 1
    assert len(second_store.records("tool_effect")) == 1
    second_store.close()


def test_resume_after_mixed_duplicate_route_restores_identical_feedback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("rootwalk.runtime.time.time", lambda: 1_000.0)
    run_id = "resume-mixed-duplicate"
    first_query: dict[str, Any] = {
        "decision_summary": "list once",
        "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
    }
    mixed_query: dict[str, Any] = {
        "decision_summary": "recall one and execute one",
        "operations": [
            {"type": "query", "tool": "list_entities", "arguments": {}},
            {
                "type": "query",
                "tool": "list_metrics",
                "arguments": {"entity_id": "bank:entity:Tomcat01"},
            },
        ],
    }
    finalize: dict[str, Any] = {
        "decision_summary": "stop",
        "operations": [{"type": "finalize", "abstain_reason": "fixture"}],
    }

    baseline_model = CapturingModel([first_query, mixed_query, finalize])
    baseline_store = RunStore(tmp_path / "baseline" / "run.sqlite3")
    baseline_result = asyncio.run(
        RootWalkRuntime(
            CountingNavigationBank(PUBLIC_INPUT),
            baseline_model,
            baseline_store,
            ArtifactStore(tmp_path / "baseline" / "artifacts"),
        ).run(run_id, tmp_path / "baseline" / "bundle")
    )
    assert baseline_result.status == "completed_partial"
    baseline_context = cast_dict(
        json.loads(cast(str, baseline_model.inputs[2][1]["content"]))
    )
    baseline_store.close()

    database = tmp_path / "crash" / "run.sqlite3"
    artifact_root = tmp_path / "crash" / "artifacts"
    first_store = RunStore(database)
    first_environment = CountingNavigationBank(PUBLIC_INPUT)
    first_runtime = RootWalkRuntime(
        first_environment,
        CapturingModel([first_query, mixed_query]),
        first_store,
        ArtifactStore(artifact_root),
    )
    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "execute_query"))
    asyncio.run(resume_runtime_until(first_runtime, run_id, "route"))
    assert first_environment.calls == ["list_entities"]
    assert len(first_store.records("tool_effect")) == 1
    first_store.close()

    recovery_model = CapturingModel([finalize])
    recovery_environment = CountingNavigationBank(PUBLIC_INPUT)
    second_store = RunStore(database)
    result = asyncio.run(
        RootWalkRuntime(
            recovery_environment,
            recovery_model,
            second_store,
            ArtifactStore(artifact_root),
        ).run(run_id, tmp_path / "crash" / "bundle")
    )

    assert result.status == "completed_partial"
    assert recovery_environment.calls == ["list_metrics"]
    recovered_context = cast_dict(
        json.loads(cast(str, recovery_model.inputs[0][1]["content"]))
    )
    assert recovered_context == baseline_context
    feedback = recovered_context["feedback"]
    assert len(feedback) == 1
    assert feedback[0]["execution"] == "novel_operations_executed"
    assert feedback[0]["existing_results"][0]["operation_index"] == 0
    effects = second_store.records("tool_effect")
    assert len(effects) == 2
    assert effects[1].stable_key.endswith(":1")
    assert result.report["budget"]["tool_attempts_used"] == 2
    assert result.report["budget"]["no_progress_rounds"] == 0
    second_store.close()


class AheadState(TypedDict):
    domain_revision: int


async def write_ahead_checkpoint(path: Path, run_id: str) -> None:
    async def finish(state: AheadState) -> dict[str, Any]:
        del state
        return {}

    graph = StateGraph(AheadState)
    graph.add_node("finish", finish)  # pyright: ignore[reportUnknownMemberType]
    graph.add_edge(START, "finish")
    graph.add_edge("finish", END)
    async with AsyncSqliteSaver.from_conn_string(str(path)) as saver:
        await graph.compile(checkpointer=saver).ainvoke(  # pyright: ignore[reportUnknownMemberType]
            {"domain_revision": 999},
            config={"configurable": {"thread_id": run_id}},
            durability="sync",
        )


def test_checkpoint_ahead_fails_closed_without_external_calls(tmp_path: Path) -> None:
    run_id = "checkpoint-ahead"
    database = tmp_path / "run.sqlite3"
    asyncio.run(write_ahead_checkpoint(database, run_id))
    store = RunStore(database)
    model = NoCallModel()
    CountingBank.calls = 0

    result = asyncio.run(
        RootWalkRuntime(
            CountingBank(PUBLIC_INPUT),
            model,
            store,
            ArtifactStore(tmp_path / "artifact-store"),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "runtime_failure"
    assert "checkpoint is ahead" in str(result.report["failure"])
    assert model.calls == CountingBank.calls == 0
    store.close()


class UsageSequenceModel:
    def __init__(self, responses: list[tuple[dict[str, Any], dict[str, Any]]]) -> None:
        self.responses = responses
        self.calls = 0

    async def decide(self, messages: Sequence[Message]) -> ModelCallResult:
        del messages
        decision_payload, usage = self.responses[self.calls]
        self.calls += 1
        decision = Decision.model_validate(decision_payload)
        return ModelCallResult(
            decision=decision,
            usage=usage,
            provider_model="usage-sequence",
            raw_response={"decision": decision_payload},
        )


def test_cumulative_input_overage_blocks_response_operations(tmp_path: Path) -> None:
    model = UsageSequenceModel(
        [
            (
                {
                    "decision_summary": "inspect entities",
                    "operations": [
                        {"type": "query", "tool": "list_entities", "arguments": {}}
                    ],
                },
                {"prompt_tokens": 100_000, "completion_tokens": 1},
            ),
            (
                {
                    "decision_summary": "finish",
                    "operations": [{"type": "finalize", "abstain_reason": "enough"}],
                },
                {"prompt_tokens": 50_000, "completion_tokens": 1},
            ),
        ]
    )

    store, result = run_runtime(tmp_path, [], model=model)

    assert result.status == "budget_exhausted"
    assert model.calls == 2
    assert result.report["budget"]["input_tokens_used"] == 150_000
    assert len(store.records("tool_effect")) == 1
    assert result.report["abstain_reason"] is None
    store.close()


def test_query_batch_cannot_exceed_tool_attempt_budget(tmp_path: Path) -> None:
    model = CapturingModel(
        [
            {
                "decision_summary": "too many queries",
                "operations": [
                    {"type": "query", "tool": "list_entities", "arguments": {}},
                    {"type": "query", "tool": "list_metrics", "arguments": {}},
                ],
            }
        ]
    )

    store, result = run_runtime(tmp_path, [], model=model, max_tool_attempts=1)

    assert result.status == "budget_exhausted"
    assert not [
        record
        for record in store.records("attempt")
        if cast_dict(record.payload).get("tool")
    ]
    assert result.report["budget"]["tool_attempts_used"] == 0
    store.close()


def test_artifact_budget_exhaustion_seals_with_gap_and_no_orphan(
    tmp_path: Path,
) -> None:
    model = NoCallModel()

    store, result = run_runtime(tmp_path, [], model=model, max_artifact_bytes=1)

    assert result.status == "budget_exhausted"
    assert model.calls == 0
    assert result.report["failure"] == "artifact_bytes"
    assert len(result.report["gaps"]) == 1
    gap = result.report["gaps"][0]
    assert (gap["kind"], gap["status"], gap["used_bytes"], gap["max_bytes"]) == (
        "artifact_budget",
        "partial",
        0,
        1,
    )
    assert gap["requested_bytes"] > 1
    assert result.report["budget"]["artifact_bytes_used"] == 0
    assert not list((tmp_path / "artifact-store").iterdir())
    assert (result.bundle_path / "report.json").is_file()
    store.close()


def test_external_input_artifacts_do_not_consume_run_artifact_budget(
    tmp_path: Path,
) -> None:
    records: list[dict[str, Any]] = [
        {
            "decision_summary": "inspect entities",
            "operations": [{"type": "query", "tool": "list_entities", "arguments": {}}],
        }
    ]

    store, result = run_runtime(tmp_path, records, max_model_calls=1)

    artifact_root = tmp_path / "artifact-store"
    embedded_bytes = sum(path.stat().st_size for path in artifact_root.iterdir())
    external = [
        cast_dict(record.payload)
        for record in store.records("artifact")
        if cast_dict(record.payload).get("storage") == "external"
    ]
    assert external
    assert result.report["budget"]["artifact_bytes_used"] == embedded_bytes
    assert all(not (artifact_root / item["content_hash"]).exists() for item in external)
    store.close()


def test_resume_reuses_frozen_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 1_000.0
    monkeypatch.setattr("rootwalk.runtime.time.time", lambda: now)
    run_id = "resume-deadline"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        first_store,
        ArtifactStore(artifact_root),
        RuntimeConfig(wall_clock_seconds=30),
    )

    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "build_context"))
    deadline = cast_dict(first_store.records("manifest")[0].payload)["deadline_at"]
    first_store.close()

    now = 1_031.0
    model = NoCallModel()
    second_store = RunStore(database)
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(wall_clock_seconds=999),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "budget_exhausted"
    assert model.calls == 0
    assert (
        cast_dict(second_store.records("manifest")[0].payload)["deadline_at"]
        == deadline
    )
    second_store.close()


def test_resume_inherits_artifact_usage_and_frozen_limit(tmp_path: Path) -> None:
    run_id = "resume-artifact-budget"
    database = tmp_path / "run.sqlite3"
    artifact_root = tmp_path / "artifact-store"
    first_store = RunStore(database)
    first_artifacts = ArtifactStore(artifact_root)
    first_runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        NoCallModel(),
        first_store,
        first_artifacts,
        RuntimeConfig(max_model_calls=1, max_artifact_bytes=100_000),
    )

    asyncio.run(interrupt_runtime_after(first_runtime, run_id, "build_context"))
    used_before_resume = first_artifacts.bytes_used
    assert used_before_resume > 0
    first_store.close()

    second_store = RunStore(database)
    model = CapturingModel(
        [
            {
                "decision_summary": "finish",
                "operations": [{"type": "finalize", "abstain_reason": "done"}],
            }
        ]
    )
    result = asyncio.run(
        RootWalkRuntime(
            BankEnvironment(PUBLIC_INPUT),
            model,
            second_store,
            ArtifactStore(artifact_root),
            RuntimeConfig(max_model_calls=1, max_artifact_bytes=1),
        ).run(run_id, tmp_path / "bundle")
    )

    assert result.status == "completed_partial"
    assert result.report["budget"]["max_artifact_bytes"] == 100_000
    assert result.report["budget"]["artifact_bytes_used"] >= used_before_resume
    second_store.close()


class TimingOutModel:
    async def decide(self, messages: Sequence[Message]) -> object:
        del messages
        raise TimeoutError("provider outcome is unknown")


class InvalidResponseModel:
    async def decide(self, messages: Sequence[Message]) -> object:
        del messages
        raise ModelResponseError(
            "provider exhausted schema retries",
            responses=[{"status": "invalid", "content": "not a Decision"}],
        )


class FrontierTimeoutThenAbstainModel:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, messages: Sequence[Message]) -> ModelCallResult:
        del messages
        self.calls += 1
        if self.calls == 1:
            raise ModelTimeoutError("provider request timed out")
        payload = {
            "decision_summary": "close after the transient provider timeout",
            "operations": [
                {
                    "type": "finalize",
                    "abstain_reason": "no evidence was collected",
                }
            ],
        }
        return ModelCallResult(
            decision=Decision.model_validate(payload),
            usage={},
            provider_model="timeout-then-abstain",
            raw_response={"decision": payload},
        )


def test_unknown_model_attempt_keeps_token_reservation(tmp_path: Path) -> None:
    store, result = run_runtime(tmp_path, [], model=TimingOutModel())

    assert result.status == "runtime_failure"
    assert result.report["budget"]["input_tokens_used"] > 0
    assert result.report["budget"]["output_tokens_used"] == 2_000
    ledger = [cast_dict(record.payload) for record in store.records("budget_ledger")]
    assert ledger[0]["settlement"] == "reserved_unknown"
    assert not [item for item in ledger if item["event"] == "model_call_completed"]
    store.close()


def test_frontier_retries_provider_timeout_before_physical_closeout(
    tmp_path: Path,
) -> None:
    model = FrontierTimeoutThenAbstainModel()

    store, result = run_runtime(
        tmp_path,
        [],
        model=model,
        context_strategy="frontier_reflection",
    )

    assert result.status == "completed_partial"
    assert result.model_calls == 2
    assert model.calls == 2
    timeouts = [
        cast_dict(record.payload)
        for record in store.records("investigation_event")
        if cast_dict(record.payload).get("event_type") == "model_timeout"
    ]
    assert len(timeouts) == 1
    assert result.report["abstain_reason"] == "no evidence was collected"
    store.close()


def test_frontier_provider_timeout_at_physical_closeout_is_safe_abstention(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        TimingOutModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(
            context_strategy="frontier_reflection",
            wall_clock_seconds=1,
            model_timeout_seconds=0.1,
        ),
    )

    result = asyncio.run(runtime.run(RUN_ID, tmp_path / "bundle"))

    assert result.status == "completed_partial"
    assert result.report["outcome_type"] == "abstention"
    assert "physical closeout boundary" in cast(str, result.report["abstain_reason"])
    assert result.report["failure"] is None
    safe_closeout = [
        cast_dict(record.payload)
        for record in store.records("investigation_event")
        if cast_dict(record.payload).get("event_type") == "runtime_physical_abstention"
    ]
    assert len(safe_closeout) == 1
    assert safe_closeout[0]["model_conclusion_synthesized"] is False
    store.close()


def test_frontier_invalid_response_at_physical_closeout_is_safe_abstention(
    tmp_path: Path,
) -> None:
    store = RunStore(tmp_path / "run.sqlite3")
    runtime = RootWalkRuntime(
        BankEnvironment(PUBLIC_INPUT),
        InvalidResponseModel(),
        store,
        ArtifactStore(tmp_path / "artifact-store"),
        RuntimeConfig(
            context_strategy="frontier_reflection",
            wall_clock_seconds=1,
            model_timeout_seconds=0.1,
        ),
    )

    result = asyncio.run(runtime.run(RUN_ID, tmp_path / "bundle"))

    assert result.status == "completed_partial"
    assert result.report["outcome_type"] == "abstention"
    assert "schema-valid Decision" in cast(str, result.report["abstain_reason"])
    assert result.report["failure"] is None
    events = [
        cast_dict(record.payload) for record in store.records("investigation_event")
    ]
    assert any(item.get("event_type") == "model_invalid_response" for item in events)
    safe_closeout = next(
        item
        for item in events
        if item.get("event_type") == "runtime_physical_abstention"
    )
    assert safe_closeout["policy"] == ("provider-invalid-response-safe-closeout-v1")
    assert safe_closeout["model_conclusion_synthesized"] is False
    store.close()


def test_repeated_invalid_submission_stops_as_no_progress(tmp_path: Path) -> None:
    invalid: dict[str, Any] = {
        "decision_summary": "repeat invalid request",
        "operations": [{"type": "query", "tool": "not_a_tool", "arguments": {}}],
    }
    model = CapturingModel([invalid, invalid, invalid])

    store, result = run_runtime(tmp_path, [], model=model)

    assert result.status == "no_progress"
    assert len(model.inputs) == 3
    ledger = [cast_dict(record.payload) for record in store.records("budget_ledger")]
    assert (
        len([item for item in ledger if item["event"] == "validator_retry_started"])
        == 2
    )
    assert result.report["budget"]["no_progress_rounds"] == 3
    store.close()


def cast_dict(value: object) -> dict[str, Any]:
    assert isinstance(value, dict)
    return cast(dict[str, Any], value)
