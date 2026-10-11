"""Minimal LangGraph investigation loop for one prepared Bank task."""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, cast, get_type_hints

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.errors import NodeCancelledError
from langgraph.graph import END, START, StateGraph
from opentelemetry import trace
from opentelemetry.trace import SpanContext, TraceFlags
from pydantic import TypeAdapter, ValidationError

from .anomalies import material_anomalies, material_anomalies_from_evidence
from .context import (
    NAVIGATION_TOOLS,
    ContextStrategy,
    ValidatorMode,
    candidate_disposition_queue,
    candidate_schedule,
    comparison_coverage,
    compile_context,
    frontier_active_anomaly_ids,
    frontier_profile_seeds,
    frontier_service_profile_seeds,
    has_strict_directional_metric_shift,
)
from .contracts import (
    AdvanceFrontierOperation,
    AnomalyDisposition,
    AnomalyDispositionProposal,
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
    Operation,
    QueryOperation,
    Reflection,
    SubmitPathOperation,
    TimeRange,
    UpdateHypothesisOperation,
)
from .contracts import (
    Path as CausalPath,
)
from .domain import classify_observation, ground_path
from .environment import RuntimeEnvironment
from .model import (
    Message,
    ModelCallResult,
    ModelResponseError,
    ModelTimeoutError,
    aggregate_model_usage,
    redact_sensitive,
    safe_exception_summary,
)
from .observability import (
    ObservabilityConfig,
    context_span,
    generic_span,
    model_span,
    record_model_result,
    record_span_content,
    record_span_outcome,
    recovery_run_span,
    run_span,
    tool_span,
    validation_span,
)
from .occurrence import (
    OccurrenceAtom,
    OccurrenceGroup,
    build_occurrence_groups,
    matching_occurrence_group,
)
from .openrca import dominant_market_fault_type, mechanism_signature_rank
from .store import (
    ArtifactBudgetExceeded,
    ArtifactStore,
    BundleExporter,
    RunStore,
    StableKeyConflict,
    checkpoint_relation,
)
from .version import runtime_version_manifest

QUERY_TOOLS = (
    "list_entities",
    "list_metrics",
    "query_metric",
    "query_metric_profile",
    "query_impact_profile",
    "query_trace_operation_profile",
    "query_service_profile",
    "query_log_samples",
    "summarize_log_patterns",
    "query_trace_neighbors",
    "query_trace_edges",
    "read_artifact_slice",
    "summarize_anomalies",
)
QUERY_CANDIDATE_TOOL = "query_candidate"
CLOSEOUT_ROUND_BUDGET = 4
DUPLICATE_SUMMARY_MAX_CHARS = 2_000
MAX_FAULT_TYPE_CANDIDATES = 90

TOOL_DESCRIPTIONS = {
    "list_entities": (
        "List all authorized canonical entity IDs; takes no arguments. Navigation "
        "only: the result is not causal evidence."
    ),
    "list_metrics": (
        "List metric_ids and log_names, optionally for one canonical entity_id. "
        "Navigation only: catalog presence is not causal evidence."
    ),
    "query_metric": (
        "Query one canonical entity_id and exact metric_id returned by list_metrics or "
        "summarize_anomalies in a named period. Returns requested-period points plus "
        "a complete paired reference/incident summary. Empty names and aliases are "
        "invalid."
    ),
    "query_metric_profile": (
        "Compare one to sixteen exact metrics across both task windows. Returns "
        "typed shape, quantile, burst, onset, optional peer, and registered ratio "
        "facts. When following initial_profile_seeds, pass its exact fault_type so "
        "the Adapter can validate metric semantics. This is Evidence, not a "
        "root-cause verdict."
    ),
    "query_impact_profile": (
        "Compare business-entry latency, success, error, and request-volume signals "
        "across reference and incident windows. Returns source-backed impact onset "
        "Evidence with evidence_role=impact_only; it may corroborate timing and "
        "business effect but can never be a root-cause subject."
    ),
    "query_trace_operation_profile": (
        "Aggregate named operations for one canonical entity across reference and "
        "incident windows. Returns typed operation latency/count shifts, onset "
        "intervals, competing reference episodes, and identifiability. Use it to "
        "corroborate or weaken a metric hypothesis; it is Evidence, not a verdict."
    ),
    "query_service_profile": (
        "Aggregate one canonical service across its pod members and service-level "
        "trace edges for one exact public fault_type. Returns typed multi-replica "
        "consensus and mechanism facts. This is Evidence, not a root-cause verdict."
    ),
    "query_log_samples": (
        "Read bounded requested-period log samples and complete reference/incident "
        "counts in one scan for an entity with a registered log source. Omit log_name "
        "or use an exact value returned in summarize_anomalies.log_names or "
        "list_metrics.log_names. Topology source filenames are not log_name values. "
        "contains is an optional literal substring, not regex."
    ),
    "summarize_log_patterns": (
        "Discover deterministic, versioned log templates and paired reference/incident "
        "counts before exact sample drill-down."
    ),
    "query_trace_neighbors": (
        "Read source-backed topology neighbors for one canonical entity_id; direction "
        "must be incoming, outgoing, or both."
    ),
    "query_trace_edges": (
        "Compare reference and incident call count, error rate, latency quantiles, and "
        "anomaly onset on directed raw-trace edges."
    ),
    "read_artifact_slice": (
        "Read by an exact 64-character SHA-256 from a prior artifact_refs entry; "
        "filenames are invalid. This reads existing material and does not create new "
        "independent evidence."
    ),
    "summarize_anomalies": (
        "Compare the unverified reference and incident windows for navigation only; "
        "takes an optional limit."
    ),
    QUERY_CANDIDATE_TOOL: (
        "Select the first candidate_id listed by comparison_coverage; on the first "
        "candidate query Runtime deterministically expands it to the persisted "
        "overview's first eight unique exact direct metric series, each with paired "
        "reference and incident query_metric calls, plus paired calls for any "
        "distinct source_metric_id."
    ),
}


def _compact_paired_log_summary(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    source = cast(Mapping[str, object], raw)
    counts_complete = source.get("counts_complete")
    samples_truncated = source.get("samples_truncated")
    if not isinstance(counts_complete, bool) or not isinstance(samples_truncated, bool):
        return None
    compact: dict[str, object] = {
        "counts_complete": counts_complete,
        "samples_truncated": samples_truncated,
    }
    for period in ("reference", "incident"):
        raw_period_counts = source.get(period)
        if not isinstance(raw_period_counts, Mapping):
            return None
        period_counts = cast(Mapping[str, object], raw_period_counts)
        matched_count = period_counts.get("matched_count")
        raw_log_name_counts = period_counts.get("log_name_counts")
        if (
            not isinstance(matched_count, int)
            or isinstance(matched_count, bool)
            or matched_count < 0
            or not isinstance(raw_log_name_counts, Mapping)
        ):
            return None
        log_name_counts = cast(Mapping[object, object], raw_log_name_counts)
        names: dict[str, int] = {}
        for name, count in log_name_counts.items():
            if (
                not isinstance(name, str)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
            ):
                return None
            names[name] = count
        if sum(names.values()) != matched_count:
            return None
        compact[period] = {
            "matched_count": matched_count,
            "log_name_counts": dict(sorted(names.items())),
        }
    return compact


def _has_complete_paired_log_summary(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    summary = _compact_paired_log_summary(
        cast(Mapping[str, object], value).get("paired_log_summary")
    )
    return summary is not None and summary["counts_complete"] is True


def _has_actionable_percent_pressure(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    summary = _compact_paired_metric_summary(
        cast(Mapping[str, object], value).get("paired_metric_summary")
    )
    if summary is None or summary["counts_complete"] is not True:
        return False
    incident = cast(Mapping[str, object], summary["incident"])
    direction = summary["pressure_direction"]
    if summary["scale_hint"] != "0..100":
        return False
    if direction == "higher_is_more_pressure":
        count, extreme = (
            incident["above_reference_max_count"],
            incident["max"],
        )
        return bool(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 2
            and isinstance(extreme, (int, float))
            and not isinstance(extreme, bool)
            and extreme >= 80
        )
    if direction == "lower_is_more_pressure":
        count, extreme = (
            incident["below_reference_min_count"],
            incident["min"],
        )
        return bool(
            isinstance(count, int)
            and not isinstance(count, bool)
            and count >= 2
            and isinstance(extreme, (int, float))
            and not isinstance(extreme, bool)
            and extreme <= 20
        )
    return False


def _has_strict_directional_metric_shift(
    value: object, pressure_direction: object = None
) -> bool:
    """Return whether a complete paired summary is a strong, directed incident shift."""
    if not isinstance(value, Mapping):
        return False
    summary = _compact_paired_metric_summary(
        cast(Mapping[str, object], value).get("paired_metric_summary")
    )
    return has_strict_directional_metric_shift(summary, pressure_direction)


def _has_strict_source_shift(candidate: Mapping[str, Any]) -> bool:
    return has_strict_directional_metric_shift(
        candidate.get("source_paired_metric_summary"),
        candidate.get("source_pressure_direction"),
    )


def _has_strict_direct_shift(candidate: Mapping[str, Any]) -> bool:
    return bool(
        candidate.get("metric_role") == "direct"
        and has_strict_directional_metric_shift(
            candidate.get("paired_metric_summary"),
            candidate.get("candidate_pressure_direction"),
        )
    )


def select_terminal_candidate(
    candidates: Sequence[Mapping[str, Any]],
    supported_hypotheses_by_subject: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Select by v4, then optionally switch subject within the exact same reason."""
    if not candidates:
        return None, None

    def v4_key(item: Mapping[str, Any]) -> tuple[int, int, int, str]:
        selection = cast(Mapping[str, Any], item["selection"])
        return (
            -int(
                selection["source_strict"] is True
                and selection["has_log_source"] is False
            ),
            -int(selection["direct_strict"] is True),
            cast(int, selection["terminal_queue_rank"]),
            cast(str, cast(Mapping[str, Any], item["hypothesis"])["hypothesis_id"]),
        )

    base = min(candidates, key=v4_key)
    base_hypothesis = cast(Mapping[str, Any], base["hypothesis"])
    same_reason = [
        item
        for item in candidates
        if cast(Mapping[str, Any], item["hypothesis"]).get("statement")
        == base_hypothesis.get("statement")
    ]
    matching_subjects = {
        cast(str, hypothesis["subject"])
        for item in same_reason
        for hypothesis in [cast(Mapping[str, Any], item["hypothesis"])]
        if hypothesis.get("subject") in supported_hypotheses_by_subject
    }
    if len(matching_subjects) != 1:
        return dict(base), None
    subject = next(iter(matching_subjects))

    def strategy_s_key(item: Mapping[str, Any]) -> tuple[int, int, int, int, str]:
        hypothesis = cast(Mapping[str, Any], item["hypothesis"])
        selection = cast(Mapping[str, Any], item["selection"])
        return (
            -int(
                selection["source_strict"] is True
                and selection["has_log_source"] is False
            ),
            -int(hypothesis.get("subject") == subject),
            -int(selection["direct_strict"] is True),
            cast(int, selection["terminal_queue_rank"]),
            cast(str, hypothesis["hypothesis_id"]),
        )

    winner = min(same_reason, key=strategy_s_key)
    if winner.get("candidate_id") == base.get("candidate_id") and cast(
        Mapping[str, Any], winner["hypothesis"]
    ).get("hypothesis_id") == base_hypothesis.get("hypothesis_id"):
        return dict(base), None
    return dict(winner), {
        "supported_subject_match": True,
        "unique_supported_terminal_subject": subject,
        "base_candidate_id": base.get("candidate_id"),
        "base_hypothesis_id": base_hypothesis.get("hypothesis_id"),
        "overlap_supported_hypothesis_ids": sorted(
            supported_hypotheses_by_subject[subject]
        ),
    }


_FAULT_TYPE_FAMILIES = {
    "high CPU usage": frozenset({"cpu"}),
    "high JVM CPU load": frozenset({"cpu"}),
    "high memory usage": frozenset({"memory"}),
    "JVM Out of Memory (OOM) Heap": frozenset({"memory"}),
    "high disk I/O read usage": frozenset({"storage"}),
    "high disk space usage": frozenset({"storage"}),
    "network latency": frozenset({"network", "latency"}),
    "network packet loss": frozenset({"network"}),
}
_KNOWN_CANDIDATE_FAMILIES = frozenset(
    {"cpu", "memory", "storage", "network", "latency"}
)


def select_family_compatible_terminal_candidate(
    candidates: Sequence[Mapping[str, Any]],
    winner: Mapping[str, Any],
    candidate_families: Mapping[str, object],
    base_policy: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Apply FAM2 only when the completed v5 winner is explicitly incompatible."""
    hypothesis = cast(Mapping[str, Any], winner["hypothesis"])
    statement = cast(str, hypothesis["statement"])
    candidate_id = cast(str, winner["candidate_id"])
    family = candidate_families.get(candidate_id)
    allowed = _FAULT_TYPE_FAMILIES.get(statement)
    if allowed is None or family not in _KNOWN_CANDIDATE_FAMILIES or family in allowed:
        return dict(winner), None
    excluded = sorted(
        cast(str, item["candidate_id"])
        for item in candidates
        if isinstance(item.get("candidate_id"), str)
        and (
            candidate_allowed := _FAULT_TYPE_FAMILIES.get(
                cast(str, cast(Mapping[str, Any], item["hypothesis"])["statement"])
            )
        )
        is not None
        and (
            candidate_family := candidate_families.get(cast(str, item["candidate_id"]))
        )
        in _KNOWN_CANDIDATE_FAMILIES
        and candidate_family not in candidate_allowed
    )
    compatible = [
        item for item in candidates if item.get("candidate_id") not in set(excluded)
    ]
    if not compatible:
        return dict(winner), None
    selected = min(
        compatible,
        key=lambda item: (
            cast(
                int, cast(Mapping[str, Any], item["selection"])["terminal_queue_rank"]
            ),
            cast(str, cast(Mapping[str, Any], item["hypothesis"])["hypothesis_id"]),
        ),
    )
    if selected.get("candidate_id") == candidate_id and cast(
        Mapping[str, Any], selected["hypothesis"]
    ).get("hypothesis_id") == hypothesis.get("hypothesis_id"):
        return dict(winner), None
    return dict(selected), {
        "known_mismatch_filter_applied": True,
        "pre_family_policy": base_policy,
        "base_candidate_id": candidate_id,
        "base_hypothesis_id": hypothesis.get("hypothesis_id"),
        "base_statement": statement,
        "base_resource_family": family,
        "excluded_candidate_ids": excluded,
    }


def candidate_metric_pair_is_valid(
    reference_evidence_id: str,
    incident_evidence_id: str,
    evidence_by_id: Mapping[str, Evidence],
    observations: Mapping[str, Observation],
    subject: str,
    metric_id: object,
) -> bool:
    if reference_evidence_id == incident_evidence_id or not isinstance(metric_id, str):
        return False
    for evidence_id, period in (
        (reference_evidence_id, "reference"),
        (incident_evidence_id, "incident"),
    ):
        evidence = evidence_by_id.get(evidence_id)
        if (
            evidence is None
            or evidence.predicate != "query_metric.result"
            or subject not in evidence.subject_ids
        ):
            return False
        matching = [
            observation
            for observation_id in evidence.observation_refs
            for observation in [observations.get(observation_id)]
            if observation is not None
            and observation.tool == "query_metric"
            and observation.status in {"success", "partial"}
            and observation.arguments.get("entity_id") == subject
            and observation.arguments.get("metric_id") == metric_id
            and observation.arguments.get("period", "incident") == period
        ]
        if len(matching) != 1:
            return False
    return True


def supported_candidate_binding(
    hypothesis: Hypothesis,
    bindings: Sequence[Mapping[str, Any]],
    queue_by_id: Mapping[str, tuple[int, Mapping[str, Any]]],
    evidence_by_id: Mapping[str, Evidence],
    observations: Mapping[str, Observation],
) -> str | None:
    candidate_ids = {
        event.get("candidate_id")
        for event in bindings
        if isinstance(event.get("candidate_id"), str) and event.get("candidate_id")
    }
    if (
        hypothesis.status != "supported"
        or not bindings
        or len(candidate_ids) != 1
        or not all(
            event.get("candidate_id") in candidate_ids
            and event.get("subject") == hypothesis.subject
            and event.get("selected_fault_type") == hypothesis.statement
            and event.get("status") in {"open", "supported"}
            for event in bindings
        )
        or bindings[-1].get("status") != "supported"
    ):
        return None
    candidate_id = next(iter(candidate_ids))
    if not isinstance(candidate_id, str) or candidate_id not in queue_by_id:
        return None
    candidate = queue_by_id[candidate_id][1]
    candidate_evidence_ids = candidate.get("evidence_ids")
    fault_types = [candidate.get("fault_type")]
    aliases = candidate.get("fault_type_aliases", [])
    if isinstance(aliases, list):
        fault_types.extend(cast(list[object], aliases))
    latest = bindings[-1]
    original = latest.get("original_operation")
    applied = latest.get("applied_operation")
    reference_evidence_id = latest.get("reference_evidence_id")
    incident_evidence_id = latest.get("incident_evidence_id")
    original_refs: set[str] = (
        set(cast(Sequence[str], original.get("supporting_evidence_refs", ())))
        if isinstance(original, Mapping)
        and isinstance(original.get("supporting_evidence_refs"), list)
        else set()
    )
    applied_refs: set[str] = (
        set(cast(Sequence[str], applied.get("supporting_evidence_refs", ())))
        if isinstance(applied, Mapping)
        and isinstance(applied.get("supporting_evidence_refs"), list)
        else set()
    )
    pair_refs = {reference_evidence_id, incident_evidence_id}
    if (
        candidate.get("entity_id") != hypothesis.subject
        or hypothesis.statement not in fault_types
        or not isinstance(original, Mapping)
        or not isinstance(applied, Mapping)
        or original.get("status") != "supported"
        or original.get("subject") != hypothesis.subject
        or original.get("statement") != hypothesis.statement
        or applied.get("status") != "supported"
        or applied.get("subject") != hypothesis.subject
        or applied.get("statement") != hypothesis.statement
        or not isinstance(reference_evidence_id, str)
        or not isinstance(incident_evidence_id, str)
        or not isinstance(candidate_evidence_ids, Mapping)
        or candidate_evidence_ids.get("reference") != reference_evidence_id
        or candidate_evidence_ids.get("incident") != incident_evidence_id
        or not pair_refs
        <= original_refs & applied_refs & set(hypothesis.supporting_evidence_refs)
        or not candidate_metric_pair_is_valid(
            reference_evidence_id,
            incident_evidence_id,
            evidence_by_id,
            observations,
            hypothesis.subject,
            candidate.get("metric_id"),
        )
    ):
        return None
    return candidate_id


def _compact_paired_metric_summary(raw: object) -> dict[str, object] | None:
    if not isinstance(raw, Mapping):
        return None
    source = cast(Mapping[str, object], raw)
    counts_complete = source.get("counts_complete")
    scale_hint = source.get("scale_hint")
    pressure_direction = source.get("pressure_direction")
    hint_basis = source.get("hint_basis")
    if (
        not isinstance(counts_complete, bool)
        or scale_hint not in {"0..100", "unknown"}
        or pressure_direction
        not in {"lower_is_more_pressure", "higher_is_more_pressure", "unknown"}
        or hint_basis != "heuristic from metric name"
    ):
        return None
    compact: dict[str, object] = {
        "counts_complete": counts_complete,
        "scale_hint": scale_hint,
        "pressure_direction": pressure_direction,
        "hint_basis": hint_basis,
    }
    sample_counts: dict[str, int] = {}
    for period in ("reference", "incident"):
        raw_stats = source.get(period)
        if not isinstance(raw_stats, Mapping):
            return None
        stats = cast(Mapping[str, object], raw_stats)
        sample_count = stats.get("sample_count")
        bounds = [stats.get(name) for name in ("min", "median", "max")]
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 0
            or (sample_count == 0 and any(bound is not None for bound in bounds))
            or (
                sample_count > 0
                and (
                    any(
                        not isinstance(bound, (int, float)) or isinstance(bound, bool)
                        for bound in bounds
                    )
                    or not cast(float, bounds[0])
                    <= cast(float, bounds[1])
                    <= cast(float, bounds[2])
                )
            )
        ):
            return None
        compact[period] = {
            "sample_count": sample_count,
            "min": bounds[0],
            "median": bounds[1],
            "max": bounds[2],
        }
        sample_counts[period] = sample_count
    incident = cast(Mapping[str, object], source["incident"])
    excursion_counts: dict[str, int | None] = {}
    for name in ("above_reference_max_count", "below_reference_min_count"):
        count = incident.get(name)
        if sample_counts["reference"] == 0:
            if count is not None:
                return None
        elif (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            or count > sample_counts["incident"]
        ):
            return None
        excursion_counts[name] = count
    if (
        sum(count or 0 for count in excursion_counts.values())
        > sample_counts["incident"]
    ):
        return None
    if sample_counts["reference"] and sample_counts["incident"]:
        reference = cast(Mapping[str, float], compact["reference"])
        compact_incident = cast(Mapping[str, float], compact["incident"])
        if (bool(excursion_counts["above_reference_max_count"])) != (
            compact_incident["max"] > reference["max"]
        ) or (bool(excursion_counts["below_reference_min_count"])) != (
            compact_incident["min"] < reference["min"]
        ):
            return None
    cast(dict[str, object], compact["incident"]).update(excursion_counts)
    return compact


def _observability_provenance(config: ObservabilityConfig) -> dict[str, object]:
    return {
        "enabled": config.enabled,
        "capture_content": config.capture_content,
        "sample_ratio": config.sample_ratio,
        "endpoint_sha256": (
            hashlib.sha256(config.endpoint.encode()).hexdigest()
            if config.endpoint is not None
            else None
        ),
    }


class DecisionModel(Protocol):
    async def decide(self, messages: Sequence[Message]) -> object: ...


class QueryExecutor(Protocol):
    @property
    def sandbox_requested(self) -> Literal["auto", "required", "off"]: ...

    @property
    def sandbox_effective(self) -> Literal["bwrap", "spawn"]: ...

    @property
    def bwrap_version(self) -> str | None: ...

    @property
    def bwrap_executable_sha256(self) -> str | None: ...

    def execute(
        self, tool: str, args: Mapping[str, Any], timeout_s: float
    ) -> dict[str, Any]: ...


class _GraphState(TypedDict, total=False):
    route: str
    phase: str
    model_input_ref: str
    model_call_number: int
    estimated_input_tokens: int
    decision_id: str
    pending_refs: list[str]
    domain_revision: int


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    wall_clock_seconds: int = 300
    cleanup_grace_seconds: int = 10
    max_model_calls: int = 12
    max_context_tokens: int = 12_000
    max_input_tokens: int = 144_000
    max_output_tokens: int = 24_000
    max_tool_attempts: int = 40
    max_artifact_bytes: int = 64 * 1024 * 1024
    tool_timeout_seconds: float = 20.0
    max_validator_retries: int = 2
    max_no_progress_rounds: int = 3
    context_strategy: ContextStrategy = "evidence"
    validator_mode: ValidatorMode = "enforce"
    model_provider: str = "unknown"
    model_id: str = "unknown"
    model_endpoint: str | None = None
    model_timeout_seconds: float | None = None
    max_output_tokens_per_call: int = 2_000
    reasoning_effort: str | None = None
    query_sandbox_requested: Literal["auto", "required", "off"] = "off"
    query_sandbox_effective: Literal["bwrap", "spawn"] = "spawn"
    query_sandbox_bwrap_version: str | None = None
    query_sandbox_bwrap_executable_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.wall_clock_seconds < 1:
            raise ValueError("wall_clock_seconds must be at least one")
        if self.cleanup_grace_seconds < 0:
            raise ValueError("cleanup_grace_seconds must be non-negative")
        if self.max_model_calls < 1:
            raise ValueError("max_model_calls must be at least one")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be at least one")
        if self.max_input_tokens < 1 or self.max_output_tokens < 1:
            raise ValueError("token budgets must be at least one")
        if self.max_tool_attempts < 0 or self.max_validator_retries < 0:
            raise ValueError("attempt budgets must be non-negative")
        if self.max_artifact_bytes < 1:
            raise ValueError("max_artifact_bytes must be at least one")
        if self.tool_timeout_seconds <= 0:
            raise ValueError("tool_timeout_seconds must be greater than zero")
        if self.max_no_progress_rounds < 1:
            raise ValueError("max_no_progress_rounds must be at least one")
        if self.context_strategy not in {
            "recent",
            "evidence",
            "frontier_reflection",
        }:
            raise ValueError(
                "context_strategy must be recent, evidence, or frontier_reflection"
            )
        if self.validator_mode not in {"enforce", "observe_only"}:
            raise ValueError("validator_mode must be enforce or observe_only")
        if not self.model_provider or not self.model_id:
            raise ValueError("model_provider and model_id must be non-empty")
        if self.model_timeout_seconds is not None and self.model_timeout_seconds <= 0:
            raise ValueError("model_timeout_seconds must be greater than zero")
        if self.max_output_tokens_per_call < 1:
            raise ValueError("max_output_tokens_per_call must be at least one")
        if self.query_sandbox_requested == "required" and (
            self.query_sandbox_effective != "bwrap"
        ):
            raise ValueError("required query sandbox must use bubblewrap")
        if self.query_sandbox_requested == "off" and (
            self.query_sandbox_effective != "spawn"
        ):
            raise ValueError("disabled query sandbox must use spawn")
        if self.query_sandbox_effective == "bwrap":
            if not self.query_sandbox_bwrap_version:
                raise ValueError("bubblewrap sandbox must record its version")
            digest = self.query_sandbox_bwrap_executable_sha256
            if digest is None or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError("bubblewrap sandbox must record its executable SHA256")
        elif (
            self.query_sandbox_bwrap_version is not None
            or self.query_sandbox_bwrap_executable_sha256 is not None
        ):
            raise ValueError("spawn query worker cannot record bubblewrap provenance")


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    run_id: str
    status: str
    model_calls: int
    report: dict[str, Any]
    bundle_path: Path


class RootWalkRuntime:
    """Run one bounded investigation; all domain effects land in ``store``."""

    def __init__(
        self,
        environment: RuntimeEnvironment,
        model: DecisionModel,
        store: RunStore,
        artifacts: ArtifactStore,
        config: RuntimeConfig = RuntimeConfig(),
        query_executor: QueryExecutor | None = None,
        tracer: trace.Tracer | None = None,
        observability_config: ObservabilityConfig = ObservabilityConfig(),
    ) -> None:
        self.environment = environment
        self.model = model
        self.store = store
        self.artifacts = artifacts
        self.config = config
        self._max_artifact_bytes = config.max_artifact_bytes
        self.artifacts.set_max_bytes(self._max_artifact_bytes)
        self.query_executor = query_executor
        actual = (
            (
                query_executor.sandbox_requested,
                query_executor.sandbox_effective,
                query_executor.bwrap_version,
                query_executor.bwrap_executable_sha256,
            )
            if query_executor is not None
            else ("off", "spawn", None, None)
        )
        configured = (
            config.query_sandbox_requested,
            config.query_sandbox_effective,
            config.query_sandbox_bwrap_version,
            config.query_sandbox_bwrap_executable_sha256,
        )
        if actual != configured:
            raise ValueError(
                "query executor sandbox provenance does not match RuntimeConfig"
            )
        self.observability_config = observability_config
        self.tracer = (
            tracer
            if observability_config.enabled and tracer is not None
            else trace.get_tracer(__name__)
            if observability_config.enabled
            else trace.NoOpTracer()
        )
        optional_tools = {
            "query_impact_profile": self._unsupported_query_impact_profile,
            "query_trace_operation_profile": (
                self._unsupported_query_trace_operation_profile
            ),
        }
        self.tools: dict[str, Callable[..., dict[str, Any]]] = {}
        for name in QUERY_TOOLS:
            method = getattr(environment, name, None)
            if callable(method):
                self.tools[name] = cast(Callable[..., dict[str, Any]], method)
            elif name in optional_tools:
                self.tools[name] = optional_tools[name]
            else:
                raise AttributeError(f"runtime environment is missing tool {name}")
        self.tools["read_artifact_slice"] = self._read_artifact_slice
        self.tools[QUERY_CANDIDATE_TOOL] = self._query_candidate
        self.graph: Any = None

        self._run_id = ""
        self._execution_attempt_id = ""
        self._bundle_dir = Path()
        self._model_calls: int = 0
        self._provider_attempts: int = 0
        self._model_usage: dict[str, int | float] = {}
        self._input_tokens_used: int = 0
        self._output_tokens_used: int = 0
        self._tool_attempts = 0
        self._validator_retries = 0
        self._no_progress_rounds = 0
        self._started_at = 0.0
        self._deadline_at = 0.0
        self._base_revision = 0
        self._decision: Decision | None = None
        self._decision_id = ""
        self._pending_operation: Operation | None = None
        self._pending_candidate_disposition: dict[str, Any] | None = None
        self._weak_finding_rejection: dict[str, Any] | None = None
        self._query_calls: list[tuple[int, QueryOperation, dict[str, Any]]] = []
        self._candidate_query_origins: dict[int, tuple[str, int]] = {}
        self._feedback: list[dict[str, Any]] = []
        self._gaps: list[dict[str, Any]] = []
        self._observations: dict[str, Observation] = {}
        self._evidence: dict[str, Evidence] = {}
        self._assessments: dict[str, EvidenceAssessment] = {}
        self._material_anomalies: dict[str, MaterialAnomaly] = {}
        self._anomaly_dispositions: dict[str, AnomalyDisposition] = {}
        self._reflected_observation_ids: set[str] = set()
        self._current_best_hypothesis_id: str | None = None
        self._hypotheses: dict[str, Hypothesis] = {}
        self._paths: dict[str, CausalPath] = {}
        self._grounding: dict[str, dict[str, Any]] = {}
        self._final_request: FinalizeOperation | None = None
        self._terminal_status: str | None = None
        self._failure: str | None = None
        self._report: dict[str, Any] | None = None

    async def run(
        self,
        run_id: str,
        bundle_dir: str | Path,
        *,
        execution_attempt_id: str | None = None,
        previous_span_context: SpanContext | None = None,
    ) -> RuntimeResult:
        if not run_id.strip():
            raise ValueError("run_id must be non-empty")
        if execution_attempt_id is not None and not execution_attempt_id.strip():
            raise ValueError("execution_attempt_id must be non-empty when provided")
        self._execution_attempt_id = execution_attempt_id or uuid.uuid4().hex
        if previous_span_context is None and self.observability_config.enabled:
            previous_span_context = self._load_previous_span_context(run_id)
        span = (
            recovery_run_span(
                self.tracer,
                self.observability_config,
                run_id=run_id,
                execution_attempt_id=self._execution_attempt_id,
                previous_span_context=previous_span_context,
            )
            if previous_span_context is not None
            else run_span(
                self.tracer,
                self.observability_config,
                run_id=run_id,
                execution_attempt_id=self._execution_attempt_id,
            )
        )
        with span as root_span:
            self._save_root_span_context(run_id, root_span.get_span_context())
            record_span_content(
                root_span,
                self.observability_config,
                input_value={
                    "run_id": run_id,
                    "task": self.environment.task.model_dump(mode="json"),
                },
            )
            result = await self._run(run_id, bundle_dir)
            record_span_content(
                root_span,
                self.observability_config,
                output_value=result.report,
            )
            record_span_outcome(root_span, result.status)
            return result

    def _load_previous_span_context(self, run_id: str) -> SpanContext | None:
        try:
            records = reversed(self.store.records("late_audit"))
            for record in records:
                payload = record.payload
                if (
                    not isinstance(payload, dict)
                    or payload.get("audit_type") != "otel_root_span_context"
                    or payload.get("run_id") != run_id
                ):
                    continue
                saved = cast(dict[str, object], payload)
                trace_id = saved.get("trace_id")
                span_id = saved.get("span_id")
                trace_flags = saved.get("trace_flags")
                is_remote = saved.get("is_remote")
                if (
                    not isinstance(trace_id, str)
                    or not isinstance(span_id, str)
                    or not isinstance(trace_flags, int)
                    or isinstance(trace_flags, bool)
                    or not isinstance(is_remote, bool)
                ):
                    continue
                context = SpanContext(
                    trace_id=int(trace_id, 16),
                    span_id=int(span_id, 16),
                    is_remote=is_remote,
                    trace_flags=TraceFlags(trace_flags),
                )
                if context.is_valid:
                    return context
        except Exception:
            pass
        return None

    def _save_root_span_context(self, run_id: str, context: SpanContext) -> None:
        if not context.is_valid:
            return
        try:
            trace_id = f"{context.trace_id:032x}"
            span_id = f"{context.span_id:016x}"
            self.store.append(
                "late_audit",
                f"otel_root_span_context:{run_id}:{self._execution_attempt_id}:{span_id}",
                {
                    "audit_type": "otel_root_span_context",
                    "run_id": run_id,
                    "execution_attempt_id": self._execution_attempt_id,
                    "trace_id": trace_id,
                    "span_id": span_id,
                    "trace_flags": int(context.trace_flags),
                    "is_remote": context.is_remote,
                },
            )
        except Exception:
            pass

    async def _run(self, run_id: str, bundle_dir: str | Path) -> RuntimeResult:
        if self._run_id:
            raise RuntimeError("a RootWalkRuntime instance may run only once")
        self._run_id = run_id
        self._bundle_dir = Path(bundle_dir)
        if self.store.seal_state(run_id) is not None:
            self._restore_from_records()
            return self._export_result()

        with generic_span(
            self.tracer,
            "rootwalk.persistence",
            self.observability_config,
            run_id=self._run_id,
            execution_attempt_id=self._execution_attempt_id,
            attributes={"record_count": 3},
        ) as persistence_span:
            record_span_content(
                persistence_span,
                self.observability_config,
                input_value={
                    "run_id": self._run_id,
                    "runtime_config": asdict(self.config),
                },
            )
            self._bootstrap()
            manifest_record = self.store.get(f"manifest:{self._run_id}")
            record_span_content(
                persistence_span,
                self.observability_config,
                output_value=(
                    manifest_record.payload if manifest_record is not None else None
                ),
            )
        graph_config: RunnableConfig = {
            "configurable": {"thread_id": run_id},
            "recursion_limit": (
                10_000
                if self.config.context_strategy == "frontier_reflection"
                else self.config.max_model_calls * 5 + 10
            ),
        }
        async with AsyncSqliteSaver.from_conn_string(str(self.store.path)) as saver:
            self.graph = self._compile_graph(saver)
            checkpoint = await saver.aget(graph_config)
            graph_input: _GraphState | None
            if checkpoint is None:
                self._restore_from_records()
                graph_input = {
                    "route": "build_context",
                    "pending_refs": [],
                    "domain_revision": self.store.domain_revision,
                }
            else:
                saved = checkpoint.get("channel_values", {})
                saved_revision = saved.get("domain_revision", 0)
                if not isinstance(saved_revision, int):
                    saved_revision = -1
                relation = checkpoint_relation(
                    self.store.domain_revision, saved_revision
                )
                self._restore_from_records(saved)
                if relation == "checkpoint_ahead":
                    self._terminal_status = "runtime_failure"
                    self._failure = "LangGraph checkpoint is ahead of the domain ledger"
                    self._seal()
                    graph_input = None
                else:
                    graph_input = None
            if self._report is None:
                with generic_span(
                    self.tracer,
                    "rootwalk.graph",
                    self.observability_config,
                    run_id=self._run_id,
                    execution_attempt_id=self._execution_attempt_id,
                    attributes={"node_count": 7},
                ) as graph_span:
                    record_span_content(
                        graph_span,
                        self.observability_config,
                        input_value=graph_input,
                    )
                    try:
                        await self.graph.ainvoke(
                            graph_input,
                            config=graph_config,
                            durability="sync",
                        )
                    except (asyncio.CancelledError, NodeCancelledError):
                        self._terminal_status = "cancelled"
                        self._seal()
                    except ArtifactBudgetExceeded as exc:
                        self._handle_artifact_budget(exc)
                        self._seal()
                    except Exception as exc:
                        self._terminal_status = "runtime_failure"
                        self._failure = f"{type(exc).__name__}: {exc}"
                        self._seal()
                    record_span_outcome(
                        graph_span, self._terminal_status or "completed"
                    )
                    record_span_content(
                        graph_span,
                        self.observability_config,
                        output_value=self._report,
                    )
        if self._report is None:
            raise RuntimeError("runtime ended without a report")
        return self._export_result()

    def _export_result(self) -> RuntimeResult:
        if self._report is None:
            raise RuntimeError("cannot export without a report")
        with generic_span(
            self.tracer,
            "rootwalk.export",
            self.observability_config,
            run_id=self._run_id,
            execution_attempt_id=self._execution_attempt_id,
            attributes={"record_count": self.store.domain_revision},
        ) as export_span:
            record_span_content(
                export_span,
                self.observability_config,
                input_value=self._report,
            )
            bundle = BundleExporter(self.store, self.artifacts).export(self._bundle_dir)
            sealed = self.store.seal_state(self._run_id)
            if sealed is None:
                raise RuntimeError("bundle exported before run seal")
            self.store.mark_outbox_done(self._run_id, sealed.report_revision)
            record_span_content(
                export_span,
                self.observability_config,
                output_value={
                    "status": "exported",
                    "domain_revision": self.store.domain_revision,
                    "report_revision": sealed.report_revision,
                },
            )
        return RuntimeResult(
            run_id=self._run_id,
            status=cast(str, self._report["status"]),
            model_calls=self._model_calls,
            report=self._report,
            bundle_path=bundle,
        )

    def _compile_graph(self, checkpointer: Any) -> Any:
        graph = StateGraph(_GraphState)
        graph.add_node("build_context", self._build_context)
        graph.add_node("call_model", self._call_model)
        graph.add_node("route", self._route)
        graph.add_node("execute_query", self._execute_query)
        graph.add_node("apply_hypothesis", self._apply_hypothesis)
        graph.add_node("stage_path", self._stage_path)
        graph.add_node("finalize", self._finalize)
        graph.add_edge(START, "build_context")
        graph.add_conditional_edges(
            "build_context",
            self._state_route,
            {"call_model": "call_model", "finalize": "finalize"},
        )
        graph.add_conditional_edges(
            "call_model",
            self._state_route,
            {
                "build_context": "build_context",
                "route": "route",
                "finalize": "finalize",
            },
        )
        graph.add_conditional_edges(
            "route",
            self._state_route,
            {
                "build_context": "build_context",
                "execute_query": "execute_query",
                "apply_hypothesis": "apply_hypothesis",
                "stage_path": "stage_path",
                "finalize": "finalize",
            },
        )
        graph.add_edge("execute_query", "build_context")
        graph.add_edge("apply_hypothesis", "build_context")
        graph.add_edge("stage_path", "build_context")
        graph.add_conditional_edges(
            "finalize",
            self._state_route,
            {"build_context": "build_context", "end": END},
        )
        return graph.compile(checkpointer=checkpointer)

    def _bootstrap(self) -> None:
        manifest_key = f"manifest:{self._run_id}"
        versions = runtime_version_manifest()
        observability = _observability_provenance(self.observability_config)
        existing = self.store.get(manifest_key)
        manifest: dict[str, Any]
        if existing is not None:
            if not isinstance(existing.payload, dict):
                raise ValueError("manifest record must be an object")
            manifest = cast(dict[str, Any], existing.payload)
            deadline_at = manifest.get("deadline_at")
            if not isinstance(deadline_at, (int, float)):
                raise ValueError("manifest deadline_at must be numeric")
            self._deadline_at = float(deadline_at)
            self._validate_frozen_manifest(manifest, versions, observability)
            artifact_limit = manifest.get("max_artifact_bytes")
            if artifact_limit is not None:
                if not isinstance(artifact_limit, int) or artifact_limit < 1:
                    raise ValueError("manifest max_artifact_bytes must be positive")
                self._max_artifact_bytes = artifact_limit
                self.artifacts.set_max_bytes(artifact_limit)
            started_at = manifest.get("started_at")
            if isinstance(started_at, (int, float)):
                self._started_at = float(started_at)
            else:
                configured_seconds = manifest.get("wall_clock_seconds")
                self._started_at = self._deadline_at - (
                    float(configured_seconds)
                    if isinstance(configured_seconds, (int, float))
                    else self.config.wall_clock_seconds
                )
        else:
            self._started_at = time.time()
            self._deadline_at = self._started_at + self.config.wall_clock_seconds
            manifest = {
                "run_id": self._run_id,
                **versions,
                "model_provider": self.config.model_provider,
                "model_id": self.config.model_id,
                "model_endpoint": redact_sensitive(self.config.model_endpoint),
                "model_timeout_seconds": self.config.model_timeout_seconds,
                "observability": observability,
                "context_strategy": self.config.context_strategy,
                "validator_mode": self.config.validator_mode,
                "query_sandbox_requested": self.config.query_sandbox_requested,
                "query_sandbox_effective": self.config.query_sandbox_effective,
                "query_sandbox_bwrap_version": (
                    self.config.query_sandbox_bwrap_version
                ),
                "query_sandbox_bwrap_executable_sha256": (
                    self.config.query_sandbox_bwrap_executable_sha256
                ),
                "wall_clock_limit_seconds": self.config.wall_clock_seconds,
                "started_at": self._started_at,
                "cleanup_grace_seconds": self.config.cleanup_grace_seconds,
                "deadline_at": self._deadline_at,
                "max_model_calls": self.config.max_model_calls,
                "max_input_tokens": self.config.max_input_tokens,
                "max_output_tokens": self.config.max_output_tokens,
                "max_output_tokens_per_call": self.config.max_output_tokens_per_call,
                "max_tool_attempts": self.config.max_tool_attempts,
                "max_artifact_bytes": self._max_artifact_bytes,
                "tool_timeout_seconds": self.config.tool_timeout_seconds,
                "max_validator_retries": self.config.max_validator_retries,
                "max_no_progress_rounds": self.config.max_no_progress_rounds,
                "reasoning_effort": self.config.reasoning_effort,
                "data_version": self.environment.dataset_id,
            }
        self._append_bootstrap_record("manifest", manifest_key, manifest)
        self._append_bootstrap_record(
            "task",
            f"task:{self._run_id}",
            self.environment.task.model_dump(mode="json"),
        )
        self._append_bootstrap_record(
            "topology",
            f"topology:{self._run_id}",
            self.environment.topology.model_dump(mode="json"),
        )

    def _append_bootstrap_record(
        self, kind: str, stable_key: str, payload: object
    ) -> None:
        existing = self.store.get(stable_key)
        if existing is None:
            self.store.append(kind, stable_key, payload)
        elif existing.kind != kind or existing.payload != payload:
            raise StableKeyConflict(
                f"stable_key {stable_key!r} already identifies another record"
            )

    def _validate_frozen_manifest(
        self,
        manifest: Mapping[str, Any],
        versions: Mapping[str, object] | None = None,
        observability: Mapping[str, object] | None = None,
    ) -> None:
        for field, current in (
            ("context_strategy", self.config.context_strategy),
            ("validator_mode", self.config.validator_mode),
            ("model_timeout_seconds", self.config.model_timeout_seconds),
            ("query_sandbox_requested", self.config.query_sandbox_requested),
            ("query_sandbox_effective", self.config.query_sandbox_effective),
            (
                "query_sandbox_bwrap_version",
                self.config.query_sandbox_bwrap_version,
            ),
            (
                "query_sandbox_bwrap_executable_sha256",
                self.config.query_sandbox_bwrap_executable_sha256,
            ),
            (
                "observability",
                dict(
                    observability
                    or _observability_provenance(self.observability_config)
                ),
            ),
            *(versions or runtime_version_manifest()).items(),
        ):
            frozen = manifest.get(field)
            if frozen is not None and frozen != current:
                raise ValueError(
                    f"cannot resume run with different {field}: "
                    f"frozen={frozen!r}, current={current!r}"
                )

    @staticmethod
    def _state_route(state: _GraphState) -> str:
        return state.get("route", "finalize")

    def _transition(
        self,
        phase: str,
        route: str,
        pending_refs: Sequence[str] = (),
        *,
        model_input_ref: str | None = None,
        model_call_number: int | None = None,
        estimated_input_tokens: int | None = None,
    ) -> _GraphState:
        result: _GraphState = {
            "route": route,
            "phase": phase,
            "pending_refs": list(pending_refs),
            "domain_revision": self.store.domain_revision,
        }
        if model_input_ref is not None:
            result["model_input_ref"] = model_input_ref
        if model_call_number is not None:
            result["model_call_number"] = model_call_number
        if estimated_input_tokens is not None:
            result["estimated_input_tokens"] = estimated_input_tokens
        if self._decision_id:
            result["decision_id"] = self._decision_id
        return result

    async def _build_context(self, state: _GraphState) -> _GraphState:
        with context_span(
            self.tracer,
            self.observability_config,
            run_id=self._run_id,
            execution_attempt_id=self._execution_attempt_id,
            attributes={"evidence_count": len(self._evidence)},
        ) as span:
            result = await self._build_context_untraced(state)
            input_hash = result.get("model_input_ref")
            if input_hash is not None:
                span.set_attribute("input_hash", input_hash)
            context_output: dict[str, object] = {
                "model_call_number": result.get("model_call_number"),
                "estimated_input_tokens": result.get("estimated_input_tokens"),
            }
            if isinstance(input_hash, str):
                context_output["messages"] = json.loads(
                    self.artifacts.read_bytes(input_hash)
                )
            call_number = result.get("model_call_number")
            if isinstance(call_number, int):
                manifest_record = self.store.get(
                    f"context_manifest:{self._run_id}:{call_number}"
                )
                if manifest_record is not None:
                    context_output["manifest"] = manifest_record.payload
            record_span_content(
                span,
                self.observability_config,
                input_value=dict(state),
                output_value=context_output,
            )
            return result

    async def _build_context_untraced(self, state: _GraphState) -> _GraphState:
        del state
        if self._terminal_status is not None:
            return self._transition("build_context:terminal", "finalize")
        if self.artifacts.bytes_used > self._max_artifact_bytes:
            self._handle_artifact_budget(
                ArtifactBudgetExceeded(
                    limit=self._max_artifact_bytes,
                    used=self.artifacts.bytes_used,
                    requested=0,
                )
            )
            return self._transition("build_context:artifact_budget", "finalize")
        if self._deadline_expired():
            self._exhaust_budget("deadline")
            return self._transition("build_context:deadline", "finalize")
        if self._apply_deferred_finalize():
            return self._transition("build_context:deferred_finalize", "finalize")
        unbounded_reasoning = self.config.context_strategy == "frontier_reflection"
        if not unbounded_reasoning and self._model_calls >= self.config.max_model_calls:
            self._exhaust_budget("model_calls")
            return self._transition("build_context:budget_exhausted", "finalize")

        call_number = self._model_calls + 1
        self._base_revision = self.store.domain_revision
        model_calls_remaining = (
            None
            if unbounded_reasoning
            else max(0, self.config.max_model_calls - self._model_calls)
        )
        provider_attempts_remaining = (
            None
            if unbounded_reasoning
            else max(0, self.config.max_model_calls - self._provider_attempts)
        )
        input_tokens_remaining = (
            None
            if unbounded_reasoning
            else max(0, self.config.max_input_tokens - self._input_tokens_used)
        )
        closeout_reasons: list[str] = []
        has_supported_open_hypothesis = any(
            hypothesis.status == "open" and hypothesis.supporting_evidence_refs
            for hypothesis in self._hypotheses.values()
        )
        now = time.time()
        deadline_seconds_remaining = max(0, int(self._deadline_at - now))
        has_typed_finding_candidate = any(
            evidence.semantic_quality == "typed"
            and evidence.evidence_kind in {"metric_profile", "service_profile"}
            and isinstance(evidence.value, Mapping)
            and isinstance(
                summary := cast(Mapping[str, object], evidence.value).get(
                    "finding_gate_summary"
                ),
                Mapping,
            )
            and (
                cast(Mapping[str, object], summary).get("fault_type_ready") is True
                and (
                    isinstance(
                        cast(Mapping[str, object], summary).get("mechanism_signature"),
                        str,
                    )
                    or cast(Mapping[str, object], summary).get(
                        "signature_free_finding_ready"
                    )
                    is True
                )
                or cast(Mapping[str, object], summary).get("material_event_ready")
                is True
                and mechanism_signature_rank(cast(Mapping[str, object], summary)) >= 3
            )
            for evidence in self._evidence.values()
        )
        has_live_hypothesis = any(
            hypothesis.status in {"open", "supported"}
            for hypothesis in self._hypotheses.values()
        )
        has_closeout_candidate = bool(
            has_live_hypothesis
            or has_typed_finding_candidate
            or self._material_anomalies
            or self._evidence
        )
        if has_supported_open_hypothesis and not unbounded_reasoning:
            assert model_calls_remaining is not None
            assert input_tokens_remaining is not None
            if model_calls_remaining <= CLOSEOUT_ROUND_BUDGET:
                closeout_reasons.append("decision_rounds")
            if input_tokens_remaining <= (
                CLOSEOUT_ROUND_BUDGET * self.config.max_context_tokens
            ):
                closeout_reasons.append("input_tokens")
        if has_closeout_candidate:
            model_timeout = self.config.model_timeout_seconds
            closeout_reserve_seconds = int(
                (model_timeout if model_timeout is not None else 60.0) + 60.0
            )
            if deadline_seconds_remaining <= closeout_reserve_seconds:
                closeout_reasons.append("wall_clock_reserve")
        paths = [
            {
                **item.model_dump(mode="json"),
                "grounding": self._grounding[item.path_id or ""],
            }
            for item in self._paths.values()
        ]
        compiled = compile_context(
            task=self.environment.task,
            tools={
                name: (
                    f"{inspect.signature(self.tools[name])}: {TOOL_DESCRIPTIONS[name]}"
                )
                for name in (*QUERY_TOOLS, QUERY_CANDIDATE_TOOL)
                if name in self.tools
            },
            budget={
                "model_calls_used": self._model_calls,
                "max_model_calls": self.config.max_model_calls,
                "provider_attempts_used": self._provider_attempts,
                "provider_attempts_remaining": provider_attempts_remaining,
                "model_calls_remaining": model_calls_remaining,
                "input_tokens_used": self._input_tokens_used,
                "max_input_tokens": self.config.max_input_tokens,
                "input_tokens_remaining": input_tokens_remaining,
                "output_tokens_used": self._output_tokens_used,
                "max_output_tokens": self.config.max_output_tokens,
                "output_tokens_remaining": (
                    None
                    if unbounded_reasoning
                    else max(
                        0, self.config.max_output_tokens - self._output_tokens_used
                    )
                ),
                "tool_attempts_used": self._tool_attempts,
                "max_tool_attempts": self.config.max_tool_attempts,
                "tool_attempts_remaining": max(
                    0, self.config.max_tool_attempts - self._tool_attempts
                ),
                "validator_retries_used": self._validator_retries,
                "max_validator_retries": (
                    None if unbounded_reasoning else self.config.max_validator_retries
                ),
                "validator_retries_remaining": (
                    None
                    if unbounded_reasoning
                    else max(
                        0,
                        self.config.max_validator_retries - self._validator_retries,
                    )
                ),
                "no_progress_rounds": self._no_progress_rounds,
                "max_no_progress_rounds": self.config.max_no_progress_rounds,
                "no_progress_rounds_remaining": (
                    None
                    if unbounded_reasoning
                    else max(
                        0,
                        self.config.max_no_progress_rounds - self._no_progress_rounds,
                    )
                ),
                "wall_clock_seconds_used": max(0, int(now - self._started_at)),
                "max_wall_clock_seconds": self.config.wall_clock_seconds,
                "deadline_seconds_remaining": deadline_seconds_remaining,
                "artifact_bytes_used": self.artifacts.bytes_used,
                "max_artifact_bytes": self._max_artifact_bytes,
                "artifact_bytes_remaining": max(
                    0, self._max_artifact_bytes - self.artifacts.bytes_used
                ),
                "closeout_required": bool(closeout_reasons),
                "closeout_reason": "+".join(closeout_reasons) or None,
            },
            revision=self._base_revision,
            observations=list(self._observations.values()),
            evidence=list(self._evidence.values()),
            hypotheses=list(self._hypotheses.values()),
            paths=paths,
            gaps=self._gaps,
            feedback=self._feedback,
            assessments=list(self._assessments.values()),
            material_anomalies=list(self._material_anomalies.values()),
            anomaly_dispositions=list(self._anomaly_dispositions.values()),
            reflected_observation_ids=sorted(self._reflected_observation_ids),
            current_best_hypothesis_id=self._current_best_hypothesis_id,
            strategy=self.config.context_strategy,
            validator_mode=self.config.validator_mode,
            max_tokens=self.config.max_context_tokens,
            topology_summary=self._topology_summary(),
            candidate_disposition_required=(
                self.config.context_strategy != "frontier_reflection"
                and self._minimum_candidate_disposition_required()
            ),
        )
        messages = compiled.messages
        input_hash = self.artifacts.put_json(messages)
        if input_hash != compiled.manifest["message_hash"]:
            raise RuntimeError("context message hash does not match persisted artifact")
        self._register_embedded_artifact(input_hash)
        self.store.append(
            "model_input",
            f"model_input:{self._run_id}:{call_number}",
            {"call_number": call_number, "content_hash": input_hash},
        )
        manifest = {**compiled.manifest, "call_number": call_number}
        self.store.append(
            "context_manifest",
            f"context_manifest:{self._run_id}:{call_number}",
            manifest,
        )
        self._feedback = []
        if compiled.status == "context_limit":
            self._terminal_status = "context_limit"
            self._record_budget("context_limit")
            return self._transition("build_context:context_limit", "finalize")
        return self._transition(
            "build_context:ready",
            "call_model",
            [
                f"model_input:{self._run_id}:{call_number}",
                f"context_manifest:{self._run_id}:{call_number}",
            ],
            model_input_ref=input_hash,
            model_call_number=call_number,
            estimated_input_tokens=cast(int, manifest["estimated_tokens"]),
        )

    async def _call_model(self, state: _GraphState) -> _GraphState:
        call_number = state.get("model_call_number", self._model_calls + 1)
        self._decision_id = f"decision:{self._run_id}:{call_number:04d}"
        existing_decision = self.store.get(self._decision_id)
        if existing_decision is not None:
            self._decision = self._decision_from_payload(existing_decision.payload)
            self._model_calls = max(self._model_calls, call_number)
            completed_key = self._budget_key(
                "model_call_completed", call_number=call_number
            )
            if self.store.get(completed_key) is None:
                response_record = self.store.get(f"model_response:{self._decision_id}")
                raw_response_payload: object = (
                    response_record.payload if response_record is not None else {}
                )
                raw_usage: object = (
                    cast(dict[str, object], raw_response_payload).get("usage", {})
                    if isinstance(raw_response_payload, dict)
                    else {}
                )
                if isinstance(raw_usage, Mapping):
                    usage = dict(cast(Mapping[str, Any], raw_usage))
                    if self._call_has_provider_attempts(call_number):
                        self._complete_audited_model_call(call_number, usage)
                    else:
                        self._accumulate_usage(usage)
                        self._settle_model_usage(call_number, usage)
            return self._transition("call_model:reused", "route", [self._decision_id])

        if (
            self.store.get(
                self._budget_key("model_call_started", call_number=call_number)
            )
            is not None
        ):
            call_number = self._model_calls + 1
            self._decision_id = f"decision:{self._run_id}:{call_number:04d}"

        if self._deadline_expired():
            self._exhaust_budget("deadline")
            return self._transition("call_model:deadline", "finalize")
        estimated_input = state.get("estimated_input_tokens")
        if not isinstance(estimated_input, int):
            manifest = self.store.get(f"context_manifest:{self._run_id}:{call_number}")
            estimated_input = (
                cast(dict[str, Any], manifest.payload).get("estimated_tokens", 0)
                if manifest is not None and isinstance(manifest.payload, dict)
                else 0
            )
        if not isinstance(estimated_input, int) or estimated_input < 0:
            raise ValueError("context estimated_tokens must be a non-negative integer")
        provider_audited = self._install_model_attempt_hook(call_number)
        if not self._reserve_model_call(
            call_number, estimated_input, reserve_tokens=not provider_audited
        ):
            if provider_audited:
                self._clear_model_attempt_hook()
            return self._transition("call_model:budget_exhausted", "finalize")
        call_started_at = time.time()
        call_started = time.perf_counter()
        usage: dict[str, Any] = {}
        try:
            input_ref = state.get("model_input_ref")
            if input_ref is None:
                raise RuntimeError("call_model reached without model_input_ref")
            raw_prompt = self.artifacts.read_bytes(input_ref).decode()
            raw_messages = json.loads(raw_prompt)
            if not isinstance(raw_messages, list):
                raise ValueError("model input artifact must contain a message list")
            messages = cast(list[dict[str, Any]], raw_messages)
            with model_span(
                self.tracer,
                self.observability_config,
                run_id=self._run_id,
                execution_attempt_id=self._execution_attempt_id,
                operation_id=f"model:{call_number}",
                model=self.config.model_id,
                prompt=raw_prompt,
            ) as span:
                try:
                    response = await asyncio.wait_for(
                        cast(Awaitable[object], self.model.decide(messages)),
                        timeout=max(0.0, self._deadline_at - time.time()),
                    )
                finally:
                    if provider_audited:
                        self._clear_model_attempt_hook()
                requests = getattr(response, "requests", ())
                if provider_audited:
                    self._reconcile_provider_attempts(call_number, requests)
                call_ended_at = time.time()
                call_duration_ms = (time.perf_counter() - call_started) * 1_000
                decision, raw_response, provider_model, usage = self._unpack_response(
                    response
                )
                raw_response = redact_sensitive(raw_response)
                response_hash = self.artifacts.put_json(raw_response)
                record_model_result(
                    span,
                    self.observability_config,
                    response=self.artifacts.read_bytes(response_hash).decode(),
                    usage=usage,
                )
                span.set_attribute("llm.model_name", provider_model)
                record_span_outcome(span, "success")
            try:
                self._persist_model_requests(call_number, requests)
            except Exception:
                if not provider_audited:
                    raise
        except asyncio.CancelledError:
            requests = getattr(self.model, "last_requests", ())
            if provider_audited:
                self._reconcile_provider_attempts(call_number, requests)
            try:
                self._persist_model_requests(call_number, requests)
            except Exception:
                if not provider_audited:
                    raise
            known_usage = aggregate_model_usage(requests)
            if not provider_audited:
                self._accumulate_usage(known_usage)
            self._record_budget("model_call_interrupted", known_usage)
            self.store.append(
                "attempt",
                f"attempt:model:{self._run_id}:{call_number}:cancelled",
                {
                    "attempt_id": f"model:{call_number}",
                    "call_number": call_number,
                    "status": "cancelled",
                    "execution_attempt_id": self._execution_attempt_id,
                    "usage": known_usage,
                    "started_at": call_started_at,
                    "ended_at": time.time(),
                    "duration_ms": (time.perf_counter() - call_started) * 1_000,
                },
            )
            raise
        except ArtifactBudgetExceeded as exc:
            known_usage = usage or aggregate_model_usage(
                getattr(self.model, "last_requests", ())
            )
            if not provider_audited:
                self._accumulate_usage(known_usage)
            self._record_budget("model_call_artifact_limit", known_usage)
            self.store.append(
                "attempt",
                f"attempt:model:{self._run_id}:{call_number}:error",
                {
                    "attempt_id": f"model:{call_number}",
                    "status": "outcome_unknown",
                    "error_type": type(exc).__name__,
                    "summary": str(exc),
                    "call_number": call_number,
                    "execution_attempt_id": self._execution_attempt_id,
                    "started_at": call_started_at,
                    "ended_at": time.time(),
                    "duration_ms": (time.perf_counter() - call_started) * 1_000,
                    "usage": known_usage,
                },
            )
            self._handle_artifact_budget(exc)
            return self._transition("call_model:artifact_budget", "finalize")
        except Exception as exc:
            requests = getattr(self.model, "last_requests", ())
            if provider_audited:
                self._reconcile_provider_attempts(call_number, requests)
            try:
                self._persist_model_requests(call_number, requests)
            except Exception:
                if not provider_audited:
                    raise
            known_usage = aggregate_model_usage(
                requests
                or (exc.responses if isinstance(exc, ModelResponseError) else ())
            )
            if not provider_audited:
                self._accumulate_usage(known_usage)
            if (
                not provider_audited
                and requests
                and all(item.get("response") is not None for item in requests)
            ):
                self._settle_model_usage(call_number, known_usage)
            else:
                # Unknown in-flight usage keeps its reservation; known usage is
                # still reported and survives recovery through the budget ledger.
                self._record_budget("model_call_failed", known_usage)
            if isinstance(exc, ModelResponseError):
                for response_index, failed_response in enumerate(exc.responses):
                    response_hash = self.artifacts.put_json(failed_response)
                    self._register_embedded_artifact(response_hash)
                    self.store.append(
                        "model_response",
                        (
                            f"model_response:{self._run_id}:{call_number}:"
                            f"invalid:{response_index}"
                        ),
                        {
                            "attempt_id": f"model:{call_number}",
                            "content_hash": response_hash,
                            "status": "invalid",
                        },
                    )
            self.store.append(
                "attempt",
                f"attempt:model:{self._run_id}:{call_number}:error",
                {
                    "attempt_id": f"model:{call_number}",
                    "status": "outcome_unknown",
                    "error_type": type(exc).__name__,
                    "summary": safe_exception_summary(exc),
                    "call_number": call_number,
                    "execution_attempt_id": self._execution_attempt_id,
                    "started_at": call_started_at,
                    "ended_at": time.time(),
                    "duration_ms": (time.perf_counter() - call_started) * 1_000,
                    "provider_request_count": len(requests) if requests else None,
                    "usage": known_usage,
                },
            )
            if self.config.context_strategy == "frontier_reflection" and isinstance(
                exc, (ModelTimeoutError, TimeoutError, ModelResponseError)
            ):
                invalid_response = isinstance(exc, ModelResponseError)
                failure_summary = safe_exception_summary(exc)
                failure_feedback = {
                    "type": (
                        "provider_invalid_response"
                        if invalid_response
                        else "provider_timeout"
                    ),
                    "call_number": call_number,
                    "summary": failure_summary,
                    "execution": "no_decision_committed",
                }
                if failure_feedback not in self._feedback:
                    self._feedback.append(failure_feedback)
                self._event(
                    "model_invalid_response" if invalid_response else "model_timeout",
                    failure_feedback,
                )
                if not self._deadline_expired() and not (
                    self._frontier_wall_closeout_required()
                ):
                    self._finish_round(False)
                    return self._transition(
                        "call_model:provider_failure_retry", "build_context"
                    )
                unresolved = sorted(
                    anomaly_id
                    for anomaly_id in self._material_anomalies
                    if anomaly_id not in self._anomaly_dispositions
                    or self._anomaly_dispositions[anomaly_id].disposition
                    == "unresolved"
                )
                self._final_request = FinalizeOperation(
                    type="finalize",
                    hypothesis_ids=[],
                    path_ids=[],
                    unresolved_questions=(
                        [
                            "Resolve the preserved material anomaly slots after "
                            "the provider becomes available."
                        ]
                        if unresolved
                        else []
                    ),
                    abstain_reason=(
                        "The model provider failed to return a schema-valid Decision "
                        "at the physical closeout boundary. "
                        if invalid_response
                        else "The model provider timed out at the physical closeout "
                        "boundary. "
                    )
                    + (
                        "Runtime emitted no Finding and preserved "
                        f"{len(unresolved)} unresolved material anomaly slot(s)."
                    ),
                )
                self._terminal_status = "completed_partial"
                self._failure = None
                self._event(
                    "runtime_physical_abstention",
                    {
                        "policy": (
                            "provider-invalid-response-safe-closeout-v1"
                            if invalid_response
                            else "provider-timeout-safe-closeout-v1"
                        ),
                        "call_number": call_number,
                        "unresolved_anomaly_ids": unresolved,
                        "model_conclusion_synthesized": False,
                    },
                )
                return self._transition(
                    "call_model:provider_failure_safe_closeout", "finalize"
                )
            if self._terminal_status != "budget_exhausted":
                self._terminal_status = (
                    "budget_exhausted"
                    if self._deadline_expired()
                    else "runtime_failure"
                )
                self._failure = safe_exception_summary(exc)
            return self._transition("call_model:failed", "finalize")

        self._register_embedded_artifact(response_hash)
        response_payload = {
            "decision_id": self._decision_id,
            "content_hash": response_hash,
            "provider_model": provider_model,
            "usage": usage,
        }
        operation_count = len(decision.operations)
        candidate_expansions: list[dict[str, Any]] = []
        candidate_expansion_error: str | None = None
        has_candidate_query = False
        if all(operation.type == "query" for operation in decision.operations):
            logical_queries = cast(list[QueryOperation], decision.operations)
            has_candidate_query = any(
                operation.tool == QUERY_CANDIDATE_TOOL for operation in logical_queries
            )
            expanded, origins, expansion_error = self._expand_candidate_queries(
                logical_queries
            )
            candidate_expansion_error = expansion_error
            if expansion_error is None and origins:
                operation_count = len(expanded)
                candidate_expansions = self._candidate_expansion_metadata(
                    expanded,
                    origins,
                    auto_fill_all=self._is_first_candidate_decision(),
                    logical_operation_count=len(logical_queries),
                )
        operation_ids = [self._operation_id(index) for index in range(operation_count)]
        decision_payload = decision.model_dump(mode="json")
        decision_payload.update(
            {
                "decision_id": self._decision_id,
                "operation_ids": operation_ids,
                "model_response_ref": response_hash,
                "runtime_base_revision": self._base_revision,
            }
        )
        if has_candidate_query:
            decision_payload["candidate_expansions"] = candidate_expansions
        if candidate_expansion_error is not None:
            decision_payload["candidate_expansion_error"] = candidate_expansion_error
        success_attempt = {
            "attempt_id": f"model:{call_number}",
            "status": "success",
            "decision_id": self._decision_id,
            "provider_model": provider_model,
            "usage": usage,
            "call_number": call_number,
            "execution_attempt_id": self._execution_attempt_id,
            "started_at": call_started_at,
            "ended_at": call_ended_at,
            "duration_ms": call_duration_ms,
            "provider_request_count": len(requests) if requests else None,
        }
        self.store.append_many(
            [
                (
                    "model_response",
                    f"model_response:{self._decision_id}",
                    response_payload,
                ),
                ("decision", self._decision_id, decision_payload),
                (
                    "attempt",
                    f"attempt:model:{self._run_id}:{call_number}:success",
                    success_attempt,
                ),
            ]
        )
        if provider_audited:
            self._complete_audited_model_call(call_number, usage)
        else:
            self._accumulate_usage(usage)
            self._settle_model_usage(call_number, usage)
        self._decision = decision
        return self._transition("call_model:committed", "route", [self._decision_id])

    async def _route(self, state: _GraphState) -> _GraphState:
        with validation_span(
            self.tracer,
            self.observability_config,
            run_id=self._run_id,
            execution_attempt_id=self._execution_attempt_id,
            attributes={
                "operation_count": len(self._decision.operations)
                if self._decision is not None
                else 0
            },
        ) as validation:
            result = await self._route_untraced(state)
            record_span_content(
                validation,
                self.observability_config,
                input_value=(
                    self._decision.model_dump(mode="json")
                    if self._decision is not None
                    else None
                ),
                output_value=result,
            )
            return result

    async def _route_untraced(self, state: _GraphState) -> _GraphState:
        del state
        if self._terminal_status is not None:
            return self._transition("route:terminal", "finalize")
        if self._deadline_expired():
            self._exhaust_budget("deadline")
            return self._transition("route:deadline", "finalize")
        if self._decision is None:
            self._terminal_status = "runtime_failure"
            self._failure = "route reached without a decision"
            return self._transition("route:failed", "finalize")
        if (
            self._decision.base_revision is not None
            and self._decision.base_revision != self._base_revision
        ):
            return self._reject_batch(
                f"base_revision must be {self._base_revision} for this round"
            )

        operations = self._decision.operations
        frontier_error = self._validate_frontier_decision(self._decision)
        if frontier_error is not None:
            if (
                frontier_error.startswith(
                    "advance_frontier requires every anomaly on the current active page"
                )
                and self._decision.reflection is not None
                and self._frontier_reflection_changes_state(self._decision.reflection)
            ):
                return self._commit_partial_frontier_certificate(frontier_error)
            return self._reject_batch(frontier_error)
        if all(operation.type == "query" for operation in operations):
            self._candidate_query_origins = {}
            logical_operations = cast(list[QueryOperation], operations)
            if self._requires_initial_overview():
                frontier_initial = bool(
                    self.config.context_strategy == "frontier_reflection"
                    and any(
                        operation.tool == "summarize_anomalies"
                        for operation in logical_operations
                    )
                    and all(
                        operation.tool
                        in {"summarize_anomalies", "list_entities", "list_metrics"}
                        for operation in logical_operations
                    )
                )
                legacy_initial = bool(
                    len(logical_operations) == 1
                    and logical_operations[0].tool == "summarize_anomalies"
                )
                if not (frontier_initial or legacy_initial):
                    overview_requirement = (
                        "include summarize_anomalies and only navigation catalogs in "
                        "the first frontier batch"
                        if self.config.context_strategy == "frontier_reflection"
                        else "submit only one summarize_anomalies operation"
                    )
                    return self._reject_batch(
                        f"broad overview required: {overview_requirement}"
                    )
            decision_record = self.store.get(self._decision_id)
            decision_payload = (
                cast(dict[str, Any], decision_record.payload)
                if decision_record is not None
                and isinstance(decision_record.payload, dict)
                else {}
            )
            if (
                any(
                    operation.tool == QUERY_CANDIDATE_TOOL
                    for operation in logical_operations
                )
                and "candidate_expansions" not in decision_payload
            ):
                self._terminal_status = "runtime_failure"
                self._failure = (
                    "persisted query_candidate decision is missing candidate_expansions"
                )
                return self._transition(
                    "route:persisted_candidate_expansion_missing", "finalize"
                )
            persisted_expansions = (
                decision_payload.get("candidate_expansions")
                if decision_payload
                else None
            )
            if any(
                operation.tool == QUERY_CANDIDATE_TOOL
                for operation in logical_operations
            ) and not isinstance(persisted_expansions, list):
                self._terminal_status = "runtime_failure"
                self._failure = "persisted candidate_expansions must be a list"
                return self._transition(
                    "route:persisted_candidate_expansion_invalid", "finalize"
                )
            if isinstance(persisted_expansions, list):
                persisted_expansions = cast(list[object], persisted_expansions)
            query_operations, origins, expansion_error = self._expand_candidate_queries(
                logical_operations, persisted_expansions=persisted_expansions
            )
            self._candidate_query_origins = origins
            if expansion_error is not None:
                recorded_expansion_error = decision_payload.get(
                    "candidate_expansion_error"
                )
                if not (
                    persisted_expansions == []
                    and recorded_expansion_error == expansion_error
                ):
                    self._terminal_status = "runtime_failure"
                    self._failure = expansion_error
                    return self._transition(
                        "route:persisted_candidate_expansion_invalid", "finalize"
                    )
                return self._reject_batch(expansion_error)
            if origins:
                pending_pairs = self._pending_metric_pair_dispositions()
                if pending_pairs:
                    return self._reject_batch(
                        "complete exact metric pairs must be handled by a supported, "
                        "refuted, or inconclusive update_hypothesis before selecting "
                        "another candidate; pending pairs: "
                        + json.dumps(pending_pairs[:1], ensure_ascii=False)
                    )
            diversity_error = self._first_metric_batch_error(
                query_operations, candidate_batch=bool(origins)
            )
            if diversity_error is not None:
                return self._reject_batch(diversity_error)
            errors: list[str] = []
            calls: list[tuple[int, QueryOperation, dict[str, Any]]] = []
            duplicates: list[tuple[int, Observation]] = []
            batch_signatures: dict[str, int] = {}
            for index, operation in enumerate(query_operations):
                if self._deadline_expired():
                    self._exhaust_budget("deadline")
                    return self._transition("route:deadline", "finalize")
                try:
                    execution_operation, arguments = self._validated_query_call(
                        operation
                    )
                except ValueError as exc:
                    errors.append(f"operation {index}: {exc}")
                    continue
                signature = self._hash_json(
                    {
                        "tool": execution_operation.tool,
                        "arguments": execution_operation.arguments,
                    }
                )
                if signature in batch_signatures:
                    errors.append(
                        f"operation {index}: duplicates operation "
                        f"{batch_signatures[signature]} after normalization"
                    )
                    continue
                batch_signatures[signature] = index
                duplicate = self._prior_query_observation(execution_operation)
                if duplicate is not None:
                    duplicates.append((index, duplicate))
                    continue
                calls.append((index, execution_operation, arguments))
            if errors:
                return self._reject_batch("; ".join(errors))
            if not calls:
                return self._recall_duplicate_batch(duplicates)
            if duplicates:
                self._record_duplicate_recall(
                    duplicates, execution="novel_operations_executed"
                )
            self._query_calls = calls
            pending = sum(
                self.store.get(f"tool_effect:{self._operation_key(index)}") is None
                for index, _operation, _arguments in calls
            )
            if self._tool_attempts + pending > self.config.max_tool_attempts:
                self._exhaust_budget("tool_attempts")
                return self._transition("route:tool_budget_exhausted", "finalize")
            self._commit_frontier_reflection(self._decision)
            return self._transition(
                "route:query",
                "execute_query",
                [self._operation_key(index) for index, _operation, _arguments in calls],
            )

        operation = operations[0]
        self._pending_candidate_disposition = None
        frontier_handle_map: dict[str, str] = {}
        if self.config.context_strategy == "frontier_reflection":
            frontier_handle_map = self._frontier_proposal_handle_map(self._decision)
            self._commit_frontier_reflection(self._decision)
            if isinstance(operation, AdvanceFrontierOperation):
                self._finish_round(True)
                return self._transition(
                    "route:advance_frontier",
                    "build_context",
                    [self._operation_key(0)],
                )
            if isinstance(operation, FinalizeOperation) and frontier_handle_map:
                operation = operation.model_copy(
                    update={
                        "hypothesis_ids": [
                            frontier_handle_map.get(hypothesis_id, hypothesis_id)
                            for hypothesis_id in operation.hypothesis_ids
                        ]
                    }
                )
        if (
            operation.type == "update_hypothesis"
            and self.config.context_strategy != "frontier_reflection"
        ):
            operation, binding_error = self._bind_candidate_disposition(operation)
            if binding_error is not None:
                return self._reject_batch(binding_error)
        error = self._validate_state_operation(operation)
        if error is not None:
            return self._reject_batch(error)
        if self.config.context_strategy != "frontier_reflection":
            self._commit_frontier_reflection(self._decision)
        self._pending_operation = operation
        routes = {
            "update_hypothesis": "apply_hypothesis",
            "submit_path": "stage_path",
            "finalize": "finalize",
        }
        return self._transition(
            f"route:{operation.type}", routes[operation.type], [self._operation_key(0)]
        )

    async def _execute_query(self, state: _GraphState) -> _GraphState:
        del state
        progress = self._query_batch_is_new()
        pending = sum(
            self.store.get(f"tool_effect:{self._operation_key(index)}") is None
            for index, _operation, _arguments in self._query_calls
        )
        if self._tool_attempts + pending > self.config.max_tool_attempts:
            self._exhaust_budget("tool_attempts")
            return self._transition(
                "execute_query:tool_budget_exhausted", "build_context"
            )
        for index, operation, arguments in self._query_calls:
            effect_key = f"tool_effect:{self._operation_key(index)}"
            if self.store.get(effect_key) is not None:
                continue
            if self._deadline_expired():
                self._exhaust_budget("deadline")
                return self._transition("execute_query:deadline", "build_context")
            operation_id = self._operation_id(index)
            attempt_id = self._start_tool_attempt(index, operation)
            tool_started_at = time.time()
            tool_started = time.perf_counter()
            candidate_audit = self._candidate_audit(index)
            result: dict[str, Any]
            with tool_span(
                self.tracer,
                self.observability_config,
                run_id=self._run_id,
                execution_attempt_id=self._execution_attempt_id,
                operation_id=self._operation_key(index),
                attributes={
                    "tool.name": operation.tool,
                    "input_hash": self._hash_json(arguments),
                    "input_count": len(
                        json.dumps(
                            arguments,
                            allow_nan=False,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        ).encode()
                    ),
                },
            ) as span:
                record_span_content(
                    span,
                    self.observability_config,
                    input_value={"tool": operation.tool, "arguments": arguments},
                )
                try:
                    result = await self._execute_tool(operation.tool, arguments)
                except Exception as exc:
                    result = {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "retryable": False,
                        "summary": str(exc),
                        "coverage": self._unknown_coverage(),
                        "artifact_refs": [],
                    }
                result_hash = self.artifacts.put_json(result)
                span.set_attribute("output_hash", result_hash)
                span.set_attribute(
                    "output_count", len(self.artifacts.read_bytes(result_hash))
                )
                record_span_content(
                    span,
                    self.observability_config,
                    output_value=result,
                )
                record_span_outcome(
                    span, str(result.get("status", "error")), result.get("error_type")
                )
            tool_ended_at = time.time()
            tool_duration_ms = (time.perf_counter() - tool_started) * 1_000
            self._register_embedded_artifact(result_hash)
            source_refs = [
                ArtifactRef.model_validate(item)
                for item in cast(list[object], result.get("artifact_refs", []))
            ]
            for source_ref in source_refs:
                self._register_external_artifact(source_ref.content_hash)
            payload_ref = ArtifactRef(
                content_hash=result_hash,
                selector={"operation_id": self._operation_key(index)},
            )
            observation_id = self._entity_id("observation", index)
            status = str(result.get("status", "error"))
            coverage = Coverage.model_validate_json(
                json.dumps(result.get("coverage", self._unknown_coverage()))
            )
            gap_message = self._gap_message(status, result, coverage)
            raw_summary = result.get("summary")
            summary = (
                json.dumps(
                    self._evidence_value(result),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                if operation.tool in NAVIGATION_TOOLS
                else raw_summary
                if isinstance(raw_summary, str) or raw_summary is None
                else json.dumps(raw_summary, sort_keys=True, separators=(",", ":"))
            )
            observation = Observation.model_validate(
                {
                    "observation_id": observation_id,
                    "tool": operation.tool,
                    "arguments": operation.arguments,
                    "status": status,
                    "coverage": coverage,
                    "artifact_refs": [payload_ref, *source_refs],
                    "attempt_id": attempt_id,
                    "summary": summary,
                    "error_type": result.get("error_type"),
                    "retryable": result.get("retryable"),
                    "extraction_gap": gap_message,
                }
            )
            evidence_id: str | None = None
            evidence: Evidence | None = None
            gap: dict[str, Any] | None = None
            disposition = classify_observation(observation)
            records: list[tuple[str, str, object]] = [
                (
                    "query_payload",
                    f"query_payload:{self._operation_key(index)}",
                    {
                        "operation_id": operation_id,
                        "content_hash": result_hash,
                        "tool": operation.tool,
                        **candidate_audit,
                    },
                ),
                ("observation", observation_id, observation),
            ]
            registered_anomalies = list(material_anomalies([observation]))
            frontier_proposal_evidence = bool(
                self.config.context_strategy == "frontier_reflection"
                and operation.tool == "summarize_anomalies"
            )
            if operation.tool in NAVIGATION_TOOLS and not frontier_proposal_evidence:
                pass
            elif disposition in {"observation_only", "negative_fact"}:
                evidence = self._extract_evidence(
                    index,
                    operation,
                    result,
                    observation,
                    payload_ref,
                    source_refs,
                    negative_fact=disposition == "negative_fact",
                )
                evidence_id = evidence.evidence_id
                records.append(("evidence", evidence.evidence_id, evidence))
            else:
                gap = {
                    "observation_id": observation_id,
                    "tool": operation.tool,
                    "status": observation.status,
                    "reason": gap_message
                    or "no trustworthy positive evidence was extracted",
                }
                records.append(
                    (
                        "investigation_event",
                        self._event_key("gap", index),
                        {"event_type": "gap", **gap},
                    )
                )
            if evidence is not None:
                registered_anomalies.extend(material_anomalies_from_evidence(evidence))
            registered_anomalies = [
                self._merge_material_anomaly(item) for item in registered_anomalies
            ]
            records.extend(
                (
                    "material_anomaly",
                    f"material_anomaly:{item.anomaly_id}:{observation_id}",
                    item,
                )
                for item in registered_anomalies
            )
            records.extend(
                [
                    (
                        "tool_effect",
                        effect_key,
                        {
                            "operation_id": operation_id,
                            "observation_id": observation_id,
                            "evidence_id": evidence_id,
                            "status": observation.status,
                            "call_number": self._model_calls,
                            "decision_id": self._decision_id,
                            "execution_attempt_id": self._execution_attempt_id,
                            "started_at": tool_started_at,
                            "ended_at": tool_ended_at,
                            "duration_ms": tool_duration_ms,
                            "error_type": observation.error_type,
                            **candidate_audit,
                        },
                    ),
                    (
                        "attempt",
                        f"{attempt_id}:finish",
                        {
                            "attempt_id": attempt_id,
                            "operation_id": operation_id,
                            "status": observation.status,
                            "call_number": self._model_calls,
                            "decision_id": self._decision_id,
                            "execution_attempt_id": self._execution_attempt_id,
                            "started_at": tool_started_at,
                            "ended_at": tool_ended_at,
                            "duration_ms": tool_duration_ms,
                            "error_type": observation.error_type,
                            "arguments": arguments,
                            **candidate_audit,
                        },
                    ),
                ]
            )
            automatic_no_information_reflection = bool(
                evidence is None
                and not registered_anomalies
                and gap is not None
                and observation.status
                in {"no_data", "error", "cancelled", "unsupported"}
            )
            if automatic_no_information_reflection:
                records.append(
                    (
                        "investigation_event",
                        f"event:{operation_id}:automatic_no_information_reflection",
                        {
                            "event_type": "automatic_no_information_reflection",
                            "observation_id": observation_id,
                            "tool": operation.tool,
                            "status": observation.status,
                            "reason": "no Evidence or material anomaly was produced",
                            "policy": "gap-only-no-state-change-v1",
                        },
                    )
                )
            with generic_span(
                self.tracer,
                "rootwalk.evidence",
                self.observability_config,
                run_id=self._run_id,
                execution_attempt_id=self._execution_attempt_id,
                operation_id=self._operation_key(index),
                attributes={
                    "record_count": len(records),
                    "observation_count": 1,
                    "evidence_count": int(evidence is not None),
                },
            ) as evidence_span:
                record_span_content(
                    evidence_span,
                    self.observability_config,
                    input_value=observation.model_dump(mode="json"),
                    output_value={
                        "evidence": (
                            evidence.model_dump(mode="json")
                            if evidence is not None
                            else None
                        ),
                        "gap": gap,
                    },
                )
                self.store.append_many(records)
                self._observations[observation_id] = observation
                if automatic_no_information_reflection:
                    self._reflected_observation_ids.add(observation_id)
                self._material_anomalies.update(
                    {item.anomaly_id: item for item in registered_anomalies}
                )
                if evidence is not None:
                    self._evidence[evidence.evidence_id] = evidence
                elif gap is not None:
                    self._gaps.append(gap)
        self._finish_round(progress)
        return self._transition(
            "execute_query:committed",
            "build_context",
            [
                observation_id
                for index, _operation, _arguments in self._query_calls
                for observation_id in [self._entity_id("observation", index)]
                if observation_id in self._observations
            ],
        )

    async def _execute_tool(
        self, tool: str, arguments: Mapping[str, Any]
    ) -> dict[str, Any]:
        local_artifact = (
            tool == "read_artifact_slice"
            and self.store.get(f"artifact:{arguments.get('artifact_id', '')}")
            is not None
        )
        if self.query_executor is None or local_artifact:
            return self.tools[tool](**dict(arguments))
        remaining = max(0.001, self._deadline_at - time.time())
        return await asyncio.to_thread(
            self.query_executor.execute,
            tool,
            arguments,
            min(self.config.tool_timeout_seconds, remaining),
        )

    def _unsupported_query_impact_profile(self) -> dict[str, Any]:
        return self._unsupported_optional_tool("query_impact_profile")

    def _unsupported_query_trace_operation_profile(
        self, entity_id: str, limit: int = 20
    ) -> dict[str, Any]:
        del entity_id, limit
        return self._unsupported_optional_tool("query_trace_operation_profile")

    def _unsupported_optional_tool(self, tool: str) -> dict[str, Any]:
        return {
            "status": "unsupported",
            "error_type": "unsupported_tool",
            "retryable": False,
            "summary": f"runtime environment does not implement optional tool {tool}",
            "coverage": self._unknown_coverage(),
            "artifact_refs": [],
        }

    def _merge_material_anomaly(self, item: MaterialAnomaly) -> MaterialAnomaly:
        existing = self._material_anomalies.get(item.anomaly_id)
        if existing is None:
            compatible = [
                candidate
                for candidate in self._material_anomalies.values()
                if candidate.subject_id == item.subject_id
                and set(candidate.mechanism_candidates) & set(item.mechanism_candidates)
            ]
            if len(compatible) == 1:
                existing = compatible[0]
        if existing is None:
            return item
        facts: dict[str, Any] = dict(existing.priority_facts)
        for name, value in item.priority_facts.items():
            prior = facts.get(name)
            if isinstance(value, bool):
                facts[name] = bool(value or prior)
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                facts[name] = max(
                    float(value),
                    float(prior)
                    if isinstance(prior, (int, float)) and not isinstance(prior, bool)
                    else 0.0,
                )
            else:
                facts[name] = value
        return existing.model_copy(
            update={
                "signal_ids": sorted({*existing.signal_ids, *item.signal_ids}),
                "mechanism_candidates": sorted(
                    {
                        *existing.mechanism_candidates,
                        *item.mechanism_candidates,
                    }
                ),
                "source_observation_ids": sorted(
                    {
                        *existing.source_observation_ids,
                        *item.source_observation_ids,
                    }
                ),
                "proposal_ids": sorted({*existing.proposal_ids, *item.proposal_ids}),
                "priority_facts": facts,
            }
        )

    def _query_candidate(self, candidate_id: str) -> dict[str, Any]:
        del candidate_id
        raise RuntimeError("query_candidate must be expanded before execution")

    async def _apply_hypothesis(self, state: _GraphState) -> _GraphState:
        del state
        operation = cast(UpdateHypothesisOperation, self._pending_operation)
        statement = operation.statement
        if operation.status == "supported":
            public_fault_type = self._resolve_public_fault_type(
                statement, self._public_fault_types()
            )
            if public_fault_type is not None:
                statement = public_fault_type
        if operation.action == "create":
            hypothesis_id = self._entity_id("hypothesis", 0)
        else:
            hypothesis_id = cast(str, operation.hypothesis_id)
        stable_key = f"hypothesis:{hypothesis_id}:{self._decision_id}"
        progress = self._hypothesis_is_progress(operation, stable_key)
        if self.store.get(stable_key) is not None:
            self._finish_round(progress)
            return self._transition(
                "apply_hypothesis:reused", "build_context", [hypothesis_id]
            )
        hypothesis = Hypothesis(
            hypothesis_id=hypothesis_id,
            subject=operation.subject,
            statement=statement,
            status=operation.status,
            supporting_evidence_refs=operation.supporting_evidence_refs,
            opposing_evidence_refs=operation.opposing_evidence_refs,
            open_questions=operation.open_questions,
            created_from=[
                *operation.supporting_evidence_refs,
                *operation.opposing_evidence_refs,
            ],
        )
        records: list[tuple[str, str, object]] = [
            ("hypothesis", stable_key, hypothesis)
        ]
        if self._pending_candidate_disposition is not None:
            disposition_event = {
                **self._pending_candidate_disposition,
                "hypothesis_id": hypothesis_id,
            }
            records.append(
                (
                    "investigation_event",
                    self._event_key("candidate_disposition_applied"),
                    disposition_event,
                )
            )
        quality_errors = (
            self._hypothesis_corroboration_errors(
                operation.subject,
                statement,
                operation.supporting_evidence_refs,
            )
            if operation.status == "supported"
            else []
        )
        if self.config.validator_mode == "observe_only" and quality_errors:
            records.append(
                (
                    "investigation_event",
                    self._event_key("hypothesis_quality_audit"),
                    {
                        "event_type": "hypothesis_quality_audit",
                        "visibility": "audit_only",
                        "hypothesis_id": hypothesis_id,
                        "errors": quality_errors,
                    },
                )
            )
        if statement != operation.statement:
            records.append(
                (
                    "investigation_event",
                    self._event_key("public_vocabulary_normalized"),
                    {
                        "event_type": "public_vocabulary_normalized",
                        "original_statement_hash": hashlib.sha256(
                            operation.statement.encode()
                        ).hexdigest(),
                        "selected_public_fault_type": statement,
                    },
                )
            )
        with generic_span(
            self.tracer,
            "rootwalk.investigation_graph",
            self.observability_config,
            run_id=self._run_id,
            execution_attempt_id=self._execution_attempt_id,
            operation_id=self._operation_key(0),
            attributes={"node_count": 1},
        ) as investigation_span:
            record_span_content(
                investigation_span,
                self.observability_config,
                input_value=operation.model_dump(mode="json"),
                output_value=hypothesis.model_dump(mode="json"),
            )
            self.store.append_many(records)
        self._hypotheses[hypothesis_id] = hypothesis
        self._event(
            "hypothesis_applied",
            {"hypothesis_id": hypothesis_id, "action": operation.action},
        )
        self._finish_round(progress)
        return self._transition(
            "apply_hypothesis:committed", "build_context", [hypothesis_id]
        )

    async def _stage_path(self, state: _GraphState) -> _GraphState:
        del state
        operation = cast(SubmitPathOperation, self._pending_operation)
        path_id = self._entity_id("path", 0)
        stable_key = f"path:{path_id}"
        progress = self._path_is_new(operation, stable_key)
        if self.store.get(stable_key) is not None:
            self._finish_round(progress)
            return self._transition("stage_path:reused", "build_context", [path_id])
        path = CausalPath(
            path_id=path_id,
            hypothesis_id=operation.hypothesis_id,
            root_cause_entity_id=operation.root_cause_entity_id,
            symptom_entity_ids=operation.symptom_entity_ids,
            steps=operation.steps,
        )
        hypothesis = self._hypotheses[path.hypothesis_id]
        claim_evidence_refs = {
            *hypothesis.supporting_evidence_refs,
            *hypothesis.opposing_evidence_refs,
        }
        grounding = ground_path(
            path,
            self.environment.topology,
            self._evidence.values(),
            self._observations.values(),
            visible_evidence_ids=set(self._evidence),
            claim_evidence_refs=claim_evidence_refs,
        )
        grounding_payload = asdict(grounding)
        comparison_errors = self._comparison_quality_errors([operation.hypothesis_id])
        path_records: list[tuple[str, str, object]] = [
            (
                "path",
                stable_key,
                {**path.model_dump(mode="json"), "grounding": grounding_payload},
            )
        ]
        if self.config.validator_mode == "observe_only" and comparison_errors:
            path_records.append(
                (
                    "investigation_event",
                    self._event_key("comparison_quality_audit"),
                    {
                        "event_type": "comparison_quality_audit",
                        "visibility": "audit_only",
                        "gate": "submit_path",
                        "hypothesis_ids": [operation.hypothesis_id],
                        "errors": comparison_errors,
                    },
                )
            )
        with generic_span(
            self.tracer,
            "rootwalk.investigation_graph",
            self.observability_config,
            run_id=self._run_id,
            execution_attempt_id=self._execution_attempt_id,
            operation_id=self._operation_key(0),
            attributes={"node_count": 1},
        ) as investigation_span:
            record_span_content(
                investigation_span,
                self.observability_config,
                input_value=operation.model_dump(mode="json"),
                output_value={
                    **path.model_dump(mode="json"),
                    "grounding": grounding_payload,
                },
            )
            self.store.append_many(path_records)
        self._paths[path_id] = path
        self._grounding[path_id] = grounding_payload
        quality_errors = self._grounding_errors(grounding_payload)
        quality_feedback = [
            {
                "feedback_id": f"path_quality:{path_id}:{index}",
                "type": "path_quality",
                "path_id": path_id,
                "error": error,
            }
            for index, error in enumerate(quality_errors)
        ]
        if self.config.validator_mode == "enforce":
            self._feedback.extend(quality_feedback)
        elif quality_feedback:
            self._event(
                "path_quality_audit",
                {"visibility": "audit_only", "feedback": quality_feedback},
            )
        self._event(
            "path_staged",
            {"path_id": path_id, "grounding": grounding_payload},
        )
        self._finish_round(progress)
        return self._transition("stage_path:committed", "build_context", [path_id])

    async def _finalize(self, state: _GraphState) -> _GraphState:
        with validation_span(
            self.tracer,
            self.observability_config,
            run_id=self._run_id,
            execution_attempt_id=self._execution_attempt_id,
            operation_id=(self._operation_key(0) if self._decision_id else None),
        ) as validation:
            result = await self._finalize_untraced(state)
            record_span_content(
                validation,
                self.observability_config,
                input_value=(
                    self._pending_operation.model_dump(mode="json")
                    if self._pending_operation is not None
                    else None
                ),
                output_value=self._report or result,
            )
            return result

    async def _finalize_untraced(self, state: _GraphState) -> _GraphState:
        del state
        if self._terminal_status is not None:
            self._seal()
            return self._transition(
                "finalize:sealed", "end", [f"report:{self._run_id}"]
            )
        operation = cast(FinalizeOperation, self._pending_operation)
        auto_selected: tuple[str, list[str]] | None = None
        if operation.abstain_reason is not None:
            supported = [
                hypothesis_id
                for hypothesis_id, hypothesis in self._hypotheses.items()
                if hypothesis.status == "supported"
            ]
            publishable_paths = [
                path_id
                for path_id, path in self._paths.items()
                if path.hypothesis_id in supported
                and not self._grounding_errors(self._grounding[path_id])
            ]
            if len(supported) == 1:
                selected_paths = (
                    publishable_paths if len(publishable_paths) == 1 else []
                )
                candidate_operation = operation.model_copy(
                    update={
                        "hypothesis_ids": supported,
                        "path_ids": selected_paths,
                        "abstain_reason": None,
                    }
                )
                reference_errors, quality_errors = self._finalization_errors(
                    candidate_operation
                )
                comparison_errors = self._comparison_quality_errors(supported)
                if not reference_errors and (
                    self._strategy_e_enabled()
                    or self.config.validator_mode == "observe_only"
                    or not quality_errors + comparison_errors
                ):
                    auto_selected = supported[0], selected_paths
                    operation = candidate_operation
        reference_errors, quality_errors = self._finalization_errors(operation)
        comparison_errors = self._comparison_quality_errors(operation.hypothesis_ids)
        selected_finding = bool(
            not reference_errors
            and self._operation_selects_supported_findings(operation)
        )
        weak_rejection = (
            self._strategy_e_weak_finding(operation) if selected_finding else None
        )
        persisted_minimum_rejection = self._minimum_disposition_rejection()
        minimum_disposition_feedback = (
            self._minimum_candidate_disposition_feedback() if selected_finding else None
        )
        if minimum_disposition_feedback is not None and self._consume_validator_retry():
            if minimum_disposition_feedback not in self._feedback:
                self._feedback.append(minimum_disposition_feedback)
            if not (
                persisted_minimum_rejection is not None
                and persisted_minimum_rejection[0] == self._decision_id
            ):
                self._event(
                    "finalize_rejected",
                    {
                        "decision_id": self._decision_id,
                        "policy": minimum_disposition_feedback["policy"],
                        "errors": [minimum_disposition_feedback["required_correction"]],
                        "feedback": [minimum_disposition_feedback],
                    },
                )
            self._finish_round(True)
            return self._transition("finalize:minimum_disposition", "build_context")
        if (
            selected_finding
            and weak_rejection is not None
            and persisted_minimum_rejection is not None
            and persisted_minimum_rejection[0] != self._decision_id
            and self._minimum_candidate_disposition_incomplete()
        ):
            operation = operation.model_copy(
                update={
                    "hypothesis_ids": [],
                    "path_ids": [],
                    "abstain_reason": (
                        "minimum candidate disposition coverage remained incomplete "
                        "after one correction"
                    ),
                }
            )
            weak_rejection = None
            auto_selected = None
            quality_errors = []
            comparison_errors = []
        if weak_rejection is not None:
            self._weak_finding_rejection = weak_rejection
            operation = operation.model_copy(
                update={
                    "hypothesis_ids": [],
                    "path_ids": [],
                    "abstain_reason": (
                        "selected Finding did not satisfy the pre-registered "
                        "strong-Finding policy"
                    ),
                }
            )
            auto_selected = None
            quality_errors = []
            comparison_errors = []
        rejected_errors = reference_errors + (
            [*quality_errors, *comparison_errors]
            if self.config.validator_mode == "enforce"
            else []
        )
        deferred_intent = self._deferred_finalize_intent(
            operation,
            reference_errors,
            quality_errors,
            comparison_errors,
        )
        if (
            weak_rejection is None
            and self.config.validator_mode == "observe_only"
            and quality_errors
        ):
            self._event(
                "finalize_quality_audit",
                {"visibility": "audit_only", "errors": quality_errors},
            )
        if (
            weak_rejection is None
            and self.config.validator_mode == "observe_only"
            and comparison_errors
        ):
            self._event(
                "comparison_quality_audit",
                {
                    "visibility": "audit_only",
                    "gate": "finalize",
                    "hypothesis_ids": operation.hypothesis_ids,
                    "errors": comparison_errors,
                },
            )
        if rejected_errors:
            feedback = [
                {
                    "feedback_id": f"exit_gate:{self._decision_id}:{index}",
                    "type": "validation_error" if reference_errors else "exit_gate",
                    "category": "authorization" if reference_errors else "quality",
                    "error": error,
                }
                for index, error in enumerate(rejected_errors)
            ]
            self._feedback.extend(
                item for item in feedback if item not in self._feedback
            )
            rejected_record = (
                "investigation_event",
                self._event_key("finalize_rejected"),
                {
                    "event_type": "finalize_rejected",
                    "errors": rejected_errors,
                    "feedback": feedback,
                },
            )
            records = [rejected_record]
            if deferred_intent is not None:
                records.append(
                    (
                        "investigation_event",
                        self._event_key("deferred_finalize_intent"),
                        deferred_intent,
                    )
                )
            self.store.append_many(records)
            if not self._continue_after_validation_rejection():
                self._seal()
                return self._transition(
                    "finalize:no_progress",
                    "end",
                    [f"report:{self._run_id}"],
                )
            return self._transition("finalize:rejected", "build_context")

        if auto_selected is not None:
            self._event(
                "finalization_auto_selected",
                {
                    "hypothesis_id": auto_selected[0],
                    "path_ids": auto_selected[1],
                    "reason": "unique supported hypothesis",
                },
            )
        self._final_request = operation
        selected_evidence = {
            evidence_id
            for hypothesis_id in operation.hypothesis_ids
            for evidence_id in (
                *self._hypotheses[hypothesis_id].supporting_evidence_refs,
                *self._hypotheses[hypothesis_id].opposing_evidence_refs,
            )
        }
        selected_supporting_evidence = {
            evidence_id
            for hypothesis_id in operation.hypothesis_ids
            for evidence_id in self._hypotheses[hypothesis_id].supporting_evidence_refs
        }
        grounded = all(
            (
                bool(operation.path_ids),
                bool(selected_evidence),
                bool(selected_supporting_evidence),
                not operation.unresolved_questions,
            )
        )
        for path_id in operation.path_ids:
            values = self._grounding[path_id]
            grounded = grounded and all(
                (
                    values["reference_grounding"] == "pass",
                    values["structural_grounding"] == "pass",
                    values["temporal_consistency"] == "consistent",
                    values["coverage"] == "complete",
                )
            )
        grounded = grounded and all(
            self._evidence[evidence_id].observation_refs
            and all(
                self._observations[observation_id].coverage.completeness == "complete"
                for observation_id in self._evidence[evidence_id].observation_refs
            )
            for evidence_id in selected_evidence
        )
        self._terminal_status = (
            "completed_grounded" if grounded else "completed_partial"
        )
        self._seal()
        return self._transition("finalize:sealed", "end", [f"report:{self._run_id}"])

    def _validated_query_call(
        self, operation: QueryOperation
    ) -> tuple[QueryOperation, dict[str, Any]]:
        arguments = self._validate_query(operation)
        return (
            operation
            if arguments == operation.arguments
            else operation.model_copy(update={"arguments": arguments}),
            arguments,
        )

    def _validate_query(self, operation: QueryOperation) -> dict[str, Any]:
        method = self.tools.get(operation.tool)
        if method is None:
            raise ValueError(f"unknown query tool {operation.tool!r}")
        signature = inspect.signature(method)
        try:
            bound = signature.bind(**operation.arguments)
        except TypeError as exc:
            raise ValueError(f"invalid parameters for {operation.tool}: {exc}") from exc
        hints = get_type_hints(method)
        validated: dict[str, Any] = {}
        for name, value in bound.arguments.items():
            try:
                validated[name] = TypeAdapter(hints[name]).validate_python(
                    value, strict=True
                )
            except (KeyError, ValidationError) as exc:
                raise ValueError(
                    f"invalid parameter {name!r} for {operation.tool}"
                ) from exc
        if operation.tool == "query_metric":
            entity_id = validated.get("entity_id")
            metric_id = validated.get("metric_id")
            if not isinstance(metric_id, str) or not metric_id.strip():
                raise ValueError("metric_id must be a non-empty exact catalog value")
            catalog = self.environment.list_metrics(cast(str, entity_id))
            if metric_id not in cast(list[object], catalog.get("metric_ids", [])):
                raise ValueError(
                    "metric_id is not available for entity_id; use an exact value "
                    "returned by list_metrics or summarize_anomalies"
                )
        elif operation.tool == "query_metric_profile":
            entity_id = cast(str, validated.get("entity_id"))
            raw_metric_ids = validated.get("metric_ids")
            metric_ids = (
                cast(list[str], raw_metric_ids)
                if isinstance(raw_metric_ids, list)
                else []
            )
            allowed = cast(
                list[str],
                self.environment.list_metrics(entity_id).get("metric_ids", []),
            )
            invalid_metric_ids = [
                metric_id for metric_id in metric_ids if metric_id not in allowed
            ]
            if not metric_ids or invalid_metric_ids:
                raise ValueError(
                    "metric_ids must be non-empty exact values available for "
                    "entity_id; "
                    "invalid="
                    + json.dumps(invalid_metric_ids)
                    + "; allowed="
                    + json.dumps(allowed)
                )
            for peer in cast(list[str], validated.get("peer_entity_ids") or []):
                peer_result = self.environment.list_metrics(peer)
                if peer_result.get("error_type") == "invalid_entity":
                    raise ValueError("each peer_entity_id must identify a known entity")
        elif operation.tool == "query_log_samples":
            entity_id = cast(str, validated.get("entity_id"))
            log_name = validated.get("log_name")
            if log_name is not None:
                allowed = cast(
                    list[str],
                    self.environment.list_metrics(entity_id).get("log_names", []),
                )
                if log_name not in allowed:
                    queue = candidate_disposition_queue(
                        list(self._observations.values()),
                        list(self._evidence.values()),
                        list(self._hypotheses.values()),
                    )
                    active = queue[0] if queue else {}
                    raw_names = active.get("log_names")
                    candidate_names = (
                        sorted(
                            {
                                name
                                for name in cast(list[object], raw_names)
                                if isinstance(name, str) and name in allowed
                            }
                        )
                        if active.get("entity_id") == entity_id
                        and isinstance(raw_names, list)
                        else []
                    )
                    normalized = log_name.strip().casefold()
                    if normalized.endswith(".log"):
                        normalized = normalized[:-4]
                    matches = [
                        name
                        for name in candidate_names
                        if (
                            name.strip().casefold()[:-4]
                            if name.strip().casefold().endswith(".log")
                            else name.strip().casefold()
                        )
                        == normalized
                    ]
                    if len(matches) == 1:
                        validated["log_name"] = matches[0]
                    elif (
                        isinstance(validated.get("contains"), str)
                        and cast(str, validated["contains"]).strip()
                    ):
                        validated.pop("log_name")
                    else:
                        raise ValueError(
                            "log_name is not available for entity_id; use one exact "
                            "log_names value: "
                            + json.dumps(candidate_names or allowed, ensure_ascii=False)
                        )
        elif operation.tool == "summarize_log_patterns":
            entity_id = cast(str, validated.get("entity_id"))
            log_name = validated.get("log_name")
            if log_name is not None:
                allowed = cast(
                    list[str],
                    self.environment.list_metrics(entity_id).get("log_names", []),
                )
                if log_name not in allowed:
                    raise ValueError(
                        "log_name is not available for entity_id; use an exact catalog "
                        "value"
                    )
        elif operation.tool in {"query_trace_neighbors", "query_trace_edges"} and (
            validated.get("direction", "both") not in {"incoming", "outgoing", "both"}
        ):
            raise ValueError("direction must be incoming, outgoing, or both")
        elif operation.tool == "read_artifact_slice":
            artifact_id = validated.get("artifact_id")
            if (
                not isinstance(artifact_id, str)
                or len(artifact_id) != 64
                or any(character not in "0123456789abcdef" for character in artifact_id)
            ):
                raise ValueError(
                    "artifact_id must be an exact 64-character lowercase SHA-256 "
                    "from artifact_refs"
                )
        return validated

    def _validate_state_operation(self, operation: Operation) -> str | None:
        topology_entities = {
            node.canonical_id for node in self.environment.topology.nodes
        }
        if operation.type == "update_hypothesis":
            if operation.subject not in topology_entities:
                return f"subject is outside topology: {operation.subject}"
            if operation.action == "revise":
                existing = self._hypotheses.get(cast(str, operation.hypothesis_id))
                if existing is None:
                    return f"unknown hypothesis: {operation.hypothesis_id}"
                if operation.subject != existing.subject:
                    return (
                        "revise cannot change hypothesis subject; keep subject "
                        f"{existing.subject!r} or create a new hypothesis for "
                        f"{operation.subject!r}"
                    )
                public_fault_types = self._public_fault_types()
                existing_fault_type = self._resolve_public_fault_type(
                    existing.statement, public_fault_types
                )
                revised_fault_type = self._resolve_public_fault_type(
                    operation.statement, public_fault_types
                )
                if (
                    existing_fault_type is not None
                    and revised_fault_type is not None
                    and existing_fault_type != revised_fault_type
                    and self._pending_candidate_disposition is None
                ):
                    return (
                        "revise cannot change the resolved public fault type from "
                        f"{existing_fault_type!r} to {revised_fault_type!r}; create a "
                        "new hypothesis"
                    )
            if (
                operation.status == "supported"
                and not operation.supporting_evidence_refs
            ):
                return "supported hypothesis requires supporting evidence"
            refs = {
                *operation.supporting_evidence_refs,
                *operation.opposing_evidence_refs,
            }
            missing = sorted(refs - self._evidence.keys())
            if missing:
                return f"unknown evidence references: {missing}"
            pending_pairs = self._pending_metric_pair_dispositions()
            candidate_registry, has_candidate_ids = self._overview_candidate_registry()
            if pending_pairs and has_candidate_ids:
                covered_pairs = [
                    pair
                    for pair in pending_pairs
                    if refs & set(pair["reference_evidence_ids"])
                    and refs & set(pair["incident_evidence_ids"])
                ]
                fault_types_by_series: dict[tuple[str, str], list[str]] = {}
                for (
                    entity_id,
                    metric_id,
                    _family,
                    fault_type,
                ) in candidate_registry.values():
                    values = fault_types_by_series.setdefault(
                        (entity_id, metric_id), []
                    )
                    if fault_type not in values:
                        values.append(fault_type)
                resolved_fault_type = self._resolve_public_fault_type(
                    operation.statement, self._public_fault_types()
                )
                pair = covered_pairs[0] if len(covered_pairs) == 1 else None
                valid_identity = pair is not None
                if pair is not None:
                    series = (
                        cast(str, pair["entity_id"]),
                        cast(str, pair["metric_id"]),
                    )
                    fault_types = fault_types_by_series.get(series, [])
                    valid_identity = (
                        operation.subject == pair["entity_id"]
                        and resolved_fault_type in fault_types
                    )
                    if (
                        valid_identity
                        and fault_types
                        and resolved_fault_type != fault_types[0]
                        and not self._has_log_evidence(operation.subject, refs)
                        and not (
                            self._subject_log_source_state(
                                operation.subject, cast(str, pair["metric_id"])
                            )
                            == "absent"
                            and self._has_strict_metric_fallback(
                                operation.subject,
                                operation.statement,
                                operation.supporting_evidence_refs,
                            )
                        )
                    ):
                        return (
                            "alias public fault type requires same-subject successful "
                            "or partial query_log_samples evidence with a complete "
                            "paired_log_summary in candidate mode; use the primary "
                            f"fault type {fault_types[0]!r} or obtain qualifier "
                            "corroboration first"
                        )
                if not valid_identity:
                    return (
                        "update_hypothesis must address exactly one pending exact "
                        "metric pair with the matching subject, evidence from both "
                        "reference and incident, and a uniquely resolved public fault "
                        "type declared for that series; pending pairs: "
                        + json.dumps(pending_pairs[:1], ensure_ascii=False)
                    )
                if (
                    operation.status in {"refuted", "inconclusive"}
                    and self.environment.capabilities.logs
                    and self._subject_log_source_state(
                        operation.subject, cast(str, covered_pairs[0]["metric_id"])
                    )
                    == "present"
                    and not self._has_terminal_log_gap(
                        operation.subject,
                        after_hypothesis_id=(
                            operation.hypothesis_id
                            if operation.action == "revise"
                            else None
                        ),
                    )
                    and any(
                        evidence_id in self._evidence
                        and _has_actionable_percent_pressure(
                            self._evidence[evidence_id].value
                        )
                        for evidence_id in refs
                    )
                ):
                    return (
                        "a known 0..100 pressure metric with at least two "
                        "incident envelope crossings must remain open until an "
                        "exact same-subject log corroboration attempt completes"
                    )
            elif pending_pairs and not any(
                operation.subject == pair["entity_id"]
                and bool(refs & set(pair["reference_evidence_ids"]))
                and bool(refs & set(pair["incident_evidence_ids"]))
                for pair in pending_pairs
            ):
                return (
                    "update_hypothesis must address one pending exact metric pair "
                    "with the matching subject and evidence from both reference and "
                    "incident; pending pairs: "
                    + json.dumps(pending_pairs[:1], ensure_ascii=False)
                )
            navigation_refs = self._navigation_evidence_refs(
                operation.supporting_evidence_refs
            )
            if operation.status == "supported" and navigation_refs:
                return (
                    f"navigation results cannot support a hypothesis: {navigation_refs}"
                )
            corroboration_errors = self._hypothesis_corroboration_errors(
                operation.subject,
                operation.statement,
                operation.supporting_evidence_refs,
            )
            if (
                operation.status == "supported"
                and self.config.validator_mode == "enforce"
                and corroboration_errors
            ):
                return "; ".join(corroboration_errors)
            public_fault_types = self._public_fault_types()
            if (
                operation.status == "supported"
                and public_fault_types
                and self._resolve_public_fault_type(
                    operation.statement, public_fault_types
                )
                is None
            ):
                return (
                    "supported hypothesis statement must exactly match one public "
                    "fault type, or uniquely contain one after Unicode casefold and "
                    "whitespace normalization; allowed values: "
                    + json.dumps(public_fault_types, ensure_ascii=False)
                )
        elif operation.type == "submit_path":
            if operation.hypothesis_id not in self._hypotheses:
                return f"unknown hypothesis: {operation.hypothesis_id}"
            comparison_errors = self._comparison_quality_errors(
                [operation.hypothesis_id]
            )
            if self.config.validator_mode == "enforce" and comparison_errors:
                return "; ".join(comparison_errors)
            claimed_entities = {
                operation.root_cause_entity_id,
                *operation.symptom_entity_ids,
                *(step.from_entity for step in operation.steps),
                *(step.to_entity for step in operation.steps),
            }
            unknown_entities = sorted(claimed_entities - topology_entities)
            if unknown_entities:
                return f"path entities are outside topology: {unknown_entities}"
            refs = {
                evidence_id
                for step in operation.steps
                for evidence_id in (
                    *step.supporting_evidence_refs,
                    *step.opposing_evidence_refs,
                )
            }
            missing = sorted(refs - self._evidence.keys())
            if missing:
                return f"unknown evidence references: {missing}"
        return None

    def _finalization_errors(
        self, operation: FinalizeOperation
    ) -> tuple[list[str], list[str]]:
        reference_errors: list[str] = []
        quality_errors: list[str] = []
        supported_hypotheses = [
            hypothesis
            for hypothesis in self._hypotheses.values()
            if hypothesis.status == "supported"
        ]
        pending_pairs = self._pending_metric_pair_dispositions()
        if (
            operation.abstain_reason is not None
            and not supported_hypotheses
            and pending_pairs
            and self._has_bulk_candidate_schedule()
        ):
            quality_errors.append(
                "cannot abstain while a complete candidate pair remains undisposed: "
                + json.dumps(pending_pairs[0], ensure_ascii=False)
            )
        missing_hypotheses = sorted(
            set(operation.hypothesis_ids) - self._hypotheses.keys()
        )
        missing_paths = sorted(set(operation.path_ids) - self._paths.keys())
        if missing_hypotheses:
            reference_errors.append(f"unknown hypotheses: {missing_hypotheses}")
        if missing_paths:
            reference_errors.append(f"unknown paths: {missing_paths}")
        if reference_errors:
            return reference_errors, quality_errors
        if operation.occurrence_evidence_refs:
            quality_errors.extend(self._occurrence_selection_errors(operation))
            return reference_errors, quality_errors
        if (
            self.config.context_strategy == "frontier_reflection"
            and operation.abstain_reason is None
            and not operation.occurrence_evidence_refs
            and operation.unresolved_questions
        ):
            quality_errors.append(
                "formal Finding cannot retain unresolved_questions; continue "
                "investigation or abstain"
            )
        if (
            self.config.context_strategy == "frontier_reflection"
            and operation.abstain_reason is None
            and not operation.occurrence_evidence_refs
            and self.environment.task.expected_fault_count is not None
            and len(operation.hypothesis_ids)
            != self.environment.task.expected_fault_count
        ):
            quality_errors.append(
                "Finding count must match task.expected_fault_count: "
                f"expected {self.environment.task.expected_fault_count}, got "
                f"{len(operation.hypothesis_ids)}"
            )
        if (
            self.config.context_strategy == "frontier_reflection"
            and operation.abstain_reason is None
            and not operation.occurrence_evidence_refs
        ):
            quality_errors.extend(
                self._anomaly_coverage_errors(operation.hypothesis_ids)
            )
        public_fault_types = self._public_fault_types()
        for hypothesis_id in operation.hypothesis_ids:
            hypothesis = self._hypotheses[hypothesis_id]
            status = hypothesis.status
            if status != "supported":
                quality_errors.append(
                    f"hypothesis {hypothesis_id} has status {status}; "
                    "only supported hypotheses may become findings"
                )
            elif not hypothesis.supporting_evidence_refs:
                reference_errors.append(
                    f"supported hypothesis {hypothesis_id} has no supporting evidence"
                )
            elif (
                self.config.context_strategy == "frontier_reflection"
                and hypothesis.open_questions
            ):
                quality_errors.append(
                    f"hypothesis {hypothesis_id} retains open_questions; explain "
                    "them through anomaly dispositions, continue investigation, "
                    "or abstain"
                )
            else:
                navigation_refs = self._navigation_evidence_refs(
                    hypothesis.supporting_evidence_refs
                )
                if navigation_refs:
                    reference_errors.append(
                        f"hypothesis {hypothesis_id} relies on navigation results: "
                        f"{navigation_refs}"
                    )
                elif (
                    self.config.context_strategy != "frontier_reflection"
                    and public_fault_types
                    and hypothesis.statement not in public_fault_types
                ):
                    reference_errors.append(
                        f"hypothesis {hypothesis_id} statement must exactly match one "
                        "public fault type; allowed values: "
                        + json.dumps(public_fault_types, ensure_ascii=False)
                    )
                if self.config.context_strategy == "frontier_reflection":
                    quality_errors.extend(
                        f"hypothesis {hypothesis_id}: {error}"
                        for error in self._frontier_finding_errors(
                            hypothesis_id,
                            self._latest_frontier_assessments(),
                        )
                    )
        for path_id in operation.path_ids:
            path = self._paths[path_id]
            if path.hypothesis_id not in operation.hypothesis_ids:
                reference_errors.append(
                    f"path {path_id} hypothesis {path.hypothesis_id} is not selected"
                )
            quality_errors.extend(
                f"path {path_id}: {error}"
                for error in self._grounding_errors(self._grounding[path_id])
            )
        return reference_errors, quality_errors

    def _occurrence_selection_errors(self, operation: FinalizeOperation) -> list[str]:
        refs = operation.occurrence_evidence_refs
        if not refs:
            return []
        errors: list[str] = []
        if self.environment.task.requested_outputs != [
            "root cause occurrence datetime"
        ]:
            errors.append(
                "occurrence recommendation requires a task requesting only root "
                "cause occurrence datetime"
            )
        pending_profiles = self._frontier_pending_material_profile_pairs()
        if pending_profiles:
            errors.append(
                "occurrence recommendation requires complete registered material "
                f"mechanism profiling; {len(pending_profiles)} subject/fault pair(s) "
                "remain unprofiled"
            )
        if not any(
            observation.tool == "query_impact_profile"
            for observation in self._observations.values()
        ):
            errors.append(
                "occurrence recommendation requires a query_impact_profile attempt "
                "to anchor causal time ordering; no_data is acceptable"
            )
        candidate_groups = self._occurrence_candidate_groups()
        selected_group = matching_occurrence_group(candidate_groups, refs)
        if selected_group is None:
            errors.append(
                "occurrence_evidence_refs must exactly match one eligible "
                "same-subject source-backed onset group; Runtime does not choose "
                "among candidates"
            )
        if len(refs) > 8:
            errors.append(
                "occurrence recommendation accepts at most eight Evidence refs"
            )
        missing = sorted(set(refs) - self._evidence.keys())
        if missing:
            errors.append(f"unknown occurrence Evidence refs: {missing}")
            return errors
        selected = [self._evidence[evidence_id] for evidence_id in refs]
        if any(item.semantic_quality != "typed" for item in selected):
            errors.append("occurrence recommendation requires typed Evidence")
        if any(item.onset_interval is None for item in selected):
            errors.append(
                "occurrence recommendation requires source-backed onset intervals"
            )
            return errors
        subject_sets = [set(item.subject_ids) for item in selected]
        if not subject_sets or not set.intersection(*subject_sets):
            errors.append("occurrence Evidence must share one subject")
        onsets = [cast(TimeRange, item.onset_interval) for item in selected]
        overlap_start = max(item.start for item in onsets)
        overlap_end = min(item.end for item in onsets)
        if overlap_start >= overlap_end:
            errors.append("occurrence Evidence onset intervals must overlap")
        incident = self.environment.task.incident_range
        if any(
            item.start < incident.start or item.end > incident.end for item in onsets
        ):
            errors.append("occurrence Evidence must stay inside the incident window")
        if len(selected) == 1:
            evidence = selected[0]
            summary = (
                cast(Mapping[str, object], evidence.value).get("finding_gate_summary")
                if isinstance(evidence.value, Mapping)
                else None
            )
            if not isinstance(summary, Mapping) or (
                cast(Mapping[str, object], summary).get("fault_type_ready") is not True
            ):
                errors.append(
                    "one occurrence Evidence ref requires a typed fault-ready "
                    "event; otherwise provide corroborating overlapping Evidence"
                )
        return errors

    def _occurrence_candidate_groups(self) -> list[OccurrenceGroup]:
        atoms: list[OccurrenceAtom] = []
        for evidence in self._evidence.values():
            if (
                evidence.semantic_quality != "typed"
                or evidence.onset_interval is None
                or not evidence.subject_ids
            ):
                continue
            summary = (
                cast(Mapping[str, object], evidence.value).get("finding_gate_summary")
                if isinstance(evidence.value, Mapping)
                else None
            )
            values: Mapping[str, object] = (
                cast(Mapping[str, object], summary)
                if isinstance(summary, Mapping)
                else {}
            )
            atoms.append(
                OccurrenceAtom(
                    evidence_id=evidence.evidence_id,
                    subject_id=evidence.subject_ids[0],
                    start=evidence.onset_interval.start,
                    end=evidence.onset_interval.end,
                    fault_type_ready=values.get("fault_type_ready") is True,
                    material_event_ready=(
                        values.get("material_event_ready") is True
                        and isinstance(values.get("mechanism_signature"), str)
                    ),
                )
            )
        return build_occurrence_groups(atoms)

    def _deferred_finalize_intent(
        self,
        operation: FinalizeOperation,
        reference_errors: Sequence[str],
        quality_errors: Sequence[str],
        comparison_errors: Sequence[str],
    ) -> dict[str, Any] | None:
        if (
            self._unbounded_reasoning_enabled()
            or self.config.validator_mode != "enforce"
            or self.config.max_model_calls - self._provider_attempts
            > CLOSEOUT_ROUND_BUDGET
            or reference_errors
            or comparison_errors
            or operation.abstain_reason is None
            or operation.hypothesis_ids
            or operation.path_ids
            or not isinstance(self._pending_operation, FinalizeOperation)
            or operation.model_dump(mode="json")
            != self._pending_operation.model_dump(mode="json")
            or not self._decision_id
        ):
            return None

        pending_pairs = self._pending_metric_pair_dispositions()
        if len(pending_pairs) != 1:
            return None
        pair_identity = pending_pairs[0]
        expected_error = (
            "cannot abstain while a complete candidate pair remains undisposed: "
            + json.dumps(pair_identity, ensure_ascii=False)
        )
        if list(quality_errors) != [expected_error]:
            return None

        source_record = self.store.get(self._decision_id)
        if (
            source_record is None
            or source_record.kind != "decision"
            or not isinstance(source_record.payload, dict)
        ):
            return None
        source_decision = dict(cast(dict[str, Any], source_record.payload))
        try:
            persisted_decision = self._decision_from_payload(source_decision)
        except (TypeError, ValueError, ValidationError):
            return None
        if (
            len(persisted_decision.operations) != 1
            or not isinstance(persisted_decision.operations[0], FinalizeOperation)
            or persisted_decision.operations[0].model_dump(mode="json")
            != operation.model_dump(mode="json")
        ):
            return None

        return {
            "event_type": "deferred_finalize_intent",
            "visibility": "audit_only",
            "policy": "deferred-model-abstain-v1",
            "source_decision_id": self._decision_id,
            "source_decision": source_decision,
            "original_operation": operation.model_dump(mode="json"),
            "pair_identity": pair_identity,
            "pair_identity_hash": self._hash_json(pair_identity),
            "rejection_error": expected_error,
        }

    def _apply_deferred_finalize(self) -> bool:
        intent_records = [
            record
            for record in self.store.records("investigation_event")
            if isinstance(record.payload, dict)
            and record.payload.get("event_type") == "deferred_finalize_intent"
        ]
        if not intent_records:
            return False
        intent_record = intent_records[-1]
        intent = cast(dict[str, Any], intent_record.payload)
        source_decision_id = intent.get("source_decision_id")
        if (
            intent.get("policy") != "deferred-model-abstain-v1"
            or not isinstance(source_decision_id, str)
            or not source_decision_id
            or intent_record.stable_key
            != (f"event:{self._run_id}:{source_decision_id}:0:deferred_finalize_intent")
        ):
            return False

        source_decision = intent.get("source_decision")
        original_operation = intent.get("original_operation")
        pair_identity = intent.get("pair_identity")
        if (
            not isinstance(source_decision, Mapping)
            or not isinstance(original_operation, Mapping)
            or not isinstance(pair_identity, Mapping)
        ):
            return False
        source_record = self.store.get(source_decision_id)
        if (
            source_record is None
            or source_record.kind != "decision"
            or source_record.payload != dict(cast(Mapping[str, Any], source_decision))
        ):
            return False

        try:
            decision = self._decision_from_payload(source_record.payload)
            operation = FinalizeOperation.model_validate(
                dict(cast(Mapping[str, Any], original_operation))
            )
        except (TypeError, ValueError, ValidationError):
            return False
        if (
            len(decision.operations) != 1
            or not isinstance(decision.operations[0], FinalizeOperation)
            or decision.operations[0].model_dump(mode="json")
            != operation.model_dump(mode="json")
            or operation.abstain_reason is None
            or operation.hypothesis_ids
            or operation.path_ids
        ):
            return False

        pair = dict(cast(Mapping[str, Any], pair_identity))
        entity_id = pair.get("entity_id")
        metric_id = pair.get("metric_id")
        reference_ids = pair.get("reference_evidence_ids")
        incident_ids = pair.get("incident_evidence_ids")
        if (
            not isinstance(entity_id, str)
            or not entity_id
            or not isinstance(metric_id, str)
            or not metric_id
            or not isinstance(reference_ids, list)
            or not reference_ids
            or not all(
                isinstance(item, str) and item
                for item in cast(list[object], reference_ids)
            )
            or not isinstance(incident_ids, list)
            or not incident_ids
            or not all(
                isinstance(item, str) and item
                for item in cast(list[object], incident_ids)
            )
            or intent.get("pair_identity_hash") != self._hash_json(pair)
        ):
            return False
        rejection_error = intent.get("rejection_error")
        rejection_prefix = (
            "cannot abstain while a complete candidate pair remains undisposed: "
        )
        if not isinstance(rejection_error, str) or not rejection_error.startswith(
            rejection_prefix
        ):
            return False
        try:
            rejected_pair = json.loads(rejection_error[len(rejection_prefix) :])
        except json.JSONDecodeError:
            return False
        if rejected_pair != pair:
            return False
        rejected_key = f"event:{self._run_id}:{source_decision_id}:0:finalize_rejected"
        rejected_record = self.store.get(rejected_key)
        if (
            rejected_record is None
            or rejected_record.kind != "investigation_event"
            or not isinstance(rejected_record.payload, dict)
            or rejected_record.payload.get("event_type") != "finalize_rejected"
            or rejected_record.payload.get("errors") != [rejection_error]
        ):
            return False

        series_bindings = [
            (record, cast(dict[str, Any], record.payload))
            for record in self.store.records("investigation_event")
            if isinstance(record.payload, dict)
            and record.payload.get("event_type") == "candidate_disposition_applied"
            and record.payload.get("subject") == entity_id
            and record.payload.get("metric_id") == metric_id
        ]
        if not series_bindings:
            return False
        binding_record, binding = series_bindings[-1]
        if (
            binding.get("reference_evidence_id") not in reference_ids
            or binding.get("incident_evidence_id") not in incident_ids
        ):
            return False
        terminal_status = binding.get("status")
        hypothesis_id = binding.get("hypothesis_id")
        if (
            terminal_status not in {"supported", "refuted", "inconclusive"}
            or not isinstance(hypothesis_id, str)
            or not hypothesis_id
        ):
            return False
        hypothesis = self._hypotheses.get(hypothesis_id)
        if (
            hypothesis is None
            or hypothesis.status != terminal_status
            or hypothesis.subject != entity_id
            or binding.get("selected_fault_type") != hypothesis.statement
        ):
            return False
        bound_refs = {
            binding.get("reference_evidence_id"),
            binding.get("incident_evidence_id"),
        }
        hypothesis_refs = {
            *hypothesis.supporting_evidence_refs,
            *hypothesis.opposing_evidence_refs,
        }
        applied_operation = binding.get("applied_operation")
        if (
            len(bound_refs) != 2
            or not all(isinstance(item, str) and item for item in bound_refs)
            or not bound_refs <= hypothesis_refs
            or not isinstance(binding.get("original_operation"), Mapping)
            or not isinstance(applied_operation, Mapping)
            or applied_operation.get("subject") != entity_id
            or applied_operation.get("statement") != hypothesis.statement
            or applied_operation.get("status") != terminal_status
        ):
            return False
        applied_refs = {
            *(
                cast(list[object], applied_operation.get("supporting_evidence_refs"))
                if isinstance(applied_operation.get("supporting_evidence_refs"), list)
                else []
            ),
            *(
                cast(list[object], applied_operation.get("opposing_evidence_refs"))
                if isinstance(applied_operation.get("opposing_evidence_refs"), list)
                else []
            ),
        }
        if not bound_refs <= applied_refs:
            return False

        if (
            any(
                item.status in {"open", "supported"}
                for item in self._hypotheses.values()
            )
            or self._pending_metric_pair_dispositions()
        ):
            return False
        reference_errors, quality_errors = self._finalization_errors(operation)
        comparison_errors = self._comparison_quality_errors(operation.hypothesis_ids)
        if reference_errors or quality_errors or comparison_errors:
            return False

        applied_key = (
            f"event:{self._run_id}:{source_decision_id}:0:deferred_finalize_applied"
        )
        self.store.append(
            "investigation_event",
            applied_key,
            {
                "event_type": "deferred_finalize_applied",
                "visibility": "audit_only",
                "policy": "deferred-model-abstain-v1",
                "intent_ref": intent_record.stable_key,
                "source_decision_id": source_decision_id,
                "disposition_ref": binding_record.stable_key,
                "original_operation": operation.model_dump(mode="json"),
                "pair_identity": pair,
                "terminal_status": terminal_status,
                "status": "completed_partial",
            },
        )
        self._final_request = operation
        self._terminal_status = "completed_partial"
        self._seal()
        return True

    @staticmethod
    def _grounding_errors(values: Mapping[str, Any]) -> list[str]:
        errors: list[str] = []
        if values["reference_grounding"] != "pass":
            errors.append("evidence reference grounding failed")
        if values["structural_grounding"] == "fail":
            errors.append("structural grounding failed")
        if values["temporal_consistency"] == "contradicted":
            errors.append("temporal consistency is contradicted")
        return errors

    def _reject_batch(self, summary: str) -> _GraphState:
        feedback = {
            "type": "validation_error",
            "decision_id": self._decision_id,
            "summary": summary,
            "execution": "zero_operations_executed",
        }
        if feedback not in self._feedback:
            self._feedback.append(feedback)
        self._event("decision_rejected", feedback)
        if not self._continue_after_validation_rejection():
            return self._transition("route:no_progress", "finalize")
        return self._transition("route:rejected", "build_context")

    def _commit_partial_frontier_certificate(self, summary: str) -> _GraphState:
        """Keep valid Reflection state while refusing an incomplete page advance."""
        assert self._decision is not None
        self._commit_frontier_reflection(self._decision)
        feedback = {
            "type": "validation_error",
            "decision_id": self._decision_id,
            "summary": summary,
            "execution": "reflection_committed_advance_not_executed",
        }
        if feedback not in self._feedback:
            self._feedback.append(feedback)
        self._event("frontier_page_certificate_partial", feedback)
        self._consume_validator_retry()
        self._finish_round(True)
        return self._transition("route:partial_frontier_certificate", "build_context")

    def _recall_duplicate_batch(
        self, duplicates: Sequence[tuple[int, Observation]]
    ) -> _GraphState:
        self._record_duplicate_recall(duplicates, execution="zero_operations_executed")
        self._finish_round(False)
        if self._terminal_status is not None:
            return self._transition("route:no_progress", "finalize")
        return self._transition("route:duplicate_recalled", "build_context")

    def _record_duplicate_recall(
        self,
        duplicates: Sequence[tuple[int, Observation]],
        *,
        execution: str,
    ) -> None:
        existing_results: list[dict[str, Any]] = []
        summaries: list[str] = []
        for index, observation in duplicates:
            evidence_index = [
                {
                    "evidence_id": evidence.evidence_id,
                    "subject_ids": evidence.subject_ids,
                    "predicate": evidence.predicate,
                }
                for evidence in self._evidence.values()
                if observation.observation_id in evidence.observation_refs
            ]
            existing_results.append(
                {
                    "operation_index": index,
                    "observation": self._duplicate_observation(observation),
                    "derived_evidence_index": evidence_index,
                    **self._candidate_audit(index),
                }
            )
            summaries.append(
                f"operation {index} reused prior observation "
                f"{observation.observation_id}"
            )
        feedback = {
            "type": "duplicate_query_recall",
            "decision_id": self._decision_id,
            "summary": "; ".join(summaries),
            "execution": execution,
            "existing_results": existing_results,
        }
        if feedback not in self._feedback:
            self._feedback.append(feedback)
        self._event("duplicate_query_recalled", feedback)

    @staticmethod
    def _duplicate_observation(observation: Observation) -> dict[str, Any]:
        payload = observation.model_dump(
            mode="json",
            include={
                "observation_id",
                "tool",
                "arguments",
                "status",
                "artifact_refs",
            },
        )
        if observation.tool in NAVIGATION_TOOLS or observation.summary is None:
            return payload
        payload["summary"] = observation.summary[:DUPLICATE_SUMMARY_MAX_CHARS]
        if len(observation.summary) > DUPLICATE_SUMMARY_MAX_CHARS:
            payload["summary_truncated"] = True
        return payload

    def _extract_evidence(
        self,
        index: int,
        operation: QueryOperation,
        result: dict[str, Any],
        observation: Observation,
        payload_ref: ArtifactRef,
        source_refs: list[ArtifactRef],
        *,
        negative_fact: bool = False,
    ) -> Evidence:
        subjects: list[str] = []
        entity_id = operation.arguments.get("entity_id")
        if isinstance(entity_id, str):
            subjects.append(entity_id)
        service_id = operation.arguments.get("service_id")
        if isinstance(service_id, str):
            attributed_members = result.get("attributed_member_entity_ids")
            if isinstance(attributed_members, list) and attributed_members:
                subjects.extend(
                    item
                    for item in cast(list[object], attributed_members)
                    if isinstance(item, str)
                )
            else:
                subjects.append(service_id)
        entity_ids = operation.arguments.get("entity_ids")
        if isinstance(entity_ids, list):
            subjects.extend(str(item) for item in entity_ids)
        if not subjects:
            subjects.extend(self.environment.task.entry_entities)
        observed = (
            observation.coverage.observed_range or observation.coverage.requested_range
        )
        onset_scope = observed
        if operation.tool in {
            "query_metric_profile",
            "query_impact_profile",
            "query_service_profile",
            "query_trace_edges",
            "query_trace_operation_profile",
        }:
            incident = self.environment.task.incident_range
            scoped_start = max(observed.start, incident.start)
            scoped_end = min(observed.end, incident.end)
            if scoped_start < scoped_end:
                onset_scope = TimeRange(start=scoped_start, end=scoped_end)
        onset_interval = self._evidence_onset(result, onset_scope)
        fingerprint = self._hash_json(
            {
                "dataset_id": self.environment.dataset_id,
                "tool": operation.tool,
                "arguments": operation.arguments,
                "source_refs": [item.content_hash for item in source_refs],
            }
        )
        return Evidence(
            evidence_id=self._entity_id("evidence", index),
            observation_refs=[observation.observation_id],
            subject_ids=list(dict.fromkeys(subjects)),
            predicate=(
                f"{operation.tool}.no_data_in_complete_healthy_range"
                if negative_fact
                else f"{operation.tool}.result"
            ),
            value=(
                {
                    "status": "no_data",
                    "scope": "requested_range",
                    "qualification": "complete healthy collection",
                }
                if negative_fact
                else self._evidence_value(result)
            ),
            observed_interval=observed,
            onset_interval=onset_interval,
            time_uncertainty=(
                "deterministic first anomalous sample; source clock and sampling "
                "uncertainty remain"
                if onset_interval is not None
                else "source coverage only; onset not inferred"
            ),
            artifact_refs=[payload_ref, *source_refs],
            extractor=Extractor(
                kind="deterministic",
                name="rootwalk.m1.query_result",
                version="1",
                parameters_hash=self._hash_json({"version": 1}),
            ),
            source_fingerprint=fingerprint,
            dataset_snapshot_id=self.environment.dataset_id,
            evidence_kind={
                "query_metric": "metric",
                "query_metric_profile": "metric_profile",
                "query_impact_profile": "impact_profile",
                "query_service_profile": "service_profile",
                "query_log_samples": "log_sample",
                "summarize_log_patterns": "log_pattern",
                "query_trace_neighbors": "topology",
                "query_trace_edges": "trace_edge",
                "query_trace_operation_profile": "trace_operation_profile",
                "summarize_anomalies": "proposal_overview",
            }.get(operation.tool, "navigation"),
            mechanism_dimension={
                "query_metric": "signal",
                "query_metric_profile": "resource_profile",
                "query_impact_profile": "business_impact",
                "query_service_profile": "service_consensus",
                "query_log_samples": "mechanism",
                "summarize_log_patterns": "mechanism",
                "query_trace_neighbors": "structural",
                "query_trace_edges": "propagation",
                "query_trace_operation_profile": "operation_profile",
            }.get(operation.tool),
            independence_group=fingerprint,
            semantic_quality=(
                "inferred"
                if operation.tool
                in {
                    "query_metric",
                    "query_log_samples",
                    "query_trace_neighbors",
                }
                else "typed"
                if operation.tool
                in {
                    "query_metric_profile",
                    "query_impact_profile",
                    "query_service_profile",
                    "summarize_log_patterns",
                    "query_trace_edges",
                    "query_trace_operation_profile",
                }
                else "unknown"
            ),
        )

    @staticmethod
    def _evidence_onset(
        result: Mapping[str, Any], observed: TimeRange
    ) -> TimeRange | None:
        candidates: list[float] = []
        gate_summary = result.get("finding_gate_summary")
        if isinstance(gate_summary, Mapping):
            gate_values = cast(Mapping[str, object], gate_summary)
            onset = gate_values.get("earliest_strong_onset_timestamp")
            if isinstance(onset, (int, float)) and not isinstance(onset, bool):
                if gate_values.get("fault_type_ready") is True or (
                    gate_values.get("material_event_ready") is True
                    and isinstance(gate_values.get("mechanism_signature"), str)
                ):
                    lower = gate_values.get(
                        "earliest_strong_onset_lower_bound_timestamp"
                    )
                    lower_timestamp = (
                        float(lower)
                        if isinstance(lower, (int, float))
                        and not isinstance(lower, bool)
                        else float(onset)
                    )
                    zone = observed.start.tzinfo or timezone.utc
                    start = datetime.fromtimestamp(
                        min(lower_timestamp, float(onset)), timezone.utc
                    ).astimezone(zone)
                    end = datetime.fromtimestamp(
                        max(lower_timestamp, float(onset)) + 1.0, timezone.utc
                    ).astimezone(zone)
                    start = max(observed.start, start)
                    end = min(observed.end, end)
                    if start < end:
                        return TimeRange(start=start, end=end)
                else:
                    candidates.append(float(onset))
        raw_profiles = result.get("metric_profiles")
        if isinstance(raw_profiles, list):
            for raw_profile in cast(list[object], raw_profiles):
                if not isinstance(raw_profile, Mapping):
                    continue
                profile = cast(Mapping[str, object], raw_profile)
                if (
                    profile.get("strong_incident_change") is not True
                    or profile.get("finding_sample_ready") is not True
                ):
                    continue
                burst = profile.get("incident_burst")
                if isinstance(burst, Mapping):
                    onset = cast(Mapping[str, object], burst).get("onset_timestamp")
                    if isinstance(onset, (int, float)) and not isinstance(onset, bool):
                        candidates.append(float(onset))
        source_intervals: list[tuple[float, float]] = []
        for collection_name in ("profiles", "operations"):
            raw_collection = result.get(collection_name)
            if not isinstance(raw_collection, list):
                continue
            for raw_item in cast(list[object], raw_collection):
                if not isinstance(raw_item, Mapping):
                    continue
                raw_interval = cast(Mapping[str, object], raw_item).get(
                    "onset_interval"
                )
                if not isinstance(raw_interval, Mapping):
                    continue
                interval = cast(Mapping[str, object], raw_interval)
                start = interval.get("start")
                end = interval.get("end")
                if (
                    isinstance(start, (int, float))
                    and not isinstance(start, bool)
                    and isinstance(end, (int, float))
                    and not isinstance(end, bool)
                    and float(start) < float(end)
                ):
                    source_intervals.append((float(start), float(end)))
        if source_intervals:
            raw_start, raw_end = min(source_intervals)
            zone = observed.start.tzinfo or timezone.utc
            start = max(
                observed.start,
                datetime.fromtimestamp(raw_start, timezone.utc).astimezone(zone),
            )
            end = min(
                observed.end,
                datetime.fromtimestamp(raw_end, timezone.utc).astimezone(zone),
            )
            if start < end:
                return TimeRange(start=start, end=end)
        for collection, field in (
            (result.get("patterns"), "first_seen"),
            (result.get("edge_profiles"), "anomaly_onset_timestamp"),
        ):
            if not isinstance(collection, list):
                continue
            for raw_item in cast(list[object], collection):
                if not isinstance(raw_item, Mapping):
                    continue
                onset = cast(Mapping[str, object], raw_item).get(field)
                if isinstance(onset, (int, float)) and not isinstance(onset, bool):
                    candidates.append(float(onset))
        if not candidates:
            return None
        zone = observed.start.tzinfo or timezone.utc
        onset = datetime.fromtimestamp(min(candidates), timezone.utc).astimezone(zone)
        if onset < observed.start or onset >= observed.end:
            return None
        end = min(observed.end, onset + timedelta(seconds=1))
        return TimeRange(start=onset, end=end)

    @staticmethod
    def _evidence_value(result: Mapping[str, Any]) -> Any:
        value: dict[str, Any] = {"status": result.get("status")}
        if result.get("summary") is not None:
            value["summary"] = result["summary"]
        paired_log_summary = _compact_paired_log_summary(
            result.get("paired_log_summary")
        )
        if paired_log_summary is not None:
            value["paired_log_summary"] = paired_log_summary
        paired_metric_summary = _compact_paired_metric_summary(
            result.get("paired_metric_summary")
        )
        if paired_metric_summary is not None:
            value["paired_metric_summary"] = paired_metric_summary
        for name in (
            "count",
            "method",
            "pattern_count",
            "omitted_pattern_count",
            "public_fault_types",
            "sample_count",
            "returned",
            "scanned_series",
            "scored_series",
            "returned_lines",
            "returned_bytes",
            "template_version",
            "trace_duration_unit",
            "trace_profile_source",
            "evidence_level",
            "evidence_role",
        ):
            if name in result:
                value[name] = result[name]
        for source, target in (("samples", "sample_count"), ("edges", "edge_count")):
            if isinstance(result.get(source), list):
                value[target] = len(cast(list[object], result[source]))
        for name in (
            "content",
            "edges",
            "entities",
            "limitations",
            "log_names",
            "metric_ids",
            "unknown_counts",
            "metric_profiles",
            "derived_metric_profiles",
            "peer_comparison",
            "peer_entity_ids",
            "service_profile_seeds",
            "service_id",
            "member_entity_ids",
            "attributed_member_entity_ids",
            "attribution_basis",
            "member_profiles",
            "trace_edge_profiles",
            "finding_gate_summary",
            "patterns",
            "edge_profiles",
            "profiles",
            "operations",
        ):
            if name in result:
                value[name] = result[name]
        if isinstance(result.get("samples"), list) and (
            result.get("log_name") or result.get("contains")
        ):
            value["samples"] = result["samples"]
        raw_public_fault_types = result.get("public_fault_types")
        public_fault_types: set[str] = set()
        if isinstance(raw_public_fault_types, list):
            public_fault_types.update(
                item
                for item in cast(list[object], raw_public_fault_types)
                if isinstance(item, str) and item
            )
        raw_fault_candidates = result.get("fault_type_candidates")
        if isinstance(raw_fault_candidates, list):
            compact_candidates: list[dict[str, Any]] = []
            seen_candidates: set[str] = set()
            for raw_candidate in cast(list[object], raw_fault_candidates):
                if len(compact_candidates) >= MAX_FAULT_TYPE_CANDIDATES:
                    break
                if not isinstance(raw_candidate, Mapping):
                    continue
                candidate = cast(Mapping[str, Any], raw_candidate)
                fault_type = candidate.get("fault_type")
                candidate_id = candidate.get("candidate_id")
                candidate_rank = candidate.get("candidate_rank")
                has_candidate_identity = (
                    candidate_id is not None or candidate_rank is not None
                )
                text_fields = {
                    name: candidate.get(name)
                    for name in (
                        "fault_type",
                        "entity_id",
                        "metric_id",
                        "resource_family",
                    )
                }
                score = candidate.get("score")
                requires_pair = candidate.get("requires_paired_exact")
                if (
                    not isinstance(fault_type, str)
                    or fault_type not in public_fault_types
                    or (
                        has_candidate_identity
                        and (
                            not isinstance(candidate_id, str)
                            or not candidate_id
                            or not isinstance(candidate_rank, int)
                            or isinstance(candidate_rank, bool)
                            or candidate_rank < 0
                        )
                    )
                    or (candidate_id or fault_type) in seen_candidates
                    or not all(
                        isinstance(item, str) and item for item in text_fields.values()
                    )
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                    or not isinstance(requires_pair, bool)
                ):
                    continue
                compact_candidate: dict[str, Any] = {
                    **text_fields,
                    "score": score,
                    "requires_paired_exact": requires_pair,
                }
                materiality = candidate.get("materiality")
                if materiality in {"material", "navigation_only"}:
                    compact_candidate["materiality"] = materiality
                    basis = candidate.get("materiality_basis")
                    if isinstance(basis, str) and basis:
                        compact_candidate["materiality_basis"] = basis
                overview_score = candidate.get("overview_score")
                if isinstance(overview_score, (int, float)) and not isinstance(
                    overview_score, bool
                ):
                    compact_candidate["overview_score"] = overview_score
                source_metric_id = candidate.get("source_metric_id")
                if isinstance(source_metric_id, str) and source_metric_id:
                    compact_candidate["source_metric_id"] = source_metric_id
                for list_field in ("metric_group", "peer_entity_ids"):
                    raw_items = candidate.get(list_field)
                    if isinstance(raw_items, list) and all(
                        isinstance(item, str) and item
                        for item in cast(list[object], raw_items)
                    ):
                        compact_candidate[list_field] = cast(list[str], raw_items)
                peer_outlier = candidate.get("peer_outlier")
                peer_group_size = candidate.get("peer_group_size")
                peer_score_ratio = candidate.get("peer_score_ratio")
                if isinstance(peer_outlier, bool):
                    compact_candidate["peer_outlier"] = peer_outlier
                if isinstance(peer_group_size, int) and not isinstance(
                    peer_group_size, bool
                ):
                    compact_candidate["peer_group_size"] = peer_group_size
                if isinstance(peer_score_ratio, (int, float)) and not isinstance(
                    peer_score_ratio, bool
                ):
                    compact_candidate["peer_score_ratio"] = peer_score_ratio
                metric_role = candidate.get("metric_role")
                if metric_role in {"direct", "ambiguous"}:
                    compact_candidate["metric_role"] = metric_role
                for direction_field in (
                    "candidate_pressure_direction",
                    "source_pressure_direction",
                ):
                    direction = candidate.get(direction_field)
                    if direction in {
                        "lower_is_more_pressure",
                        "higher_is_more_pressure",
                        "unknown",
                    }:
                        compact_candidate[direction_field] = direction
                has_log_source = candidate.get("has_log_source")
                if isinstance(has_log_source, bool):
                    compact_candidate["has_log_source"] = has_log_source
                log_names = candidate.get("log_names")
                if isinstance(log_names, list) and all(
                    isinstance(item, str) and item
                    for item in cast(list[object], log_names)
                ):
                    compact_candidate["log_names"] = sorted(cast(list[str], log_names))
                if isinstance(candidate_id, str):
                    compact_candidate["candidate_id"] = candidate_id
                    compact_candidate["candidate_rank"] = candidate_rank
                raw_aliases = candidate.get("fault_type_aliases")
                if isinstance(raw_aliases, list):
                    compact_candidate["fault_type_aliases"] = [
                        alias
                        for alias in cast(list[object], raw_aliases)
                        if isinstance(alias, str)
                        and alias in public_fault_types
                        and alias != fault_type
                    ]
                for window in ("reference", "incident"):
                    raw_stats = candidate.get(window)
                    if isinstance(raw_stats, Mapping):
                        compact_candidate[window] = {
                            name: raw_stats[name]
                            for name in ("min", "median", "max")
                            if name in raw_stats
                            and isinstance(raw_stats[name], (int, float))
                            and not isinstance(raw_stats[name], bool)
                        }
                compact_candidates.append(compact_candidate)
                seen_candidates.add(candidate_id or fault_type)
            value["fault_type_candidates"] = compact_candidates
        signals = result.get("signals")
        if isinstance(signals, list):
            compact_signals: list[dict[str, Any]] = []
            for raw_signal in cast(list[object], signals):
                if not isinstance(raw_signal, Mapping):
                    continue
                signal = cast(Mapping[str, Any], raw_signal)
                compact = {
                    name: signal[name]
                    for name in (
                        "entity_id",
                        "metric_id",
                        "resource_family",
                        "overview_score",
                        "relative_median_shift",
                        "relative_extreme_shift",
                        "stats_line",
                    )
                    if name in signal
                }
                for window in ("reference", "incident"):
                    raw_stats = signal.get(window)
                    if isinstance(raw_stats, Mapping):
                        compact[window] = {
                            name: raw_stats[name]
                            for name in ("count", "min", "median", "max")
                            if name in raw_stats
                        }
                raw_candidates = signal.get("direct_metric_candidates")
                if isinstance(raw_candidates, list):
                    compact["direct_metric_candidates"] = [
                        item
                        for item in cast(list[object], raw_candidates)
                        if isinstance(item, str)
                    ][:2]
                compact_signals.append(compact)
            value["signals"] = compact_signals
        return value

    @staticmethod
    def _gap_message(
        status: str, result: Mapping[str, Any], coverage: Coverage
    ) -> str | None:
        if status == "no_data" and all(
            (
                coverage.completeness == "complete",
                (coverage.collection_health or "").strip().lower() in {"healthy", "ok"},
                coverage.truncation is None,
            )
        ):
            return None
        if status in {"no_data", "unsupported", "error", "cancelled"}:
            return str(result.get("summary") or f"query ended with {status}")
        if result.get("gap"):
            return str(cast(Mapping[str, Any], result["gap"]).get("meaning"))
        return None

    def _restore_from_records(self, state: Mapping[str, Any] | None = None) -> None:
        self._observations = {
            item.observation_id: item
            for record in self.store.records("observation")
            for item in [Observation.model_validate_json(json.dumps(record.payload))]
        }
        self._evidence = {
            item.evidence_id: item
            for record in self.store.records("evidence")
            for item in [Evidence.model_validate_json(json.dumps(record.payload))]
        }
        self._assessments = {
            item.assessment_id: item
            for record in self.store.records("evidence_assessment")
            for item in [EvidenceAssessment.model_validate(record.payload)]
        }
        self._material_anomalies = {}
        for record in self.store.records("material_anomaly"):
            item = MaterialAnomaly.model_validate(record.payload)
            self._material_anomalies[item.anomaly_id] = item
        self._anomaly_dispositions = {}
        for record in self.store.records("anomaly_disposition"):
            item = AnomalyDisposition.model_validate(record.payload)
            self._anomaly_dispositions[item.anomaly_id] = item
        self._reflected_observation_ids = set()
        self._current_best_hypothesis_id = None
        for record in self.store.records("reflection"):
            if not isinstance(record.payload, dict):
                raise ValueError("reflection record must be an object")
            payload = cast(dict[str, Any], record.payload)
            based_on = payload.get("based_on_observation_ids")
            if isinstance(based_on, list):
                self._reflected_observation_ids.update(
                    item
                    for item in cast(list[object], based_on)
                    if isinstance(item, str)
                )
            current_best = payload.get("resolved_current_best_hypothesis_id")
            self._current_best_hypothesis_id = (
                current_best if isinstance(current_best, str) else None
            )
        self._hypotheses = {}
        for record in self.store.records("hypothesis"):
            item = Hypothesis.model_validate(record.payload)
            self._hypotheses[item.hypothesis_id] = item
        context_records = self.store.records("context_manifest")
        feedback_after = context_records[-1].revision if context_records else 0
        self._paths = {}
        self._grounding = {}
        recent_path_ids: set[str] = set()
        for record in self.store.records("path"):
            if not isinstance(record.payload, dict):
                raise ValueError("path record must be an object")
            payload = dict(cast(dict[str, Any], record.payload))
            grounding = payload.pop("grounding")
            item = CausalPath.model_validate(payload)
            path_id = cast(str, item.path_id)
            self._paths[path_id] = item
            self._grounding[path_id] = cast(dict[str, Any], grounding)
            if record.revision > feedback_after:
                recent_path_ids.add(path_id)

        self._gaps = []
        self._feedback = []
        for record in self.store.records("investigation_event"):
            if not isinstance(record.payload, dict):
                continue
            payload = cast(dict[str, Any], record.payload)
            event_type = payload.get("event_type")
            if event_type == "gap":
                self._gaps.append(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "event_type"
                    }
                )
            elif event_type == "automatic_no_information_reflection":
                observation_id = payload.get("observation_id")
                if isinstance(observation_id, str):
                    self._reflected_observation_ids.add(observation_id)
            elif event_type == "decision_rejected" and record.revision > feedback_after:
                self._feedback.append(
                    {
                        key: value
                        for key, value in payload.items()
                        if key != "event_type"
                    }
                )
            elif (
                event_type == "duplicate_query_recalled"
                and record.revision > feedback_after
            ):
                feedback = {
                    key: value for key, value in payload.items() if key != "event_type"
                }
                if feedback not in self._feedback:
                    self._feedback.append(feedback)
            elif (
                event_type == "finalize_rejected"
                and self.config.validator_mode == "enforce"
                and record.revision > feedback_after
            ):
                saved_feedback = payload.get("feedback")
                if isinstance(saved_feedback, list):
                    self._feedback.extend(
                        item
                        for raw_item in cast(list[object], saved_feedback)
                        if isinstance(raw_item, dict)
                        for item in [cast(dict[str, Any], raw_item)]
                        if item not in self._feedback
                    )
                else:
                    self._feedback.append(
                        {
                            "feedback_id": record.stable_key,
                            "type": "exit_gate",
                            "errors": payload.get("errors", []),
                        }
                    )
        if self.config.validator_mode == "enforce":
            for path_id, grounding in self._grounding.items():
                if path_id not in recent_path_ids:
                    continue
                self._feedback.extend(
                    {
                        "feedback_id": f"restored:path_quality:{path_id}:{index}",
                        "type": "path_quality",
                        "path_id": path_id,
                        "error": error,
                    }
                    for index, error in enumerate(self._grounding_errors(grounding))
                )

        for record in self.store.records("budget_ledger"):
            if not isinstance(record.payload, dict):
                continue
            payload = cast(dict[str, Any], record.payload)
            calls = payload.get("model_calls_used")
            if isinstance(calls, int):
                self._model_calls = max(self._model_calls, calls)
            cumulative = payload.get("cumulative_usage")
            if isinstance(cumulative, dict):
                self._model_usage = {
                    name: value
                    for name, value in cast(dict[str, Any], cumulative).items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            for field, attribute in (
                ("input_tokens_used", "_input_tokens_used"),
                ("output_tokens_used", "_output_tokens_used"),
                ("provider_attempts_used", "_provider_attempts"),
                ("tool_attempts_used", "_tool_attempts"),
                ("validator_retries_used", "_validator_retries"),
                ("no_progress_rounds", "_no_progress_rounds"),
            ):
                value = payload.get(field)
                if isinstance(value, int):
                    setattr(self, attribute, value)
            if payload.get("event") == "exhausted":
                self._terminal_status = "budget_exhausted"
                reason = payload.get("reason")
                self._failure = str(reason) if reason is not None else None

        if (
            not self._unbounded_reasoning_enabled()
            and self._no_progress_rounds >= self.config.max_no_progress_rounds
        ):
            self._terminal_status = "no_progress"
        elif not self._unbounded_reasoning_enabled() and (
            self._input_tokens_used > self.config.max_input_tokens
            or self._output_tokens_used > self.config.max_output_tokens
        ):
            self._terminal_status = "budget_exhausted"
            self._failure = "model_tokens"

        saved = state or {}
        decision_id = saved.get("decision_id")
        if isinstance(decision_id, str):
            record = self.store.get(decision_id)
            if record is not None:
                self._decision_id = decision_id
                self._decision = self._decision_from_payload(record.payload)
                if all(item.type == "query" for item in self._decision.operations):
                    query_operations = cast(
                        list[QueryOperation], self._decision.operations
                    )
                    decision_payload = cast(dict[str, Any], record.payload)
                    if (
                        any(
                            item.tool == QUERY_CANDIDATE_TOOL
                            for item in query_operations
                        )
                        and "candidate_expansions" not in decision_payload
                    ):
                        raise ValueError(
                            "persisted query_candidate decision is missing "
                            "candidate_expansions"
                        )
                    if any(
                        item.tool == QUERY_CANDIDATE_TOOL for item in query_operations
                    ) and not isinstance(
                        decision_payload.get("candidate_expansions"), list
                    ):
                        raise ValueError(
                            "persisted candidate_expansions must be a list"
                        )
                    operations, origins, error = self._expand_candidate_queries(
                        query_operations,
                        persisted_expansions=(
                            decision_payload.get("candidate_expansions")
                        ),
                    )
                    if error is not None:
                        raise ValueError(error)
                    self._candidate_query_origins = origins
                    self._query_calls = []
                    for index, operation in enumerate(operations):
                        if saved.get("route") == "execute_query":
                            execution_operation, arguments = self._validated_query_call(
                                operation
                            )
                        else:
                            execution_operation = operation
                            arguments = dict(operation.arguments)
                        if self._prior_query_observation(execution_operation) is None:
                            self._query_calls.append(
                                (index, execution_operation, arguments)
                            )
                else:
                    operation = self._decision.operations[0]
                    if (
                        operation.type == "update_hypothesis"
                        and saved.get("route") == "apply_hypothesis"
                    ):
                        bound, error = self._bind_candidate_disposition(operation)
                        if error is not None:
                            raise ValueError(error)
                        self._pending_operation = bound
                    else:
                        self._pending_operation = operation

        reports = self.store.records("report")
        if reports:
            report = reports[-1].payload
            if not isinstance(report, dict):
                raise ValueError("report record must be an object")
            self._report = dict(cast(dict[str, Any], report))
            self._terminal_status = cast(str, self._report["status"])

    @staticmethod
    def _decision_from_payload(payload: object) -> Decision:
        if not isinstance(payload, dict):
            raise ValueError("decision record must be an object")
        fields = Decision.model_fields
        return Decision.model_validate(
            {
                key: value
                for key, value in cast(dict[str, Any], payload).items()
                if key in fields
            }
        )

    def _unpack_response(
        self, response: object
    ) -> tuple[Decision, object, str, dict[str, Any]]:
        if isinstance(response, ModelCallResult):
            return (
                response.decision,
                response.raw_response,
                response.provider_model,
                response.usage,
            )
        if isinstance(response, Decision):
            return response, response.model_dump(mode="json"), "adapter", {}
        if isinstance(response, Mapping):
            decision = Decision.model_validate(response)
            return decision, dict(cast(Mapping[str, Any], response)), "adapter", {}
        decision = getattr(response, "decision", None)
        if isinstance(decision, Decision):
            raw = getattr(response, "raw_response", decision.model_dump(mode="json"))
            usage = getattr(response, "usage", {})
            if not isinstance(usage, Mapping):
                raise TypeError("model usage must be a mapping")
            return (
                decision,
                raw,
                str(getattr(response, "provider_model", "adapter")),
                dict(cast(Mapping[str, Any], usage)),
            )
        raise TypeError("model decide() must return a Decision or ModelCallResult")

    def _accumulate_usage(self, usage: Mapping[str, Any]) -> None:
        for name, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                self._model_usage[name] = self._model_usage.get(name, 0) + value

    def _install_model_attempt_hook(self, call_number: int) -> bool:
        setter = getattr(self.model, "set_attempt_hook", None)
        if not callable(setter):
            return False
        typed_setter = cast(
            Callable[[Callable[[str, Mapping[str, Any]], None] | None], None],
            setter,
        )
        typed_setter(
            lambda event, attempt: self._provider_attempt_event(
                call_number, event, attempt
            )
        )
        return True

    def _clear_model_attempt_hook(self) -> None:
        setter = getattr(self.model, "set_attempt_hook", None)
        if callable(setter):
            cast(Callable[[object], None], setter)(None)

    def _provider_attempt_event(
        self, call_number: int, event: str, attempt: Mapping[str, Any]
    ) -> None:
        request_index = attempt.get("request_index")
        if not isinstance(request_index, int) or isinstance(request_index, bool):
            raise ValueError("provider request_index must be an integer")
        attempt_key = f"attempt:provider:{self._run_id}:{call_number}:{request_index}"
        if event == "started":
            request: object = redact_sensitive(attempt.get("request"))
            encoded = json.dumps(
                request,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
            reserved_input = (len(encoded) + 2) // 3
            raw_max_tokens = (
                cast(Mapping[str, object], request).get("max_tokens")
                if isinstance(request, Mapping)
                else None
            )
            reserved_output: int = (
                raw_max_tokens
                if isinstance(raw_max_tokens, int)
                and not isinstance(raw_max_tokens, bool)
                and raw_max_tokens > 0
                else self.config.max_output_tokens_per_call
            )
            if (
                not self._unbounded_reasoning_enabled()
                and self._provider_attempts >= self.config.max_model_calls
            ):
                self._exhaust_budget("model_calls")
                raise RuntimeError("provider request budget exhausted")
            if not self._unbounded_reasoning_enabled() and (
                self._input_tokens_used + reserved_input > self.config.max_input_tokens
                or self._output_tokens_used + reserved_output
                > self.config.max_output_tokens
            ):
                self._exhaust_budget("model_tokens")
                raise RuntimeError("provider token budget exhausted")
            request_hash = self.artifacts.put_json(cast(object, request))
            previous = (
                self._provider_attempts,
                self._input_tokens_used,
                self._output_tokens_used,
            )
            self._provider_attempts += 1
            self._input_tokens_used += reserved_input
            self._output_tokens_used += reserved_output
            try:
                self.store.append_many(
                    [
                        (
                            "artifact",
                            f"artifact:{request_hash}",
                            {
                                "content_hash": request_hash,
                                "storage": "embedded",
                                "media_type": "application/json",
                            },
                        ),
                        (
                            "budget_ledger",
                            f"budget:{self._run_id}:provider:{call_number}:"
                            f"{request_index}:started",
                            self._budget_payload(
                                "provider_request_started",
                                call_number=call_number,
                                provider_request_index=request_index,
                                reserved_input_tokens=reserved_input,
                                reserved_output_tokens=reserved_output,
                                settlement="reserved_unknown",
                            ),
                        ),
                        (
                            "attempt",
                            f"{attempt_key}:start",
                            {
                                "attempt_id": attempt_key,
                                "call_number": call_number,
                                "provider_request_index": request_index,
                                "status": "started",
                                "execution_attempt_id": self._execution_attempt_id,
                                "model_input_ref": request_hash,
                                "reserved_input_tokens": reserved_input,
                                "reserved_output_tokens": reserved_output,
                                "started_at": attempt.get("started_at"),
                            },
                        ),
                    ]
                )
            except Exception:
                (
                    self._provider_attempts,
                    self._input_tokens_used,
                    self._output_tokens_used,
                ) = previous
                raise
            return
        if event != "finished":
            raise ValueError("provider attempt event must be started or finished")
        finish_key = f"{attempt_key}:finish"
        if self.store.get(finish_key) is not None:
            self._persist_model_requests(call_number, [attempt])
            return
        started = self.store.get(f"{attempt_key}:start")
        if started is None or not isinstance(started.payload, dict):
            raise RuntimeError("provider attempt finished without a reservation")
        started_payload = cast(dict[str, Any], started.payload)
        reserved_input = cast(int, started_payload["reserved_input_tokens"])
        reserved_output = cast(int, started_payload["reserved_output_tokens"])
        raw_usage = attempt.get("usage")
        usage = (
            dict(cast(Mapping[str, Any], raw_usage))
            if isinstance(raw_usage, Mapping)
            else {}
        )
        actual_input = self._usage_tokens(usage, "input_tokens", "prompt_tokens")
        actual_output = self._usage_tokens(usage, "output_tokens", "completion_tokens")
        previous_input = self._input_tokens_used
        previous_output = self._output_tokens_used
        previous_usage = dict(self._model_usage)
        if actual_input is not None:
            self._input_tokens_used += actual_input - reserved_input
        if actual_output is not None:
            self._output_tokens_used += actual_output - reserved_output
        self._accumulate_usage(usage)
        finish_payload = {
            key: redact_sensitive(value)
            for key, value in attempt.items()
            if key not in {"request", "response"}
        }
        finish_payload.update(
            attempt_id=attempt_key,
            call_number=call_number,
            provider_request_index=request_index,
            execution_attempt_id=self._execution_attempt_id,
        )
        try:
            self.store.append_many(
                [
                    (
                        "budget_ledger",
                        f"budget:{self._run_id}:provider:{call_number}:"
                        f"{request_index}:finished",
                        self._budget_payload(
                            "provider_request_finished",
                            usage,
                            call_number=call_number,
                            provider_request_index=request_index,
                            reserved_input_tokens=reserved_input,
                            reserved_output_tokens=reserved_output,
                            charged_input_tokens=(
                                actual_input
                                if actual_input is not None
                                else reserved_input
                            ),
                            charged_output_tokens=(
                                actual_output
                                if actual_output is not None
                                else reserved_output
                            ),
                            settlement=(
                                "actual"
                                if actual_input is not None
                                and actual_output is not None
                                else "reserved_unknown"
                            ),
                        ),
                    ),
                    ("attempt", finish_key, finish_payload),
                ]
            )
        except Exception:
            self._input_tokens_used = previous_input
            self._output_tokens_used = previous_output
            self._model_usage = previous_usage
            raise
        self._persist_model_requests(call_number, [attempt])
        if not self._unbounded_reasoning_enabled() and (
            self._input_tokens_used > self.config.max_input_tokens
            or self._output_tokens_used > self.config.max_output_tokens
        ):
            self._exhaust_budget("model_tokens")

    def _reconcile_provider_attempts(
        self, call_number: int, attempts: Sequence[Mapping[str, Any]]
    ) -> None:
        for attempt in attempts:
            request_index = attempt.get("request_index")
            if not isinstance(request_index, int) or isinstance(request_index, bool):
                continue
            finish_key = (
                f"attempt:provider:{self._run_id}:{call_number}:{request_index}:finish"
            )
            if self.store.get(finish_key) is not None:
                try:
                    self._persist_model_requests(call_number, [attempt])
                except Exception:
                    pass
                continue
            try:
                self._provider_attempt_event(call_number, "finished", attempt)
            except Exception:
                raw_usage = attempt.get("usage")
                if self.store.get(finish_key) is None and isinstance(
                    raw_usage, Mapping
                ):
                    self._accumulate_usage(cast(Mapping[str, Any], raw_usage))

    def _persist_model_requests(
        self, call_number: int, requests: Sequence[Mapping[str, Any]]
    ) -> None:
        for fallback_index, request in enumerate(requests, 1):
            raw_index = request.get("request_index")
            index = (
                raw_index
                if isinstance(raw_index, int)
                and not isinstance(raw_index, bool)
                and raw_index > 0
                else fallback_index
            )
            payload = {
                key: redact_sensitive(value)
                for key, value in request.items()
                if key not in {"request", "response"}
            }
            for field, reference in (
                ("request", "model_input_ref"),
                ("response", "model_response_ref"),
            ):
                if request.get(field) is not None:
                    content_hash = self.artifacts.put_json(
                        redact_sensitive(request[field])
                    )
                    self._register_embedded_artifact(content_hash)
                    payload[reference] = content_hash
            payload.update(
                call_number=call_number,
                attempt_id=f"model:{call_number}",
                execution_attempt_id=self._execution_attempt_id,
            )
            self.store.append(
                "model_request",
                f"model_request:{self._run_id}:{call_number}:{index}",
                payload,
            )

    def _reserve_model_call(
        self, call_number: int, estimated_input: int, *, reserve_tokens: bool = True
    ) -> bool:
        key = self._budget_key("model_call_started", call_number=call_number)
        if self.store.get(key) is not None:
            return True
        reserved_input = estimated_input if reserve_tokens else 0
        reserved_output = (
            self.config.max_output_tokens_per_call if reserve_tokens else 0
        )
        if not self._unbounded_reasoning_enabled() and (
            self._model_calls >= self.config.max_model_calls
            or self._input_tokens_used + reserved_input > self.config.max_input_tokens
            or self._output_tokens_used + reserved_output
            > self.config.max_output_tokens
        ):
            self._exhaust_budget(
                "model_calls"
                if self._model_calls >= self.config.max_model_calls
                else "model_tokens"
            )
            return False
        self._model_calls = max(self._model_calls, call_number)
        self._input_tokens_used += reserved_input
        self._output_tokens_used += reserved_output
        self._record_budget(
            "model_call_started",
            call_number=call_number,
            reserved_input_tokens=reserved_input,
            reserved_output_tokens=reserved_output,
            settlement="reserved_unknown",
        )
        self.store.append(
            "attempt",
            f"attempt:model:{self._run_id}:{call_number}:start",
            {
                "attempt_id": f"model:{call_number}",
                "status": "started",
                "reserved_input_tokens": reserved_input,
                "reserved_output_tokens": reserved_output,
                "call_number": call_number,
                "execution_attempt_id": self._execution_attempt_id,
                "started_at": time.time(),
            },
        )
        return True

    def _settle_model_usage(self, call_number: int, usage: Mapping[str, Any]) -> None:
        completed_key = self._budget_key(
            "model_call_completed", call_number=call_number
        )
        if self.store.get(completed_key) is not None:
            return
        started = self.store.get(
            self._budget_key("model_call_started", call_number=call_number)
        )
        if started is None or not isinstance(started.payload, dict):
            raise RuntimeError("model usage cannot be settled without a reservation")
        reserved_input = cast(dict[str, Any], started.payload).get(
            "reserved_input_tokens", 0
        )
        reserved_output = cast(dict[str, Any], started.payload).get(
            "reserved_output_tokens", 0
        )
        if not isinstance(reserved_input, int) or not isinstance(reserved_output, int):
            raise ValueError("model reservation tokens must be integers")
        actual_input = self._usage_tokens(usage, "input_tokens", "prompt_tokens")
        actual_output = self._usage_tokens(usage, "output_tokens", "completion_tokens")
        if actual_input is not None:
            self._input_tokens_used += actual_input - reserved_input
        if actual_output is not None:
            self._output_tokens_used += actual_output - reserved_output
        self._record_budget(
            "model_call_completed",
            usage,
            call_number=call_number,
            reserved_input_tokens=reserved_input,
            reserved_output_tokens=reserved_output,
            charged_input_tokens=(
                actual_input if actual_input is not None else reserved_input
            ),
            charged_output_tokens=(
                actual_output if actual_output is not None else reserved_output
            ),
            settlement=(
                "actual"
                if actual_input is not None and actual_output is not None
                else "reserved_unknown"
            ),
        )
        if not self._unbounded_reasoning_enabled() and (
            self._input_tokens_used > self.config.max_input_tokens
            or self._output_tokens_used > self.config.max_output_tokens
        ):
            self._exhaust_budget("model_tokens")

    def _complete_audited_model_call(
        self, call_number: int, usage: Mapping[str, Any]
    ) -> None:
        completed_key = self._budget_key(
            "model_call_completed", call_number=call_number
        )
        if self.store.get(completed_key) is not None:
            return
        self._record_budget(
            "model_call_completed",
            usage,
            call_number=call_number,
            charged_input_tokens=0,
            charged_output_tokens=0,
            settlement="provider_attempts",
        )

    def _call_has_provider_attempts(self, call_number: int) -> bool:
        return any(
            isinstance(record.payload, dict)
            and cast(dict[str, Any], record.payload).get("event")
            == "provider_request_started"
            and cast(dict[str, Any], record.payload).get("call_number") == call_number
            for record in self.store.records("budget_ledger")
        )

    @staticmethod
    def _usage_tokens(usage: Mapping[str, Any], *names: str) -> int | None:
        for name in names:
            value = usage.get(name)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return max(0, int(value))
        return None

    def _start_tool_attempt(self, index: int, operation: QueryOperation) -> str:
        operation_id = self._operation_id(index)
        prior_attempts = sum(
            1
            for record in self.store.records("attempt")
            if isinstance(record.payload, dict)
            and record.payload.get("tool") is not None
            and record.payload.get("operation_id") == operation_id
        )
        attempt_id = f"attempt:{self._operation_key(index)}:{prior_attempts + 1}"
        self._tool_attempts += 1
        self.store.append(
            "attempt",
            f"{attempt_id}:start",
            {
                "attempt_id": attempt_id,
                "operation_id": operation_id,
                "status": "started",
                "tool": operation.tool,
                "call_number": self._model_calls,
                "decision_id": self._decision_id,
                "execution_attempt_id": self._execution_attempt_id,
                "started_at": time.time(),
                **self._candidate_audit(index),
            },
        )
        self._record_budget("tool_attempt_started", tool=operation.tool)
        return attempt_id

    def _candidate_audit(self, index: int) -> dict[str, Any]:
        origin = self._candidate_query_origins.get(index)
        return (
            {
                "candidate_id": origin[0],
                "logical_operation_index": origin[1],
            }
            if origin is not None
            else {}
        )

    def _consume_validator_retry(self) -> bool:
        key = self._budget_key("validator_retry_started", call_number=self._model_calls)
        if self.store.get(key) is not None:
            return True
        if (
            self.config.context_strategy != "frontier_reflection"
            and self._validator_retries >= self.config.max_validator_retries
        ):
            return False
        self._validator_retries += 1
        self._record_budget(
            "validator_retry_started",
            call_number=self._model_calls,
            decision_id=self._decision_id,
        )
        return True

    def _continue_after_validation_rejection(self) -> bool:
        if self._consume_validator_retry():
            self._finish_round(True)
            return True
        self._no_progress_rounds = max(
            self._no_progress_rounds,
            self.config.max_no_progress_rounds - 1,
        )
        self._finish_round(False)
        self._terminal_status = "no_progress"
        self._failure = "validator_retries"
        return False

    def _finish_round(self, progress: bool) -> None:
        key = self._budget_key("round_completed", call_number=self._model_calls)
        existing = self.store.get(key)
        if existing is not None:
            if isinstance(existing.payload, dict):
                rounds = cast(dict[str, Any], existing.payload).get(
                    "no_progress_rounds"
                )
                if isinstance(rounds, int):
                    self._no_progress_rounds = rounds
            if (
                not self._unbounded_reasoning_enabled()
                and self._no_progress_rounds >= self.config.max_no_progress_rounds
            ):
                self._terminal_status = "no_progress"
            return
        self._no_progress_rounds = 0 if progress else self._no_progress_rounds + 1
        self._record_budget(
            "round_completed", call_number=self._model_calls, progress=progress
        )
        if (
            not self._unbounded_reasoning_enabled()
            and self._no_progress_rounds >= self.config.max_no_progress_rounds
        ):
            self._terminal_status = "no_progress"

    def _query_batch_is_new(self) -> bool:
        current_prefix = f"{self._run_id}:observation:{self._decision_id}:"
        prior = {
            self._hash_json({"tool": item.tool, "arguments": item.arguments})
            for item in self._observations.values()
            if not item.observation_id.startswith(current_prefix)
        }
        return any(
            self._hash_json({"tool": operation.tool, "arguments": operation.arguments})
            not in prior
            for _index, operation, _arguments in self._query_calls
        )

    def _prior_query_observation(self, operation: QueryOperation) -> Observation | None:
        current_prefix = f"{self._run_id}:observation:{self._decision_id}:"
        signature = self._hash_json(
            {"tool": operation.tool, "arguments": operation.arguments}
        )
        for observation in self._observations.values():
            if observation.observation_id.startswith(current_prefix):
                continue
            if observation.status not in {"success", "partial", "no_data"} and not (
                observation.status in {"error", "cancelled", "unsupported"}
                and observation.retryable is False
            ):
                continue
            if (
                self._hash_json(
                    {"tool": observation.tool, "arguments": observation.arguments}
                )
                == signature
            ):
                return observation
        return None

    def _requires_initial_overview(self) -> bool:
        return (
            not self.environment.task.entry_entities
            and "summarize_anomalies" in self.tools
            and not self._has_successful_overview()
        )

    def _has_successful_overview(self) -> bool:
        return any(
            observation.tool == "summarize_anomalies"
            and observation.status in {"success", "partial"}
            for observation in self._observations.values()
        )

    def _has_successful_metric_pair(self) -> bool:
        periods_by_metric: dict[tuple[str, str], set[str]] = {}
        for observation in self._observations.values():
            if observation.tool != "query_metric" or observation.status not in {
                "success",
                "partial",
            }:
                continue
            entity_id = observation.arguments.get("entity_id")
            metric_id = observation.arguments.get("metric_id")
            period = observation.arguments.get("period")
            if not all(
                isinstance(value, str) and value
                for value in (entity_id, metric_id, period)
            ):
                continue
            periods_by_metric.setdefault(
                (cast(str, entity_id), cast(str, metric_id)), set()
            ).add(cast(str, period))
        return any(
            {"reference", "incident"} <= periods
            for periods in periods_by_metric.values()
        )

    def _pending_metric_pair_dispositions(self) -> list[dict[str, Any]]:
        candidate_registry, candidate_mode = self._overview_candidate_registry()
        direct_candidate_series = {
            (entity_id, metric_id)
            for entity_id, metric_id, _family, _fault_type in (
                candidate_registry.values()
            )
        }
        evidence_by_observation: dict[str, set[str]] = {}
        for evidence in self._evidence.values():
            for observation_id in evidence.observation_refs:
                evidence_by_observation.setdefault(observation_id, set()).add(
                    evidence.evidence_id
                )

        pairs: dict[tuple[str, str], dict[str, set[str]]] = {}
        for observation in self._observations.values():
            if observation.tool != "query_metric" or observation.status not in {
                "success",
                "partial",
            }:
                continue
            entity_id = observation.arguments.get("entity_id")
            metric_id = observation.arguments.get("metric_id")
            period = observation.arguments.get("period", "incident")
            if (
                not isinstance(entity_id, str)
                or not entity_id
                or not isinstance(metric_id, str)
                or not metric_id
                or period not in {"reference", "incident"}
            ):
                continue
            if candidate_mode and (entity_id, metric_id) not in direct_candidate_series:
                continue
            pairs.setdefault((entity_id, metric_id), {}).setdefault(
                period, set()
            ).update(evidence_by_observation.get(observation.observation_id, set()))

        pending: list[dict[str, Any]] = []
        for (entity_id, metric_id), periods in sorted(pairs.items()):
            if set(periods) != {"reference", "incident"}:
                continue
            handled = any(
                hypothesis.subject == entity_id
                and hypothesis.status in {"supported", "refuted", "inconclusive"}
                and bool(
                    periods["reference"]
                    & {
                        *hypothesis.supporting_evidence_refs,
                        *hypothesis.opposing_evidence_refs,
                    }
                )
                and bool(
                    periods["incident"]
                    & {
                        *hypothesis.supporting_evidence_refs,
                        *hypothesis.opposing_evidence_refs,
                    }
                )
                for hypothesis in self._hypotheses.values()
            )
            if not handled:
                pending.append(
                    {
                        "entity_id": entity_id,
                        "metric_id": metric_id,
                        "reference_evidence_ids": sorted(periods["reference"]),
                        "incident_evidence_ids": sorted(periods["incident"]),
                    }
                )
        return pending

    def _frontier_unreflected_observation_ids(self) -> list[str]:
        return sorted(set(self._observations) - self._reflected_observation_ids)

    def _frontier_subject_is_impact_only(self, subject: str) -> bool:
        return any(
            node.canonical_id == subject and node.evidence_role == "impact_only"
            for node in self.environment.topology.nodes
        )

    def _frontier_profiled_pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for evidence in self._evidence.values():
            if not isinstance(evidence.value, Mapping):
                continue
            summary = cast(Mapping[str, object], evidence.value).get(
                "finding_gate_summary"
            )
            if not isinstance(summary, Mapping):
                continue
            fault_type = cast(Mapping[str, object], summary).get("requested_fault_type")
            if isinstance(fault_type, str):
                pairs.update((subject, fault_type) for subject in evidence.subject_ids)
        for observation in self._observations.values():
            if observation.tool not in {
                "query_metric_profile",
                "query_service_profile",
            } or observation.status not in {"success", "partial", "no_data"}:
                continue
            subject = observation.arguments.get(
                "entity_id"
            ) or observation.arguments.get("service_id")
            fault_type = observation.arguments.get("fault_type")
            if isinstance(subject, str) and isinstance(fault_type, str):
                pairs.add((subject, fault_type))
        return pairs

    def _frontier_pending_active_profile_pairs(self) -> set[tuple[str, str]]:
        profiled = self._frontier_profiled_pairs()
        active_ids = set(
            frontier_active_anomaly_ids(
                list(self._material_anomalies.values()),
                list(self._anomaly_dispositions.values()),
                list(self._evidence.values()),
            )
        )
        return {
            (item.subject_id, mechanism)
            for item in self._material_anomalies.values()
            if item.anomaly_id in active_ids
            for mechanism in item.mechanism_candidates
            if (item.subject_id, mechanism) not in profiled
        }

    def _frontier_pending_material_profile_pairs(self) -> set[tuple[str, str]]:
        profiled = self._frontier_profiled_pairs()
        return {
            (item.subject_id, mechanism)
            for item in self._material_anomalies.values()
            for mechanism in item.mechanism_candidates
            if (item.subject_id, mechanism) not in profiled
        }

    def _frontier_pending_challenger_page(self) -> list[dict[str, Any]]:
        profiled = self._frontier_profiled_pairs()
        material_pairs = {
            (item.subject_id, mechanism)
            for item in self._material_anomalies.values()
            for mechanism in item.mechanism_candidates
        }
        pending_page = [
            seed
            for seed in frontier_profile_seeds(
                list(self._observations.values()), materiality="navigation_only"
            )
            if (str(seed.get("subject")), str(seed.get("fault_type")))
            not in material_pairs
            if (str(seed.get("subject")), str(seed.get("fault_type"))) not in profiled
        ]
        return pending_page[:8]

    def _frontier_has_typed_ready_leader(self) -> bool:
        return any(
            evidence.semantic_quality == "typed"
            and evidence.evidence_kind in {"metric_profile", "service_profile"}
            and isinstance(evidence.value, Mapping)
            and isinstance(
                summary := cast(Mapping[str, object], evidence.value).get(
                    "finding_gate_summary"
                ),
                Mapping,
            )
            and (
                cast(Mapping[str, object], summary).get("fault_type_ready") is True
                and (
                    isinstance(
                        cast(Mapping[str, object], summary).get("mechanism_signature"),
                        str,
                    )
                    or cast(Mapping[str, object], summary).get(
                        "signature_free_finding_ready"
                    )
                    is True
                )
                or (
                    cast(Mapping[str, object], summary).get("material_event_ready")
                    is True
                    and mechanism_signature_rank(cast(Mapping[str, object], summary))
                    >= 3
                )
            )
            for evidence in self._evidence.values()
        )

    def _frontier_referenced_hypothesis_ids(self, decision: Decision) -> set[str]:
        reflection = decision.reflection
        if reflection is None:
            return set()
        referenced = {
            assessment.hypothesis_id for assessment in reflection.evidence_update
        }
        referenced.update(
            update.hypothesis_id for update in reflection.hypothesis_update
        )
        if reflection.current_best_hypothesis_id is not None:
            referenced.add(reflection.current_best_hypothesis_id)
        if reflection.next_check is not None:
            referenced.update(reflection.next_check.distinguishes_hypothesis_ids)
        for operation in decision.operations:
            if isinstance(operation, FinalizeOperation):
                referenced.update(operation.hypothesis_ids)
            elif isinstance(operation, SubmitPathOperation):
                referenced.add(operation.hypothesis_id)
        return referenced

    def _frontier_proposal_handles(self, decision: Decision) -> set[str]:
        reflection = decision.reflection
        if reflection is None or reflection.new_hypothesis is None:
            return set()
        handles = {"new_hypothesis"}
        if reflection.new_hypothesis.client_id is not None:
            handles.add(reflection.new_hypothesis.client_id)
        unknown_references = (
            self._frontier_referenced_hypothesis_ids(decision)
            - set(self._hypotheses)
            - handles
        )
        if len(unknown_references) == 1:
            handles.update(unknown_references)
        return handles

    def _frontier_proposal_handle_map(self, decision: Decision) -> dict[str, str]:
        reflection = decision.reflection
        if reflection is None or reflection.new_hypothesis is None:
            return {}
        hypothesis_id = self._entity_id("hypothesis", 0)
        return {
            handle: hypothesis_id
            for handle in self._frontier_proposal_handles(decision)
        }

    def _frontier_reflection_changes_state(self, reflection: Reflection) -> bool:
        latest_assessments = self._latest_frontier_assessments()
        for proposal in reflection.evidence_update:
            existing = latest_assessments.get(
                (proposal.evidence_id, proposal.hypothesis_id)
            )
            if existing is None or any(
                getattr(existing, name) != getattr(proposal, name)
                for name in ("impact", "reason_code", "explanation")
            ):
                return True
        for proposal in reflection.anomaly_disposition_update:
            existing = self._anomaly_dispositions.get(proposal.anomaly_id)
            if existing is None or any(
                getattr(existing, name) != getattr(proposal, name)
                for name in (
                    "disposition",
                    "hypothesis_id",
                    "evidence_refs",
                    "explanation",
                )
            ):
                return True
        for update in reflection.hypothesis_update:
            existing = self._hypotheses.get(update.hypothesis_id)
            if existing is None or (
                existing.status != update.status
                or existing.open_questions != update.open_questions
            ):
                return True
        if reflection.clear_current_best:
            return self._current_best_hypothesis_id is not None
        return bool(
            reflection.current_best_hypothesis_id is not None
            and reflection.current_best_hypothesis_id
            != self._current_best_hypothesis_id
        )

    def _frontier_direct_support_profile_ids(
        self,
        hypothesis_id: str,
        assessments: Mapping[tuple[str, str], EvidenceAssessment],
        *,
        subject: str,
    ) -> set[str]:
        result: set[str] = set()
        for item in assessments.values():
            evidence = self._evidence.get(item.evidence_id)
            if (
                item.hypothesis_id != hypothesis_id
                or item.impact != "supports"
                or evidence is None
                or subject not in evidence.subject_ids
                or evidence.semantic_quality != "typed"
                or evidence.evidence_kind not in {"metric_profile", "service_profile"}
                or not isinstance(evidence.value, Mapping)
            ):
                continue
            summary = cast(Mapping[str, object], evidence.value).get(
                "finding_gate_summary"
            )
            if not isinstance(summary, Mapping):
                continue
            values = cast(Mapping[str, object], summary)
            if values.get("boundary_aligned") is not True and (
                (
                    values.get("fault_type_ready") is True
                    and (
                        isinstance(values.get("mechanism_signature"), str)
                        or values.get("signature_free_finding_ready") is True
                    )
                )
                or (
                    values.get("material_event_ready") is True
                    and isinstance(values.get("mechanism_signature"), str)
                )
            ):
                result.add(item.evidence_id)
        return result

    def _frontier_eligible_profile_ids_for_subject(self, subject: str) -> set[str]:
        result: set[str] = set()
        for evidence in self._evidence.values():
            if (
                subject not in evidence.subject_ids
                or evidence.semantic_quality != "typed"
                or evidence.evidence_kind not in {"metric_profile", "service_profile"}
                or not isinstance(evidence.value, Mapping)
            ):
                continue
            summary = cast(Mapping[str, object], evidence.value).get(
                "finding_gate_summary"
            )
            if not isinstance(summary, Mapping):
                continue
            values = cast(Mapping[str, object], summary)
            if values.get("boundary_aligned") is not True and (
                (
                    values.get("fault_type_ready") is True
                    and (
                        isinstance(values.get("mechanism_signature"), str)
                        or values.get("signature_free_finding_ready") is True
                    )
                )
                or (
                    values.get("material_event_ready") is True
                    and isinstance(values.get("mechanism_signature"), str)
                )
            ):
                result.add(evidence.evidence_id)
        return result

    def _frontier_hypothesis_creation_profile_ids(self) -> set[str]:
        return {
            evidence_id
            for evidence in self._evidence.values()
            for evidence_id in [evidence.evidence_id]
            if any(
                evidence_id in self._frontier_eligible_profile_ids_for_subject(subject)
                for subject in evidence.subject_ids
            )
        }

    def _frontier_profile_strength(
        self, evidence_ids: Sequence[str]
    ) -> tuple[int, float]:
        strengths: list[tuple[int, float]] = []
        for evidence_id in evidence_ids:
            evidence = self._evidence.get(evidence_id)
            if evidence is None or not isinstance(evidence.value, Mapping):
                continue
            summary = cast(Mapping[str, object], evidence.value).get(
                "finding_gate_summary"
            )
            if not isinstance(summary, Mapping):
                continue
            values = cast(Mapping[str, object], summary)
            raw_score = values.get("root_cause_score")
            if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
                continue
            strengths.append((mechanism_signature_rank(values), float(raw_score)))
        return max(strengths, default=(-1, 0.0))

    def _frontier_stronger_profile_evidence(self, hypothesis_id: str) -> set[str]:
        hypothesis = self._hypotheses[hypothesis_id]
        current_support = self._frontier_direct_support_profile_ids(
            hypothesis_id,
            self._latest_frontier_assessments(),
            subject=hypothesis.subject,
        )
        current_strength = self._frontier_profile_strength(sorted(current_support))
        stronger: set[str] = set()
        for evidence in self._evidence.values():
            if any(subject == hypothesis.subject for subject in evidence.subject_ids):
                continue
            if evidence.evidence_id not in {
                evidence_id
                for subject in evidence.subject_ids
                for evidence_id in self._frontier_eligible_profile_ids_for_subject(
                    subject
                )
            }:
                continue
            if (
                self._frontier_profile_strength([evidence.evidence_id])
                > current_strength
            ):
                stronger.add(evidence.evidence_id)
        return stronger

    def _frontier_eligible_leader_ids(self) -> list[str]:
        assessments = self._latest_frontier_assessments()
        return sorted(
            hypothesis_id
            for hypothesis_id, hypothesis in self._hypotheses.items()
            if hypothesis.status in {"open", "supported"}
            and (
                self._frontier_direct_support_profile_ids(
                    hypothesis_id,
                    assessments,
                    subject=hypothesis.subject,
                )
                or (
                    not self._frontier_eligible_profile_ids_for_subject(
                        hypothesis.subject
                    )
                    and any(
                        item.hypothesis_id == hypothesis_id
                        and item.impact == "supports"
                        for item in assessments.values()
                    )
                )
            )
            and not any(
                item.hypothesis_id == hypothesis_id and item.impact == "refutes"
                for item in assessments.values()
            )
        )

    def _validate_frontier_decision(self, decision: Decision) -> str | None:
        if self.config.context_strategy != "frontier_reflection":
            return None
        operations = decision.operations
        occurrence_only = self.environment.task.requested_outputs == [
            "root cause occurrence datetime"
        ]
        if any(operation.type == "update_hypothesis" for operation in operations):
            return (
                "frontier_reflection owns hypothesis changes; use Decision.reflection "
                "instead of update_hypothesis"
            )
        if any(
            operation.type == "query" and operation.tool == QUERY_CANDIDATE_TOOL
            for operation in operations
        ):
            return (
                "query_candidate is a legacy fixed-queue macro; frontier_reflection "
                "must issue evidence-grounded raw query tools"
            )
        unreflected = self._frontier_unreflected_observation_ids()
        reflection = decision.reflection
        has_advance = any(
            operation.type == "advance_frontier" for operation in operations
        )
        if has_advance and (len(operations) != 1 or reflection is None):
            return (
                "advance_frontier must be the sole operation and requires a Reflection"
            )
        incomplete_advance_error: str | None = None
        if has_advance and reflection is not None:
            if not self._frontier_reflection_changes_state(reflection):
                if self._current_best_hypothesis_id in self._hypotheses:
                    stronger_evidence = self._frontier_stronger_profile_evidence(
                        self._current_best_hypothesis_id
                    )
                    if stronger_evidence:
                        return (
                            "leader upgrade is required before advance_frontier: "
                            "another subject has strictly stronger direct typed "
                            "mechanism Evidence. A rejected Decision is atomic, so "
                            "resubmit the Hypothesis, same-subject supports, and "
                            "explicit current-best replacement together. Stronger "
                            "Evidence IDs: " + json.dumps(sorted(stronger_evidence))
                        )
                return (
                    "advance_frontier requires at least one actual "
                    "EvidenceAssessment, hypothesis, current-best, or anomaly "
                    "disposition state change; decision_summary text is not state"
                )
            active_ids = frontier_active_anomaly_ids(
                list(self._material_anomalies.values()),
                list(self._anomaly_dispositions.values()),
                list(self._evidence.values()),
            )
            latest_dispositions: dict[str, AnomalyDispositionProposal] = dict(
                self._anomaly_dispositions
            )
            latest_dispositions.update(
                {
                    item.anomaly_id: item
                    for item in reflection.anomaly_disposition_update
                }
            )
            incomplete_active = [
                anomaly_id
                for anomaly_id in active_ids
                if anomaly_id not in latest_dispositions
                or latest_dispositions[anomaly_id].disposition == "unresolved"
            ]
            if incomplete_active:
                incomplete_advance_error = (
                    "advance_frontier requires every anomaly on the current active "
                    "page to have a resolved Evidence-backed disposition; incomplete "
                    "active anomaly IDs: " + json.dumps(incomplete_active)
                )
        closeout_reflection = bool(
            not unreflected
            and reflection is not None
            and reflection.can_finalize
            and reflection.next_check is None
            and reflection.new_hypothesis is None
            and len(operations) == 1
            and operations[0].type == "finalize"
        )
        state_advance_reflection = bool(
            not unreflected
            and reflection is not None
            and has_advance
            and not reflection.can_finalize
            and reflection.next_check is None
            and reflection.new_hypothesis is None
            and not reflection.new_hypothesis_is_current_best
        )
        hard_closeout_abstention = bool(
            unreflected
            and reflection is None
            and len(operations) == 1
            and isinstance(operations[0], FinalizeOperation)
            and operations[0].abstain_reason is not None
            and self._frontier_wall_closeout_required()
        )
        historical_reflection = closeout_reflection or state_advance_reflection
        if unreflected and reflection is None and not hard_closeout_abstention:
            return (
                "frontier_reflection requires Decision.reflection for new "
                "Observations: " + json.dumps(unreflected)
            )
        if not unreflected and reflection is not None and not historical_reflection:
            return (
                "without a novel Observation, reflection is allowed only as a "
                "state-changing advance_frontier or a closeout state transition; "
                "both require next_check=null and no new_hypothesis"
            )
        if reflection is None:
            return None
        if unreflected and reflection.based_on_observation_ids != unreflected:
            return (
                "reflection based_on_observation_ids must exactly match the ordered "
                "unreflected Observation IDs: " + json.dumps(unreflected)
            )
        if historical_reflection:
            based_on = set(reflection.based_on_observation_ids)
            unknown_based_on = sorted(based_on - self._reflected_observation_ids)
            if unknown_based_on:
                return (
                    "closeout reflection may cite only already reflected "
                    "Observations: " + json.dumps(unknown_based_on)
                )
            referenced_observations = {
                observation_id
                for update in reflection.evidence_update
                for evidence in [self._evidence.get(update.evidence_id)]
                if evidence is not None
                for observation_id in evidence.observation_refs
            }
            referenced_observations.update(
                observation_id
                for disposition in reflection.anomaly_disposition_update
                for evidence_id in disposition.evidence_refs
                if evidence_id in self._evidence
                for observation_id in self._evidence[evidence_id].observation_refs
            )
            missing_based_on = sorted(referenced_observations - based_on)
            if missing_based_on:
                return (
                    "closeout reflection based_on_observation_ids must cover the "
                    "Observations behind every updated Evidence reference: "
                    + json.dumps(missing_based_on)
                )
        pending_challengers = self._frontier_pending_challenger_page()
        challenger_discovery_required = bool(
            not occurrence_only
            and not self._frontier_pending_active_profile_pairs()
            and not self._frontier_has_typed_ready_leader()
            and pending_challengers
            and not self._frontier_wall_closeout_required()
        )
        if challenger_discovery_required:
            expected = [
                (
                    "query_metric_profile",
                    str(seed["subject"]),
                    str(seed["fault_type"]),
                    tuple(cast(list[str], seed["metric_ids"])),
                )
                for seed in pending_challengers
            ]
            actual = [
                (
                    operation.tool,
                    operation.arguments.get("entity_id"),
                    operation.arguments.get("fault_type"),
                    tuple(cast(list[str], operation.arguments.get("metric_ids", []))),
                )
                for operation in operations
                if isinstance(operation, QueryOperation)
            ]
            if actual != expected:
                return (
                    "bounded challenger discovery is required before deep Trace, "
                    "advance, or finalization because no typed-ready leader exists; "
                    "query the exact visible frontier_control/challenger_profile_seeds "
                    f"in order ({len(pending_challengers)} seed(s)); the Context owns "
                    "their full arguments, so validation feedback does not duplicate "
                    "them"
                )
        topology_entities = {
            node.canonical_id for node in self.environment.topology.nodes
        }
        proposal = reflection.new_hypothesis
        referenced_hypothesis_ids = self._frontier_referenced_hypothesis_ids(decision)
        proposal_handles = self._frontier_proposal_handles(decision)
        unknown_references: set[str] = (
            referenced_hypothesis_ids - set(self._hypotheses) - proposal_handles
        )
        if unknown_references:
            return (
                "reflection references unknown hypotheses: "
                + json.dumps(sorted(unknown_references))
                + "; to create a hypothesis, populate Reflection.new_hypothesis, "
                "optionally set its client_id to the temporary ID, and use that exact "
                "handle in evidence_update, hypothesis_update, current-best, next "
                "check, dispositions, and finalize"
            )
        for assessment in reflection.evidence_update:
            if assessment.evidence_id not in self._evidence:
                return f"unknown assessment evidence: {assessment.evidence_id}"
            if assessment.hypothesis_id not in self._hypotheses and not (
                proposal is not None and assessment.hypothesis_id in proposal_handles
            ):
                return f"unknown assessment hypothesis: {assessment.hypothesis_id}"
        for disposition in reflection.anomaly_disposition_update:
            anomaly = self._material_anomalies.get(disposition.anomaly_id)
            if anomaly is None:
                return f"unknown material anomaly: {disposition.anomaly_id}"
            missing_evidence = sorted(
                set(disposition.evidence_refs) - self._evidence.keys()
            )
            if missing_evidence:
                return (
                    f"anomaly {disposition.anomaly_id} cites unknown Evidence: "
                    f"{missing_evidence}"
                )
            if (
                disposition.hypothesis_id is not None
                and disposition.hypothesis_id not in self._hypotheses
                and disposition.hypothesis_id not in proposal_handles
            ):
                return (
                    f"anomaly {disposition.anomaly_id} references unknown "
                    f"hypothesis: {disposition.hypothesis_id}"
                )
            if disposition.disposition == "concurrent_noncausal":
                if disposition.hypothesis_id is None:
                    return (
                        f"concurrent_noncausal anomaly {disposition.anomaly_id} "
                        "requires the explicit winning hypothesis_id"
                    )
                root_subject = (
                    proposal.subject
                    if proposal is not None
                    and disposition.hypothesis_id in proposal_handles
                    else self._hypotheses[disposition.hypothesis_id].subject
                )
                eligible_root_support = self._frontier_eligible_profile_ids_for_subject(
                    root_subject
                )
                latest_assessments = self._latest_frontier_assessments()
                supporting_root_refs = {
                    item.evidence_id
                    for item in latest_assessments.values()
                    if item.hypothesis_id == disposition.hypothesis_id
                    and item.impact == "supports"
                } | {
                    item.evidence_id
                    for item in reflection.evidence_update
                    if item.hypothesis_id == disposition.hypothesis_id
                    and item.impact == "supports"
                }
                refuted = any(
                    item.hypothesis_id == disposition.hypothesis_id
                    and item.impact == "refutes"
                    for item in [
                        *latest_assessments.values(),
                        *reflection.evidence_update,
                    ]
                )
                if refuted or not (supporting_root_refs & eligible_root_support):
                    return (
                        f"concurrent_noncausal anomaly {disposition.anomaly_id} "
                        "requires an unrefuted winning Hypothesis with explicit "
                        "same-subject supports from a non-boundary "
                        "mechanism_signature or "
                        "signature_free_finding_ready=true profile"
                    )
            if disposition.disposition == "root_cause":
                disposition_subject = (
                    proposal.subject
                    if proposal is not None
                    and disposition.hypothesis_id in proposal_handles
                    else self._hypotheses[cast(str, disposition.hypothesis_id)].subject
                )
                if anomaly.subject_id != disposition_subject:
                    return (
                        f"root-cause anomaly {disposition.anomaly_id} subject "
                        f"{anomaly.subject_id!r} differs from hypothesis subject "
                        f"{disposition_subject!r}"
                    )
        for update in reflection.hypothesis_update:
            if update.hypothesis_id not in self._hypotheses:
                if (
                    proposal is not None
                    and update.hypothesis_id in proposal_handles
                    and update.status in {"open", "supported"}
                ):
                    continue
                return f"unknown frontier hypothesis: {update.hypothesis_id}"
        if (
            reflection.current_best_hypothesis_id is not None
            and reflection.current_best_hypothesis_id not in self._hypotheses
            and reflection.current_best_hypothesis_id not in proposal_handles
        ):
            return (
                "unknown current_best_hypothesis_id: "
                f"{reflection.current_best_hypothesis_id}"
            )
        if (
            reflection.current_best_hypothesis_id in self._hypotheses
            and self._frontier_subject_is_impact_only(
                self._hypotheses[reflection.current_best_hypothesis_id].subject
            )
        ):
            return "current-best hypothesis subject is impact_only"
        if proposal is not None:
            if proposal.subject not in topology_entities:
                return f"new hypothesis subject is outside topology: {proposal.subject}"
            if self._frontier_subject_is_impact_only(proposal.subject):
                return (
                    f"new hypothesis subject {proposal.subject!r} is impact_only; "
                    "impact Evidence may support timing or business effect but not "
                    "a root-cause candidate"
                )
            missing = sorted(set(proposal.derived_from) - self._evidence.keys())
            if missing:
                return f"new hypothesis cites unknown Evidence: {missing}"
            navigation_refs = set(self._navigation_evidence_refs(proposal.derived_from))
            if set(proposal.derived_from) <= navigation_refs:
                return (
                    "new hypothesis requires at least one non-navigation Evidence "
                    "item; adapter proposals and overview results may schedule checks "
                    "but cannot anchor a hypothesis"
                )
            live_count = sum(
                hypothesis.status in {"open", "supported"}
                for hypothesis in self._hypotheses.values()
            )
            if live_count >= 12:
                return "frontier already contains twelve live hypotheses"
        eligible_proposal_refs: set[str] = (
            self._frontier_eligible_profile_ids_for_subject(proposal.subject)
            if proposal is not None
            else set()
        )
        proposal_any_supports = any(
            item.hypothesis_id in proposal_handles
            and item.impact == "supports"
            and item.evidence_id in self._evidence
            and proposal is not None
            and proposal.subject in self._evidence[item.evidence_id].subject_ids
            for item in reflection.evidence_update
        )
        proposal_support_refs = {
            item.evidence_id
            for item in reflection.evidence_update
            if item.hypothesis_id in proposal_handles
            and item.impact == "supports"
            and item.evidence_id in eligible_proposal_refs
        }
        proposal_supports = bool(
            proposal_support_refs
            or (not eligible_proposal_refs and proposal_any_supports)
        ) and not any(
            item.hypothesis_id in proposal_handles and item.impact == "refutes"
            for item in reflection.evidence_update
        )
        selected_proposal = bool(
            proposal is not None
            and (
                reflection.new_hypothesis_is_current_best
                or reflection.current_best_hypothesis_id in proposal_handles
            )
        )
        if selected_proposal and eligible_proposal_refs and not proposal_support_refs:
            return (
                "new current-best hypothesis requires an explicit supports "
                "EvidenceAssessment for one same-subject typed, eligible mechanism "
                "profile; rival exclusions and derived_from provenance do not select "
                "a leader"
            )
        if (
            proposal is not None
            and proposal_support_refs
            and self._current_best_hypothesis_id in self._hypotheses
            and not selected_proposal
        ):
            current_id = self._current_best_hypothesis_id
            current = self._hypotheses[current_id]
            current_support = self._frontier_direct_support_profile_ids(
                current_id,
                self._latest_frontier_assessments(),
                subject=current.subject,
            )
            proposal_strength = self._frontier_profile_strength(
                sorted(proposal_support_refs)
            )
            current_strength = self._frontier_profile_strength(sorted(current_support))
            if proposal_strength > current_strength:
                return (
                    "new proposal has strictly stronger direct typed mechanism "
                    f"Evidence {proposal_strength} than current-best "
                    f"{current_strength}. A rejected Decision is atomic and does "
                    "not persist the proposal or assessment; resubmit "
                    "new_hypothesis, same-subject evidence_update supports, and "
                    "new_hypothesis_is_current_best=true together (or select its "
                    "exact client handle as current_best_hypothesis_id)."
                )
        if not occurrence_only and self._current_best_hypothesis_id in self._hypotheses:
            current_id = self._current_best_hypothesis_id
            stronger_evidence = self._frontier_stronger_profile_evidence(current_id)
            proposal_upgrades = bool(
                selected_proposal and proposal_support_refs & stronger_evidence
            )
            selected_existing = reflection.current_best_hypothesis_id
            existing_upgrades = bool(
                selected_existing in self._hypotheses
                and selected_existing != current_id
                and (
                    self._frontier_direct_support_profile_ids(
                        selected_existing,
                        self._latest_frontier_assessments(),
                        subject=self._hypotheses[selected_existing].subject,
                    )
                    | {
                        item.evidence_id
                        for item in reflection.evidence_update
                        if item.hypothesis_id == selected_existing
                        and item.impact == "supports"
                        and item.evidence_id
                        in self._frontier_eligible_profile_ids_for_subject(
                            self._hypotheses[selected_existing].subject
                        )
                    }
                )
                & stronger_evidence
            )
            if stronger_evidence and not (proposal_upgrades or existing_upgrades):
                return (
                    "leader upgrade is required because another subject has strictly "
                    "stronger direct typed mechanism Evidence. A rejected Decision "
                    "is atomic; create or select its Hypothesis, add same-subject "
                    "supports, and explicitly replace current-best in one Decision. "
                    "Stronger Evidence IDs: " + json.dumps(sorted(stronger_evidence))
                )
        if not occurrence_only and self._current_best_hypothesis_id is None:
            eligible_leaders = self._frontier_eligible_leader_ids()
            if proposal_supports:
                eligible_leaders = [*eligible_leaders, *sorted(proposal_handles)]
            selected_leader = bool(
                reflection.new_hypothesis_is_current_best
                or reflection.current_best_hypothesis_id is not None
            )
            if eligible_leaders and not selected_leader:
                return (
                    "leader selection is required because current-best is empty and "
                    "explicit supports exist; set "
                    "Reflection.current_best_hypothesis_id to one exact eligible ID "
                    "or set new_hypothesis_is_current_best=true for the supported "
                    "proposal; eligible IDs: " + json.dumps(eligible_leaders)
                )
        live_hypotheses = any(
            hypothesis.status in {"open", "supported"}
            for hypothesis in self._hypotheses.values()
        )
        creation_profiles = self._frontier_hypothesis_creation_profile_ids()
        hard_closeout_abstention_now = bool(
            len(operations) == 1
            and isinstance(operations[0], FinalizeOperation)
            and operations[0].abstain_reason is not None
            and self._frontier_wall_closeout_required()
        )
        if (
            not live_hypotheses
            and creation_profiles
            and proposal is None
            and self.environment.task.requested_outputs
            != ["root cause occurrence datetime"]
            and not hard_closeout_abstention_now
        ):
            return (
                "hypothesis creation is required because no live hypothesis exists "
                "and eligible typed/material-ready profile Evidence is visible; "
                "populate Reflection.new_hypothesis, cite one eligible profile in "
                "derived_from and evidence_update supports, and select it as current "
                "best. Eligible Evidence IDs: " + json.dumps(sorted(creation_profiles))
            )
        allowed_distinguish_ids = set(self._hypotheses)
        allowed_distinguish_ids.update(proposal_handles)
        next_check = reflection.next_check
        query_operations = [
            operation for operation in operations if operation.type == "query"
        ]
        if reflection.can_finalize:
            if len(operations) != 1 or operations[0].type != "finalize":
                return "can_finalize requires exactly one finalize operation"
            finalize_operation = operations[0]
            if finalize_operation.occurrence_evidence_refs:
                occurrence_errors = self._occurrence_selection_errors(
                    finalize_operation
                )
                if occurrence_errors:
                    return "; ".join(occurrence_errors)
            elif finalize_operation.abstain_reason is None:
                anomaly_errors = self._anomaly_coverage_errors(
                    finalize_operation.hypothesis_ids,
                    proposed=reflection.anomaly_disposition_update,
                    proposal_handles=proposal_handles,
                    prospective_support={
                        hypothesis_id: {
                            item.evidence_id
                            for item in reflection.evidence_update
                            if item.hypothesis_id == hypothesis_id
                            and item.impact == "supports"
                        }
                        for hypothesis_id in {
                            item.hypothesis_id for item in reflection.evidence_update
                        }
                    },
                )
                if anomaly_errors:
                    return "; ".join(anomaly_errors)
            updates_by_id = {
                update.hypothesis_id: update for update in reflection.hypothesis_update
            }
            for selected_id in finalize_operation.hypothesis_ids:
                update = updates_by_id.get(selected_id)
                prospective_status = (
                    update.status
                    if update is not None
                    else (
                        self._hypotheses[selected_id].status
                        if selected_id in self._hypotheses
                        else None
                    )
                )
                if prospective_status != "supported":
                    return (
                        "can_finalize requires every selected hypothesis to be "
                        "supported prospectively; add a same-reflection supported "
                        f"hypothesis_update for {selected_id}"
                    )
        else:
            if has_advance:
                if next_check is not None:
                    return "advance_frontier requires next_check=null"
            elif len(query_operations) != len(operations):
                return (
                    "reflection may be followed only by query operations or one "
                    "advance_frontier operation"
                )
            elif next_check is not None:
                tools = [operation.tool for operation in query_operations]
                compact_tools = list(dict.fromkeys(tools))
                if next_check.operation_tools not in (tools, compact_tools):
                    return (
                        "next_check.operation_tools must match either the complete "
                        "query tool sequence or its first-use ordered unique sequence"
                    )
                unknown = sorted(
                    set(next_check.distinguishes_hypothesis_ids)
                    - allowed_distinguish_ids
                )
                if unknown:
                    return f"next_check distinguishes unknown hypotheses: {unknown}"
        proposed_by_pair = {
            (item.evidence_id, item.hypothesis_id): item
            for item in reflection.evidence_update
        }
        for update in reflection.hypothesis_update:
            if update.status != "supported":
                continue
            impacts = {
                assessment.impact
                for assessment in self._latest_frontier_assessments(
                    proposed_by_pair
                ).values()
                if assessment.hypothesis_id == update.hypothesis_id
            }
            if "supports" not in impacts:
                return (
                    f"supported hypothesis {update.hypothesis_id} requires an explicit "
                    "supporting EvidenceAssessment"
                )
            if "refutes" in impacts:
                return (
                    f"supported hypothesis {update.hypothesis_id} still has refuting "
                    "EvidenceAssessment"
                )
            gate_errors = self._frontier_finding_errors(
                update.hypothesis_id,
                self._latest_frontier_assessments(proposed_by_pair),
                subject=(
                    proposal.subject
                    if proposal is not None and update.hypothesis_id in proposal_handles
                    else self._hypotheses[update.hypothesis_id].subject
                ),
                statement=(
                    proposal.statement
                    if proposal is not None and update.hypothesis_id in proposal_handles
                    else self._hypotheses[update.hypothesis_id].statement
                ),
            )
            if gate_errors:
                return "; ".join(gate_errors)
        return incomplete_advance_error

    def _anomaly_coverage_errors(
        self,
        selected_hypothesis_ids: Sequence[str],
        *,
        proposed: Sequence[Any] = (),
        proposal_handles: set[str] | None = None,
        prospective_support: Mapping[str, set[str]] | None = None,
    ) -> list[str]:
        if not self._material_anomalies:
            return []
        handles = proposal_handles or set()
        selected = set(selected_hypothesis_ids)
        latest: dict[str, Any] = {
            anomaly_id: item for anomaly_id, item in self._anomaly_dispositions.items()
        }
        latest.update({item.anomaly_id: item for item in proposed})
        unresolved = sorted(
            anomaly_id
            for anomaly_id in self._material_anomalies
            if anomaly_id not in latest
            or latest[anomaly_id].disposition == "unresolved"
        )
        errors: list[str] = []
        if unresolved:
            errors.append(
                "formal Finding requires every material anomaly to have an "
                "Evidence-backed disposition; unresolved anomalies: "
                + json.dumps(unresolved[:12])
                + (f" (+{len(unresolved) - 12} more)" if len(unresolved) > 12 else "")
            )
        root_hypotheses = {
            item.hypothesis_id
            for item in latest.values()
            if item.disposition == "root_cause" and item.hypothesis_id is not None
        }
        unresolved_selected = selected - root_hypotheses - handles
        if unresolved_selected:
            errors.append(
                "every selected Finding hypothesis requires a root_cause anomaly "
                "disposition: " + json.dumps(sorted(unresolved_selected))
            )
        unselected_roots = root_hypotheses - selected - handles
        if unselected_roots:
            stale_roots = [
                {
                    "anomaly_id": item.anomaly_id,
                    "hypothesis_id": item.hypothesis_id,
                }
                for item in latest.values()
                if item.disposition == "root_cause"
                and item.hypothesis_id in unselected_roots
            ]
            errors.append(
                "root_cause anomaly dispositions reference unselected hypotheses; "
                "revise these dispositions before selecting a different root: "
                + json.dumps(stale_roots, sort_keys=True)
            )
        for item in latest.values():
            if (
                item.disposition != "root_cause"
                or item.hypothesis_id in handles
                or item.hypothesis_id not in self._hypotheses
            ):
                continue
            supporting = set(
                self._hypotheses[item.hypothesis_id].supporting_evidence_refs
            )
            supporting.update(
                assessment.evidence_id
                for assessment in self._latest_frontier_assessments().values()
                if assessment.hypothesis_id == item.hypothesis_id
                and assessment.impact == "supports"
            )
            supporting.update(
                (prospective_support or {}).get(item.hypothesis_id, set())
            )
            missing_support = sorted(set(item.evidence_refs) - supporting)
            if missing_support:
                errors.append(
                    f"root_cause anomaly {item.anomaly_id} cites Evidence not "
                    "supporting its hypothesis: " + json.dumps(missing_support)
                )
        return errors

    def _frontier_finding_errors(
        self,
        hypothesis_id: str,
        assessments: Mapping[tuple[str, str], EvidenceAssessment],
        *,
        subject: str | None = None,
        statement: str | None = None,
    ) -> list[str]:
        existing = self._hypotheses.get(hypothesis_id)
        if existing is not None:
            subject = subject or existing.subject
            statement = statement or existing.statement
        if isinstance(subject, str) and self._frontier_subject_is_impact_only(subject):
            return ["formal Finding subject is impact_only"]
        profiled_pairs = {
            (profiled_subject, requested_fault_type)
            for evidence in self._evidence.values()
            if evidence.evidence_kind in {"metric_profile", "service_profile"}
            and isinstance(evidence.value, Mapping)
            for profiled_subject in evidence.subject_ids
            for summary in [
                cast(Mapping[str, object], evidence.value).get("finding_gate_summary")
            ]
            if isinstance(summary, Mapping)
            for requested_fault_type in [
                cast(Mapping[str, object], summary).get("requested_fault_type")
            ]
            if isinstance(requested_fault_type, str)
        }
        profiled_pairs.update(
            (profiled_subject, fault_type)
            for observation in self._observations.values()
            if observation.tool in {"query_metric_profile", "query_service_profile"}
            and observation.status in {"success", "partial", "no_data"}
            for profiled_subject in [
                observation.arguments.get("entity_id")
                or observation.arguments.get("service_id")
            ]
            for fault_type in [observation.arguments.get("fault_type")]
            if isinstance(profiled_subject, str) and isinstance(fault_type, str)
        )
        ready_storage_subjects = {
            profiled_subject
            for profiled_subject, fault_type in profiled_pairs
            if any(
                token in fault_type.casefold()
                for token in ("disk", "read i/o", "write i/o")
            )
            and any(
                evidence.evidence_kind == "metric_profile"
                and profiled_subject in evidence.subject_ids
                and isinstance(evidence.value, Mapping)
                and isinstance(
                    summary := cast(Mapping[str, object], evidence.value).get(
                        "finding_gate_summary"
                    ),
                    Mapping,
                )
                and cast(Mapping[str, object], summary).get("requested_fault_type")
                == fault_type
                and cast(Mapping[str, object], summary).get("fault_type_ready") is True
                for evidence in self._evidence.values()
            )
        }
        pending_storage_refinements = [
            (str(seed["subject"]), str(seed["fault_type"]))
            for seed in frontier_profile_seeds(list(self._observations.values()))
            if seed.get("resource_family") == "storage"
            and str(seed["subject"]) in ready_storage_subjects
            and (str(seed["subject"]), str(seed["fault_type"])) not in profiled_pairs
        ]
        if pending_storage_refinements and not self._frontier_wall_closeout_required():
            return [
                "ready storage candidate requires same-subject storage sibling "
                "refinement before Finding: " + json.dumps(pending_storage_refinements)
            ]
        supporting = [
            item
            for item in assessments.values()
            if item.hypothesis_id == hypothesis_id and item.impact == "supports"
        ]
        refuting = [
            item
            for item in assessments.values()
            if item.hypothesis_id == hypothesis_id and item.impact == "refutes"
        ]
        if not supporting:
            return ["frontier Finding requires supporting EvidenceAssessment"]
        if refuting:
            return ["frontier Finding cannot retain refuting EvidenceAssessment"]
        typed_evidence = [
            self._evidence[item.evidence_id]
            for item in supporting
            if item.evidence_id in self._evidence
            and self._evidence[item.evidence_id].semantic_quality == "typed"
        ]
        if not typed_evidence:
            return [
                "frontier Finding requires a registered typed Evidence Extractor; "
                "inferred or unregistered results may support only Recommendation"
            ]
        if any(
            evidence.evidence_kind in {"log_pattern", "trace_edge"}
            and evidence.onset_interval is not None
            for evidence in typed_evidence
        ):
            return []
        public_fault_types = self._public_fault_types()
        statement_fault_type = (
            self._resolve_public_fault_type(statement, public_fault_types)
            if statement is not None and public_fault_types
            else None
        )
        proof_bound_summaries: list[Mapping[str, object]] = []
        for evidence in typed_evidence:
            if (
                evidence.evidence_kind not in {"metric_profile", "service_profile"}
                or not isinstance(evidence.value, Mapping)
                or subject not in evidence.subject_ids
            ):
                continue
            summary = cast(Mapping[str, Any], evidence.value).get(
                "finding_gate_summary"
            )
            if (
                not isinstance(summary, Mapping)
                or summary.get("fault_type_ready") is not True
            ):
                continue
            values = cast(Mapping[str, object], summary)
            proof_bound_summaries.append(values)
        resolved_fault_type = statement_fault_type or dominant_market_fault_type(
            proof_bound_summaries
        )
        if (
            proof_bound_summaries
            and not any(
                summary.get("boundary_aligned") is not True
                and (
                    isinstance(summary.get("mechanism_signature"), str)
                    or summary.get("signature_free_finding_ready") is True
                )
                for summary in proof_bound_summaries
            )
            and not frontier_service_profile_seeds(list(self._observations.values()))
        ):
            return [
                "formal Finding requires an explicit non-boundary "
                "mechanism_signature or signature_free_finding_ready=true from an "
                "audited material effect or multi-replica consensus"
            ]
        if public_fault_types and resolved_fault_type is None:
            return [
                "Finding requires an exact registered fault type bound by typed "
                "profile Evidence or uniquely named in the statement"
            ]
        if resolved_fault_type is not None and frontier_service_profile_seeds(
            list(self._observations.values())
        ):
            matching_profiles: list[tuple[Evidence, Mapping[str, Any]]] = []
            for evidence in typed_evidence:
                if (
                    evidence.evidence_kind not in {"metric_profile", "service_profile"}
                    or not isinstance(evidence.value, Mapping)
                    or subject not in evidence.subject_ids
                ):
                    continue
                summary = cast(Mapping[str, Any], evidence.value).get(
                    "finding_gate_summary"
                )
                if (
                    isinstance(summary, Mapping)
                    and summary.get("requested_fault_type") == resolved_fault_type
                    and summary.get("fault_type_ready") is True
                ):
                    matching_profiles.append(
                        (evidence, cast(Mapping[str, Any], summary))
                    )
            if not matching_profiles:
                available_refs = sorted(
                    evidence.evidence_id
                    for evidence in self._evidence.values()
                    if evidence.evidence_kind in {"metric_profile", "service_profile"}
                    and subject in evidence.subject_ids
                    and isinstance(evidence.value, Mapping)
                    and isinstance(
                        summary := cast(Mapping[str, object], evidence.value).get(
                            "finding_gate_summary"
                        ),
                        Mapping,
                    )
                    and cast(Mapping[str, object], summary).get("requested_fault_type")
                    == resolved_fault_type
                    and cast(Mapping[str, object], summary).get("fault_type_ready")
                    is True
                )
                return [
                    "metric Finding requires typed profile Evidence requested "
                    f"for fault_type {resolved_fault_type!r} with "
                    "fault_type_ready=true; "
                    "derived_from does not imply supports, so add an explicit "
                    "EvidenceAssessment supports for one of: "
                    + json.dumps(available_refs)
                ]

            selected_has_mechanism = any(
                summary.get("boundary_aligned") is not True
                and (
                    isinstance(summary.get("mechanism_signature"), str)
                    or summary.get("signature_free_finding_ready") is True
                )
                for _evidence, summary in matching_profiles
            )
            if not selected_has_mechanism:
                profiled_service_pairs = {
                    (service_id, requested_fault_type)
                    for evidence in self._evidence.values()
                    if evidence.evidence_kind == "service_profile"
                    and isinstance(evidence.value, Mapping)
                    for service_id in [
                        cast(Mapping[str, object], evidence.value).get("service_id")
                    ]
                    for summary in [
                        cast(Mapping[str, object], evidence.value).get(
                            "finding_gate_summary"
                        )
                    ]
                    if isinstance(service_id, str) and isinstance(summary, Mapping)
                    for requested_fault_type in [
                        cast(Mapping[str, object], summary).get("requested_fault_type")
                    ]
                    if isinstance(requested_fault_type, str)
                }
                profiled_service_pairs.update(
                    (service_id, fault_type)
                    for observation in self._observations.values()
                    if observation.tool == "query_service_profile"
                    and observation.status in {"success", "partial", "no_data"}
                    for service_id in [observation.arguments.get("service_id")]
                    for fault_type in [observation.arguments.get("fault_type")]
                    if isinstance(service_id, str) and isinstance(fault_type, str)
                )
                pending_mechanism_seeds = [
                    (str(seed["service_id"]), str(seed["fault_type"]))
                    for seed in frontier_service_profile_seeds(
                        list(self._observations.values())
                    )
                    if str(seed.get("mechanism_family"))
                    in {"latency", "packet_transport", "retransmission"}
                    and (str(seed["service_id"]), str(seed["fault_type"]))
                    not in profiled_service_pairs
                ]
                if (
                    pending_mechanism_seeds
                    and not self._frontier_wall_closeout_required()
                ):
                    return [
                        "generic-pressure Finding has unresolved higher-rank "
                        "mechanism-capable service seeds: "
                        + json.dumps(pending_mechanism_seeds)
                        + "; query these dynamic-frontier competitors, not every "
                        "remaining seed"
                    ]
                return [
                    "formal Finding requires an explicit non-boundary "
                    "mechanism_signature or signature_free_finding_ready=true from "
                    "an audited material effect or multi-replica consensus"
                ]

            summaries_by_subject: dict[str, list[Mapping[str, object]]] = {}
            for evidence in self._evidence.values():
                if evidence.evidence_kind not in {
                    "metric_profile",
                    "service_profile",
                } or not isinstance(evidence.value, Mapping):
                    continue
                summary = cast(Mapping[str, Any], evidence.value).get(
                    "finding_gate_summary"
                )
                if (
                    not isinstance(summary, Mapping)
                    or summary.get("fault_type_ready") is not True
                ):
                    continue
                values = cast(Mapping[str, object], summary)
                for candidate_subject in evidence.subject_ids:
                    summaries_by_subject.setdefault(candidate_subject, []).append(
                        values
                    )
            best_by_subject: dict[str, tuple[int, float, str]] = {}
            for candidate_subject, summaries in summaries_by_subject.items():
                dominant = dominant_market_fault_type(summaries)
                scores = [
                    float(score)
                    for item in summaries
                    if item.get("requested_fault_type") == dominant
                    for score in [item.get("root_cause_score")]
                    if isinstance(score, (int, float)) and not isinstance(score, bool)
                ]
                if dominant is not None and scores:
                    dominant_summaries = [
                        item
                        for item in summaries
                        if item.get("requested_fault_type") == dominant
                    ]
                    mechanism_rank = int(
                        any(
                            isinstance(item.get("mechanism_signature"), str)
                            and item.get("boundary_aligned") is not True
                            for item in dominant_summaries
                        )
                    )
                    best_by_subject[candidate_subject] = (
                        mechanism_rank,
                        max(scores),
                        dominant,
                    )
            expected_count = self.environment.task.expected_fault_count or 1
            top_candidates = {
                (candidate_subject, candidate[2])
                for candidate_subject, candidate in sorted(
                    best_by_subject.items(),
                    key=lambda item: (
                        -item[1][0],
                        -item[1][1],
                        item[0],
                        item[1][2],
                    ),
                )[:expected_count]
            }
            if (subject, resolved_fault_type) not in top_candidates:
                return [
                    "Finding is outranked by stronger eligible profiled "
                    "subject/fault pairs: "
                    + json.dumps(sorted(top_candidates))
                    + "; ranking is lexicographic: non-boundary mechanism_signature "
                    "first, then root_cause_score. Create or support the exact top "
                    "subject/fault hypothesis with its typed Evidence before finalize"
                ]
            return []
        if (
            public_fault_types
            and resolved_fault_type is not None
            and subject is not None
        ):
            best_by_pair: dict[tuple[str, str], tuple[int, float]] = {}
            for evidence in self._evidence.values():
                if evidence.evidence_kind not in {
                    "metric_profile",
                    "service_profile",
                } or not isinstance(evidence.value, Mapping):
                    continue
                summary = cast(Mapping[str, object], evidence.value).get(
                    "finding_gate_summary"
                )
                if (
                    not isinstance(summary, Mapping)
                    or summary.get("fault_type_ready") is not True
                ):
                    continue
                summary_values = cast(Mapping[str, object], summary)
                fault_type: object = summary_values.get("requested_fault_type")
                score: object = summary_values.get("root_cause_score")
                if (
                    not isinstance(fault_type, str)
                    or fault_type not in public_fault_types
                    or not isinstance(score, (int, float))
                    or isinstance(score, bool)
                ):
                    continue
                rank = mechanism_signature_rank(summary_values)
                for candidate_subject in evidence.subject_ids:
                    pair = (candidate_subject, fault_type)
                    candidate = (rank, float(score))
                    if candidate > best_by_pair.get(pair, (-1, -1.0)):
                        best_by_pair[pair] = candidate
            expected_count = self.environment.task.expected_fault_count or 1
            top_candidates = {
                pair
                for pair, _rank in sorted(
                    best_by_pair.items(),
                    key=lambda item: (
                        -item[1][0],
                        -item[1][1],
                        item[0][0],
                        item[0][1],
                    ),
                )[:expected_count]
            }
            if top_candidates and (subject, resolved_fault_type) not in top_candidates:
                return [
                    "Finding is outranked by more causally specific eligible typed "
                    "profiles: "
                    + json.dumps(sorted(top_candidates))
                    + "; rank uses explicit mechanism specificity before numeric "
                    "score. "
                    "Revisit the dynamic frontier and support the exact leading pair"
                ]
        for evidence in typed_evidence:
            if evidence.evidence_kind != "metric_profile" or not isinstance(
                evidence.value, Mapping
            ):
                continue
            value = cast(Mapping[str, Any], evidence.value)
            gate_summary = value.get("finding_gate_summary")
            if (
                isinstance(gate_summary, Mapping)
                and gate_summary.get("single_component_ready") is True
                and evidence.onset_interval is not None
            ):
                return []
            profiles = value.get("metric_profiles")
            peer_comparison = value.get("peer_comparison")
            if (
                not isinstance(profiles, list)
                or len(cast(list[object], profiles)) < 2
                or evidence.onset_interval is None
                or not isinstance(peer_comparison, list)
            ):
                continue
            strong_profiles = sum(
                isinstance(raw_profile, Mapping)
                and raw_profile.get("strong_incident_change") is True
                for raw_profile in cast(list[object], profiles)
            )
            if strong_profiles < 2:
                continue
            isolated = any(
                isinstance(raw_peer, Mapping)
                and isinstance(raw_peer.get("peer_count"), int)
                and cast(int, raw_peer["peer_count"]) >= 2
                and (
                    isinstance(raw_peer.get("robust_z"), (int, float))
                    and not isinstance(raw_peer.get("robust_z"), bool)
                    and cast(float, raw_peer["robust_z"]) >= 3.0
                    or isinstance(raw_peer.get("percentile"), (int, float))
                    and not isinstance(raw_peer.get("percentile"), bool)
                    and cast(float, raw_peer["percentile"]) >= 0.9
                )
                for raw_peer in cast(list[object], peer_comparison)
            )
            if isolated:
                return []
        return [
            "single-component metric Finding requires a typed multi-metric profile, "
            "incident onset, and peer isolation; otherwise keep a Recommendation"
        ]

    def _frontier_wall_closeout_required(self) -> bool:
        model_timeout = self.config.model_timeout_seconds
        reserve_seconds = (model_timeout if model_timeout is not None else 60.0) + 60.0
        return self._deadline_at - time.time() <= reserve_seconds

    def _latest_frontier_assessments(
        self,
        proposed: Mapping[tuple[str, str], object] | None = None,
    ) -> dict[tuple[str, str], EvidenceAssessment]:
        latest: dict[tuple[str, str], EvidenceAssessment] = {
            (item.evidence_id, item.hypothesis_id): item
            for item in self._assessments.values()
        }
        if proposed:
            digest = self._frontier_observation_digest(
                self._frontier_unreflected_observation_ids()
            )
            for index, raw in enumerate(proposed.values()):
                if isinstance(raw, EvidenceAssessment):
                    assessment = raw
                else:
                    proposal = cast(Any, raw)
                    assessment = EvidenceAssessment(
                        **proposal.model_dump(mode="python"),
                        assessment_id=self._entity_id("assessment", index),
                        observation_digest=digest,
                    )
                latest[(assessment.evidence_id, assessment.hypothesis_id)] = assessment
        return latest

    def _latest_anomaly_dispositions(
        self,
        proposed: Sequence[AnomalyDisposition] = (),
    ) -> dict[str, AnomalyDisposition]:
        latest = dict(self._anomaly_dispositions)
        latest.update({item.anomaly_id: item for item in proposed})
        return latest

    def _frontier_observation_digest(self, observation_ids: Sequence[str]) -> str:
        return self._hash_json(
            [
                self._observations[observation_id].model_dump(mode="json")
                for observation_id in observation_ids
            ]
        )

    def _commit_frontier_reflection(self, decision: Decision) -> None:
        if self.config.context_strategy != "frontier_reflection":
            return
        reflection = decision.reflection
        if reflection is None:
            return
        stable_key = f"reflection:{self._decision_id}"
        if self.store.get(stable_key) is not None:
            return
        digest = self._frontier_observation_digest(reflection.based_on_observation_ids)
        new_hypothesis_id = (
            self._entity_id("hypothesis", 0)
            if reflection.new_hypothesis is not None
            else None
        )
        proposal_handles = self._frontier_proposal_handles(decision)
        new_assessments = [
            EvidenceAssessment(
                **proposal.model_copy(
                    update={
                        "hypothesis_id": (
                            new_hypothesis_id
                            if proposal.hypothesis_id in proposal_handles
                            else proposal.hypothesis_id
                        )
                    }
                ).model_dump(mode="python"),
                assessment_id=self._entity_id("assessment", index),
                observation_digest=digest,
            )
            for index, proposal in enumerate(reflection.evidence_update)
        ]
        new_dispositions = [
            AnomalyDisposition(
                **proposal.model_copy(
                    update={
                        "hypothesis_id": (
                            new_hypothesis_id
                            if proposal.hypothesis_id in proposal_handles
                            else proposal.hypothesis_id
                        )
                    }
                ).model_dump(mode="python"),
                disposition_id=self._entity_id("anomaly_disposition", index),
                observation_digest=digest,
            )
            for index, proposal in enumerate(reflection.anomaly_disposition_update)
        ]
        proposed_map = {
            (item.evidence_id, item.hypothesis_id): item for item in new_assessments
        }
        latest = self._latest_frontier_assessments(proposed_map)
        records: list[tuple[str, str, object]] = [
            ("evidence_assessment", f"assessment:{item.assessment_id}", item)
            for item in new_assessments
        ]
        records.extend(
            (
                "anomaly_disposition",
                f"anomaly_disposition:{item.disposition_id}",
                item,
            )
            for item in new_dispositions
        )
        updated_hypotheses: dict[str, Hypothesis] = {}
        for update in reflection.hypothesis_update:
            if update.hypothesis_id in proposal_handles:
                continue
            existing = self._hypotheses[update.hypothesis_id]
            relevant = [
                item
                for item in latest.values()
                if item.hypothesis_id == update.hypothesis_id
            ]
            updated = existing.model_copy(
                update={
                    "status": update.status,
                    "supporting_evidence_refs": sorted(
                        {
                            item.evidence_id
                            for item in relevant
                            if item.impact == "supports"
                            and not self._navigation_evidence_refs([item.evidence_id])
                        }
                    ),
                    "opposing_evidence_refs": sorted(
                        {
                            item.evidence_id
                            for item in relevant
                            if item.impact == "refutes"
                        }
                    ),
                    "open_questions": update.open_questions,
                    "created_from": list(
                        dict.fromkeys(
                            [
                                *existing.created_from,
                                *(item.evidence_id for item in relevant),
                            ]
                        )
                    ),
                }
            )
            updated_hypotheses[updated.hypothesis_id] = updated
            records.append(
                (
                    "hypothesis",
                    f"hypothesis:{updated.hypothesis_id}:{self._decision_id}",
                    updated,
                )
            )
        if reflection.new_hypothesis is not None:
            proposal = reflection.new_hypothesis
            assert new_hypothesis_id is not None
            proposal_update = next(
                (
                    update
                    for update in reflection.hypothesis_update
                    if update.hypothesis_id in proposal_handles
                ),
                None,
            )
            relevant = [
                item
                for item in latest.values()
                if item.hypothesis_id == new_hypothesis_id
            ]
            created = Hypothesis(
                hypothesis_id=new_hypothesis_id,
                subject=proposal.subject,
                statement=proposal.statement,
                status=(
                    proposal_update.status if proposal_update is not None else "open"
                ),
                supporting_evidence_refs=sorted(
                    {
                        item.evidence_id
                        for item in relevant
                        if item.impact == "supports"
                        and not self._navigation_evidence_refs([item.evidence_id])
                    }
                ),
                opposing_evidence_refs=sorted(
                    {item.evidence_id for item in relevant if item.impact == "refutes"}
                ),
                open_questions=(
                    proposal_update.open_questions
                    if proposal_update is not None
                    else proposal.open_questions
                ),
                created_from=list(
                    dict.fromkeys(
                        [
                            *proposal.derived_from,
                            *(item.evidence_id for item in relevant),
                        ]
                    )
                ),
            )
            updated_hypotheses[new_hypothesis_id] = created
            records.append(
                (
                    "hypothesis",
                    f"hypothesis:{new_hypothesis_id}:{self._decision_id}",
                    created,
                )
            )
        resolved_current_best = self._current_best_hypothesis_id
        if reflection.clear_current_best:
            resolved_current_best = None
        elif reflection.current_best_hypothesis_id is not None:
            resolved_current_best = reflection.current_best_hypothesis_id
        if reflection.new_hypothesis_is_current_best or (
            reflection.current_best_hypothesis_id in proposal_handles
        ):
            resolved_current_best = new_hypothesis_id
        reflection_payload = {
            **reflection.model_dump(mode="json"),
            "observation_digest": digest,
            "resolved_new_hypothesis_id": new_hypothesis_id,
            "resolved_current_best_hypothesis_id": resolved_current_best,
        }
        records.append(("reflection", stable_key, reflection_payload))
        self.store.append_many(records)
        for item in new_assessments:
            self._assessments[item.assessment_id] = item
        self._anomaly_dispositions.update(
            {item.anomaly_id: item for item in new_dispositions}
        )
        self._hypotheses.update(updated_hypotheses)
        self._reflected_observation_ids.update(reflection.based_on_observation_ids)
        self._current_best_hypothesis_id = resolved_current_best
        self._event(
            "reflection_applied",
            {
                "reflection_ref": stable_key,
                "observation_digest": digest,
                "assessment_count": len(new_assessments),
                "anomaly_disposition_count": len(new_dispositions),
                "new_hypothesis_id": new_hypothesis_id,
                "current_best_hypothesis_id": resolved_current_best,
            },
        )

    def _bind_candidate_disposition(
        self, operation: UpdateHypothesisOperation
    ) -> tuple[UpdateHypothesisOperation, str | None]:
        persisted = self._committed_candidate_disposition()
        if persisted is not None:
            self._pending_candidate_disposition = persisted[1]
            return persisted[0], None

        queue = candidate_disposition_queue(
            list(self._observations.values()),
            list(self._evidence.values()),
            list(self._hypotheses.values()),
        )
        if not queue:
            return operation, None
        candidate = queue[0]
        primary = candidate.get("fault_type")
        subject = candidate.get("entity_id")
        raw_evidence_ids = candidate.get("evidence_ids")
        if (
            not isinstance(primary, str)
            or not primary
            or not isinstance(subject, str)
            or not subject
            or not isinstance(raw_evidence_ids, Mapping)
        ):
            return operation, "candidate disposition queue head is malformed"

        allowed_fault_types = [primary]
        raw_aliases = candidate.get("fault_type_aliases", [])
        if isinstance(raw_aliases, list):
            allowed_fault_types.extend(
                alias
                for alias in cast(list[object], raw_aliases)
                if isinstance(alias, str) and alias and alias not in allowed_fault_types
            )
        explicit_other = next(
            (
                fault_type
                for fault_type in self._public_fault_types()
                if fault_type not in allowed_fault_types
                and self._resolve_public_fault_type(operation.statement, [fault_type])
                is not None
            ),
            None,
        )
        if explicit_other is not None:
            return (
                operation,
                "candidate disposition statement names a public fault type outside "
                f"the current pair: {explicit_other!r}; allowed values: "
                + json.dumps(allowed_fault_types, ensure_ascii=False),
            )
        statement = (
            self._resolve_public_fault_type(operation.statement, allowed_fault_types)
            or primary
        )

        evidence_ids = cast(Mapping[str, object], raw_evidence_ids)
        pair_refs = [
            cast(str, evidence_ids[period])
            for period in ("reference", "incident")
            if isinstance(evidence_ids.get(period), str) and evidence_ids[period]
        ]
        if len(pair_refs) != 2:
            return operation, "candidate disposition queue head lacks paired evidence"
        pair_ref_set = set(pair_refs)
        matching_open = [
            hypothesis
            for hypothesis in self._hypotheses.values()
            if hypothesis.status == "open"
            and hypothesis.subject == subject
            and (
                pair_ref_set
                <= {
                    *hypothesis.supporting_evidence_refs,
                    *hypothesis.opposing_evidence_refs,
                }
                or self._resolve_public_fault_type(
                    hypothesis.statement, allowed_fault_types
                )
                is not None
            )
        ]
        if len(matching_open) > 1:
            return operation, "candidate disposition matches multiple open hypotheses"
        existing_open = matching_open[0] if matching_open else None

        supporting_refs = pair_refs if operation.status in {"open", "supported"} else []
        if operation.status in {"open", "supported"}:
            supporting_refs.extend(
                evidence_id
                for evidence_id in self._candidate_supporting_log_evidence_ids(subject)
                if evidence_id not in pair_ref_set
            )
        opposing_refs = (
            pair_refs if operation.status in {"refuted", "inconclusive"} else []
        )
        bound = UpdateHypothesisOperation.model_validate(
            {
                **operation.model_dump(mode="json"),
                "action": "revise" if existing_open is not None else "create",
                "hypothesis_id": (
                    existing_open.hypothesis_id if existing_open is not None else None
                ),
                "subject": subject,
                "statement": statement,
                "supporting_evidence_refs": supporting_refs,
                "opposing_evidence_refs": opposing_refs,
            }
        )
        self._pending_candidate_disposition = {
            "event_type": "candidate_disposition_applied",
            "visibility": "audit_only",
            "decision_id": self._decision_id,
            "candidate_id": candidate.get("candidate_id"),
            "action": bound.action,
            "hypothesis_id": bound.hypothesis_id,
            "status": bound.status,
            "subject": bound.subject,
            "metric_id": candidate.get("metric_id"),
            "primary_fault_type": primary,
            "allowed_fault_types": allowed_fault_types,
            "resource_family": candidate.get("resource_family"),
            "log_selector_hint": candidate.get("log_selector_hint"),
            "selected_fault_type": bound.statement,
            "reference_evidence_id": pair_refs[0],
            "incident_evidence_id": pair_refs[1],
            "log_evidence_ids": sorted(set(supporting_refs) - pair_ref_set),
            "original_operation": operation.model_dump(mode="json"),
            "applied_operation": bound.model_dump(mode="json"),
        }
        return bound, None

    def _committed_candidate_disposition(
        self,
    ) -> tuple[UpdateHypothesisOperation, dict[str, Any]] | None:
        record = self.store.get(self._event_key("candidate_disposition_applied"))
        if record is None or not isinstance(record.payload, dict):
            return None
        event = cast(dict[str, Any], record.payload)
        payload = event.get("applied_operation")
        if not isinstance(payload, Mapping):
            raise ValueError(
                "persisted candidate disposition must contain an operation"
            )
        operation = UpdateHypothesisOperation.model_validate(
            dict(cast(Mapping[str, Any], payload))
        )
        bound_hypothesis_id = event.get("hypothesis_id")
        hypothesis_id = (
            bound_hypothesis_id
            if isinstance(bound_hypothesis_id, str) and bound_hypothesis_id
            else operation.hypothesis_id or self._entity_id("hypothesis", 0)
        )
        if self.store.get(f"hypothesis:{hypothesis_id}:{self._decision_id}") is None:
            return None
        return operation, event

    def _candidate_supporting_log_evidence_ids(self, subject: str) -> list[str]:
        result: list[str] = []
        for evidence_id, evidence in self._evidence.items():
            if (
                subject not in evidence.subject_ids
                or not _has_complete_paired_log_summary(evidence.value)
            ):
                continue
            if any(
                observation_id in self._observations
                and (observation := self._observations[observation_id]).tool
                == "query_log_samples"
                and observation.status in {"success", "partial"}
                and any(
                    isinstance(observation.arguments.get(name), str)
                    and bool(cast(str, observation.arguments[name]).strip())
                    for name in ("log_name", "contains")
                )
                for observation_id in evidence.observation_refs
            ):
                result.append(evidence_id)
        return sorted(result)

    def _has_log_evidence(self, subject: str, evidence_ids: set[str]) -> bool:
        require_paired_summary = self._overview_candidate_registry()[1]

        def qualifies(observation_id: str) -> bool:
            observation = self._observations.get(observation_id)
            if (
                observation is None
                or observation.tool != "query_log_samples"
                or observation.status not in {"success", "partial"}
            ):
                return False
            if not require_paired_summary:
                return True
            return any(
                isinstance(observation.arguments.get(name), str)
                and bool(cast(str, observation.arguments[name]).strip())
                for name in ("log_name", "contains")
            )

        return any(
            evidence_id in self._evidence
            and subject in self._evidence[evidence_id].subject_ids
            and (
                not require_paired_summary
                or _has_complete_paired_log_summary(self._evidence[evidence_id].value)
            )
            and any(
                qualifies(observation_id)
                for observation_id in self._evidence[evidence_id].observation_refs
            )
            for evidence_id in evidence_ids
        )

    def _has_terminal_log_gap(
        self, subject: str, *, after_hypothesis_id: str | None
    ) -> bool:
        if after_hypothesis_id is None:
            return False
        match = re.search(r":(\d+):\d+$", after_hypothesis_id)
        if match is None:
            return False
        hypothesis_round = int(match.group(1))
        return any(
            observation.tool == "query_log_samples"
            and observation.arguments.get("entity_id") == subject
            and any(
                isinstance(observation.arguments.get(name), str)
                and bool(cast(str, observation.arguments[name]).strip())
                for name in ("log_name", "contains")
            )
            and (
                observation_match := re.search(
                    r":(\d+):\d+$", observation.observation_id
                )
            )
            is not None
            and int(observation_match.group(1)) > hypothesis_round
            and (
                observation.status == "no_data"
                or (
                    observation.status in {"error", "cancelled", "unsupported"}
                    and observation.retryable is False
                )
            )
            for observation in self._observations.values()
        )

    def _candidate_records_for_series(
        self, entity_id: str, metric_id: str
    ) -> list[dict[str, object]]:
        registry, _has_candidate_ids = self._overview_candidate_registry()
        records: list[dict[str, object]] = []
        for observation in self._observations.values():
            if (
                observation.tool != "summarize_anomalies"
                or observation.status not in {"success", "partial"}
                or observation.summary is None
            ):
                continue
            try:
                summary = json.loads(observation.summary)
            except json.JSONDecodeError:
                continue
            if not isinstance(summary, Mapping):
                continue
            raw_candidates = cast(Mapping[str, object], summary).get(
                "fault_type_candidates"
            )
            if not isinstance(raw_candidates, list):
                continue
            for raw_candidate in cast(list[object], raw_candidates):
                if not isinstance(raw_candidate, Mapping):
                    continue
                candidate = cast(Mapping[str, object], raw_candidate)
                candidate_id = candidate.get("candidate_id")
                if (
                    not isinstance(candidate_id, str)
                    or candidate_id not in registry
                    or registry[candidate_id][:2] != (entity_id, metric_id)
                ):
                    continue
                records.append(dict(candidate))
        return records

    @staticmethod
    def _candidate_fault_types(candidate: Mapping[str, object]) -> set[str]:
        values: set[str] = set()
        if isinstance(candidate.get("fault_type"), str):
            values.add(cast(str, candidate["fault_type"]))
        raw_aliases = candidate.get("fault_type_aliases")
        if isinstance(raw_aliases, list):
            values.update(
                str(alias)
                for alias in cast(list[object], raw_aliases)
                if isinstance(alias, str) and alias
            )
        return values

    def _subject_log_source_state(
        self, subject: str, metric_id: str
    ) -> Literal["present", "absent", "unknown"]:
        persisted = {
            candidate["has_log_source"]
            for candidate in self._candidate_records_for_series(subject, metric_id)
            if isinstance(candidate.get("has_log_source"), bool)
        }
        if persisted:
            if len(persisted) != 1:
                return "unknown"
            return "present" if persisted.pop() is True else "absent"
        result = self.environment.list_metrics(subject)
        log_names = result.get("log_names")
        if result.get("status") != "success" or not isinstance(log_names, list):
            return "unknown"
        typed_log_names = cast(list[object], log_names)
        if not all(isinstance(item, str) and item for item in typed_log_names):
            return "unknown"
        return "present" if typed_log_names else "absent"

    def _paired_metric_summaries(
        self, evidence_ids: Sequence[str]
    ) -> dict[tuple[str, str], dict[str, object]]:
        summaries: dict[tuple[str, str], list[dict[str, object]]] = {}
        for evidence_id in evidence_ids:
            evidence = self._evidence.get(evidence_id)
            if evidence is None:
                continue
            summary = _compact_paired_metric_summary(
                cast(Mapping[str, object], evidence.value).get("paired_metric_summary")
                if isinstance(evidence.value, Mapping)
                else None
            )
            if summary is None or summary["counts_complete"] is not True:
                continue
            for observation_id in evidence.observation_refs:
                observation = self._observations.get(observation_id)
                if (
                    observation is None
                    or observation.tool != "query_metric"
                    or observation.status not in {"success", "partial"}
                ):
                    continue
                entity_id = observation.arguments.get("entity_id")
                metric_id = observation.arguments.get("metric_id")
                if (
                    not isinstance(entity_id, str)
                    or entity_id not in evidence.subject_ids
                    or not isinstance(metric_id, str)
                    or not metric_id
                ):
                    continue
                summaries.setdefault((entity_id, metric_id), []).append(summary)
        result: dict[tuple[str, str], dict[str, object]] = {}
        for series, values in summaries.items():
            canonical = {
                json.dumps(
                    value,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                for value in values
            }
            if len(canonical) == 1:
                result[series] = values[0]
        return result

    def _has_strict_metric_fallback(
        self, subject: str, statement: str, evidence_ids: Sequence[str]
    ) -> bool:
        accepted, _blocker = self._strict_metric_fallback_result(
            subject, statement, evidence_ids
        )
        return accepted

    def _strict_metric_fallback_result(
        self, subject: str, statement: str, evidence_ids: Sequence[str]
    ) -> tuple[bool, dict[str, object] | None]:
        resolved_fault_type = self._resolve_public_fault_type(
            statement, self._public_fault_types()
        )
        supporting_pairs = {
            series: summary
            for series, summary in self._paired_metric_summaries(evidence_ids).items()
            if series[0] == subject
        }
        if resolved_fault_type is None or len(supporting_pairs) != 1:
            return False, None
        ((series, summary),) = supporting_pairs.items()
        candidates = self._candidate_records_for_series(*series)
        matching = [
            candidate
            for candidate in candidates
            if resolved_fault_type in self._candidate_fault_types(candidate)
        ]
        directions = {
            candidate.get("candidate_pressure_direction") for candidate in matching
        }
        if (
            not matching
            or {candidate.get("metric_role") for candidate in matching} != {"direct"}
            or not _has_strict_directional_metric_shift(
                {"paired_metric_summary": summary},
                next(iter(directions)) if len(directions) == 1 else None,
            )
        ):
            return False, None
        families = {
            candidate.get("resource_family")
            for candidate in candidates
            if isinstance(candidate.get("resource_family"), str)
        }
        if len(families) != 1:
            return False, None
        family = next(iter(families))
        aliases = {
            alias
            for candidate in candidates
            for alias in self._candidate_fault_types(candidate)
        }
        observed_pairs = self._paired_metric_summaries(list(self._evidence))
        for other_series, other_summary in observed_pairs.items():
            if other_series[0] == subject:
                continue
            other_candidates = self._candidate_records_for_series(*other_series)
            matching_candidate = next(
                (
                    candidate
                    for candidate in other_candidates
                    if candidate.get("resource_family") == family
                    and bool(self._candidate_fault_types(candidate) & aliases)
                    and candidate.get("metric_role") == "direct"
                ),
                None,
            )
            other_direction = (
                matching_candidate.get("candidate_pressure_direction")
                if matching_candidate is not None
                else None
            )
            if matching_candidate is not None and _has_strict_directional_metric_shift(
                {"paired_metric_summary": other_summary}, other_direction
            ):
                return False, {
                    "reason_code": "competing_subject",
                    "subject": other_series[0],
                    "metric_id": other_series[1],
                    "candidate_id": matching_candidate.get("candidate_id"),
                    "evidence_refs": sorted(
                        evidence_id
                        for evidence_id in self._evidence
                        if other_series in self._paired_metric_summaries([evidence_id])
                    ),
                    "paired_metric_summary": other_summary,
                    "required_correction": {"status": "inconclusive"},
                }
        return True, None

    def _expand_candidate_queries(
        self,
        operations: Sequence[QueryOperation],
        *,
        persisted_expansions: object = None,
    ) -> tuple[list[QueryOperation], dict[int, tuple[str, int]], str | None]:
        candidate_operations = [
            operation
            for operation in operations
            if operation.tool == QUERY_CANDIDATE_TOOL
        ]
        if not candidate_operations:
            return list(operations), {}, None
        if len(candidate_operations) != len(operations):
            return [], {}, "query_candidate cannot be mixed with other query tools"
        if len(candidate_operations) not in {1, 2}:
            return [], {}, "query_candidate batch must contain one or two candidates"

        registry, _has_candidate_ids = self._overview_candidate_registry()
        selected: list[tuple[str, tuple[str, str, str, str]]] = []
        for operation in candidate_operations:
            if set(operation.arguments) != {"candidate_id"}:
                return [], {}, "query_candidate accepts only candidate_id"
            candidate_id = operation.arguments.get("candidate_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                return [], {}, "candidate_id must be a non-empty persisted value"
            candidate = registry.get(candidate_id)
            if candidate is None:
                return [], {}, f"unknown persisted candidate_id: {candidate_id}"
            selected.append((candidate_id, candidate))

        candidate_ids = [candidate_id for candidate_id, _candidate in selected]
        if len(set(candidate_ids)) != len(candidate_ids):
            return [], {}, "query_candidate batch must not repeat candidate_id"
        comparison = comparison_coverage(list(self._observations.values()))
        expected = cast(list[str], comparison["next_candidate_ids"])
        initial_schedule = (
            candidate_schedule(list(self._observations.values()))
            if self._is_first_candidate_decision()
            else []
        )
        if initial_schedule:
            selected = [
                (candidate_id, registry[candidate_id])
                for candidate_id in initial_schedule
            ]
            candidate_ids = list(initial_schedule)
        elif comparison["satisfied"] is False and candidate_ids != expected:
            return (
                [],
                {},
                "query_candidate must select next_candidate_ids exactly in order: "
                + json.dumps(expected),
            )

        expanded: list[QueryOperation] = []
        origins: dict[int, tuple[str, int]] = {}
        for logical_index, (
            candidate_id,
            (entity_id, metric_id, _family, _fault_type),
        ) in enumerate(selected):
            metric_ids = [metric_id]
            if initial_schedule:
                source_metric_id = next(
                    (
                        candidate.get("source_metric_id")
                        for candidate in self._candidate_records_for_series(
                            entity_id, metric_id
                        )
                        if candidate.get("candidate_id") == candidate_id
                    ),
                    None,
                )
                if (
                    isinstance(source_metric_id, str)
                    and source_metric_id
                    and source_metric_id != metric_id
                ):
                    metric_ids.append(source_metric_id)
            for expanded_metric_id in metric_ids:
                for period in ("reference", "incident"):
                    child_index = len(expanded)
                    expanded.append(
                        QueryOperation(
                            type="query",
                            tool="query_metric",
                            arguments={
                                "entity_id": entity_id,
                                "metric_id": expanded_metric_id,
                                "period": period,
                            },
                        )
                    )
                    origins[child_index] = (candidate_id, logical_index)
        canonical_expansions = self._candidate_expansion_metadata(
            expanded,
            origins,
            auto_fill_all=bool(initial_schedule),
            logical_operation_count=len(candidate_operations),
        )
        persisted_error = self._persisted_candidate_expansion_error(
            persisted_expansions, canonical_expansions
        )
        if persisted_error is not None:
            return (
                [],
                {},
                persisted_error,
            )
        return expanded, origins, None

    def _is_first_candidate_decision(self) -> bool:
        return not any(
            record.stable_key != self._decision_id
            and isinstance(record.payload, dict)
            and bool(cast(dict[str, Any], record.payload).get("candidate_expansions"))
            for record in self.store.records("decision")
        )

    def _has_bulk_candidate_schedule(self) -> bool:
        return any(
            isinstance(record.payload, dict)
            and isinstance(
                cast(dict[str, Any], record.payload).get("candidate_expansions"), list
            )
            and len(
                cast(
                    list[object],
                    cast(dict[str, Any], record.payload)["candidate_expansions"],
                )
            )
            > 2
            for record in self.store.records("decision")
        )

    def _candidate_expansion_metadata(
        self,
        expanded: Sequence[QueryOperation],
        origins: Mapping[int, tuple[str, int]],
        *,
        auto_fill_all: bool,
        logical_operation_count: int,
    ) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for candidate_id, logical_index in sorted(
            set(origins.values()), key=lambda item: item[1]
        ):
            expansion: dict[str, Any] = {
                "candidate_id": candidate_id,
                "logical_operation_index": logical_index,
                "children": [
                    {
                        "operation_id": self._operation_id(child_index),
                        "period": expanded[child_index].arguments["period"],
                    }
                    for child_index, origin in origins.items()
                    if origin == (candidate_id, logical_index)
                ],
            }
            if auto_fill_all or logical_index >= logical_operation_count:
                expansion["auto_filled"] = True
            result.append(expansion)
        return result

    @staticmethod
    def _persisted_candidate_expansion_error(
        persisted: object,
        canonical: Sequence[Mapping[str, Any]],
    ) -> str | None:
        if persisted is None:
            return None
        if not isinstance(persisted, list):
            return "persisted candidate_expansions must be a list"
        raw_items = cast(list[object], persisted)
        if len(raw_items) != len(canonical):
            return (
                "persisted candidate_expansions count does not match canonical schedule"
            )
        if not all(isinstance(item, dict) for item in raw_items):
            return "persisted candidate_expansions entries must be objects"
        items = [cast(dict[str, Any], item) for item in raw_items]
        candidate_ids = [item.get("candidate_id") for item in items]
        if any(not isinstance(item, str) or not item for item in candidate_ids):
            return "persisted candidate_expansions contains an unknown candidate_id"
        if len(set(candidate_ids)) != len(candidate_ids):
            return "persisted candidate_expansions contains duplicate candidate_id"
        canonical_ids = [item.get("candidate_id") for item in canonical]
        if any(candidate_id not in canonical_ids for candidate_id in candidate_ids):
            return "persisted candidate_expansions contains an unknown candidate_id"
        if candidate_ids != canonical_ids:
            return "persisted candidate_expansions candidate_id order is not canonical"
        for schedule_rank, (item, expected) in enumerate(zip(items, canonical)):
            if set(item) != set(expected):
                return (
                    "persisted candidate_expansions fields do not match canonical "
                    f"schedule at rank {schedule_rank}"
                )
            if item.get("logical_operation_index") != schedule_rank:
                return (
                    "persisted candidate_expansions logical_operation_index is not "
                    f"canonical at rank {schedule_rank}"
                )
            if item.get("children") != expected.get("children"):
                return (
                    "persisted candidate_expansions children do not match canonical "
                    f"operation_id/period metadata at rank {schedule_rank}"
                )
            if item.get("auto_filled") != expected.get("auto_filled"):
                return (
                    "persisted candidate_expansions auto_filled does not match "
                    f"canonical semantics at rank {schedule_rank}"
                )
        return None

    def _overview_candidate_registry(
        self,
    ) -> tuple[dict[str, tuple[str, str, str, str]], bool]:
        registry: dict[str, tuple[str, str, str, str]] = {}
        ambiguous: set[str] = set()
        for observation in self._observations.values():
            if (
                observation.tool != "summarize_anomalies"
                or observation.status not in {"success", "partial"}
                or observation.summary is None
            ):
                continue
            try:
                summary = json.loads(observation.summary)
            except json.JSONDecodeError:
                continue
            if not isinstance(summary, Mapping):
                continue
            raw_candidates = cast(Mapping[str, object], summary).get(
                "fault_type_candidates"
            )
            if not isinstance(raw_candidates, list):
                continue
            for raw_candidate in cast(list[object], raw_candidates):
                if not isinstance(raw_candidate, Mapping):
                    continue
                candidate = cast(Mapping[str, object], raw_candidate)
                candidate_id = candidate.get("candidate_id")
                if not isinstance(candidate_id, str) or not candidate_id:
                    continue
                entity_id = candidate.get("entity_id")
                metric_id = candidate.get("metric_id")
                family = candidate.get("resource_family")
                fault_type = candidate.get("fault_type")
                if candidate.get("requires_paired_exact") is not True or not all(
                    isinstance(value, str) and value
                    for value in (entity_id, metric_id, family, fault_type)
                ):
                    continue
                value = (
                    cast(str, entity_id),
                    cast(str, metric_id),
                    cast(str, family),
                    cast(str, fault_type),
                )
                existing = registry.get(candidate_id)
                if existing is not None and existing != value:
                    ambiguous.add(candidate_id)
                else:
                    registry[candidate_id] = value
        for candidate_id in ambiguous:
            registry.pop(candidate_id, None)
        return registry, bool(registry)

    def _first_metric_batch_error(
        self,
        operations: Sequence[QueryOperation],
        *,
        candidate_batch: bool = False,
    ) -> str | None:
        if self.config.context_strategy == "frontier_reflection":
            return None
        _registry, has_candidate_ids = self._overview_candidate_registry()
        if (
            has_candidate_ids
            and not candidate_batch
            and any(operation.tool == "query_metric" for operation in operations)
        ):
            return (
                "raw query_metric is disabled while persisted candidate_id values "
                "are available; use query_candidate"
            )
        if (
            self.environment.task.entry_entities
            or "query_metric" not in self.tools
            or not self._has_successful_overview()
            or self._has_successful_metric_pair()
            or not any(operation.tool == "query_metric" for operation in operations)
        ):
            return None
        if has_candidate_ids:
            return None
        candidates = self._overview_metric_candidates()
        guidance = (
            "copy entity_id and metric_id exactly from persisted overview "
            f"candidates: {json.dumps(self._overview_candidate_index(candidates))}"
        )
        if len(operations) not in {2, 4} or any(
            operation.tool != "query_metric" for operation in operations
        ):
            return (
                "first exact metric batch must contain two or four query_metric "
                "operations: one paired metric, or paired metrics from two different "
                f"resource families; {guidance}"
            )

        groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
        for operation in operations:
            entity_id = operation.arguments.get("entity_id")
            metric_id = operation.arguments.get("metric_id")
            period = operation.arguments.get("period")
            matches = [
                (entity, family)
                for (entity, family), metric_ids in candidates.items()
                if entity == entity_id and metric_id in metric_ids
            ]
            if len(matches) != 1:
                return (
                    f"query_metric is not in the persisted overview mapping; {guidance}"
                )
            if period not in {"reference", "incident"}:
                return (
                    "first exact metric batch requires explicit paired reference and "
                    f"incident periods; {guidance}"
                )
            groups.setdefault(matches[0], []).append((cast(str, metric_id), period))

        valid_groups = len(groups) == len(operations) // 2 and (
            len(operations) == 2 or len({family for _, family in groups}) == 2
        )
        valid_pairs = all(
            len(values) == 2
            and len({metric_id for metric_id, _period in values}) == 1
            and {period for _metric_id, period in values} == {"reference", "incident"}
            for values in groups.values()
        )
        if not valid_groups or not valid_pairs:
            return (
                "first exact metric batch requires one metric paired across reference "
                "and incident, or two such pairs from different resource families; "
                + guidance
            )
        return None

    def _overview_metric_candidates(self) -> dict[tuple[str, str], set[str]]:
        signal_candidates: dict[tuple[str, str], set[str]] = {}
        fault_candidates: dict[tuple[str, str], set[str]] = {}
        for observation in self._observations.values():
            if (
                observation.tool != "summarize_anomalies"
                or observation.status not in {"success", "partial"}
                or observation.summary is None
            ):
                continue
            try:
                summary = json.loads(observation.summary)
            except json.JSONDecodeError:
                continue
            if not isinstance(summary, Mapping):
                continue
            raw_signals = cast(Mapping[str, object], summary).get("signals")
            if isinstance(raw_signals, list):
                for raw_signal in cast(list[object], raw_signals):
                    if not isinstance(raw_signal, Mapping):
                        continue
                    signal = cast(Mapping[str, object], raw_signal)
                    entity_id = signal.get("entity_id")
                    family = signal.get("resource_family")
                    metric_id = signal.get("metric_id")
                    if not all(
                        isinstance(value, str) and value
                        for value in (entity_id, family)
                    ):
                        continue
                    key = (cast(str, entity_id), cast(str, family))
                    metrics = signal_candidates.setdefault(key, set())
                    if isinstance(metric_id, str) and metric_id:
                        metrics.add(metric_id)
                    direct = signal.get("direct_metric_candidates")
                    if isinstance(direct, list):
                        metrics.update(
                            item
                            for item in cast(list[object], direct)
                            if isinstance(item, str) and item
                        )
            raw_fault_candidates = cast(Mapping[str, object], summary).get(
                "fault_type_candidates"
            )
            if isinstance(raw_fault_candidates, list):
                for raw_candidate in cast(list[object], raw_fault_candidates):
                    if not isinstance(raw_candidate, Mapping):
                        continue
                    candidate = cast(Mapping[str, object], raw_candidate)
                    entity_id = candidate.get("entity_id")
                    family = candidate.get("resource_family")
                    metric_id = candidate.get("metric_id")
                    if not all(
                        isinstance(value, str) and value
                        for value in (entity_id, family, metric_id)
                    ):
                        continue
                    fault_candidates.setdefault(
                        (cast(str, entity_id), cast(str, family)), set()
                    ).add(cast(str, metric_id))
        return fault_candidates or signal_candidates

    @staticmethod
    def _overview_candidate_index(
        candidates: Mapping[tuple[str, str], set[str]],
    ) -> list[dict[str, object]]:
        return [
            {
                "entity_id": entity_id,
                "resource_family": family,
                "metric_ids": sorted(metric_ids),
            }
            for (entity_id, family), metric_ids in sorted(candidates.items())
        ]

    def _navigation_evidence_refs(self, evidence_ids: Sequence[str]) -> list[str]:
        return sorted(
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id in self._evidence
            and any(
                self._observations[observation_id].tool in NAVIGATION_TOOLS
                for observation_id in self._evidence[evidence_id].observation_refs
                if observation_id in self._observations
            )
        )

    def _has_trace_inspection(self, subject: str) -> bool:
        return any(
            observation.tool == "query_trace_neighbors"
            and observation.arguments.get("entity_id") == subject
            and observation.status in {"success", "partial"}
            for observation in self._observations.values()
        )

    def _hypothesis_corroboration_errors(
        self, subject: str, statement: str, evidence_ids: Sequence[str]
    ) -> list[str]:
        errors: list[str] = []
        metric_observations = [
            self._observations[observation_id]
            for evidence_id in evidence_ids
            if evidence_id in self._evidence
            and subject in self._evidence[evidence_id].subject_ids
            for observation_id in self._evidence[evidence_id].observation_refs
            if observation_id in self._observations
            and self._observations[observation_id].tool == "query_metric"
        ]
        if not metric_observations:
            return ["supported hypothesis requires query_metric evidence"]
        _registry, candidate_mode = self._overview_candidate_registry()
        if not candidate_mode:
            if not self.environment.capabilities.logs:
                return errors
            metric_id = metric_observations[0].arguments.get("metric_id")
            log_state = (
                self._subject_log_source_state(subject, metric_id)
                if isinstance(metric_id, str) and metric_id
                else "unknown"
            )
            if log_state == "present" and not self._has_log_evidence(
                subject, set(evidence_ids)
            ):
                errors.append(
                    "supported hypothesis requires successful or partial "
                    "query_log_samples corroboration"
                )
            elif log_state == "unknown":
                errors.append("subject log availability is unknown")
            return errors
        metric_pairs = {
            series: summary
            for series, summary in self._paired_metric_summaries(evidence_ids).items()
            if series[0] == subject
        }
        if len(metric_pairs) != 1:
            errors.append(
                "supported hypothesis requires one same-subject exact query_metric "
                "with a complete paired_metric_summary"
            )
            return errors
        metric_id = next(iter(metric_pairs))[1]
        resolved_fault_type = self._resolve_public_fault_type(
            statement, self._public_fault_types()
        )
        if resolved_fault_type is None or not any(
            resolved_fault_type in self._candidate_fault_types(candidate)
            and candidate.get("metric_role") == "direct"
            for candidate in self._candidate_records_for_series(subject, metric_id)
        ):
            errors.append(
                "supported hypothesis requires a persisted direct candidate pair "
                "for the same subject and public fault type"
            )
            return errors
        log_state = self._subject_log_source_state(subject, metric_id)
        if log_state == "present" and not self._has_log_evidence(
            subject, set(evidence_ids)
        ):
            errors.append(
                "supported hypothesis requires successful or partial "
                "query_log_samples corroboration; candidate mode also requires a "
                "non-empty log_name or contains filter and a complete "
                "paired_log_summary"
            )
        elif log_state == "absent":
            accepted, blocker = self._strict_metric_fallback_result(
                subject, statement, evidence_ids
            )
            if not accepted:
                error = (
                    "a subject without logs requires a persisted direct candidate and "
                    "a complete paired metric with known direction, at least 20% "
                    "direction-aligned incident envelope crossings, an aligned "
                    "incident median shift, and no second matching subject"
                )
                if blocker is not None:
                    error += "; strict_fallback_blocker=" + json.dumps(
                        blocker, ensure_ascii=False, sort_keys=True
                    )
                errors.append(error)
        elif log_state == "unknown":
            errors.append(
                "subject log availability is unknown; support fails closed until "
                "candidate metadata or a successful metric inventory proves it"
            )
        return errors

    def _comparison_quality_errors(self, hypothesis_ids: Sequence[str]) -> list[str]:
        if self.config.context_strategy == "frontier_reflection":
            return []
        supported_ids = [
            hypothesis_id
            for hypothesis_id in hypothesis_ids
            if hypothesis_id in self._hypotheses
            and self._hypotheses[hypothesis_id].status == "supported"
        ]
        coverage = comparison_coverage(list(self._observations.values()))
        if not supported_ids or coverage["satisfied"] is True:
            return []
        return [
            "supported hypothesis comparison coverage is incomplete: "
            f"required_count={coverage['required_count']}, "
            f"completed={coverage['completed']}, "
            "next_candidate_ids=" + json.dumps(coverage["next_candidate_ids"])
        ]

    def _public_fault_types(self) -> list[str]:
        values: set[str] = set()
        for observation in self._observations.values():
            if observation.tool != "summarize_anomalies" or not observation.summary:
                continue
            try:
                summary = json.loads(observation.summary)
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(summary, Mapping):
                continue
            raw_types = cast(Mapping[str, object], summary).get("public_fault_types")
            if not isinstance(raw_types, list):
                continue
            values.update(
                item
                for item in cast(list[object], raw_types)
                if isinstance(item, str) and item
            )
        return sorted(values)

    @staticmethod
    def _resolve_public_fault_type(
        statement: str, public_fault_types: Sequence[str]
    ) -> str | None:
        if statement in public_fault_types:
            return statement
        normalized_statement = " ".join(statement.split()).casefold()
        matches = [
            fault_type
            for fault_type in public_fault_types
            if (normalized_fault_type := " ".join(fault_type.split()).casefold())
            and re.search(
                rf"(?<!\w){re.escape(normalized_fault_type)}(?!\w)",
                normalized_statement,
            )
        ]
        return matches[0] if len(matches) == 1 else None

    def _hypothesis_is_progress(
        self, operation: UpdateHypothesisOperation, stable_key: str
    ) -> bool:
        prior_subjects: set[str] = set()
        prior_links: set[tuple[str, str, str]] = set()
        for record in self.store.records("hypothesis"):
            if record.stable_key == stable_key or not isinstance(record.payload, dict):
                continue
            payload = cast(dict[str, Any], record.payload)
            subject = payload.get("subject")
            if not isinstance(subject, str):
                continue
            prior_subjects.add(subject)
            for relation, field in (
                ("supporting", "supporting_evidence_refs"),
                ("opposing", "opposing_evidence_refs"),
            ):
                for evidence_id in cast(list[object], payload.get(field, [])):
                    prior_links.add((subject, relation, str(evidence_id)))
        new_links = {
            (operation.subject, relation, evidence_id)
            for relation, evidence_ids in (
                ("supporting", operation.supporting_evidence_refs),
                ("opposing", operation.opposing_evidence_refs),
            )
            for evidence_id in evidence_ids
        }
        return operation.subject not in prior_subjects or bool(new_links - prior_links)

    def _path_is_new(self, operation: SubmitPathOperation, stable_key: str) -> bool:
        signature = self._hash_json(operation.model_dump(mode="json"))
        for record in self.store.records("path"):
            if record.stable_key == stable_key or not isinstance(record.payload, dict):
                continue
            payload = cast(dict[str, Any], record.payload)
            comparable = {
                name: payload.get(name)
                for name in type(operation).model_fields
                if name != "type"
            }
            comparable["type"] = "submit_path"
            if self._hash_json(comparable) == signature:
                return False
        return True

    def _last_finalize_error_count(self) -> int | None:
        for record in reversed(self.store.records("investigation_event")):
            if record.stable_key == self._event_key("finalize_rejected"):
                continue
            if (
                isinstance(record.payload, dict)
                and record.payload.get("event_type") == "finalize_rejected"
            ):
                payload = cast(dict[str, Any], record.payload)
                errors = payload.get("errors")
                return (
                    len(cast(list[object], errors))
                    if isinstance(errors, list)
                    else None
                )
        return None

    def _deadline_expired(self) -> bool:
        return self._deadline_at > 0 and time.time() >= self._deadline_at

    def _exhaust_budget(self, reason: str) -> None:
        if self._terminal_status is not None:
            return
        self._terminal_status = "budget_exhausted"
        self._failure = reason
        self._record_budget("exhausted", reason=reason)

    def _handle_artifact_budget(self, exc: ArtifactBudgetExceeded) -> None:
        gap = {
            "kind": "artifact_budget",
            "status": "partial",
            "reason": str(exc),
            "used_bytes": exc.used,
            "requested_bytes": exc.requested,
            "max_bytes": exc.limit,
        }
        if gap not in self._gaps:
            self._gaps.append(gap)
        self._event("gap", gap, index=-1)
        self._exhaust_budget("artifact_bytes")

    def _record_budget(
        self,
        event: str,
        usage: Mapping[str, Any] | None = None,
        *,
        call_number: int | None = None,
        **details: Any,
    ) -> None:
        self.store.append(
            "budget_ledger",
            self._budget_key(event, call_number=call_number),
            self._budget_payload(event, usage, **details),
        )

    def _budget_payload(
        self,
        event: str,
        usage: Mapping[str, Any] | None = None,
        **details: Any,
    ) -> dict[str, Any]:
        return {
            "event": event,
            "model_calls_used": self._model_calls,
            "max_model_calls": self.config.max_model_calls,
            "provider_attempts_used": self._provider_attempts,
            "input_tokens_used": self._input_tokens_used,
            "max_input_tokens": self.config.max_input_tokens,
            "output_tokens_used": self._output_tokens_used,
            "max_output_tokens": self.config.max_output_tokens,
            "tool_attempts_used": self._tool_attempts,
            "max_tool_attempts": self.config.max_tool_attempts,
            "validator_retries_used": self._validator_retries,
            "max_validator_retries": self.config.max_validator_retries,
            "no_progress_rounds": self._no_progress_rounds,
            "max_no_progress_rounds": self.config.max_no_progress_rounds,
            "artifact_bytes_used": self.artifacts.bytes_used,
            "max_artifact_bytes": self._max_artifact_bytes,
            "deadline_at": self._deadline_at,
            "usage": dict(usage or {}),
            "cumulative_usage": self._model_usage,
            **details,
        }

    def _budget_key(self, event: str, *, call_number: int | None = None) -> str:
        suffix = (
            f"model:{call_number}:{event}"
            if call_number is not None
            else f"{self.store.revision + 1}:{event}"
        )
        return f"budget:{self._run_id}:{suffix}"

    def _event(
        self, event_type: str, payload: Mapping[str, Any], *, index: int = 0
    ) -> None:
        self.store.append(
            "investigation_event",
            self._event_key(event_type, index),
            {"event_type": event_type, **payload},
        )

    def _event_key(self, event_type: str, index: int = 0) -> str:
        return f"event:{self._run_id}:{self._decision_id}:{index}:{event_type}"

    def _unbounded_reasoning_enabled(self) -> bool:
        return self.config.context_strategy == "frontier_reflection"

    def _strategy_e_enabled(self) -> bool:
        return bool(
            not self._unbounded_reasoning_enabled()
            and self.config.validator_mode == "enforce"
            and self.environment.task.expected_fault_count == 1
            and self._overview_candidate_registry()[1]
        )

    def _minimum_candidate_disposition_coverage(self) -> dict[str, Any]:
        queue = candidate_disposition_queue(
            list(self._observations.values()), list(self._evidence.values()), ()
        )
        targets: list[dict[str, str]] = []
        seen_series: set[tuple[str, str]] = set()
        seen_families: set[str] = set()
        for candidate in queue:
            values = tuple(
                candidate.get(name)
                for name in (
                    "candidate_id",
                    "entity_id",
                    "metric_id",
                    "resource_family",
                )
            )
            if not all(isinstance(value, str) and value for value in values):
                continue
            candidate_id, entity_id, metric_id, family = cast(
                tuple[str, str, str, str], values
            )
            series = (entity_id, metric_id)
            if series in seen_series or family in seen_families:
                continue
            targets.append(
                {
                    "candidate_id": candidate_id,
                    "entity_id": entity_id,
                    "metric_id": metric_id,
                    "resource_family": family,
                }
            )
            seen_series.add(series)
            seen_families.add(family)
            if len(targets) == 4:
                break

        latest: dict[str, tuple[int, dict[str, Any]]] = {}
        target_ids = {target["candidate_id"] for target in targets}
        for record in self.store.records("investigation_event"):
            if not isinstance(record.payload, dict):
                continue
            event = cast(dict[str, Any], record.payload)
            candidate_id = event.get("candidate_id")
            if (
                event.get("event_type") == "candidate_disposition_applied"
                and isinstance(candidate_id, str)
                and candidate_id in target_ids
                and (
                    candidate_id not in latest
                    or record.revision > latest[candidate_id][0]
                )
            ):
                latest[candidate_id] = record.revision, event

        completed: list[dict[str, str]] = []
        for target in targets:
            latest_event = latest.get(target["candidate_id"])
            if latest_event is None:
                continue
            event = latest_event[1]
            if (
                event.get("status") in {"supported", "refuted", "inconclusive"}
                and event.get("subject") == target["entity_id"]
                and event.get("metric_id") == target["metric_id"]
                and event.get("resource_family") == target["resource_family"]
            ):
                completed.append(target)
        completed_ids = {target["candidate_id"] for target in completed}
        remaining = [
            target for target in targets if target["candidate_id"] not in completed_ids
        ]
        return {
            "policy": "finding-min-disposition-v2",
            "required_count": len(targets),
            "completed_count": len(completed),
            "target_candidate_ids": [target["candidate_id"] for target in targets],
            "completed_candidate_ids": [target["candidate_id"] for target in completed],
            "remaining_candidate_ids": [target["candidate_id"] for target in remaining],
            "remaining_resource_families": [
                target["resource_family"] for target in remaining
            ],
            "satisfied": len(completed) == len(targets),
        }

    def _minimum_disposition_budget_available(self) -> bool:
        return bool(
            self._validator_retries < self.config.max_validator_retries
            and not self._deadline_expired()
            and self.config.max_model_calls - self._model_calls > CLOSEOUT_ROUND_BUDGET
            and self.config.max_model_calls - self._provider_attempts
            > CLOSEOUT_ROUND_BUDGET
            and self.config.max_input_tokens - self._input_tokens_used
            > CLOSEOUT_ROUND_BUDGET * self.config.max_context_tokens
            and self.config.max_output_tokens - self._output_tokens_used
            > CLOSEOUT_ROUND_BUDGET * self.config.max_output_tokens_per_call
            and self._tool_attempts < self.config.max_tool_attempts
        )

    def _minimum_candidate_disposition_required(self) -> bool:
        return bool(
            self._minimum_candidate_disposition_gate()
            and any(
                hypothesis.status == "supported"
                for hypothesis in self._hypotheses.values()
            )
        )

    def _operation_selects_supported_findings(
        self, operation: FinalizeOperation
    ) -> bool:
        return bool(
            operation.abstain_reason is None
            and operation.hypothesis_ids
            and all(
                hypothesis_id in self._hypotheses
                and self._hypotheses[hypothesis_id].status == "supported"
                for hypothesis_id in operation.hypothesis_ids
            )
        )

    def _minimum_disposition_rejection(
        self,
    ) -> tuple[str | None, dict[str, Any] | None] | None:
        for record in self.store.records("investigation_event"):
            if not isinstance(record.payload, dict):
                continue
            event = cast(dict[str, Any], record.payload)
            if event.get("event_type") == "finalize_rejected" and event.get(
                "policy"
            ) in {
                "weak-finding-min-disposition-v1",
                "finding-min-disposition-v2",
            }:
                raw_feedback = event.get("feedback")
                feedback = (
                    cast(dict[str, Any], raw_feedback[0])
                    if isinstance(raw_feedback, list)
                    and raw_feedback
                    and isinstance(raw_feedback[0], dict)
                    else None
                )
                decision_id = event.get("decision_id")
                if not isinstance(decision_id, str) and feedback is not None:
                    feedback_id = feedback.get("feedback_id")
                    prefix = "minimum_disposition:"
                    decision_id = (
                        feedback_id[len(prefix) :]
                        if isinstance(feedback_id, str)
                        and feedback_id.startswith(prefix)
                        else None
                    )
                return (
                    decision_id if isinstance(decision_id, str) else None,
                    feedback,
                )
        return None

    def _minimum_candidate_disposition_incomplete(self) -> bool:
        coverage = self._minimum_candidate_disposition_coverage()
        return bool(
            self._strategy_e_enabled()
            and coverage["required_count"]
            and not coverage["satisfied"]
        )

    def _minimum_candidate_disposition_gate(self) -> bool:
        return bool(
            self._minimum_candidate_disposition_incomplete()
            and self._minimum_disposition_budget_available()
        )

    def _minimum_candidate_disposition_feedback(self) -> dict[str, Any] | None:
        persisted = self._minimum_disposition_rejection()
        if persisted is not None:
            decision_id, feedback = persisted
            return (
                dict(feedback)
                if decision_id == self._decision_id and feedback is not None
                else None
            )
        retry_started = self._current_validator_retry_started()
        if retry_started:
            if not self._minimum_candidate_disposition_incomplete():
                return None
        elif not self._minimum_candidate_disposition_gate():
            return None
        coverage = self._minimum_candidate_disposition_coverage()
        return {
            "feedback_id": f"minimum_disposition:{self._decision_id}",
            "type": "exit_gate",
            "category": "quality",
            "policy": coverage["policy"],
            "required_count": coverage["required_count"],
            "completed_count": coverage["completed_count"],
            "remaining_candidate_ids": coverage["remaining_candidate_ids"],
            "remaining_resource_families": coverage["remaining_resource_families"],
            "required_correction": (
                "Use update_hypothesis to disposition each remaining candidate in "
                "order as supported, refuted, or inconclusive before finalizing."
            ),
        }

    def _current_validator_retry_started(self) -> bool:
        record = self.store.get(
            self._budget_key("validator_retry_started", call_number=self._model_calls)
        )
        if record is None or not isinstance(record.payload, dict):
            return False
        payload = cast(dict[str, Any], record.payload)
        decision_id = payload.get("decision_id")
        return bool(
            payload.get("event") == "validator_retry_started"
            and payload.get("model_calls_used") == self._model_calls
            and (decision_id is None or decision_id == self._decision_id)
        )

    def _strategy_e_weak_finding(
        self, operation: FinalizeOperation
    ) -> dict[str, Any] | None:
        if not self._strategy_e_enabled() or operation.abstain_reason is not None:
            return None
        queue = candidate_disposition_queue(
            list(self._observations.values()), list(self._evidence.values())
        )
        initial_top1_candidate_id = queue[0].get("candidate_id") if queue else None
        initial_top1_subject = queue[0].get("entity_id") if queue else None
        initial_top1_fault_type = queue[0].get("fault_type") if queue else None
        initial_top1_source_strict = bool(queue and _has_strict_source_shift(queue[0]))
        hypothesis_id = (
            operation.hypothesis_ids[0] if len(operation.hypothesis_ids) == 1 else None
        )
        hypothesis = (
            self._hypotheses.get(hypothesis_id)
            if isinstance(hypothesis_id, str)
            else None
        )
        disposition_events = [
            cast(dict[str, Any], record.payload)
            for record in self.store.records("investigation_event")
            if isinstance(record.payload, dict)
            and record.payload.get("event_type") == "candidate_disposition_applied"
            and record.payload.get("hypothesis_id") == hypothesis_id
        ]
        bound_candidate_ids = {
            candidate_id
            for event in disposition_events
            for candidate_id in [event.get("candidate_id")]
            if isinstance(candidate_id, str) and candidate_id
        }
        bindings_converge = bool(disposition_events) and all(
            isinstance(event.get("candidate_id"), str)
            and bool(cast(str, event["candidate_id"]))
            for event in disposition_events
        )
        candidate_id = (
            next(iter(bound_candidate_ids)) if len(bound_candidate_ids) == 1 else None
        )
        candidate = next(
            (
                item
                for item in queue
                if candidate_id is not None and item.get("candidate_id") == candidate_id
            ),
            None,
        )
        bindings_converge = bool(
            bindings_converge
            and len(bound_candidate_ids) == 1
            and hypothesis is not None
            and all(
                event.get("candidate_id") == candidate_id
                and event.get("subject") == hypothesis.subject
                and event.get("selected_fault_type") == hypothesis.statement
                and event.get("status") in {"open", "supported"}
                for event in disposition_events
            )
            and disposition_events[-1].get("status") == "supported"
        )
        source_strict = bool(
            candidate is not None and _has_strict_source_shift(candidate)
        )
        raw_rank = candidate.get("candidate_rank") if candidate is not None else None
        candidate_rank = (
            raw_rank
            if isinstance(raw_rank, int)
            and not isinstance(raw_rank, bool)
            and raw_rank >= 0
            else None
        )
        selected_fault_type = hypothesis.statement if hypothesis is not None else None
        selected_subject = hypothesis.subject if hypothesis is not None else None
        candidate_fault_types: set[str] = set()
        if candidate is not None:
            primary_fault_type = candidate.get("fault_type")
            if isinstance(primary_fault_type, str):
                candidate_fault_types.add(primary_fault_type)
            raw_aliases = candidate.get("fault_type_aliases", [])
            if isinstance(raw_aliases, list):
                candidate_fault_types.update(
                    alias
                    for alias in cast(list[object], raw_aliases)
                    if isinstance(alias, str)
                )
        higher_rank_same_fault_candidate_ids = [
            str(item["candidate_id"])
            for item in queue
            if selected_fault_type
            in {
                item.get("fault_type"),
                *cast(list[object], item.get("fault_type_aliases", [])),
            }
            and item.get("entity_id") == selected_subject
            and isinstance(item.get("candidate_rank"), int)
            and not isinstance(item.get("candidate_rank"), bool)
            and candidate_rank is not None
            and cast(int, item["candidate_rank"]) < candidate_rank
            and isinstance(item.get("candidate_id"), str)
        ]
        strong = bool(
            hypothesis is not None
            and hypothesis.status == "supported"
            and bindings_converge
            and candidate is not None
            and selected_fault_type in candidate_fault_types
            and source_strict
            and candidate_rank == 0
        )
        if strong:
            return None
        return {
            "event_type": "weak_finding_rejected",
            "visibility": "audit_only",
            "policy": "strong-source-no-higher-peer-v1",
            "abstain_reason": (
                "selected Finding did not satisfy the pre-registered strong-Finding "
                "policy"
            ),
            "hypothesis_id": hypothesis_id,
            "candidate_id": candidate_id,
            "source_strict": source_strict,
            "candidate_rank": candidate_rank,
            "higher_rank_same_fault_candidate_ids": (
                higher_rank_same_fault_candidate_ids
            ),
            "initial_top1_candidate_id": initial_top1_candidate_id,
            "initial_top1_subject": initial_top1_subject,
            "initial_top1_fault_type": initial_top1_fault_type,
            "initial_top1_source_strict": initial_top1_source_strict,
            "original_finalize_operation": operation.model_dump(mode="json"),
        }

    def _benchmark_recommendation(
        self,
        status: str,
        request: FinalizeOperation | None,
        selected_hypotheses: Sequence[str],
    ) -> dict[str, Any] | None:
        if self.config.context_strategy == "frontier_reflection":
            return self._frontier_benchmark_recommendation(
                status, request, selected_hypotheses
            )
        if (
            status != "completed_partial"
            or request is None
            or not isinstance(request.abstain_reason, str)
            or not request.abstain_reason.strip()
            or selected_hypotheses
            or self.environment.task.expected_fault_count != 1
            or not self._overview_candidate_registry()[1]
            or request.abstain_reason
            == (
                "minimum candidate disposition coverage remained incomplete after "
                "one correction"
            )
        ):
            return None

        queue = candidate_disposition_queue(
            list(self._observations.values()), list(self._evidence.values())
        )
        queue_by_id = {
            str(item["candidate_id"]): (rank, item)
            for rank, item in enumerate(queue)
            if isinstance(item.get("candidate_id"), str)
        }
        bindings: dict[str, list[dict[str, Any]]] = {}
        for record in self.store.records("investigation_event"):
            if not isinstance(record.payload, dict):
                continue
            event = cast(dict[str, Any], record.payload)
            hypothesis_id = event.get("hypothesis_id")
            if (
                event.get("event_type") == "candidate_disposition_applied"
                and isinstance(hypothesis_id, str)
                and hypothesis_id
            ):
                bindings.setdefault(hypothesis_id, []).append(event)
        ever_open = {
            str(payload["hypothesis_id"])
            for record in self.store.records("hypothesis")
            if isinstance(record.payload, dict)
            for payload in [cast(dict[str, Any], record.payload)]
            if payload.get("status") == "open"
            and isinstance(payload.get("hypothesis_id"), str)
        }

        eligible: list[
            tuple[str, Hypothesis, str, int, Mapping[str, Any], bool, bool, bool]
        ] = []
        for hypothesis_id, hypothesis in self._hypotheses.items():
            if hypothesis.status != "inconclusive":
                continue
            hypothesis_bindings = bindings.get(hypothesis_id, [])
            candidate_ids = {
                event.get("candidate_id")
                for event in hypothesis_bindings
                if isinstance(event.get("candidate_id"), str)
                and event.get("candidate_id")
            }
            if (
                not hypothesis_bindings
                or len(candidate_ids) != 1
                or not all(
                    event.get("candidate_id") in candidate_ids
                    and event.get("subject") == hypothesis.subject
                    and event.get("selected_fault_type") == hypothesis.statement
                    and event.get("status") in {"open", "inconclusive"}
                    for event in hypothesis_bindings
                )
                or hypothesis_bindings[-1].get("status") != hypothesis.status
            ):
                continue
            candidate_id = next(iter(candidate_ids))
            if not isinstance(candidate_id, str) or candidate_id not in queue_by_id:
                continue
            terminal_rank, candidate = queue_by_id[candidate_id]
            source_strict = _has_strict_source_shift(candidate)
            has_log_source = candidate.get("has_log_source") is True
            direct_strict = _has_strict_direct_shift(candidate)
            eligible.append(
                (
                    hypothesis_id,
                    hypothesis,
                    candidate_id,
                    terminal_rank,
                    candidate,
                    source_strict,
                    has_log_source,
                    direct_strict,
                )
            )

        recommendations: list[dict[str, Any]] = []
        for (
            hypothesis_id,
            hypothesis,
            candidate_id,
            terminal_rank,
            _candidate,
            source_strict,
            has_log_source,
            direct_strict,
        ) in eligible:
            selection = {
                "inconclusive": True,
                "source_strict": source_strict,
                "has_log_source": has_log_source,
                "direct_strict": direct_strict,
                "ever_open": hypothesis_id in ever_open,
                "terminal_queue_rank": terminal_rank,
            }
            recommendations.append(
                {
                    "candidate_id": candidate_id,
                    "hypothesis": hypothesis.model_dump(mode="json"),
                    "selection": selection,
                }
            )
        if not recommendations:
            return self._rejected_supported_recommendation(queue)
        supported_hypotheses_by_subject: dict[str, list[str]] = {}
        for hypothesis_id, hypothesis in self._hypotheses.items():
            if (
                supported_candidate_binding(
                    hypothesis,
                    bindings.get(hypothesis_id, []),
                    queue_by_id,
                    self._evidence,
                    self._observations,
                )
                is not None
            ):
                supported_hypotheses_by_subject.setdefault(
                    hypothesis.subject, []
                ).append(hypothesis_id)
        selected, switch = select_terminal_candidate(
            recommendations, supported_hypotheses_by_subject
        )
        if selected is None:
            return None
        if switch is not None:
            selected = {
                **selected,
                "selection": {
                    **cast(dict[str, Any], selected["selection"]),
                    **switch,
                },
            }
        pre_family_policy = (
            "terminal-candidate-v5" if switch is not None else "terminal-candidate-v4"
        )
        family_selected, family_audit = select_family_compatible_terminal_candidate(
            recommendations,
            selected,
            {
                candidate_id: cast(str, candidate.get("resource_family"))
                for candidate_id, (_rank, candidate) in queue_by_id.items()
                if isinstance(candidate.get("resource_family"), str)
            },
            pre_family_policy,
        )
        if family_audit is not None:
            selected = {
                **family_selected,
                "selection": {
                    **cast(dict[str, Any], family_selected["selection"]),
                    **family_audit,
                },
            }
        return {
            "schema_version": "1",
            "policy": (
                "terminal-candidate-v6"
                if family_audit is not None
                else pre_family_policy
            ),
            "confidence": "low",
            "usage": "openrca_export_fallback_only",
            "is_finding": False,
            **selected,
            "eligible_terminal_count": len(recommendations),
            "audit_ref": self._event_key("benchmark_recommendation_selected"),
        }

    def _occurrence_recommendation(
        self, status: str, request: FinalizeOperation | None
    ) -> dict[str, Any] | None:
        if (
            status != "completed_partial"
            or request is None
            or not request.occurrence_evidence_refs
            or self._occurrence_selection_errors(request)
        ):
            return None
        selected = [
            self._evidence[evidence_id]
            for evidence_id in request.occurrence_evidence_refs
        ]
        onsets = [cast(TimeRange, item.onset_interval) for item in selected]
        overlap_start = max(item.start for item in onsets)
        overlap_end = min(item.end for item in onsets)
        occurrence = overlap_start + (overlap_end - overlap_start) / 2
        common_subject_ids: set[str] = set(selected[0].subject_ids)
        for item in selected[1:]:
            common_subject_ids.intersection_update(item.subject_ids)
        common_subjects = sorted(common_subject_ids)
        return {
            "schema_version": "1",
            "policy": "source-backed-occurrence-v1",
            "confidence": "low",
            "is_finding": False,
            "subject_id": common_subjects[0],
            "evidence_refs": list(request.occurrence_evidence_refs),
            "onset_interval": {
                "start": overlap_start.isoformat(),
                "end": overlap_end.isoformat(),
                "end_exclusive": True,
            },
            "occurrence_datetime": occurrence.isoformat(),
            "audit_ref": self._event_key("occurrence_recommendation_selected"),
        }

    def _frontier_benchmark_recommendation(
        self,
        status: str,
        request: FinalizeOperation | None,
        selected_hypotheses: Sequence[str],
    ) -> dict[str, Any] | None:
        if (
            status != "completed_partial"
            or request is None
            or not isinstance(request.abstain_reason, str)
            or not request.abstain_reason.strip()
            or selected_hypotheses
            or self.environment.task.expected_fault_count != 1
            or self.environment.task.requested_outputs
            == ["root cause occurrence datetime"]
            or self._current_best_hypothesis_id not in self._hypotheses
        ):
            return None
        hypothesis_id = self._current_best_hypothesis_id
        hypothesis = self._hypotheses[hypothesis_id]
        if hypothesis.status not in {"open", "supported"}:
            return None
        assessments = self._latest_frontier_assessments()
        if any(
            item.hypothesis_id == hypothesis_id and item.impact == "refutes"
            for item in assessments.values()
        ):
            return None
        supporting = {
            item.evidence_id
            for item in assessments.values()
            if item.hypothesis_id == hypothesis_id and item.impact == "supports"
        }
        eligible: list[tuple[int, float, str, str, str]] = []
        all_ready: list[tuple[int, float, str, str]] = []
        for evidence in self._evidence.values():
            if (
                evidence.semantic_quality != "typed"
                or evidence.evidence_kind not in {"metric_profile", "service_profile"}
                or not isinstance(evidence.value, Mapping)
            ):
                continue
            summary = cast(Mapping[str, object], evidence.value).get(
                "finding_gate_summary"
            )
            if not isinstance(summary, Mapping):
                continue
            values = cast(Mapping[str, object], summary)
            fault_type = values.get("requested_fault_type")
            signature = values.get("mechanism_signature")
            raw_score = values.get("root_cause_score")
            if (
                (
                    values.get("fault_type_ready") is not True
                    and values.get("material_event_ready") is not True
                )
                or values.get("boundary_aligned") is True
                or not isinstance(fault_type, str)
                or not isinstance(signature, str)
                or not isinstance(raw_score, (int, float))
                or isinstance(raw_score, bool)
            ):
                continue
            rank = mechanism_signature_rank(values)
            score = float(raw_score)
            for subject in evidence.subject_ids:
                all_ready.append((rank, score, subject, evidence.evidence_id))
                if subject == hypothesis.subject and evidence.evidence_id in supporting:
                    eligible.append(
                        (
                            rank,
                            score,
                            evidence.evidence_id,
                            fault_type,
                            signature,
                        )
                    )
        if not eligible:
            return None
        rank, score, evidence_id, fault_type, signature = sorted(
            eligible,
            key=lambda item: (-item[0], -item[1], item[2]),
        )[0]
        if any(
            subject != hypothesis.subject and (other_rank, other_score) >= (rank, score)
            for other_rank, other_score, subject, _other_evidence_id in all_ready
        ):
            return None
        latest_dispositions = self._latest_anomaly_dispositions()
        unresolved = sorted(
            anomaly_id
            for anomaly_id in self._material_anomalies
            if anomaly_id not in latest_dispositions
            or latest_dispositions[anomaly_id].disposition == "unresolved"
        )
        if not unresolved:
            return None
        candidate_id = f"frontier:{hypothesis_id}"
        return {
            "schema_version": "1",
            "policy": "frontier-supported-v1",
            "confidence": "low",
            "usage": "openrca_export_fallback_only",
            "is_finding": False,
            "candidate_id": candidate_id,
            "hypothesis": hypothesis.model_dump(mode="json"),
            "selection": {
                "current_best_hypothesis_id": hypothesis_id,
                "supporting_evidence_id": evidence_id,
                "public_fault_type": fault_type,
                "mechanism_signature": signature,
                "mechanism_rank": rank,
                "root_cause_score": score,
                "unresolved_anomaly_ids": unresolved,
            },
            "audit_ref": self._event_key("benchmark_recommendation_selected"),
        }

    def _rejected_supported_recommendation(
        self, queue: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        rejection = self._weak_finding_rejection
        if rejection is None:
            return None
        hypothesis_id = rejection.get("hypothesis_id")
        candidate_id = rejection.get("candidate_id")
        if not isinstance(hypothesis_id, str) or not isinstance(candidate_id, str):
            return None
        hypothesis = self._hypotheses.get(hypothesis_id)
        if (
            hypothesis is None
            or hypothesis.status != "supported"
            or sum(item.status == "supported" for item in self._hypotheses.values())
            != 1
        ):
            return None
        candidate = next(
            (item for item in queue if item.get("candidate_id") == candidate_id), None
        )
        if candidate is None:
            return None
        bindings = [
            (record.stable_key, cast(dict[str, Any], record.payload))
            for record in self.store.records("investigation_event")
            if isinstance(record.payload, dict)
            and record.payload.get("event_type") == "candidate_disposition_applied"
            and record.payload.get("hypothesis_id") == hypothesis_id
        ]
        if not bindings:
            return None
        allowed_fault_types = [candidate.get("fault_type")]
        raw_aliases = candidate.get("fault_type_aliases", [])
        if isinstance(raw_aliases, list):
            allowed_fault_types.extend(cast(list[object], raw_aliases))
        allowed_fault_types = [
            value for value in allowed_fault_types if isinstance(value, str) and value
        ]
        expected_log_hint = candidate.get("log_selector_hint")
        if not isinstance(expected_log_hint, Mapping) or not isinstance(
            expected_log_hint.get("exact_log_names"), list
        ):
            return None
        if not all(
            event.get("candidate_id") == candidate_id
            and event.get("subject") == hypothesis.subject
            and event.get("selected_fault_type") == hypothesis.statement
            and event.get("status") in {"open", "supported"}
            and event.get("allowed_fault_types") == allowed_fault_types
            and event.get("resource_family") == candidate.get("resource_family")
            and event.get("log_selector_hint") == expected_log_hint
            for _stable_key, event in bindings
        ):
            return None
        disposition_ref, disposition = bindings[-1]
        if (
            disposition.get("status") != "supported"
            or disposition.get("candidate_id") != rejection.get("candidate_id")
            or disposition.get("hypothesis_id") != rejection.get("hypothesis_id")
            or disposition.get("subject") != hypothesis.subject
            or disposition.get("selected_fault_type") != hypothesis.statement
        ):
            return None
        original = disposition.get("original_operation")
        applied = disposition.get("applied_operation")
        if not isinstance(original, Mapping) or not isinstance(applied, Mapping):
            return None
        original = cast(Mapping[str, Any], original)
        applied = cast(Mapping[str, Any], applied)
        if (
            original.get("status") != "supported"
            or original.get("subject") != hypothesis.subject
            or not isinstance(original.get("statement"), str)
            or self._resolve_public_fault_type(
                cast(str, original["statement"]), allowed_fault_types
            )
            != hypothesis.statement
            or applied.get("status") != "supported"
            or applied.get("subject") != hypothesis.subject
            or applied.get("statement") != hypothesis.statement
        ):
            return None
        pair_refs = {
            disposition.get("reference_evidence_id"),
            disposition.get("incident_evidence_id"),
        }
        if len(pair_refs) != 2 or not all(
            isinstance(evidence_id, str) and evidence_id for evidence_id in pair_refs
        ):
            return None
        reference_evidence_id = cast(str, disposition["reference_evidence_id"])
        incident_evidence_id = cast(str, disposition["incident_evidence_id"])
        if not self._has_candidate_metric_pair_evidence(
            reference_evidence_id,
            incident_evidence_id,
            hypothesis.subject,
            candidate.get("metric_id"),
        ):
            return None
        raw_original_refs = original.get("supporting_evidence_refs")
        raw_applied_refs = applied.get("supporting_evidence_refs")
        raw_logged_refs = disposition.get("log_evidence_ids")
        if not all(
            isinstance(value, list)
            for value in (raw_original_refs, raw_applied_refs, raw_logged_refs)
        ):
            return None
        original_refs = {
            item
            for item in cast(list[object], raw_original_refs)
            if isinstance(item, str)
        }
        applied_refs = {
            item
            for item in cast(list[object], raw_applied_refs)
            if isinstance(item, str)
        }
        final_refs = set(hypothesis.supporting_evidence_refs)
        if not pair_refs <= original_refs & applied_refs & final_refs:
            return None
        logged_refs = {
            item
            for item in cast(list[object], raw_logged_refs)
            if isinstance(item, str)
        }
        filtered_log_evidence_ids = sorted(
            evidence_id
            for evidence_id in original_refs & applied_refs & final_refs & logged_refs
            if self._is_filtered_candidate_log_evidence(evidence_id, hypothesis.subject)
        )
        if not filtered_log_evidence_ids:
            return None
        original_finalize = rejection.get("original_finalize_operation")
        if (
            rejection.get("event_type") != "weak_finding_rejected"
            or rejection.get("visibility") != "audit_only"
            or rejection.get("policy") != "strong-source-no-higher-peer-v1"
            or rejection.get("abstain_reason")
            != (
                "selected Finding did not satisfy the pre-registered "
                "strong-Finding policy"
            )
            or not isinstance(original_finalize, Mapping)
            or original_finalize.get("type") != "finalize"
            or original_finalize.get("abstain_reason") is not None
            or original_finalize.get("hypothesis_ids") != [hypothesis_id]
        ):
            return None
        return {
            "schema_version": "1",
            "policy": "rejected-supported-filtered-log-v1",
            "confidence": "low",
            "usage": "openrca_export_fallback_only",
            "is_finding": False,
            "candidate_id": candidate_id,
            "hypothesis": hypothesis.model_dump(mode="json"),
            "filtered_log_evidence_ids": filtered_log_evidence_ids,
            "weak_finding_rejected_ref": self._event_key("weak_finding_rejected"),
            "disposition_ref": disposition_ref,
            "audit_ref": self._event_key("benchmark_recommendation_selected"),
        }

    def _has_candidate_metric_pair_evidence(
        self,
        reference_evidence_id: str,
        incident_evidence_id: str,
        subject: str,
        metric_id: object,
    ) -> bool:
        if reference_evidence_id == incident_evidence_id or not isinstance(
            metric_id, str
        ):
            return False
        for evidence_id, period in (
            (reference_evidence_id, "reference"),
            (incident_evidence_id, "incident"),
        ):
            evidence = self._evidence.get(evidence_id)
            if evidence is None or subject not in evidence.subject_ids:
                return False
            if not any(
                observation_id in self._observations
                and (observation := self._observations[observation_id]).tool
                == "query_metric"
                and observation.status in {"success", "partial"}
                and observation.arguments.get("entity_id") == subject
                and observation.arguments.get("metric_id") == metric_id
                and observation.arguments.get("period", "incident") == period
                for observation_id in evidence.observation_refs
            ):
                return False
        return True

    def _is_filtered_candidate_log_evidence(
        self,
        evidence_id: str,
        subject: str,
    ) -> bool:
        evidence = self._evidence.get(evidence_id)
        if (
            evidence is None
            or subject not in evidence.subject_ids
            or not _has_complete_paired_log_summary(evidence.value)
        ):
            return False
        for observation_id in evidence.observation_refs:
            observation = self._observations.get(observation_id)
            log_name = observation.arguments.get("log_name") if observation else None
            if (
                observation is None
                or observation.tool != "query_log_samples"
                or observation.status not in {"success", "partial"}
                or not isinstance(log_name, str)
                or not log_name.strip()
            ):
                continue
            value = cast(Mapping[str, Any], evidence.value)
            summary = cast(Mapping[str, Any], value["paired_log_summary"])
            incident = summary.get("incident")
            if not isinstance(incident, Mapping):
                continue
            incident = cast(Mapping[str, Any], incident)
            counts = incident.get("log_name_counts")
            if (
                isinstance(incident.get("matched_count"), int)
                and cast(int, incident["matched_count"]) > 0
                and isinstance(counts, Mapping)
                and isinstance(cast(Mapping[str, Any], counts).get(log_name), int)
                and cast(int, cast(Mapping[str, Any], counts)[log_name]) > 0
            ):
                return True
        return False

    def _seal(self) -> None:
        if self._report is not None:
            return
        status = self._terminal_status or "runtime_failure"
        final_manifest_key = f"manifest:final:{self._run_id}"
        existing_final_manifest = self.store.get(final_manifest_key)
        existing_ended_at: object = (
            cast(dict[str, object], existing_final_manifest.payload).get("ended_at")
            if (
                existing_final_manifest is not None
                and isinstance(existing_final_manifest.payload, dict)
            )
            else None
        )
        if isinstance(existing_ended_at, (int, float)):
            ended_at = float(existing_ended_at)
        else:
            ended_at = time.time()
        wall_clock_seconds = max(0.0, ended_at - self._started_at)
        request = self._final_request
        selected_hypotheses = request.hypothesis_ids if request else []
        selected_paths = request.path_ids if request else []
        occurrence_recommendation = self._occurrence_recommendation(status, request)
        benchmark_recommendation = self._benchmark_recommendation(
            status, request, selected_hypotheses
        )
        outcome_type = (
            "finding"
            if selected_hypotheses
            else "occurrence_recommendation"
            if occurrence_recommendation is not None
            else "recommendation"
            if benchmark_recommendation is not None
            else "abstention"
            if request is not None and request.abstain_reason is not None
            else "incomplete"
        )
        report: dict[str, Any] = {
            "run_id": self._run_id,
            "task_id": self.environment.task.task_id,
            "status": status,
            "started_at": self._started_at,
            "ended_at": ended_at,
            "wall_clock_seconds": wall_clock_seconds,
            "model_calls": self._model_calls,
            "model_usage": self._model_usage,
            "observability": _observability_provenance(self.observability_config),
            "query_sandbox_requested": self.config.query_sandbox_requested,
            "query_sandbox_effective": self.config.query_sandbox_effective,
            "query_sandbox_bwrap_version": self.config.query_sandbox_bwrap_version,
            "query_sandbox_bwrap_executable_sha256": (
                self.config.query_sandbox_bwrap_executable_sha256
            ),
            "budget": {
                "deadline_at": self._deadline_at,
                "provider_attempts_used": self._provider_attempts,
                "input_tokens_used": self._input_tokens_used,
                "output_tokens_used": self._output_tokens_used,
                "tool_attempts_used": self._tool_attempts,
                "validator_retries_used": self._validator_retries,
                "no_progress_rounds": self._no_progress_rounds,
                "artifact_bytes_used": self.artifacts.bytes_used,
                "max_artifact_bytes": self._max_artifact_bytes,
            },
            "hypothesis_ids": selected_hypotheses,
            "path_ids": selected_paths,
            "occurrence_evidence_refs": (
                request.occurrence_evidence_refs if request else []
            ),
            "findings": [
                self._hypotheses[item].model_dump(mode="json")
                for item in selected_hypotheses
            ],
            "paths": [
                {
                    **self._paths[item].model_dump(mode="json"),
                    "grounding": self._grounding[item],
                }
                for item in selected_paths
            ],
            "unresolved_questions": (request.unresolved_questions if request else []),
            "abstain_reason": request.abstain_reason if request else None,
            "gaps": self._gaps,
            "failure": self._failure,
            "candidates": [
                {
                    "hypothesis": hypothesis.model_dump(mode="json"),
                    "paths": [
                        {
                            **path.model_dump(mode="json"),
                            "grounding": self._grounding[path_id],
                        }
                        for path_id, path in self._paths.items()
                        if path.hypothesis_id == hypothesis_id
                    ],
                    "selected": hypothesis_id in selected_hypotheses,
                }
                for hypothesis_id, hypothesis in self._hypotheses.items()
            ],
        }
        if benchmark_recommendation is not None:
            report["benchmark_recommendation"] = benchmark_recommendation
        if occurrence_recommendation is not None:
            report["occurrence_recommendation"] = occurrence_recommendation
        if self.config.context_strategy == "frontier_reflection":
            report.update(
                outcome_type=outcome_type,
                evidence_assessments=[
                    item.model_dump(mode="json") for item in self._assessments.values()
                ],
                material_anomalies=[
                    item.model_dump(mode="json")
                    for item in self._material_anomalies.values()
                ],
                anomaly_dispositions=[
                    item.model_dump(mode="json")
                    for item in self._anomaly_dispositions.values()
                ],
                anomaly_coverage={
                    "total": len(self._material_anomalies),
                    "resolved": sum(
                        item.disposition != "unresolved"
                        for item in self._anomaly_dispositions.values()
                    ),
                    "unresolved": sum(
                        anomaly_id not in self._anomaly_dispositions
                        or self._anomaly_dispositions[anomaly_id].disposition
                        == "unresolved"
                        for anomaly_id in self._material_anomalies
                    ),
                },
                reflections=[
                    record.payload for record in self.store.records("reflection")
                ],
                current_best_hypothesis_id=self._current_best_hypothesis_id,
            )
        report_ref = f"report:{self._run_id}"
        manifest_record = self.store.get(f"manifest:{self._run_id}")
        manifest = (
            dict(cast(dict[str, Any], manifest_record.payload))
            if manifest_record is not None and isinstance(manifest_record.payload, dict)
            else {"run_id": self._run_id, "started_at": self._started_at}
        )
        manifest.update(
            {
                "ended_at": ended_at,
                "wall_clock_seconds": wall_clock_seconds,
            }
        )
        with generic_span(
            self.tracer,
            "rootwalk.persistence",
            self.observability_config,
            run_id=self._run_id,
            execution_attempt_id=self._execution_attempt_id,
            attributes={"record_count": 3},
        ) as persistence_span:
            record_span_content(
                persistence_span,
                self.observability_config,
                input_value=manifest,
                output_value=report,
            )
            self.store.append("manifest", final_manifest_key, manifest)
            report_records: list[tuple[str, str, object]] = [
                ("report", report_ref, report)
            ]
            if self._weak_finding_rejection is not None:
                report_records.append(
                    (
                        "investigation_event",
                        self._event_key("weak_finding_rejected"),
                        self._weak_finding_rejection,
                    )
                )
            if benchmark_recommendation is not None:
                report_records.append(
                    (
                        "investigation_event",
                        self._event_key("benchmark_recommendation_selected"),
                        {
                            "event_type": "benchmark_recommendation_selected",
                            "visibility": "audit_only",
                            "recommendation": benchmark_recommendation,
                        },
                    )
                )
            if occurrence_recommendation is not None:
                report_records.append(
                    (
                        "investigation_event",
                        self._event_key("occurrence_recommendation_selected"),
                        {
                            "event_type": "occurrence_recommendation_selected",
                            "visibility": "audit_only",
                            "recommendation": occurrence_recommendation,
                        },
                    )
                )
            report_records.append(
                (
                    "investigation_event",
                    self._event_key("run_sealed"),
                    {"event_type": "run_sealed", "status": status},
                )
            )
            self.store.append_many_and_seal(
                report_records, self._run_id, status, report_ref
            )
            self._report = report

    def _register_embedded_artifact(self, content_hash: str) -> None:
        self.store.append(
            "artifact",
            f"artifact:{content_hash}",
            {
                "content_hash": content_hash,
                "storage": "embedded",
                "media_type": "application/json",
            },
        )

    def _register_external_artifact(self, content_hash: str) -> None:
        if self.store.get(f"artifact:{content_hash}") is not None:
            return
        self.store.append(
            "artifact",
            f"artifact:{content_hash}",
            {
                "content_hash": content_hash,
                "storage": "external",
                "uri": f"bank://{self.environment.dataset_id}/{content_hash}",
            },
        )

    def _operation_id(self, index: int) -> list[str | int]:
        return [self._run_id, self._decision_id, index]

    def _operation_key(self, index: int) -> str:
        return f"{self._run_id}:{self._decision_id}:{index}"

    def _entity_id(self, kind: str, index: int) -> str:
        return f"{self._run_id}:{kind}:{self._decision_id}:{index}"

    def _unknown_coverage(self) -> dict[str, Any]:
        return {
            "requested_range": self.environment.task.allowed_query_range.model_dump(
                mode="json"
            ),
            "observed_range": None,
            "sample_interval": None,
            "completeness": "unknown",
            "sampling": None,
            "truncation": None,
            "collection_health": "unknown",
            "basis": "query was not completed",
        }

    def _topology_summary(self) -> dict[str, Any]:
        topology = self.environment.topology
        return {
            "snapshot_id": topology.snapshot_id,
            "source": topology.source,
            "valid_range": topology.valid_range.model_dump(mode="json"),
            "coverage": topology.coverage.model_dump(
                mode="json", exclude={"missing_parent_samples"}
            ),
            "nodes": [
                {
                    "canonical_id": node.canonical_id,
                    "entity_type": node.entity_type,
                    "sources": node.sources,
                }
                for node in topology.nodes
            ],
            "edges": [
                {
                    "edge_id": edge.edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation_type": edge.relation_type,
                    "fact_kind": edge.fact_kind,
                }
                for edge in topology.edges
            ],
        }

    def _read_artifact_slice(
        self,
        artifact_id: str,
        *,
        start_line: int = 1,
        max_lines: int = 50,
        max_bytes: int = 16 * 1024,
    ) -> dict[str, Any]:
        record = self.store.get(f"artifact:{artifact_id}")
        if record is None or not isinstance(record.payload, dict):
            return self.environment.read_artifact_slice(
                artifact_id,
                start_line=start_line,
                max_lines=max_lines,
                max_bytes=max_bytes,
            )
        if record.payload.get("storage") != "embedded":
            return self.environment.read_artifact_slice(
                artifact_id,
                start_line=start_line,
                max_lines=max_lines,
                max_bytes=max_bytes,
            )
        if start_line < 1 or max_lines < 1 or max_bytes < 1:
            return {
                "status": "error",
                "error_type": "invalid_arguments",
                "retryable": False,
                "summary": "start_line, max_lines, and max_bytes must be positive",
                "coverage": self._unknown_coverage(),
                "artifact_refs": [],
            }
        lines = self.artifacts.read_bytes(artifact_id).splitlines(keepends=True)
        selected: list[bytes] = []
        size = 0
        truncated = False
        for line in lines[start_line - 1 :]:
            if len(selected) == max_lines or size + len(line) > max_bytes:
                truncated = True
                break
            selected.append(line)
            size += len(line)
        return {
            "status": "partial"
            if truncated
            else ("success" if selected else "no_data"),
            "artifact_id": artifact_id,
            "content": b"".join(selected).decode("utf-8"),
            "start_line": start_line,
            "returned_lines": len(selected),
            "returned_bytes": size,
            "coverage": {
                **self._unknown_coverage(),
                "completeness": "partial" if truncated else "complete",
                "collection_health": "healthy",
                "truncation": "line or byte bound reached" if truncated else None,
                "basis": "content-addressed RootWalk artifact",
            },
            "artifact_refs": [
                {
                    "content_hash": artifact_id,
                    "selector": {
                        "start_line": start_line,
                        "returned_lines": len(selected),
                        "returned_bytes": size,
                    },
                }
            ],
        }

    @staticmethod
    def _hash_json(value: object) -> str:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return hashlib.sha256(payload).hexdigest()
