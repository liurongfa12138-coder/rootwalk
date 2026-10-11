"""Read-only providers for prepared OpenRCA public slices."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from bisect import insort
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from .contracts import Task, TimeRange, TopologyEdge, TopologyNode, TopologySnapshot
from .environment import CapabilityManifest

Result = dict[str, Any]

_MAX_LIST_LIMIT = 100
_MAX_ARTIFACT_LINES = 200
_MAX_ARTIFACT_BYTES = 64 * 1024
_FORBIDDEN_SOURCE_FILES = frozenset({"query.csv", "record.csv"})
_FORBIDDEN_SOURCE_NAME_TOKENS = frozenset(
    {
        "query",
        "queries",
        "record",
        "records",
    }
)
_FORBIDDEN_EVALUATION_TOKENS = frozenset(
    {
        "answer",
        "answers",
        "gold",
        "groundtruth",
        "label",
        "labels",
        "rootcause",
    }
)
_SAFE_LABEL_POLICY_VALUES = frozenset(
    {
        "excluded",
        "false",
        "none",
        "label-free public case spec; evaluation labels excluded",
        "public instruction/task_index allowlist; evaluation labels excluded",
        (
            "public field allowlist; scoring_points only in sibling hidden_eval, "
            "not referenced by public artifacts"
        ),
    }
)
_RESOURCE_FAMILIES = ("cpu", "memory", "network", "storage", "latency", "other")


def _boundary_tokens(value: str) -> tuple[str, ...]:
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    return tuple(
        token for token in re.split(r"[^a-z0-9]+", separated.casefold()) if token
    )


def _has_token_pair(tokens: Sequence[str], left: str, right: str) -> bool:
    return any(
        first == left and second == right for first, second in zip(tokens, tokens[1:])
    )


@lru_cache(maxsize=512)
def _forbidden_source_name(value: str) -> bool:
    tokens = _boundary_tokens(value)
    return bool(
        _FORBIDDEN_SOURCE_NAME_TOKENS.intersection(tokens)
        or _forbidden_evaluation_name(value)
    )


def _forbidden_evaluation_name(value: str) -> bool:
    tokens = _boundary_tokens(value)
    return bool(
        _FORBIDDEN_EVALUATION_TOKENS.intersection(tokens)
        or _has_token_pair(tokens, "ground", "truth")
        or _has_token_pair(tokens, "root", "cause")
        or _has_token_pair(tokens, "expected", "answer")
        or _has_token_pair(tokens, "scoring", "point")
        or _has_token_pair(tokens, "scoring", "points")
        or _has_token_pair(tokens, "selected", "source")
        or _has_token_pair(tokens, "target", "component")
    )


def _forbidden_csv_header(value: str) -> bool:
    return _forbidden_evaluation_name(value)


def _safe_evaluation_metadata_kind(value: str) -> str | None:
    tokens = _boundary_tokens(value)
    if tokens == ("label", "policy"):
        return "label_policy"
    if tokens == ("labels", "included"):
        return "labels_included"
    return None


def _safe_evaluation_metadata_value(kind: str, value: object) -> bool:
    if value is None or value is False:
        return True
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.strip().casefold().split())
    if kind == "label_policy":
        if normalized in _SAFE_LABEL_POLICY_VALUES:
            return True
        tokens = _boundary_tokens(normalized)
        return bool(
            normalized.endswith("labels excluded")
            and not {"answer", "answers", "embedded", "included", "true"}.intersection(
                tokens
            )
        )
    return normalized in {"excluded", "false", "none"}


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _robust_mad(values: Sequence[float]) -> float | None:
    if not values:
        return None
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _metric_semantic_type(metric_id: str) -> str:
    normalized = metric_id.casefold()
    if any(
        token in normalized
        for token in (
            "error",
            "failure",
            "oom",
            "restart",
            "stopped",
            "start_time",
            "state",
            "status",
            "health",
        )
    ):
        return "event"
    if any(token in normalized for token in ("_total", "counter", "count")):
        return "counter"
    if any(token in normalized for token in ("rate", "_s", "per_second")):
        return "rate"
    return "gauge"


def _series_shape(
    values: Sequence[float], timestamps: Sequence[float]
) -> dict[str, Any]:
    if not values:
        return {
            "sample_count": 0,
            "min": None,
            "mean": None,
            "median": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mad": None,
            "largest_change": None,
        }
    pairs = sorted(zip(timestamps, values), key=lambda item: item[0])
    largest_change: dict[str, float] | None = None
    for (left_time, left), (right_time, right) in zip(pairs, pairs[1:]):
        change = abs(right - left)
        if largest_change is None or change > largest_change["absolute_change"]:
            largest_change = {
                "timestamp": right_time,
                "absolute_change": change,
                "signed_change": right - left,
                "gap_seconds": right_time - left_time,
            }
    return {
        "sample_count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
        "mad": _robust_mad(values),
        "largest_change": largest_change,
    }


def _profile_metric_role(metric_id: str) -> tuple[str | None, int]:
    """Classify generic request-volume/latency names without dataset aliases."""
    separated = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", metric_id)
    tokens = tuple(
        token for token in re.split(r"[^a-z0-9]+", separated.casefold()) if token
    )
    leaf = tokens[-1] if tokens else ""
    compact = "".join(tokens)
    if compact in {"sr", "successrate", "succeerate"} or (
        bool({"success", "succee"} & set(tokens))
        and bool({"rate", "ratio"} & set(tokens))
    ):
        return "success", 0
    if bool({"error", "errors", "failure", "failures", "failed"} & set(tokens)):
        return "error", 0
    if leaf in {"count", "cnt", "num"}:
        return "count", 0
    if leaf in {"mrt", "latency", "duration"}:
        return "latency", 0
    if leaf in {"requests", "calls", "volume", "throughput"}:
        return "count", 1
    if (
        leaf in {"elapsed", "delay"}
        or "time" in tokens
        and bool({"response", "avg", "average", "mean"} & set(tokens))
    ):
        return "latency", 1
    return None, 100


def _profile_effect_size(
    reference_count: float,
    incident_count: float,
    reference_latency: float | None,
    incident_latency: float | None,
) -> tuple[float, dict[str, float | None]]:
    """Return a bounded symmetric shift and its independently auditable parts."""
    count_effect = _symmetric_relative_effect(reference_count, incident_count)
    latency_effect = (
        _symmetric_relative_effect(reference_latency, incident_latency)
        if reference_latency is not None and incident_latency is not None
        else None
    )
    candidates = [count_effect]
    if latency_effect is not None:
        candidates.append(latency_effect)
    effect = max(candidates, key=abs)
    return round(effect, 6), {
        "count": round(count_effect, 6),
        "latency": round(latency_effect, 6) if latency_effect is not None else None,
    }


def _symmetric_relative_effect(reference: float, incident: float) -> float:
    denominator = abs(reference) + abs(incident)
    if denominator == 0:
        return 0.0
    return 2.0 * (incident - reference) / denominator


def _profile_sample_interval(
    *point_groups: Sequence[tuple[float, float, float | None]],
) -> float | None:
    gaps = [
        right[0] - left[0]
        for points in point_groups
        for left, right in zip(
            sorted(points, key=lambda point: point[0]),
            sorted(points, key=lambda point: point[0])[1:],
        )
        if right[0] > left[0]
    ]
    return statistics.median(gaps) if gaps else None


def _profile_latency_stats(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _profile_onset_markers(
    reference_points: Sequence[tuple[float, float, float | None]],
    incident_points: Sequence[tuple[float, float, float | None]],
    *,
    count_is_harmful: bool = True,
    value_direction: str = "higher",
    minimum_consecutive_points: int = 1,
    minimum_relative_change: float = 0.0,
) -> dict[str, Any]:
    """Bracket the first incident sample outside the observed reference envelope."""
    if not reference_points or not incident_points:
        return {
            "last_normal": None,
            "first_abnormal": None,
            "first_anomalous": None,
            "onset_interval": None,
            "segment_start": None,
            "segment_end": None,
            "consecutive_points": 0,
        }
    reference_counts = [point[1] for point in reference_points]
    reference_latencies = [
        point[2] for point in reference_points if point[2] is not None
    ]
    count_min = min(reference_counts)
    count_max = max(reference_counts)
    latency_min = _percentile(reference_latencies, 0.05)
    latency_max = _percentile(reference_latencies, 0.95)

    def abnormal(point: tuple[float, float, float | None]) -> bool:
        _timestamp, count, latency = point
        return bool(
            count_is_harmful
            and (count < count_min or count > count_max)
            or (
                latency is not None
                and (
                    value_direction == "higher"
                    and latency_max is not None
                    and latency > latency_max
                    or value_direction == "lower"
                    and latency_min is not None
                    and latency < latency_min
                )
            )
        )

    def material(point: tuple[float, float, float | None]) -> bool:
        if not abnormal(point):
            return False
        _timestamp, count, latency = point
        changes: list[float] = []
        if count_is_harmful and (count < count_min or count > count_max):
            bound = count_min if count < count_min else count_max
            changes.append(abs(count - bound) / max(abs(bound), 1e-12))
        if latency is not None:
            if value_direction == "higher" and latency_max is not None:
                changes.append((latency - latency_max) / max(abs(latency_max), 1e-12))
            elif value_direction == "lower" and latency_min is not None:
                changes.append((latency_min - latency) / max(abs(latency_min), 1e-12))
        return bool(changes and max(changes) >= minimum_relative_change)

    ordered_incident = sorted(incident_points, key=lambda point: point[0])
    candidate_segments: list[tuple[int, int]] = []
    segment_start: int | None = None
    for index, point in enumerate(ordered_incident):
        if abnormal(point):
            if segment_start is None:
                segment_start = index
            continue
        if segment_start is not None:
            candidate_segments.append((segment_start, index))
            segment_start = None
    if segment_start is not None:
        candidate_segments.append((segment_start, len(ordered_incident)))
    selected_segment = None
    for start, end in candidate_segments:
        material_index = next(
            (index for index in range(start, end) if material(ordered_incident[index])),
            None,
        )
        if (
            material_index is not None
            and end - material_index >= minimum_consecutive_points
        ):
            selected_segment = (material_index, end)
            break
    abnormal_index = selected_segment[0] if selected_segment is not None else None
    if abnormal_index is None:
        return {
            "last_normal": ordered_incident[-1][0],
            "first_abnormal": None,
            "first_anomalous": None,
            "onset_interval": None,
            "segment_start": None,
            "segment_end": None,
            "consecutive_points": 0,
        }
    first_abnormal = ordered_incident[abnormal_index][0]
    if abnormal_index > 0:
        last_normal = ordered_incident[abnormal_index - 1][0]
    else:
        reference_normal = [
            point[0] for point in reference_points if point[0] < first_abnormal
        ]
        last_normal = max(reference_normal) if reference_normal else None
    onset_interval = (
        {"start": last_normal, "end": first_abnormal}
        if last_normal is not None and last_normal < first_abnormal
        else None
    )
    segment_end_index = cast(tuple[int, int], selected_segment)[1]
    segment = ordered_incident[abnormal_index:segment_end_index]
    return {
        "last_normal": last_normal,
        "first_abnormal": first_abnormal,
        "first_anomalous": first_abnormal,
        "onset_interval": onset_interval,
        "segment_start": segment[0][0],
        "segment_end": segment[-1][0],
        "consecutive_points": len(segment),
    }


def _burst_profile(
    incident_values: Sequence[float],
    incident_timestamps: Sequence[float],
    *,
    reference_min: float | None,
    reference_max: float | None,
    direction: str,
) -> dict[str, Any]:
    def crosses(value: float) -> bool:
        if direction == "lower_is_more_pressure":
            return reference_min is not None and value < reference_min
        return reference_max is not None and value > reference_max

    pairs = sorted(zip(incident_timestamps, incident_values), key=lambda item: item[0])
    gaps = [
        right[0] - left[0]
        for left, right in zip(pairs, pairs[1:])
        if right[0] > left[0]
    ]
    interval = statistics.median(gaps) if gaps else 0.0
    segments: list[dict[str, Any]] = []
    current: list[tuple[float, float]] = []
    for pair in pairs:
        if crosses(pair[1]):
            current.append(pair)
            continue
        if current:
            segments.append(
                {
                    "start": current[0][0],
                    "end": current[-1][0] + interval,
                    "sample_count": len(current),
                    "duration_seconds": max(
                        interval, current[-1][0] - current[0][0] + interval
                    ),
                }
            )
            current = []
    if current:
        segments.append(
            {
                "start": current[0][0],
                "end": current[-1][0] + interval,
                "sample_count": len(current),
                "duration_seconds": max(
                    interval, current[-1][0] - current[0][0] + interval
                ),
            }
        )
    longest = max(segments, key=lambda item: item["duration_seconds"], default=None)
    onset = cast(float, segments[0]["start"]) if segments else None
    onset_lower_bound = (
        max(timestamp for timestamp, _value in pairs if timestamp < onset)
        if onset is not None and any(timestamp < onset for timestamp, _value in pairs)
        else onset - interval
        if onset is not None and interval > 0
        else None
    )
    return {
        "crossing_count": sum(item["sample_count"] for item in segments),
        "crossing_fraction": (
            sum(item["sample_count"] for item in segments) / len(pairs)
            if pairs
            else 0.0
        ),
        "segment_count": len(segments),
        "longest_segment": longest,
        "onset_timestamp": onset,
        "onset_lower_bound_timestamp": onset_lower_bound,
        "segments": segments[:20],
    }


def _pressure_direction_aligned(profile: Mapping[str, Any]) -> bool:
    direction = profile.get("pressure_direction")
    reference = profile.get("reference")
    incident = profile.get("incident")
    if (
        direction not in {"higher_is_more_pressure", "lower_is_more_pressure"}
        or not isinstance(reference, Mapping)
        or not isinstance(incident, Mapping)
    ):
        return False
    reference_values = cast(Mapping[str, object], reference)
    incident_values = cast(Mapping[str, object], incident)
    fields = ("median", "mean", "p95", "max", "min")
    comparisons = [
        (
            float(cast(float, incident_values[field])),
            float(cast(float, reference_values[field])),
        )
        for field in fields
        if isinstance(incident_values.get(field), (int, float))
        and not isinstance(incident_values.get(field), bool)
        and isinstance(reference_values.get(field), (int, float))
        and not isinstance(reference_values.get(field), bool)
    ]
    if direction == "higher_is_more_pressure":
        return any(current > baseline for current, baseline in comparisons)
    return any(current < baseline for current, baseline in comparisons)


def _incident_resource_burst(profile: Mapping[str, Any]) -> bool:
    if (
        profile.get("strong_incident_change") is not True
        or profile.get("scale_hint") != "0..100"
        or profile.get("pressure_direction") != "higher_is_more_pressure"
    ):
        return False
    reference = profile.get("reference")
    incident = profile.get("incident")
    burst = profile.get("incident_burst")
    if not all(isinstance(item, Mapping) for item in (reference, incident, burst)):
        return False
    reference_values = cast(Mapping[str, object], reference)
    incident_values = cast(Mapping[str, object], incident)
    burst_values = cast(Mapping[str, object], burst)
    reference_p95 = reference_values.get("p95")
    incident_p95 = incident_values.get("p95")
    longest = burst_values.get("longest_segment")
    return bool(
        isinstance(reference_p95, (int, float))
        and not isinstance(reference_p95, bool)
        and isinstance(incident_p95, (int, float))
        and not isinstance(incident_p95, bool)
        and float(incident_p95) >= 40.0
        and float(incident_p95) - float(reference_p95) >= 30.0
        and isinstance(longest, Mapping)
        and isinstance(cast(Mapping[str, object], longest).get("sample_count"), int)
        and not isinstance(
            cast(Mapping[str, object], longest).get("sample_count"), bool
        )
        and cast(int, cast(Mapping[str, object], longest)["sample_count"]) >= 3
    )


_LOG_DYNAMIC_PATTERNS = (
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", re.IGNORECASE), "<uuid>"),
    (re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b"), "<ip>"),
    (re.compile(r"\b0x[0-9a-f]+\b", re.IGNORECASE), "<hex>"),
    (re.compile(r"\b[0-9a-f]{16,}\b", re.IGNORECASE), "<id>"),
    (re.compile(r"\b\d+(?:\.\d+)?\b"), "<num>"),
)


def _normalize_log_template(text: str) -> str:
    normalized = " ".join(text.strip().split())[:2_000]
    for pattern, replacement in _LOG_DYNAMIC_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    return normalized


def _log_severity(text: str) -> str:
    normalized = text.casefold()
    if any(token in normalized for token in ("fatal", "panic", "critical")):
        return "critical"
    if any(
        token in normalized
        for token in ("error", "exception", "failed", "failure", "oom")
    ):
        return "error"
    if any(token in normalized for token in ("warn", "timeout", "retry")):
        return "warning"
    return "info"


def _trace_error_indicator(raw: object) -> float:
    normalized = str(raw).strip().casefold()
    if normalized in {"", "0", "1", "ok", "unset", "success", "false"}:
        return 0.0
    try:
        return 1.0 if float(normalized) >= 2 else 0.0
    except ValueError:
        return 1.0


def market_output_component(original_id: str) -> str:
    """Map a Market telemetry identity to the public evaluator component name."""
    node_pod = re.fullmatch(r"node-\d+\.(.+-\d+)", original_id)
    if node_pod is not None:
        return node_pod.group(1)
    if ".source." in original_id or ".destination." in original_id:
        return original_id.split(".", 1)[0]
    runtime = re.fullmatch(r"([^.]+)\.ts:\d+", original_id)
    if runtime is not None:
        return runtime.group(1)
    service = re.fullmatch(r"(.+)-(?:grpc|http)", original_id)
    if service is not None:
        return service.group(1)
    return original_id


def market_service_component(original_id: str) -> str:
    """Map a Market pod or endpoint identity to its evaluator service."""
    component = market_output_component(original_id)
    if re.fullmatch(r"node-\d+", component):
        return component
    service = re.sub(r"-\d+$", "", component)
    return service[:-1] if service.endswith("2") else service


def trace_status_error_indicator(value: str | int | float) -> float:
    """Normalize mixed OTel, gRPC, and HTTP success encodings to a 0/1 error."""
    text = str(value).strip().casefold()
    if text in {"ok", "unset"}:
        return 0.0
    try:
        numeric = float(text)
    except ValueError as exc:
        raise ValueError(f"unsupported trace status_code: {value!r}") from exc
    if numeric in {0.0, 1.0} or 200.0 <= numeric < 400.0:
        return 0.0
    return 1.0


def network_transport_error_indicator(value: str) -> float:
    """Return 1 for source-backed proxy transport failures, otherwise 0."""
    text = value.casefold()
    markers = (
        "http2.remote_reset",
        "connection error",
        "i/o timeout",
        "transport is closing",
        "connection reset",
        "upstream connect error",
    )
    return float(any(marker in text for marker in markers))


def peer_isolated_trace_member(
    member_scores: Mapping[str, float], *, latency: bool
) -> str | None:
    """Return one trace member only when it is clearly isolated from three peers."""
    if len(member_scores) < 3:
        return None
    ranked = sorted(member_scores.items(), key=lambda item: (-item[1], item[0]))
    top_member, top_score = ranked[0]
    second_score = ranked[1][1]
    peer_median = statistics.median(score for _member, score in ranked[1:])
    if latency:
        isolated = bool(
            top_score > 0
            and top_score >= 1.5 * max(peer_median, 1e-12)
            and top_score >= 1.25 * max(second_score, 1e-12)
        )
    else:
        isolated = bool(
            top_score >= 0.01
            and top_score >= second_score + 0.01
            and top_score >= 1.5 * max(peer_median, 1e-12)
        )
    return top_member if isolated else None


def dominant_market_fault_type(
    summaries: Sequence[Mapping[str, object]],
) -> str | None:
    """Choose a public mechanism while suppressing same-node iowait symptoms."""
    candidates: list[tuple[int, float, str, set[str]]] = []
    for summary in summaries:
        requested = summary.get("requested_fault_type")
        score = summary.get("root_cause_score")
        raw_qualifying = summary.get("qualifying_metric_ids")
        if (
            summary.get("fault_type_ready") is not True
            or not isinstance(requested, str)
            or not isinstance(score, (int, float))
            or isinstance(score, bool)
            or not isinstance(raw_qualifying, list)
        ):
            continue
        qualifying = {
            metric_id
            for metric_id in cast(list[object], raw_qualifying)
            if isinstance(metric_id, str)
        }
        mechanism_rank = mechanism_signature_rank(summary)
        candidates.append((mechanism_rank, float(score), requested, qualifying))
    disk_ready = any(
        " disk " in f" {fault_type.casefold()} " for _, _, fault_type, _ in candidates
    )
    if disk_ready:
        candidates = [
            candidate
            for candidate in candidates
            if not (
                "cpu" in candidate[2].casefold()
                and any("iowait" in metric.casefold() for metric in candidate[3])
            )
        ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda item: (-item[0], -item[1], item[2]))[0][2]


def mechanism_signature_rank(summary: Mapping[str, object]) -> int:
    """Rank typed mechanisms by causal specificity, independent of score scale."""
    if summary.get("boundary_aligned") is True:
        return 0
    signature = summary.get("mechanism_signature")
    if not isinstance(signature, str):
        return 0
    if signature == "availability_state_transition":
        return 5
    if signature in {"connection_state_transition", "state_transition"}:
        return 4
    if "trace" in signature or signature == "incident_event_burst":
        return 3
    if signature in {
        "sustained_high_saturation",
        "incident_resource_burst",
        "capacity_change_rate",
        "packet_count_without_payload_volume",
        "service_packet_count_without_payload_volume",
    }:
        return 2
    return 1


AUDITED_SIGNATURE_FREE_MATERIAL_EFFECT = "audited_signature_free_material_effect"


@dataclass(frozen=True, slots=True)
class _OpenRCAProfile:
    environments: frozenset[str]
    log_sources: tuple[str, ...]
    source_schema_required: bool = False


_BANK_PROFILE = _OpenRCAProfile(
    frozenset({"openrca_bank", "synthetic_bank"}),
    ("log/log_service.csv",),
)
_MARKET_PROFILE = _OpenRCAProfile(
    frozenset({"openrca_market"}),
    ("log/log_proxy.csv", "log/log_service.csv"),
    source_schema_required=True,
)
_TELECOM_PROFILE = _OpenRCAProfile(
    frozenset({"openrca_telecom"}),
    (),
)
_MARKET_SOURCE_FILES: dict[str, dict[str, object]] = {
    "log/log_proxy.csv": {
        "columns": ("log_id", "timestamp", "cmdb_id", "log_name", "value"),
        "timestamp_unit": "seconds",
        "entity_key": "cmdb_id",
        "value_kind": "free-form log text",
        "nullable_columns": (),
    },
    "log/log_service.csv": {
        "columns": ("log_id", "timestamp", "cmdb_id", "log_name", "value"),
        "timestamp_unit": "seconds",
        "entity_key": "cmdb_id",
        "value_kind": "free-form log text",
        "nullable_columns": ("value",),
    },
    **{
        f"metric/metric_{name}.csv": {
            "columns": ("timestamp", "cmdb_id", "kpi_name", "value"),
            "timestamp_unit": "seconds",
            "entity_key": "cmdb_id",
            "metric_key": "kpi_name",
            "value_kind": "numeric",
        }
        for name in ("container", "mesh", "node", "runtime")
    },
    "metric/metric_service.csv": {
        "columns": ("service", "timestamp", "rr", "sr", "mrt", "count"),
        "timestamp_unit": "seconds",
        "entity_key": "service",
        "metric_layout": "wide",
        "numeric_metric_columns": ("rr", "sr", "mrt", "count"),
    },
    "trace/trace_span.csv": {
        "columns": (
            "timestamp",
            "cmdb_id",
            "span_id",
            "trace_id",
            "duration",
            "type",
            "status_code",
            "operation_name",
            "parent_span",
        ),
        "timestamp_unit": "milliseconds",
        "entity_key": "cmdb_id",
        "trace_key": "trace_id",
        "span_key": "span_id",
        "parent_key": "parent_span",
        "nullable_columns": ("type", "parent_span"),
    },
}
_COMMON_ARTIFACT_PATHS = frozenset(
    {
        "dataset_profile.json",
        "entity_mapping.json",
        "metric_catalog.json",
        "metric_window_stats.jsonl",
        "task.json",
        "topology.json",
        "vocabulary.json",
    }
)
_BANK_TELEMETRY_HEADERS: dict[str, tuple[str, ...]] = {
    "telemetry/log/log_service.csv": (
        "log_id",
        "timestamp",
        "cmdb_id",
        "log_name",
        "value",
        "_source_record",
        "_source_member",
    ),
    "telemetry/metric/metric_app.csv": (
        "timestamp",
        "rr",
        "sr",
        "cnt",
        "mrt",
        "tc",
        "_source_record",
        "_source_member",
    ),
    "telemetry/metric/metric_container.csv": (
        "timestamp",
        "cmdb_id",
        "kpi_name",
        "value",
        "_source_record",
        "_source_member",
    ),
    "telemetry/trace/trace_span.csv": (
        "timestamp",
        "cmdb_id",
        "parent_id",
        "span_id",
        "trace_id",
        "duration",
        "_source_record",
        "_source_member",
    ),
}
_SYNTHETIC_BANK_TELEMETRY_HEADERS: dict[str, tuple[str, ...]] = {
    path: tuple(column for column in headers if column != "_source_member")
    for path, headers in _BANK_TELEMETRY_HEADERS.items()
}
_SYNTHETIC_BANK_TELEMETRY_HEADERS["telemetry/trace/trace_span.csv"] = (
    "trace_id",
    "span_id",
    "parent_id",
    "cmdb_id",
    "timestamp",
    "_source_record",
)
_TELECOM_TELEMETRY_HEADERS: dict[str, tuple[str, ...]] = {
    "telemetry/metric/metric_container.csv": (
        "timestamp",
        "cmdb_id",
        "kpi_name",
        "value",
        "_source_record",
        "_source_member",
    ),
    "telemetry/trace/trace_span.csv": (
        "timestamp",
        "cmdb_id",
        "span_id",
        "trace_id",
        "duration",
        "type",
        "status_code",
        "operation_name",
        "parent_span",
        "_source_record",
        "_source_member",
    ),
}


class OpenRCAEnvironment:
    """A deterministic, read-only view of one prepared OpenRCA ``public_input``."""

    profile = _BANK_PROFILE

    capabilities = CapabilityManifest(
        metrics=True,
        logs=True,
        trace=True,
        topology=True,
        coverage_limits=(
            "frozen public_input only",
            "upstream collection completeness is generally unknown",
            "no changes or source-code capability",
        ),
    )

    def __init__(self, public_input: str | Path) -> None:
        root = Path(public_input).resolve(strict=True)
        if not root.is_dir() or root.name != "public_input":
            raise ValueError("OpenRCA environment requires a public_input directory")
        self.root = root

        manifest_path = root / "manifest.json"
        if (
            manifest_path.is_symlink()
            or manifest_path.resolve(strict=True).parent != root
        ):
            raise ValueError("manifest.json must be a regular file inside public_input")
        manifest = json.loads(manifest_path.read_text())
        self.dataset_id = str(manifest["dataset_id"])
        self._artifacts_by_path: dict[str, dict[str, Any]] = {}
        self._artifacts_by_id: dict[str, dict[str, Any]] = {}

        raw_artifacts = manifest.get("artifacts")
        if not isinstance(raw_artifacts, list) or not all(
            isinstance(item, dict)
            and set(cast(dict[str, object], item)) == {"path", "bytes", "sha256"}
            and isinstance(cast(dict[str, object], item).get("path"), str)
            and isinstance(cast(dict[str, object], item).get("bytes"), int)
            and not isinstance(cast(dict[str, object], item).get("bytes"), bool)
            and isinstance(cast(dict[str, object], item).get("sha256"), str)
            for item in cast(list[object], raw_artifacts)
        ):
            raise ValueError("manifest artifacts require only path, bytes, and sha256")
        artifacts = cast(list[dict[str, Any]], raw_artifacts)
        artifact_paths = [str(item["path"]) for item in artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("duplicate artifact path in manifest")
        forbidden = sorted(
            path
            for path in artifact_paths
            if Path(path).name.lower() in _FORBIDDEN_SOURCE_FILES
            or _forbidden_source_name(Path(path).name)
        )
        actual_files = {
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() or path.is_symlink()
        }
        forbidden.extend(
            path
            for path in actual_files
            if (
                Path(path).name.lower() in _FORBIDDEN_SOURCE_FILES
                or _forbidden_source_name(Path(path).name)
            )
            and path not in forbidden
        )
        if forbidden:
            raise ValueError(f"forbidden source file in public_input: {forbidden[0]}")
        expected_files = set(artifact_paths) | {"manifest.json"}
        if actual_files != expected_files:
            missing = sorted(expected_files - actual_files)
            extra = sorted(actual_files - expected_files)
            raise ValueError(
                f"public_input file set mismatch: missing={missing}, extra={extra}"
            )

        for item in artifacts:
            relative = str(item["path"])
            candidate = Path(relative)
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError(f"unsafe manifest artifact path: {relative}")
            original = root / candidate
            if original.is_symlink():
                raise ValueError(f"manifest artifact must not be a symlink: {relative}")
            resolved = original.resolve(strict=True)
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    f"manifest artifact escapes public_input: {relative}"
                ) from exc
            if not resolved.is_file() or resolved.stat().st_size != int(item["bytes"]):
                raise ValueError(f"manifest artifact size mismatch: {relative}")
            artifact_id = str(item["sha256"])
            if self._file_hash(resolved) != artifact_id:
                raise ValueError(f"manifest artifact hash mismatch: {relative}")
            artifact: dict[str, Any] = {
                "artifact_id": artifact_id,
                "content_hash": artifact_id,
                "path": relative,
                "bytes": int(item["bytes"]),
                "resolved": resolved,
            }
            if artifact_id in self._artifacts_by_id:
                raise ValueError("duplicate artifact content hash in manifest")
            self._artifacts_by_path[relative] = artifact
            self._artifacts_by_id[artifact_id] = artifact

        self.task = Task.model_validate_json(
            self._artifact_path("task.json").read_text()
        )
        if self.task.environment not in self.profile.environments:
            raise ValueError(
                f"unsupported task environment for adapter: {self.task.environment}"
            )
        self._validate_artifact_allowlist(set(artifact_paths))
        self._reject_evaluation_labels(manifest, "manifest.json")
        for relative in sorted(self._artifacts_by_path):
            suffix = Path(relative).suffix.lower()
            if suffix == ".json":
                payload = json.loads(self._artifact_path(relative).read_text())
                self._reject_evaluation_labels(payload, relative)
            elif suffix == ".jsonl":
                with self._artifact_path(relative).open(encoding="utf-8") as stream:
                    for line_number, line in enumerate(stream, 1):
                        payload = json.loads(line)
                        self._reject_evaluation_labels(
                            payload, f"{relative}:{line_number}"
                        )
            elif suffix == ".csv":
                self._validate_csv_input_boundary(relative)
        self._validate_control_json_shapes()

        # These two files are the externally visible domain contracts.  Parsing
        # them through the strict models rejects unknown or malformed fields.
        topology_text = self._artifact_path("topology.json").read_text()
        raw_topology: object = json.loads(topology_text)
        raw_nodes = (
            cast(dict[str, object], raw_topology).get("nodes")
            if isinstance(raw_topology, dict)
            else None
        )
        self._implicit_impact_role_entity_ids = frozenset(
            str(cast(dict[str, object], node).get("canonical_id"))
            for node in cast(list[object], raw_nodes or [])
            if isinstance(node, dict)
            and cast(dict[str, object], node).get("entity_type")
            in {"business_entry", "business_test", "impact"}
            and "evidence_role" not in cast(dict[str, object], node)
        )
        self.topology = TopologySnapshot.model_validate_json(topology_text)
        if self._implicit_impact_role_entity_ids:
            self.topology = self.topology.model_copy(
                update={
                    "nodes": [
                        node.model_copy(update={"evidence_role": "impact_only"})
                        if node.canonical_id in self._implicit_impact_role_entity_ids
                        else node
                        for node in self.topology.nodes
                    ]
                }
            )
        if self.task.reference_range is None:
            raise ValueError("OpenRCA task requires a reference range")
        self._reference_health = self.task.reference_range.health_status
        self._entities = {node.canonical_id: node for node in self.topology.nodes}
        self._validate_topology_sources()
        self._add_deployment_edges()

        self._source_schema_files: dict[str, dict[str, object]] = {}
        self._cloudbed: str | None = None
        if self.profile.source_schema_required:
            self._load_market_source_schema(manifest)

        catalog = cast(
            dict[str, object],
            json.loads(self._artifact_path("metric_catalog.json").read_text()),
        )
        raw_metrics = catalog.get("metrics")
        if not isinstance(raw_metrics, list):
            raise ValueError("metric_catalog metrics must be an array")
        self._metric_ids = frozenset(
            str(metric) for metric in cast(list[object], raw_metrics)
        )
        raw_log_names = catalog["log_names"]
        if isinstance(raw_log_names, dict):
            log_names = [str(name) for name in cast(dict[str, object], raw_log_names)]
        elif isinstance(raw_log_names, list):
            log_names = [str(name) for name in cast(list[object], raw_log_names)]
        else:
            raise ValueError("metric_catalog log_names must be an object or array")
        self._log_names = frozenset(log_names)
        raw_schemas = catalog.get("schemas")
        self._catalog_schemas = (
            cast(dict[str, object], raw_schemas)
            if isinstance(raw_schemas, dict)
            else {}
        )
        self._trace_duration_unit = "source-native; " + str(catalog["duration_unit"])
        self._validate_catalog_sources(catalog)
        self._metric_bindings = self._load_metric_bindings(catalog)

        vocabulary = json.loads(self._artifact_path("vocabulary.json").read_text())
        raw_entities: object = vocabulary.get("entities")
        if not isinstance(raw_entities, list):
            raise ValueError("vocabulary.json requires non-empty string entities")
        entities = cast(list[object], raw_entities)
        if not entities or not all(
            isinstance(item, str) and item.strip() for item in entities
        ):
            raise ValueError("vocabulary.json requires non-empty string entities")
        self._public_entities = frozenset(cast(list[str], entities))
        self._has_public_candidate_mapping = any(
            entity.public_candidate_match is not None
            for entity in self._entities.values()
        )
        raw_fault_types: object = vocabulary.get("fault_types")
        if not isinstance(raw_fault_types, list):
            raise ValueError("vocabulary.json requires non-empty string fault_types")
        fault_types = cast(list[object], raw_fault_types)
        if not fault_types or not all(
            isinstance(item, str) and item.strip() for item in fault_types
        ):
            raise ValueError("vocabulary.json requires non-empty string fault_types")
        self._public_fault_types = tuple(cast(list[str], fault_types))

        self._metric_stats: list[dict[str, Any]] = []
        self._stats_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        self._service_profile_seed_index: dict[tuple[str, str], Result] = {}
        catalog_metrics_by_entity: dict[str, set[str]] = {}
        with self._artifact_path("metric_window_stats.jsonl").open(
            encoding="utf-8"
        ) as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                row["_line_number"] = line_number
                canonical_id = str(row["canonical_id"])
                metric_id = str(row["metric_id"])
                key = (canonical_id, metric_id)
                self._metric_stats.append(row)
                self._stats_by_pair[key] = row
                if metric_id in self._metric_ids:
                    catalog_metrics_by_entity.setdefault(canonical_id, set()).add(
                        metric_id
                    )
        self._catalog_metrics_by_entity = {
            entity_id: sorted(metric_ids)
            for entity_id, metric_ids in catalog_metrics_by_entity.items()
        }
        if self.profile.source_schema_required and set(self._metric_bindings) != set(
            self._stats_by_pair
        ):
            raise ValueError(
                "metric catalog bindings must exactly cover metric_window_stats"
            )

    def _allowed_artifact_paths(self) -> frozenset[str]:
        environment = self.task.environment
        if environment in {"openrca_bank", "synthetic_bank"}:
            return _COMMON_ARTIFACT_PATHS | frozenset(_BANK_TELEMETRY_HEADERS)
        if environment == "openrca_telecom":
            return (
                _COMMON_ARTIFACT_PATHS
                | frozenset(_TELECOM_TELEMETRY_HEADERS)
                | {"trace_edge_window_stats.jsonl"}
            )
        if environment == "openrca_market":
            return (
                _COMMON_ARTIFACT_PATHS
                | {"source_schema.json"}
                | {f"telemetry/{source_path}" for source_path in _MARKET_SOURCE_FILES}
            )
        raise ValueError(f"unsupported OpenRCA environment: {environment}")

    def _validate_control_json_shapes(self) -> None:
        vocabulary = json.loads(self._artifact_path("vocabulary.json").read_text())
        self._require_object_keys(
            vocabulary,
            required={"entities", "fault_types"},
            allowed={"entities", "fault_types", "source"},
            location="vocabulary.json",
        )
        catalog = json.loads(self._artifact_path("metric_catalog.json").read_text())
        catalog_values = self._require_object_keys(
            catalog,
            required={"duration_unit", "log_names", "metrics"},
            allowed={
                "duration_unit",
                "log_names",
                "metric_bindings",
                "metrics",
                "schemas",
            },
            location="metric_catalog.json",
        )
        raw_bindings = catalog_values.get("metric_bindings", [])
        if not isinstance(raw_bindings, list):
            raise ValueError("metric_catalog.json.metric_bindings must be an array")
        binding_keys = {
            "canonical_id",
            "entity_field",
            "evidence_role",
            "layout",
            "metric_field",
            "metric_id",
            "original_entity_key",
            "original_entity_keys",
            "source_member_filter",
            "source_metric",
            "source_path",
            "timestamp_unit",
            "value_field",
            "value_transform",
        }
        for index, binding in enumerate(cast(list[object], raw_bindings)):
            self._require_object_keys(
                binding,
                required={
                    "canonical_id",
                    "entity_field",
                    "layout",
                    "metric_id",
                    "original_entity_key",
                    "source_path",
                    "timestamp_unit",
                    "value_field",
                },
                allowed=binding_keys,
                location=f"metric_catalog.json.metric_bindings[{index}]",
            )
        entity_mapping: object = json.loads(
            self._artifact_path("entity_mapping.json").read_text()
        )
        if not isinstance(entity_mapping, list):
            raise ValueError("entity_mapping.json must be an array")
        entity_keys = {
            "canonical_id",
            "entity_type",
            "evidence_role",
            "mapping_evidence",
            "mapping_status",
            "normalization",
            "original_id",
            "public_candidate_match",
            "source_aliases",
            "sources",
        }
        for index, item in enumerate(cast(list[object], entity_mapping)):
            self._require_object_keys(
                item,
                required={"canonical_id", "entity_type", "original_id", "sources"},
                allowed=entity_keys,
                location=f"entity_mapping.json[{index}]",
            )
        dataset_profile = json.loads(
            self._artifact_path("dataset_profile.json").read_text()
        )
        dataset_keys = {
            "candidate_component_count",
            "capabilities",
            "cloudbed",
            "dataset",
            "dataset_id",
            "description",
            "entities",
            "files",
            "impact_only_entity_count",
            "impact_only_metric_series_count",
            "limitations",
            "metric_series_count",
            "prepared_trace_contract",
            "selected_rows",
            "selection_policy",
            "source_commit",
            "source_date",
            "source_rows_selected",
            "topology",
            "topology_edge_count",
            "trace_edge_profile_count",
        }
        self._require_object_keys(
            dataset_profile,
            required=set(),
            allowed=dataset_keys,
            location="dataset_profile.json",
        )
        stat_keys = {
            "canonical_id",
            "entity_id",
            "evidence_role",
            "incident",
            "max_observed_gap_seconds",
            "median_observed_interval_seconds",
            "metric_id",
            "reference",
            "source_entity_keys",
            "source_members",
            "value_unit",
        }
        summary_keys = {
            "count",
            "example_source_records",
            "max",
            "mean",
            "median",
            "min",
            "p95",
            "p99",
        }
        with self._artifact_path("metric_window_stats.jsonl").open(
            encoding="utf-8"
        ) as stream:
            for line_number, line in enumerate(stream, 1):
                row = self._require_object_keys(
                    json.loads(line),
                    required={
                        "canonical_id",
                        "entity_id",
                        "metric_id",
                        "value_unit",
                    },
                    allowed=stat_keys,
                    location=f"metric_window_stats.jsonl:{line_number}",
                )
                if not {"reference", "incident"}.intersection(row):
                    raise ValueError(
                        f"metric_window_stats.jsonl:{line_number} has no task window"
                    )
                for period in ("reference", "incident"):
                    if period not in row:
                        continue
                    self._require_object_keys(
                        row[period],
                        required={"count", "max", "mean", "median", "min"},
                        allowed=summary_keys,
                        location=(f"metric_window_stats.jsonl:{line_number}.{period}"),
                    )

    @staticmethod
    def _require_object_keys(
        payload: object,
        *,
        required: set[str],
        allowed: set[str],
        location: str,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError(f"{location} must be an object")
        values = cast(dict[str, Any], payload)
        missing = sorted(required - set(values))
        extra = sorted(set(values) - allowed)
        if missing or extra:
            raise ValueError(
                f"{location} schema mismatch: missing={missing}, extra={extra}"
            )
        return values

    def _validate_artifact_allowlist(self, artifact_paths: set[str]) -> None:
        missing = sorted(_COMMON_ARTIFACT_PATHS - artifact_paths)
        if missing:
            raise ValueError(f"required public artifact is missing: {missing[0]}")
        unauthorized = sorted(artifact_paths - self._allowed_artifact_paths())
        if unauthorized:
            raise ValueError(
                f"artifact path is not authorized for {self.task.environment}: "
                f"{unauthorized[0]}"
            )

    def _authorized_source_paths(self) -> frozenset[str]:
        return frozenset(
            path.removeprefix("telemetry/")
            for path in self._allowed_artifact_paths()
            if path.startswith("telemetry/")
        )

    def _validate_topology_sources(self) -> None:
        allowed = self._authorized_source_paths()
        for node in self.topology.nodes:
            unauthorized = sorted(
                (set(node.sources) | set(node.source_aliases)) - allowed
            )
            if unauthorized:
                raise ValueError(
                    f"topology node source is not authorized: {unauthorized[0]}"
                )

    def _validate_catalog_sources(self, catalog: Mapping[str, object]) -> None:
        raw_bindings = catalog.get("metric_bindings")
        if raw_bindings is None:
            return
        if not isinstance(raw_bindings, list):
            raise ValueError("metric_catalog metric_bindings must be an array")
        allowed = self._authorized_source_paths()
        for raw_binding in cast(list[object], raw_bindings):
            if not isinstance(raw_binding, dict):
                raise ValueError("metric binding must be an object")
            source_path = cast(dict[str, object], raw_binding).get("source_path")
            if not isinstance(source_path, str) or source_path not in allowed:
                raise ValueError(
                    f"metric binding source is not authorized: {source_path!r}"
                )

    def _add_deployment_edges(self) -> None:
        """Derive pod-to-node placement from source-backed container aliases."""
        node_by_original = {
            node.original_id: node.canonical_id
            for node in self.topology.nodes
            if node.entity_type == "node"
        }
        existing = {
            (edge.source_id, edge.target_id, edge.relation_type)
            for edge in self.topology.edges
        }
        derived: list[TopologyEdge] = []
        for node in self.topology.nodes:
            if node.entity_type != "pod":
                continue
            aliases = node.source_aliases.get("metric/metric_container.csv", [])
            hosts = {
                alias.split(".", 1)[0]
                for alias in aliases
                if "." in alias and alias.split(".", 1)[0] in node_by_original
            }
            for host in sorted(hosts):
                target_id = node_by_original[host]
                identity = (node.canonical_id, target_id, "runs_on")
                if identity in existing:
                    continue
                digest = hashlib.sha256("\0".join(identity).encode()).hexdigest()[:16]
                derived.append(
                    TopologyEdge(
                        edge_id=f"deployment-{digest}",
                        source_id=node.canonical_id,
                        target_id=target_id,
                        relation_type="runs_on",
                        fact_kind="derived",
                        valid_range=self.task.allowed_query_range,
                    )
                )
                existing.add(identity)
        if derived:
            self.topology = self.topology.model_copy(
                update={"edges": [*self.topology.edges, *derived]}
            )

    def _load_market_source_schema(self, manifest: Mapping[str, object]) -> None:
        schema = cast(
            dict[str, object],
            json.loads(self._artifact_path("source_schema.json").read_text()),
        )
        if (
            schema.get("schema_version") != 1
            or schema.get("dataset") != "OpenRCA Market"
            or schema.get("holdout_content_included") is not False
            or schema.get("labels_included") is not False
        ):
            raise ValueError("unsupported OpenRCA Market source schema")
        raw_source = schema.get("source")
        if not isinstance(raw_source, dict) or cast(dict[str, object], raw_source) != {
            "repository": "cdreetz/OpenRCA",
            "revision": "7a29a7c3600cc07b7760a6c5097d36ac8573f039",
        }:
            raise ValueError("OpenRCA Market source identity mismatch")
        revision = cast(dict[str, object], raw_source)["revision"]
        if (
            manifest.get("source_commit") != revision
            or self.task.source_commit != revision
        ):
            raise ValueError("OpenRCA Market source commit mismatch")
        raw_scope = schema.get("scope_rules")
        if not isinstance(raw_scope, dict):
            raise ValueError("OpenRCA Market scope rules mismatch")
        scope = cast(dict[str, object], raw_scope)
        if any(
            scope.get(name) != expected
            for name, expected in {
                "cloudbeds_are_independent": True,
                "partitioning": "cloudbed/date",
                "files_per_daily_partition": 8,
                "timezone": "Asia/Shanghai",
                "do_not_merge_cloudbeds": True,
            }.items()
        ):
            raise ValueError("OpenRCA Market scope rules mismatch")
        raw_files = schema.get("files")
        if not isinstance(raw_files, dict):
            raise ValueError("OpenRCA Market source file set mismatch")
        files = cast(dict[str, object], raw_files)
        if set(files) != set(_MARKET_SOURCE_FILES):
            raise ValueError("OpenRCA Market source file set mismatch")
        telemetry_files = {
            path.removeprefix("telemetry/")
            for path in self._artifacts_by_path
            if path.startswith("telemetry/")
        }
        if telemetry_files != set(_MARKET_SOURCE_FILES):
            raise ValueError("OpenRCA Market telemetry artifact set mismatch")
        for path, expected in _MARKET_SOURCE_FILES.items():
            raw = files[path]
            if not isinstance(raw, dict):
                raise ValueError(f"invalid OpenRCA Market source schema: {path}")
            item = cast(dict[str, object], raw)
            for name, value in expected.items():
                actual = (
                    item.get(name, []) if name == "nullable_columns" else item.get(name)
                )
                if isinstance(value, tuple):
                    if (
                        not isinstance(actual, list)
                        or tuple(cast(list[object], actual)) != value
                    ):
                        raise ValueError(
                            f"OpenRCA Market source schema mismatch: {path}.{name}"
                        )
                elif actual != value:
                    raise ValueError(
                        f"OpenRCA Market source schema mismatch: {path}.{name}"
                    )
            relative = f"telemetry/{path}"
            if relative not in self._artifacts_by_path:
                raise ValueError(
                    f"OpenRCA Market telemetry artifact is missing: {path}"
                )
            with self._artifact_path(relative).open(
                newline="", encoding="utf-8"
            ) as stream:
                columns = next(csv.reader(stream), None)
            expected_columns = [
                *cast(tuple[str, ...], expected["columns"]),
                "_source_record",
                "_source_member",
            ]
            if columns is None or columns != expected_columns:
                raise ValueError(f"OpenRCA Market telemetry columns mismatch: {path}")
        cloudbed = manifest.get("cloudbed")
        if not isinstance(cloudbed, str) or not cloudbed.strip():
            raise ValueError("OpenRCA Market manifest requires cloudbed")
        if cloudbed not in self.dataset_id or cloudbed not in self.task.task_id:
            raise ValueError("OpenRCA Market identifiers must include cloudbed")
        prefix = f"market:{cloudbed}:"
        if any(not entity_id.startswith(prefix) for entity_id in self._entities):
            raise ValueError(
                "OpenRCA Market canonical entity IDs must include cloudbed"
            )
        raw_dataset_profile: object = json.loads(
            self._artifact_path("dataset_profile.json").read_text()
        )
        if not isinstance(raw_dataset_profile, dict):
            raise ValueError("OpenRCA Market prepared trace contract mismatch")
        dataset_profile = cast(dict[str, object], raw_dataset_profile)
        if dataset_profile.get("prepared_trace_contract") != {
            "cloudbed": cloudbed,
            "join_keys": ["trace_id", "parent_span", "span_id"],
            "rows_filtered_to_allowed_window": True,
            "source_path": "trace/trace_span.csv",
            "timestamp_unit": "milliseconds",
        }:
            raise ValueError("OpenRCA Market prepared trace contract mismatch")
        self._cloudbed = cloudbed
        self._source_schema_files = {
            path: cast(dict[str, object], files[path]) for path in sorted(files)
        }

    def _load_metric_bindings(
        self, catalog: Mapping[str, object]
    ) -> dict[tuple[str, str], dict[str, Any]]:
        if not self.profile.source_schema_required:
            raw_bindings = catalog.get("metric_bindings")
            if raw_bindings is None:
                return {}
            if not isinstance(raw_bindings, list):
                raise ValueError("metric_catalog metric_bindings must be an array")
            bindings: dict[tuple[str, str], dict[str, Any]] = {}
            for raw in cast(list[object], raw_bindings):
                if not isinstance(raw, dict):
                    raise ValueError("metric binding must be an object")
                item = cast(dict[str, object], raw)
                canonical_id = self._binding_string(item, "canonical_id")
                metric_id = self._binding_string(item, "metric_id")
                values = {
                    name: self._binding_string(item, name)
                    for name in (
                        "source_path",
                        "layout",
                        "entity_field",
                        "original_entity_key",
                        "value_field",
                        "timestamp_unit",
                    )
                }
                for optional in ("evidence_role", "metric_field", "source_metric"):
                    if optional in item:
                        values[optional] = self._binding_string(item, optional)
                values["original_entity_keys"] = [values["original_entity_key"]]
                values.update(self._binding_source_member_filter(item))
                key = (canonical_id, metric_id)
                entity = self._entities.get(canonical_id)
                if (
                    key in bindings
                    or entity is None
                    or metric_id not in self._metric_ids
                    or values["source_path"] not in entity.sources
                    or values["original_entity_key"]
                    not in entity.source_aliases.get(
                        values["source_path"], [entity.original_id]
                    )
                ):
                    raise ValueError("invalid or duplicate metric binding")
                bindings[key] = values
            return bindings
        raw_bindings = catalog.get("metric_bindings")
        if not isinstance(raw_bindings, list):
            raise ValueError("OpenRCA Market metric_catalog requires metric_bindings")
        bindings: dict[tuple[str, str], dict[str, Any]] = {}
        for raw in cast(list[object], raw_bindings):
            if not isinstance(raw, dict):
                raise ValueError("OpenRCA Market metric binding must be an object")
            item = cast(dict[str, object], raw)
            required = (
                "canonical_id",
                "metric_id",
                "source_path",
                "layout",
                "entity_field",
                "original_entity_key",
                "value_field",
                "timestamp_unit",
            )
            values: dict[str, Any] = {
                name: self._binding_string(item, name) for name in required
            }
            raw_entity_keys = item.get("original_entity_keys")
            if raw_entity_keys is None:
                original_entity_keys = [values["original_entity_key"]]
            elif not isinstance(raw_entity_keys, list) or not all(
                isinstance(value, str) and value
                for value in cast(list[object], raw_entity_keys)
            ):
                raise ValueError(
                    "metric binding original_entity_keys must be non-empty strings"
                )
            else:
                original_entity_keys = cast(list[str], raw_entity_keys)
            if (
                not original_entity_keys
                or len(original_entity_keys) != len(set(original_entity_keys))
                or values["original_entity_key"] not in original_entity_keys
            ):
                raise ValueError("metric binding original entity keys are invalid")
            values["original_entity_keys"] = original_entity_keys
            values.update(self._binding_source_member_filter(item))
            if "evidence_role" in item:
                values["evidence_role"] = self._binding_string(item, "evidence_role")
            canonical_id = values.pop("canonical_id")
            metric_id = values.pop("metric_id")
            key = (canonical_id, metric_id)
            if key in bindings or metric_id not in self._metric_ids:
                raise ValueError("invalid or duplicate OpenRCA Market metric binding")
            try:
                source_spec = self._source_schema_files[values["source_path"]]
            except KeyError as exc:
                raise ValueError(
                    "metric binding source_path is not in source_schema"
                ) from exc
            expected_layout = str(source_spec.get("metric_layout", "long"))
            entity = self._entities.get(canonical_id)
            binding_entity_matches = bool(
                entity is not None
                and all(
                    raw_key == entity.original_id
                    or (
                        entity.public_candidate_match == entity.original_id
                        and (
                            market_output_component(raw_key) == entity.original_id
                            or market_service_component(raw_key) == entity.original_id
                        )
                    )
                    for raw_key in original_entity_keys
                )
            )
            trace_binding = (
                values["source_path"] == "trace/trace_span.csv"
                and values["layout"] == "trace"
            )
            log_binding = (
                values["source_path"] == "log/log_proxy.csv"
                and values["layout"] == "log"
            )
            if (
                (
                    not trace_binding
                    and not log_binding
                    and not values["source_path"].startswith("metric/")
                )
                or (
                    not trace_binding
                    and not log_binding
                    and values["layout"] != expected_layout
                )
                or values["entity_field"] != source_spec.get("entity_key")
                or values["timestamp_unit"] != source_spec.get("timestamp_unit")
                or not binding_entity_matches
                or entity is None
                or values["source_path"] not in entity.sources
            ):
                raise ValueError("OpenRCA Market metric binding does not match schema")
            if trace_binding:
                transform = self._binding_string(item, "value_transform")
                expected_trace_values = {
                    "duration": "numeric",
                    "status_code": "otel_status_error_indicator",
                }
                if expected_trace_values.get(values["value_field"]) != transform:
                    raise ValueError("invalid OpenRCA Market trace metric binding")
                values["value_transform"] = transform
            elif log_binding:
                transform = self._binding_string(item, "value_transform")
                if (
                    values["value_field"] != "value"
                    or transform != "network_transport_error_indicator"
                ):
                    raise ValueError("invalid OpenRCA Market log metric binding")
                values["value_transform"] = transform
            elif expected_layout == "wide":
                columns = source_spec.get("numeric_metric_columns")
                if (
                    not isinstance(columns, list)
                    or values["value_field"] not in columns
                ):
                    raise ValueError("invalid OpenRCA Market wide metric binding")
            else:
                values["metric_field"] = self._binding_string(item, "metric_field")
                values["source_metric"] = self._binding_string(item, "source_metric")
                if (
                    values["metric_field"] != source_spec.get("metric_key")
                    or values["value_field"] != "value"
                ):
                    raise ValueError("invalid OpenRCA Market long metric binding")
            self._artifact_path(f"telemetry/{values['source_path']}")
            bindings[key] = values
        return bindings

    @staticmethod
    def _binding_source_member_filter(
        item: Mapping[str, object],
    ) -> dict[str, list[str]]:
        raw_filter = item.get("source_member_filter")
        if raw_filter is None:
            return {}
        if (
            not isinstance(raw_filter, list)
            or not raw_filter
            or not all(
                isinstance(value, str) and value and not _forbidden_source_name(value)
                for value in cast(list[object], raw_filter)
            )
        ):
            raise ValueError("metric binding source_member_filter is invalid")
        return {"source_member_filter": cast(list[str], raw_filter)}

    @staticmethod
    def _binding_string(item: Mapping[str, object], name: str) -> str:
        value = item.get(name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"metric binding requires {name}")
        return value

    def _metric_binding(self, entity: TopologyNode, metric_id: str) -> dict[str, Any]:
        binding = self._metric_bindings.get((entity.canonical_id, metric_id))
        if binding is not None:
            return binding
        if self.profile.source_schema_required:
            raise ValueError("metric binding is missing")
        if entity.entity_type == "business_test":
            return {
                "source_path": "metric/metric_app.csv",
                "layout": "wide",
                "entity_field": "tc",
                "original_entity_key": entity.original_id,
                "value_field": metric_id,
                "timestamp_unit": "seconds",
            }
        return {
            "source_path": "metric/metric_container.csv",
            "layout": "long",
            "entity_field": "cmdb_id",
            "original_entity_key": entity.original_id,
            "value_field": "value",
            "timestamp_unit": "seconds",
            "metric_field": "kpi_name",
            "source_metric": metric_id,
        }

    def list_entities(self) -> Result:
        entities = [
            node.model_dump(mode="json")
            for node in sorted(self.topology.nodes, key=lambda item: item.canonical_id)
        ]
        return {
            "status": "success",
            "entities": entities,
            "count": len(entities),
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis="public topology inventory; collection completeness is unknown",
            ),
            "artifact_refs": [
                self._artifact_ref("entity_mapping.json", {"kind": "entity_inventory"}),
                self._artifact_ref("topology.json", {"kind": "nodes"}),
            ],
        }

    def list_metrics(self, entity_id: str | None = None) -> Result:
        entity = None
        if entity_id is None:
            metric_ids = sorted(self._metric_ids)
        else:
            try:
                entity = self._entity(entity_id)
            except ValueError as exc:
                return self._error("invalid_entity", str(exc))
            metric_ids = list(self._catalog_metrics_by_entity.get(entity_id, []))
        status = "success" if metric_ids else "no_data"
        result: Result = {
            "status": status,
            "entity_id": entity_id,
            "metric_ids": metric_ids,
            "log_names": (
                sorted(self._log_names)
                if entity is None
                or any(source in entity.sources for source in self.profile.log_sources)
                else []
            ),
            "count": len(metric_ids),
            "schemas": self._catalog_schemas
            or {
                "app": "wide rows: tc entity with rr, sr, cnt, mrt metrics",
                "container": "long rows: cmdb_id, kpi_name, value",
            },
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis=(
                    "precomputed public metric inventory; "
                    "source collection health unknown"
                ),
            ),
            "artifact_refs": [
                self._artifact_ref("metric_catalog.json", {"entity_id": entity_id}),
                self._artifact_ref(
                    "metric_window_stats.jsonl", {"entity_id": entity_id}
                ),
            ],
        }
        if not metric_ids:
            result["gap"] = self._gap()
        return result

    def query_metric(
        self,
        entity_id: str,
        metric_id: str,
        *,
        period: str = "incident",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> Result:
        try:
            entity = self._entity(entity_id)
            period_name, window = self._window(period, start, end)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        if metric_id not in self._metric_ids:
            return self._error("invalid_metric", f"unknown metric_id: {metric_id}")
        if (entity_id, metric_id) not in self._stats_by_pair:
            return self._no_data(
                window,
                f"metric {metric_id!r} is not present for {entity_id}",
                basis="metric inventory has unknown source collection completeness",
            )

        binding = self._metric_binding(entity, metric_id)
        relative = f"telemetry/{binding['source_path']}"
        entity_column = binding["entity_field"]
        metric_column = binding["value_field"]
        source_metric = binding.get("source_metric")
        metric_field = binding.get("metric_field")
        original_entity = binding.get("original_entity_key", entity.original_id)
        original_entities = set(
            cast(list[str], binding.get("original_entity_keys", [original_entity]))
        )
        source_member_filter = set(
            cast(list[str], binding.get("source_member_filter", []))
        )

        values: list[float] = []
        timestamps: list[float] = []
        points: list[dict[str, int | float | str]] = []
        paired_values: dict[str, list[float]] = {
            "reference": [],
            "incident": [],
        }
        task_windows = {
            "reference": cast(TimeRange, self.task.reference_range),
            "incident": self.task.incident_range,
        }
        try:
            with self._artifact_path(relative).open(
                newline="", encoding="utf-8"
            ) as stream:
                for row in csv.DictReader(stream):
                    self._validate_row_source_member(row, relative)
                    if row[entity_column] not in original_entities:
                        continue
                    if source_member_filter and row.get("_source_member") not in (
                        source_member_filter
                    ):
                        continue
                    if metric_field is not None and row[metric_field] != source_metric:
                        continue
                    timestamp = float(row["timestamp"])
                    if binding.get("timestamp_unit") == "milliseconds":
                        timestamp /= 1000
                    raw_value = row[metric_column]
                    transform = binding.get("value_transform")
                    value = (
                        trace_status_error_indicator(raw_value)
                        if transform == "otel_status_error_indicator"
                        else network_transport_error_indicator(raw_value)
                        if transform == "network_transport_error_indicator"
                        else float(raw_value)
                    )
                    for task_period, task_window in task_windows.items():
                        if self._contains(task_window, timestamp):
                            paired_values[task_period].append(value)
                    if not self._contains(window, timestamp):
                        continue
                    values.append(value)
                    timestamps.append(timestamp)
                    points.append(
                        {
                            "timestamp": self._source_timestamp(timestamp),
                            "value": value,
                            "source_record": int(row["_source_record"]),
                            "source_path": binding["source_path"],
                            "original_entity_key": row[entity_column],
                        }
                    )
        except (OSError, csv.Error, KeyError, ValueError) as exc:
            return self._error("source_read_error", str(exc), window)

        scale_hint, pressure_direction = self._metric_name_hints(metric_id)
        paired_metric_summary: Result = {
            "counts_complete": True,
            "scale_hint": scale_hint,
            "pressure_direction": pressure_direction,
            "hint_basis": "heuristic from metric name",
        }
        for task_period, task_values in paired_values.items():
            paired_metric_summary[task_period] = {
                "sample_count": len(task_values),
                "min": min(task_values) if task_values else None,
                "median": statistics.median(task_values) if task_values else None,
                "max": max(task_values) if task_values else None,
            }
        reference_values = paired_values["reference"]
        incident_values = paired_values["incident"]
        reference_min = min(reference_values) if reference_values else None
        reference_max = max(reference_values) if reference_values else None
        incident_summary = cast(Result, paired_metric_summary["incident"])
        incident_summary.update(
            above_reference_max_count=(
                sum(value > reference_max for value in incident_values)
                if reference_max is not None
                else None
            ),
            below_reference_min_count=(
                sum(value < reference_min for value in incident_values)
                if reference_min is not None
                else None
            ),
        )
        if not values:
            result = self._no_data(
                window,
                "no matching metric samples in the requested range",
                basis="source collection completeness and health are unknown",
            )
            result["paired_metric_summary"] = paired_metric_summary
            return result
        sample_interval, observed_range = self._observed(timestamps, window)
        return {
            "status": "success",
            "entity_id": entity_id,
            "metric_id": metric_id,
            "period": period_name,
            "timestamp_unit": "seconds",
            "value_unit": "source-native; not inferred",
            "sample_count": len(values),
            "summary": {
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
            },
            "paired_metric_summary": paired_metric_summary,
            "points": points,
            "coverage": self._coverage(
                window,
                completeness="unknown",
                basis=(
                    "all matching rows in the frozen slice; "
                    "upstream collection health unknown"
                ),
                observed_range=observed_range,
                sample_interval=sample_interval,
            ),
            "artifact_refs": [
                self._artifact_ref(
                    relative,
                    {
                        "entity_id": entity_id,
                        "metric_id": metric_id,
                        "source_path": binding["source_path"],
                        "original_entity_keys": sorted(original_entities),
                        "source_records": [
                            point["source_record"] for point in points[:5]
                        ],
                    },
                )
            ],
        }

    def _scan_metric_group(
        self, entity_id: str, metric_ids: Sequence[str]
    ) -> tuple[
        dict[tuple[str, str], tuple[list[float], list[float]]],
        dict[str, dict[str, Any]],
        set[str],
    ]:
        entity = self._entity(entity_id)
        bindings = {
            metric_id: self._metric_binding(entity, metric_id)
            for metric_id in metric_ids
        }
        grouped: dict[str, list[tuple[str, dict[str, Any], set[str]]]] = {}
        for metric_id, binding in bindings.items():
            grouped.setdefault(cast(str, binding["source_path"]), []).append(
                (
                    metric_id,
                    binding,
                    set(
                        cast(
                            list[str],
                            binding.get(
                                "original_entity_keys",
                                [
                                    binding.get(
                                        "original_entity_key", entity.original_id
                                    )
                                ],
                            ),
                        )
                    ),
                )
            )
        series: dict[tuple[str, str], tuple[list[float], list[float]]] = {
            (metric_id, period): ([], [])
            for metric_id in metric_ids
            for period in ("reference", "incident")
        }
        artifact_refs: dict[str, dict[str, Any]] = {}
        task_windows = {
            "reference": cast(TimeRange, self.task.reference_range),
            "incident": self.task.incident_range,
        }
        for source_path, source_bindings in grouped.items():
            relative = f"telemetry/{source_path}"
            selected_records: set[int] = set()
            with self._artifact_path(relative).open(
                newline="", encoding="utf-8"
            ) as stream:
                for row in csv.DictReader(stream):
                    self._validate_row_source_member(row, relative)
                    for metric_id, binding, aliases in source_bindings:
                        entity_field = cast(str, binding["entity_field"])
                        if row[entity_field] not in aliases:
                            continue
                        source_member_filter = set(
                            cast(list[str], binding.get("source_member_filter", []))
                        )
                        if (
                            source_member_filter
                            and row.get("_source_member") not in source_member_filter
                        ):
                            continue
                        metric_field = binding.get("metric_field")
                        if metric_field is not None and row[
                            cast(str, metric_field)
                        ] != (binding.get("source_metric")):
                            continue
                        timestamp = float(row["timestamp"])
                        if binding.get("timestamp_unit") == "milliseconds":
                            timestamp /= 1000
                        period = next(
                            (
                                name
                                for name, window in task_windows.items()
                                if self._contains(window, timestamp)
                            ),
                            None,
                        )
                        if period is None:
                            continue
                        raw_value = row[cast(str, binding["value_field"])]
                        transform = binding.get("value_transform")
                        value = (
                            trace_status_error_indicator(raw_value)
                            if transform == "otel_status_error_indicator"
                            else network_transport_error_indicator(raw_value)
                            if transform == "network_transport_error_indicator"
                            else float(raw_value)
                        )
                        timestamps, values = series[(metric_id, period)]
                        timestamps.append(timestamp)
                        values.append(value)
                        if len(selected_records) < 40:
                            selected_records.add(int(row["_source_record"]))
            artifact_ref = self._artifact_ref(
                relative,
                {
                    "entity_id": entity_id,
                    "metric_ids": sorted(
                        metric_id for metric_id, _binding, _aliases in source_bindings
                    ),
                    "source_path": source_path,
                    "source_records": sorted(selected_records),
                },
            )
            encoded = json.dumps(artifact_ref, separators=(",", ":"), sort_keys=True)
            artifact_refs[encoded] = artifact_ref
        statuses = {
            "success"
            if all(
                series[(metric_id, period)][1] for period in ("reference", "incident")
            )
            else "no_data"
            for metric_id in metric_ids
        }
        return series, artifact_refs, statuses

    def _metric_evidence_role(self, entity_id: str, metric_id: str) -> str:
        entity = self._entities[entity_id]
        binding = self._metric_bindings.get((entity_id, metric_id), {})
        return str(
            self._stats_by_pair.get((entity_id, metric_id), {}).get(
                "evidence_role", binding.get("evidence_role", entity.evidence_role)
            )
        )

    def _impact_metric_pairs(
        self,
    ) -> list[tuple[str, str, str, list[tuple[str, str]]]]:
        pairs: list[tuple[str, str, str, list[tuple[str, str]]]] = []
        for entity_id, metric_ids in sorted(self._catalog_metrics_by_entity.items()):
            entity = self._entities.get(entity_id)
            if entity is None or entity_id in self._implicit_impact_role_entity_ids:
                continue
            groups: dict[
                tuple[str, str, tuple[str, ...]],
                dict[str, list[tuple[int, str]]],
            ] = {}
            for metric_id in metric_ids:
                role, rank = _profile_metric_role(metric_id)
                if role is None:
                    continue
                binding = self._metric_binding(entity, metric_id)
                if binding.get("layout") != "wide" and not (
                    binding.get("layout") == "long"
                    and entity.evidence_role == "impact_only"
                    and self._metric_evidence_role(entity_id, metric_id)
                    == "impact_only"
                ):
                    continue
                source_path = binding.get("source_path")
                entity_field = binding.get("entity_field")
                if not isinstance(source_path, str) or not isinstance(
                    entity_field, str
                ):
                    continue
                aliases = tuple(
                    sorted(
                        cast(
                            list[str],
                            binding.get(
                                "original_entity_keys",
                                [
                                    binding.get(
                                        "original_entity_key", entity.original_id
                                    )
                                ],
                            ),
                        )
                    )
                )
                groups.setdefault(
                    (source_path, entity_field, aliases),
                    {"count": [], "error": [], "latency": [], "success": []},
                )[role].append((rank, metric_id))
            for roles in groups.values():
                if not roles["count"] or not roles["latency"]:
                    continue
                count_metric = min(roles["count"])[1]
                latency_metric = min(roles["latency"])[1]
                auxiliary = [
                    (role, min(roles[role])[1])
                    for role in ("success", "error")
                    if roles[role]
                ]
                pairs.append((entity_id, count_metric, latency_metric, auxiliary))
        return pairs

    @staticmethod
    def _impact_points(
        count_series: tuple[list[float], list[float]],
        latency_series: tuple[list[float], list[float]],
    ) -> list[tuple[float, float, float | None]]:
        counts: dict[float, list[float]] = {}
        latencies: dict[float, list[float]] = {}
        for timestamp, value in zip(*count_series, strict=True):
            counts.setdefault(timestamp, []).append(value)
        for timestamp, value in zip(*latency_series, strict=True):
            latencies.setdefault(timestamp, []).append(value)
        return [
            (
                timestamp,
                sum(counts[timestamp]),
                statistics.fmean(latencies[timestamp]),
            )
            for timestamp in sorted(counts.keys() & latencies.keys())
        ]

    def query_impact_profile(self) -> Result:
        """Compare source-backed request volume and latency as impact-only evidence."""
        pairs = self._impact_metric_pairs()
        if not pairs:
            return {
                "status": "no_data",
                "evidence_role": "impact_only",
                "profiles": [],
                "reason": (
                    "no source-backed metric group exposes both request volume "
                    "and latency as impact evidence"
                ),
                "coverage": self._coverage(
                    self.task.allowed_query_range,
                    completeness="unknown",
                    basis="generic metric binding, evidence role, and semantics",
                ),
                "artifact_refs": [],
                "gap": self._gap(),
            }
        profiles: list[dict[str, Any]] = []
        source_refs: dict[str, dict[str, Any]] = {}
        for entity_id, count_metric, latency_metric, auxiliary_metrics in pairs:
            selected_metric_ids = [
                count_metric,
                latency_metric,
                *(metric_id for _role, metric_id in auxiliary_metrics),
            ]
            try:
                series, artifact_refs, _statuses = self._scan_metric_group(
                    entity_id, selected_metric_ids
                )
            except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
                return self._error("source_read_error", str(exc))
            profile_refs = list(artifact_refs.values())
            for encoded, artifact_ref in artifact_refs.items():
                source_refs[encoded] = artifact_ref
            reference_points = self._impact_points(
                series[(count_metric, "reference")],
                series[(latency_metric, "reference")],
            )
            incident_points = self._impact_points(
                series[(count_metric, "incident")],
                series[(latency_metric, "incident")],
            )
            if not reference_points or not incident_points:
                continue
            reference_count = sum(point[1] for point in reference_points)
            incident_count = sum(point[1] for point in incident_points)
            reference_latency_values = [
                cast(float, point[2]) for point in reference_points
            ]
            incident_latency_values = [
                cast(float, point[2]) for point in incident_points
            ]
            reference_latency = _percentile(reference_latency_values, 0.95)
            incident_latency = _percentile(incident_latency_values, 0.95)
            _combined_effect, effect_components = _profile_effect_size(
                reference_count,
                incident_count,
                reference_latency,
                incident_latency,
            )
            latency_effect = effect_components["latency"]
            latency_markers = _profile_onset_markers(
                reference_points,
                incident_points,
                count_is_harmful=False,
                minimum_consecutive_points=2,
                minimum_relative_change=0.2,
            )
            harmful_kpis: list[dict[str, Any]] = [
                {
                    "metric_name": latency_metric,
                    "harmful_direction": "higher",
                    "effect_size": latency_effect,
                    **latency_markers,
                }
            ]
            for role, metric_id in auxiliary_metrics:
                reference_timestamps, reference_values = series[
                    (metric_id, "reference")
                ]
                incident_timestamps, incident_values = series[(metric_id, "incident")]
                if not reference_values or not incident_values:
                    continue
                reference_value = statistics.median(reference_values)
                incident_value = statistics.median(incident_values)
                raw_effect = _symmetric_relative_effect(reference_value, incident_value)
                harmful_effect = -raw_effect if role == "success" else raw_effect
                auxiliary_markers = _profile_onset_markers(
                    [
                        (timestamp, 0.0, value)
                        for timestamp, value in zip(
                            reference_timestamps, reference_values, strict=True
                        )
                    ],
                    [
                        (timestamp, 0.0, value)
                        for timestamp, value in zip(
                            incident_timestamps, incident_values, strict=True
                        )
                    ],
                    count_is_harmful=False,
                    value_direction="lower" if role == "success" else "higher",
                    minimum_consecutive_points=2,
                    minimum_relative_change=0.1,
                )
                harmful_kpis.append(
                    {
                        "metric_name": metric_id,
                        "harmful_direction": (
                            "lower" if role == "success" else "higher"
                        ),
                        "effect_size": round(harmful_effect, 6),
                        **auxiliary_markers,
                    }
                )
                effect_components[role] = round(harmful_effect, 6)
            observable_kpis = [
                item
                for item in harmful_kpis
                if isinstance(item.get("effect_size"), (int, float))
                and not isinstance(item.get("effect_size"), bool)
                and float(item["effect_size"]) > 0
                and isinstance(item.get("first_anomalous"), (int, float))
                and not isinstance(item.get("first_anomalous"), bool)
            ]
            selected_harmful = (
                min(observable_kpis, key=lambda item: float(item["first_anomalous"]))
                if observable_kpis
                else None
            )
            markers = (
                {
                    key: selected_harmful[key]
                    for key in (
                        "last_normal",
                        "first_abnormal",
                        "first_anomalous",
                        "onset_interval",
                        "segment_start",
                        "segment_end",
                        "consecutive_points",
                    )
                }
                if selected_harmful is not None
                else latency_markers
            )
            impact_observable = selected_harmful is not None
            harmful_effects = [
                float(item["effect_size"])
                for item in harmful_kpis
                if isinstance(item.get("effect_size"), (int, float))
                and not isinstance(item.get("effect_size"), bool)
            ]
            effect_size = max(harmful_effects) if harmful_effects else 0.0
            sample_interval = _profile_sample_interval(
                reference_points, incident_points
            )
            reference_window = cast(TimeRange, self.task.reference_range)
            expected_sample_count = (
                {
                    "reference": round(
                        (reference_window.end - reference_window.start).total_seconds()
                        / sample_interval
                    ),
                    "incident": round(
                        (
                            self.task.incident_range.end
                            - self.task.incident_range.start
                        ).total_seconds()
                        / sample_interval
                    ),
                }
                if sample_interval is not None and sample_interval > 0
                else None
            )
            profiles.append(
                {
                    "entity_id": entity_id,
                    "metric_name": latency_metric,
                    "count_metric_name": count_metric,
                    "evidence_role": "impact_only",
                    **markers,
                    "reference_count": reference_count,
                    "incident_count": incident_count,
                    "reference_latency": reference_latency,
                    "incident_latency": incident_latency,
                    "reference_latency_stats": _profile_latency_stats(
                        reference_latency_values
                    ),
                    "incident_latency_stats": _profile_latency_stats(
                        incident_latency_values
                    ),
                    "sample_interval_seconds": sample_interval,
                    "sample_count": {
                        "reference": len(reference_points),
                        "incident": len(incident_points),
                    },
                    "expected_sample_count": expected_sample_count,
                    "latency_statistic": "p95 of source latency series",
                    "latency_unit": "source-native; not inferred",
                    "effect_size": effect_size,
                    "effect_size_components": effect_components,
                    "effect_size_basis": "symmetric relative shift",
                    "harmful_kpis": harmful_kpis,
                    "identifiability": (
                        "impact_observable" if impact_observable else "unidentifiable"
                    ),
                    "identifiability_reason": (
                        "a harmful KPI crossed the reference envelope"
                        if impact_observable
                        else "request count alone is not a harmful impact signal"
                        if not any(
                            isinstance(item.get("effect_size"), (int, float))
                            and not isinstance(item.get("effect_size"), bool)
                            and float(item["effect_size"]) > 0
                            for item in harmful_kpis
                        )
                        else "harmful KPI shift lacks a sustained material onset"
                    ),
                    "competing_artifact_refs": [],
                    "modal_lag_seconds": None,
                    "modal_lag_reason": (
                        "source has no independent collection timestamp"
                    ),
                    "artifact_refs": profile_refs,
                }
            )
        profiles.sort(
            key=lambda profile: (
                -abs(cast(float, profile["effect_size"])),
                cast(str, profile["entity_id"]),
                cast(str, profile["metric_name"]),
            )
        )
        if not profiles:
            return {
                "status": "no_data",
                "evidence_role": "impact_only",
                "profiles": [],
                "reason": (
                    "paired request-volume and latency groups have no samples in "
                    "both task windows"
                ),
                "coverage": self._coverage(
                    self.task.allowed_query_range,
                    completeness="unknown",
                    basis="source-backed impact metric groups",
                ),
                "artifact_refs": list(source_refs.values()),
                "gap": self._gap(),
            }
        return {
            "status": "success",
            "evidence_role": "impact_only",
            "profiles": profiles,
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis=(
                    "paired request-volume and latency rows from generic metric "
                    "bindings; upstream collection health unknown"
                ),
            ),
            "artifact_refs": list(source_refs.values()),
        }

    def query_metric_profile(
        self,
        entity_id: str,
        metric_ids: list[str],
        *,
        peer_entity_ids: list[str] | None = None,
        derived_ratios: list[dict[str, str]] | None = None,
        fault_type: str | None = None,
    ) -> Result:
        """Return typed paired shapes without interpreting them as a root cause."""
        try:
            entity = self._entity(entity_id)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        if entity_id in self._implicit_impact_role_entity_ids:
            return self._error(
                "ambiguous_evidence_role",
                "business impact entity requires an explicit evidence_role; reprepare",
            )
        if (
            not metric_ids
            or len(metric_ids) > 16
            or len(metric_ids) != len(set(metric_ids))
        ):
            return self._error(
                "invalid_arguments",
                "metric_ids must contain one to sixteen unique exact catalog values",
            )
        if any(
            metric_id not in self._catalog_metrics_by_entity.get(entity_id, [])
            for metric_id in metric_ids
        ):
            return self._error(
                "invalid_metric",
                "every metric_id must be available on the requested entity",
            )
        metric_roles = {
            self._metric_evidence_role(entity_id, metric_id) for metric_id in metric_ids
        }
        if len(metric_roles) != 1:
            return self._error(
                "invalid_arguments",
                "one metric profile cannot mix root-cause and impact evidence",
            )
        evidence_role = next(iter(metric_roles))
        if evidence_role not in {"root_cause_candidate", "impact_only"}:
            return self._error(
                "invalid_arguments", "metrics have an unsupported evidence_role"
            )
        if entity.evidence_role == "impact_only":
            evidence_role = "impact_only"
        if evidence_role == "impact_only" and fault_type is not None:
            return self._error(
                "invalid_arguments",
                "impact-only metrics cannot be profiled as a root-cause fault type",
            )
        if fault_type is not None and fault_type not in self._public_fault_types:
            return self._error(
                "invalid_arguments",
                "fault_type must be an exact public vocabulary value",
            )
        if (
            fault_type is not None
            and self.profile is _MARKET_PROFILE
            and not self._market_fault_matches_entity(entity_id, fault_type)
        ):
            return self._error(
                "invalid_arguments",
                "fault_type is incompatible with the requested entity type",
            )
        peers = (
            peer_entity_ids
            if peer_entity_ids is not None
            else self._default_peer_entities(entity_id, metric_ids)
        )
        if len(peers) > 32 or len(peers) != len(set(peers)):
            return self._error(
                "invalid_arguments", "peer_entity_ids must be unique and at most 32"
            )
        try:
            for peer in peers:
                self._entity(peer)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))

        try:
            series, artifact_refs, statuses = self._scan_metric_group(
                entity_id, metric_ids
            )
        except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
            return self._error("source_read_error", str(exc))
        profiles: dict[str, dict[str, Any]] = {}
        for metric_id in metric_ids:
            reference_timestamps, reference_values = series[(metric_id, "reference")]
            incident_timestamps, incident_values = series[(metric_id, "incident")]
            scale_hint, direction = self._metric_name_hints(metric_id)
            reference_shape = _series_shape(reference_values, reference_timestamps)
            incident_shape = _series_shape(incident_values, incident_timestamps)
            burst = _burst_profile(
                incident_values,
                incident_timestamps,
                reference_min=cast(float | None, reference_shape["min"]),
                reference_max=cast(float | None, reference_shape["max"]),
                direction=str(direction),
            )
            relative_median_shift = (
                abs(
                    cast(float, incident_shape["median"])
                    - cast(float, reference_shape["median"])
                )
                / max(
                    abs(cast(float, incident_shape["median"])),
                    abs(cast(float, reference_shape["median"])),
                    1.0,
                )
                if incident_shape["median"] is not None
                and reference_shape["median"] is not None
                else 0.0
            )
            relative_p95_shift = (
                abs(
                    cast(float, incident_shape["p95"])
                    - cast(float, reference_shape["p95"])
                )
                / max(
                    abs(cast(float, incident_shape["p95"])),
                    abs(cast(float, reference_shape["p95"])),
                    1.0,
                )
                if incident_shape["p95"] is not None
                and reference_shape["p95"] is not None
                else 0.0
            )
            relative_mean_shift = (
                abs(
                    cast(float, incident_shape["mean"])
                    - cast(float, reference_shape["mean"])
                )
                / max(
                    abs(cast(float, incident_shape["mean"])),
                    abs(cast(float, reference_shape["mean"])),
                    1.0,
                )
                if incident_shape["mean"] is not None
                and reference_shape["mean"] is not None
                else 0.0
            )
            longest_segment = cast(
                Mapping[str, Any] | None, burst.get("longest_segment")
            )
            raw_reference_change = reference_shape["largest_change"]
            raw_incident_change = incident_shape["largest_change"]
            reference_change = (
                cast(Mapping[str, Any], raw_reference_change).get(
                    "absolute_change", 0.0
                )
                if isinstance(raw_reference_change, Mapping)
                else 0.0
            )
            incident_change = (
                cast(Mapping[str, Any], raw_incident_change).get("absolute_change", 0.0)
                if isinstance(raw_incident_change, Mapping)
                else 0.0
            )
            change_ratio = (
                float(incident_change) / max(float(reference_change), 1e-12)
                if isinstance(reference_change, (int, float))
                and not isinstance(reference_change, bool)
                and isinstance(incident_change, (int, float))
                and not isinstance(incident_change, bool)
                else 0.0
            )
            sparse_extreme = bool(
                cast(float, burst["crossing_fraction"]) >= 0.1
                and cast(float, burst["crossing_fraction"]) < 0.2
                and relative_p95_shift >= 0.5
                and longest_segment is not None
                and isinstance(longest_segment.get("duration_seconds"), (int, float))
                and not isinstance(longest_segment.get("duration_seconds"), bool)
                and cast(float, longest_segment["duration_seconds"]) >= 120.0
            )
            capacity_change = bool(
                fault_type is not None
                and "space" in self._semantic_concepts(fault_type)
                and "space" in self._semantic_concepts(metric_id)
                and cast(float, burst["crossing_fraction"]) >= 0.1
                and change_ratio >= 5.0
            )
            semantic_type = _metric_semantic_type(metric_id)
            event_transition = bool(
                semantic_type == "event"
                and reference_shape["min"] is not None
                and reference_shape["max"] is not None
                and incident_shape["min"] is not None
                and incident_shape["max"] is not None
                and (
                    cast(float, incident_shape["min"])
                    < cast(float, reference_shape["min"])
                    or cast(float, incident_shape["max"])
                    > cast(float, reference_shape["max"])
                )
            )
            incident_event_burst = bool(
                semantic_type == "event"
                and incident_shape["median"] == 0
                and isinstance(incident_shape["max"], (int, float))
                and not isinstance(incident_shape["max"], bool)
                and cast(float, incident_shape["max"]) > 0
            )
            if incident_event_burst and burst["onset_timestamp"] is None:
                event_points = sorted(
                    (timestamp, value)
                    for timestamp, value in zip(
                        incident_timestamps, incident_values, strict=True
                    )
                    if value > 0
                )
                if event_points:
                    prior_timestamps = [
                        timestamp
                        for timestamp in incident_timestamps
                        if timestamp < event_points[0][0]
                    ]
                    burst.update(
                        crossing_count=len(event_points),
                        crossing_fraction=len(event_points) / len(incident_values),
                        segment_count=1,
                        longest_segment={
                            "start": event_points[0][0],
                            "end": event_points[-1][0],
                            "sample_count": len(event_points),
                            "duration_seconds": max(
                                0.0, event_points[-1][0] - event_points[0][0]
                            ),
                        },
                        onset_timestamp=event_points[0][0],
                        onset_lower_bound_timestamp=(
                            max(prior_timestamps)
                            if prior_timestamps
                            else event_points[0][0]
                        ),
                        segments=[],
                    )
            sustained_distribution_shift = bool(
                relative_mean_shift >= 0.2 and relative_p95_shift >= 0.2
            )
            observed_gaps = [
                right - left
                for timestamps in (reference_timestamps, incident_timestamps)
                for left, right in zip(sorted(timestamps), sorted(timestamps)[1:])
                if right > left
            ]
            raw_interval = self._stats_by_pair.get((entity_id, metric_id), {}).get(
                "median_observed_interval_seconds"
            )
            sample_interval = (
                statistics.median(observed_gaps)
                if observed_gaps
                else float(raw_interval)
                if isinstance(raw_interval, (int, float))
                and not isinstance(raw_interval, bool)
                and raw_interval > 0
                else None
            )
            reference_window = cast(TimeRange, self.task.reference_range)
            expected_samples = (
                {
                    "reference": max(
                        1,
                        round(
                            (
                                reference_window.end - reference_window.start
                            ).total_seconds()
                            / sample_interval
                        ),
                    ),
                    "incident": max(
                        1,
                        round(
                            (
                                self.task.incident_range.end
                                - self.task.incident_range.start
                            ).total_seconds()
                            / sample_interval
                        ),
                    ),
                }
                if sample_interval is not None and sample_interval > 0
                else None
            )
            sample_counts = {
                "reference": len(reference_values),
                "incident": len(incident_values),
            }
            missing_evidence_reasons = [
                f"{period} has fewer than 3 samples"
                for period, count in sample_counts.items()
                if count < 3
            ]
            if expected_samples is not None:
                missing_evidence_reasons.extend(
                    f"{period} covers less than half of expected samples"
                    for period, count in sample_counts.items()
                    if count / expected_samples[period] < 0.5
                )
            finding_sample_ready = not missing_evidence_reasons
            profiles[metric_id] = {
                "metric_id": metric_id,
                "evidence_role": evidence_role,
                "semantic_type": semantic_type,
                "semantic_quality": "inferred",
                "scale_hint": scale_hint,
                "pressure_direction": direction,
                "reference": reference_shape,
                "incident": incident_shape,
                "incident_burst": burst,
                "sample_interval_seconds": sample_interval,
                "sample_count": sample_counts,
                "expected_sample_count": expected_samples,
                "finding_sample_ready": finding_sample_ready,
                "missing_evidence_reasons": missing_evidence_reasons,
                "relative_median_shift": relative_median_shift,
                "relative_mean_shift": relative_mean_shift,
                "relative_p95_shift": relative_p95_shift,
                "strong_incident_change": bool(
                    (
                        cast(float, burst["crossing_fraction"]) >= 0.2
                        and (relative_median_shift >= 0.05 or relative_p95_shift >= 0.1)
                    )
                    or sparse_extreme
                    or capacity_change
                    or event_transition
                    or incident_event_burst
                    or sustained_distribution_shift
                ),
                "change_shape": (
                    "state_transition"
                    if event_transition
                    else "incident_event_burst"
                    if incident_event_burst
                    else "distribution_shift"
                    if sustained_distribution_shift
                    else "capacity_change"
                    if capacity_change
                    else "sparse_extreme"
                    if sparse_extreme
                    else "sustained"
                ),
                "largest_change_ratio": change_ratio,
            }
            series[(metric_id, "reference")] = (
                reference_timestamps,
                reference_values,
            )
            series[(metric_id, "incident")] = (
                incident_timestamps,
                incident_values,
            )

        derived_profiles: list[dict[str, Any]] = []
        for raw_ratio in derived_ratios or []:
            if set(raw_ratio) != {"name", "numerator", "denominator"}:
                return self._error(
                    "invalid_arguments",
                    "derived ratios require name, numerator, and denominator",
                )
            name = raw_ratio["name"]
            numerator = raw_ratio["numerator"]
            denominator = raw_ratio["denominator"]
            if (
                not name
                or numerator not in metric_ids
                or denominator not in metric_ids
                or numerator == denominator
            ):
                return self._error(
                    "invalid_arguments",
                    "derived ratio inputs must be distinct selected metric_ids",
                )
            periods: dict[str, Any] = {}
            for period in ("reference", "incident"):
                numerator_points = dict(zip(*series[(numerator, period)], strict=True))
                denominator_points = dict(
                    zip(*series[(denominator, period)], strict=True)
                )
                timestamps = sorted(numerator_points.keys() & denominator_points.keys())
                ratio_timestamps: list[float] = []
                ratios: list[float] = []
                for timestamp in timestamps:
                    denominator_value = denominator_points[timestamp]
                    if denominator_value == 0:
                        continue
                    ratio_timestamps.append(timestamp)
                    ratios.append(numerator_points[timestamp] / denominator_value)
                periods[period] = _series_shape(ratios, ratio_timestamps)
            derived_profiles.append(
                {
                    "name": name,
                    "formula": "ratio",
                    "numerator": numerator,
                    "denominator": denominator,
                    "independence_group": sorted({numerator, denominator}),
                    **periods,
                }
            )

        peer_comparison: list[dict[str, Any]] = []
        for metric_id in metric_ids:
            target = self._stats_by_pair.get((entity_id, metric_id), {})
            target_score = self._paired_stats_shift(target)
            peer_scores = [
                score
                for peer in peers
                if (peer, metric_id) in self._stats_by_pair
                and (
                    score := self._paired_stats_shift(
                        self._stats_by_pair.get((peer, metric_id), {})
                    )
                )
                is not None
            ]
            if target_score is None or not peer_scores:
                continue
            center = statistics.median(peer_scores)
            mad = _robust_mad(peer_scores)
            peer_comparison.append(
                {
                    "metric_id": metric_id,
                    "target_score": target_score,
                    "peer_count": len(peer_scores),
                    "peer_median": center,
                    "peer_mad": mad,
                    "robust_z": (
                        (target_score - center) / (1.4826 * mad)
                        if mad is not None and mad > 0
                        else None
                    ),
                    "percentile": sum(score < target_score for score in peer_scores)
                    / len(peer_scores),
                }
            )
        strong_metric_ids = sorted(
            metric_id
            for metric_id, profile in profiles.items()
            if profile.get("strong_incident_change") is True
        )
        isolated_metric_ids = sorted(
            cast(str, item["metric_id"])
            for item in peer_comparison
            if (
                isinstance(item.get("peer_count"), int)
                and cast(int, item["peer_count"]) >= 2
                and isinstance(item.get("target_score"), (int, float))
                and not isinstance(item.get("target_score"), bool)
                and cast(float, item["target_score"]) > 0.0
                and (
                    isinstance(item.get("robust_z"), (int, float))
                    and not isinstance(item.get("robust_z"), bool)
                    and cast(float, item["robust_z"]) >= 3.0
                    or isinstance(item.get("percentile"), (int, float))
                    and not isinstance(item.get("percentile"), bool)
                    and cast(float, item["percentile"]) >= 0.9
                )
            )
        )
        event_transition_metric_ids = sorted(
            metric_id
            for metric_id, profile in profiles.items()
            if profile.get("change_shape") == "state_transition"
        )
        saturation_metric_ids = sorted(
            metric_id
            for metric_id, profile in profiles.items()
            if profile.get("strong_incident_change") is True
            and profile.get("scale_hint") == "0..100"
            and isinstance(
                cast(Mapping[str, object], profile["incident"]).get("p95"),
                (int, float),
            )
            and not isinstance(
                cast(Mapping[str, object], profile["incident"]).get("p95"), bool
            )
            and float(
                cast(float, cast(Mapping[str, object], profile["incident"])["p95"])
            )
            >= 80.0
            and cast(float, profile.get("relative_mean_shift", 0.0)) >= 0.2
        )
        resource_burst_metric_ids = sorted(
            metric_id
            for metric_id, profile in profiles.items()
            if _incident_resource_burst(profile)
        )
        qualifying_metric_ids = sorted(
            metric_id
            for metric_id in (
                (set(strong_metric_ids) & set(isolated_metric_ids))
                | set(event_transition_metric_ids)
                | set(saturation_metric_ids)
                | set(resource_burst_metric_ids)
            )
            if profiles[metric_id].get("finding_sample_ready") is True
        )
        incident_start = self.task.incident_range.start.timestamp()
        qualifying_onsets = [
            cast(
                float,
                cast(Mapping[str, Any], profiles[metric_id]["incident_burst"])[
                    "onset_timestamp"
                ],
            )
            for metric_id in qualifying_metric_ids
            if isinstance(profiles[metric_id].get("incident_burst"), Mapping)
            and isinstance(
                cast(Mapping[str, Any], profiles[metric_id]["incident_burst"]).get(
                    "onset_timestamp"
                ),
                (int, float),
            )
        ]
        boundary_aligned = bool(qualifying_onsets) and all(
            cast(
                float,
                cast(Mapping[str, Any], profiles[metric_id]["incident_burst"])[
                    "onset_timestamp"
                ],
            )
            <= incident_start
            + max(
                1.0,
                float(profiles[metric_id].get("sample_interval_seconds") or 1.0),
            )
            for metric_id in qualifying_metric_ids
            if isinstance(profiles[metric_id].get("incident_burst"), Mapping)
            and isinstance(
                cast(Mapping[str, Any], profiles[metric_id]["incident_burst"]).get(
                    "onset_timestamp"
                ),
                (int, float),
            )
        )
        anchor_metric_ids = (
            sorted(
                metric_id
                for metric_id in qualifying_metric_ids
                if self._fault_metric_compatible(fault_type, metric_id)
                and _pressure_direction_aligned(profiles[metric_id])
            )
            if fault_type is not None
            else []
        )
        root_cause_score = 0.0
        for metric_id in anchor_metric_ids:
            profile = profiles[metric_id]
            burst = cast(Mapping[str, Any], profile["incident_burst"])
            root_cause_score += min(
                1.0, cast(float, profile["relative_p95_shift"])
            ) * cast(float, burst["crossing_fraction"])
        mechanism_signature: str | None = None
        state_transition_metrics = sorted(
            metric_id
            for metric_id in anchor_metric_ids
            if profiles[metric_id].get("change_shape") == "state_transition"
        )
        saturation_metrics = sorted(
            metric_id
            for metric_id in anchor_metric_ids
            if profiles[metric_id].get("scale_hint") == "0..100"
            and isinstance(
                cast(Mapping[str, object], profiles[metric_id]["incident"]).get("p95"),
                (int, float),
            )
            and not isinstance(
                cast(Mapping[str, object], profiles[metric_id]["incident"]).get("p95"),
                bool,
            )
            and float(
                cast(
                    float,
                    cast(Mapping[str, object], profiles[metric_id]["incident"])["p95"],
                )
            )
            >= 80.0
            and cast(float, profiles[metric_id].get("relative_mean_shift", 0.0)) >= 0.2
        )
        if state_transition_metrics:
            transition_concepts = {
                concept
                for metric_id in state_transition_metrics
                for concept in self._semantic_concepts(metric_id)
            }
            if "availability" in transition_concepts:
                mechanism_signature = "availability_state_transition"
                root_cause_score += 0.9
            elif "connection" in transition_concepts:
                mechanism_signature = "connection_state_transition"
                root_cause_score += 0.75
            else:
                mechanism_signature = "state_transition"
                root_cause_score += 0.75
        elif any(
            profiles[metric_id].get("change_shape") == "incident_event_burst"
            for metric_id in anchor_metric_ids
        ):
            mechanism_signature = "incident_event_burst"
            root_cause_score += 0.5
        elif saturation_metrics:
            mechanism_signature = "sustained_high_saturation"
            root_cause_score += 0.75
        elif set(resource_burst_metric_ids) & set(anchor_metric_ids):
            mechanism_signature = "incident_resource_burst"
            root_cause_score += 0.5
        if fault_type is not None and "retransmission" in self._semantic_concepts(
            fault_type
        ):
            transmit_packets = next(
                (
                    metric_id
                    for metric_id in anchor_metric_ids
                    if "transmit_packets" in metric_id.casefold()
                ),
                None,
            )
            transmit_volume = next(
                (
                    metric_id
                    for metric_id in metric_ids
                    if "transmit_mb" in metric_id.casefold()
                ),
                None,
            )
            if (
                transmit_packets is not None
                and transmit_volume is not None
                and profiles[transmit_volume]["strong_incident_change"] is False
            ):
                packet = profiles[transmit_packets]
                volume = profiles[transmit_volume]
                packet_reference = cast(Mapping[str, Any], packet["reference"])["p95"]
                packet_incident = cast(Mapping[str, Any], packet["incident"])["p95"]
                volume_reference = cast(Mapping[str, Any], volume["reference"])["p95"]
                volume_incident = cast(Mapping[str, Any], volume["incident"])["p95"]
                if all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in (
                        packet_reference,
                        packet_incident,
                        volume_reference,
                        volume_incident,
                    )
                ):
                    packet_ratio = float(packet_incident) / max(
                        float(packet_reference), 1e-12
                    )
                    volume_ratio = float(volume_incident) / max(
                        float(volume_reference), 1e-12
                    )
                    if (
                        packet_ratio >= 1.3
                        and packet_ratio / max(volume_ratio, 0.01) >= 1.3
                    ):
                        mechanism_signature = "packet_count_without_payload_volume"
                        root_cause_score += 0.35
        if (
            fault_type is not None
            and "space" in self._semantic_concepts(fault_type)
            and len(anchor_metric_ids) >= 2
            and any(
                profiles[metric_id].get("change_shape") == "capacity_change"
                for metric_id in anchor_metric_ids
            )
        ):
            mechanism_signature = "capacity_change_rate"
            root_cause_score += 0.75
        anomaly_signature = None
        if (
            mechanism_signature is None
            and len(anchor_metric_ids) == 1
            and anchor_metric_ids[0] in strong_metric_ids
            and anchor_metric_ids[0] in isolated_metric_ids
        ):
            anomaly_signature = "single_metric_peer_isolated"
        fault_type_ready = bool(
            fault_type is not None
            and anchor_metric_ids
            and (len(anchor_metric_ids) >= 2 or mechanism_signature is not None)
        )
        material_effect_metric_ids = sorted(
            metric_id
            for metric_id in anchor_metric_ids
            if isinstance(profiles[metric_id].get("relative_p95_shift"), (int, float))
            and not isinstance(profiles[metric_id].get("relative_p95_shift"), bool)
            and float(profiles[metric_id]["relative_p95_shift"]) >= 0.2
        )
        trace_error_metric_ids = sorted(
            metric_id
            for metric_id in qualifying_metric_ids
            if metric_id in {"trace_incoming_error_count", "trace_incoming_error_rate"}
            and metric_id in isolated_metric_ids
        )
        peer_isolated_trace_error_event = {
            "trace_incoming_error_count",
            "trace_incoming_error_rate",
        } <= set(trace_error_metric_ids)
        queue_congestion_metric_ids = sorted(
            metric_id
            for metric_id in strong_metric_ids
            if "queue" in metric_id.casefold()
            and profiles[metric_id].get("finding_sample_ready") is True
            and _pressure_direction_aligned(profiles[metric_id])
        )
        peer_isolated_queue_metric_ids = sorted(
            set(queue_congestion_metric_ids) & set(isolated_metric_ids)
        )
        peer_isolated_queue_congestion_event = bool(
            fault_type is not None
            and "latency" in self._semantic_concepts(fault_type)
            and len(queue_congestion_metric_ids) >= 2
            and peer_isolated_queue_metric_ids
            and not boundary_aligned
        )
        if peer_isolated_trace_error_event and mechanism_signature is None:
            mechanism_signature = "peer_isolated_trace_error_event"
            root_cause_score += 0.5
        elif peer_isolated_queue_congestion_event and mechanism_signature is None:
            mechanism_signature = "peer_isolated_queue_congestion_event"
            root_cause_score += 0.4
        material_event_ready = bool(
            fault_type_ready
            or peer_isolated_trace_error_event
            or peer_isolated_queue_congestion_event
        )
        signature_free_finding_ready = bool(
            fault_type_ready
            and mechanism_signature is None
            and material_effect_metric_ids
        )
        material_event_basis = (
            "typed_fault_ready_profile"
            if fault_type_ready
            else (
                "peer_isolated_trace_error_event"
                if peer_isolated_trace_error_event
                else (
                    "peer_isolated_queue_congestion_event"
                    if peer_isolated_queue_congestion_event
                    else None
                )
            )
        )
        if fault_type_ready:
            root_cause_score += 0.15
        if boundary_aligned:
            root_cause_score *= 0.1
        onset_metric_ids = anchor_metric_ids if fault_type_ready else strong_metric_ids
        finding_onset_candidates: list[tuple[float, float]] = []
        for metric_id in onset_metric_ids:
            raw_burst = profiles[metric_id].get("incident_burst")
            if not isinstance(raw_burst, Mapping):
                continue
            burst_values = cast(Mapping[str, object], raw_burst)
            longest = burst_values.get("longest_segment")
            resource_burst_onset = (
                cast(Mapping[str, object], longest).get("start")
                if mechanism_signature == "incident_resource_burst"
                and metric_id in resource_burst_metric_ids
                and isinstance(longest, Mapping)
                else None
            )
            raw_onset = (
                resource_burst_onset
                if isinstance(resource_burst_onset, (int, float))
                and not isinstance(resource_burst_onset, bool)
                else burst_values.get("onset_timestamp")
            )
            if not isinstance(raw_onset, (int, float)) or isinstance(raw_onset, bool):
                continue
            raw_lower = (
                float(raw_onset)
                - float(profiles[metric_id].get("sample_interval_seconds") or 0.0)
                if resource_burst_onset is not None
                else burst_values.get("onset_lower_bound_timestamp")
            )
            lower = (
                float(raw_lower)
                if isinstance(raw_lower, (int, float))
                and not isinstance(raw_lower, bool)
                else float(raw_onset)
            )
            finding_onset_candidates.append((float(raw_onset), lower))
        finding_onset = (
            min(finding_onset_candidates, key=lambda item: item[0])
            if finding_onset_candidates
            else None
        )
        finding_gate_summary = {
            "strong_metric_ids": strong_metric_ids,
            "insufficient_sample_metric_ids": sorted(
                metric_id
                for metric_id, profile in profiles.items()
                if profile.get("finding_sample_ready") is not True
            ),
            "sampling_coverage_ready": any(
                profile.get("finding_sample_ready") is True
                for profile in profiles.values()
            ),
            "peer_isolated_metric_ids": isolated_metric_ids,
            "qualifying_metric_ids": qualifying_metric_ids,
            "strong_metric_count": len(strong_metric_ids),
            "qualifying_metric_count": len(qualifying_metric_ids),
            "earliest_strong_onset_timestamp": (
                finding_onset[0] if finding_onset is not None else None
            ),
            "earliest_strong_onset_lower_bound_timestamp": (
                finding_onset[1] if finding_onset is not None else None
            ),
            "boundary_aligned": boundary_aligned,
            "root_cause_score": round(root_cause_score, 6),
            "requested_fault_type": fault_type,
            "fault_type_anchor_metric_ids": anchor_metric_ids,
            "direction_aligned_metric_ids": sorted(
                metric_id
                for metric_id, profile in profiles.items()
                if _pressure_direction_aligned(profile)
            ),
            "fault_type_ready": fault_type_ready,
            "material_event_ready": material_event_ready,
            "material_event_basis": material_event_basis,
            "material_event_metric_ids": (
                trace_error_metric_ids
                if peer_isolated_trace_error_event
                else queue_congestion_metric_ids
                if peer_isolated_queue_congestion_event
                else []
            ),
            "mechanism_signature": mechanism_signature,
            "signature_free_finding_ready": signature_free_finding_ready,
            "material_effect_metric_ids": material_effect_metric_ids,
            "anomaly_signature": anomaly_signature,
            "single_component_ready": (
                fault_type_ready
                if fault_type is not None
                else len(qualifying_metric_ids) >= 2
            ),
            "trace_required": False,
        }
        if evidence_role == "impact_only":
            finding_gate_summary.update(
                {
                    "evidence_role": "impact_only",
                    "finding_eligible": False,
                    "finding_ineligible_reason": (
                        "impact-only evidence cannot identify a root cause"
                    ),
                    "strong_metric_ids": [],
                    "peer_isolated_metric_ids": [],
                    "qualifying_metric_ids": [],
                    "strong_metric_count": 0,
                    "qualifying_metric_count": 0,
                    "earliest_strong_onset_timestamp": None,
                    "earliest_strong_onset_lower_bound_timestamp": None,
                    "root_cause_score": 0.0,
                    "fault_type_anchor_metric_ids": [],
                    "direction_aligned_metric_ids": [],
                    "fault_type_ready": False,
                    "material_event_ready": False,
                    "material_event_basis": None,
                    "material_event_metric_ids": [],
                    "mechanism_signature": None,
                    "signature_free_finding_ready": False,
                    "material_effect_metric_ids": [],
                    "anomaly_signature": None,
                    "single_component_ready": False,
                }
            )
        return {
            "status": (
                "success"
                if statuses and statuses <= {"success"}
                else "partial"
                if statuses & {"success", "partial"}
                else "no_data"
            ),
            "entity_id": entity_id,
            "evidence_role": evidence_role,
            "metric_profiles": list(profiles.values()),
            "derived_metric_profiles": derived_profiles,
            "peer_comparison": peer_comparison,
            "peer_entity_ids": peers,
            "requested_fault_type": fault_type,
            "finding_gate_summary": finding_gate_summary,
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis=(
                    "paired raw metric scans with deterministic shape, burst, and "
                    "registered ratio extraction; source collection health unknown"
                ),
            ),
            "artifact_refs": list(artifact_refs.values()),
        }

    def query_service_profile(
        self,
        service_id: str,
        fault_type: str,
        *,
        member_entity_ids: list[str] | None = None,
    ) -> Result:
        """Aggregate typed pod profiles and trace edges for one service."""
        try:
            service = self._entity(service_id)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        if service.entity_type != "service":
            return self._error(
                "invalid_arguments", "service_id must identify a canonical service"
            )
        if fault_type not in self._public_fault_types:
            return self._error(
                "invalid_arguments",
                "fault_type must be an exact public vocabulary value",
            )
        service_name = market_service_component(
            service.public_candidate_match or service.original_id
        )
        available_members = sorted(
            node.canonical_id
            for node in self.topology.nodes
            if node.entity_type == "pod"
            and market_service_component(
                node.public_candidate_match or node.original_id
            )
            == service_name
        )
        members = member_entity_ids or available_members
        if (
            not members
            or len(members) != len(set(members))
            or any(member not in available_members for member in members)
        ):
            return self._error(
                "invalid_arguments",
                "member_entity_ids must be unique pod members of service_id",
            )
        common_metrics = set(self._catalog_metrics_by_entity.get(members[0], []))
        for member in members[1:]:
            common_metrics &= set(self._catalog_metrics_by_entity.get(member, []))
        anchor_metrics = sorted(
            metric_id
            for metric_id in common_metrics
            if self._fault_metric_compatible(fault_type, metric_id)
            and not metric_id.startswith("log::")
        )
        if not anchor_metrics:
            return self._no_data(
                self.task.allowed_query_range,
                f"no common fault-compatible metric is registered for {service_id}",
                basis="service membership and metric catalogs",
            )
        resource_family = self._resource_family(anchor_metrics[0])
        support_metrics = sorted(
            (
                metric_id
                for metric_id in common_metrics
                if metric_id not in anchor_metrics
                and not metric_id.startswith("log::")
                and self._resource_family(metric_id) == resource_family
            ),
            key=self._direct_metric_rank,
        )
        metric_ids = [*anchor_metrics, *support_metrics][:16]
        fault_concepts = self._semantic_concepts(fault_type)
        needs_trace = bool(fault_concepts & {"latency", "loss", "corruption"})
        if needs_trace and not self._service_profile_seed_index:
            self.summarize_anomalies(limit=1)
        seed_metadata = self._service_profile_seed_index.get(
            (service_id, fault_type), {}
        )
        consensus_member_count = seed_metadata.get("consensus_member_count", 0)
        trace_seed_corroborated = bool(
            isinstance(consensus_member_count, int)
            and not isinstance(consensus_member_count, bool)
            and consensus_member_count * 4 >= len(members) * 3
        )
        trace_result: Result = (
            self.query_trace_edges(service_id, direction="incoming", limit=100)
            if needs_trace
            else {"status": "no_data", "edge_profiles": [], "artifact_refs": []}
        )
        trace_mechanism_edges: list[str] = []
        for raw_edge in cast(list[object], trace_result.get("edge_profiles", [])):
            if not isinstance(raw_edge, Mapping):
                continue
            edge = cast(Mapping[str, Any], raw_edge)
            reference = edge.get("reference")
            incident = edge.get("incident")
            if not isinstance(reference, Mapping) or not isinstance(incident, Mapping):
                continue
            reference_values = cast(Mapping[str, Any], reference)
            incident_values = cast(Mapping[str, Any], incident)
            reference_calls = int(reference_values.get("call_count") or 0)
            incident_calls = int(incident_values.get("call_count") or 0)
            error_delta = float(incident_values.get("error_rate") or 0.0) - float(
                reference_values.get("error_rate") or 0.0
            )
            reference_p95 = float(reference_values.get("latency_p95") or 0.0)
            incident_p95 = float(incident_values.get("latency_p95") or 0.0)
            latency_ratio = incident_p95 / max(reference_p95, 1e-12)
            error_mechanism = bool(
                fault_concepts & {"loss", "corruption"}
                and reference_calls >= 5
                and incident_calls >= 5
                and error_delta >= 0.01
            )
            latency_mechanism = bool(
                "latency" in fault_concepts
                and reference_calls >= 5
                and incident_calls >= 5
                and reference_p95 > 0
                and latency_ratio >= 1.5
            )
            if error_mechanism or latency_mechanism:
                edge_id = edge.get("edge_id")
                if isinstance(edge_id, str):
                    trace_mechanism_edges.append(edge_id)

        member_profiles: list[dict[str, Any]] = []
        source_artifacts: dict[str, Result] = {}
        statuses: set[str] = set()
        trace_dominant_profile = bool(
            needs_trace and "retransmission" not in fault_concepts
        )
        if trace_dominant_profile:
            statuses.add(str(trace_result.get("status", "error")))
        else:
            for member in members:
                result = self.query_metric_profile(
                    member,
                    metric_ids,
                    peer_entity_ids=[peer for peer in members if peer != member],
                    fault_type=fault_type,
                )
                statuses.add(str(result.get("status", "error")))
                for raw_ref in cast(list[object], result.get("artifact_refs", [])):
                    if isinstance(raw_ref, Mapping):
                        ref = dict(cast(Mapping[str, Any], raw_ref))
                        source_artifacts[json.dumps(ref, sort_keys=True)] = ref
                member_profiles.append(
                    {
                        "entity_id": member,
                        "status": result.get("status"),
                        "finding_gate_summary": result.get("finding_gate_summary"),
                        "metric_profiles": result.get("metric_profiles", []),
                    }
                )
        for raw_ref in cast(list[object], trace_result.get("artifact_refs", [])):
            if isinstance(raw_ref, Mapping):
                ref = dict(cast(Mapping[str, Any], raw_ref))
                source_artifacts[json.dumps(ref, sort_keys=True)] = ref

        member_trace_scores: dict[str, float] = {}
        for raw_edge in cast(list[object], trace_result.get("edge_profiles", [])):
            if not isinstance(raw_edge, Mapping):
                continue
            edge = cast(Mapping[str, Any], raw_edge)
            incident = edge.get("incident")
            if not isinstance(incident, Mapping):
                continue
            incident_values = cast(Mapping[str, Any], incident)
            if "latency" in fault_concepts:
                raw_score = incident_values.get("latency_p95")
            else:
                raw_score = incident_values.get("error_rate")
            if not isinstance(raw_score, (int, float)) or isinstance(raw_score, bool):
                continue
            endpoint = edge.get("target_id")
            if isinstance(endpoint, str) and endpoint in members:
                member_trace_scores[endpoint] = max(
                    float(raw_score), member_trace_scores.get(endpoint, 0.0)
                )
        if not trace_mechanism_edges:
            member_trace_scores.clear()
        attributed_member = peer_isolated_trace_member(
            member_trace_scores, latency="latency" in fault_concepts
        )
        attributed_member_entity_ids = (
            [attributed_member] if attributed_member is not None else []
        )

        ready_members = [
            item
            for item in member_profiles
            if isinstance(item.get("finding_gate_summary"), Mapping)
            and cast(Mapping[str, object], item["finding_gate_summary"]).get(
                "fault_type_ready"
            )
            is True
        ]
        mechanism_members: list[str] = []
        if "retransmission" in fault_concepts:
            for item in member_profiles:
                profiles = {
                    str(profile.get("metric_id")): profile
                    for raw_profile in cast(list[object], item["metric_profiles"])
                    if isinstance(raw_profile, Mapping)
                    for profile in [cast(Mapping[str, Any], raw_profile)]
                }
                packet = next(
                    (
                        profile
                        for metric_id, profile in profiles.items()
                        if "network_transmit_packets" in metric_id.casefold()
                    ),
                    None,
                )
                volume = next(
                    (
                        profile
                        for metric_id, profile in profiles.items()
                        if "network_transmit_mb" in metric_id.casefold()
                    ),
                    None,
                )
                if packet is None or volume is None:
                    continue
                packet_reference = cast(Mapping[str, Any], packet["reference"])["p95"]
                packet_incident = cast(Mapping[str, Any], packet["incident"])["p95"]
                volume_reference = cast(Mapping[str, Any], volume["reference"])["p95"]
                volume_incident = cast(Mapping[str, Any], volume["incident"])["p95"]
                if not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in (
                        packet_reference,
                        packet_incident,
                        volume_reference,
                        volume_incident,
                    )
                ):
                    continue
                packet_ratio = float(packet_incident) / max(
                    float(packet_reference), 1e-12
                )
                volume_ratio = float(volume_incident) / max(
                    float(volume_reference), 1e-12
                )
                if (
                    packet_ratio >= 1.3
                    and packet_ratio / max(volume_ratio, 0.01) >= 1.3
                ):
                    mechanism_members.append(cast(str, item["entity_id"]))

        consensus_ready = len(ready_members) >= 2
        mechanism_signature: str | None = None
        consensus_signature = "multi_replica_consensus" if consensus_ready else None
        if len(mechanism_members) >= 3:
            mechanism_signature = "service_packet_count_without_payload_volume"
        elif trace_mechanism_edges and trace_seed_corroborated:
            mechanism_signature = "service_trace_mechanism"
        fault_type_ready = bool(consensus_ready or mechanism_signature is not None)
        member_scores = [
            float(score)
            for item in ready_members
            for summary in [cast(Mapping[str, object], item["finding_gate_summary"])]
            for score in [summary.get("root_cause_score")]
            if isinstance(score, (int, float)) and not isinstance(score, bool)
        ]
        consensus_bonus = (
            0.75 if mechanism_signature is not None else 0.5 if consensus_ready else 0.0
        )
        root_cause_score = (
            statistics.median(member_scores) if member_scores else 0.0
        ) + consensus_bonus
        member_onsets = [
            float(onset)
            for item in ready_members
            for summary in [cast(Mapping[str, object], item["finding_gate_summary"])]
            for onset in [summary.get("earliest_strong_onset_timestamp")]
            if isinstance(onset, (int, float)) and not isinstance(onset, bool)
        ]
        return {
            "status": (
                "success"
                if statuses and statuses <= {"success"}
                else "partial"
                if statuses & {"success", "partial"}
                else "no_data"
            ),
            "service_id": service_id,
            "member_entity_ids": members,
            "attributed_member_entity_ids": attributed_member_entity_ids,
            "attribution_basis": (
                "peer-isolated member trace edge"
                if attributed_member_entity_ids
                else "service-level trace or multi-replica consensus"
            ),
            "member_profiles": member_profiles,
            "trace_edge_profiles": trace_result.get("edge_profiles", []),
            "finding_gate_summary": {
                "requested_fault_type": fault_type,
                "fault_type_ready": fault_type_ready,
                "single_component_ready": fault_type_ready,
                "root_cause_score": round(root_cause_score, 6),
                "mechanism_signature": mechanism_signature,
                "consensus_signature": consensus_signature,
                "signature_free_finding_ready": bool(
                    consensus_ready and mechanism_signature is None
                ),
                "material_effect_metric_ids": (
                    anchor_metrics if consensus_ready else []
                ),
                "seed_consensus_member_count": consensus_member_count,
                "trace_seed_corroborated": trace_seed_corroborated,
                "boundary_aligned": False,
                "strong_metric_count": sum(
                    count
                    for item in ready_members
                    for count in [
                        cast(Mapping[str, object], item["finding_gate_summary"]).get(
                            "strong_metric_count", 0
                        )
                    ]
                    if isinstance(count, int) and not isinstance(count, bool)
                ),
                "qualifying_metric_count": sum(
                    count
                    for item in ready_members
                    for count in [
                        cast(Mapping[str, object], item["finding_gate_summary"]).get(
                            "qualifying_metric_count", 0
                        )
                    ]
                    if isinstance(count, int) and not isinstance(count, bool)
                ),
                "fault_type_anchor_metric_ids": anchor_metrics,
                "qualifying_metric_ids": anchor_metrics,
                "strong_metric_ids": anchor_metrics if fault_type_ready else [],
                "ready_member_count": len(ready_members),
                "mechanism_member_count": len(mechanism_members),
                "trace_mechanism_edge_ids": trace_mechanism_edges,
                "earliest_strong_onset_timestamp": (
                    min(member_onsets) if member_onsets else None
                ),
                "trace_required": False,
            },
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis=(
                    "paired member metric profiles plus service-level trace-edge "
                    "aggregation; upstream collection health unknown"
                ),
            ),
            "artifact_refs": list(source_artifacts.values()),
        }

    def _default_peer_entities(
        self, entity_id: str, metric_ids: Sequence[str]
    ) -> list[str]:
        target = self._entities[entity_id]
        if self.profile is _MARKET_PROFILE and target.entity_type == "pod":
            target_group = market_service_component(
                target.public_candidate_match or target.original_id
            )
            candidates = [
                node.canonical_id
                for node in self.topology.nodes
                if node.canonical_id != entity_id
                and node.entity_type == "pod"
                and market_service_component(
                    node.public_candidate_match or node.original_id
                )
                == target_group
            ]
        else:
            candidates = [
                node.canonical_id
                for node in self.topology.nodes
                if node.canonical_id != entity_id
                and node.entity_type == target.entity_type
            ]
        return [
            candidate
            for candidate in candidates
            if all(
                metric_id in self._catalog_metrics_by_entity.get(candidate, [])
                for metric_id in metric_ids
            )
        ][:32]

    @classmethod
    def _paired_stats_shift(cls, stats: Mapping[str, Any]) -> float | None:
        reference = stats.get("reference")
        incident = stats.get("incident")
        if not isinstance(reference, Mapping) or not isinstance(incident, Mapping):
            return None
        reference_values = cast(Mapping[str, object], reference)
        incident_values = cast(Mapping[str, object], incident)
        values = [
            reference_values.get("median"),
            reference_values.get("mean"),
            reference_values.get("max"),
            incident_values.get("median"),
            incident_values.get("mean"),
            incident_values.get("max"),
        ]
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            return None
        (
            reference_median,
            reference_mean,
            reference_max,
            incident_median,
            incident_mean,
            incident_max,
        ) = cast(list[float], values)
        median_shift = abs(incident_median - reference_median) / max(
            abs(incident_median), abs(reference_median), 1.0
        )
        extreme_shift = abs(incident_max - reference_max) / max(
            abs(incident_max), abs(reference_max), 1.0
        )
        mean_shift = abs(incident_mean - reference_mean) / max(
            abs(incident_mean), abs(reference_mean), 1.0
        )
        p95_shift = 0.0
        if (
            isinstance(reference_values.get("p95"), (int, float))
            and not isinstance(reference_values.get("p95"), bool)
            and isinstance(incident_values.get("p95"), (int, float))
            and not isinstance(incident_values.get("p95"), bool)
        ):
            reference_p95 = float(cast(float, reference_values["p95"]))
            incident_p95 = float(cast(float, incident_values["p95"]))
            p95_shift = abs(incident_p95 - reference_p95) / max(
                abs(incident_p95), abs(reference_p95), 1.0
            )
        return max(median_shift, mean_shift, p95_shift, extreme_shift)

    def query_log_samples(
        self,
        entity_id: str,
        *,
        period: str = "incident",
        start: str | datetime | None = None,
        end: str | datetime | None = None,
        log_name: str | None = None,
        contains: str | None = None,
        limit: int = 20,
    ) -> Result:
        try:
            entity = self._entity(entity_id)
            period_name, window = self._window(period, start, end)
            self._bounded_limit(limit)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        if log_name is not None and log_name not in self._log_names:
            return self._error(
                "invalid_log_name", f"unknown log_name: {log_name}", window
            )

        sources = [
            source for source in self.profile.log_sources if source in entity.sources
        ]
        if not sources:
            return self._no_data(
                window,
                f"no log source is registered for {entity_id}",
                basis="entity inventory; collection completeness is unknown",
            )

        ranked_samples: list[tuple[tuple[float, str, int], dict[str, Any]]] = []
        match_count = 0
        paired_counts: dict[str, Counter[str]] = {
            "reference": Counter(),
            "incident": Counter(),
        }
        task_windows = {
            "reference": cast(TimeRange, self.task.reference_range),
            "incident": self.task.incident_range,
        }
        try:
            for source in sources:
                relative = f"telemetry/{source}"
                source_entities = set(
                    entity.source_aliases.get(source, [entity.original_id])
                )
                with self._artifact_path(relative).open(
                    newline="", encoding="utf-8"
                ) as stream:
                    for row in csv.DictReader(stream):
                        self._validate_row_source_member(row, relative)
                        if row["cmdb_id"] not in source_entities:
                            continue
                        if log_name is not None and row["log_name"] != log_name:
                            continue
                        text = row["value"] or None
                        if contains is not None and (
                            text is None or contains not in text
                        ):
                            continue
                        timestamp = float(row["timestamp"])
                        for task_period, task_window in task_windows.items():
                            if self._contains(task_window, timestamp):
                                paired_counts[task_period][row["log_name"]] += 1
                        if not self._contains(window, timestamp):
                            continue
                        record = int(row["_source_record"])
                        sample: dict[str, Any] = {
                            "log_id": row["log_id"],
                            "timestamp": self._source_timestamp(timestamp),
                            "timestamp_unit": "seconds",
                            "log_name": row["log_name"],
                            "text": text,
                            "source_path": source,
                            "original_entity_key": row["cmdb_id"],
                            "source_record": record,
                        }
                        match_count += 1
                        insort(
                            ranked_samples,
                            ((timestamp, source, record), sample),
                        )
                        if len(ranked_samples) > limit:
                            ranked_samples.pop()
        except (OSError, csv.Error, KeyError, ValueError) as exc:
            return self._error("source_read_error", str(exc), window)

        truncated = match_count > limit
        paired_log_summary = {
            "counts_complete": True,
            "samples_truncated": truncated,
            **{
                task_period: {
                    "matched_count": sum(counts.values()),
                    "log_name_counts": dict(sorted(counts.items())),
                }
                for task_period, counts in paired_counts.items()
            },
        }
        if not ranked_samples:
            result = self._no_data(
                window,
                "no matching log samples in the requested range",
                basis="source collection completeness and health are unknown",
            )
            result["paired_log_summary"] = paired_log_summary
            return result
        samples = [sample for _key, sample in ranked_samples[:limit]]
        selected_records: dict[str, list[int]] = {}
        for sample in samples:
            selected_records.setdefault(str(sample["source_path"]), []).append(
                int(sample["source_record"])
            )
        timestamps = [float(sample["timestamp"]) for sample in samples]
        sample_interval, observed_range = self._observed(timestamps, window)
        return {
            "status": "partial" if truncated else "success",
            "entity_id": entity_id,
            "period": period_name,
            "log_name": log_name,
            "contains": contains,
            "samples": samples,
            "paired_log_summary": paired_log_summary,
            "coverage": self._coverage(
                window,
                completeness="unknown",
                basis="bounded raw log samples; upstream collection health unknown",
                observed_range=observed_range,
                sample_interval=sample_interval,
                truncation=(
                    f"limited to first {limit} matching rows" if truncated else None
                ),
            ),
            "artifact_refs": [
                self._artifact_ref(
                    f"telemetry/{source}",
                    {
                        "log_ids": [
                            sample["log_id"]
                            for sample in samples
                            if sample["source_path"] == source
                        ],
                        "source_records": selected_records[source],
                    },
                )
                for source in sorted(selected_records)
            ],
        }

    def summarize_log_patterns(
        self,
        entity_id: str,
        *,
        log_name: str | None = None,
        limit: int = 20,
    ) -> Result:
        """Discover stable log templates before exact sample drill-down."""
        try:
            entity = self._entity(entity_id)
            self._bounded_limit(limit)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        if log_name is not None and log_name not in self._log_names:
            return self._error("invalid_log_name", f"unknown log_name: {log_name}")
        sources = [
            source for source in self.profile.log_sources if source in entity.sources
        ]
        if not sources:
            return self._no_data(
                self.task.allowed_query_range,
                f"no log source is registered for {entity_id}",
                basis="entity inventory; collection completeness is unknown",
            )
        patterns: dict[str, dict[str, Any]] = {}
        task_windows = {
            "reference": cast(TimeRange, self.task.reference_range),
            "incident": self.task.incident_range,
        }
        selected_records: dict[str, set[int]] = {}
        try:
            for source in sources:
                relative = f"telemetry/{source}"
                source_entities = set(
                    entity.source_aliases.get(source, [entity.original_id])
                )
                with self._artifact_path(relative).open(
                    newline="", encoding="utf-8"
                ) as stream:
                    for row in csv.DictReader(stream):
                        self._validate_row_source_member(row, relative)
                        if row["cmdb_id"] not in source_entities:
                            continue
                        if log_name is not None and row["log_name"] != log_name:
                            continue
                        text = row["value"] or ""
                        template = _normalize_log_template(text)
                        if not template:
                            continue
                        timestamp = float(row["timestamp"])
                        period = next(
                            (
                                name
                                for name, window in task_windows.items()
                                if self._contains(window, timestamp)
                            ),
                            None,
                        )
                        if period is None:
                            continue
                        template_id = hashlib.sha256(template.encode()).hexdigest()
                        pattern = patterns.setdefault(
                            template_id,
                            {
                                "template_id": template_id,
                                "template": template,
                                "severity": _log_severity(text),
                                "reference_count": 0,
                                "incident_count": 0,
                                "log_name_counts": Counter(),
                                "incident_buckets": Counter(),
                                "first_seen": None,
                                "examples": {"reference": [], "incident": []},
                            },
                        )
                        count_key = f"{period}_count"
                        pattern[count_key] = cast(int, pattern[count_key]) + 1
                        cast(Counter[str], pattern["log_name_counts"])[
                            row["log_name"]
                        ] += 1
                        if period == "incident":
                            bucket = int(timestamp // 60 * 60)
                            cast(Counter[int], pattern["incident_buckets"])[bucket] += 1
                            first_seen = pattern["first_seen"]
                            pattern["first_seen"] = (
                                timestamp
                                if first_seen is None
                                else min(float(cast(float, first_seen)), timestamp)
                            )
                        examples = cast(
                            dict[str, list[dict[str, Any]]], pattern["examples"]
                        )[period]
                        sample = {
                            "log_id": row["log_id"],
                            "timestamp": self._source_timestamp(timestamp),
                            "log_name": row["log_name"],
                            "text": text,
                            "source_path": source,
                            "source_record": int(row["_source_record"]),
                        }
                        if len(examples) < 2:
                            examples.append(sample)
                        else:
                            examples[-1] = sample
        except (OSError, csv.Error, KeyError, ValueError) as exc:
            return self._error("source_read_error", str(exc))

        ranked: list[dict[str, Any]] = []
        severity_rank = {"critical": 0, "error": 1, "warning": 2, "info": 3}
        for pattern in patterns.values():
            reference_count = cast(int, pattern["reference_count"])
            incident_count = cast(int, pattern["incident_count"])
            buckets = cast(Counter[int], pattern.pop("incident_buckets"))
            peak_bucket = buckets.most_common(1)[0] if buckets else None
            examples_by_period = cast(
                dict[str, list[dict[str, Any]]], pattern["examples"]
            )
            for examples in examples_by_period.values():
                for example in examples:
                    selected_records.setdefault(
                        cast(str, example["source_path"]), set()
                    ).add(cast(int, example["source_record"]))
            pattern["log_name_counts"] = dict(
                sorted(cast(Counter[str], pattern["log_name_counts"]).items())
            )
            pattern["incident_excess"] = max(0, incident_count - reference_count)
            pattern["new_in_incident"] = incident_count > 0 and reference_count == 0
            pattern["peak_bucket"] = (
                {"start": peak_bucket[0], "count": peak_bucket[1]}
                if peak_bucket
                else None
            )
            ranked.append(pattern)
        ranked.sort(
            key=lambda item: (
                -int(cast(bool, item["new_in_incident"])),
                severity_rank.get(cast(str, item["severity"]), 4),
                -cast(int, item["incident_excess"]),
                -cast(int, item["incident_count"]),
                cast(str, item["template_id"]),
            )
        )
        returned = ranked[:limit]
        return {
            "status": "partial" if len(ranked) > limit else "success",
            "entity_id": entity_id,
            "log_name": log_name,
            "patterns": returned,
            "pattern_count": len(ranked),
            "omitted_pattern_count": max(0, len(ranked) - len(returned)),
            "template_version": "rootwalk-log-template-v1",
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis=(
                    "complete paired-window scan for registered entity aliases; "
                    "upstream collection health unknown"
                ),
                truncation=(
                    f"limited to top {limit} templates" if len(ranked) > limit else None
                ),
            ),
            "artifact_refs": [
                self._artifact_ref(
                    f"telemetry/{source}",
                    {"source_records": sorted(records)},
                )
                for source, records in sorted(selected_records.items())
            ],
        }

    @staticmethod
    def _trace_operation_points(
        samples: Sequence[tuple[float, float, int]], window: TimeRange
    ) -> list[tuple[float, float, float | None]]:
        start = window.start.timestamp()
        duration = window.end.timestamp() - start
        if duration <= 0:
            return []
        bucket_count = 30
        bucket_width = max(1.0, duration / bucket_count)
        buckets: list[list[float]] = [[] for _index in range(bucket_count)]
        for timestamp, latency, _record in samples:
            index = min(int((timestamp - start) / bucket_width), bucket_count - 1)
            if index >= 0:
                buckets[index].append(latency)
        return [
            (
                start + index * bucket_width,
                float(len(latencies)),
                _percentile(latencies, 0.95),
            )
            for index, latencies in enumerate(buckets)
        ]

    def query_trace_operation_profile(self, entity_id: str, limit: int = 20) -> Result:
        """Compare named trace operations without assigning causal meaning."""
        try:
            entity = self._entity(entity_id)
            self._bounded_limit(limit)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        trace_source = "trace/trace_span.csv"
        trace_path = f"telemetry/{trace_source}"
        if trace_path not in self._artifacts_by_path:
            return {
                "status": "no_data",
                "entity_id": entity_id,
                "operations": [],
                "reason": "no raw trace artifact is registered",
                "coverage": self._coverage(
                    self.task.allowed_query_range,
                    completeness="unknown",
                    basis="adapter artifact inventory",
                ),
                "artifact_refs": [],
                "gap": self._gap(),
            }
        if trace_source not in entity.sources:
            return {
                "status": "no_data",
                "entity_id": entity_id,
                "operations": [],
                "reason": "the entity has no source-backed trace identity",
                "coverage": self._coverage(
                    self.task.allowed_query_range,
                    completeness="unknown",
                    basis="topology source inventory",
                ),
                "artifact_refs": [
                    self._artifact_ref("topology.json", {"entity_id": entity_id})
                ],
                "gap": self._gap(),
            }
        aliases = set(entity.source_aliases.get(trace_source, [entity.original_id]))
        periods = {
            "reference": cast(TimeRange, self.task.reference_range),
            "incident": self.task.incident_range,
        }
        grouped: dict[
            tuple[str, str | None],
            dict[str, list[tuple[float, float, int]]],
        ] = {}
        try:
            with self._artifact_path(trace_path).open(
                newline="", encoding="utf-8"
            ) as stream:
                reader = csv.DictReader(stream)
                fields = set(reader.fieldnames or [])
                required = {
                    "timestamp",
                    "cmdb_id",
                    "duration",
                    "operation_name",
                    "_source_record",
                }
                if not required <= fields:
                    missing = sorted(required - fields)
                    return {
                        "status": "no_data",
                        "entity_id": entity_id,
                        "operations": [],
                        "reason": (
                            "trace source cannot identify operations; missing "
                            + ", ".join(missing)
                        ),
                        "coverage": self._coverage(
                            self.task.allowed_query_range,
                            completeness="unknown",
                            basis="raw trace header",
                        ),
                        "artifact_refs": [
                            self._artifact_ref(trace_path, {"selector": "header"})
                        ],
                        "gap": self._gap(),
                    }
                for row in reader:
                    self._validate_row_source_member(row, trace_path)
                    if row["cmdb_id"] not in aliases:
                        continue
                    operation_name = row["operation_name"].strip()
                    if not operation_name:
                        continue
                    timestamp = float(row["timestamp"]) / 1000.0
                    period = next(
                        (
                            name
                            for name, window in periods.items()
                            if self._contains(window, timestamp)
                        ),
                        None,
                    )
                    if period is None:
                        continue
                    trace_type = (row.get("type") or "").strip() or None
                    grouped.setdefault(
                        (operation_name, trace_type),
                        {"reference": [], "incident": []},
                    )[period].append(
                        (
                            timestamp,
                            float(row["duration"]),
                            int(row["_source_record"]),
                        )
                    )
        except (OSError, csv.Error, KeyError, TypeError, ValueError) as exc:
            return self._error("source_read_error", str(exc))
        operations: list[dict[str, Any]] = []
        for (operation_name, trace_type), samples in grouped.items():
            reference_samples = samples["reference"]
            incident_samples = samples["incident"]
            if not reference_samples and not incident_samples:
                continue
            reference_latency = _percentile(
                [sample[1] for sample in reference_samples], 0.95
            )
            incident_latency = _percentile(
                [sample[1] for sample in incident_samples], 0.95
            )
            reference_latency_stats = _profile_latency_stats(
                [sample[1] for sample in reference_samples]
            )
            incident_latency_stats = _profile_latency_stats(
                [sample[1] for sample in incident_samples]
            )
            effect_size, effect_components = _profile_effect_size(
                float(len(reference_samples)),
                float(len(incident_samples)),
                reference_latency,
                incident_latency,
            )
            reference_points = self._trace_operation_points(
                reference_samples, periods["reference"]
            )
            incident_points = self._trace_operation_points(
                incident_samples, periods["incident"]
            )
            markers = _profile_onset_markers(
                reference_points,
                incident_points,
                count_is_harmful=False,
                minimum_consecutive_points=2,
                minimum_relative_change=0.2,
            )
            reference_records = [sample[2] for sample in reference_samples]
            incident_records = [sample[2] for sample in incident_samples]
            artifact_ref = self._artifact_ref(
                trace_path,
                {
                    "entity_id": entity_id,
                    "original_entity_keys": sorted(aliases),
                    "operation_name": operation_name,
                    "trace_type": trace_type,
                    "reference_source_records": reference_records[:10],
                    "incident_source_records": incident_records[:10],
                    "matched_source_record_count": len(reference_records)
                    + len(incident_records),
                },
            )
            reference_p99 = reference_latency_stats["p99"]
            competing_reference = bool(
                reference_p99 is not None
                and incident_latency is not None
                and float(reference_p99) >= incident_latency
            )
            if not reference_samples or not incident_samples:
                identifiability = "unidentifiable"
                identifiability_reason = "samples are missing from one task window"
            elif reference_latency is None or incident_latency is None:
                identifiability = "unidentifiable"
                identifiability_reason = "latency is unavailable in one task window"
            elif incident_latency <= reference_latency:
                identifiability = "unidentifiable"
                identifiability_reason = "incident latency shift is not positive"
            elif competing_reference:
                identifiability = "unidentifiable"
                identifiability_reason = (
                    "reference contains a comparable latency episode"
                )
            else:
                identifiability = "identifiable"
                identifiability_reason = (
                    "incident p95 exceeds the reference p99 envelope"
                )
            operations.append(
                {
                    "entity_id": entity_id,
                    "metric_name": "duration",
                    "operation_name": operation_name,
                    "trace_type": trace_type,
                    "evidence_role": "component_operation",
                    **markers,
                    "reference_count": len(reference_samples),
                    "incident_count": len(incident_samples),
                    "reference_latency": reference_latency,
                    "incident_latency": incident_latency,
                    "reference_latency_stats": reference_latency_stats,
                    "incident_latency_stats": incident_latency_stats,
                    "sample_interval_seconds": _profile_sample_interval(
                        reference_points, incident_points
                    ),
                    "sample_count": {
                        "reference": len(reference_samples),
                        "incident": len(incident_samples),
                    },
                    "expected_sample_count": None,
                    "expected_sample_count_reason": (
                        "trace traffic volume has no declared sampling contract"
                    ),
                    "latency_statistic": "p95",
                    "latency_unit": self._trace_duration_unit,
                    "effect_size": effect_size,
                    "effect_size_components": effect_components,
                    "effect_size_basis": "symmetric relative shift",
                    "identifiability": identifiability,
                    "identifiability_reason": identifiability_reason,
                    "competing_artifact_refs": (
                        [artifact_ref] if competing_reference else []
                    ),
                    "modal_lag_seconds": None,
                    "modal_lag_reason": (
                        "trace rows have no independent collection timestamp"
                    ),
                    "artifact_refs": [artifact_ref],
                }
            )
        operations.sort(
            key=lambda operation: (
                -abs(cast(float, operation["effect_size"])),
                cast(str, operation["operation_name"]),
                str(operation["trace_type"] or ""),
            )
        )
        if not operations:
            return {
                "status": "no_data",
                "entity_id": entity_id,
                "operations": [],
                "reason": (
                    "no named trace operation for the entity is present in the "
                    "reference or incident window"
                ),
                "coverage": self._coverage(
                    self.task.allowed_query_range,
                    completeness="unknown",
                    basis="source-backed entity aliases and raw trace rows",
                ),
                "artifact_refs": [
                    self._artifact_ref(
                        trace_path,
                        {
                            "entity_id": entity_id,
                            "original_entity_keys": sorted(aliases),
                        },
                    )
                ],
                "gap": self._gap(),
            }
        truncated = len(operations) > limit
        selected = operations[:limit]
        return {
            "status": "partial" if truncated else "success",
            "entity_id": entity_id,
            "operations": selected,
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis=(
                    "all matching named operation rows in the frozen trace slice; "
                    "upstream collection health unknown"
                ),
                truncation=(
                    f"limited to top {limit} operations by absolute effect size"
                    if truncated
                    else None
                ),
            ),
            "artifact_refs": [
                cast(list[dict[str, Any]], operation["artifact_refs"])[0]
                for operation in selected
            ],
        }

    def query_trace_neighbors(
        self,
        entity_id: str,
        *,
        direction: str = "both",
        limit: int = 20,
    ) -> Result:
        try:
            self._entity(entity_id)
            self._bounded_limit(limit)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        if direction not in {"incoming", "outgoing", "both"}:
            return self._error(
                "invalid_arguments", "direction must be incoming, outgoing, or both"
            )

        edges = [
            edge
            for edge in self.topology.edges
            if (direction in {"outgoing", "both"} and edge.source_id == entity_id)
            or (direction in {"incoming", "both"} and edge.target_id == entity_id)
        ]
        edges.sort(key=lambda edge: edge.edge_id)
        if not edges:
            result = self._no_data(
                self.topology.valid_range,
                "no source-backed neighbor edge is present in the partial topology",
                basis="topology.json is partial and no trace edge was synthesized",
                completeness="partial",
            )
            result["trace_duration_unit"] = self._trace_duration_unit
            result["trace_data_scope"] = "prepared_topology_only"
            return result
        truncated = len(edges) > limit
        selected = edges[:limit]
        topology_completeness = getattr(
            self.topology.coverage,
            "status",
            getattr(self.topology.coverage, "completeness", "unknown"),
        )
        return {
            "status": "partial"
            if topology_completeness != "complete" or truncated
            else "success",
            "entity_id": entity_id,
            "direction": direction,
            "edges": [edge.model_dump(mode="json") for edge in selected],
            "edge_counts_scope": (
                "full allowed query range; not recomputed from trace CSV"
            ),
            "trace_timestamp_unit": "milliseconds",
            "trace_duration_unit": self._trace_duration_unit,
            "trace_data_scope": "prepared_topology_only",
            "coverage": self._coverage(
                self.topology.valid_range,
                completeness=topology_completeness,
                basis=(
                    "source-backed topology.json edges; raw trace rows were not scanned"
                ),
                truncation=(
                    f"limited to first {limit} edges by edge_id" if truncated else None
                ),
            ),
            "artifact_refs": [
                self._artifact_ref(
                    "topology.json",
                    {"edge_ids": [edge.edge_id for edge in selected]},
                )
            ],
        }

    def query_trace_edges(
        self,
        entity_id: str,
        *,
        direction: str = "both",
        limit: int = 20,
    ) -> Result:
        """Compare incident and reference behavior on directed trace edges."""
        try:
            self._entity(entity_id)
            self._bounded_limit(limit)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        if direction not in {"incoming", "outgoing", "both"}:
            return self._error(
                "invalid_arguments", "direction must be incoming, outgoing, or both"
            )
        trace_path = "telemetry/trace/trace_span.csv"
        if trace_path not in self._artifacts_by_path:
            return self._no_data(
                self.task.allowed_query_range,
                "no raw trace source is registered for dynamic edge analysis",
                basis="adapter capability inventory",
            )
        scoped_entity_ids = {entity_id}
        entity = self._entities[entity_id]
        if entity.entity_type == "service":
            service_name = market_service_component(
                entity.public_candidate_match or entity.original_id
            )
            scoped_entity_ids.update(
                node.canonical_id
                for node in self.topology.nodes
                if node.entity_type == "pod"
                and market_service_component(
                    node.public_candidate_match or node.original_id
                )
                == service_name
            )
        allowed_edges = {
            (edge.source_id, edge.target_id): edge
            for edge in self.topology.edges
            if edge.relation_type == "calls"
            and (
                direction in {"outgoing", "both"}
                and edge.source_id in scoped_entity_ids
                or direction in {"incoming", "both"}
                and edge.target_id in scoped_entity_ids
            )
        }
        if not allowed_edges:
            return self._no_data(
                self.task.allowed_query_range,
                "no source-backed adjacent trace edge is present",
                basis="prepared topology coverage is partial",
                completeness="partial",
            )
        trace_stats_path = "trace_edge_window_stats.jsonl"
        if trace_stats_path in self._artifacts_by_path:
            profiles: list[dict[str, Any]] = []
            selected_lines: list[int] = []
            selected_records: set[int] = set()
            try:
                with self._artifact_path(trace_stats_path).open(
                    encoding="utf-8"
                ) as stream:
                    for line_number, line in enumerate(stream, 1):
                        raw: object = json.loads(line)
                        if not isinstance(raw, dict):
                            raise ValueError("trace edge profile row must be an object")
                        item = dict(cast(dict[str, Any], raw))
                        pair = (item.get("source_id"), item.get("target_id"))
                        if pair not in allowed_edges:
                            continue
                        expected_edge = allowed_edges[cast(tuple[str, str], pair)]
                        if item.get("edge_id") != expected_edge.edge_id:
                            raise ValueError(
                                "trace edge profile does not match topology edge"
                            )
                        raw_records = item.pop("source_records", [])
                        if not isinstance(raw_records, list) or not all(
                            isinstance(record, int) and not isinstance(record, bool)
                            for record in cast(list[object], raw_records)
                        ):
                            raise ValueError(
                                "trace edge profile source_records are invalid"
                            )
                        selected_records.update(cast(list[int], raw_records))
                        selected_lines.append(line_number)
                        profiles.append(item)
            except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
                return self._error("source_read_error", str(exc))
            profiles.sort(
                key=lambda item: (
                    -abs(cast(float, item.get("error_rate_delta") or 0.0)),
                    -cast(
                        int,
                        cast(Mapping[str, Any], item["incident"])["call_count"],
                    ),
                    cast(str, item["edge_id"]),
                )
            )
            truncated = len(profiles) > limit
            refs = [
                self._artifact_ref(
                    trace_stats_path,
                    {"line_numbers": selected_lines, "edge_count": len(profiles)},
                )
            ]
            if selected_records:
                refs.append(
                    self._artifact_ref(
                        trace_path,
                        {"source_records": sorted(selected_records)},
                    )
                )
            return {
                "status": "partial" if truncated else "success",
                "entity_id": entity_id,
                "direction": direction,
                "edge_profiles": profiles[:limit],
                "trace_duration_unit": self._trace_duration_unit,
                "evidence_level": "incident_correlation",
                "trace_profile_source": "precomputed_source_backed_edge_profiles",
                "coverage": self._coverage(
                    self.task.allowed_query_range,
                    completeness="partial",
                    basis=(
                        "prepared raw-trace parent-child edge profiles compared across "
                        "task windows; ambiguous and duplicate span keys skipped"
                    ),
                    truncation=(f"limited to top {limit} edges" if truncated else None),
                ),
                "artifact_refs": refs,
            }
        relevant_entity_ids = {
            entity_id for pair in allowed_edges for entity_id in pair
        }
        alias_index: dict[str, list[str]] = {}
        for node in self.topology.nodes:
            if node.canonical_id not in relevant_entity_ids:
                continue
            aliases = node.source_aliases.get(
                "trace/trace_span.csv", [node.original_id]
            )
            for alias in aliases:
                alias_index.setdefault(alias, []).append(node.canonical_id)
        spans: dict[tuple[str, str], dict[str, Any]] = {}
        seen_keys: set[tuple[str, str]] = set()
        duplicate_keys: set[tuple[str, str]] = set()
        try:
            with self._artifact_path(trace_path).open(
                newline="", encoding="utf-8"
            ) as stream:
                reader = csv.reader(stream)
                header = next(reader)
                column = {name: header.index(name) for name in header}
                for row in reader:
                    if "_source_member" in column:
                        self._validate_row_source_member(
                            {"_source_member": row[column["_source_member"]]},
                            trace_path,
                        )
                    timestamp = float(row[column["timestamp"]]) / 1000
                    if not self._contains(self.task.allowed_query_range, timestamp):
                        continue
                    key = (row[column["trace_id"]], row[column["span_id"]])
                    if key in seen_keys:
                        duplicate_keys.add(key)
                        continue
                    seen_keys.add(key)
                    entity_ids = alias_index.get(row[column["cmdb_id"]])
                    if not entity_ids:
                        continue
                    spans[key] = {
                        "trace_id": row[column["trace_id"]],
                        "span_id": row[column["span_id"]],
                        "parent_span": row[column["parent_span"]],
                        "entity_ids": entity_ids,
                        "timestamp": timestamp,
                        "duration": float(row[column["duration"]]),
                        "error": _trace_error_indicator(row[column["status_code"]]),
                        "source_record": int(row[column["_source_record"]]),
                    }
        except (
            OSError,
            csv.Error,
            KeyError,
            ValueError,
            IndexError,
            StopIteration,
        ) as exc:
            return self._error("source_read_error", str(exc))
        edge_values: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {
            pair: {
                period: {
                    "durations": [],
                    "errors": [],
                    "timestamps": [],
                    "source_records": [],
                }
                for period in ("reference", "incident")
            }
            for pair in allowed_edges
        }
        task_windows = {
            "reference": cast(TimeRange, self.task.reference_range),
            "incident": self.task.incident_range,
        }
        for key, child in spans.items():
            if key in duplicate_keys:
                continue
            parent_key = (child["trace_id"], child["parent_span"])
            parent = spans.get(parent_key)
            if parent is None or parent_key in duplicate_keys:
                continue
            matching_pairs = [
                (source_id, target_id)
                for source_id in cast(list[str], parent["entity_ids"])
                for target_id in cast(list[str], child["entity_ids"])
                if (source_id, target_id) in allowed_edges
            ]
            if not matching_pairs:
                continue
            timestamp = cast(float, child["timestamp"])
            period = next(
                (
                    name
                    for name, window in task_windows.items()
                    if self._contains(window, timestamp)
                ),
                None,
            )
            if period is None:
                continue
            for pair in matching_pairs:
                values = edge_values[pair][period]
                values["durations"].append(cast(float, child["duration"]))
                values["errors"].append(cast(float, child["error"]))
                values["timestamps"].append(timestamp)
                values["source_records"].append(
                    float(cast(int, child["source_record"]))
                )
        profiles: list[dict[str, Any]] = []
        selected_records: set[int] = set()
        for pair, periods in edge_values.items():
            reference = periods["reference"]
            incident = periods["incident"]
            reference_durations = reference["durations"]
            incident_durations = incident["durations"]
            reference_p95 = _percentile(reference_durations, 0.95)
            reference_p99 = _percentile(reference_durations, 0.99)
            latency_threshold = max(
                reference_p99 or 0.0,
                1.5 * (reference_p95 or 0.0),
            )
            incident_error_rate = (
                statistics.fmean(incident["errors"]) if incident["errors"] else None
            )
            reference_error_rate = (
                statistics.fmean(reference["errors"]) if reference["errors"] else None
            )
            anomaly_timestamps = [
                timestamp
                for timestamp, duration, error in zip(
                    incident["timestamps"],
                    incident_durations,
                    incident["errors"],
                    strict=True,
                )
                if error > 0 or latency_threshold > 0 and duration > latency_threshold
            ]
            for record in [
                *reference["source_records"][:5],
                *incident["source_records"][:5],
            ]:
                selected_records.add(int(record))
            profiles.append(
                {
                    "edge_id": allowed_edges[pair].edge_id,
                    "source_id": pair[0],
                    "target_id": pair[1],
                    "reference": {
                        "call_count": len(reference_durations),
                        "error_count": int(sum(reference["errors"])),
                        "error_rate": reference_error_rate,
                        "latency_p50": _percentile(reference_durations, 0.50),
                        "latency_p95": reference_p95,
                        "latency_p99": reference_p99,
                    },
                    "incident": {
                        "call_count": len(incident_durations),
                        "error_count": int(sum(incident["errors"])),
                        "error_rate": incident_error_rate,
                        "latency_p50": _percentile(incident_durations, 0.50),
                        "latency_p95": _percentile(incident_durations, 0.95),
                        "latency_p99": _percentile(incident_durations, 0.99),
                    },
                    "error_rate_delta": (
                        incident_error_rate - reference_error_rate
                        if incident_error_rate is not None
                        and reference_error_rate is not None
                        else None
                    ),
                    "anomaly_onset_timestamp": (
                        min(anomaly_timestamps) if anomaly_timestamps else None
                    ),
                    "evidence_level": "incident_correlation",
                }
            )
        profiles.sort(
            key=lambda item: (
                -abs(cast(float, item.get("error_rate_delta") or 0.0)),
                -cast(int, cast(Mapping[str, Any], item["incident"])["call_count"]),
                cast(str, item["edge_id"]),
            )
        )
        truncated = len(profiles) > limit
        return {
            "status": "partial" if truncated else "success",
            "entity_id": entity_id,
            "direction": direction,
            "edge_profiles": profiles[:limit],
            "trace_duration_unit": self._trace_duration_unit,
            "evidence_level": "incident_correlation",
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="partial",
                basis=(
                    "raw trace parent-child joins compared across task windows; "
                    "ambiguous and duplicate span keys skipped"
                ),
                truncation=(f"limited to top {limit} edges" if truncated else None),
            ),
            "artifact_refs": [
                self._artifact_ref(
                    trace_path,
                    {"source_records": sorted(selected_records)},
                )
            ],
        }

    def read_artifact_slice(
        self,
        artifact_id: str,
        *,
        start_line: int = 1,
        max_lines: int = 50,
        max_bytes: int = 16 * 1024,
    ) -> Result:
        artifact = self._artifacts_by_id.get(artifact_id)
        if artifact is None:
            return self._error(
                "unknown_artifact",
                "artifact_id must be an exact content hash registered in manifest.json",
            )
        if Path(str(artifact["path"])).suffix.casefold() == ".csv":
            return self._error(
                "raw_telemetry_slice_forbidden",
                "raw CSV slices are unavailable; use a schema-aware telemetry query",
            )
        if start_line < 1:
            return self._error("invalid_arguments", "start_line must be at least 1")
        if not 1 <= max_lines <= _MAX_ARTIFACT_LINES:
            return self._error(
                "invalid_arguments",
                f"max_lines must be between 1 and {_MAX_ARTIFACT_LINES}",
            )
        if not 1 <= max_bytes <= _MAX_ARTIFACT_BYTES:
            return self._error(
                "invalid_arguments",
                f"max_bytes must be between 1 and {_MAX_ARTIFACT_BYTES}",
            )

        chunks: list[bytes] = []
        returned_lines = 0
        returned_bytes = 0
        truncated = False
        try:
            with self._checked_artifact_path(artifact).open("rb") as stream:
                for line_number, raw_line in enumerate(stream, 1):
                    if line_number < start_line:
                        continue
                    if (
                        returned_lines == max_lines
                        or returned_bytes + len(raw_line) > max_bytes
                    ):
                        truncated = True
                        break
                    chunks.append(raw_line)
                    returned_lines += 1
                    returned_bytes += len(raw_line)
        except (OSError, ValueError) as exc:
            return self._error("source_read_error", str(exc))

        if not chunks:
            result = {
                "status": "partial" if truncated else "no_data",
                "artifact_id": artifact_id,
                "path": artifact["path"],
                "content": "",
                "start_line": start_line,
                "returned_lines": 0,
                "returned_bytes": 0,
                "coverage": self._coverage(
                    self.task.allowed_query_range,
                    completeness="partial" if truncated else "complete",
                    basis="manifest-whitelisted physical-line slice",
                    truncation="byte bound reached before a complete line"
                    if truncated
                    else None,
                ),
                "artifact_refs": [
                    self._artifact_ref(artifact["path"], {"start_line": start_line})
                ],
            }
            if truncated:
                result["gap"] = self._gap()
            return result
        return {
            "status": "partial" if truncated else "success",
            "artifact_id": artifact_id,
            "path": artifact["path"],
            "content": b"".join(chunks).decode("utf-8"),
            "start_line": start_line,
            "returned_lines": returned_lines,
            "returned_bytes": returned_bytes,
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="partial" if truncated else "complete",
                basis="manifest-whitelisted physical-line slice",
                truncation="line or byte bound reached" if truncated else None,
            ),
            "artifact_refs": [
                self._artifact_ref(
                    artifact["path"],
                    {
                        "start_line": start_line,
                        "returned_lines": returned_lines,
                        "returned_bytes": returned_bytes,
                    },
                )
            ],
        }

    def summarize_anomalies(
        self,
        *,
        entity_ids: list[str] | None = None,
        limit: int = 12,
    ) -> Result:
        try:
            self._bounded_limit(limit)
            selected_entities = (
                None
                if entity_ids is None
                else frozenset(str(item) for item in entity_ids)
            )
            if selected_entities is not None:
                for entity_id in selected_entities:
                    self._entity(entity_id)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))

        signals: list[dict[str, Any]] = []
        unknown_series: list[dict[str, Any]] = []
        for row in self._metric_stats:
            if (
                selected_entities is not None
                and row["canonical_id"] not in selected_entities
            ):
                continue
            reference, incident = row.get("reference"), row.get("incident")
            if (
                not reference
                or not incident
                or not reference["count"]
                or not incident["count"]
            ):
                available = [
                    name
                    for name, window in (
                        ("reference", reference),
                        ("incident", incident),
                    )
                    if window and window["count"]
                ]
                unknown_series.append(
                    {
                        "entity_id": row["canonical_id"],
                        "metric_id": row["metric_id"],
                        "available_windows": available,
                        "missing_windows": [
                            name
                            for name in ("reference", "incident")
                            if name not in available
                        ],
                        "reason": "not scored because both windows are required",
                        "stats_line": row["_line_number"],
                    }
                )
                continue
            reference_median = float(reference["median"])
            incident_median = float(incident["median"])
            denominator = max(abs(reference_median), abs(incident_median), 1.0)
            relative_shift = (incident_median - reference_median) / denominator
            upper_extreme = max(
                0.0,
                (float(incident["max"]) - float(reference["max"]))
                / max(abs(float(incident["max"])), abs(float(reference["max"])), 1.0),
            )
            lower_extreme = min(
                0.0,
                (float(incident["min"]) - float(reference["min"]))
                / max(abs(float(incident["min"])), abs(float(reference["min"])), 1.0),
            )
            relative_extreme_shift = max(
                (upper_extreme, lower_extreme), key=lambda value: (abs(value), value)
            )
            mean_delta = abs(float(incident["mean"]) - float(reference["mean"]))
            relative_mean_shift = (
                float(incident["mean"]) - float(reference["mean"])
            ) / max(
                abs(float(incident["mean"])),
                abs(float(reference["mean"])),
                1.0,
            )
            relative_p95_shift = 0.0
            if (
                isinstance(reference.get("p95"), (int, float))
                and not isinstance(reference.get("p95"), bool)
                and isinstance(incident.get("p95"), (int, float))
                and not isinstance(incident.get("p95"), bool)
            ):
                relative_p95_shift = (
                    float(incident["p95"]) - float(reference["p95"])
                ) / max(
                    abs(float(incident["p95"])),
                    abs(float(reference["p95"])),
                    1.0,
                )
            extreme_delta = max(
                abs(float(incident["max"]) - float(reference["max"])),
                abs(float(incident["min"]) - float(reference["min"])),
                1e-12,
            )
            persistence = min(1.0, mean_delta / extreme_delta)
            entity_id = str(row["canonical_id"])
            metric_id = str(row["metric_id"])
            resource_family = self._resource_family(metric_id)
            evidence_role = str(
                row.get(
                    "evidence_role",
                    self._entities[entity_id].evidence_role,
                )
            )
            signals.append(
                {
                    "entity_id": entity_id,
                    "metric_id": metric_id,
                    "evidence_role": evidence_role,
                    "resource_family": resource_family,
                    "overview_score": max(
                        abs(relative_shift),
                        abs(relative_mean_shift),
                        abs(relative_p95_shift),
                        abs(relative_extreme_shift),
                    ),
                    "persistence_score": max(
                        abs(relative_shift),
                        abs(relative_mean_shift),
                        abs(relative_p95_shift),
                        abs(relative_extreme_shift),
                    )
                    * (0.5 + 0.5 * persistence),
                    "relative_median_shift": relative_shift,
                    "relative_mean_shift": relative_mean_shift,
                    "relative_p95_shift": relative_p95_shift,
                    "relative_extreme_shift": relative_extreme_shift,
                    "reference": reference,
                    "incident": incident,
                    "value_unit": row["value_unit"],
                    "stats_line": row["_line_number"],
                }
            )
        signals.sort(
            key=lambda item: (
                -item["overview_score"],
                item["entity_id"],
                item["metric_id"],
            )
        )
        root_cause_signals = self._root_cause_candidate_signals(signals)
        root_cause_catalog_metrics = self._root_cause_candidate_metrics()
        candidate_signals = (
            [
                signal
                for signal in root_cause_signals
                if (
                    self._entities[str(signal["entity_id"])].public_candidate_match
                    in self._public_entities
                    if self._has_public_candidate_mapping
                    else market_output_component(
                        self._entities[str(signal["entity_id"])].original_id
                    )
                    in self._public_entities
                )
            ]
            if self.profile is _MARKET_PROFILE
            else root_cause_signals
        )
        fault_type_candidates = self._select_fault_type_candidates(
            list(self._public_fault_types),
            candidate_signals,
            root_cause_catalog_metrics,
            {entity_id: entity.sources for entity_id, entity in self._entities.items()},
            sorted(self._log_names),
        )
        service_profile_seeds = (
            self._select_service_profile_seeds(candidate_signals)
            if self.profile is _MARKET_PROFILE
            else []
        )
        self._service_profile_seed_index = {
            (str(seed["service_id"]), str(seed["fault_type"])): dict(seed)
            for seed in service_profile_seeds
        }
        scored_series = len(signals)
        truncated = scored_series > limit
        signals = self._diversify(signals, limit)
        for signal in signals:
            signal["direct_metric_candidates"] = (
                self._rank_direct_metric_candidates(
                    root_cause_catalog_metrics.get(str(signal["entity_id"]), []),
                    str(signal["metric_id"]),
                    str(signal["resource_family"]),
                )
                if signal.get("evidence_role") == "root_cause_candidate"
                and self._entities[str(signal["entity_id"])].evidence_role
                == "root_cause_candidate"
                else []
            )
        if not signals:
            result = self._no_data(
                self.task.allowed_query_range,
                "no metric series had samples in both windows",
                basis=(
                    "precomputed metric window statistics; "
                    "source collection health unknown"
                ),
            )
            result["public_fault_types"] = list(self._public_fault_types)
            result["fault_type_candidates"] = []
            result["log_names"] = sorted(self._log_names)
            result["artifact_refs"].append(
                self._artifact_ref("vocabulary.json", {"kind": "public_fault_types"})
            )
            return result
        return {
            "status": "partial" if truncated else "success",
            "signals": signals,
            "public_fault_types": list(self._public_fault_types),
            "log_names": sorted(self._log_names),
            "fault_type_candidates": fault_type_candidates,
            "service_profile_seeds": service_profile_seeds,
            "unknown_series": unknown_series,
            "unknown_counts": {
                "reference_only": sum(
                    item["available_windows"] == ["reference"]
                    for item in unknown_series
                ),
                "incident_only": sum(
                    item["available_windows"] == ["incident"] for item in unknown_series
                ),
                "neither": sum(
                    not item["available_windows"] for item in unknown_series
                ),
            },
            "scanned_series": scored_series + len(unknown_series),
            "scored_series": scored_series,
            "returned": len(signals),
            "method": (
                "overview_score is the larger absolute value of relative_median_shift "
                "and relative_extreme_shift; relative_extreme_shift is the largest "
                "signed incident min/max excursion outside the reference envelope, "
                "normalized by the larger absolute paired bound; results are selected "
                "round-robin across resource_family buckets ordered by each family's "
                "top score; families are inferred from metric_id keywords for cpu, "
                "memory, network, storage, and latency, with other as fallback; "
                "direct_metric_candidates contain at most two same-entity, same-family "
                "catalog metrics ranked by non-proxy, direct-name, percent-name, then "
                "metric_id; fault_type_candidates first choose one source nomination "
                "per fault type and entity, then its best same-family direct metric "
                "with specific fault-concept overlap; cross-entity order compares the "
                "direct metric's direction-aware envelope crossing and normalized "
                "bounded positive peak before the source signal, using incident min "
                "undercut for "
                "lower-is-more-pressure metrics; candidate score is the source peak "
                "rounded to six decimals; a distinct direct metric supplies metric_id, "
                "reference, and incident, while source_metric_id records the original "
                "nomination; without a compatible direct metric the source remains "
                "ambiguous; metric_role classifies the actual metric as direct only "
                "when its non-proxy/direct rank passes, and has_log_source reflects "
                "the validated entity source inventory for this provider; "
                "and always require paired exact validation; "
                "navigation only, not a root-cause ranking"
            ),
            "reference_health": self._reference_health,
            "limitations": [
                "The reference window is not certified healthy.",
                "Sampling and collection completeness are unknown.",
                "This is an anomaly overview, not a root-cause ranking.",
            ],
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis=(
                    "precomputed two-window statistics; "
                    "source collection health unknown"
                ),
                truncation=(
                    f"returned {limit} signals by resource_family round-robin"
                    if truncated
                    else None
                ),
            ),
            "artifact_refs": [
                self._artifact_ref(
                    "metric_window_stats.jsonl",
                    {
                        "signal_line_numbers": [
                            signal["stats_line"] for signal in signals
                        ],
                        "unknown_line_numbers": [
                            item["stats_line"] for item in unknown_series
                        ],
                        "fault_candidate_line_numbers": [
                            self._stats_by_pair[
                                (candidate["entity_id"], candidate["metric_id"])
                            ]["_line_number"]
                            for candidate in fault_type_candidates
                        ],
                    },
                ),
                self._artifact_ref("vocabulary.json", {"kind": "public_fault_types"}),
            ],
        }

    def _root_cause_candidate_signals(
        self, signals: Sequence[Mapping[str, Any]]
    ) -> list[Result]:
        return [
            dict(signal)
            for signal in signals
            if signal.get("evidence_role") == "root_cause_candidate"
            and isinstance(signal.get("entity_id"), str)
            and cast(str, signal["entity_id"]) in self._entities
            and cast(str, signal["entity_id"])
            not in self._implicit_impact_role_entity_ids
            and self._entities[cast(str, signal["entity_id"])].evidence_role
            == "root_cause_candidate"
        ]

    def _root_cause_candidate_metrics(self) -> dict[str, list[str]]:
        return {
            entity_id: [
                metric_id
                for metric_id in metric_ids
                if self._stats_by_pair[(entity_id, metric_id)].get(
                    "evidence_role", self._entities[entity_id].evidence_role
                )
                == "root_cause_candidate"
            ]
            for entity_id, metric_ids in self._catalog_metrics_by_entity.items()
            if self._entities[entity_id].evidence_role == "root_cause_candidate"
            and entity_id not in self._implicit_impact_role_entity_ids
        }

    def _select_service_profile_seeds(
        self, signals: Sequence[Mapping[str, Any]]
    ) -> list[Result]:
        signals_by_entity: dict[str, list[Mapping[str, Any]]] = {}
        for signal in signals:
            entity_id = signal.get("entity_id")
            if isinstance(entity_id, str):
                signals_by_entity.setdefault(entity_id, []).append(signal)
        groups: dict[str, list[str]] = {}
        for node in self.topology.nodes:
            if node.entity_type != "pod":
                continue
            service_name = market_service_component(
                node.public_candidate_match or node.original_id
            )
            groups.setdefault(service_name, []).append(node.canonical_id)
        service_ids = {
            market_service_component(node.public_candidate_match or node.original_id): (
                node.canonical_id
            )
            for node in self.topology.nodes
            if node.entity_type == "service"
        }
        beams: list[list[Result]] = []
        for fault_type in self._public_fault_types:
            if not fault_type.startswith("container "):
                continue
            ranked: list[tuple[tuple[float, float, str], Result]] = []
            for service_name, raw_members in groups.items():
                members = sorted(raw_members)
                service_id = service_ids.get(service_name)
                if service_id is None or len(members) < 2:
                    continue
                member_scores: list[float] = []
                for member in members:
                    compatible = [
                        float(
                            signal.get(
                                "persistence_score",
                                signal.get("overview_score", 0.0),
                            )
                        )
                        for signal in signals_by_entity.get(member, [])
                        if self._fault_metric_compatible(
                            fault_type, str(signal.get("metric_id", ""))
                        )
                    ]
                    member_scores.append(max(compatible, default=0.0))
                active = [score for score in member_scores if score >= 0.05]
                if len(active) < 2:
                    continue
                consensus_ratio = len(active) / len(members)
                score = statistics.median(active)
                candidate: Result = {
                    "service_id": service_id,
                    "fault_type": fault_type,
                    "mechanism_family": (
                        "retransmission"
                        if "retransmission" in self._semantic_concepts(fault_type)
                        else "latency"
                        if "latency" in self._semantic_concepts(fault_type)
                        else "packet_transport"
                        if self._semantic_concepts(fault_type) & {"loss", "corruption"}
                        else fault_type.casefold()
                    ),
                    "member_entity_ids": members,
                    "consensus_member_count": len(active),
                    "member_count": len(members),
                    "consensus_ratio": consensus_ratio,
                    "overview_score": score,
                }
                ranked.append(((-consensus_ratio, -score, service_id), candidate))
            if ranked:
                beam: list[Result] = []
                for candidate_rank, (_key, candidate) in enumerate(
                    sorted(ranked, key=lambda item: item[0])[:2]
                ):
                    candidate["candidate_rank"] = candidate_rank
                    beam.append(candidate)
                beams.append(beam)
        return [
            candidate
            for candidate_rank in range(2)
            for beam in beams
            for candidate in beam
            if candidate["candidate_rank"] == candidate_rank
        ][:16]

    @classmethod
    def _resource_family(cls, metric_id: str) -> str:
        concepts = cls._semantic_concepts(metric_id)
        for family in ("latency", "cpu", "memory", "network", "storage"):
            if family in concepts:
                return family
        return "other"

    @classmethod
    def _rank_direct_metric_candidates(
        cls,
        metric_ids: list[str],
        current_metric_id: str,
        resource_family: str,
    ) -> list[str]:
        candidates = [
            metric_id
            for metric_id in metric_ids
            if metric_id != current_metric_id
            and cls._resource_family(metric_id) == resource_family
        ]
        candidates.sort(key=cls._direct_metric_rank)
        return candidates[:2]

    @staticmethod
    def _direct_metric_rank(metric_id: str) -> tuple[bool, bool, bool, str]:
        name = metric_id.lower()
        words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", metric_id)
        words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", words)
        tokens = frozenset(re.sub(r"[^a-zA-Z0-9%]+", " ", words).lower().split())
        compact = "".join(character for character in name if character.isalnum())
        proxy = (
            name == "cnt"
            or "singlecpu" in compact
            or "singlecore" in compact
            or any(
                token in name
                for token in (
                    "idle",
                    "free",
                    "available",
                    "remaining",
                    "counter",
                    "traffic",
                )
            )
            or bool(tokens & {"count", "total"})
        )
        direct = "read" in tokens or any(
            token in name
            for token in (
                "used",
                "usage",
                "util",
                "load",
                "latency",
                "response",
                "loss",
                "error",
                "drop",
                "space",
                "heap",
            )
        )
        percent = "%" in name or "percent" in name or "perc" in name
        return proxy, not direct, not percent, metric_id

    @staticmethod
    def _metric_name_hints(metric_id: str) -> tuple[str, str]:
        words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", metric_id)
        words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", words)
        tokens = frozenset(re.sub(r"[^a-zA-Z0-9%]+", " ", words).lower().split())
        compact = "".join(
            character for character in metric_id.casefold() if character.isalnum()
        )
        success_rate = compact in {"sr", "successrate", "succeerate"} or bool(
            tokens & {"success", "succee"} and tokens & {"rate", "ratio"}
        )
        direction = (
            "lower_is_more_pressure"
            if success_rate or tokens & {"idle", "free", "available", "remaining"}
            else "higher_is_more_pressure"
            if tokens
            & {
                "used",
                "usage",
                "util",
                "utilization",
                "load",
                "latency",
                "loss",
                "error",
                "errors",
                "drop",
                "dropped",
                "queue",
                "active",
                "connect",
                "connection",
                "session",
                "read",
                "space",
                "heap",
            }
            else "unknown"
        )
        scale = (
            "0..100"
            if success_rate or "%" in metric_id or tokens & {"percent", "perc"}
            else "unknown"
        )
        return scale, direction

    @classmethod
    def _candidate_pressure_direction(cls, fault_type: str, metric_id: str) -> str:
        _scale, direction = cls._metric_name_hints(metric_id)
        fault_concepts = cls._semantic_concepts(fault_type)
        compact_metric = "".join(
            character for character in metric_id.casefold() if character.isalnum()
        )
        if (
            "network" in fault_concepts
            and fault_concepts & {"latency", "loss"}
            and any(name in compact_metric for name in ("bandwidth", "throughput"))
        ):
            return "lower_is_more_pressure"
        return direction

    @classmethod
    def _directional_envelope_score(
        cls, fault_type: str, signal: Mapping[str, Any]
    ) -> tuple[bool, float]:
        reference = cast(Mapping[str, Any], signal["reference"])
        incident = cast(Mapping[str, Any], signal["incident"])
        direction = cls._candidate_pressure_direction(
            fault_type, str(signal["metric_id"])
        )
        if direction == "lower_is_more_pressure":
            reference_bound = float(reference["min"])
            incident_bound = float(incident["min"])
            delta = reference_bound - incident_bound
        elif direction == "higher_is_more_pressure":
            reference_bound = float(reference["max"])
            incident_bound = float(incident["max"])
            delta = incident_bound - reference_bound
        else:
            overview_score = signal.get("overview_score")
            if isinstance(overview_score, (int, float)) and not isinstance(
                overview_score, bool
            ):
                return False, min(1.0, max(0.0, abs(float(overview_score))))
            relative_shift = signal.get("relative_median_shift")
            if (
                isinstance(relative_shift, (int, float))
                and not isinstance(relative_shift, bool)
                and relative_shift
            ):
                return False, min(1.0, abs(float(relative_shift)))
            deltas = (
                float(incident["max"]) - float(reference["max"]),
                float(incident["min"]) - float(reference["min"]),
            )
            peak = max(
                abs(delta)
                / max(
                    abs(float(incident[bound])),
                    abs(float(reference[bound])),
                    1e-12,
                )
                for delta, bound in zip(deltas, ("max", "min"), strict=True)
            )
            return False, min(1.0, peak)
        peak = min(
            1.0,
            max(
                0.0,
                delta / max(abs(incident_bound), abs(reference_bound), 1e-12),
            ),
        )
        return delta > 0.0, peak

    @classmethod
    def _select_fault_type_candidates(
        cls,
        fault_types: list[str],
        signals: list[Result],
        catalog_metrics_by_entity: Mapping[str, list[str]],
        entity_sources_by_entity: Mapping[str, Sequence[str]] | None = None,
        log_names: Sequence[str] = (),
    ) -> list[Result]:
        peer_scores: dict[tuple[str, str], dict[str, float]] = {}
        for signal in signals:
            entity_id = str(signal["entity_id"])
            if ":pod:" in entity_id:
                component = entity_id.split(":pod:", 1)[1]
                group = f"pod:{market_service_component(component)}"
            elif ":node:" in entity_id:
                group = "node"
            elif ":service:" in entity_id:
                group = "service"
            else:
                continue
            peer_scores.setdefault((group, str(signal["metric_id"])), {})[entity_id] = (
                float(
                    signal.get("persistence_score", signal.get("overview_score", 0.0))
                )
            )
        peer_evidence: dict[tuple[str, str], tuple[bool, int, float, list[str]]] = {}
        for (_group, metric_id), scores_by_entity in peer_scores.items():
            if len(scores_by_entity) < 3:
                continue
            median_score = statistics.median(scores_by_entity.values())
            for entity_id, score in scores_by_entity.items():
                ratio = score / max(median_score, 0.01)
                peer_evidence[(entity_id, metric_id)] = (
                    score >= 0.1 and ratio >= 3.0,
                    len(scores_by_entity),
                    ratio,
                    sorted(item for item in scores_by_entity if item != entity_id),
                )
        beams: list[list[Result]] = []
        for fault_type in fault_types:
            fault_concepts = cls._semantic_concepts(fault_type)
            source_by_entity: dict[
                str,
                tuple[
                    tuple[Any, ...],
                    Result,
                    frozenset[str],
                    frozenset[str],
                    bool,
                    float,
                ],
            ] = {}
            for signal in signals:
                entity_id = str(signal["entity_id"])
                metric_id = str(signal["metric_id"])
                if cls.profile is _MARKET_PROFILE and metric_id.startswith("log::"):
                    continue
                if (
                    cls.profile is _MARKET_PROFILE
                    and fault_type.startswith(("node ", "container "))
                    and not cls._market_fault_matches_entity(entity_id, fault_type)
                ):
                    continue
                if not cls._fault_entity_compatible(entity_id, fault_type):
                    continue
                if metric_id not in catalog_metrics_by_entity.get(entity_id, []):
                    continue
                if cls.profile is _MARKET_PROFILE and not cls._fault_metric_compatible(
                    fault_type, metric_id
                ):
                    continue
                overlap = fault_concepts & cls._semantic_concepts(metric_id)
                specific_overlap = overlap - {"pressure"}
                if not specific_overlap:
                    continue
                crosses_envelope, normalized_peak = cls._directional_envelope_score(
                    fault_type, signal
                )
                scale_hint, pressure_direction = cls._metric_name_hints(metric_id)
                incident = cast(Mapping[str, object], signal["incident"])
                saturation_score = 0.0
                if (
                    scale_hint == "0..100"
                    and pressure_direction == "higher_is_more_pressure"
                    and isinstance(incident.get("mean"), (int, float))
                    and not isinstance(incident.get("mean"), bool)
                ):
                    saturation_score = min(
                        1.0, float(cast(float, incident["mean"])) / 100.0
                    ) * min(
                        1.0,
                        float(
                            signal.get(
                                "persistence_score", signal.get("overview_score", 0.0)
                            )
                        ),
                    )
                semantic_score = len(overlap) + len(specific_overlap)
                qualifier_matches = len(
                    overlap & {"latency", "loss", "read", "space", "jvm"}
                )
                direct_rank = cls._direct_metric_rank(metric_id)
                peer_outlier, _peer_size, peer_ratio, _peer_ids = peer_evidence.get(
                    (entity_id, metric_id), (False, 0, 0.0, [])
                )
                key = (
                    -int(peer_outlier),
                    -min(peer_ratio, 10.0),
                    -saturation_score,
                    -normalized_peak,
                    -int(crosses_envelope),
                    -qualifier_matches,
                    -semantic_score,
                    *direct_rank[:3],
                    fault_type,
                    entity_id,
                    metric_id,
                )
                if (
                    entity_id not in source_by_entity
                    or key < source_by_entity[entity_id][0]
                ):
                    source_by_entity[entity_id] = (
                        key,
                        signal,
                        overlap,
                        specific_overlap,
                        crosses_envelope,
                        normalized_peak,
                    )

            ranked_by_entity: list[tuple[tuple[Any, ...], Result]] = []
            for entity_id in sorted(source_by_entity):
                (
                    _source_key,
                    source_signal,
                    source_overlap,
                    source_specific_overlap,
                    source_crosses,
                    source_peak,
                ) = source_by_entity[entity_id]
                source_metric_id = str(source_signal["metric_id"])
                resource_family = str(source_signal["resource_family"])
                compatible_direct: list[
                    tuple[
                        tuple[Any, ...],
                        Result,
                        frozenset[str],
                        frozenset[str],
                        bool,
                        float,
                    ]
                ] = []
                for direct_signal in signals:
                    direct_metric_id = str(direct_signal["metric_id"])
                    if (
                        direct_signal["entity_id"] != entity_id
                        or direct_signal["resource_family"] != resource_family
                        or (
                            cls.profile is _MARKET_PROFILE
                            and direct_metric_id.startswith("log::")
                        )
                        or direct_metric_id
                        not in catalog_metrics_by_entity.get(entity_id, [])
                        or cls._direct_metric_rank(direct_metric_id)[:2]
                        != (False, False)
                        or (
                            cls.profile is _MARKET_PROFILE
                            and not cls._fault_metric_compatible(
                                fault_type, direct_metric_id
                            )
                        )
                    ):
                        continue
                    direct_overlap = fault_concepts & cls._semantic_concepts(
                        direct_metric_id
                    )
                    direct_specific_overlap = direct_overlap - {"pressure"}
                    if not direct_specific_overlap:
                        continue
                    direct_crosses, direct_peak = cls._directional_envelope_score(
                        fault_type, direct_signal
                    )
                    direct_qualifier_matches = len(
                        direct_overlap & {"latency", "loss", "read", "space", "jvm"}
                    )
                    direct_semantic_score = len(direct_overlap) + len(
                        direct_specific_overlap
                    )
                    compatible_direct.append(
                        (
                            (
                                -int(direct_crosses),
                                -direct_peak,
                                -direct_qualifier_matches,
                                -direct_semantic_score,
                                *cls._direct_metric_rank(direct_metric_id)[:3],
                                direct_metric_id,
                            ),
                            direct_signal,
                            direct_overlap,
                            direct_specific_overlap,
                            direct_crosses,
                            direct_peak,
                        )
                    )
                direct_choice = min(
                    compatible_direct, key=lambda item: item[0], default=None
                )
                if direct_choice is None:
                    actual_signal = source_signal
                    direct_overlap = frozenset[str]()
                    direct_specific_overlap = frozenset[str]()
                    direct_crosses = False
                    direct_peak = 0.0
                else:
                    (
                        _direct_key,
                        actual_signal,
                        direct_overlap,
                        direct_specific_overlap,
                        direct_crosses,
                        direct_peak,
                    ) = direct_choice
                actual_metric_id = str(actual_signal["metric_id"])
                entity_sources = (entity_sources_by_entity or {}).get(entity_id, ())
                has_log_source = any(
                    source in entity_sources for source in cls.profile.log_sources
                )
                has_trace_source = "trace/trace_span.csv" in entity_sources
                (
                    peer_outlier,
                    peer_group_size,
                    peer_score_ratio,
                    peer_entity_ids,
                ) = peer_evidence.get(
                    (entity_id, source_metric_id), (False, 0, 0.0, [])
                )
                candidate: Result = {
                    "fault_type": fault_type,
                    "entity_id": entity_id,
                    "metric_id": actual_metric_id,
                    "resource_family": resource_family,
                    "score": round(source_peak, 6),
                    "overview_score": float(
                        source_signal.get(
                            "overview_score",
                            max(
                                abs(
                                    float(
                                        source_signal.get("relative_median_shift", 0.0)
                                    )
                                ),
                                abs(
                                    float(
                                        source_signal.get("relative_extreme_shift", 0.0)
                                    )
                                ),
                            ),
                        )
                    ),
                    "reference": cls._compact_bounds(actual_signal["reference"]),
                    "incident": cls._compact_bounds(actual_signal["incident"]),
                    "requires_paired_exact": True,
                    "candidate_pressure_direction": cls._candidate_pressure_direction(
                        fault_type, actual_metric_id
                    ),
                    "has_log_source": has_log_source,
                    "has_trace_source": has_trace_source,
                    "log_names": sorted(log_names) if has_log_source else [],
                    "metric_role": "direct"
                    if direct_choice is not None
                    else "ambiguous",
                }
                related_metric_set: set[str] = {
                    str(signal["metric_id"])
                    for signal in signals
                    if signal["entity_id"] == entity_id
                    and signal["resource_family"] == resource_family
                    and str(signal["metric_id"])
                    in catalog_metrics_by_entity.get(entity_id, [])
                    and (
                        cls._fault_metric_compatible(
                            fault_type, str(signal["metric_id"])
                        )
                        if cls.profile is _MARKET_PROFILE
                        else bool(
                            fault_concepts
                            & cls._semantic_concepts(str(signal["metric_id"]))
                        )
                    )
                }

                def related_metric_rank(
                    related_metric: str,
                ) -> tuple[int, int, tuple[bool, bool, bool, str]]:
                    return (
                        -int(related_metric == source_metric_id),
                        -int(related_metric == actual_metric_id),
                        cls._direct_metric_rank(related_metric),
                    )

                related_metrics = sorted(
                    related_metric_set,
                    key=related_metric_rank,
                )
                support_metrics = sorted(
                    {
                        str(signal["metric_id"])
                        for signal in signals
                        if signal["entity_id"] == entity_id
                        and signal["resource_family"] == resource_family
                        and str(signal["metric_id"])
                        in catalog_metrics_by_entity.get(entity_id, [])
                        and not str(signal["metric_id"]).startswith("log::")
                    }
                    - set(related_metrics),
                    key=cls._direct_metric_rank,
                )
                mechanism_support: list[str] = []
                if "retransmission" in fault_concepts:
                    mechanism_support = sorted(
                        (
                            metric_id
                            for metric_id in support_metrics
                            if "network_transmit_mb" in metric_id.casefold()
                            or "network_receive_mb" in metric_id.casefold()
                        ),
                        key=lambda metric_id: (
                            -int("network_transmit_mb" in metric_id.casefold()),
                            metric_id,
                        ),
                    )
                candidate["metric_group"] = list(
                    dict.fromkeys(
                        [
                            source_metric_id,
                            actual_metric_id,
                            *related_metrics,
                            *mechanism_support,
                            *support_metrics,
                        ]
                    )
                )[:8]
                if peer_group_size:
                    candidate.update(
                        peer_outlier=peer_outlier,
                        peer_group_size=peer_group_size,
                        peer_score_ratio=round(peer_score_ratio, 6),
                        peer_entity_ids=peer_entity_ids,
                    )
                if actual_metric_id != source_metric_id:
                    candidate["source_metric_id"] = source_metric_id
                    candidate["source_pressure_direction"] = (
                        cls._candidate_pressure_direction(fault_type, source_metric_id)
                    )
                source_qualifier_matches = len(
                    source_overlap & {"latency", "loss", "read", "space", "jvm"}
                )
                source_semantic_score = len(source_overlap) + len(
                    source_specific_overlap
                )
                direct_qualifier_matches = len(
                    direct_overlap & {"latency", "loss", "read", "space", "jvm"}
                )
                direct_semantic_score = len(direct_overlap) + len(
                    direct_specific_overlap
                )
                source_saturation_score = cls._signal_saturation_score(
                    source_metric_id, source_signal
                )
                direct_saturation_score = cls._signal_saturation_score(
                    actual_metric_id, actual_signal
                )
                if max(source_saturation_score, direct_saturation_score) > 0:
                    candidate["saturation_score"] = round(
                        max(source_saturation_score, direct_saturation_score), 6
                    )
                materiality_reasons: list[str] = []
                if direct_crosses and direct_peak >= 0.2:
                    materiality_reasons.append("direct_directional_envelope_crossing")
                if source_crosses and source_peak >= 0.2:
                    materiality_reasons.append("source_directional_envelope_crossing")
                if peer_outlier and max(source_peak, direct_peak) >= 0.2:
                    materiality_reasons.append("peer_isolated_change")
                if max(source_saturation_score, direct_saturation_score) >= 0.4:
                    materiality_reasons.append("absolute_resource_pressure")
                candidate["materiality"] = (
                    "material" if materiality_reasons else "navigation_only"
                )
                candidate["materiality_basis"] = (
                    "+".join(materiality_reasons)
                    if materiality_reasons
                    else "no_registered_materiality_rule_matched"
                )
                ranked_by_entity.append(
                    (
                        (
                            -int(cls.profile is _TELECOM_PROFILE)
                            * max(source_saturation_score, direct_saturation_score),
                            -int(direct_crosses),
                            -int(peer_outlier),
                            -min(peer_score_ratio, 10.0),
                            -int(cls.profile is _MARKET_PROFILE and has_log_source),
                            -direct_peak,
                            -int(source_crosses),
                            -source_peak,
                            -direct_qualifier_matches,
                            -direct_semantic_score,
                            -source_qualifier_matches,
                            -source_semantic_score,
                            *cls._direct_metric_rank(actual_metric_id)[:3],
                            fault_type,
                            entity_id,
                            source_metric_id,
                            actual_metric_id,
                        ),
                        candidate,
                    )
                )
            if ranked_by_entity:
                beam: list[Result] = []
                beam_limit = (
                    6 if cls.profile in {_MARKET_PROFILE, _TELECOM_PROFILE} else 2
                )
                for candidate_rank, (_key, candidate) in enumerate(
                    sorted(ranked_by_entity, key=lambda item: item[0])[:beam_limit]
                ):
                    candidate["candidate_id"] = cls._fault_candidate_id(candidate)
                    candidate["candidate_rank"] = candidate_rank
                    beam.append(candidate)
                beams.append(beam)
        beam_limit = 6 if cls.profile in {_MARKET_PROFILE, _TELECOM_PROFILE} else 2
        candidate_limit = (
            90 if cls.profile in {_MARKET_PROFILE, _TELECOM_PROFILE} else 32
        )
        return [
            candidate
            for candidate_rank in range(beam_limit)
            for beam in beams
            for candidate in beam
            if candidate["candidate_rank"] == candidate_rank
        ][:candidate_limit]

    @staticmethod
    def _market_fault_matches_entity(entity_id: str, fault_type: str) -> bool:
        if fault_type == "container process termination":
            return ":pod:" in entity_id
        return (
            fault_type.startswith("node ")
            if ":node:" in entity_id
            else fault_type.startswith("container ")
        )

    @classmethod
    def _fault_entity_compatible(cls, entity_id: str, fault_type: str) -> bool:
        if cls.profile is _MARKET_PROFILE and fault_type.startswith(
            ("node ", "container ")
        ):
            return cls._market_fault_matches_entity(entity_id, fault_type)
        return True

    @classmethod
    def _signal_saturation_score(
        cls, metric_id: str, signal: Mapping[str, object]
    ) -> float:
        scale_hint, pressure_direction = cls._metric_name_hints(metric_id)
        incident = signal.get("incident")
        if (
            scale_hint != "0..100"
            or pressure_direction != "higher_is_more_pressure"
            or not isinstance(incident, Mapping)
            or not isinstance(
                cast(Mapping[str, object], incident).get("mean"), (int, float)
            )
            or isinstance(cast(Mapping[str, object], incident).get("mean"), bool)
        ):
            return 0.0
        shift = signal.get("relative_mean_shift")
        if not isinstance(shift, (int, float)) or isinstance(shift, bool):
            return 0.0
        return min(
            1.0,
            float(cast(float, cast(Mapping[str, object], incident)["mean"])) / 100.0,
        ) * min(1.0, abs(float(shift)))

    @staticmethod
    def _fault_candidate_id(candidate: Mapping[str, Any]) -> str:
        canonical = json.dumps(
            {
                name: candidate[name]
                for name in (
                    "fault_type",
                    "entity_id",
                    "resource_family",
                    "metric_id",
                )
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return "fc1:" + hashlib.sha256(canonical).hexdigest()

    @staticmethod
    def _semantic_concepts(value: str) -> frozenset[str]:
        words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
        words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", words)
        name = words.lower()
        tokens = frozenset(re.sub(r"[^a-zA-Z0-9%]+", " ", words).lower().split())
        concepts: set[str] = set()
        rules = {
            "cpu": ("cpu", "processor"),
            "memory": ("memory", "mem", "heap"),
            "network": ("network", "net", "tcp", "packet"),
            "latency": ("latency", "mrt", "response", "duration"),
            "loss": ("loss", "error", "drop", "reject", "fail", "oom"),
            "storage": ("disk", "dsk", "storage", "filesystem", "inode", "iops"),
            "space": ("space", "capacity"),
            "pressure": ("usage", "used", "util", "load", "percent", "perc"),
            "retransmission": ("retrans",),
            "corruption": ("corrupt", "checksum"),
            "termination": ("termination", "terminated", "killed", "restart"),
            "spike": ("spike",),
        }
        for concept, synonyms in rules.items():
            if any(synonym in name for synonym in synonyms):
                concepts.add(concept)
        if tokens & {"read", "reads"}:
            concepts.add("read")
        if tokens & {"write", "writes"}:
            concepts.add("write")
        if tokens & {"r", "rkb"} or any(token.startswith("r_") for token in tokens):
            concepts.add("read")
        if tokens & {"w", "wkb"} or any(token.startswith("w_") for token in tokens):
            concepts.add("write")
        if "fs" in tokens or "io" in tokens:
            concepts.add("storage")
        if (
            "storage" in concepts
            and "pressure" in concepts
            and tokens & {"disk", "dsk", "fs", "inode", "inodes"}
        ):
            concepts.add("space")
        if "network" in concepts and tokens & {"packet", "packets"}:
            concepts.add("packet_count")
        if "jvm" in tokens:
            concepts.add("jvm")
        return frozenset(concepts)

    @classmethod
    def _fault_metric_compatible(cls, fault_type: str, metric_id: str) -> bool:
        fault = cls._semantic_concepts(fault_type)
        metric = cls._semantic_concepts(metric_id)
        families = fault & {"cpu", "memory", "network", "storage"}
        if families and not families & metric:
            return False
        if "retransmission" in fault:
            return "network" in metric and "packet_count" in metric
        if "corruption" in fault:
            return "network" in metric and "loss" in metric
        qualifiers = fault & {"read", "write", "space", "latency", "loss"}
        if qualifiers:
            return bool(qualifiers & metric)
        return bool(families & metric)

    @staticmethod
    def _compact_bounds(value: object) -> Result:
        bounds = cast(Mapping[str, Any], value)
        return {name: bounds[name] for name in ("min", "median", "max")}

    @staticmethod
    def _diversify(signals: list[Result], limit: int) -> list[Result]:
        buckets = {
            family: [
                signal for signal in signals if signal["resource_family"] == family
            ]
            for family in _RESOURCE_FAMILIES
        }
        family_order = sorted(
            (family for family in _RESOURCE_FAMILIES if buckets[family]),
            key=lambda family: (
                -float(buckets[family][0]["overview_score"]),
                _RESOURCE_FAMILIES.index(family),
            ),
        )
        selected: list[Result] = []
        for index in range(
            max((len(buckets[family]) for family in family_order), default=0)
        ):
            for family in family_order:
                if index < len(buckets[family]):
                    selected.append(buckets[family][index])
                    if len(selected) == limit:
                        return selected
        return selected

    def _artifact_path(self, relative: str) -> Path:
        try:
            artifact = self._artifacts_by_path[relative]
        except KeyError as exc:
            raise ValueError(
                f"artifact is not registered in manifest: {relative}"
            ) from exc
        return self._checked_artifact_path(artifact)

    def _checked_artifact_path(self, artifact: Mapping[str, Any]) -> Path:
        original = self.root / str(artifact["path"])
        if original.is_symlink():
            raise ValueError(f"artifact path became a symlink: {artifact['path']}")
        try:
            resolved = original.resolve(strict=True)
            resolved.relative_to(self.root)
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"artifact path escaped or disappeared: {artifact['path']}"
            ) from exc
        if resolved != artifact["resolved"] or not resolved.is_file():
            raise ValueError(
                f"artifact path changed after initialization: {artifact['path']}"
            )
        return resolved

    @classmethod
    def _reject_evaluation_labels(cls, payload: Any, location: str) -> None:
        if isinstance(payload, dict):
            mapping = cast(dict[str, Any], payload)
            for key, value in mapping.items():
                metadata_kind = _safe_evaluation_metadata_kind(str(key))
                if metadata_kind is not None and not _safe_evaluation_metadata_value(
                    metadata_kind, value
                ):
                    raise ValueError(
                        f"unsafe evaluation metadata {key!r} in {location}"
                    )
            labels = sorted(
                str(key)
                for key in mapping
                if _safe_evaluation_metadata_kind(str(key)) is None
                and _forbidden_evaluation_name(str(key))
            )
            if labels:
                raise ValueError(
                    f"forbidden evaluation label {labels[0]!r} in {location}"
                )
            for key, value in mapping.items():
                if _safe_evaluation_metadata_kind(str(key)) is None:
                    cls._reject_evaluation_labels(value, location)
        elif isinstance(payload, list):
            for value in cast(list[Any], payload):
                cls._reject_evaluation_labels(value, location)

    def _validate_csv_input_boundary(self, relative: str) -> None:
        try:
            with self._artifact_path(relative).open(
                newline="", encoding="utf-8"
            ) as stream:
                reader = csv.DictReader(stream)
                headers = reader.fieldnames
                if headers is None or not headers:
                    raise ValueError(f"CSV artifact has no header: {relative}")
                forbidden_headers = [
                    header for header in headers if _forbidden_csv_header(header)
                ]
                if forbidden_headers:
                    raise ValueError(
                        "forbidden evaluation header "
                        f"{forbidden_headers[0]!r} in {relative}"
                    )
                expected_headers = self._expected_csv_headers(relative)
                if tuple(headers) not in expected_headers:
                    raise ValueError(f"CSV header contract mismatch: {relative}")
        except (OSError, UnicodeError, csv.Error) as exc:
            raise ValueError(f"invalid CSV artifact {relative}: {exc}") from exc

    @staticmethod
    def _validate_row_source_member(
        row: Mapping[str, str | None], relative: str
    ) -> None:
        if "_source_member" not in row:
            return
        source_member = row.get("_source_member")
        if not isinstance(source_member, str) or not source_member:
            raise ValueError(f"CSV source member is missing in {relative}")
        if _forbidden_source_name(source_member):
            raise ValueError(
                f"forbidden evaluation source member {source_member!r} in {relative}"
            )

    def _expected_csv_headers(self, relative: str) -> tuple[tuple[str, ...], ...]:
        if self.task.environment == "openrca_bank":
            headers = _BANK_TELEMETRY_HEADERS.get(relative)
            variants = (
                (headers, tuple(item for item in headers if item != "_source_member"))
                if headers is not None
                else None
            )
        elif self.task.environment == "synthetic_bank":
            headers = _SYNTHETIC_BANK_TELEMETRY_HEADERS.get(relative)
            variants = (
                (headers, (*headers, "_source_member")) if headers is not None else None
            )
        elif self.task.environment == "openrca_telecom":
            headers = _TELECOM_TELEMETRY_HEADERS.get(relative)
            variants = (headers,) if headers is not None else None
        elif self.task.environment == "openrca_market":
            source_path = relative.removeprefix("telemetry/")
            spec = _MARKET_SOURCE_FILES.get(source_path)
            headers = (
                (
                    *cast(tuple[str, ...], spec["columns"]),
                    "_source_record",
                    "_source_member",
                )
                if spec is not None
                else None
            )
            variants = (headers,) if headers is not None else None
        else:
            variants = None
        if variants is None:
            raise ValueError(f"CSV artifact is not authorized: {relative}")
        return variants

    def _artifact_ref(self, relative: str, selector: Mapping[str, Any]) -> Result:
        artifact = self._artifacts_by_path[relative]
        return {
            "content_hash": artifact["content_hash"],
            "selector": {"path": relative, **dict(selector)},
        }

    def _entity(self, canonical_id: str) -> TopologyNode:
        try:
            return self._entities[canonical_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown canonical entity_id: {canonical_id}; "
                "raw/original IDs are not accepted"
            ) from exc

    def _window(
        self,
        period: str,
        start: str | datetime | None,
        end: str | datetime | None,
    ) -> tuple[str, TimeRange]:
        if start is not None or end is not None:
            if start is None or end is None:
                raise ValueError("custom ranges require both start and end")
            window = TimeRange(start=self._datetime(start), end=self._datetime(end))
            period_name = "custom"
        elif period == "incident":
            window, period_name = self.task.incident_range, period
        elif period == "reference":
            if self.task.reference_range is None:
                raise ValueError("this task has no reference range")
            window, period_name = self.task.reference_range, period
        elif period == "allowed":
            window, period_name = self.task.allowed_query_range, period
        elif period == "custom":
            raise ValueError("custom ranges require both start and end")
        else:
            raise ValueError("period must be reference, incident, allowed, or custom")
        allowed = self.task.allowed_query_range
        if window.start < allowed.start or window.end > allowed.end:
            raise ValueError("requested range is outside allowed_query_range")
        return period_name, window

    @staticmethod
    def _file_hash(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _datetime(value: str | datetime) -> datetime:
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
        if parsed.tzinfo is None:
            raise ValueError("custom range timestamps must include a timezone")
        return parsed

    @staticmethod
    def _bounded_limit(limit: int) -> None:
        if isinstance(limit, bool) or not 1 <= limit <= _MAX_LIST_LIMIT:
            raise ValueError(
                f"limit must be an integer between 1 and {_MAX_LIST_LIMIT}"
            )

    @staticmethod
    def _source_timestamp(timestamp: float) -> int | float:
        return int(timestamp) if timestamp.is_integer() else timestamp

    @staticmethod
    def _contains(window: TimeRange, timestamp_seconds: float) -> bool:
        return window.start.timestamp() <= timestamp_seconds < window.end.timestamp()

    def _observed(
        self, timestamps: list[float], requested: TimeRange
    ) -> tuple[float | None, Result | None]:
        unique = sorted(set(timestamps))
        if len(unique) < 2:
            return None, None
        gaps = [right - left for left, right in zip(unique, unique[1:]) if right > left]
        sample_interval = statistics.median(gaps) if gaps else None
        observed_end = min(
            requested.end.timestamp(),
            unique[-1] + (sample_interval or 0),
        )
        if observed_end <= unique[0]:
            return sample_interval, None
        return sample_interval, {
            "start": self._epoch_iso(unique[0], requested.start),
            "end": self._epoch_iso(observed_end, requested.start),
            "end_exclusive": True,
        }

    @staticmethod
    def _epoch_iso(timestamp: float, timezone_source: datetime) -> str:
        zone = timezone_source.tzinfo or timezone.utc
        return (
            datetime.fromtimestamp(timestamp, timezone.utc).astimezone(zone).isoformat()
        )

    @staticmethod
    def _range(window: TimeRange) -> Result:
        return window.model_dump(mode="json", include={"start", "end", "end_exclusive"})

    def _coverage(
        self,
        requested: TimeRange,
        *,
        completeness: str,
        basis: str,
        observed_range: Result | None = None,
        sample_interval: float | None = None,
        truncation: str | None = None,
    ) -> Result:
        return {
            "requested_range": self._range(requested),
            "observed_range": observed_range,
            "sample_interval": sample_interval,
            "completeness": completeness,
            "sampling": "unknown",
            "truncation": truncation,
            "collection_health": "unknown",
            "basis": basis,
        }

    @staticmethod
    def _gap() -> Result:
        return {
            "kind": "coverage_gap",
            "meaning": (
                "empty or omitted data is not evidence that the event was absent"
            ),
        }

    def _no_data(
        self,
        window: TimeRange,
        summary: str,
        *,
        basis: str,
        completeness: str = "unknown",
    ) -> Result:
        result: Result = {
            "status": "no_data",
            "summary": summary,
            "coverage": self._coverage(window, completeness=completeness, basis=basis),
            "artifact_refs": [],
        }
        if completeness != "complete":
            result["gap"] = self._gap()
        return result

    def _error(
        self,
        error_type: str,
        summary: str,
        window: TimeRange | None = None,
    ) -> Result:
        return {
            "status": "error",
            "error_type": error_type,
            "retryable": False,
            "summary": summary,
            "coverage": self._coverage(
                window or self.task.allowed_query_range,
                completeness="unknown",
                basis="query was not completed",
            ),
            "artifact_refs": [],
        }


class BankEnvironment(OpenRCAEnvironment):
    """Read-only provider for a prepared OpenRCA Bank public slice."""

    profile = _BANK_PROFILE

    def _business_impact_consensus(self) -> dict[str, Any]:
        cached = getattr(self, "_business_impact_consensus_cache", None)
        if isinstance(cached, dict):
            return cast(dict[str, Any], cached)
        events: list[dict[str, Any]] = []
        artifact_refs: dict[str, dict[str, Any]] = {}
        for node in self.topology.nodes:
            if node.entity_type != "business_test":
                continue
            available = set(self._catalog_metrics_by_entity.get(node.canonical_id, []))
            metric_ids = sorted(available.intersection({"mrt", "sr"}))
            if not metric_ids:
                continue
            result = super().query_metric_profile(node.canonical_id, metric_ids)
            for raw_ref in cast(list[object], result.get("artifact_refs", [])):
                if isinstance(raw_ref, Mapping):
                    ref = dict(cast(Mapping[str, Any], raw_ref))
                    artifact_refs[
                        json.dumps(ref, separators=(",", ":"), sort_keys=True)
                    ] = ref
            for raw_profile in cast(list[object], result.get("metric_profiles", [])):
                if not isinstance(raw_profile, Mapping):
                    continue
                profile = cast(Mapping[str, Any], raw_profile)
                metric_id = profile.get("metric_id")
                reference = profile.get("reference")
                incident = profile.get("incident")
                burst = profile.get("incident_burst")
                if (
                    metric_id not in {"mrt", "sr"}
                    or not isinstance(reference, Mapping)
                    or not isinstance(incident, Mapping)
                    or not isinstance(burst, Mapping)
                ):
                    continue
                reference_mean = cast(Mapping[str, object], reference).get("mean")
                incident_mean = cast(Mapping[str, object], incident).get("mean")
                onset = cast(Mapping[str, object], burst).get("onset_timestamp")
                if (
                    not isinstance(reference_mean, (int, float))
                    or isinstance(reference_mean, bool)
                    or not isinstance(incident_mean, (int, float))
                    or isinstance(incident_mean, bool)
                    or not isinstance(onset, (int, float))
                    or isinstance(onset, bool)
                ):
                    continue
                denominator = max(
                    abs(float(reference_mean)), abs(float(incident_mean)), 1.0
                )
                bad_shift = (
                    (float(incident_mean) - float(reference_mean)) / denominator
                    if metric_id == "mrt"
                    else (float(reference_mean) - float(incident_mean)) / denominator
                )
                threshold = 0.2 if metric_id == "mrt" else 0.1
                if bad_shift >= threshold:
                    events.append(
                        {
                            "entity_id": node.canonical_id,
                            "metric_id": metric_id,
                            "onset_timestamp": float(onset),
                            "bad_shift": round(bad_shift, 6),
                        }
                    )
        best: list[dict[str, Any]] = []
        for start in sorted({float(item["onset_timestamp"]) for item in events}):
            cluster = [
                item
                for item in events
                if start <= float(item["onset_timestamp"]) <= start + 300.0
            ]
            candidate_key = (
                len({str(item["entity_id"]) for item in cluster}),
                len(cluster),
                -(
                    max(float(item["onset_timestamp"]) for item in cluster)
                    - min(float(item["onset_timestamp"]) for item in cluster)
                )
                if cluster
                else 0.0,
                -start,
            )
            best_key = (
                len({str(item["entity_id"]) for item in best}),
                len(best),
                -(
                    max(float(item["onset_timestamp"]) for item in best)
                    - min(float(item["onset_timestamp"]) for item in best)
                )
                if best
                else 0.0,
                -min(float(item["onset_timestamp"]) for item in best)
                if best
                else float("-inf"),
            )
            if candidate_key > best_key:
                best = cluster
        entity_ids = sorted({str(item["entity_id"]) for item in best})
        ready = len(entity_ids) >= 2
        result = {
            "policy": "independent-business-impact-consensus-v1",
            "ready": ready,
            "consensus_onset_timestamp": (
                statistics.median(float(item["onset_timestamp"]) for item in best)
                if ready
                else None
            ),
            "source_entity_ids": entity_ids if ready else [],
            "events": best if ready else [],
            "tolerance_seconds": 300.0,
            "artifact_refs": list(artifact_refs.values()) if ready else [],
        }
        self._business_impact_consensus_cache = result
        return result

    @staticmethod
    def _apply_business_impact_alignment(
        result: Result, consensus: Mapping[str, object]
    ) -> Result:
        raw_summary = result.get("finding_gate_summary")
        if not isinstance(raw_summary, dict):
            return result
        summary = cast(dict[str, Any], raw_summary)
        raw_onset = summary.get("earliest_strong_onset_timestamp")
        consensus_onset = consensus.get("consensus_onset_timestamp")
        distance = (
            abs(float(raw_onset) - float(consensus_onset))
            if isinstance(raw_onset, (int, float))
            and not isinstance(raw_onset, bool)
            and isinstance(consensus_onset, (int, float))
            and not isinstance(consensus_onset, bool)
            else None
        )
        tolerance = consensus.get("tolerance_seconds")
        aligned = bool(
            consensus.get("ready") is True
            and summary.get("fault_type_ready") is True
            and summary.get("signature_free_finding_ready") is True
            and isinstance(distance, float)
            and isinstance(tolerance, (int, float))
            and not isinstance(tolerance, bool)
            and distance <= float(tolerance)
        )
        summary["business_impact_alignment"] = {
            key: value for key, value in consensus.items() if key != "artifact_refs"
        } | {
            "candidate_onset_distance_seconds": distance,
            "aligned": aligned,
        }
        summary["business_impact_aligned"] = aligned
        summary["business_impact_score_bonus"] = 0.5 if aligned else 0.0
        if aligned:
            raw_score = summary.get("root_cause_score")
            if isinstance(raw_score, (int, float)) and not isinstance(raw_score, bool):
                summary["root_cause_score"] = round(float(raw_score) + 0.5, 6)
            raw_refs = result.setdefault("artifact_refs", [])
            if not isinstance(raw_refs, list):
                raise ValueError("metric profile artifact_refs must be an array")
            result_refs = cast(list[dict[str, Any]], raw_refs)
            existing = {
                json.dumps(ref, separators=(",", ":"), sort_keys=True)
                for ref in result_refs
            }
            for raw_ref in cast(list[object], consensus.get("artifact_refs", [])):
                if not isinstance(raw_ref, Mapping):
                    continue
                ref = dict(cast(Mapping[str, Any], raw_ref))
                encoded = json.dumps(ref, separators=(",", ":"), sort_keys=True)
                if encoded not in existing:
                    result_refs.append(ref)
                    existing.add(encoded)
        return result

    def query_metric_profile(
        self,
        entity_id: str,
        metric_ids: list[str],
        *,
        peer_entity_ids: list[str] | None = None,
        derived_ratios: list[dict[str, str]] | None = None,
        fault_type: str | None = None,
    ) -> Result:
        result = super().query_metric_profile(
            entity_id,
            metric_ids,
            peer_entity_ids=peer_entity_ids,
            derived_ratios=derived_ratios,
            fault_type=fault_type,
        )
        entity = self._entities.get(entity_id)
        if (
            entity is None
            or entity.evidence_role == "impact_only"
            or any(
                self._metric_evidence_role(entity_id, metric_id) == "impact_only"
                for metric_id in metric_ids
                if (entity_id, metric_id) in self._stats_by_pair
            )
        ):
            return result
        consensus = self._business_impact_consensus()
        return self._apply_business_impact_alignment(result, consensus)


class MarketEnvironment(OpenRCAEnvironment):
    """Read-only provider for a prepared OpenRCA Market public slice."""

    profile = _MARKET_PROFILE


class TelecomEnvironment(OpenRCAEnvironment):
    """Read-only provider for a prepared OpenRCA Telecom public slice."""

    profile = _TELECOM_PROFILE

    capabilities = CapabilityManifest(
        metrics=True,
        logs=False,
        trace=True,
        topology=True,
        coverage_limits=(
            "frozen public_input only",
            "upstream collection completeness is generally unknown",
            "the source dataset does not provide logs",
            "no changes or source-code capability",
        ),
    )

    @staticmethod
    def _semantic_concepts(value: str) -> frozenset[str]:
        """Add Telecom vocabulary without teaching it to the harness core."""
        concepts = set(OpenRCAEnvironment._semantic_concepts(value))
        words = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", value)
        words = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", words)
        name = words.casefold()
        tokens = frozenset(re.sub(r"[^a-zA-Z0-9%]+", " ", words).lower().split())
        if any(token in name for token in ("database", "db", "tnsping")):
            concepts.add("database")
        if any(token in name for token in ("connection", "connect", "sess")):
            concepts.add("connection")
        if {"incoming", "error"} <= tokens or {"trace", "error"} <= tokens:
            concepts.update({"connection", "loss"})
        if any(token in name for token in ("delay", "result time")):
            concepts.add("latency")
        if tokens & {"close", "closed", "off", "state", "hang"}:
            concepts.add("availability")
        if tokens & {"proc", "process"}:
            concepts.add("process")
        if "queue" in tokens:
            concepts.update({"network", "pressure"})
        return frozenset(concepts)

    @classmethod
    def _fault_metric_compatible(cls, fault_type: str, metric_id: str) -> bool:
        fault = cls._semantic_concepts(fault_type)
        metric = cls._semantic_concepts(metric_id)
        if "database" in fault:
            if "database" not in metric and not metric & {
                "connection",
                "availability",
                "process",
            }:
                return False
            qualifiers = fault & {"connection", "availability"}
            if "availability" in qualifiers and metric & {
                "availability",
                "connection",
            }:
                return True
            return bool(qualifiers & metric) if qualifiers else True
        return super()._fault_metric_compatible(fault_type, metric_id)

    @staticmethod
    def _metric_name_hints(metric_id: str) -> tuple[str, str]:
        scale, direction = OpenRCAEnvironment._metric_name_hints(metric_id)
        normalized = metric_id.casefold()
        if normalized in {"on_off_state", "hang"}:
            direction = "lower_is_more_pressure"
        elif normalized == "container_cpu_used":
            scale = "0..100"
            direction = "higher_is_more_pressure"
        elif normalized in {
            "proc_used_pct",
            "proc_user_used_pct",
            "sess_active",
            "sess_connect",
            "sess_used_temp",
            "sess_used_undo",
            "session_pct",
        }:
            direction = "higher_is_more_pressure"
        return scale, direction

    @classmethod
    def _fault_entity_compatible(cls, entity_id: str, fault_type: str) -> bool:
        concepts = cls._semantic_concepts(fault_type)
        if "database" in concepts:
            return ":service:" in entity_id
        if "network" in concepts:
            return ":node:" in entity_id
        if "cpu" in concepts:
            return ":node:" in entity_id or ":pod:" in entity_id
        return True

    def query_metric_profile(
        self,
        entity_id: str,
        metric_ids: list[str],
        *,
        peer_entity_ids: list[str] | None = None,
        derived_ratios: list[dict[str, str]] | None = None,
        fault_type: str | None = None,
    ) -> Result:
        return super().query_metric_profile(
            entity_id,
            metric_ids,
            peer_entity_ids=peer_entity_ids,
            derived_ratios=derived_ratios,
            fault_type=fault_type,
        )

    def query_service_profile(
        self,
        service_id: str,
        fault_type: str,
        *,
        member_entity_ids: list[str] | None = None,
    ) -> Result:
        """Profile an atomic service that owns metrics without replica entities."""
        try:
            service = self._entity(service_id)
        except ValueError as exc:
            return self._error("invalid_arguments", str(exc))
        if service.entity_type != "service":
            return self._error(
                "invalid_arguments", "service_id must identify a canonical service"
            )
        if fault_type not in self._public_fault_types:
            return self._error(
                "invalid_arguments",
                "fault_type must be an exact public vocabulary value",
            )
        if member_entity_ids and member_entity_ids != [service_id]:
            return self._error(
                "invalid_arguments",
                "atomic service has no replica members; omit member_entity_ids",
            )
        metric_ids = sorted(
            (
                metric_id
                for metric_id in self._catalog_metrics_by_entity.get(service_id, [])
                if self._fault_metric_compatible(fault_type, metric_id)
            ),
            key=self._direct_metric_rank,
        )[:16]
        if not metric_ids:
            return self._no_data(
                self.task.allowed_query_range,
                f"no fault-compatible direct metric is registered for {service_id}",
                basis="atomic service metric catalog",
            )
        metric_result = self.query_metric_profile(
            service_id,
            metric_ids,
            fault_type=fault_type,
        )
        trace_result = self.query_trace_edges(
            service_id, direction="incoming", limit=100
        )
        raw_summary = metric_result.get("finding_gate_summary")
        summary = (
            dict(cast(Mapping[str, object], raw_summary))
            if isinstance(raw_summary, Mapping)
            else {
                "requested_fault_type": fault_type,
                "fault_type_ready": False,
                "single_component_ready": False,
                "root_cause_score": 0.0,
            }
        )
        summary["atomic_service_profile"] = True
        refs: dict[str, Result] = {}
        for result in (metric_result, trace_result):
            for raw_ref in cast(list[object], result.get("artifact_refs", [])):
                if isinstance(raw_ref, Mapping):
                    ref = dict(cast(Mapping[str, Any], raw_ref))
                    refs[json.dumps(ref, sort_keys=True)] = ref
        return {
            "status": metric_result.get("status", "error"),
            "service_id": service_id,
            "member_entity_ids": [],
            "attributed_member_entity_ids": [],
            "attribution_basis": "atomic service direct metrics",
            "member_profiles": [
                {
                    "entity_id": service_id,
                    "status": metric_result.get("status"),
                    "finding_gate_summary": summary,
                    "metric_profiles": metric_result.get("metric_profiles", []),
                }
            ],
            "metric_profiles": metric_result.get("metric_profiles", []),
            "peer_comparison": metric_result.get("peer_comparison", []),
            "trace_edge_profiles": trace_result.get("edge_profiles", []),
            "finding_gate_summary": summary,
            "coverage": self._coverage(
                self.task.allowed_query_range,
                completeness="unknown",
                basis=(
                    "paired atomic-service metrics plus incoming trace-edge profiles; "
                    "upstream collection health unknown"
                ),
            ),
            "artifact_refs": list(refs.values()),
        }


def load_openrca_environment(public_input: str | Path) -> OpenRCAEnvironment:
    """Select a fixed OpenRCA adapter from the manifest-bound public task."""
    root = Path(public_input).resolve(strict=True)
    task_path = root / "task.json"
    if (
        not root.is_dir()
        or root.name != "public_input"
        or task_path.is_symlink()
        or task_path.resolve(strict=True).parent != root
    ):
        raise ValueError("OpenRCA environment requires a safe public_input directory")
    environment = Task.model_validate_json(task_path.read_text()).environment
    adapters: dict[str, type[OpenRCAEnvironment]] = {
        "openrca_bank": BankEnvironment,
        "openrca_market": MarketEnvironment,
        "openrca_telecom": TelecomEnvironment,
        "synthetic_bank": BankEnvironment,
    }
    try:
        adapter = adapters[environment]
    except KeyError as exc:
        raise ValueError(
            f"unsupported OpenRCA task environment: {environment}"
        ) from exc
    return adapter(root)
