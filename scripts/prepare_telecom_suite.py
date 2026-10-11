#!/usr/bin/env python3
# pyright: basic
"""Prepare label-free OpenRCA Telecom public inputs with one source-day scan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterator, TextIO, cast
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rootwalk.openrca import TelecomEnvironment  # noqa: E402

SOURCE_REVISION = "7a29a7c3600cc07b7760a6c5097d36ac8573f039"
TELECOM_SLICE_SCHEMA_VERSION = "rootwalk-dataset-slice-v2"
TZ = ZoneInfo("Asia/Shanghai")
ROOT_MARKERS = frozenset({"", "0", "-1", "none", "null", "root"})
TELECOM_COMPONENTS = tuple(
    [f"os_{index:03d}" for index in range(1, 23)]
    + [f"docker_{index:03d}" for index in range(1, 9)]
    + [f"db_{index:03d}" for index in range(1, 14)]
)
TELECOM_FAULT_TYPES = (
    "CPU fault",
    "network delay",
    "network loss",
    "db connection limit",
    "db close",
)
TASK_OUTPUTS = {
    "task_1": ("root cause occurrence datetime",),
    "task_3": ("root cause component",),
    "task_4": ("root cause occurrence datetime", "root cause reason"),
    "task_6": ("root cause component", "root cause reason"),
    "task_7": (
        "root cause occurrence datetime",
        "root cause component",
        "root cause reason",
    ),
}
SOURCE_FILES = {
    "metric/metric_app.csv": (
        "serviceName",
        "startTime",
        "avg_time",
        "num",
        "succee_num",
        "succee_rate",
    ),
    "metric/metric_container.csv": (
        "itemid",
        "name",
        "bomc_id",
        "timestamp",
        "value",
        "cmdb_id",
    ),
    "metric/metric_middleware.csv": (
        "itemid",
        "name",
        "bomc_id",
        "timestamp",
        "value",
        "cmdb_id",
    ),
    "metric/metric_node.csv": (
        "itemid",
        "name",
        "bomc_id",
        "timestamp",
        "value",
        "cmdb_id",
    ),
    "metric/metric_service.csv": (
        "itemid",
        "name",
        "bomc_id",
        "timestamp",
        "value",
        "cmdb_id",
    ),
    "trace/trace_span.csv": (
        "callType",
        "startTime",
        "elapsedTime",
        "success",
        "traceId",
        "id",
        "pid",
        "cmdb_id",
        "dsName",
        "serviceName",
    ),
}
NORMALIZED_METRIC_PATH = "metric/metric_container.csv"
NORMALIZED_TRACE_PATH = "trace/trace_span.csv"
ROOT_CAUSE_CANDIDATE_ROLE = "root_cause_candidate"
IMPACT_ONLY_ROLE = "impact_only"
csv.field_size_limit(16 * 1024 * 1024)


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


@dataclass(frozen=True, slots=True)
class CaseSpec:
    row_id: int
    task_index: str
    question: str
    reference_start: datetime
    reference_end: datetime
    incident_start: datetime
    incident_end: datetime
    source_revision: str

    @property
    def case_id(self) -> str:
        return f"openrca-telecom-row-{self.row_id:04d}"

    @property
    def allowed_start(self) -> datetime:
        return self.reference_start

    @property
    def allowed_end(self) -> datetime:
        return self.incident_end


@dataclass(slots=True)
class MetricSeries:
    canonical_id: str
    entity_id: str
    metric_id: str
    evidence_role: str
    source_members: set[str] = field(default_factory=set)
    timestamps: list[float] = field(default_factory=list)
    periods: dict[str, list[float]] = field(
        default_factory=lambda: {"reference": [], "incident": []}
    )
    records: dict[str, list[int]] = field(
        default_factory=lambda: {"reference": [], "incident": []}
    )


@dataclass(slots=True)
class CaseState:
    spec: CaseSpec
    public_input: Path
    metric_stream: TextIO
    metric_writer: csv.DictWriter[str]
    trace_stream: TextIO
    trace_writer: csv.DictWriter[str]
    metric_series: dict[tuple[str, str], MetricSeries] = field(default_factory=dict)
    entity_sources: dict[str, set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    entity_aliases: dict[str, dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    span_index: dict[tuple[str, str], tuple[str, str, int, int, float, float]] = field(
        default_factory=dict
    )
    span_conflicts: set[tuple[str, str]] = field(default_factory=set)
    duplicate_span_keys: int = 0
    source_rows: Counter[str] = field(default_factory=Counter)
    trace_metric_buckets: dict[tuple[str, str, int], dict[str, int | float]] = field(
        default_factory=dict
    )
    impact_entities: set[str] = field(default_factory=set)


def _parse_datetime(value: object, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty ISO datetime")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include an explicit UTC offset")
    return parsed.astimezone(TZ)


def load_case_specs(path: Path) -> list[CaseSpec]:
    """Load a strict label-free case specification."""
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Telecom case spec must be an object")
    payload = cast(dict[str, object], raw)
    if set(payload) != {"source_revision", "cases"}:
        raise ValueError("Telecom case spec must contain source_revision and cases")
    revision = payload["source_revision"]
    if revision != SOURCE_REVISION:
        raise ValueError("Telecom case spec source revision mismatch")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Telecom case spec cases must be a non-empty array")
    required = {
        "row_id",
        "task_index",
        "question",
        "reference_start",
        "reference_end",
        "incident_start",
        "incident_end",
    }
    specs: list[CaseSpec] = []
    for raw_case in cast(list[object], raw_cases):
        if not isinstance(raw_case, dict) or set(raw_case) != required:
            raise ValueError("each Telecom case must contain exactly the public fields")
        case = cast(dict[str, object], raw_case)
        row_id = case["row_id"]
        if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
            raise ValueError("row_id must be a non-negative integer")
        task_index = case["task_index"]
        question = case["question"]
        if not isinstance(task_index, str) or not task_index.strip():
            raise ValueError("task_index must be a non-empty string")
        if task_index not in TASK_OUTPUTS:
            raise ValueError(f"unsupported Telecom task_index: {task_index}")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        reference_start = _parse_datetime(
            case["reference_start"], field_name="reference_start"
        )
        reference_end = _parse_datetime(
            case["reference_end"], field_name="reference_end"
        )
        incident_start = _parse_datetime(
            case["incident_start"], field_name="incident_start"
        )
        incident_end = _parse_datetime(case["incident_end"], field_name="incident_end")
        if not (
            reference_start < reference_end == incident_start < incident_end
            and reference_end - reference_start == timedelta(minutes=30)
            and incident_end - incident_start == timedelta(minutes=30)
        ):
            raise ValueError(
                "Telecom cases require adjacent 30-minute reference and "
                "incident windows"
            )
        specs.append(
            CaseSpec(
                row_id=row_id,
                task_index=task_index,
                question=question,
                reference_start=reference_start,
                reference_end=reference_end,
                incident_start=incident_start,
                incident_end=incident_end,
                source_revision=cast(str, revision),
            )
        )
    if len({spec.row_id for spec in specs}) != len(specs):
        raise ValueError("Telecom case row_id values must be unique")
    if len({spec.case_id for spec in specs}) != len(specs):
        raise ValueError("Telecom case IDs must be unique")
    return sorted(specs, key=lambda spec: spec.row_id)


def _source_day(source_day: Path, specs: list[CaseSpec]) -> Path:
    root = source_day.expanduser().resolve(strict=True)
    if not root.is_dir() or root.is_symlink():
        raise ValueError("source-day must be a regular directory")
    expected_day = {spec.incident_start.strftime("%Y_%m_%d") for spec in specs}
    if expected_day != {root.name}:
        raise ValueError("source-day name must match every case incident date")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != ".DS_Store"
    }
    if actual != set(SOURCE_FILES):
        raise ValueError(
            f"Telecom source-day file set mismatch: "
            f"missing={sorted(set(SOURCE_FILES) - actual)}, "
            f"extra={sorted(actual - set(SOURCE_FILES))}"
        )
    for relative, expected_header in SOURCE_FILES.items():
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError(f"Telecom source file must be regular: {relative}")
        with candidate.open(newline="", encoding="utf-8-sig") as stream:
            header = next(csv.reader(stream), None)
        if header != list(expected_header):
            raise ValueError(f"Telecom source columns mismatch: {relative}")
    return root


def _entity_kind(original_id: str) -> str:
    if re.fullmatch(r"os_\d{3}", original_id):
        return "node"
    if re.fullmatch(r"docker_\d{3}", original_id):
        return "pod"
    if re.fullmatch(r"db_\d{3}", original_id):
        return "service"
    return "auxiliary"


def _canonical_id(
    original_id: str, *, evidence_role: str = ROOT_CAUSE_CANDIDATE_ROLE
) -> str:
    if evidence_role == IMPACT_ONLY_ROLE:
        return f"telecom:business_entry:{original_id}"
    return f"telecom:{_entity_kind(original_id)}:{original_id}"


def _matching_states(states: list[CaseState], timestamp_ms: int) -> Iterator[CaseState]:
    timestamp = timestamp_ms / 1000
    for state in states:
        if (
            state.spec.allowed_start.timestamp()
            <= timestamp
            < state.spec.allowed_end.timestamp()
        ):
            yield state


def _period(state: CaseState, timestamp_ms: int) -> str:
    timestamp = timestamp_ms / 1000
    return (
        "reference" if timestamp < state.spec.incident_start.timestamp() else "incident"
    )


def _timestamp_seconds(timestamp_ms: int) -> str:
    return (
        str(timestamp_ms // 1000)
        if timestamp_ms % 1000 == 0
        else f"{timestamp_ms / 1000:.3f}"
    )


def _record_metric(
    state: CaseState,
    *,
    source_member: str,
    source_record: int,
    timestamp_ms: int,
    entity_id: str,
    metric_id: str,
    value: float,
    evidence_role: str,
) -> None:
    if (
        evidence_role == ROOT_CAUSE_CANDIDATE_ROLE
        and entity_id not in TELECOM_COMPONENTS
    ):
        return
    if evidence_role not in {ROOT_CAUSE_CANDIDATE_ROLE, IMPACT_ONLY_ROLE}:
        raise ValueError(f"unsupported Telecom metric evidence role: {evidence_role}")
    canonical_id = _canonical_id(entity_id, evidence_role=evidence_role)
    state.metric_writer.writerow(
        {
            "timestamp": _timestamp_seconds(timestamp_ms),
            "cmdb_id": entity_id,
            "kpi_name": metric_id,
            "value": format(value, ".15g"),
            "_source_record": source_record,
            "_source_member": source_member,
        }
    )
    state.source_rows[source_member] += 1
    entity_key = canonical_id if evidence_role == IMPACT_ONLY_ROLE else entity_id
    state.entity_sources[entity_key].add(NORMALIZED_METRIC_PATH)
    state.entity_aliases[entity_key][NORMALIZED_METRIC_PATH].add(entity_id)
    if evidence_role == IMPACT_ONLY_ROLE:
        state.impact_entities.add(entity_id)
    key = (canonical_id, metric_id)
    series = state.metric_series.setdefault(
        key,
        MetricSeries(
            canonical_id=canonical_id,
            entity_id=entity_id,
            metric_id=metric_id,
            evidence_role=evidence_role,
        ),
    )
    if series.evidence_role != evidence_role:
        raise ValueError("Telecom metric series cannot mix evidence roles")
    series.source_members.add(source_member)
    timestamp = timestamp_ms / 1000
    period = _period(state, timestamp_ms)
    series.timestamps.append(timestamp)
    series.periods[period].append(value)
    if len(series.records[period]) < 5:
        series.records[period].append(source_record)


def _scan_metrics(source_day: Path, states: list[CaseState]) -> None:
    for relative in sorted(path for path in SOURCE_FILES if path.startswith("metric/")):
        with (source_day / relative).open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for source_record, row in enumerate(reader, 2):
                if relative == "metric/metric_app.csv":
                    timestamp_ms = int(row["startTime"])
                    entity_id = row["serviceName"]
                    evidence_role = IMPACT_ONLY_ROLE
                    metrics = (
                        (name, row[name])
                        for name in ("avg_time", "num", "succee_num", "succee_rate")
                    )
                else:
                    timestamp_ms = int(row["timestamp"])
                    entity_id = row["cmdb_id"]
                    evidence_role = ROOT_CAUSE_CANDIDATE_ROLE
                    metrics = ((row["name"], row["value"]),)
                matching = list(_matching_states(states, timestamp_ms))
                if not matching:
                    continue
                for metric_id, raw_value in metrics:
                    try:
                        value = float(raw_value)
                    except ValueError as exc:
                        raise ValueError(
                            f"non-numeric Telecom metric at {relative}:{source_record}"
                        ) from exc
                    for state in matching:
                        _record_metric(
                            state,
                            source_member=relative,
                            source_record=source_record,
                            timestamp_ms=timestamp_ms,
                            entity_id=entity_id,
                            metric_id=metric_id,
                            value=value,
                            evidence_role=evidence_role,
                        )


def _trace_entity(row: dict[str, str]) -> str:
    data_source = row["dsName"].strip()
    if row["callType"].casefold() == "jdbc" and data_source.startswith("db_"):
        return data_source
    return row["cmdb_id"].strip()


def _trace_status(value: str) -> str:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "success"}:
        return "0"
    if normalized in {"false", "0", "failure", "failed"}:
        return "2"
    raise ValueError(f"unsupported Telecom trace success value: {value!r}")


def _scan_trace(source_day: Path, states: list[CaseState]) -> None:
    relative = NORMALIZED_TRACE_PATH
    with (source_day / relative).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        for source_record, row in enumerate(reader, 2):
            timestamp_ms = int(row["startTime"])
            matching = list(_matching_states(states, timestamp_ms))
            if not matching:
                continue
            entity_id = _trace_entity(row)
            if entity_id not in TELECOM_COMPONENTS:
                continue
            trace_id = row["traceId"].strip()
            span_id = row["id"].strip()
            parent_span = row["pid"].strip()
            if not trace_id or not span_id:
                raise ValueError(
                    f"empty Telecom trace identity at {relative}:{source_record}"
                )
            operation_name = (
                row["dsName"].strip()
                or row["serviceName"].strip()
                or row["callType"].strip()
            )
            normalized = {
                "timestamp": timestamp_ms,
                "cmdb_id": entity_id,
                "span_id": span_id,
                "trace_id": trace_id,
                "duration": row["elapsedTime"],
                "type": row["callType"],
                "status_code": _trace_status(row["success"]),
                "operation_name": operation_name,
                "parent_span": parent_span,
                "_source_record": source_record,
                "_source_member": relative,
            }
            for state in matching:
                state.trace_writer.writerow(normalized)
                state.source_rows[relative] += 1
                state.entity_sources[entity_id].add(relative)
                state.entity_aliases[entity_id][relative].add(entity_id)
                key = (trace_id, span_id)
                indexed = (
                    _canonical_id(entity_id),
                    parent_span,
                    timestamp_ms,
                    source_record,
                    float(row["elapsedTime"]),
                    0.0 if normalized["status_code"] == "0" else 1.0,
                )
                previous = state.span_index.get(key)
                if previous is None:
                    state.span_index[key] = indexed
                elif previous == indexed:
                    state.duplicate_span_keys += 1
                else:
                    state.span_conflicts.add(key)
                if row["callType"].casefold() == "jdbc" and entity_id.startswith("db_"):
                    period = _period(state, timestamp_ms)
                    bucket_ms = timestamp_ms - timestamp_ms % 60_000
                    aggregate = state.trace_metric_buckets.setdefault(
                        (entity_id, period, bucket_ms),
                        {"calls": 0, "errors": 0, "source_record": source_record},
                    )
                    aggregate["calls"] = int(aggregate["calls"]) + 1
                    if normalized["status_code"] != "0":
                        aggregate["errors"] = int(aggregate["errors"]) + 1


def _write_trace_metrics(states: list[CaseState]) -> None:
    for state in states:
        for (entity_id, _period_name, timestamp_ms), aggregate in sorted(
            state.trace_metric_buckets.items()
        ):
            calls = int(aggregate["calls"])
            errors = int(aggregate["errors"])
            source_record = int(aggregate["source_record"])
            for metric_id, value in (
                ("trace_incoming_call_count", float(calls)),
                ("trace_incoming_error_count", float(errors)),
                (
                    "trace_incoming_error_rate",
                    float(errors) / calls if calls else 0.0,
                ),
            ):
                _record_metric(
                    state,
                    source_member=NORMALIZED_TRACE_PATH,
                    source_record=source_record,
                    timestamp_ms=timestamp_ms,
                    entity_id=entity_id,
                    metric_id=metric_id,
                    value=value,
                    evidence_role=ROOT_CAUSE_CANDIDATE_ROLE,
                )


def _range(start: datetime, end: datetime) -> dict[str, str]:
    return {"start": start.isoformat(), "end": end.isoformat()}


def _nodes(state: CaseState) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for original_id in TELECOM_COMPONENTS:
        sources = sorted(state.entity_sources.get(original_id, set()))
        aliases = state.entity_aliases.get(original_id, {})
        nodes.append(
            {
                "canonical_id": _canonical_id(original_id),
                "original_id": original_id,
                "entity_type": _entity_kind(original_id),
                "sources": sources,
                "source_aliases": {
                    source: sorted(values) for source, values in sorted(aliases.items())
                },
                "normalization": "exact Telecom public component identity",
                "public_candidate_match": original_id,
                "mapping_status": "source_exact",
                "mapping_evidence": (
                    "pinned OpenRCA Telecom public component vocabulary"
                ),
                "evidence_role": ROOT_CAUSE_CANDIDATE_ROLE,
            }
        )
    for original_id in sorted(state.impact_entities):
        canonical_id = _canonical_id(original_id, evidence_role=IMPACT_ONLY_ROLE)
        sources = sorted(state.entity_sources.get(canonical_id, set()))
        aliases = state.entity_aliases.get(canonical_id, {})
        nodes.append(
            {
                "canonical_id": canonical_id,
                "original_id": original_id,
                "entity_type": "business_entry",
                "sources": sources,
                "source_aliases": {
                    source: sorted(values) for source, values in sorted(aliases.items())
                },
                "normalization": "exact Telecom metric_app serviceName identity",
                "public_candidate_match": None,
                "mapping_status": IMPACT_ONLY_ROLE,
                "mapping_evidence": (
                    "metric_app business-entry telemetry is retained for impact "
                    "correlation and excluded from the root-cause candidate vocabulary"
                ),
                "evidence_role": IMPACT_ONLY_ROLE,
            }
        )
    return nodes


def _topology(state: CaseState, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    known = {str(node["canonical_id"]) for node in nodes}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    matched = missing = internal = sentinel = ambiguous = 0
    root_markers: Counter[str] = Counter()
    missing_samples: list[dict[str, Any]] = []
    for (trace_id, span_id), child in sorted(state.span_index.items()):
        child_key = (trace_id, span_id)
        parent_key = (trace_id, child[1])
        if child_key in state.span_conflicts or parent_key in state.span_conflicts:
            ambiguous += 1
            continue
        normalized_parent = child[1].strip().casefold()
        if normalized_parent in ROOT_MARKERS:
            sentinel += 1
            root_markers[child[1]] += 1
            continue
        parent = state.span_index.get(parent_key)
        if parent is None:
            missing += 1
            if len(missing_samples) < 10:
                missing_samples.append(
                    {
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_id": child[1],
                        "source_record": child[3],
                    }
                )
            continue
        matched += 1
        if parent[0] == child[0]:
            internal += 1
            continue
        if parent[0] not in known or child[0] not in known:
            continue
        pair = (parent[0], child[0])
        edge = edges.setdefault(
            pair,
            {
                "source_id": pair[0],
                "target_id": pair[1],
                "relation_type": "calls",
                "fact_kind": "observed",
                "span_pairs": 0,
                "first_observed_ms": child[2],
                "last_observed_ms": child[2],
                "samples": [],
            },
        )
        edge["span_pairs"] += 1
        edge["first_observed_ms"] = min(edge["first_observed_ms"], child[2])
        edge["last_observed_ms"] = max(edge["last_observed_ms"], child[2])
        samples = cast(list[dict[str, object]], edge["samples"])
        if len(samples) < 3:
            samples.append(
                {
                    "trace_id": trace_id,
                    "parent_span_id": child[1],
                    "child_span_id": span_id,
                    "parent_source_record": parent[3],
                    "child_source_record": child[3],
                }
            )
    ordered_edges: list[dict[str, Any]] = []
    for index, pair in enumerate(sorted(edges)):
        edge = dict(edges[pair])
        edge["edge_id"] = f"edge-{index:03d}"
        ordered_edges.append(edge)
    return {
        "snapshot_id": f"{state.spec.case_id}-topology-v1",
        "source": "Telecom traceId plus pid/id parent join",
        "valid_range": _range(state.spec.allowed_start, state.spec.allowed_end),
        "nodes": nodes,
        "edges": ordered_edges,
        "coverage": {
            "status": "partial",
            "unique_spans": len(state.span_index),
            "matched_parent_spans": matched,
            "missing_parent_spans": missing,
            "root_like_sentinel_spans": sentinel,
            "root_marker_values": dict(sorted(root_markers.items())),
            "intra_entity_pairs": internal,
            "duplicate_span_keys": state.duplicate_span_keys,
            "conflicting_span_keys": len(state.span_conflicts),
            "ambiguous_links_skipped": ambiguous,
            "missing_parent_samples": missing_samples,
            "note": (
                "partial component topology from source trace parent joins within "
                "the allowed window"
            ),
        },
    }


def _metric_stats(state: CaseState) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    for key in sorted(state.metric_series):
        series = state.metric_series[key]
        timestamps = sorted(series.timestamps)
        deltas = [
            right - left
            for left, right in zip(timestamps, timestamps[1:])
            if right > left
        ]
        item: dict[str, Any] = {
            "canonical_id": series.canonical_id,
            "entity_id": series.entity_id,
            "source_entity_keys": [series.entity_id],
            "metric_id": series.metric_id,
            "evidence_role": series.evidence_role,
            "source_members": sorted(series.source_members),
            "max_observed_gap_seconds": max(deltas) if deltas else None,
            "median_observed_interval_seconds": (
                statistics.median(deltas) if deltas else None
            ),
            "value_unit": "source-native; not inferred",
        }
        for period in ("reference", "incident"):
            values = series.periods[period]
            if not values:
                continue
            item[period] = {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "p95": _percentile(values, 0.95),
                "p99": _percentile(values, 0.99),
                "example_source_records": series.records[period],
            }
        stats.append(item)
    return stats


def _metric_bindings(state: CaseState) -> list[dict[str, Any]]:
    """Describe every normalized series, including its source and evidence role."""
    return [
        {
            "canonical_id": series.canonical_id,
            "metric_id": series.metric_id,
            "source_path": NORMALIZED_METRIC_PATH,
            "layout": "long",
            "entity_field": "cmdb_id",
            "original_entity_key": series.entity_id,
            "metric_field": "kpi_name",
            "source_metric": series.metric_id,
            "value_field": "value",
            "timestamp_unit": "seconds",
            "source_member_filter": sorted(series.source_members),
            "evidence_role": series.evidence_role,
        }
        for series in sorted(
            state.metric_series.values(),
            key=lambda item: (item.canonical_id, item.metric_id),
        )
    ]


def _trace_edge_stats(
    state: CaseState, topology: dict[str, Any]
) -> list[dict[str, Any]]:
    edge_by_pair = {
        (str(edge["source_id"]), str(edge["target_id"])): str(edge["edge_id"])
        for edge in cast(list[dict[str, object]], topology["edges"])
        if edge["relation_type"] == "calls"
    }
    values: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = {
        pair: {
            period: {
                "durations": [],
                "errors": [],
                "timestamps": [],
                "source_records": [],
            }
            for period in ("reference", "incident")
        }
        for pair in edge_by_pair
    }
    for key, child in state.span_index.items():
        if key in state.span_conflicts:
            continue
        parent_key = (key[0], child[1])
        parent = state.span_index.get(parent_key)
        if parent is None or parent_key in state.span_conflicts:
            continue
        pair = (parent[0], child[0])
        if pair not in values:
            continue
        period = _period(state, child[2])
        bucket = values[pair][period]
        bucket["durations"].append(child[4])
        bucket["errors"].append(child[5])
        bucket["timestamps"].append(child[2] / 1000)
        bucket["source_records"].append(float(child[3]))

    profiles: list[dict[str, Any]] = []
    for pair in sorted(values):
        periods = values[pair]
        reference = periods["reference"]
        incident = periods["incident"]
        reference_durations = reference["durations"]
        incident_durations = incident["durations"]
        reference_p95 = (
            _percentile(reference_durations, 0.95) if reference_durations else None
        )
        reference_p99 = (
            _percentile(reference_durations, 0.99) if reference_durations else None
        )
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
        profiles.append(
            {
                "edge_id": edge_by_pair[pair],
                "source_id": pair[0],
                "target_id": pair[1],
                "reference": {
                    "call_count": len(reference_durations),
                    "error_count": int(sum(reference["errors"])),
                    "error_rate": reference_error_rate,
                    "latency_p50": (
                        _percentile(reference_durations, 0.50)
                        if reference_durations
                        else None
                    ),
                    "latency_p95": reference_p95,
                    "latency_p99": reference_p99,
                },
                "incident": {
                    "call_count": len(incident_durations),
                    "error_count": int(sum(incident["errors"])),
                    "error_rate": incident_error_rate,
                    "latency_p50": (
                        _percentile(incident_durations, 0.50)
                        if incident_durations
                        else None
                    ),
                    "latency_p95": (
                        _percentile(incident_durations, 0.95)
                        if incident_durations
                        else None
                    ),
                    "latency_p99": (
                        _percentile(incident_durations, 0.99)
                        if incident_durations
                        else None
                    ),
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
                "source_records": sorted(
                    {
                        int(record)
                        for record in (
                            reference["source_records"][:5]
                            + incident["source_records"][:5]
                        )
                    }
                ),
            }
        )
    return profiles


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _finish_case(state: CaseState) -> None:
    state.metric_stream.flush()
    state.trace_stream.flush()
    nodes = _nodes(state)
    topology = _topology(state, nodes)
    metric_stats = _metric_stats(state)
    metric_bindings = _metric_bindings(state)
    trace_edge_stats = _trace_edge_stats(state, topology)
    _write_json(
        state.public_input / "task.json",
        {
            "task_id": state.spec.case_id,
            "environment": "openrca_telecom",
            "question": state.spec.question,
            "incident_range": _range(
                state.spec.incident_start, state.spec.incident_end
            ),
            "timezone": "Asia/Shanghai",
            "allowed_query_range": _range(
                state.spec.allowed_start, state.spec.allowed_end
            ),
            "reference_range": {
                **_range(state.spec.reference_start, state.spec.reference_end),
                "health_status": "unverified",
            },
            "entry_entities": [],
            "allowed_scope": {
                "entities": [str(node["canonical_id"]) for node in nodes],
                "datasets": [],
                "tools": [],
                "telemetry_root": "telemetry",
                "entity_inventory_ref": "entity_mapping.json",
            },
            "requested_outputs": list(TASK_OUTPUTS[state.spec.task_index]),
            "dataset_manifest_ref": "manifest.json",
            "source_row_id": state.spec.row_id,
            "task_index": state.spec.task_index,
            "expected_fault_count": 1,
            "fault_count_source": "label-free public case spec",
            "data_mode": "offline_window_slice",
            "source_commit": state.spec.source_revision,
        },
    )
    _write_json(state.public_input / "entity_mapping.json", nodes)
    _write_json(state.public_input / "topology.json", topology)
    _write_json(
        state.public_input / "vocabulary.json",
        {
            "entities": list(TELECOM_COMPONENTS),
            "fault_types": list(TELECOM_FAULT_TYPES),
            "source": {
                "repository": "microsoft/OpenRCA",
                "revision": state.spec.source_revision,
                "path": ("rca/baseline/rca_agent/prompt/basic_prompt_Telecom.py"),
            },
        },
    )
    _write_json(
        state.public_input / "metric_catalog.json",
        {
            "duration_unit": "milliseconds",
            "log_names": [],
            "metrics": sorted(
                {item.metric_id for item in state.metric_series.values()}
            ),
            "metric_bindings": metric_bindings,
            "schemas": {
                "metric": "long rows: cmdb_id, kpi_name, value",
                "trace": "traceId/id/pid normalized to trace_id/span_id/parent_span",
            },
        },
    )
    stats_path = state.public_input / "metric_window_stats.jsonl"
    with stats_path.open("w", encoding="utf-8", newline="") as stream:
        for item in metric_stats:
            stream.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    trace_stats_path = state.public_input / "trace_edge_window_stats.jsonl"
    with trace_stats_path.open("w", encoding="utf-8", newline="") as stream:
        for item in trace_edge_stats:
            stream.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            )
    _write_json(
        state.public_input / "dataset_profile.json",
        {
            "dataset": "OpenRCA Telecom",
            "source_date": state.spec.incident_start.strftime("%Y_%m_%d"),
            "source_rows_selected": dict(sorted(state.source_rows.items())),
            "candidate_component_count": len(TELECOM_COMPONENTS),
            "impact_only_entity_count": len(state.impact_entities),
            "impact_only_metric_series_count": sum(
                item.evidence_role == IMPACT_ONLY_ROLE
                for item in state.metric_series.values()
            ),
            "metric_series_count": len(metric_stats),
            "topology_edge_count": len(topology["edges"]),
            "trace_edge_profile_count": len(trace_edge_stats),
            "prepared_trace_contract": {
                "join_keys": ["trace_id", "parent_span", "span_id"],
                "rows_filtered_to_allowed_window": True,
                "source_path": NORMALIZED_TRACE_PATH,
                "timestamp_unit": "milliseconds",
                "jdbc_dsName_as_database_component": True,
            },
            "limitations": [
                (
                    "Only exact public root-cause candidates enter the candidate "
                    "vocabulary; metric_app business-entry entities are impact-only."
                ),
                "Rows use the exact half-open allowed query range.",
                "The source dataset contains no log files.",
                "Evaluation labels and label-like source files are excluded.",
            ],
        },
    )
    artifacts = [
        {
            "path": path.relative_to(state.public_input).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(state.public_input.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    ]
    _write_json(
        state.public_input / "manifest.json",
        {
            "schema_version": TELECOM_SLICE_SCHEMA_VERSION,
            "dataset_id": state.spec.case_id,
            "source_commit": state.spec.source_revision,
            "label_policy": "label-free public case spec; evaluation labels excluded",
            "artifacts": artifacts,
        },
    )
    TelecomEnvironment(state.public_input)


def prepare_telecom_suite(
    source_day: Path, case_spec_path: Path, output_root: Path
) -> list[Path]:
    """Prepare several Telecom cases while scanning the raw source day once."""
    specs = load_case_specs(case_spec_path)
    verified_source_day = _source_day(source_day, specs)
    resolved_output = output_root.expanduser().resolve()
    if resolved_output.is_symlink():
        raise ValueError("output-root must not be a symlink")
    try:
        resolved_output.relative_to(REPO_ROOT)
    except ValueError:
        pass
    else:
        raise ValueError("output-root must be outside the git worktree")
    resolved_output.mkdir(parents=True, exist_ok=True)
    if (
        verified_source_day == resolved_output
        or verified_source_day in resolved_output.parents
    ):
        raise ValueError("output-root must not contain the source day")
    stage = Path(tempfile.mkdtemp(prefix=".tmp-openrca-telecom-", dir=resolved_output))
    staged_cases: list[Path] = []
    try:
        with ExitStack() as stack:
            states: list[CaseState] = []
            for spec in specs:
                public_input = stage / spec.case_id / "public_input"
                metric_path = public_input / "telemetry" / NORMALIZED_METRIC_PATH
                trace_path = public_input / "telemetry" / NORMALIZED_TRACE_PATH
                metric_path.parent.mkdir(parents=True, exist_ok=True)
                trace_path.parent.mkdir(parents=True, exist_ok=True)
                metric_stream = stack.enter_context(
                    metric_path.open("w", newline="", encoding="utf-8")
                )
                trace_stream = stack.enter_context(
                    trace_path.open("w", newline="", encoding="utf-8")
                )
                metric_writer = csv.DictWriter(
                    metric_stream,
                    fieldnames=[
                        "timestamp",
                        "cmdb_id",
                        "kpi_name",
                        "value",
                        "_source_record",
                        "_source_member",
                    ],
                    lineterminator="\n",
                )
                trace_writer = csv.DictWriter(
                    trace_stream,
                    fieldnames=[
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
                    ],
                    lineterminator="\n",
                )
                metric_writer.writeheader()
                trace_writer.writeheader()
                states.append(
                    CaseState(
                        spec=spec,
                        public_input=public_input,
                        metric_stream=metric_stream,
                        metric_writer=metric_writer,
                        trace_stream=trace_stream,
                        trace_writer=trace_writer,
                    )
                )
            _scan_metrics(verified_source_day, states)
            _scan_trace(verified_source_day, states)
            _write_trace_metrics(states)
            for state in states:
                _finish_case(state)
                staged_cases.append(state.public_input.parent)
        published: list[Path] = []
        for staged in staged_cases:
            destination = resolved_output / staged.name
            if destination.exists() or destination.is_symlink():
                raise FileExistsError(f"output case already exists: {destination}")
            os.replace(staged, destination)
            published.append(destination / "public_input")
        return published
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-day", required=True, type=Path)
    parser.add_argument("--case-spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(argv)
    public_inputs = prepare_telecom_suite(
        args.source_day, args.case_spec, args.output_root
    )
    print(
        json.dumps(
            {
                "public_inputs": [str(path) for path in public_inputs],
                "source_revision": SOURCE_REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
