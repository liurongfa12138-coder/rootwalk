#!/usr/bin/env python3
# pyright: basic
"""Prepare one label-free OpenRCA Market public_input from a local source day."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterator, cast
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rootwalk.openrca import (  # noqa: E402
    MarketEnvironment,
    market_output_component,
    market_service_component,
)

DEFAULT_SOURCE_SCHEMA_PATH = (
    REPO_ROOT.parent / "rootwalk-data" / "openrca-market-public-schema-v1.json"
)
EXPECTED_SOURCE_SCHEMA_SHA256 = (
    "d5a68e3c85a06a3f3140579ceb0149cc14a2440560b35ef0ab7ca8ea786ea593"
)
EXPECTED_SOURCE_REVISION = "7a29a7c3600cc07b7760a6c5097d36ac8573f039"
MARKET_CANDIDATE_SOURCE = {
    "repository": "microsoft/OpenRCA",
    "revision": "c1bd4af7f635171a1c31cdd567c07d698dff6abc",
    "path": "rca/baseline/rca_agent/prompt/basic_prompt_Market.py",
    "sha256": "d54e2f710290019446c3b9a6d4636f0c2140aba14096c081ff7d2ca1d82fb23e",
}
MARKET_COMPONENTS = (
    "node-1",
    "node-2",
    "node-3",
    "node-4",
    "node-5",
    "node-6",
    "frontend-0",
    "frontend-1",
    "frontend-2",
    "frontend2-0",
    "shippingservice-0",
    "shippingservice-1",
    "shippingservice-2",
    "shippingservice2-0",
    "checkoutservice-0",
    "checkoutservice-1",
    "checkoutservice-2",
    "checkoutservice2-0",
    "currencyservice-0",
    "currencyservice-1",
    "currencyservice-2",
    "currencyservice2-0",
    "adservice-0",
    "adservice-1",
    "adservice-2",
    "adservice2-0",
    "emailservice-0",
    "emailservice-1",
    "emailservice-2",
    "emailservice2-0",
    "cartservice-0",
    "cartservice-1",
    "cartservice-2",
    "cartservice2-0",
    "productcatalogservice-0",
    "productcatalogservice-1",
    "productcatalogservice-2",
    "productcatalogservice2-0",
    "recommendationservice-0",
    "recommendationservice-1",
    "recommendationservice-2",
    "recommendationservice2-0",
    "paymentservice-0",
    "paymentservice-1",
    "paymentservice-2",
    "paymentservice2-0",
    "frontend",
    "shippingservice",
    "checkoutservice",
    "currencyservice",
    "adservice",
    "emailservice",
    "cartservice",
    "productcatalogservice",
    "recommendationservice",
    "paymentservice",
)
MARKET_FAULT_TYPES = (
    "container CPU load",
    "container memory load",
    "container network packet retransmission",
    "container network packet corruption",
    "container network latency",
    "container packet loss",
    "container process termination",
    "container read I/O load",
    "container write I/O load",
    "node CPU load",
    "node CPU spike",
    "node memory consumption",
    "node disk read I/O consumption",
    "node disk write I/O consumption",
    "node disk space consumption",
)
MARKET_NODE_COMPONENTS = frozenset(
    component for component in MARKET_COMPONENTS if re.fullmatch(r"node-\d+", component)
)
MARKET_POD_COMPONENTS = frozenset(
    component
    for component in MARKET_COMPONENTS
    if component not in MARKET_NODE_COMPONENTS and re.search(r"-\d+$", component)
)
MARKET_SERVICE_COMPONENTS = frozenset(MARKET_COMPONENTS) - (
    MARKET_NODE_COMPONENTS | MARKET_POD_COMPONENTS
)
LABEL_LIKE_FILE_NAMES = frozenset(
    {
        "query.csv",
        "record.csv",
        "selected_source.json",
        "selected_sources.json",
        "selected-source.json",
        "selected-source.csv",
        "selected_sources.csv",
        "selected-source.txt",
    }
)
LABEL_LIKE_KEYS = frozenset(
    {
        "answer",
        "groundtruth",
        "label",
        "labels",
        "record",
        "scoringpoints",
        "selectedsource",
        "selectedsources",
    }
)
FORBIDDEN_PATH_PARTS = frozenset(
    {
        "openrcamarketstrictunseenaudit",
        "selectedsource",
        "selectedsources",
    }
)
ROOT_MARKERS = frozenset({"", "0", "-1", "none", "null", "root"})
TZ = ZoneInfo("Asia/Shanghai")
csv.field_size_limit(16 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class CaseSpec:
    cloudbed: str
    row_id: int
    question: str
    allowed_start: datetime
    allowed_end: datetime
    reference_start: datetime
    reference_end: datetime
    incident_start: datetime
    incident_end: datetime
    source_revision: str
    public_component_vocabulary: tuple[str, ...]
    public_fault_vocabulary: tuple[str, ...]

    @property
    def case_id(self) -> str:
        return f"openrca-market-{self.cloudbed}-row-{self.row_id:04d}"

    @property
    def source_date(self) -> str:
        return self.allowed_start.astimezone(TZ).strftime("%Y_%m_%d")


@dataclass(slots=True)
class MetricSeries:
    canonical_id: str
    entity_ids: set[str]
    metric_id: str
    source_path: str
    timestamps: list[float] = field(default_factory=list)
    periods: dict[str, list[float]] = field(
        default_factory=lambda: {"reference": [], "incident": []}
    )
    records: dict[str, list[int]] = field(
        default_factory=lambda: {"reference": [], "incident": []}
    )


@dataclass(slots=True)
class PreparedState:
    spec: CaseSpec
    source_day: Path
    source_day_fd: int
    public_input: Path
    source_schema: dict[str, Any]
    source_schema_bytes: bytes
    file_rows: dict[str, int] = field(default_factory=dict)
    file_records: dict[str, int] = field(default_factory=dict)
    entity_sources: dict[tuple[str, str], set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    entity_aliases: dict[tuple[str, str], dict[str, set[str]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(set))
    )
    log_names: Counter[str] = field(default_factory=Counter)
    metric_series: dict[tuple[str, str, str], MetricSeries] = field(
        default_factory=dict
    )
    span_index: dict[tuple[str, str], tuple[str, str, int, int]] = field(
        default_factory=dict
    )
    span_conflicts: set[tuple[str, str]] = field(default_factory=set)
    duplicate_span_keys: int = 0


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _jsonl_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_entries(root: Path) -> list[dict[str, object]]:
    _ensure_tree_has_no_symlinks(root, label="case-dir")
    entries: list[dict[str, object]] = []
    for path in sorted(
        candidate for candidate in root.rglob("*") if candidate.is_file()
    ):
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _normalize_token(value: str) -> str:
    return re.sub(r"[-_\s]+", "", value.casefold())


def _reject_forbidden_path(path: Path, *, label: str) -> None:
    normalized_parts = {_normalize_token(part) for part in path.parts}
    overlap = sorted(normalized_parts & FORBIDDEN_PATH_PARTS)
    if overlap:
        raise ValueError(f"{label} points at a forbidden path segment: {overlap[0]}")


def _reject_label_content(value: object, *, location: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = _normalize_token(str(key))
            if normalized in LABEL_LIKE_KEYS:
                raise ValueError(f"label-like key is not allowed in {location}: {key}")
            _reject_label_content(item, location=location)
        return
    if isinstance(value, list):
        for item in value:
            _reject_label_content(item, location=location)


def _read_verified_regular_file_bytes(path: Path, *, label: str) -> tuple[Path, bytes]:
    raw_path = path.expanduser()
    _reject_forbidden_path(raw_path, label=label)
    if raw_path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = raw_path.resolve(strict=True)
    _reject_forbidden_path(resolved, label=label)
    parent_fd = -1
    file_fd = -1
    try:
        parent_fd = os.open(
            resolved.parent,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        file_fd = os.open(
            resolved.name,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=parent_fd,
        )
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ValueError(f"{label} must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return resolved, b"".join(chunks)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if parent_fd >= 0:
            os.close(parent_fd)


def _load_verified_source_schema(
    source_schema_path: Path | None,
) -> tuple[Path, dict[str, Any], bytes]:
    resolved, payload = _read_verified_regular_file_bytes(
        source_schema_path or DEFAULT_SOURCE_SCHEMA_PATH,
        label="source-schema",
    )
    schema_sha256 = hashlib.sha256(payload).hexdigest()
    if schema_sha256 != EXPECTED_SOURCE_SCHEMA_SHA256:
        raise ValueError(
            "source-schema sha256 does not match the verified Market schema"
        )
    schema = cast(dict[str, Any], json.loads(payload.decode("utf-8")))
    revision = str(cast(dict[str, object], schema["source"])["revision"])
    if revision != EXPECTED_SOURCE_REVISION:
        raise ValueError(
            "source-schema revision does not match the verified Market schema"
        )
    return resolved, schema, payload


VERIFIED_SOURCE_SCHEMA_PATH, SOURCE_SCHEMA, SOURCE_SCHEMA_BYTES = (
    _load_verified_source_schema(None)
)
MARKET_SOURCE_REVISION = str(
    cast(dict[str, object], SOURCE_SCHEMA["source"])["revision"]
)
EXPECTED_FILES = tuple(sorted(cast(dict[str, object], SOURCE_SCHEMA["files"])))


def _reject_overlap(
    left: Path,
    right: Path,
    *,
    left_label: str,
    right_label: str,
) -> None:
    if _is_relative_to(left, right) or _is_relative_to(right, left):
        raise ValueError(f"{left_label} and {right_label} must not overlap")


def _ensure_tree_has_no_symlinks(root: Path, *, label: str) -> None:
    if root.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            relative = candidate.relative_to(root).as_posix()
            raise ValueError(f"{label} contains a symlink: {relative}")


@contextmanager
def _open_source_member(source_day_fd: int, relative: str) -> Iterator[TextIOWrapper]:
    dir_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    dir_fds: list[int] = []
    file_fd = -1
    try:
        current_fd = os.dup(source_day_fd)
        dir_fds.append(current_fd)
        for part in Path(relative).parts[:-1]:
            next_fd = os.open(part, dir_flags, dir_fd=current_fd)
            if not stat.S_ISDIR(os.fstat(next_fd).st_mode):
                raise ValueError(
                    f"source-day member parent must be a directory: {relative}"
                )
            dir_fds.append(next_fd)
            current_fd = next_fd
        file_fd = os.open(Path(relative).name, file_flags, dir_fd=current_fd)
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ValueError(f"source-day member must be a regular file: {relative}")
        with os.fdopen(file_fd, "r", encoding="utf-8", newline="") as stream:
            file_fd = -1
            yield cast(TextIOWrapper, stream)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        for fd in reversed(dir_fds):
            os.close(fd)


def _parse_decimal(value: str, *, field_name: str) -> Decimal:
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return result


def _parse_range(value: object, *, name: str) -> tuple[datetime, datetime]:
    if not isinstance(value, dict) or value.get("end_exclusive") is not True:
        raise ValueError(f"{name} must be an explicit half-open range")
    start_raw = value.get("start")
    end_raw = value.get("end")
    if not isinstance(start_raw, str) or not isinstance(end_raw, str):
        raise ValueError(f"{name} start/end must be strings")
    try:
        start = datetime.fromisoformat(start_raw)
        end = datetime.fromisoformat(end_raw)
    except ValueError as exc:
        raise ValueError(f"{name} must use ISO timestamps") from exc
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError(f"{name} must include a timezone")
    if start.utcoffset() != timedelta(hours=8) or end.utcoffset() != timedelta(hours=8):
        raise ValueError(f"{name} must use Asia/Shanghai UTC+8")
    if end <= start:
        raise ValueError(f"{name} end must be after start")
    return start, end


def _load_case_spec(case_spec_path: Path) -> CaseSpec:
    _resolved, payload = _read_verified_regular_file_bytes(
        case_spec_path,
        label="case-spec",
    )
    raw = cast(
        dict[str, object],
        json.loads(payload.decode("utf-8")),
    )
    _reject_label_content(raw, location="case spec")
    expected = {
        "allowed_query_range",
        "cloudbed",
        "incident_range",
        "question",
        "reference_range",
        "row_id",
        "source_revision",
    }
    optional = {"public_component_vocabulary", "public_fault_vocabulary"}
    extra = sorted(set(raw) - expected - optional)
    missing = sorted(expected - set(raw))
    if missing:
        raise ValueError(f"case spec is missing required keys: {missing}")
    if extra:
        raise ValueError(f"case spec has unsupported keys: {extra}")

    cloudbed = raw["cloudbed"]
    question = raw["question"]
    row_id = raw["row_id"]
    source_revision = raw["source_revision"]
    if not isinstance(cloudbed, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", cloudbed
    ):
        raise ValueError("cloudbed must be a non-empty slug")
    if isinstance(row_id, bool) or not isinstance(row_id, int) or row_id < 0:
        raise ValueError("row_id must be a non-negative integer")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if not isinstance(source_revision, str):
        raise ValueError("source_revision must be a string")
    if source_revision != MARKET_SOURCE_REVISION:
        raise ValueError("source_revision must match the public Market schema revision")

    allowed_start, allowed_end = _parse_range(
        raw["allowed_query_range"], name="allowed_query_range"
    )
    reference_start, reference_end = _parse_range(
        raw["reference_range"], name="reference_range"
    )
    incident_start, incident_end = _parse_range(
        raw["incident_range"], name="incident_range"
    )
    if reference_end != incident_start:
        raise ValueError("reference_range must end exactly at incident_range start")
    if (allowed_start, allowed_end) != (reference_start, incident_end):
        raise ValueError(
            "allowed_query_range must exactly span reference_range + incident_range"
        )
    source_date = allowed_start.astimezone(TZ).date()
    if (allowed_end - timedelta(microseconds=1)).astimezone(TZ).date() != source_date:
        raise ValueError(
            "single-day Market prep requires all ranges to stay in one date"
        )

    def vocabulary(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        raw_items = raw.get(name)
        if raw_items is None:
            return default
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError(f"{name} must be a non-empty string array")
        items = tuple(
            item.strip() for item in raw_items if isinstance(item, str) and item.strip()
        )
        if len(items) != len(raw_items) or len(set(items)) != len(items):
            raise ValueError(f"{name} must contain unique non-empty strings")
        return items

    components = vocabulary("public_component_vocabulary", MARKET_COMPONENTS)
    fault_types = vocabulary("public_fault_vocabulary", MARKET_FAULT_TYPES)

    return CaseSpec(
        cloudbed=cloudbed,
        row_id=row_id,
        question=question.strip(),
        allowed_start=allowed_start,
        allowed_end=allowed_end,
        reference_start=reference_start,
        reference_end=reference_end,
        incident_start=incident_start,
        incident_end=incident_end,
        source_revision=source_revision,
        public_component_vocabulary=components,
        public_fault_vocabulary=fault_types,
    )


def _validate_source_day(source_day: Path, spec: CaseSpec) -> tuple[Path, int]:
    expanded = source_day.expanduser()
    _reject_forbidden_path(expanded, label="source-day")
    if expanded.is_symlink():
        raise ValueError("source-day must not be a symlink")
    resolved = expanded.resolve(strict=True)
    _reject_forbidden_path(resolved, label="source-day")
    directory_fd = -1
    try:
        directory_fd = os.open(
            resolved,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
            raise ValueError("source-day must be a directory")
        if resolved.name != spec.source_date:
            raise ValueError(
                f"source-day directory name must match the case date {spec.source_date}"
            )
        actual_files = sorted(
            path.relative_to(resolved).as_posix()
            for path in resolved.rglob("*")
            if path.is_file()
        )
        for relative in actual_files:
            name = Path(relative).name.lower()
            if name in LABEL_LIKE_FILE_NAMES:
                raise ValueError(f"label-like file is not allowed: {relative}")
        if tuple(actual_files) != EXPECTED_FILES:
            missing = sorted(set(EXPECTED_FILES) - set(actual_files))
            extra = sorted(set(actual_files) - set(EXPECTED_FILES))
            raise ValueError(
                f"source-day file set mismatch: missing={missing}, extra={extra}"
            )
        for relative in EXPECTED_FILES:
            if (resolved / relative).is_symlink():
                raise ValueError(f"source-day member must not be a symlink: {relative}")
        return resolved, directory_fd
    except Exception:
        if directory_fd >= 0:
            os.close(directory_fd)
        raise


def _range_payload(start: datetime, end: datetime) -> dict[str, object]:
    return {"start": start.isoformat(), "end": end.isoformat(), "end_exclusive": True}


def _source_member(spec: CaseSpec, relative: str) -> str:
    return f"Market/{spec.cloudbed}/telemetry/{spec.source_date}/{relative}"


def _canonical_id(spec: CaseSpec, *, kind: str, original_id: str) -> str:
    return f"market:{spec.cloudbed}:{kind}:{original_id}"


def _market_identity(
    spec: CaseSpec, *, source_path: str, kind: str, original_id: str
) -> tuple[str, str]:
    if source_path == "metric/metric_mesh.csv":
        return "relation", original_id
    component = market_output_component(original_id)
    if component not in spec.public_component_vocabulary:
        return kind, original_id
    if component in MARKET_NODE_COMPONENTS:
        return "node", component
    if component in MARKET_POD_COMPONENTS:
        return "pod", component
    if component in MARKET_SERVICE_COMPONENTS:
        return "service", component
    return kind, original_id


def _record_period(spec: CaseSpec, seconds: Decimal) -> str | None:
    reference_start = Decimal(str(spec.reference_start.timestamp()))
    reference_end = Decimal(str(spec.reference_end.timestamp()))
    incident_end = Decimal(str(spec.incident_end.timestamp()))
    if reference_start <= seconds < reference_end:
        return "reference"
    if reference_end <= seconds < incident_end:
        return "incident"
    return None


def _write_filtered_copy(
    target_path: Path,
    *,
    header: list[str],
    state: PreparedState,
    relative: str,
    seconds_columns: bool,
) -> list[dict[str, str]]:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    selected_rows: list[dict[str, str]] = []
    allowed_start = Decimal(str(state.spec.allowed_start.timestamp()))
    allowed_end = Decimal(str(state.spec.allowed_end.timestamp()))
    allowed_start_ms = allowed_start * 1000
    allowed_end_ms = allowed_end * 1000
    with _open_source_member(state.source_day_fd, relative) as source_stream:
        reader = csv.DictReader(source_stream)
        if reader.fieldnames != header:
            raise ValueError(f"{relative} columns do not match the public schema")
        with target_path.open("w", newline="", encoding="utf-8") as target_stream:
            writer = csv.DictWriter(
                target_stream,
                fieldnames=[*header, "_source_record", "_source_member"],
            )
            writer.writeheader()
            for source_record, row in enumerate(reader, 1):
                timestamp = _parse_decimal(
                    row["timestamp"], field_name=f"{relative}.timestamp"
                )
                in_window = (
                    allowed_start <= timestamp < allowed_end
                    if seconds_columns
                    else allowed_start_ms <= timestamp < allowed_end_ms
                )
                state.file_records[relative] = source_record
                if not in_window:
                    continue
                selected = dict(row)
                selected["_source_record"] = str(source_record)
                selected["_source_member"] = _source_member(state.spec, relative)
                writer.writerow(selected)
                selected_rows.append(selected)
    state.file_rows[relative] = len(selected_rows)
    return selected_rows


def _entity_sources(
    state: PreparedState,
    *,
    kind: str,
    original_id: str,
    source_path: str,
) -> None:
    identity = _market_identity(
        state.spec,
        source_path=source_path,
        kind=kind,
        original_id=original_id,
    )
    state.entity_sources[identity].add(source_path)
    state.entity_aliases[identity][source_path].add(original_id)
    if identity[0] == "pod" and source_path.startswith(("log/", "trace/")):
        service = market_service_component(identity[1])
        if service in MARKET_SERVICE_COMPONENTS:
            service_identity = ("service", service)
            state.entity_sources[service_identity].add(source_path)
            state.entity_aliases[service_identity][source_path].add(original_id)


def _metric_series(
    state: PreparedState,
    *,
    kind: str,
    original_id: str,
    metric_id: str,
    source_path: str,
) -> MetricSeries:
    normalized_kind, normalized_id = _market_identity(
        state.spec,
        source_path=source_path,
        kind=kind,
        original_id=original_id,
    )
    key = (normalized_kind, normalized_id, metric_id)
    series = state.metric_series.get(key)
    if series is None:
        series = MetricSeries(
            canonical_id=_canonical_id(
                state.spec, kind=normalized_kind, original_id=normalized_id
            ),
            entity_ids={original_id},
            metric_id=metric_id,
            source_path=source_path,
        )
        state.metric_series[key] = series
    elif series.source_path != source_path:
        raise ValueError(
            "multiple Market source files map to the same canonical metric series"
        )
    else:
        series.entity_ids.add(original_id)
    return series


def _collect_logs(
    state: PreparedState, relative: str, rows: list[dict[str, str]]
) -> None:
    for row in rows:
        cmdb_id = row["cmdb_id"]
        _entity_sources(state, kind="cmdb", original_id=cmdb_id, source_path=relative)
        state.log_names[row["log_name"]] += 1


def _collect_long_metrics(
    state: PreparedState,
    relative: str,
    metric_prefix: str,
    rows: list[dict[str, str]],
) -> None:
    for row in rows:
        cmdb_id = row["cmdb_id"]
        _entity_sources(state, kind="cmdb", original_id=cmdb_id, source_path=relative)
        value = float(_parse_decimal(row["value"], field_name=f"{relative}.value"))
        seconds = _parse_decimal(row["timestamp"], field_name=f"{relative}.timestamp")
        period = _record_period(state.spec, seconds)
        metric_id = f"{metric_prefix}::{row['kpi_name']}"
        series = _metric_series(
            state,
            kind="cmdb",
            original_id=cmdb_id,
            metric_id=metric_id,
            source_path=relative,
        )
        series.timestamps.append(float(seconds))
        if period is not None:
            series.periods[period].append(value)
            series.records[period].append(int(row["_source_record"]))


def _collect_service_metrics(
    state: PreparedState, relative: str, rows: list[dict[str, str]]
) -> None:
    for row in rows:
        service = row["service"]
        _entity_sources(
            state, kind="service", original_id=service, source_path=relative
        )
        seconds = _parse_decimal(row["timestamp"], field_name=f"{relative}.timestamp")
        period = _record_period(state.spec, seconds)
        for field_name in ("rr", "sr", "mrt", "count"):
            value = float(
                _parse_decimal(row[field_name], field_name=f"{relative}.{field_name}")
            )
            series = _metric_series(
                state,
                kind="service",
                original_id=service,
                metric_id=f"service::{field_name}",
                source_path=relative,
            )
            series.timestamps.append(float(seconds))
            if period is not None:
                series.periods[period].append(value)
                series.records[period].append(int(row["_source_record"]))


def _collect_trace(
    state: PreparedState, relative: str, rows: list[dict[str, str]]
) -> None:
    for row in rows:
        cmdb_id = row["cmdb_id"]
        _entity_sources(state, kind="cmdb", original_id=cmdb_id, source_path=relative)
        trace_id = row["trace_id"]
        span_id = row["span_id"]
        parent_span = row["parent_span"]
        timestamp = _parse_decimal(row["timestamp"], field_name=f"{relative}.timestamp")
        if timestamp != timestamp.to_integral_value():
            raise ValueError(
                f"{relative}.timestamp must be an integer millisecond value"
            )
        key = (trace_id, span_id)
        kind, component = _market_identity(
            state.spec,
            source_path=relative,
            kind="cmdb",
            original_id=cmdb_id,
        )
        value = (
            _canonical_id(state.spec, kind=kind, original_id=component),
            parent_span,
            int(timestamp),
            int(row["_source_record"]),
        )
        if key in state.span_index:
            state.duplicate_span_keys += 1
            if state.span_index[key][:3] != value[:3]:
                state.span_conflicts.add(key)
            continue
        state.span_index[key] = value


def _prepare_telemetry(state: PreparedState) -> None:
    file_specs = cast(dict[str, dict[str, object]], state.source_schema["files"])
    for relative in EXPECTED_FILES:
        spec = cast(dict[str, object], file_specs[relative])
        header = [str(column) for column in cast(list[object], spec["columns"])]
        target_path = state.public_input / "telemetry" / relative
        seconds_columns = str(spec["timestamp_unit"]) == "seconds"
        rows = _write_filtered_copy(
            target_path,
            header=header,
            state=state,
            relative=relative,
            seconds_columns=seconds_columns,
        )
        if relative.startswith("log/"):
            _collect_logs(state, relative, rows)
        elif relative == "metric/metric_service.csv":
            _collect_service_metrics(state, relative, rows)
        elif relative.startswith("metric/"):
            _collect_long_metrics(
                state,
                relative,
                Path(relative).stem.removeprefix("metric_"),
                rows,
            )
        else:
            _collect_trace(state, relative, rows)
    _aggregate_pod_metrics_to_services(state)


def _aggregate_pod_metrics_to_services(state: PreparedState) -> None:
    for (kind, pod, metric_id), series in list(state.metric_series.items()):
        if kind != "pod" or series.source_path != "metric/metric_container.csv":
            continue
        service = market_service_component(pod)
        if service not in MARKET_SERVICE_COMPONENTS:
            continue
        key = ("service", service, metric_id)
        current = state.metric_series.get(key)
        if current is None or _series_candidate_rank(series) < _series_candidate_rank(
            current
        ):
            state.metric_series[key] = MetricSeries(
                canonical_id=_canonical_id(
                    state.spec, kind="service", original_id=service
                ),
                entity_ids=set(series.entity_ids),
                metric_id=metric_id,
                source_path=series.source_path,
                timestamps=list(series.timestamps),
                periods={
                    period: list(series.periods[period])
                    for period in ("reference", "incident")
                },
                records={
                    period: list(series.records[period])
                    for period in ("reference", "incident")
                },
            )
        elif current.source_path != series.source_path:
            raise ValueError("service aggregate metric source mismatch")
        service_identity = ("service", service)
        state.entity_sources[service_identity].add(series.source_path)
        state.entity_aliases[service_identity][series.source_path].update(
            series.entity_ids
        )


def _series_candidate_rank(series: MetricSeries) -> tuple[float, str]:
    reference = series.periods["reference"]
    incident = series.periods["incident"]
    if not reference or not incident:
        return (float("inf"), min(series.entity_ids))
    reference_median = statistics.median(reference)
    incident_median = statistics.median(incident)
    relative_shift = (incident_median - reference_median) / max(
        abs(reference_median), abs(incident_median), 1.0
    )
    upper_extreme = max(
        0.0,
        (max(incident) - max(reference))
        / max(abs(max(incident)), abs(max(reference)), 1.0),
    )
    lower_extreme = min(
        0.0,
        (min(incident) - min(reference))
        / max(abs(min(incident)), abs(min(reference)), 1.0),
    )
    score = max(abs(relative_shift), abs(upper_extreme), abs(lower_extreme))
    return (-score, min(series.entity_ids))


def _entity_mapping(state: PreparedState) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for kind, original_id in sorted(state.entity_sources):
        canonical_id = _canonical_id(state.spec, kind=kind, original_id=original_id)
        nodes.append(
            {
                "canonical_id": canonical_id,
                "original_id": original_id,
                "entity_type": kind,
                "sources": sorted(state.entity_sources[(kind, original_id)]),
                "source_aliases": {
                    source: sorted(values)
                    for source, values in sorted(
                        state.entity_aliases[(kind, original_id)].items()
                    )
                },
                "normalization": (
                    "market public component identity"
                    if original_id in state.spec.public_component_vocabulary
                    else "exact raw identity"
                ),
                "public_candidate_match": (
                    original_id
                    if original_id in state.spec.public_component_vocabulary
                    else None
                ),
                "mapping_status": (
                    "normalized"
                    if original_id in state.spec.public_component_vocabulary
                    else "source_exact"
                ),
                "mapping_evidence": (
                    "pinned OpenRCA Market public candidate schema"
                    if original_id in state.spec.public_component_vocabulary
                    else "source schema entity key plus cloudbed namespace"
                ),
            }
        )
    return nodes


def _topology(state: PreparedState, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    known_entities = {str(node["canonical_id"]) for node in nodes}
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    matched = missing = internal = sentinel = ambiguous = 0
    root_markers: Counter[str] = Counter()
    missing_samples: list[dict[str, Any]] = []
    for (trace_id, span_id), (
        entity_id,
        parent_span,
        timestamp_ms,
        source_record,
    ) in sorted(state.span_index.items()):
        parent_key = (trace_id, parent_span)
        if (
            trace_id,
            span_id,
        ) in state.span_conflicts or parent_key in state.span_conflicts:
            ambiguous += 1
            continue
        lowered_parent = parent_span.strip().lower()
        if lowered_parent in ROOT_MARKERS:
            sentinel += 1
            root_markers[parent_span] += 1
            continue
        parent = state.span_index.get(parent_key)
        if parent is None:
            missing += 1
            if len(missing_samples) < 10:
                missing_samples.append(
                    {
                        "trace_id": trace_id,
                        "span_id": span_id,
                        "parent_id": parent_span,
                        "source_record": source_record,
                    }
                )
            continue
        matched += 1
        if parent[0] == entity_id:
            internal += 1
            continue
        if parent[0] not in known_entities or entity_id not in known_entities:
            continue
        pair = (parent[0], entity_id)
        edge = edges.setdefault(
            pair,
            {
                "source_id": pair[0],
                "target_id": pair[1],
                "relation_type": "calls",
                "fact_kind": "observed",
                "span_pairs": 0,
                "first_observed_ms": timestamp_ms,
                "last_observed_ms": timestamp_ms,
                "samples": [],
            },
        )
        edge["span_pairs"] += 1
        edge["first_observed_ms"] = min(
            cast(int, edge["first_observed_ms"]), timestamp_ms
        )
        edge["last_observed_ms"] = max(
            cast(int, edge["last_observed_ms"]), timestamp_ms
        )
        samples = cast(list[dict[str, object]], edge["samples"])
        if len(samples) < 3:
            samples.append(
                {
                    "trace_id": trace_id,
                    "parent_span_id": parent_span,
                    "child_span_id": span_id,
                    "parent_source_record": parent[3],
                    "child_source_record": source_record,
                }
            )
    node_by_id = {str(node["canonical_id"]): node for node in nodes}
    service_ids = {
        str(node["original_id"]): str(node["canonical_id"])
        for node in nodes
        if node["entity_type"] == "service"
    }
    for (source_id, target_id), edge in list(edges.items()):
        source_node = node_by_id[source_id]
        target_node = node_by_id[target_id]
        source_component = str(source_node.get("public_candidate_match") or "")
        target_component = str(target_node.get("public_candidate_match") or "")
        source_service = service_ids.get(market_service_component(source_component))
        target_service = service_ids.get(market_service_component(target_component))
        if (
            source_service is None
            or target_service is None
            or source_service == target_service
        ):
            continue
        pair = (source_service, target_service)
        aggregate = edges.setdefault(
            pair,
            {
                "source_id": source_service,
                "target_id": target_service,
                "relation_type": "calls",
                "fact_kind": "derived",
                "span_pairs": 0,
                "first_observed_ms": edge["first_observed_ms"],
                "last_observed_ms": edge["last_observed_ms"],
                "samples": [],
            },
        )
        aggregate["span_pairs"] += edge["span_pairs"]
        aggregate["first_observed_ms"] = min(
            cast(int, aggregate["first_observed_ms"]),
            cast(int, edge["first_observed_ms"]),
        )
        aggregate["last_observed_ms"] = max(
            cast(int, aggregate["last_observed_ms"]),
            cast(int, edge["last_observed_ms"]),
        )
        samples = cast(list[dict[str, object]], aggregate["samples"])
        for sample in cast(list[dict[str, object]], edge["samples"]):
            if len(samples) == 3:
                break
            samples.append(sample)

    ordered_edges: list[dict[str, Any]] = []
    for index, pair in enumerate(sorted(edges)):
        edge = dict(edges[pair])
        edge["edge_id"] = f"edge-{index:03d}"
        ordered_edges.append(edge)
    coverage = {
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
            "partial pod topology from trace_id + parent_span/span_id plus "
            "derived service edges within the allowed window"
        ),
    }
    return {
        "snapshot_id": f"{state.spec.case_id}-topology-v1",
        "source": "trace/trace_span.csv parent_span/span_id joined within trace_id",
        "valid_range": _range_payload(state.spec.allowed_start, state.spec.allowed_end),
        "nodes": nodes,
        "edges": ordered_edges,
        "coverage": coverage,
    }


def _metric_bindings(state: PreparedState) -> list[dict[str, Any]]:
    bindings: list[dict[str, Any]] = []
    for key in sorted(state.metric_series):
        kind, original_id, metric_id = key
        series = state.metric_series[key]
        if series.source_path == "metric/metric_service.csv":
            bindings.append(
                {
                    "canonical_id": _canonical_id(
                        state.spec, kind=kind, original_id=original_id
                    ),
                    "metric_id": metric_id,
                    "source_path": "metric/metric_service.csv",
                    "layout": "wide",
                    "entity_field": "service",
                    "original_entity_key": sorted(series.entity_ids)[0],
                    "original_entity_keys": sorted(series.entity_ids),
                    "value_field": metric_id.split("::", 1)[1],
                    "timestamp_unit": "seconds",
                }
            )
            continue
        _prefix, source_metric = metric_id.split("::", 1)
        bindings.append(
            {
                "canonical_id": _canonical_id(
                    state.spec, kind=kind, original_id=original_id
                ),
                "metric_id": metric_id,
                "source_path": series.source_path,
                "layout": "long",
                "entity_field": "cmdb_id",
                "original_entity_key": sorted(series.entity_ids)[0],
                "original_entity_keys": sorted(series.entity_ids),
                "metric_field": "kpi_name",
                "source_metric": source_metric,
                "value_field": "value",
                "timestamp_unit": "seconds",
            }
        )
    return bindings


def _metric_stats(state: PreparedState) -> list[dict[str, Any]]:
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
            "entity_id": key[1],
            "source_entity_keys": sorted(series.entity_ids),
            "metric_id": series.metric_id,
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
                "example_source_records": series.records[period][:5],
            }
        stats.append(item)
    return stats


def _task_payload(state: PreparedState, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_id": state.spec.case_id,
        "environment": "openrca_market",
        "question": state.spec.question,
        "incident_range": _range_payload(
            state.spec.incident_start, state.spec.incident_end
        ),
        "timezone": "Asia/Shanghai",
        "allowed_query_range": _range_payload(
            state.spec.allowed_start, state.spec.allowed_end
        ),
        "reference_range": {
            **_range_payload(state.spec.reference_start, state.spec.reference_end),
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
        "requested_outputs": ["root cause component", "root cause reason"],
        "dataset_manifest_ref": "manifest.json",
        "source_row_id": state.spec.row_id,
        "expected_fault_count": 1,
        "fault_count_source": "label-free public case spec",
        "data_mode": "offline_window_slice",
        "source_commit": state.spec.source_revision,
    }


def _dataset_profile(
    state: PreparedState,
    *,
    topology: dict[str, Any],
    metric_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "cloudbed": state.spec.cloudbed,
        "files": len(EXPECTED_FILES),
        "source_date": state.spec.source_date,
        "selected_rows": {
            path: state.file_rows.get(path, 0) for path in EXPECTED_FILES
        },
        "prepared_trace_contract": {
            "cloudbed": state.spec.cloudbed,
            "join_keys": ["trace_id", "parent_span", "span_id"],
            "rows_filtered_to_allowed_window": True,
            "source_path": "trace/trace_span.csv",
            "timestamp_unit": "milliseconds",
        },
        "metric_series_count": len(metric_stats),
        "topology_edge_count": len(cast(list[object], topology["edges"])),
        "limitations": [
            "Only the verified local source day is included.",
            "Rows are filtered with the exact half-open allowed_query_range.",
            "Evaluation labels and label-like source files are rejected.",
        ],
    }


def _write_public_input(state: PreparedState) -> None:
    nodes = _entity_mapping(state)
    topology = _topology(state, nodes)
    bindings = _metric_bindings(state)
    metric_stats = _metric_stats(state)
    _write_json(state.public_input / "task.json", _task_payload(state, nodes))
    _write_json(state.public_input / "entity_mapping.json", nodes)
    _write_json(state.public_input / "topology.json", topology)
    (state.public_input / "source_schema.json").write_bytes(state.source_schema_bytes)
    _write_json(
        state.public_input / "vocabulary.json",
        {
            "entities": list(state.spec.public_component_vocabulary),
            "fault_types": list(state.spec.public_fault_vocabulary),
            "source": MARKET_CANDIDATE_SOURCE,
        },
    )
    _write_json(
        state.public_input / "metric_catalog.json",
        {
            "duration_unit": "unknown",
            "log_names": dict(sorted(state.log_names.items())),
            "metrics": sorted(
                series.metric_id for series in state.metric_series.values()
            ),
            "metric_bindings": bindings,
            "schemas": {
                "long": "cmdb_id, kpi_name, value",
                "service": "wide rr, sr, mrt, count",
            },
        },
    )
    stats_path = state.public_input / "metric_window_stats.jsonl"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("wb") as stream:
        for item in metric_stats:
            stream.write(_jsonl_bytes(item))
    _write_json(
        state.public_input / "dataset_profile.json",
        _dataset_profile(state, topology=topology, metric_stats=metric_stats),
    )
    artifacts = [
        {
            "path": path.relative_to(state.public_input).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(
            candidate
            for candidate in state.public_input.rglob("*")
            if candidate.is_file() and candidate.name != "manifest.json"
        )
    ]
    _write_json(
        state.public_input / "manifest.json",
        {
            "schema_version": "rootwalk-dataset-slice-v1",
            "dataset_id": state.spec.case_id,
            "cloudbed": state.spec.cloudbed,
            "source_commit": state.spec.source_revision,
            "label_policy": "label-free public case spec; evaluation labels excluded",
            "artifacts": artifacts,
        },
    )


def prepare_market_public_input(
    source_day: Path,
    case_spec_path: Path,
    output_root: Path,
    *,
    source_schema_path: Path | None = None,
) -> Path:
    """Prepare one OpenRCA Market public_input and return its published path."""
    _resolved_source_schema_path, source_schema, source_schema_bytes = (
        _load_verified_source_schema(source_schema_path)
    )
    spec = _load_case_spec(case_spec_path)
    verified_source_day, source_day_fd = _validate_source_day(source_day, spec)
    try:
        raw_output_root = output_root.expanduser()
        if raw_output_root.is_symlink():
            raise ValueError("output-root must not be a symlink")
        resolved_output_root = raw_output_root.resolve()
        if _is_relative_to(resolved_output_root, REPO_ROOT):
            raise ValueError("output-root must be outside the git worktree")
        _reject_overlap(
            verified_source_day,
            resolved_output_root,
            left_label="source-day",
            right_label="output-root",
        )
        resolved_output_root.mkdir(parents=True, exist_ok=True)
        case_dir = resolved_output_root / spec.case_id
        _reject_overlap(
            verified_source_day,
            case_dir,
            left_label="source-day",
            right_label="case-dir",
        )
        stage = Path(
            tempfile.mkdtemp(prefix=f".tmp-{spec.case_id}-", dir=resolved_output_root)
        )
        public_input = stage / spec.case_id / "public_input"
        state = PreparedState(
            spec=spec,
            source_day=verified_source_day,
            source_day_fd=source_day_fd,
            public_input=public_input,
            source_schema=source_schema,
            source_schema_bytes=source_schema_bytes,
        )
        _prepare_telemetry(state)
        _write_public_input(state)
        MarketEnvironment(public_input)
        staged_case_dir = public_input.parent
        if case_dir.is_symlink():
            raise ValueError("case-dir must not be a symlink")
        if case_dir.exists():
            if not case_dir.is_dir():
                raise ValueError("case-dir must be a directory")
            _ensure_tree_has_no_symlinks(case_dir, label="case-dir")
            existing_public_input = case_dir / "public_input"
            if not existing_public_input.is_dir():
                raise ValueError(
                    "existing case-dir must contain a public_input directory"
                )
            MarketEnvironment(existing_public_input)
            if _tree_entries(case_dir) != _tree_entries(staged_case_dir):
                raise FileExistsError(
                    "output case already exists with different contents; "
                    "choose a new output-root"
                )
            return case_dir / "public_input"
        os.replace(staged_case_dir, case_dir)
        return case_dir / "public_input"
    finally:
        os.close(source_day_fd)
        if "stage" in locals() and stage.exists():
            shutil.rmtree(stage)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-day", required=True, type=Path)
    parser.add_argument("--case-spec", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--source-schema", type=Path, default=None)
    args = parser.parse_args(argv)
    public_input = prepare_market_public_input(
        args.source_day,
        args.case_spec,
        args.output_root,
        source_schema_path=args.source_schema,
    )
    print(
        json.dumps(
            {
                "case_id": public_input.parent.name,
                "public_input": str(public_input),
                "source_revision": MARKET_SOURCE_REVISION,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
