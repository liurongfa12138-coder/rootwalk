#!/usr/bin/env python3
# pyright: basic
"""Prepare the frozen, label-free OpenRCA Bank D9 suite."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import tempfile
import zlib
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

SPEC_PATH = Path(__file__).resolve().parents[1] / "benchmarks/openrca-bank-d9-v1.json"
V2_SPEC_PATH = (
    Path(__file__).resolve().parents[1] / "benchmarks/openrca-bank-d9-v2.json"
)
DEFAULT_ROW_IDS = (
    2,
    4,
    9,
    17,
    18,
    24,
    59,
    71,
    72,
    75,
    76,
    100,
    105,
    115,
    122,
    123,
    124,
    128,
)
SELECTION_SHA256 = "2a9d4b9d2395e934bfaad18ab0e94cead04a681051b37bade6889050db8650a0"
CASE_PLAN_SHA256 = "b5bf71395b2d94cf654bb92746d672c0a8c9c2160619523be3e5464872e10188"
V2_ROW_IDS = tuple(row_id for row_id in DEFAULT_ROW_IDS if row_id != 123)
V2_SELECTION_SHA256 = "921babc8d78ec3955d0fe55c75a27d2c521d7a5ca629099f37a459954dec356b"
V2_CASE_PLAN_SHA256 = "c8c4f6acbec6371bdc7423dab98aec779285597390d5cdf26b0e55e87a161819"
V2_SOURCE_FILES_SHA256 = (
    "fe62934a0e800e6f37c30a5ed02d88f043584ef061b5fa376daf214c4a94b7b6"
)
V2_SOURCE_COMMIT = "7a29a7c3600cc07b7760a6c5097d36ac8573f039"
V2_SOURCE_URL_TEMPLATE = (
    "https://hf-mirror.com/datasets/{repository}/resolve/{commit}/{path}?download=true"
)
FROZEN_SUITES: dict[str, dict[str, object]] = {
    "openrca-bank-d9-v1": {
        "row_ids": DEFAULT_ROW_IDS,
        "selection_sha256": SELECTION_SHA256,
        "case_plan_sha256": CASE_PLAN_SHA256,
    },
    "openrca-bank-d9-v2": {
        "row_ids": V2_ROW_IDS,
        "selection_sha256": V2_SELECTION_SHA256,
        "case_plan_sha256": V2_CASE_PLAN_SHA256,
        "source_files_sha256": V2_SOURCE_FILES_SHA256,
        "source_commit": V2_SOURCE_COMMIT,
        "source_url_template": V2_SOURCE_URL_TEMPLATE,
    },
}
MAX_DAILY_SOURCE_RECORDS = 100_000_000
MAX_JS_SAFE_INTEGER = 9_007_199_254_740_991
PROVENANCE_FIELDS = (
    "name",
    "size",
    "compressed_size",
    "offset",
    "crc32",
    "sha256",
    "archive_bytes",
)
TZ = ZoneInfo("Asia/Shanghai")
SOURCE_FILES = (
    ("metric/metric_app.csv", 1000, "tc"),
    ("metric/metric_container.csv", 1000, "cmdb_id"),
    ("log/log_service.csv", 1000, "cmdb_id"),
    ("trace/trace_span.csv", 1, "cmdb_id"),
)
ROOT_CAUSE_CANDIDATE_ROLE = "root_cause_candidate"
IMPACT_ONLY_ROLE = "impact_only"
REQUIRED_COLUMNS = {
    "metric/metric_app.csv": {"timestamp", "tc"},
    "metric/metric_container.csv": {"timestamp", "cmdb_id", "kpi_name", "value"},
    "log/log_service.csv": {"timestamp", "cmdb_id", "log_id", "log_name", "value"},
    "trace/trace_span.csv": {
        "timestamp",
        "cmdb_id",
        "trace_id",
        "span_id",
        "parent_id",
    },
}
csv.field_size_limit(16 * 1024 * 1024)


@dataclass(frozen=True)
class CasePlan:
    row_id: int
    task_index: str
    date: str
    reference_start: datetime
    incident_start: datetime
    incident_end: datetime
    allowed_start: datetime
    allowed_end: datetime
    source_dates: tuple[str, ...]
    instruction_sha256: str

    @property
    def case_id(self) -> str:
        return f"bank-row-{self.row_id:04d}"


@dataclass
class CaseState:
    plan: CasePlan
    question: str
    public: Path
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    entity_sources: dict[str, set[str]] = field(default_factory=dict)
    metric_times: dict[tuple[str, str, str], set[int]] = field(default_factory=dict)
    metric_values: dict[tuple[str, str, str], dict[str, list[float]]] = field(
        default_factory=dict
    )
    metric_records: dict[tuple[str, str, str], dict[str, list[int]]] = field(
        default_factory=dict
    )
    metrics: set[str] = field(default_factory=set)
    log_names: Counter[str] = field(default_factory=Counter)


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
    ).encode()


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
    ).encode()


def _canonical_sha256(value: object) -> str:
    payload = (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(moment: datetime) -> str:
    return moment.isoformat()


def _range(start: datetime, end: datetime) -> dict[str, object]:
    return {"start": _iso(start), "end": _iso(end), "end_exclusive": True}


def _parse_range(value: object, name: str) -> tuple[datetime, datetime]:
    if not isinstance(value, dict) or value.get("end_exclusive") is not True:
        raise ValueError(f"{name} must be an explicit half-open range")
    try:
        start = datetime.fromisoformat(str(value["start"]))
        end = datetime.fromisoformat(str(value["end"]))
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid {name}") from exc
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise ValueError(f"invalid {name}")
    if start.utcoffset() != timedelta(hours=8) or end.utcoffset() != timedelta(hours=8):
        raise ValueError(f"{name} must use Asia/Shanghai UTC+8")
    return start, end


def _source_dates(start: datetime, end: datetime) -> tuple[str, ...]:
    current = start.astimezone(TZ).date()
    last = (end.astimezone(TZ) - timedelta(microseconds=1)).date()
    result: list[str] = []
    while current <= last:
        result.append(current.strftime("%Y_%m_%d"))
        current += timedelta(days=1)
    return tuple(result)


def case_plan_sha256(plans: list[CasePlan]) -> str:
    """Hash ordered row/task/range/source-date fields, excluding instructions."""
    projection = [
        {
            "allowed_query_range": _range(plan.allowed_start, plan.allowed_end),
            "incident_range": _range(plan.incident_start, plan.incident_end),
            "reference_range": _range(plan.reference_start, plan.incident_start),
            "row_id": plan.row_id,
            "source_dates": list(plan.source_dates),
            "task_index": plan.task_index,
        }
        for plan in plans
    ]
    return _canonical_sha256(projection)


def source_files_sha256(source_files: object) -> str:
    if not isinstance(source_files, list):
        raise ValueError("mirror suite requires a source_files array")
    normalized: list[dict[str, object]] = []
    seen: set[str] = set()
    for raw in source_files:
        if not isinstance(raw, dict):
            raise ValueError("source file entry must be an object")
        path = raw.get("path")
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if (
            not isinstance(path, str)
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or path in seen
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError("source file entry is invalid")
        seen.add(path)
        normalized.append({"bytes": size, "path": path, "sha256": digest})
    if [item["path"] for item in normalized] != sorted(seen):
        raise ValueError("source_files must be sorted by path")
    return _canonical_sha256(normalized)


def mirror_source_index(spec: dict[str, Any]) -> dict[str, dict[str, object]]:
    source_files = spec.get("source_files")
    source_files_sha256(source_files)
    return {
        cast(str, entry["path"]): cast(dict[str, object], entry)
        for entry in cast(list[dict[str, object]], source_files)
    }


def suite_identity(spec: dict[str, Any]) -> dict[str, object]:
    suite_id = spec.get("suite_id")
    if suite_id == "openrca-bank-d9-v1":
        return {
            "row_ids": DEFAULT_ROW_IDS,
            "selection_sha256": SELECTION_SHA256,
            "case_plan_sha256": CASE_PLAN_SHA256,
        }
    if not isinstance(suite_id, str) or suite_id not in FROZEN_SUITES:
        raise ValueError(f"unknown frozen suite_id: {suite_id!r}")
    return FROZEN_SUITES[suite_id]


def plan_cases(
    spec_path: Path = SPEC_PATH,
    row_ids: tuple[int, ...] | list[int] | None = None,
    reference_minutes: int = 30,
) -> tuple[dict[str, Any], list[CasePlan]]:
    """Validate the frozen spec and return plans in requested row order."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    identity = suite_identity(spec)
    if spec.get("selection_sha256") != identity["selection_sha256"]:
        raise ValueError(
            "suite selection SHA-256 does not match the frozen D9 selection"
        )
    if spec.get("timezone") != "Asia/Shanghai":
        raise ValueError("suite timezone must be Asia/Shanghai")
    if spec.get("task_index") != "task_6":
        raise ValueError("suite must contain only task_6 cases")
    if spec.get("expected_fault_count") != 1:
        raise ValueError("frozen suite must be audited as exactly one fault per case")
    if not isinstance(spec.get("bank_schema_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", spec["bank_schema_sha256"]
    ):
        raise ValueError("suite requires a valid bank_schema_sha256")
    if reference_minutes <= 0 or reference_minutes != spec.get("reference_minutes"):
        raise ValueError("reference-minutes must match the frozen suite value of 30")

    raw_cases = spec.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("suite cases must be a list")
    by_row: dict[int, CasePlan] = {}
    for raw in raw_cases:
        if not isinstance(raw, dict) or isinstance(raw.get("row_id"), bool):
            raise ValueError("each suite case must have an integer row_id")
        row_id = int(raw["row_id"])
        if row_id in by_row:
            raise ValueError(f"duplicate suite row_id: {row_id}")
        if raw.get("task_index") != "task_6":
            raise ValueError(f"row {row_id} is not task_6")
        reference_start, reference_end = _parse_range(
            raw.get("reference_range"), f"row {row_id} reference_range"
        )
        incident_start, incident_end = _parse_range(
            raw.get("incident_range"), f"row {row_id} incident_range"
        )
        allowed_start, allowed_end = _parse_range(
            raw.get("allowed_query_range"), f"row {row_id} allowed_query_range"
        )
        if incident_end - incident_start != timedelta(
            minutes=int(spec.get("incident_minutes", 0))
        ):
            raise ValueError(f"row {row_id} incident duration does not match the spec")
        if (
            reference_end != incident_start
            or incident_start - reference_start != timedelta(minutes=reference_minutes)
        ):
            raise ValueError(
                f"row {row_id} reference range is not contiguous and frozen"
            )
        if (allowed_start, allowed_end) != (reference_start, incident_end):
            raise ValueError(
                f"row {row_id} allowed range does not cover reference + incident"
            )
        case_date = incident_start.astimezone(TZ).strftime("%Y_%m_%d")
        if raw.get("date") != case_date:
            raise ValueError(f"row {row_id} date does not match its incident start")
        instruction_sha256 = raw.get("instruction_sha256")
        if not isinstance(instruction_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", instruction_sha256
        ):
            raise ValueError(f"row {row_id} requires a valid instruction_sha256")
        by_row[row_id] = CasePlan(
            row_id=row_id,
            task_index="task_6",
            date=case_date,
            reference_start=reference_start,
            incident_start=incident_start,
            incident_end=incident_end,
            allowed_start=allowed_start,
            allowed_end=allowed_end,
            source_dates=_source_dates(allowed_start, allowed_end),
            instruction_sha256=instruction_sha256,
        )

    frozen_ids = tuple(by_row)
    expected_ids = tuple(cast(tuple[int, ...], identity["row_ids"]))
    if frozen_ids != expected_ids:
        raise ValueError("suite row IDs or their frozen order changed")
    if case_plan_sha256(list(by_row.values())) != identity["case_plan_sha256"]:
        raise ValueError("suite case plan SHA-256 does not match the frozen D9 plan")
    expected_source_hash = identity.get("source_files_sha256")
    if expected_source_hash is not None:
        mirror = spec.get("source_mirror")
        if (
            not isinstance(mirror, dict)
            or mirror.get("repository") != "cdreetz/OpenRCA"
            or mirror.get("commit") != identity["source_commit"]
            or mirror.get("url_template") != identity["source_url_template"]
        ):
            raise ValueError("suite mirror identity does not match the frozen source")
        actual_source_hash = source_files_sha256(spec.get("source_files"))
        if (
            spec.get("source_files_sha256") != expected_source_hash
            or actual_source_hash != expected_source_hash
        ):
            raise ValueError("suite source file SHA-256 does not match the frozen plan")
        expected_paths = {"Bank/query.csv"} | {
            f"Bank/telemetry/{source_date}/{relative}"
            for plan in by_row.values()
            for source_date in plan.source_dates
            for relative, _multiplier, _entity in SOURCE_FILES
        }
        actual_paths = {
            cast(str, item["path"])
            for item in cast(list[dict[str, object]], spec["source_files"])
        }
        if actual_paths != expected_paths:
            raise ValueError("source_files do not match the frozen case windows")
        if spec.get("source_total_bytes") != sum(
            cast(int, item["bytes"])
            for item in cast(list[dict[str, object]], spec["source_files"])
        ):
            raise ValueError("source_total_bytes does not match source_files")
    selected = expected_ids if row_ids is None else tuple(row_ids)
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("row-ids must be non-empty and unique")
    unknown = sorted(set(selected) - set(by_row))
    if unknown:
        raise ValueError(f"row-ids are not in the frozen suite: {unknown}")
    return spec, [by_row[row_id] for row_id in selected]


def _safe_input(root: Path, relative: str) -> Path:
    candidate = root / relative
    if candidate.is_symlink():
        raise ValueError(f"source file must not be a symlink: {relative}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"source path escapes its root: {relative}") from exc
    if not resolved.is_file():
        raise ValueError(f"source path is not a file: {relative}")
    return resolved


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _inside_git_worktree(path: Path) -> bool:
    return any(
        (parent / ".git").is_file() or (parent / ".git/HEAD").is_file()
        for parent in (path, *path.parents)
    )


def _validate_roots(
    source_root: Path, upstream_code_root: Path, output_root: Path
) -> tuple[Path, Path, Path]:
    source = source_root.expanduser().resolve(strict=True)
    upstream = upstream_code_root.expanduser().resolve(strict=True)
    if not source.is_dir() or not upstream.is_dir():
        raise ValueError("source-root and upstream-code-root must be directories")
    if output_root.is_symlink():
        raise ValueError("output-root must not be a symlink")
    output = output_root.expanduser().resolve(strict=False)
    if output == Path(output.anchor):
        raise ValueError("refusing to use a filesystem root as output-root")
    if _inside_git_worktree(output):
        raise ValueError("output-root must be outside every Git working tree")
    if _paths_overlap(output, source) or _paths_overlap(output, upstream):
        raise ValueError("output-root must be separate from both input roots")
    if output.exists() and not output.is_dir():
        raise ValueError("output-root exists and is not a directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    return source, upstream, output


def _load_provenance(
    source_root: Path,
    member: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    key = hashlib.sha256(member.encode()).hexdigest()[:16]
    relative = f".openrca/provenance/{key}.json"
    path = _safe_input(source_root, relative)
    value = json.loads(path.read_text(encoding="utf-8"))
    mirror = spec.get("source_mirror") if isinstance(value, dict) else None
    extra_fields = (
        (
            "source_kind",
            "source_commit",
            "source_url",
            "crc_computed",
        )
        if mirror
        else ()
    )
    if not isinstance(value, dict) or any(
        field not in value for field in (*PROVENANCE_FIELDS, *extra_fields)
    ):
        raise ValueError(f"incomplete fetch provenance for {member}")
    projected = {field: value[field] for field in (*PROVENANCE_FIELDS, *extra_fields)}
    if projected["name"] != member:
        raise ValueError(f"fetch provenance member mismatch for {member}")
    if mirror:
        source_entry = mirror_source_index(spec).get(member)
        if source_entry is None:
            raise ValueError(f"mirror source manifest does not contain {member}")
        template = cast(str, cast(dict[str, object], mirror)["url_template"])
        expected_url = template.format(
            repository=cast(dict[str, object], mirror)["repository"],
            commit=cast(dict[str, object], mirror)["commit"],
            path=member,
        )
        if (
            projected["size"] != source_entry["bytes"]
            or projected["sha256"] != source_entry["sha256"]
            or projected["compressed_size"] is not None
            or projected["offset"] is not None
            or projected["archive_bytes"] is not None
            or projected["source_kind"] != "huggingface_mirror"
            or projected["source_commit"] != cast(dict[str, object], mirror)["commit"]
            or projected["source_url"] != expected_url
            or projected["crc_computed"] is not True
        ):
            raise ValueError(
                f"fetch provenance does not match mirror spec for {member}"
            )
    else:
        archive = cast(dict[str, object], spec["official_archive"])
        archive_bytes = cast(int, archive["bytes"])
        if projected["archive_bytes"] != archive_bytes:
            raise ValueError(f"fetch provenance archive size mismatch for {member}")
        if any(
            isinstance(projected[field], bool) or not isinstance(projected[field], int)
            for field in ("size", "compressed_size", "offset")
        ):
            raise ValueError(f"invalid numeric fetch provenance for {member}")
        if (
            projected["size"] < 0
            or projected["compressed_size"] < 0
            or projected["offset"] < 0
            or projected["offset"] + projected["compressed_size"] > archive_bytes
        ):
            raise ValueError(f"out-of-range fetch provenance for {member}")
    if not isinstance(projected["crc32"], str) or not re.fullmatch(
        r"[0-9a-f]{8}", projected["crc32"]
    ):
        raise ValueError(f"invalid provenance CRC for {member}")
    if not isinstance(projected["sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", projected["sha256"]
    ):
        raise ValueError(f"invalid provenance SHA-256 for {member}")
    return projected


def _verify_provenance_content(
    member: str,
    provenance: dict[str, Any],
    *,
    size: int,
    crc32: int,
    sha256: str,
) -> None:
    if provenance["size"] != size:
        raise ValueError(f"source size does not match fetch provenance for {member}")
    if provenance["crc32"] != f"{crc32 & 0xFFFFFFFF:08x}":
        raise ValueError(f"source CRC does not match fetch provenance for {member}")
    if provenance["sha256"] != sha256:
        raise ValueError(f"source SHA-256 does not match fetch provenance for {member}")


def _verify_small_source(
    source_root: Path,
    member: str,
    source: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    provenance = _load_provenance(source_root, member, spec)
    data = source.read_bytes()
    _verify_provenance_content(
        member,
        provenance,
        size=len(data),
        crc32=zlib.crc32(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )
    return provenance


def _load_queries(
    query_path: Path, plans: list[CasePlan], expected_selection_hash: str
) -> tuple[dict[int, str], str]:
    wanted = {plan.row_id: plan for plan in plans}
    selected: dict[int, str] = {}
    with query_path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
            instruction_index = header.index("instruction")
            task_index = header.index("task_index")
        except (StopIteration, ValueError) as exc:
            raise ValueError(
                "query.csv requires instruction and task_index columns"
            ) from exc
        for row_id, row in enumerate(reader):
            if row_id not in wanted:
                continue
            if len(row) <= max(instruction_index, task_index):
                raise ValueError(f"query.csv row {row_id} is incomplete")
            instruction = row[instruction_index]
            actual_task = row[task_index]
            plan = wanted[row_id]
            if actual_task != plan.task_index:
                raise ValueError(
                    f"query.csv row {row_id} is {actual_task!r}; only task_6 is allowed"
                )
            actual_hash = hashlib.sha256(instruction.encode()).hexdigest()
            if actual_hash != plan.instruction_sha256:
                raise ValueError(f"query.csv row {row_id} instruction SHA-256 mismatch")
            selected[row_id] = instruction
    missing = sorted(set(wanted) - set(selected))
    if missing:
        raise ValueError(f"query.csv does not contain selected rows: {missing}")
    projection = [
        {
            "instruction": selected[plan.row_id],
            "row_id": plan.row_id,
            "task_index": plan.task_index,
        }
        for plan in plans
    ]
    actual_selection_hash = _canonical_sha256(projection)
    if actual_selection_hash != expected_selection_hash:
        raise ValueError("query.csv public selection SHA-256 does not match the suite")
    return selected, actual_selection_hash


def _load_vocabulary(
    upstream_root: Path, expected_schema_sha256: str
) -> tuple[list[str], list[str]]:
    direct = upstream_root / "bank_schema.py"
    schema_relative = (
        "bank_schema.py"
        if direct.exists()
        else "rca/baseline/rca_agent/prompt/basic_prompt_Bank.py"
    )
    schema_path = _safe_input(upstream_root, schema_relative)
    if _sha256(schema_path) != expected_schema_sha256:
        raise ValueError("bank_schema.py SHA-256 does not match the frozen suite")
    tree = ast.parse(schema_path.read_text(encoding="utf-8"), filename=str(schema_path))
    candidates: str | None = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "cand"
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                candidates = value
            break
    if candidates is None or "## POSSIBLE ROOT CAUSE COMPONENTS:" not in candidates:
        raise ValueError("bank_schema.py has no public Bank candidate vocabulary")
    reason_text, entity_text = candidates.split("## POSSIBLE ROOT CAUSE COMPONENTS:", 1)
    reasons = [
        line[2:].strip() for line in reason_text.splitlines() if line.startswith("- ")
    ]
    entities = [
        line[2:].strip() for line in entity_text.splitlines() if line.startswith("- ")
    ]
    if not reasons or not entities:
        raise ValueError("bank_schema.py public candidate vocabulary is empty")
    return entities, reasons


def _verify_upstream_commit(upstream_root: Path, expected: str) -> None:
    if not (upstream_root / ".git").exists():
        return
    result = subprocess.run(
        ["git", "-C", str(upstream_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != expected:
        raise ValueError("upstream-code-root is not at the frozen OpenRCA commit")


def _new_profile() -> dict[str, Any]:
    return {
        "source_provenance": [],
        "source_members": [],
        "source_rows": 0,
        "source_bytes": 0,
        "source_columns": None,
        "source_time_min_ms": None,
        "source_time_max_ms": None,
        "invalid_timestamp_rows": 0,
        "selected_rows": 0,
        "selected_time_min_ms": None,
        "selected_time_max_ms": None,
        "source_entities": Counter(),
        "selected_entities": Counter(),
        "selected_rows_by_period": {"reference": 0, "incident": 0},
    }


def _record_selected(
    state: CaseState,
    relative: str,
    entity_field: str,
    fields: list[str],
    row: dict[str, str],
    timestamp_ms: int,
    source_record: int,
) -> None:
    profile = state.profiles[relative]
    profile["selected_rows"] += 1
    current_min = profile["selected_time_min_ms"]
    current_max = profile["selected_time_max_ms"]
    profile["selected_time_min_ms"] = (
        timestamp_ms if current_min is None else min(current_min, timestamp_ms)
    )
    profile["selected_time_max_ms"] = (
        timestamp_ms if current_max is None else max(current_max, timestamp_ms)
    )
    period = (
        "reference"
        if timestamp_ms < int(state.plan.incident_start.timestamp() * 1000)
        else "incident"
    )
    profile["selected_rows_by_period"][period] += 1
    entity = row[entity_field]
    profile["selected_entities"][entity] += 1
    state.entity_sources.setdefault(entity, set()).add(relative)

    if relative == "metric/metric_container.csv":
        evidence_role = ROOT_CAUSE_CANDIDATE_ROLE
        items = [(row["kpi_name"], row["value"])]
    elif relative == "metric/metric_app.csv":
        evidence_role = IMPACT_ONLY_ROLE
        items = [
            (name, row[name]) for name in fields if name not in {"timestamp", "tc"}
        ]
    else:
        evidence_role = ROOT_CAUSE_CANDIDATE_ROLE
        items = []
    for metric, raw_value in items:
        state.metrics.add(metric)
        key = (entity, metric, evidence_role)
        state.metric_times.setdefault(key, set()).add(timestamp_ms)
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        periods = state.metric_values.setdefault(key, {})
        periods.setdefault(period, []).append(value)
        examples = state.metric_records.setdefault(key, {}).setdefault(period, [])
        if len(examples) < 3:
            examples.append(source_record)
    if relative == "log/log_service.csv":
        state.log_names[row["log_name"]] += 1


def source_record_id(source_date: str, daily_row: int) -> int:
    """Encode YYYYMMDD plus a daily row while staying JS-number safe."""
    if not 0 < daily_row < MAX_DAILY_SOURCE_RECORDS:
        raise ValueError("daily source row must be between 1 and 99,999,999")
    value = int(source_date.replace("_", "")) * MAX_DAILY_SOURCE_RECORDS + daily_row
    if value > MAX_JS_SAFE_INTEGER:
        raise ValueError("source record exceeds the JavaScript safe integer range")
    return value


def _scan_sources_once(
    source_root: Path, states: list[CaseState], spec: dict[str, Any]
) -> None:
    by_date: dict[str, list[CaseState]] = {}
    for state in states:
        for source_date in state.plan.source_dates:
            by_date.setdefault(source_date, []).append(state)

    for source_date in sorted(by_date):
        day_states = by_date[source_date]
        for relative, multiplier, entity_field in SOURCE_FILES:
            source_member = f"Bank/telemetry/{source_date}/{relative}"
            source = _safe_input(source_root, source_member)
            provenance = _load_provenance(source_root, source_member, spec)
            targets: dict[int, tuple[Any, csv.DictWriter]] = {}
            source_min: int | None = None
            source_max: int | None = None
            source_entities: Counter[str] = Counter()
            source_digest = hashlib.sha256()
            source_crc = 0
            source_bytes = 0
            invalid = 0
            count = 0
            try:
                with source.open("rb") as incoming:

                    def decoded_lines() -> Any:
                        nonlocal source_crc, source_bytes
                        first = True
                        for raw_line in incoming:
                            source_digest.update(raw_line)
                            source_crc = zlib.crc32(raw_line, source_crc)
                            source_bytes += len(raw_line)
                            encoding = "utf-8-sig" if first else "utf-8"
                            first = False
                            yield raw_line.decode(encoding)

                    reader = csv.DictReader(decoded_lines())
                    fields = list(reader.fieldnames or ())
                    if not fields or not REQUIRED_COLUMNS[relative].issubset(fields):
                        raise ValueError(f"unexpected columns in {source_member}")
                    if "_source_record" in fields:
                        raise ValueError(
                            f"source already contains _source_record: {source_member}"
                        )
                    output_fields = fields + ["_source_record", "_source_member"]
                    for state in day_states:
                        target = state.public / "telemetry" / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        new_file = not target.exists()
                        handle = target.open("a", newline="", encoding="utf-8")
                        writer = csv.DictWriter(handle, fieldnames=output_fields)
                        if new_file:
                            writer.writeheader()
                        targets[state.plan.row_id] = (handle, writer)
                        profile = state.profiles.setdefault(relative, _new_profile())
                        if profile["source_columns"] not in (None, fields):
                            raise ValueError(
                                f"columns changed between dates for {relative}"
                            )
                        profile["source_columns"] = fields

                    for count, row in enumerate(reader, 1):
                        source_record = source_record_id(source_date, count)
                        try:
                            timestamp_ms = int(Decimal(row["timestamp"]) * multiplier)
                        except (InvalidOperation, ValueError, ArithmeticError):
                            invalid += 1
                            continue
                        source_min = (
                            timestamp_ms
                            if source_min is None
                            else min(source_min, timestamp_ms)
                        )
                        source_max = (
                            timestamp_ms
                            if source_max is None
                            else max(source_max, timestamp_ms)
                        )
                        source_entities[row[entity_field]] += 1
                        for state in day_states:
                            start_ms = int(state.plan.allowed_start.timestamp() * 1000)
                            end_ms = int(state.plan.allowed_end.timestamp() * 1000)
                            if not start_ms <= timestamp_ms < end_ms:
                                continue
                            selected = dict(row)
                            selected["_source_record"] = str(source_record)
                            selected["_source_member"] = source_member
                            targets[state.plan.row_id][1].writerow(selected)
                            _record_selected(
                                state,
                                relative,
                                entity_field,
                                fields,
                                row,
                                timestamp_ms,
                                source_record,
                            )
            finally:
                for handle, _writer in targets.values():
                    handle.close()

            _verify_provenance_content(
                source_member,
                provenance,
                size=source_bytes,
                crc32=source_crc,
                sha256=source_digest.hexdigest(),
            )
            for state in day_states:
                profile = state.profiles[relative]
                profile["source_members"].append(source_member)
                profile["source_provenance"].append(provenance)
                profile["source_rows"] += count
                profile["source_bytes"] += source_bytes
                profile["source_entities"].update(source_entities)
                profile["invalid_timestamp_rows"] += invalid
                if source_min is not None:
                    old_min = profile["source_time_min_ms"]
                    profile["source_time_min_ms"] = (
                        source_min if old_min is None else min(old_min, source_min)
                    )
                if source_max is not None:
                    old_max = profile["source_time_max_ms"]
                    profile["source_time_max_ms"] = (
                        source_max if old_max is None else max(old_max, source_max)
                    )


def _entity_mapping(state: CaseState, candidates: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw_id, sources in sorted(state.entity_sources.items()):
        if "metric/metric_app.csv" in sources:
            result.append(
                {
                    "canonical_id": f"bank:test:{raw_id}",
                    "evidence_role": IMPACT_ONLY_ROLE,
                    "entity_type": "business_test",
                    "mapping_evidence": (
                        "metric_app business-test identity retained independently "
                        "from resource identifiers"
                    ),
                    "mapping_status": "business_aggregate_unmapped",
                    "normalization": "exact string identity; no fuzzy matching",
                    "original_id": raw_id,
                    "public_candidate_match": None,
                    "sources": ["metric/metric_app.csv"],
                }
            )
        resource_sources = sources - {"metric/metric_app.csv"}
        if resource_sources:
            exact = raw_id in candidates
            result.append(
                {
                    "canonical_id": f"bank:entity:{raw_id}",
                    "evidence_role": ROOT_CAUSE_CANDIDATE_ROLE,
                    "entity_type": "pod" if exact else "unknown",
                    "mapping_evidence": (
                        "public candidate vocabulary and exact cmdb_id equality"
                        if exact
                        else "raw resource telemetry identifier only"
                    ),
                    "mapping_status": (
                        "exact_candidate_match" if exact else "unresolved"
                    ),
                    "normalization": "exact string identity; no fuzzy matching",
                    "original_id": raw_id,
                    "public_candidate_match": raw_id if exact else None,
                    "sources": sorted(resource_sources),
                }
            )
    return result


def _topology(state: CaseState, mappings: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = {
        item["original_id"]: item["canonical_id"]
        for item in mappings
        if item["evidence_role"] == ROOT_CAUSE_CANDIDATE_ROLE
    }
    span_index: dict[tuple[str, str], tuple[str, str, int, int]] = {}
    conflicts: set[tuple[str, str]] = set()
    duplicate_keys = 0
    trace_path = state.public / "telemetry/trace/trace_span.csv"
    with trace_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["trace_id"], row["span_id"])
            value = (
                row["cmdb_id"],
                row["parent_id"],
                int(Decimal(row["timestamp"])),
                int(row["_source_record"]),
            )
            if key in span_index:
                duplicate_keys += 1
                if span_index[key][:3] != value[:3]:
                    conflicts.add(key)
            else:
                span_index[key] = value

    edges: dict[tuple[str, str], dict[str, Any]] = {}
    missing = matched = internal = sentinel = ambiguous = 0
    root_markers: Counter[str] = Counter()
    missing_samples: list[dict[str, Any]] = []
    for (trace_id, span_id), (
        entity,
        parent_id,
        timestamp_ms,
        record,
    ) in span_index.items():
        parent_key = (trace_id, parent_id)
        if (trace_id, span_id) in conflicts or parent_key in conflicts:
            ambiguous += 1
            continue
        if parent_id in {"", "0", "-1", "None", "null", "NULL", "root"}:
            sentinel += 1
            root_markers[parent_id] += 1
            continue
        parent = span_index.get(parent_key)
        if parent is None:
            missing += 1
            if len(missing_samples) < 10:
                missing_samples.append(
                    {
                        "parent_id": parent_id,
                        "source_record": record,
                        "span_id": span_id,
                        "trace_id": trace_id,
                    }
                )
            continue
        matched += 1
        if parent[0] == entity:
            internal += 1
            continue
        pair = (canonical[parent[0]], canonical[entity])
        edge = edges.setdefault(
            pair,
            {
                "fact_kind": "observed",
                "first_observed_ms": timestamp_ms,
                "last_observed_ms": timestamp_ms,
                "relation_type": "calls",
                "samples": [],
                "source_id": pair[0],
                "span_pairs": 0,
                "target_id": pair[1],
            },
        )
        edge["span_pairs"] += 1
        edge["first_observed_ms"] = min(edge["first_observed_ms"], timestamp_ms)
        edge["last_observed_ms"] = max(edge["last_observed_ms"], timestamp_ms)
        if len(edge["samples"]) < 3:
            edge["samples"].append(
                {
                    "child_source_record": record,
                    "child_span_id": span_id,
                    "parent_source_record": parent[3],
                    "parent_span_id": parent_id,
                    "trace_id": trace_id,
                }
            )
    ordered_edges = []
    for index, pair in enumerate(sorted(edges)):
        edge = edges[pair]
        edge["edge_id"] = f"edge-{index:03d}"
        ordered_edges.append(edge)
    coverage = {
        "ambiguous_links_skipped": ambiguous,
        "conflicting_span_keys": len(conflicts),
        "duplicate_span_keys": duplicate_keys,
        "intra_entity_pairs": internal,
        "matched_parent_spans": matched,
        "missing_parent_samples": missing_samples,
        "missing_parent_spans": missing,
        "note": (
            "No complete topology or alias inferred; parent absence can reflect "
            "sampling or the half-open slice boundary."
        ),
        "root_like_sentinel_spans": sentinel,
        "root_marker_values": dict(sorted(root_markers.items())),
        "status": "partial",
        "unique_spans": len(span_index),
    }
    return {
        "coverage": coverage,
        "edges": ordered_edges,
        "nodes": mappings,
        "snapshot_id": f"{state.plan.case_id}-topology-v1",
        "source": "trace_span.csv parent/child pairs joined within trace_id",
        "valid_range": _range(state.plan.allowed_start, state.plan.allowed_end),
    }


def _metric_stats(
    state: CaseState, mappings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    entities = {(item["original_id"], item["evidence_role"]): item for item in mappings}
    stats: list[dict[str, Any]] = []
    for entity, metric, evidence_role in sorted(state.metric_values):
        key = (entity, metric, evidence_role)
        periods = state.metric_values[key]
        stamps = sorted(state.metric_times[key])
        mapping = entities[(entity, evidence_role)]
        deltas = [
            (right - left) / 1000
            for left, right in zip(stamps, stamps[1:])
            if right > left
        ]
        item: dict[str, Any] = {
            "canonical_id": mapping["canonical_id"],
            "entity_id": entity,
            "evidence_role": mapping["evidence_role"],
            "max_observed_gap_seconds": max(deltas) if deltas else None,
            "median_observed_interval_seconds": statistics.median(deltas)
            if deltas
            else None,
            "metric_id": metric,
            "value_unit": "source-native; not inferred",
        }
        for period in ("reference", "incident"):
            values = periods.get(period)
            if values:
                item[period] = {
                    "count": len(values),
                    "example_source_records": state.metric_records[key][period],
                    "max": max(values),
                    "mean": statistics.fmean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                }
        stats.append(item)
    return stats


def _metric_bindings(
    state: CaseState, mappings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Bind every Bank metric series to its source layout and evidence role."""
    entities = {(item["original_id"], item["evidence_role"]): item for item in mappings}
    bindings: list[dict[str, Any]] = []
    for entity, metric, evidence_role in sorted(state.metric_values):
        mapping = entities[(entity, evidence_role)]
        impact_only = evidence_role == IMPACT_ONLY_ROLE
        source_path = (
            "metric/metric_app.csv" if impact_only else "metric/metric_container.csv"
        )
        item = {
            "canonical_id": mapping["canonical_id"],
            "entity_field": "tc" if impact_only else "cmdb_id",
            "evidence_role": mapping["evidence_role"],
            "layout": "wide" if impact_only else "long",
            "metric_id": metric,
            "original_entity_key": entity,
            "source_member_filter": state.profiles[source_path]["source_members"],
            "source_path": source_path,
            "timestamp_unit": "seconds",
            "value_field": metric if impact_only else "value",
        }
        if not impact_only:
            item.update({"metric_field": "kpi_name", "source_metric": metric})
        bindings.append(item)
    return bindings


def _finish_case(
    state: CaseState,
    spec: dict[str, Any],
    candidate_entities: list[str],
    fault_types: list[str],
    query_provenance: dict[str, Any],
) -> None:
    mappings = _entity_mapping(state, candidate_entities)
    topology = _topology(state, mappings)
    task = {
        "allowed_query_range": _range(state.plan.allowed_start, state.plan.allowed_end),
        "allowed_scope": {
            "entity_inventory_ref": "entity_mapping.json",
            "telemetry_root": "telemetry",
        },
        "data_mode": "offline_window_slice",
        "dataset_manifest_ref": "manifest.json",
        "entry_entities": [],
        "entry_reason": "The public question does not identify an alerted entity.",
        "environment": "openrca_bank",
        "expected_fault_count": spec["expected_fault_count"],
        "fault_count_source": "audited frozen suite specification",
        "incident_range": _range(state.plan.incident_start, state.plan.incident_end),
        "question": state.question,
        "reference_range": {
            **_range(state.plan.reference_start, state.plan.incident_start),
            "health_status": "unverified",
        },
        "requested_outputs": ["root cause component", "root cause reason"],
        "selection_policy": spec["selection_rule"],
        "source_commit": spec["upstream_commit"],
        "source_row_id": state.plan.row_id,
        "task_id": f"openrca-{state.plan.case_id}",
        "task_index": state.plan.task_index,
        "timezone": "Asia/Shanghai",
    }
    _write_json(state.public / "task.json", task)
    _write_json(state.public / "entity_mapping.json", mappings)
    _write_json(state.public / "topology.json", topology)
    _write_json(
        state.public / "vocabulary.json",
        {
            "entities": candidate_entities,
            "fault_types": fault_types,
            "source": "OpenRCA public Bank prompt; not case labels",
        },
    )
    _write_json(
        state.public / "metric_catalog.json",
        {
            "duration_unit": (
                "not specified by the inspected public Bank schema; preserve raw values"
            ),
            "log_names": dict(sorted(state.log_names.items())),
            "metric_bindings": _metric_bindings(state, mappings),
            "metrics": sorted(state.metrics),
        },
    )
    stats = _metric_stats(state, mappings)
    stats_path = state.public / "metric_window_stats.jsonl"
    with stats_path.open("wb") as stream:
        for item in stats:
            stream.write(_jsonl_bytes(item))

    profile_files: dict[str, Any] = {}
    for relative, _multiplier, _entity_field in SOURCE_FILES:
        profile = state.profiles[relative]
        target = state.public / "telemetry" / relative
        profile_files[relative] = {
            "coverage": {
                "boundary_policy": (
                    "half-open timestamp filter; no parent rows inferred "
                    "outside the window"
                ),
                "collection_health": "unknown",
                "completeness": "unknown",
                "requested_range": [
                    _iso(state.plan.allowed_start),
                    _iso(state.plan.allowed_end),
                ],
                "sampling": "unknown",
                "truncated_by_row_limit": False,
            },
            "invalid_timestamp_rows": profile["invalid_timestamp_rows"],
            "selected_bytes": target.stat().st_size,
            "selected_rows": profile["selected_rows"],
            "selected_rows_by_period": profile["selected_rows_by_period"],
            "selected_sha256": _sha256(target),
            "selected_time_range_utc": (
                [
                    datetime.fromtimestamp(
                        profile["selected_time_min_ms"] / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                    datetime.fromtimestamp(
                        profile["selected_time_max_ms"] / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                ]
                if profile["selected_time_min_ms"] is not None
                else None
            ),
            "source_bytes": profile["source_bytes"],
            "source_columns": profile["source_columns"],
            "source_entities": dict(sorted(profile["source_entities"].items())),
            "source_member": (
                profile["source_members"][0]
                if len(profile["source_members"]) == 1
                else None
            ),
            "source_members": profile["source_members"],
            "source_provenance": profile["source_provenance"],
            "source_record_basis": (
                "YYYYMMDD * 100,000,000 + one-based record number within the daily CSV"
            ),
            "source_rows": profile["source_rows"],
            "source_sha256": (
                profile["source_provenance"][0]["sha256"]
                if len(profile["source_provenance"]) == 1
                else None
            ),
            "source_time_range_utc": (
                [
                    datetime.fromtimestamp(
                        profile["source_time_min_ms"] / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                    datetime.fromtimestamp(
                        profile["source_time_max_ms"] / 1000,
                        tz=timezone.utc,
                    ).isoformat(),
                ]
                if profile["source_time_min_ms"] is not None
                else None
            ),
            "selected_entities": dict(sorted(profile["selected_entities"].items())),
            "timestamp_unit": "milliseconds"
            if relative.startswith("trace/")
            else "seconds",
            "timezone": "UTC epoch; window interpreted in Asia/Shanghai",
        }
    unresolved = [
        item["original_id"]
        for item in mappings
        if item["mapping_status"] == "unresolved"
    ]
    business = [
        item["original_id"]
        for item in mappings
        if item["mapping_status"] == "business_aggregate_unmapped"
    ]
    dataset_profile = {
        "capabilities": {
            "changes": False,
            "logs": True,
            "metrics": True,
            "source_code": False,
            "topology": "partial",
            "traces": True,
        },
        "dataset_id": state.plan.case_id,
        "entities": {
            "business_aggregates": business,
            "count": len(mappings),
            "exact_public_candidate_matches": sum(
                item["mapping_status"] == "exact_candidate_match" for item in mappings
            ),
            "unresolved": unresolved,
        },
        "impact_only_entity_count": sum(
            item["evidence_role"] == IMPACT_ONLY_ROLE for item in mappings
        ),
        "impact_only_metric_series_count": sum(
            item["evidence_role"] == IMPACT_ONLY_ROLE for item in stats
        ),
        "files": profile_files,
        "limitations": [
            "The pre-incident reference window is not certified healthy.",
            "Sampling and collection completeness are unknown.",
            (
                "Exact name identity is the only normalization; unresolved IDs "
                "are not guessed."
            ),
            "Evaluation-only fields and files are excluded from public_input.",
        ],
        "metric_series_count": len(stats),
        "selection_policy": spec["selection_rule"],
        "source_commit": spec["upstream_commit"],
        "topology": topology["coverage"],
        "topology_edge_count": len(topology["edges"]),
    }
    _write_json(state.public / "dataset_profile.json", dataset_profile)

    artifacts = _tree_entries(state.public, exclude={"manifest.json"})
    provenance: list[dict[str, Any]] = []
    for relative, _multiplier, _entity_field in SOURCE_FILES:
        profile = state.profiles[relative]
        provenance.extend(profile["source_provenance"])
    source_identity = (
        {"source_mirror": spec["source_mirror"]}
        if "source_mirror" in spec
        else {
            "official_archive": {
                **spec["official_archive"],
                "download_mode": "provided local source-root",
                "full_archive_sha256": None,
            }
        }
    )
    manifest = {
        "artifacts": artifacts,
        "dataset_id": state.plan.case_id,
        "label_policy": (
            "public instruction/task_index allowlist; evaluation labels excluded"
        ),
        **source_identity,
        "preparation_script_sha256": _sha256(Path(__file__)),
        "public_query_provenance": query_provenance,
        "schema_version": "rootwalk-dataset-slice-v2",
        "bank_schema_sha256": spec["bank_schema_sha256"],
        "source_commit": spec["upstream_commit"],
        "telemetry_provenance": provenance,
        "window": _range(state.plan.allowed_start, state.plan.allowed_end),
    }
    _write_json(state.public / "manifest.json", manifest)


def _tree_entries(root: Path, exclude: set[str] | None = None) -> list[dict[str, Any]]:
    skipped = exclude or set()
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"output tree contains a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in skipped:
            continue
        entries.append(
            {"bytes": path.stat().st_size, "path": relative, "sha256": _sha256(path)}
        )
    return entries


def _build_suite(
    source_root: Path,
    upstream_root: Path,
    stage: Path,
    spec: dict[str, Any],
    plans: list[CasePlan],
    frozen_plans: list[CasePlan],
    spec_path: Path,
) -> None:
    private_query = source_root / "private_source/Bank/query.csv"
    query_relative = (
        "private_source/Bank/query.csv" if private_query.exists() else "Bank/query.csv"
    )
    query = _safe_input(source_root, query_relative)
    query_provenance = _verify_small_source(source_root, "Bank/query.csv", query, spec)
    questions, selection_hash = _load_queries(
        query, frozen_plans, str(spec["selection_sha256"])
    )
    candidate_entities, fault_types = _load_vocabulary(
        upstream_root, str(spec["bank_schema_sha256"])
    )
    _verify_upstream_commit(upstream_root, spec["upstream_commit"])
    states = [
        CaseState(
            plan=plan,
            question=questions[plan.row_id],
            public=stage / plan.case_id / "public_input",
        )
        for plan in plans
    ]
    _scan_sources_once(source_root, states, spec)
    for state in states:
        _finish_case(
            state,
            spec,
            candidate_entities,
            fault_types,
            query_provenance,
        )

    outputs = _tree_entries(stage)
    cases = []
    for state in states:
        manifest = state.public / "manifest.json"
        cases.append(
            {
                "case_id": state.plan.case_id,
                "manifest_sha256": _sha256(manifest),
                "public_input": f"{state.plan.case_id}/public_input",
                "row_id": state.plan.row_id,
            }
        )
    suite_manifest = {
        "cases": cases,
        "outputs": outputs,
        "reference_minutes": spec["reference_minutes"],
        "row_ids": [plan.row_id for plan in plans],
        "schema_version": "rootwalk-openrca-prepared-suite-v1",
        "case_plan_hash_contract": (
            "SHA-256 of compact sorted-key JSON plus LF for the ordered "
            "row_id/task_index/reference_range/incident_range/"
            "allowed_query_range/source_dates projection"
        ),
        "case_plan_sha256": case_plan_sha256(frozen_plans),
        "selection_hash_contract": spec["selection_hash_contract"],
        "selection_rule": spec["selection_rule"],
        "selection_sha256": selection_hash,
        "bank_schema_sha256": spec["bank_schema_sha256"],
        "suite_id": spec["suite_id"],
        "suite_spec_sha256": _sha256(spec_path),
        "upstream_commit": spec["upstream_commit"],
    }
    if "source_mirror" in spec:
        suite_manifest.update(
            {
                "source_files_sha256": spec["source_files_sha256"],
                "source_mirror": spec["source_mirror"],
                "source_total_bytes": spec["source_total_bytes"],
            }
        )
    _write_json(stage / "suite_manifest.json", suite_manifest)


def prepare_suite(
    source_root: Path,
    upstream_code_root: Path,
    output_root: Path,
    row_ids: tuple[int, ...] | list[int] | None = None,
    reference_minutes: int = 30,
    spec_path: Path = SPEC_PATH,
) -> Path:
    """Build the requested frozen cases without overwriting a differing output."""
    source, upstream, output = _validate_roots(
        source_root, upstream_code_root, output_root
    )
    spec, frozen_plans = plan_cases(spec_path, None, reference_minutes)
    if row_ids is None:
        plans = frozen_plans
    else:
        selected = tuple(row_ids)
        if not selected or len(selected) != len(set(selected)):
            raise ValueError("row-ids must be non-empty and unique")
        by_row = {plan.row_id: plan for plan in frozen_plans}
        unknown = sorted(set(selected) - set(by_row))
        if unknown:
            raise ValueError(f"row-ids are not in the frozen suite: {unknown}")
        plans = [by_row[row_id] for row_id in selected]
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        _build_suite(source, upstream, stage, spec, plans, frozen_plans, spec_path)
        if output.exists():
            if _tree_entries(output) != _tree_entries(stage):
                raise FileExistsError(
                    "output-root already exists with different contents; "
                    "choose a new path"
                )
            return output
        os.replace(stage, output)
        return output
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _row_ids(values: list[str]) -> tuple[int, ...]:
    parts = [part for value in values for part in value.split(",") if part]
    try:
        return tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("row IDs must be integers") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--upstream-code-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--suite-spec", type=Path, default=SPEC_PATH)
    parser.add_argument(
        "--row-ids",
        nargs="+",
        help="space- or comma-separated row IDs; defaults to the suite spec",
    )
    parser.add_argument("--reference-minutes", type=int, default=30)
    args = parser.parse_args(argv)
    output = prepare_suite(
        args.source_root,
        args.upstream_code_root,
        args.output_root,
        _row_ids(args.row_ids) if args.row_ids else None,
        args.reference_minutes,
        args.suite_spec,
    )
    manifest = json.loads((output / "suite_manifest.json").read_text())
    print(
        json.dumps(
            {
                "cases": len(manifest["cases"]),
                "output_root": str(output),
                "selection_sha256": manifest["selection_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
