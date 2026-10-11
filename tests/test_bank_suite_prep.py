# ruff: noqa: E501 -- exact upstream instructions must not be line-wrapped
from __future__ import annotations

import csv
import hashlib
import io
import json
import stat
import struct
import sys
import zipfile
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

from rootwalk.openrca import BankEnvironment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import fetch_openrca_bank as fetcher_module  # noqa: E402
from scripts.fetch_openrca_bank import (  # noqa: E402
    download_mirror_file,
    fetch_openrca_bank,
    required_members,
    validate_content_range,
)
from scripts.prepare_bank_suite import (  # noqa: E402
    CASE_PLAN_SHA256,
    DEFAULT_ROW_IDS,
    IMPACT_ONLY_ROLE,
    MAX_JS_SAFE_INTEGER,
    SELECTION_SHA256,
    SOURCE_FILES,
    SPEC_PATH,
    V2_CASE_PLAN_SHA256,
    V2_SELECTION_SHA256,
    V2_SPEC_PATH,
    case_plan_sha256,
    plan_cases,
    prepare_suite,
    source_files_sha256,
    source_record_id,
)

TZ = ZoneInfo("Asia/Shanghai")
FROZEN_INSTRUCTIONS = {
    2: "During the specified time range of March 6, 2021, from 06:00 to 06:30, there was a single failure in the system. The component responsible for this failure and the reason behind it remain unknown. Please determine the root cause component and the root cause reason.",
    4: "A single failure occurred within the time range of March 6, 2021, from 00:30 to 01:00. The specific component affected by this failure and the reason behind it are currently unknown. You are tasked with identifying the root cause component and the root cause reason for this failure.",
    9: "Within the specified time range of March 7, 2021, from 00:30 to 01:00, the system experienced a failure. The component responsible for this failure and the underlying reasons remain unknown. Please determine the root cause component and the root cause reason for this failure.",
    17: "During the specified time range of March 9, 2021, from 11:00 to 11:30, there was one documented failure in the system. Currently, the root cause component and the underlying reason for this failure are unknown and need to be identified.",
    18: "A failure was detected during the time range of March 9, 2021, from 13:00 to 13:30. The specific component that caused this failure and the underlying reason for it are currently unknown. Please identify the root cause component and the root cause reason for this failure.",
    24: "During the specified time range of March 9, 2021, from 21:00 to 21:30, a failure occurred in the system. The exact component causing the failure and the underlying reason for this failure are currently unknown. Please identify the root cause component and the root cause reason for this incident.",
    59: "During the time range of March 9, 2021, from 15:30 to 16:00, there was a failure in the system. The exact component that caused the failure and the reason behind it are currently unknown. You are tasked with identifying the root cause component and the root cause reason for this failure.",
    71: "On March 6, 2021, between 03:30 and 04:00, the system experienced a single failure. Both the root cause component and the reason for this failure are unknown. Your task is to identify the root cause component and determine the reason for the failure.",
    72: "On March 6, 2021, between 20:30 and 21:00, a failure was observed in the system. However, the specific component responsible for this failure and the underlying reason remain unknown. You need to determine the root cause component and root cause reason for the observed failure.",
    75: "Within the time range of March 7, 2021, from 08:00 to 08:30, there was one reported failure in the system. The exact component responsible for this failure and the underlying reason for it are currently unknown. Your task is to identify the root cause component and the root cause reason.",
    76: "During the specified time range of March 7, 2021, from 11:00 to 11:30, a failure was detected in the system. The exact component affected by this failure and the underlying reasons are currently unknown. You are tasked with identifying the root cause component and the root cause reason.",
    100: "During the time range of March 24, 2021, from 19:00 to 19:30, there was one reported failure in the system. The specific component responsible for this failure, as well as the reason behind it, are currently unknown. Please identify the root cause component and the reason for the failure.",
    105: "During the time range of March 25, 2021, from 14:30 to 15:00, one failure occurred in the system. The component responsible for this failure and the underlying reason are currently unknown. You are tasked with identifying the root cause component and the root cause reason.",
    115: "During the specified time range of March 23, 2021, from 01:00 to 01:30, the system experienced one failure. The exact component that caused the failure and the reason behind it are currently unknown. You need to identify the root cause component and the root cause reason.",
    122: "On March 25, 2021, between 09:00 and 09:30, there was a single failure observed in the system. Currently, the root cause component and the reason for this failure are unknown. The goal is to identify the root cause component and determine the reason for this failure.",
    123: "On March 23, 2021, during the time range of 00:00 to 00:30, there was a recorded failure within the system. The details regarding which specific component failed and the reason behind this failure remain unknown. You are tasked with identifying the root cause component and the root cause reason for this failure.",
    124: "On March 23, 2021, between 18:30 and 19:00, the system experienced a single failure. The components affected and the reasons behind this failure are currently unknown. Your task is to identify the root cause component and the underlying reason for this failure.",
    128: "During the time range of March 25, 2021, from 11:30 to 12:00, a failure occurred, but the exact component responsible for the failure and the underlying reason is currently unknown. Your task is to identify the root cause component and the root cause reason for the incident.",
}


def _epoch(value: str, *, milliseconds: bool = False) -> str:
    result = int(datetime.fromisoformat(value).timestamp())
    return str(result * 1000 if milliseconds else result)


def _write_csv(path: Path, fields: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _write_source_provenance(source: Path, spec_path: Path, member: str) -> None:
    path = source / member
    data = path.read_bytes()
    archive_bytes = json.loads(spec_path.read_text())["official_archive"]["bytes"]
    key = hashlib.sha256(member.encode()).hexdigest()[:16]
    provenance = source / f".openrca/provenance/{key}.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(
        json.dumps(
            {
                "name": member,
                "size": len(data),
                "compressed_size": max(1, len(data) // 2),
                "offset": int(hashlib.sha256(member.encode()).hexdigest()[:8], 16)
                % 1_000_000_000,
                "crc32": f"{zlib.crc32(data) & 0xFFFFFFFF:08x}",
                "sha256": hashlib.sha256(data).hexdigest(),
                "archive_bytes": archive_bytes,
            }
        )
    )


def _synthetic_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "source"
    upstream = tmp_path / "upstream"
    upstream.mkdir(parents=True)
    (upstream / "bank_schema.py").write_text(
        'cand = """## POSSIBLE ROOT CAUSE REASONS:\n\n'
        "- high CPU usage\n\n"
        "## POSSIBLE ROOT CAUSE COMPONENTS:\n\n"
        '- Tomcat01\n- Tomcat02\n"""\n',
        encoding="utf-8",
    )
    spec = json.loads(SPEC_PATH.read_text())
    spec["bank_schema_sha256"] = hashlib.sha256(
        (upstream / "bank_schema.py").read_bytes()
    ).hexdigest()
    spec_path = tmp_path / "synthetic-suite.json"
    spec_path.write_text(json.dumps(spec))
    query_rows = [
        ["unused", "task_1", f"SECRET_LABEL_{row_id}"] for row_id in range(129)
    ]
    for row_id, instruction in FROZEN_INSTRUCTIONS.items():
        query_rows[row_id] = [instruction, "task_6", f"SECRET_LABEL_{row_id}"]
    _write_csv(
        source / "Bank/query.csv",
        ["instruction", "task_index", "scoring_points"],
        query_rows,
    )
    _write_csv(
        source / "Bank/record.csv",
        ["datetime", "component", "reason"],
        [["SECRET_RECORD", "SECRET_COMPONENT", "SECRET_REASON"]],
    )

    day = source / "Bank/telemetry/2021_03_06"
    seconds = [
        _epoch("2021-03-06T00:00:00+08:00"),
        _epoch("2021-03-06T00:30:00+08:00"),
        _epoch("2021-03-06T01:00:00+08:00"),
        _epoch("2021-03-06T05:30:00+08:00"),
        _epoch("2021-03-06T06:00:00+08:00"),
        _epoch("2021-03-06T06:30:00+08:00"),
    ]
    _write_csv(
        day / "metric/metric_app.csv",
        ["timestamp", "rr", "sr", "cnt", "mrt", "tc"],
        [
            [timestamp, "100", "100", str(index), "1", "ServiceTest1"]
            for index, timestamp in enumerate(seconds)
        ],
    )
    _write_csv(
        day / "metric/metric_container.csv",
        ["timestamp", "cmdb_id", "kpi_name", "value"],
        [
            [timestamp, "Tomcat01", "cpu", str(index)]
            for index, timestamp in enumerate(seconds)
        ],
    )
    _write_csv(
        day / "log/log_service.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [
            [f"log-{index}", timestamp, "Tomcat01", "service", f"message-{index}"]
            for index, timestamp in enumerate(seconds)
        ],
    )
    trace_rows: list[list[str]] = []
    for label, timestamp in (
        ("early", "2021-03-06T00:31:00+08:00"),
        ("late", "2021-03-06T06:01:00+08:00"),
    ):
        raw = _epoch(timestamp, milliseconds=True)
        trace_rows.extend(
            [
                [raw, "Tomcat01", "0", f"parent-{label}", f"trace-{label}", "1"],
                [
                    raw,
                    "Tomcat02",
                    f"parent-{label}",
                    f"child-{label}",
                    f"trace-{label}",
                    "2",
                ],
            ]
        )
    _write_csv(
        day / "trace/trace_span.csv",
        ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"],
        trace_rows,
    )
    _write_source_provenance(source, spec_path, "Bank/query.csv")
    for relative, _multiplier, _entity in SOURCE_FILES:
        _write_source_provenance(
            source, spec_path, f"Bank/telemetry/2021_03_06/{relative}"
        )
    return source, upstream, spec_path


def _write_cross_midnight_day(
    source: Path,
    spec_path: Path,
    source_date: str,
    timestamp: str,
    value: str,
) -> None:
    day = source / f"Bank/telemetry/{source_date}"
    second = _epoch(timestamp)
    millisecond = _epoch(timestamp, milliseconds=True)
    _write_csv(
        day / "metric/metric_app.csv",
        ["timestamp", "rr", "sr", "cnt", "mrt", "tc"],
        [[second, "100", "100", value, "1", "ServiceTest1"]],
    )
    _write_csv(
        day / "metric/metric_container.csv",
        ["timestamp", "cmdb_id", "kpi_name", "value"],
        [[second, "Tomcat01", "cpu", value]],
    )
    _write_csv(
        day / "log/log_service.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [[f"log-{value}", second, "Tomcat01", "service", f"message-{value}"]],
    )
    _write_csv(
        day / "trace/trace_span.csv",
        ["timestamp", "cmdb_id", "parent_id", "span_id", "trace_id", "duration"],
        [
            [
                millisecond,
                "Tomcat01",
                "0",
                f"span-{value}",
                f"trace-{value}",
                "1",
            ]
        ],
    )
    for relative, _multiplier, _entity in SOURCE_FILES:
        _write_source_provenance(
            source, spec_path, f"Bank/telemetry/{source_date}/{relative}"
        )


def _hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _append_source_rows(
    source: Path,
    spec_path: Path,
    member: str,
    rows: list[dict[str, str]],
) -> None:
    path = source / member
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = list(reader.fieldnames or ())
        existing = list(reader)
    _write_csv(
        path,
        fields,
        [[row.get(field, "") for field in fields] for row in [*existing, *rows]],
    )
    _write_source_provenance(source, spec_path, member)


def test_planner_has_the_frozen_rows_and_explicit_midnight_window() -> None:
    spec, plans = plan_cases()

    assert tuple(plan.row_id for plan in plans) == DEFAULT_ROW_IDS
    assert spec["selection_sha256"] == SELECTION_SHA256
    assert case_plan_sha256(plans) == CASE_PLAN_SHA256
    midnight = next(plan for plan in plans if plan.row_id == 123)
    assert midnight.source_dates == ("2021_03_22", "2021_03_23")
    assert midnight.allowed_start.isoformat() == "2021-03-22T23:30:00+08:00"
    assert midnight.allowed_end.isoformat() == "2021-03-23T00:30:00+08:00"


def test_same_day_cases_share_one_source_scan_and_are_bank_compatible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    private_query = source / "private_source/Bank/query.csv"
    private_query.parent.mkdir(parents=True)
    (source / "Bank/query.csv").replace(private_query)
    private_query.chmod(0o600)
    source_files = {
        (source / "Bank/telemetry/2021_03_06" / relative).resolve()
        for relative, _multiplier, _entity in SOURCE_FILES
    }
    calls = {path: 0 for path in source_files}
    original_open = Path.open

    def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        resolved = self.resolve()
        if resolved in calls:
            calls[resolved] += 1
        return cast(Any, original_open)(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counting_open)
    output = prepare_suite(
        source, upstream, tmp_path / "prepared", [2, 4], spec_path=spec_path
    )

    assert set(calls.values()) == {1}
    late = output / "bank-row-0002/public_input"
    early = output / "bank-row-0004/public_input"
    assert [
        row["cnt"] for row in _read_csv(late / "telemetry/metric/metric_app.csv")
    ] == [
        "3",
        "4",
    ]
    assert [
        row["cnt"] for row in _read_csv(early / "telemetry/metric/metric_app.csv")
    ] == [
        "0",
        "1",
    ]
    assert BankEnvironment(late).task.source_row_id == 2
    assert (
        BankEnvironment(early).query_metric("bank:entity:Tomcat01", "cpu")["status"]
        == "success"
    )
    assert len(BankEnvironment(late).topology.edges) == 1
    assert len(BankEnvironment(early).topology.edges) == 1


def test_business_test_metrics_are_queryable_impact_only_not_root_candidates(
    tmp_path: Path,
) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    output = prepare_suite(
        source, upstream, tmp_path / "prepared", [2], spec_path=spec_path
    )
    public = output / "bank-row-0002/public_input"
    environment = BankEnvironment(public)
    impact_id = "bank:test:ServiceTest1"
    root_id = "bank:entity:Tomcat01"

    impact = environment.query_metric_profile(impact_id, ["cnt"])
    assert impact["status"] == "success"
    assert impact["evidence_role"] == IMPACT_ONLY_ROLE
    assert impact["metric_profiles"][0]["evidence_role"] == IMPACT_ONLY_ROLE
    assert impact["metric_profiles"][0]["reference"]["median"] == 3
    assert impact["metric_profiles"][0]["incident"]["median"] == 4

    nodes = json.loads((public / "entity_mapping.json").read_text())
    impact_node = next(node for node in nodes if node["canonical_id"] == impact_id)
    assert impact_node["entity_type"] == "business_test"
    assert impact_node["evidence_role"] == IMPACT_ONLY_ROLE
    assert impact_node["public_candidate_match"] is None
    root_node = next(node for node in nodes if node["canonical_id"] == root_id)
    assert root_node["evidence_role"] == "root_cause_candidate"

    stats = [
        json.loads(line)
        for line in (public / "metric_window_stats.jsonl").read_text().splitlines()
    ]
    impact_stat = next(
        item
        for item in stats
        if item["canonical_id"] == impact_id and item["metric_id"] == "cnt"
    )
    assert impact_stat["evidence_role"] == IMPACT_ONLY_ROLE
    root_stat = next(
        item
        for item in stats
        if item["canonical_id"] == root_id and item["metric_id"] == "cpu"
    )
    assert root_stat["evidence_role"] == "root_cause_candidate"

    catalog = json.loads((public / "metric_catalog.json").read_text())
    impact_binding = next(
        item
        for item in catalog["metric_bindings"]
        if item["canonical_id"] == impact_id and item["metric_id"] == "cnt"
    )
    assert impact_binding["evidence_role"] == IMPACT_ONLY_ROLE
    assert impact_binding["source_path"] == "metric/metric_app.csv"
    assert impact_binding["source_member_filter"] == [
        "Bank/telemetry/2021_03_06/metric/metric_app.csv"
    ]
    root_binding = next(
        item
        for item in catalog["metric_bindings"]
        if item["canonical_id"] == root_id and item["metric_id"] == "cpu"
    )
    assert root_binding["evidence_role"] == "root_cause_candidate"

    overview = environment.summarize_anomalies()
    candidate_ids = {
        candidate["entity_id"] for candidate in overview["fault_type_candidates"]
    }
    assert root_id in candidate_ids
    assert impact_id not in candidate_ids
    profile = json.loads((public / "dataset_profile.json").read_text())
    assert profile["impact_only_entity_count"] == 1
    assert profile["impact_only_metric_series_count"] == 4


def test_same_raw_id_keeps_impact_and_resource_identities_separate(
    tmp_path: Path,
) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    day = "Bank/telemetry/2021_03_06"
    raw_id = "ServiceTest1"
    reference_second = _epoch("2021-03-06T05:30:00+08:00")
    incident_second = _epoch("2021-03-06T06:00:00+08:00")
    _append_source_rows(
        source,
        spec_path,
        f"{day}/metric/metric_container.csv",
        [
            {
                "timestamp": reference_second,
                "cmdb_id": raw_id,
                "kpi_name": "cnt",
                "value": "30",
            },
            {
                "timestamp": incident_second,
                "cmdb_id": raw_id,
                "kpi_name": "cnt",
                "value": "40",
            },
        ],
    )
    _append_source_rows(
        source,
        spec_path,
        f"{day}/log/log_service.csv",
        [
            {
                "log_id": "resource-collision",
                "timestamp": incident_second,
                "cmdb_id": raw_id,
                "log_name": "service",
                "value": "same raw identifier, different resource identity",
            }
        ],
    )
    _append_source_rows(
        source,
        spec_path,
        f"{day}/trace/trace_span.csv",
        [
            {
                "timestamp": _epoch("2021-03-06T06:00:00+08:00", milliseconds=True),
                "cmdb_id": raw_id,
                "parent_id": "0",
                "span_id": "resource-collision",
                "trace_id": "resource-collision",
                "duration": "3",
            }
        ],
    )

    output = prepare_suite(
        source, upstream, tmp_path / "prepared", [2], spec_path=spec_path
    )
    public = output / "bank-row-0002/public_input"
    environment = BankEnvironment(public)
    impact_id = f"bank:test:{raw_id}"
    resource_id = f"bank:entity:{raw_id}"

    nodes = json.loads((public / "entity_mapping.json").read_text())
    colliding = [node for node in nodes if node["original_id"] == raw_id]
    assert {node["canonical_id"] for node in colliding} == {impact_id, resource_id}
    by_role = {node["evidence_role"]: node for node in colliding}
    assert by_role[IMPACT_ONLY_ROLE]["sources"] == ["metric/metric_app.csv"]
    assert by_role["root_cause_candidate"]["sources"] == [
        "log/log_service.csv",
        "metric/metric_container.csv",
        "trace/trace_span.csv",
    ]

    impact = environment.query_metric_profile(impact_id, ["cnt"])
    resource = environment.query_metric_profile(resource_id, ["cnt"])
    assert impact["metric_profiles"][0]["reference"]["median"] == 3
    assert impact["metric_profiles"][0]["incident"]["median"] == 4
    assert impact["evidence_role"] == IMPACT_ONLY_ROLE
    assert resource["metric_profiles"][0]["reference"]["median"] == 30
    assert resource["metric_profiles"][0]["incident"]["median"] == 40
    assert resource["evidence_role"] == "root_cause_candidate"

    stats = [
        json.loads(line)
        for line in (public / "metric_window_stats.jsonl").read_text().splitlines()
    ]
    cnt_stats = [
        item
        for item in stats
        if item["entity_id"] == raw_id and item["metric_id"] == "cnt"
    ]
    assert {(item["canonical_id"], item["evidence_role"]) for item in cnt_stats} == {
        (impact_id, IMPACT_ONLY_ROLE),
        (resource_id, "root_cause_candidate"),
    }

    catalog = json.loads((public / "metric_catalog.json").read_text())
    cnt_bindings = {
        item["canonical_id"]: item
        for item in catalog["metric_bindings"]
        if item["metric_id"] == "cnt" and item["original_entity_key"] == raw_id
    }
    assert cnt_bindings[impact_id]["source_path"] == "metric/metric_app.csv"
    assert cnt_bindings[impact_id]["layout"] == "wide"
    assert cnt_bindings[impact_id]["evidence_role"] == IMPACT_ONLY_ROLE
    assert cnt_bindings[resource_id]["source_path"] == ("metric/metric_container.csv")
    assert cnt_bindings[resource_id]["layout"] == "long"
    assert cnt_bindings[resource_id]["evidence_role"] == "root_cause_candidate"

    assert all(
        edge.source_id != impact_id and edge.target_id != impact_id
        for edge in environment.topology.edges
    )
    candidates = environment.summarize_anomalies()["fault_type_candidates"]
    assert all(candidate["entity_id"] != impact_id for candidate in candidates)


def test_cross_midnight_case_requires_and_scans_both_source_days(
    tmp_path: Path,
) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    _write_cross_midnight_day(
        source, spec_path, "2021_03_23", "2021-03-23T00:15:00+08:00", "23"
    )
    with pytest.raises(FileNotFoundError):
        prepare_suite(
            source,
            upstream,
            tmp_path / "missing-prior-day",
            [123],
            spec_path=spec_path,
        )

    _write_cross_midnight_day(
        source, spec_path, "2021_03_22", "2021-03-22T23:45:00+08:00", "22"
    )
    output = prepare_suite(
        source, upstream, tmp_path / "prepared", [123], spec_path=spec_path
    )
    public = output / "bank-row-0123/public_input"
    app_rows = _read_csv(public / "telemetry/metric/metric_app.csv")
    assert [row["cnt"] for row in app_rows] == ["22", "23"]
    source_records = [int(row["_source_record"]) for row in app_rows]
    assert source_records == [
        source_record_id("2021_03_22", 1),
        source_record_id("2021_03_23", 1),
    ]
    assert len(set(source_records)) == 2
    assert all(value <= MAX_JS_SAFE_INTEGER for value in source_records)
    assert [int(float(value)) for value in json.loads(json.dumps(source_records))] == (
        source_records
    )
    assert [row["_source_member"] for row in app_rows] == [
        "Bank/telemetry/2021_03_22/metric/metric_app.csv",
        "Bank/telemetry/2021_03_23/metric/metric_app.csv",
    ]
    profile = json.loads((public / "dataset_profile.json").read_text())
    assert profile["files"]["metric/metric_app.csv"]["source_members"] == [
        "Bank/telemetry/2021_03_22/metric/metric_app.csv",
        "Bank/telemetry/2021_03_23/metric/metric_app.csv",
    ]
    assert BankEnvironment(public).task.reference_range is not None


def test_outputs_strip_labels_have_complete_hashes_and_repeat_stably(
    tmp_path: Path,
) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    output = prepare_suite(
        source, upstream, tmp_path / "prepared", [2, 4], spec_path=spec_path
    )
    first = _hashes(output)

    manifest = json.loads((output / "suite_manifest.json").read_text())
    assert manifest["row_ids"] == [2, 4]
    assert manifest["selection_sha256"] == SELECTION_SHA256
    assert manifest["case_plan_sha256"] == CASE_PLAN_SHA256
    assert manifest["upstream_commit"] == "c1bd4af7f635171a1c31cdd567c07d698dff6abc"
    case_manifest = json.loads(
        (output / "bank-row-0002/public_input/manifest.json").read_text()
    )
    assert case_manifest["schema_version"] == "rootwalk-dataset-slice-v2"
    provenance_fields = {
        "name",
        "size",
        "compressed_size",
        "offset",
        "crc32",
        "sha256",
        "archive_bytes",
    }
    assert len(case_manifest["telemetry_provenance"]) == 4
    assert all(
        provenance_fields <= item.keys()
        for item in case_manifest["telemetry_provenance"]
    )
    profile = json.loads(
        (output / "bank-row-0002/public_input/dataset_profile.json").read_text()
    )
    assert profile["files"]["metric/metric_app.csv"]["source_sha256"]
    assert profile["files"]["metric/metric_app.csv"]["source_time_range_utc"]
    assert profile["files"]["metric/metric_app.csv"]["source_entities"]
    assert json.loads(
        (output / "bank-row-0002/public_input/metric_catalog.json").read_text()
    )["log_names"] == {"service": 2}
    declared = {item["path"]: item["sha256"] for item in manifest["outputs"]}
    assert declared == {
        path: digest for path, digest in first.items() if path != "suite_manifest.json"
    }
    public_bytes = b"".join(
        path.read_bytes()
        for path in output.rglob("*")
        if path.is_file() and "public_input" in path.parts
    )
    assert b"SECRET_LABEL" not in public_bytes
    assert b"SECRET_RECORD" not in public_bytes
    assert b"scoring_points" not in public_bytes
    assert b"record.csv" not in public_bytes

    assert (
        prepare_suite(source, upstream, output, [2, 4], spec_path=spec_path) == output
    )
    assert _hashes(output) == first


@pytest.mark.parametrize(
    ("task_index", "instruction", "message"),
    [
        ("task_5", FROZEN_INSTRUCTIONS[2], "only task_6"),
        (
            "task_6",
            FROZEN_INSTRUCTIONS[2] + " ",
            "instruction SHA-256 mismatch",
        ),
    ],
)
def test_rejects_non_task6_and_instruction_drift(
    tmp_path: Path, task_index: str, instruction: str, message: str
) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    query = source / "Bank/query.csv"
    rows = _read_csv(query)
    rows[2]["task_index"] = task_index
    rows[2]["instruction"] = instruction
    _write_csv(query, list(rows[0]), [[row[key] for key in rows[0]] for row in rows])
    _write_source_provenance(source, spec_path, "Bank/query.csv")

    with pytest.raises(ValueError, match=message):
        prepare_suite(source, upstream, tmp_path / "prepared", [2], spec_path=spec_path)


def test_rejects_output_overlap_and_source_symlink_escape(tmp_path: Path) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    with pytest.raises(ValueError, match="separate from both input roots"):
        prepare_suite(source, upstream, source / "generated", [2], spec_path=spec_path)

    trace = source / "Bank/telemetry/2021_03_06/trace/trace_span.csv"
    outside = tmp_path / "outside.csv"
    outside.write_bytes(trace.read_bytes())
    trace.unlink()
    trace.symlink_to(outside)
    with pytest.raises(ValueError, match="must not be a symlink"):
        prepare_suite(source, upstream, tmp_path / "prepared", [2], spec_path=spec_path)

    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    (repository / ".git/HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(ValueError, match="outside every Git"):
        prepare_suite(
            source,
            upstream,
            repository / "prepared",
            [2],
            spec_path=spec_path,
        )


def test_rejects_raw_source_and_schema_tampering(tmp_path: Path) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    metric = source / "Bank/telemetry/2021_03_06/metric/metric_app.csv"
    metric.write_bytes(metric.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source (size|CRC|SHA-256)"):
        prepare_suite(
            source, upstream, tmp_path / "raw-tamper", [2], spec_path=spec_path
        )

    source, upstream, spec_path = _synthetic_inputs(tmp_path / "schema-case")
    schema = upstream / "bank_schema.py"
    schema.write_bytes(schema.read_bytes() + b"# tampered\n")
    with pytest.raises(ValueError, match="bank_schema.py SHA-256"):
        prepare_suite(
            source,
            upstream,
            tmp_path / "schema-tamper",
            [2],
            spec_path=spec_path,
        )


def test_failed_rebuild_keeps_existing_output_and_cleans_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, upstream, spec_path = _synthetic_inputs(tmp_path)
    output = prepare_suite(
        source, upstream, tmp_path / "prepared", [2], spec_path=spec_path
    )
    before = _hashes(output)

    def fail_finish(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("injected finish failure")

    monkeypatch.setattr("scripts.prepare_bank_suite._finish_case", fail_finish)
    with pytest.raises(RuntimeError, match="injected finish failure"):
        prepare_suite(source, upstream, output, [2], spec_path=spec_path)

    assert _hashes(output) == before
    assert not list(tmp_path.glob(".prepared.tmp-*"))


def test_source_record_rejects_daily_overflow() -> None:
    with pytest.raises(ValueError, match="99,999,999"):
        source_record_id("2021_03_06", 100_000_000)


def test_public_suite_spec_is_valid_json_and_contains_only_public_selection() -> None:
    spec = json.loads(SPEC_PATH.read_text())

    assert [case["row_id"] for case in spec["cases"]] == list(DEFAULT_ROW_IDS)
    assert all(case["task_index"] == "task_6" for case in spec["cases"])
    assert all(
        hashlib.sha256(FROZEN_INSTRUCTIONS[case["row_id"]].encode()).hexdigest()
        == case["instruction_sha256"]
        for case in spec["cases"]
    )
    projection = [
        {
            "instruction": FROZEN_INSTRUCTIONS[case["row_id"]],
            "row_id": case["row_id"],
            "task_index": case["task_index"],
        }
        for case in spec["cases"]
    ]
    canonical = (
        json.dumps(
            projection,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == SELECTION_SHA256
    assert "one documented failure" in FROZEN_INSTRUCTIONS[17]
    assert "A failure was detected" in FROZEN_INSTRUCTIONS[18]
    serialized = SPEC_PATH.read_text()
    assert "scoring_points" not in serialized
    assert "ground_truth" not in serialized
    members = required_members(spec)
    assert len(members) == len(set(members)) == 29
    assert "Bank/telemetry/2021_03_22/trace/trace_span.csv" in members
    assert "Bank/record.csv" not in members


def test_mirror_suite_is_frozen_to_available_public_files() -> None:
    spec, plans = plan_cases(V2_SPEC_PATH)

    assert len(plans) == 17
    assert 123 not in {plan.row_id for plan in plans}
    assert all("2021_03_22" not in plan.source_dates for plan in plans)
    assert spec["selection_sha256"] == V2_SELECTION_SHA256
    assert spec["case_plan_sha256"] == V2_CASE_PLAN_SHA256
    assert spec["source_files_sha256"] == source_files_sha256(spec["source_files"])
    members = required_members(spec)
    assert len(members) == len(set(members)) == 25
    assert set(members) == {
        item["path"] for item in cast(list[dict[str, object]], spec["source_files"])
    }
    assert "Bank/record.csv" not in members


def test_mirror_fetch_is_allowlisted_verified_and_reusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contents = {
        "Bank/query.csv": b"task_index,instruction,scoring_points\ntask_6,q,secret\n",
        "Bank/telemetry/2021_03_06/metric/metric_app.csv": b"timestamp,tc\n1,x\n",
    }
    entries = [
        {
            "bytes": len(content),
            "path": name,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for name, content in sorted(contents.items())
    ]
    spec = {
        "source_mirror": {
            "repository": "example/OpenRCA",
            "commit": "a" * 40,
            "url_template": "https://mirror.invalid/{repository}/{commit}/{path}",
        },
        "source_files": entries,
    }
    spec_path = tmp_path / "mirror-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    def planned(_path: Path) -> tuple[dict[str, Any], list[Any]]:
        return spec, []

    def members(_spec: dict[str, Any]) -> list[str]:
        return list(contents)

    monkeypatch.setattr(fetcher_module, "plan_cases", planned)
    monkeypatch.setattr(fetcher_module, "required_members", members)
    calls: list[str] = []

    def download(
        url: str, part: Path, expected_size: int, expected_sha256: str
    ) -> tuple[str, str]:
        name = next(name for name in contents if url.endswith(name))
        content = contents[name]
        assert len(content) == expected_size
        assert hashlib.sha256(content).hexdigest() == expected_sha256
        part.write_bytes(content)
        calls.append(name)
        return expected_sha256, f"{zlib.crc32(content) & 0xFFFFFFFF:08x}"

    source = tmp_path / "mirror-source"
    first = fetcher_module.fetch_openrca_bank(
        source,
        spec_path,
        mirror=True,
        mirror_fetcher=download,
    )

    assert calls == list(contents)
    assert len(first) == 2
    assert (source / "private_source/Bank/query.csv").read_bytes() == contents[
        "Bank/query.csv"
    ]
    assert (
        stat.S_IMODE((source / "private_source/Bank/query.csv").stat().st_mode) == 0o600
    )
    assert not (source / "Bank/query.csv").exists()
    assert all(item["source_kind"] == "huggingface_mirror" for item in first)

    def no_download(*_args: object) -> tuple[str, str]:
        raise AssertionError("verified mirror files must be reused")

    second = fetcher_module.fetch_openrca_bank(
        source,
        spec_path,
        mirror=True,
        mirror_fetcher=no_download,
    )
    assert second == first


def test_mirror_download_resumes_and_checks_the_pinned_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"abcdef"
    part = tmp_path / "source.part"
    part.write_bytes(content[:2])
    requests: list[object] = []

    class Response(io.BytesIO):
        headers = {"Content-Range": "bytes 2-5/6"}

        def getcode(self) -> int:
            return 206

    def open_range(request: object, timeout: int) -> Response:
        requests.append(request)
        assert timeout == 240
        return Response(content[2:])

    monkeypatch.setattr(fetcher_module.urllib.request, "urlopen", open_range)
    digest, crc32 = download_mirror_file(
        "https://mirror.invalid/file",
        part,
        len(content),
        hashlib.sha256(content).hexdigest(),
    )

    assert part.read_bytes() == content
    assert digest == hashlib.sha256(content).hexdigest()
    assert crc32 == f"{zlib.crc32(content) & 0xFFFFFFFF:08x}"
    assert len(requests) == 1
    assert cast(Any, requests[0]).get_header("Range") == "bytes=2-"


@pytest.mark.parametrize(
    ("status", "content_range", "message"),
    [
        (200, None, "resume Range"),
        (206, "bytes 1-5/6", "Content-Range"),
    ],
)
def test_mirror_resume_rejects_invalid_http_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    content_range: str | None,
    message: str,
) -> None:
    part = tmp_path / "source.part"
    part.write_bytes(b"ab")

    class Response(io.BytesIO):
        headers = {"Content-Range": content_range} if content_range else {}

        def getcode(self) -> int:
            return status

    def open_range(*_args: object, **_kwargs: object) -> Response:
        return Response(b"cdef")

    monkeypatch.setattr(fetcher_module.urllib.request, "urlopen", open_range)

    with pytest.raises(RuntimeError, match=message):
        download_mirror_file(
            "https://mirror.invalid/file",
            part,
            6,
            hashlib.sha256(b"abcdef").hexdigest(),
        )
    assert part.read_bytes() == b"ab"


def test_mirror_download_rejects_bad_digest_and_part_symlink(
    tmp_path: Path,
) -> None:
    part = tmp_path / "source.part"
    part.write_bytes(b"abcdef")
    with pytest.raises(ValueError, match="size or SHA-256"):
        download_mirror_file(
            "https://mirror.invalid/file",
            part,
            6,
            hashlib.sha256(b"abcdeg").hexdigest(),
        )
    assert not part.exists()

    target = tmp_path / "target"
    target.write_bytes(b"abcdef")
    part.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symlink"):
        download_mirror_file(
            "https://mirror.invalid/file",
            part,
            6,
            hashlib.sha256(b"abcdef").hexdigest(),
        )


def test_mirror_download_resumes_after_a_short_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content = b"abcdef"
    part = tmp_path / "source.part"
    part.write_bytes(content[:2])
    responses = iter(
        [
            ("bytes 2-5/6", content[2:4]),
            ("bytes 4-5/6", content[4:]),
        ]
    )

    class Response(io.BytesIO):
        def __init__(self, content_range: str, body: bytes) -> None:
            super().__init__(body)
            self.headers = {"Content-Range": content_range}

        def getcode(self) -> int:
            return 206

    def open_range(*_args: object, **_kwargs: object) -> Response:
        return Response(*next(responses))

    monkeypatch.setattr(fetcher_module.urllib.request, "urlopen", open_range)
    with pytest.raises(ValueError, match="size or SHA-256"):
        download_mirror_file(
            "https://mirror.invalid/file",
            part,
            len(content),
            hashlib.sha256(content).hexdigest(),
        )
    assert part.read_bytes() == content[:4]

    digest, _crc32 = download_mirror_file(
        "https://mirror.invalid/file",
        part,
        len(content),
        hashlib.sha256(content).hexdigest(),
    )
    assert part.read_bytes() == content
    assert digest == hashlib.sha256(content).hexdigest()


def test_planner_rejects_a_coherently_tampered_window(tmp_path: Path) -> None:
    spec = json.loads(SPEC_PATH.read_text())
    case = spec["cases"][0]
    case["reference_range"] = {
        "start": "2021-03-06T06:00:00+08:00",
        "end": "2021-03-06T06:30:00+08:00",
        "end_exclusive": True,
    }
    case["incident_range"] = {
        "start": "2021-03-06T06:30:00+08:00",
        "end": "2021-03-06T07:00:00+08:00",
        "end_exclusive": True,
    }
    case["allowed_query_range"] = {
        "start": "2021-03-06T06:00:00+08:00",
        "end": "2021-03-06T07:00:00+08:00",
        "end_exclusive": True,
    }
    path = tmp_path / "tampered-suite.json"
    path.write_text(json.dumps(spec))

    with pytest.raises(ValueError, match="case plan SHA-256"):
        plan_cases(path)


def _synthetic_archive(spec: dict[str, Any]) -> tuple[bytes, dict[str, str]]:
    members = {"Bank/query.csv": "instruction,task_index\npublic,task_6\n"}
    for name in required_members(spec)[1:]:
        if name.endswith("metric_app.csv"):
            content = "timestamp,tc\n1,test\n"
        elif name.endswith("metric_container.csv"):
            content = "timestamp,cmdb_id,kpi_name,value\n1,Tomcat01,cpu,1\n"
        elif name.endswith("log_service.csv"):
            content = "log_id,timestamp,cmdb_id,log_name,value\n1,1,Tomcat01,app,ok\n"
        else:
            content = (
                "timestamp,cmdb_id,parent_id,span_id,trace_id,duration\n"
                "1,Tomcat01,0,span,trace,1\n"
            )
        members[name] = content
    members["Bank/record.csv"] = "SECRET_RECORD\n"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return buffer.getvalue(), members


def _archive_spec(path: Path, archive: bytes) -> None:
    end = archive.rfind(b"PK\x05\x06")
    eocd = struct.unpack_from("<4s4H2IH", archive, end)
    directory = archive[eocd[6] : eocd[6] + eocd[5]]
    spec = json.loads(SPEC_PATH.read_text())
    spec["official_archive"] = {
        "bytes": len(archive),
        "central_directory_sha256": hashlib.sha256(directory).hexdigest(),
        "file_id": "synthetic",
        "name": "Bank.zip",
    }
    path.write_text(json.dumps(spec))


def test_range_fetcher_downloads_only_allowlisted_members_and_reuses_hashes(
    tmp_path: Path,
) -> None:
    spec = json.loads(SPEC_PATH.read_text())
    archive, members = _synthetic_archive(spec)
    spec_path = tmp_path / "suite.json"
    _archive_spec(spec_path, archive)
    calls: list[tuple[int, int]] = []

    def fetch_range(
        _url: str, start: int, end: int, total: int, destination: Path
    ) -> None:
        assert total == len(archive)
        calls.append((start, end))
        destination.write_bytes(archive[start : end + 1])

    source = tmp_path / "downloaded"
    results = fetch_openrca_bank(source, spec_path, fetch_range)

    assert len(results) == 29
    assert len(calls) == 30  # one central-directory tail plus 29 allowed members
    query = source / "private_source/Bank/query.csv"
    assert query.read_text() == members["Bank/query.csv"]
    assert stat.S_IMODE(query.stat().st_mode) == 0o600
    assert not (source / "Bank/record.csv").exists()
    assert not (source / "private_source/Bank/record.csv").exists()
    assert (
        source / "Bank/telemetry/2021_03_06/trace/trace_span.csv"
    ).read_text() == members["Bank/telemetry/2021_03_06/trace/trace_span.csv"]
    assert all(result["crc_verified"] for result in results)
    assert all(result["sha256_verified"] for result in results)

    assert fetch_openrca_bank(source, spec_path, fetch_range) == results
    assert len(calls) == 30

    archive_path = tmp_path / "Bank.zip"
    archive_path.write_bytes(archive)

    def no_network(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("local archive mode attempted a network range request")

    local_source = tmp_path / "local-download"
    local_results = fetch_openrca_bank(
        local_source,
        spec_path,
        no_network,
        archive_path=archive_path,
    )

    def without_location(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {key: value for key, value in item.items() if key != "source_url"}
            for item in items
        ]

    assert without_location(local_results) == without_location(results)
    assert {
        item["local_path"]: hashlib.sha256(
            (source / item["local_path"]).read_bytes()
        ).hexdigest()
        for item in results
    } == {
        item["local_path"]: hashlib.sha256(
            (local_source / item["local_path"]).read_bytes()
        ).hexdigest()
        for item in local_results
    }
    remote_provenance = [
        json.loads(path.read_text())
        for path in sorted((source / ".openrca/provenance").glob("*.json"))
    ]
    local_provenance = [
        json.loads(path.read_text())
        for path in sorted((local_source / ".openrca/provenance").glob("*.json"))
    ]
    assert without_location(local_provenance) == without_location(remote_provenance)
    assert all(item["source_url"] == "local-archive:Bank.zip" for item in local_results)


def test_range_fetcher_rejects_bad_content_range_and_git_destination(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="unexpected Content-Range"):
        validate_content_range("bytes 0-8/10", 0, 9, 10)

    spec = json.loads(SPEC_PATH.read_text())
    archive, _members = _synthetic_archive(spec)
    spec_path = tmp_path / "suite.json"
    _archive_spec(spec_path, archive)
    repository = tmp_path / "repository"
    (repository / ".git").mkdir(parents=True)
    (repository / ".git/HEAD").write_text("ref: refs/heads/main\n")
    with pytest.raises(ValueError, match="outside every Git"):
        fetch_openrca_bank(repository / "data", spec_path)

    source = tmp_path / "symlink-source"
    outside = tmp_path / "outside-cache"
    source.mkdir()
    outside.mkdir()
    (source / ".openrca").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must not be a symlink"):
        fetch_openrca_bank(source, spec_path)

    archive_path = tmp_path / "Bank.zip"
    archive_path.write_bytes(archive)
    archive_link = tmp_path / "Bank-link.zip"
    archive_link.symlink_to(archive_path)
    with pytest.raises(ValueError, match="archive must not be a symlink"):
        fetch_openrca_bank(
            tmp_path / "archive-symlink-output",
            spec_path,
            archive_path=archive_link,
        )

    tampered = bytearray(archive)
    end = archive.rfind(b"PK\x05\x06")
    directory_offset = struct.unpack_from("<4s4H2IH", archive, end)[6]
    tampered[directory_offset] ^= 1
    tampered_archive = tmp_path / "Bank-tampered.zip"
    tampered_archive.write_bytes(tampered)
    with pytest.raises(RuntimeError, match="central directory SHA-256"):
        fetch_openrca_bank(
            tmp_path / "tampered-archive-output",
            spec_path,
            archive_path=tampered_archive,
        )

    git_archive_root = tmp_path / "archive-repository"
    (git_archive_root / ".git").mkdir(parents=True)
    (git_archive_root / ".git/HEAD").write_text("ref: refs/heads/main\n")
    git_archive = git_archive_root / "Bank.zip"
    git_archive.write_bytes(archive)
    with pytest.raises(ValueError, match="archive must be outside every Git"):
        fetch_openrca_bank(
            tmp_path / "git-archive-output",
            spec_path,
            archive_path=git_archive,
        )
