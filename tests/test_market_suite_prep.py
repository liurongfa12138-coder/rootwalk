# pyright: reportPrivateUsage=false, reportMissingImports=false, reportUnknownMemberType=false

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest

from rootwalk.openrca import MarketEnvironment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_market_suite as prepare_market_suite_module  # noqa: E402
from scripts.prepare_market_suite import (  # noqa: E402
    DEFAULT_SOURCE_SCHEMA_PATH,
    EXPECTED_FILES,
    MARKET_COMPONENTS,
    MARKET_FAULT_TYPES,
    MARKET_SOURCE_REVISION,
    CaseSpec,
    prepare_market_public_input,
)


def _write_csv(path: Path, fields: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _synthetic_source_day(root: Path, date: str = "2022_01_01") -> Path:
    day = root / date
    ref = 1640966400
    inc = 1640967000
    end = 1640967600
    _write_csv(
        day / "log/log_proxy.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [
            ["proxy-ref", ref, "pod-a", "proxy", "reference proxy"],
            ["proxy-inc", inc + 0.75, "pod-a", "proxy", "incident proxy"],
            ["proxy-end", end, "pod-a", "proxy", "drop"],
        ],
    )
    _write_csv(
        day / "log/log_service.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [
            ["service-ref", ref, "pod-a", "service", "reference service"],
            ["service-inc", inc + 0.50, "pod-a", "service", ""],
            ["service-end", end, "pod-a", "service", "drop"],
        ],
    )
    for name, base in {
        "container": 1.0,
        "mesh": 2.0,
        "node": 3.0,
        "runtime": 4.0,
    }.items():
        _write_csv(
            day / f"metric/metric_{name}.csv",
            ["timestamp", "cmdb_id", "kpi_name", "value"],
            [
                [ref, "pod-a", "cpu", base],
                [inc, "pod-a", "cpu", base + 10],
                [end, "pod-a", "cpu", 99],
            ],
        )
    _write_csv(
        day / "metric/metric_service.csv",
        ["service", "timestamp", "rr", "sr", "mrt", "count"],
        [
            ["checkout", ref, 1, 2, 3, 4],
            ["checkout", inc, 11, 12, 13, 14],
            ["checkout", end, 99, 99, 99, 99],
        ],
    )
    _write_csv(
        day / "trace/trace_span.csv",
        [
            "timestamp",
            "cmdb_id",
            "span_id",
            "trace_id",
            "duration",
            "type",
            "status_code",
            "operation_name",
            "parent_span",
        ],
        [
            [inc * 1000, "pod-a", "span-parent", "trace-1", 8, "", 0, "root", ""],
            [
                inc * 1000 + 1,
                "pod-b",
                "span-child",
                "trace-1",
                6,
                "SERVER",
                0,
                "child",
                "span-parent",
            ],
            [end * 1000, "pod-z", "span-outside", "trace-x", 5, "", 0, "drop", ""],
        ],
    )
    return day


def _case_spec(path: Path, *, cloudbed: str = "cloudbed-a") -> Path:
    spec = {
        "cloudbed": cloudbed,
        "row_id": 7,
        "question": "Identify the synthetic Market root cause.",
        "allowed_query_range": {
            "start": "2022-01-01T00:00:00+08:00",
            "end": "2022-01-01T00:20:00+08:00",
            "end_exclusive": True,
        },
        "reference_range": {
            "start": "2022-01-01T00:00:00+08:00",
            "end": "2022-01-01T00:10:00+08:00",
            "end_exclusive": True,
        },
        "incident_range": {
            "start": "2022-01-01T00:10:00+08:00",
            "end": "2022-01-01T00:20:00+08:00",
            "end_exclusive": True,
        },
        "source_revision": MARKET_SOURCE_REVISION,
        "public_component_vocabulary": ["pod-a", "pod-b", "checkout"],
        "public_fault_vocabulary": ["cpu", "latency"],
    }
    _write_json(path, spec)
    return path


def test_prepare_market_public_input_happy_path(tmp_path: Path) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    output_root = tmp_path / "published"

    public_input = prepare_market_public_input(source_day, case_spec, output_root)

    environment = MarketEnvironment(public_input)
    assert public_input == (
        output_root / "openrca-market-cloudbed-a-row-0007/public_input"
    )
    actual_files = sorted(
        path.relative_to(public_input).as_posix()
        for path in public_input.rglob("*")
        if path.is_file()
    )
    assert set(EXPECTED_FILES).issubset(
        {
            relative.removeprefix("telemetry/")
            for relative in actual_files
            if relative.startswith("telemetry/")
        }
    )

    task = cast(dict[str, Any], json.loads((public_input / "task.json").read_text()))
    assert task["environment"] == "openrca_market"
    assert task["dataset_manifest_ref"] == "manifest.json"
    assert task["allowed_scope"]["entities"] == [
        "market:cloudbed-a:cmdb:pod-a",
        "market:cloudbed-a:cmdb:pod-b",
        "market:cloudbed-a:relation:pod-a",
        "market:cloudbed-a:service:checkout",
    ]
    vocabulary = cast(
        dict[str, Any], json.loads((public_input / "vocabulary.json").read_text())
    )
    assert vocabulary["entities"] == ["pod-a", "pod-b", "checkout"]
    assert vocabulary["fault_types"] == ["cpu", "latency"]

    proxy_rows = list(
        csv.DictReader(
            (public_input / "telemetry/log/log_proxy.csv").open(encoding="utf-8")
        )
    )
    assert [row["log_id"] for row in proxy_rows] == ["proxy-ref", "proxy-inc"]
    assert proxy_rows[1]["_source_record"] == "2"
    assert proxy_rows[1]["_source_member"].endswith("log/log_proxy.csv")

    service_rows = list(
        csv.DictReader(
            (public_input / "telemetry/log/log_service.csv").open(encoding="utf-8")
        )
    )
    assert [row["log_id"] for row in service_rows] == ["service-ref", "service-inc"]

    topology = cast(
        dict[str, Any], json.loads((public_input / "topology.json").read_text())
    )
    assert topology["edges"][0]["source_id"] == "market:cloudbed-a:cmdb:pod-a"
    assert topology["edges"][0]["target_id"] == "market:cloudbed-a:cmdb:pod-b"
    assert topology["edges"][0]["samples"][0]["parent_source_record"] == 1
    assert topology["coverage"]["status"] == "partial"

    catalog = cast(
        dict[str, Any], json.loads((public_input / "metric_catalog.json").read_text())
    )
    assert catalog["metrics"] == [
        "container::cpu",
        "mesh::cpu",
        "node::cpu",
        "runtime::cpu",
        "service::count",
        "service::mrt",
        "service::rr",
        "service::sr",
    ]
    assert len(catalog["metric_bindings"]) == 8

    stats = [
        json.loads(line)
        for line in (public_input / "metric_window_stats.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(stats) == 8
    service_count = next(
        item for item in stats if item["metric_id"] == "service::count"
    )
    assert service_count["reference"]["median"] == 4.0
    assert service_count["incident"]["median"] == 14.0

    dataset_profile = cast(
        dict[str, Any], json.loads((public_input / "dataset_profile.json").read_text())
    )
    assert dataset_profile["prepared_trace_contract"] == {
        "cloudbed": "cloudbed-a",
        "join_keys": ["trace_id", "parent_span", "span_id"],
        "rows_filtered_to_allowed_window": True,
        "source_path": "trace/trace_span.csv",
        "timestamp_unit": "milliseconds",
    }
    assert (public_input / "source_schema.json").read_bytes() == (
        DEFAULT_SOURCE_SCHEMA_PATH.read_bytes()
    )

    logs = environment.query_log_samples(
        "market:cloudbed-a:cmdb:pod-a", period="incident", limit=5
    )
    assert [sample["log_id"] for sample in logs["samples"]] == [
        "service-inc",
        "proxy-inc",
    ]
    assert logs["samples"][0]["timestamp"] == pytest.approx(1640967000.5)
    assert logs["samples"][1]["timestamp"] == pytest.approx(1640967000.75)

    trace = environment.query_trace_neighbors("market:cloudbed-a:cmdb:pod-a")
    assert trace["edges"][0]["target_id"] == "market:cloudbed-a:cmdb:pod-b"

    metric = environment.query_metric(
        "market:cloudbed-a:service:checkout", "service::count", period="incident"
    )
    assert metric["points"][0]["value"] == 14.0


def test_market_defaults_use_the_pinned_public_candidate_schema(tmp_path: Path) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    raw = cast(dict[str, Any], json.loads(case_spec.read_text()))
    raw.pop("public_component_vocabulary")
    raw.pop("public_fault_vocabulary")
    _write_json(case_spec, raw)

    public_input = prepare_market_public_input(
        source_day, case_spec, tmp_path / "published"
    )
    vocabulary = cast(
        dict[str, Any], json.loads((public_input / "vocabulary.json").read_text())
    )

    assert vocabulary["entities"] == list(MARKET_COMPONENTS)
    assert vocabulary["fault_types"] == list(MARKET_FAULT_TYPES)
    assert vocabulary["source"] == prepare_market_suite_module.MARKET_CANDIDATE_SOURCE


def test_prepare_market_aggregates_pod_evidence_to_service(tmp_path: Path) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    for path in source_day.rglob("*.csv"):
        path.write_text(
            path.read_text()
            .replace("pod-a", "node-6.cartservice2-0")
            .replace("pod-b", "node-5.checkoutservice-0")
        )
    _write_csv(
        source_day / "metric/metric_container.csv",
        ["timestamp", "cmdb_id", "kpi_name", "value"],
        [
            [1640966400, "node-6.cartservice2-0", "cpu", 1],
            [1640967000, "node-6.cartservice2-0", "cpu", 11],
            [1640966400, "node-5.cartservice-0", "cpu", 5],
            [1640967000, "node-5.cartservice-0", "cpu", 5],
            [1640967600, "node-6.cartservice2-0", "cpu", 99],
        ],
    )
    case_spec = _case_spec(tmp_path / "case.json")
    payload = cast(dict[str, Any], json.loads(case_spec.read_text()))
    payload["public_component_vocabulary"] = [
        "cartservice2-0",
        "checkoutservice-0",
        "cartservice",
        "checkoutservice",
    ]
    payload["public_fault_vocabulary"] = ["container CPU load"]
    _write_json(case_spec, payload)

    public_input = prepare_market_public_input(
        source_day, case_spec, tmp_path / "published"
    )
    environment = MarketEnvironment(public_input)
    service_id = "market:cloudbed-a:service:cartservice"

    assert environment.list_metrics(service_id)["metric_ids"] == ["container::cpu"]
    metric = environment.query_metric(service_id, "container::cpu", period="incident")
    assert metric["points"][0]["original_entity_key"] == ("node-6.cartservice2-0")
    logs = environment.query_log_samples(service_id, period="incident")
    assert {sample["log_name"] for sample in logs["samples"]} == {
        "proxy",
        "service",
    }
    candidates = environment.summarize_anomalies()["fault_type_candidates"]
    assert any(candidate["entity_id"] == service_id for candidate in candidates)


def test_prepare_market_public_input_rejects_label_like_files(tmp_path: Path) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    (source_day / "selected_source.json").write_text("{}", encoding="utf-8")
    case_spec = _case_spec(tmp_path / "case.json")

    with pytest.raises(ValueError, match="label-like file"):
        prepare_market_public_input(source_day, case_spec, tmp_path / "published")


def test_prepare_market_public_input_rejects_label_like_case_spec_keys(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    payload = cast(dict[str, Any], json.loads(case_spec.read_text()))
    payload["La_Be-LS"] = "forbidden"
    _write_json(case_spec, payload)

    with pytest.raises(ValueError, match="label-like key"):
        prepare_market_public_input(source_day, case_spec, tmp_path / "published")


def test_prepare_market_public_input_rejects_case_spec_symlink(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    symlink_spec = tmp_path / "case-link.json"
    symlink_spec.symlink_to(case_spec)

    with pytest.raises(ValueError, match="case-spec must not be a symlink"):
        prepare_market_public_input(source_day, symlink_spec, tmp_path / "published")


def test_prepare_market_public_input_rejects_case_spec_resolved_forbidden_path(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    forbidden_dir = tmp_path / "OpenRCA-Market_Strict Unseen Audit"
    _case_spec(forbidden_dir / "case.json")
    safe_dir = tmp_path / "safe"
    safe_dir.symlink_to(forbidden_dir, target_is_directory=True)

    with pytest.raises(ValueError, match="case-spec points at a forbidden path"):
        prepare_market_public_input(
            source_day, safe_dir / "case.json", tmp_path / "published"
        )


def test_prepare_market_public_input_rejects_source_day_symlink(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    symlink_day = tmp_path / "linked-day"
    symlink_day.symlink_to(source_day, target_is_directory=True)

    with pytest.raises(ValueError, match="source-day must not be a symlink"):
        prepare_market_public_input(symlink_day, case_spec, tmp_path / "published")


def test_prepare_market_public_input_rejects_source_day_member_symlink(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    target = source_day / "metric/metric_container.csv"
    replacement = tmp_path / "metric_container.real.csv"
    target.rename(replacement)
    target.symlink_to(replacement)

    with pytest.raises(ValueError, match=r"source-day member must not be a symlink"):
        prepare_market_public_input(source_day, case_spec, tmp_path / "published")


def test_prepare_market_public_input_rejects_output_root_inside_source_day(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")

    with pytest.raises(ValueError, match="source-day and output-root must not overlap"):
        prepare_market_public_input(source_day, case_spec, source_day / "published")


def test_prepare_market_public_input_rejects_source_day_inside_output_root(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    source_day = _synthetic_source_day(source_root)
    case_spec = _case_spec(tmp_path / "case.json")

    with pytest.raises(ValueError, match="source-day and output-root must not overlap"):
        prepare_market_public_input(source_day, case_spec, source_root)


def test_prepare_market_public_input_is_idempotent_for_same_inputs(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    output_root = tmp_path / "published"
    source_schema = tmp_path / "market-schema.json"
    source_schema.write_bytes(DEFAULT_SOURCE_SCHEMA_PATH.read_bytes())

    first = prepare_market_public_input(
        source_day,
        case_spec,
        output_root,
        source_schema_path=source_schema,
    )
    second = prepare_market_public_input(
        source_day,
        case_spec,
        output_root,
        source_schema_path=source_schema,
    )

    assert second == first
    assert json.loads((first / "manifest.json").read_text()) == json.loads(
        (second / "manifest.json").read_text()
    )


def test_prepare_market_public_input_rejects_existing_case_with_different_contents(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    output_root = tmp_path / "published"
    public_input = prepare_market_public_input(source_day, case_spec, output_root)
    original_manifest = (public_input / "manifest.json").read_text()
    _write_csv(
        source_day / "metric/metric_service.csv",
        ["service", "timestamp", "rr", "sr", "mrt", "count"],
        [
            ["checkout", 1640966400, 1, 2, 3, 4],
            ["checkout", 1640967000, 11, 12, 13, 77],
            ["checkout", 1640967600, 99, 99, 99, 99],
        ],
    )

    with pytest.raises(FileExistsError, match="different contents"):
        prepare_market_public_input(source_day, case_spec, output_root)

    assert (public_input / "manifest.json").read_text() == original_manifest
    metric = MarketEnvironment(public_input).query_metric(
        "market:cloudbed-a:service:checkout", "service::count", period="incident"
    )
    assert metric["points"][0]["value"] == 14.0


def test_prepare_market_public_input_rejects_existing_case_dir_symlink_content(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    output_root = tmp_path / "published"
    public_input = prepare_market_public_input(source_day, case_spec, output_root)
    target = public_input / "entity_mapping.json"
    replacement = tmp_path / "entity_mapping.real.json"
    target.rename(replacement)
    target.symlink_to(replacement)

    with pytest.raises(ValueError, match="case-dir contains a symlink"):
        prepare_market_public_input(source_day, case_spec, output_root)


def test_prepare_market_public_input_cleans_stage_on_failure(
    tmp_path: Path,
) -> None:
    source_day = _synthetic_source_day(tmp_path / "source")
    case_spec = _case_spec(tmp_path / "case.json")
    output_root = tmp_path / "published"
    _write_csv(
        source_day / "metric/metric_container.csv",
        ["timestamp", "cmdb_id", "value"],
        [[1640966400, "pod-a", 1]],
    )

    with pytest.raises(ValueError, match="columns do not match"):
        prepare_market_public_input(source_day, case_spec, output_root)

    assert not (output_root / "openrca-market-cloudbed-a-row-0007").exists()
    assert sorted(path.name for path in output_root.glob(".tmp-*")) == []


def test_prepare_market_public_input_reads_from_validated_source_day_fd(
    tmp_path: Path, monkeypatch: Any
) -> None:
    source_root = tmp_path / "source"
    source_day = _synthetic_source_day(source_root)
    case_spec = _case_spec(tmp_path / "case.json")
    output_root = tmp_path / "published"
    original_validate = prepare_market_suite_module._validate_source_day
    swapped = False

    def wrapped_validate(path: Path, spec: CaseSpec) -> tuple[Path, int]:
        nonlocal swapped
        resolved, directory_fd = original_validate(path, spec)
        if not swapped:
            swapped = True
            moved = tmp_path / "pinned-original"
            resolved.rename(moved)
            replacement = _synthetic_source_day(source_root)
            _write_csv(
                replacement / "metric/metric_service.csv",
                ["service", "timestamp", "rr", "sr", "mrt", "count"],
                [
                    ["checkout", 1640966400, 1, 2, 3, 4],
                    ["checkout", 1640967000, 11, 12, 13, 777],
                    ["checkout", 1640967600, 99, 99, 99, 99],
                ],
            )
        return resolved, directory_fd

    monkeypatch.setattr(
        prepare_market_suite_module,
        "_validate_source_day",
        wrapped_validate,
    )

    public_input = prepare_market_public_input(source_day, case_spec, output_root)
    metric = MarketEnvironment(public_input).query_metric(
        "market:cloudbed-a:service:checkout", "service::count", period="incident"
    )
    assert metric["points"][0]["value"] == 14.0
