# pyright: reportPrivateUsage=false

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pytest

from rootwalk.openrca import (
    MarketEnvironment,
    dominant_market_fault_type,
    load_openrca_environment,
    market_output_component,
    market_service_component,
    peer_isolated_trace_member,
)
from rootwalk.query_worker import QueryWorker

SOURCE_SCHEMA: dict[str, object] = {
    "schema_version": 1,
    "dataset": "OpenRCA Market",
    "source": {
        "repository": "cdreetz/OpenRCA",
        "revision": "7a29a7c3600cc07b7760a6c5097d36ac8573f039",
    },
    "scope_rules": {
        "cloudbeds_are_independent": True,
        "partitioning": "cloudbed/date",
        "files_per_daily_partition": 8,
        "timezone": "Asia/Shanghai",
        "do_not_merge_cloudbeds": True,
    },
    "files": {
        "log/log_proxy.csv": {
            "columns": ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
            "timestamp_unit": "seconds",
            "entity_key": "cmdb_id",
            "value_kind": "free-form log text",
        },
        "log/log_service.csv": {
            "columns": ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
            "timestamp_unit": "seconds",
            "entity_key": "cmdb_id",
            "value_kind": "free-form log text",
            "nullable_columns": ["value"],
        },
        **{
            f"metric/metric_{name}.csv": {
                "columns": ["timestamp", "cmdb_id", "kpi_name", "value"],
                "timestamp_unit": "seconds",
                "entity_key": "cmdb_id",
                "metric_key": "kpi_name",
                "value_kind": "numeric",
            }
            for name in ("container", "mesh", "node", "runtime")
        },
        "metric/metric_service.csv": {
            "columns": ["service", "timestamp", "rr", "sr", "mrt", "count"],
            "timestamp_unit": "seconds",
            "entity_key": "service",
            "metric_layout": "wide",
            "numeric_metric_columns": ["rr", "sr", "mrt", "count"],
        },
        "trace/trace_span.csv": {
            "columns": [
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
            "timestamp_unit": "milliseconds",
            "entity_key": "cmdb_id",
            "trace_key": "trace_id",
            "span_key": "span_id",
            "parent_key": "parent_span",
            "nullable_columns": ["type", "parent_span"],
        },
    },
    "adapter_requirements": [
        "convert all timestamps to timezone-aware instants before window filtering",
        "treat metric_service as a wide metric table rather than a "
        "kpi_name/value table",
        "keep source filename and original entity key on every normalized record",
        "treat empty log value, trace type, and parent_span fields as missing data "
        "rather than parse failures",
        "build trace parent-child links within trace_id using parent_span and span_id",
        "keep cloudbed identity in every canonical entity and case identifier",
        "validate all rows against the allowed half-open query window before exposing "
        "the public slice",
    ],
    "holdout_content_included": False,
    "labels_included": False,
}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("node-5.cartservice-2", "cartservice-2"),
        ("cartservice-2.source.cartservice.redis-cart", "cartservice-2"),
        ("cartservice.ts:8088", "cartservice"),
        ("cartservice-grpc", "cartservice"),
        ("node-5", "node-5"),
    ],
)
def test_market_output_component_uses_public_candidate_names(
    raw: str, expected: str
) -> None:
    assert market_output_component(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("node-5.cartservice-0", "cartservice"),
        ("node-6.cartservice2-0", "cartservice"),
        ("cartservice-0.destination.frontend.cartservice", "cartservice"),
        ("cartservice-grpc", "cartservice"),
        ("node-5", "node-5"),
    ],
)
def test_market_service_component_collapses_deployment_variants(
    raw: str, expected: str
) -> None:
    assert market_service_component(raw) == expected


def test_market_disk_mechanism_dominates_same_node_iowait_cpu_symptom() -> None:
    assert (
        dominant_market_fault_type(
            [
                {
                    "requested_fault_type": "node CPU load",
                    "fault_type_ready": True,
                    "root_cause_score": 0.71,
                    "qualifying_metric_ids": [
                        "node::system.cpu.iowait",
                        "node::system.cpu.pct_usage",
                    ],
                },
                {
                    "requested_fault_type": "node disk read I/O consumption",
                    "fault_type_ready": True,
                    "root_cause_score": 0.48,
                    "qualifying_metric_ids": [
                        "node::system.io.r_await",
                        "node::system.io.util",
                    ],
                },
            ]
        )
        == "node disk read I/O consumption"
    )


def test_peer_isolated_trace_member_requires_clear_replica_separation() -> None:
    assert (
        peer_isolated_trace_member(
            {
                "adservice-0": 14.0,
                "adservice-1": 17.4,
                "adservice-2": 19.0,
                "adservice2-0": 68.0,
            },
            latency=True,
        )
        == "adservice2-0"
    )
    assert (
        peer_isolated_trace_member(
            {
                "adservice-0": 14.0,
                "adservice-1": 17.4,
                "adservice-2": 19.0,
                "adservice2-0": 22.0,
            },
            latency=True,
        )
        is None
    )


def test_market_fault_candidates_prefer_log_backed_compatible_entity() -> None:
    def signal(entity_id: str, incident_max: float) -> dict[str, Any]:
        return {
            "entity_id": entity_id,
            "metric_id": "container::network_errors",
            "resource_family": "network",
            "overview_score": incident_max,
            "reference": {"min": 0.0, "median": 0.0, "max": 0.0},
            "incident": {"min": 0.0, "median": 0.0, "max": incident_max},
        }

    service = "market:cloudbed-a:service:cartservice"
    node = "market:cloudbed-a:node:node-5"
    candidates = MarketEnvironment._select_fault_type_candidates(
        ["container network packet corruption"],
        [signal(node, 10.0), signal(service, 1.0)],
        {
            node: ["container::network_errors"],
            service: ["container::network_errors"],
        },
        {
            node: ["metric/metric_node.csv"],
            service: ["log/log_service.csv", "metric/metric_container.csv"],
        },
        ["service-log"],
    )

    assert [candidate["entity_id"] for candidate in candidates] == [service]
    assert candidates[0]["has_log_source"] is True


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _refresh_manifest_artifact(root: Path, relative: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    path = root / relative
    artifact = next(item for item in manifest["artifacts"] if item["path"] == relative)
    artifact["bytes"] = path.stat().st_size
    artifact["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)


def _write_csv(
    path: Path, fields: list[str], rows: list[list[object]], cloudbed: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*fields, "_source_record", "_source_member"])
        for index, row in enumerate(rows, 1):
            writer.writerow(
                [
                    *row,
                    index,
                    f"Market/{cloudbed}/telemetry/2022_01_01/"
                    f"{path.parts[-2]}/{path.name}",
                ]
            )


def _window_stats(reference: float, incident: float) -> dict[str, object]:
    return {
        "reference": {
            "count": 1,
            "min": reference,
            "max": reference,
            "mean": reference,
            "median": reference,
            "example_source_records": [1],
        },
        "incident": {
            "count": 1,
            "min": incident,
            "max": incident,
            "mean": incident,
            "median": incident,
            "example_source_records": [2],
        },
        "max_observed_gap_seconds": 600.0,
        "median_observed_interval_seconds": 600.0,
        "value_unit": "source-native; not inferred",
    }


def _market_public_input(root: Path, cloudbed: str = "cloudbed-a") -> Path:
    root.mkdir(parents=True)
    reference_start = int(
        datetime.fromisoformat("2022-01-01T00:00:00+08:00").timestamp()
    )
    incident_start = reference_start + 600
    allowed_end = incident_start + 600
    window = {
        "start": "2022-01-01T00:00:00+08:00",
        "end": "2022-01-01T00:20:00+08:00",
        "end_exclusive": True,
    }
    reference = {
        "start": window["start"],
        "end": "2022-01-01T00:10:00+08:00",
        "end_exclusive": True,
        "health_status": "unverified",
    }
    incident = {
        "start": reference["end"],
        "end": window["end"],
        "end_exclusive": True,
    }
    cmdb_id = f"market:{cloudbed}:cmdb:pod-a"
    child_id = f"market:{cloudbed}:cmdb:pod-b"
    service_id = f"market:{cloudbed}:service:checkout"
    nodes = [
        {
            "canonical_id": cmdb_id,
            "original_id": "pod-a",
            "entity_type": "component",
            "sources": [
                "log/log_proxy.csv",
                "log/log_service.csv",
                "metric/metric_container.csv",
                "metric/metric_mesh.csv",
                "metric/metric_node.csv",
                "metric/metric_runtime.csv",
                "trace/trace_span.csv",
            ],
            "normalization": "exact raw identity",
            "mapping_status": "synthetic_exact",
        },
        {
            "canonical_id": child_id,
            "original_id": "pod-b",
            "entity_type": "component",
            "sources": ["trace/trace_span.csv"],
            "normalization": "exact raw identity",
            "mapping_status": "synthetic_exact",
        },
        {
            "canonical_id": service_id,
            "original_id": "checkout",
            "entity_type": "service",
            "sources": ["metric/metric_service.csv"],
            "normalization": "exact raw identity",
            "mapping_status": "synthetic_exact",
        },
    ]
    task = {
        "task_id": f"openrca-market-{cloudbed}-row-0001",
        "environment": "openrca_market",
        "question": "Identify the synthetic root cause.",
        "incident_range": incident,
        "timezone": "Asia/Shanghai",
        "allowed_query_range": window,
        "reference_range": reference,
        "entry_entities": [],
        "allowed_scope": {
            "entities": [node["canonical_id"] for node in nodes],
            "tools": [],
            "telemetry_root": "telemetry",
            "entity_inventory_ref": "entity_mapping.json",
        },
        "requested_outputs": ["root cause component", "root cause reason"],
        "dataset_manifest_ref": "manifest.json",
        "source_row_id": 1,
        "task_index": "synthetic",
        "expected_fault_count": 1,
        "fault_count_source": "synthetic fixture",
        "data_mode": "synthetic_fixture",
        "source_commit": "7a29a7c3600cc07b7760a6c5097d36ac8573f039",
    }
    topology: dict[str, object] = {
        "snapshot_id": f"openrca-market-{cloudbed}-topology-v1",
        "source": "synthetic trace parent_span/span_id join",
        "valid_range": window,
        "nodes": nodes,
        "edges": [
            {
                "edge_id": "edge-000",
                "source_id": cmdb_id,
                "target_id": child_id,
                "relation_type": "calls",
                "fact_kind": "observed",
                "span_pairs": 1,
                "samples": [
                    {
                        "trace_id": "trace-1",
                        "parent_span_id": "span-parent",
                        "child_span_id": "span-child",
                        "parent_source_record": 1,
                        "child_source_record": 2,
                    }
                ],
            }
        ],
        "coverage": {
            "status": "partial",
            "unique_spans": 2,
            "matched_parent_spans": 1,
            "missing_parent_spans": 0,
            "root_like_sentinel_spans": 1,
            "root_marker_values": {"": 1},
            "intra_entity_pairs": 0,
            "duplicate_span_keys": 0,
            "conflicting_span_keys": 0,
            "ambiguous_links_skipped": 0,
            "missing_parent_samples": [],
            "note": "synthetic partial topology",
        },
    }
    _write_json(root / "task.json", task)
    _write_json(root / "topology.json", topology)
    _write_json(root / "entity_mapping.json", nodes)
    _write_json(root / "source_schema.json", SOURCE_SCHEMA)
    _write_json(
        root / "vocabulary.json",
        {"entities": ["pod-a", "pod-b", "checkout"], "fault_types": ["cpu"]},
    )

    metric_bindings: list[dict[str, str]] = []
    metric_stats: list[dict[str, object]] = []
    metrics: list[str] = []
    for index, name in enumerate(("container", "mesh", "node", "runtime"), 1):
        source_path = f"metric/metric_{name}.csv"
        metric_id = f"{name}::cpu"
        metrics.append(metric_id)
        metric_bindings.append(
            {
                "canonical_id": cmdb_id,
                "metric_id": metric_id,
                "source_path": source_path,
                "layout": "long",
                "entity_field": "cmdb_id",
                "original_entity_key": "pod-a",
                "metric_field": "kpi_name",
                "source_metric": "cpu",
                "value_field": "value",
                "timestamp_unit": "seconds",
            }
        )
        metric_stats.append(
            {
                "canonical_id": cmdb_id,
                "entity_id": "pod-a",
                "metric_id": metric_id,
                **_window_stats(float(index), float(index + 10)),
            }
        )
        _write_csv(
            root / "telemetry" / source_path,
            ["timestamp", "cmdb_id", "kpi_name", "value"],
            [
                [reference_start, "pod-a", "cpu", index],
                [incident_start, "pod-a", "cpu", index + 10],
                [allowed_end, "pod-a", "cpu", 99],
            ],
            cloudbed,
        )

    service_metrics = {"rr": 11, "sr": 12, "mrt": 13, "count": 14}
    for metric, incident_value in service_metrics.items():
        metric_id = f"service::{metric}"
        metrics.append(metric_id)
        metric_bindings.append(
            {
                "canonical_id": service_id,
                "metric_id": metric_id,
                "source_path": "metric/metric_service.csv",
                "layout": "wide",
                "entity_field": "service",
                "original_entity_key": "checkout",
                "value_field": metric,
                "timestamp_unit": "seconds",
            }
        )
        metric_stats.append(
            {
                "canonical_id": service_id,
                "entity_id": "checkout",
                "metric_id": metric_id,
                **_window_stats(float(incident_value - 10), float(incident_value)),
            }
        )
    _write_csv(
        root / "telemetry/metric/metric_service.csv",
        ["service", "timestamp", "rr", "sr", "mrt", "count"],
        [
            ["checkout", reference_start, 1, 2, 3, 4],
            ["checkout", incident_start, 11, 12, 13, 14],
            ["checkout", allowed_end, 99, 99, 99, 99],
        ],
        cloudbed,
    )
    _write_json(
        root / "metric_catalog.json",
        {
            "duration_unit": "unknown",
            "log_names": {"proxy": 2, "service": 2},
            "metrics": sorted(metrics),
            "metric_bindings": metric_bindings,
            "schemas": {
                "long": "cmdb_id, kpi_name, value",
                "service": "wide rr, sr, mrt, count",
            },
        },
    )
    (root / "metric_window_stats.jsonl").write_text(
        "".join(
            json.dumps(item, separators=(",", ":")) + "\n" for item in metric_stats
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "dataset_profile.json",
        {
            "cloudbed": cloudbed,
            "files": 8,
            "prepared_trace_contract": {
                "cloudbed": cloudbed,
                "join_keys": ["trace_id", "parent_span", "span_id"],
                "rows_filtered_to_allowed_window": True,
                "source_path": "trace/trace_span.csv",
                "timestamp_unit": "milliseconds",
            },
        },
    )

    _write_csv(
        root / "telemetry/log/log_proxy.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [
            ["proxy-ref", reference_start, "pod-a", "proxy", "reference proxy"],
            ["proxy-inc", incident_start, "pod-a", "proxy", "incident proxy"],
            ["proxy-end", allowed_end, "pod-a", "proxy", "excluded"],
        ],
        cloudbed,
    )
    _write_csv(
        root / "telemetry/log/log_service.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [
            ["service-ref", reference_start, "pod-a", "service", "reference"],
            ["service-inc", incident_start, "pod-a", "service", ""],
            ["service-end", allowed_end, "pod-a", "service", "excluded"],
        ],
        cloudbed,
    )
    _write_csv(
        root / "telemetry/trace/trace_span.csv",
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
            [
                incident_start * 1000,
                "pod-a",
                "span-parent",
                "trace-1",
                8,
                "",
                0,
                "root",
                "",
            ],
            [
                incident_start * 1000 + 1,
                "pod-b",
                "span-child",
                "trace-1",
                6,
                "SERVER",
                0,
                "child",
                "span-parent",
            ],
        ],
        cloudbed,
    )

    _write_json(
        root / "manifest.json",
        {
            "schema_version": "rootwalk-dataset-slice-v1",
            "dataset_id": f"openrca-market-{cloudbed}-row-0001",
            "cloudbed": cloudbed,
            "source_commit": "7a29a7c3600cc07b7760a6c5097d36ac8573f039",
            "label_policy": "synthetic public fixture; evaluation labels excluded",
            "artifacts": [
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(root.rglob("*"))
                if path.is_file()
            ],
        },
    )
    return root


def test_market_negative_label_policy_is_accepted(tmp_path: Path) -> None:
    root = _market_public_input(tmp_path / "public_input")

    environment = MarketEnvironment(root)

    assert environment.dataset_id == "openrca-market-cloudbed-a-row-0001"


def test_market_factory_queries_long_and_wide_metrics(tmp_path: Path) -> None:
    root = _market_public_input(tmp_path / "public_input")
    environment = load_openrca_environment(root)

    assert isinstance(environment, MarketEnvironment)
    cmdb_metrics = environment.list_metrics("market:cloudbed-a:cmdb:pod-a")
    assert cmdb_metrics["metric_ids"] == [
        "container::cpu",
        "mesh::cpu",
        "node::cpu",
        "runtime::cpu",
    ]
    for metric_id in cmdb_metrics["metric_ids"]:
        result = environment.query_metric(
            "market:cloudbed-a:cmdb:pod-a", metric_id, period="incident"
        )
        assert result["status"] == "success"
        assert result["sample_count"] == 1
        assert result["points"][0]["source_path"].startswith("metric/")
        assert result["points"][0]["original_entity_key"] == "pod-a"
        assert result["paired_metric_summary"] == {
            "counts_complete": True,
            "scale_hint": "unknown",
            "pressure_direction": "unknown",
            "hint_basis": "heuristic from metric name",
            "reference": {
                "sample_count": 1,
                "min": 1.0 + cmdb_metrics["metric_ids"].index(metric_id),
                "median": 1.0 + cmdb_metrics["metric_ids"].index(metric_id),
                "max": 1.0 + cmdb_metrics["metric_ids"].index(metric_id),
            },
            "incident": {
                "sample_count": 1,
                "min": 11.0 + cmdb_metrics["metric_ids"].index(metric_id),
                "median": 11.0 + cmdb_metrics["metric_ids"].index(metric_id),
                "max": 11.0 + cmdb_metrics["metric_ids"].index(metric_id),
                "above_reference_max_count": 1,
                "below_reference_min_count": 0,
            },
        }

    service_metrics = environment.list_metrics("market:cloudbed-a:service:checkout")[
        "metric_ids"
    ]
    assert service_metrics == [
        "service::count",
        "service::mrt",
        "service::rr",
        "service::sr",
    ]
    count = environment.query_metric(
        "market:cloudbed-a:service:checkout", "service::count", period="incident"
    )
    assert count["sample_count"] == 1
    assert count["points"][0]["value"] == 14.0

    empty_requested_window = environment.query_metric(
        "market:cloudbed-a:service:checkout",
        "service::count",
        start="2022-01-01T00:05:00+08:00",
        end="2022-01-01T00:06:00+08:00",
    )
    assert empty_requested_window["status"] == "no_data"
    assert (
        empty_requested_window["paired_metric_summary"]
        == count["paired_metric_summary"]
    )


def test_market_logs_trace_and_cloudbed_namespace(tmp_path: Path) -> None:
    first = MarketEnvironment(_market_public_input(tmp_path / "a/public_input"))
    second = MarketEnvironment(
        _market_public_input(tmp_path / "b/public_input", "cloudbed-b")
    )

    logs = first.query_log_samples(
        "market:cloudbed-a:cmdb:pod-a", period="incident", limit=10
    )
    assert logs["status"] == "success"
    assert {sample["source_path"] for sample in logs["samples"]} == {
        "log/log_proxy.csv",
        "log/log_service.csv",
    }
    assert {sample["text"] for sample in logs["samples"]} == {
        "incident proxy",
        None,
    }
    assert logs["paired_log_summary"] == {
        "counts_complete": True,
        "samples_truncated": False,
        "reference": {
            "matched_count": 2,
            "log_name_counts": {"proxy": 1, "service": 1},
        },
        "incident": {
            "matched_count": 2,
            "log_name_counts": {"proxy": 1, "service": 1},
        },
    }
    filtered = first.query_log_samples(
        "market:cloudbed-a:cmdb:pod-a",
        period="incident",
        contains="proxy",
    )
    assert [sample["log_id"] for sample in filtered["samples"]] == ["proxy-inc"]
    assert filtered["paired_log_summary"]["reference"] == {
        "matched_count": 1,
        "log_name_counts": {"proxy": 1},
    }
    no_incident_matches = first.query_log_samples(
        "market:cloudbed-a:cmdb:pod-a",
        period="incident",
        contains="reference",
    )
    assert no_incident_matches["status"] == "no_data"
    assert no_incident_matches["paired_log_summary"] == {
        "counts_complete": True,
        "samples_truncated": False,
        "reference": {
            "matched_count": 2,
            "log_name_counts": {"proxy": 1, "service": 1},
        },
        "incident": {"matched_count": 0, "log_name_counts": {}},
    }
    trace = first.query_trace_neighbors("market:cloudbed-a:cmdb:pod-a")
    assert trace["edges"][0]["target_id"] == "market:cloudbed-a:cmdb:pod-b"
    assert trace["trace_timestamp_unit"] == "milliseconds"
    assert trace["trace_duration_unit"] == "source-native; unknown"
    assert second.list_entities()["entities"][0]["canonical_id"].startswith(
        "market:cloudbed-b:"
    )
    assert not {
        item["canonical_id"] for item in first.list_entities()["entities"]
    }.intersection(item["canonical_id"] for item in second.list_entities()["entities"])


def test_market_analysis_tools_preserve_shape_patterns_and_dynamic_edges(
    tmp_path: Path,
) -> None:
    market = MarketEnvironment(_market_public_input(tmp_path / "public_input"))
    entity_id = "market:cloudbed-a:cmdb:pod-a"

    metric = market.query_metric_profile(
        entity_id,
        ["container::cpu", "mesh::cpu"],
        derived_ratios=[
            {
                "name": "container_to_mesh_cpu",
                "numerator": "container::cpu",
                "denominator": "mesh::cpu",
            }
        ],
    )
    assert metric["status"] == "success"
    assert metric["metric_profiles"][0]["incident"]["p95"] is not None
    assert metric["metric_profiles"][0]["incident_burst"]["onset_timestamp"]
    assert metric["derived_metric_profiles"][0]["formula"] == "ratio"
    assert "single_component_ready" in metric["finding_gate_summary"]

    logs = market.summarize_log_patterns(entity_id, limit=10)
    assert logs["status"] == "success"
    assert logs["template_version"] == "rootwalk-log-template-v1"
    assert any(item["new_in_incident"] for item in logs["patterns"])
    assert all(len(item["template_id"]) == 64 for item in logs["patterns"])

    trace = market.query_trace_edges(entity_id, direction="outgoing")
    assert trace["status"] == "success"
    assert trace["evidence_level"] == "incident_correlation"
    assert trace["edge_profiles"][0]["source_id"] == entity_id
    assert trace["edge_profiles"][0]["incident"]["call_count"] == 1
    assert trace["edge_profiles"][0]["reference"]["call_count"] == 0


def test_market_log_sample_keeps_matched_source_alias(tmp_path: Path) -> None:
    root = _market_public_input(tmp_path / "public_input")
    topology_path = root / "topology.json"
    topology = cast(dict[str, Any], json.loads(topology_path.read_text()))
    node = next(
        item
        for item in topology["nodes"]
        if item["canonical_id"] == "market:cloudbed-a:cmdb:pod-a"
    )
    node["source_aliases"] = {"log/log_proxy.csv": ["proxy-pod-a"]}
    _write_json(topology_path, topology)
    incident_start = datetime.fromisoformat("2022-01-01T00:10:00+08:00").timestamp()
    _write_csv(
        root / "telemetry/log/log_proxy.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [["proxy-inc", incident_start, "proxy-pod-a", "proxy", "incident proxy"]],
        "cloudbed-a",
    )
    _refresh_manifest_artifact(root, "topology.json")
    _refresh_manifest_artifact(root, "telemetry/log/log_proxy.csv")

    result = MarketEnvironment(root).query_log_samples(
        "market:cloudbed-a:cmdb:pod-a", period="incident", limit=10
    )

    assert result["entity_id"] == "market:cloudbed-a:cmdb:pod-a"
    samples_by_source = {sample["source_path"]: sample for sample in result["samples"]}
    assert samples_by_source["log/log_proxy.csv"]["original_entity_key"] == (
        "proxy-pod-a"
    )
    assert samples_by_source["log/log_service.csv"]["original_entity_key"] == "pod-a"


def test_market_source_schema_and_manifest_remain_fail_closed(tmp_path: Path) -> None:
    root = _market_public_input(tmp_path / "public_input")
    schema_path = root / "source_schema.json"
    schema = cast(dict[str, Any], json.loads(schema_path.read_text()))
    schema["files"]["trace/trace_span.csv"]["parent_key"] = "parent_id"
    _write_json(schema_path, schema)
    manifest_path = root / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    artifact = next(
        item for item in manifest["artifacts"] if item["path"] == "source_schema.json"
    )
    artifact["bytes"] = schema_path.stat().st_size
    artifact["sha256"] = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="parent_key"):
        MarketEnvironment(root)


def test_market_rejects_cross_file_scope_and_extra_telemetry(tmp_path: Path) -> None:
    root = _market_public_input(tmp_path / "scope/public_input")
    topology_path = root / "topology.json"
    topology = cast(dict[str, Any], json.loads(topology_path.read_text()))
    topology["nodes"][0]["sources"].remove("metric/metric_container.csv")
    _write_json(topology_path, topology)
    _refresh_manifest_artifact(root, "topology.json")
    with pytest.raises(ValueError, match="metric binding does not match schema"):
        MarketEnvironment(root)

    root = _market_public_input(tmp_path / "extra/public_input")
    extra = root / "telemetry/metric/extra.csv"
    _write_csv(extra, ["timestamp"], [[1]], "cloudbed-a")
    manifest_path = root / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    manifest["artifacts"].append(
        {
            "path": extra.relative_to(root).as_posix(),
            "bytes": extra.stat().st_size,
            "sha256": hashlib.sha256(extra.read_bytes()).hexdigest(),
        }
    )
    _write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="artifact path is not authorized"):
        MarketEnvironment(root)


def test_market_rejects_source_commit_mismatch(tmp_path: Path) -> None:
    root = _market_public_input(tmp_path / "public_input")
    manifest_path = root / "manifest.json"
    manifest = cast(dict[str, Any], json.loads(manifest_path.read_text()))
    manifest["source_commit"] = "different"
    _write_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="source commit mismatch"):
        MarketEnvironment(root)


def test_market_logs_keep_fractional_time_and_global_order(tmp_path: Path) -> None:
    root = _market_public_input(tmp_path / "public_input")
    incident_start = datetime.fromisoformat("2022-01-01T00:10:00+08:00").timestamp()
    _write_csv(
        root / "telemetry/log/log_proxy.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [
            ["late-9", incident_start + 9, "pod-a", "proxy", "late"],
            ["late-8", incident_start + 8, "pod-a", "proxy", "late"],
            ["late-7", incident_start + 7, "pod-a", "proxy", "late"],
            ["early-proxy", incident_start + 0.75, "pod-a", "proxy", "early"],
        ],
        "cloudbed-a",
    )
    _write_csv(
        root / "telemetry/log/log_service.csv",
        ["log_id", "timestamp", "cmdb_id", "log_name", "value"],
        [["early-service", incident_start + 0.5, "pod-a", "service", "early"]],
        "cloudbed-a",
    )
    _refresh_manifest_artifact(root, "telemetry/log/log_proxy.csv")
    _refresh_manifest_artifact(root, "telemetry/log/log_service.csv")

    result = MarketEnvironment(root).query_log_samples(
        "market:cloudbed-a:cmdb:pod-a", period="incident", limit=2
    )
    assert result["status"] == "partial"
    assert [sample["log_id"] for sample in result["samples"]] == [
        "early-service",
        "early-proxy",
    ]
    assert result["samples"][0]["timestamp"] == incident_start + 0.5
    assert result["paired_log_summary"]["reference"]["matched_count"] == 0
    assert result["paired_log_summary"]["incident"] == {
        "matched_count": 5,
        "log_name_counts": {"proxy": 4, "service": 1},
    }


def test_market_query_worker_uses_environment_factory(tmp_path: Path) -> None:
    root = _market_public_input(tmp_path / "public_input")
    worker = QueryWorker(root, sandbox="off")
    try:
        result = worker.execute(
            "query_metric",
            {
                "entity_id": "market:cloudbed-a:service:checkout",
                "metric_id": "service::count",
                "period": "incident",
            },
            10.0,
        )
    finally:
        worker.close()

    assert result["status"] == "success"
    assert result["points"][0]["value"] == 14.0
