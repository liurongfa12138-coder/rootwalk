"""Route unit tests to synthetic Bank data and isolate official-data checks."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n"


def _build_synthetic_public_input(root: Path) -> Path:
    root.mkdir()
    window = {
        "start": "2021-03-04T17:30:00+08:00",
        "end": "2021-03-04T18:30:00+08:00",
        "end_exclusive": True,
    }
    incident = {
        "start": "2021-03-04T18:00:00+08:00",
        "end": "2021-03-04T18:30:00+08:00",
        "end_exclusive": True,
    }
    reference = {
        "start": "2021-03-04T17:30:00+08:00",
        "end": "2021-03-04T18:00:00+08:00",
        "end_exclusive": True,
        "health_status": "unverified",
    }
    nodes = [
        {
            "canonical_id": "bank:entity:Tomcat01",
            "original_id": "Tomcat01",
            "entity_type": "pod",
            "sources": [
                "metric/metric_container.csv",
                "log/log_service.csv",
                "trace/trace_span.csv",
            ],
            "mapping_status": "synthetic_exact",
        },
        {
            "canonical_id": "bank:entity:Tomcat02",
            "original_id": "Tomcat02",
            "entity_type": "pod",
            "sources": ["metric/metric_container.csv", "trace/trace_span.csv"],
            "mapping_status": "synthetic_exact",
        },
        {
            "canonical_id": "bank:entity:MG02",
            "original_id": "MG02",
            "entity_type": "pod",
            "sources": ["metric/metric_container.csv", "trace/trace_span.csv"],
            "mapping_status": "synthetic_exact",
        },
        {
            "canonical_id": "bank:test:ServiceTest1",
            "original_id": "ServiceTest1",
            "entity_type": "business_test",
            "sources": ["metric/metric_app.csv"],
            "mapping_status": "synthetic_exact",
        },
    ]
    task = {
        "task_id": "synthetic-bank-task",
        "environment": "synthetic_bank",
        "question": "Identify the supported synthetic root cause.",
        "incident_range": incident,
        "timezone": "Asia/Shanghai",
        "allowed_query_range": window,
        "reference_range": reference,
        "entry_entities": [],
        "allowed_scope": {
            "entities": [node["canonical_id"] for node in nodes],
            "tools": [
                "list_entities",
                "list_metrics",
                "query_metric",
                "query_log_samples",
                "query_trace_neighbors",
                "read_artifact_slice",
                "summarize_anomalies",
            ],
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
    }
    topology = {
        "snapshot_id": "synthetic-bank-topology-v1",
        "source": "synthetic fixture",
        "valid_range": window,
        "nodes": nodes,
        "edges": [
            {
                "edge_id": "synthetic-edge-1",
                "source_id": "bank:entity:Tomcat01",
                "target_id": "bank:entity:Tomcat02",
                "relation_type": "calls",
                "fact_kind": "observed",
                "span_pairs": 1,
                "samples": [
                    {
                        "trace_id": "synthetic-trace",
                        "parent_span_id": "synthetic-parent",
                        "child_span_id": "synthetic-child",
                        "parent_source_record": 1,
                        "child_source_record": 2,
                    }
                ],
            }
        ],
        "coverage": {
            "requested_range": window,
            "observed_range": window,
            "completeness": "partial",
            "sampling": "synthetic",
            "basis": "generated unit-test fixture",
        },
    }
    stats: list[dict[str, object]] = [
        {
            "entity_id": "Tomcat01",
            "canonical_id": "bank:entity:Tomcat01",
            "metric_id": "OSLinux-CPU_CPU_CPUCpuUtil",
            "value_unit": "source-native; synthetic",
            "reference": {
                "count": 2,
                "min": 10.0,
                "max": 11.0,
                "mean": 10.5,
                "median": 10.5,
                "example_source_records": [1, 2],
            },
            "incident": {
                "count": 2,
                "min": 80.0,
                "max": 90.0,
                "mean": 85.0,
                "median": 85.0,
                "example_source_records": [3, 4],
            },
        },
        {
            "entity_id": "ServiceTest1",
            "canonical_id": "bank:test:ServiceTest1",
            "metric_id": "cnt",
            "value_unit": "source-native; synthetic",
            "reference": {
                "count": 1,
                "min": 10.0,
                "max": 10.0,
                "mean": 10.0,
                "median": 10.0,
                "example_source_records": [1],
            },
            "incident": {
                "count": 1,
                "min": 20.0,
                "max": 20.0,
                "mean": 20.0,
                "median": 20.0,
                "example_source_records": [2],
            },
        },
    ]
    family_metrics = [
        "synthetic_cpu_peak",
        "synthetic_memory_signal",
        "synthetic_network_signal",
        "synthetic_storage_signal",
        "synthetic_latency_signal",
        "synthetic_misc_signal",
    ]
    zero_baseline_metrics = [f"synthetic_tiny_signal_{index}" for index in range(6)]
    for index, metric_id in enumerate(family_metrics, 1):
        short_peak = metric_id == "synthetic_cpu_peak"
        stats.append(
            {
                "entity_id": nodes[2]["original_id"],
                "canonical_id": nodes[2]["canonical_id"],
                "metric_id": metric_id,
                "value_unit": "source-native; synthetic",
                "reference": {
                    "count": 3,
                    "min": 10.0,
                    "max": 10.0,
                    "mean": 10.0,
                    "median": 10.0,
                    "example_source_records": [index],
                },
                "incident": {
                    "count": 3,
                    "min": 10.0 if short_peak else 20.0,
                    "max": 100.0 if short_peak else 20.0,
                    "mean": 40.0 if short_peak else 20.0,
                    "median": 10.0 if short_peak else 20.0,
                    "example_source_records": [index + 10],
                },
            }
        )
    for index, metric_id in enumerate(zero_baseline_metrics, 20):
        stats.append(
            {
                "entity_id": nodes[2]["original_id"],
                "canonical_id": nodes[2]["canonical_id"],
                "metric_id": metric_id,
                "value_unit": "source-native; synthetic",
                "reference": {
                    "count": 3,
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "median": 0.0,
                    "example_source_records": [index],
                },
                "incident": {
                    "count": 3,
                    "min": 0.001,
                    "max": 0.001,
                    "mean": 0.001,
                    "median": 0.001,
                    "example_source_records": [index + 10],
                },
            }
        )
    files = {
        "dataset_profile.json": _json(
            {
                "dataset_id": "synthetic-bank-v1",
                "description": "generated unit-test fixture",
            }
        ),
        "task.json": _json(task),
        "topology.json": _json(topology),
        "entity_mapping.json": _json(nodes),
        "metric_catalog.json": _json(
            {
                "metrics": [
                    "OSLinux-CPU_CPU_CPUCpuUtil",
                    "cnt",
                    *family_metrics,
                    *zero_baseline_metrics,
                ],
                "log_names": ["localhost_access_log"],
                "duration_unit": "synthetic milliseconds",
            }
        ),
        "vocabulary.json": _json(
            {
                "entities": [node["original_id"] for node in nodes],
                "fault_types": ["synthetic public fault type"],
            }
        ),
        "metric_window_stats.jsonl": "".join(_json(row) for row in stats),
        "telemetry/metric/metric_app.csv": (
            "timestamp,rr,sr,cnt,mrt,tc,_source_record\n"
            "1614850200,100,100,10,1,ServiceTest1,1\n"
            "1614852000,100,100,20,1,ServiceTest1,2\n"
        ),
        "telemetry/metric/metric_container.csv": (
            "timestamp,cmdb_id,kpi_name,value,_source_record\n"
            "1614850200,Tomcat01,OSLinux-CPU_CPU_CPUCpuUtil,10,1\n"
            "1614850260,Tomcat01,OSLinux-CPU_CPU_CPUCpuUtil,11,2\n"
            "1614852000,Tomcat01,OSLinux-CPU_CPU_CPUCpuUtil,80,3\n"
            "1614852060,Tomcat01,OSLinux-CPU_CPU_CPUCpuUtil,90,4\n"
        ),
        "telemetry/log/log_service.csv": (
            "log_id,timestamp,cmdb_id,log_name,value,_source_record\n"
            "synthetic-log,1614852001,Tomcat01,localhost_access_log,"
            "synthetic failure,1\n"
        ),
        "telemetry/trace/trace_span.csv": (
            "trace_id,span_id,parent_id,cmdb_id,timestamp,_source_record\n"
            "synthetic-trace,synthetic-parent,root,Tomcat01,1614852000,1\n"
            "synthetic-trace,synthetic-child,synthetic-parent,Tomcat02,1614852001,2\n"
        ),
    }
    artifacts: list[dict[str, object]] = []
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = content.encode()
        path.write_bytes(encoded)
        artifacts.append(
            {
                "path": relative,
                "bytes": len(encoded),
                "sha256": hashlib.sha256(encoded).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        _json(
            {
                "schema_version": "rootwalk-dataset-slice-v1",
                "dataset_id": "synthetic-bank-v1",
                "artifacts": artifacts,
            }
        )
    )
    return root


@pytest.fixture(scope="session")
def synthetic_public_input(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _build_synthetic_public_input(
        tmp_path_factory.mktemp("rootwalk-synthetic-bank") / "public_input"
    )


@pytest.fixture(scope="session")
def official_public_input() -> Path:
    configured = os.environ.get("ROOTWALK_OPENRCA_BANK_ROOT")
    if not configured:
        pytest.fail(
            "openrca_bank tests require ROOTWALK_OPENRCA_BANK_ROOT; "
            "default CI must deselect them with -m 'not openrca_bank'"
        )
    root = Path(configured).expanduser().resolve()
    if not root.is_dir():
        pytest.fail(f"ROOTWALK_OPENRCA_BANK_ROOT is not a directory: {root}")
    return root


@pytest.fixture(autouse=True)
def _select_public_input(request: pytest.FixtureRequest) -> Iterator[None]:
    node = cast(pytest.Item, getattr(request, "node"))
    module = cast(ModuleType | None, getattr(node, "module", None))
    is_official = node.get_closest_marker("openrca_bank") is not None
    if module is None or not hasattr(module, "PUBLIC_INPUT"):
        if is_official:
            request.getfixturevalue("official_public_input")
        yield
        return
    original = getattr(module, "PUBLIC_INPUT")
    fixture = "official_public_input" if is_official else "synthetic_public_input"
    setattr(module, "PUBLIC_INPUT", cast(Path, request.getfixturevalue(fixture)))
    try:
        yield
    finally:
        setattr(module, "PUBLIC_INPUT", original)
