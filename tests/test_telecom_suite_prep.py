# pyright: reportPrivateUsage=false

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from rootwalk.openrca import (
    TelecomEnvironment,
    load_openrca_environment,
    mechanism_signature_rank,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_telecom_suite import (
    IMPACT_ONLY_ROLE,
    SOURCE_REVISION,
    TELECOM_SLICE_SCHEMA_VERSION,
    prepare_telecom_suite,
)


def _write_csv(
    path: Path, fieldnames: list[str], rows: list[dict[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _source_day(root: Path) -> Path:
    day = root / "2020_05_27"
    common_fields = ["itemid", "name", "bomc_id", "timestamp", "value", "cmdb_id"]
    _write_csv(
        day / "metric/metric_app.csv",
        ["serviceName", "startTime", "avg_time", "num", "succee_num", "succee_rate"],
        [
            {
                "serviceName": "osb_001",
                "startTime": 1590512400000,
                "avg_time": 1,
                "num": 10,
                "succee_num": 10,
                "succee_rate": 1,
            },
            {
                "serviceName": "osb_001",
                "startTime": 1590514800000,
                "avg_time": 20,
                "num": 10,
                "succee_num": 5,
                "succee_rate": 0.5,
            },
        ],
    )
    _write_csv(
        day / "metric/metric_container.csv",
        common_fields,
        [
            {
                "itemid": 1,
                "name": "container_cpu_used",
                "bomc_id": "container-cpu",
                "timestamp": 1590513000000,
                "value": 2,
                "cmdb_id": "docker_001",
            },
            {
                "itemid": 1,
                "name": "container_cpu_used",
                "bomc_id": "container-cpu",
                "timestamp": 1590514800000,
                "value": 90,
                "cmdb_id": "docker_001",
            },
        ],
    )
    _write_csv(
        day / "metric/metric_middleware.csv",
        common_fields,
        [
            {
                "itemid": 2,
                "name": "connected_clients",
                "bomc_id": "redis-clients",
                "timestamp": 1590513000000,
                "value": 4,
                "cmdb_id": "redis_001",
            }
        ],
    )
    _write_csv(
        day / "metric/metric_node.csv",
        common_fields,
        [
            {
                "itemid": 3,
                "name": "Received_queue",
                "bomc_id": "node-queue",
                "timestamp": 1590513000000,
                "value": 0,
                "cmdb_id": "os_001",
            },
            {
                "itemid": 4,
                "name": "Received_queue",
                "bomc_id": "node-queue-low",
                "timestamp": 1590513000000,
                "value": 0,
                "cmdb_id": "os_002",
            },
            {
                "itemid": 4,
                "name": "Received_queue",
                "bomc_id": "node-queue-low",
                "timestamp": 1590514800000,
                "value": 1,
                "cmdb_id": "os_002",
            },
            {
                "itemid": 3,
                "name": "Received_queue",
                "bomc_id": "node-queue",
                "timestamp": 1590514800000,
                "value": 5,
                "cmdb_id": "os_001",
            },
        ],
    )
    _write_csv(
        day / "metric/metric_service.csv",
        common_fields,
        [
            {
                "itemid": 4,
                "name": "Sess_Connect",
                "bomc_id": "db-session",
                "timestamp": 1590513000000,
                "value": 10,
                "cmdb_id": "db_003",
            },
            {
                "itemid": 4,
                "name": "Sess_Connect",
                "bomc_id": "db-session",
                "timestamp": 1590514800000,
                "value": 40,
                "cmdb_id": "db_003",
            },
        ],
    )
    trace_fields = [
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
    ]
    _write_csv(
        day / "trace/trace_span.csv",
        trace_fields,
        [
            {
                "callType": "RemoteProcess",
                "startTime": 1590513000000,
                "elapsedTime": 5,
                "success": "True",
                "traceId": "trace-0",
                "id": "parent-0",
                "pid": "None",
                "cmdb_id": "docker_001",
                "dsName": "",
                "serviceName": "service-call",
            },
            {
                "callType": "OSB",
                "startTime": 1590514800000,
                "elapsedTime": 3,
                "success": "True",
                "traceId": "trace-node-1",
                "id": "node-1",
                "pid": "None",
                "cmdb_id": "os_001",
                "dsName": "",
                "serviceName": "osb_001",
            },
            {
                "callType": "OSB",
                "startTime": 1590514800000,
                "elapsedTime": 3,
                "success": "True",
                "traceId": "trace-node-2",
                "id": "node-2",
                "pid": "None",
                "cmdb_id": "os_002",
                "dsName": "",
                "serviceName": "osb_001",
            },
            {
                "callType": "JDBC",
                "startTime": 1590513001000,
                "elapsedTime": 2,
                "success": "True",
                "traceId": "trace-0",
                "id": "child-0",
                "pid": "parent-0",
                "cmdb_id": "docker_001",
                "dsName": "db_003",
                "serviceName": "",
            },
            {
                "callType": "RemoteProcess",
                "startTime": 1590514800000,
                "elapsedTime": 8,
                "success": "True",
                "traceId": "trace-1",
                "id": "parent-1",
                "pid": "None",
                "cmdb_id": "docker_001",
                "dsName": "",
                "serviceName": "service-call",
            },
            {
                "callType": "JDBC",
                "startTime": 1590514801000,
                "elapsedTime": 40,
                "success": "False",
                "traceId": "trace-1",
                "id": "child-1",
                "pid": "parent-1",
                "cmdb_id": "docker_001",
                "dsName": "db_003",
                "serviceName": "",
            },
        ],
    )
    return day


def _case_spec(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "source_revision": SOURCE_REVISION,
                "cases": [
                    {
                        "row_id": 32,
                        "task_index": "task_6",
                        "question": "Find the component and reason.",
                        "reference_start": "2020-05-27T01:00:00+08:00",
                        "reference_end": "2020-05-27T01:30:00+08:00",
                        "incident_start": "2020-05-27T01:30:00+08:00",
                        "incident_end": "2020-05-27T02:00:00+08:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prepare_telecom_suite_normalizes_metrics_and_trace(tmp_path: Path) -> None:
    public_input = prepare_telecom_suite(
        _source_day(tmp_path / "source"),
        _case_spec(tmp_path / "case-spec.json"),
        tmp_path / "prepared",
    )[0]
    environment = TelecomEnvironment(public_input)
    assert isinstance(load_openrca_environment(public_input), TelecomEnvironment)
    assert environment.task.environment == "openrca_telecom"
    assert environment.capabilities.logs is False
    manifest = json.loads((public_input / "manifest.json").read_text())
    assert manifest["schema_version"] == TELECOM_SLICE_SCHEMA_VERSION

    impact_id = "telecom:business_entry:osb_001"
    impact_profile = environment.query_metric_profile(impact_id, ["avg_time"])
    assert impact_profile["status"] == "success"
    impact_metric = impact_profile["metric_profiles"][0]
    assert impact_metric["reference"]["median"] == 1
    assert impact_metric["incident"]["median"] == 20
    impact_ref = impact_profile["artifact_refs"][0]
    assert impact_ref["selector"]["path"] == ("telemetry/metric/metric_container.csv")
    assert impact_ref["selector"]["source_records"] == [2, 3]

    with (public_input / "telemetry/metric/metric_container.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        normalized_rows = list(csv.DictReader(stream))
    impact_rows = [
        row
        for row in normalized_rows
        if row["_source_member"] == "metric/metric_app.csv"
    ]
    assert len(impact_rows) == 8
    assert {row["cmdb_id"] for row in impact_rows} == {"osb_001"}
    assert {int(row["_source_record"]) for row in impact_rows} == {2, 3}

    nodes = json.loads((public_input / "entity_mapping.json").read_text())
    impact_node = next(node for node in nodes if node["canonical_id"] == impact_id)
    assert impact_node["evidence_role"] == IMPACT_ONLY_ROLE
    assert impact_node["public_candidate_match"] is None
    assert impact_node["sources"] == ["metric/metric_container.csv"]

    stats = [
        json.loads(line)
        for line in (public_input / "metric_window_stats.jsonl")
        .read_text()
        .splitlines()
    ]
    impact_stat = next(
        item
        for item in stats
        if item["canonical_id"] == impact_id and item["metric_id"] == "avg_time"
    )
    assert impact_stat["evidence_role"] == IMPACT_ONLY_ROLE
    assert impact_stat["source_members"] == ["metric/metric_app.csv"]
    assert impact_stat["reference"]["example_source_records"] == [2]
    assert impact_stat["incident"]["example_source_records"] == [3]

    catalog = json.loads((public_input / "metric_catalog.json").read_text())
    impact_binding = next(
        item
        for item in catalog["metric_bindings"]
        if item["canonical_id"] == impact_id and item["metric_id"] == "avg_time"
    )
    assert impact_binding["source_path"] == "metric/metric_container.csv"
    assert impact_binding["source_member_filter"] == ["metric/metric_app.csv"]
    assert impact_binding["evidence_role"] == IMPACT_ONLY_ROLE

    vocabulary = json.loads((public_input / "vocabulary.json").read_text())
    assert "osb_001" not in vocabulary["entities"]
    profile = json.loads((public_input / "dataset_profile.json").read_text())
    assert profile["candidate_component_count"] == 43
    assert profile["impact_only_entity_count"] == 1
    assert profile["impact_only_metric_series_count"] == 4
    overview = environment.summarize_anomalies()
    impact_signals = [
        signal for signal in overview["signals"] if signal["entity_id"] == impact_id
    ]
    assert impact_signals
    assert all(signal["evidence_role"] == IMPACT_ONLY_ROLE for signal in impact_signals)
    assert all(not signal["direct_metric_candidates"] for signal in impact_signals)
    candidate_signals = environment._root_cause_candidate_signals(overview["signals"])
    assert all(signal["entity_id"] != impact_id for signal in candidate_signals)
    candidates = overview["fault_type_candidates"]
    assert all(candidate["entity_id"] != impact_id for candidate in candidates)

    cpu = environment.query_metric("telecom:pod:docker_001", "container_cpu_used")
    assert cpu["status"] == "success"
    assert cpu["summary"]["max"] == 90
    cpu_profile = environment.query_metric_profile(
        "telecom:pod:docker_001", ["container_cpu_used"], fault_type="CPU fault"
    )
    cpu_metric = cpu_profile["metric_profiles"][0]
    assert cpu_metric["strong_incident_change"] is True
    assert cpu_metric["finding_sample_ready"] is False
    assert cpu_metric["missing_evidence_reasons"] == [
        "reference has fewer than 3 samples",
        "incident has fewer than 3 samples",
    ]
    cpu_gate = cpu_profile["finding_gate_summary"]
    assert cpu_gate["strong_metric_ids"] == ["container_cpu_used"]
    assert cpu_gate["earliest_strong_onset_timestamp"] == 1590514800
    assert cpu_gate["insufficient_sample_metric_ids"] == ["container_cpu_used"]
    assert cpu_gate["sampling_coverage_ready"] is False
    assert cpu_gate["fault_type_ready"] is False
    assert cpu_gate["material_event_ready"] is False
    assert cpu_gate["mechanism_signature"] is None
    db = environment.query_metric("telecom:service:db_003", "Sess_Connect")
    assert db["paired_metric_summary"]["reference"]["median"] == 10
    assert db["paired_metric_summary"]["incident"]["median"] == 40
    trace_errors = environment.query_metric(
        "telecom:service:db_003", "trace_incoming_error_count"
    )
    assert trace_errors["paired_metric_summary"]["reference"]["max"] == 0
    assert trace_errors["paired_metric_summary"]["incident"]["max"] == 1

    edges = environment.query_trace_edges("telecom:service:db_003")
    assert edges["status"] in {"success", "partial"}
    assert edges["trace_profile_source"] == "precomputed_source_backed_edge_profiles"
    assert {reference["selector"]["path"] for reference in edges["artifact_refs"]} == {
        "trace_edge_window_stats.jsonl",
        "telemetry/trace/trace_span.csv",
    }
    assert edges["edge_profiles"][0]["source_id"] == "telecom:pod:docker_001"
    assert edges["edge_profiles"][0]["target_id"] == "telecom:service:db_003"
    assert edges["edge_profiles"][0]["incident"]["error_rate"] == 1.0
    service = environment.query_service_profile(
        "telecom:service:db_003", "db connection limit"
    )
    service_gate = service["finding_gate_summary"]
    assert service_gate["atomic_service_profile"] is True
    assert service_gate["strong_metric_ids"] == [
        "Sess_Connect",
        "trace_incoming_error_count",
        "trace_incoming_error_rate",
    ]
    assert service_gate["sampling_coverage_ready"] is False
    assert service_gate["fault_type_ready"] is False
    assert service_gate["material_event_ready"] is False
    assert service_gate["mechanism_signature"] is None
    network = environment.query_metric_profile(
        "telecom:node:os_001", ["Received_queue"], fault_type="network loss"
    )
    assert network["finding_gate_summary"]["fault_type_ready"] is False
    assert network["finding_gate_summary"]["mechanism_signature"] is None
    assert network["finding_gate_summary"]["anomaly_signature"] in {
        None,
        "single_metric_peer_isolated",
    }
    low_queue = environment.query_metric_profile(
        "telecom:node:os_002", ["Received_queue"], fault_type="network loss"
    )
    assert low_queue["finding_gate_summary"]["fault_type_ready"] is False
    assert low_queue["finding_gate_summary"]["material_event_ready"] is False


def test_telecom_reader_remains_compatible_with_v1_manifest(tmp_path: Path) -> None:
    public_input = prepare_telecom_suite(
        _source_day(tmp_path / "source"),
        _case_spec(tmp_path / "case-spec.json"),
        tmp_path / "prepared",
    )[0]
    manifest_path = public_input / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = "rootwalk-dataset-slice-v1"
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )

    environment = TelecomEnvironment(public_input)

    assert environment.dataset_id == "openrca-telecom-row-0032"
    assert environment.task.environment == "openrca_telecom"


def test_telecom_profile_maps_public_fault_semantics() -> None:
    concepts = TelecomEnvironment._semantic_concepts("db connection limit")
    assert {"database", "connection"} <= concepts
    assert TelecomEnvironment._fault_metric_compatible(
        "db connection limit", "Sess_Connect"
    )
    assert TelecomEnvironment._fault_metric_compatible("db close", "On_Off_State")
    assert not TelecomEnvironment._fault_metric_compatible(
        "network loss", "Received_queue"
    )
    assert mechanism_signature_rank(
        {"mechanism_signature": "availability_state_transition"}
    ) > mechanism_signature_rank({"mechanism_signature": "connection_state_transition"})
    assert TelecomEnvironment._metric_name_hints("Sess_Active")[1] == (
        "higher_is_more_pressure"
    )
    assert not TelecomEnvironment._fault_metric_compatible(
        "db close", "container_cpu_used"
    )


@pytest.mark.parametrize("label_name", ["query.csv", "record.csv"])
def test_prepare_telecom_suite_rejects_label_files(
    tmp_path: Path, label_name: str
) -> None:
    source_day = _source_day(tmp_path / "source")
    (source_day / label_name).write_text("label\nforbidden\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source-day file set mismatch"):
        prepare_telecom_suite(
            source_day,
            _case_spec(tmp_path / "case-spec.json"),
            tmp_path / "prepared",
        )
