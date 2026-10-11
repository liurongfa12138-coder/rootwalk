from datetime import datetime, timedelta, timezone

from rootwalk.anomalies import material_anomalies, material_anomalies_from_evidence
from rootwalk.contracts import Evidence, Extractor, TimeRange


def typed_profile_evidence(
    *,
    fault_type_ready: bool = True,
    qualifying_metric_count: int = 1,
    material_event_ready: bool = False,
) -> Evidence:
    start = datetime(2020, 5, 27, 4, 30, tzinfo=timezone.utc)
    observed = TimeRange(start=start, end=start + timedelta(hours=1))
    return Evidence(
        evidence_id="evidence-cpu-profile",
        observation_refs=["observation-cpu-profile"],
        subject_ids=["telecom:pod:docker_001"],
        predicate="query_metric_profile.result",
        value={
            "finding_gate_summary": {
                "requested_fault_type": "CPU fault",
                "fault_type_ready": fault_type_ready,
                "material_event_ready": material_event_ready,
                "material_event_basis": (
                    "peer_isolated_trace_error_event" if material_event_ready else None
                ),
                "material_event_metric_ids": (
                    ["container_cpu_used"] if material_event_ready else []
                ),
                "fault_type_anchor_metric_ids": ["container_cpu_used"],
                "qualifying_metric_ids": ["container_cpu_used"],
                "qualifying_metric_count": qualifying_metric_count,
                "root_cause_score": 0.92,
            }
        },
        observed_interval=observed,
        onset_interval=observed,
        time_uncertainty="fixture",
        artifact_refs=[],
        extractor=Extractor(
            kind="deterministic",
            name="metric-profile",
            version="1",
            parameters_hash="fixture",
        ),
        source_fingerprint="fixture-source",
        dataset_snapshot_id="fixture-dataset",
        evidence_kind="metric_profile",
        mechanism_dimension="resource_profile",
        independence_group="fixture-source",
        semantic_quality="typed",
    )


def test_material_anomalies_group_fault_aliases_by_source_event() -> None:
    observations = [
        {
            "observation_id": "overview-1",
            "tool": "summarize_anomalies",
            "summary": """
            {
              "fault_type_candidates": [
                {
                  "candidate_id": "delay-proposal",
                  "entity_id": "node-1",
                  "resource_family": "network",
                  "fault_type": "network delay",
                  "metric_id": "receive_errors",
                  "source_metric_id": "receive_queue",
                  "metric_group": ["receive_queue", "receive_errors"],
                  "materiality": "material",
                  "materiality_basis": "adapter_registered_fault_candidate",
                  "overview_score": 1.0,
                  "peer_outlier": true
                },
                {
                  "candidate_id": "loss-proposal",
                  "entity_id": "node-1",
                  "resource_family": "network",
                  "fault_type": "network loss",
                  "metric_id": "receive_errors",
                  "source_metric_id": "receive_queue",
                  "metric_group": ["receive_queue", "receive_errors"],
                  "materiality": "material",
                  "materiality_basis": "adapter_registered_fault_candidate",
                  "overview_score": 1.0,
                  "peer_outlier": true
                }
              ]
            }
            """,
        }
    ]

    anomalies = material_anomalies(observations)

    assert len(anomalies) == 1
    assert anomalies[0].subject_id == "node-1"
    assert anomalies[0].signal_ids == ["receive_errors", "receive_queue"]
    assert anomalies[0].mechanism_candidates == ["network delay", "network loss"]
    assert anomalies[0].proposal_ids == ["delay-proposal", "loss-proposal"]


def test_material_anomalies_ignore_navigation_only_proposals() -> None:
    anomalies = material_anomalies(
        [
            {
                "observation_id": "overview-1",
                "tool": "summarize_anomalies",
                "summary": """
                {"fault_type_candidates":[{
                  "candidate_id":"candidate-1",
                  "entity_id":"node-1",
                  "resource_family":"cpu",
                  "fault_type":"cpu pressure",
                  "metric_id":"cpu",
                  "materiality":"navigation_only"
                }]}
                """,
            }
        ]
    )

    assert anomalies == []


def test_fault_type_ready_typed_evidence_registers_material_anomaly() -> None:
    evidence = typed_profile_evidence()

    anomalies = material_anomalies_from_evidence(evidence)

    assert len(anomalies) == 1
    assert anomalies[0].subject_id == "telecom:pod:docker_001"
    assert anomalies[0].signal_family == "resource_profile"
    assert anomalies[0].signal_ids == ["container_cpu_used"]
    assert anomalies[0].mechanism_candidates == ["CPU fault"]
    assert anomalies[0].source_observation_ids == ["observation-cpu-profile"]
    assert anomalies[0].proposal_ids == ["evidence-cpu-profile"]
    assert anomalies[0].materiality_basis == "typed_fault_ready_profile"
    assert anomalies[0].priority_facts == {
        "fault_type_ready": True,
        "material_event_ready": False,
        "qualifying_metric_count": 1,
        "root_cause_score": 0.92,
    }


def test_non_ready_material_typed_event_registers_without_proving_mechanism() -> None:
    anomalies = material_anomalies_from_evidence(
        typed_profile_evidence(
            fault_type_ready=False,
            qualifying_metric_count=2,
            material_event_ready=True,
        )
    )

    assert len(anomalies) == 1
    assert anomalies[0].mechanism_candidates == ["CPU fault"]
    assert anomalies[0].materiality_basis == "peer_isolated_trace_error_event"
    assert anomalies[0].priority_facts["fault_type_ready"] is False
    assert anomalies[0].priority_facts["material_event_ready"] is True


def test_non_ready_typed_evidence_does_not_register_material_anomaly() -> None:
    assert (
        material_anomalies_from_evidence(
            typed_profile_evidence(fault_type_ready=False, qualifying_metric_count=0)
        )
        == []
    )
