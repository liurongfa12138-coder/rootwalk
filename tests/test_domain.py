from dataclasses import FrozenInstanceError
from datetime import UTC, datetime, timedelta

import pytest

from rootwalk.contracts import (
    ArtifactRef,
    CausalStep,
    Coverage,
    Evidence,
    Extractor,
    Observation,
    Path,
    TimeRange,
    TopologyEdge,
    TopologySnapshot,
)
from rootwalk.domain import (
    EvidenceLedger,
    PathGrounding,
    check_temporal_consistency,
    classify_observation,
    ground_path,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW = TimeRange(start=START, end=START + timedelta(minutes=10))
ARTIFACT = ArtifactRef(content_hash="a" * 64, selector="row:1")


def coverage(
    completeness: str = "complete", collection_health: str = "healthy"
) -> Coverage:
    return Coverage.model_validate(
        {
            "requested_range": WINDOW,
            "observed_range": WINDOW,
            "completeness": completeness,
            "collection_health": collection_health,
            "basis": "frozen test fixture",
        }
    )


def observation(
    status: str, *, completeness: str = "complete", health: str = "healthy"
) -> Observation:
    return Observation.model_validate(
        {
            "observation_id": f"observation-{status}",
            "tool": "query_metric",
            "arguments": {},
            "status": status,
            "coverage": coverage(completeness, health),
            "artifact_refs": [],
            "attempt_id": "attempt-1",
        }
    )


def evidence(
    evidence_id: str,
    fingerprint: str,
    *,
    subject: str = "A",
    onset: TimeRange | None = WINDOW,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        observation_refs=["observation-success"],
        subject_ids=[subject],
        predicate="latency_increased",
        value=True,
        observed_interval=WINDOW,
        onset_interval=onset,
        time_uncertainty="bounded by sample interval",
        artifact_refs=[ARTIFACT],
        extractor=Extractor(
            kind="deterministic",
            name="test",
            version="1",
            parameters_hash="params",
        ),
        source_fingerprint=fingerprint,
        dataset_snapshot_id="snapshot-1",
    )


def topology(
    *edges: TopologyEdge,
    coverage_status: str = "complete",
    extra_entities: tuple[str, ...] = (),
) -> TopologySnapshot:
    entities = set(extra_entities) | {
        entity for edge in edges for entity in (edge.source_id, edge.target_id)
    }
    node_types = {"pod": "pod", "node": "node"}
    return TopologySnapshot.model_validate(
        {
            "snapshot_id": "topology-1",
            "source": "fixture",
            "valid_range": WINDOW,
            "nodes": [
                {
                    "canonical_id": entity,
                    "original_id": entity,
                    "entity_type": node_types.get(entity, "service"),
                    "sources": ["fixture"],
                }
                for entity in sorted(entities)
            ],
            "edges": list(edges),
            "coverage": coverage(coverage_status),
        }
    )


def edge(
    edge_id: str,
    source: str,
    target: str,
    relation: str = "calls",
) -> TopologyEdge:
    return TopologyEdge(
        edge_id=edge_id,
        source_id=source,
        target_id=target,
        relation_type=relation,
        fact_kind="observed",
    )


def path(
    source: str,
    target: str,
    mechanism: str,
    basis: list[str],
    *,
    evidence_refs: list[str] | None = None,
) -> Path:
    return Path(
        hypothesis_id="hypothesis-1",
        root_cause_entity_id=source,
        symptom_entity_ids=[target],
        steps=[
            CausalStep(
                step_id="step-1",
                from_entity=source,
                to_entity=target,
                mechanism=mechanism,
                basis_topology_edge_ids=basis,
                supporting_evidence_refs=evidence_refs or [],
                status="proposed",
            )
        ],
    )


def test_evidence_ledger_deduplicates_source_fingerprint() -> None:
    ledger = EvidenceLedger()

    assert ledger.add(evidence("evidence-1", "same-source"))
    assert not ledger.add(evidence("evidence-2", "same-source"))
    assert len(ledger) == 2
    assert ledger.independent_count == 1
    assert ledger.canonical_evidence_id("evidence-2") == "evidence-1"


@pytest.mark.parametrize("status", ["unsupported", "error", "cancelled"])
def test_failed_observations_are_gaps(status: str) -> None:
    assert classify_observation(observation(status)) == "gap"


def test_no_data_needs_known_healthy_complete_coverage_for_negative_fact() -> None:
    assert classify_observation(observation("no_data", health="unknown")) == "gap"
    assert classify_observation(observation("no_data", completeness="unknown")) == "gap"
    assert classify_observation(observation("no_data")) == "negative_fact"
    assert classify_observation(observation("success")) == "observation_only"


@pytest.mark.parametrize(
    ("source", "target", "mechanism"),
    [
        ("B", "A", "dependency_failure"),
        ("A", "B", "request_load"),
    ],
)
def test_calls_edge_supports_both_declared_influence_directions(
    source: str, target: str, mechanism: str
) -> None:
    call = edge("call", "A", "B")

    result = ground_path(path(source, target, mechanism, ["call"]), topology(call))

    assert result.structural_grounding == "pass"


def test_runs_on_only_supports_node_to_pod_resource_impact() -> None:
    placement = edge("placement", "pod", "node", "runs_on")
    snapshot = topology(placement)

    allowed = ground_path(
        path("node", "pod", "resource_failure", ["placement"]), snapshot
    )
    reversed_path = ground_path(
        path("pod", "node", "resource_failure", ["placement"]), snapshot
    )

    assert allowed.structural_grounding == "pass"
    assert reversed_path.structural_grounding == "fail"


def test_same_entity_zero_step_needs_local_evidence() -> None:
    local_path = Path(
        hypothesis_id="hypothesis-1",
        root_cause_entity_id="A",
        symptom_entity_ids=["A"],
        steps=[],
    )

    item = evidence("local-evidence", "local-source", subject="A")
    result = ground_path(
        local_path,
        topology(extra_entities=("A",)),
        [item],
        claim_evidence_refs=[item.evidence_id],
    )

    assert result == PathGrounding("pass", "pass", "consistent", "complete")
    with pytest.raises(FrozenInstanceError):
        result.structural_grounding = "fail"  # type: ignore[misc]


def test_same_entity_zero_step_without_evidence_is_partial() -> None:
    local_path = Path(
        hypothesis_id="hypothesis-1",
        root_cause_entity_id="A",
        symptom_entity_ids=["A"],
        steps=[],
    )

    result = ground_path(local_path, topology(extra_entities=("A",)))

    assert result.structural_grounding == "pass"
    assert result.reference_grounding == "fail"
    assert result.coverage == "partial"


def test_unknown_path_entity_fails_structural_grounding() -> None:
    call = edge("call", "A", "B")
    unknown_root = ground_path(
        path("missing", "B", "request_load", ["call"]), topology(call)
    )
    unknown_symptom = ground_path(
        path("A", "missing", "request_load", ["call"]), topology(call)
    )
    unknown_step_endpoint = Path(
        hypothesis_id="hypothesis-1",
        root_cause_entity_id="A",
        symptom_entity_ids=["B"],
        steps=[
            CausalStep(
                step_id="step-1",
                from_entity="A",
                to_entity="missing",
                mechanism="request_load",
                basis_topology_edge_ids=[],
                status="proposed",
            ),
            CausalStep(
                step_id="step-2",
                from_entity="missing",
                to_entity="B",
                mechanism="request_load",
                basis_topology_edge_ids=[],
                status="proposed",
            ),
        ],
    )

    assert unknown_root.structural_grounding == "fail"
    assert unknown_symptom.structural_grounding == "fail"
    assert (
        ground_path(unknown_step_endpoint, topology(call)).structural_grounding
        == "fail"
    )


def test_missing_basis_is_partial_but_known_endpoint_conflict_fails() -> None:
    call = edge("call", "A", "B")
    snapshot = topology(call)

    missing = ground_path(path("A", "B", "request_load", []), snapshot)
    conflict = ground_path(path("A", "C", "request_load", ["call"]), snapshot)

    assert missing.structural_grounding == "partial"
    assert missing.coverage == "partial"
    assert conflict.structural_grounding == "fail"


def test_unknown_topology_coverage_stays_unknown() -> None:
    call = edge("call", "A", "B")

    result = ground_path(
        path("A", "B", "request_load", ["call"]),
        topology(call, coverage_status="unknown"),
    )

    assert result.coverage == "unknown"


def test_legal_but_unrelated_evidence_does_not_claim_semantic_correctness() -> None:
    item = evidence("evidence-1", "source-1", subject="unrelated")
    call = edge("call", "A", "B")
    candidate = path(
        "A", "B", "request_load", ["call"], evidence_refs=[item.evidence_id]
    )

    result = ground_path(candidate, topology(call), [item])

    assert result.reference_grounding == "pass"
    assert result.semantic_correctness == "not_evaluated"


def test_missing_or_invisible_evidence_reference_fails_grounding() -> None:
    item = evidence("evidence-1", "source-1")
    call = edge("call", "A", "B")
    candidate = path(
        "A", "B", "request_load", ["call"], evidence_refs=[item.evidence_id]
    )

    missing = ground_path(candidate, topology(call))
    invisible = ground_path(
        candidate, topology(call), [item], visible_evidence_ids=set()
    )

    assert missing.reference_grounding == "fail"
    assert invisible.reference_grounding == "fail"


def test_temporal_consistency_uses_intervals_and_tolerance() -> None:
    cause = TimeRange(start=START, end=START + timedelta(minutes=2))
    later_symptom = TimeRange(
        start=START + timedelta(minutes=4), end=START + timedelta(minutes=5)
    )
    overlap = TimeRange(
        start=START + timedelta(minutes=1), end=START + timedelta(minutes=3)
    )
    earlier_symptom = TimeRange(
        start=START - timedelta(minutes=3), end=START - timedelta(minutes=2)
    )

    assert check_temporal_consistency(cause, later_symptom) == "consistent"
    assert check_temporal_consistency(cause, overlap) == "unknown"
    assert check_temporal_consistency(None, later_symptom) == "unknown"
    assert check_temporal_consistency(cause, earlier_symptom) == "contradicted"
    assert (
        check_temporal_consistency(
            cause, later_symptom, clock_tolerance=timedelta(minutes=3)
        )
        == "unknown"
    )
