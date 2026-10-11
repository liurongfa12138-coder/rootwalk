from pathlib import Path

import pytest
from pydantic import ValidationError

from rootwalk.contracts import (
    AnomalyDispositionProposal,
    ArtifactRef,
    Decision,
    MaterialAnomaly,
    Task,
    TopologyNode,
    TopologySnapshot,
)

PUBLIC_INPUT = (
    Path(__file__).resolve().parents[2]
    / "rootwalk-handoff"
    / "rootwalk-development-handoff"
    / "rootwalk-data-prep"
    / "slices"
    / "bank-row-0001"
    / "public_input"
)


@pytest.mark.openrca_bank
def test_real_task_and_topology_parse() -> None:
    task = Task.model_validate_json((PUBLIC_INPUT / "task.json").read_text())
    topology = TopologySnapshot.model_validate_json(
        (PUBLIC_INPUT / "topology.json").read_text()
    )

    assert task.task_id == "openrca-bank-row-0001"
    assert topology.snapshot_id == "bank-row-0001-topology-v1"
    assert len(topology.edges) == 32


def test_artifact_ref_accepts_lowercase_sha256() -> None:
    artifact = ArtifactRef(content_hash="a" * 64, selector="row:1")

    assert artifact.content_hash == "a" * 64


def test_topology_node_defaults_to_root_cause_candidate_and_serializes_role() -> None:
    node = TopologyNode.model_validate(
        {
            "canonical_id": "service-a",
            "original_id": "service-a",
            "entity_type": "service",
            "sources": ["fixture"],
        }
    )

    assert node.evidence_role == "root_cause_candidate"
    assert node.model_dump(mode="json")["evidence_role"] == "root_cause_candidate"


def test_topology_node_accepts_impact_only_and_rejects_unknown_role() -> None:
    payload = {
        "canonical_id": "service-a::impact-only",
        "original_id": "service-a",
        "entity_type": "service",
        "sources": ["fixture"],
        "evidence_role": "impact_only",
    }

    node = TopologyNode.model_validate(payload)
    assert node.model_dump(mode="json")["evidence_role"] == "impact_only"

    with pytest.raises(ValidationError, match="evidence_role"):
        TopologyNode.model_validate({**payload, "evidence_role": "unknown"})


@pytest.mark.parametrize("content_hash", ["sha256:source", "A" * 64, "a" * 63])
def test_artifact_ref_rejects_noncanonical_hash(content_hash: str) -> None:
    with pytest.raises(ValidationError, match="content_hash"):
        ArtifactRef(content_hash=content_hash, selector="row:1")


def test_extra_fields_are_forbidden() -> None:
    payload: dict[str, object] = {
        "decision_summary": "inspect the incident",
        "operations": [
            {
                "type": "query",
                "tool": "query_metric",
                "arguments": {},
                "depends_on": ["another-operation"],
            }
        ],
    }

    with pytest.raises(ValidationError, match="depends_on"):
        Decision.model_validate(payload)


def test_mixed_operation_batch_is_forbidden() -> None:
    payload: dict[str, object] = {
        "decision_summary": "query and finish",
        "operations": [
            {"type": "query", "tool": "query_metric", "arguments": {}},
            {
                "type": "finalize",
                "hypothesis_ids": [],
                "path_ids": [],
                "unresolved_questions": ["No complete trace coverage"],
                "abstain_reason": "Insufficient evidence",
            },
        ],
    }

    with pytest.raises(ValidationError, match="only query operations may be batched"):
        Decision.model_validate(payload)


def test_advance_frontier_is_a_schema_valid_single_state_operation() -> None:
    decision = Decision.model_validate(
        {
            "decision_summary": "commit the reflected page and expose the next one",
            "reflection": {
                "based_on_observation_ids": ["observation-1"],
                "evidence_update": [],
                "anomaly_disposition_update": [],
                "hypothesis_update": [],
                "current_best_hypothesis_id": None,
                "new_hypothesis": None,
                "new_hypothesis_is_current_best": False,
                "next_check": None,
                "can_finalize": False,
            },
            "operations": [{"type": "advance_frontier"}],
        }
    )

    assert decision.operations[0].type == "advance_frontier"


def test_finalize_accepts_occurrence_recommendation_as_distinct_terminal_mode() -> None:
    decision = Decision.model_validate(
        {
            "decision_summary": "recommend one corroborated occurrence interval",
            "operations": [
                {
                    "type": "finalize",
                    "occurrence_evidence_refs": ["evidence-1", "evidence-2"],
                    "unresolved_questions": ["component and mechanism remain open"],
                }
            ],
        }
    )

    assert decision.operations[0].type == "finalize"


def test_finalize_rejects_occurrence_mixed_with_finding_or_abstention() -> None:
    for extra in (
        {"hypothesis_ids": ["hypothesis-1"]},
        {"abstain_reason": "insufficient evidence"},
    ):
        with pytest.raises(ValidationError, match="occurrence recommendation"):
            Decision.model_validate(
                {
                    "decision_summary": "invalid mixed terminal mode",
                    "operations": [
                        {
                            "type": "finalize",
                            "occurrence_evidence_refs": ["evidence-1"],
                            **extra,
                        }
                    ],
                }
            )


def test_reflection_clear_current_best_conflicts_with_selecting_one() -> None:
    payload: dict[str, object] = {
        "decision_summary": "conflicting current-best ownership",
        "reflection": {
            "based_on_observation_ids": ["observation-1"],
            "evidence_update": [],
            "anomaly_disposition_update": [],
            "hypothesis_update": [],
            "current_best_hypothesis_id": "hypothesis-1",
            "clear_current_best": True,
            "new_hypothesis": None,
            "new_hypothesis_is_current_best": False,
            "next_check": None,
            "can_finalize": True,
        },
        "operations": [
            {
                "type": "finalize",
                "abstain_reason": "fixture",
            }
        ],
    }

    with pytest.raises(ValidationError, match="clear_current_best conflicts"):
        Decision.model_validate(payload)


def test_query_batch_is_bounded() -> None:
    payload: dict[str, object] = {
        "decision_summary": "query too many metrics at once",
        "operations": [
            {
                "type": "query",
                "tool": "query_metric",
                "arguments": {"metric_id": f"metric-{index}"},
            }
            for index in range(9)
        ],
    }

    with pytest.raises(ValidationError, match="at most 8 items"):
        Decision.model_validate(payload)


def test_frontier_reflection_keeps_evidence_impact_separate_from_status() -> None:
    decision = Decision.model_validate(
        {
            "decision_summary": "test bursty pressure before changing focus",
            "reflection": {
                "based_on_observation_ids": ["observation-1"],
                "evidence_update": [
                    {
                        "evidence_id": "evidence-1",
                        "hypothesis_id": "hypothesis-1",
                        "impact": "weakens",
                        "reason_code": "stable_median",
                        "explanation": "weakens sustained saturation only",
                    }
                ],
                "current_best_hypothesis_id": "hypothesis-1",
                "next_check": {
                    "operation_tools": ["query_metric"],
                    "distinguishes_hypothesis_ids": ["hypothesis-1"],
                    "purpose": "separate bursty from sustained pressure",
                    "expected_outcomes": ["burst present", "burst absent"],
                },
                "can_finalize": False,
            },
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {"entity_id": "node-1", "metric_id": "cpu"},
                }
            ],
        }
    )

    assert decision.reflection is not None
    assert decision.reflection.evidence_update[0].impact == "weakens"


def test_frontier_reflection_accepts_redundant_new_current_best_handle() -> None:
    decision = Decision.model_validate(
        {
            "decision_summary": "promote the newly grounded hypothesis",
            "reflection": {
                "based_on_observation_ids": ["observation-1"],
                "evidence_update": [],
                "hypothesis_update": [],
                "current_best_hypothesis_id": "new-best",
                "new_hypothesis": {
                    "client_id": "new-best",
                    "subject": "node-1",
                    "statement": "typed CPU pressure",
                    "derived_from": ["evidence-1"],
                    "open_questions": [],
                },
                "new_hypothesis_is_current_best": True,
                "next_check": {
                    "operation_tools": ["query_metric"],
                    "distinguishes_hypothesis_ids": ["new-best"],
                    "purpose": "check the remaining competitor",
                    "expected_outcomes": ["competitor weaker"],
                },
                "can_finalize": False,
            },
            "operations": [
                {
                    "type": "query",
                    "tool": "query_metric",
                    "arguments": {"entity_id": "node-1", "metric_id": "cpu"},
                }
            ],
        }
    )

    assert decision.reflection is not None
    assert decision.reflection.current_best_hypothesis_id == "new-best"


def test_frontier_reflection_rejects_conflicting_new_current_best_handle() -> None:
    with pytest.raises(ValidationError, match="conflicts with the new current-best"):
        Decision.model_validate(
            {
                "decision_summary": "conflicting current best",
                "reflection": {
                    "based_on_observation_ids": ["observation-1"],
                    "evidence_update": [],
                    "hypothesis_update": [],
                    "current_best_hypothesis_id": "existing-hypothesis",
                    "new_hypothesis": {
                        "client_id": "new-best",
                        "subject": "node-1",
                        "statement": "typed CPU pressure",
                        "derived_from": ["evidence-1"],
                        "open_questions": [],
                    },
                    "new_hypothesis_is_current_best": True,
                    "next_check": None,
                    "can_finalize": True,
                },
                "operations": [
                    {
                        "type": "finalize",
                        "hypothesis_ids": ["new-best"],
                        "path_ids": [],
                        "unresolved_questions": [],
                        "abstain_reason": None,
                    }
                ],
            }
        )


def test_frontier_reflection_rejects_duplicate_assessment_pair() -> None:
    assessment = {
        "evidence_id": "evidence-1",
        "hypothesis_id": "hypothesis-1",
        "impact": "supports",
        "reason_code": "aligned",
        "explanation": "aligned",
    }
    with pytest.raises(ValidationError, match="assess an evidence/hypothesis pair"):
        Decision.model_validate(
            {
                "decision_summary": "duplicate",
                "reflection": {
                    "based_on_observation_ids": ["observation-1"],
                    "evidence_update": [assessment, assessment],
                    "next_check": {
                        "operation_tools": ["query_metric"],
                        "purpose": "check",
                        "expected_outcomes": ["yes"],
                    },
                    "can_finalize": False,
                },
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_metric",
                        "arguments": {},
                    }
                ],
            }
        )


def test_material_anomaly_is_mechanism_neutral_and_deduplicated() -> None:
    anomaly = MaterialAnomaly(
        anomaly_id="anomaly-1",
        subject_id="node-1",
        signal_family="network",
        signal_ids=["receive_queue"],
        mechanism_candidates=["delay", "loss"],
        source_observation_ids=["observation-1"],
        proposal_ids=["proposal-delay", "proposal-loss"],
        materiality_basis="adapter_registered_candidate",
        priority_facts={"overview_score": 1.0},
    )

    assert anomaly.signal_ids == ["receive_queue"]
    assert anomaly.mechanism_candidates == ["delay", "loss"]


def test_resolved_anomaly_disposition_requires_evidence() -> None:
    with pytest.raises(ValidationError, match="requires at least one evidence_ref"):
        AnomalyDispositionProposal(
            anomaly_id="anomaly-1",
            disposition="background",
            explanation="present throughout both windows",
        )


def test_root_cause_anomaly_disposition_requires_hypothesis() -> None:
    with pytest.raises(ValidationError, match="requires hypothesis_id"):
        AnomalyDispositionProposal(
            anomaly_id="anomaly-1",
            disposition="root_cause",
            evidence_refs=["evidence-1"],
            explanation="direct state transition",
        )


def test_unresolved_anomaly_disposition_may_name_the_gap_without_evidence() -> None:
    disposition = AnomalyDispositionProposal(
        anomaly_id="anomaly-1",
        disposition="unresolved",
        explanation="no deployment or propagation relation is observable",
    )

    assert disposition.evidence_refs == []


def test_reflection_rejects_duplicate_anomaly_dispositions() -> None:
    item = {
        "anomaly_id": "anomaly-1",
        "disposition": "unresolved",
        "explanation": "needs another check",
    }
    with pytest.raises(ValidationError, match="disposition an anomaly only once"):
        Decision.model_validate(
            {
                "decision_summary": "duplicate anomaly disposition",
                "reflection": {
                    "based_on_observation_ids": ["observation-1"],
                    "anomaly_disposition_update": [item, item],
                    "next_check": {
                        "operation_tools": ["query_metric_profile"],
                        "purpose": "resolve the anomaly",
                        "expected_outcomes": ["mechanism supported", "not supported"],
                    },
                    "can_finalize": False,
                },
                "operations": [
                    {
                        "type": "query",
                        "tool": "query_metric_profile",
                        "arguments": {},
                    }
                ],
            }
        )
