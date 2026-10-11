import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pytest

from rootwalk.context import (
    POLICY_VERSION,
    TOKEN_ESTIMATOR_VERSION,
    candidate_disposition_queue,
    candidate_log_selector_hint,
    candidate_schedule,
    compile_context,
    frontier_profile_seeds,
)
from rootwalk.contracts import (
    AllowedScope,
    AnomalyDisposition,
    ArtifactRef,
    Coverage,
    Evidence,
    EvidenceAssessment,
    Extractor,
    Hypothesis,
    MaterialAnomaly,
    Observation,
    Task,
    TimeRange,
)
from rootwalk.version import PROMPT_VERSION, TOOL_VERSION

START = datetime(2026, 1, 1, tzinfo=UTC)
WINDOW = TimeRange(start=START, end=START + timedelta(minutes=10))


def task(question: str = "Why is checkout unhealthy?") -> Task:
    return Task(
        task_id="task-1",
        environment="test",
        question=question,
        incident_range=WINDOW,
        timezone="UTC",
        allowed_query_range=WINDOW,
        entry_entities=["checkout"],
        allowed_scope=AllowedScope(
            entities=["checkout", "database"],
            tools=["query_metric", "read_artifact_slice"],
        ),
        requested_outputs=["root_cause"],
        dataset_manifest_ref="manifest-1",
    )


def artifact(char: str = "a", visibility: str = "model_readable") -> ArtifactRef:
    return ArtifactRef.model_validate(
        {
            "content_hash": char * 64,
            "selector": "row:1",
            "visibility": visibility,
        }
    )


def evidence(
    evidence_id: str,
    fingerprint: str,
    *,
    subject: str = "database",
    value: object = True,
    refs: list[ArtifactRef] | None = None,
) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        observation_refs=["observation-1"],
        subject_ids=[subject],
        predicate="latency_increased",
        value=cast(Any, value),
        observed_interval=WINDOW,
        time_uncertainty="one sample",
        artifact_refs=refs or [artifact()],
        extractor=Extractor(
            kind="deterministic",
            name="test",
            version="1",
            parameters_hash="params",
        ),
        source_fingerprint=fingerprint,
        dataset_snapshot_id="snapshot-1",
    )


def hypothesis(
    *, supporting: list[str] | None = None, opposing: list[str] | None = None
) -> Hypothesis:
    return Hypothesis(
        hypothesis_id="hypothesis-1",
        subject="database",
        statement="database saturation caused checkout errors",
        status="open",
        supporting_evidence_refs=supporting or [],
        opposing_evidence_refs=opposing or [],
    )


def observation(number: int, summary: str) -> Observation:
    return Observation(
        observation_id=f"observation-{number}",
        tool="query_metric",
        arguments={"name": f"metric-{number}"},
        status="success",
        coverage=Coverage(
            requested_range=WINDOW,
            observed_range=WINDOW,
            completeness="complete",
            collection_health="healthy",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id=f"attempt-{number}",
        summary=summary,
    )


def public_metric_pair(
    requires_paired_exact: bool | None = True,
) -> tuple[list[Observation], list[Evidence]]:
    candidate: dict[str, Any] = {
        "fault_type": "high memory usage",
        "entity_id": "database",
        "resource_family": "memory",
        "metric_id": "memory-used",
    }
    if requires_paired_exact is not None:
        candidate["requires_paired_exact"] = requires_paired_exact
    overview = Observation(
        observation_id="observation-overview",
        tool="summarize_anomalies",
        arguments={},
        status="success",
        coverage=Coverage(
            requested_range=WINDOW,
            completeness="unknown",
            basis="fixture",
        ),
        artifact_refs=[],
        attempt_id="attempt-overview",
        summary=json.dumps(
            {
                "public_fault_types": ["high memory usage"],
                "fault_type_candidates": [candidate],
            }
        ),
    )
    metric_observations = [
        observation(index, json.dumps({"mean": value})).model_copy(
            update={
                "observation_id": f"observation-{period}",
                "arguments": {
                    "entity_id": "database",
                    "metric_id": "memory-used",
                    "period": period,
                },
            }
        )
        for index, (period, value) in enumerate(
            (("reference", 95.0), ("incident", 98.0)), 1
        )
    ]
    metric_evidence = [
        evidence(
            f"evidence-{period}",
            f"source-{period}",
            value={"summary": {"mean": value}},
        ).model_copy(
            update={
                "observation_refs": [metric_observations[index].observation_id],
                "predicate": "query_metric.result",
            }
        )
        for index, (period, value) in enumerate(
            (("reference", 95.0), ("incident", 98.0))
        )
    ]
    return [overview, *metric_observations], metric_evidence


def context(result: object) -> dict[str, Any]:
    messages = cast(Any, result).messages
    return cast(dict[str, Any], json.loads(messages[1]["content"]))


def test_same_state_has_identical_messages_manifest_and_hash() -> None:
    kwargs = {
        "task": task(),
        "tools": {"query_metric": "query_metric(name: str)"},
        "budget": {"model_calls_remaining": 4},
        "revision": 12,
        "observations": [observation(1, "old"), observation(2, "new")],
        "evidence": [evidence("evidence-2", "source-2")],
        "hypotheses": [hypothesis(supporting=["evidence-2"])],
        "strategy": "evidence",
        "validator_mode": "enforce",
        "max_tokens": 20_000,
    }

    first = compile_context(**kwargs)  # type: ignore[arg-type]
    second = compile_context(**kwargs)  # type: ignore[arg-type]

    assert first == second
    assert first.manifest["base_revision"] == 12
    assert POLICY_VERSION == "m2-context-v83"
    assert first.manifest["policy_version"] == "m2-context-v83"
    assert PROMPT_VERSION == "m124"
    assert TOOL_VERSION == "openrca-public-input-v82"
    assert first.manifest["token_estimator_version"] == TOKEN_ESTIMATOR_VERSION
    assert len(first.manifest["message_hash"]) == 64


def test_frontier_context_exposes_unreflected_observation_without_legacy_queue() -> (
    None
):
    item = observation(1, "cpu maximum increased while median stayed stable")
    item_evidence = evidence("evidence-1", "source-1", subject="checkout").model_copy(
        update={"observation_refs": [item.observation_id]}
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(entity_id: str, metric_id: str)"},
        {"deadline_seconds_remaining": 300},
        9,
        observations=[item],
        evidence=[item_evidence],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    payload = context(result)
    assert result.manifest["policy_version"] == "frontier-reflection-v55"
    assert payload["frontier_control"] == {
        "reflection_required": True,
        "unreflected_observation_ids": ["observation-1"],
        "max_live_hypotheses": 12,
        "max_new_hypotheses_per_reflection": 1,
        "max_query_operations": 8,
        "challenger_discovery_required": False,
        "advance_frontier_contract": {
            "state_change_required": True,
            "unresolved_active_anomaly_ids": [],
            "every_id_requires_resolved_disposition": True,
            "next_check_must_be_null": True,
        },
        "page_certificate": {
            "policy": "missing-slots-v1",
            "output_field": "Decision.reflection.anomaly_disposition_update",
            "missing_slot_ids": [],
            "missing_slots": [],
            "fill_only_missing_slots": True,
            "partial_valid_updates_are_persisted": True,
            "prose_does_not_fill_slots": True,
        },
        "response_schema_constraints": {
            "anomaly_disposition_update_min_items": 0,
            "actionable_anomaly_ids": [],
            "unsupported_anomaly_ids": [],
            "applies_before_physical_closeout": True,
        },
        "recommendation_certificate": {
            "current_best_hypothesis_id": None,
            "eligible_profile_refs": [],
            "explicit_support_refs": [],
            "missing_explicit_support_refs": [],
            "derived_from_is_not_polarity": True,
        },
        "leader_selection_contract": {
            "required": False,
            "eligible_hypothesis_ids": [],
            "output_field": "Decision.reflection.current_best_hypothesis_id",
            "prose_does_not_select_leader": True,
        },
        "leader_upgrade_contract": {
            "required": False,
            "current_best_hypothesis_id": None,
            "current_best_strength": {
                "mechanism_rank": -1,
                "root_cause_score": 0.0,
            },
            "stronger_profiles": [],
            "eligible_existing_hypothesis_ids": [],
            "new_hypothesis_fields": [
                "new_hypothesis",
                "evidence_update supports",
                "new_hypothesis_is_current_best",
            ],
            "existing_hypothesis_fields": [
                "evidence_update supports",
                "current_best_hypothesis_id",
            ],
            "rejected_decision_state_persisted": False,
            "prose_does_not_upgrade_leader": True,
        },
        "hypothesis_creation_contract": {
            "required": False,
            "eligible_profiles": [],
            "proposal_field": "Decision.reflection.new_hypothesis",
            "temporary_handle": "new_hypothesis",
            "required_companion_fields": [
                "evidence_update supports",
                "new_hypothesis_is_current_best",
            ],
            "prose_does_not_create_hypothesis": True,
        },
        "occurrence_selection_contract": {
            "enabled": False,
            "outcome_type": "occurrence_recommendation",
            "output_field": "finalize.occurrence_evidence_refs",
            "requires_complete_anomaly_coverage": False,
            "impact_probe_required": False,
            "impact_evidence_refs": [],
            "selection_ready": False,
            "pending_material_profile_pair_count": 0,
            "candidate_count": 0,
            "strong_candidate_count": 0,
            "candidates": [],
            "runtime_selects_candidate": False,
            "closeout_supersedes_generic_frontier_gates": False,
            "prose_does_not_select_occurrence": True,
        },
        "semantic_rewrite": "forbidden",
    }
    assert payload["recent_observations"][0]["derived_evidence_ids"] == ["evidence-1"]
    assert "candidate_evidence_summary" not in payload
    assert "does not rewrite" in result.messages[0]["content"]


def test_frontier_context_projects_impact_and_trace_operation_evidence() -> None:
    item = observation(1, "typed impact and operation profiles")
    impact = evidence("impact-profile", "impact-source").model_copy(
        update={
            "observation_refs": [item.observation_id],
            "evidence_kind": "impact_profile",
            "mechanism_dimension": "business_impact",
            "semantic_quality": "typed",
            "value": {
                "status": "success",
                "profiles": [{"entity_id": "database", "effect_size": 0.8}],
                "ignored": "not projected",
            },
        }
    )
    operation = evidence("operation-profile", "operation-source").model_copy(
        update={
            "observation_refs": [item.observation_id],
            "evidence_kind": "trace_operation_profile",
            "mechanism_dimension": "operation_profile",
            "semantic_quality": "typed",
            "value": {
                "status": "success",
                "operations": [{"operation_name": "request", "effect_size": 0.7}],
                "ignored": "not projected",
            },
        }
    )

    result = compile_context(
        task(),
        {},
        {},
        13,
        observations=[item],
        evidence=[impact, operation],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    projected = {item["evidence_id"]: item for item in context(result)["evidence"]}
    assert projected[impact.evidence_id]["value"] == {
        "status": "success",
        "profiles": [{"entity_id": "database", "effect_size": 0.8}],
        "omitted_item_count": 0,
    }
    assert projected[operation.evidence_id]["value"] == {
        "status": "success",
        "operations": [{"operation_name": "request", "effect_size": 0.7}],
        "omitted_item_count": 0,
    }


def test_frontier_compacts_unreflected_profile_arguments() -> None:
    item = Observation(
        observation_id="profile-observation",
        tool="query_metric_profile",
        arguments={
            "entity_id": "checkout",
            "fault_type": "CPU pressure",
            "metric_ids": ["cpu", "run_queue", "throttle"],
        },
        status="success",
        coverage=Coverage(
            requested_range=WINDOW, completeness="complete", basis="fixture"
        ),
        artifact_refs=[],
        attempt_id="profile-attempt",
    )

    result = compile_context(
        task(),
        {"query_metric_profile": "query_metric_profile(...)"},
        {},
        1,
        observations=[item],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    arguments = context(result)["recent_observations"][0]["arguments"]
    assert arguments == {
        "entity_id": "checkout",
        "fault_type": "CPU pressure",
        "metric_count": 3,
    }


def test_frontier_context_surfaces_deterministic_finding_gate_index() -> None:
    item = observation(1, "typed metric profile")
    item_evidence = evidence(
        "profile-evidence", "profile-source", subject="checkout"
    ).model_copy(
        update={
            "observation_refs": [item.observation_id],
            "evidence_kind": "metric_profile",
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "single_component_ready": True,
                    "fault_type_ready": True,
                    "boundary_aligned": False,
                    "requested_fault_type": "CPU pressure",
                    "mechanism_signature": "sustained_high_saturation",
                    "root_cause_score": 1.0,
                    "strong_metric_count": 2,
                    "qualifying_metric_count": 2,
                    "qualifying_metric_ids": ["cpu", "throttle"],
                    "earliest_strong_onset_timestamp": 1_767_225_600.0,
                    "trace_required": False,
                }
            },
        }
    )

    result = compile_context(
        task(),
        {"query_metric_profile": "query_metric_profile(...)"},
        {"deadline_seconds_remaining": 300},
        10,
        observations=[item],
        evidence=[item_evidence],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    gate = context(result)["finding_gate_index"][0]
    assert gate["evidence_id"] == "profile-evidence"
    assert gate["single_component_ready"] is True
    assert gate["qualifying_metric_count"] == 2
    creation = context(result)["frontier_control"]["hypothesis_creation_contract"]
    assert creation["required"] is True
    assert creation["eligible_profiles"][0]["evidence_id"] == "profile-evidence"


def test_frontier_context_creates_hypothesis_from_signature_free_ready_profile() -> (
    None
):
    item = observation(1, "typed peer-isolated resource profile")
    item_evidence = evidence(
        "profile-evidence", "profile-source", subject="checkout"
    ).model_copy(
        update={
            "observation_refs": [item.observation_id],
            "evidence_kind": "metric_profile",
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "fault_type_ready": True,
                    "material_event_ready": True,
                    "boundary_aligned": False,
                    "requested_fault_type": "high CPU usage",
                    "mechanism_signature": None,
                    "signature_free_finding_ready": True,
                    "material_effect_metric_ids": ["cpu", "load"],
                    "root_cause_score": 0.7,
                    "strong_metric_count": 3,
                    "qualifying_metric_count": 3,
                }
            },
        }
    )

    result = compile_context(
        task(),
        {"query_metric_profile": "query_metric_profile(...)"},
        {"deadline_seconds_remaining": 300},
        10,
        observations=[item],
        evidence=[item_evidence],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    creation = context(result)["frontier_control"]["hypothesis_creation_contract"]
    assert creation["required"] is True
    assert creation["eligible_profiles"] == [
        {
            "evidence_id": item_evidence.evidence_id,
            "subject_ids": ["checkout"],
            "fault_type": "high CPU usage",
            "mechanism_signature": None,
            "root_cause_score": 0.7,
        }
    ]


def test_frontier_context_requires_nonempty_disposition_when_page_is_actionable() -> (
    None
):
    item = observation(1, "typed winner profile")
    item_evidence = evidence(
        "profile-evidence", "profile-source", subject="database"
    ).model_copy(
        update={
            "observation_refs": [item.observation_id],
            "evidence_kind": "metric_profile",
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "requested_fault_type": "high CPU usage",
                    "fault_type_ready": True,
                    "material_event_ready": True,
                    "boundary_aligned": False,
                    "mechanism_signature": None,
                    "signature_free_finding_ready": True,
                    "material_effect_metric_ids": ["cpu", "load"],
                    "root_cause_score": 0.7,
                }
            },
        }
    )
    current = hypothesis().model_copy(
        update={"hypothesis_id": "winner", "subject": "database"}
    )
    assessment = EvidenceAssessment(
        assessment_id="winner-support",
        evidence_id=item_evidence.evidence_id,
        hypothesis_id=current.hypothesis_id,
        impact="supports",
        reason_code="typed_profile",
        explanation="the audited material effect supports the winner",
        observation_digest="a" * 64,
    )
    anomaly = MaterialAnomaly(
        anomaly_id="root-anomaly",
        subject_id="database",
        signal_family="cpu",
        signal_ids=["cpu", "load"],
        mechanism_candidates=["high CPU usage"],
        source_observation_ids=[item.observation_id],
        proposal_ids=["root-proposal"],
        materiality_basis="fixture",
    )

    result = compile_context(
        task(),
        {},
        {"deadline_seconds_remaining": 300, "closeout_required": False},
        4,
        observations=[item],
        evidence=[item_evidence],
        hypotheses=[current],
        assessments=[assessment],
        material_anomalies=[anomaly],
        current_best_hypothesis_id=current.hypothesis_id,
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    constraints = context(result)["frontier_control"]["response_schema_constraints"]
    assert constraints == {
        "anomaly_disposition_update_min_items": 1,
        "actionable_anomaly_ids": [anomaly.anomaly_id],
        "unsupported_anomaly_ids": [],
        "applies_before_physical_closeout": True,
    }


def test_frontier_context_exposes_occurrence_only_evidence_consensus() -> None:
    occurrence_task = task().model_copy(
        update={"requested_outputs": ["root cause occurrence datetime"]}
    )
    first = evidence("onset-1", "onset-source-1", subject="database").model_copy(
        update={
            "onset_interval": WINDOW,
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "strong_metric_count": 1,
                    "qualifying_metric_count": 1,
                    "peer_isolated_metric_ids": ["latency"],
                    "sampling_coverage_ready": True,
                }
            },
        }
    )
    second = evidence("onset-2", "onset-source-2", subject="database").model_copy(
        update={
            "onset_interval": WINDOW,
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "strong_metric_count": 0,
                    "qualifying_metric_count": 0,
                    "peer_isolated_metric_ids": [],
                    "sampling_coverage_ready": True,
                }
            },
        }
    )
    impact_window = TimeRange(
        start=START + timedelta(minutes=5),
        end=START + timedelta(minutes=6),
    )
    impact_observation = observation(2, "typed impact profile").model_copy(
        update={"tool": "query_impact_profile", "arguments": {}}
    )
    impact = evidence("impact-1", "impact-source", subject="checkout").model_copy(
        update={
            "observation_refs": [impact_observation.observation_id],
            "subject_ids": [],
            "onset_interval": impact_window,
            "semantic_quality": "typed",
            "evidence_kind": "impact_profile",
        }
    )

    result = compile_context(
        occurrence_task,
        {},
        {},
        1,
        observations=[impact_observation],
        evidence=[first, second, impact],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    contract = context(result)["frontier_control"]["occurrence_selection_contract"]
    assert contract["enabled"] is True
    assert contract["requires_complete_anomaly_coverage"] is False
    assert contract["impact_probe_required"] is False
    assert contract["impact_evidence_refs"] == ["impact-1"]
    assert contract["selection_ready"] is True
    assert contract["candidate_count"] == 1
    assert contract["strong_candidate_count"] == 1
    assert contract["runtime_selects_candidate"] is False
    assert contract["closeout_supersedes_generic_frontier_gates"] is True
    window_json = WINDOW.model_dump(mode="json")
    assert contract["candidates"] == [
        {
            "subject_id": "database",
            "onset_interval": {
                "start": window_json["start"],
                "end": window_json["end"],
            },
            "evidence_refs": ["onset-1", "onset-2"],
            "evidence_count": 2,
            "fault_types": [],
            "fault_type_ready": False,
            "material_event_ready": False,
            "impact_alignment": {
                "relation": "overlaps_transition",
                "impact_evidence_ref": "impact-1",
                "distance_seconds": 0.0,
                "navigation_rank": 0,
            },
            "occurrence_support": {
                "ready": True,
                "basis": "impact_proximal_strong_signal",
                "strong_metric_count": 1,
                "qualifying_metric_count": 1,
                "peer_isolated_metric_ids": ["latency"],
                "sampling_coverage_ready": True,
            },
        }
    ]


def test_frontier_recommendation_certificate_requires_explicit_profile_support() -> (
    None
):
    item = observation(1, "typed metric profile")
    item_evidence = evidence(
        "profile-evidence", "profile-source", subject="database"
    ).model_copy(
        update={
            "observation_refs": [item.observation_id],
            "evidence_kind": "metric_profile",
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "fault_type_ready": True,
                    "boundary_aligned": False,
                    "mechanism_signature": "availability_state_transition",
                    "root_cause_score": 1.0,
                    "requested_fault_type": "db close",
                }
            },
        }
    )
    current = hypothesis().model_copy(
        update={
            "hypothesis_id": "current-best",
            "subject": "database",
            "created_from": [item_evidence.evidence_id],
        }
    )

    missing = compile_context(
        task(),
        {},
        {},
        1,
        observations=[item],
        evidence=[item_evidence],
        hypotheses=[current],
        current_best_hypothesis_id=current.hypothesis_id,
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )
    certificate = context(missing)["frontier_control"]["recommendation_certificate"]
    assert certificate["eligible_profile_refs"] == [item_evidence.evidence_id]
    assert certificate["missing_explicit_support_refs"] == [item_evidence.evidence_id]

    assessment = EvidenceAssessment(
        assessment_id="assessment-1",
        evidence_id=item_evidence.evidence_id,
        hypothesis_id=current.hypothesis_id,
        impact="supports",
        reason_code="typed_profile",
        explanation="the typed transition supports the current best",
        observation_digest="a" * 64,
    )
    complete = compile_context(
        task(),
        {},
        {},
        2,
        observations=[item],
        evidence=[item_evidence],
        hypotheses=[current],
        assessments=[assessment],
        current_best_hypothesis_id=current.hypothesis_id,
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )
    complete_certificate = context(complete)["frontier_control"][
        "recommendation_certificate"
    ]
    assert complete_certificate["explicit_support_refs"] == [item_evidence.evidence_id]
    assert complete_certificate["missing_explicit_support_refs"] == []

    unselected = compile_context(
        task(),
        {},
        {},
        3,
        observations=[item],
        evidence=[item_evidence],
        hypotheses=[current],
        assessments=[assessment],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )
    leader_contract = context(unselected)["frontier_control"][
        "leader_selection_contract"
    ]
    assert leader_contract["required"] is True
    assert leader_contract["eligible_hypothesis_ids"] == [current.hypothesis_id]


def test_frontier_context_requires_atomic_upgrade_to_stronger_subject() -> None:
    item = observation(1, "two typed mechanism profiles")
    weak = evidence("weak-profile", "weak-source", subject="checkout").model_copy(
        update={
            "observation_refs": [item.observation_id],
            "evidence_kind": "metric_profile",
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "fault_type_ready": False,
                    "material_event_ready": True,
                    "boundary_aligned": False,
                    "mechanism_signature": "peer_isolated_queue_congestion_event",
                    "root_cause_score": 0.4,
                    "requested_fault_type": "network delay",
                }
            },
        }
    )
    strong = evidence("strong-profile", "strong-source", subject="database").model_copy(
        update={
            "observation_refs": [item.observation_id],
            "evidence_kind": "metric_profile",
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "fault_type_ready": False,
                    "material_event_ready": True,
                    "boundary_aligned": False,
                    "mechanism_signature": "peer_isolated_trace_error_event",
                    "root_cause_score": 0.5,
                    "requested_fault_type": "db connection limit",
                }
            },
        }
    )
    current = hypothesis().model_copy(
        update={
            "hypothesis_id": "current-best",
            "subject": "checkout",
            "created_from": [weak.evidence_id],
        }
    )
    assessment = EvidenceAssessment(
        assessment_id="weak-support",
        evidence_id=weak.evidence_id,
        hypothesis_id=current.hypothesis_id,
        impact="supports",
        reason_code="typed_material_event",
        explanation="the queue event supports the provisional leader",
        observation_digest="a" * 64,
    )

    result = compile_context(
        task(),
        {},
        {"deadline_seconds_remaining": 300},
        4,
        observations=[item],
        evidence=[weak, strong],
        hypotheses=[current],
        assessments=[assessment],
        current_best_hypothesis_id=current.hypothesis_id,
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    control = context(result)["frontier_control"]
    upgrade = control["leader_upgrade_contract"]
    assert upgrade["required"] is True
    assert upgrade["current_best_strength"] == {
        "mechanism_rank": 1,
        "root_cause_score": 0.4,
    }
    assert upgrade["stronger_profiles"] == [
        {
            "evidence_id": strong.evidence_id,
            "subject_ids": ["database"],
            "fault_type": "db connection limit",
            "mechanism_signature": "peer_isolated_trace_error_event",
            "mechanism_rank": 3,
            "root_cause_score": 0.5,
            "eligible_existing_hypothesis_ids": [],
        }
    ]
    assert upgrade["rejected_decision_state_persisted"] is False
    creation = control["hypothesis_creation_contract"]
    assert creation["required"] is True
    assert creation["eligible_profiles"] == [
        {
            "evidence_id": strong.evidence_id,
            "subject_ids": ["database"],
            "fault_type": "db connection limit",
            "mechanism_signature": "peer_isolated_trace_error_event",
            "root_cause_score": 0.5,
        }
    ]


def test_frontier_profile_seeds_are_subject_and_family_diverse() -> None:
    overview = Observation(
        observation_id="overview",
        tool="summarize_anomalies",
        arguments={},
        status="success",
        coverage=Coverage(
            requested_range=WINDOW, completeness="unknown", basis="fixture"
        ),
        artifact_refs=[],
        attempt_id="overview-attempt",
        summary=json.dumps(
            {
                "public_fault_types": ["cpu", "disk"],
                "fault_type_candidates": [
                    {
                        "candidate_id": f"candidate-{index}",
                        "candidate_rank": index,
                        "fault_type": fault_type,
                        "entity_id": subject,
                        "metric_id": metric,
                        "metric_group": [metric],
                        "resource_family": (
                            "storage" if fault_type == "disk" else "cpu"
                        ),
                        "score": 1.0,
                        "overview_score": 1.0,
                        "requires_paired_exact": True,
                    }
                    for index, (fault_type, subject, metric) in enumerate(
                        [
                            ("cpu", "pod-a", "cpu-a"),
                            ("disk", "pod-a", "disk-a"),
                            ("cpu", "pod-b", "cpu-b"),
                            ("disk", "node-a", "disk-node"),
                        ]
                    )
                ],
            }
        ),
    )

    seeds = frontier_profile_seeds([overview])

    assert [seed["subject"] for seed in seeds] == [
        "pod-a",
        "pod-a",
        "pod-b",
        "node-a",
    ]


def test_frontier_storage_sibling_uses_same_subject_metric_template() -> None:
    overview = Observation(
        observation_id="overview-storage",
        tool="summarize_anomalies",
        arguments={},
        status="success",
        coverage=Coverage(
            requested_range=WINDOW, completeness="unknown", basis="fixture"
        ),
        artifact_refs=[],
        attempt_id="overview-storage-attempt",
        summary=json.dumps(
            {
                "fault_type_candidates": [
                    {
                        "candidate_id": "pod-read",
                        "candidate_rank": 0,
                        "fault_type": "container read I/O load",
                        "entity_id": "market:pod:email",
                        "metric_id": "container-read",
                        "metric_group": ["container-read"],
                        "resource_family": "storage",
                        "score": 1.0,
                        "overview_score": 1.0,
                        "requires_paired_exact": True,
                        "materiality": "navigation_only",
                    },
                    {
                        "candidate_id": "pod-write",
                        "candidate_rank": 1,
                        "fault_type": "container write I/O load",
                        "entity_id": "market:pod:email",
                        "metric_id": "container-write",
                        "metric_group": ["container-write"],
                        "resource_family": "storage",
                        "score": 0.9,
                        "overview_score": 0.9,
                        "requires_paired_exact": True,
                        "materiality": "navigation_only",
                    },
                    {
                        "candidate_id": "node-write",
                        "candidate_rank": 10,
                        "fault_type": "container write I/O load",
                        "entity_id": "market:node:host",
                        "metric_id": "node-write",
                        "metric_group": ["node-write"],
                        "resource_family": "storage",
                        "score": 0.1,
                        "overview_score": 0.1,
                        "requires_paired_exact": True,
                        "materiality": "navigation_only",
                    },
                ]
            }
        ),
    )

    seeds = frontier_profile_seeds([overview], limit=2, materiality="navigation_only")

    assert seeds == [
        {
            "subject": "market:pod:email",
            "fault_type": "container read I/O load",
            "resource_family": "storage",
            "metric_ids": ["container-read"],
        },
        {
            "subject": "market:pod:email",
            "fault_type": "container write I/O load",
            "resource_family": "storage",
            "metric_ids": ["container-write"],
        },
    ]


def test_frontier_context_separates_material_and_navigation_profile_seeds() -> None:
    overview = Observation(
        observation_id="overview",
        tool="summarize_anomalies",
        arguments={},
        status="success",
        coverage=Coverage(
            requested_range=WINDOW, completeness="unknown", basis="fixture"
        ),
        artifact_refs=[],
        attempt_id="overview-attempt",
        summary=json.dumps(
            {
                "fault_type_candidates": [
                    {
                        "candidate_id": "material-db",
                        "candidate_rank": 1,
                        "fault_type": "connection pressure",
                        "entity_id": "database",
                        "metric_id": "sessions",
                        "metric_group": ["sessions"],
                        "resource_family": "other",
                        "score": 1.0,
                        "overview_score": 1.0,
                        "requires_paired_exact": True,
                        "materiality": "material",
                    },
                    {
                        "candidate_id": "navigation-cpu",
                        "candidate_rank": 2,
                        "fault_type": "cpu pressure",
                        "entity_id": "checkout",
                        "metric_id": "cpu",
                        "metric_group": ["cpu"],
                        "resource_family": "cpu",
                        "score": 0.8,
                        "overview_score": 0.8,
                        "requires_paired_exact": True,
                        "materiality": "navigation_only",
                    },
                ]
            }
        ),
    )
    anomaly = MaterialAnomaly(
        anomaly_id="anomaly-database",
        subject_id="database",
        signal_family="other",
        signal_ids=["sessions"],
        mechanism_candidates=["connection pressure"],
        source_observation_ids=[overview.observation_id],
        proposal_ids=["material-db"],
        materiality_basis="fixture",
    )

    result = compile_context(
        task=task(),
        tools={"query_metric_profile": "query_metric_profile(...)"},
        budget={},
        revision=1,
        observations=[overview],
        material_anomalies=[anomaly],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    payload = context(result)
    assert [item["subject"] for item in payload["initial_profile_seeds"]] == [
        "database"
    ]
    assert [item["subject"] for item in payload["challenger_profile_seeds"]] == [
        "checkout"
    ]


def test_frontier_context_pages_unresolved_material_anomalies_first() -> None:
    anomalies = [
        MaterialAnomaly(
            anomaly_id=f"anomaly-{index}",
            subject_id="database" if index else "checkout",
            signal_family="latency",
            signal_ids=[f"latency-{index}"],
            mechanism_candidates=["dependency latency"],
            source_observation_ids=["observation-1"],
            proposal_ids=[f"proposal-{index}"],
            materiality_basis="fixture",
            priority_facts={"overview_score": 1.0 - index / 10},
        )
        for index in range(2)
    ]
    resolved = AnomalyDisposition(
        disposition_id="disposition-1",
        anomaly_id="anomaly-0",
        disposition="background",
        evidence_refs=["evidence-1"],
        explanation="present before the incident",
        observation_digest="a" * 64,
    )

    result = compile_context(
        task=task(),
        tools={"query_metric": "query_metric(...)"},
        budget={},
        revision=1,
        observations=[],
        evidence=[],
        hypotheses=[],
        material_anomalies=anomalies,
        anomaly_dispositions=[resolved],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    payload = context(result)
    registry = payload["material_anomaly_registry"]
    assert registry["total_count"] == 2
    assert registry["resolved_count"] == 1
    assert registry["unresolved_count"] == 1
    assert registry["active"][0]["anomaly_id"] == "anomaly-1"
    assert registry["next_unresolved_anomaly_ids"] == ["anomaly-1"]
    assert payload["frontier_control"]["advance_frontier_contract"][
        "unresolved_active_anomaly_ids"
    ] == ["anomaly-1"]
    certificate = payload["frontier_control"]["page_certificate"]
    assert certificate["missing_slot_ids"] == ["anomaly-1"]
    assert certificate["missing_slots"][0] == {
        "anomaly_id": "anomaly-1",
        "coverage_complete": False,
        "profile_evidence_refs": [],
        "observation_refs": [],
    }


def test_frontier_pages_unprofiled_anomalies_before_complete() -> None:
    complete = MaterialAnomaly(
        anomaly_id="anomaly-complete",
        subject_id="checkout",
        signal_family="cpu",
        signal_ids=["cpu"],
        mechanism_candidates=["CPU pressure"],
        source_observation_ids=["observation-1"],
        proposal_ids=["proposal-complete"],
        materiality_basis="fixture",
    )
    incomplete = MaterialAnomaly(
        anomaly_id="anomaly-incomplete",
        subject_id="database",
        signal_family="database",
        signal_ids=["sessions"],
        mechanism_candidates=["connection pressure"],
        source_observation_ids=["observation-1"],
        proposal_ids=["proposal-incomplete"],
        materiality_basis="fixture",
    )
    profile = evidence(
        "complete-profile", "complete-source", subject="checkout"
    ).model_copy(
        update={
            "evidence_kind": "metric_profile",
            "semantic_quality": "typed",
            "value": {
                "finding_gate_summary": {
                    "requested_fault_type": "CPU pressure",
                    "fault_type_ready": False,
                }
            },
        }
    )

    result = compile_context(
        task(),
        {},
        {},
        1,
        evidence=[profile],
        material_anomalies=[complete, incomplete],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    active_ids = [
        item["anomaly_id"]
        for item in context(result)["material_anomaly_registry"]["active"]
    ]
    assert active_ids == ["anomaly-incomplete", "anomaly-complete"]

    ready_profile = profile.model_copy(
        update={
            "value": {
                "finding_gate_summary": {
                    "requested_fault_type": "CPU pressure",
                    "fault_type_ready": True,
                    "mechanism_signature": "sustained_high_saturation",
                }
            }
        }
    )
    leader_context = compile_context(
        task(),
        {},
        {},
        2,
        evidence=[ready_profile],
        material_anomalies=[complete, incomplete],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )
    leader_active_ids = [
        item["anomaly_id"]
        for item in context(leader_context)["material_anomaly_registry"]["active"]
    ]
    assert leader_active_ids == ["anomaly-complete", "anomaly-incomplete"]


def test_frontier_anomaly_registry_exposes_complete_negative_mechanism_evidence() -> (
    None
):
    anomaly = MaterialAnomaly(
        anomaly_id="anomaly-1",
        subject_id="database",
        signal_family="database",
        signal_ids=["sessions"],
        mechanism_candidates=["connection pressure"],
        source_observation_ids=["observation-1"],
        proposal_ids=["proposal-1"],
        materiality_basis="fixture",
    )
    profile = evidence(
        "profile-1",
        "profile-source",
        value={
            "finding_gate_summary": {
                "requested_fault_type": "connection pressure",
                "fault_type_ready": False,
                "mechanism_signature": None,
                "root_cause_score": 0.0,
            }
        },
    ).model_copy(
        update={"evidence_kind": "metric_profile", "semantic_quality": "typed"}
    )

    result = compile_context(
        task=task(),
        tools={},
        budget={},
        revision=1,
        evidence=[profile],
        material_anomalies=[anomaly],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    coverage = context(result)["material_anomaly_registry"]["active"][0][
        "mechanism_coverage"
    ]
    assert coverage["complete"] is True
    assert coverage["all_profiled_not_ready"] is True
    assert coverage["negative_evidence_ids"] == ["profile-1"]
    slot = context(result)["frontier_control"]["page_certificate"]["missing_slots"][0]
    assert slot["profile_evidence_refs"] == ["profile-1"]
    assert slot["observation_refs"] == ["observation-1"]


def test_frontier_context_keeps_root_dispositions_and_compacts_assessments() -> None:
    anomaly = MaterialAnomaly(
        anomaly_id="anomaly-root",
        subject_id="checkout",
        signal_family="cpu",
        signal_ids=["cpu"],
        mechanism_candidates=["cpu pressure"],
        source_observation_ids=["observation-1"],
        proposal_ids=["proposal-1"],
        materiality_basis="fixture",
    )
    profile = evidence("profile-root", "profile-source", subject="checkout")
    hypothesis_item = Hypothesis(
        hypothesis_id="hypothesis-root",
        subject="checkout",
        statement="checkout CPU pressure",
        status="supported",
        supporting_evidence_refs=[profile.evidence_id],
    )
    disposition = AnomalyDisposition(
        disposition_id="disposition-root",
        anomaly_id=anomaly.anomaly_id,
        disposition="root_cause",
        hypothesis_id=hypothesis_item.hypothesis_id,
        evidence_refs=[profile.evidence_id],
        explanation="typed profile identifies the root event",
        observation_digest="a" * 64,
    )
    assessment = EvidenceAssessment(
        assessment_id="assessment-root",
        evidence_id=profile.evidence_id,
        hypothesis_id=hypothesis_item.hypothesis_id,
        impact="supports",
        reason_code="typed_profile",
        explanation="x" * 500,
        observation_digest="b" * 64,
    )

    result = compile_context(
        task=task(),
        tools={},
        budget={},
        revision=1,
        evidence=[profile],
        hypotheses=[hypothesis_item],
        assessments=[assessment],
        material_anomalies=[anomaly],
        anomaly_dispositions=[disposition],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    payload = context(result)
    assert payload["material_anomaly_registry"]["root_cause_dispositions"] == [
        {
            "anomaly_id": anomaly.anomaly_id,
            "hypothesis_id": hypothesis_item.hypothesis_id,
            "evidence_refs": [profile.evidence_id],
        }
    ]
    assert payload["evidence_assessments"] == [
        {
            "hypothesis_id": hypothesis_item.hypothesis_id,
            "supports": [profile.evidence_id],
            "weakens": [],
            "refutes": [],
        }
    ]


def test_frontier_context_surfaces_registered_service_profile_seeds() -> None:
    overview = Observation(
        observation_id="overview-service",
        tool="summarize_anomalies",
        arguments={},
        status="success",
        coverage=Coverage(
            requested_range=WINDOW, completeness="unknown", basis="fixture"
        ),
        artifact_refs=[],
        attempt_id="overview-service-attempt",
        summary=json.dumps(
            {
                "service_profile_seeds": [
                    {
                        "service_id": "service:checkout",
                        "fault_type": "container network packet corruption",
                        "member_entity_ids": ["pod:checkout-0", "pod:checkout-1"],
                        "consensus_member_count": 2,
                        "member_count": 2,
                        "consensus_ratio": 1.0,
                        "overview_score": 0.8,
                    }
                ]
            }
        ),
    )

    result = compile_context(
        task(),
        {"query_service_profile": "query_service_profile(...)"},
        {"deadline_seconds_remaining": 300},
        10,
        observations=[overview],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )

    payload = context(result)
    assert payload["initial_service_profile_seeds"] == [
        {
            "service_id": "service:checkout",
            "fault_type": "container network packet corruption",
            "mechanism_family": "packet_transport",
            "member_entity_ids": ["pod:checkout-0", "pod:checkout-1"],
            "consensus_member_count": 2,
            "member_count": 2,
            "consensus_ratio": 1.0,
            "overview_score": 0.8,
        }
    ]
    assert payload["pending_service_profile_seed_count"] == 1

    checked = Observation(
        observation_id="checked-service",
        tool="query_service_profile",
        arguments={
            "service_id": "service:checkout",
            "fault_type": "container network packet corruption",
            "member_entity_ids": ["pod:checkout-0", "pod:checkout-1"],
        },
        status="no_data",
        coverage=Coverage(
            requested_range=WINDOW, completeness="unknown", basis="fixture"
        ),
        artifact_refs=[],
        attempt_id="checked-service-attempt",
        summary="no matching trace mechanism",
    )
    checked_result = compile_context(
        task(),
        {"query_service_profile": "query_service_profile(...)"},
        {"deadline_seconds_remaining": 300},
        11,
        observations=[overview, checked],
        strategy="frontier_reflection",
        max_tokens=20_000,
        topology_summary={"nodes": [], "edges": []},
    )
    checked_payload = context(checked_result)
    assert checked_payload["initial_service_profile_seeds"] == []
    assert checked_payload["pending_service_profile_seed_count"] == 0


def test_evidence_strategy_keeps_counterevidence_before_large_support() -> None:
    counter = evidence("evidence-9", "counter-source", value=False)
    support = evidence("evidence-10", "support-source", value="x" * 8_000)
    focus = hypothesis(supporting=[support.evidence_id], opposing=[counter.evidence_id])
    compact_pair = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 4},
        7,
        evidence=[counter, evidence("evidence-10", "support-source", value="")],
        hypotheses=[focus],
        max_tokens=20_000,
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 4},
        7,
        evidence=[support, counter],
        hypotheses=[focus],
        max_tokens=cast(int, compact_pair.manifest["estimated_tokens"]),
    )

    assert result.status == "ready"
    assert result.manifest["selected_refs"]["evidence_ids"] == [counter.evidence_id]
    assert support.evidence_id in result.manifest["omitted_refs"]["evidence_ids"]
    assert context(result)["opposing_evidence_index"]["count"] == 1


def test_context_preserves_compact_paired_metric_summary() -> None:
    paired = {
        "counts_complete": True,
        "scale_hint": "unknown",
        "pressure_direction": "higher_is_more_pressure",
        "hint_basis": "heuristic from metric name",
        "reference": {
            "sample_count": 3,
            "min": 1.0,
            "median": 2.0,
            "max": 3.0,
        },
        "incident": {
            "sample_count": 3,
            "min": 2.0,
            "median": 2.0,
            "max": 4.0,
            "above_reference_max_count": 2,
            "below_reference_min_count": 0,
        },
    }
    metric_evidence = evidence(
        "evidence-metric-summary",
        "metric-summary-source",
        value={"status": "success", "paired_metric_summary": paired},
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 4},
        1,
        evidence=[metric_evidence],
        max_tokens=20_000,
    )

    projected = context(result)["evidence"]
    assert projected[0]["value"]["paired_metric_summary"] == paired


def test_evidence_deduplicates_source_fingerprint_after_priority_sort() -> None:
    support = evidence("evidence-1", "same-source")
    counter = evidence("evidence-2", "same-source", value=False)

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 4},
        9,
        evidence=[support, counter],
        hypotheses=[
            hypothesis(supporting=[support.evidence_id], opposing=[counter.evidence_id])
        ],
        max_tokens=20_000,
    )

    assert result.manifest["selected_refs"]["evidence_ids"] == [counter.evidence_id]
    assert result.manifest["omitted_refs"]["evidence_ids"] == [support.evidence_id]
    assert result.manifest["omission_reasons"][f"evidence:{support.evidence_id}"] == (
        f"duplicate_source_fingerprint:{counter.evidence_id}"
    )


def test_observe_only_hides_quality_feedback_and_non_readable_refs() -> None:
    audit_hash = "b" * 64
    feedback = [
        {
            "feedback_id": "feedback-path",
            "type": "path_quality",
            "error": "SECRET_PATH_ERROR",
        },
        {
            "feedback_id": "feedback-exit",
            "type": "exit_gate",
            "error": "SECRET_EXIT_ERROR",
        },
        {
            "feedback_id": "feedback-schema",
            "type": "validation_error",
            "category": "schema",
            "summary": "SCHEMA_ERROR_VISIBLE",
            "artifact_refs": [artifact("b", "audit_only").model_dump(mode="json")],
        },
        {
            "feedback_id": "feedback-auth",
            "type": "validation_error",
            "category": "authorization",
            "summary": "AUTH_ERROR_VISIBLE",
        },
    ]
    gaps = [
        {
            "gap_id": "audit-gap",
            "reason": "SECRET_AUDIT_GAP",
            "visibility": "audit_only",
        }
    ]

    observed = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 4},
        10,
        evidence=[
            evidence(
                "evidence-1",
                "source-1",
                refs=[artifact("a"), artifact("b", "audit_only")],
            )
        ],
        hypotheses=[hypothesis()],
        gaps=gaps,
        feedback=feedback,
        validator_mode="observe_only",
        max_tokens=20_000,
    )
    message = json.dumps(observed.messages, sort_keys=True)

    assert "SCHEMA_ERROR_VISIBLE" in message
    assert "AUTH_ERROR_VISIBLE" in message
    assert "SECRET_PATH_ERROR" not in message
    assert "SECRET_EXIT_ERROR" not in message
    assert "SECRET_AUDIT_GAP" not in message
    assert audit_hash not in message
    assert context(observed)["gap_index"]["count"] == 0

    enforced = compile_context(
        task(),
        {},
        {},
        10,
        feedback=feedback,
        validator_mode="enforce",
        max_tokens=20_000,
    )
    assert "SECRET_PATH_ERROR" in json.dumps(enforced.messages)


def test_required_context_over_limit_is_explicit_and_untruncated() -> None:
    result = compile_context(
        task("必须完整保留" * 100),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 4},
        3,
        max_tokens=1,
    )

    assert result.status == "context_limit"
    assert result.manifest["context_limit_reason"] == ("required_context_exceeds_limit")
    assert result.manifest["estimated_tokens"] > 1
    assert "必须完整保留" * 100 in result.messages[1]["content"]


def test_finalize_projects_topology_before_required_context_limit() -> None:
    support = evidence("evidence-1", "support")
    open_focus = hypothesis(supporting=[support.evidence_id])
    supported_focus = open_focus.model_copy(update={"status": "supported"})
    nodes = [
        {
            "canonical_id": "database" if index == 0 else f"service-{index:02d}",
            "entity_type": "service",
            "sources": ["x" * 512],
        }
        for index in range(60)
    ]
    topology = {
        "snapshot_id": "large-finalize",
        "source": "fixture",
        "valid_range": {"start": "reference", "end": "incident"},
        "coverage": {"status": "partial"},
        "nodes": nodes,
        "edges": [
            {
                "edge_id": f"edge-{index:02d}",
                "source_id": nodes[index]["canonical_id"],
                "target_id": nodes[index + 1]["canonical_id"],
            }
            for index in range(32)
        ],
    }

    nonfinalize = compile_context(
        task(),
        {},
        {"model_calls_remaining": 8},
        3,
        evidence=[support],
        hypotheses=[open_focus],
        topology_summary=topology,
        max_tokens=12_000,
    )
    assert nonfinalize.status == "context_limit"
    assert context(nonfinalize)["topology"] == topology

    finalized = compile_context(
        task(),
        {},
        {"model_calls_remaining": 8},
        3,
        evidence=[support],
        hypotheses=[supported_focus],
        topology_summary=topology,
        max_tokens=12_000,
    )
    projected = context(finalized)
    assert finalized.status == "ready"
    assert finalized.manifest["estimated_tokens"] <= 12_000
    assert projected["convergence"]["required_next_operation"] == "finalize"
    assert projected["current_focus"]["hypothesis_id"] == supported_focus.hypothesis_id
    assert projected["topology"] == {
        "snapshot_id": "large-finalize",
        "source": "fixture",
        "valid_range": {"start": "reference", "end": "incident"},
        "coverage": {"status": "partial"},
        "node_count": 60,
        "edge_count": 32,
        "nodes_omitted": 60,
        "edges_omitted": 32,
        "projection": "finalize summary; nodes and edges omitted",
    }


def test_large_scope_and_topology_are_projected_before_the_first_model_call() -> None:
    entities = [f"service-{index:03d}" for index in range(271)]
    scoped_task = task().model_copy(
        update={
            "entry_entities": [],
            "allowed_scope": AllowedScope(
                entities=entities,
                tools=["list_entities", "summarize_anomalies"],
                entity_inventory_ref="topology.json",
            ),
        }
    )
    topology: dict[str, Any] = {
        "snapshot_id": "large",
        "nodes": [
            {"canonical_id": entity, "entity_type": "service", "sources": []}
            for entity in entities
        ],
        "edges": [
            {
                "edge_id": f"edge-{index:03d}",
                "source_id": entities[index],
                "target_id": entities[index + 1],
            }
            for index in range(142)
        ],
    }

    result = compile_context(
        scoped_task,
        {
            "list_entities": "list_entities()",
            "summarize_anomalies": "summarize_anomalies()",
        },
        {},
        1,
        topology_summary=topology,
        max_tokens=12_000,
    )
    payload = context(result)

    assert result.status == "ready"
    assert payload["task"]["allowed_scope"]["entities"] == []
    assert payload["task"]["allowed_scope"]["entities_total"] == 271
    assert payload["task"]["allowed_scope"]["entities_omitted"] == 271
    assert payload["topology"]["nodes"] == []
    assert payload["topology"]["edges"] == []
    assert payload["topology"]["node_count"] == 271
    assert payload["topology"]["edge_count"] == 142


def test_token_estimator_uses_bytes_div_three_at_limit_boundary() -> None:
    baseline = compile_context(task(), {}, {}, 3, max_tokens=20_000)
    encoded_bytes = len(
        json.dumps(
            baseline.messages,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    estimate = (encoded_bytes + 2) // 3

    assert TOKEN_ESTIMATOR_VERSION == "utf8-bytes-div-3-v2"
    assert baseline.manifest["estimated_tokens"] == estimate
    assert compile_context(task(), {}, {}, 3, max_tokens=estimate).status == "ready"
    assert (
        compile_context(task(), {}, {}, 3, max_tokens=estimate - 1).status
        == "context_limit"
    )


def test_recent_uses_remaining_budget_for_newest_complete_observations() -> None:
    newest = observation(2, "newest")
    older = observation(1, "older-" + "x" * 8_000)
    newest_only = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {},
        2,
        observations=[newest],
        strategy="recent",
        max_tokens=20_000,
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {},
        2,
        observations=[older, newest],
        strategy="recent",
        max_tokens=cast(int, newest_only.manifest["estimated_tokens"]) + 100,
    )

    interactions = context(result)["recent_interactions"]
    assert [item["observation_id"] for item in interactions] == ["observation-2"]
    assert result.manifest["omission_reasons"]["observation:observation-1"] == (
        "max_tokens"
    )


def test_recent_exposes_only_evidence_ids_derived_from_selected_observations() -> None:
    newest = observation(2, "newest")
    older = observation(1, "older-" + "x" * 8_000)
    linked = evidence("evidence-1", "source-1")
    newest_only = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {},
        2,
        observations=[newest],
        strategy="recent",
        max_tokens=20_000,
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {},
        2,
        observations=[older, newest],
        evidence=[linked],
        strategy="recent",
        max_tokens=cast(int, newest_only.manifest["estimated_tokens"]) + 200,
    )

    interactions = context(result)["recent_interactions"]
    assert [item["observation_id"] for item in interactions] == ["observation-2"]
    assert "derived_evidence_index" not in interactions[0]
    assert result.manifest["selected_refs"]["evidence_ids"] == []
    assert result.manifest["omission_reasons"]["evidence:evidence-1"] == (
        "strategy_recent_not_in_selected_interaction"
    )
    old_status = next(
        item
        for item in context(result)["recent_tool_statuses"]
        if item["observation_id"] == older.observation_id
    )
    assert old_status["arguments"] == older.arguments
    assert old_status["derived_evidence_ids"] == [linked.evidence_id]

    selected = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {},
        2,
        observations=[older],
        evidence=[linked],
        strategy="recent",
        max_tokens=20_000,
    )
    indexed = context(selected)["recent_interactions"][0]["derived_evidence_index"]
    assert indexed == [
        {
            "evidence_id": linked.evidence_id,
            "predicate": linked.predicate,
            "subject_ids": linked.subject_ids,
        }
    ]
    assert selected.manifest["selected_refs"]["evidence_ids"] == [linked.evidence_id]
    assert (
        "observation ids are not evidence ids"
        in selected.messages[0]["content"].lower()
    )


def test_recent_orders_runtime_ids_by_decision_before_operation_index() -> None:
    newer = observation(1, "newer").model_copy(
        update={"observation_id": "run:observation:decision:run:0002:0"}
    )
    older = observation(2, "older-" + "x" * 8_000).model_copy(
        update={"observation_id": "run:observation:decision:run:0001:1"}
    )
    linked = evidence("evidence-new", "source-new").model_copy(
        update={"observation_refs": [newer.observation_id]}
    )
    newest_only = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {},
        2,
        observations=[newer],
        evidence=[linked],
        strategy="recent",
        max_tokens=20_000,
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {},
        2,
        observations=[older, newer],
        evidence=[linked],
        strategy="recent",
        max_tokens=cast(int, newest_only.manifest["estimated_tokens"]) + 100,
    )

    interactions = context(result)["recent_interactions"]
    assert [item["observation_id"] for item in interactions] == [newer.observation_id]
    assert interactions[0]["derived_evidence_index"][0]["evidence_id"] == (
        linked.evidence_id
    )


def test_focus_support_precedes_new_same_subject_evidence() -> None:
    support = evidence("evidence-1", "support", value="support")
    distractor = evidence("evidence-99", "distractor", value="x" * 4_000)
    focus = hypothesis(supporting=[support.evidence_id])
    support_only = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 8},
        2,
        evidence=[support],
        hypotheses=[focus],
        max_tokens=20_000,
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 8},
        2,
        evidence=[distractor, support],
        hypotheses=[focus],
        max_tokens=cast(int, support_only.manifest["estimated_tokens"]) + 50,
    )

    assert result.manifest["selected_refs"]["evidence_ids"] == [support.evidence_id]
    assert distractor.evidence_id in result.manifest["omitted_refs"]["evidence_ids"]


def test_evidence_and_path_precede_stale_navigation_observation() -> None:
    support = evidence("evidence-1", "support")
    focus = hypothesis(supporting=[support.evidence_id])
    path = {
        "path_id": "path-1",
        "hypothesis_id": focus.hypothesis_id,
        "root_cause_entity_id": "database",
        "symptom_entity_ids": ["checkout"],
        "steps": [],
        "grounding": {
            "reference_grounding": "pass",
            "structural_grounding": "pass",
            "temporal_consistency": "consistent",
        },
    }
    baseline = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 8},
        2,
        evidence=[support],
        hypotheses=[focus],
        paths=[path],
        max_tokens=20_000,
    )
    stale_navigation = {
        "observation_id": "run:observation:decision:run:0001:0",
        "attempt_id": "attempt-nav",
        "tool": "list_metrics",
        "arguments": {"entity_id": "database"},
        "status": "success",
        "summary": json.dumps({"metric_ids": ["x" * 4_000]}),
    }
    newer_query = {
        "observation_id": "run:observation:decision:run:0002:0",
        "attempt_id": "attempt-query",
        "tool": "query_metric",
        "arguments": {"name": "latency"},
        "status": "success",
        "summary": "new result",
    }

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 8},
        2,
        observations=[stale_navigation, newer_query],
        evidence=[support],
        hypotheses=[focus],
        paths=[path],
        max_tokens=cast(int, baseline.manifest["estimated_tokens"]) + 200,
    )

    projected = context(result)
    assert [item["evidence_id"] for item in projected["evidence"]] == [
        support.evidence_id
    ]
    assert [item["path_id"] for item in projected["paths"]] == ["path-1"]
    assert (
        result.manifest["omission_reasons"][
            "observation:run:observation:decision:run:0001:0"
        ]
        == "max_tokens"
    )


def test_convergence_stage_matrix_and_closeout_guidance() -> None:
    support = evidence("evidence-1", "support")
    open_focus = hypothesis(supporting=[support.evidence_id])
    unsupported_open = hypothesis()
    inconclusive_focus = open_focus.model_copy(update={"status": "inconclusive"})
    refuted_focus = open_focus.model_copy(update={"status": "refuted"})
    supported_focus = open_focus.model_copy(update={"status": "supported"})
    path = {
        "path_id": "path-1",
        "hypothesis_id": supported_focus.hypothesis_id,
        "root_cause_entity_id": "database",
        "symptom_entity_ids": ["checkout"],
        "steps": [],
        "grounding": {
            "reference_grounding": "pass",
            "structural_grounding": "pass",
            "temporal_consistency": "consistent",
        },
    }
    ample = {
        "model_calls_remaining": 10,
        "input_tokens_remaining": 200_000,
        "output_tokens_remaining": 10_000,
        "tool_attempts_remaining": 10,
        "validator_retries_remaining": 2,
        "no_progress_rounds_remaining": 3,
        "deadline_seconds_remaining": 100,
        "closeout_required": False,
    }
    cases = [
        ((), (), (), 4, "query"),
        ((), (support,), (), 3, "update_hypothesis"),
        ((open_focus,), (support,), (), 3, "update_hypothesis"),
        ((unsupported_open,), (support,), (), 4, "query"),
        ((inconclusive_focus,), (support,), (), 4, "query_different_candidate"),
        ((refuted_focus,), (support,), (), 4, "query_different_candidate"),
        ((supported_focus,), (support,), (), 1, "finalize"),
        ((supported_focus,), (support,), (path,), 1, "finalize"),
    ]
    for hypotheses, evidence_items, paths, rounds, operation in cases:
        result = compile_context(
            task(),
            {"query_metric": "query_metric(name: str)"},
            ample,
            3,
            hypotheses=hypotheses,
            evidence=evidence_items,
            paths=paths,
            max_tokens=20_000,
        )
        convergence = context(result)["convergence"]
        assert convergence == {
            "minimum_rounds": rounds,
            "required_next_operation": operation,
            "closeout_required": False,
        }

    closeout = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {**ample, "model_calls_remaining": 2, "closeout_required": True},
        3,
        hypotheses=[open_focus],
        evidence=[support],
        max_tokens=20_000,
    )
    assert context(closeout)["convergence"] == {
        "minimum_rounds": 3,
        "required_next_operation": "finalize",
        "closeout_required": True,
    }
    assert "Runtime assigns hypothesis and path IDs" in closeout.messages[0]["content"]
    assert "convergence.required_next_operation" in closeout.messages[0]["content"]

    provider_closeout = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {
            **ample,
            "model_calls_remaining": 4,
            "provider_attempts_remaining": 3,
        },
        3,
        hypotheses=[open_focus],
        evidence=[support],
        max_tokens=20_000,
    )
    assert context(provider_closeout)["convergence"] == {
        "minimum_rounds": 3,
        "required_next_operation": "finalize",
        "closeout_required": True,
    }

    for closed_focus in (inconclusive_focus, refuted_focus):
        closing = compile_context(
            task(),
            {},
            {**ample, "closeout_required": True},
            3,
            hypotheses=[closed_focus],
            evidence=[support],
            max_tokens=20_000,
        )
        assert context(closing)["convergence"] == {
            "minimum_rounds": 4,
            "required_next_operation": "finalize",
            "closeout_required": True,
        }

    validator_exhausted = compile_context(
        task(),
        {},
        {**ample, "validator_retries_remaining": 0},
        3,
        hypotheses=[inconclusive_focus],
        evidence=[support],
        max_tokens=20_000,
    )
    assert context(validator_exhausted)["convergence"] == {
        "minimum_rounds": 4,
        "required_next_operation": "query_different_candidate",
        "closeout_required": False,
    }

    exact_input_budget = compile_context(
        task(),
        {},
        {**ample, "input_tokens_remaining": 4 * 20_000},
        3,
        hypotheses=[refuted_focus],
        evidence=[support],
        max_tokens=20_000,
    )
    assert context(exact_input_budget)["convergence"] == {
        "minimum_rounds": 4,
        "required_next_operation": "query_different_candidate",
        "closeout_required": False,
    }

    for exhausted_budget in (
        {"model_calls_remaining": 3},
        {"input_tokens_remaining": 0},
        {"tool_attempts_remaining": 0},
        {"deadline_seconds_remaining": 0},
        {"no_progress_rounds_remaining": 0},
    ):
        exhausted = compile_context(
            task(),
            {},
            {**ample, **exhausted_budget},
            3,
            hypotheses=[inconclusive_focus],
            evidence=[support],
            max_tokens=20_000,
        )
        assert context(exhausted)["convergence"] == {
            "minimum_rounds": 4,
            "required_next_operation": "finalize",
            "closeout_required": True,
        }

    invalid_path = {
        **path,
        "grounding": {
            "reference_grounding": "pass",
            "structural_grounding": "fail",
            "temporal_consistency": "consistent",
        },
    }
    invalid = compile_context(
        task(),
        {},
        ample,
        3,
        hypotheses=[supported_focus],
        evidence=[support],
        paths=[invalid_path],
        max_tokens=20_000,
    )
    assert context(invalid)["convergence"] == {
        "minimum_rounds": 1,
        "required_next_operation": "finalize",
        "closeout_required": False,
    }


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_metric_backed_open_hypothesis_requires_nonmetric_corroboration(
    validator_mode: str,
) -> None:
    tools = {
        "query_log_samples": "query_log_samples(entity_id: str)",
        "query_trace_neighbors": "query_trace_neighbors(entity_id: str)",
    }
    metric_observation = observation(1, "metric")
    metric_evidence = evidence("evidence-metric", "metric").model_copy(
        update={"observation_refs": [metric_observation.observation_id]}
    )
    log_observation = observation(2, "log").model_copy(
        update={"tool": "query_log_samples"}
    )
    log_evidence = evidence("evidence-log", "log").model_copy(
        update={"observation_refs": [log_observation.observation_id]}
    )
    metric_focus = hypothesis(supporting=[metric_evidence.evidence_id])
    ample = {"model_calls_remaining": 10, "tool_attempts_remaining": 10}

    metric_only = compile_context(
        task(),
        tools,
        ample,
        3,
        observations=[metric_observation],
        evidence=[metric_evidence],
        hypotheses=[metric_focus],
        validator_mode=cast(Any, validator_mode),
        max_tokens=20_000,
    )
    assert context(metric_only)["convergence"] == {
        "minimum_rounds": 4,
        "required_next_operation": "query_corroboration",
        "closeout_required": False,
    }

    no_data_log = compile_context(
        task(),
        tools,
        ample,
        3,
        observations=[
            metric_observation,
            log_observation.model_copy(update={"status": "no_data"}),
        ],
        evidence=[metric_evidence, log_evidence],
        hypotheses=[metric_focus],
        validator_mode=cast(Any, validator_mode),
        max_tokens=20_000,
    )
    assert context(no_data_log)["convergence"]["required_next_operation"] == (
        "query_corroboration"
    )

    other_subject_log = evidence(
        "evidence-other-log", "other-log", subject="checkout"
    ).model_copy(update={"observation_refs": [log_observation.observation_id]})
    cross_subject_focus = hypothesis(
        supporting=[metric_evidence.evidence_id, other_subject_log.evidence_id]
    )
    cross_subject = compile_context(
        task(),
        tools,
        ample,
        3,
        observations=[metric_observation, log_observation],
        evidence=[metric_evidence, other_subject_log],
        hypotheses=[cross_subject_focus],
        validator_mode=cast(Any, validator_mode),
        max_tokens=20_000,
    )
    assert context(cross_subject)["convergence"]["required_next_operation"] == (
        "query_corroboration"
    )

    corroborated = compile_context(
        task(),
        tools,
        ample,
        3,
        observations=[metric_observation, log_observation],
        evidence=[metric_evidence, log_evidence],
        hypotheses=[metric_focus],
        validator_mode=cast(Any, validator_mode),
        max_tokens=20_000,
    )
    assert context(corroborated)["convergence"] == {
        "minimum_rounds": 3,
        "required_next_operation": "update_hypothesis",
        "closeout_required": False,
    }


def test_supported_hypothesis_requires_metric_and_corroboration() -> None:
    metric_observation = observation(1, "metric")
    metric_evidence = evidence("evidence-metric", "metric")
    metric_only_focus = hypothesis(supporting=[metric_evidence.evidence_id]).model_copy(
        update={"status": "supported"}
    )

    metric_only = compile_context(
        task(),
        {"query_log_samples": "query_log_samples(entity_id: str)"},
        {"model_calls_remaining": 10, "tool_attempts_remaining": 10},
        3,
        observations=[metric_observation],
        evidence=[metric_evidence],
        hypotheses=[metric_only_focus],
        max_tokens=20_000,
    )

    assert context(metric_only)["convergence"]["required_next_operation"] == (
        "query_corroboration"
    )

    trace_observation = observation(2, "trace").model_copy(
        update={"tool": "query_trace_neighbors"}
    )
    trace_evidence = evidence("evidence-trace", "trace").model_copy(
        update={"observation_refs": [trace_observation.observation_id]}
    )
    corroborated_focus = hypothesis(
        supporting=[metric_evidence.evidence_id, trace_evidence.evidence_id]
    ).model_copy(update={"status": "supported"})
    corroborated = compile_context(
        task(),
        {
            "query_log_samples": "query_log_samples(entity_id: str)",
            "query_trace_neighbors": "query_trace_neighbors(entity_id: str)",
        },
        {"model_calls_remaining": 10, "tool_attempts_remaining": 10},
        3,
        observations=[metric_observation, trace_observation],
        evidence=[metric_evidence, trace_evidence],
        hypotheses=[corroborated_focus],
        max_tokens=20_000,
    )

    assert context(corroborated)["convergence"]["required_next_operation"] == (
        "query_corroboration"
    )

    trace_only_focus = hypothesis(supporting=[trace_evidence.evidence_id]).model_copy(
        update={"status": "supported"}
    )
    trace_only = compile_context(
        task(),
        {"query_trace_neighbors": "query_trace_neighbors(entity_id: str)"},
        {"model_calls_remaining": 10, "tool_attempts_remaining": 10},
        3,
        observations=[trace_observation],
        evidence=[trace_evidence],
        hypotheses=[trace_only_focus],
        max_tokens=20_000,
    )
    assert context(trace_only)["convergence"]["required_next_operation"] == "query"


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_unclaimed_public_metric_pair_replaces_closed_focus_without_projection_loss(
    validator_mode: str,
) -> None:
    observations, pair_evidence = public_metric_pair()
    pair_evidence = [
        item.model_copy(update={"time_uncertainty": "x" * 8_000})
        for item in pair_evidence
    ]
    closed = hypothesis().model_copy(update={"status": "refuted"})
    required = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 10},
        4,
        observations=observations,
        evidence=pair_evidence,
        hypotheses=[closed],
        validator_mode=cast(Any, validator_mode),
        max_tokens=1,
    )
    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 10},
        4,
        observations=observations,
        evidence=pair_evidence,
        hypotheses=[closed],
        validator_mode=cast(Any, validator_mode),
        max_tokens=cast(int, required.manifest["estimated_tokens"]) + 100,
    )

    projected = context(result)
    assert result.status == "ready"
    assert projected["current_focus"] is None
    assert projected["convergence"]["required_next_operation"] == "update_hypothesis"
    assert projected["pending_metric_pairs"] == [
        {
            "entity_id": "database",
            "resource_family": "memory",
            "metric_id": "memory-used",
            "fault_type": "high memory usage",
            "missing_period": None,
            "reference": {
                "evidence_id": "evidence-reference",
                "observation_id": "observation-reference",
                "value": {"summary": {"mean": 95.0}},
            },
            "incident": {
                "evidence_id": "evidence-incident",
                "observation_id": "observation-incident",
                "value": {"summary": {"mean": 98.0}},
            },
        }
    ]
    assert set(result.manifest["selected_refs"]["evidence_ids"]) == {
        "evidence-reference",
        "evidence-incident",
    }
    assert set(result.manifest["selected_refs"]["observation_ids"]) == {
        "observation-reference",
        "observation-incident",
    }
    for kind in ("evidence_ids", "observation_ids"):
        assert set(result.manifest["selected_refs"][kind]).isdisjoint(
            result.manifest["omitted_refs"][kind]
        )


def test_new_schema_pending_pair_preserves_candidate_identity() -> None:
    observations, pair_evidence = public_metric_pair()
    overview_payload = json.loads(cast(str, observations[0].summary))
    overview_payload["fault_type_candidates"][0].update(
        candidate_id="candidate-memory",
        candidate_rank=0,
        score=0.9,
        source_metric_id="memory-free",
    )
    observations[0] = observations[0].model_copy(
        update={"summary": json.dumps(overview_payload)}
    )

    result = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 10},
        4,
        observations=observations,
        evidence=pair_evidence,
        max_tokens=20_000,
    )

    pair = context(result)["candidate_evidence_summary"]["pairs"][0]
    assert pair["candidate_id"] == "candidate-memory"
    assert pair["source_metric_id"] == "memory-free"
    assert context(result)["convergence"]["required_next_operation"] == (
        "update_hypothesis"
    )


@pytest.mark.parametrize(
    ("status", "required_next_operation"),
    [
        ("supported", "finalize"),
        ("refuted", "query_different_candidate"),
        ("inconclusive", "query_different_candidate"),
    ],
)
def test_non_open_same_subject_hypothesis_disposes_public_metric_pair(
    status: str,
    required_next_operation: str,
) -> None:
    observations, pair_evidence = public_metric_pair()
    pair_ids = [item.evidence_id for item in pair_evidence]
    handled = hypothesis(
        supporting=pair_ids if status == "supported" else None,
        opposing=pair_ids if status != "supported" else None,
    ).model_copy(
        update={"status": status, "statement": "high memory usage"},
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 10},
        4,
        observations=observations,
        evidence=pair_evidence,
        hypotheses=[handled],
        max_tokens=20_000,
    )

    projected = context(result)
    assert projected["pending_metric_pairs"] == []
    assert projected["current_focus"]["hypothesis_id"] == handled.hypothesis_id
    assert (
        projected["convergence"]["required_next_operation"] == required_next_operation
    )


@pytest.mark.parametrize(
    ("status", "required_next_operation"),
    [
        ("open", "update_hypothesis"),
        ("supported", "finalize"),
    ],
)
def test_active_hypothesis_is_not_displaced_by_pending_metric_pair(
    status: str,
    required_next_operation: str,
) -> None:
    observations, pair_evidence = public_metric_pair()
    active = hypothesis(
        supporting=(
            [item.evidence_id for item in pair_evidence]
            if status == "open"
            else [pair_evidence[0].evidence_id]
        )
    ).model_copy(
        update={
            "status": status,
            "subject": "database" if status == "open" else "checkout",
        }
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 10},
        4,
        observations=observations,
        evidence=pair_evidence,
        hypotheses=[active],
        max_tokens=20_000,
    )

    projected = context(result)
    assert projected["pending_metric_pairs"] == []
    assert projected["current_focus"]["hypothesis_id"] == active.hypothesis_id
    assert (
        projected["convergence"]["required_next_operation"] == required_next_operation
    )


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
@pytest.mark.parametrize(
    ("present_period", "missing_period"),
    [("reference", "incident"), ("incident", "reference")],
)
def test_incomplete_public_metric_pair_requests_only_missing_period(
    validator_mode: str,
    present_period: str,
    missing_period: str,
) -> None:
    observations, pair_evidence = public_metric_pair()
    selected = 0 if present_period == "reference" else 1
    present_observation = observations[selected + 1]
    if present_period == "incident":
        arguments = dict(present_observation.arguments)
        arguments.pop("period")
        present_observation = present_observation.model_copy(
            update={"arguments": arguments}
        )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 10},
        4,
        observations=[observations[0], present_observation],
        evidence=[pair_evidence[selected]],
        validator_mode=cast(Any, validator_mode),
        max_tokens=20_000,
    )

    projected = context(result)
    pair = projected["pending_metric_pairs"][0]
    assert projected["current_focus"] is None
    assert projected["convergence"]["required_next_operation"] == "query_metric"
    assert pair["missing_period"] == missing_period
    assert pair[present_period]["evidence_id"] == f"evidence-{present_period}"
    assert pair[missing_period] is None
    prompt = result.messages[0]["content"]
    assert "pending_metric_pairs is non-empty, it takes priority" in prompt
    assert "Process the first listed pair before any later pair" in prompt
    assert "Each exact series must create or revise a separate hypothesis" in prompt
    assert "never combine multiple pairs in one hypothesis" in prompt
    assert "fault_type and fault_type_aliases form the" in prompt
    assert "complete allowed alias set" in prompt
    assert "Prefer the primary fault_type" in prompt
    assert "non-primary alias requires paired exact log Evidence" in prompt
    assert "strict metric fallback" in prompt
    assert "first query only that missing side" in prompt
    assert "both the reference and incident evidence_id values" in prompt
    assert "not issue query_candidate until pending_metric_pairs is empty" in prompt
    for remaining, required, closeout_required in (
        (4, "query_metric", False),
        (3, "finalize", True),
    ):
        boundary = compile_context(
            task(),
            {"query_metric": "query_metric(name: str)"},
            {"model_calls_remaining": remaining},
            4,
            observations=[observations[0], present_observation],
            evidence=[pair_evidence[selected]],
            validator_mode=cast(Any, validator_mode),
            max_tokens=20_000,
        )
        assert context(boundary)["convergence"] == {
            "minimum_rounds": 4,
            "required_next_operation": required,
            "closeout_required": closeout_required,
        }
    for closeout_budget in (
        {"model_calls_remaining": 10, "tool_attempts_remaining": 0},
        {"model_calls_remaining": 10, "closeout_required": True},
    ):
        closing = compile_context(
            task(),
            {"query_metric": "query_metric(name: str)"},
            closeout_budget,
            4,
            observations=[observations[0], observations[selected + 1]],
            evidence=[pair_evidence[selected]],
            validator_mode=cast(Any, validator_mode),
            max_tokens=20_000,
        )
        assert context(closing)["convergence"] == {
            "minimum_rounds": 4,
            "required_next_operation": "finalize",
            "closeout_required": True,
        }


@pytest.mark.parametrize("requires_paired_exact", [None, False])
def test_non_paired_fault_candidate_does_not_create_pending_pair(
    requires_paired_exact: bool | None,
) -> None:
    observations, pair_evidence = public_metric_pair(requires_paired_exact)

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 10},
        4,
        observations=observations,
        evidence=pair_evidence,
        max_tokens=20_000,
    )

    assert context(result)["pending_metric_pairs"] == []


def test_same_metric_pair_hypothesis_handles_all_fault_aliases() -> None:
    observations, pair_evidence = public_metric_pair()
    overview_payload = json.loads(cast(str, observations[0].summary))
    first_candidate = overview_payload["fault_type_candidates"][0]
    first_candidate.update(
        candidate_id="candidate-memory",
        candidate_rank=0,
        score=0.9,
        reference={"min": 90.0, "median": 95.0, "max": 97.0},
        incident={"min": 95.0, "median": 98.0, "max": 99.0},
    )
    second_candidate = {
        **first_candidate,
        "candidate_id": "candidate-memory-alias",
        "fault_type": "memory leak",
    }
    remaining_candidate = {
        **first_candidate,
        "candidate_id": "candidate-storage",
        "fault_type": "high disk space usage",
        "entity_id": "checkout",
        "metric_id": "disk-used",
        "resource_family": "storage",
    }
    overview_payload["public_fault_types"].extend(
        ["memory leak", "high disk space usage"]
    )
    overview_payload["fault_type_candidates"].extend(
        [second_candidate, remaining_candidate]
    )
    observations[0] = observations[0].model_copy(
        update={"summary": json.dumps(overview_payload)}
    )
    pair_ids = [item.evidence_id for item in pair_evidence]
    before_disposition = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
        3,
        observations=observations,
        evidence=pair_evidence,
        max_tokens=20_000,
    )
    pending_pairs = context(before_disposition)["candidate_evidence_summary"]["pairs"]
    assert len(pending_pairs) == 1
    assert pending_pairs[0]["candidate_id"] == "candidate-memory"
    assert pending_pairs[0]["fault_type_aliases"] == ["memory leak"]

    handled = hypothesis(opposing=pair_ids).model_copy(
        update={"status": "refuted", "statement": "high memory usage"}
    )

    result = compile_context(
        task(),
        {"query_metric": "query_metric(name: str)"},
        {"model_calls_remaining": 10},
        4,
        observations=observations,
        evidence=pair_evidence,
        hypotheses=[handled],
        max_tokens=20_000,
    )

    projected = context(result)
    assert projected["pending_metric_pairs"] == []
    assert projected["convergence"]["required_next_operation"] == "query_candidate"
    beam = projected["candidate_beam"]
    assert [group["fault_type"] for group in beam] == ["high disk space usage"]
    assert beam[0]["candidates"][0]["pending"] is True


def test_observe_only_path_convergence_does_not_depend_on_hidden_grounding() -> None:
    support = evidence("evidence-1", "support")
    focus = hypothesis(supporting=[support.evidence_id]).model_copy(
        update={"status": "supported"}
    )
    valid_path = {
        "path_id": "path-1",
        "hypothesis_id": focus.hypothesis_id,
        "root_cause_entity_id": "database",
        "symptom_entity_ids": ["checkout"],
        "steps": [],
        "grounding": {
            "reference_grounding": "pass",
            "structural_grounding": "pass",
            "temporal_consistency": "consistent",
        },
    }
    invalid_path = {
        **valid_path,
        "grounding": {
            "reference_grounding": "fail",
            "structural_grounding": "fail",
            "temporal_consistency": "contradicted",
        },
    }
    budget = {"model_calls_remaining": 10, "input_tokens_remaining": 200_000}

    enforced_valid = compile_context(
        task(),
        {},
        budget,
        3,
        evidence=[support],
        hypotheses=[focus],
        paths=[valid_path],
        validator_mode="enforce",
        max_tokens=20_000,
    )
    observed_valid = compile_context(
        task(),
        {},
        budget,
        3,
        evidence=[support],
        hypotheses=[focus],
        paths=[valid_path],
        validator_mode="observe_only",
        max_tokens=20_000,
    )
    observed_invalid = compile_context(
        task(),
        {},
        budget,
        3,
        evidence=[support],
        hypotheses=[focus],
        paths=[invalid_path],
        validator_mode="observe_only",
        max_tokens=20_000,
    )
    enforced_invalid = compile_context(
        task(),
        {},
        budget,
        3,
        evidence=[support],
        hypotheses=[focus],
        paths=[invalid_path],
        validator_mode="enforce",
        max_tokens=20_000,
    )

    expected_ready = {
        "minimum_rounds": 1,
        "required_next_operation": "finalize",
        "closeout_required": False,
    }
    assert context(enforced_valid)["convergence"] == expected_ready
    assert context(observed_valid)["convergence"] == expected_ready
    assert context(observed_invalid)["convergence"] == expected_ready
    assert context(enforced_invalid)["convergence"] == expected_ready
    assert "grounding" not in context(observed_invalid)["paths"][0]


def test_system_prompt_defines_public_anomaly_vocabulary_semantics() -> None:
    result = compile_context(task(), {}, {}, 1, max_tokens=20_000)
    prompt = result.messages[0]["content"]

    assert "public_fault_types" in prompt
    assert "public output vocabulary, not case labels" in prompt
    assert "final reason must exactly match one evidence-supported value" in prompt
    assert "relative_extreme_shift" in prompt
    assert "peak or trough crossed the reference envelope" in prompt
    assert "exact paired query validates it" in prompt
    assert "same entity and resource_family" in prompt
    assert "over any single overview_score" in prompt
    assert "every symptom reachable from root_cause_entity_id" in prompt
    assert "must not contain self-loop steps" in prompt
    assert "direct unresolved step" in prompt
    assert "empty basis_topology_edge_ids" in prompt
    assert "never invent or reverse a topology edge" in prompt
    assert "A supported Hypothesis may be finalized without a Path" in prompt
    assert "never invent a Path merely to finish" in prompt
    assert "proxy metrics for navigation only" in prompt
    assert "Idle, free, available, and remaining measures" in prompt
    assert "inverse pressure indicators" in prompt
    assert "incident minimum that crosses below the reference minimum" in prompt
    assert "an upward change does not support it" in prompt
    assert "downward extreme can create an open hypothesis" in prompt
    assert "requires same-subject query_log_samples corroboration" in prompt
    assert "Similar medians alone must not refute" in prompt
    assert "or mark the candidate inconclusive" in prompt
    assert "Single-core series and traffic or counter series" in prompt
    assert "remain proxy metrics for navigation only" in prompt
    assert "paired exact metric on the same entity" in prompt
    assert "directly match one public_fault_types value" in prompt
    assert (
        "used, usage, utilization, load, latency, loss, read, space, and heap" in prompt
    )
    assert (
        "Peer entities moving in the same direction cannot replace direct evidence"
        in prompt
    )
    assert "revise that hypothesis at most once to inconclusive or refuted" in prompt
    assert "never rewrite the same inconclusive or refuted hypothesis" in prompt
    assert "query a different candidate" in prompt
    assert "hypothesis_index.fault_type is non-null" in prompt
    assert "that resolved public fault type" in prompt
    assert "changing either subject or fault type" in prompt
    assert "Decision base_revision may be omitted" in prompt
    assert "copy the current context_policy.base_revision exactly" in prompt
    assert "never reuse a base_revision from an earlier decision or feedback" in prompt
    assert "a relative increase alone does not establish high pressure" in prompt
    assert "about 0.8 or higher on a known 0..1 scale" in prompt
    assert "about 80 or higher on a known 0..100 percentage scale" in prompt
    assert "when units or ranges are unknown, do not apply these thresholds" in prompt
    assert "low-level small increase must remain inconclusive" in prompt
    assert (
        "source-to-target direction may support only mechanism=request_load" in prompt
    )
    assert "target-to-source may support only mechanism=dependency_failure" in prompt
    assert "No other mechanism may cite basis_topology_edge_ids" in prompt
    assert "task.entry_entities is empty" in prompt
    assert "first response must contain exactly one operation" in prompt
    assert "summarize_anomalies with no catalog or trace query" in prompt
    assert "all metric-candidate queries must use query_candidate" in prompt
    assert "expands the first candidate batch to at most eight" in prompt
    assert "Submit the first listed" in prompt
    assert "never hand-write entity_id, metric_id, or period" in prompt
    assert "candidate metric_id is the actual exact series" in prompt
    assert "source_metric_id, when present" in prompt
    assert "only the proxy or inverse nomination origin" in prompt
    assert "interpret candidate reference and incident statistics" in prompt
    assert "as belonging to metric_id" in prompt
    assert "independently justified log or trace queries remain allowed" in prompt
    assert "Signals and direct_metric_candidates explain the overview" in prompt
    assert "only for later expansion" in prompt
    assert "candidate_beam is empty because overview candidates have no IDs" in prompt
    assert "Whenever pending_metric_pairs is non-empty" in prompt
    assert "inspect at least one other resource_family" in prompt
    assert "before scanning another entity in the same family" in prompt
    assert "reference_range.health_status is neither healthy nor verified" in prompt
    assert "only as a comparator, not as a healthy baseline" in prompt
    assert "high in both windows without a clear incident-local threshold" in prompt
    assert "treat it as persistent background" in prompt
    assert "may nominate a candidate but cannot support it alone" in prompt
    assert "multiple components show the same resource_family signal" in prompt
    assert "require independent causal evidence" in prompt
    assert "supports both the component and reason" in prompt
    assert "otherwise keep it inconclusive and advance to the next candidate" in prompt
    assert "disclose coverage and reference health" in prompt
    assert "first create an open candidate hypothesis" in prompt
    assert "Metric evidence alone may support only the strict no-log fallback" in prompt
    assert "same-subject query_log_samples" in prompt
    assert "visible candidate log_names, summarize_anomalies.log_names" in prompt
    assert "do not guess a contains value" in prompt
    assert "never use a topology source filename as log_name" in prompt
    assert "has_log_source=true" in prompt
    assert "query_trace_neighbors may guide topology traversal" in prompt
    assert "never establishes causal support" in prompt
    assert "has_log_source=false, do not issue a doomed log or trace query" in prompt
    assert "at least 20 percent direction-aligned incident envelope crossings" in prompt
    assert "adjacent observable entities" in prompt
    assert "static topology edge is not evidence" in prompt
    assert "is not evidence that an anomaly occurred" in prompt
    assert "supported hypothesis requires both" in prompt
    assert "successful or partial query_log_samples evidence" in prompt
    assert "complete paired_log_summary" in prompt
    assert "a non-empty log_name or contains selector" in prompt
    assert "broad unfiltered log Evidence is navigation only" in prompt
    assert "do not query reference and incident separately" in prompt
    assert "Use paired log_name_counts to discover a rare log_name" in prompt
    assert "total-count increase alone is not causal corroboration" in prompt
    assert "log_selector_hint.exact_log_names" in prompt
    assert "reference count is equal or higher" in prompt
    assert "count change alone neither proves nor refutes it" in prompt
    assert "One When candidate_evidence_summary" not in prompt
    assert "One query returns bounded samples" in prompt
    assert "Broad log Evidence keeps counts but omits sample text" in prompt
    assert "compact paired_metric_summary" in prompt
    assert "excursion counts only when counts_complete=true" in prompt
    assert "above_reference_max_count and below_reference_min_count" in prompt
    assert "even when medians are stable" in prompt
    assert "heuristics from the metric name" in prompt
    assert "A single crossing or unknown direction remains insufficient" in prompt
    assert "at least two direction-aligned incident envelope crossings" in prompt
    assert "must remain open until an exact same-subject log query" in prompt


def test_public_fault_vocabulary_survives_navigation_payload_omission() -> None:
    baseline = compile_context(task(), {}, {}, 1, max_tokens=20_000)
    navigation: dict[str, Any] = {
        "observation_id": "observation-1",
        "attempt_id": "attempt-1",
        "tool": "summarize_anomalies",
        "arguments": {},
        "status": "partial",
        "summary": json.dumps(
            {
                "public_fault_types": [
                    "public-type-b",
                    "public-type-a",
                    "public-type-b",
                ],
                "signals": ["x" * 10_000],
            }
        ),
    }

    result = compile_context(
        task(),
        {"summarize_anomalies": "summarize_anomalies()"},
        {},
        1,
        observations=[navigation],
        max_tokens=cast(int, baseline.manifest["estimated_tokens"]) + 400,
    )

    projected = context(result)
    assert result.status == "ready"
    assert projected["public_output_vocabulary"] == {
        "fault_types": ["public-type-b", "public-type-a"]
    }
    assert projected["recent_interactions"] == []
    assert result.manifest["omission_reasons"]["observation:observation-1"] == (
        "max_tokens"
    )


def test_candidate_beam_is_atomic_pending_and_validator_symmetric() -> None:
    candidates: list[dict[str, Any]] = [
        {
            "candidate_id": "candidate-a0",
            "candidate_rank": 0,
            "fault_type": "public-type-a",
            "entity_id": "entity-a",
            "metric_id": "metric-a0",
            "resource_family": "family-a",
            "score": 0.9,
            "reference": {"min": 0.1, "median": 0.2, "max": 0.3},
            "incident": {"min": 0.8, "median": 0.9, "max": 1.0},
            "requires_paired_exact": True,
        },
        {
            "candidate_id": "candidate-a1",
            "candidate_rank": 1,
            "fault_type": "public-type-a",
            "entity_id": "entity-b",
            "metric_id": "metric-a1",
            "source_metric_id": "metric-a1-free",
            "resource_family": "family-a",
            "score": 0.8,
            "reference": {"min": 10, "median": 20, "max": 30},
            "incident": {"min": 80, "median": 90, "max": 100},
            "requires_paired_exact": True,
        },
        {
            "candidate_id": "candidate-b0",
            "candidate_rank": 0,
            "fault_type": "public-type-b",
            "entity_id": "entity-c",
            "metric_id": "metric-b0",
            "resource_family": "family-b",
            "score": 0.7,
            "reference": {"min": 1, "median": 2, "max": 3},
            "incident": {"min": 4, "median": 5, "max": 6},
            "requires_paired_exact": True,
        },
    ]
    overview: dict[str, Any] = {
        "observation_id": "overview-1",
        "attempt_id": "attempt-overview",
        "tool": "summarize_anomalies",
        "arguments": {},
        "status": "partial",
        "summary": json.dumps(
            {
                "public_fault_types": ["public-type-a", "public-type-b"],
                "fault_type_candidates": candidates,
                "signals": ["x" * 12_000],
            }
        ),
    }
    paired = [
        {
            "observation_id": f"metric-{period}",
            "attempt_id": f"attempt-{period}",
            "tool": "query_metric",
            "arguments": {
                "entity_id": "entity-a",
                "metric_id": "metric-a0",
                "period": period,
            },
            "status": "success",
            "summary": "paired",
        }
        for period in ("reference", "incident")
    ]
    baseline = compile_context(task(), {}, {}, 1, max_tokens=20_000)
    outputs: list[list[dict[str, Any]]] = []
    comparison_outputs: list[dict[str, Any]] = []
    for mode in ("observe_only", "enforce"):
        result = compile_context(
            task(),
            {"query_candidate": "query_candidate(candidate_id: str)"},
            {},
            2,
            observations=[overview, *paired],
            validator_mode=cast(Any, mode),
            max_tokens=cast(int, baseline.manifest["estimated_tokens"]) + 2_000,
        )
        projected = context(result)
        assert result.status == "ready"
        assert all(
            item["observation_id"] != "overview-1"
            for item in projected["recent_interactions"]
        )
        assert projected["convergence"]["required_next_operation"] == "query_candidate"
        outputs.append(cast(list[dict[str, Any]], projected["candidate_beam"]))
        comparison_outputs.append(
            cast(dict[str, Any], projected["comparison_coverage"])
        )

    assert outputs[0] == outputs[1]
    assert (
        comparison_outputs[0]
        == comparison_outputs[1]
        == {
            "required_count": 2,
            "completed": 1,
            "next_candidate_ids": ["candidate-a1"],
            "satisfied": False,
        }
    )
    assert outputs[0] == [
        {
            "fault_type": "public-type-a",
            "candidates": [
                {
                    "candidate_id": "candidate-a1",
                    "candidate_rank": 1,
                    "entity_id": "entity-b",
                    "source_metric_id": "metric-a1-free",
                    "resource_family": "family-a",
                    "score": 0.8,
                    "reference": {"min": 10, "median": 20, "max": 30},
                    "incident": {"min": 80, "median": 90, "max": 100},
                    "period_status": {},
                    "pending": True,
                },
            ],
        },
        {
            "fault_type": "public-type-b",
            "candidates": [
                {
                    "candidate_id": "candidate-b0",
                    "candidate_rank": 0,
                    "entity_id": "entity-c",
                    "resource_family": "family-b",
                    "score": 0.7,
                    "reference": {"min": 1, "median": 2, "max": 3},
                    "incident": {"min": 4, "median": 5, "max": 6},
                    "period_status": {},
                    "pending": True,
                }
            ],
        },
    ]
    assert all(
        "metric_id" not in candidate
        for group in outputs[0]
        for candidate in group["candidates"]
    )


def candidate_evidence_summary_fixture(
    *, long_ids: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = [
        {
            "candidate_id": f"candidate-{index}",
            "candidate_rank": index,
            "fault_type": f"fault-{index}",
            "entity_id": f"entity-{index}",
            "metric_id": f"metric-{index}",
            "source_metric_id": f"source-metric-{index}",
            "resource_family": f"family-{index}",
            "score": 1.0 - index / 100,
            "overview_score": 1.0 - index / 100,
            "reference": {"min": 10, "median": 20, "max": 30},
            "incident": {"min": 40, "median": 50, "max": 90},
            "requires_paired_exact": True,
            "metric_role": "direct",
            "has_log_source": False,
        }
        for index in range(8)
    ]
    candidates.insert(
        1,
        {
            **candidates[0],
            "candidate_id": "candidate-0-alias",
            "fault_type": "fault-0-alias",
        },
    )
    observations: list[dict[str, Any]] = [
        {
            "observation_id": "overview-top-eight",
            "attempt_id": "attempt-overview",
            "tool": "summarize_anomalies",
            "arguments": {},
            "status": "success",
            "summary": json.dumps(
                {
                    "public_fault_types": [item["fault_type"] for item in candidates],
                    "fault_type_candidates": candidates,
                }
            ),
        }
    ]
    evidence_items: list[dict[str, Any]] = []
    prefix = "run-" + "x" * 96 if long_ids else ""
    for index in range(8):
        paired_summary = {
            "counts_complete": True,
            "scale_hint": "0..100",
            "pressure_direction": "higher_is_more_pressure",
            "reference": {"sample_count": 10, "min": 10, "median": 20, "max": 30},
            "incident": {
                "sample_count": 10,
                "min": 40,
                "median": 50,
                "max": 90,
                "above_reference_max_count": 5,
                "below_reference_min_count": 0,
            },
        }
        for period in ("reference", "incident"):
            observation_id = f"{prefix}observation-{index}-{period}"
            evidence_id = f"{prefix}evidence-{index}-{period}"
            observations.append(
                {
                    "observation_id": observation_id,
                    "attempt_id": f"attempt-{index}-{period}",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": f"entity-{index}",
                        "metric_id": f"metric-{index}",
                        "period": period,
                    },
                    "status": "success",
                }
            )
            evidence_items.append(
                {
                    "evidence_id": evidence_id,
                    "observation_refs": [observation_id],
                    "subject_ids": [f"entity-{index}"],
                    "predicate": "query_metric.result",
                    "value": {"paired_metric_summary": paired_summary},
                    "source_fingerprint": f"source-{index}-{period}",
                }
            )
    return observations, evidence_items


def test_minimum_disposition_context_advances_complete_candidate_families() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()

    def disposed(index: int, status: str) -> dict[str, Any]:
        refs = [f"evidence-{index}-reference", f"evidence-{index}-incident"]
        return {
            "hypothesis_id": f"hypothesis-{index}",
            "subject": f"entity-{index}",
            "statement": f"fault-{index}",
            "status": status,
            "supporting_evidence_refs": (
                refs if status in {"open", "supported"} else []
            ),
            "opposing_evidence_refs": (
                refs if status in {"refuted", "inconclusive"} else []
            ),
        }

    hypotheses = [disposed(0, "supported"), disposed(1, "refuted")]

    def compile_with(
        selected_hypotheses: list[dict[str, Any]], *, required: bool = False
    ) -> dict[str, Any]:
        return context(
            compile_context(
                task(),
                {"query_metric": "query_metric(name: str)"},
                {
                    "model_calls_remaining": 10,
                    "provider_attempts_remaining": 10,
                    "input_tokens_remaining": 200_000,
                    "output_tokens_remaining": 10_000,
                    "tool_attempts_remaining": 10,
                    "deadline_seconds_remaining": 100,
                },
                3,
                observations=observations,
                evidence=evidence_items,
                hypotheses=selected_hypotheses,
                max_tokens=20_000,
                candidate_disposition_required=required,
            )
        )

    blocked = compile_with(hypotheses)
    assert blocked["current_focus"]["hypothesis_id"] == "hypothesis-0"
    assert blocked["candidate_evidence_summary"]["pairs"] == []
    assert blocked["convergence"]["required_next_operation"] == "finalize"

    advancing = compile_with(hypotheses, required=True)
    assert advancing["current_focus"] is None
    assert advancing["candidate_evidence_summary"]["pairs"][0]["candidate_id"] == (
        "candidate-2"
    )
    assert advancing["convergence"]["required_next_operation"] == ("update_hypothesis")

    open_hypothesis = disposed(2, "open")
    open_first = compile_with([*hypotheses, open_hypothesis], required=True)
    assert open_first["current_focus"]["hypothesis_id"] == "hypothesis-2"
    assert open_first["convergence"]["required_next_operation"] == ("update_hypothesis")


def add_source_metric_pair(
    observations: list[dict[str, Any]],
    evidence_items: list[dict[str, Any]],
    *,
    index: int,
    paired_summary: Mapping[str, Any],
) -> None:
    for period in ("reference", "incident"):
        observation_id = f"observation-{index}-source-{period}"
        evidence_id = f"evidence-{index}-source-{period}"
        observations.append(
            {
                "observation_id": observation_id,
                "attempt_id": f"attempt-{index}-source-{period}",
                "tool": "query_metric",
                "arguments": {
                    "entity_id": f"entity-{index}",
                    "metric_id": f"source-metric-{index}",
                    "period": period,
                },
                "status": "success",
            }
        )
        evidence_items.append(
            {
                "evidence_id": evidence_id,
                "observation_refs": [observation_id],
                "subject_ids": [f"entity-{index}"],
                "predicate": "query_metric.result",
                "value": {"paired_metric_summary": dict(paired_summary)},
                "source_fingerprint": f"source-{index}-source-{period}",
            }
        )


def test_top_eight_candidate_summary_is_compact_and_validator_symmetric() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture(long_ids=True)
    overview = json.loads(cast(str, observations[0]["summary"]))
    overview["fault_type_candidates"].extend(
        [
            {
                **overview["fault_type_candidates"][-1],
                "candidate_id": "candidate-8",
                "candidate_rank": 8,
                "fault_type": "fault-8",
                "entity_id": "entity-0",
                "metric_id": "source-metric-0",
                "resource_family": "family-8",
            },
            {
                **overview["fault_type_candidates"][-1],
                "candidate_id": "candidate-9",
                "candidate_rank": 9,
                "fault_type": "fault-9",
                "entity_id": "entity-9",
                "metric_id": "metric-9",
                "source_metric_id": "source-metric-9",
                "resource_family": "family-9",
            },
        ]
    )
    overview["public_fault_types"].extend(["fault-8", "fault-9"])
    observations[0]["summary"] = json.dumps(overview)
    add_source_metric_pair(
        observations,
        evidence_items,
        index=0,
        paired_summary=cast(
            dict[str, Any],
            cast(dict[str, Any], evidence_items[0]["value"])["paired_metric_summary"],
        ),
    )
    assert candidate_schedule(observations) == [
        f"candidate-{index}" for index in range(8)
    ]
    assert [
        item["candidate_id"]
        for item in candidate_disposition_queue(observations, evidence_items)
    ] == [f"candidate-{index}" for index in range(8)]
    summaries: list[dict[str, Any]] = []
    encoded_summaries: list[str] = []
    for mode in ("observe_only", "enforce"):
        result = compile_context(
            task(),
            {"query_candidate": "query_candidate(candidate_id: str)"},
            {"model_calls_remaining": 10, "tool_attempts_remaining": 24},
            3,
            observations=observations,
            evidence=evidence_items,
            validator_mode=cast(Any, mode),
            max_tokens=12_000,
        )
        projected = context(result)
        assert result.status == "ready"
        assert result.manifest["estimated_tokens"] <= 12_000
        assert projected["pending_metric_pairs"] == []
        assert projected["candidate_beam"] == []
        assert not any(
            item["tool"] == "query_metric" for item in projected["recent_tool_statuses"]
        )
        assert projected["convergence"]["required_next_operation"] == (
            "update_hypothesis"
        )
        summary = cast(dict[str, Any], projected["candidate_evidence_summary"])
        assert summary["remaining_count"] == 7
        assert summary["remaining_candidate_ids"] == [
            f"candidate-{index}" for index in range(1, 8)
        ]
        assert len(summary["pairs"]) == 1
        assert summary["pairs"][0]["candidate_rank"] == 0
        assert summary["pairs"][0]["fault_type_aliases"] == ["fault-0-alias"]
        assert summary["pairs"][0]["metric_role"] == "direct"
        assert summary["pairs"][0]["has_log_source"] is False
        assert set(summary["pairs"][0]["evidence_ids"]) == {
            "reference",
            "incident",
        }
        summaries.append(summary)
        encoded_summaries.append(
            json.dumps(
                summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True
            )
        )

    assert summaries[0] == summaries[1]
    assert encoded_summaries[0] == encoded_summaries[1]


def test_candidate_log_selector_hint_is_fault_specific_and_token_bounded() -> None:
    names = [
        "apache_access_log",
        "download_log",
        "application_log",
        "jvm_error",
        "gc",
        "slow",
        "tcp_timeout",
    ]

    memory_hint = candidate_log_selector_hint("high memory usage", "memory", names)
    storage_hint = candidate_log_selector_hint(
        "high disk I/O read usage", "storage", names
    )
    network_hint = candidate_log_selector_hint("network latency", "network", names)
    cpu_hint = candidate_log_selector_hint("high CPU usage", "cpu", names)

    assert memory_hint == {
        "terms": ["gc", "error", "oom", "heap"],
        "exact_log_names": ["gc", "jvm_error"],
    }
    assert storage_hint is not None
    assert storage_hint["exact_log_names"] == ["slow", "jvm_error"]
    assert network_hint is not None
    assert network_hint["exact_log_names"] == ["tcp_timeout"]
    assert cpu_hint is not None
    assert cpu_hint["exact_log_names"] == []
    assert candidate_log_selector_hint("unknown fault", "other", names) is None


def test_candidate_summary_includes_exact_log_selector_hint() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()
    overview = json.loads(cast(str, observations[0]["summary"]))
    overview["fault_type_candidates"][0].update(
        fault_type="high memory usage",
        resource_family="memory",
        has_log_source=True,
        log_names=["localhost_access_log", "gc", "error"],
    )
    observations[0]["summary"] = json.dumps(overview)

    projected = context(
        compile_context(
            task(),
            {"query_candidate": "query_candidate(candidate_id: str)"},
            {"model_calls_remaining": 10},
            3,
            observations=observations,
            evidence=evidence_items,
            max_tokens=12_000,
        )
    )

    pair = next(
        item
        for item in projected["candidate_evidence_summary"]["pairs"]
        if item["candidate_id"] == "candidate-0"
    )
    assert pair["log_selector_hint"] == {
        "terms": ["gc", "error", "oom", "heap"],
        "exact_log_names": ["gc", "error"],
    }


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_candidate_context_omits_non_publishable_paths(
    validator_mode: str,
) -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()
    pair_ids = ["evidence-0-reference", "evidence-0-incident"]
    supported = hypothesis(supporting=pair_ids).model_copy(
        update={
            "subject": "entity-0",
            "statement": "fault-0",
            "status": "supported",
        }
    )
    invalid_path = {
        "path_id": "path-invalid",
        "hypothesis_id": supported.hypothesis_id,
        "root_cause_entity_id": "entity-0",
        "symptom_entity_ids": ["entity-1"],
        "steps": [],
        "grounding": {
            "reference_grounding": "pass",
            "structural_grounding": "fail",
            "temporal_consistency": "unknown",
        },
    }

    projected = context(
        compile_context(
            task(),
            {"query_candidate": "query_candidate(candidate_id: str)"},
            {"model_calls_remaining": 10},
            4,
            observations=observations,
            evidence=evidence_items,
            hypotheses=[supported],
            paths=[invalid_path],
            validator_mode=cast(Any, validator_mode),
            max_tokens=20_000,
        )
    )

    assert projected["convergence"]["required_next_operation"] == "finalize"
    assert projected["path_index"] == []
    assert projected["paths"] == []


def test_candidate_summary_merges_source_pair_without_raw_projection() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()
    paired_summary = cast(
        dict[str, Any],
        cast(dict[str, Any], evidence_items[0]["value"])["paired_metric_summary"],
    )
    add_source_metric_pair(
        observations,
        evidence_items,
        index=0,
        paired_summary=paired_summary,
    )

    result = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 10, "tool_attempts_remaining": 20},
        3,
        observations=observations,
        evidence=evidence_items,
        max_tokens=12_000,
    )

    projected = context(result)
    pair = projected["candidate_evidence_summary"]["pairs"][0]
    assert result.status == "ready"
    assert result.manifest["estimated_tokens"] <= 12_000
    assert pair["source_paired_metric_summary"] == paired_summary
    assert pair["source_evidence_ids"] == {
        "reference": "evidence-0-source-reference",
        "incident": "evidence-0-source-incident",
    }
    assert set(result.manifest["selected_refs"]["evidence_ids"]) == {
        *pair["evidence_ids"].values(),
        *pair["source_evidence_ids"].values(),
    }
    assert not any(
        item.get("evidence_id", "").startswith("evidence-0-source")
        for item in projected["evidence"]
    )


def test_candidate_priority_is_source_then_overview_then_direct_crossing() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()
    overview = json.loads(cast(str, observations[0]["summary"]))
    overview["fault_type_candidates"] = [
        item
        for item in overview["fault_type_candidates"]
        if item["candidate_id"] in {f"candidate-{index}" for index in range(4)}
    ]
    scores = [0.1, 0.9, 0.8, 0.8]
    for candidate, score in zip(overview["fault_type_candidates"], scores):
        candidate["overview_score"] = score
    observations[0]["summary"] = json.dumps(overview)
    observations[:] = [
        item
        for item in observations
        if item["tool"] == "summarize_anomalies"
        or item["arguments"]["entity_id"] in {f"entity-{index}" for index in range(4)}
    ]
    evidence_items[:] = [
        item
        for item in evidence_items
        if item["subject_ids"][0] in {f"entity-{index}" for index in range(4)}
    ]
    for index, crossings in enumerate((1, 1, 5, 2)):
        for item in evidence_items:
            if item["subject_ids"] == [f"entity-{index}"]:
                cast(
                    dict[str, Any],
                    cast(dict[str, Any], item["value"])["paired_metric_summary"][
                        "incident"
                    ],
                )["above_reference_max_count"] = crossings
    strict_source = json.loads(
        json.dumps(
            cast(dict[str, Any], evidence_items[0]["value"])["paired_metric_summary"]
        )
    )
    strict_source["incident"]["above_reference_max_count"] = 5
    add_source_metric_pair(
        observations,
        evidence_items,
        index=0,
        paired_summary=strict_source,
    )

    assert [
        item["candidate_id"]
        for item in candidate_disposition_queue(observations, evidence_items)
    ] == ["candidate-0", "candidate-1", "candidate-2", "candidate-3"]


def test_complete_candidate_disposition_is_not_replaced_by_closeout_finalize() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()

    projected = context(
        compile_context(
            task(),
            {"query_candidate": "query_candidate(candidate_id: str)"},
            {"model_calls_remaining": 1, "closeout_required": True},
            3,
            observations=observations,
            evidence=evidence_items,
            max_tokens=12_000,
        )
    )

    assert projected["convergence"] == {
        "minimum_rounds": 3,
        "required_next_operation": "update_hypothesis",
        "closeout_required": True,
    }


def test_provider_closeout_requests_finalize_with_pending_pair() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()

    projected = context(
        compile_context(
            task(),
            {"query_candidate": "query_candidate(candidate_id: str)"},
            {
                "model_calls_remaining": 8,
                "provider_attempts_remaining": 3,
            },
            3,
            observations=observations,
            evidence=evidence_items,
            max_tokens=12_000,
        )
    )

    assert projected["convergence"] == {
        "minimum_rounds": 3,
        "required_next_operation": "finalize",
        "closeout_required": True,
    }


def test_candidate_summary_uses_paired_evidence_priority() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()
    overview = json.loads(cast(str, observations[0]["summary"]))
    for candidate in overview["fault_type_candidates"]:
        if candidate["candidate_id"] in {"candidate-0", "candidate-0-alias"}:
            candidate["overview_score"] = 0.99
    observations[0]["summary"] = json.dumps(overview)
    weak = cast(dict[str, Any], evidence_items[0]["value"])["paired_metric_summary"]
    weak["incident"] = {
        **cast(dict[str, Any], weak["incident"]),
        "median": 10,
        "max": 20,
        "above_reference_max_count": 0,
    }
    cast(dict[str, Any], evidence_items[1]["value"])["paired_metric_summary"] = weak

    projected = context(
        compile_context(
            task(),
            {"query_candidate": "query_candidate(candidate_id: str)"},
            {"model_calls_remaining": 10, "tool_attempts_remaining": 24},
            3,
            observations=observations,
            evidence=evidence_items,
            max_tokens=12_000,
        )
    )

    assert projected["candidate_evidence_summary"]["pairs"][0]["candidate_id"] == (
        "candidate-1"
    )


def test_active_focus_collapses_and_closed_focus_restores_candidate_summaries() -> None:
    observations, evidence_items = candidate_evidence_summary_fixture()
    pair_ids = ["evidence-0-reference", "evidence-0-incident"]
    open_focus = hypothesis(supporting=pair_ids).model_copy(
        update={"subject": "entity-0", "statement": "fault-0"}
    )
    opened = compile_context(
        task(),
        {"query_log_samples": "query_log_samples(entity_id: str)"},
        {"model_calls_remaining": 8, "tool_attempts_remaining": 20},
        4,
        observations=observations,
        evidence=evidence_items,
        hypotheses=[open_focus],
        max_tokens=12_000,
    )
    opened_context = context(opened)
    opened_summary = opened_context["candidate_evidence_summary"]
    assert [item["candidate_id"] for item in opened_summary["pairs"]] == ["candidate-0"]
    assert opened_summary["remaining_count"] == 7
    assert opened_summary["remaining_candidate_ids"] == [
        f"candidate-{index}" for index in range(1, 8)
    ]
    assert opened_context["convergence"]["required_next_operation"] == (
        "update_hypothesis"
    )
    assert not any(
        item["tool"] == "query_metric"
        for item in opened_context["recent_tool_statuses"]
    )
    assert set(opened.manifest["selected_refs"]["evidence_ids"]) == set(pair_ids)
    assert {item["evidence_id"] for item in opened_context["evidence"]} <= set(pair_ids)

    closed = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 8, "tool_attempts_remaining": 20},
        5,
        observations=observations,
        evidence=evidence_items,
        hypotheses=[open_focus.model_copy(update={"status": "inconclusive"})],
        max_tokens=12_000,
    )
    closed_summary = context(closed)["candidate_evidence_summary"]
    assert len(closed_summary["pairs"]) == 1
    assert closed_summary["pairs"][0]["candidate_id"] == "candidate-1"
    assert closed_summary["remaining_count"] == 6


def test_candidate_investigation_and_finalize_project_topology() -> None:
    observations, pair_evidence = public_metric_pair()
    overview = json.loads(cast(str, observations[0].summary))
    overview["fault_type_candidates"][0].update(
        candidate_id="candidate-memory",
        candidate_rank=0,
        score=0.9,
        reference={"min": 90.0, "median": 95.0, "max": 97.0},
        incident={"min": 95.0, "median": 98.0, "max": 99.0},
    )
    observations[0] = observations[0].model_copy(
        update={"summary": json.dumps(overview)}
    )
    topology = {
        "snapshot_id": "snapshot",
        "coverage": {"status": "partial"},
        "nodes": [{"canonical_id": "database"}],
        "edges": [{"edge_id": "edge-1"}],
    }
    investigating = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
        2,
        observations=observations,
        evidence=pair_evidence,
        topology_summary=topology,
        max_tokens=20_000,
    )
    assert context(investigating)["topology"] == {
        "snapshot_id": "snapshot",
        "coverage": {"status": "partial"},
    }

    supported = hypothesis(
        supporting=[item.evidence_id for item in pair_evidence]
    ).model_copy(update={"status": "supported", "statement": "high memory usage"})
    finalize_stage = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
        3,
        observations=observations,
        evidence=pair_evidence,
        hypotheses=[supported],
        topology_summary=topology,
        max_tokens=20_000,
    )
    assert context(finalize_stage)["topology"] == {
        "snapshot_id": "snapshot",
        "coverage": {"status": "partial"},
        "node_count": 1,
        "edge_count": 1,
        "nodes_omitted": 1,
        "edges_omitted": 1,
        "projection": "finalize summary; nodes and edges omitted",
    }


def test_pending_pair_and_open_focus_take_priority_over_comparison_gate() -> None:
    observations, pair_evidence = public_metric_pair()
    overview = json.loads(cast(str, observations[0].summary))
    first = overview["fault_type_candidates"][0]
    first.update(
        candidate_id="candidate-memory",
        candidate_rank=0,
        score=0.9,
        reference={"min": 90.0, "median": 95.0, "max": 97.0},
        incident={"min": 95.0, "median": 98.0, "max": 99.0},
    )
    overview["fault_type_candidates"].append(
        {
            **first,
            "candidate_id": "candidate-cpu",
            "entity_id": "checkout",
            "metric_id": "cpu-used",
            "resource_family": "cpu",
        }
    )
    observations[0] = observations[0].model_copy(
        update={"summary": json.dumps(overview)}
    )
    pending = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
        2,
        observations=observations,
        evidence=pair_evidence,
        max_tokens=20_000,
    )
    assert context(pending)["comparison_coverage"]["satisfied"] is False
    assert context(pending)["convergence"]["required_next_operation"] == (
        "update_hypothesis"
    )
    assert context(pending)["candidate_beam"] == []

    open_hypothesis = hypothesis(
        supporting=[item.evidence_id for item in pair_evidence]
    ).model_copy(update={"statement": "high memory usage"})
    opened = compile_context(
        task(),
        {
            "query_candidate": "query_candidate(candidate_id: str)",
            "query_log_samples": "query_log_samples(entity_id: str)",
        },
        {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
        3,
        observations=observations,
        evidence=pair_evidence,
        hypotheses=[open_hypothesis],
        max_tokens=20_000,
    )
    assert context(opened)["comparison_coverage"]["satisfied"] is False
    assert context(opened)["convergence"]["required_next_operation"] == (
        "query_corroboration"
    )

    log_observation = observation(9, "corroborating log").model_copy(
        update={
            "observation_id": "observation-log",
            "tool": "query_log_samples",
            "arguments": {"entity_id": "database"},
        }
    )
    log_evidence = evidence("evidence-log", "source-log").model_copy(
        update={"observation_refs": ["observation-log"]}
    )
    supported = open_hypothesis.model_copy(
        update={
            "status": "supported",
            "statement": "high memory usage",
            "supporting_evidence_refs": [
                *(item.evidence_id for item in pair_evidence),
                log_evidence.evidence_id,
            ],
        }
    )
    incomplete_log = compile_context(
        task(),
        {
            "query_candidate": "query_candidate(candidate_id: str)",
            "query_log_samples": "query_log_samples(entity_id: str)",
        },
        {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
        4,
        observations=[*observations, log_observation],
        evidence=[*pair_evidence, log_evidence],
        hypotheses=[supported],
        max_tokens=20_000,
    )
    assert context(incomplete_log)["convergence"]["required_next_operation"] == (
        "query_candidate"
    )

    log_evidence = log_evidence.model_copy(
        update={
            "value": {
                "status": "partial",
                "paired_log_summary": {
                    "counts_complete": True,
                    "samples_truncated": True,
                    "reference": {
                        "matched_count": 1,
                        "log_name_counts": {"gc": 1},
                    },
                    "incident": {
                        "matched_count": 2,
                        "log_name_counts": {"access": 1, "gc": 1},
                    },
                },
            }
        }
    )
    broad_log = compile_context(
        task(),
        {
            "query_candidate": "query_candidate(candidate_id: str)",
            "query_log_samples": "query_log_samples(entity_id: str)",
        },
        {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
        4,
        observations=[*observations, log_observation],
        evidence=[*pair_evidence, log_evidence],
        hypotheses=[supported],
        max_tokens=20_000,
    )
    assert context(broad_log)["convergence"]["required_next_operation"] == (
        "query_candidate"
    )
    log_observation = log_observation.model_copy(
        update={"arguments": {"entity_id": "database", "log_name": "gc"}}
    )
    compare_before_path = compile_context(
        task(),
        {
            "query_candidate": "query_candidate(candidate_id: str)",
            "query_log_samples": "query_log_samples(entity_id: str)",
        },
        {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
        4,
        observations=[*observations, log_observation],
        evidence=[*pair_evidence, log_evidence],
        hypotheses=[supported],
        max_tokens=20_000,
    )
    assert (
        context(compare_before_path)["convergence"]["required_next_operation"]
        == "query_candidate"
    )


def test_terminal_candidate_hypotheses_are_compacted_symmetrically() -> None:
    observations, pair_evidence = public_metric_pair()
    overview = json.loads(cast(str, observations[0].summary))
    overview["public_fault_types"].append("high CPU usage")
    overview["fault_type_candidates"][0].update(
        candidate_id="candidate-memory",
        candidate_rank=0,
    )
    observations[0] = observations[0].model_copy(
        update={"summary": json.dumps(overview)}
    )
    terminal = [
        hypothesis().model_copy(
            update={
                "hypothesis_id": "hypothesis-a",
                "statement": "high memory usage",
                "status": "refuted",
            }
        ),
        hypothesis().model_copy(
            update={
                "hypothesis_id": "hypothesis-b",
                "subject": "checkout",
                "statement": "high CPU usage caused checkout latency",
                "status": "inconclusive",
            }
        ),
    ]
    summaries: list[dict[str, Any]] = []
    for mode in ("observe_only", "enforce"):
        result = compile_context(
            task(),
            {"query_metric": "query_metric(name: str)"},
            {"model_calls_remaining": 10},
            4,
            observations=observations,
            evidence=pair_evidence,
            hypotheses=terminal,
            validator_mode=cast(Any, mode),
            max_tokens=12_000,
        )

        assert result.status == "ready"
        assert context(result)["current_focus"] is None
        assert context(result)["hypothesis_index"] == []
        assert context(result)["opposing_evidence_index"]["count"] == 0
        summaries.append(context(result)["terminal_hypothesis_summary"])

    assert (
        summaries[0]
        == summaries[1]
        == {"count": 2, "status_counts": {"refuted": 1, "inconclusive": 1}}
    )


def test_active_focus_compacts_beam_and_closed_focus_restores_it() -> None:
    candidates = [
        {
            "candidate_id": f"fc1:{group:02d}{rank:02d}" + "a" * 60,
            "candidate_rank": rank,
            "fault_type": "high CPU usage" if group == 0 else f"fault-{group}",
            "entity_id": "database" if group == 0 else f"entity-{group}-{rank}",
            "metric_id": "cpu-used" if group == 0 else f"metric-{group}-{rank}",
            "resource_family": "cpu" if group == 0 else f"family-{group}",
            "score": 1.0 - (rank / 10),
            "reference": {"min": 0.1, "median": 0.2, "max": 0.3},
            "incident": {"min": 0.8, "median": 0.9, "max": 1.0},
            "requires_paired_exact": True,
        }
        for group in range(8)
        for rank in range(2)
    ]
    overview: dict[str, Any] = {
        "observation_id": "overview-large-beam",
        "attempt_id": "overview-attempt",
        "tool": "summarize_anomalies",
        "arguments": {},
        "status": "partial",
        "summary": json.dumps(
            {
                "public_fault_types": [
                    "high CPU usage",
                    *(f"fault-{group}" for group in range(1, 8)),
                ],
                "fault_type_candidates": candidates,
            }
        ),
    }
    metric_observations: list[dict[str, Any]] = [
        {
            "observation_id": f"metric-{period}",
            "attempt_id": f"metric-{period}-attempt",
            "tool": "query_metric",
            "arguments": {
                "entity_id": "database",
                "metric_id": "cpu-used",
                "period": period,
            },
            "status": "success",
        }
        for period in ("reference", "incident")
    ]
    next_candidate_reference: dict[str, Any] = {
        "observation_id": "metric-next-reference",
        "attempt_id": "metric-next-reference-attempt",
        "tool": "query_metric",
        "arguments": {
            "entity_id": "entity-1-0",
            "metric_id": "metric-1-0",
            "period": "reference",
        },
        "status": "success",
    }
    pair_evidence = [
        evidence(f"evidence-{period}", f"source-{period}").model_copy(
            update={"observation_refs": [f"metric-{period}"]}
        )
        for period in ("reference", "incident")
    ]
    log_gap: dict[str, Any] = {
        "observation_id": "run:observation:decision:run:0002:0",
        "attempt_id": "log-attempt",
        "tool": "query_log_samples",
        "arguments": {"entity_id": "database", "log_name": "error"},
        "status": "no_data",
    }
    active = hypothesis(
        supporting=[item.evidence_id for item in pair_evidence]
    ).model_copy(
        update={
            "hypothesis_id": "run:hypothesis:decision:run:0001:0",
            "subject": "database",
            "statement": "high CPU usage " + "needs corroboration " * 40,
            "status": "open",
            "open_questions": ["No corroborating log data was available."],
        }
    )
    modes: list[list[dict[str, Any]]] = []
    for mode in ("observe_only", "enforce"):
        result = compile_context(
            task(),
            {
                "query_log_samples": "query_log_samples(entity_id: str)",
                "query_trace_neighbors": "query_trace_neighbors(entity_id: str)",
            },
            {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
            5,
            observations=[
                overview,
                *metric_observations,
                next_candidate_reference,
                log_gap,
            ],
            evidence=pair_evidence,
            hypotheses=[active],
            gaps=[
                {
                    "observation_id": "run:observation:decision:run:0002:0",
                    "reason": "no log data",
                }
            ],
            validator_mode=cast(Any, mode),
            max_tokens=12_000,
        )
        projected = context(result)
        assert result.status == "ready"
        assert projected["convergence"]["required_next_operation"] == (
            "update_hypothesis"
        )
        assert projected["candidate_beam"] == []
        modes.append(cast(list[dict[str, Any]], projected["candidate_beam"]))

        closed = active.model_copy(update={"status": "refuted"})
        reopened = compile_context(
            task(),
            {"query_candidate": "query_candidate(candidate_id: str)"},
            {"model_calls_remaining": 8, "tool_attempts_remaining": 8},
            6,
            observations=[overview, *metric_observations, next_candidate_reference],
            evidence=pair_evidence,
            hypotheses=[closed],
            validator_mode=cast(Any, mode),
            max_tokens=20_000,
        )
        reopened_beam = cast(list[dict[str, Any]], context(reopened)["candidate_beam"])
        assert len(reopened_beam) == 4
        assert reopened_beam[0]["fault_type"] == "fault-1"
        assert reopened_beam[0]["candidates"][0]["period_status"] == {
            "reference": "success"
        }
        assert all(
            candidate["pending"] is True
            for group in reopened_beam
            for candidate in group["candidates"]
        )
    assert modes[0] == modes[1]


def test_legacy_overview_without_candidate_ids_keeps_empty_beam() -> None:
    overview: dict[str, Any] = {
        "observation_id": "overview-legacy",
        "attempt_id": "attempt-legacy",
        "tool": "summarize_anomalies",
        "arguments": {},
        "status": "partial",
        "summary": json.dumps(
            {
                "fault_type_candidates": [
                    {
                        "fault_type": "public-type",
                        "entity_id": "entity-a",
                        "metric_id": "metric-a",
                        "resource_family": "family-a",
                        "score": 0.5,
                        "requires_paired_exact": True,
                    }
                ]
            }
        ),
    }

    result = compile_context(
        task(), {}, {}, 1, observations=[overview], max_tokens=20_000
    )

    assert context(result)["candidate_beam"] == []


@pytest.mark.parametrize("validator_mode", ["observe_only", "enforce"])
def test_candidate_beam_no_data_is_observed_and_advances_or_finishes(
    validator_mode: str,
) -> None:
    candidate: dict[str, Any] = {
        "candidate_id": "candidate-a",
        "candidate_rank": 0,
        "fault_type": "public-type",
        "entity_id": "entity-a",
        "metric_id": "metric-a",
        "resource_family": "family-a",
        "score": 0.9,
        "requires_paired_exact": True,
    }
    overview: dict[str, Any] = {
        "observation_id": "overview",
        "attempt_id": "overview-attempt",
        "tool": "summarize_anomalies",
        "arguments": {},
        "status": "success",
        "summary": json.dumps({"fault_type_candidates": [candidate]}),
    }
    reference_no_data: dict[str, Any] = {
        "observation_id": "metric-reference",
        "attempt_id": "reference-attempt",
        "tool": "query_metric",
        "arguments": {
            "entity_id": "entity-a",
            "metric_id": "metric-a",
            "period": "reference",
        },
        "status": "no_data",
    }

    pending = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 10},
        2,
        observations=[overview, reference_no_data],
        evidence=[evidence("unrelated", "source")],
        validator_mode=cast(Any, validator_mode),
        max_tokens=20_000,
    )
    projected = context(pending)
    beam_candidate = projected["candidate_beam"][0]["candidates"][0]
    assert beam_candidate["period_status"] == {"reference": "no_data"}
    assert beam_candidate["pending"] is True
    assert projected["pending_metric_pairs"] == []
    assert projected["convergence"]["required_next_operation"] == "query_candidate"
    assert projected["comparison_coverage"] == {
        "required_count": 1,
        "completed": 0,
        "next_candidate_ids": ["candidate-a"],
        "satisfied": False,
    }

    incident_success: dict[str, Any] = {
        "observation_id": "metric-incident",
        "attempt_id": "incident-attempt",
        "tool": "query_metric",
        "arguments": {
            "entity_id": "entity-a",
            "metric_id": "metric-a",
            "period": "incident",
        },
        "status": "success",
    }
    exhausted = compile_context(
        task(),
        {"query_candidate": "query_candidate(candidate_id: str)"},
        {"model_calls_remaining": 10},
        3,
        observations=[overview, reference_no_data, incident_success],
        evidence=[evidence("unrelated", "source")],
        validator_mode=cast(Any, validator_mode),
        max_tokens=20_000,
    )
    projected = context(exhausted)
    assert projected["candidate_beam"] == []
    assert projected["pending_metric_pairs"] == []
    assert projected["comparison_coverage"] == {
        "required_count": 1,
        "completed": 1,
        "next_candidate_ids": [],
        "satisfied": True,
    }
    assert projected["convergence"] == {
        "minimum_rounds": 1,
        "required_next_operation": "finalize",
        "closeout_required": False,
    }


@pytest.mark.parametrize(
    ("overview_status", "candidate_update"),
    [
        ("error", {}),
        ("success", {"requires_paired_exact": False}),
        ("success", {"resource_family": ""}),
    ],
)
def test_candidate_beam_excludes_candidates_not_registered_by_runtime_rules(
    overview_status: str,
    candidate_update: dict[str, Any],
) -> None:
    candidate: dict[str, Any] = {
        "candidate_id": "candidate-a",
        "candidate_rank": 0,
        "fault_type": "public-type",
        "entity_id": "entity-a",
        "metric_id": "metric-a",
        "resource_family": "family-a",
        "score": 0.9,
        "requires_paired_exact": True,
        **candidate_update,
    }
    overview: dict[str, Any] = {
        "observation_id": "overview",
        "attempt_id": "overview-attempt",
        "tool": "summarize_anomalies",
        "arguments": {},
        "status": overview_status,
        "summary": json.dumps({"fault_type_candidates": [candidate]}),
    }

    result = compile_context(
        task(), {}, {}, 1, observations=[overview], max_tokens=20_000
    )

    assert context(result)["candidate_beam"] == []
