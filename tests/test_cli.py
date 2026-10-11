import csv
import fcntl
import hashlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import NoReturn, cast

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from pytest import MonkeyPatch
from typer.testing import CliRunner

import rootwalk.cli as cli_module
from rootwalk.cli import app
from rootwalk.contracts import Hypothesis, Task
from rootwalk.model import RecordedModelAdapter
from rootwalk.observability import ObservabilityConfig
from rootwalk.store import ArtifactStore, BundleExporter, RunStore
from rootwalk.version import installed_source_revision, runtime_version_manifest

runner = CliRunner()
PUBLIC_INPUT = (
    Path(__file__).resolve().parents[2]
    / "rootwalk-handoff"
    / "rootwalk-development-handoff"
    / "rootwalk-data-prep"
    / "slices"
    / "bank-row-0001"
    / "public_input"
)


def test_openrca_export_reason_uses_proof_bound_public_fault_type(
    tmp_path: Path,
) -> None:
    evidence_id = "evidence-profile"
    (tmp_path / "evidence.jsonl").write_text(
        json.dumps(
            {
                "kind": "evidence",
                "stable_key": evidence_id,
                "revision": 1,
                "payload": {
                    "evidence_id": evidence_id,
                    "subject_ids": ["market:node-1"],
                    "value": {
                        "finding_gate_summary": {
                            "requested_fault_type": "node disk space consumption",
                            "fault_type_ready": True,
                            "root_cause_score": 1.0,
                            "qualifying_metric_ids": ["node::system.disk.pct_usage"],
                        }
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    finding = Hypothesis(
        hypothesis_id="hypothesis",
        subject="market:node-1",
        statement=(
            "Node disk space consumption on node-1 is supported by a typed profile."
        ),
        status="supported",
        supporting_evidence_refs=[evidence_id],
    )

    assert cli_module._market_finding_reason(  # pyright: ignore[reportPrivateUsage]
        tmp_path, finding
    ) == ("node disk space consumption")


def test_openrca_export_reason_accepts_material_event_fault_binding(
    tmp_path: Path,
) -> None:
    evidence_id = "evidence-material-profile"
    (tmp_path / "evidence.jsonl").write_text(
        json.dumps(
            {
                "payload": {
                    "evidence_id": evidence_id,
                    "subject_ids": ["telecom:node:os_017"],
                    "value": {
                        "finding_gate_summary": {
                            "requested_fault_type": "network delay",
                            "fault_type_ready": False,
                            "material_event_ready": True,
                            "mechanism_signature": (
                                "peer_isolated_queue_congestion_event"
                            ),
                        }
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    finding = Hypothesis(
        hypothesis_id="hypothesis",
        subject="telecom:node:os_017",
        statement=(
            "Candidate root cause: network delay at telecom:node:os_017, "
            "supported by a peer-isolated queue congestion event."
        ),
        status="supported",
        supporting_evidence_refs=[evidence_id],
    )

    assert (
        cli_module._market_finding_reason(  # pyright: ignore[reportPrivateUsage]
            tmp_path, finding
        )
        == "network delay"
    )


def test_export_occurrence_datetime_uses_supporting_evidence_onset(
    tmp_path: Path,
) -> None:
    evidence_id = "evidence-onset"
    trace_id = "evidence-trace"
    (tmp_path / "evidence.jsonl").write_text(
        json.dumps(
            {
                "payload": {
                    "evidence_id": evidence_id,
                    "evidence_kind": "metric_profile",
                    "onset_interval": {
                        "start": "2020-05-27T02:22:00+08:00",
                        "end": "2020-05-27T02:24:00+08:00",
                    },
                    "value": {
                        "finding_gate_summary": {
                            "fault_type_ready": False,
                            "material_event_ready": True,
                            "boundary_aligned": False,
                            "mechanism_signature": ("peer_isolated_trace_error_event"),
                        }
                    },
                }
            }
        )
        + "\n"
        + json.dumps(
            {
                "payload": {
                    "evidence_id": trace_id,
                    "evidence_kind": "trace_edge",
                    "onset_interval": {
                        "start": "2020-05-27T02:05:00+08:00",
                        "end": "2020-05-27T02:06:00+08:00",
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    finding = Hypothesis(
        hypothesis_id="hypothesis",
        subject="telecom:service:db_007",
        statement="db close",
        status="supported",
        supporting_evidence_refs=[evidence_id, trace_id],
    )
    task = Task.model_validate_json(
        json.dumps(
            {
                "task_id": "task",
                "environment": "openrca_telecom",
                "question": "Find the incident.",
                "incident_range": {
                    "start": "2020-05-27T02:00:00+08:00",
                    "end": "2020-05-27T02:30:00+08:00",
                },
                "timezone": "Asia/Shanghai",
                "allowed_query_range": {
                    "start": "2020-05-27T01:30:00+08:00",
                    "end": "2020-05-27T02:30:00+08:00",
                },
                "entry_entities": [],
                "allowed_scope": {},
                "requested_outputs": ["root cause occurrence datetime"],
                "dataset_manifest_ref": "manifest.json",
            }
        )
    )

    assert (
        cli_module._finding_occurrence_datetime(  # pyright: ignore[reportPrivateUsage]
            tmp_path, finding, task
        )
        == "2020-05-27 02:23:00"
    )


def test_export_occurrence_prefers_identifiable_trace_operation_onset(
    tmp_path: Path,
) -> None:
    metric_id = "metric-profile"
    trace_id = "trace-operation-profile"
    (tmp_path / "evidence.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "payload": {
                            "evidence_id": metric_id,
                            "evidence_kind": "metric_profile",
                            "onset_interval": {
                                "start": "2020-05-27T02:22:00+08:00",
                                "end": "2020-05-27T02:24:00+08:00",
                            },
                            "value": {
                                "finding_gate_summary": {
                                    "fault_type_ready": True,
                                    "boundary_aligned": False,
                                    "mechanism_signature": "incident_resource_burst",
                                }
                            },
                        }
                    }
                ),
                json.dumps(
                    {
                        "payload": {
                            "evidence_id": trace_id,
                            "evidence_kind": "trace_operation_profile",
                            "onset_interval": {
                                "start": "2020-05-27T02:05:00+08:00",
                                "end": "2020-05-27T02:06:00+08:00",
                            },
                            "value": {
                                "operations": [
                                    {
                                        "operation_name": "database call",
                                        "identifiability": "identifiable",
                                    }
                                ]
                            },
                        }
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    finding = Hypothesis(
        hypothesis_id="hypothesis",
        subject="telecom:service:db_007",
        statement="CPU fault",
        status="supported",
        supporting_evidence_refs=[metric_id, trace_id],
    )
    task = Task.model_validate_json(
        json.dumps(
            {
                "task_id": "task",
                "environment": "openrca_telecom",
                "question": "Find the incident.",
                "incident_range": {
                    "start": "2020-05-27T02:00:00+08:00",
                    "end": "2020-05-27T02:30:00+08:00",
                },
                "timezone": "Asia/Shanghai",
                "allowed_query_range": {
                    "start": "2020-05-27T01:30:00+08:00",
                    "end": "2020-05-27T02:30:00+08:00",
                },
                "entry_entities": [],
                "allowed_scope": {},
                "requested_outputs": ["root cause occurrence datetime"],
                "dataset_manifest_ref": "manifest.json",
            }
        )
    )

    assert (
        cli_module._finding_occurrence_datetime(  # pyright: ignore[reportPrivateUsage]
            tmp_path, finding, task
        )
        == "2020-05-27 02:05:30"
    )


def _recorded_decisions(tmp_path: Path) -> Path:
    path = tmp_path / "decisions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "decision_summary": "No supported finding",
                    "operations": [
                        {
                            "type": "finalize",
                            "abstain_reason": "No supported finding",
                        }
                    ],
                }
            ]
        )
    )
    return path


def _benchmark_recommendation_bundle(
    tmp_path: Path,
    *,
    status: str = "completed_partial",
    binding_candidate_id: str = "candidate-1",
    policy: str = "terminal-candidate-v1",
    hypothesis_status: str = "inconclusive",
    direct_strict: object = True,
    v5_supported_subject_match: object = True,
    v5_supported_binding: bool = True,
    v5_duplicate_hypothesis: bool = False,
    v6_base_resource_family: object = "memory",
    scenario_policy: str | None = None,
    context_policy_version: str | None = None,
    omit_context_policy: bool = False,
    mixed_context_policy_version: str | None = None,
    terminal_policy_version: int | None = None,
    omit_terminal_policy: bool = False,
    manifest_code_revision: str | None = None,
) -> Path:
    task = json.loads((PUBLIC_INPUT / "task.json").read_text())
    topology = json.loads((PUBLIC_INPUT / "topology.json").read_text())
    subject = topology["nodes"][0]["canonical_id"]
    scenario = scenario_policy or policy
    hypothesis: dict[str, object] = {
        "hypothesis_id": "hypothesis-1",
        "subject": subject,
        "statement": "high CPU usage",
        "status": hypothesis_status,
        "supporting_evidence_refs": [],
        "opposing_evidence_refs": [],
        "open_questions": [],
        "created_from": [],
        "supersedes": None,
    }
    base_subject = topology["nodes"][1]["canonical_id"]
    base_hypothesis: dict[str, object] = {
        **hypothesis,
        "hypothesis_id": "hypothesis-base",
        "subject": base_subject,
    }
    supported_hypothesis: dict[str, object] = {
        **hypothesis,
        "hypothesis_id": "hypothesis-supported",
        "status": "supported",
        "supporting_evidence_refs": [
            "evidence:candidate-1:reference",
            "evidence:candidate-1:incident",
        ],
        "created_from": [
            "evidence:candidate-1:reference",
            "evidence:candidate-1:incident",
        ],
    }
    recommendation: dict[str, object] = {
        "schema_version": "1",
        "policy": policy,
        "confidence": "low",
        "usage": "openrca_export_fallback_only",
        "is_finding": False,
        "candidate_id": "candidate-1",
        "hypothesis": hypothesis,
        "selection": {
            "inconclusive": hypothesis_status == "inconclusive",
            "source_strict": True,
            "ever_open": True,
            **(
                {"terminal_queue_rank": 0}
                if policy
                in {
                    "terminal-candidate-v3",
                    "terminal-candidate-v4",
                    "terminal-candidate-v5",
                }
                else {"initial_queue_rank": 0}
            ),
            **(
                {"has_log_source": False}
                if policy
                in {
                    "terminal-candidate-v2",
                    "terminal-candidate-v3",
                    "terminal-candidate-v4",
                    "terminal-candidate-v5",
                }
                else {}
            ),
            **(
                {"direct_strict": direct_strict}
                if policy in {"terminal-candidate-v4", "terminal-candidate-v5"}
                else {}
            ),
            **(
                {
                    "supported_subject_match": v5_supported_subject_match,
                    "unique_supported_terminal_subject": subject,
                    "base_candidate_id": "candidate-base",
                    "base_hypothesis_id": "hypothesis-base",
                    "overlap_supported_hypothesis_ids": ["hypothesis-supported"],
                }
                if policy == "terminal-candidate-v5"
                else {}
            ),
        },
        "eligible_terminal_count": (
            2 if scenario in {"terminal-candidate-v5", "terminal-candidate-v6"} else 1
        ),
        "audit_ref": "event:recommendation",
    }
    report: dict[str, object] = {
        "run_id": "recommendation-run",
        "task_id": task["task_id"],
        "status": status,
        "hypothesis_ids": [],
        "path_ids": [],
        "findings": [],
        "paths": [],
        "abstain_reason": "insufficient support",
        "benchmark_recommendation": recommendation,
        "candidates": [
            {"hypothesis": hypothesis, "paths": [], "selected": False},
            *(
                [
                    {"hypothesis": base_hypothesis, "paths": [], "selected": False},
                    {
                        "hypothesis": supported_hypothesis,
                        "paths": [],
                        "selected": False,
                    },
                ]
                if scenario in {"terminal-candidate-v5", "terminal-candidate-v6"}
                else []
            ),
        ],
    }
    if policy == "terminal-candidate-v5":
        cast(dict[str, object], recommendation["selection"]).update(
            source_strict=False,
            has_log_source=False,
            direct_strict=False,
            ever_open=False,
            terminal_queue_rank=1,
        )
        if v5_duplicate_hypothesis:
            cast(list[object], report["candidates"]).append(
                {"hypothesis": hypothesis, "paths": [], "selected": False}
            )
    elif policy == "terminal-candidate-v6":
        recommendation.update(candidate_id="candidate-base", hypothesis=base_hypothesis)
        cast(dict[str, object], recommendation["selection"]).clear()
        cast(dict[str, object], recommendation["selection"]).update(
            inconclusive=True,
            source_strict=False,
            has_log_source=False,
            direct_strict=True,
            ever_open=False,
            terminal_queue_rank=0,
            known_mismatch_filter_applied=True,
            pre_family_policy="terminal-candidate-v5",
            base_candidate_id="candidate-1",
            base_hypothesis_id="hypothesis-1",
            base_statement="high CPU usage",
            base_resource_family=v6_base_resource_family,
            excluded_candidate_ids=["candidate-1"],
        )
    elif policy == "terminal-candidate-v4" and scenario == "terminal-candidate-v6":
        recommendation.update(candidate_id="candidate-base", hypothesis=base_hypothesis)
        cast(dict[str, object], recommendation["selection"]).clear()
        cast(dict[str, object], recommendation["selection"]).update(
            inconclusive=True,
            source_strict=False,
            has_log_source=False,
            direct_strict=True,
            ever_open=False,
            terminal_queue_rank=0,
        )
    with RunStore(tmp_path / "recommendation.sqlite3") as store:
        manifest: dict[str, object] = {}
        if not omit_terminal_policy:
            effective_epoch = terminal_policy_version
            if effective_epoch is None and policy == "terminal-candidate-v6":
                effective_epoch = 6
            if effective_epoch is not None:
                manifest["terminal_policy_version"] = effective_epoch
        if manifest_code_revision is not None:
            manifest["code_revision"] = manifest_code_revision
        if manifest:
            store.append("manifest", "manifest:recommendation", manifest)
        store.append("task", "task:recommendation", task)
        store.append("topology", "topology:recommendation", topology)
        if not omit_context_policy:
            effective_version = context_policy_version or (
                "m2-context-v81"
                if policy == "terminal-candidate-v6"
                else "m2-context-v80"
            )
            store.append(
                "context_manifest",
                "context:1",
                {"policy_version": effective_version},
            )
            if mixed_context_policy_version is not None:
                store.append(
                    "context_manifest",
                    "context:2",
                    {"policy_version": mixed_context_policy_version},
                )
        if scenario in {"terminal-candidate-v5", "terminal-candidate-v6"}:
            coverage = {
                "requested_range": task["allowed_query_range"],
                "observed_range": task["allowed_query_range"],
                "completeness": "complete",
                "collection_health": "healthy",
                "basis": "fixture",
            }
            candidates = [
                {
                    "candidate_id": "candidate-base",
                    "candidate_rank": 0,
                    "fault_type": "high CPU usage",
                    "entity_id": base_subject,
                    "metric_id": "metric-base",
                    "resource_family": "cpu",
                    "score": 2.0,
                    "requires_paired_exact": True,
                    "metric_role": "direct",
                    "has_log_source": False,
                    "candidate_pressure_direction": "higher_is_more_pressure",
                },
                {
                    "candidate_id": "candidate-1",
                    "candidate_rank": 1,
                    "fault_type": "high CPU usage",
                    "entity_id": subject,
                    "metric_id": "metric-supported-subject",
                    "resource_family": (
                        "memory" if scenario == "terminal-candidate-v6" else "cpu"
                    ),
                    "score": 1.0,
                    "requires_paired_exact": True,
                    "metric_role": "ambiguous",
                    "has_log_source": False,
                    "candidate_pressure_direction": "higher_is_more_pressure",
                },
            ]
            store.append(
                "observation",
                "overview:v5",
                {
                    "observation_id": "overview:v5",
                    "tool": "summarize_anomalies",
                    "arguments": {},
                    "status": "success",
                    "coverage": coverage,
                    "artifact_refs": [],
                    "attempt_id": "attempt:overview:v5",
                    "summary": json.dumps(
                        {
                            "public_fault_types": ["high CPU usage"],
                            "fault_type_candidates": candidates,
                        }
                    ),
                },
            )
            paired = {
                "counts_complete": True,
                "scale_hint": "0..100",
                "pressure_direction": "higher_is_more_pressure",
                "hint_basis": "heuristic from metric name",
                "reference": {
                    "sample_count": 2,
                    "min": 10,
                    "median": 20,
                    "max": 30,
                },
                "incident": {
                    "sample_count": 2,
                    "min": 80,
                    "median": 90,
                    "max": 95,
                    "above_reference_max_count": 2,
                    "below_reference_min_count": 0,
                },
            }
            for candidate in candidates:
                for period in ("reference", "incident"):
                    observation_id = f"{candidate['candidate_id']}:{period}"
                    evidence_id = f"evidence:{candidate['candidate_id']}:{period}"
                    store.append(
                        "observation",
                        observation_id,
                        {
                            "observation_id": observation_id,
                            "tool": "query_metric",
                            "arguments": {
                                "entity_id": candidate["entity_id"],
                                "metric_id": candidate["metric_id"],
                                "period": period,
                            },
                            "status": "success",
                            "coverage": coverage,
                            "artifact_refs": [],
                            "attempt_id": f"attempt:{observation_id}",
                        },
                    )
                    store.append(
                        "evidence",
                        evidence_id,
                        {
                            "evidence_id": evidence_id,
                            "observation_refs": [observation_id],
                            "subject_ids": [candidate["entity_id"]],
                            "predicate": "query_metric.result",
                            "value": {"paired_metric_summary": paired},
                            "observed_interval": task["allowed_query_range"],
                            "time_uncertainty": "fixture",
                            "artifact_refs": [],
                            "extractor": {
                                "kind": "deterministic",
                                "name": "fixture",
                                "version": "1",
                                "parameters_hash": "fixture",
                            },
                            "source_fingerprint": evidence_id,
                            "dataset_snapshot_id": "fixture",
                        },
                    )
            store.append(
                "investigation_event",
                "event:binding:base",
                {
                    "event_type": "candidate_disposition_applied",
                    "hypothesis_id": "hypothesis-base",
                    "candidate_id": "candidate-base",
                    "subject": base_subject,
                    "selected_fault_type": "high CPU usage",
                    "status": "inconclusive",
                },
            )
            supported_refs = [
                "evidence:candidate-1:reference",
                "evidence:candidate-1:incident",
            ]
            supported_operation = {
                "type": "update_hypothesis",
                "action": "create",
                "hypothesis_id": None,
                "subject": subject,
                "statement": "high CPU usage",
                "status": "supported",
                "supporting_evidence_refs": supported_refs,
                "opposing_evidence_refs": [],
                "open_questions": [],
            }
            if v5_supported_binding:
                store.append(
                    "investigation_event",
                    "event:binding:supported-subject",
                    {
                        "event_type": "candidate_disposition_applied",
                        "hypothesis_id": "hypothesis-supported",
                        "candidate_id": "candidate-1",
                        "subject": subject,
                        "selected_fault_type": "high CPU usage",
                        "status": "supported",
                        "reference_evidence_id": supported_refs[0],
                        "incident_evidence_id": supported_refs[1],
                        "original_operation": supported_operation,
                        "applied_operation": supported_operation,
                    },
                )
        store.append(
            "investigation_event",
            "event:binding",
            {
                "event_type": "candidate_disposition_applied",
                "hypothesis_id": "hypothesis-1",
                "candidate_id": binding_candidate_id,
                "subject": subject,
                "selected_fault_type": "high CPU usage",
                "status": hypothesis_status,
            },
        )
        store.append(
            "investigation_event",
            "event:recommendation",
            {
                "event_type": "benchmark_recommendation_selected",
                "visibility": "audit_only",
                "recommendation": recommendation,
            },
        )
        store.append("report", "report:recommendation", report)
        return BundleExporter(store, ArtifactStore(tmp_path / "cas")).export(
            tmp_path / "recommendation-bundle"
        )


def _frontier_recommendation_bundle(
    tmp_path: Path,
    *,
    impact: str = "supports",
    add_equal_peer: bool = False,
    resolved_anomaly: bool = False,
    material_only: bool = False,
) -> Path:
    task = json.loads((PUBLIC_INPUT / "task.json").read_text())
    topology = json.loads((PUBLIC_INPUT / "topology.json").read_text())
    subject = topology["nodes"][0]["canonical_id"]
    peer_subject = topology["nodes"][1]["canonical_id"]
    hypothesis = {
        "hypothesis_id": "hypothesis-frontier",
        "subject": subject,
        "statement": "typed CPU pressure remains the best recommendation",
        "status": "open",
        "supporting_evidence_refs": ["evidence-frontier"],
        "opposing_evidence_refs": [],
        "open_questions": ["one material event remains unresolved"],
        "created_from": ["evidence-frontier"],
        "supersedes": None,
    }
    anomaly: dict[str, object] = {
        "anomaly_id": "anomaly-unresolved",
        "subject_id": peer_subject,
        "signal_family": "latency",
        "signal_ids": ["latency"],
        "mechanism_candidates": ["dependency latency"],
        "source_observation_ids": ["observation-frontier"],
        "proposal_ids": ["proposal-frontier"],
        "materiality_basis": "fixture",
        "priority_facts": {},
    }
    mechanism_signature = (
        "peer_isolated_trace_error_event"
        if material_only
        else "sustained_high_saturation"
    )
    selection = {
        "current_best_hypothesis_id": "hypothesis-frontier",
        "supporting_evidence_id": "evidence-frontier",
        "public_fault_type": "high CPU usage",
        "mechanism_signature": mechanism_signature,
        "mechanism_rank": 3 if material_only else 2,
        "root_cause_score": 0.5 if material_only else 1.0,
        "unresolved_anomaly_ids": ([] if resolved_anomaly else ["anomaly-unresolved"]),
    }
    recommendation = {
        "schema_version": "1",
        "policy": "frontier-supported-v1",
        "confidence": "low",
        "usage": "openrca_export_fallback_only",
        "is_finding": False,
        "candidate_id": "frontier:hypothesis-frontier",
        "hypothesis": hypothesis,
        "selection": selection,
        "audit_ref": "event:frontier-recommendation",
    }
    report: dict[str, object] = {
        "run_id": "frontier-recommendation-run",
        "task_id": task["task_id"],
        "status": "completed_partial",
        "outcome_type": "recommendation",
        "hypothesis_ids": [],
        "path_ids": [],
        "findings": [],
        "paths": [],
        "abstain_reason": "one material event remains unresolved",
        "current_best_hypothesis_id": "hypothesis-frontier",
        "benchmark_recommendation": recommendation,
        "candidates": [{"hypothesis": hypothesis, "paths": [], "selected": False}],
        "material_anomalies": [anomaly],
        "anomaly_dispositions": (
            [
                {
                    "disposition_id": "disposition-resolved",
                    "anomaly_id": "anomaly-unresolved",
                    "disposition": "background",
                    "hypothesis_id": None,
                    "evidence_refs": ["evidence-frontier"],
                    "explanation": "the only anomaly is resolved",
                    "observation_digest": "b" * 64,
                }
            ]
            if resolved_anomaly
            else []
        ),
        "anomaly_coverage": {
            "total": 1,
            "resolved": int(resolved_anomaly),
            "unresolved": int(not resolved_anomaly),
        },
    }
    coverage = {
        "requested_range": task["allowed_query_range"],
        "observed_range": task["allowed_query_range"],
        "completeness": "complete",
        "collection_health": "healthy",
        "basis": "fixture",
    }
    evidence = {
        "evidence_id": "evidence-frontier",
        "observation_refs": ["observation-frontier"],
        "subject_ids": [subject],
        "predicate": "query_metric_profile.result",
        "value": {
            "finding_gate_summary": {
                "requested_fault_type": "high CPU usage",
                "fault_type_ready": not material_only,
                "material_event_ready": material_only,
                "material_event_basis": (
                    "peer_isolated_trace_error_event" if material_only else None
                ),
                "boundary_aligned": False,
                "mechanism_signature": mechanism_signature,
                "root_cause_score": 0.5 if material_only else 1.0,
            }
        },
        "observed_interval": task["incident_range"],
        "onset_interval": task["incident_range"],
        "time_uncertainty": "fixture",
        "artifact_refs": [],
        "extractor": {
            "kind": "deterministic",
            "name": "fixture",
            "version": "1",
            "parameters_hash": "fixture",
        },
        "source_fingerprint": "frontier-source",
        "dataset_snapshot_id": "frontier-fixture",
        "evidence_kind": "metric_profile",
        "mechanism_dimension": "resource_profile",
        "independence_group": "frontier-source",
        "semantic_quality": "typed",
    }
    with RunStore(tmp_path / "frontier-recommendation.sqlite3") as store:
        store.append(
            "manifest",
            "manifest:frontier-recommendation",
            {"terminal_policy_version": 26},
        )
        store.append("task", "task:frontier-recommendation", task)
        store.append("topology", "topology:frontier-recommendation", topology)
        store.append(
            "context_manifest",
            "context:frontier-recommendation",
            {"policy_version": "frontier-reflection-v16"},
        )
        store.append(
            "observation",
            "observation-frontier",
            {
                "observation_id": "observation-frontier",
                "tool": "query_metric_profile",
                "arguments": {},
                "status": "success",
                "coverage": coverage,
                "artifact_refs": [],
                "attempt_id": "attempt-frontier",
            },
        )
        store.append("evidence", "evidence-frontier", evidence)
        if add_equal_peer:
            store.append(
                "evidence",
                "evidence-peer",
                {
                    **evidence,
                    "evidence_id": "evidence-peer",
                    "subject_ids": [peer_subject],
                    "source_fingerprint": "peer-source",
                },
            )
        store.append(
            "evidence_assessment",
            "assessment-frontier",
            {
                "assessment_id": "assessment-frontier",
                "evidence_id": "evidence-frontier",
                "hypothesis_id": "hypothesis-frontier",
                "impact": impact,
                "reason_code": "typed_profile",
                "explanation": "typed profile explicitly supports the recommendation",
                "observation_digest": "a" * 64,
            },
        )
        store.append("hypothesis", "hypothesis-frontier", hypothesis)
        store.append("material_anomaly", "anomaly-unresolved", anomaly)
        if resolved_anomaly:
            store.append(
                "anomaly_disposition",
                "disposition-resolved",
                cast(list[dict[str, object]], report["anomaly_dispositions"])[0],
            )
        store.append(
            "investigation_event",
            "event:frontier-recommendation",
            {
                "event_type": "benchmark_recommendation_selected",
                "visibility": "audit_only",
                "recommendation": recommendation,
            },
        )
        store.append("report", "report:frontier-recommendation", report)
        return BundleExporter(store, ArtifactStore(tmp_path / "frontier-cas")).export(
            tmp_path / "frontier-recommendation-bundle"
        )


def _occurrence_recommendation_bundle(
    tmp_path: Path, *, tamper_midpoint: bool = False
) -> Path:
    task = json.loads((PUBLIC_INPUT / "task.json").read_text())
    task["requested_outputs"] = ["root cause occurrence datetime"]
    topology = json.loads((PUBLIC_INPUT / "topology.json").read_text())
    subject = topology["nodes"][0]["canonical_id"]
    onset = task["incident_range"]
    start = datetime.fromisoformat(onset["start"])
    end = datetime.fromisoformat(onset["end"])
    midpoint = start + (end - start) / 2
    evidence_refs = ["occurrence-evidence-1", "occurrence-evidence-2"]
    recommendation = {
        "schema_version": "1",
        "policy": "source-backed-occurrence-v1",
        "confidence": "low",
        "is_finding": False,
        "subject_id": subject,
        "evidence_refs": evidence_refs,
        "onset_interval": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "end_exclusive": True,
        },
        "occurrence_datetime": (
            (midpoint + timedelta(seconds=1)).isoformat()
            if tamper_midpoint
            else midpoint.isoformat()
        ),
        "audit_ref": "event:occurrence-recommendation",
    }
    report = {
        "run_id": "occurrence-recommendation-run",
        "task_id": task["task_id"],
        "status": "completed_partial",
        "outcome_type": "occurrence_recommendation",
        "hypothesis_ids": [],
        "path_ids": [],
        "occurrence_evidence_refs": evidence_refs,
        "findings": [],
        "paths": [],
        "abstain_reason": None,
        "occurrence_recommendation": recommendation,
    }
    coverage = {
        "requested_range": task["allowed_query_range"],
        "observed_range": task["allowed_query_range"],
        "completeness": "complete",
        "collection_health": "healthy",
        "basis": "fixture",
    }
    with RunStore(tmp_path / "occurrence-recommendation.sqlite3") as store:
        store.append(
            "manifest",
            "manifest:occurrence-recommendation",
            {"terminal_policy_version": 51},
        )
        store.append("task", "task:occurrence-recommendation", task)
        store.append("topology", "topology:occurrence-recommendation", topology)
        store.append(
            "context_manifest",
            "context:occurrence-recommendation",
            {"policy_version": "frontier-reflection-v44"},
        )
        store.append(
            "observation",
            "occurrence-impact-observation",
            {
                "observation_id": "occurrence-impact-observation",
                "tool": "query_impact_profile",
                "arguments": {},
                "status": "no_data",
                "coverage": coverage,
                "artifact_refs": [],
                "attempt_id": "occurrence-impact-attempt",
                "extraction_gap": "fixture has no impact source",
            },
        )
        for index, evidence_id in enumerate(evidence_refs):
            observation_id = f"occurrence-observation-{index}"
            store.append(
                "observation",
                observation_id,
                {
                    "observation_id": observation_id,
                    "tool": "query_metric_profile",
                    "arguments": {},
                    "status": "success",
                    "coverage": coverage,
                    "artifact_refs": [],
                    "attempt_id": f"occurrence-attempt-{index}",
                },
            )
            store.append(
                "evidence",
                evidence_id,
                {
                    "evidence_id": evidence_id,
                    "observation_refs": [observation_id],
                    "subject_ids": [subject],
                    "predicate": "source_backed_onset",
                    "value": {"candidate": index},
                    "observed_interval": task["incident_range"],
                    "onset_interval": onset,
                    "time_uncertainty": "fixture",
                    "artifact_refs": [],
                    "extractor": {
                        "kind": "deterministic",
                        "name": "occurrence-fixture",
                        "version": "1",
                        "parameters_hash": "fixture",
                    },
                    "source_fingerprint": f"occurrence-source-{index}",
                    "dataset_snapshot_id": "occurrence-fixture",
                    "evidence_kind": "metric_profile",
                    "mechanism_dimension": "resource_profile",
                    "independence_group": f"occurrence-source-{index}",
                    "semantic_quality": "typed",
                },
            )
        store.append(
            "investigation_event",
            "event:occurrence-recommendation",
            {
                "event_type": "occurrence_recommendation_selected",
                "visibility": "audit_only",
                "recommendation": recommendation,
            },
        )
        store.append("report", "report:occurrence-recommendation", report)
        return BundleExporter(store, ArtifactStore(tmp_path / "occurrence-cas")).export(
            tmp_path / "occurrence-recommendation-bundle"
        )


def _initial_top1_recommendation_bundle(
    tmp_path: Path,
    *,
    weak_initial_top1_candidate_id: str = "candidate-top1",
    context_policy_version: str | None = None,
    terminal_policy_version: int | None = None,
) -> Path:
    task = json.loads((PUBLIC_INPUT / "task.json").read_text())
    topology = json.loads((PUBLIC_INPUT / "topology.json").read_text())
    subject = topology["nodes"][0]["canonical_id"]
    recommendation: dict[str, object] = {
        "schema_version": "1",
        "policy": "initial-top1-v1",
        "confidence": "low",
        "usage": "openrca_export_fallback_only",
        "is_finding": False,
        "candidate_id": "candidate-top1",
        "subject": subject,
        "statement": "high CPU usage",
        "selection": {
            "initial_queue_rank": 0,
            "source_strict": False,
            "trigger": "weak_finding_rejected",
        },
        "rejected_candidate_id": "candidate-selected",
        "weak_finding_rejected_ref": "event:weak",
        "audit_ref": "event:recommendation",
    }
    report: dict[str, object] = {
        "run_id": "initial-top1-run",
        "task_id": task["task_id"],
        "status": "completed_partial",
        "hypothesis_ids": [],
        "path_ids": [],
        "findings": [],
        "paths": [],
        "abstain_reason": (
            "selected Finding did not satisfy the pre-registered strong-Finding policy"
        ),
        "benchmark_recommendation": recommendation,
    }
    with RunStore(tmp_path / "initial-top1.sqlite3") as store:
        if terminal_policy_version is not None:
            store.append(
                "manifest",
                "manifest:initial-top1",
                {"terminal_policy_version": terminal_policy_version},
            )
        store.append("task", "task:initial-top1", task)
        store.append("topology", "topology:initial-top1", topology)
        if context_policy_version is not None:
            store.append(
                "context_manifest",
                "context:initial-top1",
                {"policy_version": context_policy_version},
            )
        store.append(
            "investigation_event",
            "event:weak",
            {
                "event_type": "weak_finding_rejected",
                "visibility": "audit_only",
                "policy": "strong-source-no-higher-peer-v1",
                "hypothesis_id": "hypothesis-selected",
                "candidate_id": "candidate-selected",
                "source_strict": True,
                "candidate_rank": 1,
                "higher_rank_same_fault_candidate_ids": ["candidate-top1"],
                "initial_top1_candidate_id": weak_initial_top1_candidate_id,
                "initial_top1_subject": subject,
                "initial_top1_fault_type": "high CPU usage",
                "initial_top1_source_strict": False,
                "original_finalize_operation": {
                    "type": "finalize",
                    "hypothesis_ids": ["hypothesis-selected"],
                    "path_ids": [],
                    "unresolved_questions": [],
                    "abstain_reason": None,
                },
            },
        )
        store.append(
            "investigation_event",
            "event:recommendation",
            {
                "event_type": "benchmark_recommendation_selected",
                "visibility": "audit_only",
                "recommendation": recommendation,
            },
        )
        store.append("report", "report:initial-top1", report)
        return BundleExporter(
            store, ArtifactStore(tmp_path / "initial-top1-cas")
        ).export(tmp_path / "initial-top1-bundle")


def _rejected_supported_recommendation_bundle(
    tmp_path: Path,
    *,
    log_name: str = "request_failed",
    contains_only: bool = False,
    corruption: str | None = None,
) -> Path:
    task = json.loads((PUBLIC_INPUT / "task.json").read_text())
    topology = json.loads((PUBLIC_INPUT / "topology.json").read_text())
    subject = topology["nodes"][0]["canonical_id"]
    pair_refs = ["reference", "reference" if corruption == "same-pair" else "incident"]
    hypothesis: dict[str, object] = {
        "hypothesis_id": "hypothesis-supported",
        "subject": subject,
        "statement": "high CPU usage",
        "status": "supported",
        "supporting_evidence_refs": [*pair_refs, "log-evidence"],
        "opposing_evidence_refs": [],
        "open_questions": [],
        "created_from": [*pair_refs, "log-evidence"],
        "supersedes": None,
    }
    recommendation: dict[str, object] = {
        "schema_version": "1",
        "policy": "rejected-supported-filtered-log-v1",
        "confidence": "low",
        "usage": "openrca_export_fallback_only",
        "is_finding": False,
        "candidate_id": "candidate-supported",
        "hypothesis": hypothesis,
        "filtered_log_evidence_ids": ["log-evidence"],
        "weak_finding_rejected_ref": "event:weak",
        "disposition_ref": "event:disposition",
        "audit_ref": "event:recommendation",
    }
    supporting_refs = [*pair_refs, "log-evidence"]
    operation = {
        "type": "update_hypothesis",
        "action": "create",
        "hypothesis_id": None,
        "subject": subject,
        "statement": "high CPU usage",
        "status": "supported",
        "supporting_evidence_refs": supporting_refs,
        "opposing_evidence_refs": [],
        "open_questions": [],
    }
    coverage = {
        "requested_range": task["allowed_query_range"],
        "observed_range": task["allowed_query_range"],
        "completeness": "complete",
        "collection_health": "healthy",
        "basis": "fixture",
    }
    report: dict[str, object] = {
        "run_id": "rejected-supported-run",
        "task_id": task["task_id"],
        "status": "completed_partial",
        "hypothesis_ids": [],
        "path_ids": [],
        "findings": [],
        "paths": [],
        "abstain_reason": (
            "selected Finding did not satisfy the pre-registered strong-Finding policy"
        ),
        "benchmark_recommendation": recommendation,
        "candidates": [{"hypothesis": hypothesis, "paths": [], "selected": False}],
    }
    with RunStore(tmp_path / "rejected-supported.sqlite3") as store:
        store.append("task", "task:rejected-supported", task)
        store.append("topology", "topology:rejected-supported", topology)
        store.append(
            "observation",
            "overview-observation",
            {
                "observation_id": "overview-observation",
                "tool": "summarize_anomalies",
                "arguments": {},
                "status": "success",
                "coverage": coverage,
                "artifact_refs": [],
                "attempt_id": "overview-attempt",
                "summary": json.dumps(
                    {
                        "public_fault_types": ["high CPU usage"],
                        "fault_type_candidates": [
                            {
                                "candidate_id": "candidate-supported",
                                "candidate_rank": 1,
                                "fault_type": "high CPU usage",
                                "entity_id": subject,
                                "metric_id": "cpu_metric",
                                "resource_family": "cpu",
                                "score": 1.0,
                                "requires_paired_exact": True,
                                "metric_role": "direct",
                                "has_log_source": True,
                                "log_names": ["request_failed"],
                            }
                        ],
                    }
                ),
            },
        )
        paired_metric_summary = {
            "counts_complete": True,
            "scale_hint": "0..100",
            "pressure_direction": "higher_is_more_pressure",
            "hint_basis": "heuristic from metric name",
            "reference": {"sample_count": 2, "min": 10, "median": 20, "max": 30},
            "incident": {
                "sample_count": 2,
                "min": 80,
                "median": 90,
                "max": 95,
                "above_reference_max_count": 2,
                "below_reference_min_count": 0,
            },
        }
        for period in ("reference", "incident"):
            store.append(
                "observation",
                f"metric-observation-{period}",
                {
                    "observation_id": f"metric-observation-{period}",
                    "tool": "query_metric",
                    "arguments": {
                        "entity_id": subject,
                        "metric_id": (
                            "other_metric"
                            if corruption == "wrong-metric" and period == "incident"
                            else "cpu_metric"
                        ),
                        "period": period,
                    },
                    "status": "success",
                    "coverage": coverage,
                    "artifact_refs": [],
                    "attempt_id": f"metric-attempt-{period}",
                },
            )
            store.append(
                "evidence",
                period,
                {
                    "evidence_id": period,
                    "observation_refs": [f"metric-observation-{period}"],
                    "subject_ids": [subject],
                    "predicate": "query_metric.result",
                    "value": {"paired_metric_summary": paired_metric_summary},
                    "observed_interval": task["allowed_query_range"],
                    "time_uncertainty": "fixture",
                    "artifact_refs": [],
                    "extractor": {
                        "kind": "deterministic",
                        "name": "fixture",
                        "version": "1",
                        "parameters_hash": "fixture",
                    },
                    "source_fingerprint": f"fixture-{period}",
                    "dataset_snapshot_id": "fixture",
                },
            )
        store.append(
            "observation",
            "log-observation",
            {
                "observation_id": "log-observation",
                "tool": "query_log_samples",
                "arguments": (
                    {"entity_id": subject, "contains": "failure"}
                    if contains_only
                    else {"entity_id": subject, "log_name": log_name}
                ),
                "status": "success",
                "coverage": coverage,
                "artifact_refs": [],
                "attempt_id": "log-attempt",
            },
        )
        store.append(
            "evidence",
            "log-evidence",
            {
                "evidence_id": "log-evidence",
                "observation_refs": ["log-observation"],
                "subject_ids": [subject],
                "predicate": "query_log_samples.result",
                "value": {
                    "paired_log_summary": {
                        "counts_complete": True,
                        "samples_truncated": (
                            "false" if corruption == "incomplete-log" else False
                        ),
                        "reference": {"matched_count": 0, "log_name_counts": {}},
                        "incident": {
                            "matched_count": 1,
                            "log_name_counts": {log_name: 1},
                        },
                    }
                },
                "observed_interval": task["allowed_query_range"],
                "time_uncertainty": "fixture",
                "artifact_refs": [],
                "extractor": {
                    "kind": "deterministic",
                    "name": "fixture",
                    "version": "1",
                    "parameters_hash": "fixture",
                },
                "source_fingerprint": "fixture-log",
                "dataset_snapshot_id": "fixture",
            },
        )
        if corruption == "legacy-first-binding":
            store.append(
                "investigation_event",
                "event:old-binding",
                {
                    "event_type": "candidate_disposition_applied",
                    "hypothesis_id": "hypothesis-supported",
                    "candidate_id": "candidate-supported",
                    "subject": subject,
                    "selected_fault_type": "high CPU usage",
                    "status": "open",
                },
            )
        store.append(
            "investigation_event",
            "event:disposition",
            {
                "event_type": "candidate_disposition_applied",
                "hypothesis_id": "hypothesis-supported",
                "candidate_id": "candidate-supported",
                "subject": subject,
                "metric_id": "cpu_metric",
                "selected_fault_type": "high CPU usage",
                "status": "supported",
                "allowed_fault_types": (
                    None if corruption == "old-disposition" else ["high CPU usage"]
                ),
                "resource_family": "cpu",
                "log_selector_hint": {
                    "terms": ["cpu", "load"],
                    "exact_log_names": [],
                },
                "reference_evidence_id": "reference",
                "incident_evidence_id": pair_refs[1],
                "log_evidence_ids": ["log-evidence"],
                "original_operation": operation,
                "applied_operation": operation,
            },
        )
        store.append(
            "investigation_event",
            "event:weak",
            {
                "event_type": "weak_finding_rejected",
                "visibility": "audit_only",
                "policy": (
                    "tampered"
                    if corruption == "weak-policy"
                    else "strong-source-no-higher-peer-v1"
                ),
                "abstain_reason": (
                    "selected Finding did not satisfy the pre-registered "
                    "strong-Finding policy"
                ),
                "hypothesis_id": "hypothesis-supported",
                "candidate_id": "candidate-supported",
                "source_strict": False,
                "candidate_rank": 1,
                "higher_rank_same_fault_candidate_ids": [],
                "initial_top1_candidate_id": "candidate-supported",
                "initial_top1_subject": subject,
                "initial_top1_fault_type": "high CPU usage",
                "initial_top1_source_strict": False,
                "original_finalize_operation": {
                    "type": "finalize",
                    "hypothesis_ids": ["hypothesis-supported"],
                    "path_ids": [],
                    "unresolved_questions": [],
                    "abstain_reason": None,
                },
            },
        )
        if corruption == "duplicate-evidence":
            duplicate = store.get("reference")
            assert duplicate is not None
            store.append("evidence", "duplicate-reference", duplicate.payload)
        if corruption == "duplicate-observation":
            duplicate = store.get("metric-observation-reference")
            assert duplicate is not None
            store.append("observation", "duplicate-observation", duplicate.payload)
        if corruption == "v4-pool":
            terminal = {
                **hypothesis,
                "hypothesis_id": "hypothesis-inconclusive",
                "status": "inconclusive",
            }
            cast(list[object], report["candidates"]).append(
                {"hypothesis": terminal, "paths": [], "selected": False}
            )
            store.append(
                "investigation_event",
                "event:terminal-binding",
                {
                    "event_type": "candidate_disposition_applied",
                    "hypothesis_id": "hypothesis-inconclusive",
                    "candidate_id": "candidate-supported",
                    "subject": subject,
                    "selected_fault_type": "high CPU usage",
                    "status": "inconclusive",
                },
            )
        store.append(
            "investigation_event",
            "event:recommendation",
            {
                "event_type": "benchmark_recommendation_selected",
                "visibility": "audit_only",
                "recommendation": recommendation,
            },
        )
        store.append("report", "report:rejected-supported", report)
        return BundleExporter(store, ArtifactStore(tmp_path / "p0-cas")).export(
            tmp_path / "rejected-supported-bundle"
        )


def test_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "doctor" in result.stdout
    assert "diff" in result.stdout
    assert "evaluate" in result.stdout
    assert "export" in result.stdout
    assert "probe-model" in result.stdout
    assert "replay" in result.stdout
    assert "run" in result.stdout
    run_help = runner.invoke(app, ["run", "--help"])
    assert run_help.exit_code == 0
    assert "--otel" in run_help.stdout
    assert "--otel-endpoint" in run_help.stdout
    assert "--otel-capture-" in run_help.stdout
    assert "--query-sandbox" in run_help.stdout
    export_help = runner.invoke(app, ["export", "--help"])
    assert export_help.exit_code == 0
    assert "--self-contained" in export_help.stdout
    assert "--external-root" in export_help.stdout


def test_export_can_explicitly_embed_external_artifacts(tmp_path: Path) -> None:
    content = b"portable telemetry\n"
    content_hash = hashlib.sha256(content).hexdigest()
    with RunStore(tmp_path / "run.sqlite3") as store:
        store.append("task", "task:1", {"task_id": "task-1"})
        store.append("topology", "topology:1", {"snapshot_id": "topology-1"})
        store.append(
            "observation",
            "observation:1",
            {"artifact_refs": [{"content_hash": content_hash}]},
        )
        store.append("report", "report:1", {"status": "partial"})
        store.append(
            "artifact",
            f"artifact:{content_hash}",
            {
                "content_hash": content_hash,
                "storage": "external",
                "uri": f"bank://fixture/{content_hash}",
            },
        )
        bundle = BundleExporter(store, ArtifactStore(tmp_path / "cas")).export(
            tmp_path / "bundle"
        )
    external = tmp_path / "public_input"
    artifact = external / "telemetry/source.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(content)
    (external / "manifest.json").write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "path": "telemetry/source.csv",
                        "bytes": len(content),
                        "sha256": content_hash,
                    }
                ]
            }
        )
    )

    output = tmp_path / "portable"
    result = runner.invoke(
        app,
        [
            "export",
            str(bundle),
            "--self-contained",
            "--external-root",
            str(external),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.stdout)
    assert summary["external_artifacts_embedded"] == 1
    assert (output / "artifacts" / content_hash).read_bytes() == content
    replay_result = runner.invoke(app, ["replay", str(output)])
    assert replay_result.exit_code == 0, replay_result.output
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["bundle_size_bytes"] += 1
    manifest_path.write_text(json.dumps(manifest))
    rejected = runner.invoke(app, ["replay", str(output)])
    assert rejected.exit_code == 1
    assert "bundle size mismatch" in rejected.stderr


def test_diff_same_recorded_bundle() -> None:
    bundle = Path(__file__).resolve().parents[1] / "examples" / "bank-recorded-run"

    result = runner.invoke(app, ["diff", str(bundle), str(bundle)])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["same"] is True


def test_public_synthetic_example_preserves_provenance() -> None:
    bundle = Path(__file__).resolve().parents[1] / "examples" / "bank-recorded-run"
    manifest = json.loads((bundle / "manifest.json").read_text())
    # The committed example is generated from synthetic telemetry and keeps the
    # complete version manifest that produced it.
    current_fields = set(runtime_version_manifest())
    assert all(manifest[name] for name in current_fields)
    assert len(manifest["code_revision"]) == 64
    contexts = [
        json.loads(line)["payload"]
        for line in (bundle / "context_manifests.jsonl").read_text().splitlines()
    ]
    assert contexts and all(item["token_estimator_version"] for item in contexts)

    result = runner.invoke(app, ["replay", str(bundle)])
    assert result.exit_code == 0, result.output
    assert json.loads((bundle / "manifest.json").read_text()) == manifest


def test_doctor_reports_presence_without_exposing_config(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "config.toml"
    secret = "secret-token-value"
    config.write_text(f"token = '{secret}'\n")
    monkeypatch.setenv("ROOTWALK_CONFIG", str(config))

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Python: 3.11." in result.stdout
    assert "Config: present" in result.stdout
    assert secret not in result.stdout
    assert str(config) not in result.stdout


def test_probe_model_reports_missing_key_without_leaking_options(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)

    result = runner.invoke(
        app,
        [
            "probe-model",
            "--model",
            "synthetic-endpoint",
            "--base-url",
            "https://ark.invalid/api/v3",
        ],
    )

    assert result.exit_code == 1
    assert "ARK_API_KEY" in result.stderr
    assert "synthetic-endpoint" not in result.stderr
    assert "https://ark.invalid" not in result.stderr


def test_live_model_configuration_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="--model-provider must be ark"):
        cli_module._live_model_configuration(  # pyright: ignore[reportPrivateUsage]
            "unknown", None, None
        )


def test_run_reports_missing_key_without_creating_a_bundle(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    bundle = tmp_path / "bundle"

    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "missing-key",
            "--model",
            "synthetic-endpoint",
            "--base-url",
            "https://ark.invalid/api/v3",
        ],
    )

    assert result.exit_code == 1
    assert "ARK_API_KEY" in result.stderr
    assert "synthetic-endpoint" not in result.stderr
    assert "https://ark.invalid" not in result.stderr
    assert not bundle.exists()


def test_live_run_persists_the_model_adapter_timeout(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    adapter_arguments: dict[str, object] = {}

    def fake_adapter(**kwargs: object) -> RecordedModelAdapter:
        adapter_arguments.update(kwargs)
        return RecordedModelAdapter(
            [
                {
                    "decision_summary": "No supported finding",
                    "operations": [
                        {
                            "type": "finalize",
                            "abstain_reason": "No supported finding",
                        }
                    ],
                }
            ]
        )

    monkeypatch.setattr("rootwalk.cli.ModelAdapter", fake_adapter)
    bundle = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "live-timeout",
            "--model",
            "synthetic-endpoint",
            "--base-url",
            "https://ark.invalid/api/v3",
            "--model-timeout-s",
            "7.5",
            "--max-model-calls",
            "18",
            "--max-input-tokens",
            "216000",
            "--max-tool-attempts",
            "40",
        ],
    )

    assert result.exit_code == 0, result.output
    assert adapter_arguments["timeout_s"] == 7.5
    environment = json.loads((bundle / "manifest.json").read_text())
    assert environment["model_timeout_seconds"] == 7.5
    assert environment["max_model_calls"] == 18
    assert environment["max_input_tokens"] == 216_000
    assert environment["max_tool_attempts"] == 40


def test_live_run_persists_ark_provider(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    adapter_arguments: dict[str, object] = {}

    def fake_adapter(**kwargs: object) -> RecordedModelAdapter:
        adapter_arguments.update(kwargs)
        return RecordedModelAdapter(
            [
                {
                    "decision_summary": "No supported finding",
                    "operations": [
                        {
                            "type": "finalize",
                            "abstain_reason": "No supported finding",
                        }
                    ],
                }
            ]
        )

    monkeypatch.setattr("rootwalk.cli.ModelAdapter", fake_adapter)
    bundle = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "ark-provider",
            "--model",
            "ark-model",
            "--base-url",
            "https://ark.invalid/api/v3",
        ],
    )

    assert result.exit_code == 0, result.output
    assert adapter_arguments["base_url"] == "https://ark.invalid/api/v3"
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["model_provider"] == "ark"
    assert manifest["model_id"] == "ark-model"
    assert manifest["model_endpoint"] == "https://ark.invalid/api/v3"


@pytest.mark.parametrize(
    ("option", "value", "message"),
    [
        (
            "--context-strategy",
            "everything",
            "--context-strategy must be recent, evidence, or frontier_reflection",
        ),
        (
            "--validator-mode",
            "disabled",
            "--validator-mode must be observe_only or enforce",
        ),
        (
            "--query-sandbox",
            "maybe",
            "--query-sandbox must be auto, required, or off",
        ),
    ],
)
def test_run_rejects_invalid_mechanism_controls_before_runtime(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    option: str,
    value: str,
    message: str,
) -> None:
    def unexpected_runtime(*_args: object, **_kwargs: object) -> NoReturn:
        raise AssertionError("invalid controls must not create a runtime")

    monkeypatch.setattr("rootwalk.cli.RootWalkRuntime", unexpected_runtime)
    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(tmp_path / "bundle"),
            "--run-id",
            "invalid-control",
            "--recorded-decisions",
            str(_recorded_decisions(tmp_path)),
            option,
            value,
        ],
    )

    assert result.exit_code == 1
    assert message in result.stderr
    assert "invalid controls must not create" not in result.output


def test_run_persists_mechanism_controls_in_compiled_context(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "mechanism-controls",
            "--recorded-decisions",
            str(_recorded_decisions(tmp_path)),
            "--context-strategy",
            "recent",
            "--validator-mode",
            "observe_only",
            "--query-sandbox",
            "off",
            "--max-model-calls",
            "24",
            "--max-output-tokens",
            "1000",
            "--max-input-tokens",
            "216000",
            "--max-tool-attempts",
            "40",
        ],
    )

    assert result.exit_code == 0, result.output
    environment = json.loads((bundle / "manifest.json").read_text())
    assert environment["model_timeout_seconds"] is None
    assert environment["context_strategy"] == "recent"
    assert environment["validator_mode"] == "observe_only"
    assert environment["query_sandbox_requested"] == "off"
    assert environment["query_sandbox_effective"] == "spawn"
    assert environment["query_sandbox_bwrap_version"] is None
    assert environment["query_sandbox_bwrap_executable_sha256"] is None
    assert environment["max_model_calls"] == 24
    assert environment["max_output_tokens_per_call"] == 1_000
    assert environment["max_output_tokens"] == 24_000
    assert environment["max_input_tokens"] == 216_000
    assert environment["max_tool_attempts"] == 40
    report = json.loads((bundle / "report.json").read_text())
    assert report["query_sandbox_requested"] == "off"
    assert report["query_sandbox_effective"] == "spawn"
    assert report["query_sandbox_bwrap_version"] is None
    assert report["query_sandbox_bwrap_executable_sha256"] is None
    saved = json.loads(
        (bundle / "context_manifests.jsonl").read_text().splitlines()[0]
    )["payload"]
    assert saved["strategy"] == "recent"
    assert saved["validator_mode"] == "observe_only"
    messages = json.loads((bundle / "artifacts" / saved["model_input_ref"]).read_text())
    context = json.loads(messages[1]["content"])
    assert context["context_policy"] == {
        "strategy": "recent",
        "validator_mode": "observe_only",
        "base_revision": 3,
    }


def test_run_observability_is_disabled_by_default(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected_provider(config: ObservabilityConfig) -> NoReturn:
        del config
        raise AssertionError("disabled observability must not create a provider")

    monkeypatch.setenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "https://collector.invalid/v1/traces?token=secret",
    )
    monkeypatch.setenv("ROOTWALK_OTEL_CAPTURE_CONTENT", "true")
    monkeypatch.setenv("ROOTWALK_OTEL_SAMPLE_RATIO", "0.25")
    monkeypatch.setattr("rootwalk.cli.create_tracer_provider", unexpected_provider)
    bundle = tmp_path / "bundle"
    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "otel-disabled",
            "--recorded-decisions",
            str(_recorded_decisions(tmp_path)),
        ],
    )

    assert result.exit_code == 0, result.output
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["observability"] == {
        "enabled": False,
        "capture_content": False,
        "sample_ratio": 1.0,
        "endpoint_sha256": None,
    }


class TrackingProvider:
    def __init__(self) -> None:
        self.exporter = InMemorySpanExporter()
        self.provider = TracerProvider()
        self.provider.add_span_processor(SimpleSpanProcessor(self.exporter))
        self.force_flush_timeout: int | None = None
        self.shutdown_called = False

    def get_tracer(self, name: str) -> trace.Tracer:
        return self.provider.get_tracer(name)

    def force_flush(self, timeout_millis: int) -> bool:
        self.force_flush_timeout = timeout_millis
        return self.provider.force_flush(timeout_millis)

    def shutdown(self) -> None:
        self.shutdown_called = True
        self.provider.shutdown()


def test_run_enables_otel_from_env_and_flushes(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    endpoint = "https://collector.invalid/v1/traces"
    provider = TrackingProvider()
    configs: list[ObservabilityConfig] = []

    def create_provider(config: ObservabilityConfig) -> TrackingProvider:
        configs.append(config)
        return provider

    monkeypatch.setenv("ROOTWALK_OTEL_ENABLED", "true")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", endpoint)
    monkeypatch.setattr("rootwalk.cli.create_tracer_provider", create_provider)
    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(tmp_path / "bundle"),
            "--run-id",
            "otel-env",
            "--recorded-decisions",
            str(_recorded_decisions(tmp_path)),
        ],
    )

    assert result.exit_code == 0, result.output
    assert configs == [
        ObservabilityConfig(enabled=True, capture_content=False, endpoint=endpoint)
    ]
    assert provider.force_flush_timeout == 10_000
    assert provider.shutdown_called
    assert any(
        span.name == "rootwalk.run" for span in provider.exporter.get_finished_spans()
    )


def test_run_otel_automatically_links_across_cli_reopen(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    providers: list[TrackingProvider] = []

    def create_provider(config: ObservabilityConfig) -> TrackingProvider:
        assert config.enabled
        provider = TrackingProvider()
        providers.append(provider)
        return provider

    monkeypatch.setattr("rootwalk.cli.create_tracer_provider", create_provider)
    decisions = _recorded_decisions(tmp_path)
    bundle = tmp_path / "bundle"
    arguments = [
        "run",
        "--data-root",
        str(PUBLIC_INPUT),
        "--output",
        str(bundle),
        "--run-id",
        "otel-cli-resume",
        "--recorded-decisions",
        str(decisions),
        "--otel",
    ]

    first_result = runner.invoke(app, arguments)
    assert first_result.exit_code == 0, first_result.output
    bundle.rename(tmp_path / "first-bundle")
    second_result = runner.invoke(app, arguments)
    assert second_result.exit_code == 0, second_result.output

    first = next(
        span
        for span in providers[0].exporter.get_finished_spans()
        if span.name == "rootwalk.run"
    )
    second = next(
        span
        for span in providers[1].exporter.get_finished_spans()
        if span.name == "rootwalk.run"
    )
    assert first.context is not None and second.context is not None
    assert first.context.trace_id != second.context.trace_id
    assert len(second.links) == 1
    assert second.links[0].context.span_id == first.context.span_id


class FailingCleanupProvider:
    def __init__(self, secret: str) -> None:
        self.secret = secret
        self.force_flush_called = False
        self.shutdown_called = False

    def get_tracer(self, name: str) -> trace.Tracer:
        del name
        return trace.NoOpTracer()

    def force_flush(self, timeout_millis: int) -> bool:
        del timeout_millis
        self.force_flush_called = True
        raise RuntimeError(self.secret)

    def shutdown(self) -> None:
        self.shutdown_called = True
        raise RuntimeError(self.secret)


def test_run_otel_cleanup_failure_is_silent(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    secret = "never-print-otel-header"
    provider = FailingCleanupProvider(secret)
    configs: list[ObservabilityConfig] = []

    def create_provider(config: ObservabilityConfig) -> FailingCleanupProvider:
        configs.append(config)
        return provider

    monkeypatch.setenv("OTEL_EXPORTER_OTLP_HEADERS", secret)
    monkeypatch.setattr("rootwalk.cli.create_tracer_provider", create_provider)
    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(tmp_path / "bundle"),
            "--run-id",
            "otel-failed-cleanup",
            "--recorded-decisions",
            str(_recorded_decisions(tmp_path)),
            "--otel",
            "--otel-endpoint",
            "https://collector.invalid/v1/traces",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "completed_partial"
    assert configs[0].enabled and not configs[0].capture_content
    assert provider.force_flush_called and provider.shutdown_called
    assert secret not in result.output


def test_recorded_run_replay_and_empty_openrca_export(tmp_path: Path) -> None:
    decisions = _recorded_decisions(tmp_path)
    bundle = tmp_path / "bundle"
    work = tmp_path / ".bundle.work"

    run_result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "cli-recorded",
            "--recorded-decisions",
            str(decisions),
        ],
    )

    assert run_result.exit_code == 0, run_result.output
    assert (work / "run.sqlite3").is_file()
    assert (work / "artifacts").is_dir()
    assert (work / "run.lock").is_file()

    completed_result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "cli-recorded",
            "--recorded-decisions",
            str(decisions),
        ],
    )
    assert completed_result.exit_code == 1
    assert "use rootwalk replay" in completed_result.stderr

    replay_result = runner.invoke(app, ["replay", str(bundle)])
    assert replay_result.exit_code == 0, replay_result.output
    assert json.loads(replay_result.stdout) == {
        "finding_count": 0,
        "model_calls": 1,
        "path_count": 0,
        "run_id": "cli-recorded",
        "status": "completed_partial",
    }

    export_result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])
    assert export_result.exit_code == 0, export_result.output
    assert list(csv.DictReader(io.StringIO(export_result.stdout))) == [
        {"row_id": "1", "prediction": "{}"}
    ]

    bundle.rename(tmp_path / "first-bundle")
    resumed_result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "cli-recorded",
            "--recorded-decisions",
            str(decisions),
        ],
    )
    assert resumed_result.exit_code == 0, resumed_result.output
    assert json.loads(resumed_result.stdout)["model_calls"] == 1
    assert (bundle / "manifest.json").is_file()

    (bundle / "report.json").write_text("{}\n")
    invalid_replay = runner.invoke(app, ["replay", str(bundle)])
    assert invalid_replay.exit_code == 1
    assert "hash mismatch" in invalid_replay.stderr

    bundle.rename(tmp_path / "tampered-bundle")
    wrong_run = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "different-run",
            "--recorded-decisions",
            str(decisions),
        ],
    )
    assert wrong_run.exit_code == 1
    assert "different run_id" in wrong_run.stderr


@pytest.mark.parametrize(
    "policy",
    [
        "terminal-candidate-v1",
        "terminal-candidate-v2",
        "terminal-candidate-v3",
        "terminal-candidate-v4",
        "terminal-candidate-v5",
        "terminal-candidate-v6",
    ],
)
def test_openrca_export_uses_audited_low_confidence_benchmark_recommendation(
    tmp_path: Path,
    policy: str,
) -> None:
    bundle = _benchmark_recommendation_bundle(tmp_path, policy=policy)

    replay = runner.invoke(app, ["replay", str(bundle)])
    exported = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.stdout)["finding_count"] == 0
    assert exported.exit_code == 0, exported.output
    assert json.loads(exported.stderr)["event"] == (
        "low_confidence_benchmark_recommendation_exported"
    )
    assert json.loads(exported.stderr)["policy"] == policy
    assert list(csv.DictReader(io.StringIO(exported.stdout))) == [
        {
            "row_id": "1",
            "prediction": json.dumps(
                {
                    "1": {
                        "root cause component": (
                            "Tomcat02"
                            if policy == "terminal-candidate-v6"
                            else "Tomcat01"
                        ),
                        "root cause reason": "high CPU usage",
                    }
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    ]


def test_openrca_export_uses_audited_frontier_recommendation(
    tmp_path: Path,
) -> None:
    bundle = _frontier_recommendation_bundle(tmp_path)

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 0, result.output
    row = next(csv.DictReader(io.StringIO(result.stdout)))
    prediction = json.loads(row["prediction"])
    assert prediction["1"]["root cause reason"] == "high CPU usage"
    assert "frontier-supported-v1" in result.stderr


def test_openrca_export_accepts_material_event_recommendation(tmp_path: Path) -> None:
    bundle = _frontier_recommendation_bundle(tmp_path, material_only=True)

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 0, result.output
    assert "frontier-supported-v1" in result.stderr


def test_openrca_export_uses_audited_occurrence_only_recommendation(
    tmp_path: Path,
) -> None:
    bundle = _occurrence_recommendation_bundle(tmp_path)

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 0, result.output
    row = next(csv.DictReader(io.StringIO(result.stdout)))
    prediction = json.loads(row["prediction"])
    assert set(prediction["1"]) == {"root cause occurrence datetime"}
    assert "source-backed-occurrence-v1" in result.stderr


def test_openrca_export_rejects_tampered_occurrence_midpoint(tmp_path: Path) -> None:
    bundle = _occurrence_recommendation_bundle(tmp_path, tamper_midpoint=True)

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "onset audit is invalid" in result.stderr


@pytest.mark.parametrize(
    ("impact", "add_equal_peer", "message"),
    [
        ("refutes", False, "lacks an explicit supports assessment"),
        ("supports", True, "equal or stronger peer"),
    ],
)
def test_openrca_export_rejects_invalid_frontier_recommendation(
    tmp_path: Path,
    impact: str,
    add_equal_peer: bool,
    message: str,
) -> None:
    bundle = _frontier_recommendation_bundle(
        tmp_path, impact=impact, add_equal_peer=add_equal_peer
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert message in result.stderr


def test_openrca_export_rejects_frontier_recommendation_without_unresolved_anomaly(
    tmp_path: Path,
) -> None:
    bundle = _frontier_recommendation_bundle(tmp_path, resolved_anomaly=True)

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "unresolved audit is invalid" in result.stderr


@pytest.mark.parametrize("policy", ["terminal-candidate-v2", "terminal-candidate-v4"])
def test_openrca_export_rejects_refuted_terminal_recommendation(
    tmp_path: Path, policy: str
) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy=policy,
        hypothesis_status="refuted",
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "invalid fields" in result.stderr


def test_openrca_export_rejects_tampered_v4_direct_strict(tmp_path: Path) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path, policy="terminal-candidate-v4", direct_strict="true"
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "invalid fields" in result.stderr


def test_openrca_export_rejects_tampered_v5_supported_subject_match(
    tmp_path: Path,
) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy="terminal-candidate-v5",
        v5_supported_subject_match=False,
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "invalid fields" in result.stderr or "not reproducible" in result.stderr


def test_openrca_export_rejects_v5_supported_candidate_without_raw_binding(
    tmp_path: Path,
) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy="terminal-candidate-v5",
        v5_supported_binding=False,
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "selection is not reproducible" in result.stderr


def test_openrca_export_rejects_duplicate_v5_hypothesis_id(tmp_path: Path) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy="terminal-candidate-v5",
        v5_duplicate_hypothesis=True,
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "report candidate is invalid" in result.stderr


def test_openrca_export_rejects_tampered_v6_family_audit(tmp_path: Path) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy="terminal-candidate-v6",
        v6_base_resource_family="cpu",
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "not reproducible" in result.stderr


@pytest.mark.parametrize(
    "reported_policy", ["terminal-candidate-v5", "terminal-candidate-v4"]
)
def test_openrca_export_rejects_v81_terminal_policy_downgrade(
    tmp_path: Path, reported_policy: str
) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy=reported_policy,
        scenario_policy="terminal-candidate-v6",
        context_policy_version="m2-context-v81",
        terminal_policy_version=6,
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "terminal policy downgrade" in result.stderr


def test_openrca_export_rejects_v81_initial_top1_downgrade(tmp_path: Path) -> None:
    bundle = _initial_top1_recommendation_bundle(
        tmp_path,
        context_policy_version="m2-context-v81",
        terminal_policy_version=6,
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "cannot use initial-top1-v1" in result.stderr


def test_openrca_export_rejects_mixed_or_missing_v6_policy_provenance(
    tmp_path: Path,
) -> None:
    (tmp_path / "mixed").mkdir()
    (tmp_path / "missing").mkdir()
    mixed = _benchmark_recommendation_bundle(
        tmp_path / "mixed",
        policy="terminal-candidate-v6",
        context_policy_version="m2-context-v81",
        mixed_context_policy_version="m2-context-v80",
    )
    missing = _benchmark_recommendation_bundle(
        tmp_path / "missing",
        policy="terminal-candidate-v6",
        omit_context_policy=True,
        omit_terminal_policy=True,
    )

    mixed_result = runner.invoke(app, ["export", str(mixed), "--format", "openrca"])
    missing_result = runner.invoke(app, ["export", str(missing), "--format", "openrca"])

    assert mixed_result.exit_code == 1
    assert "must be consistent" in mixed_result.stderr
    assert missing_result.exit_code == 1
    assert "invalid fields" in missing_result.stderr


def test_openrca_export_rejects_v80_self_reported_v6(tmp_path: Path) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy="terminal-candidate-v6",
        context_policy_version="m2-context-v80",
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "requires context policy v81" in result.stderr


def test_openrca_export_rejects_future_terminal_policy_epoch(tmp_path: Path) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy="terminal-candidate-v6",
        context_policy_version="m2-context-v81",
        terminal_policy_version=61,
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "newer than this exporter" in result.stderr


def test_openrca_export_keeps_v80_v5_without_applying_fam2(tmp_path: Path) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy="terminal-candidate-v5",
        scenario_policy="terminal-candidate-v6",
        context_policy_version="m2-context-v80",
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.stderr)["policy"] == "terminal-candidate-v5"


@pytest.mark.parametrize("terminal_policy_version", [None, 5])
def test_openrca_export_current_code_requires_current_terminal_epoch(
    tmp_path: Path, terminal_policy_version: int | None
) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path,
        policy="terminal-candidate-v4",
        terminal_policy_version=terminal_policy_version,
        omit_terminal_policy=terminal_policy_version is None,
        manifest_code_revision=installed_source_revision(),
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "current code revision requires" in result.stderr


@pytest.mark.parametrize("policy", ["terminal-candidate-v1", "terminal-candidate-v4"])
def test_openrca_export_rejects_unbound_benchmark_recommendation(
    tmp_path: Path, policy: str
) -> None:
    bundle = _benchmark_recommendation_bundle(
        tmp_path, binding_candidate_id="different-candidate", policy=policy
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "does not match its audit and disposition records" in result.stderr


def test_openrca_export_uses_rejected_supported_filtered_log_policy(
    tmp_path: Path,
) -> None:
    bundle = _rejected_supported_recommendation_bundle(tmp_path)

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 0, result.output
    marker = json.loads(result.stderr)
    assert marker["policy"] == "rejected-supported-filtered-log-v1"
    assert marker["hypothesis_id"] == "hypothesis-supported"
    assert list(csv.DictReader(io.StringIO(result.stdout)))[0]["prediction"] != "{}"


def test_openrca_export_rejects_contains_only_log_recommendation(
    tmp_path: Path,
) -> None:
    bundle = _rejected_supported_recommendation_bundle(tmp_path, contains_only=True)

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "filtered log evidence is invalid" in result.stderr


@pytest.mark.parametrize(
    "corruption",
    [
        "same-pair",
        "wrong-metric",
        "incomplete-log",
        "duplicate-evidence",
        "duplicate-observation",
        "legacy-first-binding",
        "old-disposition",
        "weak-policy",
        "v4-pool",
    ],
)
def test_openrca_export_rejects_invalid_p0_provenance(
    tmp_path: Path, corruption: str
) -> None:
    bundle = _rejected_supported_recommendation_bundle(tmp_path, corruption=corruption)

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1


def test_investigation_event_reader_rejects_duplicate_stable_keys(
    tmp_path: Path,
) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    record = {
        "kind": "investigation_event",
        "stable_key": "event:duplicate",
        "payload": {"event_type": "fixture"},
    }
    (bundle / "investigation_events.jsonl").write_text(
        "\n".join([json.dumps(record), json.dumps(record)]) + "\n"
    )

    with pytest.raises(ValueError, match="duplicate investigation event stable keys"):
        cli_module._investigation_event_records(bundle)  # pyright: ignore[reportPrivateUsage]


def test_openrca_export_uses_audited_initial_top1_recommendation(
    tmp_path: Path,
) -> None:
    bundle = _initial_top1_recommendation_bundle(tmp_path)

    replay = runner.invoke(app, ["replay", str(bundle)])
    exported = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert replay.exit_code == 0, replay.output
    assert json.loads(replay.stdout)["finding_count"] == 0
    assert exported.exit_code == 0, exported.output
    marker = json.loads(exported.stderr)
    assert marker["policy"] == "initial-top1-v1"
    assert marker["rejected_candidate_id"] == "candidate-selected"
    assert list(csv.DictReader(io.StringIO(exported.stdout))) == [
        {
            "row_id": "1",
            "prediction": json.dumps(
                {
                    "1": {
                        "root cause component": "Tomcat01",
                        "root cause reason": "high CPU usage",
                    }
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        }
    ]


def test_openrca_export_rejects_mismatched_initial_top1_audit(
    tmp_path: Path,
) -> None:
    bundle = _initial_top1_recommendation_bundle(
        tmp_path, weak_initial_top1_candidate_id="different-candidate"
    )

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 1
    assert "does not match its weak-finding audit records" in result.stderr


def test_openrca_export_ignores_recommendation_on_failed_terminal_run(
    tmp_path: Path,
) -> None:
    bundle = _benchmark_recommendation_bundle(tmp_path, status="runtime_failure")

    result = runner.invoke(app, ["export", str(bundle), "--format", "openrca"])

    assert result.exit_code == 0, result.output
    assert result.stderr == ""
    assert list(csv.DictReader(io.StringIO(result.stdout))) == [
        {"row_id": "1", "prediction": "{}"}
    ]


def test_run_rejects_a_concurrent_writer_lock(tmp_path: Path) -> None:
    decisions = _recorded_decisions(tmp_path)
    bundle = tmp_path / "bundle"
    work = tmp_path / ".bundle.work"
    work.mkdir()

    with (work / "run.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = runner.invoke(
            app,
            [
                "run",
                "--data-root",
                str(PUBLIC_INPUT),
                "--output",
                str(bundle),
                "--run-id",
                "locked-run",
                "--recorded-decisions",
                str(decisions),
            ],
        )

    assert result.exit_code == 1
    assert "locked by another run" in result.stderr
    assert not (work / "run.sqlite3").exists()


def test_run_rejects_symlink_work_directory(
    tmp_path: Path,
) -> None:
    decisions = _recorded_decisions(tmp_path)
    bundle = tmp_path / "bundle"
    outside = tmp_path / "outside"
    outside.mkdir()
    bundle.with_name(".bundle.work").symlink_to(outside, target_is_directory=True)

    result = runner.invoke(
        app,
        [
            "run",
            "--data-root",
            str(PUBLIC_INPUT),
            "--output",
            str(bundle),
            "--run-id",
            "symlink-work",
            "--recorded-decisions",
            str(decisions),
        ],
    )

    assert result.exit_code == 1
    assert "work directory must not be a symlink" in result.stderr
    assert list(outside.iterdir()) == []


def test_pinned_work_directory_cannot_be_redirected_after_open(
    tmp_path: Path,
) -> None:
    work = tmp_path / ".bundle.work"
    outside = tmp_path / "outside"
    outside.mkdir()

    with cli_module._work_directory(work) as (pinned, _descriptor):  # pyright: ignore[reportPrivateUsage]
        moved = tmp_path / "original-work"
        work.rename(moved)
        work.symlink_to(outside, target_is_directory=True)
        (pinned / "probe").write_text("inside pinned directory")

    assert (moved / "probe").read_text() == "inside pinned directory"
    assert list(outside.iterdir()) == []
