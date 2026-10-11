"""Mechanism-neutral material anomaly registration from adapter proposals."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, cast

from .contracts import Evidence, MaterialAnomaly, Observation


def _record(item: Observation | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(item, Observation):
        return item.model_dump(mode="json")
    raw = dict(item)
    payload = raw.get("payload")
    return (
        dict(cast(Mapping[str, Any], payload)) if isinstance(payload, Mapping) else raw
    )


def _overview_payload(item: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if item.get("tool") != "summarize_anomalies":
        return None
    result = item.get("result")
    if isinstance(result, Mapping):
        return cast(Mapping[str, Any], result)
    summary = item.get("summary")
    if not isinstance(summary, str):
        return None
    try:
        parsed: object = json.loads(summary)
    except json.JSONDecodeError:
        return None
    return cast(Mapping[str, Any], parsed) if isinstance(parsed, Mapping) else None


def _anomaly_id(key: tuple[str, str, str]) -> str:
    encoded = json.dumps(key, ensure_ascii=False, separators=(",", ":")).encode()
    return f"anomaly:{hashlib.sha256(encoded).hexdigest()[:24]}"


def material_anomalies(
    observations: Sequence[Observation | Mapping[str, Any]],
) -> list[MaterialAnomaly]:
    """Group adapter-admitted proposals into events before choosing a mechanism."""
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for raw_observation in observations:
        observation = _record(raw_observation)
        payload = _overview_payload(observation)
        if payload is None:
            continue
        observation_id = observation.get("observation_id")
        candidates = payload.get("fault_type_candidates")
        if not isinstance(observation_id, str) or not isinstance(candidates, list):
            continue
        for raw_candidate in cast(list[object], candidates):
            if not isinstance(raw_candidate, Mapping):
                continue
            candidate = cast(Mapping[str, Any], raw_candidate)
            if candidate.get("materiality") != "material":
                continue
            candidate_id = candidate.get("candidate_id")
            subject = candidate.get("entity_id")
            family = candidate.get("resource_family")
            fault_type = candidate.get("fault_type")
            metric_id = candidate.get("metric_id")
            source_metric_id = candidate.get("source_metric_id")
            signal = (
                source_metric_id if isinstance(source_metric_id, str) else metric_id
            )
            if not all(
                isinstance(value, str) and value
                for value in (candidate_id, subject, family, fault_type, signal)
            ):
                continue
            key = (cast(str, subject), cast(str, family), cast(str, signal))
            entry = grouped.setdefault(
                key,
                {
                    "subject_id": subject,
                    "signal_family": family,
                    "signal_ids": set[str](),
                    "mechanism_candidates": set[str](),
                    "source_observation_ids": set[str](),
                    "proposal_ids": set[str](),
                    "materiality_basis": str(
                        candidate.get("materiality_basis")
                        or "adapter_registered_candidate"
                    ),
                    "priority_facts": {},
                },
            )
            raw_group = candidate.get("metric_group")
            signals = (
                [
                    item
                    for item in cast(list[object], raw_group)
                    if isinstance(item, str)
                ]
                if isinstance(raw_group, list)
                else []
            )
            entry["signal_ids"].update([cast(str, signal), *signals])
            entry["mechanism_candidates"].add(fault_type)
            entry["source_observation_ids"].add(observation_id)
            entry["proposal_ids"].add(candidate_id)
            facts = cast(dict[str, Any], entry["priority_facts"])
            for name in ("score", "overview_score", "peer_score_ratio"):
                value = candidate.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    facts[name] = max(float(value), float(facts.get(name, 0.0)))
            if isinstance(candidate.get("peer_outlier"), bool):
                facts["peer_outlier"] = bool(
                    candidate["peer_outlier"] or facts.get("peer_outlier", False)
                )
    return [
        MaterialAnomaly(
            anomaly_id=_anomaly_id(key),
            subject_id=cast(str, value["subject_id"]),
            signal_family=cast(str, value["signal_family"]),
            signal_ids=sorted(cast(set[str], value["signal_ids"])),
            mechanism_candidates=sorted(cast(set[str], value["mechanism_candidates"])),
            source_observation_ids=sorted(
                cast(set[str], value["source_observation_ids"])
            ),
            proposal_ids=sorted(cast(set[str], value["proposal_ids"])),
            materiality_basis=cast(str, value["materiality_basis"]),
            priority_facts=cast(dict[str, Any], value["priority_facts"]),
        )
        for key, value in sorted(grouped.items())
    ]


def material_anomalies_from_evidence(evidence: Evidence) -> list[MaterialAnomaly]:
    """Promote material typed events that navigation did not admit."""
    if evidence.evidence_kind not in {"metric_profile", "service_profile"} or not (
        isinstance(evidence.value, Mapping)
    ):
        return []
    summary = cast(Mapping[str, object], evidence.value).get("finding_gate_summary")
    if not isinstance(summary, Mapping):
        return []
    values = cast(Mapping[str, object], summary)
    fault_type = values.get("requested_fault_type")
    fault_type_ready = values.get("fault_type_ready") is True
    material_event_ready = values.get("material_event_ready") is True
    qualifying_count = values.get("qualifying_metric_count")
    material_event = fault_type_ready or material_event_ready
    if not material_event or not isinstance(fault_type, str):
        return []
    signal_ids = [
        item
        for field in (
            "material_event_metric_ids",
            "fault_type_anchor_metric_ids",
            "qualifying_metric_ids",
            "strong_metric_ids",
        )
        for raw_items in [values.get(field)]
        if isinstance(raw_items, list)
        for item in cast(list[object], raw_items)
        if isinstance(item, str) and item
    ]
    score = values.get("root_cause_score")
    facts: dict[str, Any] = {
        "fault_type_ready": fault_type_ready,
        "material_event_ready": material_event_ready,
    }
    if isinstance(qualifying_count, int) and not isinstance(qualifying_count, bool):
        facts["qualifying_metric_count"] = qualifying_count
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        facts["root_cause_score"] = float(score)
    result: list[MaterialAnomaly] = []
    for subject in evidence.subject_ids:
        key = (subject, evidence.mechanism_dimension or "typed_profile", fault_type)
        result.append(
            MaterialAnomaly(
                anomaly_id=_anomaly_id(key),
                subject_id=subject,
                signal_family=evidence.mechanism_dimension or "typed_profile",
                signal_ids=sorted(set(signal_ids)) or [evidence.predicate],
                mechanism_candidates=[fault_type],
                source_observation_ids=list(evidence.observation_refs),
                proposal_ids=[evidence.evidence_id],
                materiality_basis=(
                    "typed_fault_ready_profile"
                    if fault_type_ready
                    else str(
                        values.get("material_event_basis")
                        or "typed_profile_material_event"
                    )
                ),
                priority_facts=facts,
            )
        )
    return result
