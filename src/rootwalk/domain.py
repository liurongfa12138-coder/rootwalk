"""Small, deterministic domain rules for evidence and causal paths."""

from __future__ import annotations

from collections.abc import Iterable, Set
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

from rootwalk.contracts import (
    Coverage,
    Evidence,
    Observation,
    Path,
    TimeRange,
    TopologyCoverage,
    TopologyEdge,
    TopologySnapshot,
)

ObservationDisposition = Literal["gap", "negative_fact", "observation_only"]
ReferenceGrounding = Literal["pass", "fail"]
StructuralGrounding = Literal["pass", "partial", "fail"]
TemporalConsistency = Literal["consistent", "contradicted", "unknown"]
GroundingCoverage = Literal["complete", "partial", "unknown"]


@dataclass(frozen=True, slots=True)
class PathGrounding:
    reference_grounding: ReferenceGrounding
    structural_grounding: StructuralGrounding
    temporal_consistency: TemporalConsistency
    coverage: GroundingCoverage
    semantic_correctness: Literal["not_evaluated"] = "not_evaluated"


class EvidenceLedger:
    """Keep every reference while counting each source only once."""

    def __init__(self, evidence: Iterable[Evidence] = ()) -> None:
        self._by_id: dict[str, Evidence] = {}
        self._canonical_by_source: dict[str, str] = {}
        for item in evidence:
            self.add(item)

    def add(self, evidence: Evidence) -> bool:
        """Add evidence and return whether it is an independent source."""
        existing = self._by_id.get(evidence.evidence_id)
        if existing is not None:
            if existing != evidence:
                raise ValueError(f"conflicting evidence_id: {evidence.evidence_id}")
            return False

        self._by_id[evidence.evidence_id] = evidence
        if evidence.source_fingerprint in self._canonical_by_source:
            return False
        self._canonical_by_source[evidence.source_fingerprint] = evidence.evidence_id
        return True

    @property
    def independent_count(self) -> int:
        return len(self._canonical_by_source)

    def canonical_evidence_id(self, evidence_id: str) -> str:
        evidence = self._by_id[evidence_id]
        return self._canonical_by_source[evidence.source_fingerprint]

    def get(self, evidence_id: str) -> Evidence | None:
        return self._by_id.get(evidence_id)

    def __len__(self) -> int:
        return len(self._by_id)


def classify_observation(observation: Observation) -> ObservationDisposition:
    """Classify what an observation may say; never infer business health."""
    if observation.extraction_gap:
        return "gap"
    if observation.status in {"unsupported", "error", "cancelled"}:
        return "gap"
    if observation.status == "no_data":
        coverage = observation.coverage
        healthy = (coverage.collection_health or "").strip().lower() in {
            "healthy",
            "ok",
        }
        if (
            coverage.completeness == "complete"
            and healthy
            and coverage.truncation is None
        ):
            return "negative_fact"
        return "gap"
    return "observation_only"


def ground_path(
    path: Path,
    topology: TopologySnapshot,
    evidence: Iterable[Evidence] = (),
    observations: Iterable[Observation] | None = None,
    *,
    claim_evidence_refs: Iterable[str] = (),
    visible_evidence_ids: Set[str] | None = None,
    cause_onset: TimeRange | None = None,
    symptom_onset: TimeRange | None = None,
    clock_tolerance: timedelta = timedelta(0),
) -> PathGrounding:
    """Mechanically ground a path without judging whether its claim is true."""
    if clock_tolerance < timedelta(0):
        raise ValueError("clock_tolerance must be non-negative")

    evidence_by_id = {item.evidence_id: item for item in evidence}
    refs = _path_evidence_refs(path, claim_evidence_refs)
    reference = _reference_grounding(refs, evidence_by_id, visible_evidence_ids)
    if not path.steps and not _has_local_evidence(
        path.root_cause_entity_id, refs, evidence_by_id
    ):
        reference = "fail"
    structural = _structural_grounding(path, topology)

    if not path.steps and all(
        symptom == path.root_cause_entity_id for symptom in path.symptom_entity_ids
    ):
        temporal: TemporalConsistency = "consistent"
    else:
        cause_onset = cause_onset or _onset_for_entities(
            refs, evidence_by_id, {path.root_cause_entity_id}
        )
        symptom_onset = symptom_onset or _onset_for_entities(
            refs, evidence_by_id, set(path.symptom_entity_ids)
        )
        temporal = check_temporal_consistency(
            cause_onset, symptom_onset, clock_tolerance
        )

    coverage = _grounding_coverage(
        path,
        topology,
        reference,
        structural,
        refs,
        evidence_by_id,
        observations,
    )
    return PathGrounding(reference, structural, temporal, coverage)


def check_temporal_consistency(
    cause_onset: TimeRange | None,
    symptom_onset: TimeRange | None,
    clock_tolerance: timedelta = timedelta(0),
) -> TemporalConsistency:
    if clock_tolerance < timedelta(0):
        raise ValueError("clock_tolerance must be non-negative")
    if cause_onset is None or symptom_onset is None:
        return "unknown"
    if cause_onset.start > symptom_onset.end + clock_tolerance:
        return "contradicted"
    if cause_onset.end + clock_tolerance < symptom_onset.start:
        return "consistent"
    return "unknown"


def _path_evidence_refs(path: Path, claim_evidence_refs: Iterable[str]) -> set[str]:
    return set(claim_evidence_refs) | {
        evidence_id
        for step in path.steps
        for evidence_id in (
            *step.supporting_evidence_refs,
            *step.opposing_evidence_refs,
        )
    }


def _has_local_evidence(
    entity_id: str, refs: set[str], evidence_by_id: dict[str, Evidence]
) -> bool:
    return any(
        (item := evidence_by_id.get(evidence_id)) is not None
        and entity_id in item.subject_ids
        for evidence_id in refs
    )


def _reference_grounding(
    refs: set[str],
    evidence_by_id: dict[str, Evidence],
    visible_evidence_ids: Set[str] | None,
) -> ReferenceGrounding:
    for evidence_id in refs:
        item = evidence_by_id.get(evidence_id)
        if item is None or (
            visible_evidence_ids is not None and evidence_id not in visible_evidence_ids
        ):
            return "fail"
        if not item.observation_refs or not item.subject_ids or not item.artifact_refs:
            return "fail"
    return "pass"


def _structural_grounding(
    path: Path, topology: TopologySnapshot
) -> StructuralGrounding:
    known_entities = {node.canonical_id for node in topology.nodes}
    path_entities = {
        path.root_cause_entity_id,
        *path.symptom_entity_ids,
        *(step.from_entity for step in path.steps),
        *(step.to_entity for step in path.steps),
    }
    if not path_entities.issubset(known_entities):
        return "fail"
    if not path.steps:
        if path.symptom_entity_ids and all(
            symptom == path.root_cause_entity_id for symptom in path.symptom_entity_ids
        ):
            return "pass"
        return "partial"

    edges = {edge.edge_id: edge for edge in topology.edges}
    node_types = {
        node.canonical_id: node.entity_type.strip().lower() for node in topology.nodes
    }
    result: StructuralGrounding = "pass"
    graph: dict[str, set[str]] = {}

    for step in path.steps:
        graph.setdefault(step.from_entity, set()).add(step.to_entity)
        if not step.basis_topology_edge_ids:
            result = "partial"
            continue
        for edge_id in step.basis_topology_edge_ids:
            edge = edges.get(edge_id)
            if edge is None:
                result = "partial"
            elif not _edge_carries_step(
                edge,
                step.from_entity,
                step.to_entity,
                step.mechanism,
                node_types,
            ):
                return "fail"

    reachable = {path.root_cause_entity_id}
    pending = [path.root_cause_entity_id]
    while pending:
        source = pending.pop()
        for target in graph.get(source, ()):
            if target not in reachable:
                reachable.add(target)
                pending.append(target)
    if any(symptom not in reachable for symptom in path.symptom_entity_ids):
        return "fail"
    return result


def _edge_carries_step(
    edge: TopologyEdge,
    source: str,
    target: str,
    mechanism: str,
    node_types: dict[str, str],
) -> bool:
    relation = edge.relation_type.strip().lower()
    mechanism = mechanism.strip().lower().replace("-", "_").replace(" ", "_")
    if relation in {"calls", "uses"}:
        if mechanism == "dependency_failure":
            return source == edge.target_id and target == edge.source_id
        if mechanism == "request_load":
            return source == edge.source_id and target == edge.target_id
        return False
    if relation == "runs_on":
        return (
            mechanism.startswith("resource_")
            and node_types.get(edge.source_id) == "pod"
            and node_types.get(edge.target_id) == "node"
            and source == edge.target_id
            and target == edge.source_id
        )
    return False


def _onset_for_entities(
    refs: set[str],
    evidence_by_id: dict[str, Evidence],
    entity_ids: set[str],
) -> TimeRange | None:
    intervals = [
        item.onset_interval
        for evidence_id in refs
        if (item := evidence_by_id.get(evidence_id)) is not None
        and entity_ids.intersection(item.subject_ids)
        and item.onset_interval is not None
    ]
    if not intervals:
        return None
    first = min(interval.start for interval in intervals)
    last = max(interval.end for interval in intervals)
    return TimeRange(start=first, end=last)


def _grounding_coverage(
    path: Path,
    topology: TopologySnapshot,
    reference: ReferenceGrounding,
    structural: StructuralGrounding,
    refs: set[str],
    evidence_by_id: dict[str, Evidence],
    observations: Iterable[Observation] | None,
) -> GroundingCoverage:
    statuses: list[GroundingCoverage] = []
    if not path.steps and reference == "fail":
        statuses.append("partial")
    if path.steps:
        statuses.append(_coverage_status(topology.coverage))
    if structural == "partial":
        statuses.append("partial")

    if observations is not None:
        observation_by_id = {
            observation.observation_id: observation for observation in observations
        }
        observation_refs = {
            observation_id
            for evidence_id in refs
            if (item := evidence_by_id.get(evidence_id)) is not None
            for observation_id in item.observation_refs
        }
        for observation_id in observation_refs:
            observation = observation_by_id.get(observation_id)
            if observation is None:
                statuses.append("unknown")
                continue
            statuses.append(_coverage_status(observation.coverage))
            if observation.extraction_gap:
                statuses.append("partial")

    if "unknown" in statuses:
        return "unknown"
    if "partial" in statuses:
        return "partial"
    return "complete"


def _coverage_status(
    coverage: Coverage | TopologyCoverage,
) -> GroundingCoverage:
    status = (
        coverage.completeness if isinstance(coverage, Coverage) else coverage.status
    )
    if status == "unknown":
        return "unknown"
    if status == "partial":
        return "partial"
    if isinstance(coverage, Coverage):
        if (coverage.collection_health or "").strip().lower() == "unknown":
            return "unknown"
        if coverage.truncation is not None:
            return "partial"
    return "complete"
