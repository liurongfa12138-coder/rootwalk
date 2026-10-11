"""Strict domain contracts shared by RootWalk adapters and runtime."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    model_validator,
)

NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Contract(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class TimeRange(Contract):
    start: datetime
    end: datetime
    end_exclusive: bool = True

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("time ranges must include a timezone")
        if self.end <= self.start:
            raise ValueError("time range end must be after start")
        return self


class ReferenceRange(TimeRange):
    health_status: NonEmptyStr


class AllowedScope(Contract):
    entities: list[NonEmptyStr] = Field(default_factory=list)
    datasets: list[NonEmptyStr] = Field(default_factory=list)
    tools: list[NonEmptyStr] = Field(default_factory=list)
    telemetry_root: NonEmptyStr | None = None
    entity_inventory_ref: NonEmptyStr | None = None


class Task(Contract):
    task_id: NonEmptyStr
    environment: NonEmptyStr
    question: NonEmptyStr
    incident_range: TimeRange
    timezone: NonEmptyStr
    allowed_query_range: TimeRange
    entry_entities: list[NonEmptyStr]
    allowed_scope: AllowedScope
    requested_outputs: list[NonEmptyStr]
    dataset_manifest_ref: NonEmptyStr

    # Public-source provenance retained without changing the generic facts above.
    source_row_id: Annotated[int, Field(ge=0)] | None = None
    task_index: NonEmptyStr | None = None
    reference_range: ReferenceRange | None = None
    entry_reason: NonEmptyStr | None = None
    expected_fault_count: Annotated[int, Field(gt=0)] | None = None
    fault_count_source: NonEmptyStr | None = None
    selection_policy: NonEmptyStr | None = None
    data_mode: NonEmptyStr | None = None
    source_commit: NonEmptyStr | None = None


class ArtifactRef(Contract):
    content_hash: Sha256Hex
    selector: NonEmptyStr | dict[str, JsonValue]
    visibility: Literal["model_readable", "audit_only", "evaluation_only"] = (
        "model_readable"
    )


Completeness = Literal["complete", "partial", "unknown"]


class Coverage(Contract):
    requested_range: TimeRange
    observed_range: TimeRange | None = None
    sample_interval: NonEmptyStr | float | None = None
    completeness: Completeness
    sampling: NonEmptyStr | None = None
    truncation: NonEmptyStr | None = None
    collection_health: NonEmptyStr | None = None
    basis: NonEmptyStr


class TopologyNode(Contract):
    canonical_id: NonEmptyStr
    original_id: NonEmptyStr
    entity_type: NonEmptyStr
    sources: list[NonEmptyStr]
    evidence_role: Literal["root_cause_candidate", "impact_only"] = (
        "root_cause_candidate"
    )
    source_aliases: dict[NonEmptyStr, list[NonEmptyStr]] = Field(default_factory=dict)
    normalization: NonEmptyStr | None = None
    public_candidate_match: NonEmptyStr | None = None
    mapping_status: NonEmptyStr | None = None
    mapping_evidence: NonEmptyStr | None = None


class TopologyEdgeSample(Contract):
    trace_id: NonEmptyStr
    parent_span_id: NonEmptyStr
    child_span_id: NonEmptyStr
    parent_source_record: Annotated[int, Field(ge=0)]
    child_source_record: Annotated[int, Field(ge=0)]


class MissingParentSample(Contract):
    trace_id: NonEmptyStr
    span_id: NonEmptyStr
    parent_id: NonEmptyStr
    source_record: Annotated[int, Field(ge=0)]


class TopologyCoverage(Contract):
    status: Completeness
    unique_spans: Annotated[int, Field(ge=0)]
    matched_parent_spans: Annotated[int, Field(ge=0)]
    missing_parent_spans: Annotated[int, Field(ge=0)]
    root_like_sentinel_spans: Annotated[int, Field(ge=0)]
    root_marker_values: dict[str, int]
    intra_entity_pairs: Annotated[int, Field(ge=0)]
    duplicate_span_keys: Annotated[int, Field(ge=0)]
    conflicting_span_keys: Annotated[int, Field(ge=0)]
    ambiguous_links_skipped: Annotated[int, Field(ge=0)]
    missing_parent_samples: list[MissingParentSample]
    note: NonEmptyStr


class TopologyEdge(Contract):
    edge_id: NonEmptyStr
    source_id: NonEmptyStr
    target_id: NonEmptyStr
    relation_type: NonEmptyStr
    fact_kind: Literal["observed", "declared", "derived"]
    artifact_refs: list[ArtifactRef] = []
    valid_range: TimeRange | None = None
    coverage: Coverage | None = None
    span_pairs: Annotated[int, Field(ge=0)] | None = None
    first_observed_ms: Annotated[int, Field(ge=0)] | None = None
    last_observed_ms: Annotated[int, Field(ge=0)] | None = None
    samples: list[TopologyEdgeSample] = []


class TopologySnapshot(Contract):
    snapshot_id: NonEmptyStr
    source: NonEmptyStr
    valid_range: TimeRange = Field(
        validation_alias=AliasChoices("valid_range", "window")
    )
    nodes: list[TopologyNode]
    edges: list[TopologyEdge]
    coverage: TopologyCoverage | Coverage


ObservationStatus = Literal[
    "success", "partial", "no_data", "unsupported", "error", "cancelled"
]


class Observation(Contract):
    observation_id: NonEmptyStr
    tool: NonEmptyStr
    arguments: dict[str, JsonValue]
    status: ObservationStatus
    coverage: Coverage
    artifact_refs: list[ArtifactRef]
    attempt_id: NonEmptyStr
    summary: NonEmptyStr | None = None
    error_type: NonEmptyStr | None = None
    retryable: bool | None = None
    extraction_gap: NonEmptyStr | None = None


class Extractor(Contract):
    kind: Literal["deterministic", "model", "human"]
    name: NonEmptyStr
    version: NonEmptyStr
    parameters_hash: NonEmptyStr


class Evidence(Contract):
    evidence_id: NonEmptyStr
    observation_refs: list[NonEmptyStr]
    subject_ids: list[NonEmptyStr]
    predicate: NonEmptyStr
    value: JsonValue
    unit: NonEmptyStr | None = None
    observed_interval: TimeRange
    onset_interval: TimeRange | None = None
    time_uncertainty: NonEmptyStr
    artifact_refs: list[ArtifactRef]
    extractor: Extractor
    source_fingerprint: NonEmptyStr
    dataset_snapshot_id: NonEmptyStr
    evidence_kind: NonEmptyStr = "unknown"
    mechanism_dimension: NonEmptyStr | None = None
    independence_group: NonEmptyStr | None = None
    semantic_quality: Literal["typed", "inferred", "unknown"] = "unknown"


EvidenceImpact = Literal["supports", "weakens", "refutes"]
HypothesisStatus = Literal["open", "supported", "refuted", "inconclusive"]
AnomalyDispositionKind = Literal[
    "root_cause",
    "propagated_symptom",
    "background",
    "concurrent_noncausal",
    "measurement_artifact",
    "unresolved",
]


class EvidenceAssessmentProposal(Contract):
    evidence_id: NonEmptyStr
    hypothesis_id: NonEmptyStr
    impact: EvidenceImpact
    reason_code: NonEmptyStr
    explanation: NonEmptyStr


class EvidenceAssessment(EvidenceAssessmentProposal):
    assessment_id: NonEmptyStr
    observation_digest: Sha256Hex


class MaterialAnomaly(Contract):
    """One adapter-admitted event; mechanism labels remain competing hypotheses."""

    anomaly_id: NonEmptyStr
    subject_id: NonEmptyStr
    signal_family: NonEmptyStr
    signal_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    mechanism_candidates: list[NonEmptyStr] = Field(default_factory=list)
    source_observation_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    proposal_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    materiality_basis: NonEmptyStr
    priority_facts: dict[NonEmptyStr, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def unique_members(self) -> Self:
        for name in (
            "signal_ids",
            "mechanism_candidates",
            "source_observation_ids",
            "proposal_ids",
        ):
            values = getattr(self, name)
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        return self


class AnomalyDispositionProposal(Contract):
    anomaly_id: NonEmptyStr
    disposition: AnomalyDispositionKind
    hypothesis_id: NonEmptyStr | None = None
    evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    explanation: NonEmptyStr

    @model_validator(mode="after")
    def grounded_disposition(self) -> Self:
        if self.disposition in {"root_cause", "propagated_symptom"} and (
            self.hypothesis_id is None
        ):
            raise ValueError(f"{self.disposition} disposition requires hypothesis_id")
        if self.disposition != "unresolved" and not self.evidence_refs:
            raise ValueError(
                "resolved anomaly disposition requires at least one evidence_ref"
            )
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("anomaly disposition evidence_refs must be unique")
        return self


class AnomalyDisposition(AnomalyDispositionProposal):
    disposition_id: NonEmptyStr
    observation_digest: Sha256Hex


class FrontierHypothesisProposal(Contract):
    client_id: NonEmptyStr | None = None
    subject: NonEmptyStr
    statement: NonEmptyStr
    derived_from: Annotated[list[NonEmptyStr], Field(min_length=1)]
    open_questions: list[NonEmptyStr] = Field(default_factory=list)


class FrontierHypothesisUpdate(Contract):
    hypothesis_id: NonEmptyStr
    status: HypothesisStatus
    open_questions: list[NonEmptyStr] = Field(default_factory=list)


class ReflectionNextCheck(Contract):
    operation_tools: Annotated[list[NonEmptyStr], Field(min_length=1, max_length=8)]
    distinguishes_hypothesis_ids: list[NonEmptyStr] = Field(default_factory=list)
    purpose: NonEmptyStr
    expected_outcomes: Annotated[list[NonEmptyStr], Field(min_length=1)]


class Reflection(Contract):
    based_on_observation_ids: Annotated[list[NonEmptyStr], Field(min_length=1)]
    evidence_update: list[EvidenceAssessmentProposal] = []
    anomaly_disposition_update: list[AnomalyDispositionProposal] = []
    hypothesis_update: list[FrontierHypothesisUpdate] = []
    current_best_hypothesis_id: NonEmptyStr | None = None
    clear_current_best: bool = False
    new_hypothesis: FrontierHypothesisProposal | None = None
    new_hypothesis_is_current_best: bool = False
    next_check: ReflectionNextCheck | None
    can_finalize: bool

    @model_validator(mode="after")
    def valid_control_result(self) -> Self:
        if len(self.evidence_update) > 32:
            raise ValueError("evidence_update may contain at most 32 items")
        if len(self.anomaly_disposition_update) > 64:
            raise ValueError("anomaly_disposition_update may contain at most 64 items")
        if len(self.hypothesis_update) > 12:
            raise ValueError("hypothesis_update may contain at most 12 items")
        if len(self.based_on_observation_ids) != len(
            set(self.based_on_observation_ids)
        ):
            raise ValueError("based_on_observation_ids must be unique")
        assessment_pairs = [
            (item.evidence_id, item.hypothesis_id) for item in self.evidence_update
        ]
        if len(assessment_pairs) != len(set(assessment_pairs)):
            raise ValueError(
                "one reflection may assess an evidence/hypothesis pair only once"
            )
        disposition_ids = [item.anomaly_id for item in self.anomaly_disposition_update]
        if len(disposition_ids) != len(set(disposition_ids)):
            raise ValueError("one reflection may disposition an anomaly only once")
        updated = [item.hypothesis_id for item in self.hypothesis_update]
        if len(updated) != len(set(updated)):
            raise ValueError("hypothesis_update targets must be unique")
        if self.new_hypothesis_is_current_best and self.new_hypothesis is None:
            raise ValueError("new_hypothesis_is_current_best requires new_hypothesis")
        if self.new_hypothesis_is_current_best and self.current_best_hypothesis_id:
            proposal_handles = {"new_hypothesis"}
            assert self.new_hypothesis is not None
            if self.new_hypothesis.client_id is not None:
                proposal_handles.add(self.new_hypothesis.client_id)
            if self.current_best_hypothesis_id not in proposal_handles:
                raise ValueError(
                    "current_best_hypothesis_id conflicts with the new current-best "
                    "hypothesis"
                )
        if self.clear_current_best and (
            self.current_best_hypothesis_id is not None
            or self.new_hypothesis_is_current_best
        ):
            raise ValueError(
                "clear_current_best conflicts with selecting an existing or new best"
            )
        if self.can_finalize and self.next_check is not None:
            raise ValueError("a finalizable reflection cannot request another check")
        return self


class Hypothesis(Contract):
    hypothesis_id: NonEmptyStr
    subject: NonEmptyStr
    statement: NonEmptyStr
    status: HypothesisStatus
    supporting_evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    opposing_evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    open_questions: list[NonEmptyStr] = Field(default_factory=list)
    created_from: list[NonEmptyStr] = Field(default_factory=list)
    supersedes: NonEmptyStr | None = None


CausalStepStatus = Literal[
    "proposed", "observationally_supported", "contradicted", "unresolved"
]


class CausalStep(Contract):
    step_id: NonEmptyStr
    from_entity: NonEmptyStr
    to_entity: NonEmptyStr
    mechanism: NonEmptyStr
    basis_topology_edge_ids: list[NonEmptyStr] = Field(default_factory=list)
    supporting_evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    opposing_evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    status: CausalStepStatus


class Path(Contract):
    path_id: NonEmptyStr | None = None
    hypothesis_id: NonEmptyStr
    root_cause_entity_id: NonEmptyStr
    symptom_entity_ids: list[NonEmptyStr]
    steps: list[CausalStep]


CausalPath = Path


class QueryOperation(Contract):
    type: Literal["query"]
    tool: NonEmptyStr
    arguments: dict[str, JsonValue]


class UpdateHypothesisOperation(Contract):
    type: Literal["update_hypothesis"]
    action: Literal["create", "revise"]
    hypothesis_id: NonEmptyStr | None = None
    subject: NonEmptyStr
    statement: NonEmptyStr
    status: HypothesisStatus
    supporting_evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    opposing_evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    open_questions: list[NonEmptyStr] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_target(self) -> Self:
        if self.action == "revise" and self.hypothesis_id is None:
            raise ValueError("revise requires hypothesis_id")
        if self.action == "create" and self.hypothesis_id is not None:
            raise ValueError("create cannot choose a runtime-assigned hypothesis_id")
        return self


class SubmitPathOperation(Contract):
    type: Literal["submit_path"]
    hypothesis_id: NonEmptyStr
    root_cause_entity_id: NonEmptyStr
    symptom_entity_ids: list[NonEmptyStr]
    steps: list[CausalStep]


class FinalizeOperation(Contract):
    type: Literal["finalize"]
    hypothesis_ids: list[NonEmptyStr] = Field(default_factory=list)
    path_ids: list[NonEmptyStr] = Field(default_factory=list)
    occurrence_evidence_refs: list[NonEmptyStr] = Field(default_factory=list)
    unresolved_questions: list[NonEmptyStr] = Field(default_factory=list)
    abstain_reason: NonEmptyStr | None = None

    @model_validator(mode="after")
    def conclusion_or_abstention(self) -> Self:
        causal_finding = bool(self.hypothesis_ids or self.path_ids)
        occurrence_recommendation = bool(self.occurrence_evidence_refs)
        abstention = self.abstain_reason is not None
        if sum((causal_finding, occurrence_recommendation, abstention)) != 1:
            raise ValueError(
                "provide a causal finding, occurrence recommendation, or abstain_reason"
            )
        if len(self.occurrence_evidence_refs) != len(
            set(self.occurrence_evidence_refs)
        ):
            raise ValueError("occurrence_evidence_refs must be unique")
        if len(self.occurrence_evidence_refs) > 8:
            raise ValueError("occurrence_evidence_refs may contain at most 8 items")
        return self


class AdvanceFrontierOperation(Contract):
    """Commit one Reflection and recompile the dynamic frontier without a Tool call."""

    type: Literal["advance_frontier"]


Operation: TypeAlias = Annotated[
    QueryOperation
    | UpdateHypothesisOperation
    | SubmitPathOperation
    | AdvanceFrontierOperation
    | FinalizeOperation,
    Field(discriminator="type"),
]

DecisionIntent = Literal["Explore", "Recall", "Revisit", "Backtrack"]


class Decision(Contract):
    decision_summary: NonEmptyStr
    intent: DecisionIntent | None = None
    operations: Annotated[list[Operation], Field(min_length=1, max_length=8)]
    reflection: Reflection | None = None
    schema_version: NonEmptyStr = "1"
    base_revision: Annotated[int, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def valid_batch(self) -> Self:
        if len(self.operations) > 1 and any(
            operation.type != "query" for operation in self.operations
        ):
            raise ValueError("only query operations may be batched")
        return self
