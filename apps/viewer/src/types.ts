export type JsonObject = Record<string, unknown>;

export interface BundleRecord {
  kind: string;
  payload: JsonObject;
  revision: number;
  stable_key: string;
}

export interface TopologyNode {
  canonical_id: string;
  entity_type?: string;
  [key: string]: unknown;
}

export interface TopologyEdge {
  edge_id: string;
  source_id: string;
  target_id: string;
  relation_type: string;
  fact_kind?: string;
  span_pairs?: number | null;
  [key: string]: unknown;
}

export interface Topology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  coverage?: JsonObject;
  [key: string]: unknown;
}

export interface ArtifactReference {
  content_hash?: string;
  selector?: JsonObject;
  visibility?: string;
}

export interface EvidencePayload extends JsonObject {
  evidence_id: string;
  observation_refs?: string[];
  subject_ids?: string[];
  predicate?: string;
  value?: unknown;
  observed_interval?: JsonObject;
  time_uncertainty?: string;
  artifact_refs?: ArtifactReference[];
}

export interface Hypothesis extends JsonObject {
  hypothesis_id: string;
  subject?: string;
  statement?: string;
  status?: string;
  supporting_evidence_refs?: string[];
  opposing_evidence_refs?: string[];
  open_questions?: string[];
}

export interface CausalStep extends JsonObject {
  step_id?: string;
  from_entity: string;
  to_entity: string;
  mechanism?: string;
  status?: string;
}

export interface CausalPath extends JsonObject {
  path_id?: string;
  hypothesis_id?: string;
  root_cause_entity_id?: string;
  symptom_entity_ids?: string[];
  steps?: CausalStep[];
}

export interface Candidate extends JsonObject {
  hypothesis: Hypothesis;
  paths?: CausalPath[];
  selected?: boolean;
}

export interface RunReport extends JsonObject {
  run_id: string;
  task_id?: string;
  status: string;
  abstain_reason?: string | null;
  failure?: unknown;
  candidates?: Candidate[];
  findings?: JsonObject[];
  gaps?: unknown[];
  unresolved_questions?: string[];
  paths?: CausalPath[];
}

export interface RunBundle {
  manifest: JsonObject;
  task: JsonObject;
  topology: Topology;
  report: RunReport;
  evidence: BundleRecord[];
  observations: BundleRecord[];
  decisions: BundleRecord[];
  events: BundleRecord[];
  attempts: BundleRecord[];
  budgets: BundleRecord[];
  contexts: BundleRecord[];
  modelRequests?: BundleRecord[];
  executionAudit?: BundleRecord[];
  readArtifact?: (hash: string) => Promise<string>;
  sourceLabel: string;
}

export interface TimelineItem extends BundleRecord {
  timelineKind:
    | "decision"
    | "attempt"
    | "observation"
    | "evidence"
    | "event"
    | "budget"
    | "context";
  label: string;
  operationTypes: string[];
}

export type ComparisonLayerName =
  | "environment_manifest"
  | "context_manifest"
  | "decision"
  | "observation"
  | "evidence"
  | "termination";

export interface BundleDifference {
  index: number;
  left: unknown;
  right: unknown;
}

export interface BundleComparison {
  same: boolean;
  firstDivergence: (BundleDifference & { layer: ComparisonLayerName }) | null;
  layers: Record<ComparisonLayerName, { same: boolean; differences: BundleDifference[] }>;
}
