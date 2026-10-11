import { useCallback, useEffect, useMemo, useState, type ChangeEvent } from "react";
import {
  buildTimeline,
  collectStringValues,
  compareRunBundles,
  loadExampleBundle,
  parseBundleFiles,
} from "./bundle";
import { InvestigationGraph } from "./InvestigationGraph";
import { ExecutionPanel } from "./ExecutionPanel";
import type {
  BundleRecord,
  Candidate,
  CausalPath,
  EvidencePayload,
  Hypothesis,
  JsonObject,
  RunBundle,
  TimelineItem,
} from "./types";

const displayStatus = (status: string) => status.split("_").join(" · ").toUpperCase();
const shortId = (id: string) => {
  const parts = id.split(":");
  return parts[parts.length - 1] || id;
};
const asText = (value: unknown, fallback = "Not recorded") =>
  typeof value === "string" ? value : value == null ? fallback : JSON.stringify(value, null, 2);
const stringArray = (value: unknown) => (Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []);

function StatusMark({ status }: { status: string }) {
  const state = status.includes("fail") || status.includes("error")
    ? "failed"
    : status.includes("partial") || status.includes("inconclusive")
      ? "partial"
      : "complete";
  return <span className="status-mark" data-state={state} aria-hidden="true" />;
}

function SectionHeading({ title, count }: { title: string; count?: number }) {
  return (
    <div className="section-heading">
      <h2>{title}</h2>
      {count !== undefined && <span className="count">{count.toString().padStart(2, "0")}</span>}
    </div>
  );
}

function DetailRows({ rows }: { rows: Array<[string, unknown]> }) {
  return (
    <dl className="detail-rows">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{asText(value)}</dd>
        </div>
      ))}
    </dl>
  );
}

function relationFor(hypothesis: Hypothesis, evidenceId: string) {
  if ((hypothesis.supporting_evidence_refs ?? []).includes(evidenceId)) return "supports";
  if ((hypothesis.opposing_evidence_refs ?? []).includes(evidenceId)) return "opposes";
  return undefined;
}

function EvidenceDetail({
  record,
  observations,
  candidates,
  reportGaps,
}: {
  record?: BundleRecord;
  observations: BundleRecord[];
  candidates: Candidate[];
  reportGaps: unknown[];
}) {
  if (!record) return (
    <>
      <p className="empty-state">No evidence was recorded in this bundle.</p>
      <div className="evidence-block gap-block">
        <h4><span className="relation-shape gap" />Report gaps</h4>
        {reportGaps.length ? reportGaps.map((gap, index) => <p key={index}>{asText(gap)}</p>) : <p>None recorded</p>}
      </div>
    </>
  );
  const evidence = record.payload as EvidencePayload;
  const observationIds = evidence.observation_refs ?? [];
  const linkedObservations = observations.filter((item) => observationIds.includes(asText(item.payload.observation_id, "")));
  const related = candidates
    .map(({ hypothesis }) => ({ hypothesis, relation: relationFor(hypothesis, evidence.evidence_id) }))
    .filter((item): item is { hypothesis: Hypothesis; relation: "supports" | "opposes" } => Boolean(item.relation));
  const support = related.filter((item) => item.relation === "supports");
  const opposition = related.filter((item) => item.relation === "opposes");
  const gaps = [
    ...linkedObservations.flatMap((observation) =>
      observation.payload.extraction_gap ? [asText(observation.payload.extraction_gap)] : [],
    ),
    ...related.flatMap((item) => item.hypothesis.open_questions ?? []),
  ];

  return (
    <div className="evidence-detail" data-testid="evidence-detail">
      <div className="evidence-title">
        <StatusMark status={asText(evidence.value && typeof evidence.value === "object" ? (evidence.value as JsonObject).status : "recorded")} />
        <div>
          <h3>{asText(evidence.predicate, "Evidence")}</h3>
          <code>{evidence.evidence_id}</code>
        </div>
      </div>
      <DetailRows
        rows={[
          ["Subject", (evidence.subject_ids ?? []).join(", ") || undefined],
          ["Observed", evidence.observed_interval],
          ["Value", evidence.value],
          ["Uncertainty", evidence.time_uncertainty],
        ]}
      />
      <div className="evidence-block">
        <h4>Source selector</h4>
        {evidence.artifact_refs?.length ? evidence.artifact_refs.map((ref, index) => (
          <div className="artifact-ref" key={`${ref.content_hash ?? "source"}-${index}`}>
            <DetailRows rows={[["Content hash", ref.content_hash], ["Visibility", ref.visibility]]} />
            <pre>{JSON.stringify(ref.selector ?? {}, null, 2)}</pre>
          </div>
        )) : <p>Not recorded</p>}
      </div>
      <div className="evidence-block">
        <h4>Coverage</h4>
        {observationIds.length === 0 && <p>No observation references recorded.</p>}
        {observationIds.map((observationId) => {
          const observation = linkedObservations.find((item) => item.payload.observation_id === observationId);
          return (
            <div className="coverage-record" key={observationId}>
              <code>{observationId}</code>
              <pre>{JSON.stringify(observation?.payload.coverage ?? { status: "referenced observation not selected" }, null, 2)}</pre>
            </div>
          );
        })}
      </div>
      <div className="relation-grid">
        <div>
          <h4><span className="relation-shape support" />Support</h4>
          {support.length ? support.map(({ hypothesis }) => <p key={hypothesis.hypothesis_id}>{hypothesis.statement}</p>) : <p>None recorded</p>}
        </div>
        <div>
          <h4><span className="relation-shape oppose" />Opposition</h4>
          {opposition.length ? opposition.map(({ hypothesis }) => <p key={hypothesis.hypothesis_id}>{hypothesis.statement}</p>) : <p>None recorded</p>}
        </div>
      </div>
      <div className="evidence-block gap-block">
        <h4><span className="relation-shape gap" />Gap</h4>
        {gaps.length ? gaps.map((gap) => <p key={gap}>{gap}</p>) : <p>No linked extraction gap or open question.</p>}
      </div>
      <div className="evidence-block gap-block">
        <h4><span className="relation-shape gap" />Report gaps</h4>
        {reportGaps.length ? reportGaps.map((gap, index) => <p key={index}>{asText(gap)}</p>) : <p>None recorded</p>}
      </div>
    </div>
  );
}

function Timeline({
  items,
  selectedKey,
  onSelect,
}: {
  items: TimelineItem[];
  selectedKey?: string;
  onSelect: (item: TimelineItem) => void;
}) {
  return (
    <section className="timeline-panel" aria-labelledby="timeline-title">
      <SectionHeading title="Revision rail" count={items.length} />
      <p className="section-note" id="timeline-title">Operations and evidence in committed order.</p>
      <ol className="timeline-list">
        {items.map((item) => (
          <li key={`${item.revision}:${item.stable_key}`}>
            <button
              aria-current={selectedKey === item.stable_key ? "step" : undefined}
              aria-pressed={selectedKey === item.stable_key}
              className={selectedKey === item.stable_key ? "timeline-item selected" : "timeline-item"}
              onClick={() => onSelect(item)}
              type="button"
            >
              <span className="revision">R{item.revision.toString().padStart(3, "0")}</span>
              <span className="event-mark" data-kind={item.timelineKind} aria-hidden="true" />
              <span>
                <strong>{item.timelineKind}</strong>
                <span>{item.label}</span>
                {item.operationTypes.length > 0 && <small>{item.operationTypes.join(" + ")}</small>}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </section>
  );
}

function ComparisonPanel({ primary, secondary }: { primary: RunBundle; secondary: RunBundle }) {
  const result = useMemo(() => {
    try {
      return { comparison: compareRunBundles(primary, secondary), error: undefined };
    } catch (reason) {
      return {
        comparison: undefined,
        error: reason instanceof Error ? reason.message : "The selected runs cannot be compared.",
      };
    }
  }, [primary, secondary]);
  if (!result.comparison) {
    return (
      <section className="comparison-panel" aria-labelledby="comparison-title">
        <h2 id="comparison-title">Comparison unavailable</h2>
        <p className="comparison-note" role="alert">{result.error} Select a revision-ordered RunBundle and try again.</p>
      </section>
    );
  }
  const comparison = result.comparison;
  const first = comparison.firstDivergence;
  return (
    <section className="comparison-panel" aria-labelledby="comparison-title">
      <div className="comparison-heading">
        <div>
          <h2 id="comparison-title">First observable divergence</h2>
          <p>{primary.report.run_id} ↔ {secondary.report.run_id}</p>
        </div>
        <div className="comparison-result" aria-live="polite">
          <StatusMark status={comparison.same ? "complete" : "partial"} />
          <strong>{comparison.same ? "NO OBSERVABLE DIVERGENCE" : `${first?.layer.split("_").join(" ")} · INDEX ${first?.index}`}</strong>
        </div>
      </div>
      <p className="comparison-note">
        Compared in protocol order: environment manifest, context manifest, decision, observation, evidence, termination. Difference is observational, not a causal claim.
      </p>
      <ol className="comparison-layers" aria-label="Six comparison layers">
        {Object.entries(comparison.layers).map(([layer, result]) => (
          <li key={layer}>
            <StatusMark status={result.same ? "complete" : "partial"} />
            <span>{layer.split("_").join(" ")}</span>
            <strong>{result.same ? "same" : `${result.differences.length} different`}</strong>
          </li>
        ))}
      </ol>
      {first && (
        <div className="divergence-values">
          <div>
            <h3>Primary</h3>
            <pre>{first.left == null ? "Not present" : JSON.stringify(first.left, null, 2)}</pre>
          </div>
          <div>
            <h3>Comparison</h3>
            <pre>{first.right == null ? "Not present" : JSON.stringify(first.right, null, 2)}</pre>
          </div>
        </div>
      )}
    </section>
  );
}

export function RunViewer({ bundle, comparisonBundle }: { bundle: RunBundle; comparisonBundle?: RunBundle }) {
  const longIdentity = bundle.report.run_id.length > 48;
  const runTitle = longIdentity
    ? typeof bundle.task.source_row_id === "number" ? `Case ${bundle.task.source_row_id}` : `${bundle.report.run_id.slice(0,40)}…`
    : bundle.report.run_id;
  const candidates = bundle.report.candidates ?? [];
  const paths = useMemo<CausalPath[]>(
    () => [...(bundle.report.paths ?? []), ...candidates.flatMap((candidate) => candidate.paths ?? [])],
    [bundle.report.paths, candidates],
  );
  const timeline = useMemo(() => buildTimeline(bundle), [bundle]);
  const evidenceRecords = bundle.evidence.filter((record) => typeof record.payload.evidence_id === "string");
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | undefined>(() =>
    asText(evidenceRecords[0]?.payload.evidence_id, "") || undefined,
  );
  const [selectedTimelineKey, setSelectedTimelineKey] = useState<string>();
  const [selectedEntity, setSelectedEntity] = useState<string | undefined>(() =>
    stringArray(evidenceRecords[0]?.payload.subject_ids)[0],
  );
  const selectedEvidence = evidenceRecords.find((record) => record.payload.evidence_id === selectedEvidenceId);
  const termination = bundle.report.failure
    ? ["FAILED", asText(bundle.report.failure)]
    : bundle.report.abstain_reason
      ? ["ABSTAINED", bundle.report.abstain_reason]
      : bundle.report.unresolved_questions?.length
        ? ["SEALED WITH OPEN QUESTIONS", bundle.report.unresolved_questions.join(" ")]
        : ["SEALED", "Run completed with recorded findings."];

  const selectEvidence = (record: BundleRecord) => {
    setSelectedEvidenceId(asText(record.payload.evidence_id));
    setSelectedEntity(stringArray(record.payload.subject_ids)[0]);
  };
  const selectTimeline = (item: TimelineItem) => {
    setSelectedTimelineKey(item.stable_key);
    const refs = collectStringValues(item);
    const evidence = evidenceRecords.find((record) => refs.has(asText(record.payload.evidence_id, "")));
    if (evidence) selectEvidence(evidence);
    const hypothesis = candidates.map((candidate) => candidate.hypothesis).find((candidate) => refs.has(candidate.hypothesis_id));
    if (hypothesis) {
      setSelectedEntity(hypothesis.subject);
      const linkedId = hypothesis.supporting_evidence_refs?.[0] ?? hypothesis.opposing_evidence_refs?.[0];
      if (linkedId) setSelectedEvidenceId(linkedId);
    }
  };

  return (
    <main>
      <section className="run-summary" aria-labelledby="run-title">
        <div>
          <p className="run-source">{bundle.sourceLabel}</p>
          <h1 id="run-title">{runTitle}</h1>
          {longIdentity ? <details className="run-identifiers"><summary>运行标识</summary><code>{bundle.report.run_id}</code><p className="task-id">{bundle.report.task_id ?? asText(bundle.task.task_id)}</p></details> : <p className="task-id">{bundle.report.task_id ?? asText(bundle.task.task_id)}</p>}
        </div>
        <div className="outcome">
          <div className="outcome-status"><StatusMark status={bundle.report.status} />{displayStatus(bundle.report.status)}</div>
          <p className="termination-label">{termination[0]}</p>
          {termination[1].length > 240 ? <><p>{termination[1].slice(0,240)}…</p><details className="run-identifiers"><summary>完整终止原因与未解决问题</summary><p>{termination[1]}</p></details></> : <p>{termination[1]}</p>}
        </div>
      </section>

      {comparisonBundle && <ComparisonPanel primary={bundle} secondary={comparisonBundle} />}

      <ExecutionPanel bundle={bundle} />

      <details className="investigation-section">
      <summary>证据与调查图</summary>
      <div className="investigation-layout">
        <div className="left-column">
          <section className="ruled-section" aria-labelledby="graph-title">
            <SectionHeading title="Investigation graph" count={bundle.topology.nodes.length} />
            <p className="section-note" id="graph-title">
              Observed topology is solid. Candidate causal paths are dashed vermilion.
            </p>
            <InvestigationGraph
              topology={bundle.topology}
              paths={paths}
              selectedEntity={selectedEntity}
              onSelectEntity={setSelectedEntity}
            />
          </section>

          <section className="ruled-section evidence-index" aria-labelledby="evidence-title">
            <SectionHeading title="Evidence ledger" count={evidenceRecords.length} />
            <p className="section-note" id="evidence-title">Select a record to inspect its provenance and relationship to hypotheses.</p>
            {evidenceRecords.length ? (
              <div className="evidence-list">
                {evidenceRecords.map((record) => {
                  const evidence = record.payload as EvidencePayload;
                  return (
                    <button
                      aria-pressed={selectedEvidenceId === evidence.evidence_id}
                      className={selectedEvidenceId === evidence.evidence_id ? "evidence-row selected" : "evidence-row"}
                      key={evidence.evidence_id}
                      onClick={() => selectEvidence(record)}
                      type="button"
                    >
                      <span>R{record.revision.toString().padStart(3, "0")}</span>
                      <strong>{asText(evidence.predicate, "Evidence")}</strong>
                      <small>{(evidence.subject_ids ?? []).map(shortId).join(", ") || "No subject"}</small>
                    </button>
                  );
                })}
              </div>
            ) : <p className="empty-state">No evidence records.</p>}
          </section>
        </div>

        <Timeline items={timeline} selectedKey={selectedTimelineKey} onSelect={selectTimeline} />

        <div className="right-column">
          <section className="ruled-section" aria-labelledby="detail-title">
            <SectionHeading title="Evidence detail" />
            <p className="section-note" id="detail-title">Source, coverage, support, opposition, and unresolved gaps.</p>
            <EvidenceDetail
              record={selectedEvidence}
              observations={bundle.observations}
              candidates={candidates}
              reportGaps={bundle.report.gaps ?? []}
            />
          </section>

          <section className="ruled-section" aria-labelledby="candidate-title">
            <SectionHeading title="Hypotheses & gaps" count={candidates.length} />
            <p className="section-note" id="candidate-title">Candidate statements are not confirmed findings.</p>
            {candidates.length ? candidates.map(({ hypothesis, selected }) => {
              const relation = selectedEvidenceId ? relationFor(hypothesis, selectedEvidenceId) : undefined;
              return (
                <article className={relation ? `hypothesis ${relation}` : "hypothesis"} key={hypothesis.hypothesis_id}>
                  <div className="hypothesis-status">
                    <StatusMark status={hypothesis.status ?? "open"} />
                    <strong>{displayStatus(hypothesis.status ?? "open")}</strong>
                    {selected && <span>SELECTED</span>}
                  </div>
                  <h3>{hypothesis.statement ?? "No statement recorded"}</h3>
                  <code>{hypothesis.hypothesis_id}</code>
                  <DetailRows rows={[["Subject", hypothesis.subject], ["Linked evidence", relation ?? "none"]]} />
                  {(hypothesis.open_questions ?? []).map((question) => <p className="open-question" key={question}>{question}</p>)}
                </article>
              );
            }) : <p className="empty-state">No hypotheses were recorded.</p>}
            {bundle.report.unresolved_questions?.map((question) => (
              <p className="open-question" key={question}>{question}</p>
            ))}
          </section>
        </div>
      </div>
      </details>
    </main>
  );
}

export default function App() {
  const [bundle, setBundle] = useState<RunBundle>();
  const [comparisonBundle, setComparisonBundle] = useState<RunBundle>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(true);
  const [comparisonLoading, setComparisonLoading] = useState(false);

  const loadExample = useCallback(() => {
    setLoading(true);
    setError(undefined);
    loadExampleBundle()
      .then(setBundle)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : "Could not read the recorded example."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(loadExample, [loadExample]);

  const openFiles = async (
    event: ChangeEvent<HTMLInputElement>,
    target: "primary" | "comparison",
  ) => {
    if (!event.target.files?.length) return;
    target === "primary" ? setLoading(true) : setComparisonLoading(true);
    setError(undefined);
    try {
      const parsed = await parseBundleFiles(event.target.files);
      target === "primary" ? setBundle(parsed) : setComparisonBundle(parsed);
    } catch (reason) {
      const side = target === "primary" ? "primary" : "comparison";
      setError(reason instanceof Error ? reason.message : `Could not read the ${side} RunBundle files.`);
    } finally {
      target === "primary" ? setLoading(false) : setComparisonLoading(false);
      event.target.value = "";
    }
  };

  return (
    <>
      <header className="app-header">
        <a className="wordmark" href="#top" aria-label="RootWalk viewer home">ROOT/WALK</a>
        <p>Read-only RunBundle viewer</p>
        <div className="file-controls">
          <label className="file-control">
            <span>打开运行文件夹</span>
            <input
              aria-label="打开完整 RunBundle 文件夹"
              {...{ webkitdirectory: "" }}
              multiple
              onChange={(event) => openFiles(event, "primary")}
              type="file"
            />
          </label>
          <label className="file-control">
            <span>Replace primary</span>
            <input
              accept=".json,.jsonl,application/json,application/x-ndjson"
              aria-label="Open primary RunBundle files"
              multiple
              onChange={(event) => openFiles(event, "primary")}
              type="file"
            />
          </label>
          <label className="file-control compare-control">
            <span>Compare run</span>
            <input
              accept=".json,.jsonl,application/json,application/x-ndjson"
              aria-label="Open comparison RunBundle files"
              multiple
              onChange={(event) => openFiles(event, "comparison")}
              type="file"
            />
          </label>
          {comparisonBundle && <button className="clear-comparison" type="button" onClick={() => setComparisonBundle(undefined)}>Clear</button>}
        </div>
      </header>
      <div id="top" className="load-state" aria-live="polite">
        {loading && <p><span className="loading-mark" aria-hidden="true" />Loading recorded RunBundle…</p>}
        {comparisonLoading && <p><span className="loading-mark" aria-hidden="true" />Loading comparison RunBundle…</p>}
        {error && (
          <div className="error-state" role="alert">
            <p><strong>Bundle not loaded.</strong> {error}</p>
            <button type="button" onClick={loadExample}>Reload recorded example</button>
          </div>
        )}
      </div>
      {bundle && !loading && (
        <RunViewer
          key={bundle.sourceLabel + bundle.report.run_id}
          bundle={bundle}
          comparisonBundle={comparisonBundle}
        />
      )}
      <footer>
        <span>ROOTWALK M3</span>
        <span>Local data only · no bundle mutation</span>
      </footer>
    </>
  );
}
