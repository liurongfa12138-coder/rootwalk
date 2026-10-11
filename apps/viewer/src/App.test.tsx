// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import { RunViewer } from "./App";
import type { RunBundle } from "./types";

vi.mock("cytoscape", () => ({
  default: () => ({
    destroy: vi.fn(),
    fit: vi.fn(),
    getElementById: () => ({ addClass: vi.fn() }),
    nodes: () => ({ removeClass: vi.fn() }),
    on: vi.fn(),
    resize: vi.fn(),
  }),
}));

class ResizeObserverStub {
  observe() {}
  disconnect() {}
}

globalThis.ResizeObserver = ResizeObserverStub as unknown as typeof ResizeObserver;
(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const record = (revision: number, id: string, predicate: string) => ({
  kind: "evidence",
  revision,
  stable_key: id,
  payload: {
    evidence_id: id,
    predicate,
    subject_ids: [revision === 1 ? "service:a" : "service:b"],
    observation_refs: revision === 1 ? ["observation-1", "observation-2"] : [],
    artifact_refs: [{ content_hash: `hash-${revision}`, visibility: "model_readable", selector: { row: revision } }],
    value: { status: "success", value: revision },
  },
});

const bundle: RunBundle = {
  manifest: {},
  task: { task_id: "task-1" },
  topology: {
    nodes: [{ canonical_id: "service:a" }, { canonical_id: "service:b" }],
    edges: [{ edge_id: "edge-1", source_id: "service:a", target_id: "service:b", relation_type: "calls" }],
  },
  report: {
    run_id: "run-1",
    task_id: "task-1",
    status: "completed",
    gaps: [{ kind: "coverage", detail: "report gap" }],
    candidates: [{
      hypothesis: {
        hypothesis_id: "hypothesis-1",
        subject: "service:a",
        statement: "Candidate statement",
        status: "inconclusive",
        supporting_evidence_refs: ["evidence-one"],
        opposing_evidence_refs: [],
        open_questions: ["linked gap"],
      },
      paths: [{
        path_id: "path-1",
        root_cause_entity_id: "service:a",
        symptom_entity_ids: ["service:b"],
        steps: [{ from_entity: "service:a", to_entity: "service:b", mechanism: "queue pressure", status: "proposed" }],
      }],
    }],
  },
  evidence: [record(1, "evidence-one", "first.metric"), record(2, "evidence-two", "second.metric")],
  observations: [
    { kind: "observation", revision: 3, stable_key: "observation-1", payload: { observation_id: "observation-1", coverage: { basis: "coverage-one" } } },
    { kind: "observation", revision: 4, stable_key: "observation-2", payload: { observation_id: "observation-2", coverage: { basis: "coverage-two" } } },
  ],
  decisions: [{ kind: "decision", revision: 5, stable_key: "decision-1", payload: { decision_summary: "Inspect service", operations: [{ type: "query" }] } }],
  events: [],
  attempts: [],
  budgets: [],
  contexts: [],
  sourceLabel: "test",
};

afterEach(() => {
  document.body.replaceChildren();
});

describe("RunViewer interactions", () => {
  it("shows the evidence selected from the ledger", async () => {
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => root.render(<RunViewer bundle={bundle} />));

    const evidenceButtons = host.querySelectorAll<HTMLButtonElement>(".evidence-row");
    expect(host.querySelector('[data-testid="evidence-detail"]')?.textContent).toContain("first.metric");
    expect(host.querySelector('[data-testid="evidence-detail"]')?.textContent).toContain("hash-1");
    expect(host.querySelector('[data-testid="evidence-detail"]')?.textContent).toContain("model_readable");
    expect(host.querySelector('[data-testid="evidence-detail"]')?.textContent).toContain("coverage-one");
    expect(host.querySelector('[data-testid="evidence-detail"]')?.textContent).toContain("coverage-two");
    expect(host.querySelector('[data-testid="evidence-detail"]')?.textContent).toContain("report gap");
    expect(evidenceButtons[0].getAttribute("aria-pressed")).toBe("true");
    await act(async () => evidenceButtons[1].click());
    expect(host.querySelector('[data-testid="evidence-detail"]')?.textContent).toContain("second.metric");
    expect(evidenceButtons[1].getAttribute("aria-pressed")).toBe("true");

    const timelineButton = host.querySelector<HTMLButtonElement>(".timeline-item")!;
    await act(async () => timelineButton.click());
    expect(timelineButton.getAttribute("aria-current")).toBe("step");
    expect(timelineButton.getAttribute("aria-pressed")).toBe("true");

    const entityButton = host.querySelectorAll<HTMLButtonElement>(".entity-button")[1];
    await act(async () => entityButton.click());
    expect(entityButton.getAttribute("aria-pressed")).toBe("true");
    expect(host.textContent).toContain("a calls b");
    expect(host.textContent).toContain("queue pressure");

    await act(async () => root.unmount());
  });

  it("renders the first observable divergence for a second bundle", async () => {
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    const comparison = { ...bundle, report: { ...bundle.report, run_id: "run-2" }, decisions: [] };
    await act(async () => root.render(<RunViewer bundle={bundle} comparisonBundle={comparison} />));

    expect(host.querySelectorAll(".comparison-layers li")).toHaveLength(6);
    expect(host.querySelector(".comparison-result")?.textContent).toContain("decision · INDEX 0");
    expect(host.querySelector(".divergence-values")?.textContent).toContain("Inspect service");

    await act(async () => root.unmount());
  });

  it("reports a non-monotonic comparison bundle without crashing the viewer", async () => {
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    const comparison = { ...bundle, evidence: [...bundle.evidence].reverse() };
    await act(async () => root.render(<RunViewer bundle={bundle} comparisonBundle={comparison} />));

    expect(host.querySelector('[role="alert"]')?.textContent).toContain(
      "evidence.jsonl:2 revision 1 is not strictly after 2",
    );
    expect(host.textContent).toContain("Comparison unavailable");

    await act(async () => root.unmount());
  });
});
