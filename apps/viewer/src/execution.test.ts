import { describe, expect, it } from "vitest";
import { executionRounds, usageOf } from "./execution";
import { parseBundleText } from "./bundle";

const record = (kind: string, revision: number, payload: Record<string, unknown>) => ({ kind, revision, stable_key: `${kind}-${revision}`, payload });
const base = () => parseBundleText(new Map([
  ["manifest.json", "{}"], ["task.json", "{}"], ["topology.json", '{"nodes":[],"edges":[]}'],
  ["report.json", '{"run_id":"run","status":"runtime_failure"}'],
]));

describe("execution rounds", () => {
  it("keeps failed rounds without decisions and joins provider attempts by call number", () => {
    const bundle = base();
    bundle.contexts = [record("context", 1, { call_number: 1 }), record("context", 5, { call_number: 2 })];
    bundle.decisions = [record("decision", 3, { decision_id: "decision:run:0001", decision_summary: "inspect" })];
    bundle.attempts = [record("attempt", 6, { attempt_id: "model:2", status: "started" }), record("attempt", 8, { attempt_id: "model:2", status: "outcome_unknown", usage: { prompt_tokens: 9, completion_tokens: 15 } })];
    bundle.modelRequests = [record("model_request", 7, { call_number: 2, status: "invalid_response" })];
    const rounds = executionRounds(bundle);
    expect(rounds).toHaveLength(2);
    expect(rounds[1].decision).toBeUndefined();
    expect(rounds[1].requests).toHaveLength(1);
    expect(usageOf(rounds[1].attempt.usage).total).toBe(24);
  });

  it("joins legacy tool start/finish records without counting them twice", () => {
    const bundle = base();
    bundle.decisions = [record("decision", 2, { decision_id: "decision:run:0001" })];
    bundle.attempts = [record("attempt", 3, { attempt_id: "tool-1", operation_id: ["run", "decision:run:0001", 0], tool: "query_metric", status: "started" }), record("attempt", 4, { attempt_id: "tool-1", status: "success", duration_ms: 25 })];
    bundle.observations = [record("observation", 5, { attempt_id: "tool-1", summary: "metric returned" })];
    const tool = executionRounds(bundle)[0].tools[0];
    expect(tool.attempt.tool).toBe("query_metric");
    expect(tool.attempt.duration_ms).toBe(25);
    expect(tool.observation?.summary).toBe("metric returned");
    expect(executionRounds(bundle)[0].tools).toHaveLength(1);
  });

  it("does not represent missing token usage as zero or double-count cached tokens", () => {
    expect(usageOf({}).total).toBeUndefined();
    expect(usageOf({ prompt_tokens: 100, completion_tokens: 20, cache_read_input_tokens: 80 }).total).toBe(120);
  });
});
