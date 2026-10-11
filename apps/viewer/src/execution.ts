import type { BundleRecord, JsonObject, RunBundle } from "./types";

export const object = (value: unknown): JsonObject =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? value as JsonObject : {};
export const number = (value: unknown): number | undefined =>
  typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : undefined;
export const text = (value: unknown, fallback = "未记录") => typeof value === "string" && value ? value : fallback;
export const count = (value: unknown) => number(value)?.toLocaleString("en-US") ?? "未记录";
export const duration = (milliseconds: unknown) => {
  const value = number(milliseconds);
  return value === undefined ? "未记录" : value < 1000 ? `${value.toFixed(0)} ms` : `${(value / 1000).toFixed(2)} s`;
};

export function usageOf(value: unknown) {
  const usage = object(value);
  const input = number(usage.prompt_tokens ?? usage.input_tokens);
  const output = number(usage.completion_tokens ?? usage.output_tokens);
  return {
    input, output,
    total: number(usage.total_tokens) ?? (input !== undefined && output !== undefined ? input + output : undefined),
    cached: number(usage.cache_read_input_tokens ?? usage.cached_tokens ?? object(usage.prompt_tokens_details).cached_tokens),
    reasoning: number(usage.reasoning_tokens ?? object(usage.completion_tokens_details).reasoning_tokens),
  };
}

export interface ToolExecution {
  id: string;
  attempt: JsonObject;
  observation?: JsonObject;
}

export interface ExecutionRound {
  number: number;
  revision: number;
  context?: JsonObject;
  decision?: JsonObject;
  attempt: JsonObject;
  requests: JsonObject[];
  tools: ToolExecution[];
  events: JsonObject[];
}

export function executionRounds(bundle: RunBundle): ExecutionRound[] {
  const rounds = new Map<number, ExecutionRound>();
  const ensure = (n: number, revision: number) => {
    if (!rounds.has(n)) rounds.set(n, { number: n, revision, attempt: {}, requests: [], tools: [], events: [] });
    return rounds.get(n)!;
  };
  const decisionNumbers = new Map<string, number>();
  const sorted = (records: BundleRecord[]) => [...records].sort((a, b) => a.revision - b.revision);
  for (const record of sorted(bundle.contexts)) {
    const n = number(record.payload.call_number);
    if (n !== undefined) ensure(n, record.revision).context = record.payload;
  }
  sorted(bundle.decisions).forEach((record, index) => {
    const id = text(record.payload.decision_id, record.stable_key);
    const n = number(record.payload.call_number) ?? Number(id.match(/:(\d+)$/)?.[1] ?? index + 1);
    ensure(n, record.revision).decision = record.payload;
    decisionNumbers.set(id, n);
  });
  const tools = new Map<string, { payload: JsonObject; revision: number }>();
  for (const record of sorted(bundle.attempts)) {
    const id = text(record.payload.attempt_id, record.stable_key);
    const modelCall = id.match(/^model:(\d+)$/);
    if (modelCall) {
      const round = ensure(Number(modelCall[1]), record.revision);
      round.attempt = { ...round.attempt, ...record.payload };
    } else {
      const prior = tools.get(id);
      tools.set(id, { payload: { ...prior?.payload, ...record.payload }, revision: prior?.revision ?? record.revision });
    }
  }
  for (const record of sorted(bundle.modelRequests ?? [])) {
    const n = number(record.payload.call_number);
    if (n !== undefined) ensure(n, record.revision).requests.push(record.payload);
  }
  for (const [id, item] of tools) {
    const operation = item.payload.operation_id;
    const decision = text(item.payload.decision_id, Array.isArray(operation) ? text(operation[1], "") : "");
    const n = number(item.payload.call_number) ?? decisionNumbers.get(decision);
    if (n === undefined) continue;
    const observation = bundle.observations.find((record) => record.payload.attempt_id === id)?.payload;
    ensure(n, item.revision).tools.push({ id, attempt: item.payload, observation });
  }
  const ordered = [...rounds.values()].sort((a, b) => a.number - b.number);
  for (const record of sorted(bundle.events)) {
    const n = decisionNumbers.get(text(record.payload.decision_id, ""));
    const round = n === undefined ? [...ordered].reverse().find((item) => item.revision <= record.revision) : rounds.get(n);
    round?.events.push(record.payload);
  }
  return ordered;
}

export function roundStatus(round: ExecutionRound): string {
  if (round.attempt.status === "started" && !round.decision) return "interrupted";
  return text(round.attempt.status, round.decision ? "success" : text(round.context?.status));
}
