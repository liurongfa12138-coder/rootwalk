// @vitest-environment jsdom
import { act } from "react";
import { createRoot } from "react-dom/client";
import { describe, expect, it, vi } from "vitest";
import { ExecutionPanel } from "./ExecutionPanel";
import { parseBundleText } from "./bundle";

(globalThis as typeof globalThis & { IS_REACT_ACT_ENVIRONMENT: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

describe("call inspection", () => {
  it("switches rounds, shows failure usage, and loads escaped input only when opened", async () => {
    const hash = "a".repeat(64);
    const record = (kind: string, revision: number, payload: object) => JSON.stringify({ kind, revision, stable_key: `${kind}-${revision}`, payload });
    const bundle = parseBundleText(new Map([
      ["manifest.json", "{}"], ["task.json", "{}"], ["topology.json", '{"nodes":[],"edges":[]}'],
      ["report.json", '{"run_id":"test","status":"runtime_failure","model_usage":{"prompt_tokens":9,"completion_tokens":15}}'],
      ["context_manifests.jsonl", record("context", 1, { call_number: 1, estimated_tokens: 100, max_context_tokens: 1000, model_input_ref: hash }) + "\n" + record("context", 4, { call_number: 2, estimated_tokens: 200, max_context_tokens: 1000, model_input_ref: hash })],
      ["decisions.jsonl", record("decision", 2, { decision_id: "decision:test:0001", decision_summary: "Inspect metrics", operations: [] })],
      ["attempts.jsonl", record("attempt", 6, { attempt_id: "model:2", status: "outcome_unknown", duration_ms: 250, usage: { prompt_tokens: 9, completion_tokens: 15 }, summary: "Provider failed" })],
    ]));
    bundle.readArtifact = vi.fn(async () => JSON.stringify([{ role: "user", content: '<img src=x onerror="alert(1)">untrusted input' }]));
    const host = document.createElement("div"); document.body.append(host);
    const root = createRoot(host);
    await act(async () => root.render(<ExecutionPanel bundle={bundle} />));
    expect(host.textContent).toContain("Inspect metrics");
    expect(host.textContent).toContain("10.0%");
    expect(bundle.readArtifact).not.toHaveBeenCalled();
    await act(async () => host.querySelectorAll<HTMLButtonElement>(".round-list button")[1].click());
    expect(host.querySelector(".round-detail")?.textContent).toContain("Provider failed");
    expect(host.querySelector(".round-detail")?.textContent).toContain("250 ms");
    expect(host.querySelector(".round-detail")?.textContent).toContain("20.0%");
    const details = [...host.querySelectorAll<HTMLDetailsElement>(".execution-raw")].find((item) => item.querySelector("summary")?.textContent === "模型看到的完整上下文")!;
    await act(async () => { details.open = true; details.dispatchEvent(new Event("toggle")); });
    expect(bundle.readArtifact).toHaveBeenCalledWith(hash);
    expect(host.textContent).toContain("untrusted input");
    expect(host.querySelector("img")).toBeNull();
    await act(async () => root.unmount()); host.remove();
  });

  it("does not parse or render oversized artifact content", async () => {
    const hash = "b".repeat(64);
    const record = (kind: string, revision: number, payload: object) => JSON.stringify({ kind, revision, stable_key: `${kind}-${revision}`, payload });
    const bundle = parseBundleText(new Map([
      ["manifest.json", "{}"], ["task.json", "{}"], ["topology.json", '{"nodes":[],"edges":[]}'],
      ["report.json", '{"run_id":"test","status":"completed_partial"}'],
      ["context_manifests.jsonl", record("context", 1, { call_number: 1, model_input_ref: hash })],
    ]));
    const marker = "must-not-render";
    bundle.readArtifact = vi.fn(async () => JSON.stringify({ content: `${"x".repeat(200_000)}${marker}` }));
    const parse = vi.spyOn(JSON, "parse");
    const host = document.createElement("div"); document.body.append(host);
    const root = createRoot(host);
    await act(async () => root.render(<ExecutionPanel bundle={bundle} />));
    const details = [...host.querySelectorAll<HTMLDetailsElement>(".execution-raw")].find((item) => item.querySelector("summary")?.textContent === "模型看到的完整上下文")!;

    parse.mockClear();
    await act(async () => { details.open = true; details.dispatchEvent(new Event("toggle")); });

    expect(parse).not.toHaveBeenCalled();
    expect(host.textContent).toContain("超过页面预览上限，未解析或渲染");
    expect(host.textContent).not.toContain(marker);
    parse.mockRestore();
    await act(async () => root.unmount()); host.remove();
  });
});
