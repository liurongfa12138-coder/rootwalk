import { describe, expect, it } from "vitest";
import { buildTimeline, compareRunBundles, parseBundleText, parseJsonl } from "./bundle";

const required = new Map([
  ["manifest.json", JSON.stringify({ bundle_schema_version: "1", run_id: "run-1" })],
  ["task.json", JSON.stringify({ task_id: "task-1" })],
  ["topology.json", JSON.stringify({ nodes: [], edges: [] })],
  ["report.json", JSON.stringify({ run_id: "run-1", status: "completed" })],
]);

async function sha256(source: string) {
  const digest = await globalThis.crypto.subtle.digest("SHA-256", new TextEncoder().encode(source));
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function artifactBundle(content: string, stored = content, bundlePath?: string) {
  const hash = await sha256(content);
  const path = bundlePath ?? `artifacts/${hash}`;
  const index = JSON.stringify([{ content_hash: hash, storage: "embedded", bundle_path: path }]);
  const files = new Map(required);
  files.set("artifact_index.json", index);
  files.set("manifest.json", JSON.stringify({
    artifact_index: "artifact_index.json",
    file_hashes: { "artifact_index.json": await sha256(index), [path]: hash },
  }));
  files.set(path, stored);
  return { bundle: parseBundleText(files), files, hash, path };
}

describe("RunBundle parsing", () => {
  it("parses revisioned JSONL and orders the operation timeline", () => {
    const files = new Map(required);
    files.set(
      "decisions.jsonl",
      `${JSON.stringify({
        kind: "decision",
        payload: { decision_summary: "Second", operations: [{ type: "query" }] },
        revision: 9,
        stable_key: "decision-2",
      })}\n${JSON.stringify({
        kind: "decision",
        payload: { decision_summary: "First", operations: [{ type: "finalize" }] },
        revision: 2,
        stable_key: "decision-1",
      })}\n`,
    );

    const bundle = parseBundleText(files, "test");
    expect(buildTimeline(bundle).map((item) => [item.revision, item.label])).toEqual([
      [2, "First"],
      [9, "Second"],
    ]);
  });

  it("rejects malformed records with a file and line number", () => {
    expect(() => parseJsonl('{"kind":"evidence"}', "evidence.jsonl")).toThrow(
      "evidence.jsonl:1 is not a revisioned RunBundle record",
    );
  });

  it("verifies embedded artifact declarations and content hashes before reading", async () => {
    const valid = await artifactBundle('{"message":"recorded"}');
    await expect(valid.bundle.readArtifact?.(valid.hash)).resolves.toBe('{"message":"recorded"}');

    const tampered = await artifactBundle('{"message":"recorded"}', '{"message":"changed"}');
    await expect(tampered.bundle.readArtifact?.(tampered.hash)).rejects.toThrow("SHA-256");

    const tamperedIndex = await artifactBundle('{"message":"recorded"}');
    tamperedIndex.files.set("artifact_index.json", `${tamperedIndex.files.get("artifact_index.json")} `);
    await expect(parseBundleText(tamperedIndex.files).readArtifact?.(tamperedIndex.hash)).rejects.toThrow("artifact_index.json");

    const wrongPath = await artifactBundle('{"message":"recorded"}', undefined, "artifacts/not-the-hash");
    await expect(wrongPath.bundle.readArtifact?.(wrongPath.hash)).rejects.toThrow("embedded artifact");

    const wrongManifest = await artifactBundle('{"message":"recorded"}');
    const manifest = JSON.parse(wrongManifest.files.get("manifest.json")!);
    manifest.file_hashes[wrongManifest.path] = "0".repeat(64);
    wrongManifest.files.set("manifest.json", JSON.stringify(manifest));
    await expect(parseBundleText(wrongManifest.files).readArtifact?.(wrongManifest.hash)).rejects.toThrow("embedded artifact");
  });

  it("keeps legacy bundles readable but fails artifact inspection explicitly", async () => {
    const bundle = parseBundleText(new Map(required));
    await expect(bundle.readArtifact?.("a".repeat(64))).rejects.toThrow("旧版 bundle 可继续浏览");
  });

  it("includes budget and context records in revision order", () => {
    const files = new Map(required);
    files.set("budget_ledger.jsonl", JSON.stringify({
      kind: "budget_ledger",
      payload: { event: "model_call_started", model_calls_used: 1, max_model_calls: 5 },
      revision: 3,
      stable_key: "budget-1",
    }));
    files.set("context_manifests.jsonl", JSON.stringify({
      kind: "context_manifest",
      payload: { call_number: 1, status: "ready" },
      revision: 4,
      stable_key: "context-1",
    }));

    expect(buildTimeline(parseBundleText(files)).map((item) => item.timelineKind)).toEqual([
      "budget",
      "context",
    ]);
  });

  it("finds the first divergence across the same six protocol layers as the Python diff", () => {
    const leftFiles = new Map(required);
    leftFiles.set("context_manifests.jsonl", `${JSON.stringify({ kind: "context_manifest", payload: { call_number: 1, status: "ready" }, revision: 1, stable_key: "context-1" })}\n`);
    leftFiles.set("decisions.jsonl", `${JSON.stringify({ kind: "decision", payload: { decision_id: "one", decision_summary: "same" }, revision: 2, stable_key: "decision-1" })}\n${JSON.stringify({ kind: "decision", payload: { decision_id: "two", decision_summary: "inspect A" }, revision: 3, stable_key: "decision-2" })}\n`);
    const rightFiles = new Map(leftFiles);
    rightFiles.set("decisions.jsonl", `${JSON.stringify({ kind: "decision", payload: { decision_id: "one", decision_summary: "same" }, revision: 2, stable_key: "decision-1" })}\n`);

    const comparison = compareRunBundles(parseBundleText(leftFiles), parseBundleText(rightFiles));
    expect(Object.keys(comparison.layers)).toEqual([
      "environment_manifest",
      "context_manifest",
      "decision",
      "observation",
      "evidence",
      "termination",
    ]);
    expect(comparison.firstDivergence).toMatchObject({ layer: "decision", index: 1, right: undefined });
  });

  it("locates evidence divergence and preserves structured context references", () => {
    const commonEvidence = {
      evidence_id: "evidence-1",
      observation_refs: ["observation-1"],
      subject_ids: ["service:a"],
      predicate: "latency.high",
      value: { status: "observed", p99_ms: 800 },
      artifact_refs: [{ content_hash: "hash-1", visibility: "model_readable" }],
      source_fingerprint: "source-1",
    };
    const selectedRefs = { evidence_ids: ["evidence-1"], observation_ids: ["observation-1"] };
    const omittedRefs = { evidence_ids: [], observation_ids: [] };
    const leftFiles = new Map(required);
    leftFiles.set("context_manifests.jsonl", JSON.stringify({
      kind: "context_manifest",
      payload: { call_number: 1, status: "ready", selected_refs: selectedRefs, omitted_refs: omittedRefs },
      revision: 1,
      stable_key: "context-1",
    }));
    leftFiles.set("evidence.jsonl", JSON.stringify({
      kind: "evidence",
      payload: commonEvidence,
      revision: 2,
      stable_key: "evidence-1",
    }));

    const contextRightFiles = new Map(leftFiles);
    contextRightFiles.set("context_manifests.jsonl", JSON.stringify({
      kind: "context_manifest",
      payload: {
        call_number: 1,
        status: "ready",
        selected_refs: { ...selectedRefs, evidence_ids: [] },
        omitted_refs: { ...omittedRefs, evidence_ids: ["evidence-1"] },
      },
      revision: 1,
      stable_key: "context-1",
    }));
    const contextComparison = compareRunBundles(
      parseBundleText(leftFiles),
      parseBundleText(contextRightFiles),
    );
    expect(contextComparison.firstDivergence).toMatchObject({
      layer: "context_manifest",
      left: { selected_refs: selectedRefs, omitted_refs: omittedRefs },
      right: {
        selected_refs: { ...selectedRefs, evidence_ids: [] },
        omitted_refs: { ...omittedRefs, evidence_ids: ["evidence-1"] },
      },
    });

    const evidenceRightFiles = new Map(leftFiles);
    evidenceRightFiles.set("evidence.jsonl", JSON.stringify({
      kind: "evidence",
      payload: { ...commonEvidence, value: { status: "observed", p99_ms: 1200 } },
      revision: 2,
      stable_key: "evidence-1",
    }));
    const evidenceComparison = compareRunBundles(
      parseBundleText(leftFiles),
      parseBundleText(evidenceRightFiles),
    );
    expect(evidenceComparison.firstDivergence).toMatchObject({
      layer: "evidence",
      index: 0,
      left: { value: { status: "observed", p99_ms: 800 } },
      right: { value: { status: "observed", p99_ms: 1200 } },
    });
  });

  it("ignores non-reference context content and rejects non-monotonic evidence", () => {
    const leftFiles = new Map(required);
    leftFiles.set("context_manifests.jsonl", JSON.stringify({
      kind: "context_manifest",
      payload: {
        call_number: 1,
        selected_refs: {
          evidence_ids: ["evidence-1"],
          body: "private model input",
          path: "/private/context.json",
          selector: { line: 7 },
        },
        omitted_refs: {
          observation_ids: ["observation-1"],
          body: "private omitted input",
          path: "/private/omitted.json",
        },
      },
      revision: 1,
      stable_key: "context-1",
    }));
    const rightFiles = new Map(leftFiles);
    rightFiles.set("context_manifests.jsonl", JSON.stringify({
      kind: "context_manifest",
      payload: {
        call_number: 1,
        selected_refs: {
          evidence_ids: ["evidence-1"],
          body: "different private input",
          path: "/private/context.json",
          selector: { line: 7 },
        },
        omitted_refs: {
          observation_ids: ["observation-1"],
          body: "private omitted input",
          path: "/different/private.json",
        },
      },
      revision: 1,
      stable_key: "context-1",
    }));
    expect(compareRunBundles(parseBundleText(leftFiles), parseBundleText(rightFiles)).same).toBe(true);

    const unorderedFiles = new Map(leftFiles);
    unorderedFiles.set("evidence.jsonl", [
      JSON.stringify({ kind: "evidence", revision: 4, stable_key: "evidence-2", payload: { evidence_id: "evidence-2" } }),
      JSON.stringify({ kind: "evidence", revision: 3, stable_key: "evidence-1", payload: { evidence_id: "evidence-1" } }),
    ].join("\n"));
    const unordered = parseBundleText(unorderedFiles);
    expect(() => compareRunBundles(unordered, unordered)).toThrow(
      "evidence.jsonl:2 revision 3 is not strictly after 4",
    );
  });
});
