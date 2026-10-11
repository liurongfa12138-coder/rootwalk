import type {
  BundleRecord,
  BundleComparison,
  ComparisonLayerName,
  JsonObject,
  RunBundle,
  RunReport,
  TimelineItem,
  Topology,
} from "./types";

const JSON_FILES = ["manifest.json", "task.json", "topology.json", "report.json"] as const;
const ARTIFACT_INDEX_FILE = "artifact_index.json";
const STREAM_FILES = {
  evidence: "evidence.jsonl",
  observations: "observations.jsonl",
  decisions: "decisions.jsonl",
  events: "investigation_events.jsonl",
  attempts: "attempts.jsonl",
  budgets: "budget_ledger.jsonl",
  contexts: "context_manifests.jsonl",
  modelRequests: "model_requests.jsonl",
  executionAudit: "execution_audit.jsonl",
} as const;

const isObject = (value: unknown): value is JsonObject =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value);

function parseObject(source: string, filename: string): JsonObject {
  let value: unknown;
  try {
    value = JSON.parse(source);
  } catch {
    throw new Error(`${filename} is not valid JSON.`);
  }
  if (!isObject(value)) throw new Error(`${filename} must contain a JSON object.`);
  return value;
}

function parseArtifactIndex(source: string): JsonObject[] {
  let value: unknown;
  try {
    value = JSON.parse(source);
  } catch {
    throw new Error(`${ARTIFACT_INDEX_FILE} is not valid JSON.`);
  }
  if (!Array.isArray(value) || !value.every(isObject)) {
    throw new Error(`${ARTIFACT_INDEX_FILE} must contain an array of objects.`);
  }
  return value;
}

async function sha256Hex(content: string | ArrayBuffer): Promise<string> {
  if (!globalThis.crypto?.subtle) {
    throw new Error("当前浏览器不支持 SHA-256，无法验证原始内容。");
  }
  const bytes = typeof content === "string" ? new TextEncoder().encode(content) : content;
  const digest = await globalThis.crypto.subtle.digest("SHA-256", bytes);
  return [...new Uint8Array(digest)].map((byte) => byte.toString(16).padStart(2, "0")).join("");
}

function artifactReader(
  manifest: JsonObject,
  artifactIndexSource: string | undefined,
  load: (bundlePath: string) => Promise<string | ArrayBuffer>,
) {
  return async (hash: string): Promise<string> => {
    validateArtifactHash(hash);
    if (!artifactIndexSource?.trim()) {
      throw new Error("此 RunBundle 未加载 artifact_index.json；旧版 bundle 可继续浏览，但无法验证原始内容。");
    }
    const fileHashes = isObject(manifest.file_hashes) ? manifest.file_hashes : undefined;
    const indexHash = fileHashes?.[ARTIFACT_INDEX_FILE];
    if (
      manifest.artifact_index !== ARTIFACT_INDEX_FILE ||
      typeof indexHash !== "string" ||
      await sha256Hex(artifactIndexSource) !== indexHash
    ) {
      throw new Error("RunBundle 的 artifact_index.json 完整性校验失败。");
    }
    const matches = parseArtifactIndex(artifactIndexSource).filter(
      (entry) => entry.content_hash === hash,
    );
    const expectedPath = `artifacts/${hash}`;
    if (
      matches.length !== 1 ||
      matches[0].storage !== "embedded" ||
      matches[0].bundle_path !== expectedPath ||
      fileHashes?.[expectedPath] !== hash
    ) {
      throw new Error("原始内容不是 RunBundle 中声明的 embedded artifact。");
    }
    const raw = await load(expectedPath);
    if (await sha256Hex(raw) !== hash) {
      throw new Error("原始内容 SHA-256 与引用不一致。");
    }
    return typeof raw === "string" ? raw : new TextDecoder().decode(raw);
  };
}

export function parseJsonl(source: string, filename = "stream.jsonl"): BundleRecord[] {
  return source
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      const value = parseObject(line, `${filename}:${index + 1}`);
      if (
        typeof value.kind !== "string" ||
        typeof value.revision !== "number" ||
        typeof value.stable_key !== "string" ||
        !isObject(value.payload)
      ) {
        throw new Error(`${filename}:${index + 1} is not a revisioned RunBundle record.`);
      }
      return value as unknown as BundleRecord;
    });
}

function requireObject<T extends JsonObject>(value: JsonObject, filename: string): T {
  return value as T;
}

export function parseBundleText(files: Map<string, string>, sourceLabel = "Local files"): RunBundle {
  for (const filename of JSON_FILES) {
    if (!files.has(filename)) throw new Error(`Missing required bundle file: ${filename}`);
  }

  const manifest = parseObject(files.get("manifest.json")!, "manifest.json");
  const task = parseObject(files.get("task.json")!, "task.json");
  const topology = requireObject<Topology>(
    parseObject(files.get("topology.json")!, "topology.json"),
    "topology.json",
  );
  const report = requireObject<RunReport>(
    parseObject(files.get("report.json")!, "report.json"),
    "report.json",
  );

  if (!Array.isArray(topology.nodes) || !Array.isArray(topology.edges)) {
    throw new Error("topology.json must contain nodes and edges arrays.");
  }
  if (typeof report.run_id !== "string" || typeof report.status !== "string") {
    throw new Error("report.json must contain run_id and status.");
  }

  const stream = (filename: string) => parseJsonl(files.get(filename) ?? "", filename);
  const bundle: RunBundle = {
    manifest,
    task,
    topology,
    report,
    evidence: stream(STREAM_FILES.evidence),
    observations: stream(STREAM_FILES.observations),
    decisions: stream(STREAM_FILES.decisions),
    events: stream(STREAM_FILES.events),
    attempts: stream(STREAM_FILES.attempts),
    budgets: stream(STREAM_FILES.budgets),
    contexts: stream(STREAM_FILES.contexts),
    modelRequests: stream(STREAM_FILES.modelRequests),
    executionAudit: stream(STREAM_FILES.executionAudit),
    sourceLabel,
  };
  bundle.readArtifact = artifactReader(
    manifest,
    files.get(ARTIFACT_INDEX_FILE),
    async (bundlePath) => {
      const content = files.get(bundlePath) ?? files.get(bundlePath.split("/").at(-1)!);
      if (content === undefined) throw new Error("未加载原始内容。请打开包含 artifacts 的完整 RunBundle 文件夹。");
      return content;
    },
  );
  return bundle;
}

export async function parseBundleFiles(files: FileList | File[]): Promise<RunBundle> {
  const selected = Array.from(files);
  if (selected.filter((file) => file.name === "manifest.json").length !== 1) {
    throw new Error("请选择一个 RunBundle 文件夹，其中应只有一个 manifest.json。");
  }
  const names = new Set<string>([...JSON_FILES, ARTIFACT_INDEX_FILE, ...Object.values(STREAM_FILES)]);
  const texts = await Promise.all(selected.filter((file) => names.has(file.name)).map(async (file) => [file.name, await file.text()] as const));
  const bundle = parseBundleText(new Map(texts), `${selected.length} local files`);
  const artifacts = new Map(selected.filter((file) => /^[a-f0-9]{64}$/.test(file.name)).map((file) => [file.name, file]));
  bundle.readArtifact = artifactReader(bundle.manifest, texts.find(([name]) => name === ARTIFACT_INDEX_FILE)?.[1], async (bundlePath) => {
    const file = artifacts.get(bundlePath.split("/").at(-1)!);
    if (!file) throw new Error("未加载原始内容。请打开包含 artifacts 的完整 RunBundle 文件夹。");
    return file.arrayBuffer();
  });
  return bundle;
}

function validateArtifactHash(hash: string) {
  if (!/^[a-f0-9]{64}$/.test(hash)) throw new Error("原始内容的引用无效。");
}

export async function loadExampleBundle(base = "/example-run"): Promise<RunBundle> {
  const filenames = [...JSON_FILES, ARTIFACT_INDEX_FILE, ...Object.values(STREAM_FILES)];
  const entries = await Promise.all(
    filenames.map(async (filename) => {
      const response = await fetch(`${base}/${filename}`, { cache: "no-store" });
      if ((filename === ARTIFACT_INDEX_FILE || filename === STREAM_FILES.modelRequests || filename === STREAM_FILES.executionAudit) && (response.status === 404 || response.headers.get("content-type")?.includes("text/html"))) return [filename, ""] as const;
      if (!response.ok) throw new Error(`Could not load ${filename} (${response.status}).`);
      return [filename, await response.text()] as const;
    }),
  );
  const bundle = parseBundleText(new Map(entries), "Recorded example");
  bundle.readArtifact = artifactReader(bundle.manifest, new Map(entries).get(ARTIFACT_INDEX_FILE), async (bundlePath) => {
    const response = await fetch(`${base}/${bundlePath}`);
    if (!response.ok) throw new Error("无法读取原始内容，请检查 RunBundle 的 artifacts 文件夹。");
    return response.arrayBuffer();
  });
  return bundle;
}

const payloadString = (value: unknown) => (typeof value === "string" ? value : undefined);

export function buildTimeline(bundle: RunBundle): TimelineItem[] {
  const items: TimelineItem[] = [];
  for (const record of bundle.decisions) {
    const operations = Array.isArray(record.payload.operations) ? record.payload.operations : [];
    items.push({
      ...record,
      timelineKind: "decision",
      label: payloadString(record.payload.decision_summary) ?? "Decision",
      operationTypes: operations
        .filter(isObject)
        .map((operation) => payloadString(operation.type) ?? "operation"),
    });
  }
  for (const record of bundle.attempts) {
    const tool = payloadString(record.payload.tool) ?? payloadString(record.payload.attempt_id) ?? "Attempt";
    const status = payloadString(record.payload.status);
    items.push({ ...record, timelineKind: "attempt", label: status ? `${tool} · ${status}` : tool, operationTypes: [] });
  }
  for (const record of bundle.observations) {
    const tool = payloadString(record.payload.tool) ?? "Observation";
    const status = payloadString(record.payload.status);
    items.push({ ...record, timelineKind: "observation", label: status ? `${tool} · ${status}` : tool, operationTypes: [] });
  }
  for (const record of bundle.evidence) {
    items.push({
      ...record,
      timelineKind: "evidence",
      label: payloadString(record.payload.predicate) ?? "Evidence captured",
      operationTypes: [],
    });
  }
  for (const record of bundle.events) {
    items.push({
      ...record,
      timelineKind: "event",
      label: (payloadString(record.payload.event_type) ?? "Investigation event").split("_").join(" "),
      operationTypes: [],
    });
  }
  for (const record of bundle.budgets) {
    const event = payloadString(record.payload.event)?.split("_").join(" ") ?? "Budget update";
    const used = typeof record.payload.model_calls_used === "number" ? record.payload.model_calls_used : undefined;
    const maximum = typeof record.payload.max_model_calls === "number" ? record.payload.max_model_calls : undefined;
    const count = used === undefined ? "" : ` · ${used}${maximum === undefined ? "" : `/${maximum}`} calls`;
    items.push({ ...record, timelineKind: "budget", label: `${event}${count}`, operationTypes: [] });
  }
  for (const record of bundle.contexts) {
    const call = typeof record.payload.call_number === "number" ? ` #${record.payload.call_number}` : "";
    const status = payloadString(record.payload.status);
    items.push({
      ...record,
      timelineKind: "context",
      label: `Model context${call}${status ? ` · ${status}` : ""}`,
      operationTypes: [],
    });
  }
  return items.sort((a, b) => a.revision - b.revision || a.stable_key.localeCompare(b.stable_key));
}

export function collectStringValues(value: unknown, into = new Set<string>()): Set<string> {
  if (typeof value === "string") into.add(value);
  else if (Array.isArray(value)) value.forEach((item) => collectStringValues(item, into));
  else if (isObject(value)) Object.values(value).forEach((item) => collectStringValues(item, into));
  return into;
}

const LAYER_ORDER: ComparisonLayerName[] = [
  "environment_manifest",
  "context_manifest",
  "decision",
  "observation",
  "evidence",
  "termination",
];
const CONTEXT_REF_FIELDS = [
  "observation_ids",
  "evidence_ids",
  "hypothesis_ids",
  "path_ids",
  "gap_refs",
  "feedback_refs",
] as const;

const VOLATILE_MANIFEST_FIELDS = new Set([
  "artifact_index",
  "deadline_at",
  "domain_cursor",
  "file_hashes",
  "run_id",
]);

function canonical(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(canonical);
  if (!isObject(value)) return value;
  return Object.fromEntries(Object.keys(value).sort().map((key) => [key, canonical(value[key])]));
}

const sameValue = (left: unknown, right: unknown) =>
  JSON.stringify(canonical(left)) === JSON.stringify(canonical(right));

function contextRefs(value: unknown): JsonObject {
  if (!isObject(value)) return {};
  return Object.fromEntries(
    CONTEXT_REF_FIELDS.flatMap((field) => {
      const items = value[field];
      return Array.isArray(items)
        ? [[field, items.filter((item): item is string => typeof item === "string")]]
        : [];
    }),
  );
}

function comparisonValue(layer: ComparisonLayerName, value: unknown): unknown {
  if (layer !== "context_manifest" || !isObject(value)) return value;
  const normalized: JsonObject = { ...value };
  for (const field of ["selected_refs", "omitted_refs"] as const) {
    if (field in normalized) normalized[field] = contextRefs(normalized[field]);
  }
  return normalized;
}

function payloadsInRevisionOrder(records: BundleRecord[], filename: string): JsonObject[] {
  let previousRevision: number | undefined;
  return records.map((record, index) => {
    if (!Number.isSafeInteger(record.revision)) {
      throw new Error(`${filename}:${index + 1} has invalid revision.`);
    }
    if (previousRevision !== undefined && record.revision <= previousRevision) {
      throw new Error(
        `${filename}:${index + 1} revision ${record.revision} is not strictly after ${previousRevision}.`,
      );
    }
    previousRevision = record.revision;
    return record.payload;
  });
}

function pick(value: unknown, fields: string[]): unknown {
  if (!isObject(value)) return value;
  return Object.fromEntries(fields.filter((field) => field in value).map((field) => [field, value[field]]));
}

function comparisonSummary(layer: ComparisonLayerName, value: unknown): unknown {
  if (value == null || !isObject(value)) return value;
  if (layer === "environment_manifest") {
    const manifest = isObject(value.manifest) ? value.manifest : {};
    const task = isObject(value.task) ? value.task : {};
    const topology = isObject(value.topology) ? value.topology : {};
    return {
      environment: task.environment,
      task_id: task.task_id,
      data_version: manifest.data_version,
      model_provider: manifest.model_provider,
      model_id: manifest.model_id,
      topology_id: topology.snapshot_id,
    };
  }
  const fields: Record<Exclude<ComparisonLayerName, "environment_manifest">, string[]> = {
    context_manifest: [
      "call_number",
      "status",
      "strategy",
      "message_hash",
      "estimated_tokens",
      "selected_refs",
      "omitted_refs",
    ],
    decision: ["decision_id", "decision_summary", "intent", "operations"],
    observation: ["observation_id", "tool", "arguments", "status", "summary", "artifact_refs"],
    evidence: [
      "evidence_id",
      "observation_refs",
      "subject_ids",
      "predicate",
      "value",
      "unit",
      "observed_interval",
      "onset_interval",
      "time_uncertainty",
      "artifact_refs",
      "extractor",
      "source_fingerprint",
      "dataset_snapshot_id",
    ],
    termination: ["status", "failure", "abstain_reason", "hypothesis_ids", "path_ids", "unresolved_questions"],
  };
  return pick(value, fields[layer]);
}

function comparisonLayers(bundle: RunBundle): Record<ComparisonLayerName, unknown[]> {
  const manifest = Object.fromEntries(
    Object.entries(bundle.manifest).filter(([key]) => !VOLATILE_MANIFEST_FIELDS.has(key)),
  );
  return {
    environment_manifest: [{ manifest, task: bundle.task, topology: bundle.topology }],
    context_manifest: payloadsInRevisionOrder(bundle.contexts, STREAM_FILES.contexts),
    decision: payloadsInRevisionOrder(bundle.decisions, STREAM_FILES.decisions),
    observation: payloadsInRevisionOrder(bundle.observations, STREAM_FILES.observations),
    evidence: payloadsInRevisionOrder(bundle.evidence, STREAM_FILES.evidence),
    termination: [bundle.report],
  };
}

export function compareRunBundles(left: RunBundle, right: RunBundle): BundleComparison {
  const leftLayers = comparisonLayers(left);
  const rightLayers = comparisonLayers(right);
  const layers = {} as BundleComparison["layers"];
  let firstDivergence: BundleComparison["firstDivergence"] = null;

  for (const layer of LAYER_ORDER) {
    const differences = [];
    const leftItems = leftLayers[layer];
    const rightItems = rightLayers[layer];
    for (let index = 0; index < Math.max(leftItems.length, rightItems.length); index += 1) {
      const leftValue = comparisonValue(layer, leftItems[index]);
      const rightValue = comparisonValue(layer, rightItems[index]);
      if (sameValue(leftValue, rightValue)) continue;
      const difference = {
        index,
        left: comparisonSummary(layer, leftValue),
        right: comparisonSummary(layer, rightValue),
      };
      differences.push(difference);
      if (!firstDivergence) firstDivergence = { layer, ...difference };
    }
    layers[layer] = { same: differences.length === 0, differences };
  }

  return { same: firstDivergence === null, firstDivergence, layers };
}
