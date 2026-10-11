# Inspecting AI calls

For a terminal-first audit, inspect a sealed bundle without calling a model or
tool:

```bash
rootwalk inspect /absolute/path/to/run-bundle \
  --format markdown \
  --output /tmp/rootwalk-investigation.md

rootwalk inspect /absolute/path/to/run-bundle --call 2
```

The JSON form includes each compiled context, model decision, provider request,
actual tool attempt, observation, evidence record, Runtime binding, budget event,
and its source file and line. It also reconstructs the candidate funnel and flags
common failure shapes such as constant replacement metrics, score saturation,
identity aliases that hide log sources, state-transition token concentration, and
low-confidence-only predictions. The inspector verifies every manifest-bound file
before reading it and ignores unmanifested files.

To compare repeat runs or a case suite:

```bash
rootwalk inspect-many /absolute/path/to/run-1 /absolute/path/to/run-2 \
  --output /tmp/rootwalk-investigation-summary.json
```

The aggregate preserves missing historical fields instead of treating them as zero.
It reports candidate reuse, final-selection origin, provider requests versus Runtime
model calls, query mix, constant direct candidates, identity fragmentation, and
prompt tokens spent in rounds that produced no new observations.

Open the Viewer and choose **打开运行文件夹**. Select the exported RunBundle directory,
including `artifacts/`. Files stay in the browser. The older multi-file picker still
works, but opening raw inputs or outputs requires the referenced artifact files.

The call view groups records by runtime model-call number. Each round shows:

- The returned decision summary and planned operations, followed by actual tool attempts.
- Input, output, cache-read, and reasoning tokens, when reported by the provider.
- Model-call elapsed time and individual tool elapsed times.
- Estimated Context occupancy against the configured compilation budget, selected
  and omitted references, and omission reasons. This estimate is distinct from billed
  input tokens and from the model's physical context-window limit.
- Lazy access to complete input messages and raw output. Provider-returned analysis
  text is displayed when present. Hidden reasoning that was never returned is not reconstructed.
- Every adapter-to-SDK request, including JSON repair and response-format fallback,
  with separate input/output artifacts, elapsed time, status, and reported usage.
  SDK-internal transport retries are not separate adapter requests.

Cache-read tokens are a subset of input tokens; reasoning tokens are a subset of
output tokens. They are not added to the total a second time. Runtime call counts
and provider-request counts are intentionally separate. A failed or interrupted
request may have no reported usage: missing values mean unknown, not zero.

For older bundles, the viewer recovers round associations from decision and
operation IDs. Missing timing or usage is shown as **未记录**. Historical bundles
retain the versions and hashes that produced them; opening them does not update
their provenance to the current source revision.

## Storage and tracing

`attempts.jsonl` now includes call numbers, execution IDs, start/end timestamps,
monotonic elapsed milliseconds, and provider-request counts where available.
`model_requests.jsonl` contains request-level metadata and content-addressed input
and output references. `execution_audit.jsonl` exports late audit records available
at bundle creation, including trace IDs for enabled OTEL runs.

Model failures retain usage from responses already received. Fully returned invalid
responses settle against reported tokens; interrupted requests retain an unknown
reservation while reporting the known usage separately. Failure and tool outcome
metadata is exported to OTEL without exception text or stack traces.

The CLI returns telemetry flush completion, cleanup completion, and dropped-span
counts after bounded shutdown, and persists that delivery result as a late audit
record in the working SQLite database. Because the sealed bundle is exported before
telemetry shutdown, this final delivery record is not retroactively inserted into
the initial bundle. A subsequent export includes it. The metadata-only default
does not upload prompts or response bodies.

The Viewer works without Langfuse. Langfuse Cloud v4 ingestion and structured IO
mapping have been validated with a public recorded-model smoke run. Billing remains
unverified because recorded models do not report provider usage; see
[langfuse.md](langfuse.md).
