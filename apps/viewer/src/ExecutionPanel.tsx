import { useMemo, useState } from "react";
import { count, duration, executionRounds, number, object, roundStatus, text, usageOf } from "./execution";
import type { ExecutionRound } from "./execution";
import type { RunBundle } from "./types";

const ARTIFACT_PREVIEW_CHAR_LIMIT = 200_000;

const statusLabels: Record<string, string> = {
  success: "完成", error: "失败", runtime_failure: "执行失败", invalid_response: "输出格式无效",
  cancelled: "已取消", outcome_unknown: "调用失败 · 结果未知", interrupted: "未完成",
  started: "执行中", no_data: "无数据", partial: "部分数据", unsupported: "不支持", ready: "上下文已就绪",
};
const statusText = (value: unknown) => statusLabels[text(value)] ?? text(value);
const failed = (status: string) => ["error", "runtime_failure", "invalid_response", "outcome_unknown", "cancelled", "interrupted"].includes(status);

function Values({ values }: { values: [string, string][] }) {
  return <dl className="execution-values">{values.map(([label, value]) => <div key={label}><dt>{label}</dt><dd>{value}</dd></div>)}</dl>;
}

function JsonDetail({ title, value }: { title: string; value: unknown }) {
  return <details className="execution-raw"><summary>{title}</summary><pre>{JSON.stringify(value, null, 2) ?? "未记录"}</pre></details>;
}

function ArtifactContent({ source }: { source: string }) {
  if (source.length > ARTIFACT_PREVIEW_CHAR_LIMIT) {
    return <p className="execution-note" role="status">内容共 {source.length.toLocaleString("en-US")} 个字符，超过页面预览上限，未解析或渲染；请下载完整文件查看。</p>;
  }
  let value: unknown;
  try { value = JSON.parse(source); } catch { return <pre>{source}</pre>; }
  const root = object(value);
  const messages = Array.isArray(value) ? value : Array.isArray(root.messages) ? root.messages :
    Array.isArray(root.choices) ? root.choices.map((item) => object(item).message) : undefined;
  if (!messages) return <pre>{JSON.stringify(value, null, 2)}</pre>;
  return <div className="model-messages">{messages.map((item, index) => {
    const message = object(item);
    return <section key={index}><h5>{text(message.role, "assistant")}</h5>
      {typeof message.reasoning_content === "string" && <details><summary>模型返回的分析文本</summary><pre>{message.reasoning_content}</pre></details>}
      <pre>{typeof message.content === "string" ? message.content : JSON.stringify(message.content, null, 2) ?? "未记录"}</pre>
    </section>;
  })}{Array.isArray(root.messages) && <JsonDetail title="请求参数（含输出 Schema）" value={{ ...root, messages: undefined }} />}</div>;
}

function ArtifactDetail({ bundle, hash, title }: { bundle: RunBundle; hash: unknown; title: string }) {
  const [source, setSource] = useState<string>();
  const [error, setError] = useState<string>();
  const [loading, setLoading] = useState(false);
  const load = async () => {
    if (loading || source !== undefined) return;
    if (!bundle.readArtifact || typeof hash !== "string") { setError("这次运行未记录原始内容。"); return; }
    setLoading(true); setError(undefined);
    try { setSource(await bundle.readArtifact(hash)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "读取失败"); }
    finally { setLoading(false); }
  };
  const download = () => {
    if (source === undefined) return;
    const url = URL.createObjectURL(new Blob([source], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = `${text(hash, "artifact")}.json`; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return <details className="execution-raw" onToggle={(event) => { if (event.currentTarget.open) void load(); }}>
    <summary>{title}</summary>
    {loading && <p role="status">正在读取本地记录…</p>}
    {error && <p role="alert">{error} <button type="button" onClick={() => void load()}>重试</button></p>}
    {source !== undefined && <><button type="button" className="artifact-download" onClick={download}>下载完整 JSON</button>
      <ArtifactContent source={source} /></>}
  </details>;
}

function RoundDetail({ bundle, round }: { bundle: RunBundle; round: ExecutionRound }) {
  const usage = usageOf(round.attempt.usage);
  const context = round.context ?? {};
  const estimate = number(context.estimated_tokens);
  const limit = number(context.max_context_tokens);
  const occupancy = estimate !== undefined && limit ? estimate / limit * 100 : undefined;
  const operations = Array.isArray(round.decision?.operations) ? round.decision.operations : [];
  const status = roundStatus(round);
  return <article className="round-detail" aria-label={`第 ${round.number} 轮详情`}>
    <div className="round-heading"><h3>第 {round.number} 轮</h3><span className="execution-status" data-error={failed(status)}>{statusText(status)}</span></div>
    <Values values={[["模型耗时", duration(round.attempt.duration_ms)], ["输入 Token", count(usage.input)], ["输出 Token", count(usage.output)], ["缓存命中 Token", count(usage.cached)], ["其中推理 Token", count(usage.reasoning)], ["模型请求次数", count(round.requests.length || round.attempt.provider_request_count)]]} />
    <section className="context-occupancy"><div><h4>上下文占用</h4><strong>{count(estimate)} / {count(limit)} tokens{occupancy !== undefined ? ` · ${occupancy.toFixed(1)}%` : ""}</strong></div>
      {occupancy !== undefined && <div className="context-meter" role="meter" aria-label="上下文预算占用百分比" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.min(occupancy, 100)} aria-valuetext={`${occupancy.toFixed(1)}%`}><span style={{ width: `${Math.min(occupancy, 100)}%` }} /></div>}
      <p>按本轮 Context 编译预算估算；上方输入 Token 来自模型返回，可能包含修复请求的额外输入。</p>
      <JsonDetail title="查看选入、遗漏的上下文与原因" value={{ selected_refs: context.selected_refs, omitted_refs: context.omitted_refs, omission_reasons: context.omission_reasons, policy_version: context.policy_version, token_estimator_version: context.token_estimator_version }} />
    </section>
    <section className="round-decision"><h4>AI 决策摘要</h4><p>{text(round.decision?.decision_summary, round.attempt.summary ? text(round.attempt.summary) : "这轮没有提交有效决策。")}</p>
      <p className="execution-note">展示模型返回的决策摘要与原始输入输出，不推测未返回的隐藏思维链。</p>
      {operations.length > 0 && <JsonDetail title={`模型计划执行的操作 · ${operations.length}`} value={operations} />}
      <ArtifactDetail key={`input-${text(context.model_input_ref)}`} bundle={bundle} hash={context.model_input_ref ?? context.message_hash} title="模型看到的完整上下文" />
      {Boolean(round.decision?.model_response_ref) && <ArtifactDetail key={`response-${text(round.decision?.model_response_ref)}`} bundle={bundle} hash={round.decision?.model_response_ref} title="模型原始输出" />}
    </section>
    {round.requests.length > 0 && <section><h4>实际模型请求 · {round.requests.length}</h4><p className="execution-note">包含输出修复与格式回退。每次请求分别记录，Token 已计入本轮合计。</p>
      {round.requests.map((request, index) => {
        const requestUsage = usageOf(request.usage);
        return <details className="request-detail" key={index} open={failed(text(request.status))}>
          <summary><strong>请求 {index + 1}</strong><span className="execution-status" data-error={failed(text(request.status))}>{statusText(request.status)}</span><span>{count(requestUsage.total)} tokens · {duration(request.duration_ms)}</span></summary>
          <Values values={[["输入 Token", count(requestUsage.input)], ["输出 Token", count(requestUsage.output)], ["修复 / 回退原因", text(request.retry_reason, "无")]]} />
          {Boolean(request.error_type) && <p className="execution-error">{text(request.error_type)} {text(request.error_summary, "")}</p>}
          <ArtifactDetail bundle={bundle} hash={request.model_input_ref} title="这次请求的完整输入" />
          {Boolean(request.model_response_ref) && <ArtifactDetail bundle={bundle} hash={request.model_response_ref} title="这次请求的完整输出" />}
        </details>;
      })}</section>}
    <section className="round-tools"><h4>工具执行 · {round.tools.length}</h4>
      {round.tools.length === 0 && <p className="execution-note">这轮没有记录工具调用；状态更新与结束操作见决策及 Runtime 记录。</p>}
      {round.tools.map((tool) => <details key={tool.id} className="tool-detail">
        <summary><strong>{text(tool.attempt.tool, text(tool.observation?.tool))}</strong><span className="execution-status" data-error={failed(text(tool.attempt.status))}>{statusText(tool.attempt.status === "started" ? "interrupted" : tool.attempt.status)}</span><span>{duration(tool.attempt.duration_ms)}</span></summary>
        <p>{text(tool.observation?.summary, "未记录结果摘要")}</p>
        <JsonDetail title="调用参数" value={tool.attempt.arguments ?? tool.observation?.arguments} />
        <JsonDetail title="结果、覆盖与错误详情" value={tool.observation ?? tool.attempt} />
        {Array.isArray(tool.observation?.artifact_refs) && tool.observation.artifact_refs[0] && <ArtifactDetail bundle={bundle} hash={object(tool.observation.artifact_refs[0]).content_hash} title="工具原始返回" />}
      </details>)}
    </section>
    {round.events.length > 0 && <section><h4>Runtime 执行记录 · {round.events.length}</h4>
      {round.events.map((event, index) => <JsonDetail key={index} title={text(event.event_type)} value={event} />)}
    </section>}
  </article>;
}

export function ExecutionPanel({ bundle }: { bundle: RunBundle }) {
  const rounds = useMemo(() => executionRounds(bundle), [bundle]);
  const [selected, setSelected] = useState<number>();
  const round = rounds.find((item) => item.number === selected) ?? rounds[0];
  const usage = usageOf(bundle.report.model_usage);
  const toolCount = rounds.reduce((sum, item) => sum + item.tools.length, 0);
  const observation = object(bundle.report.observability ?? bundle.manifest.observability);
  const trace = bundle.executionAudit?.find((record) => record.payload.audit_type === "otel_root_span_context")?.payload;
  return <section className="execution-panel" aria-label="AI 调用过程">
    <div className="execution-heading"><div><h2>AI 调用过程</h2><p>逐轮查看上下文、决策、Token 消耗和工具执行。</p></div><span className="execution-note">{rounds.length} 轮记录</span></div>
    <Values values={[["整次运行", duration(number(bundle.report.wall_clock_seconds) === undefined ? undefined : Number(bundle.report.wall_clock_seconds) * 1000)], ["已记录 Token", count(usage.total)], ["输入 / 输出", `${count(usage.input)} / ${count(usage.output)}`], ["工具执行", `${toolCount} 次`], ["远端 Trace", observation.enabled === true ? "已启用" : observation.enabled === false ? "未启用 · 本地记录可用" : "未记录"]]} />
    {round ? <div className="execution-layout"><nav className="round-list" aria-label="选择 AI 调用轮次">{rounds.map((item) => {
      const itemUsage = usageOf(item.attempt.usage);
      const itemStatus = roundStatus(item);
      return <button type="button" key={item.number} aria-pressed={round.number === item.number} onClick={() => setSelected(item.number)}>
        <span className="round-number">{String(item.number).padStart(2, "0")}</span>
        <span className="round-nav-body"><strong>{text(item.decision?.decision_summary, statusText(itemStatus))}</strong><span>{count(itemUsage.total)} tokens · {duration(item.attempt.duration_ms)}</span><small data-error={failed(itemStatus)}>{statusText(itemStatus)} · {item.tools.length} 次工具调用</small></span>
      </button>;
    })}</nav><RoundDetail key={round.number} bundle={bundle} round={round} /></div> : <p className="execution-empty">没有调用过程记录。请加载包含 attempts、decisions 和 context_manifests 的 RunBundle。</p>}
    {trace && <JsonDetail title="Trace 关联信息" value={trace} />}
  </section>;
}
