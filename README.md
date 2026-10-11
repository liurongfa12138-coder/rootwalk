# RootWalk

RootWalk 是面向微服务故障诊断的 Agent Harness。它把工具查询、证据、假设、候选因果路径和实际模型输入保存成可检查的 RunBundle，并在 Runtime 中限制查询范围、证据引用、预算和终止条件。

当前发布候选在固定的 12 行 OpenRCA Telecom 开发切片上达到 **8/12 strict、8.5/12 partial**。这组案例已用于误差归因和迭代，只能视为开发集结果，不能与 OpenRCA 完整榜单直接比较。历史 Bank 和 Market 实验、失败终态及拒答仍保留在评测记录中，不因本次 Telecom 提升而从分母中删除。真实遥测、评分标签、模型密钥和运行产物均不随仓库发布。

早期单案例对照仍保留：真实 Ark 在同一个 Bank 开发案例上完成 C0、C1/G0、G1 三组运行，每组 3 次，9 次得分全部为 **0.0**。它与最终 135 次 holdout 分开统计；两组实验的失败和拒答都保留在各自分母里。详见 [评测记录](docs/evaluation-report.md)。

## 环境

Python 固定为 3.11，依赖和虚拟环境由 uv 管理：

```bash
uv sync --frozen
uv run rootwalk --help
```

RunBundle 会冻结源码 revision、`uv.lock` SHA-256、直接依赖版本及其指纹、模型配置、实际 provider model、预算、沙箱 provenance 和 OTel 配置。源码 checkout 与 clean wheel 的版本清单、replay 结果已验证一致。环境边界见 [环境适配器协议](docs/environment-adapters.md)。

## 准备外部 OpenRCA 数据

真实遥测和评测标签不能放进 RootWalk Git 工作区。先准备四个互不重叠的绝对路径：

```bash
export RW_EXTERNAL_ROOT=/absolute/path/outside/rootwalk
export RW_OPENRCA_SOURCE="$RW_EXTERNAL_ROOT/source"
export RW_OPENRCA_CODE="$RW_EXTERNAL_ROOT/OpenRCA"
export RW_OPENRCA_SUITE="$RW_EXTERNAL_ROOT/public-suite"
export RW_OPENRCA_LABELS="$RW_EXTERNAL_ROOT/evaluator-labels"
```

固定上游代码版本：

```bash
git clone https://github.com/microsoft/OpenRCA.git "$RW_OPENRCA_CODE"
git -C "$RW_OPENRCA_CODE" checkout c1bd4af7f635171a1c31cdd567c07d698dff6abc
```

当前 D9 v2 使用固定第三方镜像 `cdreetz/OpenRCA@7a29a7c3600cc07b7760a6c5097d36ac8573f039`。按 v2 spec 下载并校验源文件：

```bash
uv run python scripts/fetch_openrca_bank.py \
  --source-root "$RW_OPENRCA_SOURCE" \
  --suite-spec benchmarks/openrca-bank-d9-v2.json \
  --mirror
```

脚本拒绝 Git 工作区内的 `source-root`。v2 源文件合计 17,968,974,882 bytes，文件清单哈希为 `fe62934a0e800e6f37c30a5ed02d88f043584ef061b5fa376daf214c4a94b7b6`。`query.csv` 含评分字段，只写入权限收紧的 `private_source/`。

03-04 的四个遥测文件与既有官方 profile 的大小和 SHA-256 对齐，只能作为 compatibility anchor。v2 使用的另外六天没有得到官方归档逐字节证明，不能把第三方镜像写成官方来源。

官方来源探测结果保持单独记录：`drive.google.com` 和 `drive.usercontent.google.com` 在 TCP 连接阶段超时；`www.googleapis.com` 可达，但 Drive v3 `alt=media` 在没有 API 身份时返回 HTTP 403；browser-use 导航也超时。公开检索没有找到能替代官方证明的可信镜像。

生成不含标签的 17-case 公开切片：

```bash
uv run python scripts/prepare_bank_suite.py \
  --source-root "$RW_OPENRCA_SOURCE" \
  --upstream-code-root "$RW_OPENRCA_CODE" \
  --output-root "$RW_OPENRCA_SUITE" \
  --suite-spec benchmarks/openrca-bank-d9-v2.json
```

`output-root` 必须位于所有 Git 工作区之外，并与两个输入目录分离。v2 排除 row 123，因为固定镜像缺少其参考窗口所需的 `2021_03_22` 遥测。已生成的 suite 约 2.4 GiB，包含 17 个 case 和 204 个受哈希约束的输出。

评测标签另存为 evaluator-only 包：

```bash
uv run python scripts/prepare_openrca_eval.py \
  --query "$RW_OPENRCA_SOURCE/private_source/Bank/query.csv" \
  --suite benchmarks/openrca-bank-d9-v2.json \
  --output "$RW_OPENRCA_LABELS"
```

标签包包含 17 行，目录权限为 `0700`，文件权限为 `0600`。不要把它传给 `rootwalk run`，也不要提交、缓存或随 RunBundle 发布。第三方代码和数据边界见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [安全说明](docs/security.md)。

## M0：环境和模型探针

```bash
uv run rootwalk doctor

export ARK_API_KEY=...
uv run rootwalk probe-model \
  --model ep-your-endpoint \
  --base-url https://ark.cn-beijing.volces.com/api/v3
```

`probe-model` 只发一次合成 Schema 请求。密钥从环境变量读取，错误输出不会打印密钥、模型参数或 base URL。

## M1：运行、回放和导出

recorded fixture 只验证协议和状态流：

```bash
export RW_DATA_ROOT=/path/to/public_input

uv run rootwalk run \
  --data-root "$RW_DATA_ROOT" \
  --output /tmp/rootwalk-bank-recorded \
  --run-id bank-recorded \
  --recorded-decisions examples/bank-recorded-decisions.json

uv run rootwalk replay /tmp/rootwalk-bank-recorded
uv run rootwalk export /tmp/rootwalk-bank-recorded \
  --format openrca \
  --output /tmp/rootwalk-bank-recorded.csv
```

真实模型运行把 `--recorded-decisions` 换成 `--model` 与 `--base-url`。也可以通过
`ARK_MODEL` 和 `ARK_BASE_URL` 提供这两个配置。`--reasoning-effort`、
`--model-timeout-s` 和 `--max-output-tokens` 用来固定单次模型参数。

## M2：预算、查询隔离和恢复

Runtime 限制总时长、Context Token、累计输入/输出 Token、模型调用、工具尝试、Artifact、校验重试和无进展轮数。每个 Run 新写入 Artifact 的硬上限为 64 MiB。Bank 查询在可终止子进程中串行执行，单次超时会终止子进程。

进程中断且最终 Bundle 尚未生成时，保留 `.<output-name>.work/`，用相同的 `--output` 和 `--run-id` 重跑原命令。Runtime 会读取 SQLite ledger 与 LangGraph Checkpoint，复用已提交的 Decision 和工具效果。真实 multiprocessing crash matrix 已覆盖响应孤儿、domain ahead、结果未知重试和 seal 后迟到效果。

Linux 默认使用 `--query-sandbox auto`。预检通过时，QueryWorker 进入 bwrap 文件系统与网络隔离；无法使用时退回 spawn。正式 D9 runner 固定 `--query-sandbox required`，并冻结 bwrap 版本和可执行文件 SHA-256。

## M3：机制开关、diff 和 Viewer

`run` 已开放 Context 与 Validator 机制开关：

```bash
uv run rootwalk run \
  --data-root "$RW_DATA_ROOT" \
  --output /tmp/rootwalk-c1 \
  --run-id bank-c1 \
  --recorded-decisions examples/bank-recorded-decisions.json \
  --context-strategy evidence \
  --validator-mode observe_only
```

可选值分别是 `recent/evidence` 和 `observe_only/enforce`。Runtime 已实际生成 C0、C1/G0、G1 recorded Bundle，并核对实际模型输入只出现目标变量差异。

比较两个 Bundle：

```bash
uv run rootwalk diff /path/to/control-bundle /path/to/treatment-bundle
```

`diff` 按 environment manifest、context manifest、decision、observation、evidence、termination 六层定位首次可观察差异，不做因果归因。

逐轮检查一个 Bundle，或聚合一组重复运行：

```bash
uv run rootwalk inspect /path/to/bundle \
  --format markdown \
  --output /tmp/rootwalk-investigation.md

uv run rootwalk inspect-many /path/to/run-1 /path/to/run-2 \
  --output /tmp/rootwalk-investigation-summary.json
```

`inspect` 会校验 manifest，关联 Context、provider 请求、Decision、Runtime 绑定、工具结果、Evidence 和预算，并标出候选指标塌缩、实体别名隐藏日志、无 Trace 调查和状态转换 Token 集中等问题。它只读封存记录，不重新调用模型或工具。详细字段见 [AI 调用检查](docs/call-inspection.md)。

默认 RunBundle 只保留大文件的 external 引用。需要制作可搬运副本时，显式提供受信任的 `public_input` 根目录：

```bash
uv run rootwalk export /path/to/bundle \
  --self-contained \
  --external-root /path/to/public_input \
  --output /path/to/portable-bundle
```

该模式只嵌入 external-root manifest 已登记且大小、SHA-256 均匹配的 Artifact；`evaluation_only` Artifact 会直接拒绝。普通 Bundle 和 self-contained Bundle 都通过临时目录原子发布，新 manifest 会记录总字节数和完整性范围。

启动只读 Viewer：

```bash
cd apps/viewer
npm ci
npm run dev
```

Viewer 可查看调查图、Evidence、Hypothesis、Path、Revision rail、预算和双 Bundle 对比。所有解析都在浏览器内完成；Viewer 不复算 manifest 哈希，正式检查应先运行 CLI `replay`、`diff` 或 `inspect`。

## OpenTelemetry

默认不启用 OTel，也不采集内容。显式开启时建议继续保留 `--no-otel-capture-content`：

```bash
uv run rootwalk run \
  --data-root "$RW_DATA_ROOT" \
  --output /tmp/rootwalk-otel \
  --run-id bank-otel \
  --recorded-decisions examples/bank-recorded-decisions.json \
  --otel \
  --otel-endpoint http://127.0.0.1:4318/v1/traces \
  --no-otel-capture-content \
  --otel-sample-ratio 1.0
```

CLI 的 OTLP 配置、本地 HTTP OTLP 导出、完整 stage spans、同一 run 重新打开后的自动 trace link、W3C traceparent 隔离、drop 计数、flush 边界和 wall-clock timing 已通过测试。真实 Langfuse Cloud 也已完成 metadata-only ingestion：调用树可见、内容未上传、真实模型 Token 与 RunBundle 一致。接入步骤与验收清单见 [docs/langfuse.md](docs/langfuse.md)。

## 评测

```bash
uv run rootwalk evaluate \
  --predictions /path/to/prediction.csv \
  --labels "$RW_OPENRCA_LABELS"
```

`evaluate` 在独立进程读取标签，只返回 row ID、单行分数和汇总，不把标签交给调查 Runtime。v2 的 selection SHA-256 为 `921babc8d78ec3955d0fe55c75a27d2c521d7a5ca629099f37a459954dec356b`，case plan SHA-256 为 `c8c4f6acbec6371bdc7423dab98aec779285597390d5cdf26b0e55e87a161819`。

先检查排除开发行 2 和 128 后的 135-run holdout 计划，不发模型请求，也不要求当前机器已经安装或能运行 bubblewrap：

```bash
uv run python scripts/run_bank_benchmark.py \
  --suite-root "$RW_OPENRCA_SUITE" \
  --output-root "$RW_EXTERNAL_ROOT/benchmark-runs" \
  --model ep-your-endpoint \
  --base-url https://ark.cn-beijing.volces.com/api/v3 \
  --reasoning-effort minimal \
  --repeats 3 \
  --exclude-row-id 2 \
  --exclude-row-id 128 \
  --dry-run
```

dry-run 已真实生成 135 个唯一且绑定 suite spec 与 case manifest 的 run ID，命令中标签路径命中数为 0。`sandbox_preflight.status=not_run` 表示这里只核对计划。正式执行还要加 `--confirm-paid-run`，会先完成真实 bubblewrap preflight，并固定传 `--query-sandbox required`。首次执行会把 suite identity、排除行和全部 run ID 冻结到 `run_plan.json`；非空输出目录缺少该文件，或恢复时计划不一致，都会直接失败。只有数据内容、环境与版本清单完全一致的 Bundle 才会复用。

最终 holdout 的 `run_plan.json` SHA-256 为 `d4be31d7ae9f6090f33668113a588a8b0201d8a0c5be8dfac44b8192dc422650`。135 个 RunBundle 全部生成，runner failure 为 0；其中 57 次输出诊断、78 次拒答，28 次以失败终态结束。严格得分为 0，只有 3 次出现非零部分分：C0 合计 0.5，C1/G0 合计 1.0，G1 为 0。部分得分只表示组件或原因有局部匹配，不能包装成可用准确率。

这批历史运行合计使用 12,266,563 Token，单次 wall-clock 平均 39.9048 秒，逐次相加为 5,387.14 秒。由于没有确认价目，费用保持 `null`。最新 Bank 开发集为 15 个同 revision Bundle、runner failure 为 0，strict 5/15、partial 8.0/15。预注册 Market strict-unseen 单次运行正常完成但评分为 0/1，因此 D9 当前缺的是跨数据集诊断质量，不再是 key、预算或运行链路。

## 工程检查

默认 CI 不下载真实 Bank 数据：

```bash
uv lock --check
uv run pytest -q --strict-markers -m "not openrca_bank"
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

当前结果为 `768 passed, 9 deselected`。提供固定官方切片后的全量命令与结果是：

```bash
ROOTWALK_OPENRCA_BANK_ROOT=/path/to/bank-row-0001/public_input \
  uv run pytest -q --strict-markers
# 777 passed
```

Viewer 验证：

```bash
cd apps/viewer
npm ci
npm test
npm run build
```

当前结果为 16 tests passed，build 成功。

## 文档

- [架构与恢复](docs/architecture.md)
- [适配器指南](docs/adapter-guide.md)
- [环境适配器协议](docs/environment-adapters.md)
- [验收状态](docs/acceptance.md)
- [评测说明](docs/evaluation.md)
- [评测记录](docs/evaluation-report.md)
- [当前限制](docs/limitations.md)
- [安全说明](docs/security.md)
- [Langfuse 与 OpenTelemetry](docs/langfuse.md)
- [AI call inspection in the Viewer](docs/call-inspection.md)

RootWalk 采用 [Apache License 2.0](LICENSE)。OpenRCA 的 MIT 声明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
