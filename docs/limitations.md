# RootWalk 当前限制

RootWalk 已能生成、恢复、校验、比较和查看 RunBundle，但适用范围仍然很窄。

## 数据和适配器

- CLI 已能按 `task.environment` 选择 OpenRCA Bank 或 Market `public_input`；尚未提供插件发现和远程数据源实现。
- 默认 CI 使用合成 Bank fixture。8 个固定官方事实测试仍依赖仓库外的 `bank-row-0001/public_input`。
- D9 v2 固定 17 个案例，第三方镜像源、公开 suite 和 evaluator-only 标签包已经生成。row 123 因镜像缺少 `2021_03_22` 遥测而排除；最终 holdout 又排除开发行 2 和 128，实际评测 15 行。
- 03-04 四个遥测文件与既有官方 profile compatibility anchor 对齐；其余六天没有得到官方归档逐字节证明，因此 v2 不能描述为全量官方数据。
- 真实遥测没有确认单独的再分发许可，不能提交到仓库或 CI cache。
- Runtime 已通过 `TaskSource`、`TopologyProvider`、`ToolProvider`、`ResultExporter` 和 `CapabilityManifest` 定义 Adapter Protocol，当前有 Bank 和 Market 两个本地实现，没有插件发现、版本协商或远程工具协议。详见 [environment-adapters.md](environment-adapters.md)。
- `summarize_anomalies` 只用于导航，不是根因排序器。

## 调查正确性

- `supported` Hypothesis、通过 Grounding 的 Path 和 `completed_partial` 都不等于真实根因。
- Coverage 不完整或采集健康未知时，`no_data` 只会形成 gap，无法证明目标没有异常。
- 拓扑来自冻结输入，不能代表调查时刻之外的动态依赖。
- C0、C1/G0、G1 三组真实 Ark 单案例运行各 3 次，9 次评分全部为 0.0。
- 最终 135-run holdout 已完成，严格命中仍为 0/135。部分得分只有 1.5/135，132 次为 0 分；3 次非零部分分只说明组件或原因有局部匹配，不能视为可用质量。
- 最终运行有 57 次诊断输出、78 次拒答和 28 次失败终态。completed_partial 也不等于答案正确，本次 107 次 completed_partial 没有带来任何严格命中。
- 最新 Bank 开发集复验达到 5/15 strict、8.0/15 partial；该集合已经用于误差归因，不能作为未见集成绩。
- Market 首个预注册案例暴露了大 scope 首轮上下文溢出，修复后运行链路可以完成。第二个预注册案例的 3 次运行仍为 0 分；脱敏归因确认正确原因已进入公开候选，正确组件没有进入候选。Market 的 pod、service、node 多层归一化和 service-level 候选生成尚未完成。
- 最新版本又按固定哈希预注册一个此前未运行、未评分的 Market 案例并只运行一次；运行正常完成，但 component criterion 为 0/1。该结果已冻结，不会通过重抽或改分母覆盖。
- 该 Market 失败由 `ever_open` 路径历史压过更早候选导致；移除该排序位后，同一决策轨迹的 recorded post-hoc replay 为 1/1。此结果只验证已知缺陷修复，不构成新的 unseen 泛化证据。

## 运行方式

- 一次 CLI 调用只处理一个 Task，工作目录也只允许一个 run。
- 没有服务端、任务队列、账号、多租户、审批或在线控制面。
- QueryWorker 串行执行七个 Bank 工具。Linux 上默认用 bubblewrap 隔离网络和除 runtime、包代码、当前 `public_input` 外的文件；`required` 模式在 binary、mount plan 或沙箱 UID 可读性预检失败时关闭，`off` 模式只保留进程与超时边界。D9 benchmark 固定使用 `required`。超时会终止整个子进程，下一次查询重新启动。
- 模型边界支持 Ark 和 recorded model，没有自动路由或 provider fallback
  或多供应商比较。
- 总时限、Token、模型调用、工具尝试、校验重试和无进展轮数有硬预算，每个 Run 新写入 Artifact 的硬上限为 64 MiB。这些默认值已经在 135 次 holdout 运行中实际执行，但零严格命中说明不能据此认定预算合理或已经完成多案例调优。
- 135 次运行共使用 12,266,563 Token，单次 wall-clock 平均 39.9048 秒。没有确认价目，cost 为 `null`；本次 timing 也不构成性能承诺。

## 恢复和导出

- 恢复依赖同一台机器上的 `.work` 目录、同一个 `run_id` 和相同输出路径。
- 已提交的 Decision 和工具效果会复用；未提交完成的外部调用仍可能出现结果未知。
- Checkpoint 领先领域 ledger 时会失败关闭，不会猜测应当采用哪一边。
- Run seal 后只允许 late audit。当前没有后台 outbox worker；CLI 在导出成功后同步标记 done。
- 普通 Bundle 和 self-contained Bundle 都使用临时目录原子发布。self-contained 模式会复制通过校验的 external Artifact，但不会复制 `evaluation_only` 内容。

## Diff、实验和 Viewer

- `diff` 只找六层首次可观察差异，不做因果归因；新增的第六层是 Evidence。
- CLI 已开放 `--context-strategy` 和 `--validator-mode`。D9 runner 已完成多案例矩阵并支持续跑，但三组条件都没有严格命中，尚不能证明任一机制带来可用收益。
- Viewer 完全在浏览器里运行，只读本地 JSON/JSONL；它没有后端、鉴权、共享链接或协作能力。
- Viewer 不会复算 `manifest.json` 里的 SHA-256，也不读取所有 Artifact 正文。完整性判断以 CLI 为准。
- Viewer build 仍报告一个 693.51 kB 的 Cytoscape chunk，当前没有性能基准证明它造成实际问题。

## 可观测性

- CLI 已开放 OTLP 参数，本地 HTTP OTLP、完整 stage spans、自动恢复 Link、traceparent 隔离、drop 计数、timing、flush 和失败隔离均已验证。
- 默认关闭内容采集；打开 `--otel-capture-content` 会把 prompt 或响应交给配置的 exporter，使用方要自行评估数据边界。
- exporter 失败采用 fail-open，不阻断调查，因此观测平台可能缺 span。
- 真实 Langfuse Cloud 已验证 OTLP ingestion、调用树、metadata-only 内容关闭、真实模型 Token 映射和有界 flush；恢复 Link 与后端失败边界由本地测试覆盖。自定义模型未配置价目，因此费用仍为 `null`。具体证据与剩余边界见 [langfuse.md](langfuse.md)。

## 非目标

当前版本不承诺生产自动处置、写操作、全量日志托管、实时拓扑发现、跨用户隔离或任何性能指标。Bank 开发结果不能替代 D9 的 Market strict-unseen 质量门；自托管 Langfuse 与费用计算也仍是部署侧待办。
