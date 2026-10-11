# RootWalk 开发进度

更新时间：2026-10-05

当前判断：D1～D8、D10、D11 已通过，D9 的跨数据集质量门未通过。D9 v2 历史 135-run holdout 为 0/135；最新冻结 Bank 开发集复验达到 5/15 strict、8.0/15 partial。随后预注册且只运行一次的 Market strict-unseen 为 0/1，未重抽样本。移除路径依赖的 `ever_open` 排序后，同一已揭示样本的 recorded post-hoc replay 为 1/1，但不能改写 unseen 结果。运行费用因缺少确认价目而保持 `null`。真实 Langfuse Cloud 已验证 OTLP 接收、调用树、metadata-only 内容关闭、真实模型 Token 映射和有界 flush。

## M0～M3 已完成的工程链路

- Python 3.11、uv、Typer CLI、Ark 探针和严格 Pydantic v2 Schema 已建立。
- Bank 单案例可以完成只读查询、证据提取、假设与路径校验、封存、回放、OpenRCA 导出和独立评分。
- Runtime 已限制时长、Context Token、累计输入/输出 Token、模型调用、工具尝试、Artifact、校验重试和无进展轮数；每个 Run 新写入 Artifact 的硬上限为 64 MiB。
- QueryWorker 在独立子进程中执行查询，Linux 默认尝试 bwrap；正式 D9 要求 bwrap，并保存版本与可执行文件 SHA-256。
- SQLite ledger、同步 LangGraph Checkpoint、稳定效果键、run seal 和 outbox 支持中断恢复。
- Bundle 采用临时目录加原子 rename 发布，也可显式导出经过校验的 self-contained Bundle。
- `rootwalk run` 已开放 `--context-strategy`、`--validator-mode`、`--query-sandbox` 和 OTel 参数。
- 六层 Evidence Diff、机制实验、只读 Viewer 和 OTLP 观测链路已实现。
- `TaskSource`、`TopologyProvider`、`ToolProvider`、`ResultExporter` 与 `CapabilityManifest` 已形成 Adapter Protocol 边界。

## 已验证的工程事实

- 默认干净 CI 为 `653 passed, 9 deselected`；固定官方 Bank 切片下全量 Python 测试为 `662 passed`。
- Ruff lint、Ruff format 和 Pyright 均通过。
- Viewer 为 16 tests passed，build 成功。
- D5 的真实 multiprocessing crash matrix 已通过，覆盖响应 Artifact 已写但 Decision 未提交、domain ahead、结果未知重试和 seal 后迟到效果。
- D6 已由 Runtime 实际生成 C0、C1/G0、G1 recorded Bundle，并核对实际模型输入和反馈隔离。
- clean wheel、replay 与源码 checkout 的版本清单已验证一致。Run 会冻结源码 revision、`uv.lock` SHA-256、8 个直接依赖版本及指纹、OTel 配置、模型配置和实际 provider model。
- CLI OTLP、本地 HTTP OTLP、完整 stage spans、同一 run 跨 reopen 自动 trace link、traceparent 隔离、drop 计数、flush 边界和 wall-clock timing 已通过。
- `rootwalk inspect` 与 `inspect-many` 已能从哈希校验后的 RunBundle 还原每轮 Context、provider 请求、Decision、Runtime 绑定、工具结果、Evidence 和预算，并聚合候选塌缩、身份碎片、查询组合及无新 Observation 的 Token 消耗。
- Bank `public_input` 的 12 个文件、97,346,957 bytes 和固定 manifest 已通过校验；目标指标在 incident 窗口返回 19 个样本。
- Market 适配器与公开切片准备脚本已实现，覆盖两类日志、四类 long metrics、一类 wide service metrics 和 trace。大实体集合在模型上下文中会保留完整计数和查询入口，只内联入口实体及其相邻拓扑；完整 scope、拓扑和权限检查仍保留在账本与 Runtime。
- Market 输出候选改用固定上游 `microsoft/OpenRCA@c1bd4af7f635171a1c31cdd567c07d698dff6abc` 的公开候选表；导出时按上游约定把容器、服务 endpoint、runtime 和 mesh 标识还原为评分组件名。

## 真实模型结果全部保留

早期 9 个真实 Ark 开发 run 包含 5 次 runtime failure、4 次 completed_partial，得分全部为 0.0。

同一开发案例后来又完成 C0、C1/G0、G1 三组受控运行，每组 3 次。9 次运行也全部得 0.0，其中 5 次执行失败、9 次拒答，没有输出 Finding。完整 Token、timing 和终态见 [docs/evaluation-report.md](docs/evaluation-report.md)。这些结果不能外推为总体准确率或稳定性结论。

D9 v2 最终实验另用 15 个 holdout 行，排除开发行 2 和 128。3 个条件各重复 3 次，共生成 135 个 RunBundle；严格命中仍为 0，132 次得 0 分，3 次只得到局部匹配的非零部分分，合计 1.5。这个结果覆盖多案例，但仍不能解释为可用质量。

旧 holdout 已用于误差归因，因此只作为开发集复验。最新 G1 在冻结 revision `d820ac0f...` 上运行 15 次，14 次 `completed_partial`、1 次 `no_progress`；严格命中 5/15，部分得分 8.0/15，component 9/15、reason 7/15。14 个非空输出中只有 1 个正式 Finding，其余是仅供 benchmark exporter 使用的低置信推荐。这个结果证明 fallback 与运行收尾较旧版有改善，但不能证明正式诊断质量已经改善，也不能作为新的未见集成绩。

随后预注册了两个未进入历史 RunBundle 的 Market 案例。第一个案例在模型调用前因 271 个实体和 142 条边把必选上下文推到 37,185 tokens，3 次均以 `context_limit` 结束。加入有界 scope 和拓扑投影后，3 次都能完成调查，但评分仍为 0/3。再按上游公开候选表修正 Market 的 56 个组件候选、15 个原因候选和导出名称，并在第二个预注册案例上先跑完 3 次再评分，结果仍为 0/3。脱敏归因显示正确原因已进入 16 个候选，正确组件没有进入候选，当前主要缺口是 Market 的 pod、service、node 多层实体归一化和 service-level 候选生成。

## M4 已交付的部分

- README、架构、Adapter、Evaluation、Limitations、Security、Acceptance 和第三方声明已补齐。
- Apache-2.0 LICENSE 与 OpenRCA MIT notice 已纳入构建元数据。
- 默认 CI 使用运行时生成的合成 Bank fixture，官方数据组通过 `openrca_bank` marker 隔离。
- D9 v2 固定使用 `cdreetz/OpenRCA@7a29a7c3600cc07b7760a6c5097d36ac8573f039`，排除缺少 `2021_03_22` 遥测的 row 123，保留 17 个 case。
- v2 selection、case plan、source files 三个 SHA-256 分别为 `921babc8d78ec3955d0fe55c75a27d2c521d7a5ca629099f37a459954dec356b`、`c8c4f6acbec6371bdc7423dab98aec779285597390d5cdf26b0e55e87a161819`、`fe62934a0e800e6f37c30a5ed02d88f043584ef061b5fa376daf214c4a94b7b6`。
- 镜像源文件合计 17,968,974,882 bytes。已生成约 2.4 GiB 的公开 suite，共 17 个 case、204 个输出；私有评测包包含 17 行，文件权限为 `0600`。
- 03-04 四个遥测文件与既有官方 profile compatibility anchor 对齐；另外六天没有官方归档逐字节证明。
- 官方来源探测中，两个 Drive 域名在 TCP 阶段超时；`www.googleapis.com` 可达，但无 API 身份的 Drive v3 `alt=media` 返回 HTTP 403；browser-use 导航也超时。公开检索没有找到能替代官方逐字节证明的可信镜像。
- `run_bank_benchmark.py` 的 153-run dry-run 已实际执行，153 个 run ID 全部唯一并绑定 suite spec 与 case manifest，命令中标签路径命中数为 0。
- D9 runner 已加入 `--confirm-paid-run` 和 `--max-new-runs`。达到上限的计划项记为 deferred，不进入 attempted failure/refusal 分母。
- 最终 holdout 的 `run_plan.json` SHA-256 为 `d4be31d7ae9f6090f33668113a588a8b0201d8a0c5be8dfac44b8192dc422650`。135 次运行全部产出 Bundle，runner failure 为 0。
- 最终运行中有 57 次诊断输出、78 次拒答和 28 次失败终态；状态分布为 completed_partial 107、no_progress 22、runtime_failure 3、budget_exhausted 2、context_limit 1。
- 最终运行共使用 12,266,563 Token；单次 wall-clock 平均 39.9048 秒，逐次相加为 5,387.14 秒。价目未确认，费用为 `null`。

## 仍未完成

- D9：最终 135-run holdout 已完成，但严格准确率仍为 0，只有 3 次局部匹配，总部分得分 1.5/135。后续 Bank 开发集复验有改善，Market 预注册复验仍为 0/3；当前阻塞是跨数据集诊断质量，不是批跑是否启动。成本仍因缺少确认价目而无法计算。
- D10：本地 OTLP 边界测试和真实 Langfuse Cloud ingestion 均已通过。metadata-only smoke 的同一 Trace 中包含 Root、Graph、Context、Model、Validation、Tool、Evidence、Persistence 和 Export；输入输出正文均为空，flush 成功且 dropped=0。真实 Ark generation 的 Langfuse usage 为 input 8,023、output 60、total 8,083，与 RunBundle 一致。费用仍为 `null`，因为项目没有配置该自定义模型价目。
- 当前已有完整的 holdout 统计，但没有可宣称为可用的诊断准确率；成本没有确认价目，wall-clock 只作为本次实验记录，不构成性能承诺。

## 下一步

下一步只基于公开 bundle 修复 Market 的多层实体归一化、service-level 候选生成和 direct 候选语义问题；不得重跑或替换已评分的 strict-unseen 样本。改进效果应先用开发样本验证，再预注册新的版本级 holdout 方案。确认实际价目和缓存计费口径后补算成本。

## Ark 重新注入后的单案例结果

- 合成探针成功，实际 provider model 为 `deepseek-v4-flash-ga-260731`，本次 1,407 Token。
- 在正式 Bank row 2 上复现并修复了五个通用 Harness 问题：recent Context 没有暴露合法 Evidence ID、recent 排序误用批内操作序号、单批查询失控、重复查询只报错但不召回结果、Decision repair 不足。
- 当次修复提交：`5a58d9e`、`5592f5d`、`35197e3`、`fbf3850`、`ed7477f`。
- 当前质量门：默认 Python 测试为 `430 passed, 8 deselected`，固定官方切片下为 `438 passed`；Ruff lint/format 和 Pyright 通过；Viewer 9 tests 与 build 通过。
- `ed7477f` 的 C0 Run 在 9 次模型调用、140,596 Token 后因累计模型 Token 预算结束，形成 1 个候选，无 Finding。
- `ed7477f` 的主策略 G1 Run 在 11 次模型调用、137,596 Token、40.47 秒后主动拒答；无 Runtime failure、0 次 Validator retry，形成 1 个候选，无路径、官方得分 0。
- 独立评分显示该开发案例正确项为 `Tomcat01 / high memory usage`。标签已经揭示，因此该行只用于开发复盘，不再用于未见测试或继续定向调参。
