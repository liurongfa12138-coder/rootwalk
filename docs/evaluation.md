# RootWalk 评测

RootWalk 分开检查工程回归、实验隔离和诊断质量。协议走通只能证明 Harness 工作，不能替代根因准确率评测。

## 工程回归

默认 CI 使用运行时生成的合成 Bank fixture，不下载官方遥测：

```bash
uv lock --check
uv sync --frozen
uv run pytest -q --strict-markers -m "not openrca_bank"
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

当前结果为 `768 passed, 9 deselected`。9 个 `openrca_bank` 测试只检查固定官方 Bank 事实；显式提供切片后运行全量测试：

```bash
ROOTWALK_OPENRCA_BANK_ROOT=/path/to/bank-row-0001/public_input \
  uv run pytest -q --strict-markers
```

当前全量结果为 `777 passed`。如果显式选择 `-m openrca_bank` 却没有设置 `ROOTWALK_OPENRCA_BANK_ROOT`，测试会失败关闭，不会静默跳过。

Viewer 的独立结果为 16 tests passed，build 成功：

```bash
cd apps/viewer
npm ci
npm test
npm run build
```

## OpenRCA 导出和评分

调查 Runtime 只读无标签的 `public_input`。先从封存 Bundle 导出预测：

```bash
uv run rootwalk export /path/to/run-bundle \
  --format openrca \
  --output /tmp/rootwalk-prediction.csv
```

评分在独立进程读取 evaluator-only 标签包：

```bash
uv run rootwalk evaluate \
  --predictions /tmp/rootwalk-prediction.csv \
  --labels /path/outside/rootwalk/evaluator-labels \
  --output /tmp/rootwalk-score.json
```

实现固定到 `microsoft/OpenRCA` commit `c1bd4af7f635171a1c31cdd567c07d698dff6abc` 的评分语义。输出只含 row ID、单行分数和汇总，不返回标签正文。失败和拒答都留在分母里。

## 已完成的单案例实验

同一个 Bank 开发案例已经完成 C0、C1/G0、G1 三组真实 Ark 运行，每组 3 次。模型、工具、数据范围和预算保持一致，G0 直接复用 C1。9 次评分全部为 0.0，其中 5 次执行失败、9 次拒答，没有输出 Finding。

Runtime 还实际生成了三组 recorded Bundle，并通过 `compare_actual_model_inputs()` 核对目标变量：C0/C1 比较 `context_strategy`，G0/G1 比较 `validator_mode`。两组 `observe_only` 输入都没有 `path_quality` 或 `exit_gate` 反馈泄漏。

这些结果证明运行和实验隔离可复核，也证明当前策略没有显示出质量收益。完整终态、Token 和 timing 见 [evaluation-report.md](evaluation-report.md)。单案例不能外推为总体准确率、稳定性、成本或时延结论。

`summarize_benchmark()` 会先校验每个 RunBundle，再按任务和实验条件汇总完整分母、终态、失败、拒答、Token、wall-clock、评分及重复波动。费用只在调用方显式传入输入、输出和缓存价目时计算；缺价目或缺必要 Token 字段时返回 `null`。

```python
from rootwalk.benchmark import summarize_benchmark

report = summarize_benchmark(
    bundle_paths,
    run_metadata=run_metadata,
    evaluator_summaries=evaluator_summaries,
    pricing={
        "prompt_per_million": INPUT_PRICE,
        "completion_per_million": OUTPUT_PRICE,
        "cache_read_per_million": CACHE_READ_PRICE,
    },
)
```

三个价格变量由调用方按运行时实际价目填写；没有可靠价目时传 `pricing=None`，费用会保持 `null`。

## 六层 Run Diff

```bash
uv run rootwalk diff /path/to/control-bundle /path/to/treatment-bundle
```

`diff` 会先校验两个 Bundle 的声明文件哈希，再按 environment manifest、context manifest、decision、observation、evidence、termination 的顺序找首次可观察差异。Context manifest 比较会保留 Evidence 的 selected/omitted 引用。结果只说明两次运行从哪一层开始不同，不作因果归因。

## D9 v2 数据与最终 holdout

[openrca-bank-d9-v2.json](../benchmarks/openrca-bank-d9-v2.json) 固定 17 个单故障 `task_6` 案例。row 123 被排除，只因为镜像缺少参考窗口需要的 `2021_03_22` 遥测。

v2 固定第三方镜像 `cdreetz/OpenRCA@7a29a7c3600cc07b7760a6c5097d36ac8573f039`。selection、case plan 和 source files 的 SHA-256 分别为：

```text
921babc8d78ec3955d0fe55c75a27d2c521d7a5ca629099f37a459954dec356b
c8c4f6acbec6371bdc7423dab98aec779285597390d5cdf26b0e55e87a161819
fe62934a0e800e6f37c30a5ed02d88f043584ef061b5fa376daf214c4a94b7b6
```

镜像源文件合计 17,968,974,882 bytes。03-04 的四个遥测文件与既有官方 profile compatibility anchor 的大小和 SHA-256 对齐；这个锚点不能证明 v2 另外六天与官方归档逐字节一致。

当前已经生成并核对：

- 约 2.4 GiB 的公开 suite，包含 17 个 case 和 204 个受哈希约束的输出；
- 17 行 evaluator-only 标签，目录权限为 `0700`，文件权限为 `0600`；
- 153-run dry-run，153 个 run ID 全部唯一并绑定 suite spec 与 case manifest，命令中标签路径命中数为 0；
- 最终 135-run holdout，排除开发行 2 和 128，对 15 行执行 3 个条件、每个条件 3 次重复。

若要重新生成计划，可重复执行 dry-run：

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

实际运行还要加 `--confirm-paid-run`。首次执行会写入 `run_plan.json`；恢复时 selected rows、excluded rows、suite identity、重复次数和 run ID 只要有一项变化，runner 就会失败关闭。每个 Run 都保存源码 revision、`uv.lock` SHA-256、直接依赖版本及其指纹、模型配置、实际 provider model、预算、OTel 配置和 QueryWorker 沙箱 provenance。恢复和复用 Bundle 时会核对这些字段；正式 D9 还要求 bwrap，并核对版本与可执行文件 SHA-256。

最终计划文件的 SHA-256 为 `d4be31d7ae9f6090f33668113a588a8b0201d8a0c5be8dfac44b8192dc422650`。135 个 RunBundle 全部生成，runner failure 为 0；对应的 9 个预测文件也全部成功评分。

严格命中为 0/135。部分得分为 1.5/135，即 0.011111；132 次为 0 分，只有 3 次得到非零部分分。按条件看，C0 合计 0.5，C1/G0 合计 1.0，G1 为 0。非零部分分只表示组件或原因有局部匹配，不代表诊断结果可用。

运行侧共得到 57 次诊断输出、78 次拒答和 28 次失败终态。终态分布如下：

- completed_partial：107；
- no_progress：22；
- runtime_failure：3；
- budget_exhausted：2；
- context_limit：1。

总 Token 为 12,266,563。单次 wall-clock 平均为 39.9048 秒，逐次相加为 5,387.14 秒。没有确认价目，因此费用为 `null`。

## D9 判断

正式 135-run holdout 已完成，不再受 key、启动确认或批跑预算阻塞。该结果已用于误差分析，只保留为历史基线。

随后在固定的 15 行 Bank 开发集上，以同一个 Ark endpoint、G1、Top8、24 次 provider-attempt 上限和冻结 revision `d820ac0fe6d3b97aa5c2d21dca5a724941e5f31188737130a0db064142a370f7` 完成一次新运行。15 个 RunBundle 全部生成，runner failure 为 0；14 个 `completed_partial`、1 个 `no_progress`，失败行仍保留在分母。隔离 evaluator 得到：

- strict：`5/15`（33.3%）；
- partial：`8.0/15`（53.3%）；
- component：`9/15`；
- reason：`7/15`；
- pair：`5/15`；
- nonzero：`11/15`。

相对上一版 Bank 开发基线 `3/15 strict、4.0/15 partial`，分别提升 `+2` 和 `+4.0`。这组 Bank 数据已经用于开发和误差归因，不能称为 unseen。

2026-10-04 又完成了一次固定 15 行的 Bank 开发回归。运行使用同一 Ark endpoint、G1、Top8、24 次 model/provider 上限，Git checkpoint 为 `0881434`，源码 revision 为 `2a23a25fe9648d5fb3ab3613c9e132298baf3288649fa7f2fa8d8951df5e06b3`，版本为 prompt `m48`、context policy `m2-context-v83`、tool `openrca-public-input-v35`、terminal policy epoch `6`。运行前显式校验了 Python 从当前 checkout 的 `src/` 导入，避免误用旧 editable install。

15 个 RunBundle 全部生成，runner failure 为 0，15 行均为 `completed_partial`，15 行均有非空预测。隔离 evaluator 得到：

- strict：`6/15`（40.0%）；
- partial：`7.0/15`（46.7%）；
- component：`8/15`；
- reason：`6/15`；
- pair：`6/15`；
- nonzero：`8/15`。

这次结果首次超过旧 Bank 最佳的 strict 和 pair，各增加 1 个；partial、component、reason 仍分别低 1.0、1、1。它证明 minimum-disposition、同 reason component corroboration 和 terminal family compatibility 的组合能提高严格命中，同时消除空输出；它还不是跨数据集质量结论。一次 v82 回归曾出现 provider repair 抢占最后一次请求、已完成候选却无法 finalize 的 `budget_exhausted`。当前 v83 用持久化 deferred abstain 修复该恢复窗口；v83 全量没有失败，但该分支在这次 live 全量里没有实际触发，其行为由定向恢复测试覆盖。

随后从 145 个未运行、未评分的 Market 候选中按预注册 SHA-256 顺序选定一例；选择不读取标签，公开输入 manifest SHA-256 为 `83fa8b884eb6c9a9ee51a7e10a9061fbffb4912234ea5d93a1c0d31ada213c9d`。该例只运行一次：`completed_partial`、23 次模型调用、0 次 validator retry；导出一条低置信 recommendation，正式 Finding 为空。隔离官方评分为 strict `0/1`、partial `0/1`、component `0/1`。失败结果已保留，未重抽样本。因此 D9 的跨数据集质量门仍未通过。

公开 RunBundle 显示错误选择来自 `ever_open` 历史状态，而非更强证据：预测与正确项是同一 service 下不同 pod，正确项已在 Top8。移除该路径依赖排序后，对同一决策轨迹的 recorded post-hoc replay 得到 `1/1`。因为样本标签此时已经揭示，这只能证明修复覆盖已知失败，不能改写前述 strict-unseen 的 `0/1`。

D10 已完成真实后端验收；D9 的 Market strict-unseen 质量门仍未通过，因此 RootWalk 不能宣称 v0.1 已完成。
