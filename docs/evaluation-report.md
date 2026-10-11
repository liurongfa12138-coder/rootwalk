# RootWalk 评测记录

更新时间：2026-10-05。

这份记录保留开发阶段的全部真实结果，也记录已经完成的 D9 最终 holdout。早期开发运行使用同一个 Bank 案例，但模型参数、超时和代码版本并未完全固定，因此不能把它们当作受控重复实验，也不能与最终 holdout 混算。

## 单案例开发结果

9 个已封存 RunBundle 均使用 `bank-row-0001`，并由固定版本的 OpenRCA evaluator 重新评分。9 个结果的得分都是 0.0。

| Run | 状态 | 模型调用 | 总 Token | 有 Finding | 官方得分 |
| --- | --- | ---: | ---: | --- | ---: |
| `bank-row-0001-ark` | runtime_failure | 1 | 未知 | 否 | 0.0 |
| `bank-row-0001-ark-2` | runtime_failure | 3 | 27,893 | 否 | 0.0 |
| `bank-row-0001-ark-low` | runtime_failure | 3 | 21,571 | 否 | 0.0 |
| `bank-row-0001-ark-minimal` | runtime_failure | 2 | 5,085 | 否 | 0.0 |
| `bank-row-0001-ark-minimal-2` | completed_partial | 8 | 124,684 | 否 | 0.0 |
| `bank-row-0001-ark-minimal-3` | completed_partial | 11 | 146,006 | 是 | 0.0 |
| `bank-row-0001-ark-minimal-4` | completed_partial | 10 | 152,321 | 是 | 0.0 |
| `bank-row-0001-ark-minimal-5` | completed_partial | 12 | 181,905 | 否 | 0.0 |
| `bank-row-0001-ark-timeout60` | runtime_failure | 3 | 16,367 | 否 | 0.0 |

汇总结果：5 次 runtime failure，4 次 completed_partial，7 次没有 Finding，只有 2 次输出 Finding。8 次有完整 Token 记录，总 Token 平均值为 84,479，标准差为 68,555.11。旧运行发生在累计 144,000 输入 Token 硬限制加入之前，所以不能用来证明当前预算上限失效。

这些 RunBundle 没有保存可靠的 wall-clock latency；当前也没有经用户确认的方舟实际结算价和完整缓存计费记录。因此 latency 和 cost 保持 `null`，不使用公开标价推测账单。

## 单案例受控对照

当前代码又对同一开发案例完成了 C0、C1/G0、G1 三组真实 Ark 运行，每组 3 次。模型、工具、数据范围和预算相同；G0 直接复用 C1。独立 evaluator 对 9 次结果的评分均为 0.0。

本表只纳入 `c0-v3-r1..r3`、`c1-g0-v3-r1..r3` 和 `g1-v3-r1..r3`。更早的试跑及修复前暴露 Runtime 缺陷的运行仍保留在本地，但不混入这组对照。

| 条件 | 终态 | 执行失败 | 拒答 | 平均模型调用 | 平均总 Token | 平均 wall-clock |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C0：recent + observe_only | completed_partial × 1、no_progress × 2 | 2/3 | 3/3 | 8.00 | 91,363.67 | 30.74s |
| C1/G0：evidence + observe_only | completed_partial × 1、budget_exhausted × 2 | 2/3 | 3/3 | 8.67 | 95,982.67 | 32.87s |
| G1：evidence + enforce | completed_partial × 2、budget_exhausted × 1 | 1/3 | 3/3 | 9.67 | 85,788.33 | 29.04s |

9 次运行全部保留，合计 5 次执行失败、9 次拒答，没有输出 Finding。总 Token 平均值为 91,044.89，标准差为 25,921.60；wall-clock 平均值为 30.89 秒，标准差为 9.04 秒。每次运行都记录了缓存 Token，但当前没有经用户确认的缓存价、输入价和输出价，所以费用仍为 `null`。

`compare_actual_model_inputs()` 已直接读取这些 RunBundle：C0/C1 的目标变量是 `context_strategy`，G0/G1 的目标变量是 `validator_mode`；两组 `observe_only` 输入都没有 `path_quality` 或 `exit_gate` 反馈泄漏。这只能证明实验隔离有效，零分结果说明当前调查策略没有显示出质量收益。

当前 Runtime 还会冻结源码 revision、`uv.lock` SHA-256、8 个直接依赖版本及其指纹、模型配置、实际 provider model、预算、OTel 配置和查询沙箱 provenance。clean wheel、replay 与源码 checkout 的版本清单已经核对一致。后续正式批跑不能复用这些字段不一致的旧 Bundle。

## D9 v2 数据集

D9 v2 使用固定第三方镜像 `cdreetz/OpenRCA@7a29a7c3600cc07b7760a6c5097d36ac8573f039`。v2 保留 17 个 `task_6` 单故障案例：

```text
2,4,9,17,18,24,59,71,72,75,76,100,105,115,122,124,128
```

row 123 被排除，只因为镜像缺少其参考窗口所需的 `2021_03_22` 遥测。当前 selection SHA-256 为 `921babc8d78ec3955d0fe55c75a27d2c521d7a5ca629099f37a459954dec356b`，case plan SHA-256 为 `c8c4f6acbec6371bdc7423dab98aec779285597390d5cdf26b0e55e87a161819`，source files SHA-256 为 `fe62934a0e800e6f37c30a5ed02d88f043584ef061b5fa376daf214c4a94b7b6`。

镜像源文件合计 17,968,974,882 bytes。03-04 的四个遥测文件与既有官方 profile compatibility anchor 的大小和 SHA-256 对齐；这个锚点不能证明其余六天与官方归档逐字节一致。

公开 suite 已生成，实际大小为 2,539,327,454 bytes，约 2.4 GiB，包含 17 个 case 和 204 个受哈希约束的输出。私有评测包包含 17 行，目录权限为 `0700`，两个文件权限均为 `0600`。row 到标签的映射没有进入 Runtime、公开输入或 dry-run 计划。

## 外部来源探测

当前主机连接 `drive.google.com` 和 `drive.usercontent.google.com` 时在 TCP 阶段超时。`www.googleapis.com` 可以连接，但 Drive v3 `alt=media` 在没有 API 身份时返回 HTTP 403。browser-use 导航也超时。公开检索没有找到能替代官方逐字节证明的可信镜像；当前使用的是固定第三方镜像，并受上述 compatibility anchor 限制。

## 153-run dry-run 与最终 135-run holdout

`run_bank_benchmark.py` 已真实生成 153 项 dry-run 计划，对应 17 个案例、3 个条件和每条件 3 次重复。153 个 run ID 全部唯一，并绑定 suite spec 与 case manifest；计划命令中的标签路径命中数为 0。

付费执行必须显式传 `--confirm-paid-run`，并可用 `--max-new-runs` 限制新增 Run。超过上限的计划项记为 `deferred`，不进入 attempted failure/refusal rate；planned diagnostic output rate 仍以全部计划项为分母。

最终实验排除开发行 2 和 128，用其余 15 行组成 holdout。3 个条件各重复 3 次，共完成 135 次运行。`run_plan.json` SHA-256 为 `d4be31d7ae9f6090f33668113a588a8b0201d8a0c5be8dfac44b8192dc422650`；135 个 Bundle manifest 的 `code_revision` 均为 `cb28152017c51e1db6a67999c7d4f2ce1a544b0d7f754e086eaaf18aaca95af9`。runner failure 为 0，9 个预测文件全部成功评分。

| 条件 | 运行数 | 严格命中 | 部分得分 | 非零部分分 | 诊断输出 | 拒答 | 失败终态 | 总 Token | 平均 wall-clock |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 | 45 | 0 | 0.5 | 1 | 22 | 23 | 11 | 3,862,392 | 37.74s |
| C1/G0 | 45 | 0 | 1.0 | 2 | 20 | 25 | 9 | 4,060,782 | 39.24s |
| G1 | 45 | 0 | 0 | 0 | 15 | 30 | 8 | 4,343,389 | 42.73s |
| 合计 | 135 | 0 | 1.5 | 3 | 57 | 78 | 28 | 12,266,563 | 39.9048s |

严格准确率为 **0/135**。部分得分为 **1.5/135（0.011111）**，132 次得 0 分，3 次得到非零部分分。这里的非零只表示组件或原因有局部匹配，没有一次同时满足严格答案，不能据此宣称诊断质量可用。

135 次运行的 wall-clock 逐次相加为 5,387.14 秒。终态分布为 completed_partial 107、no_progress 22、runtime_failure 3、budget_exhausted 2、context_limit 1。28 次失败终态是 Run 的状态统计，不是 runner failure；runner 本身没有漏跑计划项。没有确认模型与缓存价目，所以 cost 保持 `null`。

## 正式批跑前的单案例修复记录

Key 通过临时进程环境重新注入后，`probe-model` 成功返回 `deepseek-v4-flash-ga-260731`，使用 1,407 Token。此前的阻塞是运行进程没有环境变量，不是供应商返回无效 Key。随后将凭证保存到用户私有配置文件，权限 `0600`；文件不在仓库、RunBundle 或模型输入中。

在不读取 row 2 隐藏标签的前提下，连续单案例运行暴露并修复了 Context Evidence 引用、recent 排序、查询批次、重复查询召回和 Decision repair 问题。修复后的正式结果如下：

| 条件 | 终态 | 模型调用 | 总 Token | wall-clock | 候选 | Finding | 官方得分 |
|---|---|---:|---:|---:|---:|---:|---:|
| C0：recent + observe_only | budget_exhausted | 9 | 140,596 | 50.70s | 1 | 0 | 0.0 |
| G1：evidence + enforce | completed_partial | 11 | 137,596 | 40.47s | 1 | 0 | 0.0 |

G1 没有 Runtime failure 或 Validator retry。它维护了一个候选并在缺少支持路径时主动拒答，说明运行约束和 Evidence 引用已经生效；也说明当前调查策略仍无法识别正确根因。独立评分后揭示的正确项为 `Tomcat01 / high memory usage`，所以 row 2 现在只能作为开发复盘案例，不能继续针对该答案调参或计入未见测试集。

这两次单案例运行只保留为开发复盘，不计入最终 135-run holdout。最终实验已经执行完毕，但严格准确率仍为零；下一轮需要改进通用调查策略，并在新的未见集上重新验证。当前也没有确认价目可补算成本。

## 2026-10-05 调查策略复验

旧 Bank holdout 已用于误差归因，因此本轮 15 行 G1 结果只算开发集复验。15 次运行均为 `completed_partial`，总计 255 次模型调用、2,874,728 Token，7 次输出 Finding，8 次输出低置信 benchmark recommendation。独立评分为 3/15 strict、4.5/15 partial，平均 wall-clock 为 87.6236 秒。3 个严格命中全部来自低置信 recommendation；7 个正式 Finding 的严格命中为 0。评测文件 SHA-256 为 `3e145e321eed5d2242e6b60f88fba6c62d63dc415f4debf8cdd2f6a0ba384a54`。

Market 另选了此前没有进入历史 RunBundle 的案例。首个案例的三次运行都在首轮模型调用前触发 `context_limit`；必选上下文为 37,185 tokens，主要来自 271 个实体和 142 条拓扑边。修复为有界 scope 和拓扑投影后，三次都完成调查，但仍为 0/3。

随后把 Market 候选表改为固定上游 `microsoft/OpenRCA@c1bd4af7f635171a1c31cdd567c07d698dff6abc` 中公开的 56 个组件和 15 个原因，并按同一公开契约规范化导出名称。第二个案例在运行前完成选择和输入哈希固定，三次运行均为 `completed_partial`，合计 45 次模型调用、439,934 Token，评分仍为 0/3。脱敏归因只返回覆盖情况：正确原因位于 16 个公开候选中，正确组件没有进入候选。该结果说明下一步应补齐 Market 的多层实体归一化和 service-level 候选生成，不能只继续调模型提示词。
