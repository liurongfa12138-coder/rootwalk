# D1～D11 验收状态

更新时间：2026-10-05。D1～D8、D10、D11 已通过；D9 的 Bank 开发集达到 5/15 strict、8.0/15 partial，但预注册且只运行一次的 Market strict-unseen 为 0/1，因此跨数据集质量门未通过。开发集结果不会被写成未见集结论。

| 编号 | 状态 | 当前证据 | 未完成项 |
| --- | --- | --- | --- |
| D1 | 通过 | `uv sync --frozen`、`rootwalk --help`、recorded run、wheel 和 sdist 均已验证；clean wheel、replay 与源码 checkout 的版本清单一致 | 无 |
| D2 | 通过 | 固定 Bank 切片包含 29 个实体、346 个指标；指标、日志、Trace、Artifact 查询和 19 样本回归通过；Runtime 使用已验证的 Adapter Protocol | 无 |
| D3 | 通过 | Ark Endpoint 返回过 `deepseek-v4-flash-ga-260731`；真实 run 保存请求、响应、usage、Decision、工具结果和状态，并记录实际 provider model | 质量结果归 D9 |
| D4 | 通过 | 引用、方向、缺边、错误实体、unknown 时间、空证据和 refuted Finding 均有测试 | 无 |
| D5 | 通过 | 真实 multiprocessing crash matrix 覆盖响应孤儿、domain ahead、结果未知重试和 seal 后迟到效果；64 MiB Artifact 硬限、Checkpoint、预算和 deadline 恢复也有测试 | 无 |
| D6 | 通过 | Runtime 实际生成 C0、C1/G0、G1 recorded Bundle；实际输入 diff 和 `observe_only` 反馈隔离通过；真实 Ark 三组各跑 3 次 | 效果评测归 D9 |
| D7 | 通过 | RunBundle 原子发布、self-contained 导出、report、replay、OpenRCA exporter、独立 evaluator 和 benchmark 汇总已实现 | 无 |
| D8 | 通过 | Viewer 可查看调查图、Evidence、Hypothesis、Path、Revision rail、预算和双 Run 六层首个分歧；桌面与移动截图 review 结论为 `ship` | 单个 693.51 kB Cytoscape chunk 仍有体积警告，不阻断当前验收 |
| D9 | 部分通过 | 历史 135-run holdout 完整保留；最新冻结 Bank 开发集 15/15 bundle 同 revision、runner failure 为 0，strict 5/15、partial 8.0/15；预注册 Market strict-unseen 单次运行链路完整；其失败经 recorded post-hoc replay 修复为 1/1 | Market strict-unseen 仍记 0/1，post-hoc 不能替代新 holdout；价目未确认 |
| D10 | 通过 | 本地 OTLP 边界测试通过；真实 Langfuse Cloud metadata-only Trace 显示完整 stage 树、内容关闭、真实 Ark usage 与 RunBundle 一致，flush 完成且 dropped=0 | 自托管部署与自定义模型价目不在本次运行环境内 |
| D11 | 通过 | 默认 CI 为 `653 passed, 9 deselected`；显式官方切片全量为 `662 passed`；Ruff lint/format、Pyright、Viewer 16 tests/build 均通过 | 无 |

## 发布判断

工程交付已经覆盖 D1～D8、D10 和 D11。D9 的 Bank 开发集已有可复核提升，但 Market strict-unseen 为 0/1，当前不能标记 v0.1 完成。
