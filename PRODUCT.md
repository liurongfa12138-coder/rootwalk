# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

已确认：Viewer 使用 React、TypeScript、Vite 和 Cytoscape.js，读取本地静态 RunBundle。

## Users

主要用户是构建和调试 RCA Agent 的工程师。他们需要查看一次调查为何选择某条路径、模型看见了哪些证据，以及失败首先出现在哪一轮。

## Product Purpose

RootWalk 把微服务故障调查保存成可验证、可回放、可比较的运行记录。Viewer 要让用户快速定位 Evidence、Hypothesis、Path、Operation、预算和终止原因之间的关系。

## Positioning

RootWalk 展示领域调查事实和约束结果；通用模型调用观测由 Langfuse 承担。界面不能把“grounded”误写成“根因正确”。

## Operating Context

用户从本地 CLI 生成 RunBundle，再用只读 Viewer 检查调查图、证据详情、操作时间线和两次运行的首次可观察分歧。Viewer 不依赖在线后端，也不能修改 RunBundle。

## Capabilities and Constraints

- 所有内容来自真实 RunBundle；没有数据时显示空状态，不生成演示结论。
- 事实拓扑边、候选因果步骤、支持证据、反证和 gap 必须有清楚区别。
- 日志与模型文本是不可信输入，只能按文本渲染。
- v0.1 不含账号、多租户、任务队列、在线控制面或写操作。
- 推断：首个 Viewer 以桌面调试为主，同时保证窄屏可阅读和键盘操作。

## Evidence on Hand

- `examples/bank-recorded-run/`：真实 Bank 输入上的 recorded 协议回归包。
- `runs/bank-row-0001-ark-minimal-5/`：真实方舟调查包，本地保存且不进入源码提交。
- 当前没有可公开宣称的准确率提升、客户案例或生产效果数据。

## Product Principles

- 先展示结论状态和限制，再展示调查细节。
- 证据来源、覆盖和引用始终可追溯。
- 对比只定位首次可观察分歧，不自动宣称失败原因。
- 调试效率高于装饰，所有视觉区分都服务于事实、判断和缺口的辨认。

## Accessibility & Inclusion

键盘可访问、清晰焦点、正文对比度至少 4.5:1；状态不能只靠颜色区分。
