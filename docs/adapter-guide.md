# RootWalk 适配器指南

RootWalk 目前有 OpenRCA Bank `public_input` 数据适配器、Ark 模型适配器和
recorded model。这里记录现有接缝，方便后续接数据源或模型；当前还没有插件注册机制，
新数据源也不能只放一个文件就自动生效。

## 模型适配器

Runtime 通过 `DecisionModel` 调用模型：

```python
class DecisionModel(Protocol):
    async def decide(self, messages: Sequence[Message]) -> object: ...
```

建议返回 `ModelCallResult`，其中包括经过 Pydantic 校验的 `Decision`、原始响应、provider model 和 usage。Runtime 也接受 `Decision` 或可校验的 Mapping，但这两种返回值缺少完整调用元数据。

一个 Decision 至少包含非空的 `decision_summary` 和 Operation 列表。允许的 Operation 有 `query`、`update_hypothesis`、`submit_path`、`finalize`。只有互相独立的 query 可以批量返回，另外三类必须单独成批。Runtime 会再次检查 `base_revision`、引用和 Exit Gate，模型适配器不要绕过这些检查。

现有实现可直接复用：

- `ModelAdapter` 调用 Ark，从 `ARK_API_KEY` 读取密钥，优先请求严格 JSON Schema；服务端不支持时退回纯 JSON，并只修复一次无效响应。
- `RecordedModelAdapter` 依序读取已录制的 Decision，用于离线协议回归，不代表诊断质量。

新增模型适配器至少要验证这些行为：取消信号继续向外抛出、超时有明确错误类型、错误消息不泄露请求或响应正文、usage 能写入预算流水、无效 Decision 不会进入 Runtime。

## 数据环境的现有接口

`RootWalkRuntime` 当前直接接收 `BankEnvironment`，并使用下面的属性和七个方法：

```text
dataset_id
task
topology

list_entities()
list_metrics(entity_id=None)
query_metric(entity_id, metric_id, period="incident", start=None, end=None)
query_log_samples(entity_id, period="incident", start=None, end=None,
                  log_name=None, contains=None, limit=20)
query_trace_neighbors(entity_id, direction="both", limit=20)
read_artifact_slice(artifact_id, start_line=1, max_lines=50, max_bytes=16384)
summarize_anomalies(entity_ids=None, limit=20)
```

返回值必须是可序列化的 JSON object。查询结果至少要有 `status`；正常查询还应提供 `coverage` 和 `artifact_refs`。`artifact_refs` 用 SHA-256 内容哈希定位来源，并带 selector 和 visibility。不要把本地绝对路径交给模型。

`status=no_data` 不能自动变成负证据。只有 Coverage 明确为 `completeness=complete` 且 `collection_health=healthy` 时，Runtime 才把它登记成带条件的 negative fact；其余情况记录为 gap。

## 新数据源需要改哪些位置

当前数据面还没有抽象成可注册插件，接入另一种环境需要一起改这些入口：

- 为数据源实现严格、只读、确定性的环境类，并产出 `Task` 与 `TopologySnapshot`。
- 在 Runtime 的工具白名单和说明里登记方法，同时补参数校验。
- 在 QueryWorker 子进程里装配同一环境，确保请求和响应都能经过 JSON round-trip。
- 在 CLI `run` 里增加明确的数据源选择与构造逻辑。
- 补充最小公开 fixture，避免测试依赖开发机上的外部目录。

不要先做通用插件框架。现有第二个真实数据源出现后，再从两个实现中抽出共同 Protocol，能避免凭空设计生命周期和配置格式。

## 适配器必须守住的边界

- 初始化时校验文件集合、大小、SHA-256、路径逃逸和 symlink。
- 明确拒绝评测标签。Bank 适配器会拒绝 `query.csv`、`record.csv`，也会递归检查 `scoring_points`、`groundtruth`、`ground_truth` 和 `answer`。
- 工具参数只接受 canonical entity ID、命名时间窗和已列出的 metric ID。
- 日志查询要设上限，字符串筛选默认按字面量处理。
- Artifact 读取同时限制行数和字节数，并在每次读取时重新检查目录边界。
- 查询异常返回稳定的错误类型，不把任意异常正文或本地路径送给模型。
- 慢查询必须经过 `QueryExecutor`。当前 QueryWorker 串行复用一个 spawn 子进程，超时会终止子进程，下一次调用再启动。

## 最小验收

适配器测试至少覆盖成功、无数据、部分数据、参数错误、源文件损坏和超时终止。还要证明：

- 同一冻结输入产生稳定的 Task、Topology 和 Artifact 哈希。
- 未授权实体、工具或时间窗在查询前被拒绝。
- 未知采集完整度不会被写成“没有异常”的证据。
- 子进程超时后没有继续运行，下一次查询可以重新启动。
- 生成的 RunBundle 能通过 `replay`，同一 Bundle 与自身 `diff` 的结果为 `same=true`。
