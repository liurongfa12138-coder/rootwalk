# Langfuse / OpenTelemetry 接入

## 当前状态

观测默认关闭。RootWalk 的运行、回放和导出都不依赖 Langfuse。

`uv.lock` 已固定客户端配置：

- OpenTelemetry API、SDK 和 OTLP HTTP exporter：`1.45.0`
- 传输协议：OTLP/HTTP protobuf
- service name：`rootwalk`
- 属性映射：`openinference-rootwalk-v1`
- 内容采集：默认关闭
- 开启观测后的默认采样率：`1.0`

自托管验收时还要记录 Langfuse 的确切 release 或不可变镜像 digest，以及匹配的 Compose revision。不要使用 `latest` 或随 `main` 变化的配置。本机没有 `docker`、`podman` 或 `nerdctl`，因此本文不声称已经验证自托管部署。

## 连接方式

在独立的 Langfuse 项目中取得 public key、secret key 和 base URL。凭证只放环境变量，不要放进命令参数、日志、截图或 RunBundle。

```bash
export LANGFUSE_BASE_URL='https://<your-langfuse-host>'
export LANGFUSE_PUBLIC_KEY='...'
export LANGFUSE_SECRET_KEY='...'

auth="$(printf '%s' "${LANGFUSE_PUBLIC_KEY}:${LANGFUSE_SECRET_KEY}" | base64 | tr -d '\n')"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Basic ${auth},x-langfuse-ingestion-version=4"
unset auth

uv run rootwalk run \
  --data-root "$RW_DATA_ROOT" \
  --output /tmp/rootwalk-langfuse \
  --run-id langfuse-smoke \
  --recorded-decisions examples/bank-recorded-decisions.json \
  --otel \
  --otel-endpoint "${LANGFUSE_BASE_URL%/}/api/public/otel/v1/traces" \
  --no-otel-capture-content \
  --otel-sample-ratio 1.0
```

也可以使用通用的 `OTEL_EXPORTER_OTLP_HEADERS`。`--otel-endpoint` 接收完整的 Trace 地址，包含 `/v1/traces`。只配置地址不会开启观测，还要传入 `--otel` 或设置 `ROOTWALK_OTEL_ENABLED=true`。

2026-09-16 之后创建的 Langfuse Cloud 组织不能使用旧的
`GET /api/public/traces` 查询链路。自动验收应改用
`GET /api/public/v2/observations?fromStartTime=<from>&toStartTime=<to>`；旧接口会返回
`410 LEGACY_API_UNAVAILABLE_FOR_NEW_ORGANIZATION`。

仓库内的 `scripts/run_langfuse.sh` 会读取权限为 `600` 的 `.env.langfuse`，生成认证
Header，并按照 `ROOTWALK_OTEL_CAPTURE_CONTENT` 选择内容模式。它不会把 Key 放进命令参数：

```bash
scripts/run_langfuse.sh \
  --data-root "$RW_DATA_ROOT" \
  --output /tmp/rootwalk-langfuse \
  --run-id langfuse-smoke \
  --recorded-decisions examples/langfuse-smoke-decisions.json \
  --validator-mode observe_only
```

完整内容模式会把脱敏后的模型输入输出、Context、工具参数和结果、Evidence、Hypothesis、
Path 与最终报告发送到配置的 Langfuse 项目。要强制使用 metadata-only 模式，继续调用
`scripts/run_langfuse_metadata.sh`。

## 预期调用树

一次执行产生一个新的 `rootwalk.run` 根 Span。它的直接子节点包括启动持久化、`rootwalk.graph` 和 `rootwalk.export`。Graph 内部包含：

- 每轮 Context 编译对应一个 `rootwalk.context`；
- 每次 Runtime 模型调用只对应一个 `rootwalk.model`；
- 路由和结束校验对应 `rootwalk.validation`；
- 每次工具尝试对应 `rootwalk.tool`；
- Observation/Evidence 提交对应 `rootwalk.evidence`；
- Hypothesis 或 Path 提交对应 `rootwalk.investigation_graph`；
- 报告和封存对应最终的 `rootwalk.persistence`。

关闭内容采集后，Span 属性只保留关联 ID、哈希、数量、模型标识、映射版本和 Token 用量。Prompt、回复、日志正文、Evidence 正文、鉴权头和 API Key 都不能进入 Span。恢复执行时，新进程使用相同的 `run_id`、新的 `execution_attempt_id` 和新的 Trace，并通过 Link 关联前一次执行的根 Span。

本地 `telemetry_dropped_count` 统计 shutdown 返回前尚未确认交付的 Span，包括队列饱和、exporter 失败和 shutdown 超时。超时中的 export 后续仍可能成功，因此该计数不会回退，可能保守高估实际丢失量。

## 验收清单

只有固定版本的 Langfuse 部署证明以下行为后，才能关闭 D10：

1. OTLP 请求成功，Trace 能在 Langfuse 中看到。
2. Span 数量与 Runtime 账本一致。每次模型调用只产生一个 Generation，模型和 Token 用量一致，不重复计费。
3. Tool、Evidence、Investigation Graph、Persistence 和 Export Span 的父子关系正确，属性中只有允许的元数据。
4. 同一 Run 恢复后，新 Trace 能关联前一次执行的根 Span。
5. 在 Prompt、回复、日志和 Evidence 中放入唯一测试文本；关闭内容采集后，OTLP 载荷和 Langfuse 中都找不到该文本。
6. 无效凭证、后端断开和队列饱和都不改变 RootWalk 报告；丢弃数量单独记录，清理时间有上限。
7. 在其他条件相同的情况下，分别记录开启和关闭观测的耗时与队列开销。

仓库已经用内存 exporter、本地 HTTP 接收端和 Langfuse Cloud v4 验证 Runtime 到
OTLP 的边界。2026-10-05 的 metadata-only Cloud smoke 在一个 Trace 中收到 14 个
Observation：Root 1、Graph 1、Context 2、Model Generation 2、Validation 3、Tool 1、
Evidence 1、Persistence 2、Export 1；所有 input/output 均为空，flush 成功且 dropped=0。
随后一次真实 Ark model Generation 映射出 input 8,023、output 60、total 8,083 Token，
与 RunBundle 一致，仍未上传正文。Langfuse 项目未配置该自定义模型价目，所以 cost 为
`null`；自托管部署行为也仍需固定版本实例单独验证。
