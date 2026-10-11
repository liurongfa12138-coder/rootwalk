# RootWalk 安全说明

RootWalk 处理日志、Trace、模型输入和模型响应，这些内容可能带有内部标识或业务数据。当前实现适合受控的本地调查环境，还不是面向不可信租户的安全沙箱。

## 信任边界

| 边界 | 当前处理 |
| --- | --- |
| 冻结数据进入 BankEnvironment | 要求目录名为 `public_input`，核对精确文件集、大小、SHA-256，拒绝 symlink 和路径逃逸 |
| 数据进入模型 | 只通过白名单工具、canonical entity ID、命名时间窗和有界结果；Artifact 用内容哈希引用 |
| 模型返回 Runtime | 用严格 Pydantic Schema 校验 Decision，整批校验失败时不执行其中任何 Operation |
| 查询执行 | 请求和响应经过 JSON round-trip；Linux 默认用 bubblewrap 隔离文件系统和网络，超时直接终止子进程 |
| Runtime 落盘 | SQLite 事务、稳定键冲突检查、单写者文件锁、目录身份校验、64 MiB Artifact 硬限、内容寻址 Artifact 和原子 Bundle 发布 |
| RunBundle 被读取 | Python `replay` 和 `diff` 校验 manifest 声明的文件 SHA-256；Viewer 只做结构解析，不复算哈希 |
| 可观测数据外发 | 默认禁用；默认只保留安全元数据和内容哈希，exporter 超时或失败不阻断主流程 |

SHA-256 能发现文件与 manifest 不一致，但 manifest 没有数字签名。能同时改文件和 manifest 的人仍可重算哈希，所以 RunBundle 不能当作不可抵赖的审计凭据。

## 密钥

Ark 密钥只从 `ARK_API_KEY` 环境变量读取。不要把密钥写入项目配置、命令参数、
recorded response、RunBundle 或截图。当前适配器会把供应商异常归一成错误类型、
状态码和 request ID，不回显请求体、响应体、model 参数或 base URL。

运行真实模型前可先确认环境变量存在，但不要打印它的值：

```bash
test -n "$ARK_API_KEY"
```

如果怀疑密钥已经进入 Artifact 或 Git 历史，应先撤销密钥，再处理本地文件和历史记录。仅从工作区删除明文并不能让已经泄露的密钥恢复安全。

## 数据最小化

- 只给 `public_input` 放调查所需字段，不要混入 `query.csv`、`record.csv` 或评测答案。
- Bank 适配器会递归拒绝常见评测标签，避免答案泄漏到模型上下文。
- 日志样本和 Artifact slice 都有数量或字节上限；每个 Run 新写入 Artifact 的硬上限为 64 MiB。默认导出把大文件保留为 external 引用。
- 显式使用 `rootwalk export --self-contained --external-root ...` 时，导出器会从领域 records 重新汇总 visibility，并与 index 核对一致。任一侧出现 `evaluation_only`，即使文件换了布局或 index 被降级篡改，也会在读取正文前拒绝。
- 同一个 Artifact 同时被 `model_readable` 和 `audit_only` 引用是合法情况，汇总结果为 `model_readable`。通过 visibility 检查后，导出器还会核对 external manifest、大小、SHA-256、symlink 和路径边界。
- 模型输入和原始响应会作为 embedded Artifact 保存。RunBundle 在发布前仍要按数据分级检查，不能因为文件名是哈希就认为内容已经脱敏。
- `capture_content=False` 是 OpenTelemetry 的安全默认值。只有 exporter、存储位置和访问权限都明确时才打开正文采集。

## OpenRCA 数据准备

真实数据准备必须使用仓库外的绝对路径，并把下载源、公开切片、上游代码和评测标签分开。`fetch_openrca_bank.py` 只允许 spec 计算出的 29 个 ZIP 成员，明确拒绝 `record.csv`；带评分字段的 `query.csv` 只写入权限收紧的 `private_source/`。

`prepare_bank_suite.py` 会拒绝 Git 工作区内的输出，也会拒绝输入输出目录重叠。它只把题面和遥测写入公开切片。`prepare_openrca_eval.py` 另行生成 evaluator-only 包，目录权限为 `0700`，文件权限为 `0600`。这个标签包只能交给 `rootwalk evaluate`，不能挂载给调查 Runtime。

真实遥测没有确认单独的再分发许可。不要提交下载目录、公开切片、标签包或下载缓存，也不要把它们上传为 CI Artifact。OpenRCA 代码的 MIT 声明见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## QueryWorker 沙箱

`rootwalk run --query-sandbox auto` 是默认值：Linux 且 `bwrap`、mount plan 和输入可读性预检均通过时启用文件系统和网络隔离，否则发出不含路径的 warning 并退回普通 spawn 子进程。`required` 在任一预检失败时关闭运行；`off` 仅用于没有 bubblewrap 的可移植环境，并明确放弃系统级文件隔离。

沙箱从空 rootfs 启动，只读绑定 Python runtime、当前虚拟环境、`src/rootwalk` 包和当前 `public_input`，另提供最小系统动态库、`/proc`、`/dev` 与空 `/tmp`。mount source 若解析为 `/`、用户 home、RootWalk workspace 根或这些敏感目录的任一祖先都会直接拒绝；运行前还会以沙箱 UID 65534 验证 `public_input` 树可读。它不绑定仓库其余内容、父目录、`hidden_eval` 或环境变量凭证，并创建独立 network namespace。W3C Trace Context 只通过已存在的 JSON Pipe 请求传递。

每个 Run 的 manifest 和 report 都记录 `query_sandbox_requested`、`query_sandbox_effective`、`query_sandbox_bwrap_version` 与 `query_sandbox_bwrap_executable_sha256`。Runtime 会把这些配置与实际 QueryExecutor 核对，恢复时全部冻结，不能把原本的 bwrap 运行静默降级为 spawn 或换用另一份 executable。worker 启动失败只返回有界错误类别和 stderr 摘要哈希，不回显原始 stderr、凭证或宿主机路径。

## 环境与观测冻结

Run 还会记录源码 revision、`uv.lock` SHA-256、直接依赖版本及其指纹、模型配置、实际 provider model 和 OTel 配置。恢复或复用时若这些字段变化，Runtime 或 D9 runner 会拒绝沿用旧 Bundle。

这条 Trace Context 传递不会把 `traceparent` 或 `tracestate` 写进模型输入、RunBundle 或日志。默认关闭内容采集；完整 stage spans、drop 计数和本地 HTTP OTLP 已通过测试。真实 Langfuse Cloud 的 metadata-only ingestion、调用树、内容关闭、真实模型 Token 映射和有界 flush 也已验证；自托管实例与自定义模型价目仍是部署侧事项。步骤与验收项见 [langfuse.md](langfuse.md)。

## 本地文件

`.work` 和 RunBundle 都可能含模型正文、日志样本与诊断结论。建议把它们放在访问受控的磁盘，使用操作系统权限限制读取，并按数据保留策略清理。仓库根目录已经忽略 `runs/` 和 `.*.work/`，这只是降低误提交概率，不是保密机制。

Viewer 在浏览器内用 `File.text()` 读取用户选择的文件，没有上传后端。页面通过 React 文本节点和 `<pre>` 展示数据，没有把日志或模型文本当 HTML 注入。它仍会把完整选择内容放进浏览器内存，不要在不受信任的共享机器上打开敏感 Bundle。

## 当前缺口

- 没有 Bundle 签名、加密、密钥轮换或远程证明。
- 没有多租户隔离、RBAC、审计后台或安全删除。
- Viewer 没有哈希校验，不能替代 CLI 完整性检查。
- 非 Linux 或显式 `--query-sandbox off` 时，QueryWorker 只有进程隔离和超时边界，不是针对恶意 Python 代码的系统级沙箱。
- 依赖漏洞扫描、发布制品签名和 SBOM 尚未进入 CI。

在这些缺口补齐前，RootWalk 不应接收任意用户数据，也不应加载任意工具代码。
