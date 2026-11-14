# 阶段 4.2：PDF 证据生成与页级引用

> 本文档固化阶段 4.2 经用户逐项确认的实现契约与验收边界。

- 状态：人工验收通过，等待发布授权
- 分支：`codex/pdf-grounded-generation`
- 上游：阶段 4.1 PDF `SearchResult` 及完整 provenance
- 后续：阶段 5.1 检索评估基线

## 1. 目标与边界

```text
用户问题
  -> PDF 查询适配
  -> 精确 Top-K SearchResult
  -> 多证据选择与稳定编号
  -> 受证据约束的生成
  -> 回答状态与 Citation validation
  -> document / generation / record / page / block / bbox
```

本阶段不加入混合检索、重排器、未经评测的相似度阈值、自动查询改写、多轮会话状态、
答案 UI、生产 API、部署或正式答案评估。

## 2. PDF 查询适配与 Evidence 组织

- 查询适配器接收自然语言，校验空白、`top_k`、token 上限及 Embedder 与 manifest 兼容性。
- 适配器绑定 `index_id`，构造 `VectorSearchQuery` 并调用阶段 4.1 `VectorIndex.search()`。
- 检索结果保持原始顺序，不重排、不设分数阈值、不将故障伪装为空结果。
- Evidence 按排名分配 `[1]`、`[2]` 等临时编号，在字符预算内选择连续排名前缀。
- Chunk 只能完整进入上下文；不截断、不跳过高排名证据去容纳低排名证据。

## 3. Evidence 与 Citation 契约

一个 `[n]` 对应一个完整 PDF Chunk。答案实际使用的每个不同 Evidence 生成一条
Citation；一条 Citation 可包含多个跨页或同页 region。Citation 由应用程序从
`SearchResult.record.provenance` 构造，模型不得生成 document、generation、record、page、block 或 bbox。

核心 Citation 使用稳定 `document_id`；真实预览通过 manifest 补充 sample ID 和 PDF 路径。
本阶段不为此修改数据库 schema。

## 4. 生成输入输出与回答状态

可信规则与不可信的用户问题/Evidence 分开呈现；不可信部分使用 JSON 序列化。模型只允许返回：

```json
{
  "status": "answered",
  "answer": "使用证据的回答 [1]"
}
```

允许状态及引用要求：

| 状态 | 语义 | Citation |
| --- | --- | ---: |
| `answered` | 证据完整支持回答 | 至少 1 |
| `partial` | 只能回答部分问题 | 至少 1 |
| `insufficient_evidence` | 当前证据不足 | 0 |
| `conflicting_evidence` | 至少两条证据无法同时成立 | 至少 2 |
| `needs_clarification` | 问题含糊、不完整或多义 | 0 |

无检索结果时应用程序直接返回 `insufficient_evidence`，不调用模型。有 Evidence 时，模型在同一次
生成中提出状态和回答；不额外调用分类模型，v1 不自动修复或重试非法输出。

## 5. Citation validation

模型调用前校验所有选中 Evidence 的 record 身份、provenance schema、source regions、page、block、
bbox、Chunk 字符范围和 extraction provenance。模型调用后校验 JSON、状态、引用编号及
Citation 数量。

任何损坏或不支持的 provenance 都 fail-closed；不跳过、不猜测、不伪装为证据不足。本阶段证明引用
真实存在且可定位；陈述与证据之间的完整语义忠实度属于后续答案评估阶段。

## 6. 测试和人工验收

采用三层验证：

1. 默认离线测试使用假 Embedder、假索引和假生成器，不访问网络、PostgreSQL、真实模型或 GPU。
2. 显式 PostgreSQL 集成测试使用假查询向量验证真实持久化契约，仅允许连接名称包含 `_test` 的数据库。
3. 真实 PDF/模型预览复用 `native.pdf`、`layout.pdf`、`scan.pdf`，通过显式命令输出完整 JSON 与 Markdown。

真实预览通过通用 `TextGenerator` 边界注入 `DeepSeekTextGenerator`。适配器默认连接 DeepSeek 官方
Base URL，从当前进程或项目根目录 `.env` 读取 `DEEPSEEK_API_KEY`；可选的
`DEEPSEEK_BASE_URL` 只用于显式覆盖。OpenAI 与 DeepSeek 保留独立适配器，不把供应商分支带入 PDF
问答服务。

扫描 PDF 复合问题必须组合多条 Evidence；不可回答和含糊问题分别验证
`insufficient_evidence` 与 `needs_clarification`。少量人工预览不宣称 Recall@K、MRR 或正式忠实度得分。

## 7. 验收结果

- Ruff 格式和静态检查通过；默认测试集 269 项通过，1 项显式 PostgreSQL 测试按设计跳过。
- 显式 PostgreSQL 集成入口完成迁移并通过 2 项真实 pgvector 测试。
- DeepSeek 真实预览的 5 个固定问题全部完成：3 个 `answered`、1 个
  `insufficient_evidence`、1 个 `needs_clarification`。
- 三个可回答问题分别产生 1、4、2 条 Citation，预期 PDF 和页码全部命中；拒答与澄清均为 0
  Citation。
- 真实运行暴露并修复了两项回归：状态专属引用规则未完整写入 Prompt，以及数学方括号被误判为
  Citation。两项均保留 fail-closed 校验并增加自动化测试。
- 用户于 2026-11-15 完成人工报告检查并确认进入下一步。
