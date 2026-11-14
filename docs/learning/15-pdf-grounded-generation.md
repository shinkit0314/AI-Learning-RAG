# 学习 15：PDF 证据生成与页级引用

## 1. 本阶段解决了什么

阶段 4.1 已能从 PostgreSQL/pgvector 返回带完整 provenance 的 PDF `SearchResult`，但检索结果还
不是最终回答。阶段 4.2 补齐从自然语言问题到可审计回答的边界：

```text
自然语言问题
  -> Query Embedding
  -> 指定 index_id 的 Top-K 向量检索
  -> 完整 Evidence 预算选择与 [n] 编号
  -> DeepSeek 受证据约束生成
  -> 状态、JSON 和引用编号校验
  -> page / block / bbox Citation
```

模型只负责返回回答状态、自然语言答案和临时 Evidence 编号。稳定的 document、generation、record、
page、block、bbox 和 extraction provenance 全部由应用程序从已验证记录构造，模型没有权限生成这些
来源字段。

## 2. 为什么查询适配与底层索引分开

`VectorIndex.search()` 接收向量和 `index_id`，保持底层索引与具体 Embedding 实现无关。
`PdfRetriever.search()` 接收自然语言，负责校验问题、token 上限、模型与 manifest 的兼容性，再构造
`VectorSearchQuery`。这种分层让底层精确检索可以独立测试，也避免每个调用方重复查询向量化逻辑。

当前检索结果按向量相似度保持原始顺序，没有过滤、混合检索或重排。阶段 5.1 会先建立固定评测集
和 Recall@K/MRR 基线，之后才比较检索增强策略。

## 3. Evidence 与 Citation 不是同一个概念

- Evidence 是模型本次可以阅读的候选 Chunk。
- Citation 是答案实际引用的 Evidence。

检索返回 Top-K 后，系统按排名选择能够完整装入字符预算的连续前缀。Chunk 不截断，也不会跳过
较高排名的大 Chunk 去容纳低排名内容。模型可以看到 5 条 Evidence，但只引用其中 1 条；报告会同时
保留所有选中和排除的 Evidence，便于解释答案为何产生。

`[1]` 只表示本次 Prompt 中的临时 Evidence 编号。它在应用层映射回稳定的 record 和完整 PDF 来源。
数学表达式中的普通方括号不是 Citation；只有内容恰好为正整数的 `[n]` 才进入引用校验。

## 4. 回答状态与 fail-closed 边界

生成器只允许返回含 `status` 和 `answer` 的 JSON。状态与引用数量必须一致：

| 状态 | 引用要求 |
| --- | ---: |
| `answered` / `partial` | 至少 1 条 |
| `conflicting_evidence` | 至少 2 条 |
| `insufficient_evidence` / `needs_clarification` | 0 条 |

无检索结果或预算无法容纳完整 Evidence 时，应用程序直接返回证据不足，不调用模型。模型输出不是合法
JSON、使用不存在的编号、状态与引用数量冲突，或来源 provenance 损坏时，系统都会显式失败，不猜测
也不自动修复。

## 5. 为什么增加 DeepSeek 专用适配器

RAG 服务依赖通用 `TextGenerator.generate(prompt)` 协议，而供应商认证、Base URL、模型名称和请求参数
保留在各自适配器中。`DeepSeekTextGenerator` 使用 DeepSeek 官方 OpenAI 兼容 Responses API：默认
Base URL 为 `https://api.deepseek.com`，从当前环境或项目根目录 `.env` 读取
`DEEPSEEK_API_KEY`。这不会把 DeepSeek Key 发送给 OpenAI。

通用的是业务接口，不是把所有供应商塞入一个带大量条件分支的实现。以后出现稳定重复时，可以抽取
内部共享传输代码，但仍保留明确的供应商边界。

## 6. 三层验证和真实结果

1. 离线测试使用假 Embedder、索引和生成器，覆盖查询适配、预算选择、状态和引用校验。
2. 显式 PostgreSQL 测试验证真实 pgvector 持久化和搜索契约，只允许测试数据库。
3. 真实 PDF/DeepSeek 预览复用 `native.pdf`、`layout.pdf`、`scan.pdf`，输出完整 JSON 和 Markdown。

最终自动门禁为 Ruff 格式检查通过、Ruff 静态检查通过、269 项测试通过，1 项显式 PostgreSQL 测试
按默认规则跳过；显式 PostgreSQL 入口另有 2 项通过。

真实模型的五个固定问题得到：

| 问题类型 | 状态 | Citation |
| --- | --- | ---: |
| 期待收益定义 | `answered` | 1 |
| AI Agent 核心特征 | `answered` | 4 |
| 可扩展性与可维护性 | `answered` | 2 |
| 不存在的 2027 部署预算 | `insufficient_evidence` | 0 |
| 指代不明的“它为什么重要” | `needs_clarification` | 0 |

真实运行还发现两项单元假数据没有暴露的问题：Prompt 最初没有明确状态专属引用数量，数学公式中的
方括号也会被宽泛正则误判为 Citation。两项问题都通过收紧契约而不是放宽校验修复，并补充回归测试。

## 7. 已知限制与下一步

- Top-K 仍是纯向量排序，未做混合检索或 rerank。
- 20,000 字符预算是 v1 的确定性保护，不等同于正式 token 成本优化。
- 程序验证引用真实存在且可定位，但逐句语义忠实度仍需要人工审核或后续答案评估。
- 五个真实问题是人工预览，不代表 Recall@K、MRR 或统计显著的质量结论。
- v1 不自动改写含糊问题，不维护多轮会话，也不重试非法模型输出。

下一阶段是 5.1 检索评估基线：先固定问题和相关 Chunk 标注，再计算 Recall@K、MRR 和可重复报告。
只有建立基线后，后续切分、Embedding、混合检索和重排实验才有客观比较依据。
