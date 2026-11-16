# 阶段 5.1：检索评估基线

> 本文档固化阶段 5.1 经用户逐项确认的实现契约与验收边界。

- 状态：开发中
- 分支：`codex/retrieval-evaluation-baseline`
- 上游：阶段 4.2 PDF 证据生成与页级引用
- 后续：阶段 5.2 切分与 Embedding 实验

## 1. 目标与边界

本阶段把“检索结果看起来不错”升级为固定、可重复、可审计的质量基线：

```text
固定 PDF 与 Chunk 集合
  -> 固定问题
  -> 人工 Chunk 二元相关性
  -> 当前纯向量 Top-5
  -> Recall@1/3/5 与 MRR@5
  -> JSON / Markdown 报告
  -> 版本化人工验收基线
```

本阶段不加入 reranker、混合检索、元数据过滤、查询改写、Embedding/Chunk 参数调优、答案忠实度
评估、UI 或部署。现有少量页码级预览只作为候选问题来源，不直接包装成正式指标。

## 2. 人工相关性标注

相关性是“问题与 Chunk 的关系”，不是 Chunk 自身的永久属性。一个 `answerable` 问题可以对应多个
相关 Chunk；Chunk 包含回答问题所需的直接证据才标为相关，正确 PDF、正确页码、主题相近或只含
关键词均不足以构成相关。

正式身份使用：

```text
document_id + chunk_fingerprint
```

`page_start`、`page_end`、`section_path`、`source_block_ids`、内容类型和完整正文仅用于人工审核。
不使用会随索引代际变化的 `record_id`、用于显示排序的 `chunk_index`，也不使用只有页码的粗粒度
预期。

重叠 Chunk 只要都独立包含直接证据，就都标为相关；跨页 Chunk 仍作为一个 Chunk。候选标注与
正式 JSON 分离，只有用户明确批准的标注才能进入指标。正式评测遇到任何 `draft` 必须拒绝运行，
不得静默跳过。

## 3. 问题类型与规模

问题类型为：

| 类型 | 相关 Chunk | 正式指标 |
| --- | ---: | --- |
| `answerable` | 至少 1 个 | 进入 Recall/MRR |
| `unanswerable` | 0 个 | 仅诊断 |
| `ambiguous` | 0 个 | 仅诊断 |

先为三份 PDF 各准备 2 个问题，共 6 个试标问题，验证标注和报告流程；通过后扩充到每份 5 个，共
15 个正式问题，并增加 2 个 `unanswerable` 和 1 个 `ambiguous` 诊断问题。空相关集合不进入
Recall/MRR，不伪装成 0 分或满分。

## 4. 语料与 Chunk 兼容边界

评测集绑定固定语料文件 SHA-256、稳定 `document_id`、切分/准入 profile，以及全部已准入 Chunk
身份形成的集合指纹。实际评测报告另外记录完整 `index_id` 和 manifest。

- 只更换 Embedding 或重建索引，而 Chunk 集合不变：允许使用同一评测集比较；
- 新增或移除 PDF、Parser/OCR 输出变化、切分边界变化或准入结果变化：拒绝使用旧语料快照；
- 任一人工相关 Chunk 不存在于当前兼容语料：拒绝运行，不将契约错误伪装成 Recall 下降；
- 扩充语料时保留旧基线历史，重新审核旧问题是否出现新相关证据，并建立新版本评测集和基线。

## 5. 指标

一次检索 Top-5，计算：

```text
Recall@K = Top-K 找回的相关 Chunk 数 / 人工标注的全部相关 Chunk 数
MRR@5    = 所有正式问题的第一条相关 Chunk 倒数排名的平均值
```

报告 `Recall@1`、`Recall@3`、`Recall@5` 和 `MRR@5`。第一条相关 Chunk 未出现在 Top-5 时，
该正式问题的 reciprocal rank 为 0。所有正式问题等权宏平均，不让相关 Chunk 较多的问题获得更高
权重。

## 6. 报告与基线升级

JSON 保存数据集、语料、实际索引、汇总指标、逐问题指标、完整 Top-5、命中、遗漏和 provenance；
Markdown 保存同一信息供人工审核。报告同时比较汇总变化与逐问题变化，防止一个问题的提升掩盖
另一个问题的退化。

基线是当前经过人工验收的正式参考版本，不是自动更新的历史最高分。新实验只有在兼容条件下检查
指标权衡、逐问题退化并通过人工验收后，才能发布为新的版本化基线；旧基线不得覆盖。

## 7. 测试和真实验证

采用三层验证：

1. 默认离线测试使用受控排名，验证数据契约、兼容拒绝、Recall/MRR、空相关集合和报告完整性；
2. 显式真实评测使用三份 PDF、固定 Qwen Embedding 和 PostgreSQL/pgvector；
3. 人工审核 Markdown 候选、正式金标和最终基线报告。

真实模型和 PostgreSQL 不强制进入默认 pytest/CI；显式入口必须使用锁定依赖、本地模型和已迁移的
数据库。阶段交付前仍执行完整 Ruff 和 pytest 质量门禁。

## 8. Ragas 后续边界

Ragas 计划在里程碑 6“答案质量评估”阶段以离线、显式方式接入，候选指标包括 Faithfulness、
Response Relevancy、Context Precision/Recall 和 Factual Correctness。Ragas 不替代本阶段的人工
Chunk 金标、Recall@K/MRR，也不替代项目自定义 Citation 及 page/block/bbox 确定性校验；阶段 5.1
不新增 Ragas 依赖或实现。
