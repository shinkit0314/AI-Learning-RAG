# AI-Learning-RAG 开发路线图

> 本文档是所有新对话的项目导航入口。开始设计或编码前，必须先完整阅读本文档，再结合
> `docs/PROJECT_REQUIREMENTS.md`、当前阶段学习文档和 Git 状态确认实际进度。

- 路线图版本：`0.4`
- 最近更新日期：2026-11-15
- 当前里程碑：PDF 可检索知识库
- 当前开发阶段：PDF 证据生成与页级引用（人工验收通过，等待发布授权）
- 下一计划阶段：检索评估基线

## 1. 这份路线图解决什么问题

项目已经完成工程基线、最小纯文本 RAG，以及 PDF 解析和切分的多个阶段，但过去主要依靠独立
学习文档记录进度。独立文档能解释“这一阶段做了什么”，却不容易回答以下跨阶段问题：

- 当前处于完整 RAG 生命周期的哪里；
- 为什么下一阶段应该先做这一项；
- 每个阶段依赖什么、产出什么、明确不做什么；
- 每个阶段使用哪个分支，以及何时才允许创建；
- 自动检查通过后还需要怎样的人工验收；
- 哪些后续方向已经排队，哪些仍然只是候选方向。

本文档负责开发顺序和阶段边界；`PROJECT_REQUIREMENTS.md` 负责长期需求；`docs/learning/` 负责记录
每个已实施阶段的原理、实验和复盘。三者不能互相替代。

## 2. 完整 RAG 生命周期

~~~text
数据源
  -> 采集与导入
  -> Parser / OCR / 标准化
  -> 结构感知切分
  -> Chunk 质量门禁
  -> Embedding
  -> 向量与元数据索引
  -> Query 检索
  -> 过滤 / 混合检索 / 重排
  -> 证据组织与引用
  -> 答案生成
  -> 检索和答案评估
  -> 可观测、增量更新与版本维护
  -> API / UI / 部署
~~~

当前已经完成 `Chunk 质量门禁 -> Embedding -> 向量与元数据索引 -> Query 检索 -> 证据组织与引用
-> 答案生成` 的实现和人工验收。阶段 4.2 尚待 commit、push、PR 和合入 `develop`；合入后进入
检索评估基线，不提前创建后续阶段分支。

## 3. 阶段和分支规则

### 3.1 一个阶段对应一个分支

- 功能分支统一使用 `codex/` 前缀；
- 一个分支只解决一个已确认阶段，不混入后续阶段实现；
- 分支必须从前一阶段已经合并后的最新 `develop` 创建；
- 计划表中的未来分支名只是预留，不允许提前创建；
- PR 必须明确以 `develop` 为 base，不能依赖 GitHub 默认的 `main`；
- 前一阶段未完成人工验收和合并时，不开始下一阶段编码。

### 3.2 每个阶段的固定流程

~~~text
定位当前阶段
  -> 检查 develop 和工作区
  -> 调研现有代码与测试
  -> 逐个确认重要设计
  -> 汇总最终设计和验收标准
  -> 用户批准编码
  -> 小步实现和 review
  -> 自动质量门禁
  -> 人工验收
  -> 用户批准 commit / push / PR
  -> PR 合入 develop
  -> 更新路线图状态
~~~

### 3.3 状态定义

| 状态 | 含义 |
| --- | --- |
| 计划 | 已进入路线图，但尚未创建分支或开始设计 |
| 设计中 | 正在调研和确认方案，不能开始编码 |
| 已批准编码 | 主要设计和验收标准已确认，用户允许实现 |
| 开发中 | 正在实现、补测试和文档 |
| 自动检查通过 | Ruff 和 pytest 已通过，等待人工验收 |
| 人工验收通过 | 用户已确认结果，可准备提交或 PR |
| 已合并 | PR 已合入 `develop`，阶段完成 |
| 候选 | 方向合理，但顺序、范围或必要性尚未确认 |

## 4. 总体里程碑与阶段顺序

### 里程碑 1：工程基线

| 阶段 | 分支 | 状态 | 主要完成条件 |
| --- | --- | --- | --- |
| 项目结构、Python/uv、FastAPI、Ruff、pytest、CI | 历史阶段，无需重建 | 已合并 | 环境可复现，基础质量门禁可运行 |

学习记录：[学习 01：工程基线](learning/01-engineering-baseline.md)。

### 里程碑 2：最小纯文本 RAG 闭环

| 阶段 | 分支 | 状态 | 主要完成条件 |
| --- | --- | --- | --- |
| 纯文本标准化与稳定切分 | 历史阶段，无需重建 | 已合并 | 文档和 Chunk 身份稳定、位置可追溯 |
| 本地 Qwen3 Embedding | 历史阶段，无需重建 | 已合并 | query/document 编码分离、维度受控 |
| 内存向量检索 | 历史阶段，无需重建 | 已合并 | 幂等 upsert、精确 Top-K、来源元数据 |
| RetrievalPipeline | 历史阶段，无需重建 | 已合并 | 切分、Embedding、索引和查询串联 |
| 证据约束生成与引用 | 历史阶段，无需重建 | 已合并 | 答案只使用检索证据，引用可校验 |
| 真实 OpenAI 生成实验与复盘 | 历史阶段，无需重建 | 已合并 | 自动测试与人工真实模型实验分离 |

学习记录：[学习 02](learning/02-text-chunking.md) 至
[学习 08](learning/08-minimal-rag-retrospective.md)。

### 里程碑 3：可追溯 PDF 解析与切分

| 阶段 | 分支 | 状态 | 主要完成条件 |
| --- | --- | --- | --- |
| PDF 提取基线 | `codex/rag-pdf-baseline` | 已合并 | 三类 PDF 样本可输出统一 `ParsedDocument` |
| PDF Chunk Contract v1 | `codex/pdf-chunk-contract` | 已合并 | 页码、bbox、block 和字符范围可追溯 |
| PDF Chunk Contract v2 | `codex/pdf-chunk-contract-v2` | 已合并 | 稳定 fingerprint、profile 和来源切片身份 |
| 结构感知 PDF 切分 | `codex/pdf-structure-aware-chunking` | 已合并 | 不同内容类型有明确策略并输出 `TraceablePdfChunk` |
| 页面级 Parser/OCR 自动路由 | `codex/pdf-page-routing` | 已合并 | 快速路径、fallback、人工 OCR 复核和 provenance 可审计 |

学习记录：[学习 09](learning/09-pdf-extraction-baseline.md) 至
[学习 13](learning/13-pdf-page-routing.md)。

### 里程碑 4：PDF 可检索知识库

| 顺序 | 阶段 | 计划分支 | 状态 | 主要完成条件 |
| ---: | --- | --- | --- | --- |
| 4.0 | 建立整体开发路线图 | `codex/development-roadmap` | 已合并 | 所有新对话先读取路线图，阶段、分支和验收出口清晰 |
| 4.1 | 可追溯 PDF Chunk 索引接入 v1 | `codex/pdf-chunk-indexing` | 已合并 | 准入门禁、幂等索引、文档替换、持久化检索和完整 provenance 返回 |
| 4.2 | PDF 证据生成与页级引用 | `codex/pdf-grounded-generation` | 人工验收通过 | 检索结果能形成证据，答案引用可回到 PDF 页和 bbox |

阶段 4.1 同时建立 `native.pdf`、`layout.pdf` 和 `scan.pdf` 的检索 JSON/Markdown 人工预览；不在该
阶段加入混合检索、重排器、答案 UI 或生产部署。

阶段 4.1 的已确认技术契约、生产补强项和验收边界见
[阶段 4.1 设计文档](design/04-01-pdf-chunk-indexing.md)。
实际实现、难点和真实检索结果见
[学习 14：可追溯 PDF 索引与检索](learning/14-pdf-chunk-indexing-and-retrieval.md)。

阶段 4.2 的已确认生成和引用契约见
[阶段 4.2 设计文档](design/04-02-pdf-grounded-generation.md)，实现、DeepSeek 适配、真实模型结果和
已知边界见[学习 15：PDF 证据生成与页级引用](learning/15-pdf-grounded-generation.md)。

### 里程碑 5：检索评估与质量提升

| 顺序 | 阶段 | 计划分支 | 状态 | 主要完成条件 |
| ---: | --- | --- | --- | --- |
| 5.1 | 检索评估基线 | `codex/retrieval-evaluation-baseline` | 计划 | 固定问题、相关 Chunk 标注、Recall@K/MRR 和可重复报告 |
| 5.2 | 切分与 Embedding 实验 | `codex/retrieval-quality-experiments` | 计划 | 用同一评测集比较参数，不针对单个问题硬调 |
| 5.3 | 元数据过滤与混合检索 | `codex/hybrid-retrieval` | 计划 | 语义、关键词和过滤策略有可量化对照 |
| 5.4 | 检索重排 | `codex/retrieval-reranking` | 计划 | 重排收益、延迟和失败边界经过评测 |

必须先完成 5.1，再进入 5.2 至 5.4。没有固定评测集时，不应通过主观观察宣称检索质量提升。

### 里程碑 6：答案质量评估

| 阶段 | 计划分支 | 状态 | 主要完成条件 |
| --- | --- | --- | --- |
| 回答、忠实度与引用评估 | `codex/answer-evaluation` | 计划 | 区分检索失败和生成失败，记录质量、延迟与成本 |

### 里程碑 7：知识库维护与可观测性

| 顺序 | 阶段 | 计划分支 | 状态 | 主要完成条件 |
| ---: | --- | --- | --- | --- |
| 7.1 | 知识库生命周期 | `codex/knowledge-base-lifecycle` | 计划 | 增量导入、去重、替换、失效和版本变化可审计 |
| 7.2 | RAG 可观测性 | `codex/rag-observability` | 计划 | 解析、索引、检索和生成的关键输入输出与性能可诊断 |

PDF 索引 v1 会先实现其自身所需的幂等和替换语义；7.1 再将其扩展为多数据源的统一维护能力。

### 里程碑 8：多源知识接入

| 顺序 | 阶段 | 计划分支 | 状态 | 主要完成条件 |
| ---: | --- | --- | --- | --- |
| 8.1 | 论文网站与技术博客 | `codex/web-content-ingestion` | 计划 | 保留 URL、标题、作者、发布时间和正文位置 |
| 8.2 | PPT/PPTX | `codex/ppt-ingestion` | 计划 | 保留幻灯片、层级、文本和必要视觉位置 |
| 8.3 | 图片和视觉内容 | `codex/image-ingestion` | 计划 | OCR/多模态来源、区域和置信度可追溯 |

每一种数据源都应独立设计、实现和验收，不能在一个“大一统导入”分支中同时开发。

### 里程碑 9：应用和部署

| 顺序 | 阶段 | 计划分支 | 状态 | 主要完成条件 |
| ---: | --- | --- | --- | --- |
| 9.1 | 长期运行的 RAG API 服务 | `codex/rag-service` | 候选 | 模型生命周期、并发、超时和错误边界明确 |
| 9.2 | 学习型 UI | 尚未命名 | 候选 | 只有在检索和回答评估稳定后才确认范围 |
| 9.3 | 部署与权限 | 尚未命名 | 候选 | 根据实际使用环境再选择部署方式 |

应用形态和部署不是当前 PDF 知识库的前置条件，不提前锁定技术栈或创建分支。

## 5. 当前阶段快照

### 已完成

- 工程、测试和文档基线；
- 最小纯文本 RAG 闭环；
- PDF 提取、Chunk 契约和结构感知切分；
- 页面级 PyMuPDF / Docling / RapidOCR 自动路由；
- 隐藏 OCR 冲突的 fail-closed 人工复核；
- `TraceablePdfChunk` 的稳定身份和页级来源区域；
- PDF Chunk 的 PostgreSQL/pgvector 持久化、索引准入门禁、文档代际与 revision 并发控制；
- 保留完整 provenance 的精确 Top-K PDF 检索和三类真实 PDF 人工预览；
- 自然语言 PDF 查询适配、Top-K Evidence 预算组织、DeepSeek 受约束生成，以及可回到 page、block、
  bbox 的 fail-closed Citation。

### 最近完成

- `codex/development-roadmap` 已通过 PR #8 合入 `develop`；
- `codex/pdf-chunk-indexing` 已完成主要实现、离线质量门禁、PostgreSQL 集成验证和真实 PDF
  Top-K 预览；最终自动门禁为 Ruff 格式检查通过、Ruff 静态检查通过、225 项测试通过且 1 项
  显式 PostgreSQL 集成测试按设计跳过，并已于 2026-11-13 通过人工验收、经 PR #9 合入
  `develop`。阶段总结见[学习 14](learning/14-pdf-chunk-indexing-and-retrieval.md)。
- `codex/pdf-grounded-generation` 已完成查询适配、Evidence/Citation 契约、DeepSeek 专用适配器和
  真实 PDF/模型预览；最终自动门禁为 Ruff 格式检查通过、Ruff 静态检查通过、269 项测试通过且
  1 项显式 PostgreSQL 集成测试按设计跳过，显式 PostgreSQL 测试另有 2 项通过。五个真实问题分别
  验证单证据回答、多证据回答、不可回答和含糊问题，并已于 2026-11-15 通过人工验收。

### 当前阶段

- `codex/pdf-grounded-generation` 基于包含 PR #9 的 `develop` 完成实现和人工验收；
- 阶段 4.2 保持原始向量检索排名，不加入混合检索、重排器、答案 UI 或生产部署；
- 当前等待用户审核变更并授权 commit、push 和创建以 `develop` 为 base 的 PR。

### 下一步

1. 审核阶段 4.2 的最终 diff 和阶段复盘；
2. 获得用户明确授权后 commit、push 并创建以 `develop` 为 base 的 PR；
3. PR 合入最新 `develop` 后，将阶段 4.2 标记为已合并；
4. 此后才创建 `codex/retrieval-evaluation-baseline`，开始阶段 5.1 的设计确认。

## 6. 每阶段的统一完成定义

阶段进入人工验收前，至少需要：

- 功能范围与已确认设计一致；
- 每个行为变化都有对应测试；
- 文档、示例和配置同步；
- `uv run ruff format --check .` 通过；
- `uv run ruff check .` 通过；
- `uv run pytest` 通过；
- 真实模型或真实文档验证与离线单元测试明确分离；
- 生成可读的 JSON/Markdown 人工检查材料；
- 说明已知限制、不在本阶段处理的内容和下一阶段依赖；
- 用户完成人工验收。

未经用户确认，不 commit、push、创建 PR 或合并。自动检查通过不等于阶段已经验收完成。

## 7. 路线图维护规则

- 新阶段开始、暂停、恢复、验收或合并时更新“当前阶段快照”和对应状态；
- 变更开发顺序前先分析依赖和影响，并获得用户确认；
- 已完成阶段保留原分支名和历史，不为了统一命名重建旧分支；
- 未来分支只记录名称，不提前创建；
- 若新证据表明阶段过大，应先拆分路线图，再创建分支；
- 若阶段目标可以合并，必须确认不会降低学习价值、可测试性或人工验收清晰度；
- `README.md` 只保留简明进度，详细顺序以本文档为准；
- 长期需求变化同步更新 `PROJECT_REQUIREMENTS.md` 及其变更记录。

## 8. 尚未锁定的长期决策

以下决策只在相应阶段到来时逐个确认：

- 持久化向量索引的最终部署形态；
- metadata 与关系型存储是否需要分离；
- 混合检索、过滤和重排的具体实现；
- 检索与回答评估数据规模和质量阈值；
- 网页、PPT 和图片的 Parser/OCR/多模态模型；
- API、UI、权限和部署技术栈；
- 性能、并发、成本和硬件目标。

路线图确定开发顺序和验收边界，但不提前替未来实验锁定技术答案。
