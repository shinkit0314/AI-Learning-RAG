# 阶段 5.3 · 元数据过滤与混合检索

5.3 属于里程碑 5「检索评估与质量提升」。5.1 的版本化评测已验收，5.2 已人工验收并合入 `develop`。开始编码前核对本地 `develop`、远端跟踪引用及 GitHub 的 `develop` 均为 `83e6a45`。本地 `codex/hybrid-retrieval` 从该提交建立。5.4 在本阶段验收合并后评估检索重排。[PR #24](https://github.com/shinkit0314/AI-Learning-RAG/pull/24)

## 已确认的对照

- 使用 5.1 的 v1.0.0、三份原 PDF 的 87 个片段、600 字符切分上限、Qwen 默认查询指令和精确余弦向量检索。片段集合指纹为 `4f057af5e74ce1be7b4f2cb3b04b26a7f747eb712ddaf30b5af56066b57b2469`。三份本地 PDF 的 SHA-256 已对照样本清单。
- 15 道正式题的 Recall@1/3/5 为 0.8022、0.9511、0.9733，MRR@5 为 1.0000。三道诊断题单列。5.2 的复跑结果与 5.1 基线报告字节一致。
- [PR #9](https://github.com/shinkit0314/AI-Learning-RAG/pull/9) 引入 PostgreSQL/pgvector 有效代际检索及来源追溯。[PR #10](https://github.com/shinkit0314/AI-Learning-RAG/pull/10) 将 PDF 检索接入证据和引用。[PR #11](https://github.com/shinkit0314/AI-Learning-RAG/pull/11) 建立评测集和逐题报告。

## 现有索引与题目

[`PdfRetriever.search`](../../src/ai_learning_rag/pdf_retrieval.py) 接收自然语言和 Top-K，经 Qwen 编码后传给向量索引。当前 SQL 以 `index_id`、向量维度和文档有效代际限定候选，按精确余弦距离及记录 ID 排序。查询入口尚无文档范围或关键词模式。

| 已存字段 | 现有数据与用途 |
| --- | --- |
| `document_id` | 可指定文档。样本 ID 和文件路径在预览工具中按文档 ID 映射。 |
| `page_start`、`page_end` | 记录包含首尾的页码范围。基线中有跨页证据，开放页码筛选前需约定相交或完全落在范围内。 |
| `content_type`、`section_path` | 语料清单有 84 个正文片段和 3 个图注片段。章节路径是 JSON 数组，筛选语义需用实际章节验证。 |
| `provenance` | 保存原文块、页面区域和提取信息，可复核来源。 |

字段见[记录模型](../../src/ai_learning_rag/database_models.py)和[查询实现](../../src/ai_learning_rag/postgres_vector_index.py)。下表用本地 v1.0.0 报告和 87 个片段的目录对照，已标注证据见[正式评测配置](../../configs/pdf-retrieval-evaluation-v1.json)。

| 场景 | 已观察到的证据 | 计划检验 |
| --- | --- | --- |
| 图 2 `layout-figure-two-dimensions` | 正确图注在 `layout.pdf` 第 3 页并排第 1。全库第 4、5 条分别来自 `scan.pdf`、`native.pdf`。正文还在论文第 2 页提及 `Fig. 2`。 | 给定文档范围后，数越界条数并核对图注仍可回查。观察关键词能否区分正文引用和图注。 |
| 图 4、POMDP、10000 | 完整原题已把对应标注排第 1。目录里分别只有一条片段包含对应的字面形式。 | 检验英文图号、日文括号内缩写和中文相邻数字怎样匹配，查看混合排序有无退步。 |
| `layout-taxonomy-system-design` | 原前五条都来自 `layout.pdf`，五段标注找回三段。漏掉第 2 页的设计不足和过度设计后果，以及第 3 页的按复杂度选型框架。 | 比较关键词与混合排名能否把两段原文提到前五。按文档过滤本身不会改变这五条。 |
| 三道诊断题 | 学习率与 ImageNet 分数缺少已知答案。含糊题没有唯一所指。 | 保留各模式的原始结果供人工复核。由题外上下文提供的显式范围另行记录。 |

## 已确认 · 首个过滤场景

对应 [Issue #25](https://github.com/shinkit0314/AI-Learning-RAG/issues/25) 的 WHAT 第 2 项。

> 确认首个过滤验证场景及范围输入。用 Fig. 2 题说明谁提供检索范围、什么算越界、哪些页码和内容类型线索留作后续验证。

2026-11-21 用户确认方案 A。首轮由调用方显式指定一份 PDF，并把文档 ID 和原问题一同传入。以 `layout.pdf` 的 Fig. 2 原题比较全库与指定论文的前五条。其他文档的结果计为越界，预期从 2 条降到 0 条，第 3 页正确图注仍可回查。用可控数据补一例，确保范围外高分候选占满全库 Top-K 时仍能找回范围内片段。Azure AI Search 记录了先取 Top-K 再过滤时遗漏范围内候选的条件。[Azure AI Search 向量过滤](https://learn.microsoft.com/en-us/azure/search/vector-search-filters)

方案 B 是首轮同时支持文档、页码和内容类型，已留待后续。它可以直接定位某页的图注，也多出跨页包含规则、正文或图注语义和组合输入。出现调用方明确提出按页或内容类型限定范围的例子时再讨论扩展。评测时记录显式范围来自调用方的上下文。

## 已确认 · 首轮关键词匹配

对应 [Issue #25](https://github.com/shinkit0314/AI-Learning-RAG/issues/25) 的 WHAT 第 3 项。

> 逐项确认关键词匹配方式、混合排序与实验口径。设计文档记录选项、取舍、已确认和仍待决定的内容，并写清同一语料和标注怎样用于可复现对照。

在固定片段目录里，`Fig. 2` 字面命中正文和图注各一次，`Fig. 4`、`POMDP` 和 `10000` 各命中一个片段。四道完整原题均未原样出现在对应片段里。以 Fig. 2 为例，图注包含图号，却缺少问题里的 `represented` 和 `branches`。要求原问题的所有词同时出现，容易漏掉该图注。PostgreSQL 的 `plainto_tsquery` 默认把词项用 AND 连接，短语查询可单独验证词项顺序。[PostgreSQL 18 全文检索查询](https://www.postgresql.org/docs/18/textsearch-controls.html)

2026-11-23 用户确认首轮用方案 1。复用 PostgreSQL 原生全文检索，输入完整原题，允许片段凭部分词项匹配。用图号短语、缩写和数字单独诊断命中及分词，人工指定词的结果记在诊断里。方案复用现有数据库，常见词可能引入噪声，中文和日文的实际词边界及数字相邻情形仍需检查。[pgvector 混合检索](https://github.com/pgvector/pgvector#hybrid-search) · [PostgreSQL 18 分词诊断](https://www.postgresql.org/docs/18/textsearch-debugging.html)

方案 2 的中日文双字切分留待后续。它能给无空格文本提供更多候选词项，也增加配置与验证成本。若第一轮在多道中日文题上确认同类分词漏召回，再用独立用例比较两种分析方式。[Elasticsearch CJK 双字切分](https://www.elastic.co/docs/reference/text-analysis/analysis-cjk-bigram-tokenfilter)

## 已确认 · 首轮混合排序

对应 [Issue #25](https://github.com/shinkit0314/AI-Learning-RAG/issues/25) 的 WHAT 第 3 项，验收落在 [Issue #28](https://github.com/shinkit0314/AI-Learning-RAG/issues/28) 的 WHAT 第 1 项。

向量路的余弦相似度和关键词路的词项相关分数来自不同计算。合并时需要决定依据名次还是依据归一化后的分数。

2026-11-23 用户确认采用等权倒数排名融合（RRF）。每个片段按所在列表的名次获得一份贡献，片段在两路都出现时把两份贡献相加。一路未返回该片段时，该路贡献为零。同一片段只保留一次，记录两路名次供复核。RRF 省去跨路分数归一化，适合首轮固定语料对照。它只看名次，高置信度第一名和险胜的第一名贡献一样。Azure AI Search 和 Elasticsearch 均提供 RRF。[Azure AI Search 混合排名](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking) · [Elasticsearch RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

归一化后加权相加留待后续。它能利用同一路内的分差，也需定义归一化、单条或同分候选如何处理，以及权重如何确定。若在独立题目上反复发现 RRF 把一路高置信度的正确证据压出前五，再比较这条路径。[Elasticsearch 线性融合](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/linear-retriever)

## 已确认 · 首轮候选深度

对应 [Issue #28](https://github.com/shinkit0314/AI-Learning-RAG/issues/28) 的 WHAT 第 1 项，也属于 [Issue #25](https://github.com/shinkit0314/AI-Learning-RAG/issues/25) 的 WHAT 第 3 项。

> 使用获批的候选深度和排序方法合并两路结果。测试相同片段只出现一次、一路为空、同分排序和两路都无结果，报告能复核各路排名和合并结果。

基线只保存了每题的前五条。系统设计题的五处标注证据有两处没进向量前五，现有报告没有它们在第六名以后的排名。只有进入至少一路候选的片段才可能出现在融合后的前五条。Azure AI Search 和 Elasticsearch 都将每路候选数与最终输出数分开设置。[Azure AI Search 候选和输出数量](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking#number-of-ranked-results-in-a-hybrid-query-response) · [Elasticsearch RRF](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)

2026-11-23 用户确认方案 1。整套题目统一使用两路各最多 10 条、合并前最多 20 条，融合后最多输出 5 条。若相关片段已经进入候选，报告应标明它的融合名次。若未进入，用更深的只读排名核对实际位置。方案 2 的每路 20 条留待后续，若诊断显示多道题的相关证据落在第 11–20 名，再比较两种深度的实际差异。

## 已确认 · RRF 排名常数

对应 [Issue #28](https://github.com/shinkit0314/AI-Learning-RAG/issues/28) 的 WHAT 第 1 项。

> 使用获批的候选深度和排序方法合并两路结果。测试相同片段只出现一次、一路为空、同分排序和两路都无结果，报告能复核各路排名和合并结果。

RRF 给每次命中加上 `1 / (排名常数 + 名次)`。名次从 1 开始，排名常数全题统一。常数较大时，同一路第 1 名与第 10 名的贡献更接近。常数较小时，前几名的优势更明显。用假设片段算一遍，分数四舍五入到小数点后四位。[Azure AI Search RRF 计分](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)

| 排名常数 | 只在一路排第 1 | 两路均排第 10 | 靠前的片段 |
| --- | ---: | ---: | --- |
| 60 | 0.0164 | 0.0286 | 两路第 10 |
| 5 | 0.1667 | 0.1333 | 一路第 1 |

2026-11-23 用户确认方案 1。首轮全题固定使用 Elasticsearch 默认的排名常数 60。在两路各最多 10 条的等权融合中，所有双路命中都会排在单路命中前面。报告应检查这一规则是否影响原本靠前的相关证据。方案 2 的常数 5 留待后续，若多道题因相关片段只在一路出现而跌出前五，再用相同候选比较。[Elasticsearch RRF 参数](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/retrievers/rrf-retriever)

融合分数相同时，设计沿用现有向量检索的稳定规则，按记录 ID 升序排列。这个工程默认值只用于打破同分，报告中的显示精度不决定是否同分。[PostgreSQL 向量检索排序](../../src/ai_learning_rag/postgres_vector_index.py)

## 已确认 · 混合检索质量判定

对应 [Issue #30](https://github.com/shinkit0314/AI-Learning-RAG/issues/30) 的 WHAT 第 1 项，逐题证据由 [Issue #29](https://github.com/shinkit0314/AI-Learning-RAG/issues/29) 的报告提供。

> 按已确认标准复核量化报告，记录过滤约束是否通过、混合检索的收益与退步、样本局限，以及用户确认的检索策略结论。

v1.0.0 的 15 道正式题首条都相关，14 道在前五条找齐已标注证据。仅系统设计题找回两处原先漏掉的证据，就能让平均 Recall@5 增加约 2.67 个百分点，可能掩盖其他题的证据损失。评价用同一组标注，逐题核对首个相关结果的名次及前五条证据的增减。显式过滤用例单独检查越界和范围内证据覆盖。[5.2 逐题判定](05-02-retrieval-quality-experiments.md) · [Elasticsearch 排名评测](https://www.elastic.co/docs/api/doc/elasticsearch/operation/operation-rank-eval)

2026-11-24 用户确认方案 1，沿用 5.2 的逐题质量门槛。15 道正式题的首条都继续相关，且每题前五条保留基线已找到的直接证据。在满足这两条后，至少一道题在前五条找回新增的直接证据，或把原第 4、5 名的相关证据提到前三，才判为质量改善。用文档 ID 和片段指纹核对，遇到重叠片段再回查原文。质量结论和最终采纳决定由用户复核，耗时另记。出现退步时仍完成对照并保留结果。

方案 2 允许总体收益抵消个别退步。现有 15 道正式题不足以稳定确定可以接受哪些证据或排名损失。若以后能用更大且独立的题集确认收益和退步分布，再讨论这条路径。

## 已确认 · 首轮评测分组

对应 [Issue #29](https://github.com/shinkit0314/AI-Learning-RAG/issues/29) 的 WHAT 第 2 项。

> 用原 15 道正式题对比精确向量、纯关键词和混合检索的 Recall@1/3/5、MRR@5 及逐题找回和遗漏。另列 3 道诊断题的行为，按已批准的显式范围用例报告越界数和范围内证据覆盖。

v1.0.0 的题目和相关片段绑定同一份 87 片段快照。全库比较使用相同原题、标注及最终输出数量，才能把变化归到检索策略。过滤用例的已选文档是额外的调用方输入，适合用同题、同检索模式开关范围做成对对照。Vespa 的评测示例用同一套题比较语义、关键词和混合检索，并给出逐题召回和耗时。[Vespa 检索召回评测](https://docs.vespa.ai/en/learn/tutorials/rag-blueprint#evaluating-recall-of-the-retrieval-phase)

2026-11-24 用户确认方案 1。全库用同一批 18 道原题运行精确向量、纯关键词、混合检索，15 道正式题计入指标，3 道诊断题单列。对 Fig. 2 额外运行指定 `layout.pdf` 的精确向量和混合，分别与各自的全库结果比较。报告记录越界条数、正确图注是否进入前五，以及混合两路候选是否均遵守范围。

方案 2 再为 `native.pdf` 和 `scan.pdf` 各设计一个显式过滤用例，留待有明确调用需求或首轮发现跨文档边界问题时讨论。增加用例可以覆盖更多文档类型，同时需要人工核对新增场景与证据。

首轮每遍包含 54 次全库检索和 2 次显式过滤检索。人工提供的专用词项和扩大候选深度的排查结果只计入诊断。计时复用 5.2 的做法，在同一环境中按固定题序先热身一遍、再计时三遍。每种策略从接收原问题计时，到返回结果为止，混合检索包含融合。报告中位数及范围，耗时差异小于重复测量波动时，记录为尚无法判断。[5.2 计时方法](05-02-retrieval-quality-experiments.md)

## 开发顺序与验收

| 任务 | 可独立验收的交付 | 前置 |
| --- | --- | --- |
| [#25](https://github.com/shinkit0314/AI-Learning-RAG/issues/25) | 场景、设计选择、实验口径和获批的验收契约 | 5.2 已合并 |
| [#26](https://github.com/shinkit0314/AI-Learning-RAG/issues/26) | 显式文档过滤与真实 PDF 对照 | #25 |
| [#27](https://github.com/shinkit0314/AI-Learning-RAG/issues/27) | 可单独运行的关键词检索和精确词验证 | #25 |
| [#28](https://github.com/shinkit0314/AI-Learning-RAG/issues/28) | 可追溯的混合排序，两路遵守文档范围 | #26、#27 |
| [#29](https://github.com/shinkit0314/AI-Learning-RAG/issues/29) | 固定基线的 JSON、Markdown 量化对照与耗时 | #28 |
| [#30](https://github.com/shinkit0314/AI-Learning-RAG/issues/30) | 人工验收、合并和 5.4 交接 | #29 |

### 实现边界与自动测试

- **#26 · 文档范围**。范围是调用方给出的一个可选文档 ID。先限定到同一索引的有效文档代际，再做排序和 Top-K。测试省略范围时的原有结果，以及范围外高分足以占满全库 Top-K、范围内记录少于 K、有效文档范围为空、旧代际存在的情形。显式范围无候选时返回空列表。
- **#27 · 关键词**。复用现有 PostgreSQL 和片段正文。实现起点使用内建 `pg_catalog.simple` 生成词项，以 OR 匹配部分词项，用 PostgreSQL 原生的 `ts_rank_cd` 排序并按记录 ID 处理同分。用固定、相同的处理方式运行所有原题。测试空词项、零命中、图号相近、大小写、标点、多语言及旧代际隔离。真实语料用 `ts_debug` 核查 `Fig. 2`、`Fig. 4`、`POMDP`、`10000` 的实际匹配行为并记录限制。当前 87 个片段先用现有表查询，查询规模或计时证明需要时再讨论全文索引。[PostgreSQL 全文检索查询和排序](https://www.postgresql.org/docs/18/textsearch-controls.html) · [PostgreSQL 分词诊断](https://www.postgresql.org/docs/18/textsearch-debugging.html)
- **#28 · 融合**。两路读取同一索引的同一份有效代际快照，并在截断候选前应用相同的显式范围。测试相同片段合并后只出现一次、两路贡献和名次可复查、任一路为空、两路均为空、同分排序及结果可定位到原文。默认检索入口继续使用精确向量，纯关键词与混合通过显式选用的检索入口或实验配置运行。[PostgreSQL 事务快照](https://www.postgresql.org/docs/18/transaction-iso.html)
- **#29 · 评测**。复用 5.1 的逐题评分和快照核对。测试空排名、诊断题排除、证据增减和范围断言；JSON 保留各路候选、融合名次、片段身份和原文位置，Markdown 展示逐题结论、耗时与实际限制。快照或精确向量对照存在无法解释的变化时，先诊断差异再给候选下结论。

### 人工验收

1. 核对三份 PDF、87 个片段、集合指纹、600 字符上限和 Qwen 默认查询指令。精确向量对照与已验收的 v1.0.0 对照逐题核对前 1、3、5 名的相关片段身份、首个相关结果名次及汇总指标。
2. 对照 Fig. 2 的四组前五条。指定 `layout.pdf` 的两种检索结果及混合的两路候选都须零越界。指定范围的精确向量结果仍须找回第 3 页图注，混合的图注名次或未命中原因计入报告。抽查片段能回到 PDF 页码、原文块及页面区域。
3. 用 JSON 和 Markdown 复核系统设计题两处遗漏、精确术语题及本轮发生退步的题。根据已确认的逐题门槛给出结论，讲清是在候选阶段漏掉证据，还是在融合后跌出前五。过滤约束与混合的质量结论各有验收依据，发现质量退步仍可交付完整、可复核的实验结果。
4. 保存复现命令、配置、报告和自动门禁结果。Ruff 格式与静态检查、默认 pytest、显式 PostgreSQL 集成测试和真实 Qwen/PostgreSQL 对照完成后，由用户复核结果并确认阶段验收。

### 编码授权

对应 [Issue #25](https://github.com/shinkit0314/AI-Learning-RAG/issues/25) 的 WHAT 第 4 项。用户确认开始编码，按已汇总的 5.3 设计从 #26 开始。在独立工作树中检出 `codex/hybrid-retrieval`，基点为当时核实的 `develop` 提交 `83e6a45`，原工作树及其既有改动保留。

阶段启动时，Ruff 格式和静态检查通过，默认 pytest 303 项通过，PostgreSQL 集成模块按默认配置跳过。该次检查记录现有工作区的状态，5.3 实现完成后仍需进行真实检索和人工验收。

## 已验证的实现结果

- **#26**。同一份 87 片段索引上的 Fig. 2 向量前五条，范围外结果从 2 条降为 0 条。指定 `layout.pdf` 后，第 3 页图注仍排第 1，报告保存了片段指纹、原文块及页面区域。受控测试覆盖范围外最高分、当前代际、少于 K 和空范围。
- **#27**。真实原题的关键词前五条中，Fig. 4 和 POMDP 都在第 1 名找回标注证据，Fig. 2 和 10000 未找回。单独查 `Fig. 2` 时，对应图注排第 2，说明词项命中与完整原题排序需要分别验证。数据库把扫描页的数字解析成 `10000个磁盘的存储集群中` 这个连续词项，说明内建分析器难以独立匹配某些混在中文 OCR 文本中的编号。受控 PostgreSQL 测试验证了图号和标点、部分匹配、范围隔离、空词项及旧代际隔离。
- **#28**。RRF 测试验证了重复片段合并、空候选、同分、返回上限和范围隔离。真实 PostgreSQL 测试在两路查询之间发布新文档代际，两路仍看到同一份旧快照，后续独立查询读到新代际。真实 Fig. 2 混合结果限定前后均为零越界，图注都排第 1，指定范围的两路候选均无越界。
- **#29**。在 PostgreSQL 18.6、pgvector 0.8.7 和锁定的 Qwen GPU 环境中验证三份 PDF、87 个片段及集合指纹。复跑精确向量逐题核对通过。每遍 56 次检索，热身一遍加三遍计时，共 224 次。试验输入、报告与命令记录在原仓库的 `output/stage53` 下，正式结果位于 `experiment/comparison.md` 和 `experiment/comparison.json`。

| 检索策略 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | 检索耗时中位数（毫秒） |
| --- | ---: | ---: | ---: | ---: | ---: |
| 精确向量 | 0.8022 | 0.9511 | 0.9733 | 1.0000 | 47.30 |
| 纯关键词 | 0.2667 | 0.2667 | 0.3333 | 0.2833 | 3.88 |
| 混合 | 0.7556 | 0.9044 | 0.9400 | 0.9333 | 52.35 |

### 人工验收结论

- **显式过滤通过**。Fig. 2 的指定文档结果满足零越界和图注保留，原文可回查。
- **本轮混合配置未通过质量门槛**。`layout-taxonomy-system-design` 和 `layout-search-platforms` 的首个相关结果均从第 1 降到第 2。平台题一段相关片段只由向量路找回，向量名次为 3，融合名次为 7，前五从两段证据减为一段。
- **系统设计题原有遗漏出在候选阶段**。第 3 页证据的扩展诊断名次是向量 14、关键词 21。第 2 页证据在向量排 44，关键词没有命中。两处均未进入首轮合并列表。增加候选数能改变哪些片段可参加排序，能否进入最终前五仍需实证。
- **继续用精确向量作默认与 5.4 的质量对照**。文档范围可由调用方显式使用。首轮混合配置保留为可复现的实验对照。结果来自三份 PDF 和 15 道正式题，耗时来自本机顺序执行。

### 验收状态

Ruff 格式与静态检查通过。使用三份本地 PDF 的默认 pytest 有 312 项通过，显式门控的 PostgreSQL 集成模块另有 2 项通过。真实 Qwen/PostgreSQL 对照和基线核对完成。

2026-11-24 用户确认接收以上结论，5.3 人工验收通过。用户另行授权提交、推送和创建以 `develop` 为目标的 PR。合并后补齐 PR 及合并提交记录。[Issue #30](https://github.com/shinkit0314/AI-Learning-RAG/issues/30)
