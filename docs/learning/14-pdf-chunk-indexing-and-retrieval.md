# 学习 14：可追溯 PDF 索引与检索

> 本文总结阶段 4.1 的实际工作、关键设计、工程难点、解决方案与真实验证结果。它说明的是
> “如何把已经解析和切分的 PDF 变成可持久化检索的知识库”，不是阶段 5.1 的正式检索评测报告。

人工审核已确认本文覆盖阶段位置、实际实现、工程难点、真实验证、能力边界、验证入口和下一步，
并且接口与 Alembic 迁移入口已经和当前代码保持一致。

- 阶段：4.1 可追溯 PDF Chunk 索引接入 v1
- 分支：`codex/pdf-chunk-indexing`
- 后端：PostgreSQL 18 + pgvector 0.8.7
- Embedding：`Qwen/Qwen3-Embedding-0.6B`
- 向量维度：1024
- 验收状态：2026-11-13 人工验收通过
- 当前结论：真实 PDF 索引与精确 Top-K 检索链路已经跑通

## 1. 这一阶段在完整 RAG 中的位置

上游阶段已经能对普通 PDF、复杂版式 PDF 和扫描 PDF 进行页面级 Parser/OCR 路由，并输出带
页码、坐标、block、章节和解析来源的 `TraceablePdfChunk`。本阶段负责把这些 Chunk 变成可以
长期保存和查询的知识库记录。

完整数据流如下：

```text
PDF
  -> 页面级 Parser / OCR 路由
  -> 结构感知切分
  -> TraceablePdfChunk
  -> Chunk 索引准入门禁
  -> 确定性 Embedding 输入
  -> Qwen3 Embedding
  -> PostgreSQL + pgvector
  -> 精确 cosine Top-K 检索
  -> 原文、分数、页码、bbox 和完整 provenance
```

完成本阶段后，系统可以根据问题找出相关 PDF 原文，但还不会把原文组织成最终答案。证据组织、
回答生成和页级引用属于阶段 4.2。

## 2. 具体完成了哪些工作

### 2.1 选择 PostgreSQL + pgvector 作为持久化后端

项目评估过独立向量数据库与 PostgreSQL 方案，最终采用已有 PostgreSQL 加 pgvector：

- 向量、文档状态、代际关系和 provenance 可以处于同一个事务边界；
- 关系约束可以直接防止维度、代际和文档身份错配；
- 当前数据量适合先建立精确检索基线，无需提前增加独立服务；
- 后续仍可根据容量、延迟和评测证据演进到 HNSW、混合检索或其他后端。

当前 v1 使用 pgvector 的精确 cosine 检索，没有创建 HNSW/IVFFlat。这样可以先保证结果正确且
可重复，再根据真实容量和延迟决定近似索引参数，避免过早优化。

### 2.2 建立与数据库实现解耦的索引契约

业务编排依赖 `VectorIndex` 协议，而不是直接依赖 SQLAlchemy：

```python
get_manifest(index_id)
ensure_index(manifest)
get_document_state(index_id, document_id)
replace_document(generation, records, expected_revision)
search(query)
```

由此形成两个实现：

- `InMemoryVectorIndex`：用于快速、离线的契约测试和正确性基线；
- `PostgresVectorIndex`：用于 PostgreSQL/pgvector 的真实持久化与检索。

这使门禁、Embedding 和文档替换逻辑不被特定数据库 API 绑死，也让大部分自动化测试不需要
启动 PostgreSQL 或加载真实模型。

### 2.3 定义不可变的索引身份

`VectorIndexManifest` 固定了一个向量空间的完整身份，包括：

- 模型和模型 revision；
- tokenizer 和 tokenizer revision；
- 向量维度、归一化方式、pooling 和计算 dtype；
- query prompt、document input profile；
- 最大 token 数、超限策略和 adapter 版本。

这些字段共同生成确定性的 `index_id`。只要任何会改变向量含义的配置发生变化，就必须创建新
索引，不能在原索引中混入不兼容向量。GPU 型号、批大小等不改变向量语义的运行信息只进入
审计元数据。

索引状态使用：

```text
BUILDING -> READY -> ACTIVE -> RETIRED
                 \-> FAILED
```

只有 `ACTIVE` 索引允许查询。发布新索引时，同一逻辑知识库的旧活动索引会被原子地标记为
`RETIRED`，避免查询到半构建结果。

### 2.4 建立文档代际和幂等替换机制

文件内容、Parser/OCR 决策、切分配置、门禁策略或 provenance registry 发生变化时，会生成新的
`generation_id`。同样的输入和配置再次执行时得到相同代际，从而返回 `already_indexed`，不会
重复写入同一批记录。

替换一个文档时：

1. 读取活动 generation 和稳定文档槽位的 `expected_revision`；
2. 在事务外完成门禁、token 统计和 Embedding，缩短数据库锁定时间；
3. 在短事务中锁定 `(index_id, document_id)` 对应的稳定文档槽位；
4. 候选已发布时返回 `already_indexed`；revision 已变化时返回 `concurrent_update`，拒绝过期任务；
5. revision 匹配时写入完整的新 generation 和 vector records；
6. 原子切换 `active_generation_id` 并递增 revision；
7. 将旧 generation 标记为 `superseded`；
8. 任一步失败则整个事务回滚，查询仍看到旧的完整代际。

这样解决了重复导入、更新过程中读到半份文档，以及“较早任务最后完成并覆盖较新版本”的问题。

### 2.5 增加 fail-closed 的 Chunk 准入门禁

并不是 Parser 输出的所有内容都应该进入向量库。`PdfChunkIndexer` 在 Embedding 前逐块检查：

- 允许 `BODY_TEXT`、`TABLE`、`FORMULA`、`FIGURE_CAPTION` 和 `CODE`；
- 拒绝 `OTHER`、空内容和 ERROR warning；
- 拒绝未知 warning，避免新版上游异常被静默放行；
- 拒绝缺失或无法解析的 extraction provenance；
- OCR 内容必须存在已接受的自动或人工决定；
- 使用真实 tokenizer 计算未截断 token 数，超限时拒绝而不是静默截断。

每个接受或拒绝决定都会生成 JSON/Markdown 门禁报告，并直接列出 Chunk 内容类型、全部来源页码
和全部来源 block ID。审核者不需要先查询向量记录，就可以从报告回到 PDF 原文。拒绝是可观察、
可解释的质量控制，不是悄悄丢数据。

### 2.6 固定 document Embedding 输入

供用户查看和引用的内容始终保存原始 `chunk.content`，而用于生成向量的输入固定为：

```text
section_path

chunk.content
```

输入 profile 为 `section-path-prefix-v1`。章节路径能给短 Chunk 补充语义上下文，同时不会污染
最终展示的原文。系统保存 `embedding_input_hash` 和真实 token 数，用于审计、排错和幂等验证。

Qwen 模型、tokenizer 和 revision 被明确固定；真实预览启用 `local_files_only`，不会在运行期间
偷偷下载或切换模型。

### 2.7 建立四表持久化模型

`rag` schema 中包含四张核心表：

| 表 | 作用 |
| --- | --- |
| `vector_indexes` | 保存向量空间身份、维度和发布状态 |
| `indexed_documents` | 保存稳定文档槽位及其当前活动代际 |
| `document_generations` | 保存每次完整处理配置形成的不可变代际 |
| `vector_records` | 保存 Chunk 原文、向量、页码、输入哈希和 provenance |

数据库层同时约束：

- record 必须属于正确的 index、document 和 generation；
- 向量实际维度必须等于 manifest 声明的维度；
- 同一 generation 内 Chunk 指纹不能重复；
- 页码、token 数、provenance schema version 必须合法；
- 同一逻辑索引最多只有一个 `ACTIVE` 版本。

迁移由 Alembic 显式执行，应用启动时不会自动改数据库结构。

### 2.8 实现精确 Top-K 检索

查询过程为：

1. 使用与索引一致的 Qwen query profile 生成查询向量；
2. 校验查询向量维度、数值有限且不是零向量；
3. 只查询指定 `index_id`；
4. 只连接每个文档的 `active_generation_id`；
5. 使用 pgvector cosine distance 运算符 `<=>` 升序排序；
6. 相同距离使用 `record_id` 稳定排序；
7. 将距离转换为便于展示的相似度分数 `1 - distance`。

检索结果不仅返回文本和分数，还返回：

- 文档、generation、Chunk 和 record 身份；
- 章节路径与起止页码；
- 每个来源区域的 bbox、block、reading order 和字符范围；
- Parser/OCR 方法、版本、配置、warning 和人工复核决定。

因此后续阶段可以把答案引用精确回溯到 PDF 页面和区域，而不是只返回一段无法验证的文字。
人工审核确认当前阶段以结构化页码、block ID 和 bbox 作为追溯出口即可；PDF 可视化高亮需要的
数据已经保留，但绘制交互界面不属于阶段 4.1。

### 2.9 分离数据库权限

本地数据库使用两个不同 PostgreSQL 角色：

- `ai_learning_rag_migrator`：执行 Alembic DDL，负责建表、约束和迁移；
- `ai_learning_rag_app`：运行索引和检索，只具有必要的 DML 权限。

管理员只负责建库、安装 pgvector、创建 schema 和配置角色。真实集成测试同时验证迁移角色与
运行角色，避免出现“测试一直使用高权限账号，生产运行账号却无法写入”的假通过。

### 2.10 建立三层验证

1. **默认离线测试**：身份、门禁、token、幂等、替换、稳定排序、序列化和 SQL 适配逻辑；
2. **显式 PostgreSQL 集成测试**：迁移、应用角色 DML、维度约束、事务替换和真实 pgvector 查询；
3. **真实 PDF 人工预览**：固定 3 份 PDF、3 个问题、期望文档/页码和 Top-K，输出 JSON/Markdown。

少量人工预览只用于验证链路和可读结果，不能被称为 Recall@K 或 MRR。正式检索数据集、相关性
标注与指标属于阶段 5.1。Markdown 报告同时保留紧凑的 Top-K 对比表和每条命中的完整 content，
避免截断预览掩盖证据是否足以回答问题。

## 3. 主要难点以及如何解决

| 难点 | 问题表现 | 解决方案 | 结果 |
| --- | --- | --- | --- |
| Windows 安装 pgvector | PostgreSQL 已安装，但 `vector` 扩展不可用；VS Code 也不提供 C/C++ 编译器 | 安装 Visual Studio Build Tools/MSVC，用 PostgreSQL 18 的开发文件编译并安装 pgvector 0.8.7，再通过 `pg_available_extensions` 验证 | 两个项目数据库均可启用 `vector` 扩展 |
| 管理员、迁移和运行权限混用 | 高权限账号测试通过，不代表生产应用账号可运行 | 将建库/扩展 bootstrap、Alembic migration 和 app DML 分离；集成脚本分别要求 migrator 与 app 密码 | 最小权限下真实索引和检索通过 |
| 向量空间可能被静默污染 | 模型 revision、tokenizer、维度或 prompt 改变后，旧向量与新向量不可比较 | 将所有语义相关配置纳入 manifest 和确定性 `index_id`，禁止原地修改身份字段 | 不兼容配置只能生成新索引 |
| 文档更新可能产生半新半旧状态 | 逐条更新时失败或并发会让查询看到不完整文档 | 使用稳定文档槽位、不可变 generation、行锁和短事务原子切换活动代际 | 失败回滚，旧代际持续可查询 |
| 过期任务可能最后完成 | 两个不同 generation 同时构建时，单纯行锁会形成最后完成者获胜 | 稳定文档槽位增加 revision；发布必须携带 expected revision，冲突返回 `concurrent_update` | 较旧任务不能覆盖已经发布的新代际 |
| OCR 和 Parser 输出并非都可信 | 未解决 OCR 冲突、未知 warning 或 `OTHER` 内容可能污染索引 | 在 Embedding 前执行版本化 fail-closed 门禁，并生成逐块报告 | 真实预览中 2 个 `OTHER` 块被解释性拒绝 |
| 模型可能静默截断长文本 | 只依赖模型 `max_seq_length` 会让超长证据被无提示裁剪 | 调用真实 tokenizer、关闭 truncation、保存 token 数并在超限时拒绝 | 向量输入与保存的证据可审计 |
| Transformers 返回类型兼容 | Qwen 加载成功，但 token 统计报 `unexpected payload` | 复现发现新版返回 `BatchEncoding`；从只接受原生 `dict` 改为接受标准 `Mapping`，并增加仿真与真实 tokenizer 测试 | 修复后真实 token 统计与完整索引成功 |
| pgvector 维度隔离 | 无固定维度 `vector` 列灵活，但容易混入不同维度 | manifest 保存维度，record 冗余维度，并使用复合外键和 `vector_dims()` CHECK 双重约束 | 应用层与数据库层共同阻止错维写入 |
| 少量样例容易被误当成质量指标 | 三个问题命中并不能代表全部查询质量 | 输出明确标为 human-review preview，不计算 Recall@K/MRR | 链路验收与正式评测边界清晰 |

## 4. 真实验证结果

真实预览使用：

- `native.pdf`：原生文本 PDF；
- `layout.pdf`：复杂版式 PDF；
- `scan.pdf`：扫描 PDF，使用已经人工确认的 OCR 决定。

### 4.1 索引结果

| 样本 | 生成 Chunk | 写入向量记录 | 门禁拒绝 |
| --- | ---: | ---: | ---: |
| `native.pdf` | 8 | 8 | 0 |
| `layout.pdf` | 69 | 68 | 1 |
| `scan.pdf` | 12 | 11 | 1 |
| **合计** | **89** | **87** | **2** |

两个拒绝项的原因都是 `unsupported_content_type: other`，属于预期的质量门禁行为。

### 4.2 固定问题检索结果

| 问题 | 期望来源 | Top-1 结果 | Top-1 分数 |
| --- | --- | --- | ---: |
| 强化学习的期待收益如何定义 | `native.pdf` 第 6 页 | 命中第 6 页 | 0.8732 |
| AI Agents 的核心特征 | `layout.pdf` 第 4 页 | 命中第 4 页 | 0.8047 |
| 可扩展性和可维护性分别意味着什么 | `scan.pdf` 第 4 页 | 命中第 4 页 | 0.7970 |

三个固定问题都在 Top-1 命中预期文档和页码。这证明真实链路能够工作，但不代表已经完成大规模
检索质量评估。

其中“可扩展性和可维护性分别意味着什么”是复合问题：Rank 1 提供可维护性，Rank 2 提供可扩展
性。人工审核确认 Top-K 证据集合可以完整覆盖问题，并接受复合问题不必由单个 Rank 1 Chunk 独立
回答。该结果同时形成阶段 4.2 的明确要求：证据组织不能只取第一条命中，必须能组合多个相关
Chunk 覆盖不同子问题。

## 5. 当前已经具备与尚未具备的能力

### 已经具备

- 将三类 PDF 的可追溯 Chunk 持久化到 PostgreSQL/pgvector；
- 使用固定 Qwen 向量空间进行精确 Top-K 语义检索；
- 幂等重入和文档级原子替换；
- 拒绝不可安全索引的 Chunk，并输出审计报告；
- 从检索结果回溯到 PDF 页码、bbox、block、Parser/OCR 和 warning；
- 使用最小权限 app 角色完成真实运行；
- 通过离线测试、真实数据库集成测试和真实 PDF 预览。

### 尚未具备

- 将检索证据组织成最终自然语言答案；
- 自动生成并校验页级引用；
- 固定评测集、人工相关性标注、Recall@K、MRR 等正式指标；
- 关键词/BM25 混合检索、metadata 过滤和 reranker；
- HNSW/IVFFlat 容量与延迟调优；
- 批量导入 API、删除策略、旧代际清理和多租户；
- 生产环境的 TLS、密钥轮换、备份恢复、监控和负载测试。

人工审核接受以上延期范围，不要求为了阶段 4.1 验收提前加入后续能力。三份 PDF 和三个固定问题
用于验证真实端到端链路和人工检查出口，不等同于正式检索质量指标或生产容量结论。

人工审核接受当前旧代际处理方式：被替换的 generation 及 records 保留在原表并标记为
`superseded`，正常查询只读取 `active_generation_id`，所以它们不会污染当前检索结果。这样先保留
审计和未来回滚所需的数据；具体保留期限、回滚入口与自动清理由阶段 7.1 统一实现。

## 6. 重要实现与验证入口

- 阶段技术契约：`docs/design/04-01-pdf-chunk-indexing.md`
- 数据库准备：`docs/POSTGRES_SETUP.md`
- 索引领域契约：`src/ai_learning_rag/document_index.py`
- PDF 门禁与编排：`src/ai_learning_rag/pdf_indexing.py`
- PostgreSQL 适配器：`src/ai_learning_rag/postgres_vector_index.py`
- 数据库模型：`src/ai_learning_rag/database_models.py`
- 初始 Alembic 迁移：`migrations/versions/20261107_0001_pdf_vector_index.py`
- revision 并发控制迁移：`migrations/versions/20261112_0002_add_document_revision.py`
- 真实预览 manifest：`configs/pdf-indexing-preview-v1.json`
- 真实预览脚本：`scripts/pdf_indexing_preview.py`
- 安全运行入口：`scripts/run_pdf_indexing_preview.ps1`
- 本地产物：`output/pdf-indexing-preview/retrieval-preview.md`

## 7. 阶段结论与下一步

阶段 4.1 已经建立了可以继续演进到生产的索引与检索基础：身份稳定、权限分离、写入原子、失败
可回滚、来源可追溯、结果可人工检查。这里的“生产基础”不等于已经完成生产部署，它表示核心
数据模型和事务语义不需要为了上线而推倒重来。

下一阶段 4.2 将在此基础上完成：

```text
用户问题
  -> 当前阶段的 Top-K 检索
  -> 证据选择和上下文组织
  -> 受证据约束的答案生成
  -> PDF 页级引用
  -> 证据不足时拒绝编造
```

在阶段 4.2 完成后，阶段 5.1 再建立正式检索评测基线，用固定数据集和指标指导混合检索、切分
实验与重排，而不是根据少量示例主观调参。
