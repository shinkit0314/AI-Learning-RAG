# 阶段 4.1：可追溯 PDF Chunk 索引接入 v1

> 本文档是阶段 4.1 的实现契约。路线图负责说明开发顺序；本文档负责记录已经确认的
> 技术边界、数据模型、失败语义和验收标准。后续对话在修改本阶段实现前应先阅读本文档。

- 状态：已批准编码
- 分支：`codex/pdf-chunk-indexing`
- 最近确认日期：2026-11-07
- 上游输入：`TraceablePdfChunk`
- 下一阶段：PDF 证据生成与页级引用

## 1. 阶段目标与边界

本阶段把已经验收的 PDF 解析、OCR 路由和结构感知 Chunk 接入持久化向量索引，完成：

```text
TraceablePdfChunk
  -> 索引准入门禁
  -> 确定性 Embedding 输入
  -> 文档级原子替换
  -> PostgreSQL + pgvector 持久化
  -> 精确 Top-K 检索
  -> 完整 PDF provenance 返回
```

本阶段不加入混合检索、重排器、答案生成 UI、HNSW/IVFFlat、长期代际清理或生产部署。
这些能力必须建立在后续固定评测集、容量数据和部署目标之上。

## 2. 已确认的实现方案

### 2.1 后端与依赖边界

- 使用稳定的文档级 `VectorIndex` 协议；业务编排不依赖 SQLAlchemy 或 psycopg 类型。
- `InMemoryVectorIndex` 是离线契约测试和正确性基线。
- `PostgresVectorIndex` 是真实持久化实现。
- 使用现有 PostgreSQL，并在项目数据库中启用 pgvector 的 `vector` 扩展。
- SQLAlchemy 定义模型，Alembic 管理版本化迁移；应用启动时不得自动执行迁移。

### 2.2 数据库与最小权限

- 数据库：`ai_learning_rag`。
- schema：`rag`。
- 管理员只负责建库、角色以及安装或升级扩展。
- `ai_learning_rag_migrator` 拥有 `rag` schema 并执行 Alembic DDL。
- `ai_learning_rag_app` 只具有连接、schema 使用和必要的表/序列 DML 权限，不拥有
  `CREATE`、`ALTER` 或 `DROP` 权限。
- 本地开发允许由同一个登录账号切换角色，但代码和权限脚本必须保留角色边界。
- 密钥只进入本地环境变量或未提交的 `.env`；仓库只提供占位配置。

### 2.3 四表数据模型

1. `rag.vector_indexes`
   - 保存完整 Embedding 空间身份、维度、扩展要求和
     `BUILDING / READY / ACTIVE / FAILED / RETIRED`
     状态。
   - 新索引必须经过构建、验证和显式发布；失败时旧的 `ACTIVE` 索引继续服务。
2. `rag.indexed_documents`
   - 使用 `(index_id, document_id)` 唯一标识稳定文档槽位。
   - 保存 `active_generation_id` 和单调递增的 `revision`，并提供首次导入与并发替换时可锁定的
     稳定行。
3. `rag.document_generations`
   - 保存文件、路由、Parser/OCR 决策、切分和准入策略共同形成的代际身份。
   - 旧代际标记为 superseded，但本阶段不自动删除。
4. `rag.vector_records`
   - 保存完整 Chunk 指纹、原文、Embedding 输入哈希、token 数、向量和 provenance。
   - 关键过滤字段使用关系列，嵌套来源区域使用带版本号的 JSONB。

向量列使用无固定维度的 `vector`。`vector_indexes` 保存预期维度，records 冗余维度并通过
复合外键与 `vector_dims()` CHECK 防止跨空间写入。未来的近似索引必须按 index 和维度建立
局部表达式索引。

### 2.4 索引身份

改变下列任一项都必须产生新的 `index_id`：

- 模型、模型 revision、tokenizer 和 tokenizer revision；
- 输出维度、归一化、pooling、计算 dtype；
- query prompt 与 document input profile；
- 最大输入 token 数、超限策略；
- Embedding adapter 的契约版本。

设备、批大小、进度条和 GPU 型号不改变 `index_id`，但应进入审计元数据。

### 2.5 文档代际与并发替换

文件内容、路由、Parser/OCR 决策、切分配置、准入门禁或 provenance registry 变化时产生新
generation。处理流程为：

1. 在事务外完成解析、门禁、token 检查和 Embedding。
2. 开始处理前读取活动 generation 和 `expected_revision`。
3. 开启短事务，创建或锁定 `(index_id, document_id)` 的稳定文档槽位。
4. 锁内重新检查：候选已发布时返回 `already_indexed`；revision 与预期不同时返回
   `concurrent_update`，禁止过期任务覆盖较新结果。
5. revision 匹配时写入 generation 和 records。
6. 原子切换 `active_generation_id`、递增 revision，旧 generation 标记为 superseded。
7. 提交；任何失败都回滚，新旧状态不能混合可见。

唯一约束或可重试并发错误只允许进行有限重试，不得掩盖永久性数据错误。

### 2.6 Chunk 准入门禁

允许进入索引的类型：`BODY_TEXT`、`TABLE`、`FORMULA`、`FIGURE_CAPTION`、`CODE`。

以下情况拒绝 Embedding，并写入可审计报告：

- `OTHER`、空内容或任何 ERROR warning；
- 未登记的 warning code；
- unresolved 或人工拒绝的 OCR；
- 缺失、未登记或无法解析的 extraction provenance；
- Embedding 输入超过模型 token 上限。

每条门禁决定必须同时记录 Chunk 内容类型、全部来源页码和全部来源 block ID，使审核者无需先查询
向量记录，就能从门禁报告直接回到 PDF 原文位置。幂等重入可以只重建门禁报告，不得重复生成
向量或修改活动代际。

已登记且明确接受的 warning/OCR 决策可以继续索引，但必须原样保存在 provenance 中。门禁规则
自身使用版本号，并参与 generation 身份计算。

### 2.7 Embedding 输入

- 展示和引用证据始终保存原始 `chunk.content`。
- document Embedding 输入固定为 `section_path + "\n\n" + content`。
- input profile 为 `section-path-prefix-v1`。
- 使用模型真实 tokenizer 计算并保存 `embedding_input_token_count`。
- v1 对超限输入 fail-closed，不允许模型库静默截断。
- 保存完整 `embedding_input_hash`，用于审计与幂等检查。

### 2.8 检索

- v1 使用精确 cosine 检索，所有查询必须限定 `index_id`、维度和 active generation。
- 相同分数按 `record_id` 稳定排序。
- SQL 必须直接按 pgvector 距离运算符升序排序，再在结果列中换算展示分数，保证未来索引兼容。
- 返回原文、分数、完整 Chunk 指纹、页码、bbox、block、section path、warning 和 extraction
  provenance。
- 阶段 4.1 的人工审核接受结构化页码、block ID 与 bbox 作为追溯出口；将 bbox 绘制到 PDF 的
  可视化高亮界面属于后续 UI 阶段，不阻塞当前索引验收。

## 3. 接口契约

核心数据类型：

- `VectorIndexManifest`
- `DocumentGeneration`
- `VectorRecord`
- `VectorSearchQuery`
- `SearchResult`
- `DocumentReplaceResult`

`VectorIndex` 只暴露文档级原子语义：

```python
get_manifest(index_id)
ensure_index(manifest)
get_document_state(index_id, document_id)
get_active_generation(index_id, document_id)
replace_document(generation, records, expected_revision)
search(query)
```

`PdfChunkIndexer` 负责门禁、构建 Embedding 输入、调用 Embedder 和映射记录。数据库模型不得泄漏到
该编排层。

## 4. 迁移与扩展约束

- 管理员 bootstrap 与应用 schema 迁移分离。
- 管理员 bootstrap 负责创建 `rag` schema；迁移账号只在已有 schema 内管理对象，不需要
  数据库级 `CREATE` 权限。
- 迁移脚本应显式检查或记录所需 pgvector 版本。
- PostgreSQL/pgvector 特有约束允许在 Alembic 中使用经过评审的手写 SQL。
- JSONB provenance 必须包含 `schema_version`，读取时验证支持的版本。
- 迁移不由应用启动自动触发。

## 5. 测试与验收

### 默认离线测试

- 身份指纹、门禁、输入构建、token 超限和 provenance 序列化单元测试。
- `InMemoryVectorIndex` 与 `PostgresVectorIndex` 共享契约测试。
- 文档首次导入、幂等重入、替换失败回滚和稳定排序测试。
- 默认测试不得依赖数据库、真实模型或下载。

### 显式 PostgreSQL 集成测试

- 只允许连接名称明确包含 `_test` 的数据库，并在执行前再次校验。
- 验证迁移、维度约束、并发替换、事务回滚和精确检索。
- 测试不能自动删除数据库。

### 人工预览

- 使用版本化 manifest 固定 `native.pdf`、`layout.pdf` 和 `scan.pdf` 的问题、期望文档/页码与
  `top_k`。
- 输出 JSON 和 Markdown；真实模型只通过显式命令运行，不自动下载。
- Markdown 先提供紧凑 Top-K 对比表，再逐条展示完整检索 content；审核者不应为了检查证据完整性
  被迫打开大型 JSON。
- 人工审核按 Top-K 证据集合判断问题覆盖，不强制复合问题由单个 Rank 1 Chunk 完整回答；阶段
  4.2 必须能选择并组合多个命中，覆盖问题中的不同子问题。
- 本阶段不把少量预览包装成 Recall@K/MRR，正式评测属于阶段 5.1。

## 6. 生产边界

本阶段提供可以自然演进到生产的存储和事务基础，但不宣称已经完成生产部署。连接池、TLS、密钥
轮换、备份/PITR、恢复演练、监控、容量和负载测试、代际清理、多租户隔离及零停机迁移属于后续
部署与生命周期阶段。

人工审核已接受以上生产边界：答案生成与页级引用、正式检索评测、混合检索与重排、近似向量
索引调优、知识库生命周期以及生产运维能力均按路线图延期，不阻塞阶段 4.1 验收。当前三份 PDF
和三个固定问题只证明真实链路可运行，不作为整体检索准确率或生产容量声明。

人工审核已接受阶段 4.1 的旧代际保留策略：文档发布新 generation 后，旧 generation 及其 records
继续保存在原表中并标记为 `superseded`，正常检索只连接 `active_generation_id`，因此旧数据不会
混入当前结果。本阶段不提供自动清理和面向用户的回滚入口；保留期限、回滚流程和安全清理策略在
阶段 7.1“知识库生命周期”中统一设计。

本地 Windows PostgreSQL 用于开发验证；真实生产环境应单独选择受管理 PostgreSQL 或标准化部署，
固定扩展版本并建立备份与升级流程。
