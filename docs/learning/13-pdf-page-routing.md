# 学习 13：PDF Parser 自动路由 v1

## 本阶段位于 RAG 的哪里

~~~text
PDF 输入
  -> 页面诊断与 Parser/OCR 路由（本阶段）
  -> ParsedDocument / ParsedBlock
  -> 结构感知切分器
  -> TraceablePdfChunk
  -> Embedding / 向量索引（不在本阶段）
~~~

本阶段只解决切分之前的页面解析路线。切分算法、Embedding、向量数据库和检索流程没有改变。

## 为什么需要页面级路由

同一份 PDF 可能同时包含普通数字文字、双栏论文、扫描图片、隐藏 OCR 和损坏文字层。固定使用一个
Parser 会产生两类问题：简单页付出不必要成本，复杂页则可能得到错误阅读顺序或错误 OCR。

第一版坚持页面级路由，不在 block 级拼接多个 Parser 的正文结果。每页最终只有一个被接受的解析
来源；其他尝试只保留在诊断中。

## 顶层接口

~~~python
result = parse_pdf_auto(path, PdfRoutingConfig())
document = result.document
~~~

`PdfAutoRoutingResult` 包含统一文档、原始页面信号、逐页路由诊断、稳定 `routing_profile` 和不参与
业务身份的缓存事件。统一文档可以直接交给 `chunk_pdf_document()`。

## 路由流程

~~~text
PyMuPDF 全文轻量预检
  -> 页面来源分类
  -> 文字结构质量门禁
  -> 隐藏 OCR 验证需求
  -> 布局复杂度判断
  -> RoutePlan
  -> 相同路线连续页段
  -> Docling / RapidOCR
  -> 质量门禁与非对称 fallback
  -> 原始页码严格合并
~~~

优先级是失败安全与文字可信度、布局正确性、处理速度。只有明确简单且文字层可用的页面走
PyMuPDF 快速路径。

## 页面信号与来源分类

信号提取与路由决策分离。`PageRoutingSignals` 记录字符数量、异常 Unicode、私用区字符、bbox
异常、重复 glyph、拉丁文字碎片、混合文字体系、图片并集覆盖率、最大图片覆盖率、绘图数量和
多栏几何证据。

混合文字体系比例默认只作为诊断证据，不单独降低文字层质量。技术文档和 PPT 中的 `RAG系统`、
`AI应用` 等中英混排是正常内容，不能据此认定文字层损坏。若积累了足够标注样本，可以通过
`mixed_script_affects_text_quality` 显式启用该门禁；启用前应先做离线评测。

图片覆盖率使用区域并集，不会把重叠图片重复计数。来源分类包括普通数字文字、纯图片扫描、隐藏
OCR、损坏文字层、混合/不确定页和空白页。

内部文字质量只输出 `usable`、`suspicious` 或 `unusable`。它能发现结构损坏，不能证明合法字符
在语义上正确。例如 `属于` 和 `属千` 都可能通过 Unicode 检查。

## 隐藏 OCR 验证

“整页图片 + 文字层”页面进入 full-page OCR 验证，而不是直接信任或丢弃隐藏文字层：

- 两份规范化文字一致：保留 embedded OCR；
- embedded 文字层有硬失败而重跑结果可用：整页采用重跑 OCR；
- 双方都合理但有实质差异：页面失败并产生
  `OCR_RESULT_CONFLICT_REVIEW_REQUIRED`；
- 不做字符、句子或 block 级静默混合。

这套策略能发现少量差异并阻止其进入后续 RAG，但不会假装自动知道两个合法字符中哪个正确。

## 人工 OCR Review Contract

冲突不会通过修改代码或路由阈值解决。工具生成三个文件：

~~~text
scan.ocr-review.md         # 人工阅读，不修改
scan.ocr-review.json       # 完整证据，不修改
scan.ocr-decisions.json    # 只修改 decision 和 review_note
~~~

每页只能选择 `accept_embedded_ocr`、`accept_rerun_ocr` 或 `reject_page`。决定文件绑定原始 PDF
SHA-256、`routing_profile`、页码、`conflict_id` 和两份候选文字哈希。文件、配置或候选结果变化后，
旧决定会被拒绝，不能静默套用到新内容。

应用人工决定后，页面 provenance 增加稳定 `review_decision_id`。接受隐藏层会产生
`MANUAL_REVIEW_ACCEPTED_EMBEDDED_OCR`，接受重跑结果会产生
`MANUAL_REVIEW_ACCEPTED_RERUN_OCR`，拒绝页面会产生 `MANUAL_REVIEW_REJECTED_PAGE`。审核备注用于
审计但不参与内容身份；选择的候选和 review decision 会通过 provenance 进入最终 Chunk ID。

生成复核文件：

~~~powershell
uv run python scripts/pdf_ocr_review.py generate data/PDF/scan.pdf
~~~

编辑 decisions JSON 后应用：

~~~powershell
uv run python scripts/pdf_ocr_review.py apply data/PDF/scan.pdf
~~~

已有页面解析缓存命中时不会重新调用 Docling/OCR。没有填写决定的冲突页继续保持 failed，不能进入
Chunk。

## 复杂布局与 Docling

强复杂信号包括稳定多栏和明显阅读顺序风险。弱信号包括大量文字 block、绘图以及多个图片与正文
交错。强信号直接升级；两个弱信号升级为复杂；一个弱信号标记不确定。复杂和不确定页面都使用
Docling，只有明确简单页面使用 PyMuPDF。

运行时不会为了选择 Parser 而默认同时运行 PyMuPDF 和 Docling。两条路线的并列结果只用于离线
评测。

## 非对称 fallback

- 已知复杂页的 Docling 失败不能静默降级为 PyMuPDF 成功；
- OCR 验证失败不能回到未经验证的隐藏文字层；
- 最佳努力文字可以出现在诊断中，但失败页不能进入 Chunk；
- Parser、OCR 或模型不可用时显式失败；
- 代码不会主动清理缓存或下载模型。

## 多页面合并

PyMuPDF 预检完整文档一次。相邻且路线配置相同的页面组成连续 `page_range`，复用同一个 Docling
converter。最终结果按原始页码合并，并拒绝缺页、重复页或错误页码。

文档 ID 和文件哈希始终来自原始完整 PDF，不会为页面范围生成新的来源身份。

## Extraction provenance 与稳定身份

`ExtractionProvenance` 记录 Parser、版本、提取方式、OCR 引擎和配置 profile。文档保存去重记录，
页面、block 和 `ChunkSourceRegion` 通过 ID 引用。

- `file_sha256/document_id/source_fingerprint` 表示输入来源；
- `routing_profile` 表示页面路由配置；
- `ExtractionProvenance` 表示最终 Parser/OCR；
- `chunking_profile` 表示切分配置。

最终 Chunk ID 同时包含处理 profile 和来源 fingerprint，未采用的失败尝试不参与身份。

## 缓存和重新校准

缓存分为原始信号、路线决策和通过门禁的规范化页面结果。缓存键包含文件哈希、算法版本、配置、
Parser/OCR 版本和 schema 版本。失败与 OCR 冲突不会长期缓存，缓存损坏会显式报告并重新计算。

所有阈值位于 `configs/pdf-routing-v1.json`，可以复制为候选配置后离线评测。阈值变化只重新运行
便宜的路线决策；真正改变路线的页面才需要新的昂贵解析结果。

当前只有三份真实样本，因此 v1 阈值是保守初值，不是通用最优值。合成 fixture 验证规则边界，
真实样本验证实际表现。增加新的文档类型后必须重新校准。

## OCR 语言

默认 RapidOCR profile 为 `ch`，允许显式覆盖。隐藏文字层明显以假名或韩文为主但仍配置 `ch` 时，
路由产生 `OCR_LANGUAGE_UNSUPPORTED`，不会自动切换或下载另一个模型。纯图片页没有可靠语言证据
时只能使用调用方明确指定的 profile。

## 预览与评测

只运行轻量诊断，不加载模型：

~~~powershell
uv run python scripts/pdf_routing_preview.py `
  data/PDF/native.pdf data/PDF/layout.pdf data/PDF/scan.pdf
~~~

显式执行完整 Parser/OCR 路由：

~~~powershell
uv run python scripts/pdf_routing_preview.py --run-parsers `
  data/PDF/native.pdf data/PDF/layout.pdf data/PDF/scan.pdf
~~~

第二个命令可能初始化 Docling/OCR 模型。运行前必须确认模型已经在本地可用；本项目不会为人工验收
静默批准模型下载。

应用人工 OCR 决定并进入结构感知切分：

~~~powershell
uv run python scripts/pdf_chunking_preview.py --auto-route `
  --ocr-review-decisions output/pdf-routing-preview/scan.ocr-decisions.json `
  data/PDF/scan.pdf
~~~

离线比较候选阈值：

~~~powershell
uv run python scripts/pdf_routing_evaluate.py `
  --config configs/pdf-routing-v1.json `
  --manifest data/PDF/sample_manifest.json
~~~

## 测试策略

默认测试由纯策略单元测试、合成边界、fake backend 编排测试和缓存测试组成，不运行真实模型。
真实 Docling/OCR 结果由 JSON/Markdown 预览人工确认。

## 已知限制

- 内部文字质量不能证明少量合法字符正确；
- OCR 对照双方都合理时需要人工复核；
- 人工复核是逐页显式决定，不提供无条件接受整份文档的快捷方式；
- 当前 OCR 结果没有可靠的逐词置信度传播；
- 复杂布局规则是保守启发式，可能把简单页面升级到 Docling；
- 当前三份真实样本不足以估计生产准确率；
- v1 不恢复图片语义、流程图关系、跨页表格或手写内容；
- v1 不做 block/region 级 Parser 仲裁；
- 缓存不自动清理；
- 真实模型升级后必须重新生成 provenance、缓存和人工预览。

## 阶段验收

自动质量门禁通过后，还必须人工检查三份样本的路线、理由、Parser/OCR、冲突、warning 和最终页面
状态。人工确认之前，本阶段不视为验收完成。
