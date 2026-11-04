# 学习 09：PDF 提取基线与可追溯契约

## 我们现在处于哪里

纯文本 RAG 闭环已经完成。现在进入第三个里程碑“PDF 知识库”的第一步：

~~~text
PDF 样本
  -> 页面类型诊断
  -> 原生文本、版面解析或 OCR
  -> ParsedDocument
  -> ChunkCandidate
  -> 后续切分与 Embedding
~~~

当前已经完成 PyMuPDF 原生基线、Docling 版面解析、扫描页整页 OCR、图表隔离和第一版路由决策。暂不接入向量数据库。

## 三类评测样本

样本清单和人工标准答案保存在 data/PDF/sample_manifest.json。

| 样本 | 预期类型 | 当前用途 |
| --- | --- | --- |
| data/PDF/native.pdf | native_text | 验证原生文字、页码、bbox、公式和图文混排 |
| data/PDF/layout.pdf | complex_layout_text | 验证双栏顺序、标题、图注和页眉页脚 |
| data/PDF/scan.pdf | scanned_with_ocr | 比较已有 OCR 与强制重新 OCR 的质量 |

scan.pdf 每页都有接近整页的扫描图像，同时带有隐藏 OCR 文字层。它不能单独证明“空文本自动转 OCR”路由正确，但可以验证已有 OCR 是否值得信任。后续应再增加一个没有文字层的最小扫描 fixture。

## 人工标准答案

标准答案不是逐字抄录整份 PDF，而是少量、可稳定核对的事实：

- native.pdf：核对第 1、2、6 页的日文标题。
- layout.pdf：核对论文标题、Introduction，以及第 3 页图注。
- scan.pdf：核对章节标题、图 1-1 图注、可扩展性和可维护性标题。
- scan.pdf 的已知 OCR 错误必须保留，以验证质量检查是否能发现失败。

修改样本后必须同时更新文件哈希、页数和人工检查项，避免评测输入悄悄变化。

## 统一输出契约

src/ai_learning_rag/pdf_schema.py 定义解析器无关的数据结构：

- BoundingBox：使用左上角原点和 PDF point，格式为 x0、y0、x1、y1。
- ParsedBlock：保存页码、文本、bbox、阅读顺序、提取方法、置信度和警告。
- ParsedPage：保存页面尺寸、页面类型、解析状态以及有序 block。
- ParsedDocument：保存文件哈希、解析器名称和版本，以及连续的一页起始页码。
- ChunkCandidate：作为后续 chunk 阶段的输入，继续保留来源 block、页码和 bbox。

原生解析、版面解析和 OCR 都必须输出相同结构，下游不能依赖某个具体解析器。

## scan.pdf 的处理策略

默认流程不会盲目信任或盲目忽略已有 OCR：

~~~text
整页扫描图像
├─ 无文字层 -> needs_ocr
└─ 有文字层 -> scanned_with_ocr
               ├─ 质量合格 -> embedded_ocr
               └─ 质量不足 -> rerun_ocr
~~~

实验阶段对同一页面保留两份结果：

1. embedded_ocr：PDF 内已有的隐藏文字。
2. rerun_ocr：把页面渲染为图片后重新识别的文字。

不能静默混合两种结果。最终 block 必须记录 extraction_method，并在没有可信置信度时写入警告。

## 当前完成标准

- 三种预期样本类型已经固定。
- 每份样本都有文件哈希、页数和至少三个人工核对事实。
- 扫描件的已知 OCR 失败已经记录。
- 统一结构强制保留一页起始页码和 bbox。
- ChunkCandidate 能追溯到源 block、页码和坐标。
- 解析失败可以表示为 needs_ocr 或 failed，而不是成功返回空文本。
- PyMuPDF 已输出三个固定样本的页码、bbox、页面诊断和页眉页脚分类。
- Docling 已在论文和幻灯片样本上输出标题、正文、列表、图注、页眉和页脚。
- Docling 版面结果已经人工确认通过。
- RapidOCR 整页重跑修复了已知的中文和英文字符替换，自动检查与人工验收均已通过。
- 图表区域和其中的 OCR 文字已建立父子关系；`picture_text` 不允许进入普通 `ChunkCandidate`。
- 第 3 页图表隔离边界已经通过 bbox 覆盖图完成人工确认，FAIL-005 已关闭。

## Docling 版面解析对照

Docling 使用标准 PDF pipeline，并在原生文字样本上关闭 OCR，避免把版面解析与 OCR
质量混为一次实验。首次运行会下载官方版面模型，后续从本地缓存加载。

~~~powershell
$env:HF_HOME = "D:\github\AI-Learning-RAG\.cache\huggingface"
uv run python scripts/docling_baseline.py data/PDF/layout.pdf data/PDF/native.pdf
~~~

输出保存在 `output/pdf-docling-baseline/`，仍然遵守 `ParsedDocument` 契约。Docling 的
BOTTOMLEFT 坐标会转换成项目统一使用的 TOPLEFT 坐标。

当前对照发现：

- 论文样本可以识别章节标题、图注、列表、页眉和页脚。
- 幻灯片样本可以区分页面标题、列表项和页脚。
- Docling 的 `section_header` 不等于完整标题层级；论文主标题也可能被归为章节标题。
- 图片中的节点关系仍不是普通版面解析可以可靠输出的结构。

## 本阶段结论

PDF 提取基线的各项实验已经完成验收。第一版路线、触发条件、失败策略、不支持范围和下一阶段
chunk 输入整理在 `docs/learning/10-pdf-routing-decision.md`，等待最终路线确认后关闭本课题。

下一阶段不再继续为单个样本增加解析规则，而是实现结构感知的 chunk 构建，并保持从 chunk
到源 block、页码、bbox 和图片区域的完整追溯。
