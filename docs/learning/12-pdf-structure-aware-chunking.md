# 学习 12：结构感知 PDF 切分

状态：结构感知切分器已实现并完成自动测试，等待阶段最终人工验收。

## 本阶段位于 RAG 的哪里

PDF 进入 Embedding 之前需要经过三层：

~~~text
PDF
  -> Parser / OCR
  -> ParsedDocument / ParsedBlock
  -> 结构感知切分器（本阶段）
  -> TraceablePdfChunk
  -> Embedding / 向量索引（不在本阶段）
~~~

本阶段不读取具体 Parser 的私有结构。PyMuPDF、Docling、RapidOCR 或未来 Parser 只要能输出
`ParsedDocument`，都能调用同一个 `chunk_pdf_document()`。

## 本阶段解决的问题

结构感知切分器负责：

- 阻止不可用页面、固定页眉页脚、页码和图片内部文字进入普通 Chunk；
- 根据标题层级维护 `section_path`；
- 在章节、内容类型、页面和长度边界之间组合正文；
- 使用 `ChunkBlockSlice` 精确切分超长来源 block；
- 为正文、表格、公式、图注、代码和未知内容采用明确策略；
- 保留页码、bbox、reading order、block ID、解析方法、warning 和双重字符偏移；
- 生成稳定 `source_fingerprint`、Chunk ID 和 `chunking_profile`；
- 返回 `TraceablePdfChunk`，但不生成 Embedding。

## 顶层接口

~~~python
result = chunk_pdf_document(document, PdfChunkingConfig())
~~~

`PdfChunkingResult` 包含：

- `chunks`：按文档顺序排列的 `TraceablePdfChunk`；
- `diagnostics`：过滤、推断、降级和异常的结构化诊断；
- `status`：`completed`、`partial` 或 `failed`。

顶层接口按固定顺序执行：

~~~text
过滤与章节状态
  -> 超长正文切分
  -> 同页正文组合
  -> 跨页正文组合
  -> 表格策略
  -> 公式策略
  -> 图注策略
  -> 代码策略
  -> OTHER 隔离
  -> TraceablePdfChunk
~~~

这些步骤只改变 Chunk 边界，不修改来源文字。

## 默认配置

~~~text
strategy_version=pdf-structure-v1
length_unit=characters
target_size=400
hard_max_size=600
overlap_percent=20
~~~

当前长度单位是 Python 字符数，不是 token，也不是 UTF-8 字节或 UTF-16 码元。配置序列化为：

~~~text
pdf-structure-v1;length=characters;target=400;hard_max=600;overlap_percent=20
~~~

这个字符串就是 `chunking_profile`。任何会改变 Chunk 边界的配置或算法版本都必须反映在 profile
中，避免不同策略生成相同身份。

## 页面和 block 门禁

以下页面不会进入切分：

- `ParseStatus.NEEDS_OCR`；
- `ParseStatus.FAILED`；
- 带有已知阻断 warning 的页面；
- 带有未知 page warning 的页面。

以下 block 会被过滤，但同页其他有效内容仍可继续：

- 固定 `header`；
- 固定 `footer`；
- `printed_page_number`；
- `picture_text`。

已知可接受 warning 会进入 Chunk warning，同时在结果 diagnostics 中留下记录。未知 block warning
默认只阻断对应 block，未知 page warning 默认阻断整页。`PAGE_n_*` 文档汇总 warning 不会重复执行
门禁。

这种策略的原则是：不确定内容可以被隔离或阻断，但不能静默进入 Embedding。

## 标题和 section_path

标题只由 Parser 标记为 `TITLE` 或 `HEADING` 的 block 驱动。切分器不会把普通正文中提到的
“3.2.1 切分参数”重新识别为标题，因此引用章节编号不会意外改变当前章节。

标题层级的优先级是：

1. Parser 提供的 `heading_level`；
2. 标题开头的阿拉伯数字层级，例如 `3.2.1` 推断为三级；
3. 无法确定时保守地按一级标题处理，并产生 warning。

如果 Parser 层级和数字编号冲突，以 Parser 为准并产生诊断。层级跳跃、重复文档标题和没有正文的
标题也都会留下诊断。

标题先进入 pending 状态，不立即生成无意义的标题 Chunk。连续标题会与所属章节的第一个正文
Chunk 一起输出；标题文字只附加一次，后续 Chunk 只通过 `section_path` 保留章节上下文。新章节
是正文组合的硬边界。

## 正文策略

正文包括 `PARAGRAPH` 和 `LIST_ITEM`。

### 超长 block

超长正文按以下优先级寻找边界：

1. 句末标点；
2. 换行；
3. 空白；
4. `hard_max` 硬切。

边界至少达到目标长度的 50%，避免为了使用一个很近的标点生成过短片段。每个片段都使用
`ChunkBlockSlice` 保存精确 `source_text_start/source_text_end`。如果正文前带 pending 标题，只在
第一个片段附加标题。

### 相邻正文组合

同页正文只有在以下条件同时满足时才组合：

- 内容类型兼容；
- `section_path` 相同；
- 下一单元没有新的标题前缀；
- 组合后不超过 `hard_max`。

表格、公式、图注、代码和 `OTHER` 都会先结束当前正文 Chunk。

### overlap

overlap 只重复一个完整的正文单元，并且该单元必须不超过 `target_size * overlap_percent`。已经被
切开的正文片段、标题、表格、公式、图注、代码和 `OTHER` 都不参与 overlap。这样不会为了
重叠再次制造无法追溯的半段文字。

## 跨页正文组合

正文可以跨相邻物理页组合，但必须满足：

- 两侧都是正文；
- 页码相邻；
- `section_path` 相同；
- 下一页没有新标题前缀；
- 长度仍在限制内。

页码来自来源 block，而不是通过 Chunk 的显示位置推断。跨页时不会自动删除连字符或修复断词，
因为这会破坏原始字符偏移。行末连字符和解析方法变化会产生诊断，交给后续质量评估处理。

## 不同内容类型的策略

| 内容类型 | 当前策略 | 超长行为 | overlap |
| --- | --- | --- | --- |
| 正文 | 相邻兼容 block 组合 | 句子、换行、空白、硬切 | 仅完整正文单元 |
| 表格 | 独立 Chunk | 行、空白、硬切；不重复表头 | 否 |
| 公式 | 独立原子 Chunk | 保持完整，允许超过 hard max 并报警 | 否 |
| 图注 | 独立原子 Chunk | 保持完整，允许超过 hard max 并报警 | 否 |
| 代码 | 独立 Chunk | 定义、空行、完整行、空白、硬切 | 否 |
| OTHER | 隔离 Chunk | 保持完整，允许超过 hard max 并报警 | 否 |

### 表格

表格不会和正文或其他表格自动合并。超长表格先按换行切分，无法保持完整行时才退化到空白或
硬切。当前版本不猜测哪一行是表头，也不在后续 Chunk 中复制表头，避免生成不属于原文的新
来源范围。单行超过 `hard_max` 时会产生 `TABLE_ROW_SPLIT`。

### 公式

公式保持原子性。即使超过 `hard_max` 也不硬切 LaTeX 或数学表达式，而是产生
`FORMULA_EXCEEDS_HARD_MAX`。公式通常较短，这个例外比破坏语法更安全。

### 图注

图注保持独立，并保留 Parser 提供的 `parent_picture_id`。切分器不根据距离、“图 1”或图片顺序
重新猜测图注关系。缺少确认关系时仍输出图注，但产生 `FIGURE_CAPTION_PARENT_MISSING`。

### 代码

代码保持原始缩进和换行。超长代码优先在顶层 `def`、`async def`、`class`、JavaScript
`function` 或空行处切分，再退化到完整行、空白和硬切。所有切片重新连接后必须与原代码逐字符
一致。单行超过 `hard_max` 时产生 `CODE_LINE_SPLIT`。

### OTHER

未知类型不会静默变成正文，也不会消失。它作为 `OTHER` 独立输出并产生
`UNSUPPORTED_CONTENT_TYPE`。超长时保持完整并产生 `OTHER_EXCEEDS_HARD_MAX`。本阶段不生成
Embedding，因此“默认不索引 OTHER”只保留为后续索引阶段的门禁要求。

## 稳定身份和输出顺序

最终 `chunk_index` 按所有内容类型交错后的文档顺序从 0 连续编号，只用于展示和排序，不参与
身份计算。

`source_fingerprint` 由文档、内容类型、章节路径、来源 block、字符范围和正文计算。Chunk ID 再由
`chunking_profile + source_fingerprint` 计算。因此：

- 同一文档、同一配置重复运行会生成相同 ID；
- 修改显示顺序不会改变来源身份；
- 修改切分配置会改变 Chunk ID，但相同来源仍可通过 fingerprint 识别；
- 相同文字来自不同 block 时不会被误认为同一来源。

## 人工检查结果

使用 `scripts/pdf_chunking_preview.py` 对仓库三份真实样例生成了完整 JSON 和可读 Markdown：

~~~powershell
uv run python scripts/pdf_chunking_preview.py `
  data/PDF/native.pdf `
  data/PDF/layout.pdf `
  data/PDF/scan.pdf
~~~

结果写入被 Git 忽略的 `output/pdf-chunking-preview/`：

| 样例 | Chunk 数 | 最大 Python 字符数 | 主要观察 |
| --- | ---: | ---: | --- |
| native.pdf | 3 | 590 | footer 过滤、跨页正文组合 |
| layout.pdf | 68 | 595 | 页眉页脚过滤、超长 block 切片 |
| scan.pdf | 10 | 545 | OCR warning、overlap、跨页组合 |

这些预览直接调用 PyMuPDF 基线。PyMuPDF 当前把大多数内容标记为 `PARAGRAPH`，所以预览中的
`section_path` 为空，表格、公式和图注也不会自动获得语义类型。这个结果说明切分器不会凭空
发明上游结构；结构化类型策略由 Docling fixture 和单元测试覆盖。

`scan.pdf` 预览使用 PDF 原有隐藏 OCR 文字层，并带有 `EMBEDDED_OCR_TEXT_LAYER` warning。它只用于
观察切分行为，不代表已经完成 OCR 质量路由。是否信任隐藏层或强制 RapidOCR，必须在生成
`ParsedDocument` 之前决定。

## 已知限制和失败案例

- 切分器依赖 Parser 提供正确 block 类型和 reading order；不能修复双栏排序错误；
- 标题层级推断目前只支持标题开头的阿拉伯数字编号，不推断罗马数字或中文序号；
- 不从普通段落中重新分类标题，避免章节引用误判；
- 不恢复跨页表格，不复制表头；
- 不理解图片、图表或流程图语义；
- 不在切分阶段推断图片和图注关系；
- 代码定义边界目前显式支持 Python 和 JavaScript，其他语言主要按空行和完整行处理；
- 公式、图注和 `OTHER` 为保持原子性，异常情况下可能超过 `hard_max`，但会产生诊断；
- 不修复 OCR 错字、断词或乱码；
- 不自动比较隐藏 OCR、RapidOCR、PaddleOCR 或其他 OCR 结果；
- 长度单位仍是字符，尚未针对具体 Embedding tokenizer 优化；
- 当前短 Chunk ID 只用于学习项目展示，持久化时仍需使用完整 fingerprint 和 profile 防冲突。

## 验收范围

自动测试覆盖：正文、标题层级、连续标题、无标题、跨页、超长 block、表格、公式、图注、代码、
`OTHER`、异常页面、warning 门禁、稳定身份和全局输出顺序。

阶段结束前仍需：

1. 运行完整 Ruff 和 pytest 质量门禁；
2. 人工确认三份预览和本文档；
3. 由用户决定何时 commit、push 和创建 PR。

## 后续方向

本阶段之后有两个独立方向，不能在当前分支静默展开：

1. OCR 质量路由：比较隐藏 OCR、Docling + RapidOCR，以及必要时 PaddleOCR；
2. 索引接入：将已验收的 `TraceablePdfChunk` 接入 Embedding、向量索引和检索。

具体先后顺序应在本阶段合并后单独确认。
