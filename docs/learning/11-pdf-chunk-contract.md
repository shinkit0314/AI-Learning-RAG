# 学习 11：可追溯 PDF Chunk 输出契约

状态：输出契约已实现，结构感知切分算法尚未开始。

## 本阶段解决的问题

PDF 解析器输出的是页面和 block，而 Embedding 接收的是字符串。若只把多个 block 拼成字符串，
页码、bbox、阅读顺序和 OCR 来源就会在向量化之前丢失。本阶段定义两者之间的稳定边界：

~~~text
ParsedBlock
  -> ChunkCandidate
  -> TraceablePdfChunk
       ├─ content：交给 Embedding 的正文
       ├─ section_path：章节上下文
       └─ source_regions：正文字符范围到 PDF 区域的映射
~~~

这一阶段只定义最终输出和验证规则，不决定相邻 block 应该如何组合。组合策略属于下一阶段的
结构感知切分。

## 为什么不能只保存页码

一个 chunk 可能由同一页的多个段落组成，也可能跨越两页。只保存 `page_start` 和 `page_end`
只能说明大致来源，无法回答：

- 回答中的这句话来自哪个 block；
- 应该在页面上的哪个区域绘制高亮；
- 该文字来自原生文本、Docling 还是重跑 OCR；
- OCR 置信度和解析警告是否需要展示给用户。

因此，每个来源 block 都转换为一个 `ChunkSourceRegion`。

## ChunkSourceRegion

每个来源区域包含：

- `block_id`：稳定的原始 block 标识；
- `page_number` 和 `bbox`：PDF 页面位置；
- `block_type`：标题、段落、表格等语义类型；
- `extraction_method`：原生提取、版面解析或 OCR；
- `reading_order`：在原页面中的阅读顺序；
- `content_start` 和 `content_end`：该 block 在 chunk 正文中的字符范围；
- `confidence` 和 `warnings`：质量判断信息。

字符范围使用 Python 字符串的左闭右开规则 `[content_start, content_end)`。因此可以直接使用：

~~~python
source_text = chunk.content[region.content_start : region.content_end]
~~~

重新得到进入该 chunk 的 block 文字。

## TraceablePdfChunk

最终 chunk 包含：

- 稳定 `id`；
- `document_id`；
- 用于展示顺序的 `chunk_index`；
- 交给 Embedding 的 `content`；
- `content_type`；
- `section_path`；
- 一个或多个 `source_regions`。

`page_start`、`page_end`、`source_block_ids` 和 `bboxes` 都可以从来源区域确定性得到，避免存储
互相矛盾的重复字段。

## 稳定 ID 的设计

chunk ID 由以下内容计算 SHA-256：

- 文档 ID；
- 内容类型；
- 章节路径；
- 来源 block ID；
- chunk 正文。

`chunk_index` 不参与 ID。若只是在文档前面增加了另一个 chunk，原 chunk 的显示顺序可能变化，
但其内容和来源没有变化，不应该在未来的向量索引中被当成全新记录。

相反，即使两个 chunk 的文字完全相同，只要来源 block 不同，它们的 ID 仍然不同。

## 第一版内容类型

当前定义：

- `body_text`：标题、正文和列表组成的普通文本；
- `table`：需要独立处理的表格；
- `formula`：需要独立处理的公式；
- `figure_caption`：需要独立检索或展示的图注。
- `code`：保留换行和缩进、使用代码边界切分的代码块。

图中节点文字仍然不会进入普通 `ChunkCandidate`，因此也不会通过普通路径生成
`TraceablePdfChunk`。

Docling 的 `code` 标签会映射为 `BlockType.CODE`，再由切分器生成 `code` Chunk。代码不会执行
普通正文的空白标准化；下一阶段将优先按函数、类或空行边界切分，而不是从任意字符位置截断。

## 契约验证

构造 chunk 时会拒绝：

- 空 ID、空正文或空来源；
- 非法页码、阅读顺序或置信度；
- 超出正文长度的字符范围；
- 相互重叠的字符范围；
- 重复的来源 block；
- 不符合文档阅读顺序的来源区域。

这些检查的目的不是修复解析错误，而是在错误数据进入 Embedding 和向量索引前尽早失败。

## 下一阶段

下一阶段实现结构感知切分器，负责：

1. 根据页面状态和 block 类型过滤输入；
2. 维护标题形成的 `section_path`；
3. 在章节边界和长度限制之间组合相邻 block；
4. 为表格、公式和图注选择独立策略；
5. 输出本阶段定义的 `TraceablePdfChunk`。
