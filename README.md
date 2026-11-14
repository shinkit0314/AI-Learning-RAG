# AI-Learning-RAG

面向 AI 初学者、以学习和实践为主线的 RAG（检索增强生成）系统项目。

## 项目文档

- [整体开发路线图](docs/DEVELOPMENT_ROADMAP.md)
- [项目目标与需求](docs/PROJECT_REQUIREMENTS.md)
- [环境准备指南](docs/ENVIRONMENT_SETUP.md)
- [学习 01：工程基线](docs/learning/01-engineering-baseline.md)
- [学习 02：纯文本标准化与切分](docs/learning/02-text-chunking.md)
- [学习 03：本地 Embedding 基线](docs/learning/03-local-embedding.md)
- [学习 04：内存向量检索](docs/learning/04-in-memory-vector-search.md)
- [学习 05：串联最小检索流水线](docs/learning/05-retrieval-pipeline.md)
- [学习 06：基于证据生成与引用](docs/learning/06-grounded-generation.md)
- [学习 07：接入真实生成模型](docs/learning/07-openai-generation.md)
- [学习 08：最小 RAG 闭环复盘](docs/learning/08-minimal-rag-retrospective.md)
- [学习 09：PDF 提取基线](docs/learning/09-pdf-extraction-baseline.md)
- [学习 10：PDF 解析路由决策](docs/learning/10-pdf-routing-decision.md)
- [学习 11：可追溯 PDF Chunk 输出契约](docs/learning/11-pdf-chunk-contract.md)
- [学习 12：结构感知 PDF 切分](docs/learning/12-pdf-structure-aware-chunking.md)
- [学习 13：PDF Parser 自动路由 v1](docs/learning/13-pdf-page-routing.md)
- [学习 14：可追溯 PDF 索引与检索](docs/learning/14-pdf-chunk-indexing-and-retrieval.md)
- [学习 15：PDF 证据生成与页级引用](docs/learning/15-pdf-grounded-generation.md)

## 当前进度

第一个里程碑“工程基线”、第二个里程碑“最小 RAG 闭环”，以及 PDF 知识库的解析与切分阶段
已经通过验收。页面级 PDF Parser 自动路由 v1 会先用 PyMuPDF 做轻量诊断，再按页面选择
PyMuPDF、Docling 或 RapidOCR，并保留逐页 route reason、失败尝试和 Extraction provenance；隐藏
OCR 冲突支持逐页、可审计且 fail-closed 的人工选择。

全项目开发路线图已经通过人工验收并合入 `develop`。“可追溯 PDF Chunk 索引接入 v1”已经将
验收过的 `TraceablePdfChunk` 接入 Embedding、持久化向量索引和检索流程，并保留回到原 PDF
页面的完整 provenance；自动检查和人工验收均已通过，并已通过 PR #9 合入 `develop`。“PDF
证据生成与页级引用”已经完成自然语言查询适配、Top-K Evidence 组织、DeepSeek 受约束生成和
page/block/bbox Citation，自动检查与真实模型人工验收均已通过，当前在
`codex/pdf-grounded-generation` 分支等待发布授权。详细阶段顺序见
[整体开发路线图](docs/DEVELOPMENT_ROADMAP.md)，
阶段 4.1 的已确认数据模型、事务边界和验收标准见
[阶段 4.1 设计文档](docs/design/04-01-pdf-chunk-indexing.md)。
本地数据库和显式集成测试准备步骤见
[PostgreSQL 与 pgvector 本地验证](docs/POSTGRES_SETUP.md)。

切分器接收 Parser 无关的 `ParsedDocument`，支持章节路径、正文组合、跨页处理、可追溯超长 block
切片，以及表格、公式、图注、代码和未知内容的隔离策略。Embedding、持久化索引和检索已经作为
独立阶段完成；受证据约束的 PDF 答案生成与页级引用也已完成人工验收。阶段 4.2 合入 `develop`
后，下一阶段先建立检索评估基线，不会直接凭主观印象加入重排器。

## 环境要求

- [uv](https://docs.astral.sh/uv/getting-started/installation/)：管理 Python、虚拟环境和依赖。
- Git。

项目使用 Python 3.12。uv 会根据 `.python-version` 自动获取合适的 Python，不需要手动管理虚拟环境。

新对话不会自动安装环境。Codex 会先解释环境准备步骤，再提醒你亲自运行命令。详细说明请阅读[环境准备指南](docs/ENVIRONMENT_SETUP.md)。

## 开始使用

在 Windows PowerShell 中准备并验证环境：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
```

运行全部质量检查：

```powershell
uv run ruff format --check .
uv run ruff check .
uv run pytest
```

生成本地 PDF Chunk 人工预览：

```powershell
uv run python scripts/pdf_chunking_preview.py data/PDF/native.pdf data/PDF/layout.pdf data/PDF/scan.pdf
```

预览结果写入被 Git 忽略的 `output/pdf-chunking-preview/`，同时包含完整 JSON 和便于阅读的
Markdown。这个命令使用 PyMuPDF 基线，扫描样例会读取已有隐藏 OCR 层；它不会自动运行或比较
RapidOCR/PaddleOCR。

生成不会加载 Docling/OCR 模型的页面路由诊断预览：

```powershell
uv run python scripts/pdf_routing_preview.py data/PDF/native.pdf data/PDF/layout.pdf data/PDF/scan.pdf
```

确认本地模型已经可用后，显式运行完整页面路由预览：

```powershell
uv run python scripts/pdf_routing_preview.py --run-parsers data/PDF/native.pdf data/PDF/layout.pdf data/PDF/scan.pdf
```

完整路由结果写入 `output/pdf-routing-preview/`。未添加 `--run-parsers` 时只生成信号、阈值和路线
决策，不运行 Docling 或 OCR，也不会触发模型下载。

完成本地 Embedding、DeepSeek SDK、PostgreSQL 和 `.env` 准备后，显式运行 PDF 证据生成预览：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_pdf_grounded_generation_preview.ps1
```

该命令会把选中的 PDF Evidence 发送给 DeepSeek，可能产生 API 费用；JSON 和 Markdown 报告写入
`output/pdf-grounded-generation-preview/`。

为隐藏 OCR 冲突生成只读证据和可编辑决定文件：

```powershell
uv run python scripts/pdf_ocr_review.py generate data/PDF/scan.pdf
```

人工只修改 `output/pdf-routing-preview/scan.ocr-decisions.json` 中的 `decision` 和
`review_note`，然后应用决定：

```powershell
uv run python scripts/pdf_ocr_review.py apply data/PDF/scan.pdf
```

人工选择通过后，可以直接生成结构感知 Chunk：

```powershell
uv run python scripts/pdf_chunking_preview.py --auto-route `
  --ocr-review-decisions output/pdf-routing-preview/scan.ocr-decisions.json `
  data/PDF/scan.pdf
```

启动开发服务器：

```powershell
uv run uvicorn ai_learning_rag.main:app --reload
```

启动后可以访问：

- 健康检查：http://127.0.0.1:8000/health
- 交互式 API 文档：http://127.0.0.1:8000/docs

在交互式 API 文档中，可以通过 `POST /documents/preview` 观察文本标准化和切分结果；这个预览接口不会持久化数据。

## 项目结构

```text
src/ai_learning_rag/   应用代码
tests/                 自动化测试
docs/                  需求与学习文档
.github/workflows/     持续集成配置
```
