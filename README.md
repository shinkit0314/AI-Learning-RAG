# AI-Learning-RAG

面向 AI 初学者、以学习和实践为主线的 RAG（检索增强生成）系统项目。

## 项目文档

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

## 当前进度

第一个里程碑“工程基线”和第二个里程碑“最小 RAG 闭环”已经通过验收。PDF 知识库阶段
已经完成三类样本的解析基线、路由决策、可追溯 Chunk 输出契约和结构感知切分器。当前正在进行
结构感知切分阶段的文档与人工验收，尚未把 PDF Chunk 接入 Embedding、向量数据库或检索流程。

切分器接收 Parser 无关的 `ParsedDocument`，支持章节路径、正文组合、跨页处理、可追溯超长 block
切片，以及表格、公式、图注、代码和未知内容的隔离策略。OCR 质量门控和 Embedding 接入将作为
后续独立阶段设计。

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
