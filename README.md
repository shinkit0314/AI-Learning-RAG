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

## 当前进度

第一个里程碑“工程基线”和第二个里程碑“最小 RAG 闭环”已经通过验收。PDF 知识库阶段
已经完成三类样本的解析基线、路由决策和可追溯 Chunk 输出契约，下一步将实现结构感知切分，
再把 PDF Chunk 接入现有 Embedding 与检索流程。

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
