# AI-Learning-RAG

面向 AI 初学者、以学习和实践为主线的 RAG（检索增强生成）系统项目。

## 项目文档

- [项目目标与需求](docs/PROJECT_REQUIREMENTS.md)
- [环境准备指南](docs/ENVIRONMENT_SETUP.md)
- [学习 01：工程基线](docs/learning/01-engineering-baseline.md)

## 当前进度

第一个里程碑“工程基线”的最小版本已通过自动验证和人工确认。下一阶段将进入“最小 RAG 闭环”。

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

## 项目结构

```text
src/ai_learning_rag/   应用代码
tests/                 自动化测试
docs/                  需求与学习文档
.github/workflows/     持续集成配置
```
