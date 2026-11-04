# 环境准备指南

## 这一步解决什么问题

项目代码依赖特定版本的 Python 和第三方库。环境准备的目标是让不同对话、不同 Worktree 和不同时间安装出的依赖保持一致，避免出现“这边可以运行，换个目录就报错”的情况。

项目通过以下文件描述环境：

- `.python-version`：指定 Python 3.12。
- `pyproject.toml`：声明项目直接依赖和开发工具。
- `uv.lock`：锁定完整依赖的精确版本。
- `.venv`：当前目录实际使用的虚拟环境，不提交到 Git。

## Windows 下的准备步骤

### 第一步：进入项目目录

打开 PowerShell 后，先执行：

```powershell
Set-Location "D:\github\AI-Learning-RAG"
```

执行后，提示符应以项目路径开头：

```text
PS D:\github\AI-Learning-RAG>
```

这是必要步骤。若在 `C:\Windows\system32` 等其他目录运行命令，uv 无法找到本项目的 `pyproject.toml` 和依赖。

### 第二步：准备环境

执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1
```

这个脚本会：

1. 查找已经安装的 uv。
2. 根据 `.python-version` 选择 Python 3.12。
3. 根据 `uv.lock` 创建或同步当前目录的 `.venv`。
4. 输出实际使用的 Python 版本。

它不会替换系统 Python，也不会把依赖安装到系统环境。重复执行是安全的：环境已经一致时，uv 只会进行快速检查。

在需要运行本地 Embedding 模型的阶段，改用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -WithLocalEmbedding
```

这个开关会额外安装 `local-embedding` 可选依赖，其中包括 CUDA 12.6 版本的 PyTorch 和
Sentence Transformers。GPU 依赖体积较大，因此普通的 API 开发和自动化测试不会默认安装它们。

需要同时运行本地 Embedding 与 OpenAI 生成层时，使用：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -WithLocalEmbedding -WithOpenAI
```

`-WithOpenAI` 只安装官方 Python SDK，不会读取 API Key，也不会自动发出付费请求。

成功时应看到类似输出：

```text
[1/2] Synchronizing dependencies from uv.lock...
Resolved 32 packages
Checked 30 packages
[2/2] Verifying the project Python version...
Python 3.12.x
```

### 第三步：验证项目命令

```powershell
uv run pytest
```

当前工程基线应显示全部测试通过。`uv run` 会自动使用当前项目的 `.venv`，因此不需要手动激活虚拟环境。

## 新对话或新 Worktree 时怎么办

- 同一项目目录的新对话可以继续使用已有 `.venv`，但仍应先确认当前目录。
- 新 Worktree 位于另一个目录，不会复制 `.venv`。在新 Worktree 中重新运行 `scripts/setup.ps1` 即可。
- 新建出来的 `.venv` 不是旧环境的物理副本，但由于使用同一个 `uv.lock`，Python 依赖版本保持一致。

新对话中的 Codex 应先解释上述步骤，提醒你亲自执行环境命令，并等待你确认结果；除非你明确要求，否则不应自动安装或同步环境。

## 模型文件保存在哪里

本项目把 Hugging Face 模型缓存放在项目目录下的 `.cache/huggingface`，也就是当前项目位于
`D:\github\AI-Learning-RAG` 时，模型会保存到：

```text
D:\github\AI-Learning-RAG\.cache\huggingface
```

模型加载代码会显式使用该目录，而不是依赖 Hugging Face 默认的 C 盘用户缓存。`.cache`
已经被 `.gitignore` 排除，不会把数 GB 的模型文件提交到 Git。

## 常用命令

```powershell
# 查看项目 Python 版本
uv run python --version

# 运行格式检查
uv run ruff format --check .

# 运行代码检查
uv run ruff check .

# 运行测试
uv run pytest

# 启动开发服务器
uv run uvicorn ai_learning_rag.main:app --reload
```

遇到错误时，不要急着更换 Python 或降低依赖版本。先记录以下信息并交给 Codex 判断：

```powershell
Get-Location
uv --version
uv run python --version
```
