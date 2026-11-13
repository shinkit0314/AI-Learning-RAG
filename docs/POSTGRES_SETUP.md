# PostgreSQL 与 pgvector 本地验证

## 这一步属于哪里

这是阶段 4.1“可追溯 PDF Chunk 索引接入 v1”的显式集成环境。默认单元测试仍然完全离线；只有人工
准备好测试数据库并打开开关后，PostgreSQL 集成测试才会连接数据库。

本指南不会让应用启动时自动建库、安装扩展或执行迁移。管理员 bootstrap、迁移账号和运行账号保持
分离。

## 1. 先同步 Python 环境

进入仓库：

```powershell
Set-Location "D:\github\AI-Learning-RAG"
```

当前阶段需要 Alembic，并会在人工预览时使用本地 Embedding，因此执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup.ps1 -WithLocalEmbedding
```

该命令根据 `uv.lock` 同步项目 `.venv`，不会修改系统 Python。成功时最后应显示
`Python 3.12.x`，且前面没有 `uv sync failed`。

## 2. 管理员准备数据库和 pgvector

本项目使用两个独立数据库：

- `ai_learning_rag`：本地开发数据；
- `ai_learning_rag_test`：显式集成测试，测试保护会拒绝任何不以 `_test` 结尾的数据库。

管理员需要先创建数据库，并在每个数据库中运行 bootstrap。项目提供了可重复执行的本地脚本，
它只创建缺失数据库，并通过 `\password` 安全地提示设置 migrator/app 密码：

### 2.1 Windows 原生安装 pgvector

当前 PostgreSQL 安装在 `D:\PostgreSQL\18`。pgvector 官方 Windows 流程要求先安装 Microsoft C++
Build Tools，并选择 **Desktop development with C++** workload。安装完成后，以管理员身份打开
`x64 Native Tools Command Prompt for VS 2026`，执行：

```bat
set "PGROOT=D:\PostgreSQL\18"
cd /d %TEMP%
git clone --branch v0.8.7 https://github.com/pgvector/pgvector.git
cd pgvector
nmake /F Makefile.win
nmake /F Makefile.win install
```

成功后应存在：

```text
D:\PostgreSQL\18\share\extension\vector.control
D:\PostgreSQL\18\lib\vector.dll
```

如果 `%TEMP%\pgvector` 已存在，不要递归删除；停止并选择新的明确目录，或手动处理旧目录。
不要使用来源不明的预编译 DLL替代官方源码构建。

### 2.2 创建数据库并执行管理员 bootstrap

```powershell
& "D:\PostgreSQL\18\bin\psql.exe" -U postgres -d postgres `
  -f scripts/postgres/bootstrap_local_databases.sql
```

执行过程会依次要求输入 PostgreSQL 管理员密码、migrator 新密码和 app 新密码。密码输入不会显示字符，
也不会写入脚本。成功时最后应显示 `ai_learning_rag_test | 0.8.7`。

如需为其他环境分别管理登录用户，也可以跳过本地便利脚本，显式创建登录用户后分别授予
`ai_learning_rag_migrator` 与 `ai_learning_rag_app` NOLOGIN 角色。

真实密码不得写入脚本、Git 或聊天记录。

执行 bootstrap 前，PostgreSQL 服务器必须已经安装 pgvector 扩展文件。若出现
`extension "vector" is not available`，说明服务器侧 pgvector 尚未安装，此时停止，不要继续迁移。

## 3. 创建本地登录用户

生产边界使用两个登录用户：

- migrator 登录用户只获得 `ai_learning_rag_migrator` 角色；
- app 登录用户只获得 `ai_learning_rag_app` 角色。

可以在 `psql` 中创建登录用户，然后用 `\password 用户名` 交互设置密码，避免密码进入命令历史：

```sql
CREATE ROLE ai_learning_rag_local_migrator LOGIN;
CREATE ROLE ai_learning_rag_local_app LOGIN;
GRANT ai_learning_rag_migrator TO ai_learning_rag_local_migrator;
GRANT ai_learning_rag_app TO ai_learning_rag_local_app;
\password ai_learning_rag_local_migrator
\password ai_learning_rag_local_app
```

测试数据库可以使用独立 test 登录用户。不要把 migrator 权限授予长期运行的 app 用户。

## 4. 配置本地环境变量

复制 `.env.example` 的变量名到你自己的 `.env` 或 PowerShell 会话中，并替换占位账号。应用不会自动
读取或提交 `.env`。例如当前 PowerShell 会话可设置：

```powershell
$env:AI_LEARNING_RAG_DATABASE_URL = "postgresql+psycopg://.../ai_learning_rag"
$env:AI_LEARNING_RAG_TEST_DATABASE_URL = "postgresql+psycopg://.../ai_learning_rag_test"
```

不要把真实 URL 发到聊天中；只需报告命令成功或错误类型。

## 5. 人工执行迁移

这一步属于 schema 版本管理，会在 `rag` schema 创建四张表和 Alembic 版本表：

```powershell
uv run alembic upgrade head
```

成功时应看到迁移到 revision `20261112_0002`，并且没有自动创建或升级扩展。应用启动不会运行该
命令。

## 6. 显式运行 PostgreSQL 集成测试

先将迁移 URL临时切换为测试数据库并迁移测试库，然后打开显式开关：

```powershell
$env:AI_LEARNING_RAG_DATABASE_URL = $env:AI_LEARNING_RAG_TEST_DATABASE_URL
uv run alembic upgrade head
$env:AI_LEARNING_RAG_RUN_POSTGRES_INTEGRATION = "1"
uv run pytest -m postgres_integration
```

测试会验证 pgvector 版本、索引发布、文档原子替换和精确检索。它不会删除测试数据库，也不会执行
递归文件删除；为方便失败诊断，测试写入的唯一命名记录会保留在 test 数据库中。

本地首次验证推荐使用封装脚本一次完成第 5、6 步：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_postgres_integration.ps1
```

脚本通过安全输入框分别读取 migrator 和 app 密码：迁移只使用 migrator，真实索引读写测试只使用
app。密码只在脚本进程内构造连接 URL，不会显示或写入，完成或失败后都会清除相关环境变量。成功
结果应依次显示两个迁移完成，以及 PostgreSQL 集成测试通过。
