# 学习 07：接入真实生成模型

## 真实项目也是这样做生成层吗

是。常见的 RAG 在线请求链路是：

```text
用户问题
  -> query Embedding
  -> 向量或混合检索
  -> 可选重排
  -> 组装受长度限制的证据上下文
  -> 调用生成模型
  -> 校验答案、引用和安全规则
  -> 返回答案并记录延迟、用量和链路信息
```

不同项目会替换向量数据库、Embedding、重排器和生成模型，但“检索负责找证据，生成负责依据证据组织答案”的职责边界非常常见。

本项目暂时自行完成检索，再把 Top-K 证据发送给 OpenAI。这样可以完整学习 RAG 各环节，而不是直接使用托管文件检索隐藏实现细节。

## 为什么使用 Responses API

OpenAI 官方文档建议新的直接模型请求使用 Responses API。`OpenAITextGenerator` 只实现项目自己的 `TextGenerator` 协议，因此业务代码不知道 OpenAI SDK 的具体调用方式，未来可以替换为本地模型或其他服务。

## 第一版模型配置

| 配置 | 当前值 | 理由 |
| --- | --- | --- |
| 模型 | `gpt-6-luna` | 当前任务边界明确，优先降低成本与延迟 |
| reasoning effort | `none` | 证据摘要不需要复杂多步推理 |
| 最大输出 | 512 tokens | 限制答案长度和单次费用 |
| API 响应存储 | `store=False` | 不把响应保存为可通过 API 后续检索的 Response |

`store=False` 不等价于零数据保留。处理敏感资料前，仍需根据账户的数据控制能力、合规要求和官方数据政策进行评估。

## API Key 如何处理

Key 不写入源码、配置文件、命令参数、测试输出或 Git。官方 SDK 会从当前进程的 `OPENAI_API_KEY` 环境变量读取它。

在 PowerShell 中使用隐藏输入创建仅当前终端会话可见的环境变量：

```powershell
$secureApiKey = Read-Host "请输入 OpenAI API Key" -AsSecureString
$env:OPENAI_API_KEY = [System.Net.NetworkCredential]::new("", $secureApiKey).Password
```

关闭该 PowerShell 窗口后，此会话变量随之消失。不要把真实 Key 粘贴进对话或提交到仓库。

## 自动测试为什么不调用 API

测试使用与官方 SDK 形状相同的假客户端，验证模型名、推理等级、输出上限和 `store=False`。这样自动测试不会产生费用，也不会因为网络或账户状态变得不稳定。

真实调用属于需要人工确认的集成实验，每次运行都应明确知道将发送哪些证据以及可能产生费用。

## 首次真实闭环验证

2026-10-17，用户使用本地 Qwen Embedding 和 OpenAI 生成层完成首次人工验证：

```text
检索排名 1：manual://b，score=0.6147
检索排名 2：manual://a，score=0.5938
```

生成答案正确使用 `[1][2]` 引用，同时明确指出现有证据只说明了 RAG 的检索生成流程和 Embedding 的向量映射作用，没有充分解释“为什么 RAG 需要向量”。这验证了证据约束能够抑制无依据补充，也暴露了当前演示语料不够完整的问题。

人工验收结论：本地向量化、Top-K 检索、证据组装、真实模型生成和引用回溯均已跑通。
