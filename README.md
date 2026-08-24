# LangGraph Demo —— 渐进式学习项目

10 个可运行的 demo，从最小图一路讲到多 Agent 与流式输出。
**离线可跑**：本机 LLM 端点（vLLM/Ollama 等）可用时自动用真模型，
不可用时自动回退到脚本化假模型，输出依然完整、确定。

## 环境说明

使用 [uv](https://docs.astral.sh/uv/) 创建虚拟环境并安装依赖。
依赖版本已在 `requirements.txt` 中锁定（langgraph 1.2.11 / langchain 1.3.16 / langchain-core 1.6.0 / langchain-openai 1.6.0，本 demo 按 langgraph 1.x API 编写）：

```bash
# 安装 uv（已装可跳过）
pip install uv

# 进入项目目录（重要：demo 用同目录导入 common.py）
cd langgraph-demo

# 创建虚拟环境并安装依赖
uv venv --python 3.13
uv pip install -r requirements.txt
```

## 快速开始

```bash
# 跑单个 demo（任选其一，Windows）
.venv/Scripts/python.exe demos/01_min_graph.py

# 一键跑全部
.venv/Scripts/python.exe run_all.py
```

Linux/macOS 下解释器路径为 `.venv/bin/python`。

### 接入真模型（可选）

demo 默认探测 `http://localhost:8000/v1`。要接入本机模型，复制 `.env.example` 为 `.env` 并修改：

```bash
cp .env.example .env
```

```dotenv
LOCAL_API_URL=http://localhost:11434/v1   # Ollama（vLLM 默认 http://localhost:8000/v1）
LOCAL_MODEL_NAME=qwen2.5:7b                # 模型名
LOCAL_API_KEY=not-needed                   # 本地部署通常不需要
```

也可以直接设环境变量（优先级高于 `.env`）：`LOCAL_API_URL` / `LOCAL_MODEL_NAME` / `LOCAL_API_KEY`。

不设置 `LOCAL_MODEL_NAME` 也能跑 —— 自动进入"离线演示模式"（脚本化 Fake 模型）。

## Demo 一览（建议按序学习）

| # | 文件 | 主题 | 一句话核心 |
|---|------|------|-----------|
| 01 | `01_min_graph.py` | 最小图 | State(TypedDict) + 节点函数 + 边 + compile |
| 02 | `02_reducers.py` | Reducer | 状态合并语义：覆盖 vs `Annotated[T, reducer]` |
| 03 | `03_conditional_edges.py` | 条件边 | 路由函数返回节点名，Agent 分支逻辑的基础 |
| 04 | `04_loop_react.py` | 循环 | tools→llm 回边 + 条件退出 = ReAct Agent 心脏 |
| 05 | `05_prebuilt_react_agent.py` | 预置组件 | `create_agent` 一行顶手写几十行 |
| 06 | `06_checkpointer_memory.py` | 持久化/记忆 | checkpointer + thread_id = 多轮对话记忆 |
| 07 | `07_human_in_loop.py` | 人机协同 | `interrupt()` 暂停 + `Command(resume=)` 恢复 |
| 08 | `08_send_map_reduce.py` | 动态并行 | `Send` API：map-reduce、动态分支 |
| 09 | `09_subgraphs_multi_agent.py` | 子图/多Agent | 编译好的图当节点挂进父图，supervisor 模式 |
| 10 | `10_streaming.py` | 流式输出 | `stream()` 的 values/updates/messages 模式 |

## 知识点文档

核心知识点详解见 **[知识点说明.md](知识点说明.md)** —— 每个 demo 对应一节，
包含 API 速查、常见坑、生产建议。

## 目录结构

```
langgraph-demo/
├── README.md            # 本文件
├── 知识点说明.md         # 核心知识点详解（配合 demos 阅读）
├── run_all.py           # 一键运行全部 demo
├── demos/
│   ├── common.py        # 公共工具：UTF-8 保护 / get_llm() / 离线回退
│   ├── 01_min_graph.py
│   ├── 02_reducers.py
│   ├── 03_conditional_edges.py
│   ├── 04_loop_react.py
│   ├── 05_prebuilt_react_agent.py
│   ├── 06_checkpointer_memory.py
│   ├── 07_human_in_loop.py
│   ├── 08_send_map_reduce.py
│   ├── 09_subgraphs_multi_agent.py
│   └── 10_streaming.py
```
