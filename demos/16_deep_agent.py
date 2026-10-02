"""Demo 16 —— Deep Agent：计划清单 + 虚拟文件系统（长任务 Agent 的上下文工程）。

2025-2026 深度研究类 Agent（Deep Research / Claude Code / deepagents 库）的共同骨架：

  1. planning：显式的 todo 清单工具，Agent 自己排步骤、勾进度；
  2. 文件系统工作区：中间产物（大纲/草稿/笔记）写进【虚拟文件】，
     消息里只传文件名和摘要 —— 长任务的上下文工程核心：
     把"工作内存"从 messages 挪到 state 的 files 字典里；
  3. 工具循环：Demo 04 的 ReAct 环，只是工具换成了 write_todos/read_file/write_file。

为什么要把内容放进文件而不是聊天记录？任务一长，messages 会被中间产物
撑爆（上下文窗口 + 注意力稀释）；文件可以按需 read 回来，不需要的永远
不进上下文。这就是"沙盒工作区"模式的全部动机。

本 demo：让 Agent 写一份小报告 —— 排计划 -> 写大纲 -> 写草稿 -> 读回校对 -> 交付。
离线时 Fake 模型回放这整条工具调用轨迹；真模型时由 LLM 自主决策。

关键点：
- 自定义 state 字段（todos/files）挂在 ReAct 骨架上，这是生产 Agent 的标准形态。
- 工具函数同时更新 todos/files —— 用闭包捕获图的 state 引用做不到（LangGraph
  节点返回更新），正确做法：tools 节点里手工执行工具并 merge 进 state。
- Deep agent 三件套可继续升级：再挂子 Agent（demo 09）、沙盒执行代码、外部搜索。
"""
from typing import Annotated, TypedDict

from common import banner, get_llm
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    todos: list[dict]      # [{"step": str, "status": "pending|done"}]
    files: dict[str, str]  # 虚拟文件系统：路径 -> 内容


# ---------- 离线脚本：一条真实的 deep-agent 工具轨迹 ----------
OFFLINE_SCRIPT = [
    AIMessage(content="", tool_calls=[{
        "name": "write_todos", "id": "c1", "type": "tool_call",
        "args": {"steps": ["拟定大纲", "撰写草稿", "校对交付"]},
    }]),
    AIMessage(content="", tool_calls=[{
        "name": "write_file", "id": "c2", "type": "tool_call",
        "args": {"path": "outline.md", "content": "# 报告大纲\n1. 背景\n2. 方案\n3. 结论"},
    }]),
    AIMessage(content="", tool_calls=[{
        "name": "write_file", "id": "c3", "type": "tool_call",
        "args": {"path": "draft.md", "content": "# 报告草稿（按 outline.md 展开）\n背景：… 方案：… 结论：…"},
    }]),
    AIMessage(content="", tool_calls=[{
        "name": "read_file", "id": "c4", "type": "tool_call",
        "args": {"path": "outline.md"},
    }]),
    AIMessage(content="报告完成：outline.md 定框架，draft.md 为正文，两文件已在工作区。"),
]
llm = get_llm(responses=OFFLINE_SCRIPT)


# 工具 schema（真模型需要看得到签名与说明；离线假模型忽略）
@tool
def write_todos(steps: list[str]) -> str:
    """更新任务计划清单。steps：步骤列表。"""
    return "ok"


@tool
def read_file(path: str) -> str:
    """读取工作区文件内容。"""
    return "ok"


@tool
def write_file(path: str, content: str) -> str:
    """把内容写入工作区文件（整文件覆盖）。"""
    return "ok"


llm_with_tools = llm.bind_tools([write_todos, read_file, write_file])


# ---------- 工具实现：直接操作 state 的 todos/files ----------
def run_tool(name: str, args: dict, state: State) -> dict:
    if name == "write_todos":
        todos = [{"step": s, "status": "pending"} for s in args["steps"]]
        return {"todos": todos}, f"已排定 {len(todos)} 个步骤"
    if name == "read_file":
        content = state["files"].get(args["path"])
        return {}, content if content is not None else f"[错误] {args['path']} 不存在"
    if name == "write_file":
        # files 是普通 dict 字段：必须【整键覆盖】返回（LangGraph 不做深合并）
        files = {**state["files"], args["path"]: args["content"]}
        return {"files": files}, f"已写入 {args['path']}（{len(args['content'])} 字）"
    return {}, f"[错误] 未知工具 {name}"


def _tick_todos(todos: list[dict], tool_name: str) -> list[dict]:
    """每次调完工具，把第一个 pending 勾成 done（模拟 Agent 推进计划）。"""
    result = [dict(t) for t in todos]
    if tool_name != "write_todos":
        for t in result:
            if t["status"] == "pending":
                t["status"] = "done"
                break
    return result


# ---------- 节点：标准 ReAct 骨架 ----------
def agent(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        print(f"  [agent] 调工具: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print(f"  [agent] 最终回答: {response.content[:40]}…")
    return {"messages": [response]}


def tools(state: State) -> dict:
    last = state["messages"][-1]
    updates, tool_msgs, todos = {}, [], state["todos"]
    for tc in last.tool_calls:                       # 一条 AI 消息可以带多个调用
        upd, result = run_tool(tc["name"], tc["args"], state)
        updates.update(upd)
        todos = _tick_todos(todos, tc["name"])
        tool_msgs.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"]))
    updates["messages"] = tool_msgs
    updates["todos"] = todos
    return updates


def should_continue(state: State) -> str:
    return "tools" if state["messages"][-1].tool_calls else END


builder = StateGraph(State)
builder.add_node("agent", agent)
builder.add_node("tools", tools)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "agent")
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 16 Deep Agent：todo 计划 + 虚拟文件系统工作区")
    result = app.invoke(
        {"messages": [("user", "帮我写一份 LangGraph 技术选型报告，先规划再动笔")],
         "todos": [], "files": {}},
        config={"recursion_limit": 30},
    )

    print("\n--- 计划清单（todos）---")
    for t in result["todos"]:
        mark = "✓" if t["status"] == "done" else " "
        print(f"  [{mark}] {t['step']}")

    print("\n--- 工作区（files）---")
    for path, content in result["files"].items():
        preview = content.replace("\n", " ⏎ ")[:44]
        print(f"  {path}: {preview}…")

    print("\n--- 消息条数 vs 文件体量 ---")
    print(f"  messages 共 {len(result['messages'])} 条；"
          f"文件内容 {sum(len(v) for v in result['files'].values())} 字，"
          f"其中 read 回上下文的只有 outline.md")

    print(
        "\n要点：\n"
        "  1. 三件套 = ReAct 环 + write_todos + 虚拟文件系统，Deep Research 类 Agent 的骨架。\n"
        "  2. 上下文工程：中间产物住 files，需要时 read 回来 —— messages 始终苗条。\n"
        "  3. files 是普通 dict 字段，更新必须整键覆盖（浅拷贝再改，不深合并）。\n"
        "  4. langchain 官方 deepagents 库就是这套骨架的封装（本项目未安装，思路一致）。"
    )
