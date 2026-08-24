"""Demo 04 —— 循环（Agent 的心脏）：模型 + 工具 + 循环 = ReAct Agent。

上一个 demo 是"一次跑完"的 DAG；Agent 的本质是【循环】：
    LLM 思考 -> 调工具 -> 观察结果 -> 再思考 -> ... -> 给出最终答案

LangGraph 实现循环的方式非常朴素：
  - 一条从 tool 节点【指回】llm 节点的边（back edge）
  - 一条条件边决定"继续调工具"还是"结束"

本 demo 手写一个最小的 ReAct 循环，让你看清楚 create_react_agent（demo05）
内部到底发生了什么。

关键点：
- should_continue 检查 last message 的 tool_calls —— 这是 Agent 循环的"脉搏"。
- 工具结果要用 ToolMessage 包起来，并且 tool_call_id 必须对得上。
"""
from typing import Annotated, TypedDict

from common import banner, get_llm
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


class State(TypedDict):
    messages: Annotated[list, add_messages]


# ---------- 定义工具 ----------
@tool
def add(a: int, b: int) -> int:
    """计算两个整数相加。"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """计算两个整数相乘。"""
    return a * b


tools = [add, multiply]

# ---------- 模型：离线时脚本化回放 "先调工具 -> 拿到结果 -> 总结" ----------
OFFLINE_SCRIPT = [
    AIMessage(
        content="",
        tool_calls=[{"name": "add", "args": {"a": 3, "b": 5}, "id": "call_1", "type": "tool_call"}],
    ),
    AIMessage(
        content="",
        tool_calls=[{"name": "multiply", "args": {"a": 8, "b": 7}, "id": "call_2", "type": "tool_call"}],
    ),
    AIMessage(content="3 加 5 等于 8，再乘 7 等于 56。最终答案：56"),
]
llm = get_llm(responses=OFFLINE_SCRIPT)
llm_with_tools = llm.bind_tools(tools)   # 把工具 schema 绑给模型（假模型重写了 bind_tools 直接返回自身）


# ---------- 节点 ----------
def call_model(state: State) -> dict:
    response = llm_with_tools.invoke(state["messages"])
    if response.tool_calls:
        print(f"  [llm] 决定调工具: {[tc['name'] for tc in response.tool_calls]}")
    else:
        print(f"  [llm] 最终回答: {response.content}")
    return {"messages": [response]}


# ToolNode 是官方预置节点：执行 tool_calls、包成 ToolMessage、带 tool_call_id
tool_node = ToolNode(tools)


def should_continue(state: State) -> str:
    last = state["messages"][-1]
    if last.tool_calls:        # 模型还想调工具 -> 回到工具节点
        return "tools"
    return END                 # 没有工具调用了 -> 结束


builder = StateGraph(State)
builder.add_node("llm", call_model)
builder.add_node("tools", tool_node)
builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", should_continue, ["tools", END])  # 分支：继续 or 结束
builder.add_edge("tools", "llm")   # ★ 回边：工具结果送回模型 —— 循环诞生
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 04 循环：手写最小 ReAct（模型 + 工具 + 回边）")
    result = app.invoke({"messages": [("user", "先算 3+5，再把结果乘以 7")]})

    print("\n--- 完整消息轨迹（ReAct 的 trajectory）---")
    for m in result["messages"]:
        if m.type == "ai" and m.tool_calls:
            calls = ", ".join(f"{tc['name']}{tuple(tc['args'].values())}" for tc in m.tool_calls)
            print(f"  AI   -> 调用工具: {calls}")
        else:
            print(f"  {m.type:5s}-> {m.content}")

    print(
        "\n要点：\n"
        "  1. 循环 = 一条 tools -> llm 的回边 + 一条 llm -> END/tools 的条件边。\n"
        "  2. ToolMessage.tool_call_id 必须与 AIMessage.tool_calls[].id 对应。\n"
        "  3. 这个结构就是 create_react_agent 的骨架（下一课）。"
    )
