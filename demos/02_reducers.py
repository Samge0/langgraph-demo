"""Demo 02 —— Reducer：状态如何被"合并"，为什么聊天记录要用 add_messages。

LangGraph 节点返回的 dict 不是简单 update() 进状态：
  - 默认行为：新值【整键覆盖】旧值（demo01 里已见过）
  - 加了 Annotated[list, reducer] 后：新值与旧值由 reducer 函数合并

这是 LangGraph 最重要的机制之一。典型场景：聊天消息。
  messages: Annotated[list, add_messages]
  -> 节点返回 {"messages": [new_msg]} 时不是"替换"而是"追加"，
     并且 add_messages 还会按 message.id 去重/更新。

并行分支写同一个键时，reducer 决定了"合并语义"（拼接？求和？取最大？），
没有 reducer 的键在并行写入时会直接报 InvalidUpdateError。
"""
import operator
from typing import Annotated, TypedDict

from common import banner
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    # 演示三种合并语义：
    plain: str                                    # 1) 默认：整键覆盖
    notes: Annotated[list, operator.add]          # 2) 列表拼接
    messages: Annotated[list, add_messages]       # 3) 消息专用（含按 id 去重）


def writer_a(state: State) -> dict:
    return {
        "plain": "A 的值（会覆盖任何旧值）",
        "notes": ["A 写了一条 note"],
        "messages": [{"role": "user", "content": "你好"}],
    }


def writer_b(state: State) -> dict:
    return {
        "plain": "B 的值（把 A 的覆盖掉了！）",
        "notes": ["B 也写了一条 note"],
        "messages": [{"role": "assistant", "content": "你好呀，有什么可以帮你？"}],
    }


builder = StateGraph(State)
builder.add_node("writer_a", writer_a)
builder.add_node("writer_b", writer_b)
builder.add_edge(START, "writer_a")
builder.add_edge("writer_a", "writer_b")
builder.add_edge("writer_b", END)
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 02 Reducer：状态合并语义")
    result = app.invoke(
        {"plain": "初始值", "notes": ["初始 note"], "messages": []}
    )
    print("plain（覆盖）  :", result["plain"])
    print("notes（拼接）  :", result["notes"])
    print("messages（追加）:")
    for m in result["messages"]:
        print(f"  - {m.type:10s} {m.content}")

    print(
        "\n要点：\n"
        "  1. 无 reducer -> 覆盖；Annotated[T, reducer] -> 按 reducer 合并。\n"
        "  2. 聊天应用的状态几乎总是 messages: Annotated[list, add_messages]。\n"
        "  3. 并行节点写同一个无 reducer 的键 = InvalidUpdateError（LangGraph 帮你防竞态）。"
    )
