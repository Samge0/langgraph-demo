"""Demo 06 —— Checkpointer（持久化）与多轮记忆 / thread 概念。

图默认是"一次性的"：invoke 完状态就没了。
加上 checkpointer 后，每一步之后状态都会被存档：
  - 下次用同一 thread_id invoke 时，从上次存档【恢复】继续
  - 这就是 LangGraph 的"多轮对话记忆"实现方式
  - human-in-the-loop（demo07）、time-travel 都建立在它之上

关键概念：
- thread_id：一个会话线程的标识。同 thread = 延续，换 thread = 全新对话。
- MemorySaver：内存版 checkpointer（进程级，重启即失）。
  生产用 SqliteSaver / PostgresSaver（langgraph-checkpoint-* 包）。
- get_state(cfg) 可以随时查看检查点里的当前状态（历史也在）。
"""
from typing import Annotated, TypedDict

from common import FakeToolCallingLLM, banner, get_llm
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# ---------- 用裸 StateGraph 演示（不用预置 Agent，看清机制） ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    turns: int


def chat(state: State) -> dict:
    # 真模型：带上完整历史 invoke，模型能从检查点恢复的记忆里作答；
    # 离线：回声最后一条用户消息（演示记忆用足够了）
    llm = get_llm()
    history = state["messages"]
    n = len([m for m in history if m.type == "human"])
    if isinstance(llm, FakeToolCallingLLM):
        last_user = [m for m in history if m.type == "human"][-1].content
        reply = AIMessage(content=f"(离线回声) 你是第 {n} 次说话，刚才说：'{last_user}'")
    else:
        reply = llm.invoke(history)
    return {"messages": [reply], "turns": n}


builder = StateGraph(State)
builder.add_node("chat", chat)
builder.add_edge(START, "chat")
builder.add_edge("chat", END)

checkpointer = MemorySaver()
app = builder.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    banner("Demo 06 Checkpointer：thread / 多轮记忆 / 状态恢复")

    cfg = {"configurable": {"thread_id": "user-alice-001"}}

    print(">>> 第 1 轮: 我叫 Sam，正在学 LangGraph")
    r1 = app.invoke({"messages": [("user", "我叫 Sam，正在学 LangGraph")]}, cfg)
    print("<<<", r1["messages"][-1].content)

    print("\n>>> 第 2 轮（不带历史，只发新句）: 我刚才说我叫什么？")
    r2 = app.invoke({"messages": [("user", "我刚才说我叫什么？")]}, cfg)
    print("<<<", r2["messages"][-1].content)
    print("   ↑ 图自动带上了第 1 轮的记忆 —— 这就是 checkpointer 的作用")

    # 换一个 thread：全新会话，没有记忆
    cfg_new = {"configurable": {"thread_id": "user-bob-002"}}
    print("\n>>> 换 thread_id 后再问（Bob 的线程）: 我刚才说我叫什么？")
    r3 = app.invoke({"messages": [("user", "我刚才说我叫什么？")]}, cfg_new)
    print("<<<", r3["messages"][-1].content)
    print("   ↑ 新 thread 不共享 Alice 的记忆")

    # 检查点检查
    snap = app.get_state(cfg)
    print(f"\n[get_state] Alice 线程当前状态: turns={snap.values.get('turns')}, "
          f"消息数={len(snap.values.get('messages', []))}")

    print(
        "\n要点：\n"
        "  1. 记忆 = checkpointer + thread_id，不是模型自己\"记住\"的。\n"
        "  2. MemorySaver 只在内存；生产用 SqliteSaver/PostgresSaver。\n"
        "  3. get_state() 随时读检查点；历史快照可回溯（time travel）。\n"
        "  4. create_agent(model, tools, checkpointer=...) 同样适用。"
    )
