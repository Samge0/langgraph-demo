"""Demo 10 —— 流式输出 stream() 与 token 级流式 stream_mode="messages"。

真实产品的体验关键：边生成边展示。LangGraph 的 stream() 支持 5 种模式：

  stream_mode="values"    每步输出【完整状态】（默认之一）
  stream_mode="updates"   每步输出【增量更新】{节点名: 更新dict}
  stream_mode="custom"    节点里 get_stream_writer()("...") 自定义事件
  stream_mode="debug"     全量调试事件（含任务/状态快照）
  stream_mode="messages"  ★ LLM 的 token 级流式 + 工具调用中间态

关键点：
- 多模式可组合：stream_mode=["updates", "messages"]，事件变成 (mode, chunk) 元组。
- "messages" 模式产出 (message_chunk, metadata)，metadata 含 langgraph_node
  —— 前端据此区分"这是哪个节点在说话"。
- 图未变，只是换一种消费输出的方式。
"""
from typing import Annotated, TypedDict

from common import banner, get_llm
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


def step_one(state: State) -> dict:
    print("  [step_one] 执行")
    return {"messages": [("ai", "第一步完成")]}


def step_two(state: State) -> dict:
    # 离线假模型不支持 token 流式，这里直接产出一条消息；
    # 真模型（ChatOpenAI）在 messages 模式下会自动 token 级流式。
    llm = get_llm()
    resp = llm.invoke([HumanMessage(content="写一句关于坚持的鸡汤")]
                      if not state["messages"] else state["messages"])
    return {"messages": [AIMessage(content=resp.content)]}


builder = StateGraph(State)
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)
builder.add_edge(START, "step_one")
builder.add_edge("step_one", "step_two")
builder.add_edge("step_two", END)
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 10 流式输出：updates / messages 等模式")

    print("--- 1) stream_mode='updates'：每步的增量 ---")
    for chunk in app.stream({"messages": [("user", "讲讲坚持")]},
                            stream_mode="updates"):
        print(f"  updates: { {k: '...msgs' for k in chunk} }")

    print("\n--- 2) stream_mode='values'：每步的完整状态 ---")
    for i, snap in enumerate(app.stream({"messages": [("user", "讲讲坚持")]},
                                         stream_mode="values")):
        print(f"  第{i}步状态消息数: {len(snap['messages'])}")

    print("\n--- 3) stream_mode='messages'：token 级（真模型时逐字输出）---")
    for msg_chunk, metadata in app.stream({"messages": [("user", "讲讲坚持")]},
                                           stream_mode="messages"):
        node = metadata.get("langgraph_node", "?")
        if msg_chunk.content:
            print(f"  [{node}] {msg_chunk.content}", end="", flush=True)
    print()

    print(
        "\n\n要点：\n"
        "  1. invoke -> 等全量；stream -> 边跑边拿。\n"
        "  2. updates 适合做进度条/日志；messages 适合聊天 UI 的打字机效果。\n"
        "  3. 组合模式 stream_mode=['updates','messages'] 返回 (mode, chunk) 元组。\n"
        "  4. custom 模式：节点内 get_stream_writer() 主动吐自定义事件。"
    )
