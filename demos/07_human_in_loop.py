"""Demo 07 —— Human-in-the-loop：interrupt() 暂停图，等人类批准后恢复。

高危操作（删除/付款/发邮件）让 Agent 直接执行很危险。
LangGraph 的方案：在节点里调用 interrupt(payload) —— 图立刻暂停，
状态（含所有消息）存入检查点；把 payload 展示给人类；
人类决定后用 Command(resume=...) 恢复执行，interrupt() 处拿到返回值。

关键点：
- interrupt() 必须搭配 checkpointer + thread_id（暂停状态要存档）。
- Command(resume=value) 从断点【原样恢复】上下文继续跑。
- resume 的值会作为 interrupt() 的返回值传进节点。
- resuming 是"重新执行该节点"：interrupt 之前的代码会再跑一遍，
  所以 interrupt() 要放在节点内副作用发生之前。
"""
from typing import Annotated, TypedDict

from common import banner
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command, interrupt


class State(TypedDict):
    messages: Annotated[list, add_messages]
    order: dict
    result: str


def review_order(state: State) -> dict:
    order = state["order"]
    # ★ 图在这里暂停。payload 会出现在结果末尾的 __interrupt__ 里，展示给人类。
    decision = interrupt({
        "question": "是否批准以下退款？",
        "order": order,
        "hint": "批准请回复 approve，拒绝请回复 reject",
    })
    if decision == "approve":
        result = f"退款 {order['amount']} 元已执行（操作人已批准）"
    else:
        result = f"退款被拒绝：{order.get('reason', '人工否决')}"
    print(f"  [review_order] 决定: {decision} -> {result}")
    return {"result": result, "messages": [("ai", result)]}


builder = StateGraph(State)
builder.add_node("review_order", review_order)
builder.add_edge(START, "review_order")
builder.add_edge("review_order", END)
app = builder.compile(checkpointer=MemorySaver())

if __name__ == "__main__":
    banner("Demo 07 Human-in-the-loop：interrupt / Command(resume=...)")

    cfg = {"configurable": {"thread_id": "order-1001"}}

    print(">>> 提交退款申请: 88 元")
    result = app.invoke({"order": {"id": 1001, "amount": 88}, "messages": []}, cfg)

    # 图被暂停：结果里带 __interrupt__
    interrupts = result.get("__interrupt__", [])
    if interrupts:
        payload = interrupts[0].value
        print(f"\n⏸ 图已暂停，等待人工审批。Payload: {payload}")

        # ---- 模拟人类看到审批界面，点了"批准" ----
        print("\n>>> [人工] 批准退款")
        result = app.invoke(Command(resume="approve"), cfg)

    print(f"\n最终结果: {result['result']}")
    print(f"完整消息数: {len(result['messages'])}")

    print(
        "\n要点：\n"
        "  1. interrupt(payload) 暂停图；Command(resume=v) 恢复。\n"
        "  2. 必须有 checkpointer + thread_id，否则暂停状态无处安放。\n"
        "  3. 恢复时节点会重跑：interrupt() 前不要放不可重复的副作用。\n"
        "  4. 同一机制可实现 time-travel（从历史检查点重放/分叉）。"
    )
