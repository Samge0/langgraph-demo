"""Demo 03 —— 条件边：让图拥有"分支逻辑"，Agent 的路由就靠它。

add_conditional_edges(来源节点, 路由函数, [可选的目标节点列表])
路由函数收 state，返回【下一节点的名字】（或一组名字）。

本 demo 演示一个经典场景：情感分析路由。
  用户输入 -> classify(分类) -> 按情绪走不同处理节点 -> 汇总 -> END

关键点：
- 路由函数只返回"节点名"，它本身不修改状态；要带数据流转，把信息写进 state 再返回名字。
- 返回多个名字 = 并行扇出（配合 reducer 才安全，见 demo02）。
- 也可以从 START 接条件边（入口分流），或从同一节点接多条不同条件。
"""
from typing import Literal, TypedDict

from common import banner, get_llm
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    user_text: str
    sentiment: str
    reply: str


# ---------- 节点 ----------
def classify(state: State) -> dict:
    text = state["user_text"]
    # 真模型可用时让 LLM 分类；离线时用极简规则兜底
    llm = get_llm(responses=[AIMessage(content="positive" if "好" in text or "棒" in text else "negative")])
    answer = llm.invoke(
        f"判断下面这句话的情绪，只输出 positive 或 negative：\n{text}"
    ).content.strip().lower()
    sentiment = "positive" if "pos" in answer else "negative"
    print(f"  [classify] '{text}' -> {sentiment}")
    return {"sentiment": sentiment}


def reply_happy(state: State) -> dict:
    msg = f"太好了！听到这个消息很开心 :-D  ({state['user_text']})"
    print(f"  [reply_happy] {msg}")
    return {"reply": msg}


def reply_sad(state: State) -> dict:
    msg = f"别难过，事情会好起来的 :-)  ({state['user_text']})"
    print(f"  [reply_sad] {msg}")
    return {"reply": msg}


# ---------- 路由函数：返回节点名 ----------
def route_by_sentiment(state: State) -> Literal["reply_happy", "reply_sad"]:
    if state["sentiment"] == "positive":
        return "reply_happy"
    return "reply_sad"


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("reply_happy", reply_happy)
builder.add_node("reply_sad", reply_sad)

builder.add_edge(START, "classify")
# 条件边：classify 之后走哪条路，由 route_by_sentiment 说了算
builder.add_conditional_edges("classify", route_by_sentiment, ["reply_happy", "reply_sad"])
builder.add_edge("reply_happy", END)
builder.add_edge("reply_sad", END)
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 03 条件边：情感路由")
    for text in ["今天天气真好，心情很棒！", "这个 bug 折腾了我一整天……"]:
        print(f"\n>>> 输入: {text}")
        result = app.invoke({"user_text": text})
        print(f"<<< 回复: {result['reply']}")
