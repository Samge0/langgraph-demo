"""Demo 11 —— Command 导航：节点自己决定"下一步去哪"（LangGraph 1.x 主推写法）。

Demo 03 用 add_conditional_edges：路由函数在节点【外面】看 state 返回节点名。
LangGraph 1.x 提供了更内聚的方式：节点直接返回
    Command(goto="目标节点", update={...})
导航 + 写状态一步完成，路由逻辑跟着业务代码走。

本 demo 场景：客服工单自动分流。
  classify(LLM 分类) --Command(goto=...)--> refund / tech / escalate 三条路
  escalate 还会演示 Command 的 goto 列表：同时通知人工 + 记录台账（并行扇出）。

关键点：
- Command 从 langgraph.types 导入（不是 langgraph.graph）。
- 返回 Command 的节点不能再被 add_conditional_edges 双重导航（两个指挥官会打架）。
- goto 传列表 = 并行扇出，目标字段必须是"整个键覆盖"安全的状态（配合 reducer）。
- interrupt() 的恢复值也是用 Command(resume=...) 传的（demo 07），一家人。
"""
import operator
from typing import Annotated, Literal, TypedDict

from common import banner, get_llm
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command


class State(TypedDict):
    ticket: str
    category: str
    # 两条并行支路都要写它，必须用 reducer 合并（见 demo 02）
    log: Annotated[list[str], operator.add]
    reply: str


def _classify_llm(text: str) -> str:
    """真模型可用时让 LLM 分类；离线时关键词规则兜底，输出确定。"""
    keyword_rule = (
        "refund" if ("退款" in text or "退货" in text)
        else "tech" if ("报错" in text or "登录" in text or "崩" in text)
        else "escalate"
    )
    llm = get_llm(responses=[AIMessage(content=keyword_rule)])
    answer = llm.invoke(
        "把工单分类为 refund/tech/escalate，只输出单词本身：\n" + text
    ).content.strip().lower()
    return answer if answer in ("refund", "tech", "escalate") else "escalate"


# ---------- 节点：返回 Command，自己决定下一站 ----------
def classify(state: State) -> Command[Literal["refund", "tech", "escalate"]]:
    cat = _classify_llm(state["ticket"])
    print(f"  [classify] '{state['ticket']}' -> {cat}")
    # ★ 导航 + 写状态一步完成，不再需要外面的路由函数
    return Command(goto=cat, update={"category": cat, "log": [f"classified:{cat}"]})


def refund(state: State) -> dict:
    reply = "已登记退款申请，1-3 个工作日原路退回。（退款组）"
    print(f"  [refund] {reply}")
    return {"reply": reply, "log": ["handled:refund"]}


def tech(state: State) -> dict:
    reply = "已生成工单并附上诊断日志，转技术支持排查。（技术组）"
    print(f"  [tech] {reply}")
    return {"reply": reply, "log": ["handled:tech"]}


def escalate(state: State) -> dict:
    # ★ goto 传列表 = 并行扇出到两个节点（log 有 reducer，安全）
    print("  [escalate] 复杂工单 -> 同时通知人工 + 写台账")
    return Command(
        goto=["human_agent", "audit"],
        update={"log": ["escalated"]},
    )


def human_agent(state: State) -> dict:
    print("  [human_agent] 已打电话通知值班同学")
    return {"reply": "工单复杂，已升级人工跟进，2 小时内联系您。", "log": ["notified:human"]}


def audit(state: State) -> dict:
    print("  [audit] 台账已记录")
    return {"log": ["audit:written"]}


builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("refund", refund)
builder.add_node("tech", tech)
builder.add_node("escalate", escalate)
builder.add_node("human_agent", human_agent)
builder.add_node("audit", audit)

builder.add_edge(START, "classify")
# 注意：classify 之后【没有】add_conditional_edges —— 导航权在 Command 手里
for leaf in ("refund", "tech", "human_agent", "audit"):
    builder.add_edge(leaf, END)
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 11 Command 导航：节点自己说'下一步去哪'")
    for ticket in ["我要退款，昨天买的东西不要了", "APP 一登录就报错闪退", "你们和合作方的合同条款有争议"]:
        print(f"\n>>> 工单: {ticket}")
        result = app.invoke({"ticket": ticket, "log": []})
        print(f"<<< 回复: {result['reply']}")
        print(f"    日志: {result['log']}")

    print(
        "\n要点：\n"
        "  1. Command(goto=, update=) 让'决策'和'写状态'在同一个函数里完成。\n"
        "  2. goto 列表 = 并行扇出；被多节点同时写的字段需要 reducer。\n"
        "  3. 与条件边二选一即可，别对同一节点同时用两种导航。"
    )
