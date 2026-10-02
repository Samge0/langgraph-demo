"""Demo 17 —— Swarm 多智能体：没有主管的点对点转接（handoff 模式）。

Demo 09 的 supervisor 是"经理制"：所有活儿先汇总给主管，主管再派单。
Swarm 是"同事制"：Agent 之间直接转接，谁接得住谁处理，扁平、少一跳。

LangGraph 1.x 里 handoff 的实现配方：
  1. 每个 Agent 是一个节点，共享同一个 messages 状态（互相看得见聊天记录）；
  2. "转接"被建模成一个工具：handoff(target, note)；
  3. LLM 调 handoff 时，节点返回 Command(goto=target)（Demo 11 的知识），
     并把交接备注写进消息 —— 智能体间通信 = 消息追加，不是魔法。

本 demo 场景：前台 -> 技术支持（可来回转），处理"登录报错"工单。
离线时 Fake 模型回放：前台识别为技术问题 -> handoff(tech) -> 技术直接解决。

关键点：
- goto 的目标是【动态】的（来自工具参数），这正是 Command 相对条件边的优势：
  路由决策可以由 LLM 的工具调用实时决定。
- 消息历史全程共享；交接备注让接手者不用用户重复描述。
- 防环：转接计数超过 N 次强制 END（小模型会互相踢皮球，真实踩坑）。
"""
import operator
from typing import Annotated, Literal, TypedDict

from common import banner, get_llm
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command

MAX_HANDOFFS = 3          # 防踢皮球保险丝


class State(TypedDict):
    messages: Annotated[list, add_messages]
    handoff_count: Annotated[int, operator.add]   # 计数用 add reducer，转一次 +1


# ---------- 离线脚本：前台转技术，技术直接解答 ----------
FRONT_SCRIPT = [
    AIMessage(content="", tool_calls=[{
        "name": "handoff", "id": "h1", "type": "tool_call",
        "args": {"target": "tech", "note": "用户登录报错，涉及鉴权模块，需技术排查"},
    }]),
]
TECH_SCRIPT = [
    AIMessage(content="已定位：登录报错是 token 过期未刷新导致。请退出重登即可；仍报错的话清一次缓存。"),
]
BILLING_SCRIPT = [
    AIMessage(content="账单问题：本月扣费记录已核对，无重复扣款，详情已发您邮箱。"),
]

AGENTS: dict[str, list] = {
    "front": FRONT_SCRIPT,
    "tech": TECH_SCRIPT,
    "billing": BILLING_SCRIPT,
}


def call_agent(agent_name: str, script: list):
    """生成一个 swarm 节点：LLM 回复，或调 handoff 工具触发 Command 转接。"""
    llm = get_llm(responses=script)
    llm_with_tools = llm.bind_tools([handoff])       # 每个智能体都会"转接"这个动作

    def node(state: State) -> Command[Literal["front", "tech", "billing"]] | dict:
        response = llm_with_tools.invoke(state["messages"])
        if not response.tool_calls:                  # 正常作答 -> 结束
            print(f"  [{agent_name}] 回复: {response.content[:38]}…")
            return {"messages": [response]}

        tc = response.tool_calls[0]
        assert tc["name"] == "handoff"
        target, note = tc["args"]["target"], tc["args"]["note"]
        if target not in AGENTS:
            target = "front"
        # ★ 动态路由：目标节点来自 LLM 的工具参数；交接备注作为消息传给接手者
        if state["handoff_count"] + 1 >= MAX_HANDOFFS:
            print(f"  [{agent_name}] 转接次数达上限，强制收尾")
            return {"messages": [
                response,
                ToolMessage(content="转接次数达上限，请直接作答", tool_call_id=tc["id"], name="handoff"),
                AIMessage(content="多智能体协作已达转接上限，本次由人工跟进。"),
            ]}
        print(f"  [{agent_name}] handoff -> {target}（备注: {note[:20]}…）")
        return Command(
            goto=target,
            update={
                "messages": [
                    response,
                    ToolMessage(content=f"已转接给 {target}", tool_call_id=tc["id"], name="handoff"),
                    AIMessage(content=f"[交接自 {agent_name}] {note}"),
                ],
                "handoff_count": 1,
            },
        )

    return node


@tool
def handoff(target: str, note: str) -> str:
    """把当前对话转接给另一个智能体。target：front/tech/billing；note：交接备注。"""
    return "ok"


builder = StateGraph(State)
for name, script in AGENTS.items():
    builder.add_node(name, call_agent(name, script))
builder.add_edge(START, "front")
# 各智能体节点【不接出边】—— 出口完全由 Command(goto=) 动态决定
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 17 Swarm：Agent 之间点对点转接（handoff）")
    result = app.invoke(
        {"messages": [("user", "APP 登录一直报错 401，密码肯定没错")]},
        config={"recursion_limit": 15},
    )

    print("\n--- 消息轨迹 ---")
    for m in result["messages"]:
        if m.type == "ai" and m.tool_calls:
            tc = m.tool_calls[0]
            print(f"  AI   -> handoff({tc['args']['target']})")
        elif m.type == "tool":
            print(f"  tool-> {m.content}")
        else:
            print(f"  {m.type:5s}-> {str(m.content)[:44]}")

    print(
        f"\n要点：\n"
        f"  1. 无主管：front -> tech 直达，比 supervisor 少一跳派单。\n"
        f"  2. goto 的值来自工具参数 —— LLM 实时决定路由，这是 Command 的杀手锏。\n"
        f"  3. 转接上限 {MAX_HANDOFFS} 是必须的保险丝（小模型会互相踢皮球）。\n"
        f"  4. 官方 langgraph-swarm 库（未安装）即此模式的封装。"
    )
