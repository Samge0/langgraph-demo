"""Demo 09 —— 子图（subgraph）与多 Agent 协作。

单个图越长越难维护；LangGraph 允许把"一个编译好的图"当作【一个节点】
挂进另一个图 —— 这就是子图，也是多 Agent 架构的乐高积木。

两种衔接方式：
  A. 共享 state schema：子图直接读写父图状态（本 demo 用这种）
  B. 状态转换：父图节点里手动 invoke 子图并做字段映射

本 demo：一个"主管（supervisor）"图调度两个子图 Agent：
  - research_subgraph：负责查资料（内置自己的循环结构）
  - write_subgraph：负责成稿
主管按任务类型路由，这就是 supervisor 模式的最小实现。

关键点：
- 子图是一个 compile() 过的图，add_node("name", subgraph_app) 直接挂入。
- 子图可以有自己的 checkpointer（也可以没有）。
- langgraph 官方 supervisor 库（langgraph-supervisor 包）本质就是这套模式 +
  LLM 决定下一步路由谁。
"""
from typing import Literal, TypedDict

from common import banner
from langgraph.graph import END, START, StateGraph


# ---------- 父图共享状态 ----------
class TeamState(TypedDict):
    task: str
    research: str
    report: str


# ---------- 子图 1：research（内部有一个两步流水线） ----------
class ResearchState(TypedDict):
    task: str
    research: str


def collect(state: ResearchState) -> dict:
    findings = f"关于'{state['task']}'的 3 条要点：要点A；要点B；要点C"
    print(f"  [子图research.collect] {findings}")
    return {"research": findings}


def summarize(state: ResearchState) -> dict:
    s = state["research"] + "（已提炼）"
    print(f"  [子图research.summarize] 完成")
    return {"research": s}


research_builder = StateGraph(ResearchState)
research_builder.add_node("collect", collect)
research_builder.add_node("summarize", summarize)
research_builder.add_edge(START, "collect")
research_builder.add_edge("collect", "summarize")
research_builder.add_edge("summarize", END)
research_agent = research_builder.compile()      # ★ 编译后的图 = 一个可挂载节点


# ---------- 子图 2：write ----------
class WriteState(TypedDict):
    research: str
    report: str


def write_report(state: WriteState) -> dict:
    report = f"# 报告\n\n{state['research']}\n\n（结论：一切都好）"
    print(f"  [子图write.write_report] 报告已生成")
    return {"report": report}


write_builder = StateGraph(WriteState)
write_builder.add_node("write_report", write_report)
write_builder.add_edge(START, "write_report")
write_builder.add_edge("write_report", END)
write_agent = write_builder.compile()


# ---------- 父图：supervisor 调度 ----------
def route(state: TeamState) -> Literal["research_agent", "write_agent", "__end__"]:
    if not state.get("research"):
        return "research_agent"
    if not state.get("report"):
        return "write_agent"
    return END


builder = StateGraph(TeamState)
builder.add_node("research_agent", research_agent)   # ★ 子图当节点
builder.add_node("write_agent", write_agent)
builder.add_edge(START, "research_agent")
builder.add_conditional_edges("research_agent", route)
builder.add_conditional_edges("write_agent", route)
team = builder.compile()

if __name__ == "__main__":
    banner("Demo 09 子图与多 Agent：supervisor 调度 research + write")
    result = team.invoke({"task": "LangGraph 落地实践"})
    print(f"\nresearch: {result['research'][:40]}...")
    print(f"report  : {result['report'][:40]}...")

    print(
        "\n要点：\n"
        "  1. compile() 后的图可以整个作为一个节点挂进父图。\n"
        "  2. 共享 schema 时子图直接读写父状态；不同 schema 则在节点里手动转换。\n"
        "  3. supervisor 模式 = 主管节点/条件边决定下一个干活的子图 Agent。\n"
        "  4. 官方 langgraph-supervisor / langgraph-swarm 包是这个模式的工程化版本。"
    )
