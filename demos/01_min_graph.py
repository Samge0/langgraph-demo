"""Demo 01 —— 最小图：StateGraph / 节点 / 边 / compile。

LangGraph 的一切都从三件事开始：
  1. State        —— 用 TypedDict 定义共享状态（图里流动的"数据"）
  2. Node         —— 一个普通 Python 函数：接收 state，返回【部分】更新
  3. Edge         —— 描述节点之间的固定跳转关系

运行三部曲：build（StateGraph）-> compile()（编译成可执行图）-> invoke()（执行）。

关键点：
- 节点返回 dict 时只写"要更新的键"，LangGraph 默认按"整键覆盖"合并进状态。
- START / END 是两个保留节点：START 是入口，END 是终点。
- compile() 之后图不可再修改；同一编译产物可以反复 invoke。
"""
from typing import TypedDict

from common import banner
from langgraph.graph import START, END, StateGraph


# ---------- 1. 定义状态：TypedDict 就是"图的 schema" ----------
class State(TypedDict):
    target: str            # 输入：要问候的名字
    greeting: str          # 中间产物：第一站生成
    final: str             # 最终输出：第二站加工


# ---------- 2. 定义节点：普通函数，收 state 返回部分更新 ----------
def make_greeting(state: State) -> dict:
    """节点1：生成问候语。注意返回值只包含要更新的键。"""
    greeting = f"你好，{state['target']}！"
    print(f"  [节点 make_greeting] 生成 -> {greeting}")
    return {"greeting": greeting}          # 只更新 greeting，不动其他键


def add_emoji(state: State) -> dict:
    """节点2：给问候语加上装饰。"""
    final = state["greeting"] + " (from LangGraph)"
    print(f"  [节点 add_emoji] 加工 -> {final}")
    return {"final": final}


# ---------- 3. 组装图 ----------
builder = StateGraph(State)
builder.add_node("make_greeting", make_greeting)
builder.add_node("add_emoji", add_emoji)

builder.add_edge(START, "make_greeting")   # 入口 -> 节点1
builder.add_edge("make_greeting", "add_emoji")  # 固定边：节点1 -> 节点2
builder.add_edge("add_emoji", END)         # 节点2 -> 终点

app = builder.compile()                    # 编译后才是可执行的图

if __name__ == "__main__":
    banner("Demo 01 最小图：State -> Node -> Edge -> compile")
    result = app.invoke({"target": "LangGraph 学习者"})   # 只需提供部分状态
    print(f"\n最终状态: {result}")
    print("注意：greeting 和 final 都保留在最终状态里 —— 状态是累积的。")
