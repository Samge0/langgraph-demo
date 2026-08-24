"""Demo 08 —— Send API（map-reduce / 动态并行）。

固定边和条件边都只能路由到【事先声明】的节点；
Send(value, state) 则把"节点名"变成运行时可计算的东西，还能为每个分支
准备【私有状态】—— 这就是 map-reduce / 动态并行分支的实现基础。

典型场景：
  - 评论审核：N 条评论各自起一个审核分支（各自的私有 state）
  - map-reduce：把长任务拆成子任务并行处理，再汇总

关键点：
- Send("node_name", private_state)：private_state 是该分支独享的输入。
- 各分支的输出仍会经过 reducer 汇入共享状态（这里用 operator.add 拼接）。
- 分支数在运行前不必知道 —— 由 state 内容动态决定。
"""
import operator
from typing import Annotated, TypedDict

from common import banner
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send


class SharedState(TypedDict):
    comments: list                     # 输入：待审核评论
    verdicts: Annotated[list, operator.add]   # 输出：各分支结论（拼接合并）


class WorkerState(TypedDict):         # 分支私有状态
    comment: str


def fan_out(state: SharedState) -> list[Send]:
    """map：为每条评论生成一个 Send 分支。"""
    sends = [Send("moderate", {"comment": c}) for c in state["comments"]]
    print(f"  [fan_out] 分出 {len(sends)} 个并行分支")
    return sends


def moderate(worker: WorkerState) -> dict:
    c = worker["comment"]
    toxic = any(w in c for w in ("垃圾", "滚", "蠢"))
    verdict = f"{'违规' if toxic else '通过'}: {c}"
    print(f"  [moderate] {verdict}")
    return {"verdicts": [verdict]}


builder = StateGraph(SharedState)
builder.add_node("moderate", moderate)
builder.add_conditional_edges(START, fan_out, ["moderate"])  # 入口即扇出
builder.add_edge("moderate", END)
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 08 Send：动态并行（map-reduce）")
    result = app.invoke({
        "comments": ["这个视频太棒了", "垃圾内容，骗点击", "学到了不少", "博主真蠢"]
    })
    print("\n汇总结论:")
    for v in result["verdicts"]:
        print(f"  - {v}")
    print(
        "\n要点：\n"
        "  1. Send(节点名, 私有状态) = 运行时动态生成分支 + 各自独立输入。\n"
        "  2. 分支写回共享键必须有 reducer（operator.add），否则并行写冲突。\n"
        "  3. 分支数量不需要提前声明 —— map-reduce / 子图编排都靠它。"
    )
