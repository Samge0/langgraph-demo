"""Demo 15 —— Agentic RAG：带纠错环的检索增强生成（CRAG 模式）。

朴素 RAG 是一条直线：检索 -> 塞进 prompt -> 生成。检索到垃圾也硬着头皮答。
Agentic RAG 把检索变成【图里的环】，让 Agent 自己把关：

  retrieve(检索) -> grade(相关吗?)
       |-- 相关 -> generate(生成) -> END
       |-- 不相关 -> rewrite(改写查询) -> 再 retrieve（环）

这就是 Self-RAG / CRAG 的骨架：检索质量不行就换问法再检，而不是幻觉作答。

玩具检索器：关键词重叠打分（不依赖 embedding，离线可跑）。
真模型时 grade/rewrite/generate 由 LLM 完成；离线时脚本回放同样走完整个环。

关键点：
- 环必须有出口：hops 计数器防止无限改写（生产再配 recursion_limit）。
- grade 输出结构化二值（"yes"/"no"），条件边据此路由 —— 复习 demo 03/13。
- 把关键词检索换成向量库 + 把 grade 换成 LLM few-shot 判断，就是生产形态。
"""
from typing import TypedDict

from common import banner, get_llm
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph

# ---------- 玩具知识库 ----------
DOCS = {
    "d1": "LangGraph 的 checkpointer 负责保存每一步状态，用于多轮记忆和故障恢复。",
    "d2": "乌龙茶属于半发酵茶，产于福建、广东和台湾。",
    "d3": "LangGraph 的 Send API 可以在运行时动态扇出并行任务，做 map-reduce。",
    "d4": "Python 的 GIL 使得单进程多线程无法并行执行 CPU 密集代码。",
}


def _keyword_search(query: str, top_k: int = 2) -> list[tuple[str, str]]:
    """玩具检索器：按字符 bigram 重叠打分。生产换成向量检索。"""
    def bigrams(s: str) -> set[str]:
        s = "".join(s.split())
        return {s[i:i + 2] for i in range(len(s) - 1)}

    q = bigrams(query)
    scored = sorted(
        ((len(q & bigrams(text)), doc_id) for doc_id, text in DOCS.items()),
        reverse=True,
    )
    return [(doc_id, DOCS[doc_id]) for score, doc_id in scored[:top_k] if score > 0]


class State(TypedDict):
    question: str
    query: str                 # 当前检索词（可能被改写过）
    docs: list
    verdict: str               # 本轮检索的相关性裁决：yes / no
    hops: int                  # 改写次数，环的保险丝
    answer: str


# ---------- 离线脚本：第一轮检索不相关 -> 改写 -> 第二轮命中 ----------
OFFLINE = {
    "grade": ["no", "yes"],     # 第 1 轮不相关，第 2 轮相关
    "rewrite": ["LangGraph checkpoint 状态持久化 作用"],
    "generate": ["根据文档：checkpointer 保存图每一步的状态，用于多轮记忆与故障恢复。"],
}


def retrieve(state: State) -> dict:
    docs = _keyword_search(state["query"])
    print(f"  [retrieve] 查询='{state['query']}' -> {[(d[0], d[1][:18] + '…') for d in docs] or '空'}")
    return {"docs": docs}


def grade(state: State) -> dict:
    hits = len(state["docs"])
    # 真模型时让 LLM 判断文档能否回答问题；离线时规则：命中<2 条视为不相关（首问会触发改写）
    rule = "yes" if hits >= 2 else "no"
    llm = get_llm(responses=[AIMessage(content=OFFLINE["grade"][min(state["hops"], 1)])])
    verdict = llm.invoke(
        f"文档能否回答问题？只输出 yes/no。\n问题：{state['question']}\n命中：{hits} 条"
    ).content.strip().lower()
    verdict = verdict if verdict in ("yes", "no") else rule
    print(f"  [grade] 命中 {hits} 条 -> {verdict}")
    return {"verdict": verdict, "hops": state["hops"] + (1 if verdict == "no" else 0)}


def route_after_grade(state: State) -> str:
    if state["verdict"] == "yes" or state["hops"] >= 2:   # 相关，或保险丝烧断
        return "generate"
    return "rewrite"


def rewrite(state: State) -> dict:
    llm = get_llm(responses=[AIMessage(content=OFFLINE["rewrite"][0])])
    new_query = llm.invoke(
        f"原始问题'{state['question']}'检索效果差，改写成一个信息量更大的检索查询，只输出查询本身。"
    ).content.strip()
    print(f"  [rewrite] '{state['query']}' -> '{new_query}'")
    return {"query": new_query}


def generate(state: State) -> dict:
    if not state["docs"]:       # 保险丝烧断仍无文档：诚实说不知道，而不是编造
        print("  [generate] 知识库无相关文档 -> 诚实拒答")
        return {"answer": "知识库里没有足够相关的资料，无法可靠回答该问题。"}
    context = "\n".join(f"[{d[0]}] {d[1]}" for d in state["docs"])
    llm = get_llm(responses=[AIMessage(content=OFFLINE["generate"][0])])
    answer = llm.invoke(
        f"仅依据以下文档回答问题，不要编造。\n文档：\n{context}\n\n问题：{state['question']}"
    ).content
    print(f"  [generate] 依据 {len(state['docs'])} 条文档作答")
    return {"answer": answer}


builder = StateGraph(State)
builder.add_node("retrieve", retrieve)
builder.add_node("grade", grade)
builder.add_node("rewrite", rewrite)
builder.add_node("generate", generate)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route_after_grade, ["rewrite", "generate"])
builder.add_edge("rewrite", "retrieve")          # ★ 纠错环：改写后重新检索
builder.add_edge("generate", END)
app = builder.compile()

if __name__ == "__main__":
    banner("Demo 15 Agentic RAG：检索不相关就改写再检（CRAG 纠错环）")
    print("\n>>> 问题: checkpoint 有什么用？")
    # 首问口语化（"checkpoint 有什么用"），玩具检索命中不足 -> 触发改写环
    result = app.invoke({"question": "checkpoint 有什么用？", "query": "checkpoint 有什么用", "hops": 0})
    print(f"\n<<< 回答: {result['answer']}")
    print(f"    共改写 {result['hops']} 次（首查不相关 -> 换个说法 -> 命中）")
    print(
        "\n要点：\n"
        "  1. grade -> rewrite -> retrieve 的环是 Agentic RAG 的灵魂：质量自检 + 自纠。\n"
        "  2. hops/上限是保险丝；生产上再配 recursion_limit（默认 25）双保险。\n"
        "  3. 生产形态：向量库检索 + LLM 评分 + 检索不到就转 web_search 的 fallback 分支。"
    )
