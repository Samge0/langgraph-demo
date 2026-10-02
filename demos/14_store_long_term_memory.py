"""Demo 14 —— BaseStore 长期记忆：跨会话的用户偏好。

Demo 06 的 checkpointer 是【短期记忆】：绑在 thread 上，换一个 thread_id
就什么都不记得了。真实产品还需要【长期记忆】：昨天说过的偏好，今天新开会话
仍然生效 —— 这就是 BaseStore 的角色。

  checkpointer:  每个会话（thread）的状态快照      -> "我们聊过什么"
  store:         跨会话、按命名空间组织的键值数据库 -> "用户是谁、喜欢什么"

用法三步：
  1. 建仓：store = InMemoryStore()（生产换 PostgresSaver/RedisStore）
  2. 挂图：app = builder.compile(store=store)
  3. 节点里取：from langgraph.config import get_store; get_store().put/search

命名空间是元组 ("维度", "键")，比如 ("memories", user_id) ——
多租户隔离靠它，别把所有用户的记忆塞一个桶里。

本 demo：thread-1 用户随口说了口味偏好 -> 写入 store；
thread-2（全新会话，checkpointer 视角是陌生人）依然能读出偏好做个性化推荐。

关键点：
- put(namespace, key, value) 是整键覆盖；search(namespace) 返回条目列表。
- store 与 checkpointer 独立配置，互不影响。
- 什么时候写记忆、写什么（抽取式 vs 原文式）是产品设计问题，图只提供机制。
"""
from common import banner, get_llm
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_store
from langgraph.graph import END, START, StateGraph
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict


class State(TypedDict):
    user_id: str
    messages: list          # 本 demo 手动传消息，不用 add_messages，保持简单
    memory_saved: str
    reply: str


def _reply_llm(text: str) -> str:
    """真模型时生成回复，离线时回放脚本文本（内容依据 store 里是否有记忆生成）。"""
    llm = get_llm(responses=[AIMessage(content=text)])
    return llm.invoke(f"参考以下信息，生成一句面向用户的中文回复：\n{text}").content


# ---------- 节点 1：判断是否偏好陈述，是才写入 store ----------
PREF_CUES = ("喜欢", "偏爱", "不要", "不喝", "不爱", "口味", "习惯")


def save_memory(state: State) -> dict:
    store = get_store()                                  # 节点内拿 store
    ns = ("memories", state["user_id"])                  # ★ 命名空间 = (维度, 用户ID)
    last = state["messages"][-1]["content"]
    # 何时写记忆是产品设计问题：生产里用 LLM 判断"这句话值得长期记住吗"，
    # demo 用关键词启发。普通提问不写 —— 否则新会话会把旧偏好覆盖掉（真实踩坑）。
    if not any(cue in last for cue in PREF_CUES):
        print(f"  [save_memory] '{last[:16]}…' 不是偏好陈述 -> 不写 store")
        return {"memory_saved": ""}
    store.put(ns, "taste", {"text": last})
    print(f"  [save_memory] 已写入 store: ns={ns}")
    return {"memory_saved": last}


# ---------- 节点 2：读 store，个性化回复 ----------
def respond(state: State) -> dict:
    store = get_store()
    items = store.search(("memories", state["user_id"]))
    if items:
        prefs = "；".join(f"{i.key}: {i.value['text']}" for i in items)
        reply = _reply_llm(f"记得您的偏好（{prefs}），推荐：冻顶乌龙（无糖）。")
        print(f"  [respond] 从 store 找到 {len(items)} 条记忆: {prefs}")
    else:
        reply = _reply_llm("还没有您的偏好记录，默认推荐：拿铁。")
        print("  [respond] store 里没有该用户的记忆 -> 用默认推荐")
    return {"reply": reply}


builder = StateGraph(State)
builder.add_node("save_memory", save_memory)
builder.add_node("respond", respond)
builder.add_edge(START, "save_memory")
builder.add_edge("save_memory", "respond")
builder.add_edge("respond", END)

store = InMemoryStore()
checkpointer = InMemorySaver()
app = builder.compile(store=store, checkpointer=checkpointer)

if __name__ == "__main__":
    banner("Demo 14 BaseStore：跨会话长期记忆（checkpointer 做不到的事）")

    print("\n== 会话 1（thread-001）：用户随口提到偏好 ==")
    app.invoke(
        {"user_id": "u_42", "messages": [{"role": "user", "content": "我偏爱乌龙茶，不喝甜的"}]},
        config={"configurable": {"thread_id": "thread-001"}},
    )

    print("\n== 会话 2（thread-002）：全新会话，问推荐 ==")
    result = app.invoke(
        {"user_id": "u_42", "messages": [{"role": "user", "content": "给我推荐一杯"}]},
        config={"configurable": {"thread_id": "thread-002"}},
    )
    print(f"\n<<< 回复: {result['reply']}")

    print("\n== 换个没打过交道的用户（thread-003）==")
    result = app.invoke(
        {"user_id": "u_99", "messages": [{"role": "user", "content": "给我推荐一杯"}]},
        config={"configurable": {"thread_id": "thread-003"}},
    )
    print(f"<<< 回复: {result['reply']}")

    print(
        "\n要点：\n"
        "  1. thread-002 对 checkpointer 来说是全新对话，但 store 里有 u_42 的记忆。\n"
        "  2. 命名空间 ('memories', user_id) 做多租户隔离；u_99 查不到 -> 走默认。\n"
        "  3. 生产栈：InMemoryStore 换 PostgresStore，配合语义检索做'向量记忆'。"
    )
