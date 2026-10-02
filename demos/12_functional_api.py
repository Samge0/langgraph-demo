"""Demo 12 —— Functional API：@task + @entrypoint，用普通 Python 写工作流。

StateGraph 用"节点+边"描述拓扑；Functional API 反过来：
写【普通函数 + for/if/try】，LangGraph 在背后把执行过程记录成 checkpoint。

  @task        把函数变成可持久化的步骤（结果会被缓存，崩溃恢复后不重跑）
  @entrypoint  工作流入口，等价于 compiled graph，可以 .invoke()/.stream()

本 demo 场景：晨报流水线 fetch_news -> summarize -> compose，
然后故意让 summarize 崩一次，用同一个 thread_id 恢复：
  已完成的 fetch_news 直接从 checkpoint 恢复，不会重新抓取（调用记录可证）。

关键点：
- task() 返回 future，同步场景用 .result() 取值，异步用 await。
- entrypoint(checkpointer=...) + thread_id 是 durability 的开关：
  崩溃后 invoke(None, config) 恢复已有 thread，None 表示"接着上次跑"。
- task 结果缓存按【参数哈希】复用：同参数重跑 workflow，task 不再执行。
- 什么时候选它 vs StateGraph：编排逻辑复杂（嵌套循环/异常分支）时函数式更顺手；
  需要可视化拓扑、多入口路由时 StateGraph 更直观。
"""
import time

from common import banner
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.func import entrypoint, task

CALL_LOG = []                       # 记录每个 task 真实执行了几次
OUTAGE = {"on": True}               # 模拟下游服务故障，稍后"修复"


@task
def fetch_news(source: str) -> list[str]:
    """假装抓新闻（耗时/耗钱的操作，崩溃恢复后不应该重跑）。"""
    CALL_LOG.append(f"fetch({source})")
    time.sleep(0.05)
    return {
        "tech": ["LangGraph 1.2 发布", "新模型评测出炉"],
        "market": ["大盘收涨 1.2%", "新能源板块活跃"],
    }[source]


@task
def summarize(articles: list[str]) -> str:
    """假装摘要。OUTAGE 打开时抛错，模拟下游服务抖动。"""
    CALL_LOG.append(f"summarize({len(articles)} 篇)")
    if OUTAGE["on"]:
        raise RuntimeError("摘要服务 503（模拟瞬时故障）")
    return "；".join(a[:12] for a in articles) + " —— 一句话总结"


@task
def compose(sections: list[str]) -> str:
    CALL_LOG.append("compose()")
    return "【晨报】\n" + "\n".join(f"- {s}" for s in sections)


@entrypoint(checkpointer=InMemorySaver())
def morning_report(topic: str) -> str:
    """普通 Python：循环抓取 -> 逐段摘要 -> 拼装。"""
    sections = []
    for source in ("tech", "market"):
        articles = fetch_news(source).result()      # task 调用，.result() 取值
        sections.append(summarize(articles).result())
    return compose(sections).result()


if __name__ == "__main__":
    banner("Demo 12 Functional API：@task/@entrypoint + 崩溃恢复")
    cfg = {"configurable": {"thread_id": "morning-001"}}

    print("\n-- 第一次运行：summarize 撞上模拟故障 --")
    try:
        morning_report.invoke("AI", cfg)
    except RuntimeError as e:
        print(f"  崩溃：{e}")
    print(f"  实际执行过的步骤: {CALL_LOG}")

    print("\n-- 故障修复后，恢复同一个 thread --")
    OUTAGE["on"] = False
    report = morning_report.invoke(None, cfg)        # ★ None = 恢复，不从头开始
    print(f"  {report}")
    print(f"  全部执行记录: {CALL_LOG}")

    assert CALL_LOG.count("fetch(tech)") == 1 and CALL_LOG.count("fetch(market)") == 1, \
        "fetch 不应该重跑！"
    print(
        "\n要点：\n"
        "  1. fetch(tech)/fetch(market) 只执行了一次 —— 崩溃恢复后从 checkpoint 拿缓存。\n"
        "  2. invoke(None, cfg) 恢复已有 thread；换成重新 invoke 输入则等于重跑。\n"
        "  3. 生产里 checkpointer 换 SqliteSaver/PostgresSaver，跨进程也能恢复。"
    )
