"""Demo 18 —— 生产容错三件套：重试 / 不重试 / 兜底（RetryPolicy 实战）。

LLM 应用上生产，节点一定会以各种姿势失败。LangGraph 1.x 的容错配置挂在
【节点级】（add_node 的参数），不是图级：

  add_node("name", fn,
           retry_policy=RetryPolicy(...),   # 瞬时故障：指数退避重试
           timeout=30)                      # 挂死保护（协作为主，同步阻塞要当心）

最容易踩的认知坑（本 demo 用执行记录证明）：
  默认的 retry_on 只重试【瞬时故障】（网络/5xx 等），
  ValueError/TypeError 这类确定性 bug 默认【不重试】——
  同样的输入重跑一百次还是炸，重试只是浪费钱和时间。
  要强制重试某类异常，显式传 retry_on=ValueError。

本 demo 跑三张小图，对照三种行为：
  1. 网络抖动 + RetryPolicy        -> 退避重试后成功
  2. 确定性 bug + 默认策略          -> 1 次就抛（正确行为！）
  3. 确定性 bug + retry_on=ValueError -> 也被重试（强制覆盖默认）

外加一个环失控的兜底：GraphRecursionError 的友好处理（复习 recursion_limit）。

关键点：
- RetryPolicy(initial_interval, backoff_factor, max_attempts, jitter)：
  间隔 = initial * backoff^(次数-1)，加抖动防止惊群。
- 重试层次：ChatOpenAI 自己的 max_retries 管【单次 HTTP】，
  节点 RetryPolicy 管【整个节点】（含解析/组装逻辑），别混用。
- ValueError 不重试是【特性】不是 bug —— 它是你的代码错了，先修代码。
"""
import time

from common import banner
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy


class TransientError(Exception):
    """模拟网络抖动 / 下游 5xx —— 默认策略认为这类值得重试。"""


STATS = {"flaky_calls": 0, "bug_calls": 0}


def flaky_network(state: dict) -> dict:
    """前 2 次调用失败，第 3 次成功（模拟抖动恢复）。"""
    STATS["flaky_calls"] += 1
    if STATS["flaky_calls"] < 3:
        print(f"    第 {STATS['flaky_calls']} 次调用 -> TransientError")
        raise TransientError("connection reset by peer")
    return {"out": f"第 3 次调用成功"}


def deterministic_bug(state: dict) -> dict:
    STATS["bug_calls"] += 1
    raise ValueError("字段名拼错了，重试一万次也一样")


# 自定义 retry_on：把默认不重试的 ValueError 也纳入重试
forced_policy = RetryPolicy(max_attempts=3, initial_interval=0.01, jitter=False,
                            retry_on=ValueError)


def build(node_fn, retry):
    g = StateGraph(dict)
    g.add_node("work", node_fn, retry_policy=retry)
    g.add_edge(START, "work")
    g.add_edge("work", END)
    return g.compile()


if __name__ == "__main__":
    banner("Demo 18 生产容错：重试 / 不重试 / 兜底")

    print("\n== 1. 瞬时故障 + RetryPolicy（指数退避，默认 retry_on 放行）==")
    policy = RetryPolicy(max_attempts=5, initial_interval=0.05, backoff_factor=2,
                         jitter=False)
    start = time.perf_counter()
    r1 = build(flaky_network, policy).invoke({})
    elapsed = time.perf_counter() - start
    print(f"    -> 成功: {r1['out']}；总耗时 {elapsed:.2f}s（0.05+0.10 两次退避）")

    print("\n== 2. 确定性 bug + 默认策略（ValueError 不该重试）==")
    before = STATS["bug_calls"]
    try:
        build(deterministic_bug,
              RetryPolicy(max_attempts=5, initial_interval=0.01, jitter=False)).invoke({})
    except ValueError as e:
        print(f"    -> 直接抛出: {e}")
    print(f"    实际调用次数: {STATS['bug_calls'] - before}"
          f"（默认 retry_on 排除 ValueError，1 次就停）")

    print("\n== 3. 同样的 bug + retry_on=ValueError（强制重试）==")
    before = STATS["bug_calls"]
    try:
        build(deterministic_bug, forced_policy).invoke({})
    except ValueError:
        pass
    print(f"    实际调用次数: {STATS['bug_calls'] - before}（强制策略下重试满 3 次）")

    print("\n== 4. 环失控兜底：GraphRecursionError ==")
    loop = StateGraph(dict)
    loop.add_node("spin", lambda s: {})
    loop.add_edge(START, "spin")
    loop.add_edge("spin", "spin")                    # 无出口的环
    try:
        loop.compile().invoke({}, config={"recursion_limit": 50})
    except GraphRecursionError:
        print("    -> 捕获 GraphRecursionError：给用户友好提示 + 告警，而不是堆栈")
    print("    （recursion_limit 默认 25；Agent 循环必须配 should_continue 出口）")

    print(
        "\n要点：\n"
        "  1. retry_policy 挂在 add_node（节点级），compile() 不再收这个参数。\n"
        "  2. 默认 retry_on：网络/5xx 重试，ValueError/TypeError 不重试 —— 这是特性。\n"
        "  3. add_node(timeout=) 可加挂死保护；同步阻塞代码会延迟超时触发。\n"
        "  4. 外层永远 try GraphRecursionError，REPL/接口才不会被环打穿。"
    )
