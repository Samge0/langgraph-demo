"""Demo 05 —— create_react_agent 预置体：一行顶 demo04 的几十行。

demo04 手写的 "llm <-> tools" 循环太常用，官方把它抽成了预置组件：
    from langchain.agents import create_agent
    agent = create_agent(model, tools, system_prompt=...)

内置能力（手写都要自己补）：
  - 工具绑定（bind_tools）
  - llm<->tools 循环 + 结束条件
  - 可选 checkpointer（传参即得持久化/多轮记忆）
  - system_prompt 即 system prompt（旧版叫 prompt）

关键点：
- 输入输出都用 {"messages": [...]}，天然多轮。
- 加 checkpointer + thread_id 后，第二次 invoke 不用再带历史 —— 图会从检查点恢复。
- 旧写法 from langgraph.prebuilt import create_react_agent 在 LangGraph V1.0
  已废弃（V2.0 移除），统一改用 langchain.agents.create_agent。
"""
from common import banner, get_llm
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool


@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气（演示用假数据）。"""
    fake = {"北京": "晴 32°C", "上海": "多云 29°C", "杭州": "小雨 27°C"}
    return fake.get(city, f"{city}：晴 25°C")


OFFLINE_SCRIPT = [
    AIMessage(
        content="",
        tool_calls=[{"name": "get_weather", "args": {"city": "北京"}, "id": "c1", "type": "tool_call"}],
    ),
    AIMessage(content="北京今天晴，32°C。"),
]

llm = get_llm(responses=OFFLINE_SCRIPT)
agent = create_agent(llm, [get_weather], system_prompt="你是天气助手，回答要简洁。")

if __name__ == "__main__":
    banner("Demo 05 create_react_agent 预置 Agent")

    result = agent.invoke({"messages": [("user", "杭州今天天气怎么样？")]})
    print("最终回答:", result["messages"][-1].content)

    print(
        "\n要点：\n"
        "  1. 一个函数调用 = 完整 ReAct Agent（绑定工具/循环/结束判断都内置了）。\n"
        "  2. prompt 参数即 system prompt；tools 是 @tool 装饰的函数列表。\n"
        "  3. 需要多轮记忆时传 checkpointer=MemorySaver() 并用 thread_id（见 demo06）。"
    )
