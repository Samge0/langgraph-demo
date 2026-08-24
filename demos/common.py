"""公共工具：所有 demo 共用。

核心作用：
1. 保证 stdout 用 UTF-8（Windows 控制台默认 GBK，直接 print 中文会乱码/报错）。
2. get_llm() —— 统一的模型获取入口：
   - 优先尝试连接本地 OpenAI 兼容端点（环境变量 LOCAL_API_URL / LOCAL_MODEL_NAME），
     探活成功则用真模型（langchain_openai.ChatOpenAI）；
   - 端点不可用时自动回退到脚本化的 FakeMessagesListChatModel，
     保证所有 demo 离线也能跑通、输出确定性结果。
3. FakeToolCallingLLM —— 支持 bind_tools() 的脚本化模型，用于 ReAct demo 的离线回退。
"""
import os
import sys

# ---- 加载项目根目录的 .env（存在则读取；不覆盖已导出的环境变量）----
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---- UTF-8 输出保护（必须最先执行）----
if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage


class FakeToolCallingLLM(FakeMessagesListChatModel):
    """支持 bind_tools 的脚本化假模型。

    BaseChatModel.bind_tools() 基类实现是直接抛 NotImplementedError，
    create_react_agent / llm.bind_tools(...) 离线跑不起来；
    这里重写为"返回自身"，让脚本里预置的 tool_calls 照常回放。
    """

    def bind_tools(self, tools, **kwargs):  # noqa: D401
        return self


def _real_llm_available() -> bool:
    """探测本地 OpenAI 兼容端点是否可用（2 秒超时，不重试）。"""
    base_url = os.getenv("LOCAL_API_URL", "http://localhost:8000/v1")
    model = os.getenv("LOCAL_MODEL_NAME", "")
    if not model:
        return False
    try:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=model,
            api_key=os.getenv("LOCAL_API_KEY", "not-needed"),
            base_url=base_url,
            timeout=10,
            max_retries=0,
        )
        llm.invoke("ping")
        return True
    except Exception as e:
        print(f"探测本地 OpenAI 兼容端点时出现异常：{e}")
        return False


def get_llm(responses=None):
    """获取 LLM 实例。

    Args:
        responses: 离线回退时使用的脚本化回复列表（AIMessage 列表）。
                   会被循环消费：第 n 次调用返回第 n 条，用完后从头再来。

    Returns:
        真模型（ChatOpenAI）或脚本化假模型，二者都实现标准 ChatModel 接口。
    """
    if _real_llm_available():
        from langchain_openai import ChatOpenAI

        print(f"[LLM] 使用真实模型: {os.getenv('LOCAL_MODEL_NAME')} @ {os.getenv('LOCAL_API_URL')}")
        return ChatOpenAI(
            model=os.getenv("LOCAL_MODEL_NAME"),
            api_key=os.getenv("LOCAL_API_KEY", "not-needed"),
            base_url=os.getenv("LOCAL_API_URL", "http://localhost:8000/v1"),
            temperature=0,
        )
    print("[LLM] 本地端点不可用 -> 离线演示模式（脚本化 Fake 模型）")
    return FakeToolCallingLLM(
        responses=responses or [AIMessage(content="(脚本化回复：这是离线模式的假模型输出)")]
    )


def banner(title: str):
    """打印分节横幅，让 demo 输出更易读。"""
    line = "=" * 62
    print(f"\n{line}\n  {title}\n{line}")
