"""Demo 13 —— 结构化输出：让 LLM 的回复变成可校验的 Python 对象。

Agent 开发里一半的活是"从自然语言里可靠地抠出结构化数据"。
两条路线：

  A. 声明式（省事）：llm.with_structured_output(Schema)
     内部 = bind_tools(把 Pydantic 模型转成工具 schema) + 从 tool_call 里取参数。
     需要"真"模型支持 tool calling；本 demo 的离线假模型走不了这条路。

  B. 手写协议（万能，生产兜底）：prompt 注入格式说明 -> 解析 JSON -> Pydantic 校验
     -> 失败就把错误信息喂回去重试。任何模型（包括不支工具调用的）都能跑，
     也是接网关/小模型时最稳的做法。

本 demo 演示路线 B 的完整容错链：剥掉 ```json 围栏、截取第一个 {...}、
解析失败自动带错误反馈重试（离线脚本第一次故意回坏格式，看它自我纠正）。

关键点：
- Pydantic 的 Field(description=...) 不只是文档 —— description 会进 prompt，
  是"给模型看的提示词"。
- 永远不要裸 json.loads(LLM 输出)：围栏、前后废话、截断都会炸。
- 校验失败 -> 错误信息回填 -> 再试一次，这个"纠错环"就是最朴素的 self-correction。
"""
import json
import re

from common import FakeToolCallingLLM, banner, get_llm
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, ValidationError


# ---------- 目标 Schema：字段描述就是给模型的 prompt ----------
from typing import Literal


class ReviewInsight(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="情感倾向，只能是 positive / negative / neutral 之一")
    topics: list[str] = Field(description="评论提到的主题词列表，1~3 个")
    need_reply: bool = Field(description="是否需要人工回复：投诉/退款类为 True")


parser = JsonOutputParser(pydantic_object=ReviewInsight)

# 离线脚本：第一次故意返回"带围栏+废话+多余字段"的脏输出，第二次是纠错后的干净输出
OFFLINE_SCRIPT = [
    AIMessage(content="好的，分析结果如下：\n```json\n{\"sentiment\": \"angry\", \"topics\": [\"物流\", \"客服\"], \"need_reply\": \"yes\"}\n```"),
    AIMessage(content='{"sentiment": "negative", "topics": ["物流"], "need_reply": true}'),
]
llm = get_llm(responses=OFFLINE_SCRIPT)

SYSTEM_PROMPT = "你是评论分析助手，只输出 JSON，不要任何多余文字。\n" + parser.get_format_instructions()


def extract_json_block(text: str) -> dict:
    """从模型输出里抠出 JSON：剥围栏 -> 截取第一个 {...} -> 解析。"""
    text = re.sub(r"```(?:json)?|```", "", text)          # 去掉 ```json 围栏
    match = re.search(r"\{.*\}", text, re.DOTALL)          # 截取第一个大括号块
    if not match:
        raise ValueError("输出里找不到 JSON 对象")
    return json.loads(match.group())


def analyze(review: str, max_retries: int = 2) -> ReviewInsight:
    """带纠错环的结构化提取：解析/校验失败 -> 错误反馈 -> 重试。"""
    messages = [HumanMessage(content=f"{SYSTEM_PROMPT}\n\n评论：{review}")]
    for attempt in range(1, max_retries + 1):
        raw = llm.invoke(messages).content
        print(f"  [第 {attempt} 次] 模型原始输出: {raw[:80]}...")
        try:
            return ReviewInsight.model_validate(extract_json_block(raw))
        except (ValueError, ValidationError) as e:
            print(f"  [第 {attempt} 次] 解析/校验失败: {str(e)[:100]}")
            messages.append(AIMessage(content=raw))                       # 回放模型的话
            messages.append(HumanMessage(content=                         # 喂回错误
                f"输出不合法：{e}\n严格只输出符合 schema 的 JSON，need_reply 必须是布尔值，"
                "sentiment 只能是 positive/negative/neutral。再试一次。"))
    raise RuntimeError("重试用尽仍无法解析")


if __name__ == "__main__":
    banner("Demo 13 结构化输出：手写 JSON 协议 + 纠错环（路线 B）")
    review = "快递拖了五天，客服还踢皮球，太失望了！"
    print(f"\n>>> 评论: {review}")
    result = analyze(review)
    print(f"\n<<< 结构化对象: {result}")
    print(f"    sentiment={result.sentiment} | topics={result.topics} | need_reply={result.need_reply}")

    # 路线 A 一行流（需要真模型 + tool calling）：
    if not isinstance(llm, FakeToolCallingLLM):
        print("\n-- 路线 A：with_structured_output（真模型）--")
        structured = llm.with_structured_output(ReviewInsight)
        print(f"  {structured.invoke(f'分析这条评论：{review}')}")
    else:
        print("\n-- 路线 A：with_structured_output 需要支持 tool calling 的真模型，离线跳过 --")
        print("  用法：llm.with_structured_output(ReviewInsight).invoke('...')  # 直接返回 pydantic 对象")

    print(
        "\n要点：\n"
        "  1. Field(description=...) 会进 prompt —— schema 即提示词。\n"
        "  2. 手写协议三件套：剥围栏 / 截 {...} / Pydantic 校验，缺一个都会在生产炸。\n"
        "  3. 纠错环（错误回填重试）是 universal 兜底；with_structured_output 是舒适区首选。"
    )
