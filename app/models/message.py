"""
消息模型

定义对话中的消息结构。
"""

import re
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

_FUNC_NAME_SANITIZER_R = r"[^a-zA-Z0-9_-]"
_MESSAGE_ACTION_R = r"(?:action=*([a-z_]*))? *(.*)"
_MESSAGE_STYLE_R = r"(?:style=*([a-z_]*))? *(.*)"


class StyleEnum(str, Enum):
    """语音风格"""
    CHEERFUL = "cheerful"
    NONE = "none"
    SAD = "sad"


class ActionEnum(str, Enum):
    """消息动作类型"""
    CALL = "call"
    """用户发起通话"""
    HANGUP = "hangup"
    """用户挂断"""
    SMS = "sms"
    """短信"""
    TALK = "talk"
    """对话消息"""


class PersonaEnum(str, Enum):
    """消息角色"""
    ASSISTANT = "assistant"
    """AI 助手"""
    HUMAN = "human"
    """人类用户"""
    TOOL = "tool"
    """工具调用结果"""


class ToolModel(BaseModel):
    """工具调用模型"""
    content: str = ""
    function_arguments: str = ""
    function_name: str = ""
    tool_id: str = ""

    @property
    def is_openai_valid(self) -> bool:
        """检查是否为有效的 OpenAI 工具调用"""
        return bool(self.tool_id and self.function_name)

    def add_delta(self, delta: Any) -> "ToolModel":
        """更新工具调用（流式响应）"""
        if hasattr(delta, 'id') and delta.id:
            self.tool_id = delta.id
        if hasattr(delta, 'function'):
            if hasattr(delta.function, 'name') and delta.function.name:
                self.function_name = delta.function.name
            if hasattr(delta.function, 'arguments') and delta.function.arguments:
                self.function_arguments += delta.function.arguments
        return self

    def to_openai(self) -> dict:
        """转换为 OpenAI 格式"""
        sanitized_name = "-".join(
                    re.sub(
                        _FUNC_NAME_SANITIZER_R,
                        "-",
                        self.function_name,
                    ).split("-")
        )
        return {
            "id": self.tool_id,
            "type": "function",
            "function": {
                "name": sanitized_name,
                "arguments": self.function_arguments,
            }
        }

    def __hash__(self) -> int:
        return self.tool_id.__hash__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ToolModel):
            return False
        return self.tool_id == other.tool_id


class MessageModel(BaseModel):
    """消息模型"""
    # 不可变字段
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC), frozen=True)
    # 可编辑字段
    action: ActionEnum = ActionEnum.TALK
    content: str
    lang_short_code: str | None = None
    persona: PersonaEnum
    style: StyleEnum = StyleEnum.NONE
    tool_calls: list[ToolModel] = []

    async def translate(self, target_short_code: str) -> "MessageModel":
        """
        翻译消息（简化版，暂不支持翻译）
        
        返回消息副本。
        """
        # 简化实现：直接返回副本，不进行翻译
        return self.model_copy()

    @field_validator("created_at")
    @classmethod
    def _validate_created_at(cls, created_at: datetime) -> datetime:
        """确保时间戳有时区信息"""
        if not created_at.tzinfo:
            return created_at.replace(tzinfo=UTC)
        return created_at

    def to_openai(self) -> list[dict]:
        """转换为 OpenAI 消息格式"""
        content = " ".join([line.strip() for line in self.content.splitlines()])

        # 用户消息
        if self.persona == PersonaEnum.HUMAN:
            return [
                {
                    "role": "user",
                    "content": f"action={self.action.value} {content}",
                }
            ]

        # 助手消息（无工具调用）
        if self.persona == PersonaEnum.ASSISTANT:
            if not self.tool_calls:
                return [
                    {
                        "role": "assistant",
                        "content": f"action={self.action.value} style={self.style.value} {content}",
                    }
                ]

        # 助手消息（有工具调用）
        valid_tools = [
            tool_call for tool_call in self.tool_calls if tool_call.is_openai_valid
        ]
        res = []
        res.append(
            {
                "role": "assistant",
                "content": f"action={self.action.value} style={self.style.value} {content}",
                "tool_calls": [tool_call.to_openai() for tool_call in valid_tools],
            }
        )
        res.extend(
            {
                "role": "tool",
                "content": tool_call.content,
                "tool_call_id": tool_call.tool_id,
            }
            for tool_call in valid_tools
            if tool_call.content
        )
        return res


def _filter_action(text: str) -> str:
    """移除 action 标记"""
    res = re.match(_MESSAGE_ACTION_R, text)
    if not res:
        return text
    try:
        return res.group(2) or ""
    except ValueError:
        return text


def _filter_content(text: str) -> str:
    """移除 content= 前缀"""
    return text.replace("content=", "")


def extract_message_style(text: str) -> tuple[StyleEnum, str]:
    """
    提取消息风格
    
    示例:
    - 输入: "style=cheerful Hello!"
    - 输出: (StyleEnum.CHEERFUL, "Hello!")
    """
    text = _filter_action(text)
    text = _filter_content(text)

    default_style = StyleEnum.NONE
    res = re.match(_MESSAGE_STYLE_R, text)
    if not res:
        return default_style, text
    try:
        return (
            StyleEnum(res.group(1)),
            (res.group(2) or ""),
        )
    except ValueError:
        return default_style, text
