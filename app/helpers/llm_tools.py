"""
LLM 工具插件（简化版）

为本地对话提供的工具函数。
电话相关功能（挂断、转接、短信）已移除。
"""

import asyncio
from html import escape
from typing import Annotated, TypedDict

from pydantic import ValidationError

from app.helpers.config import CONFIG
from app.helpers.llm_utils import AbstractPlugin, add_customer_response
from app.helpers.logging import logger
from app.models.call import CallStateModel
from app.models.message import (
    ActionEnum as MessageActionEnum,
    MessageModel,
    PersonaEnum as MessagePersonaEnum,
)
from app.models.reminder import ReminderModel
from app.models.training import TrainingModel

_db = CONFIG.database.instance
_search = CONFIG.ai_search.instance


class UpdateClaimDict(TypedDict):
    field: str
    value: str


class DefaultPlugin(AbstractPlugin):
    """默认 LLM 工具插件"""

    @add_customer_response(
        [
            "我正在创建一个新的案例。",
            "好的，我们开始新的对话。",
        ]
    )
    async def new_claim(self) -> str:
        """
        创建新的案例/对话。
        
        # 行为
        1. 旧案例被保存但不再可访问
        2. 重置对话
        
        # 使用场景
        - 客户想讨论完全不同的主题
        - 客户明确要求创建新案例
        """
        # 触发后处理回调
        await self.post_callback(self.call)

        # 创建新案例
        self.call = await _db.call_create(
            CallStateModel(
                initiate=self.call.initiate.model_copy(),
                messages=[
                    MessageModel(
                        action=MessageActionEnum.CALL,
                        content="",
                        persona=MessagePersonaEnum.HUMAN,
                    ),
                ],
            )
        )
        return "案例和消息已重置"

    @add_customer_response(
        [
            "好的，我记录一下提醒。",
            "已经为您安排了一个待办事项。",
        ]
    )
    async def new_or_updated_reminder(
        self,
        description: Annotated[
            str,
            "提醒的描述，应该足够详细。例如：'回访患者了解恢复情况'",
        ],
        due_date_time: Annotated[
            str,
            "提醒触发的日期时间，ISO 格式。",
        ],
        owner: Annotated[
            str,
            "提醒的负责人。例如：'客服'、'患者'、'医生'",
        ],
        title: Annotated[
            str,
            "提醒的简短标题。例如：'术后随访'、'复查预约'",
        ],
    ) -> str:
        """
        创建或更新提醒。

        # 使用场景
        - 需要跟进某件事情
        - 安排回访或随访
        - 记录待办事项
        """
        # 检查是否已存在同名提醒
        for reminder in self.call.reminders:
            if reminder.title == title:
                try:
                    reminder.description = description
                    reminder.due_date_time = due_date_time  # pyright: ignore
                    reminder.owner = owner
                    return f'提醒 "{title}" 已更新。'
                except ValidationError as e:
                    return f'更新提醒失败 "{title}": {e.json()}'

        # 创建新提醒
        try:
            reminder = ReminderModel(
                description=description,
                due_date_time=due_date_time,  # pyright: ignore
                owner=owner,
                title=title,
            )
            self.call.reminders.append(reminder)
            return f'提醒 "{title}" 已创建。'
        except ValidationError as e:
            return f'创建提醒失败 "{title}": {e.json()}'

    @add_customer_response(
        [
            "好的，我记录一下这个信息。",
            "已经更新了您的资料。",
        ]
    )
    async def updated_claim(
        self,
        updates: Annotated[
            list[UpdateClaimDict],
            """
            要更新的字段。

            # 可用字段
            {% for field in call.initiate.claim %}
            {% if not field.description %}
            - {{ field.name }}
            {% else %}
            - '{{ field.name }}', {{ field.description }}
            {% endif %}
            {% endfor %}

            # 数据格式
            [{'field': '[字段名]', 'value': '[值]'}]

            # 示例
            - [{'field': 'patient_name_confirmed', 'value': '张三'}]
            """,
        ],
    ) -> str:
        """
        更新案例信息。
        
        # 使用场景
        - 记录患者信息
        - 更新随访状态
        - 保存对话中收集的数据
        """
        res = "# 已更新字段"
        for field in updates:
            res += f"\n- {self._update_claim_field(field)}"
        return res

    def _update_claim_field(self, update: UpdateClaimDict) -> str:
        field = update["field"]
        new_value = update["value"]

        old_value = self.call.claim.get(field, None)
        try:
            self.call.claim[field] = new_value
            CallStateModel.model_validate(self.call)
            return f'已更新 "{field}" 为 "{new_value}"'
        except ValidationError as e:
            self.call.claim[field] = old_value
            return f'更新失败 "{field}": {e.json()}'

    @add_customer_response(
        [
            "我正在为您查找相关资料。",
            "让我在知识库中搜索一下。",
        ]
    )
    async def search_document(
        self,
        queries: Annotated[
            list[str],
            "搜索查询文本列表。例如：['脑起搏器术后注意事项', '帕金森病康复指南']",
        ],
    ) -> str:
        """
        搜索知识库文档。
        
        # 使用场景
        - 查找医疗指南
        - 搜索术后护理说明
        - 获取专业信息
        """
        # 并行执行搜索
        tasks = await asyncio.gather(
            *[
                _search.training_search_all(text=query, lang="zh-CN")
                for query in queries
            ]
        )

        # 合并结果
        trainings = sorted(set(training for task in tasks for training in task or []))

        # 格式化结果
        trainings_str = "\n".join(
            [
                f"<documents>{escape(training.model_dump_json(exclude=TrainingModel.excluded_fields_for_llm()))}</documents>"
                for training in trainings
            ]
        )

        res = "# 搜索结果"
        res += f"\n{trainings_str}"
        return res

    @add_customer_response(
        [
            "好的，我放慢/加快语速。",
        ],
        before=False,
    )
    async def speech_speed(
        self,
        speed: Annotated[
            float,
            "语速，范围 0.75-1.25，1.0 为正常速度。",
        ],
    ) -> str:
        """
        调整语音速度。

        # 使用场景
        - 客户听不清楚
        - 需要更快或更慢的语速
        """
        speed = max(0.75, min(speed, 1.25))
        initial_speed = self.call.initiate.prosody_rate
        self.call.initiate.prosody_rate = speed
        return f"语速已调整为 {speed}（原为 {initial_speed}）"

    @add_customer_response(
        [
            "好的，我切换到中文。",
        ],
        before=False,
    )
    async def speech_lang(
        self,
        lang: Annotated[
            str,
            """
            语言代码。

            # 可用语言
            {% for available in call.initiate.lang.availables %}
            - {{ available.short_code }} ({{ available.pronunciations_en[0] }})
            {% endfor %}
            """,
        ],
    ) -> str:
        """
        切换对话语言。
        
        # 使用场景
        - 客户希望使用其他语言
        """
        if not any(
            lang == available.short_code
            for available in self.call.initiate.lang.availables
        ):
            return f"语言 {lang} 不可用"

        initial_lang = self.call.lang.short_code
        self.call.lang_short_code = lang
        return f"语言已切换为 {lang}（原为 {initial_lang}）"
