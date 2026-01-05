"""
History Extractor - Backend LLM for Slot Extraction from Conversation History

Extracts structured slot values from the full conversation history.
Called periodically during the call to track progress and at end for final extraction.
"""

import json
import logging
import os
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# LLM client for extraction
_llm_client: Optional[OpenAI] = None


def _get_llm_client() -> OpenAI:
    """Get or create LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _llm_client


# The questions we need to collect (matching full SOP schema)
REQUIRED_INFO = [
    {
        "id": "is_patient",
        "question": "是否是患者本人接听",
        "type": "bool",
        "required": True,
    },
    {
        "id": "symptom_improvement",
        "question": "术后症状改善情况",
        "type": "enum",
        "options": ["明显改善", "有所改善", "无变化", "有所加重"],
        "required": True,
    },
    {
        "id": "control_score",
        "question": "症状控制打分（0-10）",
        "type": "int_0_10",
        "required": True,
    },
    {
        "id": "life_quality_score",
        "question": "生活质量打分（0-10）",
        "type": "int_0_10",
        "required": False,
    },
    {
        "id": "programming_count",
        "question": "程控次数",
        "type": "int",
        "required": True,
    },
    {
        "id": "programming_satisfaction",
        "question": "程控效果满意度",
        "type": "enum",
        "options": ["满意", "一般", "不满意"],
        "required": True,
    },
    {
        "id": "side_effects",
        "question": "有没有不良反应或不舒服",
        "type": "text",
        "required": True,
    },
    {
        "id": "mental_issues",
        "question": "情绪精神方面有没有问题",
        "type": "bool",
        "required": False,
    },
    {
        "id": "medication_issues",
        "question": "吃药有没有问题",
        "type": "text",
        "required": False,
    },
    {
        "id": "pre_op_informed",
        "question": "术前医生是否说过预期效果",
        "type": "enum",
        "options": ["充分告知", "未充分告知", "不记得"],
        "required": True,
    },
    {
        "id": "device_explained",
        "question": "出院时是否讲解设备使用",
        "type": "enum",
        "options": ["已讲解清楚", "未讲解", "不记得"],
        "required": True,
    },
    {
        "id": "id_card_explained",
        "question": "识别卡作用是否清楚",
        "type": "enum",
        "options": ["清楚", "不清楚"],
        "required": True,
    },
    {
        "id": "insurance_type",
        "question": "医保类型",
        "type": "enum",
        "options": ["职工医保", "居民医保", "新农合", "自费", "其他"],
        "required": True,
    },
    {
        "id": "total_cost",
        "question": "手术总费用（大概）",
        "type": "text",
        "required": False,
    },
    {
        "id": "self_pay",
        "question": "自费部分（大概）",
        "type": "text",
        "required": False,
    },
    {
        "id": "huimin_bao",
        "question": "是否买了惠民保",
        "type": "bool",
        "required": True,
    },
    {
        "id": "other_concerns",
        "question": "其他问题或特殊情况",
        "type": "text",
        "required": False,
    },
]


def extract_from_history(messages: list[dict]) -> dict[str, Any]:
    """
    Extract all slot values from conversation history.

    Args:
        messages: Full conversation history [{"role": "user/assistant", "content": "..."}]

    Returns:
        Dict mapping slot_id to extracted value (None if not found)
    """
    if not messages:
        return {}

    # Build conversation text
    conversation = "\n".join(
        f"{'患者' if m['role'] == 'user' else '客服'}: {m['content']}" for m in messages
    )

    # Build extraction prompt
    slot_descriptions = []
    for info in REQUIRED_INFO:
        slot_type = info["type"]
        if slot_type == "bool":
            type_hint = "true 或 false"
        elif slot_type == "int_0_10":
            type_hint = "0-10的整数"
        elif slot_type == "int":
            type_hint = "整数"
        elif slot_type == "enum":
            type_hint = f"只能是: {info.get('options', [])}"
        else:
            type_hint = "字符串"

        required_str = "【必填】" if info.get("required") else ""
        slot_descriptions.append(
            f'- "{info["id"]}": {info["question"]} ({type_hint}) {required_str}'
        )

    prompt = f"""你是一个信息提取助手。请从以下对话中提取随访信息。

【对话记录】
{conversation}

【需要提取的信息】
{chr(10).join(slot_descriptions)}

【规则】
1. 只提取对话中明确提到或可以明确推断的信息
2. 如果某项信息没有提到，返回 null
3. 对于评分，"很好/好多了"可推断为7-8分，"还行/一般"为5分，"不太好"为3分
4. 对于症状改善，"好多了"="明显改善"，"有点好转"="有所改善"

只返回JSON格式:"""

    try:
        client = _get_llm_client()
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500,
        )

        result_text = response.choices[0].message.content.strip()

        # Clean up markdown code blocks
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()

        result = json.loads(result_text)
        logger.info(f"[HistoryExtractor] Extracted: {result}")
        return result

    except Exception as e:
        logger.error(f"[HistoryExtractor] Error: {e}")
        return {}


def get_missing_info(extracted: dict[str, Any]) -> list[str]:
    """
    Get list of missing REQUIRED information.

    Returns:
        List of question descriptions that still need to be asked
    """
    missing = []
    for info in REQUIRED_INFO:
        if info.get("required", False):
            slot_id = info["id"]
            if slot_id not in extracted or extracted[slot_id] is None:
                missing.append(info["question"])
    return missing


def get_collection_progress(extracted: dict[str, Any]) -> str:
    """
    Get a natural language summary of collection progress.

    Returns:
        String like "已收集: 身份确认、症状改善 | 待收集: 评分、不良反应"
    """
    collected = []
    pending = []

    for info in REQUIRED_INFO:
        slot_id = info["id"]
        question = info["question"]

        if slot_id in extracted and extracted[slot_id] is not None:
            value = extracted[slot_id]
            # Format value nicely
            if isinstance(value, bool):
                value_str = "是" if value else "否"
            elif isinstance(value, (int, float)):
                value_str = str(value)
            else:
                value_str = str(value)[:20]
            collected.append(f"{question}={value_str}")
        elif info.get("required", False):
            pending.append(question)

    collected_str = ", ".join(collected) if collected else "暂无"
    pending_str = ", ".join(pending) if pending else "无"

    return f"已收集: {collected_str}\n待收集: {pending_str}"
