"""
Slot Extractor Module

Extracts structured slot values from free-form user responses.
Uses cheap regex for numbers, LLM fallback for complex extractions.
"""

import json
import logging
import os
import re
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# =============================================================================
# LLM Client for Fallback Extraction
# =============================================================================

_llm_client: Optional[OpenAI] = None


def _get_llm_client() -> OpenAI:
    """Get or create LLM client for slot extraction."""
    global _llm_client
    if _llm_client is None:
        _llm_client = OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY", ""),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    return _llm_client


def llm_extract_slots(
    user_text: str,
    slot_defs: list[dict],
) -> dict[str, Any]:
    """
    Use LLM (qwen-turbo) to extract slots from user responses.

    Args:
        user_text: User's response
        slot_defs: Slots to extract with name, type, options

    Returns:
        Dict of extracted slot values
    """
    if not slot_defs:
        return {}

    # Build slot descriptions for prompt
    slot_desc = []
    for slot in slot_defs:
        name = slot["name"]
        slot_type = slot["type"]
        options = slot.get("options", [])

        if slot_type == "enum" and options:
            slot_desc.append(f'"{name}": 只能是以下之一: {options}')
        elif slot_type == "yes_no" or slot_type == "bool":
            slot_desc.append(f'"{name}": true 或 false')
        elif slot_type == "int_0_10":
            slot_desc.append(f'"{name}": 0-10的整数，根据用户描述推断')
        elif slot_type == "multi_select" and options:
            slot_desc.append(
                f'"{name}": 数组，可选值: {options}。如果用户说"都好了"或"改善了"，可以推断为["整体改善"]或根据上下文选择'
            )
        else:
            slot_desc.append(f'"{name}": 字符串')

    prompt = f"""你是一个槽位提取助手。从用户回复中提取信息，只返回JSON。

规则:
1. 如果用户明确说了某个值，直接提取
2. 如果用户的回答暗示了某个值，合理推断
3. 对于评分类问题，"很好/好多了"可推断为8-9分，"一般"为5分，"不好"为2-3分
4. 对于症状改善，"好多了/改善很多"=明显改善，"有点好"=有所改善
5. 只有完全无法判断时才返回null

需要提取的字段:
{chr(10).join(slot_desc)}

用户说: "{user_text}"

只返回JSON:"""

    try:
        client = _get_llm_client()
        response = client.chat.completions.create(
            model="qwen-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )

        result_text = response.choices[0].message.content.strip()
        # Clean up markdown code blocks if present
        if result_text.startswith("```"):
            result_text = result_text.split("```")[1]
            if result_text.startswith("json"):
                result_text = result_text[4:]
        result_text = result_text.strip()

        result = json.loads(result_text)
        logger.info(f"[LLM Extract] {result}")
        return result

    except Exception as e:
        logger.error(f"[LLM Extract] Error: {e}")
        return {}


# =============================================================================
# Regex Patterns for Fast Extraction
# =============================================================================

PATTERNS = {
    # Numbers (Chinese and Arabic)
    "int_0_10": re.compile(
        r"(?P<num>\d+|[零一二三四五六七八九十]+)\s*[分点]?",
        re.IGNORECASE,
    ),
    "fuzzy_range": re.compile(
        r"(?:大概|差不多|估计|约|左右)?\s*(?P<num>[\d,.]+)\s*(?P<unit>[万元块千百]?)",
        re.IGNORECASE,
    ),
    "fuzzy_pct": re.compile(
        r"(?:大概|差不多|约)?\s*(?P<num>[\d.]+)\s*[%％成]?|(?:一半|三分之一|四分之一)",
        re.IGNORECASE,
    ),
    "yes_no": re.compile(
        r"^(?P<yes>是|对|有|好的|可以|嗯|行|没问题|对的)|(?P<no>不是|没有|不|否|没|不对|不行|不是的)$",
        re.IGNORECASE,
    ),
}

# Chinese number mapping
CN_NUMBERS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}


def cn_to_int(text: str) -> Optional[int]:
    """Convert Chinese number to integer."""
    if text.isdigit():
        return int(text)
    if text in CN_NUMBERS:
        return CN_NUMBERS[text]
    # Handle 十几
    if text.startswith("十") and len(text) == 2:
        return 10 + CN_NUMBERS.get(text[1], 0)
    return None


def extract_int_0_10(text: str) -> Optional[int]:
    """Extract a score from 0-10."""
    # Direct number
    match = PATTERNS["int_0_10"].search(text)
    if match:
        num_str = match.group("num")
        num = cn_to_int(num_str)
        if num is not None and 0 <= num <= 10:
            return num

    # Descriptive patterns
    text_lower = text.lower()
    if any(w in text_lower for w in ["满分", "非常好", "特别好", "完美"]):
        return 10
    if any(w in text_lower for w in ["很好", "不错", "挺好"]):
        return 8
    if any(w in text_lower for w in ["一般", "还行", "凑合"]):
        return 5
    if any(w in text_lower for w in ["不好", "差", "糟糕"]):
        return 2
    if any(w in text_lower for w in ["很差", "非常差", "最差"]):
        return 0

    return None


def extract_yes_no(text: str) -> Optional[bool]:
    """Extract yes/no from response."""
    text = text.strip()

    # Direct patterns
    match = PATTERNS["yes_no"].match(text)
    if match:
        if match.group("yes"):
            return True
        if match.group("no"):
            return False

    # Longer responses
    if any(w in text for w in ["是的", "对的", "有的", "没问题", "可以的", "好的"]):
        return True
    if any(w in text for w in ["不是", "没有", "不对", "不行", "不可以", "不用"]):
        return False

    return None


def extract_fuzzy_range(text: str) -> Optional[dict]:
    """
    Extract fuzzy monetary range.
    "差不多5万" → {"min": 45000, "max": 55000, "estimate": 50000}
    """
    match = PATTERNS["fuzzy_range"].search(text)
    if not match:
        return None

    num_str = match.group("num").replace(",", "")
    try:
        num = float(num_str)
    except ValueError:
        return None

    unit = match.group("unit") or ""
    multiplier = 1
    if "万" in unit:
        multiplier = 10000
    elif "千" in unit:
        multiplier = 1000
    elif "百" in unit:
        multiplier = 100

    estimate = num * multiplier
    # 10% fuzzy range
    return {
        "min": int(estimate * 0.9),
        "max": int(estimate * 1.1),
        "estimate": int(estimate),
    }


def extract_fuzzy_pct(text: str) -> Optional[float]:
    """
    Extract fuzzy percentage.
    "大概一半" → 0.5
    "差不多60%" → 0.6
    """
    if "一半" in text:
        return 0.5
    if "三分之一" in text:
        return 0.33
    if "四分之一" in text:
        return 0.25
    if "三分之二" in text:
        return 0.67

    match = PATTERNS["fuzzy_pct"].search(text)
    if match and match.group("num"):
        num = float(match.group("num"))
        if num > 1:  # Percentage like 60
            return num / 100
        return num  # Already 0.x

    return None


# Synonym mapping for enum options
ENUM_SYNONYMS = {
    "明显改善": [
        "改善很多",
        "好多了",
        "好很多",
        "大大改善",
        "明显好转",
        "效果很好",
        "效果特别好",
    ],
    "有所改善": ["有改善", "有点改善", "稍微好点", "好一些", "有好转", "还可以"],
    "无变化": ["没变化", "差不多", "一样", "没什么变化", "跟以前一样"],
    "有所加重": ["严重了", "加重了", "更严重", "恶化", "不如以前", "变差"],
    "满意": ["很满意", "挺满意", "还满意", "不错"],
    "不满意": ["不太满意", "不是很满意", "有点不满意"],
}


# =============================================================================
# Phase 3: Deterministic Mappers
# =============================================================================

# Satisfaction phrase → 0-5 score
SATISFACTION_SCORE_MAP = {
    5: ["非常满意", "特别满意", "满分", "完美", "太好了"],
    4: ["很满意", "挺满意", "满意"],
    3: ["还行", "一般", "凑合", "还可以"],
    2: ["不太满意", "有点不满意"],
    1: ["不满意", "比较差"],
    0: ["非常不满意", "很差", "太差了"],
}


def satisfaction_phrase_to_score(text: str) -> Optional[int]:
    """
    Map satisfaction phrase to 0-5 score.

    Examples:
        "非常满意" → 5
        "挺满意" → 4
        "一般" → 3
        "不太满意" → 2
    """
    text_lower = text.lower()

    # Check more specific (longer) phrases first to avoid overlap
    # Order: check negative phrases before positive ones
    ordered_checks = [
        (0, ["非常不满意", "很差", "太差了"]),
        (1, ["不满意", "比较差"]),
        (2, ["不太满意", "有点不满意"]),
        (5, ["非常满意", "特别满意", "满分", "完美", "太好了"]),
        (4, ["很满意", "挺满意", "满意"]),
        (3, ["还行", "一般", "凑合", "还可以"]),
    ]

    for score, phrases in ordered_checks:
        for phrase in phrases:
            if phrase in text_lower:
                return score
    return None


# Per-product symptom keywords for multi-select tagging
PRODUCT_SYMPTOM_KEYWORDS = {
    "DBS_PD": {
        "震颤": ["震颤", "抖", "手抖", "颤抖"],
        "强直": ["强直", "僵硬", "僵", "硬"],
        "运动迟缓": ["迟缓", "慢", "动作慢", "反应慢"],
        "姿势步态": ["走路", "姿势", "步态", "平衡"],
    },
    "DBS_Dystonia": {
        "肌张力障碍": ["肌张力", "扭转", "痉挛"],
        "异常姿势": ["姿势", "歪", "扭"],
    },
    "VNS": {
        "发作频率": ["发作", "频率", "次数"],
        "发作程度": ["程度", "严重", "剧烈"],
    },
    "SNM": {
        "排尿症状": ["排尿", "尿", "小便"],
        "漏尿": ["漏尿", "尿失禁"],
    },
    "SCS": {
        "疼痛程度": ["疼", "痛", "疼痛"],
        "活动能力": ["活动", "走路", "运动"],
    },
}


def extract_product_symptoms(
    text: str,
    product_line: str = "DBS_PD",
) -> list[str]:
    """
    Extract symptom tags from text based on product line.

    Args:
        text: User response
        product_line: Product line (DBS_PD, VNS, etc.)

    Returns:
        List of matched symptom tag names
    """
    symptoms = PRODUCT_SYMPTOM_KEYWORDS.get(product_line, {})
    matched = []
    text_lower = text.lower()

    for symptom_name, keywords in symptoms.items():
        for keyword in keywords:
            if keyword in text_lower:
                matched.append(symptom_name)
                break  # Only add once per symptom

    return matched


def extract_enum(text: str, options: list[str]) -> Optional[str]:
    """
    Match text to one of the enum options.
    Uses substring matching and synonym lookup.
    """
    text_lower = text.lower()

    # Direct match
    for opt in options:
        if opt.lower() in text_lower:
            return opt

    # Synonym match
    for opt in options:
        synonyms = ENUM_SYNONYMS.get(opt, [])
        for syn in synonyms:
            if syn in text_lower:
                return opt

    # Fuzzy: first 2 chars match
    for opt in options:
        if len(opt) >= 2 and opt[:2].lower() in text_lower:
            return opt

    return None


def extract_multi_select(text: str, options: list[str]) -> list[str]:
    """
    Extract multiple selections from text.
    """
    selected = []
    text_lower = text.lower()
    for opt in options:
        if opt.lower() in text_lower:
            selected.append(opt)
    return selected


def extract_slots(
    user_text: str,
    slot_defs: list[dict],
    use_llm_fallback: bool = True,
) -> dict[str, Any]:
    """
    Extract all slots from user text using hybrid approach.

    Fast path: Regex/keyword matching (0ms)
    Fallback: LLM extraction for required slots that failed regex (200-400ms)

    Args:
        user_text: Raw user response
        slot_defs: List of slot definitions with 'name', 'type', 'options'
        use_llm_fallback: Whether to use LLM for missing required slots

    Returns:
        Dict of slot_name → extracted_value (None if not found)
    """
    results = {}

    # === Fast Path: Regex/Keyword Extraction ===
    for slot in slot_defs:
        name = slot["name"]
        slot_type = slot["type"]
        options = slot.get("options", [])

        value = None

        if slot_type == "yes_no":
            value = extract_yes_no(user_text)
        elif slot_type in ("int_0_10", "int_or_unknown"):
            value = extract_int_0_10(user_text)
            if value is None and slot_type == "int_or_unknown":
                if any(w in user_text for w in ["不知道", "不清楚", "不记得", "忘了"]):
                    value = -1
        elif slot_type == "fuzzy_range":
            value = extract_fuzzy_range(user_text)
        elif slot_type == "fuzzy_pct":
            value = extract_fuzzy_pct(user_text)
        elif slot_type == "enum":
            value = extract_enum(user_text, options)
        elif slot_type == "multi_select":
            value = extract_multi_select(user_text, options)
            if not value:
                value = None
        elif slot_type == "bool":
            value = extract_yes_no(user_text)
        elif slot_type == "text":
            if len(user_text.strip()) > 2:
                value = user_text.strip()

        results[name] = value
        if value is not None:
            logger.debug(f"[SlotExtractor:Regex] {name} ({slot_type}) = {value}")

    # === LLM Extraction for ALL Required Slots (every turn) ===
    if use_llm_fallback:
        required_slots = [slot for slot in slot_defs if slot.get("required", False)]

        if required_slots:
            logger.info(
                f"[SlotExtractor] Using LLM for {len(required_slots)} required slots"
            )
            llm_results = llm_extract_slots(user_text, required_slots)

            # Merge LLM results (LLM takes priority over regex for required slots)
            for slot in required_slots:
                name = slot["name"]
                if name in llm_results and llm_results[name] is not None:
                    # LLM result overrides regex result for required slots
                    if (
                        results.get(name) is None
                        or results.get(name) != llm_results[name]
                    ):
                        results[name] = llm_results[name]
                        logger.info(f"[SlotExtractor:LLM] {name} = {llm_results[name]}")

    return results


def map_score_to_5(score: int) -> int:
    """
    Map 0-10 score to 0-5 scale.
    0-1 → 0, 2-3 → 1, 4-5 → 2, 6-7 → 3, 8-9 → 4, 10 → 5
    """
    if score < 0:
        return -1
    if score <= 1:
        return 0
    if score <= 3:
        return 1
    if score <= 5:
        return 2
    if score <= 7:
        return 3
    if score <= 9:
        return 4
    return 5
