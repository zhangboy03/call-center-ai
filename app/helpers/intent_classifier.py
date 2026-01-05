"""
Intent Classifier Module

Classifies user utterances to determine how to handle them:
- Direct answers to SOP questions
- Side questions (RAG)
- Complaints/emotions (empathy)
- Medical requests (guardrail)
- Fillers (skip extraction)
"""

import logging
import re
from typing import Optional

from app.helpers.state_models import IntentLabel

logger = logging.getLogger(__name__)


# =============================================================================
# Keyword Patterns for Fast Classification
# =============================================================================

FILLER_PATTERNS = [
    r"^嗯+$",
    r"^哦+$",
    r"^啊+$",
    r"^好的?$",
    r"^行$",
    r"^是$",
    r"^对$",
    r"^知道了$",
    r"^明白了?$",
    r"^好吧$",
    r"^嗯嗯$",
    r"^嗯哼$",
]

QUESTION_PATTERNS = [
    r"(什么|怎么|如何|为什么|哪里|哪个|能不能|可不可以|是不是)",
    r"(吗|呢|吧)\s*[?？]?\s*$",
    r"[?？]$",
]

MEDICAL_ADVICE_PATTERNS = [
    r"(应该|需要|可以).*(吃药|用药|服药|调药)",
    r"(调参|参数|调整|调高|调低|增加|减少).*(刺激|电流|幅度)",
    r"(自己|自行).*(调|改|换)",
    r"(不舒服|疼痛|头晕|恶心|呕吐).*(怎么办|怎么处理)",
    r"(药物|药品|处方)",
]

COMPLAINT_PATTERNS = [
    r"(不满意|很差|太差|糟糕|不行|垃圾)",
    r"(投诉|反馈|意见|不开心)",
    r"(怎么回事|什么情况|搞什么)",
    r"(生气|愤怒|失望|烦)",
]

END_CALL_PATTERNS = [
    r"(再见|拜拜|挂了|先这样|没事了|就这些|回头再说)",
    r"(不说了|不聊了|挂断)",
]

CHITCHAT_PATTERNS = [
    r"(天气|吃饭|你好|谢谢|辛苦)",
    r"(哈哈|呵呵|嘿嘿)",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    """Check if text matches any pattern."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _is_filler(text: str) -> bool:
    """Check if text is just a filler."""
    text = text.strip()
    if len(text) <= 3:
        return _matches_any(text, FILLER_PATTERNS)
    return False


def classify_intent(
    text: str,
    current_step: Optional[str] = None,
) -> IntentLabel:
    """
    Classify user utterance intent using rules and keywords.

    Order of precedence:
    1. Filler/silence → SILENCE_OR_FILLER
    2. Medical advice request → ASK_MEDICAL_ADVICE
    3. End call indicators → END_CALL
    4. Complaint/emotion → COMPLAINT_OR_EMOTION
    5. Question patterns → ASK_SIDE_QUESTION
    6. Chitchat → CHITCHAT_OR_OTHER
    7. Default → ANSWER_CURRENT_QUESTION

    Args:
        text: User utterance
        current_step: Current SOP step ID (for context)

    Returns:
        IntentLabel enum value
    """
    text = text.strip()

    # Empty or very short
    if not text or len(text) < 2:
        return IntentLabel.SILENCE_OR_FILLER

    # Check for fillers
    if _is_filler(text):
        logger.debug(f"[Intent] FILLER: {text}")
        return IntentLabel.SILENCE_OR_FILLER

    # Check for medical advice requests (highest priority)
    if _matches_any(text, MEDICAL_ADVICE_PATTERNS):
        logger.info(f"[Intent] MEDICAL: {text}")
        return IntentLabel.ASK_MEDICAL_ADVICE

    # Check for end call
    if _matches_any(text, END_CALL_PATTERNS):
        logger.info(f"[Intent] END_CALL: {text}")
        return IntentLabel.END_CALL

    # Check for complaints
    if _matches_any(text, COMPLAINT_PATTERNS):
        logger.info(f"[Intent] COMPLAINT: {text}")
        return IntentLabel.COMPLAINT_OR_EMOTION

    # Check for questions (likely side questions for RAG)
    if _matches_any(text, QUESTION_PATTERNS):
        # Questions with ? or ？ are more likely to be real questions
        if "?" in text or "？" in text or len(text) > 8:
            logger.info(f"[Intent] SIDE_QUESTION: {text}")
            return IntentLabel.ASK_SIDE_QUESTION

    # Check for chitchat
    if _matches_any(text, CHITCHAT_PATTERNS) and len(text) < 15:
        logger.debug(f"[Intent] CHITCHAT: {text}")
        return IntentLabel.CHITCHAT_OR_OTHER

    # Default: assume it's an answer to the current SOP question
    logger.debug(f"[Intent] ANSWER: {text}")
    return IntentLabel.ANSWER_CURRENT_QUESTION


def get_policy_decision(
    intent: IntentLabel,
    guardrail_triggered: bool = False,
) -> str:
    """
    Determine policy decision based on intent.

    Returns:
        Policy string: extract_advance, extract_no_advance, empathy, rag, guardrail, continue, end_call
    """
    from app.helpers.state_models import PolicyDecision

    if guardrail_triggered:
        return PolicyDecision.GUARDRAIL_RESPONSE

    policy_map = {
        IntentLabel.ANSWER_CURRENT_QUESTION: PolicyDecision.EXTRACT_AND_ADVANCE,
        IntentLabel.SILENCE_OR_FILLER: PolicyDecision.CONTINUE_PROMPT,
        IntentLabel.ASK_SIDE_QUESTION: PolicyDecision.RAG_RESPONSE,
        IntentLabel.COMPLAINT_OR_EMOTION: PolicyDecision.EMPATHY_RESPONSE,
        IntentLabel.ASK_MEDICAL_ADVICE: PolicyDecision.GUARDRAIL_RESPONSE,
        IntentLabel.CHITCHAT_OR_OTHER: PolicyDecision.CONTINUE_PROMPT,
        IntentLabel.END_CALL: PolicyDecision.END_CALL,
        IntentLabel.UNCLEAR: PolicyDecision.CONTINUE_PROMPT,
    }

    return policy_map.get(intent, PolicyDecision.CONTINUE_PROMPT)
