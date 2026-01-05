"""
Guardrails Module

Safety checks and content restrictions for the SOP agent.
Phase 4: Centralized with rule IDs, template replies, and escalation flags.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GuardrailResult:
    """Result of guardrail check."""

    triggered: bool
    rule_id: str = ""  # Unique ID for logging/tracking
    rule_name: str = ""
    response: str = ""
    action: str = ""  # record_to_notes, block, warn, escalate
    should_escalate: bool = False
    log_event: bool = True  # Whether to log this event


# =============================================================================
# Guardrail Rules
# =============================================================================

GUARDRAILS = [
    {
        "id": "GR-001",
        "name": "vns_no_adjust",
        "description": "VNS患者不应自行调参",
        "product_lines": ["VNS"],
        "triggers": [
            r"自己调",
            r"自行调",
            r"自己改",
            r"自己设置",
            r"怎么调参",
            r"参数怎么调",
        ],
        "response": "这个建议您跟医生确认一下再调整，自行调整可能有风险。",
        "action": "record_to_notes",
        "should_escalate": False,
    },
    {
        "id": "GR-002",
        "name": "no_medical_advice",
        "description": "不给用药建议",
        "product_lines": None,  # All products
        "triggers": [
            r"药.*怎么吃",
            r"吃多少.*药",
            r"药量",
            r"加药",
            r"减药",
            r"换药",
            r"停药",
            r"剂量",
        ],
        "response": "用药方面建议您跟主治医生详细沟通，我们不能给具体的用药建议。",
        "action": "record_to_notes",
        "should_escalate": False,
    },
    {
        "id": "GR-003",
        "name": "no_diagnosis",
        "description": "不做诊断",
        "product_lines": None,
        "triggers": [
            r"是不是.*病",
            r"是什么病",
            r"帮我诊断",
            r"帮我看看",
            r"什么原因引起",
        ],
        "response": "这个需要医生面诊才能确定，建议您去医院检查一下。",
        "action": "record_to_notes",
        "should_escalate": False,
    },
    {
        "id": "GR-004",
        "name": "emergency_redirect",
        "description": "紧急情况转接",
        "product_lines": None,
        "triggers": [
            r"很疼|剧痛|受不了",
            r"发烧.*度|高烧",
            r"感染|化脓|红肿",
            r"出血|流血",
            r"晕倒|昏迷",
        ],
        "response": "您这个情况比较紧急，建议您尽快去医院急诊看一下，或者拨打120。",
        "action": "escalate",
        "should_escalate": True,
    },
]


# =============================================================================
# Empathy Templates (Phase 4)
# =============================================================================

EMPATHY_TEMPLATES = {
    "complaint": [
        "理解您的心情，这确实挺不容易的。",
        "您说的我都记下了，我们会反馈给相关部门。",
        "抱歉给您带来不好的体验。",
    ],
    "frustration": [
        "非常理解您的感受。",
        "您说得对，我们会改进的。",
    ],
    "worry": [
        "您不用太担心，有什么问题我们可以帮您解决。",
        "这个很常见，不用紧张。",
    ],
}


def check_guardrails(
    user_text: str,
    product_line: Optional[str] = None,
) -> Optional[GuardrailResult]:
    """
    Check user input against all guardrails.

    Args:
        user_text: Raw user input
        product_line: Current product line (e.g., "VNS", "DBS_PD")

    Returns:
        GuardrailResult if triggered, None otherwise
    """
    for rule in GUARDRAILS:
        # Check product line filter
        if rule["product_lines"] is not None:
            if product_line not in rule["product_lines"]:
                continue

        # Check triggers
        for pattern in rule["triggers"]:
            if re.search(pattern, user_text, re.IGNORECASE):
                logger.warning(
                    f"[Guardrail] Triggered: {rule['id']} ({rule['name']}) for text: {user_text[:50]}"
                )
                return GuardrailResult(
                    triggered=True,
                    rule_id=rule.get("id", ""),
                    rule_name=rule["name"],
                    response=rule["response"],
                    action=rule["action"],
                    should_escalate=rule.get("should_escalate", False),
                )

    return None


def should_escalate_to_human(
    guardrail_triggers: int,
    retry_count: int,
    unclear_responses: int,
) -> bool:
    """
    Determine if call should be escalated to human agent.

    Args:
        guardrail_triggers: Number of guardrail violations
        retry_count: Number of retry attempts on current step
        unclear_responses: Number of unclear responses total

    Returns:
        True if should escalate
    """
    if guardrail_triggers >= 3:
        logger.info("[Guardrail] Escalate: Too many guardrail triggers")
        return True
    if retry_count >= 5:
        logger.info("[Guardrail] Escalate: Too many retries")
        return True
    if unclear_responses >= 8:
        logger.info("[Guardrail] Escalate: Too many unclear responses")
        return True
    return False


# =============================================================================
# Barge-In Handling
# =============================================================================


def handle_barge_in(
    transcript: str,
    pending_slot_name: str,
    pending_slot_type: str,
) -> dict:
    """
    Handle user interruption (barge-in).

    If the interrupted text contains an answer to the pending question,
    we should advance. Otherwise, restate the question.

    Args:
        transcript: What the user said during interruption
        pending_slot_name: The slot we were trying to collect
        pending_slot_type: Type of the slot

    Returns:
        Dict with 'action' (advance/restate) and 'extracted_value' if any
    """
    from app.helpers.slot_extractor import extract_slots

    # Try to extract the pending slot from the transcript
    slot_defs = [{"name": pending_slot_name, "type": pending_slot_type}]
    extracted = extract_slots(transcript, slot_defs)

    value = extracted.get(pending_slot_name)
    if value is not None:
        logger.info(
            f"[Barge-In] User answered during interruption: {pending_slot_name}={value}"
        )
        return {
            "action": "advance",
            "extracted_value": value,
            "message": "用户在打断时回答了问题",
        }

    logger.info(f"[Barge-In] User interrupted but did not answer: {transcript[:30]}")
    return {
        "action": "restate",
        "message": "需要重新提问",
    }
