"""
State Models for Call Center AI

Defines state enums and dataclasses for call flow and SOP tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional


class CallState(Enum):
    """State of a phone call."""

    DIALING = "dialing"  # Call initiated, waiting for answer
    CONNECTED = "connected"  # Call answered, pre-SOP (identity verification)
    IN_SOP = "in_sop"  # Actively collecting SOP data
    ENDED = "ended"  # Call ended normally
    FAILED = "failed"  # Call failed (no answer, dropped, etc.)


class IntentLabel(Enum):
    """Classified intent of user utterance."""

    ANSWER_CURRENT_QUESTION = "answer"  # Direct answer to SOP question
    SILENCE_OR_FILLER = "filler"  # "嗯", "好", "行" - no real content
    ASK_SIDE_QUESTION = "side_question"  # General info question (RAG)
    COMPLAINT_OR_EMOTION = "complaint"  # Expressing frustration/emotion
    ASK_MEDICAL_ADVICE = "medical"  # Requesting medical guidance
    CHITCHAT_OR_OTHER = "chitchat"  # Off-topic chat
    END_CALL = "end_call"  # Wants to end call
    UNCLEAR = "unclear"  # Cannot determine intent


class PolicyDecision(Enum):
    """What the agent should do after intent classification."""

    EXTRACT_AND_ADVANCE = "extract_advance"  # Extract slots, may advance SOP
    EXTRACT_NO_ADVANCE = "extract_no_advance"  # Extract slots, stay on step
    EMPATHY_RESPONSE = "empathy"  # Short empathetic reply
    RAG_RESPONSE = "rag"  # Answer from knowledge base
    GUARDRAIL_RESPONSE = "guardrail"  # Safety response
    CONTINUE_PROMPT = "continue"  # Just prompt to continue
    END_CALL = "end_call"  # Close the call


@dataclass
class SOPStepState:
    """Current state within SOP flow."""

    current_step_id: str
    filled_slots: dict[str, Any] = field(default_factory=dict)
    unfilled_slots: list[str] = field(default_factory=list)
    step_completed: bool = False
    retry_count: int = 0
    last_prompt: str = ""

    def mark_slot_filled(self, slot_name: str, value: Any):
        """Mark a slot as filled with a value."""
        self.filled_slots[slot_name] = value
        if slot_name in self.unfilled_slots:
            self.unfilled_slots.remove(slot_name)

    def to_dict(self) -> dict:
        return {
            "current_step_id": self.current_step_id,
            "filled_slots": self.filled_slots,
            "unfilled_slots": self.unfilled_slots,
            "step_completed": self.step_completed,
            "retry_count": self.retry_count,
            "last_prompt": self.last_prompt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SOPStepState":
        return cls(
            current_step_id=data.get("current_step_id", "intro"),
            filled_slots=data.get("filled_slots", {}),
            unfilled_slots=data.get("unfilled_slots", []),
            step_completed=data.get("step_completed", False),
            retry_count=data.get("retry_count", 0),
            last_prompt=data.get("last_prompt", ""),
        )


@dataclass
class TurnContext:
    """Context for a single conversation turn."""

    call_id: str
    turn_id: int
    call_state: CallState
    sop_state: SOPStepState
    recent_transcript: str

    # Filled during processing
    intent_label: IntentLabel = IntentLabel.UNCLEAR
    policy_decision: PolicyDecision = PolicyDecision.CONTINUE_PROMPT
    extracted_slots: dict[str, Any] = field(default_factory=dict)
    guardrail_triggered: Optional[str] = None
    rag_used: bool = False

    # Timing (for latency tracking)
    timestamp: datetime = field(default_factory=datetime.now)
    asr_latency_ms: int = 0
    llm_latency_ms: int = 0
    tts_latency_ms: int = 0

    def to_log_dict(self) -> dict:
        """Return dict for structured logging."""
        return {
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "call_state": self.call_state.value,
            "step": self.sop_state.current_step_id,
            "intent": self.intent_label.value,
            "policy": self.policy_decision.value,
            "extracted": self.extracted_slots,
            "guardrail": self.guardrail_triggered,
            "rag": self.rag_used,
            "latency": {
                "asr": self.asr_latency_ms,
                "llm": self.llm_latency_ms,
                "tts": self.tts_latency_ms,
            },
        }
