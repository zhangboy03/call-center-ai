"""
SOP State Machine Engine

Manages the postoperative follow-up call flow, slot collection, and state persistence.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import redis
import yaml

logger = logging.getLogger(__name__)


@dataclass
class SlotDef:
    """Definition of a slot to collect."""

    name: str
    type: str  # yes_no, int_0_10, enum, multi_select, text, fuzzy_range, bool
    required: bool = True
    options: list[str] = None
    map_to_5: bool = False
    allow_unknown: bool = False

    def __post_init__(self):
        if self.options is None:
            self.options = []


@dataclass
class StepDef:
    """Definition of an SOP step."""

    id: str
    prompt: str
    slots: list[SlotDef] = field(default_factory=list)
    next: list[dict] = field(default_factory=list)
    terminal: bool = False
    status: Optional[str] = None


@dataclass
class CallState:
    """Current state of a call."""

    call_id: str
    patient_id: str
    product_line: str = "DBS_PD"
    current_step: str = "intro"
    slots_collected: dict[str, Any] = field(default_factory=dict)
    conversation_history: list[dict] = field(default_factory=list)
    retry_count: int = 0
    started_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    final_status: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "patient_id": self.patient_id,
            "product_line": self.product_line,
            "current_step": self.current_step,
            "slots_collected": self.slots_collected,
            "conversation_history": self.conversation_history,
            "retry_count": self.retry_count,
            "started_at": self.started_at.isoformat(),
            "completed": self.completed,
            "final_status": self.final_status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CallState":
        data["started_at"] = datetime.fromisoformat(data["started_at"])
        return cls(**data)


class SOPEngine:
    """
    State machine for SOP-driven calls.

    Usage:
        engine = SOPEngine(call_id="123", patient_id="P001", product_line="DBS_PD")
        prompt = engine.get_current_prompt()
        # ... user responds ...
        action = engine.process_response(user_text, extracted_slots)
        # action tells you what to do next (advance, retry, complete, etc.)
    """

    def __init__(
        self,
        call_id: str,
        patient_id: str,
        product_line: str = "DBS_PD",
        redis_client: Optional[redis.Redis] = None,
        schema_path: str = "app/resources/sop_schema.yaml",
    ):
        self.call_id = call_id
        self.patient_id = patient_id
        self.product_line = product_line
        self.redis = redis_client
        self.schema_path = Path(schema_path)

        # Load schema
        self.schema = self._load_schema()
        self.steps: dict[str, StepDef] = self._parse_steps()
        self.product_symptoms = self._get_product_symptoms()

        # Initialize or restore state
        self.state = self._load_or_create_state()

    def _load_schema(self) -> dict:
        """Load SOP schema from YAML."""
        if not self.schema_path.exists():
            logger.warning(
                f"Schema not found at {self.schema_path}, using empty schema"
            )
            return {}
        with open(self.schema_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _parse_steps(self) -> dict[str, StepDef]:
        """Parse flow steps from schema."""
        steps = {}
        for step_data in self.schema.get("flow", []):
            slots = [
                SlotDef(
                    name=s["name"],
                    type=s["type"],
                    required=s.get("required", True),
                    options=s.get("options", []),
                    map_to_5=s.get("map_to_5", False),
                    allow_unknown=s.get("allow_unknown", False),
                )
                for s in step_data.get("slots", [])
            ]
            steps[step_data["id"]] = StepDef(
                id=step_data["id"],
                prompt=step_data["prompt"],
                slots=slots,
                next=step_data.get("next", []),
                terminal=step_data.get("terminal", False),
                status=step_data.get("status"),
            )
        return steps

    def _get_product_symptoms(self) -> list[str]:
        """Get symptom tags for current product line."""
        products = self.schema.get("product_lines", {})
        if self.product_line in products:
            return products[self.product_line].get("symptom_tags", [])
        return []

    def _load_or_create_state(self) -> CallState:
        """Load state from Redis or create new."""
        if self.redis:
            key = f"sop:call:{self.call_id}"
            data = self.redis.get(key)
            if data:
                logger.info(f"[SOP] Restored state for call {self.call_id}")
                return CallState.from_dict(json.loads(data))

        # Create new state
        return CallState(
            call_id=self.call_id,
            patient_id=self.patient_id,
            product_line=self.product_line,
        )

    def save_state(self):
        """Persist state to Redis."""
        if self.redis:
            key = f"sop:call:{self.call_id}"
            self.redis.setex(
                key,
                86400 * 7,  # 7 days TTL
                json.dumps(self.state.to_dict(), ensure_ascii=False),
            )
            logger.info(f"[SOP] Saved state for call {self.call_id}")

    def get_current_step(self) -> Optional[StepDef]:
        """Get current step definition."""
        return self.steps.get(self.state.current_step)

    def get_current_prompt(self, patient_name: str = "") -> str:
        """
        Get the prompt for the current step.

        Args:
            patient_name: Patient name for personalization

        Returns:
            Formatted prompt string
        """
        step = self.get_current_step()
        if not step:
            return ""

        prompt = step.prompt
        # Template substitution
        prompt = prompt.replace("{patient_name}", patient_name)
        prompt = prompt.replace(
            "{product_symptoms}", "、".join(self.product_symptoms[:3])
        )

        return prompt

    def get_pending_slots(self) -> list[SlotDef]:
        """Get slots not yet collected for current step."""
        step = self.get_current_step()
        if not step:
            return []

        pending = []
        for slot in step.slots:
            if slot.name not in self.state.slots_collected:
                pending.append(slot)
            elif slot.required and self.state.slots_collected.get(slot.name) is None:
                pending.append(slot)

        return pending

    def get_all_required_slots(self) -> list[str]:
        """Get all required slot names across all steps."""
        required = []
        for step in self.steps.values():
            for slot in step.slots:
                if slot.required:
                    required.append(slot.name)
        return required

    def is_complete(self) -> bool:
        """Check if all required slots are filled."""
        required = self.get_all_required_slots()
        for slot_name in required:
            if slot_name not in self.state.slots_collected:
                return False
            if self.state.slots_collected[slot_name] is None:
                return False
        return True

    def process_response(
        self,
        user_text: str,
        extracted_slots: dict[str, Any],
    ) -> dict:
        """
        Process user response and determine next action.

        Args:
            user_text: Raw user transcript
            extracted_slots: Slots extracted from the response

        Returns:
            Action dict with keys: action (advance/retry/complete/escalate), next_step, message
        """
        step = self.get_current_step()
        if not step:
            return {"action": "error", "message": "No current step"}

        # Record conversation
        self.state.conversation_history.append(
            {"role": "user", "text": user_text, "step": self.state.current_step}
        )

        # Store extracted slots
        for slot_name, value in extracted_slots.items():
            if value is not None:
                self.state.slots_collected[slot_name] = value
                logger.info(f"[SOP] Collected slot {slot_name} = {value}")

        # Check if we have all required slots for this step
        pending = self.get_pending_slots()
        required_pending = [s for s in pending if s.required]

        if required_pending:
            self.state.retry_count += 1
            if self.state.retry_count >= 3:
                # Escalate or use simplified prompt
                return {
                    "action": "escalate",
                    "message": f"无法获取必填信息: {', '.join(s.name for s in required_pending)}",
                    "pending_slots": [s.name for s in required_pending],
                }
            return {
                "action": "retry",
                "message": f"请补充: {', '.join(s.name for s in required_pending)}",
                "pending_slots": [s.name for s in required_pending],
            }

        # All slots collected for this step, determine next
        self.state.retry_count = 0

        if step.terminal:
            self.state.completed = True
            self.state.final_status = step.status or "已完成"
            self.save_state()
            return {
                "action": "complete",
                "status": self.state.final_status,
                "message": "通话完成",
            }

        # Find next step
        next_step_id = self._evaluate_next(step.next)
        if next_step_id:
            self.state.current_step = next_step_id
            self.save_state()
            return {
                "action": "advance",
                "next_step": next_step_id,
                "message": f"进入下一步: {next_step_id}",
            }

        return {"action": "error", "message": "无法确定下一步"}

    def _evaluate_next(self, transitions: list[dict]) -> Optional[str]:
        """Evaluate transition conditions and return next step ID."""
        for trans in transitions:
            condition = trans.get("condition")
            goto = trans.get("goto")

            if not condition:
                # Unconditional transition
                return goto

            # Simple condition evaluation
            # Format: "slot_name == value" or "slot_name contains 'value'"
            if "==" in condition:
                parts = condition.split("==")
                slot_name = parts[0].strip()
                expected = parts[1].strip().strip("'\"")
                actual = self.state.slots_collected.get(slot_name)
                if str(actual).lower() == expected.lower():
                    return goto
            elif "contains" in condition:
                parts = condition.split("contains")
                slot_name = parts[0].strip()
                needle = parts[1].strip().strip("'\"")
                actual = self.state.slots_collected.get(slot_name, [])
                if isinstance(actual, list) and needle in actual:
                    return goto

        # Default: return first goto without condition
        for trans in transitions:
            if "condition" not in trans or not trans["condition"]:
                return trans.get("goto")

        return None

    def add_note(self, note: str):
        """Add a note to the call record."""
        existing = self.state.slots_collected.get("notes", "")
        self.state.slots_collected["notes"] = f"{existing}\n{note}".strip()
        self.save_state()


# =============================================================================
# Standalone Helper Functions (Phase 1)
# =============================================================================


def is_step_complete(step_schema: StepDef, filled_slots: dict[str, Any]) -> bool:
    """
    Check if all required slots for a step are filled.

    Args:
        step_schema: Step definition with slot requirements
        filled_slots: Dict of slot_name → value

    Returns:
        True if all required slots have non-null values
    """
    for slot in step_schema.slots:
        if not slot.required:
            continue
        value = filled_slots.get(slot.name)
        if value is None:
            return False
    return True


def get_unfilled_slots(
    step_schema: StepDef, filled_slots: dict[str, Any]
) -> list[SlotDef]:
    """
    Get list of unfilled required slots for a step.

    Args:
        step_schema: Step definition with slot requirements
        filled_slots: Dict of slot_name → value

    Returns:
        List of SlotDef objects that are required but unfilled
    """
    unfilled = []
    for slot in step_schema.slots:
        if not slot.required:
            continue
        value = filled_slots.get(slot.name)
        if value is None:
            unfilled.append(slot)
    return unfilled


def infer_final_status(
    sop_state: CallState,
    call_events: Optional[list[dict]] = None,
) -> str:
    """
    Determine the final 随访状态 based on call outcome.

    Logic:
    - If call completed normally → based on collected data
    - If call dropped/failed → 联系失败
    - If emergency escalation → 需要人工介入
    - If complaints → 有问题需处理

    Args:
        sop_state: Current call state with collected slots
        call_events: Optional list of events (escalations, guardrails, etc.)

    Returns:
        Status string: 随访成功, 联系失败, 有问题需跟进, 需人工介入
    """
    events = call_events or []

    # Check for escalation events
    for event in events:
        if event.get("type") == "escalation":
            return "需人工介入"
        if event.get("type") == "emergency":
            return "紧急情况"

    # Check if call completed
    if not sop_state.completed:
        return "联系失败"

    # Check collected data for issues
    slots = sop_state.slots_collected

    # Negative sentiment indicators
    symptom_change = slots.get("symptom_change", "")
    if symptom_change == "有所加重":
        return "有问题需跟进"

    satisfaction = slots.get("programming_satisfaction", "")
    if satisfaction == "不满意":
        return "有问题需跟进"

    # Check for complaints in notes
    notes = slots.get("notes", "")
    if "投诉" in notes or "不满" in notes or "问题" in notes:
        return "有问题需跟进"

    # Normal completion
    return "随访成功"
