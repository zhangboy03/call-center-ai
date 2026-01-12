"""
Observability Module

Per-turn logging and metrics for call center AI.
Phase 5: Structured logging, latency tracking, and config knobs.
"""

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Config Knobs
# =============================================================================

CONFIG = {
    "max_turns_per_step": 5,  # Max retries before escalation
    "intent_confidence_threshold": 0.7,  # Min confidence to trust intent
    "use_fast_model_for_empathy": True,  # Use qwen-turbo for empathy responses
    "log_slot_deltas": True,  # Log slot value changes
    "log_latencies": True,  # Log per-component latencies
    "enable_rag_cache": True,  # Cache RAG results
    "rag_cache_ttl_minutes": 30,  # RAG cache TTL
    "trace_enabled": True,  # Enable turn event tracing (Phase 0)
    "trace_dir": "traces",  # Directory for JSONL trace files
}


def get_config(key: str, default: Any = None) -> Any:
    """Get config value."""
    return CONFIG.get(key, default)


def set_config(key: str, value: Any):
    """Set config value at runtime."""
    CONFIG[key] = value
    logger.info(f"[Config] Set {key} = {value}")


# =============================================================================
# Turn Event Tracing (Phase 0)
# =============================================================================


def _hash_prompt(prompt: str) -> str:
    """Hash system prompt for deterministic comparison."""
    return hashlib.sha256(prompt.encode()).hexdigest()[:12]


@dataclass
class TurnEvent:
    """
    Structured turn event for replay and evaluation.

    Captures all decision-relevant data for a single conversation turn,
    enabling offline replay without re-calling external services.
    """

    # Identity
    call_id: str
    turn_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Input
    user_text: str = ""

    # Decisions (for replay)
    is_question: bool = False
    user_seems_done: bool = False

    # RAG
    rag_query: Optional[str] = None
    rag_results: list[dict] = field(default_factory=list)
    rag_context: str = ""

    # Extraction (cached from async)
    extraction_output: dict[str, Any] = field(default_factory=dict)
    missing_list: list[str] = field(default_factory=list)
    all_collected: bool = False

    # Prompt construction
    missing_hint: str = ""
    system_prompt_hash: str = ""

    # SOP Step Tracking (Phase 2A)
    current_step_id: str = ""  # e.g., "intro", "symptom_discussion", etc.
    step_transition: Optional[str] = None  # e.g., "intro -> symptom_discussion"
    sop_retry_count: int = 0

    # Latencies (ms)
    asr_latency_ms: int = 0
    rag_latency_ms: int = 0
    llm_first_token_ms: int = 0
    llm_total_ms: int = 0
    tts_total_ms: int = 0

    # Output
    ai_response: str = ""

    def to_dict(self) -> dict:
        """Convert to dict for JSON serialization."""
        return {
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "ts": self.timestamp,
            "user_text": self.user_text,
            "decisions": {
                "is_question": self.is_question,
                "user_seems_done": self.user_seems_done,
            },
            "rag": {
                "query": self.rag_query,
                "results": self.rag_results,
                "context_len": len(self.rag_context),
            },
            "extraction": {
                "output": self.extraction_output,
                "missing": self.missing_list,
                "all_collected": self.all_collected,
            },
            "prompt": {
                "missing_hint_type": "goodbye"
                if "告别" in self.missing_hint
                else "complete"
                if "收集完毕" in self.missing_hint
                else "pending"
                if self.missing_hint
                else "none",
                "system_prompt_hash": self.system_prompt_hash,
            },
            "latency_ms": {
                "asr": self.asr_latency_ms,
                "rag": self.rag_latency_ms,
                "llm_first_token": self.llm_first_token_ms,
                "llm_total": self.llm_total_ms,
                "tts_total": self.tts_total_ms,
            },
            "sop": {
                "current_step": self.current_step_id,
                "transition": self.step_transition,
                "retry_count": self.sop_retry_count,
            },
            "ai_response": self.ai_response[:100] + "..."
            if len(self.ai_response) > 100
            else self.ai_response,
        }


# Trace file handles (per call_id)
_trace_files: dict[str, Any] = {}


def _get_trace_file(call_id: str):
    """Get or create trace file for a call."""
    if call_id not in _trace_files:
        trace_dir = Path(get_config("trace_dir", "traces"))
        trace_dir.mkdir(exist_ok=True)

        # Use call_id and timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = trace_dir / f"call_{call_id}_{timestamp}.jsonl"
        _trace_files[call_id] = open(filepath, "a", encoding="utf-8")
        logger.info(f"[Trace] Created trace file: {filepath}")

    return _trace_files[call_id]


def log_turn_event(event: TurnEvent):
    """
    Log a turn event to JSONL file.

    Only logs if TRACE_ENABLED is True (can be toggled via env var or config).
    """
    # Check env var override first
    trace_enabled = os.environ.get("TRACE_ENABLED", "").lower() not in (
        "0",
        "false",
        "no",
    )
    if not trace_enabled:
        trace_enabled = get_config("trace_enabled", True)

    if not trace_enabled:
        return

    try:
        f = _get_trace_file(event.call_id)
        f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
        f.flush()  # Ensure immediate write
        logger.debug(f"[Trace] Logged turn {event.turn_id} for call {event.call_id}")
    except Exception as e:
        logger.error(f"[Trace] Failed to log event: {e}")


def close_trace_file(call_id: str):
    """Close trace file when call ends."""
    if call_id in _trace_files:
        try:
            _trace_files[call_id].close()
            del _trace_files[call_id]
            logger.info(f"[Trace] Closed trace file for call {call_id}")
        except Exception as e:
            logger.error(f"[Trace] Failed to close file: {e}")


# =============================================================================
# Turn Snapshot for Logging
# =============================================================================


@dataclass
class TurnSnapshot:
    """Snapshot of a single conversation turn for logging."""

    call_id: str
    turn_id: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    # Input
    user_text: str = ""

    # Classification
    intent: str = ""
    policy: str = ""

    # Extraction
    slot_deltas: dict[str, Any] = field(default_factory=dict)

    # Safety
    guardrail_event: Optional[str] = None

    # RAG
    rag_used: bool = False
    rag_query: str = ""

    # Latencies (ms)
    asr_latency: int = 0
    llm_latency: int = 0
    tts_latency: int = 0
    total_latency: int = 0

    # Current state
    sop_step: str = ""
    sop_action: str = ""

    def to_dict(self) -> dict:
        return {
            "call_id": self.call_id,
            "turn_id": self.turn_id,
            "ts": self.timestamp,
            "user_text": self.user_text[:50] + "..."
            if len(self.user_text) > 50
            else self.user_text,
            "intent": self.intent,
            "policy": self.policy,
            "slots": self.slot_deltas,
            "guardrail": self.guardrail_event,
            "rag": self.rag_used,
            "step": self.sop_step,
            "action": self.sop_action,
            "latency_ms": {
                "asr": self.asr_latency,
                "llm": self.llm_latency,
                "tts": self.tts_latency,
                "total": self.total_latency,
            },
        }

    def log(self):
        """Log this turn snapshot."""
        log_data = self.to_dict()
        logger.info(f"[Turn] {json.dumps(log_data, ensure_ascii=False)}")


# =============================================================================
# Latency Timer
# =============================================================================


class LatencyTimer:
    """Context manager for measuring latency."""

    def __init__(self, name: str):
        self.name = name
        self.start_time: float = 0
        self.elapsed_ms: int = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start_time
        self.elapsed_ms = int(elapsed * 1000)
        if get_config("log_latencies"):
            logger.debug(f"[Latency] {self.name}: {self.elapsed_ms}ms")


# =============================================================================
# Call Metrics Accumulator
# =============================================================================


@dataclass
class CallMetrics:
    """Accumulated metrics for a call."""

    call_id: str
    start_time: datetime = field(default_factory=datetime.now)

    total_turns: int = 0
    total_asr_ms: int = 0
    total_llm_ms: int = 0
    total_tts_ms: int = 0

    guardrail_triggers: int = 0
    rag_queries: int = 0
    slot_extractions: int = 0

    intent_counts: dict[str, int] = field(default_factory=dict)

    def add_turn(self, snapshot: TurnSnapshot):
        """Add a turn's metrics."""
        self.total_turns += 1
        self.total_asr_ms += snapshot.asr_latency
        self.total_llm_ms += snapshot.llm_latency
        self.total_tts_ms += snapshot.tts_latency

        if snapshot.guardrail_event:
            self.guardrail_triggers += 1
        if snapshot.rag_used:
            self.rag_queries += 1
        if snapshot.slot_deltas:
            self.slot_extractions += 1

        # Count intents
        intent = snapshot.intent
        self.intent_counts[intent] = self.intent_counts.get(intent, 0) + 1

    def summary(self) -> dict:
        """Get summary metrics."""
        avg_asr = self.total_asr_ms // max(self.total_turns, 1)
        avg_llm = self.total_llm_ms // max(self.total_turns, 1)
        avg_tts = self.total_tts_ms // max(self.total_turns, 1)

        return {
            "call_id": self.call_id,
            "duration_sec": (datetime.now() - self.start_time).total_seconds(),
            "turns": self.total_turns,
            "avg_latency_ms": {
                "asr": avg_asr,
                "llm": avg_llm,
                "tts": avg_tts,
                "total": avg_asr + avg_llm + avg_tts,
            },
            "guardrails": self.guardrail_triggers,
            "rag_queries": self.rag_queries,
            "extractions": self.slot_extractions,
            "intents": self.intent_counts,
        }

    def log_summary(self):
        """Log call summary."""
        logger.info(f"[CallSummary] {json.dumps(self.summary(), ensure_ascii=False)}")
