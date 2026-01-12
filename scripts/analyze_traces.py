#!/usr/bin/env python3
"""
Trace Analysis Script - Phase 3 Offline Evaluation

Analyzes recorded traces to compute metrics:
- Slot completion rate
- Turns to completion
- Step transitions
- Decision patterns
"""

import json
import sys
from pathlib import Path


def load_traces(trace_dir: Path) -> list[dict]:
    """Load all trace files from directory."""
    traces = []
    for jsonl_file in sorted(trace_dir.glob("*.jsonl")):
        call_data = {"file": jsonl_file.name, "turns": []}
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    call_data["turns"].append(json.loads(line))
        if call_data["turns"]:
            call_data["call_id"] = call_data["turns"][0].get("call_id", "unknown")
            traces.append(call_data)
    return traces


def analyze_call(call_data: dict) -> dict:
    """Analyze a single call's trace."""
    turns = call_data["turns"]
    if not turns:
        return {}

    # Basic metrics
    num_turns = len(turns)

    # Slot completion
    final_turn = turns[-1]
    extraction = final_turn.get("extraction", {})
    extracted = extraction.get("output", {})
    missing = extraction.get("missing", [])
    all_collected = extraction.get("all_collected", False)

    slots_filled = len([v for v in extracted.values() if v is not None])
    slots_total = len(extracted)

    # Decision patterns
    question_turns = sum(1 for t in turns if t.get("decisions", {}).get("is_question"))
    rag_triggered = sum(1 for t in turns if t.get("rag", {}).get("query"))

    # Step tracking
    steps_seen = []
    transitions = []
    for t in turns:
        sop = t.get("sop", {})
        step = sop.get("current_step", "")
        if step and (not steps_seen or steps_seen[-1] != step):
            if steps_seen:
                transitions.append(f"{steps_seen[-1]} -> {step}")
            steps_seen.append(step)

    # Latencies
    llm_latencies = [t.get("latency_ms", {}).get("llm_first_token", 0) for t in turns]
    avg_latency = sum(llm_latencies) / len(llm_latencies) if llm_latencies else 0

    return {
        "call_id": call_data["call_id"],
        "num_turns": num_turns,
        "slots_filled": slots_filled,
        "slots_total": slots_total,
        "slot_completion_pct": round(100 * slots_filled / max(slots_total, 1), 1),
        "all_collected": all_collected,
        "question_turns": question_turns,
        "rag_triggered": rag_triggered,
        "steps_seen": steps_seen,
        "transitions": transitions,
        "avg_llm_first_token_ms": round(avg_latency),
        "missing_slots": missing[:5],  # First 5 for brevity
    }


def print_summary(analyses: list[dict]) -> None:
    """Print summary report."""
    print("\n" + "=" * 60)
    print("📊 TRACE ANALYSIS REPORT")
    print("=" * 60)

    total_calls = len(analyses)
    completed_calls = sum(1 for a in analyses if a.get("all_collected"))
    avg_turns = sum(a.get("num_turns", 0) for a in analyses) / max(total_calls, 1)
    avg_slot_pct = sum(a.get("slot_completion_pct", 0) for a in analyses) / max(
        total_calls, 1
    )

    print(f"\n📞 Total Calls: {total_calls}")
    print(
        f"✅ Completed: {completed_calls} ({100 * completed_calls // max(total_calls, 1)}%)"
    )
    print(f"📝 Avg Turns: {avg_turns:.1f}")
    print(f"📋 Avg Slot Completion: {avg_slot_pct:.1f}%")

    print("\n" + "-" * 60)
    print("Per-Call Details:")
    print("-" * 60)

    for a in analyses:
        status = "✅" if a.get("all_collected") else "⏳"
        print(f"\n{status} Call {a['call_id']}")
        print(
            f"   Turns: {a['num_turns']}, Slots: {a['slots_filled']}/{a['slots_total']} ({a['slot_completion_pct']}%)"
        )
        print(f"   Questions: {a['question_turns']}, RAG: {a['rag_triggered']}")
        print(f"   Avg LLM latency: {a['avg_llm_first_token_ms']}ms")
        if a.get("steps_seen"):
            print(f"   Steps: {' → '.join(a['steps_seen'][:5])}")
        if a.get("missing_slots"):
            print(f"   Missing: {', '.join(a['missing_slots'][:3])}")

    print("\n" + "=" * 60)


def main():
    trace_dir = Path(__file__).parent.parent / "traces"

    if len(sys.argv) > 1:
        trace_dir = Path(sys.argv[1])

    if not trace_dir.exists():
        print(f"❌ Trace directory not found: {trace_dir}")
        sys.exit(1)

    print(f"📂 Loading traces from: {trace_dir}")
    traces = load_traces(trace_dir)

    if not traces:
        print("❌ No trace files found")
        sys.exit(1)

    print(f"📝 Found {len(traces)} call traces")

    analyses = [analyze_call(call) for call in traces]
    print_summary(analyses)


if __name__ == "__main__":
    main()
