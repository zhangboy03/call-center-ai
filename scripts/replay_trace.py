#!/usr/bin/env python3
"""
Replay Trace Script - Phase 3 Offline Evaluation

Replays a recorded trace through the LangGraph planner to verify decisions match.
Used for A/B testing planner changes without live calls.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.helpers.langgraph_planner import run_turn_planner


async def replay_turn(
    turn_data: dict,
    messages: list[dict],
    cached_extraction: dict,
) -> dict:
    """Replay a single turn through the LangGraph planner."""

    # Reconstruct state from trace
    user_text = turn_data.get("user_text", "")
    call_id = turn_data.get("call_id", "replay")

    # Use extraction from trace
    extraction = turn_data.get("extraction", {})
    cached_extraction = {
        "extracted": extraction.get("output", {}),
        "missing": extraction.get("missing", []),
        "all_collected": extraction.get("all_collected", False),
    }

    # Simple base prompt for replay
    base_system_prompt = "你是品驰关爱中心的客服。"

    # Run through planner
    result = await run_turn_planner(
        call_id=call_id,
        user_text=user_text,
        messages=messages,
        cached_extraction=cached_extraction,
        base_system_prompt=base_system_prompt,
        executor=None,
    )

    return {
        "original": {
            "is_question": turn_data.get("decisions", {}).get("is_question"),
            "rag_triggered": bool(turn_data.get("rag", {}).get("query")),
        },
        "replayed": {
            "is_question": result.get("is_question"),
            "rag_triggered": bool(result.get("rag_query")),
        },
        "match": (
            turn_data.get("decisions", {}).get("is_question")
            == result.get("is_question")
        ),
    }


async def replay_call(trace_file: Path) -> dict:
    """Replay an entire call trace."""

    turns = []
    with open(trace_file) as f:
        for line in f:
            if line.strip():
                turns.append(json.loads(line))

    if not turns:
        return {"error": "No turns in trace"}

    call_id = turns[0].get("call_id", "unknown")
    print(f"\n🔄 Replaying call {call_id} ({len(turns)} turns)")

    messages = []
    cached_extraction = {"extracted": {}, "missing": [], "all_collected": False}
    results = []
    matches = 0

    for i, turn_data in enumerate(turns):
        user_text = turn_data.get("user_text", "")
        ai_response = turn_data.get("ai_response", "")

        # Add to message history
        if user_text:
            messages.append({"role": "user", "content": user_text})

        # Replay
        result = await replay_turn(turn_data, messages.copy(), cached_extraction)
        results.append(result)

        match_str = "✅" if result["match"] else "❌"
        print(
            f"  Turn {i + 1}: {match_str} is_question: {result['original']['is_question']} -> {result['replayed']['is_question']}"
        )

        if result["match"]:
            matches += 1

        # Add AI response to history
        if ai_response:
            messages.append({"role": "assistant", "content": ai_response})

    accuracy = 100 * matches / len(turns) if turns else 0
    print(f"  📊 Decision accuracy: {matches}/{len(turns)} ({accuracy:.0f}%)")

    return {
        "call_id": call_id,
        "num_turns": len(turns),
        "matches": matches,
        "accuracy_pct": round(accuracy, 1),
    }


async def main():
    trace_dir = Path(__file__).parent.parent / "traces"

    if len(sys.argv) > 1:
        # Replay specific file
        trace_file = Path(sys.argv[1])
        if not trace_file.exists():
            print(f"❌ File not found: {trace_file}")
            sys.exit(1)
        await replay_call(trace_file)
    else:
        # Replay all
        print("📂 Replaying all traces...")
        total_matches = 0
        total_turns = 0

        for trace_file in sorted(trace_dir.glob("*.jsonl")):
            result = await replay_call(trace_file)
            if "matches" in result:
                total_matches += result["matches"]
                total_turns += result["num_turns"]

        if total_turns:
            print(f"\n{'=' * 50}")
            print(
                f"📊 Overall Decision Accuracy: {total_matches}/{total_turns} ({100 * total_matches / total_turns:.0f}%)"
            )


if __name__ == "__main__":
    asyncio.run(main())
