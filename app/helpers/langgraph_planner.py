"""
LangGraph Turn Planner (Phase 1)

Wraps current process_text() prompt-building logic in a LangGraph graph.
Does NOT change behavior - reproduces current decisions for traceability.

Graph flow:
    QuestionDetect -> RAGNode (if question) -> ExtractionCheck -> PromptBuilder

Usage:
    from app.helpers.langgraph_planner import run_turn_planner

    result = await run_turn_planner(
        call_id=call_id,
        user_text=user_text,
        messages=messages,
        cached_extraction=cached_extraction,
        base_system_prompt=base_system_prompt,
        executor=executor,
    )

    # result contains:
    # - system_prompt: str (ready for LLM)
    # - is_question: bool
    # - rag_context: str
    # - missing_hint: str
    # - all_collected: bool
"""

import asyncio
import logging
import os
from concurrent.futures import Executor
from typing import Any, Optional

from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


# =============================================================================
# Planner State
# =============================================================================


class PlannerState(TypedDict):
    """State passed between graph nodes."""

    # Input
    call_id: str
    user_text: str
    messages: list[dict]
    cached_extraction: dict[str, Any]
    base_system_prompt: str

    # Decisions
    is_question: bool
    user_seems_done: bool

    # RAG
    rag_query: Optional[str]
    rag_results: list[dict]
    rag_context: str

    # Extraction (from cache)
    missing: list[str]
    all_collected: bool

    # Prompt
    missing_hint: str
    system_prompt: str


# =============================================================================
# Graph Nodes
# =============================================================================


def question_detect_node(state: PlannerState) -> PlannerState:
    """
    Detect if user text is a question that needs RAG.

    Replicates logic from streaming_routes.py:298-301
    """
    user_text = state["user_text"]

    # Same heuristic as current code
    is_question = "？" in user_text or any(
        w in user_text
        for w in ["吗", "怎么", "是不是", "为什么", "什么", "能不能", "可以", "哪"]
    )

    # Detect if user seems done
    user_done_signals = [
        "没了",
        "没有了",
        "好的",
        "行",
        "嗯",
        "谢谢",
        "再见",
        "拜拜",
        "没问题",
        "好",
    ]
    user_seems_done = (
        user_text.strip() in user_done_signals or len(user_text.strip()) <= 3
    )

    logger.debug(
        f"[Planner] QuestionDetect: is_question={is_question}, user_seems_done={user_seems_done}"
    )

    return {
        **state,
        "is_question": is_question,
        "user_seems_done": user_seems_done,
    }


def rag_node(state: PlannerState) -> PlannerState:
    """
    Query RAG if this is a question.

    Note: This is called synchronously. The actual streaming_routes.py
    runs RAG in executor, but for the planner we keep it simple.
    """
    if not state["is_question"]:
        return {
            **state,
            "rag_query": None,
            "rag_results": [],
            "rag_context": "",
        }

    user_text = state["user_text"]

    try:
        from app.helpers.rag import rag_service

        results = rag_service.search(user_text, top_k=2, threshold=0.45)

        rag_context = ""
        if results:
            logger.info(
                f"[Planner] RAG found {len(results)} hits for: {user_text[:30]}"
            )
            rag_context = "\n\n【知识库参考】\n"
            for res in results:
                rag_context += f"- {res['topic']}: {res['answer'][:100]}\n"

        return {
            **state,
            "rag_query": user_text,
            "rag_results": [
                {"topic": r.get("topic", ""), "score": r.get("score", 0)}
                for r in results
            ],
            "rag_context": rag_context,
        }

    except Exception as e:
        logger.error(f"[Planner] RAG error: {e}")
        return {
            **state,
            "rag_query": user_text,
            "rag_results": [],
            "rag_context": "",
        }


def extraction_check_node(state: PlannerState) -> PlannerState:
    """
    Check cached extraction results.

    Does NOT block - just reads from cached_extraction.
    """
    cached = state["cached_extraction"]

    missing = cached.get("missing", [])
    all_collected = cached.get("all_collected", False)

    logger.debug(
        f"[Planner] ExtractionCheck: missing={missing[:3]}, all_collected={all_collected}"
    )

    return {
        **state,
        "missing": missing,
        "all_collected": all_collected,
    }


def prompt_builder_node(state: PlannerState) -> PlannerState:
    """
    Build the system prompt with RAG context and missing hints.

    Replicates logic from streaming_routes.py:337-360
    """
    base_system_prompt = state["base_system_prompt"]
    rag_context = state["rag_context"]
    all_collected = state["all_collected"]
    user_seems_done = state["user_seems_done"]
    is_question = state["is_question"]
    missing = state["missing"]

    # Determine missing_hint (same logic as streaming_routes.py)
    if all_collected and user_seems_done and not is_question:
        # All info collected and user has no more questions - say goodbye!
        missing_hint = """

【重要】信息已全部收集完毕，用户也没有其他问题了。请立即友好告别：
- 感谢对方配合随访
- 提醒有问题可拨打400电话
- 祝对方身体健康
- 说再见

示例："好的，今天的随访就到这里，感谢您的配合！后续有任何问题可以拨打我们的400热线。祝您身体健康，再见！"
"""
        logger.info("[Planner] Triggering goodbye!")
    elif all_collected:
        # Info collected but user might have questions - answer then close
        missing_hint = "\n\n【温馨提示】信息已收集完毕。如果用户问了问题就简短回答，然后友好告别。如果没有问题，就直接告别。"
    elif missing:
        missing_hint = f"\n\n【温馨提示】以下信息还没聊到，找合适的时机自然提起：{', '.join(missing[:3])}"
    else:
        missing_hint = ""

    # Build final prompt
    system_prompt = f"""{base_system_prompt}
{rag_context}
{missing_hint}"""

    logger.debug(
        f"[Planner] PromptBuilder: hint_type={'goodbye' if '告别' in missing_hint else 'complete' if '收集完毕' in missing_hint else 'pending' if missing_hint else 'none'}"
    )

    return {
        **state,
        "missing_hint": missing_hint,
        "system_prompt": system_prompt,
    }


# =============================================================================
# Graph Definition
# =============================================================================


def _should_query_rag(state: PlannerState) -> str:
    """Router: decide whether to query RAG."""
    if state["is_question"]:
        return "rag"
    return "extraction_check"


def create_planner_graph() -> StateGraph:
    """
    Create the LangGraph turn planner.

    Flow:
        question_detect -> (if question) -> rag -> extraction_check -> prompt_builder
                       -> (if not question) -> extraction_check -> prompt_builder
    """
    graph = StateGraph(PlannerState)

    # Add nodes
    graph.add_node("question_detect", question_detect_node)
    graph.add_node("rag", rag_node)
    graph.add_node("extraction_check", extraction_check_node)
    graph.add_node("prompt_builder", prompt_builder_node)

    # Add edges
    graph.set_entry_point("question_detect")

    # Conditional: RAG or skip to extraction
    graph.add_conditional_edges(
        "question_detect",
        _should_query_rag,
        {
            "rag": "rag",
            "extraction_check": "extraction_check",
        },
    )

    graph.add_edge("rag", "extraction_check")
    graph.add_edge("extraction_check", "prompt_builder")
    graph.add_edge("prompt_builder", END)

    return graph.compile()


# Singleton compiled graph
_planner_graph = None


def get_planner_graph():
    """Get or create the singleton planner graph."""
    global _planner_graph
    if _planner_graph is None:
        _planner_graph = create_planner_graph()
        logger.info("[Planner] LangGraph turn planner initialized")
    return _planner_graph


# =============================================================================
# Public API
# =============================================================================


class PlannerResult(TypedDict):
    """Result from run_turn_planner."""

    system_prompt: str
    is_question: bool
    user_seems_done: bool
    rag_query: Optional[str]
    rag_results: list[dict]
    rag_context: str
    missing: list[str]
    all_collected: bool
    missing_hint: str


async def run_turn_planner(
    call_id: str,
    user_text: str,
    messages: list[dict],
    cached_extraction: dict[str, Any],
    base_system_prompt: str,
    executor: Optional[Executor] = None,
) -> PlannerResult:
    """
    Run the LangGraph turn planner.

    Args:
        call_id: Current call ID
        user_text: User's input text
        messages: Conversation history
        cached_extraction: Cached extraction results from async extractor
        base_system_prompt: Base system prompt
        executor: Thread pool executor (for RAG, if needed)

    Returns:
        PlannerResult with system_prompt and decision metadata
    """
    graph = get_planner_graph()

    initial_state: PlannerState = {
        "call_id": call_id,
        "user_text": user_text,
        "messages": messages,
        "cached_extraction": cached_extraction,
        "base_system_prompt": base_system_prompt,
        # These will be filled by nodes
        "is_question": False,
        "user_seems_done": False,
        "rag_query": None,
        "rag_results": [],
        "rag_context": "",
        "missing": [],
        "all_collected": False,
        "missing_hint": "",
        "system_prompt": "",
    }

    # Run graph (sync for now - RAG is the only potentially slow op)
    # TODO: If latency is an issue, run RAG in executor
    if executor:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, graph.invoke, initial_state)
    else:
        result = graph.invoke(initial_state)

    return PlannerResult(
        system_prompt=result["system_prompt"],
        is_question=result["is_question"],
        user_seems_done=result["user_seems_done"],
        rag_query=result["rag_query"],
        rag_results=result["rag_results"],
        rag_context=result["rag_context"],
        missing=result["missing"],
        all_collected=result["all_collected"],
        missing_hint=result["missing_hint"],
    )


def is_planner_enabled() -> bool:
    """Check if LangGraph planner is enabled."""
    env_val = os.environ.get("USE_LANGGRAPH_PLANNER", "").lower()
    return env_val in ("1", "true", "yes")
