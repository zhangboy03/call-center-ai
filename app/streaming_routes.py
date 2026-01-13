"""
语音对话 - 自动 VAD，无需按住说话

实现真实电话体验：
- 自动检测语音开始和结束
- AI 说完后自动切换到听
- 用户可以打断 AI
"""

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import redis
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# Phase 1: LangGraph planner (optional)
from app.helpers.langgraph_planner import is_planner_enabled, run_turn_planner

# Phase 0: Turn event tracing
from app.helpers.observability import (
    TurnEvent,
    _hash_prompt,
    close_trace_file,
    log_turn_event,
)

# Phase 2A: SOPEngine for step tracking (observability only, no behavior change)
from app.helpers.sop_engine import SOPEngine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/streaming", tags=["streaming"])

executor = ThreadPoolExecutor(max_workers=4)

# Redis client for SOP state persistence
try:
    redis_client = redis.Redis(
        host=os.environ.get("REDIS_HOST", "redis"),
        port=int(os.environ.get("REDIS_PORT", 6379)),
        db=int(os.environ.get("REDIS_DB", 0)),
        decode_responses=True,
    )
    redis_client.ping()
    logger.info("[SOP] Redis connected for state persistence")
except Exception as e:
    logger.warning(f"[SOP] Redis not available: {e}. State will not persist.")
    redis_client = None


# Preloaded welcome audio cache (uses synthesize_audio defined below)

_welcome_audio_cache: dict[str, bytes] = {}


def get_preloaded_welcome_audio(patient_name: str) -> bytes:
    """Generate and cache welcome audio for patient name."""
    if patient_name in _welcome_audio_cache:
        return _welcome_audio_cache[patient_name]

    welcome_text = f"您好，我是品驰脑起搏器关爱中心的小驰。{patient_name}您好，您之前在我们这做了手术，现在差不多6个月了，我给您打电话是想做个术后随访。请问您现在方便吗？"
    logger.info(f"[TTS] Generating welcome audio for: {patient_name}")
    audio = synthesize_audio(welcome_text)  # Local function defined at ~line 780
    if audio:
        logger.info(f"[TTS] Cached welcome audio: {len(audio)} bytes")
        _welcome_audio_cache[patient_name] = audio
    return audio


def should_query_rag(text: str) -> bool:
    """Heuristic to decide if an utterance is a question worth querying RAG."""
    if not text:
        return False

    stripped = text.strip()
    question_markers = ["?", "？"]
    question_words = [
        "吗",
        "么",
        "什么",
        "怎么",
        "为何",
        "为何",
        "是否",
        "能否",
        "可以",
        "能不能",
        "可不可以",
        "哪",
        "多少",
        "咨询",
        "想了解",
        "想知道",
    ]

    if any(marker in stripped for marker in question_markers):
        return True

    if any(word in stripped for word in question_words):
        return True

    # Long statements without question cues are likely SOP answers
    if len(stripped) < 6:
        return False

    return False


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket 语音对话 with SOP-driven flow"""
    await websocket.accept()
    logger.info("WebSocket connected")

    # Get patient info from query params
    patient_id = websocket.query_params.get("patient_id", f"P-{str(uuid.uuid4())[:8]}")
    patient_name = websocket.query_params.get("patient_name", "患者")
    product_line = websocket.query_params.get("product_line", "DBS_PD")
    call_id = str(uuid.uuid4())[:8]

    logger.info(f"[Call] Started call {call_id}, patient: {patient_name}")

    # 对话历史
    messages = []

    # Base system prompt with SOP guidance (optimized for low latency)
    base_system_prompt = f"""你是品驰关爱中心的智能客服"小驰"，正在和患者/家属进行术后随访电话。

【核心规则】
- 像朋友一样温暖、自然地聊天
- 一次只问一个问题
- 不要重复问已经问过的问题！检查对话历史避免重复
- 不要连续用相同的开头词（嗯、好的、那）
- 根据对方回答给予适当反馈

【回应方式】
确认词（换着用）：嗯 / 好 / 行 / 是的 / 明白 / 了解
过渡词：那... / 对了... / 顺便问下... / 另外...


【随访问题】按顺序自然问出（跳过已回答的）：
1) 身份确认：是{patient_name}本人或家属吗？→ 对方确认身份后立即跳过！（家属也算确认）
2) 症状恢复：术后感觉怎么样？有改善吗？
3) 控制打分：满分10分，症状控制打几分？
4) 程控调参：做过几次程控？满意吗？
5) 情绪用药：情绪还好吗？吃药有问题吗？
6) 医保费用：什么医保？总费用和自费大概多少？
7) 其他情况：还有其他要反映的吗？
8) 结束：感谢配合，有问题拨打400电话，祝健康！

始终像真人一样自然说话，让患者感到温暖！"""

    # 状态
    is_speaking = False  # AI 是否在说话
    is_processing = False  # 是否正在处理用户输入

    # TTS voice - 龙安洋，成熟男声
    TTS_VOICE = "longanyang"

    # Cached extraction results (for async fire-and-forget extraction)
    cached_extraction = {"extracted": {}, "missing": [], "all_collected": False}
    extraction_in_progress = False

    # Phase 0: Turn counter for tracing
    turn_counter = [0]  # Use list to allow mutation in nested function

    # Phase 2A: SOPEngine for step tracking (observability only)
    # Initialize but do NOT use for control flow - just track current_step_id
    sop_engine = SOPEngine(
        call_id=call_id,
        patient_id=patient_id,
        product_line=product_line,
        redis_client=redis_client,
    )
    logger.info(
        f"[SOP-Trace] Initialized SOPEngine, starting step: {sop_engine.state.current_step}"
    )

    # === Simple VAD with audio buffer (proven working approach) ===
    audio_buffer = bytearray()
    last_speech_time = time.time()
    is_user_speaking = False
    SILENCE_THRESHOLD = 0.6  # User requested 0.6s for word completion
    MIN_SPEECH_DURATION = 0.3  # Minimum speech duration to process
    ENERGY_THRESHOLD = 500  # RMS threshold for speech

    # === Barge-in state (full-duplex support) ===
    barge_in_triggered = False  # Flag to cancel TTS
    barge_in_audio = bytearray()  # Preserve interrupted audio for ASR
    BARGE_IN_THRESHOLD = 600  # RMS threshold during AI speech
    BARGE_IN_DEBOUNCE = 0.3  # Seconds to wait before allowing another barge-in

    # === Persistent ASR for connection reuse ===
    persistent_asr = None  # Will be initialized with hot_words on first use

    def compute_rms(audio_bytes: bytes) -> float:
        """Compute RMS energy of audio (16-bit PCM)"""
        import struct

        if len(audio_bytes) < 2:
            return 0
        samples = struct.unpack(f"<{len(audio_bytes) // 2}h", audio_bytes)
        if not samples:
            return 0
        return (sum(s * s for s in samples) / len(samples)) ** 0.5

    async def send_msg(msg_type: str, **kwargs):
        try:
            await websocket.send_text(json.dumps({"type": msg_type, **kwargs}))
        except Exception as e:
            logger.error("Send error: %s", e)

    async def process_audio(audio_data: bytes):
        """处理用户音频 - ASR + LLM + TTS"""
        nonlocal messages, is_speaking, is_processing

        if len(audio_data) < 1600:
            logger.debug("[ASR] Audio too short: %d bytes", len(audio_data))
            return

        # ASR
        await send_msg("status", text="正在识别...")
        t0 = time.time()

        user_text = await asyncio.get_event_loop().run_in_executor(
            executor, recognize_audio, audio_data
        )

        logger.info("[ASR] %.2fs: %s", time.time() - t0, user_text)

        if not user_text or not user_text.strip():
            await send_msg("status", text="通话中")
            return

        # Process the recognized text with proper cleanup
        try:
            await process_text(user_text)
        except Exception as e:
            logger.error("[Process] Error processing text: %s", e)
        finally:
            # Ensure is_processing is always reset even if process_text errors
            pass  # is_processing is reset inside process_text

    async def process_text(user_text: str):
        """处理用户文本 - 简化版，支持优先"""
        nonlocal messages, is_speaking, is_processing

        if is_processing:
            logger.warning("[Process] Already processing, skipping: %s", user_text[:20])
            return

        is_processing = True

        if not user_text or not user_text.strip():
            is_processing = False
            return

        # Phase 0: Start timing and create turn event

        turn_counter[0] += 1
        turn_event = TurnEvent(
            call_id=call_id,
            turn_id=turn_counter[0],
            user_text=user_text,
            # Phase 2A: Track current SOP step
            current_step_id=sop_engine.state.current_step,
            sop_retry_count=sop_engine.state.retry_count,
        )

        await send_msg("user_text", text=user_text)

        # Add user message to history
        messages.append({"role": "user", "content": user_text})

        # === Backend Extraction: Async Fire-and-Forget ===
        # Use cached results from previous extraction (don't block current turn)
        from app.helpers.history_extractor import extract_from_history, get_missing_info

        nonlocal cached_extraction, extraction_in_progress

        # Use cached results for this turn (from previous async extraction)
        missing = cached_extraction.get("missing", [])
        all_info_collected = cached_extraction.get("all_collected", False)

        # Kick off async extraction for NEXT turn (fire-and-forget)
        turn_count = len([m for m in messages if m["role"] == "user"])

        if turn_count % 2 == 0 and not extraction_in_progress:
            # Define async extraction task
            async def do_extraction():
                nonlocal cached_extraction, extraction_in_progress
                extraction_in_progress = True
                try:
                    extracted = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        extract_from_history,
                        messages.copy(),  # Copy to avoid race
                    )
                    new_missing = get_missing_info(extracted)
                    logger.info(
                        f"[Extraction-Async] Collected: {list(extracted.keys())}, Missing: {new_missing}"
                    )

                    # Update cache
                    cached_extraction = {
                        "extracted": extracted,
                        "missing": new_missing,
                        "all_collected": len(new_missing) == 0,
                    }

                    # Send slot updates to frontend
                    for slot_name, slot_value in extracted.items():
                        if slot_value is not None:
                            await send_msg(
                                "slot_update", name=slot_name, value=str(slot_value)
                            )

                    if cached_extraction["all_collected"]:
                        logger.info("[Completion] All required info collected!")

                except Exception as e:
                    logger.error(f"[Extraction-Async] Error: {e}")
                finally:
                    extraction_in_progress = False

            # Fire and forget - don't await
            asyncio.create_task(do_extraction())

        # === Phase 1: Use LangGraph planner if enabled ===
        if is_planner_enabled():
            # Run the LangGraph turn planner
            planner_result = await run_turn_planner(
                call_id=call_id,
                user_text=user_text,
                messages=messages,
                cached_extraction=cached_extraction,
                base_system_prompt=base_system_prompt,
                executor=executor,
            )

            # Extract results
            is_question = planner_result["is_question"]
            user_seems_done = planner_result["user_seems_done"]
            rag_context = planner_result["rag_context"]
            rag_results_raw = planner_result["rag_results"]
            all_info_collected = planner_result["all_collected"]
            missing = planner_result["missing"]
            missing_hint = planner_result["missing_hint"]
            current_system_prompt = planner_result["system_prompt"]

            # Phase 0: Record decisions from planner
            turn_event.is_question = is_question
            turn_event.user_seems_done = user_seems_done
            turn_event.rag_query = planner_result["rag_query"]
            turn_event.rag_results = rag_results_raw
            turn_event.rag_context = rag_context
            turn_event.extraction_output = cached_extraction.get("extracted", {})
            turn_event.missing_list = missing
            turn_event.all_collected = all_info_collected
            turn_event.missing_hint = missing_hint
            turn_event.system_prompt_hash = _hash_prompt(current_system_prompt)

            logger.debug(f"[Planner] Used LangGraph planner for turn {turn_counter[0]}")
        else:
            # === Original inline logic (fallback) ===
            # RAG for user questions
            rag_context = ""
            # Simple question detection (contains ？ or question words)
            is_question = "？" in user_text or any(
                w in user_text
                for w in [
                    "吗",
                    "怎么",
                    "是不是",
                    "为什么",
                    "什么",
                    "能不能",
                    "可以",
                    "哪",
                ]
            )

            # Phase 0: Record decisions
            turn_event.is_question = is_question
            rag_results_raw = []

            if is_question:
                try:
                    from app.helpers.rag import rag_service

                    results = await asyncio.get_event_loop().run_in_executor(
                        executor,
                        lambda: rag_service.search(user_text, top_k=2, threshold=0.45),
                    )
                    if results:
                        logger.info(f"[RAG] Found {len(results)} hits for: {user_text}")
                        rag_results_raw = results  # Phase 0: Capture for tracing
                        rag_context = "\n\n【知识库参考】\n"
                        for i, res in enumerate(results):
                            rag_context += f"- {res['topic']}: {res['answer'][:100]}\n"
                except Exception as e:
                    logger.error(f"[RAG] Error: {e}")

            # Phase 0: Record RAG results
            if is_question:
                turn_event.rag_query = user_text
                turn_event.rag_results = [
                    {"topic": r.get("topic", ""), "score": r.get("score", 0)}
                    for r in rag_results_raw
                ]
                turn_event.rag_context = rag_context

            # === Build Support-First System Prompt ===
            # Check if user is saying goodbye or has no more questions
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

            # Phase 0: Record extraction state and user_seems_done
            turn_event.user_seems_done = user_seems_done
            turn_event.extraction_output = cached_extraction.get("extracted", {})
            turn_event.missing_list = missing
            turn_event.all_collected = all_info_collected

            if all_info_collected and user_seems_done and not is_question:
                # All info collected and user has no more questions - say goodbye!
                missing_hint = """

【重要】信息已全部收集完毕，用户也没有其他问题了。请立即友好告别：
- 感谢对方配合随访
- 提醒有问题可拨打400电话
- 祝对方身体健康
- 说再见

示例："好的，今天的随访就到这里，感谢您的配合！后续有任何问题可以拨打我们的400热线。祝您身体健康，再见！"
"""
                logger.info("[Completion] Triggering goodbye!")
            elif all_info_collected:
                # Info collected but user might have questions - answer then close
                missing_hint = "\n\n【温馨提示】信息已收集完毕。如果用户问了问题就简短回答，然后友好告别。如果没有问题，就直接告别。"
            elif missing:
                # DIRECTIVE: AI must ask next question immediately after answering
                missing_hint = (
                    f'\n\n【下一步】必须立即问：{missing[0]}（不要问"还有其他问题吗"）'
                )
            else:
                missing_hint = ""

            # Build explicit collected info for LLM
            extracted = cached_extraction.get("extracted", {})
            collected_items = []
            slot_display_names = {
                "is_patient": "身份确认",
                "symptom_improvement": "症状改善",
                "control_score": "症状控制打分",
                "life_quality_score": "生活质量打分",
                "programming_count": "程控次数",
                "programming_satisfaction": "程控满意度",
                "side_effects": "不良反应",
                "mental_issues": "情绪问题",
                "medication_issues": "用药问题",
                "insurance_type": "医保类型",
                "total_cost": "总费用",
                "self_pay": "自费金额",
                "other_concerns": "其他情况",
            }
            for key, value in extracted.items():
                if value is not None and key in slot_display_names:
                    display_name = slot_display_names[key]
                    if isinstance(value, bool):
                        value_str = "是" if value else "否"
                    else:
                        value_str = str(value)
                    collected_items.append(f"{display_name}：{value_str}")

            if collected_items:
                collected_info = "\n\n【已收集信息】（不要重复问这些！）\n" + "\n".join(
                    collected_items
                )
            else:
                collected_info = ""

            current_system_prompt = f"""{base_system_prompt}
{collected_info}
{rag_context}
{missing_hint}"""

            # Phase 0: Record prompt hash
            turn_event.missing_hint = missing_hint
            turn_event.system_prompt_hash = _hash_prompt(current_system_prompt)

        # === Stream LLM Response ===
        t1 = time.time()
        full_response = ""
        sentence_buffer = ""
        first_audio = True

        is_speaking = True
        await send_msg("status", text="AI 正在回复...")
        await send_msg("speaking", value=True)

        first_token_time = None
        # Use full history for context continuity (up to reasonable limit)
        history_for_llm = messages[-20:] if len(messages) > 20 else messages

        async for token in stream_llm(history_for_llm, current_system_prompt):
            if not is_speaking:
                break

            if first_token_time is None:
                first_token_time = time.time()
                logger.info("[LLM首token] %.2fs", first_token_time - t1)

            full_response += token
            sentence_buffer += token

            # OPTIMIZATION: Start TTS on comma OR period (faster first audio)
            # Comma chunks are ~5-10 chars, spoken while LLM continues
            for punct in ["，", "。", "！", "？", ",", ".", "!", "?"]:
                if punct in sentence_buffer:
                    parts = sentence_buffer.split(punct, 1)
                    sentence = parts[0] + punct
                    sentence_buffer = parts[1] if len(parts) > 1 else ""

                    if sentence.strip() and len(sentence) > 2:  # Min 2 chars
                        t_sentence = time.time()
                        logger.info(
                            "[LLM句子完成] %.2fs: %s", t_sentence - t1, sentence[:20]
                        )

                        await send_msg("ai_token", text=sentence)

                        t_tts_start = time.time()
                        chunk_count = 0
                        for audio_chunk in synthesize_audio_stream(sentence, TTS_VOICE):
                            if not is_speaking or barge_in_triggered:
                                break
                            if audio_chunk:
                                chunk_count += 1
                                if first_audio:
                                    logger.info("[TTS首块] %.2fs", time.time() - t1)
                                    first_audio = False
                                await send_msg(
                                    "audio", data=base64.b64encode(audio_chunk).decode()
                                )

                                # Poll for barge-in during TTS (full-duplex)
                                try:
                                    poll_data = await asyncio.wait_for(
                                        websocket.receive_text(), timeout=0.02
                                    )
                                    poll_msg = json.loads(poll_data)
                                    if poll_msg["type"] == "audio_frame":
                                        poll_audio = base64.b64decode(poll_msg["data"])
                                        poll_rms = compute_rms(poll_audio)
                                        if poll_rms > BARGE_IN_THRESHOLD:
                                            barge_in_triggered = True
                                            barge_in_audio.extend(poll_audio)
                                            is_speaking = False
                                            logger.info(
                                                "[Barge-in] Detected during TTS (RMS=%.0f)",
                                                poll_rms,
                                            )
                                            await send_msg("stop_audio")
                                            break
                                except asyncio.TimeoutError:
                                    pass  # No message, continue TTS
                                except Exception as e:
                                    logger.debug("[Barge-in] Poll error: %s", e)

                        logger.info(
                            "[TTS完成] %.2fs, %d块",
                            time.time() - t_tts_start,
                            chunk_count,
                        )
                    break

        if sentence_buffer.strip() and is_speaking:
            await send_msg("ai_token", text=sentence_buffer)
            for audio_chunk in synthesize_audio_stream(sentence_buffer, TTS_VOICE):
                if not is_speaking:
                    break
                if audio_chunk:
                    await send_msg("audio", data=base64.b64encode(audio_chunk).decode())

        logger.info("[LLM+TTS] 总计 %.2fs: %s", time.time() - t1, full_response[:50])

        if full_response:
            await send_msg("ai_text", text=full_response)
            messages.append({"role": "assistant", "content": full_response})

        await send_msg("audio_end")
        is_speaking = False
        is_processing = False
        await send_msg("speaking", value=False)
        await send_msg("status", text="通话中")

        # Phase 0: Complete and log turn event
        turn_event.ai_response = full_response
        turn_event.llm_total_ms = int((time.time() - t1) * 1000)
        if first_token_time:
            turn_event.llm_first_token_ms = int((first_token_time - t1) * 1000)

        # Phase 2A: Track step transitions (observability only, no behavior change)
        # Update SOP state based on extraction, but don't use for control flow
        old_step = sop_engine.state.current_step
        try:
            # Sync extracted slots to SOPEngine for step inference
            if cached_extraction.get("extracted"):
                sop_engine.state.slots.update(cached_extraction["extracted"])
            # Let SOPEngine infer step transition
            sop_engine.process_response(user_text, ai_response=full_response)
            new_step = sop_engine.state.current_step
            if old_step != new_step:
                turn_event.step_transition = f"{old_step} -> {new_step}"
                logger.info(f"[SOP-Trace] Step transition: {old_step} -> {new_step}")
        except Exception as e:
            logger.warning(f"[SOP-Trace] Step tracking error (non-fatal): {e}")

        log_turn_event(turn_event)

    try:
        # 欢迎语 - 自然、人性化的开场白
        welcome = f"您好，我是品驰脑起搏器关爱中心的小驰。{patient_name}您好，您之前在我们这做了手术，现在差不多6个月了，我给您打电话是想做个术后随访。请问您现在方便吗？"
        await send_msg("ai_text", text=welcome)
        messages.append({"role": "assistant", "content": welcome})

        is_speaking = True
        await send_msg("speaking", value=True)

        # Use preloaded welcome audio if available, fallback to streaming
        welcome_audio = get_preloaded_welcome_audio(patient_name)
        if welcome_audio:
            logger.info(
                "[TTS] Using preloaded welcome audio: %d bytes", len(welcome_audio)
            )
            await send_msg("audio", data=base64.b64encode(welcome_audio).decode())
        else:
            # Fallback: stream welcome audio if cache failed
            logger.info("[TTS] Cache miss, streaming welcome audio")
            chunk_count = 0
            for audio_chunk in synthesize_audio_stream(welcome, TTS_VOICE):
                if audio_chunk:
                    chunk_count += 1
                    await send_msg("audio", data=base64.b64encode(audio_chunk).decode())
            logger.info("[TTS] Streamed welcome audio: %d chunks", chunk_count)

        await send_msg("audio_end")
        is_speaking = False
        await send_msg("speaking", value=False)
        await send_msg("status", text="通话中")

        # 主循环 - Simple VAD-based approach
        while True:
            try:
                data = await websocket.receive_text()
                msg = json.loads(data)

                if msg["type"] == "user_audio":
                    audio_data = base64.b64decode(msg["data"])
                    rms = compute_rms(audio_data)

                    # Server-side barge-in detection during AI speech
                    if is_speaking:
                        # Use threshold slightly above VAD to filter noise
                        BARGE_IN_THRESHOLD = (
                            600  # Lowered from 800 for better detection
                        )
                        if rms > BARGE_IN_THRESHOLD:
                            # User is speaking during AI playback - trigger barge-in
                            is_speaking = False
                            audio_buffer.clear()
                            logger.info(
                                "[Barge-in] Server detected user speech (RMS=%.0f), interrupting AI",
                                rms,
                            )
                            await send_msg("barge_in", detected=True)
                        continue  # Don't process further during speech

                    # VAD: Accumulate audio and detect speech (rms already computed above)
                    current_time = time.time()

                    if rms > ENERGY_THRESHOLD:
                        # User is speaking
                        if not is_user_speaking:
                            is_user_speaking = True
                            logger.debug("[VAD] Speech started, RMS=%.0f", rms)
                        audio_buffer.extend(audio_data)
                        last_speech_time = current_time
                    else:
                        # Silence detected
                        if is_user_speaking:
                            audio_buffer.extend(audio_data)  # Include trailing audio

                            # Check if silence duration exceeded threshold
                            silence_duration = current_time - last_speech_time
                            if silence_duration >= SILENCE_THRESHOLD:
                                # User stopped speaking - process accumulated audio
                                is_user_speaking = False

                                # Check minimum speech duration
                                audio_duration = len(audio_buffer) / (
                                    16000 * 2
                                )  # 16kHz, 16-bit
                                if audio_duration >= MIN_SPEECH_DURATION:
                                    logger.info(
                                        "[VAD] Speech ended, duration=%.2fs, processing...",
                                        audio_duration,
                                    )
                                    await process_audio(bytes(audio_buffer))
                                else:
                                    logger.debug(
                                        "[VAD] Too short (%.2fs)", audio_duration
                                    )

                                audio_buffer.clear()

                elif msg["type"] == "interrupt":
                    is_speaking = False
                    audio_buffer.clear()
                    logger.info("[Interrupt] User interrupted")

                elif msg["type"] == "ping":
                    await send_msg("pong")

            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error("Error: %s", e)
                continue

    except Exception as e:
        logger.error("WebSocket error: %s", e)
    finally:
        # Phase 0: Close trace file
        close_trace_file(call_id)
        logger.info("WebSocket disconnected")


def recognize_audio(audio_data: bytes) -> str:
    """ASR - 使用 PersistentASR 复用连接，带热词增强"""
    try:
        from app.helpers.asr_client import PersistentASR

        # Module-level singleton for connection reuse
        global _persistent_asr
        if "_persistent_asr" not in globals() or _persistent_asr is None:
            # 热词列表 - 从 knowledge_base.json 提取的领域专用词
            hot_words = [
                # 产品型号
                {"text": "DBS", "weight": 5},
                {"text": "R802", "weight": 5},
                {"text": "R801", "weight": 5},
                {"text": "C701", "weight": 5},
                {"text": "C702", "weight": 5},
                {"text": "IPG", "weight": 4},
                {"text": "G102R", "weight": 4},
                {"text": "G106R", "weight": 4},
                # 产品术语
                {"text": "脑起搏器", "weight": 5},
                {"text": "脑深部电刺激", "weight": 5},
                {"text": "刺激器", "weight": 4},
                {"text": "程控器", "weight": 5},
                {"text": "充电器", "weight": 4},
                {"text": "程控", "weight": 5},
                {"text": "远程程控", "weight": 5},
                {"text": "延伸导线", "weight": 4},
                {"text": "电极", "weight": 4},
                # 品牌和App
                {"text": "品驰", "weight": 5},
                {"text": "嘉医有品", "weight": 5},
                {"text": "品驰医疗", "weight": 5},
                {"text": "品驰中台", "weight": 4},
                # 疾病和医学术语
                {"text": "帕金森", "weight": 5},
                {"text": "帕金森病", "weight": 5},
                {"text": "肌张力障碍", "weight": 4},
                {"text": "震颤", "weight": 5},
                {"text": "僵直", "weight": 4},
                {"text": "手抖", "weight": 4},
                # 医保和费用
                {"text": "医保", "weight": 4},
                {"text": "惠民保", "weight": 5},
                {"text": "识别卡", "weight": 5},
                {"text": "随访", "weight": 4},
                # 操作术语
                {"text": "充电日志", "weight": 4},
                {"text": "起搏器自诊", "weight": 4},
                {"text": "开关刺激", "weight": 4},
                {"text": "开机", "weight": 4},
                {"text": "关机", "weight": 4},
                # Scores and feedback
                {"text": "满分", "weight": 3},
                {"text": "打分", "weight": 4},
                {"text": "满意", "weight": 4},
                {"text": "不满意", "weight": 4},
            ]
            _persistent_asr = PersistentASR(hot_words=hot_words)
            _persistent_asr.start()
            logger.info("[ASR] Initialized PersistentASR connection")

        result = _persistent_asr.recognize(audio_data)
        return result or ""
    except Exception as e:
        logger.error("ASR error: %s", e)
        # Reset on error
        if "_persistent_asr" in globals():
            try:
                _persistent_asr.stop()
            except:
                pass
            globals()["_persistent_asr"] = None
        return ""


async def stream_llm(messages: list, system_prompt: str):
    """流式 LLM - 异步生成器，逐 token 返回"""
    from app.helpers.config import CONFIG

    try:
        llm_model = CONFIG.llm.selected(is_fast=True)
        client, config = await llm_model.client()

        full_messages = [{"role": "system", "content": system_prompt}]
        full_messages.extend(messages)

        response = await client.chat.completions.create(
            model=config.model,
            messages=full_messages,
            max_tokens=100,
            temperature=0.8,  # Higher for more natural, varied responses
            stream=True,
        )

        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        logger.error("LLM stream error: %s", e)
        import traceback

        traceback.print_exc()


async def call_llm(messages: list, system_prompt: str) -> str:
    """LLM - 非流式，用于简单场景"""
    result = ""
    async for token in stream_llm(messages, system_prompt):
        result += token
    return result


def synthesize_audio(text: str) -> bytes:
    """TTS - 同步合成"""
    import dashscope
    from dashscope.audio.tts_v2 import AudioFormat, SpeechSynthesizer

    api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return b""

    dashscope.api_key = api_key

    try:
        synthesizer = SpeechSynthesizer(
            model="cosyvoice-v3-flash",
            voice="longanyang",  # 龙安洋，成熟男声
            format=AudioFormat.PCM_24000HZ_MONO_16BIT,  # 24kHz 高音质
        )

        audio = synthesizer.call(text)
        return audio if isinstance(audio, bytes) else b""

    except Exception as e:
        logger.error("TTS error: %s", e)
        return b""


def synthesize_audio_stream(text: str, voice: str = "longanyang"):
    """
    TTS 流式合成 - 生成器，逐块返回音频

    Args:
        text: 要合成的文本
        voice: 音色名称 (longanyang=普通话, longanyue=粤语)
    """
    import queue
    import threading

    import dashscope
    from dashscope.audio.tts_v2 import AudioFormat, ResultCallback, SpeechSynthesizer

    api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return

    dashscope.api_key = api_key
    logger.debug(f"[TTS] Using voice: {voice}")
    audio_queue = queue.Queue()

    class StreamCallback(ResultCallback):
        def on_open(self):
            logger.debug("TTS stream opened")

        def on_complete(self):
            audio_queue.put(None)  # 结束信号

        def on_error(self, msg):
            logger.error("TTS stream error: %s", msg)
            audio_queue.put(None)

        def on_close(self):
            pass

        def on_event(self, msg):
            pass

        def on_data(self, data):
            if data:
                audio_queue.put(data)  # 实时放入队列

    def run_synthesis():
        """在后台线程中运行 TTS 合成"""
        try:
            synthesizer = SpeechSynthesizer(
                model="cosyvoice-v3-flash",
                voice=voice,  # 使用传入的音色参数
                format=AudioFormat.PCM_24000HZ_MONO_16BIT,  # 24kHz 高音质
                callback=StreamCallback(),
            )
            synthesizer.streaming_call(text)
            synthesizer.streaming_complete()
        except Exception as e:
            logger.error("TTS synthesis thread error: %s", e)
            audio_queue.put(None)

    # 启动后台线程进行合成
    thread = threading.Thread(target=run_synthesis, daemon=True)
    thread.start()

    # 实时从队列读取并返回音频块
    try:
        while True:
            chunk = audio_queue.get(timeout=15)
            if chunk is None:
                break
            yield chunk
    except queue.Empty:
        logger.warning("TTS stream timeout")
    finally:
        thread.join(timeout=1)


# =============================================================================
# 真实电话体验界面 - 自动 VAD
# =============================================================================

PHONE_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>品驰关爱中心 - AI 客服</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', sans-serif;
            background: #000;
            min-height: 100vh;
            color: white;
        }

        /* 来电界面 */
        .incoming-call {
            position: fixed;
            inset: 0;
            background: linear-gradient(180deg, #2c2c2e 0%, #1c1c1e 50%, #000 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 80px 20px 50px;
            z-index: 100;
        }

        .incoming-call.hidden { display: none; }

        .caller-avatar {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            background: linear-gradient(135deg, #5856d6, #af52de);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 48px;
            margin-bottom: 16px;
        }

        .caller-name {
            font-size: 24px;
            font-weight: 300;
            margin-bottom: 8px;
        }

        .caller-label {
            font-size: 15px;
            color: #8e8e93;
            margin-bottom: auto;
        }

        .call-actions {
            display: flex;
            gap: 80px;
        }

        .call-action {
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .action-btn {
            width: 64px;
            height: 64px;
            border-radius: 50%;
            border: none;
            font-size: 28px;
            cursor: pointer;
            margin-bottom: 8px;
        }

        .btn-decline {
            background: #ff3b30;
            color: white;
        }

        .btn-accept {
            background: #34c759;
            color: white;
        }

        .action-label {
            font-size: 13px;
            color: #8e8e93;
        }

        /* 通话界面 */
        .call-screen {
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            background: linear-gradient(180deg, #1c1c1e 0%, #000 100%);
        }

        .call-header {
            padding: 60px 20px 20px;
            text-align: center;
        }

        .call-name {
            font-size: 20px;
            font-weight: 500;
            margin-bottom: 4px;
        }

        .call-timer {
            font-size: 15px;
            color: #8e8e93;
            font-variant-numeric: tabular-nums;
        }

        .call-status {
            font-size: 13px;
            color: #30d158;
            margin-top: 4px;
        }

        /* 对话区域 */
        .conversation {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .msg {
            max-width: 80%;
            padding: 10px 14px;
            border-radius: 18px;
            font-size: 15px;
            line-height: 1.4;
        }

        .msg.user {
            background: #0a84ff;
            align-self: flex-end;
            border-bottom-right-radius: 4px;
        }

        .msg.ai {
            background: #3a3a3c;
            align-self: flex-start;
            border-bottom-left-radius: 4px;
        }

        /* 音量指示器 */
        .volume-indicator {
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 4px;
            height: 80px;
        }

        .volume-bar {
            width: 4px;
            background: #30d158;
            border-radius: 2px;
            transition: height 0.05s;
        }

        .volume-indicator.ai-speaking .volume-bar {
            background: #af52de;
        }

        /* 控制按钮 */
        .call-controls {
            padding: 30px 20px 50px;
            display: flex;
            justify-content: space-around;
        }

        .ctrl-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: none;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 24px;
            cursor: pointer;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }

        .ctrl-btn.active {
            background: white;
            color: black;
        }

        .ctrl-btn span {
            font-size: 10px;
            margin-top: 4px;
        }

        .end-call-btn {
            background: #ff3b30;
        }
    </style>
</head>
<body>
    <!-- 来电界面 -->
    <div class="incoming-call" id="incomingCall">
        <div class="caller-avatar">🤖</div>
        <div class="caller-name">品驰关爱中心</div>
        <div class="caller-label">来电中...</div>

        <div class="call-actions">
            <div class="call-action">
                <button class="action-btn btn-decline" onclick="declineCall()">📵</button>
                <span class="action-label">拒绝</span>
            </div>
            <div class="call-action">
                <button class="action-btn btn-accept" onclick="acceptCall()">📞</button>
                <span class="action-label">接听</span>
            </div>
        </div>
    </div>

    <!-- 通话界面 -->
    <div class="call-screen" id="callScreen" style="display:none;">
        <div class="call-header">
            <div class="call-name">品驰关爱中心</div>
            <div class="call-timer" id="timer">00:00</div>
            <div class="call-status" id="status">连接中...</div>
        </div>

        <div class="conversation" id="conversation"></div>

        <div class="volume-indicator" id="volumeIndicator">
            <!-- 音量条 -->
        </div>

        <div class="call-controls">
            <button class="ctrl-btn" id="btnMute" onclick="toggleMute()">
                🔊
                <span>静音</span>
            </button>
            <button class="ctrl-btn end-call-btn" onclick="endCall()">
                📵
                <span>挂断</span>
            </button>
            <button class="ctrl-btn" id="btnSpeaker" onclick="toggleSpeaker()">
                🔈
                <span>扬声器</span>
            </button>
        </div>
    </div>

    <script>
        // === 状态 ===
        let ws = null;
        let mediaStream = null;
        let audioContext = null;
        let analyser = null;
        let processor = null;
        let callStartTime = null;
        let timerInterval = null;
        let isMuted = false;
        let isAISpeaking = false;

        // VAD 状态
        let audioBuffer = [];
        let preBuffer = [];  // 预缓冲，保留语音开始前的音频
        const PRE_BUFFER_SIZE = 3;  // 保留最近 3 帧（约 750ms）
        let isSpeaking = false;
        let silenceStart = null;
        const SILENCE_THRESHOLD = 0.002;  // 更敏感的阈值
        const SILENCE_DURATION = 300;  // ms - 更快响应
        const MIN_SPEECH_DURATION = 100;  // ms - 允许更短的语音
        let speechStart = null;

        // 音频播放
        let audioQueue = [];
        let isPlaying = false;
        let currentSource = null; // For barge-in audio stop

        // === DOM ===
        const incomingCall = document.getElementById('incomingCall');
        const callScreen = document.getElementById('callScreen');
        const timer = document.getElementById('timer');
        const status = document.getElementById('status');
        const conversation = document.getElementById('conversation');
        const volumeIndicator = document.getElementById('volumeIndicator');
        const btnMute = document.getElementById('btnMute');

        // 初始化音量条
        for (let i = 0; i < 30; i++) {
            const bar = document.createElement('div');
            bar.className = 'volume-bar';
            bar.style.height = '4px';
            volumeIndicator.appendChild(bar);
        }

        // === 函数 ===
        function updateTimer() {
            if (!callStartTime) return;
            const s = Math.floor((Date.now() - callStartTime) / 1000);
            timer.textContent = `${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;
        }

        function addMessage(role, text) {
            const div = document.createElement('div');
            div.className = 'msg ' + (role === 'user' ? 'user' : 'ai');
            div.textContent = text;
            conversation.appendChild(div);
            conversation.scrollTop = conversation.scrollHeight;
        }

        function updateVolume(volume) {
            const bars = volumeIndicator.children;
            const normalized = Math.min(1, volume * 15);
            for (let i = 0; i < bars.length; i++) {
                const h = 4 + normalized * 30 * (0.5 + Math.random() * 0.5);
                bars[i].style.height = h + 'px';
            }
        }

        function declineCall() {
            window.close();
        }

        async function acceptCall() {
            try {
                // Resume AudioContext in user gesture context (required by browser policy)
                if (playbackCtx.state === 'suspended') {
                    await playbackCtx.resume();
                    console.log('AudioContext resumed in click handler');
                }

                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true, noiseSuppression: true }
                });

                incomingCall.classList.add('hidden');
                callScreen.style.display = 'flex';

                // WebSocket
                const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${proto}//${location.host}/streaming/ws`);

                ws.onopen = () => {
                    console.log('Connected');
                    status.textContent = '通话中';
                    callStartTime = Date.now();
                    timerInterval = setInterval(updateTimer, 1000);
                    startAudioProcessing();
                };

                ws.onmessage = (e) => handleMessage(JSON.parse(e.data));
                ws.onclose = () => endCall();
                ws.onerror = (e) => console.error('WS error:', e);

            } catch (e) {
                console.error(e);
                alert('无法访问麦克风');
            }
        }

        function startAudioProcessing() {
            audioContext = new AudioContext({ sampleRate: 16000 });
            const source = audioContext.createMediaStreamSource(mediaStream);

            analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);

            processor = audioContext.createScriptProcessor(4096, 1, 1);

            processor.onaudioprocess = (e) => {
                if (isMuted) return;

                const input = e.inputBuffer.getChannelData(0);

                // 计算音量
                let sum = 0;
                for (let i = 0; i < input.length; i++) sum += Math.abs(input[i]);
                const volume = sum / input.length;

                // 总是更新音量显示
                updateVolume(volume);

                // Debug: Log volume during AI speech every 500ms
                if (isAISpeaking && Math.random() < 0.02) {
                    console.log('[Mic during AI] volume:', volume.toFixed(4));
                }

                // 转换为 PCM
                const pcm = new Int16Array(input.length);
                for (let i = 0; i < input.length; i++) {
                    pcm[i] = Math.max(-32768, Math.min(32767, input[i] * 32768));
                }

                // Client-side barge-in: stop audio locally when user speaks during AI
                // Use absolute threshold since echo cancellation may affect relative values
                const BARGE_IN_THRESHOLD = 0.01; // Much higher than SILENCE_THRESHOLD
                if (isAISpeaking && volume > BARGE_IN_THRESHOLD) {
                    console.log('🛑 BARGE-IN! volume:', volume);
                    // Send audio_frame to server for server-side detection
                    const base64 = arrayBufferToBase64(pcm.buffer);
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: 'audio_frame', data: base64 }));
                    }
                    // Also stop audio locally (client-side fallback)
                    audioQueue = [];
                    if (currentSource) {
                        try { currentSource.stop(0); } catch(e) {}
                        currentSource = null;
                    }
                    isPlaying = false;
                    isAISpeaking = false;
                    status.textContent = '正在听您说话...';
                    // Continue with normal VAD - don't return, let speech be processed
                }

                // VAD 逻辑
                const hasSpeech = volume > SILENCE_THRESHOLD;
                const now = Date.now();

                if (hasSpeech) {
                    if (!isSpeaking) {
                        console.log('Speech detected, volume:', volume);
                        isSpeaking = true;
                        speechStart = now;
                        // 把预缓冲的音频加到开头（防止前几个字丢失）
                        audioBuffer = [...preBuffer];
                        preBuffer = [];
                        status.textContent = '正在听您说话...';
                    }
                    silenceStart = null;
                    audioBuffer.push(pcm);

                } else if (isSpeaking) {
                    // 静音中，继续收集
                    if (!silenceStart) {
                        silenceStart = now;
                    }
                    audioBuffer.push(pcm);

                    // 静音超时 -> 发送
                    if (now - silenceStart > SILENCE_DURATION) {
                        console.log('Silence detected, sending audio. Buffer size:', audioBuffer.length);
                        if (speechStart && now - speechStart > MIN_SPEECH_DURATION) {
                            sendAudio();
                        }
                        isSpeaking = false;
                        speechStart = null;
                        silenceStart = null;
                        audioBuffer = [];
                    }
                } else {
                    // 静音时，维护预缓冲（保留最近几帧）
                    preBuffer.push(pcm);
                    if (preBuffer.length > PRE_BUFFER_SIZE) {
                        preBuffer.shift();
                    }
                }
            };

            source.connect(processor);
            processor.connect(audioContext.destination);
        }

        function sendAudio() {
            console.log('sendAudio called, buffer length:', audioBuffer.length, 'ws state:', ws?.readyState);
            if (audioBuffer.length === 0) {
                console.log('No audio to send');
                return;
            }
            if (!ws || ws.readyState !== WebSocket.OPEN) {
                console.log('WebSocket not ready');
                return;
            }

            // 合并音频
            let totalLength = 0;
            for (const chunk of audioBuffer) totalLength += chunk.length;
            console.log('Total samples:', totalLength);

            const merged = new Int16Array(totalLength);
            let offset = 0;
            for (const chunk of audioBuffer) {
                merged.set(chunk, offset);
                offset += chunk.length;
            }

            // 发送
            const base64 = arrayBufferToBase64(merged.buffer);
            ws.send(JSON.stringify({ type: 'user_audio', data: base64 }));

            status.textContent = '处理中...';
            audioBuffer = [];
        }

        function handleMessage(msg) {
            switch (msg.type) {
                case 'status':
                    status.textContent = msg.text;
                    break;
                case 'user_text':
                    // User spoke - stop AI audio immediately (ASR-based barge-in)
                    console.log('🛑 User spoke, stopping AI audio');
                    audioQueue = [];
                    if (currentSource) {
                        try { currentSource.stop(0); } catch(e) {}
                        currentSource = null;
                    }
                    isPlaying = false;
                    isAISpeaking = false;
                    addMessage('user', msg.text);
                    break;
                case 'ai_text':
                    addMessage('ai', msg.text);
                    break;
                case 'speaking':
                    isAISpeaking = msg.value;
                    volumeIndicator.classList.toggle('ai-speaking', msg.value);
                    break;
                case 'audio':
                    console.log('Received audio:', msg.data.length, 'base64 chars');
                    const audioData = base64ToArrayBuffer(msg.data);
                    console.log('Decoded audio:', audioData.byteLength, 'bytes');
                    audioQueue.push(audioData);
                    playAudio();
                    break;
                case 'audio_end':
                    break;
                case 'barge_in':
                    // User spoke during AI speech - stop playback immediately
                    console.log('Barge-in detected, stopping audio');
                    audioQueue = [];
                    if (currentSource) {
                        try { currentSource.stop(0); } catch(e) {}
                        currentSource = null;
                    }
                    isPlaying = false;
                    isAISpeaking = false;
                    status.textContent = '正在听您说话...';
                    break;
                case 'stop_audio':
                    // Server detected barge-in, stop playback
                    console.log('🛑 Server requested stop_audio');
                    audioQueue = [];
                    if (currentSource) {
                        try { currentSource.stop(0); } catch(e) {}
                        currentSource = null;
                    }
                    isPlaying = false;
                    isAISpeaking = false;
                    status.textContent = '正在听您说话...';
                    break;
                case 'pong':
                    break;
            }
        }

        async function playAudio() {
            if (isPlaying || audioQueue.length === 0) return;
            isPlaying = true;

            while (audioQueue.length > 0 && isAISpeaking) {
                const buf = audioQueue.shift();
                await playPCM(buf);
                // Check again after each buffer - barge-in might have happened
                if (!isAISpeaking) break;
            }

            isPlaying = false;
        }

        // Global Audio Context for playback
        const playbackCtx = new window.AudioContext();

        function playPCM(pcmBuffer) {
            return new Promise(resolve => {
                try {
                    const ctx = playbackCtx; // Reuse global context
                    if (ctx.state === 'suspended') {
                        ctx.resume();
                    }

                    const pcm16 = new Int16Array(pcmBuffer);
                    const float32 = new Float32Array(pcm16.length);
                    for (let i = 0; i < pcm16.length; i++) {
                        float32[i] = pcm16[i] / 32768;
                    }

                    // 24kHz matches server Tts output
                    const buffer = ctx.createBuffer(1, float32.length, 24000);
                    buffer.getChannelData(0).set(float32);

                    const updateVol = setInterval(() => {
                        if (isAISpeaking) {
                            updateVolume(0.3 + Math.random() * 0.3);
                        }
                    }, 100);

                    const source = ctx.createBufferSource();
                    source.buffer = buffer;
                    source.connect(ctx.destination);
                    source.onended = () => {
                        clearInterval(updateVol);
                        currentSource = null;
                        resolve();
                    };
                    currentSource = source;
                    source.start();
                } catch (e) {
                    console.error('Play error:', e);
                    resolve();
                }
            });
        }

        function toggleMute() {
            isMuted = !isMuted;
            btnMute.classList.toggle('active', isMuted);
            btnMute.innerHTML = isMuted ? '🔇<span>取消静音</span>' : '🔊<span>静音</span>';
        }

        function toggleSpeaker() {
            // 模拟功能
        }

        function endCall() {
            if (ws) { ws.close(); ws = null; }
            if (mediaStream) { mediaStream.getTracks().forEach(t => t.stop()); }
            if (audioContext) { audioContext.close(); }
            if (timerInterval) { clearInterval(timerInterval); }

            callScreen.style.display = 'none';
            incomingCall.classList.remove('hidden');
            conversation.innerHTML = '';
            timer.textContent = '00:00';
        }

        function arrayBufferToBase64(buffer) {
            const bytes = new Uint8Array(buffer);
            let binary = '';
            for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
            return btoa(binary);
        }

        function base64ToArrayBuffer(base64) {
            const binary = atob(base64);
            const bytes = new Uint8Array(binary.length);
            for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
            return bytes.buffer;
        }

        // 保持连接
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000);
    </script>
</body>
</html>
"""


@router.get("/", response_class=HTMLResponse)
async def phone_page():
    """电话界面"""
    return HTMLResponse(content=PHONE_HTML)
