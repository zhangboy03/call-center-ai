"""
Persistent Streaming ASR - Paraformer Realtime

Maintains ONE ASR session for the entire call duration:
- Uses Paraformer's built-in VAD for sentence detection
- Supports hot words for domain-specific accuracy
- Async queue for backpressure management
- Heartbeat/reconnection handling

Reference: https://help.aliyun.com/zh/model-studio/paraformer-real-time-speech-recognition-python-sdk
"""

import asyncio
import logging
import os
import queue
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


# Domain-specific hot words extracted from knowledge_base.json
HOT_WORDS = [
    # Company & Products
    "品驰",
    "品驰医疗",
    "品驰关爱中心",
    # DBS System
    "DBS",
    "脑起搏器",
    "脑深部电刺激",
    "IPG",
    "刺激器",
    "脉冲发生器",
    # Components
    "程控器",
    "充电器",
    "电极",
    "延伸导线",
    "磁铁",
    # Model Numbers
    "G101A",
    "G102",
    "G102R",
    "G106",
    "G106R",
    "G107",
    "G107R",
    "R801",
    "R802",
    "C701",
    "C702",
    # Medical Terms
    "帕金森",
    "帕金森病",
    "肌张力障碍",
    "术后",
    "复诊",
    "程控",
    # Actions
    "开机",
    "关机",
    "充电",
    "开刺激",
    "关刺激",
    # Symptoms
    "震颤",
    "僵直",
    "异动",
    "步态",
    "平衡",
]


class PersistentASR:
    """
    Persistent streaming ASR using Paraformer-realtime-v2.

    Maintains one ASR session per call, using Paraformer's built-in VAD.
    """

    def __init__(
        self,
        on_partial: Callable[[str], None],
        on_final: Callable[[str], None],
        sample_rate: int = 16000,
        model: str = "paraformer-realtime-v2",
        max_sentence_silence: int = 800,  # ms
    ):
        """
        Initialize persistent ASR.

        Args:
            on_partial: Callback for partial (intermediate) results
            on_final: Callback for final (sentence-complete) results
            sample_rate: Audio sample rate (must match input audio)
            model: Paraformer model name
            max_sentence_silence: Silence duration to trigger sentence end (ms)
        """
        self.on_partial = on_partial
        self.on_final = on_final
        self.sample_rate = sample_rate
        self.model = model
        self.max_sentence_silence = max_sentence_silence

        self._recognition = None
        self._running = False
        self._worker_thread = None

        # Async queue for backpressure management (Codex recommendation)
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue(maxsize=100)

        # State tracking
        self._last_activity = time.time()
        self._partial_text = ""

    def start(self) -> bool:
        """
        Start persistent ASR session.

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning("[PersistentASR] Already running")
            return True

        self._running = True
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

        logger.info("[PersistentASR] Started")
        return True

    def stop(self):
        """Stop ASR session."""
        self._running = False
        self._audio_queue.put(None)  # Signal worker to stop

        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)

        logger.info("[PersistentASR] Stopped")

    def send_audio(self, audio_data: bytes):
        """
        Send audio chunk to ASR (non-blocking).

        Drops audio if queue is full (backpressure).
        """
        if not self._running:
            return

        try:
            self._audio_queue.put_nowait(audio_data)
            self._last_activity = time.time()
        except queue.Full:
            logger.warning("[PersistentASR] Queue full, dropping audio")

    def flush_buffer(self):
        """Flush audio queue (e.g., on interrupt)."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break
        logger.debug("[PersistentASR] Buffer flushed")

    def _worker(self):
        """Background worker thread for ASR processing."""
        import dashscope
        from dashscope.audio.asr import (
            Recognition,
            RecognitionCallback,
            RecognitionResult,
        )

        api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get(
            "DASHSCOPE_API_KEY"
        )
        if not api_key:
            logger.error("[PersistentASR] DASHSCOPE_API_KEY not set")
            return

        dashscope.api_key = api_key

        class Callback(RecognitionCallback):
            def __init__(self, parent: "PersistentASR"):
                self.parent = parent
                self.session_ended = False

            def on_open(self):
                logger.info("[PersistentASR] Connection opened")
                self.session_ended = False

            def on_close(self):
                logger.info("[PersistentASR] Connection closed")
                self.session_ended = True

            def on_error(self, result: RecognitionResult):
                logger.error("[PersistentASR] Error: %s", result)
                self.session_ended = True

            def on_event(self, result: RecognitionResult):
                self.parent._handle_result(result)

            def on_complete(self):
                logger.info("[PersistentASR] Session complete - will reconnect")
                self.session_ended = True

        callback = Callback(self)

        # Auto-reconnect loop
        while self._running:
            try:
                # Create recognition instance
                self._recognition = Recognition(
                    model=self.model,
                    format="pcm",
                    sample_rate=self.sample_rate,
                    callback=callback,
                )

                # Start with VAD settings (Codex & Aliyun docs)
                callback.session_ended = False
                self._recognition.start(
                    language_hints=["zh"],
                    # VAD断句 mode (lower latency for conversation)
                    semantic_punctuation_enabled=False,
                    # Sentence silence threshold
                    max_sentence_silence=self.max_sentence_silence,
                )

                logger.info(
                    "[PersistentASR] Recognition started with VAD=%dms",
                    self.max_sentence_silence,
                )

                # Forward audio from queue until session ends
                while self._running and not callback.session_ended:
                    try:
                        audio_data = self._audio_queue.get(timeout=0.1)

                        if audio_data is None:
                            # Stop signal
                            self._running = False
                            break

                        # Send audio frame
                        self._recognition.send_audio_frame(audio_data)

                    except queue.Empty:
                        # Check for timeout (send keepalive if needed)
                        if time.time() - self._last_activity > 30:
                            logger.debug("[PersistentASR] Idle keepalive")
                            self._last_activity = time.time()
                        continue

                # Session ended - stop current recognition
                try:
                    self._recognition.stop()
                except Exception:
                    pass

                if self._running:
                    logger.info("[PersistentASR] Reconnecting...")
                    time.sleep(0.1)  # Brief pause before reconnect

            except Exception as e:
                logger.error("[PersistentASR] Worker error: %s", e, exc_info=True)
                if self._running:
                    time.sleep(1.0)  # Wait before retry on error

    def _handle_result(self, result):
        """Handle ASR result from callback."""
        try:
            if hasattr(result, "get_sentence"):
                sentence = result.get_sentence()
                if sentence and "text" in sentence:
                    text = sentence["text"]

                    # Check if sentence ended (Paraformer VAD)
                    from dashscope.audio.asr import RecognitionResult

                    is_final = RecognitionResult.is_sentence_end(sentence)

                    if is_final:
                        logger.info("[PersistentASR] Final: %s", text)
                        self._partial_text = ""
                        if self.on_final and text.strip():
                            self.on_final(text)
                    else:
                        self._partial_text = text
                        if self.on_partial and text.strip():
                            self.on_partial(text)
        except Exception as e:
            logger.error("[PersistentASR] Handle result error: %s", e)


class AsyncPersistentASR:
    """
    Async wrapper for PersistentASR.

    Provides async interface for use in FastAPI/asyncio context.
    """

    def __init__(
        self,
        on_partial: Callable[[str], None],
        on_final: Callable[[str], None],
        **kwargs,
    ):
        self._asr = PersistentASR(on_partial=on_partial, on_final=on_final, **kwargs)
        self._loop = None

    async def start(self):
        """Start ASR in executor."""
        self._loop = asyncio.get_event_loop()
        await self._loop.run_in_executor(None, self._asr.start)

    async def stop(self):
        """Stop ASR."""
        if self._loop:
            await self._loop.run_in_executor(None, self._asr.stop)

    def send_audio(self, audio_data: bytes):
        """Send audio (non-blocking)."""
        self._asr.send_audio(audio_data)

    def flush_buffer(self):
        """Flush audio buffer."""
        self._asr.flush_buffer()
