"""
流式语音识别 (ASR) - Paraformer Realtime

使用 DashScope 的 Paraformer-realtime-v2 实现流式语音识别：
- 持续发送音频帧
- 实时返回 partial（部分）和 final（完整）结果
- 内置 VAD（语音活动检测）

参考文档：
https://help.aliyun.com/zh/model-studio/paraformer-real-time-speech-recognition-python-sdk
"""

import asyncio
import json
import logging
import os
import threading
import time
from typing import Callable

logger = logging.getLogger(__name__)


class StreamingASR:
    """
    流式语音识别
    
    使用 DashScope Recognition SDK 的流式模式
    """
    
    def __init__(
        self,
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
        model: str = "paraformer-realtime-v2",
        sample_rate: int = 16000,
    ):
        """
        初始化流式 ASR
        
        Args:
            on_partial: 收到部分识别结果时的回调
            on_final: 收到完整识别结果时的回调
            model: 模型名称
            sample_rate: 采样率
        """
        self.on_partial = on_partial
        self.on_final = on_final
        self.model = model
        self.sample_rate = sample_rate
        
        self._recognition = None
        self._running = False
        self._audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._worker_thread = None
        
        # 当前识别结果
        self._partial_text = ""
        self._last_final_time = time.time()
        
        # VAD 参数
        self._silence_timeout = 0.8  # 静音超时（秒）
        self._last_audio_time = time.time()
    
    async def start(self):
        """启动流式识别"""
        if self._running:
            return
        
        self._running = True
        self._partial_text = ""
        
        # 在后台线程运行识别（SDK 是同步的）
        self._worker_thread = threading.Thread(target=self._recognition_worker, daemon=True)
        self._worker_thread.start()
        
        logger.info("Streaming ASR started")
    
    async def stop(self):
        """停止流式识别"""
        self._running = False
        await self._audio_queue.put(None)  # 发送停止信号
        
        if self._worker_thread:
            self._worker_thread.join(timeout=2.0)
        
        logger.info("Streaming ASR stopped")
    
    async def feed_audio(self, data: bytes):
        """
        输入音频数据
        
        Args:
            data: PCM 音频数据 (16kHz, 16bit, mono)
        """
        if self._running:
            await self._audio_queue.put(data)
            self._last_audio_time = time.time()
    
    def _recognition_worker(self):
        """识别工作线程（在后台线程运行）"""
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
        
        api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            logger.error("DASHSCOPE_API_KEY not set")
            return
        
        dashscope.api_key = api_key
        
        class Callback(RecognitionCallback):
            def __init__(self, parent: 'StreamingASR'):
                self.parent = parent
            
            def on_open(self):
                logger.debug("ASR connection opened")
            
            def on_close(self):
                logger.debug("ASR connection closed")
            
            def on_error(self, result: RecognitionResult):
                logger.error("ASR error: %s", result.message)
            
            def on_event(self, result: RecognitionResult):
                # 处理识别结果
                self.parent._handle_result(result)
        
        callback = Callback(self)
        
        try:
            # 创建识别实例
            recognition = Recognition(
                model=self.model,
                format="pcm",
                sample_rate=self.sample_rate,
                callback=callback,
            )
            
            # 启动识别
            recognition.start()
            self._recognition = recognition
            
            # 持续发送音频
            loop = asyncio.new_event_loop()
            while self._running:
                try:
                    # 从队列获取音频（带超时）
                    future = asyncio.run_coroutine_threadsafe(
                        asyncio.wait_for(self._audio_queue.get(), timeout=0.1),
                        loop
                    )
                    try:
                        data = future.result(timeout=0.2)
                    except:
                        # 检查是否需要触发静音超时
                        if time.time() - self._last_audio_time > self._silence_timeout:
                            if self._partial_text:
                                # 静音超时，触发 final
                                self._trigger_final()
                        continue
                    
                    if data is None:
                        break
                    
                    # 发送音频到识别服务
                    recognition.send_audio_frame(data)
                    
                except Exception as e:
                    if self._running:
                        logger.debug("Audio queue error: %s", e)
                    continue
            
            # 停止识别
            recognition.stop()
            loop.close()
            
        except Exception as e:
            logger.error("Recognition worker error: %s", e)
    
    def _handle_result(self, result):
        """处理识别结果"""
        try:
            # 解析结果
            if hasattr(result, 'get_sentence'):
                sentences = result.get_sentence()
                if sentences:
                    for sent in sentences:
                        text = sent.get('text', '')
                        is_final = sent.get('end_time', 0) > 0  # 有结束时间表示是 final
                        
                        if is_final:
                            if text and self.on_final:
                                self.on_final(text)
                            self._partial_text = ""
                        else:
                            self._partial_text = text
                            if text and self.on_partial:
                                self.on_partial(text)
            
        except Exception as e:
            logger.error("Handle result error: %s", e)
    
    def _trigger_final(self):
        """触发 final 结果（用于静音超时）"""
        if self._partial_text and self.on_final:
            self.on_final(self._partial_text)
        self._partial_text = ""


class SimpleStreamingASR:
    """
    简化版流式 ASR
    
    使用同步调用 + 线程池，更稳定
    """
    
    def __init__(
        self,
        on_partial: Callable[[str], None] | None = None,
        on_final: Callable[[str], None] | None = None,
        sample_rate: int = 16000,
        silence_timeout: float = 0.6,  # 静音超时触发识别
    ):
        self.on_partial = on_partial
        self.on_final = on_final
        self.sample_rate = sample_rate
        self.silence_timeout = silence_timeout
        
        self._running = False
        self._audio_buffer = bytearray()
        self._last_audio_time = time.time()
        self._recognition_task = None
        self._lock = asyncio.Lock()
        
        # VAD 状态
        self._is_speaking = False
        self._silence_start = None
    
    async def start(self):
        """启动"""
        self._running = True
        self._audio_buffer.clear()
        logger.info("Simple streaming ASR started")
    
    async def stop(self):
        """停止"""
        self._running = False
        # 处理剩余音频
        if self._audio_buffer:
            await self._recognize_buffer()
        logger.info("Simple streaming ASR stopped")
    
    async def feed_audio(self, data: bytes, is_speech: bool = True):
        """
        输入音频数据
        
        Args:
            data: PCM 音频数据
            is_speech: 是否包含语音（用于 VAD）
        """
        if not self._running:
            return
        
        async with self._lock:
            self._audio_buffer.extend(data)
            
            if is_speech:
                self._is_speaking = True
                self._silence_start = None
                self._last_audio_time = time.time()
            else:
                if self._is_speaking:
                    if self._silence_start is None:
                        self._silence_start = time.time()
                    elif time.time() - self._silence_start > self.silence_timeout:
                        # 静音超时，触发识别
                        self._is_speaking = False
                        await self._recognize_buffer()
    
    async def force_recognize(self):
        """强制识别当前缓冲区"""
        async with self._lock:
            if self._audio_buffer:
                await self._recognize_buffer()
    
    async def _recognize_buffer(self):
        """识别缓冲区中的音频"""
        if not self._audio_buffer:
            return
        
        audio_data = bytes(self._audio_buffer)
        self._audio_buffer.clear()
        
        # 在线程池中运行同步识别
        loop = asyncio.get_event_loop()
        try:
            text = await loop.run_in_executor(
                None,
                self._sync_recognize,
                audio_data
            )
            if text and self.on_final:
                self.on_final(text)
        except Exception as e:
            logger.error("Recognition error: %s", e)
    
    def _sync_recognize(self, audio_data: bytes) -> str:
        """同步识别"""
        import dashscope
        from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult
        
        api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")
        
        dashscope.api_key = api_key
        
        result_text = ""
        done_event = threading.Event()
        
        class Callback(RecognitionCallback):
            def on_event(self, result: RecognitionResult):
                nonlocal result_text
                if hasattr(result, 'get_sentence'):
                    sentences = result.get_sentence()
                    if sentences:
                        for sent in sentences:
                            text = sent.get('text', '')
                            if text:
                                result_text = text
            
            def on_complete(self):
                done_event.set()
            
            def on_error(self, result: RecognitionResult):
                logger.error("Recognition error: %s", result.message)
                done_event.set()
        
        recognition = Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=self.sample_rate,
            callback=Callback(),
        )
        
        recognition.start()
        
        # 分块发送音频
        chunk_size = 3200 * 5  # 500ms
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            recognition.send_audio_frame(chunk)
        
        recognition.stop()
        done_event.wait(timeout=5.0)
        
        return result_text

