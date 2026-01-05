"""
流式语音合成 (TTS) - CosyVoice

使用 DashScope 的 CosyVoice 实现流式语音合成：
- 边合成边返回音频
- 支持打断

参考文档：
https://help.aliyun.com/zh/model-studio/cosyvoice-python-sdk
"""

import asyncio
import logging
import os
import queue
import threading
import time
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class StreamingTTS:
    """
    流式语音合成
    
    使用 DashScope TTS SDK 的回调模式，边合成边返回音频
    """
    
    def __init__(
        self,
        model: str = "cosyvoice-v3-flash",
        voice: str = "longanyang",
        sample_rate: int = 8000,
    ):
        """
        初始化流式 TTS
        
        Args:
            model: 模型名称
            voice: 音色
            sample_rate: 采样率
        """
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
    
    async def synthesize_stream(self, text: str) -> AsyncIterator[bytes]:
        """
        流式合成语音
        
        Args:
            text: 要合成的文本
        
        Yields:
            音频数据块 (PCM)
        """
        if not text.strip():
            return
        
        # 使用队列在线程之间传递音频数据
        audio_queue: queue.Queue[bytes | None] = queue.Queue()
        error_holder = [None]  # 用列表存储错误，以便在线程间传递
        
        def synthesis_worker():
            """在后台线程运行同步合成"""
            try:
                import dashscope
                from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat, ResultCallback
                
                api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
                if not api_key:
                    raise ValueError("DASHSCOPE_API_KEY not set")
                
                dashscope.api_key = api_key
                
                # 选择音频格式
                if self.sample_rate == 8000:
                    audio_format = AudioFormat.PCM_8000HZ_MONO_16BIT
                else:
                    audio_format = AudioFormat.PCM_16000HZ_MONO_16BIT
                
                class Callback(ResultCallback):
                    def on_open(self):
                        logger.debug("TTS connection opened")
                    
                    def on_complete(self):
                        logger.debug("TTS synthesis complete")
                        audio_queue.put(None)  # 发送结束信号
                    
                    def on_error(self, message):
                        logger.error("TTS error: %s", message)
                        error_holder[0] = message
                        audio_queue.put(None)
                    
                    def on_close(self):
                        logger.debug("TTS connection closed")
                    
                    def on_event(self, message):
                        pass  # 忽略事件消息
                    
                    def on_data(self, data: bytes):
                        if data:
                            audio_queue.put(data)
                
                synthesizer = SpeechSynthesizer(
                    model=self.model,
                    voice=self.voice,
                    format=audio_format,
                    callback=Callback(),
                )
                
                # 使用流式调用
                synthesizer.streaming_call(text)
                
                # 等待完成
                synthesizer.streaming_complete()
                
            except Exception as e:
                logger.error("TTS worker error: %s", e)
                error_holder[0] = str(e)
                audio_queue.put(None)
        
        # 启动后台线程
        thread = threading.Thread(target=synthesis_worker, daemon=True)
        thread.start()
        
        # 异步读取音频数据
        loop = asyncio.get_event_loop()
        while True:
            try:
                # 在线程池中等待队列数据
                data = await loop.run_in_executor(
                    None,
                    lambda: audio_queue.get(timeout=5.0)
                )
                
                if data is None:
                    break
                
                yield data
                
            except Exception as e:
                logger.error("Error reading audio queue: %s", e)
                break
        
        # 等待线程结束
        thread.join(timeout=1.0)
        
        if error_holder[0]:
            logger.error("TTS synthesis failed: %s", error_holder[0])


class SimpleTTS:
    """
    简单 TTS（非流式，但更稳定）
    
    每次调用都创建新连接，适合短句
    """
    
    def __init__(
        self,
        model: str = "cosyvoice-v3-flash",
        voice: str = "longanyang",
        sample_rate: int = 8000,
    ):
        self.model = model
        self.voice = voice
        self.sample_rate = sample_rate
    
    async def synthesize(self, text: str) -> bytes:
        """
        合成语音
        
        Args:
            text: 要合成的文本
        
        Returns:
            音频数据 (PCM)
        """
        if not text.strip():
            return b""
        
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_synthesize, text)
    
    def _sync_synthesize(self, text: str) -> bytes:
        """同步合成"""
        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
        
        api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY not set")
        
        dashscope.api_key = api_key
        
        if self.sample_rate == 8000:
            audio_format = AudioFormat.PCM_8000HZ_MONO_16BIT
        else:
            audio_format = AudioFormat.PCM_16000HZ_MONO_16BIT
        
        synthesizer = SpeechSynthesizer(
            model=self.model,
            voice=self.voice,
            format=audio_format,
        )
        
        audio_data = synthesizer.call(text)
        return audio_data if isinstance(audio_data, bytes) else b""

