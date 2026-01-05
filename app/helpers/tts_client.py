"""
阿里云 CosyVoice 实时语音合成客户端

基于 DashScope SDK 实现流式语音合成，支持边生成边播放。

使用方法：
```python
# 基本用法
async with CosyVoiceSynthesizer() as tts:
    async for audio_chunk in tts.synthesize_stream("你好，我是小驰"):
        # 处理音频数据（PCM 格式）
        play_audio(audio_chunk)

# 流式合成（配合 LLM 流式输出）
async with CosyVoiceSynthesizer() as tts:
    await tts.start_stream()
    
    async for text_chunk in llm_response_stream:
        await tts.feed_text(text_chunk)
    
    async for audio in tts.finish_stream():
        send_to_phone(audio)
```

环境变量：
- BAILIAN_API_KEY 或 DASHSCOPE_API_KEY: 百炼平台 API Key

参考文档：
https://help.aliyun.com/zh/model-studio/cosyvoice-python-sdk
"""

import asyncio
import logging
import queue
import threading
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from enum import Enum
from os import environ
from typing import Callable

_logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_MODEL = "cosyvoice-v3-flash"  # 快速版，1元/万字符
DEFAULT_VOICE = "longanyang"  # 龙安洋，成熟男声，适合客服场景
DEFAULT_SAMPLE_RATE = 8000  # 电话音频采样率

# 全局预热标记
_tts_warmed_up = False


class SynthesisState(Enum):
    """合成状态"""
    IDLE = "idle"
    STREAMING = "streaming"
    COMPLETING = "completing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SynthesisResult:
    """合成结果"""
    audio_data: bytes
    is_final: bool = False
    request_id: str = ""
    

class CosyVoiceError(Exception):
    """CosyVoice TTS 错误"""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class CosyVoiceSynthesizer:
    """
    阿里云 CosyVoice 实时语音合成器
    
    支持流式合成，边输入文本边输出音频。
    """
    
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        api_key: str | None = None,
    ):
        """
        初始化合成器
        
        Args:
            model: 模型名称，默认 cosyvoice-v3-flash
            voice: 音色名称，默认 longxiaochun
            sample_rate: 采样率，默认 8000（电话音频）
            api_key: API Key，不传则从环境变量读取
        """
        self._api_key = api_key or environ.get("BAILIAN_API_KEY") or environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "BAILIAN_API_KEY 或 DASHSCOPE_API_KEY 环境变量未设置！\n"
                "请在 .env 文件中添加：BAILIAN_API_KEY=sk-your-api-key"
            )
        
        self._model = model
        self._voice = voice
        self._sample_rate = sample_rate
        self._state = SynthesisState.IDLE
        
        # 音频队列（用于异步获取）
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._error: Exception | None = None
        self._synthesizer = None
        
        # 设置 DashScope API Key
        try:
            import dashscope
            dashscope.api_key = self._api_key
        except ImportError:
            raise ImportError(
                "dashscope 库未安装！请运行：pip install dashscope>=1.23.4"
            )
        
        _logger.info(
            "CosyVoice TTS initialized: model=%s, voice=%s, sample_rate=%d",
            model, voice, sample_rate
        )
    
    async def __aenter__(self) -> "CosyVoiceSynthesizer":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    async def close(self) -> None:
        """关闭合成器"""
        self._state = SynthesisState.COMPLETED
    
    def _get_audio_format(self):
        """获取音频格式"""
        from dashscope.audio.tts_v2 import AudioFormat
        
        if self._sample_rate == 8000:
            return AudioFormat.PCM_8000HZ_MONO_16BIT
        elif self._sample_rate == 16000:
            return AudioFormat.PCM_16000HZ_MONO_16BIT
        elif self._sample_rate == 22050:
            return AudioFormat.PCM_22050HZ_MONO_16BIT
        elif self._sample_rate == 24000:
            return AudioFormat.PCM_24000HZ_MONO_16BIT
        else:
            _logger.warning("Unsupported sample rate %d, using 16000", self._sample_rate)
            return AudioFormat.PCM_16000HZ_MONO_16BIT
    
    def _create_callback(self):
        """创建回调处理器"""
        from dashscope.audio.tts_v2 import ResultCallback
        
        audio_queue = self._audio_queue
        synthesizer = self
        
        class SynthesisCallback(ResultCallback):
            def on_open(self):
                _logger.debug("TTS connection opened")
            
            def on_complete(self):
                _logger.debug("TTS synthesis completed")
                audio_queue.put(None)  # 标记结束
            
            def on_error(self, message: str):
                _logger.error("TTS error: %s", message)
                synthesizer._error = CosyVoiceError("SYNTHESIS_ERROR", message)
                audio_queue.put(None)
            
            def on_close(self):
                _logger.debug("TTS connection closed")
            
            def on_data(self, data: bytes) -> None:
                """接收音频数据"""
                if data:
                    audio_queue.put(data)
        
        return SynthesisCallback()
    
    async def synthesize(self, text: str) -> bytes:
        """
        一次性合成（非流式）
        
        Args:
            text: 要合成的文本
        
        Returns:
            完整的音频数据（PCM 格式）
        """
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)
    
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        流式合成单段文本
        
        Args:
            text: 要合成的文本
        
        Yields:
            音频数据块（PCM 格式）
        """
        if not text or not text.strip():
            return
        
        from dashscope.audio.tts_v2 import SpeechSynthesizer
        
        self._state = SynthesisState.STREAMING
        self._audio_queue = queue.Queue()
        self._error = None
        
        callback = self._create_callback()
        
        # 在后台线程中运行合成
        def run_synthesis():
            try:
                synthesizer = SpeechSynthesizer(
                    model=self._model,
                    voice=self._voice,
                    format=self._get_audio_format(),
                    callback=callback,
                )
                
                # 流式发送文本
                # 单片不超过 2000 字符
                max_chunk_size = 2000
                for i in range(0, len(text), max_chunk_size):
                    chunk = text[i:i + max_chunk_size]
                    synthesizer.streaming_call(chunk)
                
                # 完成合成
                synthesizer.streaming_complete()
                
            except Exception as e:
                _logger.error("Synthesis error: %s", e)
                self._error = e
                self._audio_queue.put(None)
        
        # 启动合成线程
        thread = threading.Thread(target=run_synthesis, daemon=True)
        thread.start()
        
        # 异步获取音频数据
        while True:
            try:
                # 非阻塞获取
                audio = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._audio_queue.get(timeout=0.1)
                )
                
                if audio is None:
                    break
                
                yield audio
                
            except queue.Empty:
                # 队列为空，继续等待
                if not thread.is_alive() and self._audio_queue.empty():
                    break
                await asyncio.sleep(0.01)
        
        # 检查错误
        if self._error:
            raise self._error
        
        self._state = SynthesisState.COMPLETED
    
    async def start_stream(self) -> None:
        """
        开始流式合成会话
        
        用于配合 LLM 流式输出，边收到文本边合成。
        """
        from dashscope.audio.tts_v2 import SpeechSynthesizer
        
        self._state = SynthesisState.STREAMING
        self._audio_queue = queue.Queue()
        self._error = None
        
        callback = self._create_callback()
        
        self._synthesizer = SpeechSynthesizer(
            model=self._model,
            voice=self._voice,
            format=self._get_audio_format(),
            callback=callback,
        )
        
        _logger.debug("TTS stream started")
    
    async def feed_text(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        向流式会话发送文本并获取音频
        
        Args:
            text: 文本片段（单次不超过 2000 字符）
        
        Yields:
            音频数据块
        """
        if not self._synthesizer:
            raise RuntimeError("Stream not started, call start_stream() first")
        
        if not text or not text.strip():
            return
        
        # 确保单片不超过 2000 字符
        if len(text) > 2000:
            _logger.warning("Text too long (%d chars), truncating to 2000", len(text))
            text = text[:2000]
        
        # 在后台线程发送文本
        def send_text():
            try:
                self._synthesizer.streaming_call(text)
            except Exception as e:
                _logger.error("Error sending text: %s", e)
                self._error = e
        
        await asyncio.get_event_loop().run_in_executor(None, send_text)
        
        # 获取已生成的音频
        while not self._audio_queue.empty():
            audio = self._audio_queue.get_nowait()
            if audio:
                yield audio
    
    async def finish_stream(self) -> AsyncGenerator[bytes, None]:
        """
        完成流式合成并获取剩余音频
        
        Yields:
            剩余的音频数据块
        """
        if not self._synthesizer:
            return
        
        self._state = SynthesisState.COMPLETING
        
        # 在后台完成合成
        def complete():
            try:
                self._synthesizer.streaming_complete()
            except Exception as e:
                _logger.error("Error completing synthesis: %s", e)
                self._error = e
                self._audio_queue.put(None)
        
        await asyncio.get_event_loop().run_in_executor(None, complete)
        
        # 获取所有剩余音频
        while True:
            try:
                audio = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._audio_queue.get(timeout=0.5)
                )
                
                if audio is None:
                    break
                
                yield audio
                
            except queue.Empty:
                break
        
        self._synthesizer = None
        self._state = SynthesisState.COMPLETED
        
        if self._error:
            raise self._error
    
    @property
    def state(self) -> SynthesisState:
        return self._state
    
    @property
    def is_streaming(self) -> bool:
        return self._state == SynthesisState.STREAMING


# =============================================================================
# 快速同步合成（避免流式开销）
# =============================================================================

def warmup_tts(
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
) -> None:
    """
    预热 TTS（首次调用有初始化开销，预热可消除）
    """
    global _tts_warmed_up
    if _tts_warmed_up:
        return
    
    _logger.info("Warming up TTS...")
    try:
        import dashscope
        from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
        
        api_key = environ.get("BAILIAN_API_KEY") or environ.get("DASHSCOPE_API_KEY")
        if api_key:
            dashscope.api_key = api_key
            synthesizer = SpeechSynthesizer(
                model=model,
                voice=voice,
                format=AudioFormat.PCM_8000HZ_MONO_16BIT,
            )
            # 合成一个短句预热
            synthesizer.call("你好")
            _tts_warmed_up = True
            _logger.info("TTS warmed up successfully")
    except Exception as e:
        _logger.warning("TTS warmup failed: %s", e)


def synthesize_sync(
    text: str,
    model: str = DEFAULT_MODEL,
    voice: str = DEFAULT_VOICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """
    同步合成文本（每次创建新实例，确保连接正常）
    
    Args:
        text: 要合成的文本
        model: 模型
        voice: 音色
        sample_rate: 采样率
    
    Returns:
        音频数据（PCM 格式）
    """
    global _tts_warmed_up
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
    
    api_key = environ.get("BAILIAN_API_KEY") or environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")
    
    dashscope.api_key = api_key
    
    # 选择音频格式
    if sample_rate == 8000:
        audio_format = AudioFormat.PCM_8000HZ_MONO_16BIT
    elif sample_rate == 16000:
        audio_format = AudioFormat.PCM_16000HZ_MONO_16BIT
    else:
        audio_format = AudioFormat.PCM_16000HZ_MONO_16BIT
    
    # 每次创建新实例（WebSocket 连接不可复用）
    synthesizer = SpeechSynthesizer(
        model=model,
        voice=voice,
        format=audio_format,
    )
    
    # 直接调用，返回音频数据
    audio_data = synthesizer.call(text)
    _tts_warmed_up = True
    
    return audio_data if isinstance(audio_data, bytes) else b""


# 便捷函数
async def synthesize_text(
    text: str,
    voice: str = DEFAULT_VOICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """
    快速合成文本
    
    Args:
        text: 要合成的文本
        voice: 音色
        sample_rate: 采样率
    
    Returns:
        音频数据（PCM 格式）
    """
    async with CosyVoiceSynthesizer(voice=voice, sample_rate=sample_rate) as tts:
        return await tts.synthesize(text)


async def synthesize_to_file(
    text: str,
    output_path: str,
    voice: str = DEFAULT_VOICE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> None:
    """
    合成文本并保存到文件
    
    Args:
        text: 要合成的文本
        output_path: 输出文件路径（.pcm 或 .wav）
        voice: 音色
        sample_rate: 采样率
    """
    audio_data = await synthesize_text(text, voice, sample_rate)
    
    if output_path.endswith(".wav"):
        # 添加 WAV 头
        import struct
        
        num_channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8
        data_size = len(audio_data)
        
        wav_header = struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF',
            36 + data_size,
            b'WAVE',
            b'fmt ',
            16,  # PCM
            1,   # Audio format (PCM)
            num_channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
            b'data',
            data_size,
        )
        
        with open(output_path, 'wb') as f:
            f.write(wav_header)
            f.write(audio_data)
    else:
        # 直接保存 PCM
        with open(output_path, 'wb') as f:
            f.write(audio_data)
    
    _logger.info("Audio saved to %s (%d bytes)", output_path, len(audio_data))

