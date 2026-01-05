"""
Mock TTS 合成器

用于本地开发和测试，不需要真实的 TTS 服务。
生成静音或简单的音频数据。
"""

import asyncio
import struct
import math
from collections.abc import AsyncGenerator


class MockTtsSynthesizer:
    """
    Mock TTS 合成器
    
    生成简单的测试音频（静音或正弦波）。
    """
    
    def __init__(self, sample_rate: int = 8000):
        self._sample_rate = sample_rate
        self._is_streaming = False
    
    async def __aenter__(self) -> "MockTtsSynthesizer":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
    
    async def close(self) -> None:
        self._is_streaming = False
    
    async def synthesize(self, text: str) -> bytes:
        """生成测试音频"""
        # 根据文本长度生成对应时长的音频
        # 假设每个字 0.3 秒
        duration = len(text) * 0.3
        return self._generate_audio(duration)
    
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """流式生成测试音频"""
        duration = len(text) * 0.3
        chunk_duration = 0.1  # 每 100ms 一个 chunk
        
        total_samples = int(duration * self._sample_rate)
        chunk_samples = int(chunk_duration * self._sample_rate)
        
        for i in range(0, total_samples, chunk_samples):
            chunk = self._generate_audio(chunk_duration)
            yield chunk
            await asyncio.sleep(0.05)  # 模拟处理延迟
    
    async def start_stream(self) -> None:
        self._is_streaming = True
    
    async def feed_text(self, text: str) -> AsyncGenerator[bytes, None]:
        """处理文本片段"""
        if text:
            chunk = self._generate_audio(len(text) * 0.1)
            yield chunk
    
    async def finish_stream(self) -> AsyncGenerator[bytes, None]:
        """完成流式合成"""
        self._is_streaming = False
        # 生成一小段尾音
        yield self._generate_audio(0.1)
    
    def _generate_audio(self, duration: float) -> bytes:
        """
        生成测试音频
        
        生成一个简单的 440Hz 正弦波（A4 音符）
        """
        num_samples = int(duration * self._sample_rate)
        frequency = 440.0
        amplitude = 8000  # 低音量
        
        samples = []
        for i in range(num_samples):
            t = i / self._sample_rate
            value = int(amplitude * math.sin(2 * math.pi * frequency * t))
            samples.append(value)
        
        return struct.pack(f"<{len(samples)}h", *samples)
    
    @property
    def is_streaming(self) -> bool:
        return self._is_streaming

