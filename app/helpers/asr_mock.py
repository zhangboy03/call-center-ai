"""
Mock ASR 客户端

用于本地开发和测试，不需要真实的 ASR 服务。
"""

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import dataclass


@dataclass
class MockRecognitionResult:
    """Mock 识别结果"""
    text: str
    is_final: bool = False
    sentence_id: int = 0
    begin_time: int = 0
    end_time: int = 0


class MockAsrClient:
    """
    Mock ASR 客户端
    
    返回预设的测试文本，用于本地开发。
    """
    
    def __init__(self):
        self._is_recognizing = False
        self._mock_texts = [
            "你好",
            "我是张三",
            "我想咨询一下脑起搏器的术后随访",
            "好的，我知道了",
        ]
        self._text_index = 0
    
    async def __aenter__(self) -> "MockAsrClient":
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        pass
    
    async def connect(self) -> None:
        pass
    
    async def close(self) -> None:
        self._is_recognizing = False
    
    async def start_recognition(self, **kwargs) -> None:
        self._is_recognizing = True
        self._text_index = 0
    
    async def send_audio(self, audio_data: bytes) -> None:
        pass
    
    async def stream_audio(
        self,
        audio_frames: AsyncGenerator[bytes, None],
        frame_duration_ms: int = 100,
    ) -> AsyncGenerator[MockRecognitionResult, None]:
        """模拟流式识别"""
        frame_count = 0
        
        async for _ in audio_frames:
            frame_count += 1
            
            # 每 10 帧返回一个结果
            if frame_count % 10 == 0 and self._text_index < len(self._mock_texts):
                text = self._mock_texts[self._text_index]
                self._text_index += 1
                
                yield MockRecognitionResult(
                    text=text,
                    is_final=True,
                    sentence_id=self._text_index,
                )
                
                await asyncio.sleep(0.1)
    
    async def stop_recognition(self) -> MockRecognitionResult | None:
        self._is_recognizing = False
        return None
    
    @property
    def is_recognizing(self) -> bool:
        return self._is_recognizing

