"""
语音合成（TTS）配置模型

支持两种模式：
- azure: Azure Cognitive Services Speech（原有）
- cosyvoice: 阿里云 CosyVoice 实时语音合成

配置示例：
```yaml
tts:
  mode: cosyvoice
  cosyvoice:
    model: cosyvoice-v3-flash
    voice: longanyang
    sample_rate: 8000
```

可用音色列表（cosyvoice-v3-flash）：
- longanyang: 龙安洋，成熟男声，适合客服
- loongstella: Stella，英文女声
- loongbella: Bella，英文女声

可用音色列表（cosyvoice-v1/v2）：
- longxiaochun: 龙小淳，温柔女声
- longxiaochun_v2: 龙小淳 v2
"""

from functools import cached_property
from typing import Literal

from pydantic import BaseModel, Field


class CosyVoiceModel(BaseModel, frozen=True):
    """阿里云 CosyVoice 配置"""
    model: str = Field(default="cosyvoice-v3-flash", description="模型名称")
    voice: str = Field(default="longanyang", description="音色名称（龙安洋）")
    sample_rate: int = Field(default=8000, description="采样率，电话音频通常是 8000")


class AzureTtsModel(BaseModel, frozen=True):
    """Azure Speech TTS 配置（保留兼容）"""
    region: str = ""
    voice: str = "zh-CN-XiaoxiaoNeural"


class TtsModel(BaseModel):
    """
    语音合成配置
    
    支持 Azure Speech 和 阿里云 CosyVoice 两种模式。
    """
    mode: Literal["azure", "cosyvoice", "mock"] = "cosyvoice"
    
    # CosyVoice 配置
    cosyvoice: CosyVoiceModel = Field(default_factory=CosyVoiceModel)
    
    # Azure Speech 配置（保留兼容）
    azure: AzureTtsModel = Field(default_factory=AzureTtsModel)
    
    @cached_property
    def instance(self):
        """
        获取 TTS 合成器实例（缓存）
        
        根据 mode 返回对应的合成器实例。
        """
        return self.get_synthesizer()
    
    def get_synthesizer(self):
        """
        获取 TTS 合成器（新实例）
        
        根据 mode 返回对应的合成器实例。
        """
        if self.mode == "cosyvoice":
            from app.helpers.tts_client import CosyVoiceSynthesizer
            return CosyVoiceSynthesizer(
                model=self.cosyvoice.model,
                voice=self.cosyvoice.voice,
                sample_rate=self.cosyvoice.sample_rate,
            )
        
        elif self.mode == "azure":
            raise NotImplementedError("Azure Speech TTS 模式请使用原有代码")
        
        elif self.mode == "mock":
            from app.helpers.tts_mock import MockTtsSynthesizer
            return MockTtsSynthesizer()
        
        else:
            raise ValueError(f"Unknown TTS mode: {self.mode}")

