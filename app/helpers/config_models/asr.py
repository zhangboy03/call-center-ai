"""
语音识别（ASR）配置模型

支持两种模式：
- azure: Azure Cognitive Services Speech（原有）
- paraformer: 阿里云 Paraformer 实时语音识别

配置示例：
```yaml
asr:
  mode: paraformer
  paraformer:
    model: paraformer-realtime-v2
    sample_rate: 8000
    language: zh
    hot_words:
      脑起搏器: 5
      帕金森: 5
      品驰: 5
```
"""

from functools import cached_property
from typing import Literal

from pydantic import BaseModel, Field


class ParaformerModel(BaseModel, frozen=True):
    """阿里云 Paraformer 配置"""
    model: str = "paraformer-realtime-v2"
    sample_rate: int = Field(default=16000, description="采样率，16kHz 适合高质量语音识别")
    format: str = Field(default="pcm", description="音频格式")
    language: str = Field(default="zh", description="识别语言")
    enable_punctuation: bool = Field(default=True, description="是否启用标点")
    enable_timestamp: bool = Field(default=True, description="是否启用时间戳")
    hot_words: dict[str, int] = Field(
        default_factory=dict,
        description="热词字典，格式为 {'词': 权重}，权重范围 1-5"
    )


class AzureSpeechModel(BaseModel, frozen=True):
    """Azure Speech 配置（保留兼容）"""
    region: str = ""
    resource_id: str = ""


class AsrModel(BaseModel):
    """
    语音识别配置
    
    支持 Azure Speech 和 阿里云 Paraformer 两种模式。
    """
    mode: Literal["azure", "paraformer", "mock"] = "paraformer"
    
    # Paraformer 配置
    paraformer: ParaformerModel = Field(default_factory=ParaformerModel)
    
    # Azure Speech 配置（保留兼容）
    azure: AzureSpeechModel = Field(default_factory=AzureSpeechModel)
    
    @cached_property
    def instance(self):
        """
        获取 ASR 客户端实例（缓存）
        
        根据 mode 返回对应的客户端实例。
        """
        return self.get_client()
    
    def get_client(self):
        """
        获取 ASR 客户端
        
        根据 mode 返回对应的客户端实例。
        """
        if self.mode == "paraformer":
            from app.helpers.asr_client import ParaformerClient
            return ParaformerClient(model=self.paraformer.model)
        
        elif self.mode == "azure":
            raise NotImplementedError("Azure Speech 模式请使用原有代码")
        
        elif self.mode == "mock":
            # Mock 模式，用于测试
            from app.helpers.asr_mock import MockAsrClient
            return MockAsrClient()
        
        else:
            raise ValueError(f"Unknown ASR mode: {self.mode}")

