"""
LLM 配置模型

使用通义千问 / 百炼平台（阿里云）。

配置示例：
```yaml
llm:
  fast:
    model: qwen-turbo
    context: 8000
  slow:
    model: qwen-max
    context: 32000
```

环境变量：
- DASHSCOPE_API_KEY: 百炼平台 API Key
"""

import logging
from os import environ

from openai import AsyncOpenAI
from pydantic import BaseModel

from app.helpers.cache import lru_acache

# 使用标准 logging，避免循环导入
_logger = logging.getLogger(__name__)

# DashScope OpenAI 兼容端点
DASHSCOPE_OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class DeploymentModel(BaseModel, frozen=True):
    """
    单个 LLM 部署配置
    
    使用通义千问（qwen-turbo / qwen-max 等）
    """
    
    # 模型配置
    model: str
    context: int
    seed: int = 42
    temperature: float = 0.0
    
    # 端点配置（可选，默认使用 DashScope）
    endpoint: str = ""
    
    # API Key（可选，默认从环境变量读取）
    api_key: str | None = None

    @lru_acache()
    async def client(self) -> tuple[AsyncOpenAI, "DeploymentModel"]:
        """
        创建 LLM 客户端
        
        Returns:
            tuple: (AsyncOpenAI 客户端, 配置对象)
        """
        api_key = self.api_key or environ.get("DASHSCOPE_API_KEY")
        
        if not api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 环境变量未设置！\n"
                "请在 .env 文件中添加：DASHSCOPE_API_KEY=sk-your-api-key\n"
                "获取地址：https://bailian.console.aliyun.com/ → API-KEY 管理"
            )
        
        base_url = self.endpoint if self.endpoint else DASHSCOPE_OPENAI_BASE_URL
        
        _logger.info(
            "Creating DashScope client for model %s",
            self.model,
        )
        
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
        return client, self


class LlmModel(BaseModel):
    """
    LLM 配置（支持 fast/slow 双模型）
    
    - fast: 快速响应模型（qwen-turbo）
    - slow: 高质量模型（qwen-max）
    """
    
    fast: DeploymentModel
    slow: DeploymentModel

    def selected(self, is_fast: bool) -> DeploymentModel:
        """根据 is_fast 选择模型配置"""
        return self.fast if is_fast else self.slow
