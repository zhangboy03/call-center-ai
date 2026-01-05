"""
AI 搜索配置模型

支持两种模式：
- mock: Mock 搜索（本地开发用，返回空结果）
- opensearch: 阿里云 OpenSearch（待实现）

配置示例：
```yaml
ai_search:
  mode: mock  # 本地开发
  index: trainings
  top_n_documents: 5
```
"""

from functools import cached_property
from typing import Literal

from pydantic import BaseModel, Field

from app.persistence.isearch import ISearch


class AiSearchModel(BaseModel, frozen=True):
    """
    AI 搜索配置
    
    目前支持 Mock 模式，未来将支持阿里云 OpenSearch。
    """
    # 模式选择
    mode: Literal["mock", "opensearch"] = "mock"
    
    # 搜索配置
    index: str = "trainings"
    expansion_n_messages: int = Field(default=10, ge=1)
    top_n_documents: int = Field(default=5, ge=1)

    @cached_property
    def instance(self) -> ISearch:
        """根据模式返回对应的搜索实例"""
        from app.helpers.config import CONFIG
        
        if self.mode == "mock":
            from app.persistence.mock_search import MockSearch
            return MockSearch(cache=CONFIG.cache.instance)
        
        elif self.mode == "opensearch":
            # TODO: 实现阿里云 OpenSearch 支持
            raise NotImplementedError(
                "阿里云 OpenSearch 支持正在开发中。"
                "请使用 mode: mock 进行本地开发。"
            )
        
        else:
            raise ValueError(f"Unknown search mode: {self.mode}")
