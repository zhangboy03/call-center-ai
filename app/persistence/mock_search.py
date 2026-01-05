"""
Mock 搜索实现

用于本地开发，替代 Azure AI Search / 阿里云 OpenSearch。
返回空结果，不执行实际搜索。

使用方法：
    在 config.yaml 中配置：
    ```yaml
    ai_search:
      mode: mock  # 或 azure / opensearch
    ```
"""

from app.helpers.logging import logger
from app.models.readiness import ReadinessEnum
from app.models.training import TrainingModel
from app.persistence.icache import ICache
from app.persistence.isearch import ISearch


class MockSearch(ISearch):
    """
    Mock 搜索实现
    
    不执行实际搜索，返回空结果。
    用于本地开发和测试。
    """
    
    def __init__(self, cache: ICache):
        super().__init__(cache)
        logger.info("Using mock search (no actual search will be performed)")
    
    async def readiness(self) -> ReadinessEnum:
        """Mock 搜索始终就绪"""
        return ReadinessEnum.OK
    
    async def training_search_all(
        self,
        lang: str,
        text: str,
        cache_only: bool = False,
    ) -> list[TrainingModel] | None:
        """
        Mock 搜索，返回空结果
        
        在开发模式下，可以在这里返回一些硬编码的测试数据。
        """
        logger.debug(
            "Mock search called with lang=%s, text=%s (returning empty)",
            lang,
            text[:50] if text else None,
        )
        
        # 如果需要，可以返回一些测试数据
        # return [
        #     TrainingModel(
        #         id="mock-1",
        #         title="测试问答",
        #         content="这是一个测试回答。",
        #         score=0.95,
        #     ),
        # ]
        
        return None

