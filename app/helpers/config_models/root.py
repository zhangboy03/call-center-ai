"""
根配置模型

品驰关爱中心 AI 客服系统配置。
基于阿里云服务（通义千问 + CosyVoice + Paraformer）。
"""

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from app.helpers.config_models.ai_search import AiSearchModel
from app.helpers.config_models.asr import AsrModel
from app.helpers.config_models.cache import CacheModel
from app.helpers.config_models.conversation import ConversationModel
from app.helpers.config_models.database import DatabaseModel
from app.helpers.config_models.llm import LlmModel
from app.helpers.config_models.prompts import PromptsModel
from app.helpers.config_models.queue import QueueModel
from app.helpers.config_models.resources import ResourcesModel
from app.helpers.config_models.tts import TtsModel


class RootModel(BaseSettings):
    """
    应用根配置

    核心服务：
    - llm: LLM 配置（通义千问）
    - database: 数据库配置（MongoDB）
    - cache: 缓存配置（Redis）
    - queue: 消息队列配置（内存队列）

    语音服务：
    - asr: 语音识别（Paraformer）
    - tts: 语音合成（CosyVoice）

    业务配置：
    - ai_search: 知识库搜索（Mock/OpenSearch）
    - conversation: 对话配置
    - prompts: 提示词配置
    - resources: 静态资源配置
    """

    # Pydantic settings
    model_config = SettingsConfigDict(
        env_ignore_empty=True,
        env_nested_delimiter="__",
        env_prefix="",
    )

    # 基本配置
    public_domain: str = Field(default="http://localhost:8080", frozen=True)
    version: str = Field(default="0.0.0-dev", frozen=True)

    # === 核心服务 ===
    llm: LlmModel
    database: DatabaseModel
    cache: CacheModel = Field(default_factory=CacheModel)
    queue: QueueModel = Field(default_factory=QueueModel)

    # === 语音服务 ===
    asr: AsrModel = Field(default_factory=AsrModel)
    tts: TtsModel = Field(default_factory=TtsModel)

    # === 业务配置 ===
    ai_search: AiSearchModel = Field(default_factory=AiSearchModel)
    conversation: ConversationModel = Field(serialization_alias="workflow")
    prompts: PromptsModel = Field(default_factory=PromptsModel)
    resources: ResourcesModel = Field(default_factory=ResourcesModel)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],  # noqa: ARG003
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """
        配置源优先级：
        1. 环境变量
        2. .env 文件
        3. Docker secrets
        4. 初始设置
        """
        return env_settings, dotenv_settings, file_secret_settings, init_settings
