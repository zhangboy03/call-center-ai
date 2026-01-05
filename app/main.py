"""
品驰关爱中心 - AI 客服系统

基于阿里云服务的智能客服系统：
- LLM: 通义千问 (qwen-turbo / qwen-max)
- TTS: 阿里云 CosyVoice (cosyvoice-v3-flash)
- ASR: 阿里云 Paraformer (paraformer-realtime-v2)
- 数据库: MongoDB
- 缓存: Redis

目前支持：
- 本地对话测试（Web 界面）
- REST API 调用

未来支持：
- 阿里云 VMS 电话服务
- 阿里云 OpenSearch 知识库
"""

import asyncio
from contextlib import asynccontextmanager
from http import HTTPStatus

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.helpers.config import CONFIG
from app.helpers.logging import logger
from app.local_chat_routes import router as local_chat_router
from app.models.readiness import ReadinessCheckModel, ReadinessEnum, ReadinessModel
from app.streaming_routes import router as streaming_router
from app.test_platform import router as test_platform_router

# 启动日志
logger.info("call-center-ai v%s", CONFIG.version)

# 持久化实例
_cache = CONFIG.cache.instance
_db = CONFIG.database.instance
_search = CONFIG.ai_search.instance


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    """应用生命周期管理"""
    logger.info("Application starting...")

    # 预热 TTS（消除首次调用延迟）
    if CONFIG.tts.mode == "cosyvoice":
        try:
            from app.helpers.tts_client import warmup_tts

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: warmup_tts(
                    model=CONFIG.tts.cosyvoice.model,
                    voice=CONFIG.tts.cosyvoice.voice,
                ),
            )
        except Exception as e:
            logger.warning("TTS warmup failed: %s", e)

        yield

    # 关闭时的清理工作
    logger.info("Application shutting down...")


# FastAPI 应用
api = FastAPI(
    title="品驰关爱中心 AI 客服",
    description="基于阿里云的智能客服系统，支持语音对话和文字聊天。",
    version=CONFIG.version,
    lifespan=lifespan,
    contact={
        "name": "品驰医疗",
        "url": "https://www.pins-medical.com",
    },
)

# 注册路由
api.include_router(local_chat_router)
api.include_router(streaming_router)  # 流式语音对话
api.include_router(test_platform_router)  # 测试平台


@api.get("/")
async def root():
    """根路径"""
    return {
        "message": "品驰关爱中心 AI 客服系统",
        "version": CONFIG.version,
        "endpoints": {
            "test_platform": "/test/",  # 测试平台（推荐）
            "streaming": "/streaming/",  # 流式语音对话
            "local_chat": "/local-chat/",  # 旧版对话测试
            "health": "/health/readiness",
            "docs": "/docs",
        },
    }


@api.get("/health/liveness")
async def health_liveness_get() -> None:
    """
    存活检查

    检查服务是否在运行。
    """
    return


@api.get("/health/readiness", status_code=HTTPStatus.OK)
async def health_readiness_get() -> JSONResponse:
    """
    就绪检查

    检查所有依赖服务是否就绪：
    - 缓存 (Redis)
    - 数据库 (MongoDB)
    - 搜索 (Mock/OpenSearch)
    """
    # 并行检查所有组件
    cache_check, store_check, search_check = await asyncio.gather(
        _cache.readiness(),
        _db.readiness(),
        _search.readiness(),
    )

    readiness = ReadinessModel(
        status=ReadinessEnum.OK,
        checks=[
            ReadinessCheckModel(id="cache", status=cache_check),
            ReadinessCheckModel(id="store", status=store_check),
            ReadinessCheckModel(id="search", status=search_check),
            ReadinessCheckModel(id="startup", status=ReadinessEnum.OK),
        ],
    )

    # 如果任何检查失败，整体状态为失败
    status_code = HTTPStatus.OK
    for check in readiness.checks:
        if check.status != ReadinessEnum.OK:
            readiness.status = ReadinessEnum.FAIL
            status_code = HTTPStatus.SERVICE_UNAVAILABLE
            break

    return JSONResponse(
        content=readiness.model_dump(mode="json"),
        status_code=status_code,
    )


# =============================================================================
# 以下是为未来 VMS 电话服务预留的占位符
# =============================================================================

# TODO: 阿里云 VMS 电话服务集成
# - 接收来电回调
# - WebSocket 音频流处理
# - ASR + LLM + TTS 实时对话

# TODO: 阿里云 OpenSearch 知识库集成
# - 向量搜索
# - RAG 增强对话
