"""
本地对话测试路由

提供 REST API 端点用于本地对话测试。
支持文字和语音输入。
"""

import base64
from http import HTTPStatus

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.helpers.config import CONFIG
from app.helpers.local_chat import (
    VOICE_CHAT_HTML,
    ChatSession,
    chat_with_llm,
    get_or_create_session,
    get_session,
    pcm_to_wav,
    synthesize_speech,
    recognize_speech,
)
from app.helpers.logging import logger

# 创建路由器
router = APIRouter(prefix="/local-chat", tags=["Local Chat"])


class ChatRequest(BaseModel):
    """聊天请求（文字）"""
    session_id: str = Field(..., description="会话 ID")
    message: str = Field(..., min_length=1, max_length=2000, description="用户消息")
    synthesize_audio: bool = Field(default=True, description="是否合成语音")


class VoiceChatRequest(BaseModel):
    """语音聊天请求"""
    session_id: str = Field(..., description="会话 ID")
    audio_base64: str = Field(..., description="音频数据 (base64 WAV)")
    synthesize_audio: bool = Field(default=True, description="是否合成语音回复")


class ChatResponse(BaseModel):
    """聊天响应"""
    session_id: str
    response: str
    audio_base64: str | None = None


class VoiceChatResponse(BaseModel):
    """语音聊天响应"""
    session_id: str
    recognized_text: str  # ASR 识别出的文字
    response: str         # AI 回复
    audio_base64: str | None = None


class TtsRequest(BaseModel):
    """TTS 请求"""
    text: str = Field(..., min_length=1, max_length=500, description="要合成的文字")


class TtsResponse(BaseModel):
    """TTS 响应"""
    audio_base64: str


class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str
    message_count: int


class StatusResponse(BaseModel):
    """服务状态响应"""
    llm: str  # "ok", "mock", "error"
    tts: str
    asr: str


@router.get("/", response_class=HTMLResponse)
async def get_chat_page() -> HTMLResponse:
    """
    获取语音对话页面（电话模拟）
    """
    return HTMLResponse(content=VOICE_CHAT_HTML, status_code=HTTPStatus.OK)


@router.get("/status")
async def get_status() -> StatusResponse:
    """
    获取服务状态
    """
    import os
    
    # LLM 状态 - 检查环境变量
    llm_status = "ok" if os.environ.get("DASHSCOPE_API_KEY") else "error"
    
    # TTS 状态
    tts_status = "mock" if CONFIG.tts.mode == "mock" else "ok"
    
    # ASR 状态
    asr_status = "mock" if CONFIG.asr.mode == "mock" else "ok"
    
    return StatusResponse(llm=llm_status, tts=tts_status, asr=asr_status)


@router.post("/session")
async def create_session() -> SessionResponse:
    """
    创建新的聊天会话
    """
    session = get_or_create_session()
    return SessionResponse(
        session_id=session.session_id,
        message_count=len(session.messages),
    )


@router.get("/session/{session_id}")
async def get_session_info(session_id: str) -> SessionResponse:
    """
    获取会话信息
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    
    return SessionResponse(
        session_id=session.session_id,
        message_count=len(session.messages),
    )


@router.post("/chat")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    发送消息并获取 AI 回复
    """
    # 获取会话
    session = get_session(request.session_id)
    if not session:
        # 如果会话不存在，创建新会话
        session = get_or_create_session(request.session_id)
    
    # 与 LLM 对话
    response_text = await chat_with_llm(session, request.message)
    
    # 合成语音
    audio_base64 = None
    if request.synthesize_audio and CONFIG.tts.mode != "mock":
        try:
            pcm_audio = await synthesize_speech(response_text)
            if pcm_audio:
                # 转换为 WAV 格式（浏览器可以播放）
                wav_audio = pcm_to_wav(pcm_audio, sample_rate=8000)
                audio_base64 = base64.b64encode(wav_audio).decode("utf-8")
                logger.info("Audio synthesized: %d bytes WAV", len(wav_audio))
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
    
    return ChatResponse(
        session_id=session.session_id,
        response=response_text,
        audio_base64=audio_base64,
    )


@router.post("/voice-chat")
async def voice_chat(request: VoiceChatRequest) -> VoiceChatResponse:
    """
    语音对话：上传音频 → ASR 识别 → LLM 回复 → TTS 合成
    """
    import wave
    import io
    
    # 获取会话
    session = get_session(request.session_id)
    if not session:
        session = get_or_create_session(request.session_id)
    
    # 解码音频
    try:
        audio_bytes = base64.b64decode(request.audio_base64)
        logger.info("Received audio: %d bytes", len(audio_bytes))
    except Exception as e:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail=f"Invalid audio data: {e}",
        )
    
    # 从 WAV 提取 PCM 数据
    try:
        with io.BytesIO(audio_bytes) as wav_buffer:
            with wave.open(wav_buffer, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                pcm_data = wav_file.readframes(wav_file.getnframes())
                logger.info("Audio: %d Hz, %d frames", sample_rate, len(pcm_data) // 2)
    except Exception as e:
        logger.warning("Could not parse as WAV, treating as raw PCM: %s", e)
        pcm_data = audio_bytes
        sample_rate = 16000  # 默认采样率
    
    # ASR 识别
    recognized_text = await recognize_speech(pcm_data, sample_rate)
    if not recognized_text:
        recognized_text = "(无法识别语音)"
        logger.warning("ASR returned empty result")
    
    logger.info("ASR recognized: %s", recognized_text)
    
    # 与 LLM 对话
    response_text = await chat_with_llm(session, recognized_text)
    
    # TTS 合成
    audio_base64 = None
    if request.synthesize_audio and CONFIG.tts.mode != "mock":
        try:
            pcm_audio = await synthesize_speech(response_text)
            if pcm_audio:
                wav_audio = pcm_to_wav(pcm_audio, sample_rate=8000)
                audio_base64 = base64.b64encode(wav_audio).decode("utf-8")
                logger.info("TTS synthesized: %d bytes", len(wav_audio))
        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
    
    return VoiceChatResponse(
        session_id=session.session_id,
        recognized_text=recognized_text,
        response=response_text,
        audio_base64=audio_base64,
    )


@router.post("/tts")
async def synthesize_tts(request: TtsRequest) -> TtsResponse:
    """
    独立的 TTS 合成 API（用于异步获取音频）
    """
    if CONFIG.tts.mode == "mock":
        raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail="TTS is in mock mode")
    
    try:
        pcm_audio = await synthesize_speech(request.text)
        if pcm_audio:
            wav_audio = pcm_to_wav(pcm_audio, sample_rate=8000)
            return TtsResponse(audio_base64=base64.b64encode(wav_audio).decode("utf-8"))
        else:
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail="TTS failed")
    except Exception as e:
        logger.error("TTS error: %s", e)
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=str(e))


@router.delete("/session/{session_id}")
async def delete_session(session_id: str) -> JSONResponse:
    """
    删除会话
    """
    session = get_session(session_id)
    if not session:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND,
            detail=f"Session {session_id} not found",
        )
    
    # 从内存中删除会话（这里简化实现，实际上 _sessions 是私有的）
    from app.helpers.local_chat import _sessions
    del _sessions[session_id]
    
    return JSONResponse(
        content={"message": f"Session {session_id} deleted"},
        status_code=HTTPStatus.OK,
    )

