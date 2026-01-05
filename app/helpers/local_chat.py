"""
本地对话测试模块

提供一个简单的 Web 界面，用于测试完整的客服对话流程：
- 文字输入 → LLM → 文字回复
- TTS 语音合成和播放
- （可选）音频上传 → ASR → 文字

无需真实电话，在浏览器中即可测试。
"""

import asyncio
import base64
import io
import wave
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from app.helpers.config import CONFIG
from app.helpers.logging import logger


class ChatMessage(BaseModel):
    """聊天消息"""
    role: str = Field(..., description="消息角色: user/assistant/system")
    content: str = Field(..., description="消息内容")
    timestamp: datetime = Field(default_factory=datetime.now)
    audio_base64: str | None = Field(default=None, description="TTS 音频 (base64 PCM)")


class ChatSession(BaseModel):
    """聊天会话"""
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    messages: list[ChatMessage] = Field(default_factory=list)
    system_prompt: str = Field(
        default="""你是品驰关爱中心的智能客服小驰，正在进行术后随访电话。

要求：
- 每次回复控制在20-30字
- 像真人打电话一样自然简洁
- 一次只问一个问题或回应一个话题
- 语气亲切但不啰嗦"""
    )


# 内存中的会话存储（简单实现，重启后丢失）
_sessions: dict[str, ChatSession] = {}


def get_or_create_session(session_id: str | None = None) -> ChatSession:
    """获取或创建会话"""
    if session_id and session_id in _sessions:
        return _sessions[session_id]
    
    session = ChatSession()
    _sessions[session.session_id] = session
    logger.info("Created new chat session: %s", session.session_id)
    return session


def get_session(session_id: str) -> ChatSession | None:
    """获取会话"""
    return _sessions.get(session_id)


async def chat_with_llm(
    session: ChatSession,
    user_message: str,
) -> str:
    """
    与 LLM 对话
    
    Args:
        session: 聊天会话
        user_message: 用户消息
        
    Returns:
        AI 回复内容
    """
    # 添加用户消息
    session.messages.append(ChatMessage(role="user", content=user_message))
    
    # 构建消息列表
    messages = [
        {"role": "system", "content": session.system_prompt},
    ]
    
    # 添加历史消息（保留最近 10 条）
    for msg in session.messages[-10:]:
        messages.append({"role": msg.role, "content": msg.content})
    
    # 调用 LLM
    import time
    try:
        # 使用配置中的 LLM
        llm_model = CONFIG.llm.fast
        # client() 是异步方法，返回 (AsyncOpenAI, DeploymentModel)
        client, _ = await llm_model.client()
        
        logger.info("[计时] LLM开始: %s", llm_model.model)
        t0 = time.time()
        
        response = await client.chat.completions.create(
            model=llm_model.model,
            messages=messages,
            max_tokens=100,  # 允许完整回复
            temperature=0.7,
        )
        
        t1 = time.time()
        assistant_content = response.choices[0].message.content or "抱歉，我没有理解您的意思。"
        
        logger.info("[计时] LLM完成: %.2f秒, %d字", t1 - t0, len(assistant_content))
        
        # 添加助手消息
        session.messages.append(ChatMessage(role="assistant", content=assistant_content))
        
        return assistant_content
        
    except Exception as e:
        logger.error("LLM error: %s", e)
        error_msg = f"抱歉，系统出现问题：{str(e)[:50]}"
        session.messages.append(ChatMessage(role="assistant", content=error_msg))
        return error_msg


async def synthesize_speech(text: str) -> bytes | None:
    """
    将文字转换为语音
    
    使用同步调用方式，比流式更快。
    
    Args:
        text: 要合成的文字
        
    Returns:
        PCM 音频数据 (8kHz, 16bit, mono)，失败返回 None
    """
    import time
    
    try:
        # 检查 TTS 模式
        if CONFIG.tts.mode == "mock":
            logger.info("Using mock TTS, returning empty audio")
            return None
        
        logger.info("[计时] TTS开始: %d字", len(text))
        t0 = time.time()
        
        # 使用同步调用（更快）
        from app.helpers.tts_client import synthesize_sync
        
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(
            None,
            lambda: synthesize_sync(
                text=text,
                model=CONFIG.tts.cosyvoice.model,
                voice=CONFIG.tts.cosyvoice.voice,
                sample_rate=CONFIG.tts.cosyvoice.sample_rate,
            )
        )
        
        t1 = time.time()
        
        if audio_data:
            logger.info("[计时] TTS完成: %.2f秒, %d字节", t1 - t0, len(audio_data))
            return audio_data
        
        return None
        
    except Exception as e:
        logger.error("TTS error: %s", e)
        return None


async def recognize_speech(audio_data: bytes, sample_rate: int = 16000) -> str | None:
    """
    将语音转换为文字
    
    使用阿里云实时语音识别（DashScope SDK）。
    
    Args:
        audio_data: PCM 音频数据 (16bit mono)
        sample_rate: 采样率
        
    Returns:
        识别出的文字，失败返回 None
    """
    import time
    
    try:
        # 检查 ASR 模式
        if CONFIG.asr.mode == "mock":
            logger.info("Using mock ASR")
            return "这是模拟的语音识别结果"
        
        # 使用 DashScope SDK 的同步识别
        from app.helpers.asr_client import recognize_audio_sync
        
        audio_duration = len(audio_data) / sample_rate / 2  # 秒
        logger.info("[计时] ASR开始: %.1f秒音频, %d字节", audio_duration, len(audio_data))
        
        t0 = time.time()
        
        # 在线程池中运行同步识别（避免阻塞事件循环）
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,  # 使用默认线程池
            lambda: recognize_audio_sync(
                audio_data=audio_data,
                sample_rate=sample_rate,
                format="pcm",
                model="paraformer-realtime-v2",
                language="zh",
            )
        )
        
        t1 = time.time()
        logger.info("[计时] ASR完成: %.2f秒, 结果: %s", t1 - t0, result[:30] if result else "(empty)")
        
        return result
        
    except Exception as e:
        logger.error("ASR error: %s", e)
        return None


def pcm_to_wav(pcm_data: bytes, sample_rate: int = 8000, channels: int = 1, sample_width: int = 2) -> bytes:
    """
    将 PCM 数据转换为 WAV 格式
    
    Args:
        pcm_data: PCM 音频数据
        sample_rate: 采样率
        channels: 声道数
        sample_width: 采样宽度（字节）
        
    Returns:
        WAV 格式的音频数据
    """
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    
    buffer.seek(0)
    return buffer.read()


# HTML 模板 - 真实电话模拟界面（自动语音检测）
VOICE_CHAT_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>品驰关爱中心 - 来电</title>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #10b981;
            --danger: #ef4444;
            --bg-dark: #0a0a0f;
            --bg-card: #16161d;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Noto Sans SC', sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        /* 来电界面 */
        .incoming-call {
            position: fixed;
            inset: 0;
            background: linear-gradient(180deg, #1a1a2e 0%, #0a0a0f 100%);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 100;
        }
        .incoming-call.hidden { display: none; }
        
        .caller-avatar {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 3rem;
            margin-bottom: 1.5rem;
            animation: pulse-ring 2s ease-out infinite;
        }
        @keyframes pulse-ring {
            0% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.7); }
            70% { box-shadow: 0 0 0 30px rgba(99, 102, 241, 0); }
            100% { box-shadow: 0 0 0 0 rgba(99, 102, 241, 0); }
        }
        
        .caller-name { font-size: 1.5rem; font-weight: 500; margin-bottom: 0.5rem; }
        .caller-info { color: var(--text-secondary); margin-bottom: 3rem; }
        
        .call-actions {
            display: flex;
            gap: 4rem;
        }
        .call-btn {
            width: 70px;
            height: 70px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.5rem;
            transition: transform 0.2s;
        }
        .call-btn:hover { transform: scale(1.1); }
        .call-btn.accept { background: var(--primary); }
        .call-btn.reject { background: var(--danger); }
        
        /* 通话界面 */
        .call-screen {
            flex: 1;
            display: none;
            flex-direction: column;
        }
        .call-screen.active { display: flex; }
        
        .call-header {
            background: var(--bg-card);
            padding: 1rem;
            text-align: center;
            border-bottom: 1px solid #2a2a35;
        }
        .call-timer {
            font-size: 0.9rem;
            color: var(--text-secondary);
            margin-top: 0.25rem;
        }
        
        .conversation {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }
        
        .message {
            max-width: 80%;
            padding: 0.75rem 1rem;
            border-radius: 1rem;
            animation: fadeIn 0.3s;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user {
            align-self: flex-end;
            background: #2563eb;
            border-bottom-right-radius: 0.25rem;
        }
        .message.assistant {
            align-self: flex-start;
            background: var(--bg-card);
            border: 1px solid #2a2a35;
            border-bottom-left-radius: 0.25rem;
        }
        
        /* 状态栏 */
        .status-bar {
            background: var(--bg-card);
            padding: 1rem;
            text-align: center;
            border-top: 1px solid #2a2a35;
        }
        .status-text {
            font-size: 0.9rem;
            color: var(--text-secondary);
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            min-height: 1.5rem;
        }
        
        /* 音量指示器 */
        .volume-indicator {
            display: flex;
            gap: 3px;
            align-items: center;
        }
        .volume-bar {
            width: 4px;
            height: 20px;
            background: var(--primary);
            border-radius: 2px;
            transition: height 0.1s;
        }
        
        /* 波形动画 */
        .wave-animation {
            display: flex;
            gap: 3px;
            align-items: center;
        }
        .wave-animation span {
            width: 4px;
            height: 16px;
            background: var(--primary);
            border-radius: 2px;
            animation: wave 0.8s ease-in-out infinite;
        }
        .wave-animation span:nth-child(2) { animation-delay: 0.1s; }
        .wave-animation span:nth-child(3) { animation-delay: 0.2s; }
        .wave-animation span:nth-child(4) { animation-delay: 0.3s; }
        @keyframes wave {
            0%, 100% { transform: scaleY(0.5); }
            50% { transform: scaleY(1.5); }
        }
        
        /* 挂断按钮 */
        .hangup-btn {
            position: fixed;
            bottom: 2rem;
            left: 50%;
            transform: translateX(-50%);
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: var(--danger);
            border: none;
            cursor: pointer;
            font-size: 1.25rem;
            color: white;
            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4);
        }
        .hangup-btn:hover { transform: translateX(-50%) scale(1.1); }
    </style>
</head>
<body>
    <!-- 来电界面 -->
    <div class="incoming-call" id="incoming-call">
        <div class="caller-avatar">🤖</div>
        <div class="caller-name">品驰关爱中心</div>
        <div class="caller-info">术后随访电话</div>
        <div class="call-actions">
            <button class="call-btn reject" onclick="rejectCall()">📵</button>
            <button class="call-btn accept" onclick="acceptCall()">📞</button>
        </div>
    </div>
    
    <!-- 通话界面 -->
    <div class="call-screen" id="call-screen">
        <div class="call-header">
            <div>🤖 品驰关爱中心 · 小驰</div>
            <div class="call-timer" id="call-timer">00:00</div>
        </div>
        
        <div class="conversation" id="conversation"></div>
        
        <div class="status-bar">
            <div class="status-text" id="status-text">准备中...</div>
        </div>
        
        <button class="hangup-btn" onclick="hangupCall()">📵</button>
    </div>

    <script>
        let sessionId = null;
        let audioContext = null;
        let mediaStream = null;
        let analyser = null;
        let isListening = false;
        let isProcessing = false;
        let isPlaying = false;
        let callStartTime = null;
        let timerInterval = null;
        let silenceTimer = null;
        let audioChunks = [];
        let mediaRecorder = null;
        
        // VAD 参数（优化低延迟）
        const SILENCE_THRESHOLD = 0.015;  // 静音阈值（稍高更容易检测结束）
        const SILENCE_DURATION = 500;     // 静音持续时间（ms）触发发送
        const MIN_RECORD_TIME = 300;      // 最短录音时间（ms）
        
        // 拒接电话
        function rejectCall() {
            alert('您已拒接来电');
            location.reload();
        }
        
        // 接听电话
        async function acceptCall() {
            try {
                // 请求麦克风权限
                mediaStream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                        autoGainControl: true,
                    }
                });
                
                // 创建会话
                const resp = await fetch('/local-chat/session', { method: 'POST' });
                const data = await resp.json();
                sessionId = data.session_id;
                
                // 切换界面
                document.getElementById('incoming-call').classList.add('hidden');
                document.getElementById('call-screen').classList.add('active');
                
                // 启动计时器
                callStartTime = Date.now();
                timerInterval = setInterval(updateTimer, 1000);
                
                // 初始化音频分析
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(mediaStream);
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                source.connect(analyser);
                
                // AI 先打招呼
                setStatus('connecting', '正在连接...');
                await aiGreeting();
                
            } catch (e) {
                alert('无法访问麦克风: ' + e.message);
                location.reload();
            }
        }
        
        // AI 主动问候
        async function aiGreeting() {
            setStatus('thinking', '小驰正在准备...');
            
            try {
                const resp = await fetch('/local-chat/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        message: '请开始问候患者',
                        synthesize_audio: true
                    })
                });
                
                const data = await resp.json();
                addMessage('assistant', data.response);
                
                if (data.audio_base64) {
                    setStatus('playing', '小驰正在说话...');
                    await playAudio(data.audio_base64);
                }
                
                // 播放完毕，开始监听用户
                startListening();
                
            } catch (e) {
                console.error('Greeting error:', e);
                setStatus('error', '连接失败');
            }
        }
        
        // 开始监听用户说话
        function startListening() {
            if (isListening || isProcessing || isPlaying) return;
            
            isListening = true;
            audioChunks = [];
            
            setStatus('listening', '正在听您说话...');
            
            // 创建录音器
            mediaRecorder = new MediaRecorder(mediaStream, {
                mimeType: 'audio/webm;codecs=opus'
            });
            
            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    audioChunks.push(e.data);
                }
            };
            
            mediaRecorder.onstop = async () => {
                if (audioChunks.length > 0 && !isPlaying) {
                    await processUserSpeech();
                }
            };
            
            mediaRecorder.start(100);
            
            // 开始 VAD 检测
            detectSpeech();
        }
        
        // VAD 语音活动检测
        let recordStartTime = 0;
        let lastSpeechTime = 0;
        
        function detectSpeech() {
            if (!isListening || !analyser) return;
            
            const dataArray = new Uint8Array(analyser.frequencyBinCount);
            analyser.getByteFrequencyData(dataArray);
            
            // 计算音量
            const volume = dataArray.reduce((a, b) => a + b, 0) / dataArray.length / 255;
            
            const now = Date.now();
            
            if (volume > SILENCE_THRESHOLD) {
                // 检测到声音
                if (recordStartTime === 0) {
                    recordStartTime = now;
                }
                lastSpeechTime = now;
                
                // 更新状态显示音量
                updateVolumeIndicator(volume);
                
                // 清除静音计时器
                if (silenceTimer) {
                    clearTimeout(silenceTimer);
                    silenceTimer = null;
                }
            } else {
                // 静音
                if (recordStartTime > 0 && lastSpeechTime > 0) {
                    const recordDuration = now - recordStartTime;
                    const silenceDuration = now - lastSpeechTime;
                    
                    // 录音超过最短时间 && 静音超过阈值
                    if (recordDuration > MIN_RECORD_TIME && silenceDuration > SILENCE_DURATION) {
                        // 停止录音并处理
                        stopListening();
                        return;
                    }
                }
            }
            
            // 继续检测
            requestAnimationFrame(detectSpeech);
        }
        
        // 更新音量指示器
        function updateVolumeIndicator(volume) {
            const statusEl = document.getElementById('status-text');
            const barCount = 4;
            let bars = '';
            for (let i = 0; i < barCount; i++) {
                const height = Math.min(30, 8 + volume * 100 * (1 + Math.random() * 0.5));
                bars += `<div class="volume-bar" style="height:${height}px"></div>`;
            }
            statusEl.innerHTML = `<div class="volume-indicator">${bars}</div> 正在听您说话...`;
        }
        
        // 停止监听
        function stopListening() {
            if (!isListening) return;
            
            isListening = false;
            recordStartTime = 0;
            lastSpeechTime = 0;
            
            if (silenceTimer) {
                clearTimeout(silenceTimer);
                silenceTimer = null;
            }
            
            if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                mediaRecorder.stop();
            }
        }
        
        // 处理用户语音
        async function processUserSpeech() {
            if (isProcessing || audioChunks.length === 0) return;
            
            isProcessing = true;
            setStatus('processing', '正在识别...');
            
            try {
                // 转换为 WAV
                const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                const wavBlob = await convertToWav(audioBlob);
                const base64 = await blobToBase64(wavBlob);
                
                // 发送到服务器
                const resp = await fetch('/local-chat/voice-chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        audio_base64: base64,
                        synthesize_audio: true
                    })
                });
                
                const data = await resp.json();
                
                // 显示用户消息
                if (data.recognized_text && data.recognized_text !== '(无法识别语音)') {
                    addMessage('user', data.recognized_text);
                    
                    // 显示 AI 回复
                    addMessage('assistant', data.response);
                    
                    // 播放 AI 语音
                    if (data.audio_base64) {
                        setStatus('playing', '小驰正在说话...');
                        await playAudio(data.audio_base64);
                    }
                } else {
                    setStatus('error', '没有听清，请再说一遍');
                    await sleep(1500);
                }
                
                // 继续监听
                isProcessing = false;
                startListening();
                
            } catch (e) {
                console.error('Process error:', e);
                setStatus('error', '处理失败');
                isProcessing = false;
                await sleep(1500);
                startListening();
            }
        }
        
        // 播放音频
        async function playAudio(base64) {
            isPlaying = true;
            
            try {
                const binary = atob(base64);
                const bytes = new Uint8Array(binary.length);
                for (let i = 0; i < binary.length; i++) {
                    bytes[i] = binary.charCodeAt(i);
                }
                
                const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                
                await new Promise(resolve => {
                    source.onended = resolve;
                    source.start(0);
                });
                
            } catch (e) {
                console.error('Playback error:', e);
            }
            
            isPlaying = false;
        }
        
        // 添加消息
        function addMessage(role, content) {
            const container = document.getElementById('conversation');
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.textContent = content;
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        // 设置状态
        function setStatus(type, text) {
            const el = document.getElementById('status-text');
            if (type === 'listening') {
                el.innerHTML = `<div class="wave-animation"><span></span><span></span><span></span><span></span></div> ${text}`;
            } else if (type === 'playing') {
                el.innerHTML = `🔊 ${text}`;
            } else if (type === 'processing' || type === 'thinking' || type === 'connecting') {
                el.innerHTML = `⏳ ${text}`;
            } else if (type === 'error') {
                el.innerHTML = `⚠️ ${text}`;
            } else {
                el.textContent = text;
            }
        }
        
        // 更新计时器
        function updateTimer() {
            if (!callStartTime) return;
            const elapsed = Math.floor((Date.now() - callStartTime) / 1000);
            const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
            const secs = (elapsed % 60).toString().padStart(2, '0');
            document.getElementById('call-timer').textContent = `${mins}:${secs}`;
        }
        
        // 挂断电话
        function hangupCall() {
            if (timerInterval) clearInterval(timerInterval);
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
            if (audioContext) audioContext.close();
            
            alert('通话已结束');
            location.reload();
        }
        
        // 工具函数
        function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }
        
        function blobToBase64(blob) {
            return new Promise(resolve => {
                const reader = new FileReader();
                reader.onloadend = () => resolve(reader.result.split(',')[1]);
                reader.readAsDataURL(blob);
            });
        }
        
        async function convertToWav(webmBlob) {
            const ctx = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await webmBlob.arrayBuffer();
            const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
            
            const offlineCtx = new OfflineAudioContext(1, audioBuffer.duration * 16000, 16000);
            const source = offlineCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(offlineCtx.destination);
            source.start(0);
            
            const rendered = await offlineCtx.startRendering();
            return audioBufferToWav(rendered);
        }
        
        function audioBufferToWav(buffer) {
            const samples = buffer.getChannelData(0);
            const wavBuffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(wavBuffer);
            
            const writeString = (offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };
            
            writeString(0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true);
            view.setUint16(22, 1, true);
            view.setUint32(24, buffer.sampleRate, true);
            view.setUint32(28, buffer.sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, samples.length * 2, true);
            
            let offset = 44;
            for (let i = 0; i < samples.length; i++) {
                const s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                offset += 2;
            }
            
            return new Blob([wavBuffer], { type: 'audio/wav' });
        }
    </script>
</body>
</html>
"""

# HTML 模板 - 文字聊天界面（备用）
CHAT_HTML = """
        
        .message.user {
            align-self: flex-end;
        }
        
        .message.assistant {
            align-self: flex-start;
        }
        
        .message-bubble {
            padding: 0.875rem 1rem;
            border-radius: 1rem;
            line-height: 1.5;
            font-size: 0.9375rem;
        }
        
        .user .message-bubble {
            background: var(--primary);
            border-bottom-right-radius: 0.25rem;
        }
        
        .assistant .message-bubble {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-bottom-left-radius: 0.25rem;
        }
        
        .message-label {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
            padding: 0 0.25rem;
        }
        
        .user .message-label {
            text-align: right;
        }
        
        .control-panel {
            background: var(--bg-card);
            border-top: 1px solid var(--border);
            padding: 1.5rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 1rem;
        }
        
        .voice-status {
            font-size: 0.875rem;
            color: var(--text-secondary);
            min-height: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .voice-status.recording {
            color: var(--danger);
        }
        
        .voice-status.processing {
            color: var(--primary);
        }
        
        .voice-status.playing {
            color: #8b5cf6;
        }
        
        .recording-wave {
            display: flex;
            gap: 3px;
            align-items: center;
        }
        
        .recording-wave span {
            width: 3px;
            height: 12px;
            background: currentColor;
            border-radius: 2px;
            animation: wave 0.5s ease-in-out infinite;
        }
        
        .recording-wave span:nth-child(2) { animation-delay: 0.1s; }
        .recording-wave span:nth-child(3) { animation-delay: 0.2s; }
        .recording-wave span:nth-child(4) { animation-delay: 0.3s; }
        
        @keyframes wave {
            0%, 100% { transform: scaleY(0.5); }
            50% { transform: scaleY(1.5); }
        }
        
        .mic-button {
            width: 100px;
            height: 100px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(145deg, var(--primary), var(--primary-dark));
            color: white;
            font-size: 2.5rem;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 20px var(--glow);
            display: flex;
            align-items: center;
            justify-content: center;
            user-select: none;
            -webkit-user-select: none;
        }
        
        .mic-button:active,
        .mic-button.recording {
            transform: scale(1.1);
            background: linear-gradient(145deg, var(--danger), #dc2626);
            box-shadow: 0 4px 30px rgba(239, 68, 68, 0.5);
        }
        
        .mic-button:disabled {
            opacity: 0.5;
            cursor: not-allowed;
            transform: none;
        }
        
        .mic-button svg {
            width: 40px;
            height: 40px;
        }
        
        .hint {
            font-size: 0.75rem;
            color: var(--text-muted);
        }
        
        .text-input-toggle {
            font-size: 0.8rem;
            color: var(--text-secondary);
            text-decoration: underline;
            cursor: pointer;
            background: none;
            border: none;
            margin-top: 0.5rem;
        }
        
        .text-input-container {
            display: none;
            width: 100%;
            gap: 0.5rem;
            padding-top: 0.5rem;
        }
        
        .text-input-container.show {
            display: flex;
        }
        
        .text-input-container input {
            flex: 1;
            background: var(--bg-input);
            border: 1px solid var(--border);
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            color: var(--text-primary);
            font-size: 0.9375rem;
        }
        
        .text-input-container input:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        .text-input-container button {
            background: var(--primary);
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 1.25rem;
            color: white;
            font-size: 0.875rem;
            cursor: pointer;
        }
        
        .welcome-screen {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            text-align: center;
        }
        
        .welcome-icon {
            font-size: 4rem;
            margin-bottom: 1rem;
        }
        
        .welcome-title {
            font-size: 1.25rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
        }
        
        .welcome-desc {
            color: var(--text-secondary);
            font-size: 0.9rem;
            line-height: 1.6;
            max-width: 300px;
        }
        
        .start-btn {
            margin-top: 2rem;
            background: var(--primary);
            border: none;
            border-radius: 2rem;
            padding: 1rem 2.5rem;
            color: white;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .start-btn:hover {
            transform: scale(1.05);
        }
        
        .call-screen {
            display: none;
            flex: 1;
            flex-direction: column;
        }
        
        .call-screen.active {
            display: flex;
        }
        
        .welcome-screen.hidden {
            display: none;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📞 品驰关爱中心</h1>
        <div class="subtitle">语音对话测试</div>
    </div>
    
    <div class="status-bar" id="status-bar">
        <div class="status-item"><span class="status-dot"></span> 加载中...</div>
    </div>
    
    <div class="welcome-screen" id="welcome-screen">
        <div class="welcome-icon">🎙️</div>
        <div class="welcome-title">准备开始对话</div>
        <div class="welcome-desc">
            点击下方按钮开始模拟电话对话。<br>
            按住麦克风按钮说话，松开后 AI 会自动回复。
        </div>
        <button class="start-btn" onclick="startCall()">开始对话</button>
    </div>
    
    <div class="call-screen" id="call-screen">
        <div class="conversation" id="conversation"></div>
        
        <div class="control-panel">
            <div class="voice-status" id="voice-status">按住按钮开始说话</div>
            
            <button class="mic-button" id="mic-button"
                    onmousedown="startRecording()" onmouseup="stopRecording()"
                    ontouchstart="startRecording()" ontouchend="stopRecording()">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/>
                </svg>
            </button>
            
            <div class="hint">按住说话，松开发送</div>
            
            <button class="text-input-toggle" onclick="toggleTextInput()">切换到文字输入</button>
            
            <div class="text-input-container" id="text-input-container">
                <input type="text" id="text-input" placeholder="输入消息..." 
                       onkeydown="if(event.key==='Enter')sendTextMessage()">
                <button onclick="sendTextMessage()">发送</button>
            </div>
        </div>
    </div>
    
    <script>
        let sessionId = null;
        let mediaRecorder = null;
        let audioChunks = [];
        let isRecording = false;
        let isProcessing = false;
        let audioContext = null;
        let currentAudio = null;
        
        // 初始化
        document.addEventListener('DOMContentLoaded', async () => {
            await checkServices();
        });
        
        // 检查服务状态
        async function checkServices() {
            try {
                const response = await fetch('/local-chat/status');
                const data = await response.json();
                
                document.getElementById('status-bar').innerHTML = `
                    <div class="status-item">
                        <span class="status-dot ${data.llm}"></span> LLM
                    </div>
                    <div class="status-item">
                        <span class="status-dot ${data.tts}"></span> TTS
                    </div>
                    <div class="status-item">
                        <span class="status-dot ${data.asr}"></span> ASR
                    </div>
                `;
            } catch (e) {
                console.error('Services check failed:', e);
            }
        }
        
        // 开始通话
        async function startCall() {
            try {
                // 请求麦克风权限
                await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // 创建会话
                const response = await fetch('/local-chat/session', { method: 'POST' });
                const data = await response.json();
                sessionId = data.session_id;
                console.log('Session:', sessionId);
                
                // 切换界面
                document.getElementById('welcome-screen').classList.add('hidden');
                document.getElementById('call-screen').classList.add('active');
                
                // AI 主动问候
                await sendToLLM('');
                
            } catch (e) {
                alert('无法访问麦克风，请允许麦克风权限。\\n\\n错误: ' + e.message);
            }
        }
        
        // 开始录音
        async function startRecording() {
            if (isProcessing || isRecording) return;
            
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ 
                    audio: {
                        sampleRate: 16000,
                        channelCount: 1,
                        echoCancellation: true,
                        noiseSuppression: true,
                    }
                });
                
                mediaRecorder = new MediaRecorder(stream, {
                    mimeType: 'audio/webm;codecs=opus'
                });
                
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                };
                
                mediaRecorder.onstop = async () => {
                    stream.getTracks().forEach(t => t.stop());
                    
                    if (audioChunks.length > 0) {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                        await processRecording(audioBlob);
                    }
                };
                
                mediaRecorder.start(100);
                isRecording = true;
                
                document.getElementById('mic-button').classList.add('recording');
                setStatus('recording', '正在录音...');
                
            } catch (e) {
                console.error('Recording error:', e);
                alert('录音失败: ' + e.message);
            }
        }
        
        // 停止录音
        function stopRecording() {
            if (!isRecording || !mediaRecorder) return;
            
            isRecording = false;
            mediaRecorder.stop();
            document.getElementById('mic-button').classList.remove('recording');
        }
        
        // 处理录音
        async function processRecording(audioBlob) {
            if (isProcessing) return;
            isProcessing = true;
            
            const micBtn = document.getElementById('mic-button');
            micBtn.disabled = true;
            setStatus('processing', '正在识别...');
            
            try {
                // 转换为 WAV
                const wavBlob = await convertToWav(audioBlob);
                
                // 转 base64
                const base64 = await blobToBase64(wavBlob);
                
                // 发送到服务器
                const response = await fetch('/local-chat/voice-chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        audio_base64: base64,
                        synthesize_audio: true
                    })
                });
                
                const data = await response.json();
                
                // 显示用户消息（ASR 识别结果）
                if (data.recognized_text && data.recognized_text !== '(无法识别语音)') {
                    addMessage('user', data.recognized_text);
                }
                
                // 显示 AI 回复
                addMessage('assistant', data.response);
                
                // 播放语音
                if (data.audio_base64) {
                    setStatus('playing', '正在播放...');
                    await playAudio(data.audio_base64);
                }
                
                setStatus('', '按住按钮开始说话');
                
            } catch (e) {
                console.error('Processing error:', e);
                setStatus('', '处理失败，请重试');
            }
            
            isProcessing = false;
            micBtn.disabled = false;
        }
        
        // 发送文字（AI 首次问候或文字输入）
        async function sendToLLM(message) {
            if (isProcessing) return;
            isProcessing = true;
            
            const micBtn = document.getElementById('mic-button');
            micBtn.disabled = true;
            setStatus('processing', '正在思考...');
            
            try {
                let response;
                
                if (message) {
                    // 有用户输入
                    addMessage('user', message);
                    
                    response = await fetch('/local-chat/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: sessionId,
                            message: message,
                            synthesize_audio: true
                        })
                    });
                } else {
                    // AI 主动问候
                    response = await fetch('/local-chat/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            session_id: sessionId,
                            message: '你好，请开始问候我',
                            synthesize_audio: true
                        })
                    });
                }
                
                const data = await response.json();
                
                // 显示 AI 回复
                addMessage('assistant', data.response);
                
                // 播放语音
                if (data.audio_base64) {
                    setStatus('playing', '正在播放...');
                    await playAudio(data.audio_base64);
                }
                
                setStatus('', '按住按钮开始说话');
                
            } catch (e) {
                console.error('LLM error:', e);
                setStatus('', '处理失败，请重试');
            }
            
            isProcessing = false;
            micBtn.disabled = false;
        }
        
        // 添加消息
        function addMessage(role, content) {
            const container = document.getElementById('conversation');
            const label = role === 'user' ? '我' : '小驰';
            
            const div = document.createElement('div');
            div.className = `message ${role}`;
            div.innerHTML = `
                <div class="message-label">${label}</div>
                <div class="message-bubble">${escapeHtml(content)}</div>
            `;
            
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
        }
        
        // 设置状态
        function setStatus(type, text) {
            const el = document.getElementById('voice-status');
            el.className = 'voice-status ' + type;
            
            if (type === 'recording') {
                el.innerHTML = `<div class="recording-wave"><span></span><span></span><span></span><span></span></div> ${text}`;
            } else {
                el.textContent = text;
            }
        }
        
        // 播放音频
        async function playAudio(base64) {
            return new Promise(async (resolve) => {
                try {
                    if (!audioContext) {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)();
                    }
                    
                    // 解码
                    const binary = atob(base64);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) {
                        bytes[i] = binary.charCodeAt(i);
                    }
                    
                    const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
                    
                    // 播放
                    const source = audioContext.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioContext.destination);
                    source.onended = resolve;
                    source.start(0);
                    
                } catch (e) {
                    console.error('Playback error:', e);
                    resolve();
                }
            });
        }
        
        // WebM 转 WAV
        async function convertToWav(webmBlob) {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const arrayBuffer = await webmBlob.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            
            // 重采样到 16kHz
            const targetSampleRate = 16000;
            const offlineContext = new OfflineAudioContext(
                1,
                audioBuffer.duration * targetSampleRate,
                targetSampleRate
            );
            
            const source = offlineContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(offlineContext.destination);
            source.start(0);
            
            const renderedBuffer = await offlineContext.startRendering();
            
            // 转 WAV
            return audioBufferToWav(renderedBuffer);
        }
        
        // AudioBuffer 转 WAV Blob
        function audioBufferToWav(buffer) {
            const numChannels = 1;
            const sampleRate = buffer.sampleRate;
            const format = 1; // PCM
            const bitDepth = 16;
            
            const samples = buffer.getChannelData(0);
            const bytesPerSample = bitDepth / 8;
            const blockAlign = numChannels * bytesPerSample;
            
            const wavBuffer = new ArrayBuffer(44 + samples.length * bytesPerSample);
            const view = new DataView(wavBuffer);
            
            // RIFF header
            writeString(view, 0, 'RIFF');
            view.setUint32(4, 36 + samples.length * bytesPerSample, true);
            writeString(view, 8, 'WAVE');
            
            // fmt chunk
            writeString(view, 12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, format, true);
            view.setUint16(22, numChannels, true);
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * blockAlign, true);
            view.setUint16(32, blockAlign, true);
            view.setUint16(34, bitDepth, true);
            
            // data chunk
            writeString(view, 36, 'data');
            view.setUint32(40, samples.length * bytesPerSample, true);
            
            // samples
            let offset = 44;
            for (let i = 0; i < samples.length; i++) {
                const s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                offset += 2;
            }
            
            return new Blob([wavBuffer], { type: 'audio/wav' });
        }
        
        function writeString(view, offset, string) {
            for (let i = 0; i < string.length; i++) {
                view.setUint8(offset + i, string.charCodeAt(i));
            }
        }
        
        // Blob 转 Base64
        function blobToBase64(blob) {
            return new Promise((resolve) => {
                const reader = new FileReader();
                reader.onloadend = () => {
                    const base64 = reader.result.split(',')[1];
                    resolve(base64);
                };
                reader.readAsDataURL(blob);
            });
        }
        
        // HTML 转义
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // 切换文字输入
        function toggleTextInput() {
            const container = document.getElementById('text-input-container');
            container.classList.toggle('show');
            if (container.classList.contains('show')) {
                document.getElementById('text-input').focus();
            }
        }
        
        // 发送文字消息
        async function sendTextMessage() {
            const input = document.getElementById('text-input');
            const message = input.value.trim();
            if (!message) return;
            
            input.value = '';
            await sendToLLM(message);
        }
    </script>
</body>
</html>
"""

# HTML 模板 - 文字聊天界面（备用）
CHAT_HTML = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>品驰关爱中心 - 本地对话测试</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #2563eb;
            --primary-dark: #1d4ed8;
            --primary-light: #dbeafe;
            --secondary: #10b981;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-input: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border: #475569;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Noto Sans SC', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, var(--bg-dark) 0%, #1a1a2e 100%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
        }
        
        .header {
            background: var(--bg-card);
            border-bottom: 1px solid var(--border);
            padding: 1rem 1.5rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: var(--shadow);
        }
        
        .header-title {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        
        .header-title h1 {
            font-size: 1.25rem;
            font-weight: 500;
            color: var(--text-primary);
        }
        
        .header-title .badge {
            background: var(--secondary);
            color: white;
            font-size: 0.7rem;
            padding: 0.25rem 0.5rem;
            border-radius: 9999px;
            font-weight: 500;
        }
        
        .header-status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--text-secondary);
            font-size: 0.875rem;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--secondary);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .container {
            flex: 1;
            display: flex;
            flex-direction: column;
            max-width: 900px;
            width: 100%;
            margin: 0 auto;
            padding: 1rem;
        }
        
        .chat-container {
            flex: 1;
            overflow-y: auto;
            padding: 1rem 0;
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }
        
        .message {
            display: flex;
            gap: 0.75rem;
            max-width: 85%;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            align-self: flex-end;
            flex-direction: row-reverse;
        }
        
        .message.assistant {
            align-self: flex-start;
        }
        
        .avatar {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            flex-shrink: 0;
        }
        
        .user .avatar {
            background: var(--primary);
        }
        
        .assistant .avatar {
            background: linear-gradient(135deg, #6366f1, #8b5cf6);
        }
        
        .message-content {
            background: var(--bg-card);
            border-radius: 1rem;
            padding: 0.875rem 1rem;
            border: 1px solid var(--border);
        }
        
        .user .message-content {
            background: var(--primary);
            border-color: var(--primary-dark);
        }
        
        .message-text {
            line-height: 1.6;
            font-size: 0.9375rem;
        }
        
        .message-audio {
            margin-top: 0.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .play-btn {
            background: rgba(255,255,255,0.15);
            border: none;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        
        .play-btn:hover {
            background: rgba(255,255,255,0.25);
        }
        
        .play-btn svg {
            width: 16px;
            height: 16px;
            fill: white;
        }
        
        .message-time {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }
        
        .user .message-time {
            text-align: right;
            color: rgba(255,255,255,0.6);
        }
        
        .input-container {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 1rem;
            padding: 0.75rem;
            display: flex;
            gap: 0.75rem;
            align-items: flex-end;
            box-shadow: var(--shadow);
        }
        
        .input-wrapper {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        textarea {
            background: var(--bg-input);
            border: none;
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
            color: var(--text-primary);
            font-size: 0.9375rem;
            font-family: inherit;
            resize: none;
            min-height: 44px;
            max-height: 150px;
            line-height: 1.5;
        }
        
        textarea:focus {
            outline: none;
            box-shadow: 0 0 0 2px var(--primary);
        }
        
        textarea::placeholder {
            color: var(--text-muted);
        }
        
        .send-btn {
            background: var(--primary);
            border: none;
            border-radius: 0.75rem;
            padding: 0.75rem 1.5rem;
            color: white;
            font-size: 0.9375rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s, transform 0.1s;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .send-btn:hover:not(:disabled) {
            background: var(--primary-dark);
        }
        
        .send-btn:active:not(:disabled) {
            transform: scale(0.98);
        }
        
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .send-btn svg {
            width: 18px;
            height: 18px;
        }
        
        .typing-indicator {
            display: flex;
            gap: 0.25rem;
            padding: 0.5rem;
        }
        
        .typing-indicator span {
            width: 8px;
            height: 8px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-indicator span:nth-child(1) { animation-delay: -0.32s; }
        .typing-indicator span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-6px); }
        }
        
        .welcome-message {
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }
        
        .welcome-message h2 {
            font-size: 1.5rem;
            font-weight: 500;
            margin-bottom: 0.5rem;
            color: var(--text-primary);
        }
        
        .welcome-message p {
            font-size: 0.9375rem;
            line-height: 1.6;
        }
        
        .features {
            display: flex;
            gap: 1rem;
            margin-top: 1.5rem;
            flex-wrap: wrap;
            justify-content: center;
        }
        
        .feature {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            width: 140px;
            text-align: center;
        }
        
        .feature-icon {
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }
        
        .feature-title {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--text-primary);
        }
        
        .footer {
            text-align: center;
            padding: 1rem;
            color: var(--text-muted);
            font-size: 0.8rem;
        }
        
        /* 服务状态指示器 */
        .services-status {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
        }
        
        .service-badge {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            font-size: 0.75rem;
            color: var(--text-secondary);
            background: var(--bg-input);
            padding: 0.25rem 0.5rem;
            border-radius: 0.25rem;
        }
        
        .service-badge .dot {
            width: 6px;
            height: 6px;
            border-radius: 50%;
        }
        
        .service-badge .dot.ok { background: var(--secondary); }
        .service-badge .dot.mock { background: #f59e0b; }
        .service-badge .dot.error { background: #ef4444; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-title">
            <h1>🤖 品驰关爱中心</h1>
            <span class="badge">本地测试</span>
        </div>
        <div class="services-status" id="services-status">
            <!-- 由 JS 填充 -->
        </div>
    </div>
    
    <div class="container">
        <div class="chat-container" id="chat-container">
            <div class="welcome-message">
                <h2>👋 欢迎使用本地对话测试</h2>
                <p>这是一个模拟电话客服的测试环境，您可以在这里测试完整的对话流程。</p>
                <div class="features">
                    <div class="feature">
                        <div class="feature-icon">💬</div>
                        <div class="feature-title">文字对话</div>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🧠</div>
                        <div class="feature-title">AI 回复</div>
                    </div>
                    <div class="feature">
                        <div class="feature-icon">🔊</div>
                        <div class="feature-title">语音合成</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="input-container">
            <div class="input-wrapper">
                <textarea 
                    id="message-input" 
                    placeholder="输入消息... (按 Enter 发送，Shift+Enter 换行)"
                    rows="1"
                ></textarea>
            </div>
            <button class="send-btn" id="send-btn" onclick="sendMessage()">
                <span>发送</span>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                </svg>
            </button>
        </div>
    </div>
    
    <div class="footer">
        本地测试环境 · 数据不会上传 · 重启后对话会丢失
    </div>
    
    <script>
        let sessionId = null;
        let isProcessing = false;
        let audioContext = null;
        
        // 初始化
        document.addEventListener('DOMContentLoaded', async () => {
            // 检查服务状态
            await checkServices();
            
            // 创建新会话
            const response = await fetch('/local-chat/session', { method: 'POST' });
            const data = await response.json();
            sessionId = data.session_id;
            console.log('Session created:', sessionId);
            
            // 自动聚焦输入框
            document.getElementById('message-input').focus();
        });
        
        // 检查服务状态
        async function checkServices() {
            try {
                const response = await fetch('/local-chat/status');
                const data = await response.json();
                
                const container = document.getElementById('services-status');
                container.innerHTML = `
                    <div class="service-badge">
                        <span class="dot ${data.llm}"></span>
                        <span>LLM</span>
                    </div>
                    <div class="service-badge">
                        <span class="dot ${data.tts}"></span>
                        <span>TTS</span>
                    </div>
                    <div class="service-badge">
                        <span class="dot ${data.asr}"></span>
                        <span>ASR</span>
                    </div>
                `;
            } catch (e) {
                console.error('Failed to check services:', e);
            }
        }
        
        // 发送消息
        async function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (!message || isProcessing) return;
            
            isProcessing = true;
            document.getElementById('send-btn').disabled = true;
            
            // 添加用户消息
            addMessage('user', message);
            input.value = '';
            autoResize(input);
            
            // 显示 typing 指示器
            const typingId = addTypingIndicator();
            
            try {
                const response = await fetch('/local-chat/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        session_id: sessionId,
                        message: message,
                        synthesize_audio: true
                    })
                });
                
                const data = await response.json();
                
                // 移除 typing 指示器
                removeTypingIndicator(typingId);
                
                // 添加 AI 回复
                addMessage('assistant', data.response, data.audio_base64);
                
                // 自动播放语音
                if (data.audio_base64) {
                    playAudio(data.audio_base64);
                }
                
            } catch (error) {
                console.error('Chat error:', error);
                removeTypingIndicator(typingId);
                addMessage('assistant', '抱歉，发生了错误：' + error.message);
            }
            
            isProcessing = false;
            document.getElementById('send-btn').disabled = false;
            input.focus();
        }
        
        // 添加消息到聊天区域
        function addMessage(role, content, audioBase64 = null) {
            const container = document.getElementById('chat-container');
            
            // 移除欢迎消息
            const welcome = container.querySelector('.welcome-message');
            if (welcome) welcome.remove();
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${role}`;
            
            const avatar = role === 'user' ? '👤' : '🤖';
            const time = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
            
            let audioHtml = '';
            if (audioBase64) {
                audioHtml = `
                    <div class="message-audio">
                        <button class="play-btn" onclick="playAudio('${audioBase64}')">
                            <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                        </button>
                        <span style="font-size: 0.75rem; opacity: 0.7;">点击播放语音</span>
                    </div>
                `;
            }
            
            messageDiv.innerHTML = `
                <div class="avatar">${avatar}</div>
                <div class="message-content">
                    <div class="message-text">${escapeHtml(content)}</div>
                    ${audioHtml}
                    <div class="message-time">${time}</div>
                </div>
            `;
            
            container.appendChild(messageDiv);
            container.scrollTop = container.scrollHeight;
        }
        
        // 添加 typing 指示器
        function addTypingIndicator() {
            const container = document.getElementById('chat-container');
            const id = 'typing-' + Date.now();
            
            const div = document.createElement('div');
            div.id = id;
            div.className = 'message assistant';
            div.innerHTML = `
                <div class="avatar">🤖</div>
                <div class="message-content">
                    <div class="typing-indicator">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            `;
            
            container.appendChild(div);
            container.scrollTop = container.scrollHeight;
            
            return id;
        }
        
        // 移除 typing 指示器
        function removeTypingIndicator(id) {
            const el = document.getElementById(id);
            if (el) el.remove();
        }
        
        // 播放音频 (WAV base64)
        async function playAudio(base64Data) {
            try {
                if (!audioContext) {
                    audioContext = new (window.AudioContext || window.webkitAudioContext)();
                }
                
                // 解码 base64
                const binaryString = atob(base64Data);
                const bytes = new Uint8Array(binaryString.length);
                for (let i = 0; i < binaryString.length; i++) {
                    bytes[i] = binaryString.charCodeAt(i);
                }
                
                // 解码音频
                const audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
                
                // 播放
                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);
                source.start(0);
                
            } catch (e) {
                console.error('Audio playback error:', e);
            }
        }
        
        // HTML 转义
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // 自动调整 textarea 高度
        function autoResize(el) {
            el.style.height = 'auto';
            el.style.height = Math.min(el.scrollHeight, 150) + 'px';
        }
        
        // 输入框事件
        const input = document.getElementById('message-input');
        
        input.addEventListener('input', function() {
            autoResize(this);
        });
        
        input.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
    </script>
</body>
</html>
"""

