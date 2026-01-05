"""
全流式音频管道

实现微软 Call-Center-AI 风格的低延迟对话：
1. 流式 ASR - 边说边识别，支持 partial results
2. 流式 LLM - 边识别边生成，分句输出
3. 流式 TTS - 边合成边播放

架构：
    用户音频 → [audio_in队列] → ASR → [text队列] → LLM → [sentence队列] → TTS → [audio_out队列] → 播放
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Callable

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    """管道状态"""
    IDLE = "idle"           # 空闲
    LISTENING = "listening" # 正在听用户说话
    THINKING = "thinking"   # 正在生成回复
    SPEAKING = "speaking"   # 正在播放回复
    

@dataclass
class AudioFrame:
    """音频帧"""
    data: bytes
    timestamp: float = field(default_factory=time.time)
    is_silence: bool = False


@dataclass
class TextChunk:
    """文本块"""
    text: str
    is_final: bool = False  # 是否是最终结果（vs partial）
    timestamp: float = field(default_factory=time.time)


@dataclass
class Sentence:
    """完整句子，用于 TTS"""
    text: str
    timestamp: float = field(default_factory=time.time)


class StreamingPipeline:
    """
    全流式音频管道
    
    核心思想：
    - 所有组件并行运行
    - 通过队列解耦
    - 支持中断（用户打断 AI 说话）
    """
    
    def __init__(
        self,
        on_partial_text: Callable[[str], None] | None = None,
        on_final_text: Callable[[str], None] | None = None,
        on_ai_text: Callable[[str], None] | None = None,
        on_audio_out: Callable[[bytes], None] | None = None,
        on_state_change: Callable[[PipelineState], None] | None = None,
    ):
        """
        初始化管道
        
        Args:
            on_partial_text: 收到 ASR partial 结果时的回调
            on_final_text: 收到 ASR final 结果时的回调
            on_ai_text: 收到 AI 回复句子时的回调
            on_audio_out: 有音频输出时的回调
            on_state_change: 状态变化时的回调
        """
        # 队列
        self.audio_in: asyncio.Queue[AudioFrame | None] = asyncio.Queue()
        self.text_queue: asyncio.Queue[TextChunk | None] = asyncio.Queue()
        self.sentence_queue: asyncio.Queue[Sentence | None] = asyncio.Queue()
        self.audio_out: asyncio.Queue[bytes | None] = asyncio.Queue()
        
        # 回调
        self.on_partial_text = on_partial_text
        self.on_final_text = on_final_text
        self.on_ai_text = on_ai_text
        self.on_audio_out = on_audio_out
        self.on_state_change = on_state_change
        
        # 状态
        self._state = PipelineState.IDLE
        self._running = False
        self._tasks: list[asyncio.Task] = []
        
        # 中断控制
        self._interrupt_event = asyncio.Event()
        
        # 对话历史
        self.messages: list[dict] = []
        self.system_prompt = """你是品驰关爱中心的智能客服小驰，正在进行术后随访电话。
要求：
- 每次回复控制在20-30字
- 像真人打电话一样自然简洁
- 一次只问一个问题或回应一个话题"""
    
    @property
    def state(self) -> PipelineState:
        return self._state
    
    @state.setter
    def state(self, value: PipelineState):
        if self._state != value:
            self._state = value
            if self.on_state_change:
                self.on_state_change(value)
            logger.debug("Pipeline state: %s", value.value)
    
    async def start(self):
        """启动管道"""
        if self._running:
            return
        
        self._running = True
        self._interrupt_event.clear()
        
        # 启动所有处理任务
        self._tasks = [
            asyncio.create_task(self._asr_worker(), name="asr_worker"),
            asyncio.create_task(self._llm_worker(), name="llm_worker"),
            asyncio.create_task(self._tts_worker(), name="tts_worker"),
        ]
        
        logger.info("Pipeline started")
    
    async def stop(self):
        """停止管道"""
        self._running = False
        
        # 发送停止信号
        await self.audio_in.put(None)
        await self.text_queue.put(None)
        await self.sentence_queue.put(None)
        await self.audio_out.put(None)
        
        # 等待任务结束
        for task in self._tasks:
            task.cancel()
        
        self._tasks.clear()
        logger.info("Pipeline stopped")
    
    def interrupt(self):
        """中断当前播放（用户打断 AI）"""
        self._interrupt_event.set()
        # 清空输出队列
        while not self.audio_out.empty():
            try:
                self.audio_out.get_nowait()
            except asyncio.QueueEmpty:
                break
        while not self.sentence_queue.empty():
            try:
                self.sentence_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        logger.debug("Pipeline interrupted")
    
    async def feed_audio(self, data: bytes, is_silence: bool = False):
        """
        输入音频帧
        
        Args:
            data: PCM 音频数据
            is_silence: 是否是静音帧
        """
        if not self._running:
            return
        
        frame = AudioFrame(data=data, is_silence=is_silence)
        await self.audio_in.put(frame)
        
        # 如果是有效音频，切换到听状态
        if not is_silence and self.state == PipelineState.IDLE:
            self.state = PipelineState.LISTENING
    
    async def get_audio_output(self) -> bytes | None:
        """获取音频输出"""
        return await self.audio_out.get()
    
    # =========================================================================
    # ASR Worker - 流式语音识别
    # =========================================================================
    
    async def _asr_worker(self):
        """ASR 工作线程 - 流式识别"""
        from app.helpers.streaming_asr import StreamingASR
        
        asr = StreamingASR(
            on_partial=self._on_asr_partial,
            on_final=self._on_asr_final,
        )
        
        try:
            await asr.start()
            
            while self._running:
                frame = await self.audio_in.get()
                if frame is None:
                    break
                
                # 如果正在说话且用户开始讲话，中断
                if not frame.is_silence and self.state == PipelineState.SPEAKING:
                    self.interrupt()
                    self.state = PipelineState.LISTENING
                
                # 发送音频到 ASR
                await asr.feed_audio(frame.data)
            
            await asr.stop()
            
        except Exception as e:
            logger.error("ASR worker error: %s", e)
    
    def _on_asr_partial(self, text: str):
        """ASR partial 结果回调"""
        if self.on_partial_text:
            self.on_partial_text(text)
    
    def _on_asr_final(self, text: str):
        """ASR final 结果回调"""
        if text.strip():
            logger.info("[ASR] Final: %s", text)
            if self.on_final_text:
                self.on_final_text(text)
            
            # 放入文本队列
            chunk = TextChunk(text=text, is_final=True)
            asyncio.create_task(self.text_queue.put(chunk))
            
            # 切换到思考状态
            self.state = PipelineState.THINKING
    
    # =========================================================================
    # LLM Worker - 流式生成 + 分句
    # =========================================================================
    
    async def _llm_worker(self):
        """LLM 工作线程 - 流式生成"""
        from app.helpers.config import CONFIG
        
        try:
            while self._running:
                chunk = await self.text_queue.get()
                if chunk is None:
                    break
                
                if not chunk.is_final:
                    continue
                
                # 添加用户消息
                self.messages.append({"role": "user", "content": chunk.text})
                
                # 流式调用 LLM
                t0 = time.time()
                full_response = ""
                sentence_buffer = ""
                
                async for token in self._stream_llm(chunk.text):
                    if self._interrupt_event.is_set():
                        self._interrupt_event.clear()
                        break
                    
                    full_response += token
                    sentence_buffer += token
                    
                    # 检查是否有完整句子
                    sentences = self._split_sentences(sentence_buffer)
                    if len(sentences) > 1:
                        # 发送除最后一个（可能不完整）之外的所有句子
                        for sent in sentences[:-1]:
                            if sent.strip():
                                logger.info("[LLM] Sentence: %s (%.2fs)", sent, time.time() - t0)
                                if self.on_ai_text:
                                    self.on_ai_text(sent)
                                await self.sentence_queue.put(Sentence(text=sent))
                        sentence_buffer = sentences[-1]
                
                # 发送剩余的句子
                if sentence_buffer.strip() and not self._interrupt_event.is_set():
                    logger.info("[LLM] Final sentence: %s", sentence_buffer)
                    if self.on_ai_text:
                        self.on_ai_text(sentence_buffer)
                    await self.sentence_queue.put(Sentence(text=sentence_buffer))
                
                # 添加助手消息
                if full_response:
                    self.messages.append({"role": "assistant", "content": full_response})
                
                logger.info("[LLM] Total: %.2fs, %d chars", time.time() - t0, len(full_response))
                
        except Exception as e:
            logger.error("LLM worker error: %s", e)
    
    async def _stream_llm(self, user_text: str) -> AsyncIterator[str]:
        """流式调用 LLM"""
        from app.helpers.config import CONFIG
        
        llm_model = CONFIG.llm.selected
        client, _ = await llm_model.client()
        
        messages = [{"role": "system", "content": self.system_prompt}]
        # 只保留最近 10 条消息
        messages.extend(self.messages[-10:])
        
        response = await client.chat.completions.create(
            model=llm_model.model,
            messages=messages,
            max_tokens=100,
            temperature=0.7,
            stream=True,  # 流式输出
        )
        
        async for chunk in response:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _split_sentences(self, text: str) -> list[str]:
        """
        分句 - 按标点符号切分
        
        支持中文和英文标点
        """
        # 使用正则按句子结束符切分，保留分隔符
        pattern = r'([。！？.!?，,、；;：:]+)'
        parts = re.split(pattern, text)
        
        # 合并句子和标点
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            if i + 1 < len(parts):
                sentences.append(parts[i] + parts[i + 1])
            else:
                sentences.append(parts[i])
        
        # 处理最后一个部分（可能没有标点）
        if len(parts) % 2 == 1 and parts[-1]:
            sentences.append(parts[-1])
        
        return sentences if sentences else [text]
    
    # =========================================================================
    # TTS Worker - 流式合成
    # =========================================================================
    
    async def _tts_worker(self):
        """TTS 工作线程 - 流式合成"""
        from app.helpers.streaming_tts import StreamingTTS
        
        tts = StreamingTTS()
        
        try:
            while self._running:
                sentence = await self.sentence_queue.get()
                if sentence is None:
                    break
                
                if self._interrupt_event.is_set():
                    continue
                
                self.state = PipelineState.SPEAKING
                
                t0 = time.time()
                
                # 流式合成并输出
                async for audio_chunk in tts.synthesize_stream(sentence.text):
                    if self._interrupt_event.is_set():
                        break
                    
                    await self.audio_out.put(audio_chunk)
                    if self.on_audio_out:
                        self.on_audio_out(audio_chunk)
                
                logger.info("[TTS] %.2fs for '%s'", time.time() - t0, sentence.text[:20])
                
                # 如果队列空了，回到空闲状态
                if self.sentence_queue.empty():
                    self.state = PipelineState.IDLE
                
        except Exception as e:
            logger.error("TTS worker error: %s", e)

