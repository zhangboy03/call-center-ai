"""
阿里云 Paraformer 实时语音识别客户端

基于 DashScope SDK 实现，使用官方 Recognition 类。

参考文档：
https://help.aliyun.com/zh/model-studio/paraformer-real-time-speech-recognition-python-sdk

环境变量：
- DASHSCOPE_API_KEY: 百炼平台 API Key（必需）
"""

import logging
from os import environ

import dashscope
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

_logger = logging.getLogger(__name__)


class ParaformerError(Exception):
    """Paraformer ASR 错误"""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")


class _ResultCallback(RecognitionCallback):
    """识别结果回调"""

    def __init__(self):
        self.results: list[str] = []
        self.final_text: str = ""
        self.error: Exception | None = None
        self._completed = False

    def on_open(self) -> None:
        _logger.debug("ASR connection opened")

    def on_close(self) -> None:
        _logger.debug("ASR connection closed")
        self._completed = True

    def on_event(self, result: RecognitionResult) -> None:
        """处理识别结果"""
        sentence = result.get_sentence()
        if sentence and "text" in sentence:
            text = sentence["text"]
            is_end = RecognitionResult.is_sentence_end(sentence)

            _logger.debug("ASR result: %s (final=%s)", text, is_end)

            if is_end:
                # 句子结束，保存完整结果
                self.final_text = text
                self.results.append(text)
            else:
                # 中间结果
                self.final_text = text

    def on_error(self, result: RecognitionResult) -> None:
        """处理错误"""
        error_msg = str(result)
        _logger.error("ASR error: %s", error_msg)
        self.error = ParaformerError("ASR_ERROR", error_msg)
        self._completed = True

    def on_complete(self) -> None:
        """识别完成"""
        _logger.debug(
            "ASR completed, final text: %s",
            self.final_text[:50] if self.final_text else "(empty)",
        )
        self._completed = True

    @property
    def is_completed(self) -> bool:
        return self._completed


class PersistentASR:
    """
    持久化 ASR 连接 - 复用 WebSocket 避免每次握手开销

    使用方法:
    ```python
    asr = PersistentASR(hot_words=[...])
    asr.start()  # 建立连接

    # 每次识别复用连接
    text = asr.recognize(audio_data)

    asr.stop()  # 结束时关闭
    ```
    """

    def __init__(
        self,
        model: str = "paraformer-realtime-v2",
        sample_rate: int = 16000,
        language: str = "zh",
        hot_words: list | None = None,
    ):
        self._model = model
        self._sample_rate = sample_rate
        self._language = language
        self._hot_words = hot_words
        self._recognition = None
        self._callback = None
        self._is_started = False

        api_key = environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")
        dashscope.api_key = api_key

    def start(self) -> None:
        """启动持久连接"""
        import time

        t0 = time.time()

        if self._is_started:
            return

        self._callback = _ResultCallback()
        self._recognition = Recognition(
            model=self._model,
            format="pcm",
            sample_rate=self._sample_rate,
            callback=self._callback,
        )

        start_kwargs = {}
        if self._model in ["paraformer-realtime-v2", "paraformer-realtime-8k-v2"]:
            start_kwargs["language_hints"] = [self._language]
        if self._hot_words:
            start_kwargs["hotwords"] = self._hot_words

        self._recognition.start(**start_kwargs)
        self._is_started = True
        _logger.info("[PersistentASR] Started in %.0fms", (time.time() - t0) * 1000)

    def recognize(self, audio_data: bytes) -> str | None:
        """识别音频 - 使用 stop() 获取完整结果，然后自动重启连接"""
        import time

        t0 = time.time()

        if not self._is_started:
            self.start()

        # Reset callback for new recognition
        self._callback.final_text = ""
        self._callback.results = []
        self._callback._completed = False

        # Send audio in small chunks
        chunk_size = int(self._sample_rate * 0.1) * 2  # 100ms chunks
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            self._recognition.send_audio_frame(chunk)

        # Call stop() to get final result (blocks until complete)
        try:
            self._recognition.stop()
        except Exception as e:
            _logger.warning("[PersistentASR] Stop error: %s", e)

        result = self._callback.final_text
        total_ms = (time.time() - t0) * 1000
        _logger.info(
            "[PersistentASR] Recognized in %.0fms: %s",
            total_ms,
            result[:30] if result else "(empty)",
        )

        # Reset state - will restart on next recognize() call (lazy restart)
        self._is_started = False
        self._recognition = None

        return result or None

    def _restart_connection(self) -> None:
        """后台重启连接，准备下次识别"""
        import threading

        def restart():
            try:
                self._callback = _ResultCallback()
                self._recognition = Recognition(
                    model=self._model,
                    format="pcm",
                    sample_rate=self._sample_rate,
                    callback=self._callback,
                )
                start_kwargs = {}
                if self._model in [
                    "paraformer-realtime-v2",
                    "paraformer-realtime-8k-v2",
                ]:
                    start_kwargs["language_hints"] = [self._language]
                if self._hot_words:
                    start_kwargs["hotwords"] = self._hot_words
                self._recognition.start(**start_kwargs)
                self._is_started = True
                _logger.debug("[PersistentASR] Connection restarted")
            except Exception as e:
                _logger.warning("[PersistentASR] Restart failed: %s", e)
                self._is_started = False

        # Restart in background thread (non-blocking)
        threading.Thread(target=restart, daemon=True).start()

    def stop(self) -> None:
        """停止连接"""
        if self._recognition and self._is_started:
            try:
                self._recognition.stop()
            except Exception as e:
                _logger.warning("[PersistentASR] Stop error: %s", e)
        self._is_started = False
        self._recognition = None
        _logger.info("[PersistentASR] Stopped")

    @property
    def is_running(self) -> bool:
        return self._is_started


def recognize_audio_sync(
    audio_data: bytes,
    sample_rate: int = 16000,
    format: str = "pcm",
    model: str = "paraformer-realtime-v2",
    language: str = "zh",
    hot_words: list | None = None,
    timeout: float = 3.0,  # Max wait time for recognition
) -> str | None:
    """
    同步识别音频（阻塞调用）- 优化版

    Args:
        audio_data: PCM 音频数据 (16bit mono)
        sample_rate: 采样率
        format: 音频格式
        model: 模型名称
        language: 语言
        hot_words: 热词列表，格式 [{"text": "词语", "weight": 5}, ...]
        timeout: 最大等待时间（秒）

    Returns:
        识别出的文字，失败返回 None
    """
    import time

    t_start = time.time()

    api_key = environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")

    # 设置 API Key
    dashscope.api_key = api_key

    callback = _ResultCallback()

    # 创建识别器
    recognition = Recognition(
        model=model,
        format=format,
        sample_rate=sample_rate,
        callback=callback,
    )

    # 配置语言和热词
    start_kwargs = {}
    if model in ["paraformer-realtime-v2", "paraformer-realtime-8k-v2"]:
        start_kwargs["language_hints"] = [language]

    if hot_words:
        start_kwargs["hotwords"] = hot_words
        _logger.debug("Using %d hot words", len(hot_words))

    try:
        recognition.start(**start_kwargs)
        t_connected = time.time()
        _logger.debug("[ASR] Connected in %.2fms", (t_connected - t_start) * 1000)

        # 分块发送音频 - 小块减少延迟
        chunk_size = (
            int(sample_rate * 0.1) * 2
        )  # 100ms chunks (16-bit = 2 bytes per sample)
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            recognition.send_audio_frame(chunk)

        t_sent = time.time()
        _logger.debug(
            "[ASR] Audio sent in %.2fms (%d bytes)",
            (t_sent - t_connected) * 1000,
            len(audio_data),
        )

        # 结束识别
        recognition.stop()

        t_done = time.time()
        total_ms = (t_done - t_start) * 1000
        _logger.info(
            "[ASR] Total: %.0fms, result: %s",
            total_ms,
            callback.final_text[:30] if callback.final_text else "(empty)",
        )

        # 检查错误
        if callback.error:
            raise callback.error

        return callback.final_text or None

    except Exception as e:
        _logger.error(
            "[ASR] Error: %s (after %.0fms)", e, (time.time() - t_start) * 1000
        )
        return None


class ParaformerClient:
    """
    Paraformer 语音识别客户端

    封装 DashScope SDK，提供简洁的接口。
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "paraformer-realtime-v2",
    ):
        self._api_key = api_key or environ.get("DASHSCOPE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "DASHSCOPE_API_KEY 环境变量未设置！\n"
                "请在 .env 文件中添加：DASHSCOPE_API_KEY=sk-your-api-key"
            )

        self._model = model
        dashscope.api_key = self._api_key

    def recognize(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        format: str = "pcm",
        language: str = "zh",
    ) -> str | None:
        """
        识别音频

        Args:
            audio_data: PCM 音频数据
            sample_rate: 采样率
            format: 音频格式
            language: 语言

        Returns:
            识别出的文字
        """
        return recognize_audio_sync(
            audio_data=audio_data,
            sample_rate=sample_rate,
            format=format,
            model=self._model,
            language=language,
        )
