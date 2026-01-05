"""
功能标志模块（简化版）

提供默认配置，不依赖 Azure App Configuration。
"""


def callback_timeout_hour() -> int:
    """回调超时时间（小时）"""
    return 24


def slow_llm_for_chat() -> bool:
    """是否使用慢速 LLM 进行聊天"""
    return False


def answer_hard_timeout_sec() -> int:
    """回答硬超时时间（秒）"""
    return 180


def answer_soft_timeout_sec() -> int:
    """回答软超时时间（秒）"""
    return 30


def phone_silence_timeout_sec() -> int:
    """电话静音超时时间（秒）"""
    return 5


def vad_silence_timeout_ms() -> int:
    """VAD 静音超时时间（毫秒）"""
    return 500


def vad_threshold() -> float:
    """VAD 阈值"""
    return 0.5


def recording_enabled() -> bool:
    """是否启用录音"""
    return False


def voice_recognition_retry_max() -> int:
    """语音识别最大重试次数"""
    return 3

