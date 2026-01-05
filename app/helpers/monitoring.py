"""
监控模块（简化版）

提供兼容的空实现，不依赖 Azure Application Insights。
"""

import asyncio
import functools
from contextlib import contextmanager
from enum import Enum
from typing import Any, Callable


class SpanAttributeEnum(str, Enum):
    """Span 属性枚举"""
    CALL_ID = "call.id"
    CALL_PHONE_NUMBER = "call.phone_number"
    
    def attribute(self, value: Any) -> None:
        """设置属性（空实现）"""
        pass


# 空的 tracer
class DummyTracer:
    def start_as_current_span(self, name: str, **kwargs):
        @contextmanager
        def dummy_context():
            yield None
        return dummy_context()


tracer = DummyTracer()


def start_as_current_span(name: str):
    """装饰器：创建 span（空实现）"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
                return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


@contextmanager
def suppress(*exceptions):
    """上下文管理器：忽略指定异常"""
    try:
        yield
    except exceptions:
        pass


def gauge_set(metric: Any, value: float) -> None:
    """设置 gauge 指标（空实现）"""
    pass


# 空的指标
call_frames_in_latency = None
call_frames_out_latency = None

