"""
日志配置

使用 structlog 进行结构化日志记录。
"""

import os
from logging import Logger, _nameToLevel, basicConfig

from structlog import (
    configure_once,
    get_logger as structlog_get_logger,
    make_filtering_bound_logger,
)
from structlog.contextvars import merge_contextvars
from structlog.dev import ConsoleRenderer
from structlog.processors import (
    StackInfoRenderer,
    TimeStamper,
    UnicodeDecoder,
    add_log_level,
)
from structlog.stdlib import PositionalArgumentsFormatter

# 从环境变量获取日志级别，默认 INFO
SYS_LOG_LEVEL = os.environ.get("LOG_LEVEL", "WARNING")
APP_LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO")

# 配置系统日志级别
basicConfig(level=SYS_LOG_LEVEL.upper())

# 配置应用日志
configure_once(
    cache_logger_on_first_use=True,
    context_class=dict,
    wrapper_class=make_filtering_bound_logger(
        _nameToLevel.get(APP_LOG_LEVEL.upper(), 20)  # 20 = INFO
    ),
    processors=[
        merge_contextvars,
        add_log_level,
        PositionalArgumentsFormatter(),
        TimeStamper(fmt="iso", utc=True),
        StackInfoRenderer(),
        UnicodeDecoder(),
        ConsoleRenderer(),
    ],
)

# 导出 logger 实例
logger: Logger = structlog_get_logger("call-center-ai")
