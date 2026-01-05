"""
LLM 调用封装

支持两种 provider:
- Azure OpenAI / Azure AI Inference
- 通义千问 / DashScope（通过 OpenAI 兼容接口）
"""

import json
from collections.abc import AsyncGenerator, Callable
from os import environ
from typing import Any, TypeVar

import tiktoken
from json_repair import repair_json
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry,
    retry_any,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.helpers.cache import lru_cache
from app.helpers.config import CONFIG
from app.helpers.config_models.llm import DeploymentModel as LlmDeploymentModel
from app.helpers.features import slow_llm_for_chat
from app.helpers.logging import logger
from app.helpers.monitoring import start_as_current_span
from app.helpers.resources import resources_dir
from app.models.message import MessageModel

# tiktoken cache
environ["TIKTOKEN_CACHE_DIR"] = resources_dir("tiktoken")

logger.info(
    "Using LLM models %s (slow) and %s (fast)",
    CONFIG.llm.selected(False).model,
    CONFIG.llm.selected(True).model,
)

T = TypeVar("T")


class SafetyCheckError(Exception):
    pass


class MaximumTokensReachedError(Exception):
    pass


# 需要重试的异常类型
_retried_exceptions: list[type[Exception]] = []

# 尝试添加各种可能的异常
try:
    from openai import APIError, RateLimitError, APIConnectionError
    _retried_exceptions.extend([APIError, RateLimitError, APIConnectionError])
except ImportError:
    pass

try:
    from azure.core.exceptions import ServiceResponseError
    _retried_exceptions.append(ServiceResponseError)
except ImportError:
    pass


# =============================================================================
# 消息类型转换（兼容 Azure SDK 和 OpenAI SDK）
# =============================================================================

def _to_openai_messages(messages: list[Any]) -> list[ChatCompletionMessageParam]:
    """将消息列表转换为 OpenAI 格式"""
    result: list[ChatCompletionMessageParam] = []
    
    for msg in messages:
        if isinstance(msg, dict):
            result.append(msg)  # type: ignore
        elif hasattr(msg, "model_dump"):
            # Pydantic model
            result.append(msg.model_dump())  # type: ignore
        elif hasattr(msg, "content"):
            # Azure SDK message types or similar
            role = "user"
            if hasattr(msg, "__class__"):
                class_name = msg.__class__.__name__.lower()
                if "system" in class_name:
                    role = "system"
                elif "assistant" in class_name:
                    role = "assistant"
                elif "user" in class_name:
                    role = "user"
            result.append({
                "role": role,  # type: ignore
                "content": msg.content,
            })
        else:
            result.append({"role": "user", "content": str(msg)})
    
    return result


def _to_openai_tools(tools: list[Any] | None) -> list[dict] | None:
    """将工具定义转换为 OpenAI 格式"""
    if not tools:
        return None
    
    result = []
    for tool in tools:
        if isinstance(tool, dict):
            result.append(tool)
        elif hasattr(tool, "model_dump"):
            result.append(tool.model_dump())
        elif hasattr(tool, "function"):
            # Azure ChatCompletionsToolDefinition
            func = tool.function
            tool_dict = {
                "type": "function",
                "function": {
                    "name": func.name if hasattr(func, "name") else "",
                    "description": func.description if hasattr(func, "description") else "",
                    "parameters": func.parameters if hasattr(func, "parameters") else {},
                }
            }
            result.append(tool_dict)
    
    return result if result else None


# =============================================================================
# Delta 响应类型（统一 Azure 和 OpenAI 的流式响应格式）
# =============================================================================

class UnifiedDelta:
    """统一的 Delta 响应格式"""
    
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None):
        self.content = content
        self.tool_calls = tool_calls


def _convert_openai_delta(delta: Any) -> UnifiedDelta:
    """将 OpenAI delta 转换为统一格式"""
    content = getattr(delta, "content", None)
    tool_calls = getattr(delta, "tool_calls", None)
    return UnifiedDelta(content=content, tool_calls=tool_calls)


# =============================================================================
# 核心 LLM 调用函数
# =============================================================================

@start_as_current_span("llm_completion_stream")
async def completion_stream(
    max_tokens: int,
    messages: list[MessageModel],
    system: list[Any],
    tools: list[Any] = [],
) -> AsyncGenerator[UnifiedDelta, None]:
    """
    返回流式补全结果。
    
    首先尝试使用 fast LLM，失败后切换到 slow LLM。
    最多重试 3 次。
    """
    retryed = AsyncRetrying(
        reraise=True,
        retry=retry_any(
            *[retry_if_exception_type(exception) for exception in _retried_exceptions]
        ) if _retried_exceptions else retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(multiplier=0.8, max=8),
    )

    # 首先尝试主 LLM
    try:
        async for attempt in retryed:
            with attempt:
                async for chunk in _completion_stream_worker(
                    is_fast=not await slow_llm_for_chat(),
                    max_tokens=max_tokens,
                    messages=messages,
                    system=system,
                    tools=tools,
                ):
                    yield chunk
                return
    except Exception as e:
        if _retried_exceptions and not any(isinstance(e, exc) for exc in _retried_exceptions):
            raise e
        logger.warning(
            "%s error, trying with the other LLM backend",
            e.__class__.__name__,
        )

    # 使用备用 LLM 重试
    async for attempt in retryed:
        with attempt:
            async for chunk in _completion_stream_worker(
                is_fast=await slow_llm_for_chat(),
                max_tokens=max_tokens,
                messages=messages,
                system=system,
                tools=tools,
            ):
                yield chunk


async def _completion_stream_worker(
    is_fast: bool,
    max_tokens: int,
    messages: list[MessageModel],
    system: list[Any],
    tools: list[Any] = [],
) -> AsyncGenerator[UnifiedDelta, None]:
    """
    流式补全工作函数。
    
    自动检测客户端类型（OpenAI 或 Azure）并使用对应的调用方式。
    """
    # 获取客户端
    client, platform = await _use_llm(is_fast)

    # 构建消息上下文
    prompt = _limit_messages(
        context_window=platform.context,
        max_messages=20,
        max_tokens=max_tokens,
        messages=messages,
        model=platform.model,
        system=system,
        tools=tools,
    )

    # 转换为 OpenAI 格式
    openai_messages = _to_openai_messages(prompt)
    openai_tools = _to_openai_tools(tools)
    
    # 判断客户端类型并调用
    if isinstance(client, AsyncOpenAI):
        # OpenAI 兼容客户端（包括通义千问）
        async for delta in _stream_with_openai(
            client=client,
            messages=openai_messages,
            model=platform.model,
            max_tokens=max_tokens,
            tools=openai_tools,
        ):
            yield delta
    else:
        # Azure AI Inference 客户端
        async for delta in _stream_with_azure(
            client=client,
            messages=prompt,
            max_tokens=max_tokens,
            tools=tools,
        ):
            yield delta


async def _stream_with_openai(
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    model: str,
    max_tokens: int,
    tools: list[dict] | None,
) -> AsyncGenerator[UnifiedDelta, None]:
    """使用 OpenAI SDK 进行流式调用"""
    
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": True,
    }
    
    if tools:
        kwargs["tools"] = tools
    
    stream = await client.chat.completions.create(**kwargs)
    
    async for chunk in stream:
        if not chunk.choices:
            continue
        
        choice = chunk.choices[0]
        delta = choice.delta
        
        # 检查结束原因
        if choice.finish_reason == "content_filter":
            raise SafetyCheckError(f"Issue detected in text: {delta.content if delta else ''}")
        if choice.finish_reason == "length":
            logger.warning("Maximum tokens reached %s", max_tokens)
            raise MaximumTokensReachedError(f"Maximum tokens reached {max_tokens}")
        
        if delta:
            yield _convert_openai_delta(delta)


async def _stream_with_azure(
    client: Any,
    messages: list[Any],
    max_tokens: int,
    tools: list[Any],
) -> AsyncGenerator[UnifiedDelta, None]:
    """使用 Azure AI Inference SDK 进行流式调用（保留原有逻辑）"""
    
    stream = await client.complete(
        max_tokens=max_tokens,
        messages=messages,
        stream=True,
        tools=tools or None,
    )

    async for chunk in stream:
        choices = chunk.choices
        if not choices:
            continue
        
        choice = choices[0]
        delta = choice.delta
        
        if choice.finish_reason == "content_filter":
            raise SafetyCheckError(f"Issue detected in text: {delta.content if delta else ''}")
        if choice.finish_reason == "length":
            logger.warning("Maximum tokens reached %s", max_tokens)
            raise MaximumTokensReachedError(f"Maximum tokens reached {max_tokens}")
        
        if delta:
            yield UnifiedDelta(
                content=getattr(delta, "content", None),
                tool_calls=getattr(delta, "tool_calls", None),
            )


# =============================================================================
# 同步补全函数
# =============================================================================

@retry(
    reraise=True,
    retry=retry_if_exception_type(ValidationError),
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(multiplier=0.8, max=8),
)
@start_as_current_span("llm_completion_sync")
async def completion_sync(
    res_type: type[T],
    system: list[Any],
    validation_callback: Callable[[str | None], tuple[bool, str | None, T | None]],
    validate_json: bool = False,
    _previous_result: str | None = None,
    _retries_remaining: int = 3,
    _validation_error: str | None = None,
) -> T | None:
    """同步补全（用于生成 JSON 等结构化输出）"""
    
    messages = list(system)
    if _validation_error:
        messages.append({"role": "assistant", "content": _previous_result or ""})
        messages.append({"role": "user", "content": f"A validation error occurred, please retry: {_validation_error}"})

    res_content: str | None = await _completion_sync_worker(
        is_fast=False,
        json_output=validate_json,
        system=messages,
    )
    
    if validate_json and res_content:
        res_content = repair_json(json_str=res_content)  # type: ignore

    is_valid, validation_error, res_object = validation_callback(res_content)
    
    if not is_valid:
        if _retries_remaining == 0:
            logger.error("LLM validation error: %s", validation_error)
            return None
        logger.warning(
            "LLM validation error, retrying (%s retries left)",
            _retries_remaining,
        )
        return await completion_sync(
            res_type=res_type,
            system=system,
            validate_json=validate_json,
            validation_callback=validation_callback,
            _previous_result=res_content,
            _retries_remaining=_retries_remaining - 1,
            _validation_error=validation_error,
        )

    return res_object


async def _completion_sync_worker(
    is_fast: bool,
    system: list[Any],
    json_output: bool = False,
    max_tokens: int | None = None,
) -> str | None:
    """同步补全工作函数"""
    
    client, platform = await _use_llm(is_fast)

    prompt = _limit_messages(
        context_window=platform.context,
        max_tokens=max_tokens,
        messages=[],
        model=platform.model,
        system=system,
    )

    retryed = AsyncRetrying(
        reraise=True,
        retry=retry_any(
            *[retry_if_exception_type(exc) for exc in _retried_exceptions]
        ) if _retried_exceptions else retry_if_exception_type(Exception),
        stop=stop_after_attempt(10),
        wait=wait_random_exponential(multiplier=0.8, max=8),
    )
    
    result_content: str | None = None
    
    async for attempt in retryed:
        with attempt:
            if isinstance(client, AsyncOpenAI):
                # OpenAI 兼容客户端
                result_content = await _sync_with_openai(
                    client=client,
                    messages=_to_openai_messages(prompt),
                    model=platform.model,
                    max_tokens=max_tokens,
                    json_output=json_output,
                    seed=platform.seed,
                    temperature=platform.temperature,
                )
            else:
                # Azure 客户端
                result_content = await _sync_with_azure(
                    client=client,
                    messages=prompt,
                    model=platform.model,
                    max_tokens=max_tokens,
                    json_output=json_output,
                    seed=platform.seed,
                    temperature=platform.temperature,
                )
    
    return result_content


async def _sync_with_openai(
    client: AsyncOpenAI,
    messages: list[ChatCompletionMessageParam],
    model: str,
    max_tokens: int | None,
    json_output: bool,
    seed: int,
    temperature: float,
) -> str | None:
    """使用 OpenAI SDK 进行同步调用"""
    
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "seed": seed,
        "temperature": temperature,
    }
    
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    
    if json_output:
        kwargs["response_format"] = {"type": "json_object"}
    
    response = await client.chat.completions.create(**kwargs)
    choice = response.choices[0]
    
            if choice.finish_reason == "content_filter":
        raise SafetyCheckError(f"Issue detected in generation: {choice.message.content}")
    if choice.finish_reason == "length":
        raise MaximumTokensReachedError(f"Maximum tokens reached {max_tokens}")
    
    return choice.message.content


async def _sync_with_azure(
    client: Any,
    messages: list[Any],
    model: str,
    max_tokens: int | None,
    json_output: bool,
    seed: int,
    temperature: float,
) -> str | None:
    """使用 Azure SDK 进行同步调用"""
    
    response = await client.complete(
        max_tokens=max_tokens,
        messages=messages,
        model=model,
        response_format="json_object" if json_output else None,
        seed=seed,
        temperature=temperature,
    )
    
    choice = response.choices[0]
    
    if choice.finish_reason == "content_filter":
        raise SafetyCheckError(f"Issue detected in generation: {choice.message.content}")
            if choice.finish_reason == "length":
                raise MaximumTokensReachedError(f"Maximum tokens reached {max_tokens}")

    return choice.message.content


# =============================================================================
# 辅助函数
# =============================================================================

def _limit_messages(
    context_window: int,
    max_tokens: int | None,
    messages: list[MessageModel],
    model: str,
    system: list[Any],
    max_messages: int = 1000,
    tools: list[Any] | None = None,
) -> list[Any]:
    """限制消息数量以适应上下文窗口"""
    
    max_tokens = max_tokens or 0
    counter = 0
    max_context = context_window - max_tokens
    selected_messages: list[Any] = []
    tokens = 0
    total = min(len(system) + len(messages), max_messages)

    # 添加系统消息
    for message in system:
        tokens += _count_tokens(_dump_message(message), model)
        counter += 1

    # 添加工具
    for tool in tools or []:
        tokens += _count_tokens(_dump_message(tool), model)

    # 从最新到最旧添加用户消息
    for message in messages[::-1]:
        openai_message = message.to_openai()
        new_tokens = _count_tokens(
            "".join([_dump_message(x) for x in openai_message]),
            model,
        )
        if tokens + new_tokens >= max_context:
            break
        if counter >= max_messages:
            break
        counter += 1
        selected_messages += openai_message[::-1]
        tokens += new_tokens

    logger.info("Using %s/%s messages (%s tokens) as context", counter, total, tokens)
    return [
        *system,
        *selected_messages[::-1],
    ]


@lru_cache()
def _count_tokens(content: str, model: str) -> int:
    """计算 token 数量"""
    try:
        encoding_name = tiktoken.encoding_name_for_model(model)
    except KeyError:
        # 对于 qwen 等未知模型，使用 GPT-3.5 的编码
        encoding_name = tiktoken.encoding_name_for_model("gpt-3.5")
        logger.debug("Unknown model %s, using %s encoding", model, encoding_name)
    return len(tiktoken.get_encoding(encoding_name).encode(content))


def _dump_message(message: Any) -> str:
    """将消息转换为 JSON 字符串"""
    if isinstance(message, dict):
        return json.dumps(message)
    elif hasattr(message, "model_dump"):
        return json.dumps(message.model_dump())
    elif hasattr(message, "content"):
        return str(message.content)
    else:
        return str(message)


async def _use_llm(
    is_fast: bool,
) -> tuple[Any, LlmDeploymentModel]:
    """获取 LLM 客户端和配置"""
    return await CONFIG.llm.selected(is_fast).client()
