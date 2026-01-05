"""
内存队列实现

用于本地开发，替代 Azure Queue Storage。
不持久化，应用重启后队列清空。

使用方法：
    在 config.yaml 中配置：
    ```yaml
    queue:
      mode: memory  # 或 azure
      call_name: call-queue
      post_name: post-queue
      sms_name: sms-queue
      training_name: training-queue
    ```
"""

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from uuid import uuid4

from pydantic import BaseModel

from app.helpers.cache import get_scheduler
from app.helpers.logging import logger


class Message(BaseModel):
    """消息模型（与 Azure Queue Storage 兼容）"""
    content: str
    delete_token: str | None = None
    dequeue_count: int | None = 0
    message_id: str


class MemoryQueue:
    """
    内存队列实现
    
    使用 asyncio.Queue 作为底层存储。
    接口与 AzureQueueStorage 完全兼容。
    """
    
    _name: str
    _queue: asyncio.Queue[Message]
    _processing: dict[str, Message]  # message_id -> Message（正在处理的消息）
    
    def __init__(self, name: str) -> None:
        self._name = name
        self._queue = asyncio.Queue()
        self._processing = {}
        logger.info("Using memory queue '%s'", name)
    
    async def send_message(self, message: str) -> None:
        """发送消息到队列"""
        msg = Message(
            content=message,
            delete_token=str(uuid4()),
            dequeue_count=0,
            message_id=str(uuid4()),
        )
        await self._queue.put(msg)
        logger.debug("Message sent to queue '%s': %s", self._name, msg.message_id)
    
    async def receive_messages(
        self,
        max_messages: int,
        visibility_timeout: int,  # 暂时忽略，内存队列不需要
    ) -> AsyncGenerator[Message]:
        """
        接收消息
        
        从队列中获取消息，最多返回 max_messages 条。
        如果队列为空，等待最多 1 秒钟。
        """
        received = 0
        
        while received < max_messages:
            try:
                # 尝试从队列获取消息，超时 0.1 秒
                msg = await asyncio.wait_for(self._queue.get(), timeout=0.1)
                
                # 增加处理计数
                msg.dequeue_count = (msg.dequeue_count or 0) + 1
                
                # 加入正在处理的集合
                self._processing[msg.message_id] = msg
                
                received += 1
                yield msg
                
            except asyncio.TimeoutError:
                # 队列为空，退出循环
                break
    
    async def delete_message(self, message: Message) -> None:
        """删除消息（标记为已处理）"""
        if message.message_id in self._processing:
            del self._processing[message.message_id]
            logger.debug("Message deleted from queue '%s': %s", self._name, message.message_id)
    
    async def trigger(
        self,
        arg: str,
        func: Callable[..., Awaitable],
    ) -> None:
        """
        触发函数处理消息
        
        这是主要的消息处理循环，与 Azure Queue Storage 行为一致。
        """
        logger.info(
            'Memory Queue "%s" is set to trigger function "%s"',
            self._name,
            func.__name__,
        )
        
        async with get_scheduler() as scheduler:
            try:
                while True:
                    # 接收消息
                    async for message in self.receive_messages(
                        max_messages=32,
                        visibility_timeout=32 * 5,
                    ):
                        await scheduler.spawn(
                            self._process_message(
                                arg=arg,
                                func=func,
                                message=message,
                            )
                        )
                    
                    # 添加小延迟避免 CPU 过高
                    await asyncio.sleep(1)
                    
            except asyncio.CancelledError:
                logger.debug('Memory Queue "%s" trigger task cancelled', self._name)
            finally:
                await scheduler.close()
    
    async def _process_message(
        self,
        arg: str,
        func: Callable[..., Awaitable],
        message: Message,
    ) -> None:
        """处理单条消息"""
        try:
            kwargs = {arg: message}
            await func(**kwargs)
            await self.delete_message(message)
        except Exception as e:
            logger.error("Error processing message in queue '%s': %s", self._name, e)
            # 将消息放回队列（重试）
            if message.dequeue_count and message.dequeue_count < 3:
                await self._queue.put(message)
                logger.debug("Message re-queued for retry: %s", message.message_id)
    
    @property
    def queue_size(self) -> int:
        """获取队列大小（用于监控）"""
        return self._queue.qsize()
    
    @property
    def processing_count(self) -> int:
        """获取正在处理的消息数量"""
        return len(self._processing)

