"""
消息队列配置模型

目前只支持内存队列（本地开发）。
未来可扩展支持阿里云 MNS / RocketMQ。

配置示例：
```yaml
queue:
  mode: memory
  call_name: call-queue
  post_name: post-queue
  sms_name: sms-queue
  training_name: training-queue
```
"""

from functools import cached_property
from typing import Literal

from pydantic import BaseModel


class QueueModel(BaseModel, frozen=True):
    """
    消息队列配置
    
    目前支持内存队列，未来可扩展阿里云 MNS。
    """
    # 模式选择
    mode: Literal["memory", "mns"] = "memory"
    
    # 队列名称
    call_name: str = "call-queue"
    post_name: str = "post-queue"
    sms_name: str = "sms-queue"
    training_name: str = "training-queue"

    def _get_queue(self, name: str):
        """根据模式返回对应的队列实例"""
        if self.mode == "memory":
            from app.persistence.memory_queue import MemoryQueue
            return MemoryQueue(name=name)
        elif self.mode == "mns":
            # TODO: 实现阿里云 MNS 支持
            raise NotImplementedError("阿里云 MNS 支持正在开发中")
        else:
            raise ValueError(f"Unknown queue mode: {self.mode}")

    @cached_property
    def call(self):
        """Call 队列"""
        return self._get_queue(self.call_name)

    @cached_property
    def post(self):
        """Post 队列"""
        return self._get_queue(self.post_name)

    @cached_property
    def sms(self):
        """SMS 队列"""
        return self._get_queue(self.sms_name)

    @cached_property
    def training(self):
        """Training 队列"""
        return self._get_queue(self.training_name)
