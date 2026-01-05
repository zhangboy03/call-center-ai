"""
Redis 缓存实现
"""

import hashlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from uuid import uuid4

from redis.asyncio import Connection, ConnectionPool, Redis, SSLConnection
from redis.asyncio.retry import Retry
from redis.backoff import ExponentialBackoff
from redis.exceptions import (
    BusyLoadingError,
    ConnectionError as RedisConnectionError,
    RedisError,
)

from app.helpers.cache import lru_acache
from app.helpers.config_models.cache import RedisModel
from app.helpers.logging import logger
from app.models.readiness import ReadinessEnum
from app.persistence.icache import ICache


class RedisCache(ICache):
    _config: RedisModel

    def __init__(self, config: RedisModel):
        self._config = config

    async def readiness(self) -> ReadinessEnum:
        """检查 Redis 连接状态"""
        test_name = str(uuid4())
        test_value = "test"
        try:
            async with self._use_client() as client:
                assert await client.get(test_name) is None
                await client.set(test_name, test_value)
                assert (await client.get(test_name)).decode() == test_value
                await client.delete(test_name)
                assert await client.get(test_name) is None
            return ReadinessEnum.OK
        except AssertionError:
            logger.exception("Readiness test failed")
        except RedisError:
            logger.exception("Error requesting Redis")
        except Exception:
            logger.exception("Unknown error while checking Redis readiness")
        return ReadinessEnum.FAIL

    async def get(self, key: str) -> bytes | None:
        """获取缓存值"""
        sha_key = self._key_to_hash(key)
        res = None
        try:
            async with self._use_client() as client:
                res = await client.get(sha_key)
        except RedisError:
            logger.exception("Error getting value")
        return res

    async def set(
        self,
        key: str,
        ttl_sec: int,
        value: str | bytes | None,
    ) -> bool:
        """设置缓存值"""
        sha_key = self._key_to_hash(key)
        try:
            async with self._use_client() as client:
                await client.set(
                    ex=ttl_sec,
                    name=sha_key,
                    value=value if value else "",
                )
        except RedisError:
            logger.exception("Error setting value")
            return False
        return True

    async def delete(self, key: str) -> bool:
        """删除缓存值"""
        sha_key = self._key_to_hash(key)
        try:
            async with self._use_client() as client:
                await client.delete(sha_key)
        except RedisError:
            logger.exception("Error deleting value")
            return False
        return True

    @lru_acache()
    async def _use_connection_pool(self) -> ConnectionPool:
        """创建 Redis 连接池"""
        logger.info("Using Redis cache %s:%s", self._config.host, self._config.port)

        return ConnectionPool(
            db=self._config.database,
            health_check_interval=10,
            retry_on_error=[BusyLoadingError, RedisConnectionError],
            retry_on_timeout=True,
            retry=Retry(backoff=ExponentialBackoff(), retries=3),
            socket_connect_timeout=5,
            socket_timeout=1,
            connection_class=SSLConnection if self._config.ssl else Connection,
            host=self._config.host,
            port=self._config.port,
            password=self._config.password.get_secret_value()
            if self._config.password
            else None,
        )

    @asynccontextmanager
    async def _use_client(self) -> AsyncGenerator[Redis]:
        """获取 Redis 客户端"""
        async with Redis(
            auto_close_connection_pool=False,
            connection_pool=await self._use_connection_pool(),
        ) as client:
            yield client

    @staticmethod
    def _key_to_hash(key: str) -> bytes:
        """将 key 转换为 hash"""
        return hashlib.sha256(key.encode(), usedforsecurity=False).digest()
