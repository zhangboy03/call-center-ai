"""
MongoDB 存储实现

替代 Azure Cosmos DB，使用 MongoDB 作为通话记录存储。
支持本地 MongoDB 和阿里云 ApsaraDB for MongoDB。

使用方法：
    在 config.yaml 中配置：
    ```yaml
    database:
      mode: mongodb
      mongodb:
        connection_string: mongodb://localhost:27017
        database: call-center-ai
        collection: calls
    ```

依赖：
    pip install motor  # MongoDB async driver
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from os import environ
from typing import Any
from uuid import UUID, uuid4

from aiojobs import Scheduler
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorCollection
from pydantic import ValidationError
from pymongo import DESCENDING
from pymongo.errors import PyMongoError

from app.helpers.cache import lru_acache
from app.helpers.features import callback_timeout_hour
from app.helpers.logging import logger
from app.helpers.monitoring import suppress
from app.models.call import CallStateModel
from app.models.readiness import ReadinessEnum
from app.persistence.icache import ICache
from app.persistence.istore import IStore


class MongoDbModel:
    """MongoDB 配置（由 config_models/database.py 导入使用）"""
    connection_string: str
    database: str
    collection: str


class MongoDbStore(IStore):
    """
    MongoDB 存储实现
    
    完全兼容 IStore 接口，可以无缝替换 Cosmos DB。
    """
    
    _connection_string: str
    _database_name: str
    _collection_name: str
    
    def __init__(
        self,
        cache: ICache,
        connection_string: str,
        database: str,
        collection: str,
    ):
        super().__init__(cache)
        self._connection_string = connection_string
        self._database_name = database
        self._collection_name = collection
        logger.info("Using MongoDB %s/%s", database, collection)
    
    async def readiness(self) -> ReadinessEnum:
        """
        检查 MongoDB 服务的就绪状态。
        
        验证 CRUD 操作：创建、读取、更新、删除。
        """
        test_id = str(uuid4())
        test_phone = "+33612345678"
        test_doc = {
            "_id": test_id,
            "initiate": {
                "phone_number": test_phone,
            },
            "test": "readiness_check",
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        
        try:
            async with self._use_collection() as collection:
                # 1. 确保测试文档不存在
                existing = await collection.find_one({"_id": test_id})
                if existing:
                    await collection.delete_one({"_id": test_id})
                
                # 2. 创建文档
                await collection.insert_one(test_doc)
                
                # 3. 读取并验证
                read_doc = await collection.find_one({"_id": test_id})
                if not read_doc:
                    logger.error("MongoDB readiness: document not found after insert")
                    return ReadinessEnum.FAIL
                
                if read_doc.get("test") != "readiness_check":
                    logger.error("MongoDB readiness: document content mismatch")
                    return ReadinessEnum.FAIL
                
                # 4. 更新文档
                await collection.update_one(
                    {"_id": test_id},
                    {"$set": {"test": "updated"}}
                )
                
                # 5. 验证更新
                updated_doc = await collection.find_one({"_id": test_id})
                if not updated_doc or updated_doc.get("test") != "updated":
                    logger.error("MongoDB readiness: update verification failed")
                    return ReadinessEnum.FAIL
                
                # 6. 删除文档
                await collection.delete_one({"_id": test_id})
                
                # 7. 确认删除
                deleted_doc = await collection.find_one({"_id": test_id})
                if deleted_doc:
                    logger.error("MongoDB readiness: document still exists after delete")
                    return ReadinessEnum.FAIL
                
                logger.debug("MongoDB readiness check passed")
                return ReadinessEnum.OK
                
        except PyMongoError as e:
            logger.exception("MongoDB readiness check failed: %s", e)
            return ReadinessEnum.FAIL
        except Exception as e:
            logger.exception("Unknown error during MongoDB readiness check: %s", e)
            return ReadinessEnum.FAIL
    
    async def call_get(
        self,
        call_id: UUID,
    ) -> CallStateModel | None:
        """根据 call_id 获取通话记录"""
        logger.debug("Loading call %s", call_id)
        
        # 尝试从缓存获取
        cache_key = self._cache_key_call_id(call_id)
        cached = await self._cache.get(cache_key)
        if cached:
            try:
                return CallStateModel.model_validate_json(cached)
            except ValidationError as e:
                logger.debug("Cache parsing error: %s", e.errors())
        
        # 从数据库获取
        call = None
        try:
            async with self._use_collection() as collection:
                # MongoDB 使用 _id 作为主键
                doc = await collection.find_one({"_id": str(call_id)})
                if doc:
                    # 将 _id 转换为 id（兼容 Cosmos DB 格式）
                    doc["id"] = doc.pop("_id", None)
                    try:
                        call = CallStateModel.model_validate(doc)
                    except ValidationError as e:
                        logger.debug("Document parsing error: %s", e.errors())
        except PyMongoError as e:
            logger.error("Error accessing MongoDB: %s", e)
        
        # 更新缓存
        if call:
            timeout_hours = await callback_timeout_hour()
            await self._cache.set(
                key=cache_key,
                ttl_sec=max(timeout_hours, 1) * 60 * 60,
                value=call.model_dump_json(),
            )
        
        return call
    
    @asynccontextmanager
    async def call_transac(
        self,
        call: CallStateModel,
        scheduler: Scheduler,
    ) -> AsyncGenerator[None]:
        """
        事务性更新通话记录
        
        使用乐观锁模式：记录初始状态，yield 后计算差异并更新。
        """
        # 记录初始状态
        init_data = call.model_dump(mode="json", exclude_none=True)
        yield
        
        async def _exec() -> None:
            # 计算差异
            current_data = call.model_dump(mode="json", exclude_none=True)
            updates: dict[str, Any] = {}
            
            for field, new_value in current_data.items():
                init_value = init_data.get(field)
                if init_value != new_value:
                    updates[field] = new_value
            
            # 没有变化则跳过
            if not updates:
                logger.debug("No update needed for call %s", call.call_id)
                return
            
            try:
                async with self._use_collection() as collection:
                    result = await collection.update_one(
                        {"_id": str(call.call_id)},
                        {"$set": updates}
                    )
                    
                    if result.modified_count == 0:
                        logger.warning(
                            "Call %s was not updated (may not exist or no changes)",
                            call.call_id
                        )
                        return
                    
                    # 重新读取更新后的文档
                    updated_doc = await collection.find_one({"_id": str(call.call_id)})
                    if updated_doc:
                        updated_doc["id"] = updated_doc.pop("_id", None)
                        try:
                            remote_call = CallStateModel.model_validate(updated_doc)
                            # 刷新本地对象
                            for field in call.model_fields_set:
                                new_value = getattr(remote_call, field)
                                if getattr(call, field) == new_value:
                                    continue
                                with suppress(ValidationError):
                                    setattr(call, field, new_value)
                        except ValidationError:
                            logger.debug("Parsing error after update", exc_info=True)
                    
            except PyMongoError as e:
                logger.error("Error updating call in MongoDB: %s", e)
                return
            
            # 更新缓存
            cache_key = self._cache_key_call_id(call.call_id)
            timeout_hours = await callback_timeout_hour()
            await self._cache.set(
                key=cache_key,
                ttl_sec=max(timeout_hours, 1) * 60 * 60,
                value=call.model_dump_json(),
            )
        
        # 延迟执行更新
        await scheduler.spawn(_exec())
    
    async def call_create(
        self,
        call: CallStateModel,
    ) -> CallStateModel:
        """创建新的通话记录"""
        logger.debug("Creating new call %s", call.call_id)
        
        # 序列化
        data = call.model_dump(mode="json", exclude_none=True)
        # MongoDB 使用 _id 作为主键
        data["_id"] = str(call.call_id)
        
        try:
            async with self._use_collection() as collection:
                await collection.insert_one(data)
        except PyMongoError as e:
            logger.exception("Error creating call in MongoDB: %s", e)
        
        # 更新缓存
        cache_key = self._cache_key_call_id(call.call_id)
        timeout_hours = await callback_timeout_hour()
        await self._cache.set(
            key=cache_key,
            ttl_sec=max(timeout_hours, 1) * 60 * 60,
            value=call.model_dump_json(),
        )
        
        # 使电话号码缓存失效
        cache_key_phone = self._cache_key_phone_number(call.initiate.phone_number)
        await self._cache.delete(cache_key_phone)
        
        return call
    
    async def call_search_one(
        self,
        phone_number: str,
        callback_timeout: bool = True,
    ) -> CallStateModel | None:
        """按电话号码搜索最近的通话"""
        logger.debug("Loading last call for %s", phone_number)
        
        timeout = await callback_timeout_hour()
        if timeout < 1 and callback_timeout:
            logger.debug("Callback timeout is off, skipping search")
            return None
        
        # 尝试从缓存获取
        cache_key = self._cache_key_phone_number(phone_number)
        cached = await self._cache.get(cache_key)
        if cached:
            try:
                return CallStateModel.model_validate_json(cached)
            except ValidationError:
                logger.debug("Cache parsing error", exc_info=True)
        
        # 构建查询条件
        query: dict[str, Any] = {
            "$or": [
                {"initiate.phone_number": {"$regex": f"^{phone_number}$", "$options": "i"}},
                {"claim.policyholder_phone": {"$regex": f"^{phone_number}$", "$options": "i"}},
            ]
        }
        
        # 添加时间过滤
        if callback_timeout:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=timeout)
            query["created_at"] = {"$gte": cutoff_time.isoformat()}
        
        call = None
        try:
            async with self._use_collection() as collection:
                doc = await collection.find_one(
                    query,
                    sort=[("created_at", DESCENDING)]
                )
                if doc:
                    doc["id"] = doc.pop("_id", None)
                    try:
                        call = CallStateModel.model_validate(doc)
                    except ValidationError:
                        logger.debug("Document parsing error", exc_info=True)
        except PyMongoError as e:
            logger.exception("Error searching call in MongoDB: %s", e)
        
        # 更新缓存
        if call:
            await self._cache.set(
                key=cache_key,
                ttl_sec=timeout * 60 * 60,
                value=call.model_dump_json(),
            )
        
        return call
    
    async def call_search_all(
        self,
        count: int,
        phone_number: str | None = None,
    ) -> tuple[list[CallStateModel] | None, int]:
        """搜索所有通话记录"""
        logger.debug("Searching calls for %s, count %s", phone_number, count)
        
        calls, total = await asyncio.gather(
            self._search_calls_worker(count, phone_number),
            self._count_calls_worker(phone_number),
        )
        return calls, total
    
    async def _search_calls_worker(
        self,
        count: int,
        phone_number: str | None = None,
    ) -> list[CallStateModel] | None:
        """搜索通话记录（内部方法）"""
        calls: list[CallStateModel] = []
        
        # 构建查询条件
        query: dict[str, Any] = {}
        if phone_number:
            query["$or"] = [
                {"initiate.phone_number": {"$regex": f"^{phone_number}$", "$options": "i"}},
                {"claim.policyholder_phone": {"$regex": f"^{phone_number}$", "$options": "i"}},
            ]
        
        try:
            async with self._use_collection() as collection:
                cursor = collection.find(query).sort("created_at", DESCENDING).limit(count)
                async for doc in cursor:
                    if not doc:
                        continue
                    doc["id"] = doc.pop("_id", None)
                    try:
                        calls.append(CallStateModel.model_validate(doc))
                    except ValidationError:
                        logger.debug("Document parsing error", exc_info=True)
        except PyMongoError as e:
            logger.exception("Error searching calls in MongoDB: %s", e)
        
        return calls
    
    async def _count_calls_worker(
        self,
        phone_number: str | None = None,
    ) -> int:
        """计数通话记录（内部方法）"""
        # 构建查询条件
        query: dict[str, Any] = {}
        if phone_number:
            query["$or"] = [
                {"initiate.phone_number": {"$regex": f"^{phone_number}$", "$options": "i"}},
                {"claim.policyholder_phone": {"$regex": f"^{phone_number}$", "$options": "i"}},
            ]
        
        try:
            async with self._use_collection() as collection:
                return await collection.count_documents(query)
        except PyMongoError as e:
            logger.exception("Error counting calls in MongoDB: %s", e)
            return 0
    
    @lru_acache()
    async def _get_client(self) -> AsyncIOMotorClient:
        """
        获取 MongoDB 客户端（单例）
        
        支持从环境变量覆盖连接字符串。
        """
        connection_string = environ.get("MONGODB_URI") or self._connection_string
        logger.debug("Connecting to MongoDB: %s", connection_string.split("@")[-1])  # 不记录密码
        
        return AsyncIOMotorClient(
            connection_string,
            # 连接池设置
            maxPoolSize=10,
            minPoolSize=1,
            # 超时设置
            connectTimeoutMS=10000,
            serverSelectionTimeoutMS=10000,
            # 重试设置
            retryWrites=True,
            retryReads=True,
        )
    
    @asynccontextmanager
    async def _use_collection(self) -> AsyncGenerator[AsyncIOMotorCollection]:
        """获取 MongoDB collection"""
        client = await self._get_client()
        database = client[self._database_name]
        yield database[self._collection_name]

