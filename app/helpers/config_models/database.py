"""
数据库配置模型

支持 MongoDB / ApsaraDB for MongoDB（阿里云）。

配置示例：
```yaml
database:
  mode: mongodb
  mongodb:
    connection_string: mongodb://localhost:27017
    database: call-center-ai
    collection: calls
```
"""

from functools import cached_property
from os import environ
from typing import Literal

from pydantic import BaseModel

from app.persistence.istore import IStore


class MongoDbModel(BaseModel, frozen=True):
    """MongoDB 配置"""
    connection_string: str = "mongodb://localhost:27017"
    database: str = "call-center-ai"
    collection: str = "calls"

    @cached_property
    def instance(self) -> IStore:
        from app.helpers.config import CONFIG
        from app.persistence.mongodb import MongoDbStore
        
        # 支持从环境变量覆盖
        connection_string = environ.get("MONGODB_URI") or self.connection_string
        database = environ.get("MONGODB_DATABASE") or self.database
        
        return MongoDbStore(
            cache=CONFIG.cache.instance,
            connection_string=connection_string,
            database=database,
            collection=self.collection,
        )


class DatabaseModel(BaseModel):
    """
    数据库配置
    
    目前支持 MongoDB，可连接本地 MongoDB 或阿里云 ApsaraDB。
    """
    # 模式选择
    mode: Literal["mongodb"] = "mongodb"
    
    # MongoDB 配置
    mongodb: MongoDbModel = MongoDbModel()

    @cached_property
    def instance(self) -> IStore:
        """返回数据库实例"""
        return self.mongodb.instance
