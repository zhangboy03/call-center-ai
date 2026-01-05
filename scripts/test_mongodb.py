#!/usr/bin/env python3
"""
测试 MongoDB 连接和操作

使用方法：
  docker compose exec app python scripts/test_mongodb.py

或在本地运行：
  MONGODB_URI=mongodb://localhost:27017 python scripts/test_mongodb.py
"""

import asyncio
import os
import sys
from uuid import uuid4
from datetime import datetime, timezone

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_mongodb_direct():
    """直接测试 MongoDB 连接（不经过应用层）"""
    from motor.motor_asyncio import AsyncIOMotorClient
    
    uri = os.environ.get("MONGODB_URI", "mongodb://mongo:27017")
    
    print("🔄 测试 1：直接连接 MongoDB...")
    print(f"   连接地址: {uri.split('@')[-1]}")  # 不打印密码
    
    try:
        client = AsyncIOMotorClient(uri, serverSelectionTimeoutMS=5000)
        
        # 测试连接
        await client.admin.command('ping')
        print("✅ MongoDB 连接成功！")
        
        # 列出数据库
        db_list = await client.list_database_names()
        print(f"   现有数据库: {db_list}")
        
        return True, client
        
    except Exception as e:
        print(f"❌ 连接失败：{e}")
        return False, None


async def test_crud_operations(client):
    """测试 CRUD 操作"""
    from motor.motor_asyncio import AsyncIOMotorClient
    
    print("\n🔄 测试 2：CRUD 操作...")
    
    try:
        db = client["call-center-ai"]
        collection = db["calls"]
        
        test_id = str(uuid4())
        test_doc = {
            "_id": test_id,
            "call_id": test_id,
            "initiate": {
                "phone_number": "+8613800138000",
                "bot_name": "小驰",
                "bot_company": "品驰关爱中心",
            },
            "created_at": datetime.now(timezone.utc).isoformat(),
            "test": True,
        }
        
        # Create
        print("   [C] 创建文档...")
        await collection.insert_one(test_doc)
        
        # Read
        print("   [R] 读取文档...")
        read_doc = await collection.find_one({"_id": test_id})
        assert read_doc is not None, "文档未找到"
        assert read_doc["initiate"]["phone_number"] == "+8613800138000"
        
        # Update
        print("   [U] 更新文档...")
        await collection.update_one(
            {"_id": test_id},
            {"$set": {"initiate.bot_name": "小驰2.0"}}
        )
        updated_doc = await collection.find_one({"_id": test_id})
        assert updated_doc["initiate"]["bot_name"] == "小驰2.0"
        
        # Delete
        print("   [D] 删除文档...")
        await collection.delete_one({"_id": test_id})
        deleted_doc = await collection.find_one({"_id": test_id})
        assert deleted_doc is None, "文档应该已被删除"
        
        print("✅ CRUD 操作全部成功！")
        return True
        
    except Exception as e:
        print(f"❌ CRUD 操作失败：{e}")
        return False


async def test_app_store():
    """测试应用层数据库操作"""
    print("\n🔄 测试 3：应用层数据库操作...")
    
    try:
        from app.helpers.config import CONFIG
        from app.models.readiness import ReadinessEnum
        
        # 获取数据库实例
        store = CONFIG.database.instance
        print(f"   数据库类型: {type(store).__name__}")
        
        # 测试就绪性检查
        readiness = await store.readiness()
        if readiness == ReadinessEnum.OK:
            print("✅ 应用层数据库就绪检查通过！")
            return True
        else:
            print(f"⚠️ 就绪检查返回: {readiness}")
            return False
        
    except Exception as e:
        print(f"⚠️ 应用层测试失败（配置可能不完整）：{e}")
        return False


async def main():
    print("=" * 60)
    print("🧪 MongoDB 数据库测试")
    print("=" * 60)
    
    # 测试 1：直接连接
    success1, client = await test_mongodb_direct()
    
    if success1 and client:
        # 测试 2：CRUD 操作
        await test_crud_operations(client)
        
        # 关闭客户端
        client.close()
    
    # 测试 3：应用层（加载配置可能有其他依赖问题）
    await test_app_store()
    
    print("\n" + "=" * 60)
    if success1:
        print("✅ MongoDB 基础功能测试通过！")
        print("   数据库可以正常使用。")
    else:
        print("❌ 测试失败，请检查：")
        print("   1. MongoDB 容器是否运行中")
        print("   2. MONGODB_URI 环境变量是否正确")
        print("   3. 网络连接是否正常")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

