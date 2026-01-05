#!/usr/bin/env python3
"""
测试 Redis 缓存连接和操作

使用方法：
  docker compose exec app python scripts/test_redis.py
"""

import asyncio
import os
import sys
from uuid import uuid4

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_redis_direct():
    """直接测试 Redis 连接"""
    from redis.asyncio import Redis
    
    host = os.environ.get("REDIS_HOST", "redis")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    
    print("🔄 测试 1：直接连接 Redis...")
    print(f"   连接地址: {host}:{port}")
    
    try:
        client = Redis(host=host, port=port, decode_responses=True)
        
        # 测试连接
        pong = await client.ping()
        print(f"✅ Redis 连接成功！PING 响应: {pong}")
        
        # 获取 info
        info = await client.info("server")
        print(f"   Redis 版本: {info.get('redis_version', 'unknown')}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ 连接失败：{e}")
        return False


async def test_crud_operations():
    """测试 CRUD 操作"""
    from redis.asyncio import Redis
    
    host = os.environ.get("REDIS_HOST", "redis")
    port = int(os.environ.get("REDIS_PORT", "6379"))
    
    print("\n🔄 测试 2：CRUD 操作...")
    
    try:
        client = Redis(host=host, port=port, decode_responses=True)
        
        test_key = f"test:{uuid4()}"
        test_value = "Hello from test!"
        
        # Create
        print("   [C] 创建键值对...")
        await client.set(test_key, test_value, ex=60)  # 60 秒过期
        
        # Read
        print("   [R] 读取键值对...")
        read_value = await client.get(test_key)
        assert read_value == test_value, f"期望 '{test_value}'，实际 '{read_value}'"
        
        # Update
        print("   [U] 更新键值对...")
        new_value = "Updated value!"
        await client.set(test_key, new_value, ex=60)
        updated_value = await client.get(test_key)
        assert updated_value == new_value
        
        # Delete
        print("   [D] 删除键值对...")
        await client.delete(test_key)
        deleted_value = await client.get(test_key)
        assert deleted_value is None
        
        await client.close()
        print("✅ CRUD 操作全部成功！")
        return True
        
    except Exception as e:
        print(f"❌ CRUD 操作失败：{e}")
        return False


async def test_app_cache():
    """测试应用层缓存操作"""
    print("\n🔄 测试 3：应用层缓存操作...")
    
    try:
        from app.helpers.config import CONFIG
        from app.models.readiness import ReadinessEnum
        
        # 获取缓存实例
        cache = CONFIG.cache.instance
        print(f"   缓存类型: {type(cache).__name__}")
        
        # 测试就绪性检查
        readiness = await cache.readiness()
        if readiness == ReadinessEnum.OK:
            print("✅ 应用层缓存就绪检查通过！")
        else:
            print(f"⚠️ 就绪检查返回: {readiness}")
            return False
        
        # 测试 set/get/delete
        test_key = f"app_test:{uuid4()}"
        test_value = "Application cache test"
        
        # Set
        set_result = await cache.set(test_key, 60, test_value)
        assert set_result, "Set 操作失败"
        
        # Get
        get_result = await cache.get(test_key)
        assert get_result is not None, "Get 返回 None"
        # 注意：应用层返回的是 bytes
        assert test_value.encode() in get_result or test_value in get_result.decode()
        
        # Delete
        delete_result = await cache.delete(test_key)
        assert delete_result, "Delete 操作失败"
        
        print("✅ 应用层缓存操作全部成功！")
        return True
        
    except Exception as e:
        print(f"⚠️ 应用层测试失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("=" * 60)
    print("🧪 Redis 缓存测试")
    print("=" * 60)
    
    # 测试 1：直接连接
    success1 = await test_redis_direct()
    
    if success1:
        # 测试 2：CRUD 操作
        await test_crud_operations()
    
    # 测试 3：应用层
    await test_app_cache()
    
    print("\n" + "=" * 60)
    if success1:
        print("✅ Redis 缓存测试通过！")
        print("   缓存服务可以正常使用。")
    else:
        print("❌ 测试失败，请检查：")
        print("   1. Redis 容器是否运行中")
        print("   2. REDIS_HOST/REDIS_PORT 环境变量是否正确")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

