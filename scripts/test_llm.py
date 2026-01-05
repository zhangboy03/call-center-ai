#!/usr/bin/env python3
"""
测试通义千问 LLM 调用

使用方法：
  docker compose exec app python scripts/test_llm.py

或在本地运行：
  DASHSCOPE_API_KEY=sk-xxx python scripts/test_llm.py
"""

import asyncio
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_dashscope_direct():
    """直接测试 DashScope API（不经过应用层）"""
    from openai import AsyncOpenAI
    
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误：DASHSCOPE_API_KEY 环境变量未设置")
        print("   请在 .env 文件中添加：DASHSCOPE_API_KEY=sk-your-api-key")
        return False
    
    print("🔄 测试 1：直接调用 DashScope API...")
    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    try:
        response = await client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "system", "content": "你是一个友好的助手。"},
                {"role": "user", "content": "你好，请用一句话介绍自己。"}
            ],
            max_tokens=100,
        )
        
        content = response.choices[0].message.content
        print(f"✅ 成功！模型响应：\n   {content}")
        return True
        
    except Exception as e:
        print(f"❌ 调用失败：{e}")
        return False


async def test_stream():
    """测试流式调用"""
    from openai import AsyncOpenAI
    
    api_key = os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        return False
    
    print("\n🔄 测试 2：流式调用...")
    
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    try:
        stream = await client.chat.completions.create(
            model="qwen-turbo",
            messages=[
                {"role": "user", "content": "数到5，每个数字一行"}
            ],
            max_tokens=50,
            stream=True,
        )
        
        print("   流式响应：", end="")
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
        print("✅ 流式调用成功！")
        return True
        
    except Exception as e:
        print(f"❌ 流式调用失败：{e}")
        return False


async def test_app_llm():
    """测试应用层 LLM 调用（如果配置正确）"""
    print("\n🔄 测试 3：应用层 LLM 调用...")
    
    try:
        # 尝试加载应用配置
        from app.helpers.config import CONFIG
        from app.helpers.config_models.llm import DeploymentModel
        
        # 获取 fast 模型配置
        fast_config = CONFIG.llm.fast
        print(f"   模型配置：{fast_config.model}")
        
        # 创建客户端
        client, config = await fast_config.client()
        
        print(f"   客户端类型：{type(client).__name__}")
        
        # 测试调用
        response = await client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "user", "content": "说'测试成功'"}
            ],
            max_tokens=20,
        )
        
        content = response.choices[0].message.content
        print(f"✅ 应用层调用成功！响应：{content}")
        return True
        
    except Exception as e:
        print(f"⚠️ 应用层调用失败（这在配置不完整时是正常的）：{e}")
        return False


async def main():
    print("=" * 60)
    print("🧪 通义千问 LLM 测试")
    print("=" * 60)
    
    # 测试 1：直接 API 调用
    result1 = await test_dashscope_direct()
    
    if result1:
        # 测试 2：流式调用
        await test_stream()
        
        # 测试 3：应用层调用（可能失败，因为其他配置不完整）
        await test_app_llm()
    
    print("\n" + "=" * 60)
    if result1:
        print("✅ LLM 基础功能测试通过！")
        print("   通义千问 API 可以正常使用。")
    else:
        print("❌ 测试失败，请检查：")
        print("   1. DASHSCOPE_API_KEY 是否正确设置")
        print("   2. 网络是否能访问 dashscope.aliyuncs.com")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

