#!/usr/bin/env python3
"""
测试阿里云 Paraformer 实时语音识别

使用方法：
  # 使用测试音频生成器
  docker compose exec app python scripts/test_asr_ws.py
  
  # 使用真实 PCM 文件
  docker compose exec app python scripts/test_asr_ws.py /path/to/audio.pcm

环境变量：
  BAILIAN_API_KEY 或 DASHSCOPE_API_KEY: 百炼平台 API Key

注意：
  - 音频格式：PCM 16bit 单声道
  - 采样率：8000Hz（电话音频）或 16000Hz
"""

import asyncio
import os
import sys
import struct
import math

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def generate_test_audio(
    duration_sec: float = 3.0,
    sample_rate: int = 8000,
    frequency: float = 440.0,
) -> bytes:
    """
    生成测试音频（正弦波）
    
    Args:
        duration_sec: 时长（秒）
        sample_rate: 采样率
        frequency: 频率（Hz）
    
    Returns:
        PCM 16bit 单声道音频数据
    """
    samples = []
    num_samples = int(duration_sec * sample_rate)
    
    for i in range(num_samples):
        t = i / sample_rate
        # 生成正弦波
        value = int(32767 * 0.5 * math.sin(2 * math.pi * frequency * t))
        samples.append(value)
    
    # 打包为 16bit PCM
    return struct.pack(f"<{len(samples)}h", *samples)


async def generate_audio_frames(
    audio_data: bytes,
    chunk_size: int = 1600,  # 100ms @ 8kHz 16bit
    sample_rate: int = 8000,
) -> bytes:
    """
    将音频数据分割为帧并模拟实时发送
    
    Args:
        audio_data: 完整的 PCM 音频数据
        chunk_size: 每帧字节数
        sample_rate: 采样率
    
    Yields:
        音频帧
    """
    offset = 0
    while offset < len(audio_data):
        chunk = audio_data[offset:offset + chunk_size]
        if not chunk:
            break
        
        yield chunk
        offset += chunk_size
        
        # 模拟实时发送延迟
        delay = chunk_size / (sample_rate * 2)  # 2 bytes per sample
        await asyncio.sleep(delay * 0.5)  # 稍快一点


async def test_connection():
    """测试 WebSocket 连接"""
    from app.helpers.asr_client import ParaformerClient
    
    print("🔄 测试 1：WebSocket 连接...")
    
    api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误：BAILIAN_API_KEY 或 DASHSCOPE_API_KEY 环境变量未设置")
        print("   请在 .env 文件中添加：BAILIAN_API_KEY=sk-your-api-key")
        return False
    
    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        client = ParaformerClient(api_key=api_key)
        await client.connect()
        print("✅ WebSocket 连接成功！")
        await client.close()
        return True
    except Exception as e:
        print(f"❌ 连接失败：{e}")
        return False


async def test_recognition_with_silence():
    """测试识别流程（使用静音）"""
    from app.helpers.asr_client import ParaformerClient
    
    print("\n🔄 测试 2：识别流程（静音测试）...")
    
    try:
        async with ParaformerClient() as client:
            # 开始识别
            await client.start_recognition(
                sample_rate=8000,
                language_hints=["zh"],
            )
            print("   识别会话已启动")
            
            # 发送静音（全零数据）
            silence = bytes(1600)  # 100ms 静音
            for i in range(10):  # 发送 1 秒静音
                await client.send_audio(silence)
                await asyncio.sleep(0.1)
            
            print("   已发送 1 秒静音")
            
            # 停止识别
            result = await client.stop_recognition()
            print(f"   最终结果：{result.text if result else '(无)'}")
            
        print("✅ 识别流程测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 识别失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_recognition_with_file(file_path: str):
    """测试识别 PCM 文件"""
    from app.helpers.asr_client import recognize_file
    
    print(f"\n🔄 测试 3：识别 PCM 文件...")
    print(f"   文件：{file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return False
    
    file_size = os.path.getsize(file_path)
    duration = file_size / (8000 * 2)  # 假设 8kHz 16bit
    print(f"   大小：{file_size} bytes，时长约 {duration:.1f} 秒")
    
    try:
        # 添加医疗热词
        hot_words = {
            "脑起搏器": 5,
            "帕金森": 5,
            "品驰": 5,
            "DBS": 5,
            "神经调控": 4,
            "术后随访": 4,
        }
        
        result = await recognize_file(
            file_path=file_path,
            sample_rate=8000,
            hot_words=hot_words,
        )
        
        print(f"\n📝 识别结果：")
        print(f"   {result}")
        print("\n✅ 文件识别测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 识别失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_mock_client():
    """测试 Mock 客户端"""
    from app.helpers.asr_mock import MockAsrClient
    
    print("\n🔄 测试 4：Mock 客户端...")
    
    try:
        async def fake_audio():
            for _ in range(50):
                yield bytes(1600)
                await asyncio.sleep(0.01)
        
        async with MockAsrClient() as client:
            await client.start_recognition()
            
            results = []
            async for result in client.stream_audio(fake_audio()):
                results.append(result.text)
                print(f"   收到：{result.text}")
            
            await client.stop_recognition()
        
        print(f"✅ Mock 客户端测试通过！共收到 {len(results)} 条结果")
        return True
        
    except Exception as e:
        print(f"❌ Mock 测试失败：{e}")
        return False


async def main():
    print("=" * 60)
    print("🧪 阿里云 Paraformer 实时语音识别测试")
    print("=" * 60)
    
    # 检查是否有文件参数
    pcm_file = sys.argv[1] if len(sys.argv) > 1 else None
    
    # 测试 Mock 客户端（不需要 API Key）
    await test_mock_client()
    
    # 测试真实连接
    success = await test_connection()
    
    if success:
        # 测试识别流程
        await test_recognition_with_silence()
        
        # 如果提供了文件，测试文件识别
        if pcm_file:
            await test_recognition_with_file(pcm_file)
    
    print("\n" + "=" * 60)
    print("📋 测试完成！")
    print("")
    print("💡 如需测试真实音频识别，请：")
    print("   1. 准备 8kHz 16bit PCM 文件")
    print("   2. 运行：docker compose exec app python scripts/test_asr_ws.py /path/to/audio.pcm")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

