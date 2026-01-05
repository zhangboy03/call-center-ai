#!/usr/bin/env python3
"""
测试阿里云 CosyVoice 实时语音合成

使用方法：
  # 基本测试
  docker compose exec app python scripts/test_tts.py
  
  # 保存到文件
  docker compose exec app python scripts/test_tts.py --save /tmp/output.wav
  
  # 使用指定音色
  docker compose exec app python scripts/test_tts.py --voice longyuan

环境变量：
  BAILIAN_API_KEY 或 DASHSCOPE_API_KEY: 百炼平台 API Key

可用音色（cosyvoice-v3-flash）：
  - longxiaochun: 龙小淳，温柔女声（默认）
  - longxiaoxia: 龙小夏，活泼女声
  - longxiaochen: 龙小晨，成熟男声
  - longxiaobai: 龙小白，知性女声
  - longshu: 龙叔，沉稳男声
  - longyuan: 龙媛，甜美女声
"""

import asyncio
import argparse
import os
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def test_mock_tts():
    """测试 Mock TTS"""
    from app.helpers.tts_mock import MockTtsSynthesizer
    
    print("🔄 测试 1：Mock TTS...")
    
    try:
        async with MockTtsSynthesizer() as tts:
            audio = await tts.synthesize("你好，这是测试")
            print(f"   生成音频：{len(audio)} bytes")
            
            # 测试流式
            chunks = []
            async for chunk in tts.synthesize_stream("流式测试文本"):
                chunks.append(chunk)
            print(f"   流式生成：{len(chunks)} chunks, 总计 {sum(len(c) for c in chunks)} bytes")
        
        print("✅ Mock TTS 测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ Mock TTS 测试失败：{e}")
        return False


async def test_cosyvoice_connection():
    """测试 CosyVoice 连接"""
    print("\n🔄 测试 2：CosyVoice 连接...")
    
    api_key = os.environ.get("BAILIAN_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        print("❌ 错误：BAILIAN_API_KEY 或 DASHSCOPE_API_KEY 环境变量未设置")
        return False
    
    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
    
    try:
        from app.helpers.tts_client import CosyVoiceSynthesizer
        
        tts = CosyVoiceSynthesizer()
        print("✅ CosyVoice 初始化成功！")
        return True
        
    except Exception as e:
        print(f"❌ CosyVoice 初始化失败：{e}")
        return False


async def test_synthesis(voice: str = "longxiaochun", save_path: str | None = None):
    """测试语音合成"""
    from app.helpers.tts_client import CosyVoiceSynthesizer, synthesize_to_file
    
    print(f"\n🔄 测试 3：语音合成（音色：{voice}）...")
    
    test_text = "你好，我是品驰脑起搏器关爱中心的小驰。今天给您打电话是进行术后六个月的随访。请问您现在方便吗？"
    
    print(f"   测试文本：{test_text[:30]}...")
    
    try:
        async with CosyVoiceSynthesizer(voice=voice) as tts:
            chunks = []
            chunk_count = 0
            
            async for chunk in tts.synthesize_stream(test_text):
                chunks.append(chunk)
                chunk_count += 1
                # 显示进度
                print(f"\r   接收音频：{chunk_count} chunks, {sum(len(c) for c in chunks)} bytes", end="")
            
            print()  # 换行
            
            total_audio = b"".join(chunks)
            duration = len(total_audio) / (8000 * 2)  # 8kHz 16bit
            
            print(f"   总音频大小：{len(total_audio)} bytes")
            print(f"   预计时长：{duration:.1f} 秒")
        
        # 保存到文件
        if save_path:
            await synthesize_to_file(test_text, save_path, voice=voice)
            print(f"   已保存到：{save_path}")
        
        print("✅ 语音合成测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 语音合成失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_synthesis():
    """测试流式合成（模拟 LLM 输出）"""
    from app.helpers.tts_client import CosyVoiceSynthesizer
    
    print("\n🔄 测试 4：流式合成（模拟 LLM 输出）...")
    
    # 模拟 LLM 的流式输出
    llm_chunks = [
        "你好，",
        "我是小驰，",
        "来自品驰关爱中心。",
        "今天给您打电话",
        "是进行术后随访。",
    ]
    
    try:
        async with CosyVoiceSynthesizer() as tts:
            await tts.start_stream()
            
            total_audio_bytes = 0
            
            for text_chunk in llm_chunks:
                print(f"   发送文本：{text_chunk}")
                async for audio in tts.feed_text(text_chunk):
                    total_audio_bytes += len(audio)
                await asyncio.sleep(0.1)  # 模拟 LLM 延迟
            
            # 完成合成
            async for audio in tts.finish_stream():
                total_audio_bytes += len(audio)
            
            print(f"   总音频大小：{total_audio_bytes} bytes")
        
        print("✅ 流式合成测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 流式合成失败：{e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    parser = argparse.ArgumentParser(description="测试 CosyVoice TTS")
    parser.add_argument("--voice", default="longxiaochun", help="音色名称")
    parser.add_argument("--save", help="保存音频到文件路径")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧪 阿里云 CosyVoice 实时语音合成测试")
    print("=" * 60)
    
    # 测试 1: Mock TTS
    await test_mock_tts()
    
    # 测试 2: 连接
    if not await test_cosyvoice_connection():
        print("\n⚠️ CosyVoice 连接失败，跳过后续测试")
        return
    
    # 测试 3: 语音合成
    await test_synthesis(voice=args.voice, save_path=args.save)
    
    # 测试 4: 流式合成
    await test_streaming_synthesis()
    
    print("\n" + "=" * 60)
    print("📋 测试完成！")
    print("")
    print("💡 提示：")
    print("   - 如需保存音频：--save /tmp/output.wav")
    print("   - 如需换音色：--voice longyuan")
    print("   - 可用音色：longxiaochun, longyuan, longshu, longxiaobai 等")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

