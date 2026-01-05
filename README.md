# 品驰关爱中心 AI 客服系统

基于阿里云 + 通义千问的智能客服系统，用于脑起搏器患者术后随访，强调低延迟语音体验与严格的术后 SOP 流程。

## ✨ 功能特性

- 🤖 **智能对话** - 通义千问 (qwen-turbo / qwen-max)
- 🔊 **语音合成** - 阿里云 CosyVoice v3-flash
- 🎤 **语音识别** - 阿里云 Paraformer realtime v2
- 📚 **知识库 RAG** - FAISS + DashScope Embeddings，带问句判定、填充词清洗与缓存
- 📋 **SOP 状态机** - 术后随访流程驱动，可恢复中断
- 💾 **数据存储** - MongoDB
- ⚡ **高速缓存** - Redis
- 🌐 **Web 界面** - 本地对话测试

## 🚀 快速开始

### 1. 配置环境变量

```bash
cp env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 2. 安装依赖（可选本地开发）

```bash
make install   # 使用 uv 创建虚拟环境并安装依赖
```

### 3. 启动服务

```bash
docker compose up -d
```

### 4. 访问

打开浏览器访问：

```
http://localhost:8080/streaming/     # 语音对话（推荐）
http://localhost:8080/local-chat/    # 文字对话
```

## 📁 项目结构

详见 `HANDOVER.md` 与 `AGENTS.md`。

```
app/
├── main.py                 # FastAPI 入口
├── streaming_routes.py     # 语音对话 (WebSocket)
├── local_chat_routes.py    # 文字对话 API
├── helpers/
│   ├── config_models/      # 配置模型
│   ├── asr_client.py       # Paraformer ASR
│   ├── tts_client.py       # CosyVoice TTS
│   ├── rag.py              # 知识库检索 (FAISS + 清洗 + 缓存)
│   ├── streaming_routes.py # System prompt & RAG gating glue
│   ├── sop_engine.py       # SOP 状态机
│   ├── slot_extractor.py   # 槽位提取
│   └── guardrails.py       # 安全护栏
├── models/                 # 数据模型
├── resources/              # 知识库 & SOP 配置
└── persistence/            # 持久化层
```

## ⚙️ 配置说明

主配置文件：`config.yaml`

```yaml
# LLM - 通义千问
llm:
  fast:
    model: qwen-turbo
  slow:
    model: qwen-max

# TTS - CosyVoice
tts:
  mode: cosyvoice
  cosyvoice:
    model: cosyvoice-v3-flash
    voice: longanyang

# ASR - Paraformer
asr:
  mode: paraformer
  paraformer:
    model: paraformer-realtime-v2

# 数据库 - MongoDB
database:
  mode: mongodb

# 缓存 - Redis
cache:
  mode: redis
```

## 🔧 开发命令

```bash
# 安装依赖（本地）
make install

# 启动开发环境（含 Mongo/Redis）
docker compose up -d

# 查看日志
docker compose logs -f app

# 重启应用
docker compose restart app

# 进入容器
docker compose exec app bash

# 运行静态检查与单测
make test-static
make test-unit   # 或 uv run pytest

# 容器内测试脚本
docker compose exec app python scripts/test_llm.py
docker compose exec app python scripts/test_tts.py
```

## ⚡ 语音低延迟要点

- **ASR**：推荐持久 Paraformer 会话（16 kHz mono）+ 服务器 VAD，`language_hints=['zh']`、heartbeat、热词；固定 100–200 ms PCM 分帧；必要时前/后端重采样。
- **采集/回声**：启用 echoCancellation/noiseSuppression；TTS 播放时硬静音录音，打断时清空待发送音频。
- **LLM 提示**：只传最近摘要+少量对话，口语化展示下一步问题（不用槽位名）；后置长度/问号检查，不要过度限制生成。
- **槽位**：先正则/关键词，缺失必填再 LLM 回填，并做枚举/分值归一。
- **RAG**：低 top_k、缓存，简短回答后立即回到 SOP。
- **TTS**：CosyVoice 预热；统一采样率；流式播放并支持打断。

## 📋 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/streaming/` | GET | 语音对话页面（含 VAD） |
| `/streaming/ws` | WebSocket | 语音对话流 |
| `/local-chat/` | GET | 文字对话页面 |
| `/local-chat/chat` | POST | 发送消息 |
| `/health/readiness` | GET | 健康检查 |

## 🔮 路线图

- [x] LLM 对话（通义千问）
- [x] TTS 语音合成（CosyVoice）
- [x] ASR 语音识别（Paraformer）
- [x] 知识库 RAG (FAISS + Embeddings + 问句判定/缓存)
- [x] SOP 状态机 (术后随访流程，持久化)
- [x] 安全护栏 (VNS 禁止自调参等)
- [ ] 阿里云 VMS 电话服务
- [ ] Gating 规则与随访状态推断落地
- [ ] ACK 部署与监控告警

## 📜 许可证

Apache-2.0

## 🙏 致谢

基于 [Microsoft call-center-ai](https://github.com/microsoft/call-center-ai) 项目重构。
