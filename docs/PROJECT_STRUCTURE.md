# 品驰关爱中心 AI 客服系统 - 项目结构说明

## 📁 目录结构

```
call-center-ai/
├── app/                          # 🔹 主应用代码
│   ├── main.py                   # FastAPI 应用入口
│   ├── local_chat_routes.py      # 本地对话 API 路由
│   │
│   ├── helpers/                  # 🔹 辅助模块
│   │   ├── config.py             # 配置加载器
│   │   ├── config_models/        # 配置数据模型（详见下方）
│   │   ├── logging.py            # 日志配置
│   │   ├── cache.py              # 缓存工具（LRU等）
│   │   ├── features.py           # 功能标志/默认配置
│   │   ├── monitoring.py         # 监控（空实现）
│   │   ├── resources.py          # 静态资源路径
│   │   │
│   │   ├── asr_client.py         # ✅ Paraformer ASR 客户端
│   │   ├── asr_mock.py           # ASR Mock 实现
│   │   ├── tts_client.py         # ✅ CosyVoice TTS 客户端
│   │   ├── tts_mock.py           # TTS Mock 实现
│   │   │
│   │   ├── llm_worker.py         # LLM 调用核心逻辑
│   │   ├── llm_utils.py          # LLM 工具函数
│   │   ├── llm_tools.py          # LLM 工具插件定义
│   │   ├── local_chat.py         # 本地对话逻辑
│   │   │
│   │   └── pydantic_types/       # 自定义 Pydantic 类型
│   │       └── phone_numbers.py  # 电话号码验证
│   │
│   ├── models/                   # 🔹 数据模型（Pydantic）
│   │   ├── call.py               # 通话/对话会话模型
│   │   ├── message.py            # 消息模型
│   │   ├── claim.py              # 案例/工单模型
│   │   ├── reminder.py           # 提醒模型
│   │   ├── training.py           # 训练数据/知识库模型
│   │   ├── synthesis.py          # 对话摘要模型
│   │   ├── next.py               # 下一步动作模型
│   │   ├── readiness.py          # 健康检查模型
│   │   └── error.py              # 错误响应模型
│   │
│   ├── persistence/              # 🔹 数据持久化层
│   │   ├── istore.py             # 存储接口（抽象基类）
│   │   ├── icache.py             # 缓存接口
│   │   ├── isearch.py            # 搜索接口
│   │   │
│   │   ├── mongodb.py            # ✅ MongoDB 实现
│   │   ├── redis.py              # ✅ Redis 实现
│   │   ├── memory.py             # 内存缓存实现
│   │   ├── memory_queue.py       # 内存队列实现
│   │   └── mock_search.py        # Mock 搜索实现
│   │
│   └── resources/                # 静态资源
│       └── tiktoken/             # Token 计算相关
│
├── scripts/                      # 🔹 开发/测试脚本
│   ├── dev-start.sh              # 启动开发环境
│   ├── test_llm.py               # 测试 LLM
│   ├── test_tts.py               # 测试 TTS
│   ├── test_asr_ws.py            # 测试 ASR
│   ├── test_mongodb.py           # 测试 MongoDB
│   ├── test_redis.py             # 测试 Redis
│   └── delete_calls.py           # 清理通话记录
│
├── cicd/                         # 🔹 CI/CD
│   └── Dockerfile.dev            # 开发环境 Dockerfile
│
├── docs/                         # 🔹 文档
│   ├── PROJECT_STRUCTURE.md      # 本文档
│   ├── ALIYUN_QUICKSTART.md      # 阿里云快速入门
│   └── user_report.png           # 报告截图
│
├── public/                       # 公共静态文件
│   ├── loading.wav               # 加载音效
│   └── lexicon.xml               # 发音词典
│
├── config.yaml                   # ⭐ 主配置文件
├── docker-compose.yml            # ⭐ Docker Compose 配置
├── pyproject.toml                # Python 依赖配置
├── env.example                   # 环境变量模板
└── Makefile                      # 常用命令
```

---

## 🔧 配置模型说明 (`app/helpers/config_models/`)

| 文件 | 作用 | 说明 |
|------|------|------|
| `root.py` | 根配置 | 定义所有配置字段，加载 config.yaml |
| `llm.py` | LLM 配置 | 通义千问 qwen-turbo/qwen-max |
| `tts.py` | TTS 配置 | CosyVoice v3-flash |
| `asr.py` | ASR 配置 | Paraformer realtime v2 |
| `database.py` | 数据库配置 | MongoDB |
| `cache.py` | 缓存配置 | Redis |
| `queue.py` | 队列配置 | 内存队列（本地开发） |
| `ai_search.py` | 搜索配置 | Mock 搜索（待实现 OpenSearch） |
| `conversation.py` | 对话配置 | 客服场景、话术配置 |
| `prompts.py` | 提示词配置 | LLM 系统提示、TTS 话术 |
| `resources.py` | 资源配置 | 静态资源 URL |

---

## 🔄 系统工作流程

### 本地对话测试流程

```
用户输入（Web 界面）
      ↓
[local_chat_routes.py] → 接收 HTTP 请求
      ↓
[local_chat.py] → 处理对话逻辑
      ↓
┌─────────────────────────────────────────┐
│ 1. 调用 LLM (llm_worker.py)              │
│    → 通义千问 qwen-turbo                  │
│    → 返回 AI 回复                         │
│                                         │
│ 2. 调用 TTS (tts_client.py)              │
│    → CosyVoice v3-flash                 │
│    → 返回音频数据                         │
└─────────────────────────────────────────┘
      ↓
返回 JSON 响应（含文字和音频）
      ↓
Web 界面播放语音
```

### 数据流

```
config.yaml
    ↓ 加载
CONFIG (config.py)
    ↓ 提供配置
┌────────────────────────────────────────┐
│ 各个服务实例化：                          │
│ - CONFIG.llm.fast.client()  → LLM      │
│ - CONFIG.tts.instance       → TTS      │
│ - CONFIG.asr.instance       → ASR      │
│ - CONFIG.database.instance  → MongoDB  │
│ - CONFIG.cache.instance     → Redis    │
└────────────────────────────────────────┘
```

---

## 📦 核心模块说明

### 1. LLM 模块 (`llm_worker.py`, `llm_utils.py`, `llm_tools.py`)

```python
# 核心调用逻辑
client, config = await CONFIG.llm.fast.client()
response = await client.chat.completions.create(
    model=config.model,
    messages=[...],
    tools=[...],  # 工具函数定义
)
```

**工具函数** (`llm_tools.py`):

- `updated_claim()` - 更新案例信息
- `new_or_updated_reminder()` - 创建/更新提醒
- `search_document()` - 搜索知识库
- `speech_speed()` - 调整语速
- `speech_lang()` - 切换语言

### 2. TTS 模块 (`tts_client.py`)

```python
# CosyVoice 流式合成
synthesizer = CosyVoiceSynthesizer(
    model="cosyvoice-v3-flash",
    voice="longanyang",
    sample_rate=8000,
)

async for audio_chunk in synthesizer.synthesize_stream("你好"):
    yield audio_chunk
```

### 3. ASR 模块 (`asr_client.py`)

```python
# Paraformer 实时识别
client = ParaformerClient(model="paraformer-realtime-v2")

async for text in client.start_recognition(
    sample_rate=8000,
    format="pcm",
    language="zh",
):
    print(text)  # 识别结果
```

### 4. 数据持久化 (`persistence/`)

```python
# MongoDB
db = CONFIG.database.instance
call = await db.call_get(call_id)
await db.call_create(new_call)

# Redis 缓存
cache = CONFIG.cache.instance
await cache.set(key, ttl, value)
value = await cache.get(key)
```

---

## 🚀 启动方式

```bash
# 启动所有服务
docker compose up -d

# 查看日志
docker compose logs -f app

# 访问
http://localhost:8080/local-chat/
```

---

## 🔮 待实现功能

| 功能 | 状态 | 说明 |
|------|------|------|
| 本地对话测试 | ✅ 已完成 | Web 界面 |
| 通义千问 LLM | ✅ 已完成 | qwen-turbo/max |
| CosyVoice TTS | ✅ 已完成 | v3-flash |
| Paraformer ASR | ✅ 代码完成 | 需开通服务 |
| MongoDB | ✅ 已完成 | - |
| Redis | ✅ 已完成 | - |
| OpenSearch 知识库 | ⏳ 待开发 | - |
| 阿里云 VMS 电话 | ⏳ 待开发 | 需企业资质 |

---

## 📝 环境变量

必需：

```bash
DASHSCOPE_API_KEY=sk-xxx   # 百炼平台 API Key
```

可选：

```bash
LOG_LEVEL=WARNING          # 系统日志级别
APP_LOG_LEVEL=INFO         # 应用日志级别
MONGODB_URI=mongodb://...  # MongoDB 连接（覆盖 config.yaml）
```
