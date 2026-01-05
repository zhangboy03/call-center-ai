# 🚀 Call-Center-AI 阿里云版快速启动指南

本指南将帮助你在本地使用 Docker 快速启动 Call-Center-AI 阿里云版本。

---

## 📋 前置要求

### 1. 本地环境
- [x] Docker Desktop 已安装（你已有 Docker 28.5.2 ✅）
- [x] Docker Compose 已安装（你已有 v2.40.3 ✅）

### 2. 阿里云服务开通

#### 阶段 1-2（现在需要）：通义千问
1. 访问 [百炼平台控制台](https://bailian.console.aliyun.com/)
2. 开通服务（有免费额度）
3. 在「API-KEY 管理」创建 API Key
4. 记录你的 API Key（格式：`sk-xxxxxxxxxxxxxxxx`）

#### 阶段 3+（后续需要）：
- NLS 智能语音：https://nls-portal.console.aliyun.com/
- OpenSearch：https://opensearch.console.aliyun.com/
- MongoDB：https://mongodb.console.aliyun.com/
- Redis：https://kvstore.console.aliyun.com/

---

## 🎯 快速启动（3 步）

### Step 1：配置环境变量

```bash
# 进入项目目录
cd /Users/keeplearning/Desktop/Final/ccai-pinch-med/call-center-ai

# 从模板创建 .env 文件
cp env.example .env

# 编辑 .env 文件，填写你的阿里云凭证
```

打开 `.env` 文件，至少填写以下内容：

```dotenv
# 通义千问 API Key（必填）
DASHSCOPE_API_KEY=sk-你的真实APIKey

# 阿里云 AccessKey（如果需要其他服务）
ALIYUN_ACCESS_KEY_ID=你的AccessKeyID
ALIYUN_ACCESS_KEY_SECRET=你的AccessKeySecret
```

### Step 2：配置应用

```bash
# 从阿里云模板创建配置文件
cp config.aliyun.example.yaml config.yaml
```

> 💡 默认配置已经可以运行，无需修改。后续可根据需要调整。

### Step 3：启动服务

```bash
# 方式 1：使用启动脚本（推荐）
./scripts/dev-start.sh

# 方式 2：直接使用 docker compose
docker compose up --build
```

---

## ✅ 验证启动成功

### 1. 检查服务状态

```bash
docker compose ps
```

应该看到 3 个服务都是 `running` 状态：
- `call-center-ai-app`
- `call-center-ai-redis`
- `call-center-ai-mongo`

### 2. 访问应用

| 地址 | 说明 |
|------|------|
| http://localhost:8080/health/liveness | 健康检查（应返回 200） |
| http://localhost:8080/health/readiness | 就绪检查（显示各组件状态） |
| http://localhost:8080/report | Web 报告界面 |
| http://localhost:8080/docs | API 文档（Swagger） |

### 3. 查看日志

```bash
# 查看应用日志
docker compose logs -f app

# 查看所有服务日志
docker compose logs -f
```

---

## 🔧 常用命令

```bash
# 启动服务
./scripts/dev-start.sh

# 重新构建并启动（代码有重大变更时）
./scripts/dev-start.sh --build

# 启动包含数据库管理工具
./scripts/dev-start.sh --tools
# 然后访问：
#   MongoDB 管理：http://localhost:8081 (admin/admin123)
#   Redis 管理：  http://localhost:8082

# 停止服务
./scripts/dev-start.sh --down

# 停止服务并清理数据
./scripts/dev-start.sh --clean

# 查看日志
docker compose logs -f app

# 进入容器调试
docker compose exec app bash
```

---

## 🔑 环境变量说明

| 变量名 | 说明 | 必填 | 示例 |
|--------|------|------|------|
| `DASHSCOPE_API_KEY` | 通义千问 API Key | ✅ 是 | `sk-xxxx` |
| `ALIYUN_ACCESS_KEY_ID` | 阿里云 AccessKey ID | 部分服务需要 | `LTAI5xxx` |
| `ALIYUN_ACCESS_KEY_SECRET` | 阿里云 AccessKey Secret | 部分服务需要 | `xxx` |
| `ALIYUN_NLS_APP_KEY` | NLS 语音服务 AppKey | 语音功能需要 | `xxx` |
| `PUBLIC_DOMAIN` | 应用公开域名 | 否 | `http://localhost:8080` |
| `LOG_LEVEL` | 日志级别 | 否 | `info` |

---

## 🐛 常见问题

### Q: 启动时报错 "DASHSCOPE_API_KEY not set"
A: 请确保 `.env` 文件中正确设置了 `DASHSCOPE_API_KEY`。

### Q: 容器启动后立即退出
A: 查看日志排查原因：
```bash
docker compose logs app
```

### Q: 端口 8080 被占用
A: 修改 `docker-compose.yml` 中的端口映射，例如改为 `8090:8080`。

### Q: 如何更新代码后重新部署？
A: 代码已经通过 volume 挂载，大部分修改会自动热重载。如果修改了依赖或 Dockerfile，需要重新构建：
```bash
docker compose up --build
```

### Q: 如何查看 MongoDB 中的数据？
A: 启动管理工具：
```bash
./scripts/dev-start.sh --tools
```
然后访问 http://localhost:8081

---

## 📁 项目文件结构

```
call-center-ai/
├── .env                      # 环境变量（不提交 Git）
├── config.yaml               # 应用配置
├── docker-compose.yml        # Docker Compose 配置
├── env.example               # 环境变量模板
├── config.aliyun.example.yaml # 阿里云配置模板
├── scripts/
│   └── dev-start.sh          # 启动脚本
├── cicd/
│   ├── Dockerfile            # 生产镜像
│   └── Dockerfile.dev        # 开发镜像
└── app/                      # 应用代码
```

---

## 🔜 下一步

环境配置完成后，我们将开始第一个改造任务：

**阶段 2.1：将 LLM 调用从 Azure OpenAI 迁移到通义千问**

这是风险最低、影响面最大的改造，完成后你就可以开始使用通义千问进行对话了！

准备好了吗？告诉我「开始改造 LLM」，我们继续下一步。

