#!/bin/bash
# =============================================================================
# Call-Center-AI 本地开发启动脚本
# =============================================================================
# 用法：
#   ./scripts/dev-start.sh          启动所有服务
#   ./scripts/dev-start.sh --build  重新构建并启动
#   ./scripts/dev-start.sh --tools  启动包含管理工具（Mongo Express, Redis Commander）
#   ./scripts/dev-start.sh --down   停止所有服务
#   ./scripts/dev-start.sh --clean  停止服务并清理数据
# =============================================================================

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 打印带颜色的消息
info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查 .env 文件
check_env() {
    if [ ! -f ".env" ]; then
        warn ".env 文件不存在，正在从模板创建..."
        if [ -f "env.example" ]; then
            cp env.example .env
            success ".env 文件已创建，请编辑 .env 填写阿里云凭证"
            echo ""
            echo "  必填项："
            echo "    - DASHSCOPE_API_KEY (通义千问 API Key)"
            echo ""
            echo "  获取方式：https://bailian.console.aliyun.com/ → API-KEY 管理"
            echo ""
            read -p "按 Enter 键继续，或 Ctrl+C 退出编辑 .env..."
        else
            error "找不到 env.example 模板文件"
            exit 1
        fi
    fi
}

# 检查 config.yaml 文件
check_config() {
    if [ ! -f "config.yaml" ]; then
        warn "config.yaml 文件不存在"
        if [ -f "config.aliyun.example.yaml" ]; then
            info "正在从阿里云模板创建 config.yaml..."
            cp config.aliyun.example.yaml config.yaml
            success "config.yaml 已创建"
        else
            error "找不到配置模板文件"
            exit 1
        fi
    fi
}

# 启动服务
start_services() {
    local build_flag=""
    local profile_flag=""
    
    if [ "$1" == "--build" ]; then
        build_flag="--build"
        info "将重新构建 Docker 镜像..."
    fi
    
    if [ "$1" == "--tools" ] || [ "$2" == "--tools" ]; then
        profile_flag="--profile tools"
        info "将启动管理工具（Mongo Express: http://localhost:8081, Redis Commander: http://localhost:8082）"
    fi
    
    info "正在启动 Docker 服务..."
    docker compose $profile_flag up $build_flag -d
    
    echo ""
    success "服务启动完成！"
    echo ""
    echo "  📱 应用地址：    http://localhost:8080"
    echo "  📊 健康检查：    http://localhost:8080/health/liveness"
    echo "  📋 报告页面：    http://localhost:8080/report"
    echo ""
    
    if [ -n "$profile_flag" ]; then
        echo "  🔧 MongoDB 管理： http://localhost:8081 (admin/admin123)"
        echo "  🔧 Redis 管理：   http://localhost:8082"
        echo ""
    fi
    
    echo "  查看日志：docker compose logs -f app"
    echo "  停止服务：./scripts/dev-start.sh --down"
    echo ""
}

# 停止服务
stop_services() {
    info "正在停止 Docker 服务..."
    docker compose --profile tools down
    success "服务已停止"
}

# 清理数据
clean_data() {
    warn "这将删除所有本地数据（MongoDB、Redis）！"
    read -p "确认删除？(y/N) " confirm
    if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
        info "正在停止服务并清理数据..."
        docker compose --profile tools down -v
        success "数据已清理"
    else
        info "操作已取消"
    fi
}

# 显示帮助
show_help() {
    echo "Call-Center-AI 本地开发启动脚本"
    echo ""
    echo "用法: ./scripts/dev-start.sh [选项]"
    echo ""
    echo "选项:"
    echo "  (无参数)     启动所有服务"
    echo "  --build      重新构建 Docker 镜像并启动"
    echo "  --tools      启动包含管理工具（Mongo Express, Redis Commander）"
    echo "  --down       停止所有服务"
    echo "  --clean      停止服务并清理所有数据"
    echo "  --help       显示此帮助信息"
    echo ""
}

# 主逻辑
case "$1" in
    --down)
        stop_services
        ;;
    --clean)
        clean_data
        ;;
    --help|-h)
        show_help
        ;;
    *)
        check_env
        check_config
        start_services "$1" "$2"
        ;;
esac

