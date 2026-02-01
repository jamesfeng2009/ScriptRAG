#!/bin/bash

# 开发模式启动脚本
# 使用 uvicorn 启动 API 服务，支持自动重载

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║        RAG Screenplay API - 开发模式启动                      ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# 检查 Python
echo -e "\n${BLUE}检查环境...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "✗ Python3 未安装"
    exit 1
fi
echo -e "${GREEN}✓ Python3 已安装${NC}"

# 检查 uvicorn
if ! python3 -c "import uvicorn" 2>/dev/null; then
    echo -e "${YELLOW}⚠ uvicorn 未安装，正在安装...${NC}"
    pip install uvicorn
fi
echo -e "${GREEN}✓ uvicorn 已安装${NC}"

# 检查 .env 文件
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env 文件不存在${NC}"
    echo "请先运行: cp .env.example .env"
    exit 1
fi
echo -e "${GREEN}✓ .env 文件存在${NC}"

# 创建日志目录
mkdir -p logs

# 启动信息
echo -e "\n${BLUE}启动配置:${NC}"
echo "  主机: 0.0.0.0"
echo "  端口: 8000"
echo "  模式: 开发（自动重载）"
echo "  日志级别: debug"
echo ""
echo -e "${BLUE}访问地址:${NC}"
echo "  API: http://localhost:8000"
echo "  文档: http://localhost:8000/docs"
echo "  ReDoc: http://localhost:8000/redoc"
echo ""
echo -e "${YELLOW}按 Ctrl+C 停止服务${NC}"
echo ""

# 启动服务
uvicorn src.presentation.api:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --log-level debug \
  --access-log
