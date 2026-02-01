#!/bin/bash

# 启动 API 并运行测试的脚本

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║           RAG Screenplay API - 启动和测试脚本                 ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 检查 Python
echo -e "\n${BLUE}检查 Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python3 未安装${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python3 已安装${NC}"

# 检查依赖
echo -e "\n${BLUE}检查依赖...${NC}"
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}⚠ 缺少依赖，正在安装...${NC}"
    pip install -r requirements.txt
fi
echo -e "${GREEN}✓ 依赖已安装${NC}"

# 检查 .env 文件
echo -e "\n${BLUE}检查配置...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${RED}✗ .env 文件不存在${NC}"
    echo -e "${YELLOW}请先运行: cp .env.example .env${NC}"
    exit 1
fi
echo -e "${GREEN}✓ .env 文件存在${NC}"

# 检查 GLM API 密钥
if grep -q "GLM_API_KEY=your-glm-key" .env; then
    echo -e "${RED}✗ GLM_API_KEY 未配置${NC}"
    echo -e "${YELLOW}请在 .env 中设置实际的 GLM API 密钥${NC}"
    exit 1
fi
echo -e "${GREEN}✓ GLM API 密钥已配置${NC}"

# 创建日志目录
mkdir -p logs

# 启动 API
echo -e "\n${BLUE}启动 API 服务...${NC}"
python3 -m src.presentation.api > logs/api.log 2>&1 &
API_PID=$!
echo -e "${GREEN}✓ API 进程已启动 (PID: $API_PID)${NC}"

# 等待 API 启动
echo -e "\n${BLUE}等待 API 启动...${NC}"
sleep 3

# 检查 API 是否运行
if ! kill -0 $API_PID 2>/dev/null; then
    echo -e "${RED}✗ API 启动失败${NC}"
    echo -e "${YELLOW}查看日志: tail -f logs/api.log${NC}"
    exit 1
fi
echo -e "${GREEN}✓ API 已启动${NC}"

# 运行测试
echo -e "\n${BLUE}运行 API 测试...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

python3 test_api.py --wait 120

TEST_RESULT=$?

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 清理
echo -e "\n${BLUE}清理...${NC}"
kill $API_PID 2>/dev/null || true
echo -e "${GREEN}✓ API 已停止${NC}"

# 打印总结
echo -e "\n${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        测试完成                                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}✓ 所有测试通过！${NC}"
    echo -e "\n${BLUE}下一步:${NC}"
    echo "  1. 启动 API: python -m src.presentation.api"
    echo "  2. 访问文档: http://localhost:8000/docs"
    echo "  3. 提交请求: curl -X POST http://localhost:8000/generate ..."
else
    echo -e "\n${RED}✗ 某些测试失败${NC}"
    echo -e "\n${BLUE}故障排除:${NC}"
    echo "  1. 查看日志: tail -f logs/api.log"
    echo "  2. 检查配置: cat .env"
    echo "  3. 检查依赖: pip install -r requirements.txt"
fi

exit $TEST_RESULT
