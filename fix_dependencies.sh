#!/bin/bash

# 修复依赖版本冲突的脚本

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║           修复 Pydantic 和 LangSmith 版本冲突                 ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"

echo ""
echo "问题: pydantic v1 和 langsmith 版本不兼容"
echo "解决方案: 升级 langsmith 到最新版本"
echo ""

# 升级 langsmith
echo "正在升级 langsmith..."
pip install --upgrade langsmith

# 升级 langchain-core
echo "正在升级 langchain-core..."
pip install --upgrade langchain-core

# 升级 langgraph
echo "正在升级 langgraph..."
pip install --upgrade langgraph

echo ""
echo "✓ 依赖已更新"
echo ""
echo "现在尝试启动服务:"
echo "  uvicorn src.presentation.api:app --reload"
