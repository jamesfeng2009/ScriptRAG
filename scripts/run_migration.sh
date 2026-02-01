#!/bin/bash

# ============================================================================
# Alembic Migration 执行脚本
# 用于执行数据库迁移，移除 workspace 相关的表和字段
# ============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Alembic Migration - Remove Workspace"
echo "=========================================="
echo ""

# 检查是否在项目根目录
if [ ! -f "alembic.ini" ]; then
    echo "❌ 错误：未找到 alembic.ini 文件"
    echo "请在项目根目录下运行此脚本"
    exit 1
fi

# 加载环境变量
if [ -f ".env" ]; then
    echo "✓ 加载环境变量..."
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  警告：未找到 .env 文件，使用默认配置"
fi

# 显示当前数据库配置
echo ""
echo "数据库配置："
echo "  Host: ${POSTGRES_HOST:-localhost}"
echo "  Port: ${POSTGRES_PORT:-5432}"
echo "  Database: ${POSTGRES_DB:-screenplay_db}"
echo "  User: ${POSTGRES_USER:-postgres}"
echo ""

# 询问用户确认
read -p "是否继续执行 migration？这将移除 workspace 相关的表和字段 (y/N): " confirm
if [[ ! $confirm =~ ^[Yy]$ ]]; then
    echo "❌ 已取消"
    exit 0
fi

echo ""
echo "=========================================="
echo "步骤 1: 检查当前 migration 状态"
echo "=========================================="
alembic current

echo ""
echo "=========================================="
echo "步骤 2: 查看待执行的 migration"
echo "=========================================="
alembic history

echo ""
echo "=========================================="
echo "步骤 3: 执行 migration (upgrade head)"
echo "=========================================="
alembic upgrade head

echo ""
echo "=========================================="
echo "步骤 4: 验证 migration 结果"
echo "=========================================="
alembic current

echo ""
echo "✅ Migration 执行完成！"
echo ""
echo "变更内容："
echo "  ✓ 移除 workspaces 表"
echo "  ✓ 移除 screenplay_sessions.workspace_id 字段"
echo "  ✓ 移除 code_documents.workspace_id 字段"
echo "  ✓ 添加 code_documents.file_path 唯一约束"
echo ""
echo "如需回滚，请执行："
echo "  alembic downgrade -1"
echo ""
