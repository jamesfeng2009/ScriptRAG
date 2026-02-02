# Alembic 迁移使用指南

## 概述

本项目使用 Alembic 管理数据库迁移。RAG 系统的 pgvector 表结构通过 Alembic 进行版本控制。

## 迁移脚本位置

```
alembic/
├── versions/
│   ├── d4e5f6a7b8c9_add_pgvector_tables.py  # RAG pgvector 表结构
│   ├── c3d4e5f6a7b8_remove_duplicate_alembic_version.py
│   ├── b2c3d4e5f6a7_add_llm_call_logs.py
│   ├── a1b2c3d4e5f6_add_rag_fields_to_tasks.py
│   ├── 7b9c8d3e2f1a_remove_unused_tables.py
│   ├── 5f8b9c0d1e2a_add_workspace_tables.py
│   ├── 4a7b8c3d2e1f_remove_tenants_table.py
│   ├── 3f8a9b2c1d5e_remove_workspace_tables.py
│   ├── 285c3c88f28a_add_description_field_to_workspaces.py
│   └── cc267aeab914_initial_migration.py
├── env.py                    # 迁移环境配置
├── script.py.mako           # 迁移脚本模板
└── README                   # Alembic 说明
```

## 环境配置

### 1. 设置环境变量

```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=123456
export POSTGRES_DB=rag_system
```

或在 `.env` 文件中配置：

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5433
POSTGRES_USER=postgres
POSTGRES_PASSWORD=123456
POSTGRES_DB=rag_system
```

### 2. 数据库配置 (alembic.ini)

```ini
[alembic]
sqlalchemy.url = postgresql+asyncpg://postgres:123456@localhost:5433/rag_system
```

## 常用命令

### 查看迁移状态

```bash
# 查看当前数据库版本
alembic current

# 查看所有迁移历史
alembic history

# 查看待执行的迁移
alembic upgrade --sql
```

### 执行迁移

```bash
# 升级到最新版本
alembic upgrade head

# 升级一步
alembic upgrade +1

# 升级到指定版本
alembic upgrade d4e5f6a7b8c9

# 离线升级（不连接数据库，生成 SQL 文件）
alembic upgrade --sql -o migrations/
```

### 回滚迁移

```bash
# 回滚一步
alembic downgrade -1

# 回滚到指定版本
alembic downgrade d4e5f6a7b8c9

# 回滚所有迁移（清空数据库）
alembic downgrade base
```

### 生成新迁移

```bash
# 自动检测模型变更并生成迁移脚本
alembic revision -m "描述变更"

# 生成空迁移脚本（手动编写）
alembic revision -m "描述变更" --empty
```

## RAG 表结构说明

### 主要表

| 表名 | 说明 |
|------|------|
| `document_embeddings` | 文档向量嵌入主表 |
| `document_chunks` | 文档分块表（长文档支持） |
| `knowledge_nodes` | 知识图谱节点 |
| `knowledge_relations` | 知识图谱关系 |
| `retrieval_logs` | 检索历史记录 |
| `api_usage_stats` | API 使用统计 |

### 索引

- `idx_embeddings_vector`: HNSW 向量索引
- `idx_retrieval_logs_workspace`: 检索日志索引
- `idx_api_usage_date`: API 统计按日期索引

### 视图

| 视图名 | 说明 |
|--------|------|
| `daily_cost_summary` | 每日成本汇总 |
| `popular_queries` | 热门查询 |

### 函数

| 函数名 | 说明 |
|--------|------|
| `cleanup_old_logs(retention_days)` | 清理旧检索日志 |
| `cleanup_old_api_stats(retention_days)` | 清理旧 API 统计 |

## 迁移脚本示例

### 创建新迁移脚本

```bash
alembic revision -m "add_custom_field_to_documents"
```

生成的脚本模板：

```python
"""add_custom_field_to_documents

Revision ID: xxxxx
Revises: d4e5f6a7b8c9
Create Date: 2026-02-02 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'xxxxx'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade"""
    op.execute("ALTER TABLE document_embeddings ADD COLUMN custom_field VARCHAR(255)")


def downgrade() -> None:
    """Downgrade"""
    op.execute("ALTER TABLE document_embeddings DROP COLUMN custom_field")
```

### 迁移脚本最佳实践

1. **始终设置正确的 `down_revision`**
   ```python
   down_revision = 'd4e5f6a7b8c9'  # 必须指向前一个迁移版本
   ```

2. **使用 `IF EXISTS` 和 `IF NOT EXISTS`**
   ```python
   op.execute("CREATE INDEX IF NOT EXISTS idx_xxx ON table_name(column)")
   ```

3. **批量操作使用 `execute()`**
   ```python
   op.execute("INSERT INTO table(col1, col2) VALUES (val1, val2)")
   ```

4. **回滚操作要完整**
   ```python
   def downgrade() -> None:
       # 删除在 upgrade 中创建的所有对象
       op.execute("DROP TABLE IF EXISTS new_table")
   ```

## 常见问题

### 1. 迁移失败："relation already exists"

```bash
# 检查当前版本
alembic current

# 如果需要重新执行迁移，先回滚
alembic downgrade -1

# 重新升级
alembic upgrade +1
```

### 2. 迁移顺序错误

检查 `down_revision` 是否正确：

```bash
# 查看迁移链
alembic history --verbose
```

### 3. 离线迁移生成 SQL

```bash
# 生成迁移 SQL 文件
alembic upgrade --sql > migration.sql

# 查看生成的 SQL
cat migration.sql
```

### 4. 数据库未创建

```bash
# 创建数据库
createdb rag_system

# 或使用 psql
psql -U postgres -c "CREATE DATABASE rag_system;"
```

## 与 pgvector_service.py 集成

pgvector_service.py 使用 SQLAlchemy 连接数据库：

```python
from src.services.database.pgvector_service import PgVectorDBService

db = PgVectorDBService(
    host="localhost",
    port=5433,
    database="rag_system",
    user="postgres",
    password="123456"
)

await db.initialize()
```

确保 alembic.ini 中的数据库配置与服务配置一致。

## 测试迁移

```bash
# 1. 创建测试数据库
createdb rag_system_test

# 2. 执行迁移
alembic upgrade head

# 3. 验证表已创建
psql -d rag_system_test -c "\dt"

# 4. 回滚测试
alembic downgrade base

# 5. 删除测试数据库
dropdb rag_system_test
```
