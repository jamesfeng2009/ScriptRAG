# 数据库设置指南

本目录包含基于 RAG 的剧本生成多智能体系统的 PostgreSQL 数据库初始化脚本。

## 目录结构

```
scripts/
├── README_DATABASE.md              # 本文件
├── setup_database.sql              # 主设置脚本（执行所有脚本）
├── init_db.sql                     # 数据库初始化和扩展
├── create_core_tables.sql          # 核心业务表
├── create_vector_tables.sql        # 向量存储表
├── create_log_tables.sql           # 日志和审计表
├── create_functions.sql            # 数据库函数
└── performance_optimization.sql    # 性能优化配置
```

## 前置要求

### 1. PostgreSQL 17+

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install postgresql-17

# macOS (Homebrew)
brew install postgresql@17

# 启动 PostgreSQL
sudo systemctl start postgresql  # Linux
brew services start postgresql@17  # macOS
```

### 2. pgvector 扩展

```bash
# Ubuntu/Debian
sudo apt-get install postgresql-17-pgvector

# macOS (Homebrew)
brew install pgvector

# 或从源码编译
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install
```

### 3. TimescaleDB（可选，用于时序数据）

```bash
# Ubuntu/Debian
sudo apt-get install timescaledb-2-postgresql-17

# macOS (Homebrew)
brew tap timescale/tap
brew install timescaledb

# 配置 TimescaleDB
sudo timescaledb-tune
```

## 快速开始

### 1. 创建数据库和用户

```bash
# 以 postgres 用户身份登录
sudo -u postgres psql

# 在 psql 中执行：
CREATE DATABASE screenplay_db;
CREATE USER screenplay_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE screenplay_db TO screenplay_user;

# 退出 psql
\q
```

### 2. 执行设置脚本

```bash
# 方法 1：使用主设置脚本（推荐）
psql -U screenplay_user -d screenplay_db -f scripts/setup_database.sql

# 方法 2：逐个执行脚本
psql -U screenplay_user -d screenplay_db -f scripts/init_db.sql
psql -U screenplay_user -d screenplay_db -f scripts/create_core_tables.sql
psql -U screenplay_user -d screenplay_db -f scripts/create_vector_tables.sql
psql -U screenplay_user -d screenplay_db -f scripts/create_log_tables.sql
psql -U screenplay_user -d screenplay_db -f scripts/create_functions.sql
psql -U screenplay_user -d screenplay_db -f scripts/performance_optimization.sql
```

### 3. 验证安装

```bash
psql -U screenplay_user -d screenplay_db

# 在 psql 中执行：
\dx                          # 查看已安装的扩展
\dt screenplay.*             # 查看所有表
\df screenplay.*             # 查看所有函数
\dv screenplay.*             # 查看所有视图

# 查看表大小
SELECT * FROM screenplay.table_sizes;

# 查看向量数据库指标
SELECT * FROM screenplay.vector_db_metrics;
```

## 数据库架构

### 核心业务表

| 表名 | 说明 |
|------|------|
| `tenants` | 租户表（多租户支持） |
| `users` | 用户表 |
| `workspaces` | 工作空间表 |
| `screenplay_sessions` | 剧本生成会话表 |
| `outline_steps` | 大纲步骤表 |
| `screenplay_fragments` | 剧本片段表 |
| `retrieved_documents` | 检索文档表 |

### 向量存储表

| 表名 | 说明 |
|------|------|
| `code_documents` | 代码文档表（包含 1536 维向量嵌入） |

### 日志和审计表

| 表名 | 说明 |
|------|------|
| `execution_logs` | 执行日志表 |
| `llm_call_logs` | LLM 调用日志表 |
| `audit_logs` | 审计日志表 |
| `quota_usage` | 配额使用表（时序数据） |

### 关键函数

| 函数名 | 说明 |
|--------|------|
| `search_similar_documents()` | 向量相似度搜索 |
| `search_by_keywords()` | 关键词搜索 |
| `hybrid_search_documents()` | 混合搜索（向量 + 关键词） |
| `get_document_statistics()` | 文档统计 |
| `cleanup_old_logs()` | 清理旧日志 |
| `analyze_all_tables()` | 分析所有表 |
| `vacuum_and_analyze_all()` | 清理和分析所有表 |

## 性能优化

### 1. PostgreSQL 配置

编辑 `postgresql.conf`（通常位于 `/etc/postgresql/17/main/postgresql.conf`）：

```ini
# 内存配置（假设 16GB RAM）
shared_buffers = 4GB
effective_cache_size = 12GB
maintenance_work_mem = 2GB
work_mem = 256MB

# 并行查询
max_parallel_workers_per_gather = 4
max_parallel_workers = 8
max_worker_processes = 8

# 连接
max_connections = 200

# WAL
wal_buffers = 16MB
checkpoint_completion_target = 0.9
max_wal_size = 4GB
min_wal_size = 1GB

# 查询规划器（SSD 优化）
random_page_cost = 1.1
effective_io_concurrency = 200

# 日志
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
```

重启 PostgreSQL：

```bash
sudo systemctl restart postgresql
```

### 2. PgBouncer 连接池（可选）

安装 PgBouncer：

```bash
# Ubuntu/Debian
sudo apt-get install pgbouncer

# macOS
brew install pgbouncer
```

配置 `/etc/pgbouncer/pgbouncer.ini`：

```ini
[databases]
screenplay_db = host=localhost port=5432 dbname=screenplay_db

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 25
min_pool_size = 10
reserve_pool_size = 5
max_db_connections = 100
```

启动 PgBouncer：

```bash
sudo systemctl start pgbouncer
```

应用连接到 PgBouncer：

```python
# 连接到 PgBouncer 而不是直接连接 PostgreSQL
DATABASE_URL = "postgresql://screenplay_user:password@localhost:6432/screenplay_db"
```

### 3. 定期维护

创建 cron 任务进行定期维护：

```bash
# 编辑 crontab
crontab -e

# 添加以下任务：

# 每天凌晨 2 点执行 VACUUM ANALYZE
0 2 * * * psql -U screenplay_user -d screenplay_db -c "SELECT vacuum_and_analyze_all();"

# 每周日凌晨 3 点清理 90 天前的日志
0 3 * * 0 psql -U screenplay_user -d screenplay_db -c "SELECT * FROM cleanup_old_logs(90);"

# 每月 1 号凌晨 4 点重建索引
0 4 1 * * psql -U screenplay_user -d screenplay_db -c "SELECT reindex_all_tables();"
```

## 监控和诊断

### 查看慢查询

```sql
-- 需要启用 pg_stat_statements 扩展
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- 查看慢查询
SELECT * FROM screenplay.slow_queries;
```

### 查看表和索引大小

```sql
SELECT * FROM screenplay.table_sizes;
```

### 查看索引使用情况

```sql
SELECT * FROM screenplay.index_usage;
```

### 查看缓存命中率

```sql
-- 应该 > 0.99
SELECT * FROM screenplay.cache_hit_ratio;
```

### 查看连接状态

```sql
SELECT * FROM screenplay.connection_stats;
```

### 向量搜索性能测试

```sql
-- 运行 100 次向量搜索并统计性能
SELECT 
    AVG(execution_time_ms) AS avg_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) AS p95_time_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY execution_time_ms) AS p99_time_ms
FROM benchmark_vector_search('your-workspace-uuid'::UUID, 100);
```

## 备份和恢复

### 备份

```bash
# 完整备份
pg_dump -U screenplay_user -d screenplay_db -F c -f screenplay_db_backup.dump

# 仅备份 schema
pg_dump -U screenplay_user -d screenplay_db -s -f screenplay_db_schema.sql

# 仅备份数据
pg_dump -U screenplay_user -d screenplay_db -a -f screenplay_db_data.sql

# 使用 pg_basebackup（物理备份）
pg_basebackup -D /backup/postgres -Ft -z -P -U screenplay_user
```

### 恢复

```bash
# 从 dump 文件恢复
pg_restore -U screenplay_user -d screenplay_db screenplay_db_backup.dump

# 从 SQL 文件恢复
psql -U screenplay_user -d screenplay_db -f screenplay_db_backup.sql
```

### 自动备份脚本

创建 `/usr/local/bin/backup_screenplay_db.sh`：

```bash
#!/bin/bash
BACKUP_DIR="/backup/screenplay"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/screenplay_db_$DATE.dump"

mkdir -p $BACKUP_DIR

# 执行备份
pg_dump -U screenplay_user -d screenplay_db -F c -f $BACKUP_FILE

# 压缩备份
gzip $BACKUP_FILE

# 删除 30 天前的备份
find $BACKUP_DIR -name "*.dump.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_FILE.gz"
```

添加到 crontab：

```bash
# 每天凌晨 1 点备份
0 1 * * * /usr/local/bin/backup_screenplay_db.sh
```

## 迁移到 Milvus

当向量数据量超过 100 万或搜索 QPS 超过 100 时，考虑迁移到 Milvus：

### 1. 监控迁移触发条件

```sql
-- 查看向量数据库指标
SELECT * FROM screenplay.vector_db_metrics;

-- 迁移建议：
-- 1. total_vectors > 1,000,000
-- 2. 向量搜索 QPS > 100
-- 3. P99 延迟 > 500ms
-- 4. total_size > 100GB
```

### 2. 迁移步骤

1. 安装 Milvus
2. 创建 Milvus 集合（schema 与 PostgreSQL 兼容）
3. 编写数据迁移脚本
4. 实施双写验证
5. 灰度切流
6. 完全切换到 Milvus

详细迁移指南请参考设计文档。

## Docker 部署

使用 Docker Compose 快速部署：

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: screenplay_db
      POSTGRES_USER: screenplay_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./scripts:/docker-entrypoint-initdb.d
    ports:
      - "5432:5432"
    command: >
      postgres
      -c shared_buffers=4GB
      -c effective_cache_size=12GB
      -c maintenance_work_mem=2GB
      -c work_mem=256MB

volumes:
  postgres_data:
```

启动：

```bash
docker-compose up -d
```

## 故障排查

### 问题：pgvector 扩展未安装

```
ERROR: could not open extension control file "/usr/share/postgresql/17/extension/vector.control"
```

解决方案：安装 pgvector 扩展（见前置要求）

### 问题：内存不足

```
ERROR: out of memory
```

解决方案：调整 `work_mem` 和 `shared_buffers` 配置

### 问题：连接数过多

```
FATAL: sorry, too many clients already
```

解决方案：
1. 增加 `max_connections`
2. 使用 PgBouncer 连接池

### 问题：向量搜索慢

解决方案：
1. 调整 `hnsw.ef_search` 参数
2. 增加 `max_parallel_workers_per_gather`
3. 考虑迁移到 Milvus

## 参考资料

- [PostgreSQL 官方文档](https://www.postgresql.org/docs/17/)
- [pgvector 文档](https://github.com/pgvector/pgvector)
- [TimescaleDB 文档](https://docs.timescale.com/)
- [PgBouncer 文档](https://www.pgbouncer.org/)
- [Milvus 文档](https://milvus.io/docs)

## 支持

如有问题，请查看：
1. 设计文档：`.kiro/specs/rag-screenplay-multi-agent/design.md`
2. 需求文档：`.kiro/specs/rag-screenplay-multi-agent/requirements.md`
3. 项目 README：`README.md`
