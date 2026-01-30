# 数据库设置验证清单

使用此清单验证数据库设置是否正确完成。

## ✅ 前置要求检查

- [ ] PostgreSQL 17+ 已安装
- [ ] pgvector 扩展已安装
- [ ] 数据库 `screenplay_db` 已创建
- [ ] 用户 `screenplay_user` 已创建并授权

验证命令：
```bash
psql --version  # 应显示 PostgreSQL 17.x
psql -U postgres -c "SELECT extname, extversion FROM pg_available_extensions WHERE extname = 'vector';"
```

## ✅ 脚本执行检查

- [ ] `init_db.sql` 执行成功
- [ ] `create_core_tables.sql` 执行成功
- [ ] `create_vector_tables.sql` 执行成功
- [ ] `create_log_tables.sql` 执行成功
- [ ] `create_functions.sql` 执行成功
- [ ] `performance_optimization.sql` 执行成功

或者一次性执行：
```bash
psql -U screenplay_user -d screenplay_db -f scripts/setup_database.sql
```

## ✅ 扩展验证

连接到数据库：
```bash
psql -U screenplay_user -d screenplay_db
```

检查扩展：
```sql
\dx
```

应该看到：
- [x] vector
- [x] uuid-ossp

## ✅ 表验证

检查所有表：
```sql
\dt screenplay.*
```

应该看到以下表（共 11 个）：

**核心业务表（7 个）：**
- [ ] tenants
- [ ] users
- [ ] workspaces
- [ ] screenplay_sessions
- [ ] outline_steps
- [ ] screenplay_fragments
- [ ] retrieved_documents

**向量存储表（1 个）：**
- [ ] code_documents

**日志和审计表（4 个）：**
- [ ] execution_logs
- [ ] llm_call_logs
- [ ] audit_logs
- [ ] quota_usage

## ✅ 索引验证

检查向量索引：
```sql
SELECT 
    schemaname,
    tablename,
    indexname,
    indexdef
FROM pg_indexes
WHERE schemaname = 'screenplay'
    AND tablename = 'code_documents'
    AND indexname LIKE '%hnsw%';
```

应该看到：
- [ ] idx_code_documents_embedding_hnsw

## ✅ 函数验证

检查所有函数：
```sql
\df screenplay.*
```

应该看到以下函数（至少 8 个）：
- [ ] search_similar_documents
- [ ] search_by_keywords
- [ ] hybrid_search_documents
- [ ] deduplicate_search_results
- [ ] get_document_statistics
- [ ] batch_update_embeddings
- [ ] analyze_search_performance
- [ ] cleanup_unused_documents
- [ ] cleanup_old_logs
- [ ] analyze_all_tables
- [ ] reindex_all_tables
- [ ] vacuum_and_analyze_all
- [ ] benchmark_vector_search

## ✅ 视图验证

检查所有视图：
```sql
\dv screenplay.*
```

应该看到以下视图（至少 7 个）：
- [ ] vector_db_metrics
- [ ] llm_call_statistics
- [ ] agent_execution_statistics
- [ ] quota_usage_summary
- [ ] slow_queries
- [ ] table_sizes
- [ ] index_usage
- [ ] cache_hit_ratio
- [ ] connection_stats

## ✅ 触发器验证

检查触发器：
```sql
SELECT 
    trigger_name,
    event_object_table,
    action_statement
FROM information_schema.triggers
WHERE trigger_schema = 'screenplay';
```

应该看到 `update_*_updated_at` 触发器用于：
- [ ] tenants
- [ ] users
- [ ] workspaces
- [ ] screenplay_sessions
- [ ] outline_steps
- [ ] code_documents

## ✅ 功能测试

### 1. 测试表插入

```sql
-- 插入测试租户
INSERT INTO screenplay.tenants (name, plan) 
VALUES ('Test Tenant', 'free') 
RETURNING id;

-- 记录返回的 tenant_id，用于后续测试
```

### 2. 测试向量搜索函数

```sql
-- 创建测试工作空间
INSERT INTO screenplay.workspaces (tenant_id, name) 
VALUES ('your-tenant-id', 'Test Workspace') 
RETURNING id;

-- 插入测试文档（不含向量）
INSERT INTO screenplay.code_documents (
    workspace_id, 
    file_path, 
    content, 
    has_deprecated
) VALUES (
    'your-workspace-id',
    'test.py',
    'def deprecated_function(): pass',
    TRUE
);

-- 测试关键词搜索
SELECT * FROM screenplay.search_by_keywords(
    'your-workspace-id'::UUID,
    TRUE,  -- has_deprecated
    NULL,
    NULL,
    NULL,
    10
);
```

### 3. 测试统计函数

```sql
-- 查看表大小
SELECT * FROM screenplay.table_sizes;

-- 查看向量数据库指标
SELECT * FROM screenplay.vector_db_metrics;

-- 查看缓存命中率
SELECT * FROM screenplay.cache_hit_ratio;
```

### 4. 测试维护函数

```sql
-- 分析所有表
SELECT screenplay.analyze_all_tables();

-- 应该返回成功消息
```

## ✅ 性能验证

### 1. 检查数据库配置

```sql
SHOW shared_buffers;
SHOW effective_cache_size;
SHOW work_mem;
SHOW max_parallel_workers_per_gather;
SHOW hnsw.ef_search;
```

### 2. 检查索引大小

```sql
SELECT * FROM screenplay.table_sizes 
WHERE tablename = 'code_documents';
```

### 3. 检查连接状态

```sql
SELECT * FROM screenplay.connection_stats;
```

## ✅ 清理测试数据

```sql
-- 删除测试数据
DELETE FROM screenplay.workspaces WHERE name = 'Test Workspace';
DELETE FROM screenplay.tenants WHERE name = 'Test Tenant';
```

## ✅ 最终检查

运行完整验证：
```sql
-- 1. 检查所有表是否存在
SELECT COUNT(*) FROM information_schema.tables 
WHERE table_schema = 'screenplay';
-- 应该返回 11

-- 2. 检查所有函数是否存在
SELECT COUNT(*) FROM information_schema.routines 
WHERE routine_schema = 'screenplay';
-- 应该返回至少 13

-- 3. 检查所有视图是否存在
SELECT COUNT(*) FROM information_schema.views 
WHERE table_schema = 'screenplay';
-- 应该返回至少 9

-- 4. 检查所有索引是否存在
SELECT COUNT(*) FROM pg_indexes 
WHERE schemaname = 'screenplay';
-- 应该返回大量索引（50+）
```

## 🎉 验证完成

如果所有检查项都通过，数据库设置已成功完成！

### 下一步：

1. **配置 postgresql.conf**（系统级配置）
   - 编辑 `/etc/postgresql/17/main/postgresql.conf`
   - 应用推荐的内存和并行查询配置
   - 重启 PostgreSQL

2. **设置定期维护**
   - 配置 cron 任务执行 VACUUM ANALYZE
   - 配置日志清理任务
   - 配置备份任务

3. **配置 PgBouncer**（可选）
   - 安装 PgBouncer
   - 配置连接池
   - 更新应用连接字符串

4. **开始实现任务 6**
   - 导航器智能体
   - RAG 检索组件
   - 向量搜索集成

## 故障排查

### 问题：扩展未安装

```
ERROR: could not open extension control file
```

**解决方案**：
```bash
# Ubuntu/Debian
sudo apt-get install postgresql-17-pgvector

# macOS
brew install pgvector
```

### 问题：权限不足

```
ERROR: permission denied for schema screenplay
```

**解决方案**：
```sql
GRANT ALL PRIVILEGES ON SCHEMA screenplay TO screenplay_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA screenplay TO screenplay_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA screenplay TO screenplay_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA screenplay TO screenplay_user;
```

### 问题：表已存在

```
ERROR: relation "table_name" already exists
```

**解决方案**：
```sql
-- 删除所有表（谨慎使用！）
DROP SCHEMA screenplay CASCADE;
CREATE SCHEMA screenplay;

-- 重新执行设置脚本
\i scripts/setup_database.sql
```

## 参考文档

- 完整设置指南：`scripts/README_DATABASE.md`
- 设置总结：`scripts/DATABASE_SETUP_SUMMARY.md`
- 设计文档：`.kiro/specs/rag-screenplay-multi-agent/design.md`
