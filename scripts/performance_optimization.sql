-- ============================================================================
-- PostgreSQL 性能优化配置脚本
-- 包含内存配置、并行查询、连接池和向量搜索优化
-- ============================================================================

-- 注意：此脚本中的某些配置需要在 postgresql.conf 中设置并重启数据库
-- 其他配置可以在运行时通过 ALTER SYSTEM 或 ALTER DATABASE 设置

SET search_path TO screenplay, public;

-- ============================================================================
-- 1. 内存配置优化
-- ============================================================================

-- 这些配置需要在 postgresql.conf 中设置或使用 ALTER SYSTEM
-- 以下是推荐值（假设服务器有 16GB RAM）

-- shared_buffers: PostgreSQL 用于缓存数据的共享内存
-- 推荐值：系统内存的 25%（对于专用数据库服务器）
-- ALTER SYSTEM SET shared_buffers = '4GB';

-- effective_cache_size: 操作系统和数据库可用于缓存的总内存估计
-- 推荐值：系统内存的 50-75%
-- ALTER SYSTEM SET effective_cache_size = '12GB';

-- maintenance_work_mem: 维护操作（VACUUM, CREATE INDEX）使用的内存
-- 推荐值：系统内存的 5-10%，但不超过 2GB
-- ALTER SYSTEM SET maintenance_work_mem = '2GB';

-- work_mem: 每个查询操作（排序、哈希）使用的内存
-- 推荐值：根据并发连接数调整，通常 256MB-1GB
-- ALTER SYSTEM SET work_mem = '256MB';

-- 注意：以上配置需要重启 PostgreSQL 才能生效
-- 使用命令：sudo systemctl restart postgresql

-- ============================================================================
-- 2. 并行查询配置
-- ============================================================================

-- max_parallel_workers_per_gather: 每个 Gather 节点的最大并行工作进程数
-- ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- max_parallel_workers: 系统范围内的最大并行工作进程数
-- ALTER SYSTEM SET max_parallel_workers = 8;

-- max_worker_processes: 后台工作进程的最大数量
-- ALTER SYSTEM SET max_worker_processes = 8;

-- parallel_setup_cost: 启动并行工作进程的成本估计
-- ALTER SYSTEM SET parallel_setup_cost = 1000;

-- parallel_tuple_cost: 通过并行工作进程传输一个元组的成本
-- ALTER SYSTEM SET parallel_tuple_cost = 0.1;

-- min_parallel_table_scan_size: 触发并行扫描的最小表大小
-- ALTER SYSTEM SET min_parallel_table_scan_size = '8MB';

-- ============================================================================
-- 3. 连接和会话配置
-- ============================================================================

-- max_connections: 最大并发连接数
-- 推荐值：根据应用需求，通常 100-200（使用连接池时可以更少）
-- ALTER SYSTEM SET max_connections = 200;

-- idle_in_transaction_session_timeout: 空闲事务超时（毫秒）
-- 防止长时间空闲的事务占用资源
ALTER DATABASE screenplay_db SET idle_in_transaction_session_timeout = '10min';

-- statement_timeout: 语句执行超时（毫秒）
-- 防止长时间运行的查询
ALTER DATABASE screenplay_db SET statement_timeout = '60s';

-- lock_timeout: 锁等待超时（毫秒）
ALTER DATABASE screenplay_db SET lock_timeout = '30s';

-- ============================================================================
-- 4. 查询规划器配置
-- ============================================================================

-- random_page_cost: 随机页面访问的成本估计
-- 对于 SSD，设置为较低值（1.1-2.0）
ALTER DATABASE screenplay_db SET random_page_cost = 1.1;

-- effective_io_concurrency: 并发 I/O 操作数
-- 对于 SSD，可以设置为较高值
ALTER DATABASE screenplay_db SET effective_io_concurrency = 200;

-- default_statistics_target: 统计信息收集的详细程度
-- 较高的值提高查询规划质量但增加 ANALYZE 时间
ALTER DATABASE screenplay_db SET default_statistics_target = 100;

-- ============================================================================
-- 5. WAL（Write-Ahead Logging）配置
-- ============================================================================

-- wal_buffers: WAL 缓冲区大小
-- 推荐值：shared_buffers 的 3%，通常 16MB
-- ALTER SYSTEM SET wal_buffers = '16MB';

-- checkpoint_completion_target: 检查点完成目标
-- 推荐值：0.9（将检查点 I/O 分散到更长时间）
-- ALTER SYSTEM SET checkpoint_completion_target = 0.9;

-- max_wal_size: 触发检查点的最大 WAL 大小
-- ALTER SYSTEM SET max_wal_size = '4GB';

-- min_wal_size: 保留的最小 WAL 大小
-- ALTER SYSTEM SET min_wal_size = '1GB';

-- ============================================================================
-- 6. 向量搜索优化（pgvector 特定）
-- ============================================================================

-- 设置 HNSW 索引的搜索参数
-- ef_search: 搜索时的动态候选列表大小（默认 40）
-- 较大的值提高召回率但降低速度
ALTER DATABASE screenplay_db SET hnsw.ef_search = 100;

-- 为向量搜索启用并行查询
ALTER DATABASE screenplay_db SET max_parallel_workers_per_gather = 4;

-- ============================================================================
-- 7. 自动 VACUUM 配置
-- ============================================================================

-- autovacuum: 启用自动 VACUUM
-- ALTER SYSTEM SET autovacuum = on;

-- autovacuum_max_workers: 自动 VACUUM 工作进程数
-- ALTER SYSTEM SET autovacuum_max_workers = 3;

-- autovacuum_naptime: 自动 VACUUM 运行间隔
-- ALTER SYSTEM SET autovacuum_naptime = '1min';

-- 针对高频更新的表调整 autovacuum 参数
ALTER TABLE code_documents SET (
    autovacuum_vacuum_scale_factor = 0.05,
    autovacuum_analyze_scale_factor = 0.02
);

ALTER TABLE screenplay_sessions SET (
    autovacuum_vacuum_scale_factor = 0.1,
    autovacuum_analyze_scale_factor = 0.05
);

-- ============================================================================
-- 8. 日志配置
-- ============================================================================

-- log_min_duration_statement: 记录慢查询（毫秒）
-- ALTER SYSTEM SET log_min_duration_statement = 1000;

-- log_line_prefix: 日志行前缀格式
-- ALTER SYSTEM SET log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h ';

-- log_statement: 记录的语句类型
-- ALTER SYSTEM SET log_statement = 'ddl';

-- log_checkpoints: 记录检查点
-- ALTER SYSTEM SET log_checkpoints = on;

-- log_connections: 记录连接
-- ALTER SYSTEM SET log_connections = on;

-- log_disconnections: 记录断开连接
-- ALTER SYSTEM SET log_disconnections = on;

-- ============================================================================
-- 9. 创建性能监控视图
-- ============================================================================

-- 慢查询监控视图
CREATE OR REPLACE VIEW slow_queries AS
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time,
    stddev_exec_time,
    rows
FROM pg_stat_statements
WHERE mean_exec_time > 1000  -- 平均执行时间超过 1 秒
ORDER BY mean_exec_time DESC
LIMIT 20;

COMMENT ON VIEW slow_queries IS '慢查询监控视图（需要 pg_stat_statements 扩展）';

-- 表大小监控视图
CREATE OR REPLACE VIEW table_sizes AS
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_indexes_size(schemaname||'.'||tablename)) AS indexes_size,
    pg_total_relation_size(schemaname||'.'||tablename) AS total_bytes
FROM pg_tables
WHERE schemaname = 'screenplay'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

COMMENT ON VIEW table_sizes IS '表大小监控视图';

-- 索引使用情况监控视图
CREATE OR REPLACE VIEW index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE schemaname = 'screenplay'
ORDER BY idx_scan ASC, pg_relation_size(indexrelid) DESC;

COMMENT ON VIEW index_usage IS '索引使用情况监控视图';

-- 缓存命中率监控视图
CREATE OR REPLACE VIEW cache_hit_ratio AS
SELECT 
    'index hit rate' AS name,
    (sum(idx_blks_hit)) / nullif(sum(idx_blks_hit + idx_blks_read), 0) AS ratio
FROM pg_statio_user_indexes
UNION ALL
SELECT 
    'table hit rate' AS name,
    sum(heap_blks_hit) / nullif(sum(heap_blks_hit) + sum(heap_blks_read), 0) AS ratio
FROM pg_statio_user_tables;

COMMENT ON VIEW cache_hit_ratio IS '缓存命中率监控视图（应该 > 0.99）';

-- 连接状态监控视图
CREATE OR REPLACE VIEW connection_stats AS
SELECT 
    state,
    COUNT(*) AS count,
    MAX(EXTRACT(EPOCH FROM (NOW() - state_change))) AS max_duration_seconds
FROM pg_stat_activity
WHERE datname = 'screenplay_db'
GROUP BY state;

COMMENT ON VIEW connection_stats IS '连接状态监控视图';

-- ============================================================================
-- 10. 创建性能优化函数
-- ============================================================================

-- 分析所有表的函数
CREATE OR REPLACE FUNCTION analyze_all_tables()
RETURNS TEXT AS $$
DECLARE
    table_record RECORD;
    result TEXT := '';
BEGIN
    FOR table_record IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'screenplay'
    LOOP
        EXECUTE 'ANALYZE screenplay.' || table_record.tablename;
        result := result || 'Analyzed: ' || table_record.tablename || E'\n';
    END LOOP;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION analyze_all_tables IS '分析所有表以更新统计信息';

-- 重建所有索引的函数
CREATE OR REPLACE FUNCTION reindex_all_tables()
RETURNS TEXT AS $$
DECLARE
    table_record RECORD;
    result TEXT := '';
BEGIN
    FOR table_record IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'screenplay'
    LOOP
        EXECUTE 'REINDEX TABLE screenplay.' || table_record.tablename;
        result := result || 'Reindexed: ' || table_record.tablename || E'\n';
    END LOOP;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION reindex_all_tables IS '重建所有表的索引';

-- 清理和优化数据库的函数
CREATE OR REPLACE FUNCTION vacuum_and_analyze_all()
RETURNS TEXT AS $$
DECLARE
    table_record RECORD;
    result TEXT := '';
BEGIN
    FOR table_record IN 
        SELECT tablename 
        FROM pg_tables 
        WHERE schemaname = 'screenplay'
    LOOP
        EXECUTE 'VACUUM ANALYZE screenplay.' || table_record.tablename;
        result := result || 'Vacuumed and analyzed: ' || table_record.tablename || E'\n';
    END LOOP;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION vacuum_and_analyze_all IS '清理和分析所有表';

-- ============================================================================
-- 11. PgBouncer 连接池配置建议
-- ============================================================================

-- PgBouncer 是一个轻量级的 PostgreSQL 连接池
-- 配置文件示例（pgbouncer.ini）：

/*
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
reserve_pool_timeout = 3
max_db_connections = 100
max_user_connections = 100
server_idle_timeout = 600
server_lifetime = 3600
server_connect_timeout = 15
query_timeout = 0
query_wait_timeout = 120
client_idle_timeout = 0
idle_transaction_timeout = 0
log_connections = 1
log_disconnections = 1
log_pooler_errors = 1
*/

-- ============================================================================
-- 12. 性能基准测试脚本
-- ============================================================================

-- 创建性能测试函数
CREATE OR REPLACE FUNCTION benchmark_vector_search(
    p_workspace_id UUID,
    p_iterations INTEGER DEFAULT 100
)
RETURNS TABLE (
    iteration INTEGER,
    execution_time_ms NUMERIC,
    result_count INTEGER
) AS $$
DECLARE
    i INTEGER;
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    exec_time NUMERIC;
    res_count INTEGER;
    random_embedding vector(1536);
BEGIN
    FOR i IN 1..p_iterations LOOP
        -- 生成随机嵌入向量（实际使用中应该是真实的查询向量）
        random_embedding := (
            SELECT array_agg(random())::vector(1536)
            FROM generate_series(1, 1536)
        );
        
        start_time := clock_timestamp();
        
        SELECT COUNT(*) INTO res_count
        FROM search_similar_documents(p_workspace_id, random_embedding, 5, 0.7);
        
        end_time := clock_timestamp();
        exec_time := EXTRACT(EPOCH FROM (end_time - start_time)) * 1000;
        
        RETURN QUERY SELECT i, exec_time, res_count;
    END LOOP;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION benchmark_vector_search IS '向量搜索性能基准测试';

-- 使用示例：
-- SELECT 
--     AVG(execution_time_ms) AS avg_time_ms,
--     PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY execution_time_ms) AS p95_time_ms,
--     PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY execution_time_ms) AS p99_time_ms
-- FROM benchmark_vector_search('workspace-uuid'::UUID, 100);

-- ============================================================================
-- 完成性能优化配置
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Performance optimization configuration completed.';
    RAISE NOTICE '';
    RAISE NOTICE 'Database-level settings applied:';
    RAISE NOTICE '  - idle_in_transaction_session_timeout = 10min';
    RAISE NOTICE '  - statement_timeout = 60s';
    RAISE NOTICE '  - lock_timeout = 30s';
    RAISE NOTICE '  - random_page_cost = 1.1';
    RAISE NOTICE '  - effective_io_concurrency = 200';
    RAISE NOTICE '  - default_statistics_target = 100';
    RAISE NOTICE '  - hnsw.ef_search = 100';
    RAISE NOTICE '';
    RAISE NOTICE 'System-level settings (require postgresql.conf changes and restart):';
    RAISE NOTICE '  - shared_buffers = 4GB';
    RAISE NOTICE '  - effective_cache_size = 12GB';
    RAISE NOTICE '  - maintenance_work_mem = 2GB';
    RAISE NOTICE '  - work_mem = 256MB';
    RAISE NOTICE '  - max_parallel_workers_per_gather = 4';
    RAISE NOTICE '  - max_parallel_workers = 8';
    RAISE NOTICE '';
    RAISE NOTICE 'Monitoring views created:';
    RAISE NOTICE '  - slow_queries';
    RAISE NOTICE '  - table_sizes';
    RAISE NOTICE '  - index_usage';
    RAISE NOTICE '  - cache_hit_ratio';
    RAISE NOTICE '  - connection_stats';
    RAISE NOTICE '';
    RAISE NOTICE 'Maintenance functions created:';
    RAISE NOTICE '  - analyze_all_tables()';
    RAISE NOTICE '  - reindex_all_tables()';
    RAISE NOTICE '  - vacuum_and_analyze_all()';
    RAISE NOTICE '  - benchmark_vector_search()';
    RAISE NOTICE '';
    RAISE NOTICE 'For PgBouncer connection pooling, see comments in this script.';
END $$;
