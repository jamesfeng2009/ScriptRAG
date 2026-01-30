-- ============================================================================
-- 日志和审计表创建脚本
-- 包含执行日志、LLM 调用日志、审计日志和配额使用表
-- ============================================================================

SET search_path TO screenplay, public;

-- ============================================================================
-- 1. 执行日志表
-- ============================================================================
CREATE TABLE IF NOT EXISTS execution_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES screenplay_sessions(id) ON DELETE CASCADE,
    agent_name VARCHAR(100) NOT NULL,
    action VARCHAR(100) NOT NULL,
    details JSONB DEFAULT '{}',
    level VARCHAR(20) DEFAULT 'INFO',
    duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 约束
    CONSTRAINT execution_logs_agent_check CHECK (agent_name IN (
        'planner', 'navigator', 'director', 'pivot_manager', 'writer', 'compiler', 'fact_checker'
    )),
    CONSTRAINT execution_logs_level_check CHECK (level IN ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')),
    CONSTRAINT execution_logs_duration_check CHECK (duration_ms >= 0)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_execution_logs_session 
ON execution_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_execution_logs_agent 
ON execution_logs(agent_name);

CREATE INDEX IF NOT EXISTS idx_execution_logs_level 
ON execution_logs(level);

CREATE INDEX IF NOT EXISTS idx_execution_logs_created_at 
ON execution_logs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_execution_logs_session_created 
ON execution_logs(session_id, created_at DESC);

-- GIN 索引用于 JSONB 搜索
CREATE INDEX IF NOT EXISTS idx_execution_logs_details 
ON execution_logs USING gin(details);

-- 注释
COMMENT ON TABLE execution_logs IS '执行日志表，记录智能体的所有操作';
COMMENT ON COLUMN execution_logs.agent_name IS '智能体名称';
COMMENT ON COLUMN execution_logs.action IS '执行的操作';
COMMENT ON COLUMN execution_logs.details IS 'JSON 格式的详细信息';
COMMENT ON COLUMN execution_logs.level IS '日志级别';
COMMENT ON COLUMN execution_logs.duration_ms IS '操作耗时（毫秒）';

-- ============================================================================
-- 2. LLM 调用日志表
-- ============================================================================
CREATE TABLE IF NOT EXISTS llm_call_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES screenplay_sessions(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    task_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) NOT NULL,
    response_time_ms INTEGER,
    token_count INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd DECIMAL(10, 6),
    error_message TEXT,
    retry_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 约束
    CONSTRAINT llm_logs_provider_check CHECK (provider IN ('openai', 'qwen', 'minimax', 'glm', 'other')),
    CONSTRAINT llm_logs_task_check CHECK (task_type IN ('high_performance', 'lightweight', 'embedding')),
    CONSTRAINT llm_logs_status_check CHECK (status IN ('success', 'failed', 'timeout', 'rate_limited')),
    CONSTRAINT llm_logs_response_time_check CHECK (response_time_ms >= 0),
    CONSTRAINT llm_logs_token_count_check CHECK (token_count >= 0),
    CONSTRAINT llm_logs_retry_count_check CHECK (retry_count >= 0)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_llm_logs_session 
ON llm_call_logs(session_id);

CREATE INDEX IF NOT EXISTS idx_llm_logs_provider 
ON llm_call_logs(provider);

CREATE INDEX IF NOT EXISTS idx_llm_logs_model 
ON llm_call_logs(model);

CREATE INDEX IF NOT EXISTS idx_llm_logs_status 
ON llm_call_logs(status);

CREATE INDEX IF NOT EXISTS idx_llm_logs_created_at 
ON llm_call_logs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_llm_logs_provider_status 
ON llm_call_logs(provider, status);

CREATE INDEX IF NOT EXISTS idx_llm_logs_task_type 
ON llm_call_logs(task_type);

-- 部分索引：仅索引失败的调用
CREATE INDEX IF NOT EXISTS idx_llm_logs_failed 
ON llm_call_logs(created_at DESC) 
WHERE status IN ('failed', 'timeout', 'rate_limited');

-- 注释
COMMENT ON TABLE llm_call_logs IS 'LLM 调用日志表，记录所有 LLM API 调用';
COMMENT ON COLUMN llm_call_logs.provider IS 'LLM 提供商：openai, qwen, minimax, glm';
COMMENT ON COLUMN llm_call_logs.task_type IS '任务类型：high_performance, lightweight, embedding';
COMMENT ON COLUMN llm_call_logs.status IS '调用状态：success, failed, timeout, rate_limited';
COMMENT ON COLUMN llm_call_logs.cost_usd IS '调用成本（美元）';
COMMENT ON COLUMN llm_call_logs.retry_count IS '重试次数';

-- ============================================================================
-- 3. 审计日志表
-- ============================================================================
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100) NOT NULL,
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    status VARCHAR(50) DEFAULT 'success',
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 约束
    CONSTRAINT audit_logs_action_check CHECK (action IN (
        'create', 'read', 'update', 'delete', 'login', 'logout', 
        'export', 'import', 'configure', 'execute'
    )),
    CONSTRAINT audit_logs_resource_check CHECK (resource_type IN (
        'user', 'workspace', 'session', 'document', 'configuration', 'system'
    )),
    CONSTRAINT audit_logs_status_check CHECK (status IN ('success', 'failed', 'denied'))
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant 
ON audit_logs(tenant_id);

CREATE INDEX IF NOT EXISTS idx_audit_logs_user 
ON audit_logs(user_id);

CREATE INDEX IF NOT EXISTS idx_audit_logs_action 
ON audit_logs(action);

CREATE INDEX IF NOT EXISTS idx_audit_logs_resource 
ON audit_logs(resource_type, resource_id);

CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at 
ON audit_logs(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_created 
ON audit_logs(tenant_id, created_at DESC);

-- GIN 索引用于 JSONB 搜索
CREATE INDEX IF NOT EXISTS idx_audit_logs_details 
ON audit_logs USING gin(details);

-- 部分索引：仅索引失败和拒绝的操作
CREATE INDEX IF NOT EXISTS idx_audit_logs_failures 
ON audit_logs(created_at DESC) 
WHERE status IN ('failed', 'denied');

-- 注释
COMMENT ON TABLE audit_logs IS '审计日志表，记录所有用户操作和系统事件';
COMMENT ON COLUMN audit_logs.action IS '操作类型';
COMMENT ON COLUMN audit_logs.resource_type IS '资源类型';
COMMENT ON COLUMN audit_logs.resource_id IS '资源 ID';
COMMENT ON COLUMN audit_logs.ip_address IS '客户端 IP 地址';
COMMENT ON COLUMN audit_logs.user_agent IS '客户端 User-Agent';

-- ============================================================================
-- 4. 配额使用表（可选 TimescaleDB 超表）
-- ============================================================================
CREATE TABLE IF NOT EXISTS quota_usage (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    resource_type VARCHAR(50) NOT NULL,
    usage_count INTEGER NOT NULL,
    usage_limit INTEGER,
    cost_usd DECIMAL(10, 6),
    
    -- 约束
    CONSTRAINT quota_usage_resource_check CHECK (resource_type IN (
        'api_calls', 'llm_tokens', 'storage_mb', 'vector_searches', 'sessions'
    )),
    CONSTRAINT quota_usage_count_check CHECK (usage_count >= 0),
    CONSTRAINT quota_usage_limit_check CHECK (usage_limit IS NULL OR usage_limit >= 0),
    
    -- 主键
    PRIMARY KEY (time, tenant_id, resource_type)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_quota_usage_tenant 
ON quota_usage(tenant_id, time DESC);

CREATE INDEX IF NOT EXISTS idx_quota_usage_time 
ON quota_usage(time DESC);

CREATE INDEX IF NOT EXISTS idx_quota_usage_resource 
ON quota_usage(resource_type, time DESC);

-- 注释
COMMENT ON TABLE quota_usage IS '配额使用表，记录租户资源使用情况（时序数据）';
COMMENT ON COLUMN quota_usage.time IS '时间戳（按小时或天聚合）';
COMMENT ON COLUMN quota_usage.resource_type IS '资源类型';
COMMENT ON COLUMN quota_usage.usage_count IS '使用数量';
COMMENT ON COLUMN quota_usage.usage_limit IS '配额限制';
COMMENT ON COLUMN quota_usage.cost_usd IS '成本（美元）';

-- 如果安装了 TimescaleDB，可以将其转换为超表
-- 注意：需要先安装 TimescaleDB 扩展
-- SELECT create_hypertable('quota_usage', 'time', if_not_exists => TRUE);

-- ============================================================================
-- 日志保留策略（可选）
-- ============================================================================

-- 创建函数：清理旧日志
CREATE OR REPLACE FUNCTION cleanup_old_logs(retention_days INTEGER DEFAULT 90)
RETURNS TABLE (
    execution_logs_deleted BIGINT,
    llm_logs_deleted BIGINT,
    audit_logs_deleted BIGINT
) AS $$
DECLARE
    cutoff_date TIMESTAMP WITH TIME ZONE;
    exec_deleted BIGINT;
    llm_deleted BIGINT;
    audit_deleted BIGINT;
BEGIN
    cutoff_date := NOW() - (retention_days || ' days')::INTERVAL;
    
    -- 删除旧的执行日志
    DELETE FROM execution_logs WHERE created_at < cutoff_date;
    GET DIAGNOSTICS exec_deleted = ROW_COUNT;
    
    -- 删除旧的 LLM 调用日志
    DELETE FROM llm_call_logs WHERE created_at < cutoff_date;
    GET DIAGNOSTICS llm_deleted = ROW_COUNT;
    
    -- 审计日志通常保留更长时间，这里不删除
    -- 如需删除，取消注释以下代码：
    -- DELETE FROM audit_logs WHERE created_at < cutoff_date;
    -- GET DIAGNOSTICS audit_deleted = ROW_COUNT;
    audit_deleted := 0;
    
    RETURN QUERY SELECT exec_deleted, llm_deleted, audit_deleted;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION cleanup_old_logs IS '清理指定天数之前的日志（默认 90 天）';

-- 使用示例：
-- SELECT * FROM cleanup_old_logs(90);  -- 清理 90 天前的日志

-- ============================================================================
-- 日志统计视图
-- ============================================================================

-- LLM 调用统计视图
CREATE OR REPLACE VIEW llm_call_statistics AS
SELECT 
    provider,
    model,
    task_type,
    status,
    COUNT(*) AS call_count,
    AVG(response_time_ms) AS avg_response_time_ms,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY response_time_ms) AS p95_response_time_ms,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY response_time_ms) AS p99_response_time_ms,
    SUM(token_count) AS total_tokens,
    SUM(cost_usd) AS total_cost_usd,
    DATE_TRUNC('hour', created_at) AS hour
FROM llm_call_logs
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY provider, model, task_type, status, DATE_TRUNC('hour', created_at)
ORDER BY hour DESC, call_count DESC;

COMMENT ON VIEW llm_call_statistics IS 'LLM 调用统计（最近 24 小时）';

-- 智能体执行统计视图
CREATE OR REPLACE VIEW agent_execution_statistics AS
SELECT 
    agent_name,
    action,
    level,
    COUNT(*) AS execution_count,
    AVG(duration_ms) AS avg_duration_ms,
    MAX(duration_ms) AS max_duration_ms,
    DATE_TRUNC('hour', created_at) AS hour
FROM execution_logs
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY agent_name, action, level, DATE_TRUNC('hour', created_at)
ORDER BY hour DESC, execution_count DESC;

COMMENT ON VIEW agent_execution_statistics IS '智能体执行统计（最近 24 小时）';

-- 配额使用统计视图
CREATE OR REPLACE VIEW quota_usage_summary AS
SELECT 
    t.name AS tenant_name,
    t.plan,
    qu.resource_type,
    SUM(qu.usage_count) AS total_usage,
    MAX(qu.usage_limit) AS usage_limit,
    CASE 
        WHEN MAX(qu.usage_limit) > 0 THEN 
            (SUM(qu.usage_count)::FLOAT / MAX(qu.usage_limit) * 100)::NUMERIC(5,2)
        ELSE NULL
    END AS usage_percentage,
    SUM(qu.cost_usd) AS total_cost_usd,
    DATE_TRUNC('day', qu.time) AS day
FROM quota_usage qu
JOIN tenants t ON qu.tenant_id = t.id
WHERE qu.time >= NOW() - INTERVAL '30 days'
GROUP BY t.name, t.plan, qu.resource_type, DATE_TRUNC('day', qu.time)
ORDER BY day DESC, total_usage DESC;

COMMENT ON VIEW quota_usage_summary IS '配额使用汇总（最近 30 天）';

-- ============================================================================
-- 完成日志和审计表创建
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Log and audit tables created successfully.';
    RAISE NOTICE 'Tables: execution_logs, llm_call_logs, audit_logs, quota_usage';
    RAISE NOTICE 'Views: llm_call_statistics, agent_execution_statistics, quota_usage_summary';
    RAISE NOTICE 'Functions: cleanup_old_logs()';
END $$;
