-- pgvector 数据库表结构设计
-- 版本: 1.0
-- 用途: 存储代码文档的向量嵌入，支持语义搜索

-- ============================================
-- 1. 主表: 文档向量表
-- ============================================
CREATE TABLE IF NOT EXISTS document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id VARCHAR(255) NOT NULL,
    file_path TEXT NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    content TEXT NOT NULL,
    content_summary TEXT,
    embedding vector(1024),  -- embedding维度根据模型调整
    language VARCHAR(50),
    file_type VARCHAR(50),
    file_size INTEGER,
    line_count INTEGER,
    has_deprecated BOOLEAN DEFAULT FALSE,
    has_fixme BOOLEAN DEFAULT FALSE,
    has_todo BOOLEAN DEFAULT FALSE,
    has_security BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 约束
    CONSTRAINT unique_file_per_workspace UNIQUE (workspace_id, file_path)
);

-- ============================================
-- 2. 索引: 向量相似度搜索 (HNSW)
-- ============================================
-- HNSW 索引 - 高性能近似最近邻搜索
CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
ON document_embeddings 
USING hnsw (embedding vector_cosine_ops)
WHERE workspace_id IS NOT NULL;

-- 复合索引 - 按工作空间和修改时间查询
CREATE INDEX IF NOT EXISTS idx_embeddings_workspace_time 
ON document_embeddings (workspace_id, updated_at DESC);

-- 标记索引 - 快速过滤
CREATE INDEX IF NOT EXISTS idx_embeddings_markers 
ON document_embeddings (workspace_id, has_deprecated, has_fixme, has_todo, has_security);

-- 文件类型索引
CREATE INDEX IF NOT EXISTS idx_embeddings_filetype 
ON document_embeddings (workspace_id, file_type);

-- ============================================
-- 3. 表: 工作空间管理
-- ============================================
CREATE TABLE IF NOT EXISTS workspaces (
    id VARCHAR(255) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    owner_id VARCHAR(255),
    config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    max_documents INTEGER DEFAULT 10000,
    max_storage_mb INTEGER DEFAULT 1024
);

-- ============================================
-- 4. 表: 检索历史记录
-- ============================================
CREATE TABLE IF NOT EXISTS retrieval_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id VARCHAR(255) NOT NULL,
    user_id VARCHAR(255),
    query TEXT NOT NULL,
    query_embedding vector(1024),
    top_k INTEGER,
    result_count INTEGER,
    total_score FLOAT,
    search_strategy VARCHAR(50),
    duration_ms INTEGER,
    token_usage JSONB,
    cost_usd FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 检索日志索引
CREATE INDEX IF NOT EXISTS idx_retrieval_logs_workspace 
ON retrieval_logs (workspace_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_retrieval_logs_query 
ON retrieval_logs USING gin (query gin_trgm_ops);

-- ============================================
-- 5. 表: API 使用统计
-- ============================================
CREATE TABLE IF NOT EXISTS api_usage_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id VARCHAR(255),
    provider VARCHAR(50) NOT NULL,
    model VARCHAR(100) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    cost_usd FLOAT DEFAULT 0,
    request_count INTEGER DEFAULT 1,
    date DATE DEFAULT CURRENT_DATE,
    hour INTEGER DEFAULT EXTRACT(HOUR FROM NOW()),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- API 使用统计索引
CREATE INDEX IF NOT EXISTS idx_api_usage_date 
ON api_usage_stats (date, provider, model);

CREATE INDEX IF NOT EXISTS idx_api_usage_workspace 
ON api_usage_stats (workspace_id, date);

-- ============================================
-- 6. 表: 文档分块表 (支持长文档)
-- ============================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES document_embeddings(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1024),
    start_line INTEGER,
    end_line INTEGER,
    char_count INTEGER,
    token_count INTEGER,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 分块索引
CREATE INDEX IF NOT EXISTS idx_chunks_document 
ON document_chunks (document_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
ON document_chunks 
USING hnsw (embedding vector_cosine_ops);

-- ============================================
-- 7. 表: 知识图谱节点 (可选扩展)
-- ============================================
CREATE TABLE IF NOT EXISTS knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id VARCHAR(255) NOT NULL,
    node_type VARCHAR(100) NOT NULL,
    name VARCHAR(500) NOT NULL,
    content TEXT,
    embedding vector(1024),
    properties JSONB DEFAULT '{}',
    source_file VARCHAR(500),
    line_number INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 关系表
CREATE TABLE IF NOT EXISTS knowledge_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id VARCHAR(255) NOT NULL,
    source_node_id UUID NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    target_node_id UUID NOT NULL REFERENCES knowledge_nodes(id) ON DELETE CASCADE,
    relation_type VARCHAR(100) NOT NULL,
    strength FLOAT DEFAULT 1.0,
    properties JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT unique_relation UNIQUE (source_node_id, target_node_id, relation_type)
);

-- 知识图谱索引
CREATE INDEX IF NOT EXISTS idx_nodes_workspace_type 
ON knowledge_nodes (workspace_id, node_type);

CREATE INDEX IF NOT EXISTS idx_nodes_embedding 
ON knowledge_nodes 
USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_relations_source 
ON knowledge_relations (source_node_id);

CREATE INDEX IF NOT EXISTS idx_relations_target 
ON knowledge_relations (target_node_id);

-- ============================================
-- 8. 函数: 更新修改时间
-- ============================================
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 触发器
CREATE TRIGGER update_document_embeddings_updated_at 
    BEFORE UPDATE ON document_embeddings 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_knowledge_nodes_updated_at 
    BEFORE UPDATE ON knowledge_nodes 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- 9. 视图: 每日成本汇总
-- ============================================
CREATE OR REPLACE VIEW daily_cost_summary AS
SELECT 
    date,
    provider,
    model,
    SUM(cost_usd) as total_cost,
    SUM(prompt_tokens) as total_prompt_tokens,
    SUM(completion_tokens) as total_completion_tokens,
    SUM(request_count) as total_requests,
    COUNT(DISTINCT workspace_id) as workspace_count
FROM api_usage_stats
WHERE date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY date, provider, model
ORDER BY date DESC, total_cost DESC;

-- ============================================
-- 10. 视图: 热门查询
-- ============================================
CREATE OR REPLACE VIEW popular_queries AS
SELECT 
    workspace_id,
    LEFT(query, 100) as query_preview,
    COUNT(*) as query_count,
    AVG(result_count) as avg_results,
    SUM(token_usage->>'total_tokens')::integer as total_tokens,
    SUM(cost_usd) as total_cost
FROM retrieval_logs
WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
GROUP BY workspace_id, query
ORDER BY query_count DESC
LIMIT 100;

-- ============================================
-- 11. 示例数据: 插入工作空间
-- ============================================
INSERT INTO workspaces (id, name, description, owner_id)
VALUES 
    ('default', '默认工作空间', '系统默认工作空间', 'system'),
    ('project-alpha', '项目 Alpha', '主要项目工作空间', 'admin')
ON CONFLICT (id) DO NOTHING;

-- ============================================
-- 12. 维护任务: 定期清理
-- ============================================
-- 保留最近 30 天的检索日志
CREATE OR REPLACE FUNCTION cleanup_old_logs(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM retrieval_logs 
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 保留最近 90 天的 API 使用统计
CREATE OR REPLACE FUNCTION cleanup_old_api_stats(retention_days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM api_usage_stats 
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- 13. 权限设置 (根据需要调整)
-- ============================================
-- GRANT SELECT ON document_embeddings TO app_user;
-- GRANT INSERT, UPDATE, DELETE ON document_embeddings TO app_user;
-- GRANT SELECT ON workspaces TO app_user;
-- GRANT SELECT ON retrieval_logs TO app_user;
