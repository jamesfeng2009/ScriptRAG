-- ============================================================================
-- 向量存储表创建脚本
-- 使用 pgvector 扩展进行向量存储和相似度搜索
-- ============================================================================

SET search_path TO screenplay, public;

-- ============================================================================
-- 代码文档表（包含向量嵌入）
-- ============================================================================
CREATE TABLE IF NOT EXISTS code_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    workspace_id UUID NOT NULL REFERENCES workspaces(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI text-embedding-3-large 维度
    language VARCHAR(50),
    
    -- 标量字段：用于混合搜索的敏感标记
    has_deprecated BOOLEAN DEFAULT FALSE,
    has_fixme BOOLEAN DEFAULT FALSE,
    has_todo BOOLEAN DEFAULT FALSE,
    has_security BOOLEAN DEFAULT FALSE,
    
    -- 元数据
    metadata JSONB DEFAULT '{}',
    file_size INTEGER,
    line_count INTEGER,
    
    -- 时间戳
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    indexed_at TIMESTAMP WITH TIME ZONE,
    
    -- 约束
    CONSTRAINT code_docs_unique_file_per_workspace UNIQUE (workspace_id, file_path),
    CONSTRAINT code_docs_content_not_empty CHECK (LENGTH(content) > 0),
    CONSTRAINT code_docs_file_size_check CHECK (file_size >= 0),
    CONSTRAINT code_docs_line_count_check CHECK (line_count >= 0)
);

-- ============================================================================
-- 向量索引：HNSW 算法用于高性能相似度搜索
-- ============================================================================

-- HNSW 索引配置说明：
-- - m: 每个节点的最大连接数（默认 16，范围 2-100）
--   较大的 m 值提高召回率但增加内存使用和构建时间
-- - ef_construction: 构建时的动态候选列表大小（默认 64，范围 4-1000）
--   较大的值提高索引质量但增加构建时间

CREATE INDEX IF NOT EXISTS idx_code_documents_embedding_hnsw 
ON code_documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 注释：可以根据数据规模调整参数
-- 小规模（< 10万条）：m = 16, ef_construction = 64
-- 中规模（10万-100万条）：m = 24, ef_construction = 128
-- 大规模（> 100万条）：考虑迁移到 Milvus

-- ============================================================================
-- 标量字段索引：支持混合搜索
-- ============================================================================

-- 工作空间索引（最常用的过滤条件）
CREATE INDEX IF NOT EXISTS idx_code_documents_workspace 
ON code_documents(workspace_id);

-- 部分索引：仅索引标记为 TRUE 的行（节省空间）
CREATE INDEX IF NOT EXISTS idx_code_documents_deprecated 
ON code_documents(has_deprecated) 
WHERE has_deprecated = TRUE;

CREATE INDEX IF NOT EXISTS idx_code_documents_fixme 
ON code_documents(has_fixme) 
WHERE has_fixme = TRUE;

CREATE INDEX IF NOT EXISTS idx_code_documents_todo 
ON code_documents(has_todo) 
WHERE has_todo = TRUE;

CREATE INDEX IF NOT EXISTS idx_code_documents_security 
ON code_documents(has_security) 
WHERE has_security = TRUE;

-- 语言索引（用于按编程语言过滤）
CREATE INDEX IF NOT EXISTS idx_code_documents_language 
ON code_documents(language);

-- 时间戳索引
CREATE INDEX IF NOT EXISTS idx_code_documents_created_at 
ON code_documents(created_at DESC);

CREATE INDEX IF NOT EXISTS idx_code_documents_updated_at 
ON code_documents(updated_at DESC);

-- 复合索引：工作空间 + 语言（常见查询模式）
CREATE INDEX IF NOT EXISTS idx_code_documents_workspace_language 
ON code_documents(workspace_id, language);

-- GIN 索引：用于 JSONB 元数据搜索
CREATE INDEX IF NOT EXISTS idx_code_documents_metadata 
ON code_documents USING gin(metadata);

-- ============================================================================
-- 全文搜索索引（可选）
-- ============================================================================

-- 为内容创建全文搜索索引（用于关键词搜索）
-- 使用 GIN 索引提高搜索性能
CREATE INDEX IF NOT EXISTS idx_code_documents_content_fts 
ON code_documents 
USING gin(to_tsvector('english', content));

-- 为文件路径创建全文搜索索引
CREATE INDEX IF NOT EXISTS idx_code_documents_path_fts 
ON code_documents 
USING gin(to_tsvector('simple', file_path));

-- ============================================================================
-- 触发器：自动更新 updated_at
-- ============================================================================

CREATE TRIGGER update_code_documents_updated_at
    BEFORE UPDATE ON code_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 表注释
-- ============================================================================

COMMENT ON TABLE code_documents IS '代码文档表，存储代码文件及其向量嵌入';
COMMENT ON COLUMN code_documents.embedding IS '1536 维向量嵌入（OpenAI text-embedding-3-large）';
COMMENT ON COLUMN code_documents.has_deprecated IS '是否包含 @deprecated 标记';
COMMENT ON COLUMN code_documents.has_fixme IS '是否包含 FIXME 标记';
COMMENT ON COLUMN code_documents.has_todo IS '是否包含 TODO 标记';
COMMENT ON COLUMN code_documents.has_security IS '是否包含 Security 相关标记';
COMMENT ON COLUMN code_documents.metadata IS 'JSON 格式的文档元数据（函数、类、注释等）';
COMMENT ON COLUMN code_documents.indexed_at IS '向量索引时间戳';

-- ============================================================================
-- 性能优化建议
-- ============================================================================

-- 1. 定期 VACUUM 和 ANALYZE
-- VACUUM ANALYZE code_documents;

-- 2. 监控索引使用情况
-- SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read, idx_tup_fetch
-- FROM pg_stat_user_indexes
-- WHERE tablename = 'code_documents'
-- ORDER BY idx_scan DESC;

-- 3. 监控表大小
-- SELECT pg_size_pretty(pg_total_relation_size('code_documents')) AS total_size,
--        pg_size_pretty(pg_relation_size('code_documents')) AS table_size,
--        pg_size_pretty(pg_indexes_size('code_documents')) AS indexes_size;

-- 4. 向量搜索性能调优
-- 可以在查询时设置 ef_search 参数（默认 40）
-- SET hnsw.ef_search = 100;  -- 提高召回率但降低速度

-- ============================================================================
-- 迁移到 Milvus 的触发条件监控
-- ============================================================================

-- 创建视图监控向量数据规模
CREATE OR REPLACE VIEW vector_db_metrics AS
SELECT 
    COUNT(*) AS total_vectors,
    pg_size_pretty(pg_total_relation_size('code_documents')) AS total_size,
    AVG(CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END) AS embedding_coverage,
    COUNT(DISTINCT workspace_id) AS workspace_count
FROM code_documents;

COMMENT ON VIEW vector_db_metrics IS '向量数据库指标，用于监控是否需要迁移到 Milvus';

-- 迁移建议：
-- 当满足以下任一条件时，考虑迁移到 Milvus：
-- 1. total_vectors > 1,000,000（向量数量超过 100 万）
-- 2. 向量搜索 QPS > 100
-- 3. P99 延迟 > 500ms
-- 4. total_size > 100GB

-- ============================================================================
-- 完成向量存储表创建
-- ============================================================================

DO $$
DECLARE
    vector_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO vector_count FROM code_documents;
    
    RAISE NOTICE 'Vector storage tables created successfully.';
    RAISE NOTICE 'Table: code_documents with pgvector support';
    RAISE NOTICE 'HNSW index created with m=16, ef_construction=64';
    RAISE NOTICE 'Current vector count: %', vector_count;
    
    IF vector_count > 1000000 THEN
        RAISE WARNING 'Vector count exceeds 1 million. Consider migrating to Milvus for better performance.';
    END IF;
END $$;
