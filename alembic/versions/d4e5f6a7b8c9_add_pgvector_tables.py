"""Add pgvector tables for RAG system

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-02-02 15:00:00.000000

Add pgvector tables for the RAG system in the screenplay schema:
- document_embeddings: Main table for document vector embeddings
- document_chunks: Table for document chunks (long documents)
- knowledge_nodes: Table for knowledge graph nodes
- knowledge_relations: Table for knowledge graph relations
- retrieval_logs: Table for retrieval history
- api_usage_stats: Table for API usage statistics

Also creates indexes, views, and maintenance functions.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, None] = 'c3d4e5f6a7b8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create pgvector tables and indexes in screenplay schema"""
    
    # 启用 pg_trgm 扩展（用于模糊查询）
    op.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
    
    # ============================================
    # 1. 主表: 文档向量表 (在 screenplay schema)
    # ============================================
    op.execute("""
        CREATE TABLE IF NOT EXISTS screenplay.document_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id VARCHAR(255) NOT NULL,
            file_path TEXT NOT NULL,
            file_hash VARCHAR(64) NOT NULL,
            content TEXT NOT NULL,
            content_summary TEXT,
            embedding vector(1024),
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
            
            CONSTRAINT screenplay_unique_file_per_workspace UNIQUE (workspace_id, file_path)
        )
    """)
    
    # ============================================
    # 2. 索引: 向量相似度搜索 (HNSW)
    # ============================================
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_vector 
        ON screenplay.document_embeddings 
        USING hnsw (embedding vector_cosine_ops)
        WHERE workspace_id IS NOT NULL
    """)
    
    # 复合索引 - 按工作空间和修改时间查询
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_workspace_time 
        ON screenplay.document_embeddings (workspace_id, updated_at DESC)
    """)
    
    # 标记索引 - 快速过滤
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_markers 
        ON screenplay.document_embeddings (workspace_id, has_deprecated, has_fixme, has_todo, has_security)
    """)
    
    # 文件类型索引
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_embeddings_filetype 
        ON screenplay.document_embeddings (workspace_id, file_type)
    """)
    
    # ============================================
    # 3. 表: 检索历史记录
    # ============================================
    op.execute("""
        CREATE TABLE IF NOT EXISTS screenplay.retrieval_logs (
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
        )
    """)
    
    # 检索日志索引
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_retrieval_logs_workspace 
        ON screenplay.retrieval_logs (workspace_id, created_at DESC)
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_retrieval_logs_query 
        ON screenplay.retrieval_logs USING gin (query gin_trgm_ops)
    """)
    
    # ============================================
    # 4. 表: API 使用统计
    # ============================================
    op.execute("""
        CREATE TABLE IF NOT EXISTS screenplay.api_usage_stats (
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
        )
    """)
    
    # API 使用统计索引
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_api_usage_date 
        ON screenplay.api_usage_stats (date, provider, model)
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_api_usage_workspace 
        ON screenplay.api_usage_stats (workspace_id, date)
    """)
    
    # ============================================
    # 5. 表: 文档分块表 (支持长文档)
    # ============================================
    op.execute("""
        CREATE TABLE IF NOT EXISTS screenplay.document_chunks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            document_id UUID NOT NULL REFERENCES screenplay.document_embeddings(id) ON DELETE CASCADE,
            chunk_index INTEGER NOT NULL,
            content TEXT NOT NULL,
            embedding vector(1024),
            start_line INTEGER,
            end_line INTEGER,
            char_count INTEGER,
            token_count INTEGER,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)
    
    # 分块索引
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_document 
        ON screenplay.document_chunks (document_id, chunk_index)
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
        ON screenplay.document_chunks 
        USING hnsw (embedding vector_cosine_ops)
    """)
    
    # ============================================
    # 6. 表: 知识图谱节点 (可选扩展)
    # ============================================
    op.execute("""
        CREATE TABLE IF NOT EXISTS screenplay.knowledge_nodes (
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
        )
    """)
    
    # 关系表
    op.execute("""
        CREATE TABLE IF NOT EXISTS screenplay.knowledge_relations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            workspace_id VARCHAR(255) NOT NULL,
            source_node_id UUID NOT NULL REFERENCES screenplay.knowledge_nodes(id) ON DELETE CASCADE,
            target_node_id UUID NOT NULL REFERENCES screenplay.knowledge_nodes(id) ON DELETE CASCADE,
            relation_type VARCHAR(100) NOT NULL,
            strength FLOAT DEFAULT 1.0,
            properties JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            
            CONSTRAINT screenplay_unique_relation UNIQUE (source_node_id, target_node_id, relation_type)
        )
    """)
    
    # 知识图谱索引
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodes_workspace_type 
        ON screenplay.knowledge_nodes (workspace_id, node_type)
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_nodes_embedding 
        ON screenplay.knowledge_nodes 
        USING hnsw (embedding vector_cosine_ops)
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_relations_source 
        ON screenplay.knowledge_relations (source_node_id)
    """)
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_relations_target 
        ON screenplay.knowledge_relations (target_node_id)
    """)
    
    # ============================================
    # 7. 函数: 更新修改时间
    # ============================================
    op.execute("""
        CREATE OR REPLACE FUNCTION screenplay.update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ language 'plpgsql'
    """)
    
    # 触发器
    op.execute("""
        DROP TRIGGER IF EXISTS update_document_embeddings_updated_at ON screenplay.document_embeddings
    """)
    op.execute("""
        CREATE TRIGGER update_document_embeddings_updated_at 
        BEFORE UPDATE ON screenplay.document_embeddings 
        FOR EACH ROW EXECUTE FUNCTION screenplay.update_updated_at_column()
    """)
    
    op.execute("""
        DROP TRIGGER IF EXISTS update_knowledge_nodes_updated_at ON screenplay.knowledge_nodes
    """)
    op.execute("""
        CREATE TRIGGER update_knowledge_nodes_updated_at 
        BEFORE UPDATE ON screenplay.knowledge_nodes 
        FOR EACH ROW EXECUTE FUNCTION screenplay.update_updated_at_column()
    """)
    
    # ============================================
    # 8. 视图: 每日成本汇总
    # ============================================
    op.execute("""
        CREATE OR REPLACE VIEW screenplay.daily_cost_summary AS
        SELECT 
            date,
            provider,
            model,
            SUM(cost_usd) as total_cost,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            SUM(request_count) as total_requests,
            COUNT(DISTINCT workspace_id) as workspace_count
        FROM screenplay.api_usage_stats
        WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        GROUP BY date, provider, model
        ORDER BY date DESC, total_cost DESC
    """)
    
    # ============================================
    # 9. 视图: 热门查询
    # ============================================
    op.execute("""
        CREATE OR REPLACE VIEW screenplay.popular_queries AS
        SELECT 
            workspace_id,
            LEFT(query, 100) as query_preview,
            COUNT(*) as query_count,
            AVG(result_count) as avg_results,
            SUM((token_usage->>'total_tokens')::integer) as total_tokens,
            SUM(cost_usd) as total_cost
        FROM screenplay.retrieval_logs
        WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'
        GROUP BY workspace_id, query
        ORDER BY query_count DESC
        LIMIT 100
    """)
    
    # ============================================
    # 10. 维护任务: 定期清理函数
    # ============================================
    op.execute("DROP FUNCTION IF EXISTS screenplay.cleanup_old_logs(INTEGER)")
    op.execute("""
        CREATE OR REPLACE FUNCTION screenplay.cleanup_old_logs(retention_days INTEGER DEFAULT 30)
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            DELETE FROM screenplay.retrieval_logs 
            WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
            
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql
    """)
    
    op.execute("DROP FUNCTION IF EXISTS screenplay.cleanup_old_api_stats(INTEGER)")
    op.execute("""
        CREATE OR REPLACE FUNCTION screenplay.cleanup_old_api_stats(retention_days INTEGER DEFAULT 90)
        RETURNS INTEGER AS $$
        DECLARE
            deleted_count INTEGER;
        BEGIN
            DELETE FROM screenplay.api_usage_stats 
            WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
            
            GET DIAGNOSTICS deleted_count = ROW_COUNT;
            RETURN deleted_count;
        END;
        $$ LANGUAGE plpgsql
    """)


def downgrade() -> None:
    """Drop pgvector tables and indexes"""
    
    # 删除视图
    op.execute("DROP VIEW IF EXISTS screenplay.popular_queries CASCADE")
    op.execute("DROP VIEW IF EXISTS screenplay.daily_cost_summary CASCADE")
    
    # 删除函数
    op.execute("DROP FUNCTION IF EXISTS screenplay.cleanup_old_api_stats(INTEGER) CASCADE")
    op.execute("DROP FUNCTION IF EXISTS screenplay.cleanup_old_logs(INTEGER) CASCADE")
    op.execute("DROP FUNCTION IF EXISTS screenplay.update_updated_at_column() CASCADE")
    
    # 删除表（按依赖顺序）
    op.execute("DROP TABLE IF EXISTS screenplay.knowledge_relations CASCADE")
    op.execute("DROP TABLE IF EXISTS screenplay.knowledge_nodes CASCADE")
    op.execute("DROP TABLE IF EXISTS screenplay.document_chunks CASCADE")
    op.execute("DROP TABLE IF EXISTS screenplay.api_usage_stats CASCADE")
    op.execute("DROP TABLE IF EXISTS screenplay.retrieval_logs CASCADE")
    op.execute("DROP TABLE IF EXISTS screenplay.document_embeddings CASCADE")
