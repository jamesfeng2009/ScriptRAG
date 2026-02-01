-- ============================================================================
-- 核心业务表创建脚本
-- 包含用户管理、会话、大纲步骤、剧本片段和检索文档表
-- ============================================================================

SET search_path TO screenplay, public;

-- ============================================================================
-- 1. 用户表
-- ============================================================================
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    last_login_at TIMESTAMP WITH TIME ZONE,
    
    -- 约束
    CONSTRAINT users_role_check CHECK (role IN ('admin', 'user', 'viewer')),
    CONSTRAINT users_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'))
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

-- 注释
COMMENT ON TABLE users IS '用户表，存储用户认证和基本信息';
COMMENT ON COLUMN users.role IS '用户角色：admin, user, viewer';
COMMENT ON COLUMN users.password_hash IS '密码哈希值（bcrypt）';

-- ============================================================================
-- 2. 代码文档表（全局代码库，不再按工作空间隔离）
-- ============================================================================
CREATE TABLE IF NOT EXISTS code_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_path VARCHAR(1000) NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    embedding VECTOR(1536),  -- OpenAI text-embedding-3-large 维度
    file_size INTEGER NOT NULL,
    language VARCHAR(50),
    has_deprecated BOOLEAN DEFAULT FALSE,
    has_fixme BOOLEAN DEFAULT FALSE,
    has_todo BOOLEAN DEFAULT FALSE,
    has_security BOOLEAN DEFAULT FALSE,
    metadata JSONB DEFAULT '{}',
    indexed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 约束
    CONSTRAINT code_documents_unique_path UNIQUE (file_path)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_code_documents_file_path ON code_documents(file_path);
CREATE INDEX IF NOT EXISTS idx_code_documents_language ON code_documents(language);
CREATE INDEX IF NOT EXISTS idx_code_documents_deprecated ON code_documents(has_deprecated) WHERE has_deprecated = TRUE;
CREATE INDEX IF NOT EXISTS idx_code_documents_fixme ON code_documents(has_fixme) WHERE has_fixme = TRUE;
CREATE INDEX IF NOT EXISTS idx_code_documents_todo ON code_documents(has_todo) WHERE has_todo = TRUE;
CREATE INDEX IF NOT EXISTS idx_code_documents_security ON code_documents(has_security) WHERE has_security = TRUE;
CREATE INDEX IF NOT EXISTS idx_code_documents_updated_at ON code_documents(updated_at DESC);

-- 为向量列创建 HNSW 索引（高性能相似度搜索）
CREATE INDEX IF NOT EXISTS idx_code_documents_embedding ON code_documents 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 注释
COMMENT ON TABLE code_documents IS '代码文档表，存储全局代码库的文档和向量';
COMMENT ON COLUMN code_documents.embedding IS '文档的向量嵌入，用于语义搜索';
COMMENT ON COLUMN code_documents.content_hash IS '内容哈希值，用于检测文件变更';

-- ============================================================================
-- 3. 剧本生成会话表
-- ============================================================================
CREATE TABLE IF NOT EXISTS screenplay_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    user_topic TEXT NOT NULL,
    project_context TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    current_step_index INTEGER DEFAULT 0,
    current_skill VARCHAR(100) DEFAULT 'standard_tutorial',
    global_tone VARCHAR(50) DEFAULT 'professional',
    max_retries INTEGER DEFAULT 3,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    error_message TEXT,
    
    -- 约束
    CONSTRAINT sessions_status_check CHECK (status IN ('pending', 'in_progress', 'completed', 'failed', 'cancelled')),
    CONSTRAINT sessions_skill_check CHECK (current_skill IN (
        'standard_tutorial', 'warning_mode', 'visualization_analogy', 
        'research_mode', 'meme_style', 'fallback_summary'
    )),
    CONSTRAINT sessions_tone_check CHECK (global_tone IN ('professional', 'cautionary', 'engaging', 'exploratory', 'casual', 'neutral')),
    CONSTRAINT sessions_step_index_check CHECK (current_step_index >= 0),
    CONSTRAINT sessions_max_retries_check CHECK (max_retries >= 0 AND max_retries <= 10)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_sessions_user ON screenplay_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON screenplay_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON screenplay_sessions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_completed_at ON screenplay_sessions(completed_at DESC) WHERE completed_at IS NOT NULL;

-- 注释
COMMENT ON TABLE screenplay_sessions IS '剧本生成会话表，记录每次生成任务';
COMMENT ON COLUMN screenplay_sessions.status IS '会话状态：pending, in_progress, completed, failed, cancelled';
COMMENT ON COLUMN screenplay_sessions.current_skill IS '当前使用的 Skill 模式';
COMMENT ON COLUMN screenplay_sessions.global_tone IS '全局语调设置';

-- ============================================================================
-- 4. 大纲步骤表
-- ============================================================================
CREATE TABLE IF NOT EXISTS outline_steps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES screenplay_sessions(id) ON DELETE CASCADE,
    step_id INTEGER NOT NULL,
    description TEXT NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    retry_count INTEGER DEFAULT 0,
    skill_used VARCHAR(100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    -- 约束
    CONSTRAINT outline_steps_unique_step_per_session UNIQUE (session_id, step_id),
    CONSTRAINT outline_steps_status_check CHECK (status IN ('pending', 'in_progress', 'completed', 'skipped', 'failed')),
    CONSTRAINT outline_steps_step_id_check CHECK (step_id >= 0),
    CONSTRAINT outline_steps_retry_count_check CHECK (retry_count >= 0)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_outline_steps_session ON outline_steps(session_id);
CREATE INDEX IF NOT EXISTS idx_outline_steps_status ON outline_steps(status);
CREATE INDEX IF NOT EXISTS idx_outline_steps_session_step ON outline_steps(session_id, step_id);

-- 注释
COMMENT ON TABLE outline_steps IS '大纲步骤表，存储剧本生成的各个步骤';
COMMENT ON COLUMN outline_steps.step_id IS '步骤序号（从 0 开始）';
COMMENT ON COLUMN outline_steps.retry_count IS '重试次数计数器';

-- ============================================================================
-- 5. 剧本片段表
-- ============================================================================
CREATE TABLE IF NOT EXISTS screenplay_fragments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES screenplay_sessions(id) ON DELETE CASCADE,
    step_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    skill_used VARCHAR(100) NOT NULL,
    sources JSONB DEFAULT '[]',
    word_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_valid BOOLEAN DEFAULT TRUE,
    validation_notes TEXT,
    
    -- 约束
    CONSTRAINT fragments_step_id_check CHECK (step_id >= 0),
    CONSTRAINT fragments_content_not_empty CHECK (LENGTH(content) > 0)
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_fragments_session ON screenplay_fragments(session_id);
CREATE INDEX IF NOT EXISTS idx_fragments_session_step ON screenplay_fragments(session_id, step_id);
CREATE INDEX IF NOT EXISTS idx_fragments_created_at ON screenplay_fragments(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_fragments_is_valid ON screenplay_fragments(is_valid);

-- 注释
COMMENT ON TABLE screenplay_fragments IS '剧本片段表，存储生成的剧本内容';
COMMENT ON COLUMN screenplay_fragments.sources IS 'JSON 数组，包含引用的源文档';
COMMENT ON COLUMN screenplay_fragments.is_valid IS '事实检查器验证结果';

-- ============================================================================
-- 6. 检索文档表
-- ============================================================================
CREATE TABLE IF NOT EXISTS retrieved_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES screenplay_sessions(id) ON DELETE CASCADE,
    step_id INTEGER NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(500) NOT NULL,
    confidence FLOAT NOT NULL,
    metadata JSONB DEFAULT '{}',
    summary TEXT,
    retrieval_method VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- 约束
    CONSTRAINT retrieved_docs_step_id_check CHECK (step_id >= 0),
    CONSTRAINT retrieved_docs_confidence_check CHECK (confidence >= 0.0 AND confidence <= 1.0),
    CONSTRAINT retrieved_docs_method_check CHECK (retrieval_method IN ('vector', 'keyword', 'hybrid'))
);

-- 索引
CREATE INDEX IF NOT EXISTS idx_retrieved_docs_session ON retrieved_documents(session_id);
CREATE INDEX IF NOT EXISTS idx_retrieved_docs_session_step ON retrieved_documents(session_id, step_id);
CREATE INDEX IF NOT EXISTS idx_retrieved_docs_confidence ON retrieved_documents(confidence DESC);
CREATE INDEX IF NOT EXISTS idx_retrieved_docs_created_at ON retrieved_documents(created_at DESC);

-- 注释
COMMENT ON TABLE retrieved_documents IS '检索文档表，存储 RAG 检索的文档';
COMMENT ON COLUMN retrieved_documents.confidence IS '置信度分数（0.0-1.0）';
COMMENT ON COLUMN retrieved_documents.retrieval_method IS '检索方法：vector, keyword, hybrid';
COMMENT ON COLUMN retrieved_documents.metadata IS 'JSON 格式的文档元数据';

-- ============================================================================
-- 触发器：自动更新 updated_at 字段
-- ============================================================================

-- 创建通用的 updated_at 触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- 为需要的表添加触发器
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_code_documents_updated_at
    BEFORE UPDATE ON code_documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at
    BEFORE UPDATE ON screenplay_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_outline_steps_updated_at
    BEFORE UPDATE ON outline_steps
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- 完成核心业务表创建
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Core business tables created successfully.';
    RAISE NOTICE 'Tables: users, code_documents, screenplay_sessions, outline_steps, screenplay_fragments, retrieved_documents';
END $$;