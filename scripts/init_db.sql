-- ============================================================================
-- PostgreSQL 数据库初始化脚本
-- 用于基于 RAG 的剧本生成多智能体系统
-- ============================================================================

-- 启用必要的扩展
-- pgvector: 用于向量存储和相似度搜索
CREATE EXTENSION IF NOT EXISTS vector;

-- TimescaleDB: 用于时序数据（可选，用于性能指标和配额监控）
-- 注意：需要先安装 TimescaleDB，如果未安装则注释掉此行
-- CREATE EXTENSION IF NOT EXISTS timescaledb;

-- uuid-ossp: 用于生成 UUID
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 创建数据库（如果不存在）
-- 注意：此脚本假设在目标数据库中执行，如需创建新数据库，请在 psql 中单独执行
-- CREATE DATABASE screenplay_db OWNER screenplay_user;

-- 设置数据库参数
-- 这些参数优化向量搜索和并发性能
ALTER DATABASE screenplay_db SET shared_buffers = '4GB';
ALTER DATABASE screenplay_db SET effective_cache_size = '12GB';
ALTER DATABASE screenplay_db SET maintenance_work_mem = '2GB';
ALTER DATABASE screenplay_db SET work_mem = '256MB';
ALTER DATABASE screenplay_db SET max_parallel_workers_per_gather = 4;
ALTER DATABASE screenplay_db SET max_parallel_workers = 8;

-- 创建 schema（可选，用于组织表）
CREATE SCHEMA IF NOT EXISTS screenplay;

-- 设置默认 schema
SET search_path TO screenplay, public;

-- ============================================================================
-- 完成初始化
-- ============================================================================

-- 验证扩展安装
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'pgvector extension is not installed. Please install it first.';
    END IF;
    
    RAISE NOTICE 'Database initialization completed successfully.';
    RAISE NOTICE 'Extensions installed: vector, uuid-ossp';
END $$;
