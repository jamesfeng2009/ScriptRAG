"""Remove unused tables for simplified architecture

Revision ID: 7b9c8d3e2f1a
Revises: 4a7b8c3d2e1f
Create Date: 2026-02-02

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '7b9c8d3e2f1a'
down_revision: Union[str, None] = '4a7b8c3d2e1f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    删除未使用的表，精简架构

    删除的表：
    - audit_logs: 审计日志，未启用
    - code_documents: 代码文档，未使用
    - execution_logs: 执行日志，未启用
    - llm_call_logs: LLM 调用日志，未启用
    - outline_steps: 大纲步骤，已被 Task 替代
    - quota_usage: 配额使用，未启用
    - retrieved_documents: 检索文档，已被 Task 替代
    - screenplay_fragments: 剧本片段，已被 Task 替代
    - screenplay_sessions: 剧本会话，已被 Task 替代
    - users: 用户表，未使用
    - workspaces: 工作空间表，未使用
    """

    # 删除未使用的日志和监控表
    op.execute("""
        DROP TABLE IF EXISTS screenplay.audit_logs CASCADE;
    """)
    op.execute("""
        DROP TABLE IF EXISTS screenplay.code_documents CASCADE;
    """)
    op.execute("""
        DROP TABLE IF EXISTS screenplay.execution_logs CASCADE;
    """)
    op.execute("""
        DROP TABLE IF EXISTS screenplay.llm_call_logs CASCADE;
    """)
    op.execute("""
        DROP TABLE IF EXISTS screenplay.quota_usage CASCADE;
    """)

    # 删除与 screenplay_sessions 相关的表（按依赖顺序）
    op.execute("""
        DROP TABLE IF EXISTS screenplay.retrieved_documents CASCADE;
    """)
    op.execute("""
        DROP TABLE IF EXISTS screenplay.screenplay_fragments CASCADE;
    """)
    op.execute("""
        DROP TABLE IF EXISTS screenplay.outline_steps CASCADE;
    """)
    op.execute("""
        DROP TABLE IF EXISTS screenplay.screenplay_sessions CASCADE;
    """)

    # 删除未使用的用户和工作空间表
    op.execute("""
        DROP TABLE IF EXISTS screenplay.users CASCADE;
    """)
    op.execute("""
        DROP TABLE IF EXISTS screenplay.workspaces CASCADE;
    """)

    # 删除不再需要的视图
    op.execute("""
        DROP VIEW IF EXISTS screenplay.quota_usage_summary CASCADE;
    """)


def downgrade() -> None:
    """
    恢复已删除的表（仅用于开发环境，数据会丢失）
    """

    # 恢复 users 表
    op.execute("""
        CREATE TABLE screenplay.users (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            username VARCHAR(255) UNIQUE NOT NULL,
            email VARCHAR(255) UNIQUE,
            full_name VARCHAR(255),
            hashed_password VARCHAR(255),
            is_active BOOLEAN DEFAULT TRUE,
            is_superuser BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # 恢复 workspaces 表
    op.execute("""
        CREATE TABLE screenplay.workspaces (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            name VARCHAR(255) UNIQUE NOT NULL,
            description TEXT,
            settings JSONB DEFAULT '{}',
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # 恢复 screenplay_sessions 表
    op.execute("""
        CREATE TABLE screenplay.screenplay_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            topic TEXT NOT NULL,
            context TEXT DEFAULT '',
            current_skill VARCHAR(100) DEFAULT 'standard_tutorial',
            global_tone VARCHAR(100) DEFAULT 'professional',
            max_retries INTEGER DEFAULT 3,
            status VARCHAR(50) DEFAULT 'pending',
            pivot_triggered BOOLEAN DEFAULT FALSE,
            pivot_reason TEXT,
            fact_check_passed BOOLEAN DEFAULT TRUE,
            awaiting_user_input BOOLEAN DEFAULT FALSE,
            user_input_prompt TEXT,
            skill_history JSONB DEFAULT '[]',
            execution_log JSONB DEFAULT '[]',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # 恢复 outline_steps 表
    op.execute("""
        CREATE TABLE screenplay.outline_steps (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES screenplay.screenplay_sessions(id) ON DELETE CASCADE,
            step_id INTEGER NOT NULL,
            description TEXT NOT NULL,
            status VARCHAR(50) DEFAULT 'pending',
            retry_count INTEGER DEFAULT 0,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # 恢复 screenplay_fragments 表
    op.execute("""
        CREATE TABLE screenplay.screenplay_fragments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES screenplay.screenplay_sessions(id) ON DELETE CASCADE,
            step_id UUID REFERENCES screenplay.outline_steps(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            skill_used VARCHAR(100) NOT NULL,
            sources JSONB DEFAULT '[]',
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    # 恢复 retrieved_documents 表
    op.execute("""
        CREATE TABLE screenplay.retrieved_documents (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES screenplay.screenplay_sessions(id) ON DELETE CASCADE,
            content TEXT NOT NULL,
            source VARCHAR(500) NOT NULL,
            confidence FLOAT NOT NULL,
            summary TEXT,
            metadata JSONB DEFAULT '{}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
    """)

    print("WARNING: Downgrade completed but all tables are empty. Data has been lost.")
