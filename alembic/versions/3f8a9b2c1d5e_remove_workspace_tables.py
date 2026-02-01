"""Remove workspace tables and references

Revision ID: 3f8a9b2c1d5e
Revises: 285c3c88f28a
Create Date: 2026-02-01 16:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '3f8a9b2c1d5e'
down_revision: Union[str, None] = '285c3c88f28a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    移除 workspace 相关的表和字段
    
    变更内容：
    1. 删除依赖 workspace_id 的视图
    2. 从 screenplay_sessions 表移除 workspace_id 字段
    3. 从 code_documents 表移除 workspace_id 字段
    4. 删除 workspaces 表
    5. 为 code_documents.file_path 添加唯一约束
    """
    
    # 0. 删除依赖 workspace_id 的视图（如果存在）
    op.execute("DROP VIEW IF EXISTS screenplay.vector_db_metrics CASCADE")
    
    # 1. 移除 screenplay_sessions 表的 workspace_id 外键和字段
    op.drop_constraint('screenplay_sessions_workspace_id_fkey', 'screenplay_sessions', schema='screenplay', type_='foreignkey')
    op.drop_index('idx_sessions_workspace', table_name='screenplay_sessions', schema='screenplay')
    op.drop_column('screenplay_sessions', 'workspace_id', schema='screenplay')
    
    # 2. 移除 code_documents 表的 workspace_id 外键和字段
    op.drop_constraint('code_documents_workspace_id_fkey', 'code_documents', schema='screenplay', type_='foreignkey')
    op.drop_column('code_documents', 'workspace_id', schema='screenplay')
    
    # 3. 为 code_documents.file_path 添加唯一约束（先检查是否存在）
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint 
                WHERE conname = 'code_documents_unique_path' 
                AND connamespace = 'screenplay'::regnamespace
            ) THEN
                ALTER TABLE screenplay.code_documents 
                ADD CONSTRAINT code_documents_unique_path UNIQUE (file_path);
            END IF;
        END $$;
    """)
    
    # 4. 添加新的索引（如果不存在）
    op.execute("CREATE INDEX IF NOT EXISTS idx_code_documents_file_path ON screenplay.code_documents(file_path)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_code_documents_language ON screenplay.code_documents(language)")
    op.execute("CREATE INDEX IF NOT EXISTS idx_code_documents_updated_at ON screenplay.code_documents(updated_at DESC)")
    
    # 5. 删除 workspaces 表（CASCADE 会自动处理依赖）
    op.drop_table('workspaces', schema='screenplay')


def downgrade() -> None:
    """
    恢复 workspace 相关的表和字段
    
    注意：这个降级操作会丢失数据，仅用于开发环境
    """
    
    # 1. 重新创建 workspaces 表
    op.create_table(
        'workspaces',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('settings', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['tenant_id'], ['screenplay.tenants.id'], ),
        sa.PrimaryKeyConstraint('id'),
        schema='screenplay'
    )
    
    # 2. 重新创建索引
    op.create_index('idx_workspaces_tenant', 'workspaces', ['tenant_id'], schema='screenplay')
    op.create_index('idx_workspaces_is_active', 'workspaces', ['is_active'], schema='screenplay')
    op.create_index('idx_workspaces_created_at', 'workspaces', [sa.text('created_at DESC')], schema='screenplay')
    
    # 3. 移除 code_documents 的新索引和约束
    op.drop_index('idx_code_documents_updated_at', table_name='code_documents', schema='screenplay')
    op.drop_index('idx_code_documents_language', table_name='code_documents', schema='screenplay')
    op.drop_index('idx_code_documents_file_path', table_name='code_documents', schema='screenplay')
    op.drop_constraint('code_documents_unique_path', 'code_documents', schema='screenplay', type_='unique')
    
    # 4. 为 code_documents 添加 workspace_id 字段
    # 注意：这里使用一个默认的 UUID，实际使用时需要根据实际情况调整
    op.add_column('code_documents', 
                  sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=True), 
                  schema='screenplay')
    
    # 5. 为 screenplay_sessions 添加 workspace_id 字段
    op.add_column('screenplay_sessions', 
                  sa.Column('workspace_id', postgresql.UUID(as_uuid=True), nullable=True), 
                  schema='screenplay')
    
    # 6. 添加外键约束（注意：降级后需要手动填充 workspace_id 数据）
    # op.create_foreign_key('code_documents_workspace_id_fkey', 'code_documents', 'workspaces', 
    #                       ['workspace_id'], ['id'], source_schema='screenplay', referent_schema='screenplay')
    # op.create_foreign_key('screenplay_sessions_workspace_id_fkey', 'screenplay_sessions', 'workspaces', 
    #                       ['workspace_id'], ['id'], source_schema='screenplay', referent_schema='screenplay')
    
    # 7. 重新创建索引
    op.create_index('idx_sessions_workspace', 'screenplay_sessions', ['workspace_id'], schema='screenplay')
    
    print("WARNING: Downgrade completed, but workspace_id fields are NULL. Manual data migration required.")
