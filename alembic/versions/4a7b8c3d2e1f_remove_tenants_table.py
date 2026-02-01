"""Remove tenants table and simplify users table

Revision ID: 4a7b8c3d2e1f
Revises: 5f8b9c0d1e2a
Create Date: 2026-02-01 17:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4a7b8c3d2e1f'
down_revision: Union[str, None] = '5f8b9c0d1e2a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    移除 tenants 表，简化 users 表
    
    变更内容：
    1. 从 users 表移除 tenant_id 字段
    2. 删除 tenants 表
    3. 从 quota_usage 表移除 tenant_id 字段
    4. 从 audit_logs 表移除 tenant_id 字段
    """
    
    # 0. 删除依赖 tenant_id 的视图（如果存在）
    op.execute("""
        DROP VIEW IF EXISTS screenplay.quota_usage_summary CASCADE;
    """)
    
    # 1. 移除 users 表的 tenant_id 外键和字段
    op.drop_constraint('users_tenant_id_fkey', 'users', schema='screenplay', type_='foreignkey')
    op.drop_index('idx_users_tenant', table_name='users', schema='screenplay')
    op.drop_column('users', 'tenant_id', schema='screenplay')
    
    # 2. 移除 quota_usage 表的 tenant_id 字段（如果存在）
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = 'screenplay' 
                AND table_name = 'quota_usage' 
                AND column_name = 'tenant_id'
            ) THEN
                ALTER TABLE screenplay.quota_usage DROP COLUMN tenant_id CASCADE;
            END IF;
        END $$;
    """)
    
    # 3. 移除 audit_logs 表的 tenant_id 字段（如果存在）
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM information_schema.columns 
                WHERE table_schema = 'screenplay' 
                AND table_name = 'audit_logs' 
                AND column_name = 'tenant_id'
            ) THEN
                ALTER TABLE screenplay.audit_logs DROP COLUMN tenant_id CASCADE;
            END IF;
        END $$;
    """)
    
    # 4. 删除 tenants 表
    op.drop_table('tenants', schema='screenplay')


def downgrade() -> None:
    """
    恢复 tenants 表和相关字段
    
    注意：这个降级操作会丢失数据，仅用于开发环境
    """
    
    # 1. 重新创建 tenants 表
    op.create_table(
        'tenants',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('settings', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        schema='screenplay'
    )
    
    # 2. 为 users 添加 tenant_id 字段
    op.add_column('users', 
                  sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True), 
                  schema='screenplay')
    
    # 3. 为 quota_usage 添加 tenant_id 字段
    op.add_column('quota_usage', 
                  sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True), 
                  schema='screenplay')
    
    # 4. 为 audit_logs 添加 tenant_id 字段
    op.add_column('audit_logs', 
                  sa.Column('tenant_id', postgresql.UUID(as_uuid=True), nullable=True), 
                  schema='screenplay')
    
    print("WARNING: Downgrade completed, but tenant_id fields are NULL. Manual data migration required.")
