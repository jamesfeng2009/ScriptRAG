"""添加工作空间和技能表

Revision ID: 5f8b9c0d1e2a
Revises: 3f8a9b2c1d5e
Create Date: 2026-02-01 21:10:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '5f8b9c0d1e2a'
down_revision: Union[str, None] = '3f8a9b2c1d5e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """添加工作空间和技能相关表"""
    
    # 创建 workspaces 表
    op.create_table(
        'workspaces',
        sa.Column('workspace_id', sa.String(100), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('settings', postgresql.JSON, default={}),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        schema='screenplay'
    )
    
    # 创建 workspace_skills 表
    op.create_table(
        'workspace_skills',
        sa.Column('id', sa.Integer, autoincrement=True, primary_key=True),
        sa.Column('workspace_id', sa.String(100), nullable=False, index=True),
        sa.Column('skill_name', sa.String(100), nullable=False),
        sa.Column('description', sa.Text, nullable=False),
        sa.Column('tone', sa.String(50), nullable=False),
        sa.Column('compatible_with', postgresql.JSON, default=[]),
        sa.Column('prompt_config', postgresql.JSON, default={}),
        sa.Column('is_enabled', sa.Boolean, default=True),
        sa.Column('is_default', sa.Boolean, default=False),
        sa.Column('extra_data', postgresql.JSON, default={}),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.UniqueConstraint('workspace_id', 'skill_name', name='uq_workspace_skill'),
        schema='screenplay'
    )
    
    # 创建索引
    op.create_index('idx_workspace_skills_workspace_id', 'workspace_skills', ['workspace_id'], schema='screenplay')


def downgrade() -> None:
    """移除工作空间和技能表"""
    
    # 删除索引
    op.drop_index('idx_workspace_skills_workspace_id', table_name='workspace_skills', schema='screenplay')
    
    # 删除表
    op.drop_table('workspace_skills', schema='screenplay')
    op.drop_table('workspaces', schema='screenplay')
