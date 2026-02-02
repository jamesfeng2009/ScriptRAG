"""Remove duplicate alembic_version table from screenplay schema

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-02-02 14:30:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c3d4e5f6a7b8'
down_revision: Union[str, None] = 'b2c3d4e5f6a7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """删除 screenplay schema 中重复的 alembic_version 表
    
    Alembic 的版本追踪表应该只在 public schema 中存在。
    screenplay.alembic_version 是之前意外创建的重复表，应该删除以避免混淆。
    """
    op.execute("DROP TABLE IF EXISTS screenplay.alembic_version CASCADE")


def downgrade() -> None:
    """恢复 alembic_version 表（仅用于回滚时的记录）
    
    注意：这个表不应该被重新创建，因为 Alembic 应该只使用 public schema。
    """
    pass
