"""Add chat_session_id to tasks table

Revision ID: a1b2c3d4e5f8
Revises: a1b2c3d4e5f7
Create Date: 2026-02-05

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f8'
down_revision: Union[str, None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add chat_session_id column to tasks table"""
    op.add_column('tasks', sa.Column('chat_session_id', sa.String(50), nullable=True))
    
    op.create_index('idx_tasks_chat_session', 'tasks', ['chat_session_id'])


def downgrade() -> None:
    """Remove chat_session_id column from tasks table"""
    op.drop_index('idx_tasks_chat_session', table_name='tasks')
    op.drop_column('tasks', 'chat_session_id')
