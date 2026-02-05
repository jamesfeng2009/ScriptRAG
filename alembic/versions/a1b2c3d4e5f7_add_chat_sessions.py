"""Add chat_sessions table

Revision ID: a1b2c3d4e5f7
Revises: e1f2a3b4c5d6
Create Date: 2026-02-05

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f7'
down_revision: Union[str, None] = 'e1f2a3b4c5d6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create chat_sessions table
    op.create_table(
        'chat_sessions',
        sa.Column('id', sa.String(50), nullable=False, primary_key=True),
        sa.Column('topic', sa.String(500), nullable=True),
        sa.Column('mode', sa.String(20), nullable=False),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('message_history', sa.JSON(), nullable=True),
        sa.Column('related_task_id', sa.String(50), nullable=True),
        sa.Column('status', sa.String(20), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        schema='screenplay'
    )
    
    # Create indexes
    op.create_index('idx_chat_sessions_related_task', 'chat_sessions', ['related_task_id'], schema='screenplay')
    op.create_index('idx_chat_sessions_status', 'chat_sessions', ['status'], schema='screenplay')
    op.create_index('idx_chat_sessions_created', 'chat_sessions', ['created_at'], schema='screenplay')


def downgrade() -> None:
    op.drop_index('idx_chat_sessions_created', table_name='chat_sessions', schema='screenplay')
    op.drop_index('idx_chat_sessions_status', table_name='chat_sessions', schema='screenplay')
    op.drop_index('idx_chat_sessions_related_task', table_name='chat_sessions', schema='screenplay')
    op.drop_table('chat_sessions', schema='screenplay')
