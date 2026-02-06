"""Add agent_executions table for tracking agent executions

Revision ID: f1a2b3c4d5e6
Revises: e1f2a3b4c5d6
Create Date: 2026-02-06

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'a1b2c3d4e5f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create agent_executions table for tracking each agent execution"""
    op.create_table(
        'agent_executions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('execution_id', sa.String(36), nullable=False),
        sa.Column('task_id', sa.String(36), nullable=True),
        sa.Column('chat_session_id', sa.String(50), nullable=True),
        sa.Column('agent_name', sa.String(50), nullable=False),
        sa.Column('node_name', sa.String(50), nullable=False),
        sa.Column('step_id', sa.String(50), nullable=True),
        sa.Column('step_index', sa.Integer(), nullable=True),
        sa.Column('action', sa.String(100), nullable=True),
        sa.Column('input_data', sa.JSON(), nullable=True),
        sa.Column('output_data', sa.JSON(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='success'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('execution_time_ms', sa.Float(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('id', name='pk_agent_executions'),
        schema='screenplay'
    )
    
    op.create_index('ix_agent_executions_execution_id', 'agent_executions', ['execution_id'], schema='screenplay')
    op.create_index('ix_agent_executions_task_id', 'agent_executions', ['task_id'], schema='screenplay')
    op.create_index('ix_agent_executions_chat_session_id', 'agent_executions', ['chat_session_id'], schema='screenplay')
    op.create_index('ix_agent_executions_agent_name', 'agent_executions', ['agent_name'], schema='screenplay')
    op.create_index('ix_agent_executions_created_at', 'agent_executions', ['created_at'], schema='screenplay')


def downgrade() -> None:
    """Drop agent_executions table"""
    op.drop_index('ix_agent_executions_created_at', table_name='agent_executions', schema='screenplay')
    op.drop_index('ix_agent_executions_agent_name', table_name='agent_executions', schema='screenplay')
    op.drop_index('ix_agent_executions_chat_session_id', table_name='agent_executions', schema='screenplay')
    op.drop_index('ix_agent_executions_task_id', table_name='agent_executions', schema='screenplay')
    op.drop_index('ix_agent_executions_execution_id', table_name='agent_executions', schema='screenplay')
    op.drop_table('agent_executions', schema='screenplay')
