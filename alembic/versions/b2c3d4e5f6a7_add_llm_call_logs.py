"""Add llm_call_logs table for tracking LLM calls

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-02-02

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create llm_call_logs table for tracking LLM API calls"""
    op.create_table(
        'llm_call_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('task_id', sa.String(36), nullable=True),
        sa.Column('provider', sa.String(50), nullable=False),
        sa.Column('model', sa.String(100), nullable=False),
        sa.Column('request_type', sa.String(50), nullable=False),
        sa.Column('input_tokens', sa.Integer(), nullable=True),
        sa.Column('output_tokens', sa.Integer(), nullable=True),
        sa.Column('total_tokens', sa.Integer(), nullable=True),
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='success'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('error_code', sa.String(50), nullable=True),
        sa.Column('cost_estimate', sa.Float(), nullable=True),
        sa.Column('request_preview', sa.Text(), nullable=True),
        sa.Column('response_preview', sa.Text(), nullable=True),
        sa.Column('extra_data', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=True),
        sa.PrimaryKeyConstraint('id', name='pk_llm_call_logs'),
        schema='screenplay'
    )
    
    op.create_index('ix_llm_call_logs_task_id', 'llm_call_logs', ['task_id'], schema='screenplay')
    op.create_index('ix_llm_call_logs_provider', 'llm_call_logs', ['provider'], schema='screenplay')
    op.create_index('ix_llm_call_logs_created_at', 'llm_call_logs', ['created_at'], schema='screenplay')


def downgrade() -> None:
    """Drop llm_call_logs table"""
    op.drop_index('ix_llm_call_logs_created_at', table_name='llm_call_logs', schema='screenplay')
    op.drop_index('ix_llm_call_logs_provider', table_name='llm_call_logs', schema='screenplay')
    op.drop_index('ix_llm_call_logs_task_id', table_name='llm_call_logs', schema='screenplay')
    op.drop_table('llm_call_logs', schema='screenplay')
