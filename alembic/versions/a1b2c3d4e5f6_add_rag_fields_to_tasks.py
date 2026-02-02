"""Add RAG-related fields to tasks table

Revision ID: a1b2c3d4e5f6
Revises: 7b9c8d3e2f1a
Create Date: 2026-02-02

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '7b9c8d3e2f1a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add RAG-related columns to tasks table"""
    op.add_column('tasks', sa.Column('rag_analysis', sa.JSON(), nullable=True))
    op.add_column('tasks', sa.Column('rag_top_k', sa.Integer(), nullable=True))
    op.add_column('tasks', sa.Column('rag_similarity_threshold', sa.Float(), nullable=True))
    op.add_column('tasks', sa.Column('rag_enable_hybrid_search', sa.Boolean(), nullable=True))
    op.add_column('tasks', sa.Column('rag_enable_reranking', sa.Boolean(), nullable=True))


def downgrade() -> None:
    """Remove RAG-related columns from tasks table"""
    op.drop_column('tasks', 'rag_enable_reranking')
    op.drop_column('tasks', 'rag_enable_hybrid_search')
    op.drop_column('tasks', 'rag_similarity_threshold')
    op.drop_column('tasks', 'rag_top_k')
    op.drop_column('tasks', 'rag_analysis')
