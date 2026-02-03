"""Rename workspace_skills to skills table

Revision ID: e1f2a3b4c5d6
Revises: d4e5f6a7b8c9
Create Date: 2025-02-03 12:00:00.000000

"""
from typing import Optional
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e1f2a3b4c5d6'
down_revision: str = 'd4e5f6a7b8c9'
branch_labels: Optional[str] = None
depends_on: Optional[str] = None


def upgrade() -> None:
    """Rename workspace_skills table to skills and remove workspace_id column."""
    # Rename table
    op.rename_table('screenplay.workspace_skills', 'screenplay.skills')
    
    # Drop the unique constraint on (workspace_id, skill_name) - now just skill_name
    op.drop_constraint('workspace_skills_workspace_id_skill_name_key', 'screenplay.skills', type_='unique')
    
    # Drop index on workspace_id
    op.drop_index('idx_workspace_skills_workspace_id', table_name='screenplay.skills')
    
    # Drop workspace_id column
    op.drop_column('workspace_id', 'screenplay.skills')
    
    # Create index on skill_name
    op.create_index('idx_skills_skill_name', 'screenplay.skills', ['skill_name'], unique=True)


def downgrade() -> None:
    """Revert skills table back to workspace_skills with workspace_id."""
    # Drop the unique constraint
    op.drop_constraint('skills_skill_name_key', 'screenplay.skills', type_='unique')
    
    # Drop index on skill_name
    op.drop_index('idx_skills_skill_name', table_name='screenplay.skills')
    
    # Add workspace_id column
    op.add_column('workspace_id', sa.Column('workspace_id', sa.VARCHAR(length=100), nullable=True))
    
    # Create index on workspace_id
    op.create_index('idx_workspace_skills_workspace_id', 'screenplay.skills', ['workspace_id'])
    
    # Create unique constraint on (workspace_id, skill_name)
    op.create_unique_constraint('workspace_skills_workspace_id_skill_name_key', 'screenplay.skills', ['workspace_id', 'skill_name'])
    
    # Rename table back
    op.rename_table('screenplay.skills', 'screenplay.workspace_skills')
