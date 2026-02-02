"""
数据库实体类（领域层）

本模块定义了与数据库表对应的实体类，用于持久化操作。
使用 SQLAlchemy 进行 ORM 映射。

保留的表：
- tasks: 剧本生成任务记录
- documents: RAG 知识库文档
- workspace_skills: 技能配置
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()


class WorkspaceSkill(Base):
    """工作空间技能配置实体"""
    __tablename__ = 'workspace_skills'
    __table_args__ = {'schema': 'screenplay'}

    id = Column(Integer, primary_key=True, autoincrement=True)
    workspace_id = Column(String(100), nullable=False, index=True)
    skill_name = Column(String(100), nullable=False)
    description = Column(Text, nullable=False)
    tone = Column(String(50), nullable=False)
    compatible_with = Column(JSON, default=[])
    prompt_config = Column(JSON, default={})
    is_enabled = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    extra_data = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        UniqueConstraint('workspace_id', 'skill_name', name='uq_workspace_skill'),
        {'schema': 'screenplay'}
    )


class Document(Base):
    """RAG文档实体（用于剧本生成的背景知识库）"""
    __tablename__ = 'documents'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    title = Column(String(500), nullable=False)
    file_name = Column(String(500), nullable=False)
    file_path = Column(String(1000), nullable=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    category = Column(String(100), nullable=True)
    language = Column(String(50), nullable=True)
    file_size = Column(Integer, nullable=False)
    doc_metadata = Column(JSON, default={})
    indexed_at = Column(DateTime(timezone=True), server_default=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Task(Base):
    """Task 任务实体"""
    __tablename__ = 'tasks'
    __table_args__ = {'schema': 'screenplay'}
    
    task_id = Column(String(36), primary_key=True)
    status = Column(String(50), nullable=False, default='pending')
    topic = Column(Text, nullable=False)
    context = Column(Text, default='')
    current_skill = Column(String(100), default='standard_tutorial')
    screenplay = Column(Text)
    outline = Column(JSON, default=[])
    skill_history = Column(JSON, default=[])
    direction_changes = Column(JSON, default=[])
    error = Column(Text)
    request_data = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
