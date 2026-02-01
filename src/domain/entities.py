"""
数据库实体类（领域层）

本模块定义了与数据库表对应的实体类，用于持久化操作。
使用 SQLAlchemy 进行 ORM 映射。
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import UUID, uuid4
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

# 定义 VECTOR 类型（pgvector 扩展）
from sqlalchemy import TypeDecorator, String as SQLString

class VECTOR(TypeDecorator):
    """PostgreSQL pgvector VECTOR 类型"""
    impl = SQLString
    
    def __init__(self, dimensions=None):
        self.dimensions = dimensions
        super().__init__()
    
    def load_dialect_impl(self, dialect):
        return dialect.type_descriptor(SQLString())
    
    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, list):
            return f"[{','.join(map(str, value))}]"
        return str(value)
    
    def process_result_value(self, value, dialect):
        if value is None:
            return value
        # 简单解析向量字符串
        if value.startswith('[') and value.endswith(']'):
            return [float(x) for x in value[1:-1].split(',')]
        return value

Base = declarative_base()


class User(Base):
    """用户实体"""
    __tablename__ = 'users'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    username = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    full_name = Column(String(255))
    preferences = Column(JSON, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class Workspace(Base):
    """工作空间实体"""
    __tablename__ = 'workspaces'
    __table_args__ = {'schema': 'screenplay'}

    workspace_id = Column(String(100), primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    settings = Column(JSON, default={})
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


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


class ScreenplaySession(Base):
    """剧本生成会话实体"""
    __tablename__ = 'screenplay_sessions'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey('screenplay.users.id'), nullable=False)
    topic = Column(Text, nullable=False)
    context = Column(Text, default='')
    current_skill = Column(String(100), default='standard_tutorial')
    global_tone = Column(String(100), default='professional')
    max_retries = Column(Integer, default=3)
    status = Column(String(50), default='pending')
    pivot_triggered = Column(Boolean, default=False)
    pivot_reason = Column(Text)
    fact_check_passed = Column(Boolean, default=True)
    awaiting_user_input = Column(Boolean, default=False)
    user_input_prompt = Column(Text)
    skill_history = Column(JSON, default=[])
    execution_log = Column(JSON, default=[])
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 关系
    user = relationship("User", back_populates="screenplay_sessions")
    outline_steps = relationship("OutlineStep", back_populates="session", cascade="all, delete-orphan")
    screenplay_fragments = relationship("ScreenplayFragment", back_populates="session", cascade="all, delete-orphan")
    retrieved_documents = relationship("RetrievedDocument", back_populates="session", cascade="all, delete-orphan")


class OutlineStep(Base):
    """大纲步骤实体"""
    __tablename__ = 'outline_steps'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey('screenplay.screenplay_sessions.id'), nullable=False)
    step_id = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)
    status = Column(String(50), default='pending')
    retry_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 关系
    session = relationship("ScreenplaySession", back_populates="outline_steps")
    screenplay_fragments = relationship("ScreenplayFragment", back_populates="outline_step")


class ScreenplayFragment(Base):
    """剧本片段实体"""
    __tablename__ = 'screenplay_fragments'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey('screenplay.screenplay_sessions.id'), nullable=False)
    step_id = Column(PG_UUID(as_uuid=True), ForeignKey('screenplay.outline_steps.id'), nullable=False)
    content = Column(Text, nullable=False)
    skill_used = Column(String(100), nullable=False)
    sources = Column(JSON, default=[])
    doc_metadata = Column(JSON, default={})  # 重命名避免冲突
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    
    # 关系
    session = relationship("ScreenplaySession", back_populates="screenplay_fragments")
    outline_step = relationship("OutlineStep", back_populates="screenplay_fragments")


class RetrievedDocument(Base):
    """检索文档实体"""
    __tablename__ = 'retrieved_documents'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), ForeignKey('screenplay.screenplay_sessions.id'), nullable=False)
    content = Column(Text, nullable=False)
    source = Column(String(500), nullable=False)
    confidence = Column(Float, nullable=False)
    summary = Column(Text)
    doc_metadata = Column(JSON, default={})  # 重命名避免冲突
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # 关系
    session = relationship("ScreenplaySession", back_populates="retrieved_documents")


class CodeDocument(Base):
    """代码文档实体（向量存储）"""
    __tablename__ = 'code_documents'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    file_path = Column(String(1000), nullable=False, unique=True)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=False)
    embedding = Column(VECTOR(1536))  # OpenAI text-embedding-3-large 维度
    file_size = Column(Integer, nullable=False)
    language = Column(String(50))
    has_deprecated = Column(Boolean, default=False)
    has_fixme = Column(Boolean, default=False)
    has_todo = Column(Boolean, default=False)
    has_security = Column(Boolean, default=False)
    doc_metadata = Column(JSON, default={})  # 重命名避免与 SQLAlchemy metadata 冲突
    indexed_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ExecutionLog(Base):
    """执行日志实体"""
    __tablename__ = 'execution_logs'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), nullable=True)  # 可选，某些日志可能不关联会话
    agent_name = Column(String(100), nullable=False)
    action = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)
    duration_ms = Column(Integer)
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    exec_metadata = Column(JSON, default={})  # 重命名避免冲突
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class LLMCallLog(Base):
    """LLM 调用日志实体"""
    __tablename__ = 'llm_call_logs'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PG_UUID(as_uuid=True), nullable=True)
    provider = Column(String(50), nullable=False)
    model = Column(String(100), nullable=False)
    prompt_tokens = Column(Integer)
    completion_tokens = Column(Integer)
    total_tokens = Column(Integer)
    response_time_ms = Column(Integer)
    cost_usd = Column(Float)
    status = Column(String(50), nullable=False)
    error_message = Column(Text)
    call_metadata = Column(JSON, default={})  # 重命名避免冲突
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class AuditLog(Base):
    """审计日志实体"""
    __tablename__ = 'audit_logs'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), nullable=True)
    action = Column(String(100), nullable=False)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(String(255))
    old_values = Column(JSON)
    new_values = Column(JSON)
    ip_address = Column(String(45))
    user_agent = Column(Text)
    audit_metadata = Column(JSON, default={})  # 重命名避免冲突
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class QuotaUsage(Base):
    """配额使用实体（时序数据）"""
    __tablename__ = 'quota_usage'
    __table_args__ = {'schema': 'screenplay'}
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=uuid4)
    user_id = Column(PG_UUID(as_uuid=True), nullable=True)
    resource_type = Column(String(100), nullable=False)
    usage_amount = Column(Float, nullable=False)
    quota_limit = Column(Float)
    period_start = Column(DateTime(timezone=True), nullable=False)
    period_end = Column(DateTime(timezone=True), nullable=False)
    usage_metadata = Column(JSON, default={})
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())


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