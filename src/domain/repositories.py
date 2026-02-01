"""
数据访问层（Repository 模式）

本模块定义了数据访问接口和实现，提供对数据库实体的 CRUD 操作。
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.orm import selectinload

from .entities import (
    Tenant, User, Workspace, ScreenplaySession, OutlineStep,
    ScreenplayFragment, RetrievedDocument, CodeDocument,
    ExecutionLog, LLMCallLog, AuditLog, QuotaUsage
)
from .models import SharedState


class IRepository(ABC):
    """仓储接口基类"""
    
    @abstractmethod
    async def get_by_id(self, id: UUID) -> Optional[Any]:
        """根据 ID 获取实体"""
        pass
    
    @abstractmethod
    async def create(self, entity: Any) -> Any:
        """创建实体"""
        pass
    
    @abstractmethod
    async def update(self, entity: Any) -> Any:
        """更新实体"""
        pass
    
    @abstractmethod
    async def delete(self, id: UUID) -> bool:
        """删除实体"""
        pass


class ScreenplaySessionRepository(IRepository):
    """剧本会话仓储"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, id: UUID) -> Optional[ScreenplaySession]:
        """根据 ID 获取会话（包含关联数据）"""
        stmt = select(ScreenplaySession).options(
            selectinload(ScreenplaySession.outline_steps),
            selectinload(ScreenplaySession.screenplay_fragments),
            selectinload(ScreenplaySession.retrieved_documents),
            selectinload(ScreenplaySession.user),
            selectinload(ScreenplaySession.workspace)
        ).where(ScreenplaySession.id == id)
        
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create(self, entity: ScreenplaySession) -> ScreenplaySession:
        """创建会话"""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: ScreenplaySession) -> ScreenplaySession:
        """更新会话"""
        entity.updated_at = datetime.now()
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: UUID) -> bool:
        """删除会话"""
        stmt = delete(ScreenplaySession).where(ScreenplaySession.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_by_user_and_workspace(
        self, 
        user_id: UUID, 
        workspace_id: UUID,
        limit: int = 10
    ) -> List[ScreenplaySession]:
        """获取用户在特定工作空间的会话列表"""
        stmt = select(ScreenplaySession).where(
            and_(
                ScreenplaySession.user_id == user_id,
                ScreenplaySession.workspace_id == workspace_id
            )
        ).order_by(ScreenplaySession.created_at.desc()).limit(limit)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def create_from_shared_state(
        self,
        shared_state: SharedState,
        user_id: UUID,
        workspace_id: UUID
    ) -> ScreenplaySession:
        """从 SharedState 创建会话"""
        session_entity = ScreenplaySession(
            user_id=user_id,
            workspace_id=workspace_id,
            topic=shared_state.user_topic,
            context=shared_state.project_context,
            current_skill=shared_state.current_skill,
            global_tone=shared_state.global_tone,
            max_retries=shared_state.max_retries,
            pivot_triggered=shared_state.pivot_triggered,
            pivot_reason=shared_state.pivot_reason,
            fact_check_passed=shared_state.fact_check_passed,
            awaiting_user_input=shared_state.awaiting_user_input,
            user_input_prompt=shared_state.user_input_prompt,
            skill_history=shared_state.skill_history,
            execution_log=shared_state.execution_log
        )
        
        # 创建会话
        created_session = await self.create(session_entity)
        
        # 创建大纲步骤
        for step in shared_state.outline:
            outline_step = OutlineStep(
                session_id=created_session.id,
                step_id=step.step_id,
                description=step.description,
                status=step.status,
                retry_count=step.retry_count
            )
            self.session.add(outline_step)
        
        # 创建剧本片段
        for fragment in shared_state.fragments:
            # 找到对应的大纲步骤
            outline_step_stmt = select(OutlineStep).where(
                and_(
                    OutlineStep.session_id == created_session.id,
                    OutlineStep.step_id == fragment.step_id
                )
            )
            outline_step_result = await self.session.execute(outline_step_stmt)
            outline_step = outline_step_result.scalar_one_or_none()
            
            if outline_step:
                screenplay_fragment = ScreenplayFragment(
                    session_id=created_session.id,
                    step_id=outline_step.id,
                    content=fragment.content,
                    skill_used=fragment.skill_used,
                    sources=fragment.sources
                )
                self.session.add(screenplay_fragment)
        
        # 创建检索文档
        for doc in shared_state.retrieved_docs:
            retrieved_doc = RetrievedDocument(
                session_id=created_session.id,
                content=doc.content,
                source=doc.source,
                confidence=doc.confidence,
                summary=doc.summary,
                metadata=doc.metadata
            )
            self.session.add(retrieved_doc)
        
        await self.session.commit()
        return created_session
    
    async def to_shared_state(self, session_id: UUID) -> Optional[SharedState]:
        """将数据库会话转换为 SharedState"""
        session_entity = await self.get_by_id(session_id)
        if not session_entity:
            return None
        
        # 转换大纲步骤
        outline = []
        for step in session_entity.outline_steps:
            outline.append({
                "step_id": step.step_id,
                "description": step.description,
                "status": step.status,
                "retry_count": step.retry_count
            })
        
        # 转换剧本片段
        fragments = []
        for fragment in session_entity.screenplay_fragments:
            fragments.append({
                "step_id": fragment.outline_step.step_id,  # 使用关联的大纲步骤 ID
                "content": fragment.content,
                "skill_used": fragment.skill_used,
                "sources": fragment.sources
            })
        
        # 转换检索文档
        retrieved_docs = []
        for doc in session_entity.retrieved_documents:
            retrieved_docs.append({
                "content": doc.content,
                "source": doc.source,
                "confidence": doc.confidence,
                "summary": doc.summary,
                "metadata": doc.metadata
            })
        
        # 创建 SharedState
        shared_state = SharedState(
            user_topic=session_entity.topic,
            project_context=session_entity.context,
            outline=outline,
            current_step_index=0,  # 需要根据状态计算
            retrieved_docs=retrieved_docs,
            fragments=fragments,
            current_skill=session_entity.current_skill,
            skill_history=session_entity.skill_history,
            global_tone=session_entity.global_tone,
            pivot_triggered=session_entity.pivot_triggered,
            pivot_reason=session_entity.pivot_reason,
            max_retries=session_entity.max_retries,
            fact_check_passed=session_entity.fact_check_passed,
            awaiting_user_input=session_entity.awaiting_user_input,
            user_input_prompt=session_entity.user_input_prompt,
            execution_log=session_entity.execution_log,
            created_at=session_entity.created_at,
            updated_at=session_entity.updated_at
        )
        
        return shared_state


class CodeDocumentRepository(IRepository):
    """代码文档仓储"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, id: UUID) -> Optional[CodeDocument]:
        """根据 ID 获取代码文档"""
        stmt = select(CodeDocument).where(CodeDocument.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create(self, entity: CodeDocument) -> CodeDocument:
        """创建代码文档"""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: CodeDocument) -> CodeDocument:
        """更新代码文档"""
        entity.updated_at = datetime.now()
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: UUID) -> bool:
        """删除代码文档"""
        stmt = delete(CodeDocument).where(CodeDocument.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_by_workspace(self, workspace_id: UUID) -> List[CodeDocument]:
        """获取工作空间的所有代码文档"""
        stmt = select(CodeDocument).where(
            CodeDocument.workspace_id == workspace_id
        ).order_by(CodeDocument.file_path)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_by_file_path(
        self, 
        workspace_id: UUID, 
        file_path: str
    ) -> Optional[CodeDocument]:
        """根据文件路径获取代码文档"""
        stmt = select(CodeDocument).where(
            and_(
                CodeDocument.workspace_id == workspace_id,
                CodeDocument.file_path == file_path
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def search_by_markers(
        self,
        workspace_id: UUID,
        has_deprecated: Optional[bool] = None,
        has_fixme: Optional[bool] = None,
        has_todo: Optional[bool] = None,
        has_security: Optional[bool] = None,
        limit: int = 10
    ) -> List[CodeDocument]:
        """根据标记搜索代码文档"""
        conditions = [CodeDocument.workspace_id == workspace_id]
        
        if has_deprecated is not None:
            conditions.append(CodeDocument.has_deprecated == has_deprecated)
        if has_fixme is not None:
            conditions.append(CodeDocument.has_fixme == has_fixme)
        if has_todo is not None:
            conditions.append(CodeDocument.has_todo == has_todo)
        if has_security is not None:
            conditions.append(CodeDocument.has_security == has_security)
        
        stmt = select(CodeDocument).where(
            and_(*conditions)
        ).limit(limit)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()


class ExecutionLogRepository(IRepository):
    """执行日志仓储"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, id: UUID) -> Optional[ExecutionLog]:
        """根据 ID 获取执行日志"""
        stmt = select(ExecutionLog).where(ExecutionLog.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create(self, entity: ExecutionLog) -> ExecutionLog:
        """创建执行日志"""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: ExecutionLog) -> ExecutionLog:
        """更新执行日志"""
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: UUID) -> bool:
        """删除执行日志"""
        stmt = delete(ExecutionLog).where(ExecutionLog.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_by_session(
        self, 
        session_id: UUID,
        limit: int = 100
    ) -> List[ExecutionLog]:
        """获取会话的执行日志"""
        stmt = select(ExecutionLog).where(
            ExecutionLog.session_id == session_id
        ).order_by(ExecutionLog.created_at.desc()).limit(limit)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_by_agent(
        self,
        agent_name: str,
        limit: int = 100
    ) -> List[ExecutionLog]:
        """获取特定智能体的执行日志"""
        stmt = select(ExecutionLog).where(
            ExecutionLog.agent_name == agent_name
        ).order_by(ExecutionLog.created_at.desc()).limit(limit)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()


class LLMCallLogRepository(IRepository):
    """LLM 调用日志仓储"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, id: UUID) -> Optional[LLMCallLog]:
        """根据 ID 获取 LLM 调用日志"""
        stmt = select(LLMCallLog).where(LLMCallLog.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create(self, entity: LLMCallLog) -> LLMCallLog:
        """创建 LLM 调用日志"""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: LLMCallLog) -> LLMCallLog:
        """更新 LLM 调用日志"""
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: UUID) -> bool:
        """删除 LLM 调用日志"""
        stmt = delete(LLMCallLog).where(LLMCallLog.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_by_session(
        self, 
        session_id: UUID,
        limit: int = 100
    ) -> List[LLMCallLog]:
        """获取会话的 LLM 调用日志"""
        stmt = select(LLMCallLog).where(
            LLMCallLog.session_id == session_id
        ).order_by(LLMCallLog.created_at.desc()).limit(limit)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()
    
    async def get_by_provider(
        self,
        provider: str,
        limit: int = 100
    ) -> List[LLMCallLog]:
        """获取特定提供商的 LLM 调用日志"""
        stmt = select(LLMCallLog).where(
            LLMCallLog.provider == provider
        ).order_by(LLMCallLog.created_at.desc()).limit(limit)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()


class WorkspaceRepository(IRepository):
    """工作空间仓储"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, id: UUID) -> Optional[Workspace]:
        """根据 ID 获取工作空间"""
        stmt = select(Workspace).where(Workspace.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create(self, entity: Workspace) -> Workspace:
        """创建工作空间"""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: Workspace) -> Workspace:
        """更新工作空间"""
        entity.updated_at = datetime.now()
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: UUID) -> bool:
        """删除工作空间"""
        stmt = delete(Workspace).where(Workspace.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_by_tenant(self, tenant_id: UUID) -> List[Workspace]:
        """获取租户的所有工作空间"""
        stmt = select(Workspace).where(
            Workspace.tenant_id == tenant_id
        ).order_by(Workspace.name)
        
        result = await self.session.execute(stmt)
        return result.scalars().all()


class UserRepository(IRepository):
    """用户仓储"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def get_by_id(self, id: UUID) -> Optional[User]:
        """根据 ID 获取用户"""
        stmt = select(User).where(User.id == id)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create(self, entity: User) -> User:
        """创建用户"""
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def update(self, entity: User) -> User:
        """更新用户"""
        entity.updated_at = datetime.now()
        await self.session.commit()
        await self.session.refresh(entity)
        return entity
    
    async def delete(self, id: UUID) -> bool:
        """删除用户"""
        stmt = delete(User).where(User.id == id)
        result = await self.session.execute(stmt)
        await self.session.commit()
        return result.rowcount > 0
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        stmt = select(User).where(User.email == email)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
    
    async def get_by_username(self, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        stmt = select(User).where(User.username == username)
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()