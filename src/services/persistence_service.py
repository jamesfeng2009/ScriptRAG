"""
持久化服务

本模块提供 SharedState 与数据库实体之间的转换和持久化操作。
"""

import logging
from typing import Optional, List, Dict, Any
from uuid import UUID, uuid4
from datetime import datetime

from ..domain.models import SharedState, OutlineStep as DomainOutlineStep, ScreenplayFragment as DomainScreenplayFragment, RetrievedDocument as DomainRetrievedDocument
from ..domain.entities import ScreenplaySession, OutlineStep, ScreenplayFragment, RetrievedDocument, ExecutionLog, LLMCallLog
from ..services.database.orm_service import ORMDatabaseService

logger = logging.getLogger(__name__)


class PersistenceService:
    """持久化服务"""
    
    def __init__(self, orm_service: ORMDatabaseService):
        """
        初始化持久化服务
        
        Args:
            orm_service: ORM 数据库服务
        """
        self.orm_service = orm_service
    
    async def save_shared_state(
        self,
        shared_state: SharedState,
        user_id: UUID,
        workspace_id: UUID,
        session_id: Optional[UUID] = None
    ) -> UUID:
        """
        保存 SharedState 到数据库
        
        Args:
            shared_state: 共享状态对象
            user_id: 用户 ID
            workspace_id: 工作空间 ID
            session_id: 会话 ID（如果为 None 则创建新会话）
        
        Returns:
            会话 ID
        """
        async with self.orm_service.get_session() as db_session:
            session_repo = await self.orm_service.get_screenplay_session_repository(db_session)
            
            if session_id:
                # 更新现有会话
                session_entity = await session_repo.get_by_id(session_id)
                if not session_entity:
                    raise ValueError(f"Session {session_id} not found")
                
                # 更新会话属性
                session_entity.topic = shared_state.user_topic
                session_entity.context = shared_state.project_context
                session_entity.current_skill = shared_state.current_skill
                session_entity.global_tone = shared_state.global_tone
                session_entity.max_retries = shared_state.max_retries
                session_entity.pivot_triggered = shared_state.pivot_triggered
                session_entity.pivot_reason = shared_state.pivot_reason
                session_entity.fact_check_passed = shared_state.fact_check_passed
                session_entity.awaiting_user_input = shared_state.awaiting_user_input
                session_entity.user_input_prompt = shared_state.user_input_prompt
                session_entity.skill_history = shared_state.skill_history
                session_entity.execution_log = shared_state.execution_log
                
                await session_repo.update(session_entity)
                
                # 更新大纲步骤（简单起见，先删除再创建）
                # 在生产环境中应该使用更智能的更新策略
                await self._update_outline_steps(db_session, session_id, shared_state.outline)
                await self._update_screenplay_fragments(db_session, session_id, shared_state.fragments)
                await self._update_retrieved_documents(db_session, session_id, shared_state.retrieved_docs)
                
                return session_id
            else:
                # 创建新会话
                return await session_repo.create_from_shared_state(
                    shared_state, user_id, workspace_id
                ).id
    
    async def load_shared_state(self, session_id: UUID) -> Optional[SharedState]:
        """
        从数据库加载 SharedState
        
        Args:
            session_id: 会话 ID
        
        Returns:
            SharedState 对象，如果不存在则返回 None
        """
        async with self.orm_service.get_session() as db_session:
            session_repo = await self.orm_service.get_screenplay_session_repository(db_session)
            return await session_repo.to_shared_state(session_id)
    
    async def log_execution(
        self,
        session_id: Optional[UUID],
        agent_name: str,
        action: str,
        status: str,
        duration_ms: Optional[int] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        记录执行日志
        
        Args:
            session_id: 会话 ID
            agent_name: 智能体名称
            action: 执行动作
            status: 执行状态
            duration_ms: 执行时长（毫秒）
            input_data: 输入数据
            output_data: 输出数据
            error_message: 错误消息
            metadata: 元数据
        
        Returns:
            日志 ID
        """
        async with self.orm_service.get_session() as db_session:
            log_repo = await self.orm_service.get_execution_log_repository(db_session)
            
            log_entity = ExecutionLog(
                session_id=session_id,
                agent_name=agent_name,
                action=action,
                status=status,
                duration_ms=duration_ms,
                input_data=input_data,
                output_data=output_data,
                error_message=error_message,
                metadata=metadata or {}
            )
            
            created_log = await log_repo.create(log_entity)
            return created_log.id
    
    async def log_llm_call(
        self,
        session_id: Optional[UUID],
        provider: str,
        model: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
        response_time_ms: Optional[int] = None,
        cost_usd: Optional[float] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        记录 LLM 调用日志
        
        Args:
            session_id: 会话 ID
            provider: LLM 提供商
            model: 模型名称
            prompt_tokens: 提示 token 数
            completion_tokens: 完成 token 数
            total_tokens: 总 token 数
            response_time_ms: 响应时间（毫秒）
            cost_usd: 成本（美元）
            status: 调用状态
            error_message: 错误消息
            metadata: 元数据
        
        Returns:
            日志 ID
        """
        async with self.orm_service.get_session() as db_session:
            log_repo = await self.orm_service.get_llm_call_log_repository(db_session)
            
            log_entity = LLMCallLog(
                session_id=session_id,
                provider=provider,
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                response_time_ms=response_time_ms,
                cost_usd=cost_usd,
                status=status,
                error_message=error_message,
                metadata=metadata or {}
            )
            
            created_log = await log_repo.create(log_entity)
            return created_log.id
    
    async def get_session_logs(
        self, 
        session_id: UUID,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取会话的执行日志
        
        Args:
            session_id: 会话 ID
            limit: 限制数量
        
        Returns:
            日志列表
        """
        async with self.orm_service.get_session() as db_session:
            log_repo = await self.orm_service.get_execution_log_repository(db_session)
            logs = await log_repo.get_by_session(session_id, limit)
            
            return [
                {
                    "id": str(log.id),
                    "agent_name": log.agent_name,
                    "action": log.action,
                    "status": log.status,
                    "duration_ms": log.duration_ms,
                    "input_data": log.input_data,
                    "output_data": log.output_data,
                    "error_message": log.error_message,
                    "metadata": log.metadata,
                    "created_at": log.created_at.isoformat()
                }
                for log in logs
            ]
    
    async def get_llm_call_logs(
        self, 
        session_id: UUID,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        获取会话的 LLM 调用日志
        
        Args:
            session_id: 会话 ID
            limit: 限制数量
        
        Returns:
            日志列表
        """
        async with self.orm_service.get_session() as db_session:
            log_repo = await self.orm_service.get_llm_call_log_repository(db_session)
            logs = await log_repo.get_by_session(session_id, limit)
            
            return [
                {
                    "id": str(log.id),
                    "provider": log.provider,
                    "model": log.model,
                    "prompt_tokens": log.prompt_tokens,
                    "completion_tokens": log.completion_tokens,
                    "total_tokens": log.total_tokens,
                    "response_time_ms": log.response_time_ms,
                    "cost_usd": log.cost_usd,
                    "status": log.status,
                    "error_message": log.error_message,
                    "metadata": log.metadata,
                    "created_at": log.created_at.isoformat()
                }
                for log in logs
            ]
    
    async def _update_outline_steps(
        self, 
        db_session, 
        session_id: UUID, 
        outline: List[DomainOutlineStep]
    ):
        """更新大纲步骤"""
        # 删除现有步骤
        from sqlalchemy import delete
        stmt = delete(OutlineStep).where(OutlineStep.session_id == session_id)
        await db_session.execute(stmt)
        
        # 创建新步骤
        for step in outline:
            outline_step = OutlineStep(
                session_id=session_id,
                step_id=step.step_id,
                description=step.description,
                status=step.status,
                retry_count=step.retry_count
            )
            db_session.add(outline_step)
        
        await db_session.commit()
    
    async def _update_screenplay_fragments(
        self, 
        db_session, 
        session_id: UUID, 
        fragments: List[DomainScreenplayFragment]
    ):
        """更新剧本片段"""
        # 删除现有片段
        from sqlalchemy import delete
        stmt = delete(ScreenplayFragment).where(ScreenplayFragment.session_id == session_id)
        await db_session.execute(stmt)
        
        # 创建新片段
        for fragment in fragments:
            # 找到对应的大纲步骤
            from sqlalchemy import select, and_
            outline_step_stmt = select(OutlineStep).where(
                and_(
                    OutlineStep.session_id == session_id,
                    OutlineStep.step_id == fragment.step_id
                )
            )
            outline_step_result = await db_session.execute(outline_step_stmt)
            outline_step = outline_step_result.scalar_one_or_none()
            
            if outline_step:
                screenplay_fragment = ScreenplayFragment(
                    session_id=session_id,
                    step_id=outline_step.id,
                    content=fragment.content,
                    skill_used=fragment.skill_used,
                    sources=fragment.sources
                )
                db_session.add(screenplay_fragment)
        
        await db_session.commit()
    
    async def _update_retrieved_documents(
        self, 
        db_session, 
        session_id: UUID, 
        documents: List[DomainRetrievedDocument]
    ):
        """更新检索文档"""
        # 删除现有文档
        from sqlalchemy import delete
        stmt = delete(RetrievedDocument).where(RetrievedDocument.session_id == session_id)
        await db_session.execute(stmt)
        
        # 创建新文档
        for doc in documents:
            retrieved_doc = RetrievedDocument(
                session_id=session_id,
                content=doc.content,
                source=doc.source,
                confidence=doc.confidence,
                summary=doc.summary,
                metadata=doc.metadata
            )
            db_session.add(retrieved_doc)
        
        await db_session.commit()