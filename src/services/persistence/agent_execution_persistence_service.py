"""
Agent Execution 持久化服务

本模块提供 Agent 执行记录的数据库持久化功能，
用于追踪剧本生成过程中每个 agent 的执行情况。
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
from uuid import uuid4

from sqlalchemy import Column, String, Text, DateTime, JSON, select, update, delete, func
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.dialects.postgresql import insert

from src.domain.entities import Base, AgentExecution

logger = logging.getLogger(__name__)


class AgentExecutionRecord:
    """Agent 执行记录类（内存表示）"""

    def __init__(
        self,
        execution_id: str,
        task_id: Optional[str] = None,
        chat_session_id: Optional[str] = None,
        agent_name: str = "",
        node_name: str = "",
        step_id: Optional[str] = None,
        step_index: Optional[int] = None,
        action: Optional[str] = None,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error_message: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        retry_count: int = 0,
        extra_data: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None
    ):
        self.execution_id = execution_id
        self.task_id = task_id
        self.chat_session_id = chat_session_id
        self.agent_name = agent_name
        self.node_name = node_name
        self.step_id = step_id
        self.step_index = step_index
        self.action = action
        self.input_data = input_data or {}
        self.output_data = output_data or {}
        self.status = status
        self.error_message = error_message
        self.execution_time_ms = execution_time_ms
        self.retry_count = retry_count
        self.extra_data = extra_data or {}
        self.created_at = created_at or datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "execution_id": self.execution_id,
            "task_id": self.task_id,
            "chat_session_id": self.chat_session_id,
            "agent_name": self.agent_name,
            "node_name": self.node_name,
            "step_id": self.step_id,
            "step_index": self.step_index,
            "action": self.action,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "status": self.status,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "retry_count": self.retry_count,
            "extra_data": self.extra_data,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def to_entity(self) -> AgentExecution:
        """转换为数据库实体"""
        return AgentExecution(
            execution_id=self.execution_id,
            task_id=self.task_id,
            chat_session_id=self.chat_session_id,
            agent_name=self.agent_name,
            node_name=self.node_name,
            step_id=self.step_id,
            step_index=self.step_index,
            action=self.action,
            input_data=self.input_data,
            output_data=self.output_data,
            status=self.status,
            error_message=self.error_message,
            execution_time_ms=self.execution_time_ms,
            retry_count=self.retry_count,
            extra_data=self.extra_data,
            created_at=self.created_at
        )

    @classmethod
    def from_entity(cls, entity: AgentExecution) -> "AgentExecutionRecord":
        """从实体创建"""
        return cls(
            execution_id=entity.execution_id,
            task_id=entity.task_id,
            chat_session_id=entity.chat_session_id,
            agent_name=entity.agent_name,
            node_name=entity.node_name,
            step_id=entity.step_id,
            step_index=entity.step_index,
            action=entity.action,
            input_data=entity.input_data or {},
            output_data=entity.output_data or {},
            status=entity.status,
            error_message=entity.error_message,
            execution_time_ms=entity.execution_time_ms,
            retry_count=entity.retry_count,
            extra_data=entity.extra_data or {},
            created_at=entity.created_at
        )


class AgentExecutionPersistenceService:
    """Agent 执行记录持久化服务"""

    _instance = None
    _pool = None
    _session_factory = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        return cls()

    @staticmethod
    def _get_database_url() -> str:
        """获取数据库连接 URL"""
        return "postgresql+asyncpg://postgres:123456@localhost:5433/Screenplay"

    @classmethod
    def _ensure_connection(cls):
        """确保数据库连接已建立"""
        if cls._pool is None:
            database_url = cls._get_database_url()
            cls._pool = create_async_engine(
                database_url,
                poolclass=NullPool,
                echo=False
            )
            cls._session_factory = async_sessionmaker(
                cls._pool,
                class_=AsyncSession,
                expire_on_commit=False
            )
            logger.info("[AgentExecutionPersistenceService] 数据库连接已建立")

    async def create(self, record: AgentExecutionRecord) -> str:
        """创建执行记录（幂等操作）- 如果已存在则忽略"""
        try:
            self._ensure_connection()

            entity = record.to_entity()

            async with self._session_factory() as session:
                stmt = insert(AgentExecution).values(
                    execution_id=entity.execution_id,
                    task_id=entity.task_id,
                    chat_session_id=entity.chat_session_id,
                    agent_name=entity.agent_name,
                    node_name=entity.node_name,
                    step_id=entity.step_id,
                    step_index=entity.step_index,
                    action=entity.action,
                    input_data=entity.input_data,
                    output_data=entity.output_data,
                    status=entity.status,
                    error_message=entity.error_message,
                    execution_time_ms=entity.execution_time_ms,
                    retry_count=entity.retry_count,
                    extra_data=entity.extra_data,
                    created_at=entity.created_at
                ).on_conflict_do_nothing(
                    index_elements=['execution_id']
                )

                await session.execute(stmt)
                await session.commit()
                logger.info(f"[AgentExecutionPersistenceService] 执行记录已落库（幂等）: {record.execution_id} - {record.agent_name}")
                return record.execution_id

        except Exception as e:
            logger.error(f"[AgentExecutionPersistenceService] 执行记录落库失败: {e}")
            raise

    async def get_by_execution_id(self, execution_id: str) -> Optional[AgentExecutionRecord]:
        """根据执行 ID 获取记录"""
        try:
            self._ensure_connection()

            async with self._session_factory() as session:
                result = await session.execute(
                    select(AgentExecution).where(AgentExecution.execution_id == execution_id)
                )
                entity = result.scalar_one_or_none()
                if entity:
                    return AgentExecutionRecord.from_entity(entity)
                return None

        except Exception as e:
            logger.error(f"[AgentExecutionPersistenceService] 获取执行记录失败: {e}")
            return None

    async def list_by_task_id(self, task_id: str) -> List[AgentExecutionRecord]:
        """根据任务 ID 获取所有执行记录"""
        try:
            self._ensure_connection()

            async with self._session_factory() as session:
                result = await session.execute(
                    select(AgentExecution)
                    .where(AgentExecution.task_id == task_id)
                    .order_by(AgentExecution.created_at)
                )
                entities = result.scalars().all()
                return [AgentExecutionRecord.from_entity(e) for e in entities]

        except Exception as e:
            logger.error(f"[AgentExecutionPersistenceService] 获取任务执行记录失败: {e}")
            return []

    async def list_by_chat_session_id(self, chat_session_id: str) -> List[AgentExecutionRecord]:
        """根据会话 ID 获取所有执行记录"""
        try:
            self._ensure_connection()

            async with self._session_factory() as session:
                result = await session.execute(
                    select(AgentExecution)
                    .where(AgentExecution.chat_session_id == chat_session_id)
                    .order_by(AgentExecution.created_at)
                )
                entities = result.scalars().all()
                return [AgentExecutionRecord.from_entity(e) for e in entities]

        except Exception as e:
            logger.error(f"[AgentExecutionPersistenceService] 获取会话执行记录失败: {e}")
            return []

    async def update_status(
        self,
        execution_id: str,
        status: str,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        execution_time_ms: Optional[float] = None
    ) -> bool:
        """更新执行记录状态"""
        try:
            self._ensure_connection()

            async with self._session_factory() as session:
                await session.execute(
                    update(AgentExecution)
                    .where(AgentExecution.execution_id == execution_id)
                    .values(
                        status=status,
                        output_data=output_data,
                        error_message=error_message,
                        execution_time_ms=execution_time_ms
                    )
                )
                await session.commit()
                return True

        except Exception as e:
            logger.error(f"[AgentExecutionPersistenceService] 更新执行记录状态失败: {e}")
            return False

    async def count_by_agent(self, task_id: str) -> Dict[str, int]:
        """统计任务中每个 agent 的执行次数"""
        try:
            self._ensure_connection()

            async with self._session_factory() as session:
                result = await session.execute(
                    select(AgentExecution.agent_name, func.count(AgentExecution.id))
                    .where(AgentExecution.task_id == task_id)
                    .group_by(AgentExecution.agent_name)
                )
                return {row[0]: row[1] for row in result.all()}

        except Exception as e:
            logger.error(f"[AgentExecutionPersistenceService] 统计 agent 执行次数失败: {e}")
            return {}

    async def close(self):
        """关闭数据库连接"""
        if self._pool:
            await self._pool.dispose()
            self._pool = None
            self._session_factory = None
            logger.info("[AgentExecutionPersistenceService] 数据库连接已关闭")


agent_execution_service = AgentExecutionPersistenceService.get_instance()
