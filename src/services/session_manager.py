"""会话状态管理器 - 支持断点续传

本模块实现会话状态持久化功能：
1. 保存当前执行状态到数据库
2. 恢复中断的会话
3. 清理过期会话

注意：这是"断点续传"功能，而非长期记忆管理。
每次任务独立，不需要跨任务记忆。

使用示例：
    from src.services.session_manager import SessionManager

    session_manager = SessionManager(db_service)

    # 保存会话
    await session_manager.save_session(session_id, state)

    # 恢复会话
    restored_state = await session_manager.load_session(session_id)
"""

import logging
import datetime
import uuid
import json
import zlib
import base64
from typing import Dict, Any, Optional, List
from typing_extensions import TypedDict

from .incremental_storage import (
    IncrementalStorageOptimizer,
    DeltaState,
    StateDiffCalculator
)


logger = logging.getLogger(__name__)


class SessionState(TypedDict):
    """会话状态数据结构"""
    session_id: str
    workspace_id: str
    user_topic: str
    project_context: str
    outline: List[Dict[str, Any]]
    current_step_index: int
    fragments: List[Dict[str, Any]]
    current_skill: Optional[str]
    skill_history: List[Dict[str, Any]]
    created_at: str
    updated_at: str


class SessionConfig(TypedDict):
    """会话配置"""
    max_age_hours: int
    auto_save_interval: int
    enable_compression: bool


DEFAULT_CONFIG: SessionConfig = {
    "max_age_hours": 24,
    "auto_save_interval": 5,
    "enable_compression": False
}


class SessionManager:
    """
    会话状态管理器

    功能：
    1. 保存当前执行状态到数据库
    2. 恢复中断的会话
    3. 清理过期会话

    设计考量：
    - 只保存必要的状态（outline、fragments、skill_history）
    - 不保存临时数据（retrieved_docs、execution_log）
    - 自动清理过期会话，避免数据库膨胀
    """

    def __init__(
        self,
        db_service: Optional[Any] = None,
        config: Optional[SessionConfig] = None
    ):
        """
        初始化会话管理器

        Args:
            db_service: 数据库服务（可选，如果不提供则使用内存存储）
            config: 会话配置
        """
        self.db = db_service
        self.config = config or DEFAULT_CONFIG

        if db_service is None:
            logger.warning("No DB service provided, using in-memory storage")
            self._memory_store: Dict[str, SessionState] = {}

        self._session_optimizer = IncrementalStorageOptimizer(
            enable_compression=self.config.get("enable_compression", True),
            max_delta_chain=100,
            compression_threshold=50,
            max_history_count=self.config.get("max_history_count", 1000),
            max_history_days=self.config.get("max_history_days", 7)
        )

        self._session_states: Dict[str, Dict[str, Any]] = {}

        logger.info(f"SessionManager initialized (max_age={self.config['max_age_hours']}h, compression={self.config.get('enable_compression', True)})")

    def _generate_session_id(self, workspace_id: str, user_topic: str) -> str:
        """生成会话 ID"""
        unique_str = f"{workspace_id}:{user_topic}:{datetime.datetime.now().isoformat()}"
        return f"session_{uuid.uuid5(uuid.NAMESPACE_DNS, unique_str).hex[:16]}"

    async def save_session(
        self,
        session_id: str,
        state: Dict[str, Any],
        workspace_id: str = "default"
    ) -> bool:
        """
        保存会话状态

        Args:
            session_id: 会话 ID
            state: GlobalState 字典
            workspace_id: 工作空间 ID

        Returns:
            是否保存成功
        """
        try:
            session_data = self._extract_session_state(session_id, state, workspace_id)

            if self.db:
                success = await self._save_to_db(session_data)
            else:
                self._memory_store[session_id] = session_data
                success = True

            if success:
                logger.info(
                    f"Session {session_id} saved: step={state.get('current_step_index', 0)}, "
                    f"fragments={len(state.get('fragments', []))}"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {str(e)}")
            return False

    def _extract_session_state(
        self,
        session_id: str,
        state: Dict[str, Any],
        workspace_id: str
    ) -> SessionState:
        """
        从 GlobalState 提取需要保存的状态

        只保存必要的数据：
        - outline（大纲）
        - current_step_index（当前步骤）
        - fragments（已生成的片段）
        - current_skill（当前技能）
        - skill_history（技能切换历史）

        不保存：
        - retrieved_docs（临时检索结果）
        - execution_log（日志，重新生成时会重建）
        - director_feedback（临时反馈）
        """
        now = datetime.datetime.now().isoformat()

        return SessionState(
            session_id=session_id,
            workspace_id=workspace_id,
            user_topic=state.get("user_topic", ""),
            project_context=state.get("project_context", ""),
            outline=state.get("outline", []),
            current_step_index=state.get("current_step_index", 0),
            fragments=state.get("fragments", []),
            current_skill=self._extract_current_skill(state.get("fragments", [])),
            skill_history=state.get("skill_history", []),
            created_at=state.get("created_at", now),
            updated_at=now
        )

    def _extract_current_skill(self, fragments: List[Dict[str, Any]]) -> Optional[str]:
        """从片段中提取当前使用的技能"""
        if not fragments:
            return None
        return fragments[-1].get("skill_used")

    async def _save_to_db(self, session_data: SessionState) -> bool:
        """
        保存到数据库（子类可重写）

        默认实现抛出 NotImplementedError，
        具体的数据库实现需要重写此方法。

        Args:
            session_data: 会话数据

        Returns:
            是否保存成功
        """
        if self.db is None:
            raise RuntimeError("No DB service configured")

        if hasattr(self.db, 'upsert_session'):
            return await self.db.upsert_session(session_data)

        raise NotImplementedError(
            "Database save not implemented. Please provide a db_service with "
            "upsert_session() method or implement _save_to_db()"
        )

    async def load_session(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        恢复会话状态

        Args:
            session_id: 会话 ID

        Returns:
            恢复后的 GlobalState 字典，如果找不到返回 None
        """
        try:
            session_data: Optional[SessionState]

            if self.db:
                session_data = await self._load_from_db(session_id)
            else:
                session_data = self._memory_store.get(session_id)

            if not session_data:
                logger.debug(f"Session {session_id} not found")
                return None

            state = self._reconstruct_state(session_data)

            logger.info(
                f"Session {session_id} loaded: step={state.get('current_step_index', 0)}, "
                f"fragments={len(state.get('fragments', []))}"
            )

            return state

        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {str(e)}")
            return None

    def _reconstruct_state(self, session_data: SessionState) -> Dict[str, Any]:
        """
        从会话数据重建 GlobalState
        """
        return {
            "user_topic": session_data["user_topic"],
            "project_context": session_data.get("project_context", ""),
            "outline": session_data["outline"],
            "current_step_index": session_data["current_step_index"],
            "fragments": session_data["fragments"],
            "skill_history": session_data.get("skill_history", []),
            "execution_log": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "fact_check_passed": None,
            "error_flag": None,
            "retry_count": 0,
            "workflow_complete": False,
            "final_screenplay": None,
            "created_at": session_data["created_at"],
            "resumed_from_session": session_data["session_id"]
        }

    async def _load_from_db(self, session_id: str) -> Optional[SessionState]:
        """
        从数据库加载（子类可重写）

        Args:
            session_id: 会话 ID

        Returns:
            会话数据，找不到返回 None
        """
        if self.db is None:
            raise RuntimeError("No DB service configured")

        if hasattr(self.db, 'get_session'):
            return await self.db.get_session(session_id)

        raise NotImplementedError(
            "Database load not implemented. Please provide a db_service with "
            "get_session() method or implement _load_from_db()"
        )

    async def delete_session(self, session_id: str) -> bool:
        """
        删除会话

        Args:
            session_id: 会话 ID

        Returns:
            是否删除成功
        """
        try:
            if self.db:
                success = await self._delete_from_db(session_id)
            else:
                if session_id in self._memory_store:
                    del self._memory_store[session_id]
                    success = True
                else:
                    success = False

            if success:
                logger.info(f"Session {session_id} deleted")

            return success

        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {str(e)}")
            return False

    async def _delete_from_db(self, session_id: str) -> bool:
        """从数据库删除"""
        if self.db is None:
            raise RuntimeError("No DB service configured")

        if hasattr(self.db, 'delete_session'):
            return await self.db.delete_session(session_id)

        raise NotImplementedError(
            "Database delete not implemented"
        )

    async def cleanup_expired_sessions(self) -> int:
        """
        清理过期会话

        Returns:
            删除的会话数量
        """
        try:
            cutoff_time = datetime.datetime.now() - datetime.timedelta(
                hours=self.config["max_age_hours"]
            )

            if self.db:
                deleted_count = await self._cleanup_from_db(cutoff_time)
            else:
                expired_ids = [
                    sid for sid, data in self._memory_store.items()
                    if datetime.datetime.fromisoformat(data["updated_at"]) < cutoff_time
                ]
                for sid in expired_ids:
                    del self._memory_store[sid]
                deleted_count = len(expired_ids)

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired sessions")

            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {str(e)}")
            return 0

    async def _cleanup_from_db(self, cutoff_time: datetime.datetime) -> int:
        """从数据库清理过期会话"""
        if self.db is None:
            raise RuntimeError("No DB service configured")

        if hasattr(self.db, 'cleanup_expired_sessions'):
            return await self.db.cleanup_expired_sessions(cutoff_time)

        raise NotImplementedError(
            "Database cleanup not implemented"
        )

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取会话信息（不包含完整状态）

        Args:
            session_id: 会话 ID

        Returns:
            会话信息字典
        """
        session_data = await self.load_session(session_id)

        if not session_data:
            return None

        return {
            "session_id": session_id,
            "user_topic": session_data.get("user_topic"),
            "current_step_index": session_data.get("current_step_index"),
            "total_steps": len(session_data.get("outline", [])),
            "fragments_count": len(session_data.get("fragments", [])),
            "workflow_complete": session_data.get("workflow_complete", False)
        }

    def get_memory_store_size(self) -> int:
        """获取内存存储中的会话数量"""
        return len(self._memory_store)

    def list_memory_sessions(self) -> List[str]:
        """列出内存存储中的所有会话 ID"""
        return list(self._memory_store.keys())


class InMemorySessionStore:
    """
    内存会话存储（用于测试或简单场景）

    提供简单的内存存储实现，无需数据库配置。
    """

    def __init__(self, max_sessions: int = 100):
        self.sessions: Dict[str, SessionState] = {}
        self.max_sessions = max_sessions

    async def upsert_session(self, session_data: SessionState) -> bool:
        """保存会话"""
        if len(self.sessions) >= self.max_sessions:
            oldest_key = next(iter(self.sessions.keys()))
            del self.sessions[oldest_key]

        self.sessions[session_data["session_id"]] = session_data
        return True

    async def get_session(self, session_id: str) -> Optional[SessionState]:
        """获取会话"""
        return self.sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    async def cleanup_expired_sessions(self, cutoff_time: datetime.datetime) -> int:
        """清理过期会话"""
        expired_ids = [
            sid for sid, data in self.sessions.items()
            if datetime.datetime.fromisoformat(data["updated_at"]) < cutoff_time
        ]
        for sid in expired_ids:
            del self.sessions[sid]
        return len(expired_ids)

    async def save_session_incremental(
        self,
        session_id: str,
        state: Dict[str, Any]
    ) -> bool:
        """
        增量保存会话状态（使用差量存储）

        Args:
            session_id: 会话 ID
            state: 当前状态

        Returns:
            是否保存成功
        """
        try:
            if session_id not in self._session_states:
                self._session_states[session_id] = {}
                old_state = {}
            else:
                old_state = self._session_states[session_id].copy()

            new_state = self._extract_session_dict(state, session_id)
            self._session_states[session_id] = new_state

            changed_fields = list(new_state.keys())

            self._session_optimizer.record_step(
                old_state=old_state,
                new_state=new_state,
                changed_fields=changed_fields
            )

            logger.debug(
                f"Incremental save for {session_id}: "
                f"changed_fields={changed_fields}"
            )

            return True

        except Exception as e:
            logger.error(f"Failed to incrementally save session {session_id}: {str(e)}")
            return False

    def _extract_session_dict(
        self,
        state: Dict[str, Any],
        session_id: str
    ) -> Dict[str, Any]:
        """提取会话字典（用于增量存储）"""
        return {
            "session_id": session_id,
            "user_topic": state.get("user_topic", ""),
            "project_context": state.get("project_context", ""),
            "outline": state.get("outline", []),
            "current_step_index": state.get("current_step_index", 0),
            "fragments": state.get("fragments", []),
            "skill_history": state.get("skill_history", []),
            "created_at": state.get("created_at", datetime.datetime.now().isoformat())
        }

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储优化统计"""
        return self._session_optimizer.get_optimization_stats()

    def estimate_session_storage(
        self,
        state: Dict[str, Any],
        estimated_steps: int = 100
    ) -> Dict[str, Any]:
        """估算会话存储大小"""
        session_dict = self._extract_session_dict(state, "estimate")
        return self._session_optimizer.estimate_storage_size(
            state=session_dict,
            steps=estimated_steps
        )

    async def cleanup_incremental_storage(self) -> Dict[str, Any]:
        """
        清理增量存储

        Returns:
            清理统计
        """
        cleanup_stats = {
            "sessions_before": len(self._session_states),
            "sessions_after": len(self._session_states),
            "snapshots_count": 0,
            "deltas_cleaned": 0
        }

        if hasattr(self._session_optimizer, 'cleanup_policy'):
            sessions_to_remove, reason = (
                self._session_optimizer.cleanup_policy.get_sessions_to_cleanup()
            )

            for session_id in sessions_to_remove:
                if session_id in self._session_states:
                    del self._session_states[session_id]

            cleanup_stats["sessions_after"] = len(self._session_states)
            cleanup_stats["reason"] = reason

        return cleanup_stats
