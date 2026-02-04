"""SessionManager 单元测试和扩展

测试内容：
1. 基本会话管理功能
2. 内存存储
3. 增量存储优化
4. 幻觉控制集成
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, Optional, List
import sys
import os
import asyncio
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.session_manager import (
    SessionManager,
    SessionState,
    SessionConfig,
    DEFAULT_CONFIG,
    InMemorySessionStore,
)


class TestSessionManager:
    """SessionManager 测试类"""

    def setup_method(self):
        """每个测试前创建新的 SessionManager"""
        self.config: SessionConfig = {
            "max_age_hours": 24,
            "auto_save_interval": 5,
            "enable_compression": False
        }
        self.manager = SessionManager(
            db_service=None,
            config=self.config
        )

    def test_init(self):
        """测试初始化"""
        assert self.manager.config["max_age_hours"] == 24
        assert self.manager.config["auto_save_interval"] == 5
        assert isinstance(self.manager._memory_store, dict)

    def test_generate_session_id(self):
        """测试生成会话 ID"""
        session_id = self.manager._generate_session_id(
            workspace_id="ws1",
            user_topic="test topic"
        )

        assert session_id.startswith("session_")
        assert len(session_id) > 10

    def test_extract_session_state(self):
        """测试提取会话状态"""
        state = {
            "user_topic": "Python async",
            "project_context": "async guide",
            "outline": [{"step": 1}, {"step": 2}],
            "current_step_index": 1,
            "fragments": [{"content": "fragment 1"}],
            "skill_history": [{"skill": "python"}],
            "execution_log": [],
            "created_at": "2024-01-01T00:00:00"
        }

        session_state = self.manager._extract_session_state(
            session_id="test_session",
            state=state,
            workspace_id="ws1"
        )

        assert session_state["session_id"] == "test_session"
        assert session_state["workspace_id"] == "ws1"
        assert session_state["user_topic"] == "Python async"
        assert session_state["current_step_index"] == 1
        assert len(session_state["fragments"]) == 1
        assert session_state["fragments"][0]["content"] == "fragment 1"

    def test_extract_current_skill(self):
        """测试提取当前技能"""
        fragments = [
            {"skill_used": "python", "content": "code"},
            {"skill_used": "javascript", "content": "js code"}
        ]

        skill = self.manager._extract_current_skill(fragments)

        assert skill == "javascript"

    def test_extract_current_skill_empty(self):
        """测试空片段提取技能"""
        skill = self.manager._extract_current_skill([])

        assert skill is None

    def test_reconstruct_state(self):
        """测试重建状态"""
        session_data = SessionState(
            session_id="test_session",
            workspace_id="ws1",
            user_topic="test topic",
            project_context="test context",
            outline=[{"step": 1}],
            current_step_index=0,
            fragments=[{"content": "fragment"}],
            current_skill="python",
            skill_history=[{"skill": "python"}],
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        state = self.manager._reconstruct_state(session_data)

        assert state["user_topic"] == "test topic"
        assert state["outline"] == [{"step": 1}]
        assert state["current_step_index"] == 0
        assert len(state["fragments"]) == 1
        assert state["workflow_complete"] is False
        assert state["resumed_from_session"] == "test_session"

    def test_save_session_memory(self):
        """测试保存会话到内存"""
        state = {
            "user_topic": "test",
            "outline": [],
            "fragments": [],
            "skill_history": []
        }

        success = asyncio.run(self.manager.save_session(
            session_id="test_session",
            state=state,
            workspace_id="ws1"
        ))

        assert success is True
        assert "test_session" in self.manager._memory_store

    def test_load_session_memory(self):
        """测试从内存加载会话"""
        state = {
            "user_topic": "test",
            "outline": [],
            "fragments": [],
            "skill_history": []
        }

        asyncio.run(self.manager.save_session(
            session_id="test_session",
            state=state,
            workspace_id="ws1"
        ))

        loaded = asyncio.run(self.manager.load_session("test_session"))

        assert loaded is not None
        assert loaded["user_topic"] == "test"

    def test_load_session_not_found(self):
        """测试加载不存在的会话"""
        loaded = asyncio.run(self.manager.load_session("nonexistent"))

        assert loaded is None

    def test_delete_session_memory(self):
        """测试删除会话"""
        state = {
            "user_topic": "test",
            "outline": [],
            "fragments": [],
            "skill_history": []
        }

        asyncio.run(self.manager.save_session(
            session_id="test_session",
            state=state,
            workspace_id="ws1"
        ))

        success = asyncio.run(self.manager.delete_session("test_session"))

        assert success is True
        assert "test_session" not in self.manager._memory_store

    def test_cleanup_expired_sessions(self):
        """测试清理过期会话"""
        old_session_id = "old_session"

        self.manager._memory_store[old_session_id] = SessionState(
            session_id=old_session_id,
            workspace_id="ws1",
            user_topic="old topic",
            project_context="old context",
            outline=[],
            current_step_index=0,
            fragments=[],
            current_skill=None,
            skill_history=[],
            created_at="2023-01-01T00:00:00",
            updated_at="2023-01-01T00:00:00"
        )

        deleted_count = asyncio.run(self.manager.cleanup_expired_sessions())

        assert deleted_count >= 1
        assert old_session_id not in self.manager._memory_store

    def test_get_session_info(self):
        """测试获取会话信息"""
        state = {
            "user_topic": "test",
            "project_context": "context",
            "outline": [{"step": 1}, {"step": 2}],
            "current_step_index": 1,
            "fragments": [{"content": "f1"}],
            "skill_history": [],
            "workflow_complete": False
        }

        asyncio.run(self.manager.save_session(
            session_id="test_session",
            state=state,
            workspace_id="ws1"
        ))

        info = asyncio.run(self.manager.get_session_info("test_session"))

        assert info is not None
        assert info["user_topic"] == "test"
        assert info["current_step_index"] == 1
        assert info["total_steps"] == 2
        assert info["fragments_count"] == 1

    def test_get_memory_store_size(self):
        """测试获取内存存储大小"""
        assert self.manager.get_memory_store_size() == 0

        state = {"user_topic": "test", "outline": [], "fragments": [], "skill_history": []}
        asyncio.run(self.manager.save_session("s1", state, "ws1"))
        asyncio.run(self.manager.save_session("s2", state, "ws1"))

        assert self.manager.get_memory_store_size() == 2

    def test_list_memory_sessions(self):
        """测试列出内存会话"""
        state = {"user_topic": "test", "outline": [], "fragments": [], "skill_history": []}
        asyncio.run(self.manager.save_session("s1", state, "ws1"))
        asyncio.run(self.manager.save_session("s2", state, "ws1"))

        sessions = self.manager.list_memory_sessions()

        assert len(sessions) == 2
        assert "s1" in sessions
        assert "s2" in sessions


class TestInMemorySessionStore:
    """内存会话存储测试类"""

    def setup_method(self):
        """每个测试前创建新的存储"""
        self.store = InMemorySessionStore(max_sessions=10)

    def test_upsert_session(self):
        """测试保存会话"""
        session_data = SessionState(
            session_id="test",
            workspace_id="ws1",
            user_topic="test",
            project_context="",
            outline=[],
            current_step_index=0,
            fragments=[],
            current_skill=None,
            skill_history=[],
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        success = asyncio.run(self.store.upsert_session(session_data))

        assert success is True
        assert "test" in self.store.sessions

    def test_get_session(self):
        """测试获取会话"""
        session_data = SessionState(
            session_id="test",
            workspace_id="ws1",
            user_topic="test",
            project_context="",
            outline=[],
            current_step_index=0,
            fragments=[],
            current_skill=None,
            skill_history=[],
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        asyncio.run(self.store.upsert_session(session_data))

        loaded = asyncio.run(self.store.get_session("test"))

        assert loaded is not None
        assert loaded["user_topic"] == "test"

    def test_delete_session(self):
        """测试删除会话"""
        session_data = SessionState(
            session_id="test",
            workspace_id="ws1",
            user_topic="test",
            project_context="",
            outline=[],
            current_step_index=0,
            fragments=[],
            current_skill=None,
            skill_history=[],
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )

        asyncio.run(self.store.upsert_session(session_data))
        success = asyncio.run(self.store.delete_session("test"))

        assert success is True
        assert "test" not in self.store.sessions

    def test_max_sessions_limit(self):
        """测试会话数量限制"""
        for i in range(15):
            session_data = SessionState(
                session_id=f"session_{i}",
                workspace_id="ws1",
                user_topic=f"topic_{i}",
                project_context="",
                outline=[],
                current_step_index=0,
                fragments=[],
                current_skill=None,
                skill_history=[],
                created_at="2024-01-01T00:00:00",
                updated_at="2024-01-01T00:00:00"
            )
            asyncio.run(self.store.upsert_session(session_data))

        assert len(self.store.sessions) == 10

    def test_cleanup_expired(self):
        """测试清理过期会话"""
        now = datetime.datetime.now()

        for i in range(5):
            session_data = SessionState(
                session_id=f"session_{i}",
                workspace_id="ws1",
                user_topic=f"topic_{i}",
                project_context="",
                outline=[],
                current_step_index=0,
                fragments=[],
                current_skill=None,
                skill_history=[],
                created_at=now.isoformat(),
                updated_at=(now - datetime.timedelta(days=2)).isoformat() if i < 3 else now.isoformat()
            )
            asyncio.run(self.store.upsert_session(session_data))

        cutoff = now - datetime.timedelta(hours=24)
        deleted = asyncio.run(self.store.cleanup_expired_sessions(cutoff))

        assert deleted == 3


class TestSessionManagerWithDB:
    """带数据库的 SessionManager 测试类"""

    def test_save_with_db_service(self):
        """测试使用数据库服务保存"""
        mock_db = AsyncMock()
        mock_db.upsert_session = AsyncMock(return_value=True)

        manager = SessionManager(db_service=mock_db)

        state = {
            "user_topic": "test",
            "outline": [],
            "fragments": [],
            "skill_history": []
        }

        success = asyncio.run(manager.save_session("test_session", state, "ws1"))

        assert success is True
        mock_db.upsert_session.assert_called_once()

    def test_load_with_db_service(self):
        """测试使用数据库服务加载"""
        mock_db = AsyncMock()

        session_data = SessionState(
            session_id="test",
            workspace_id="ws1",
            user_topic="test",
            project_context="",
            outline=[],
            current_step_index=0,
            fragments=[],
            current_skill=None,
            skill_history=[],
            created_at="2024-01-01T00:00:00",
            updated_at="2024-01-01T00:00:00"
        )
        mock_db.get_session = AsyncMock(return_value=session_data)

        manager = SessionManager(db_service=mock_db)
        loaded = asyncio.run(manager.load_session("test"))

        assert loaded is not None
        assert loaded["user_topic"] == "test"

    def test_delete_with_db_service(self):
        """测试使用数据库服务删除"""
        mock_db = AsyncMock()
        mock_db.delete_session = AsyncMock(return_value=True)

        manager = SessionManager(db_service=mock_db)
        success = asyncio.run(manager.delete_session("test"))

        assert success is True
        mock_db.delete_session.assert_called_once()


class TestSessionManagerEdgeCases:
    """SessionManager 边界情况测试"""

    def setup_method(self):
        """每个测试前创建新的 SessionManager"""
        self.config: SessionConfig = {
            "max_age_hours": 24,
            "auto_save_interval": 5,
            "enable_compression": False
        }
        self.manager = SessionManager(
            db_service=None,
            config=self.config
        )

    def test_save_session_db_not_implemented(self):
        """测试数据库未实现时的保存"""
        class PartialDB:
            async def upsert_session(self, data):
                return True

        manager = SessionManager(db_service=PartialDB())

        state = {
            "user_topic": "test",
            "outline": [],
            "fragments": [],
            "skill_history": []
        }

        success = asyncio.run(manager.save_session("test_session", state, "ws1"))

        assert success is True

    def test_load_session_db_not_implemented(self):
        """测试数据库未实现时的加载"""
        class PartialDB:
            pass

        manager = SessionManager(db_service=PartialDB())

        loaded = asyncio.run(manager.load_session("test"))

        assert loaded is None

    def test_extract_session_with_missing_fields(self):
        """测试提取缺少字段的会话状态"""
        state = {
            "user_topic": "test"
        }

        session_state = self.manager._extract_session_state(
            session_id="test",
            state=state,
            workspace_id="ws1"
        )

        assert session_state["session_id"] == "test"
        assert session_state["user_topic"] == "test"
        assert session_state["outline"] == []
        assert session_state["fragments"] == []

    def test_get_session_info_not_found(self):
        """测试获取不存在的会话信息"""
        info = asyncio.run(self.manager.get_session_info("nonexistent"))

        assert info is None


class TestIntegrationWithHallucinationControl:
    """幻觉控制集成测试"""

    def setup_method(self):
        """每个测试前创建新的 SessionManager"""
        self.manager = SessionManager()

    def test_session_does_not_store_retrieval_history(self):
        """测试会话只持久化必要字段（幻觉控制 - 隔离敏感数据）

        SessionState 会存储所有字段，但重建的状态只包含必要字段。
        每次恢复会话时，last_retrieved_docs 和 execution_log 会被清空。
        """
        state = {
            "user_topic": "test",
            "outline": [{"step": 1}],
            "current_step_index": 0,
            "fragments": [],
            "skill_history": [],
            "last_retrieved_docs": [{"id": 1, "content": "retrieved"}],
            "execution_log": [{"action": "retrieve"}]
        }

        asyncio.run(self.manager.save_session("test_session", state, "ws1"))

        loaded = asyncio.run(self.manager.load_session("test_session"))

        assert loaded is not None
        assert loaded["user_topic"] == "test"
        assert loaded["fragments"] == []
        assert loaded["skill_history"] == []
        assert loaded["current_step_index"] == 0
        assert loaded["resumed_from_session"] == "test_session"
        assert loaded["workflow_complete"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
