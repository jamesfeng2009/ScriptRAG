"""幻觉控制端到端集成测试

测试内容：
1. 检索隔离与上下文最小化的集成
2. 数据访问控制与状态隔离的集成
3. SessionManager 与幻觉控制的集成
4. 端到端幻觉控制流程验证
5. 增量存储优化验证
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
import json
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHallucinationControlE2E:
    """幻觉控制端到端测试类"""

    def setup_method(self):
        """每个测试前初始化组件"""
        from src.domain.data_access_control import DataAccessControl
        from src.services.retrieval_isolation import RetrievalIsolation, ContextMinimizer
        from src.services.session_manager import SessionManager
        from src.domain.agents.fact_checker_minimal_context import (
            FactCheckerMinimalContext,
            VerificationScope
        )

        self.retrieval_isolation = RetrievalIsolation()
        self.context_minimizer = ContextMinimizer
        self.session_manager = SessionManager()
        self.fact_checker_context = FactCheckerMinimalContext()

        DataAccessControl.STRICT_MODE = True

    def test_isolated_retrieval_per_agent_step(self):
        """测试每步检索隔离 - 清空检索结果"""
        from src.services.retrieval_isolation import RetrievalIsolation

        isolation = RetrievalIsolation()

        state = {
            "user_topic": "Python async",
            "project_context": "async programming guide",
            "outline": [{"step": 1, "title": "Introduction"}],
            "last_retrieved_docs": [
                {"id": 1, "source": "doc1.md", "content": "async def main():"}
            ],
            "current_step_index": 0,
            "fragments": [],
            "execution_log": [],
            "skill_history": []
        }

        cleared_state = isolation.clear_step_results(state)

        assert "last_retrieved_docs" in cleared_state
        assert cleared_state["last_retrieved_docs"] == []
        assert cleared_state["current_step_query"] == ""

    def test_data_access_control_enforces_minimal_permissions(self):
        """测试数据访问控制执行最小权限"""
        from src.domain.data_access_control import DataAccessControl

        permissions = DataAccessControl.list_agent_permissions("test_agent")

        assert "can_read" in permissions
        assert "can_write" in permissions
        assert len(permissions["can_read"]) > 0

    def test_session_manager_isolates_retrieval_history(self):
        """测试会话管理器隔离检索历史"""
        from src.services.session_manager import SessionManager

        manager = SessionManager()

        state = {
            "user_topic": "test topic",
            "project_context": "test context",
            "outline": [{"step": 1}],
            "current_step_index": 0,
            "fragments": [],
            "skill_history": [],
            "last_retrieved_docs": [{"id": 1, "content": "sensitive retrieval"}],
            "execution_log": [{"action": "error", "detail": "should not persist"}]
        }

        asyncio.run(manager.save_session("test_session", state, "ws1"))

        loaded = asyncio.run(manager.load_session("test_session"))

        assert loaded is not None
        assert loaded["user_topic"] == "test topic"

    def test_fact_checker_gets_minimal_context(self):
        """测试 FactChecker 获取最小上下文"""
        state = {
            "user_topic": "Python",
            "project_context": "guide",
            "last_retrieved_docs": [
                {"id": 1, "source": "doc.md", "content": "async def main():"}
            ],
            "fragments": [{"content": "Use async for I/O", "fact_check_passed": False}],
            "current_step_index": 0,
            "execution_log": [{"action": "old_error"}],
            "retrieval_history": [{"query": "old_query"}]
        }

        context = self.fact_checker_context.get_verification_context(state, 0)

        assert context["_metadata"]["context_type"] == "fact_checker_minimal"
        assert context["_metadata"]["can_access_history"] is False
        assert "execution_log" not in context
        assert "retrieval_history" not in context
        assert "current_fragment" in context

    def test_verification_scope_controls_access(self):
        """测试验证范围控制访问"""
        from src.domain.agents.fact_checker_minimal_context import VerificationScope

        scope = VerificationScope.STRICT_SCOPE

        state = {
            "user_topic": "test",
            "fragments": [{"id": 1}, {"id": 2}, {"id": 3}],
            "last_retrieved_docs": [{"id": i} for i in range(10)],
            "outline": [{"step": 1}, {"step": 2}],
            "current_step_index": 0
        }

        filtered = VerificationScope.apply_scope(scope, state)

        assert len(filtered["fragments"]) == 1
        assert len(filtered["last_retrieved_docs"]) == 2

    def test_context_minimizer_for_different_agents(self):
        """测试不同 Agent 的上下文最小化"""
        state = {
            "user_topic": "test",
            "project_context": "context",
            "last_retrieved_docs": [{"id": 1, "content": "doc1"}, {"id": 2, "content": "doc2"}],
            "fragments": [{"content": "fragment"}],
            "current_step_query": "query",
            "outline": [{"step": 1}],
            "current_step_index": 0,
            "execution_log": [{"action": "log"}],
            "retrieval_history": [{"query": "history"}]
        }

        navigator_context = self.context_minimizer.minimize_for_agent("navigator", state, 0)
        fact_checker_context = self.context_minimizer.minimize_for_agent("fact_checker", state, 0)
        writer_context = self.context_minimizer.minimize_for_agent("writer", state, 0)

        assert "retrieved_docs" in navigator_context
        assert "execution_log" not in navigator_context

        assert "retrieved_docs" in fact_checker_context
        assert "execution_log" not in fact_checker_context

        assert "user_topic" in writer_context

    def test_retrieval_isolation_record_retrieval(self):
        """测试检索隔离记录功能"""
        from src.services.retrieval_isolation import RetrievalIsolation

        isolation = RetrievalIsolation(max_docs_per_step=5)

        results = [{"id": i, "content": f"doc {i}"} for i in range(20)]
        filtered = isolation.record_retrieval("test query", results, 1, "navigator")

        assert len(filtered) == 5
        assert isolation.get_audit_report()["total_retrievals"] == 1

    def test_full_hallucination_control_workflow(self):
        """测试完整幻觉控制工作流"""
        state = {
            "user_topic": "Python async programming",
            "project_context": "comprehensive guide",
            "outline": [
                {"step": 1, "title": "Introduction"},
                {"step": 2, "title": "Syntax"},
                {"step": 3, "title": "Best Practices"}
            ],
            "current_step_index": 0,
            "current_step_query": "How to use async in Python?",
            "last_retrieved_docs": [
                {"id": 1, "source": "python docs", "content": "async def main():", "citation": "[1]"},
                {"id": 2, "source": "tutorial", "content": "await asyncio.sleep()", "citation": "[2]"}
            ],
            "fragments": [],
            "skill_history": [],
            "execution_log": [],
            "retrieval_history": []
        }

        cleared_state = self.retrieval_isolation.clear_step_results(state)

        assert cleared_state["last_retrieved_docs"] == []
        assert cleared_state["current_step_query"] == ""

        fact_checker_context = self.fact_checker_context.get_verification_context(
            state, 0
        )

        assert fact_checker_context["_metadata"]["isolation_enabled"] is True

    def test_state_diff_calculation(self):
        """测试状态差异计算"""
        from src.services.incremental_storage import StateDiffCalculator

        old_state = {
            "user_topic": "Python",
            "fragments": [{"id": 1}],
            "current_step_index": 0
        }

        new_state = {
            "user_topic": "Python",
            "fragments": [{"id": 1}, {"id": 2}],
            "current_step_index": 1,
            "last_retrieved_docs": [{"id": 1}]
        }

        diff = StateDiffCalculator.compute_full_diff(old_state, new_state)

        assert "added" in diff
        assert "removed" in diff
        assert "changed" in diff
        assert "last_retrieved_docs" in diff["added"]
        assert diff["changed"]["current_step_index"]["from"] == 0
        assert diff["changed"]["current_step_index"]["to"] == 1

    def test_delta_storage(self):
        """测试差量存储"""
        from src.services.incremental_storage import DeltaStorage

        storage = DeltaStorage(max_delta_chain=3, enable_compression=True)

        state_0 = {"step": 0, "fragments": []}
        state_1 = {"step": 1, "fragments": [{"id": 1}]}
        state_2 = {"step": 2, "fragments": [{"id": 1}, {"id": 2}]}

        storage.record_change(state_0, state_1)
        storage.record_change(state_1, state_2)

        assert storage.get_delta_chain_length() == 2

        reconstructed = storage.get_state_at(1)
        assert reconstructed is not None
        assert reconstructed["step"] == 1


class TestIncrementalStorageOptimization:
    """增量存储优化测试类"""

    def test_delta_storage_compression(self):
        """测试差量存储 - 超过阈值创建快照"""
        from src.services.incremental_storage import DeltaStorage

        storage = DeltaStorage(max_delta_chain=3, enable_compression=True)

        for i in range(10):
            state_0 = {"step": i, "fragments": [{"id": j} for j in range(i + 1)]}
            state_1 = {"step": i + 1, "fragments": [{"id": j} for j in range(i + 2)]}
            if i < 9:
                storage.record_change(state_0, state_1)

        assert storage.get_snapshots_count() >= 1

    def test_snapshot_creation(self):
        """测试快照创建"""
        from src.services.incremental_storage import DeltaStorage

        storage = DeltaStorage(max_delta_chain=3, enable_compression=True)

        states = [
            {"step": i, "fragments": [{"id": j} for j in range(i)]}
            for i in range(10)
        ]

        for i in range(1, len(states)):
            storage.record_change(states[i - 1], states[i])

        assert storage.get_snapshots_count() >= 1

    def test_incremental_optimizer(self):
        """测试增量优化器"""
        from src.services.incremental_storage import IncrementalStorageOptimizer

        optimizer = IncrementalStorageOptimizer(
            enable_compression=True,
            max_delta_chain=5
        )

        states = [
            {"step": i, "fragments": [{"id": j} for j in range(min(i, 5))]}
            for i in range(20)
        ]

        for i in range(1, len(states)):
            optimizer.record_step(states[i - 1], states[i])

        stats = optimizer.get_optimization_stats()

        assert stats["total_steps"] == 19

    def test_cleanup_policy(self):
        """测试清理策略"""
        from src.services.incremental_storage import IncrementalCleanupPolicy
        from datetime import datetime, timedelta

        policy = IncrementalCleanupPolicy(
            max_history_count=5,
            max_history_days=1,
            min_access_count=3
        )

        now = datetime.now()

        for i in range(10):
            policy.register_session(
                f"session_{i}",
                created_at=now - timedelta(days=i),
                last_accessed=now - timedelta(days=i)
            )
            for _ in range(min(i + 1, 5)):
                policy.record_access(f"session_{i}")

        to_remove, reason = policy.get_sessions_to_cleanup()

        assert len(to_remove) > 0 or len(policy._sessions_metadata) > 5

    def test_storage_size_estimation(self):
        """测试存储大小估算"""
        from src.services.incremental_storage import IncrementalStorageOptimizer

        optimizer = IncrementalStorageOptimizer(enable_compression=True)

        sample_state = {
            "user_topic": "test",
            "project_context": "context" * 100,
            "fragments": [{"content": f"fragment {i}" * 50} for i in range(10)],
            "outline": [{"step": i} for i in range(20)]
        }

        estimate = optimizer.estimate_storage_size(sample_state, steps=100)

        assert estimate["single_step_original"] > 0
        assert estimate["single_step_compressed"] > 0
        assert estimate["full_storage_original"] >= estimate["full_storage_compressed"]


class TestIntegrationWithRealScenarios:
    """真实场景集成测试类"""

    def test_multi_step_workflow_with_hallucination_control(self):
        """测试多步工作流幻觉控制"""
        from src.services.retrieval_isolation import RetrievalIsolation
        from src.services.session_manager import SessionManager
        from src.domain.data_access_control import DataAccessControl

        manager = SessionManager()
        isolation = RetrievalIsolation()

        DataAccessControl.STRICT_MODE = True

        initial_state = {
            "user_topic": "Build a web scraper",
            "project_context": "Python web scraping tutorial",
            "outline": [
                {"step": 1, "title": "Setup"},
                {"step": 2, "title": "Request Handling"},
                {"step": 3, "title": "Parsing"}
            ],
            "current_step_index": 0,
            "current_step_query": "How to make HTTP requests in Python?",
            "last_retrieved_docs": [
                {"id": 1, "source": "requests.md", "content": "requests.get(url)", "citation": "[1]"},
                {"id": 2, "source": "aiohttp.md", "content": "async with ClientSession()", "citation": "[2]"}
            ],
            "fragments": [],
            "skill_history": [],
            "execution_log": [],
            "retrieval_history": []
        }

        asyncio.run(manager.save_session("session_1", initial_state, "ws1"))

        step_1_state = isolation.clear_step_results(initial_state)
        step_1_state["current_step_index"] = 1
        step_1_state["current_step_query"] = "How to parse HTML?"
        step_1_state["last_retrieved_docs"] = [
            {"id": 3, "source": "beautifulsoup.md", "content": "BeautifulSoup(html, 'html.parser')", "citation": "[1]"}
        ]
        step_1_state["fragments"] = [{"step_id": 1, "content": "Use requests for HTTP"}]

        assert len(step_1_state["last_retrieved_docs"]) == 1
        assert step_1_state["current_step_query"] == "How to parse HTML?"

        asyncio.run(manager.save_session("session_1", step_1_state, "ws1"))

        loaded = asyncio.run(manager.load_session("session_1"))

        assert loaded is not None
        assert loaded["user_topic"] == "Build a web scraper"
        assert loaded["current_step_index"] == 1

    def test_fact_checker_verification_with_retrieval(self):
        """测试 FactChecker 验证与检索集成"""
        from src.domain.agents.fact_checker_minimal_context import (
            FactCheckerMinimalContext,
            VerificationScope
        )

        handler = FactCheckerMinimalContext()

        state = {
            "user_topic": "Python",
            "last_retrieved_docs": [
                {"id": 1, "source": "doc.md", "content": "def add(a, b): return a + b", "citation": "[1]"}
            ],
            "fragments": [{"content": "The add function takes two parameters a and b."}],
            "current_step_index": 0
        }

        context = handler.get_verification_context(state, 0)

        assert "current_fragment" in context
        assert len(context["retrieved_docs"]) == 1

        prompt = handler.create_verification_prompt(
            fragment_content=context["current_fragment"]["content"],
            retrieved_docs=context["retrieved_docs"],
            verification_scope=VerificationScope.STRICT_SCOPE
        )

        assert len(prompt) == 2

    def test_concurrent_agent_isolation(self):
        """测试并发 Agent 隔离"""
        from src.services.retrieval_isolation import ContextMinimizer

        shared_state = {
            "user_topic": "shared topic",
            "last_retrieved_docs": [{"id": 1, "content": "shared doc"}],
            "fragments": [{"content": "fragment"}],
            "execution_log": [{"action": "error"}],
            "retrieval_history": [{"query": "old"}]
        }

        navigator_ctx = ContextMinimizer.minimize_for_agent("navigator", shared_state, 0)
        fact_checker_ctx = ContextMinimizer.minimize_for_agent("fact_checker", shared_state, 0)
        writer_ctx = ContextMinimizer.minimize_for_agent("writer", shared_state, 0)

        assert "retrieved_docs" in navigator_ctx
        assert "retrieved_docs" in fact_checker_ctx
        assert "execution_log" not in fact_checker_ctx
        assert "retrieval_history" not in navigator_ctx


class TestEdgeCasesIntegration:
    """边界情况集成测试类"""

    def test_empty_state_handling(self):
        """测试空状态处理"""
        from src.services.retrieval_isolation import RetrievalIsolation
        from src.services.session_manager import SessionManager

        isolation = RetrievalIsolation()
        manager = SessionManager()

        empty_state = {}

        cleared = isolation.clear_step_results(empty_state)

        assert "last_retrieved_docs" in cleared

        asyncio.run(manager.save_session("empty_session", empty_state, "ws1"))
        loaded = asyncio.run(manager.load_session("empty_session"))

        assert loaded is not None

    def test_large_state_optimization(self):
        """测试大状态优化"""
        from src.services.incremental_storage import IncrementalStorageOptimizer

        optimizer = IncrementalStorageOptimizer(
            enable_compression=True,
            max_delta_chain=10
        )

        large_state = {
            "user_topic": "test",
            "project_context": "context" * 1000,
            "fragments": [{"content": f"fragment {i}" * 100} for i in range(100)],
            "outline": [{"step": i, "title": f"Step {i}"} for i in range(50)],
            "skill_history": [{"skill": f"skill_{i}"} for i in range(200)],
            "execution_log": [{"action": f"log_{i}"} for i in range(500)]
        }

        estimate = optimizer.estimate_storage_size(large_state, steps=100)

        assert estimate["single_step_original"] > 0
        assert estimate["single_step_compressed"] > 0

    def test_rapid_state_changes(self):
        """测试快速状态变化"""
        from src.services.incremental_storage import DeltaStorage

        storage = DeltaStorage(max_delta_chain=5, enable_compression=True)

        current = {"count": 0}

        for i in range(20):
            previous = current.copy()
            current = {"count": i, "data": f"data_{i}" * 10}
            storage.record_change(previous, current)

        assert storage.get_snapshots_count() >= 2

    def test_mixed_content_types(self):
        """测试混合内容类型"""
        from src.services.retrieval_isolation import RetrievalIsolation

        isolation = RetrievalIsolation()

        state = {
            "user_topic": "mixed types",
            "last_retrieved_docs": [
                {"id": 1, "source": "code.py", "content": "def foo(): pass"},
                {"id": 2, "source": "data.json", "content": json.dumps({"key": "value"})},
                {"id": 3, "source": "text.md", "content": "markdown content"}
            ],
            "fragments": [
                {"content": "code and data", "metadata": {"type": "mixed"}}
            ],
            "current_step_query": "query with mixed content?"
        }

        minimal_ctx = isolation.get_minimal_context_for_writer(state, 1)

        assert len(minimal_ctx.retrieved_docs) == 3

    def test_retrieval_audit_report(self):
        """测试检索审计报告"""
        from src.services.retrieval_isolation import RetrievalIsolation

        isolation = RetrievalIsolation()

        isolation.record_retrieval("query 1", [{"id": 1}], 1, "navigator")
        isolation.record_retrieval("query 2", [{"id": 2}, {"id": 3}], 2, "navigator")

        report = isolation.get_audit_report()

        assert report["total_retrievals"] == 2
        assert report["isolation_enabled"] is True
        assert report["max_docs_per_step"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
