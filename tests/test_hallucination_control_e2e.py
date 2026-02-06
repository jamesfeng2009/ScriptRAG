"""幻觉控制端到端集成测试

测试内容：
1. 检索隔离与上下文最小化的集成
2. 数据访问控制与状态隔离的集成
3. 端到端幻觉控制流程验证
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List, Optional
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestHallucinationControlE2E:
    """幻觉控制端到端测试类"""

    def setup_method(self):
        """每个测试前初始化组件"""
        from src.domain.data_access_control import DataAccessControl
        from src.services.retrieval_isolation import RetrievalIsolation, ContextMinimizer
        from src.domain.agents.fact_checker_minimal_context import (
            FactCheckerMinimalContext,
            VerificationScope
        )

        self.retrieval_isolation = RetrievalIsolation()
        self.context_minimizer = ContextMinimizer
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
            "retrieved_docs": [
                {"id": 1, "source": "doc1.md", "content": "async def main():"}
            ],
            "current_step_index": 0,
            "fragments": [],
            "execution_log": [],
            "skill_history": []
        }

        cleared_state = isolation.clear_step_results(state)

        assert "retrieved_docs" in cleared_state
        assert cleared_state["retrieved_docs"] == []
        assert cleared_state["current_step_query"] == ""

    def test_data_access_control_enforces_minimal_permissions(self):
        """测试数据访问控制执行最小权限"""
        from src.domain.data_access_control import DataAccessControl

        permissions = DataAccessControl.list_agent_permissions("test_agent")

        assert "can_read" in permissions
        assert "can_write" in permissions
        assert len(permissions["can_read"]) > 0

    def test_fact_checker_gets_minimal_context(self):
        """测试 FactChecker 获取最小上下文"""
        state = {
            "user_topic": "Python",
            "project_context": "guide",
            "retrieved_docs": [
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
            "retrieved_docs": [{"id": i} for i in range(10)],
            "outline": [{"step": 1}, {"step": 2}],
            "current_step_index": 0
        }

        filtered = VerificationScope.apply_scope(scope, state)

        assert len(filtered["fragments"]) == 1
        assert len(filtered["retrieved_docs"]) == 2

    def test_context_minimizer_for_different_agents(self):
        """测试不同 Agent 的上下文最小化"""
        state = {
            "user_topic": "test",
            "project_context": "context",
            "retrieved_docs": [{"id": 1, "content": "doc1"}, {"id": 2, "content": "doc2"}],
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
            "retrieved_docs": [
                {"id": 1, "source": "python docs", "content": "async def main():", "citation": "[1]"},
                {"id": 2, "source": "tutorial", "content": "await asyncio.sleep()", "citation": "[2]"}
            ],
            "fragments": [],
            "skill_history": [],
            "execution_log": [],
            "retrieval_history": []
        }

        cleared_state = self.retrieval_isolation.clear_step_results(state)

        assert cleared_state["retrieved_docs"] == []
        assert cleared_state["current_step_query"] == ""

        fact_checker_context = self.fact_checker_context.get_verification_context(
            state, 0
        )

        assert fact_checker_context["_metadata"]["isolation_enabled"] is True

    def test_fact_checker_verification_with_retrieval(self):
        """测试 FactChecker 验证与检索集成"""
        from src.domain.agents.fact_checker_minimal_context import (
            FactCheckerMinimalContext
        )

        handler = FactCheckerMinimalContext()

        state = {
            "user_topic": "Python",
            "retrieved_docs": [
                {"id": 1, "source": "doc.md", "content": "def add(a, b): return a + b", "citation": "[1]"}
            ],
            "fragments": [{"content": "The add function takes two parameters a and b."}],
            "current_step_index": 0
        }

        context = handler.get_verification_context(state, 0)

        assert "current_fragment" in context
        assert len(context["retrieved_docs"]) == 1

        messages = handler.create_verification_messages(
            fragment_content=context["current_fragment"]["content"],
            retrieved_docs=context["retrieved_docs"]
        )

        assert len(messages) == 2

    def test_concurrent_agent_isolation(self):
        """测试并发 Agent 隔离"""
        from src.services.retrieval_isolation import ContextMinimizer

        shared_state = {
            "user_topic": "shared topic",
            "retrieved_docs": [{"id": 1, "content": "shared doc"}],
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

        isolation = RetrievalIsolation()

        empty_state = {}

        cleared = isolation.clear_step_results(empty_state)

        assert "retrieved_docs" in cleared

    def test_mixed_content_types(self):
        """测试混合内容类型"""
        from src.services.retrieval_isolation import RetrievalIsolation

        isolation = RetrievalIsolation()

        state = {
            "user_topic": "mixed types",
            "retrieved_docs": [
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
