"""检索隔离模块集成测试

测试内容：
1. 检索结果隔离
2. 最小化上下文获取
3. ContextMinimizer 功能
4. 幻觉控制场景
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.retrieval_isolation import (
    RetrievalIsolation,
    RetrievalRecord,
    MinimalContext,
    ContextMinimizer,
    IsolationLevel,
)


class TestRetrievalIsolation:
    """检索隔离测试类"""

    def setup_method(self):
        """每个测试前创建新的隔离器"""
        self.isolation = RetrievalIsolation(max_docs_per_step=5)

    def test_init(self):
        """测试初始化"""
        assert self.isolation.max_docs_per_step == 5
        assert self.isolation._isolation_enabled is True
        assert self.isolation.retrieval_history == []

    def test_record_retrieval(self):
        """测试记录检索结果"""
        results = [{"id": i, "content": f"doc {i}"} for i in range(10)]

        filtered = self.isolation.record_retrieval(
            query="test query",
            results=results,
            step_index=0
        )

        assert len(filtered) == 5
        assert len(self.isolation.retrieval_history) == 1

        record = self.isolation.retrieval_history[0]
        assert record.step_index == 0
        assert record.query == "test query"
        assert record.doc_count == 5

    def test_multiple_retrievals(self):
        """测试多次检索"""
        for i in range(3):
            results = [{"id": i, "step": i}]
            self.isolation.record_retrieval(
                query=f"query {i}",
                results=results,
                step_index=i
            )

        assert len(self.isolation.retrieval_history) == 3

        audit = self.isolation.get_audit_report()
        assert audit["total_retrievals"] == 3
        assert len(audit["retrieval_history"]) == 3

    def test_disable_isolation(self):
        """测试禁用隔离"""
        self.isolation.disable_isolation()

        results = [{"id": i} for i in range(10)]
        filtered = self.isolation.record_retrieval(
            query="test",
            results=results,
            step_index=0
        )

        assert len(filtered) == 10
        assert self.isolation._isolation_enabled is False

    def test_enable_isolation(self):
        """测试启用隔离"""
        self.isolation.disable_isolation()
        self.isolation.enable_isolation()

        assert self.isolation._isolation_enabled is True

    def test_clear_step_results(self):
        """测试清空步骤结果"""
        state = {
            "last_retrieved_docs": [{"id": 1}, {"id": 2}],
            "current_step_query": "test query",
            "current_step_index": 0
        }

        cleared = self.isolation.clear_step_results(state)

        assert cleared["last_retrieved_docs"] == []
        assert cleared["current_step_query"] == ""

    def test_prepare_retrieval(self):
        """测试准备检索"""
        state = {
            "current_step_index": 5
        }

        context = self.isolation.prepare_retrieval(state, "test query")

        assert context.current_step_query == "test query"
        assert context.doc_count == 0


class TestMinimalContext:
    """最小上下文测试类"""

    def test_minimal_context_creation(self):
        """测试创建最小上下文"""
        context = MinimalContext(
            current_step_query="async programming benefits",
            retrieved_docs=[{"id": 1}, {"id": 2}],
            doc_count=2,
            sources=["doc1.md", "doc2.md"],
            citations=["[1]", "[2]"]
        )

        assert context.current_step_query == "async programming benefits"
        assert context.doc_count == 2
        assert len(context.sources) == 2


class TestContextMinimizer:
    """上下文最小化器测试类"""

    def test_minimize_for_fact_checker(self):
        """测试 FactChecker 上下文最小化"""
        state = {
            "user_topic": "Python async",
            "project_context": "Async programming guide",
            "outline": [{"step_id": 1}, {"step_id": 2}],
            "last_retrieved_docs": [{"id": i, "source": f"doc{i}.md"} for i in range(10)],
            "fragments": [{"content": "fragment content", "skill_used": "python"}],
            "current_step_index": 0
        }

        minimized = ContextMinimizer.minimize_for_agent(
            agent_type="fact_checker",
            state=state,
            current_step_index=0
        )

        assert minimized["user_topic"] == "Python async"
        assert len(minimized["retrieved_docs"]) == 3
        assert minimized["doc_count"] == 3
        assert minimized["current_fragment"] is not None
        assert "content" in minimized["current_fragment"]

    def test_minimize_for_writer(self):
        """测试 Writer 上下文最小化"""
        state = {
            "user_topic": "test",
            "last_retrieved_docs": [{"id": i, "source": f"doc{i}.md"} for i in range(10)],
            "fragments": [],
            "current_step_index": 0
        }

        minimized = ContextMinimizer.minimize_for_agent(
            agent_type="writer",
            state=state,
            current_step_index=0
        )

        assert minimized["doc_count"] == 5
        assert minimized["current_fragment"] is None

    def test_get_fact_checker_context(self):
        """测试获取 FactChecker 专用上下文"""
        state = {
            "user_topic": "async programming",
            "project_context": "guide",
            "outline": [{"step": 1}],
            "last_retrieved_docs": [{"id": 1, "source": "doc.md", "content": "async improves I/O"}],
            "fragments": [{"content": "fragment", "skill_used": "python"}],
            "current_step_index": 0
        }

        context = ContextMinimizer.get_fact_checker_context(
            state=state,
            current_fragment_index=0
        )

        assert context["verification_scope"]["can_access_history"] is False
        assert context["verification_scope"]["require_citations"] is True
        assert context["fragment_to_verify"] is not None

    def test_fallback_to_navigator_config(self):
        """测试未知 Agent 回退到 navigator 配置"""
        state = {
            "user_topic": "test",
            "last_retrieved_docs": [{"id": i} for i in range(15)],
            "fragments": [],
            "current_step_index": 0
        }

        minimized = ContextMinimizer.minimize_for_agent(
            agent_type="unknown_agent",
            state=state,
            current_step_index=0
        )

        assert len(minimized["retrieved_docs"]) == 10


class TestHallucinationControl:
    """幻觉控制场景测试"""

    def test_isolation_prevents_history_access(self):
        """测试隔离防止访问历史检索结果"""
        isolation = RetrievalIsolation(max_docs_per_step=5)

        state = {
            "last_retrieved_docs": [{"id": 1, "content": "old result"}],
            "current_step_index": 1
        }

        context = isolation.get_minimal_context_for_factchecker(
            state=state,
            current_fragment_index=0
        )

        assert context.retrieved_docs == [{"id": 1, "content": "old result"}]

        isolation.record_retrieval(
            query="new query",
            results=[{"id": 2, "content": "new result"}],
            step_index=1
        )

        assert len(isolation.retrieval_history) == 1
        assert isolation.retrieval_history[0].doc_count == 1

        assert isolation.get_audit_report()["total_retrievals"] == 1

    def test_minimal_context_reduces_token_usage(self):
        """测试最小上下文减少 token 使用"""
        state = {
            "user_topic": "test",
            "project_context": "test context",
            "outline": [{"step": i} for i in range(100)],
            "last_retrieved_docs": [{"id": i, "content": f"doc {i}" * 100} for i in range(50)],
            "fragments": [{"content": f"fragment {i}" * 50} for i in range(100)],
            "execution_log": [{"action": f"log {i}"} for i in range(200)],
            "current_step_index": 0
        }

        minimized = ContextMinimizer.minimize_for_agent(
            agent_type="fact_checker",
            state=state,
            current_step_index=0
        )

        assert len(minimized["retrieved_docs"]) == 3
        assert minimized["current_fragment"] is not None
        assert "execution_log" not in minimized
        assert len(minimized["user_topic"]) > 0

    def test_citations_required_for_fact_checker(self):
        """测试 FactChecker 需要引用标注"""
        state = {
            "user_topic": "test",
            "project_context": "test",
            "last_retrieved_docs": [
                {"id": 1, "source": "doc1.md", "citation": "[1]"},
                {"id": 2, "source": "doc2.md", "citation": "[2]"}
            ],
            "fragments": [],
            "current_step_index": 0
        }

        context = ContextMinimizer.get_fact_checker_context(
            state=state,
            current_fragment_index=0
        )

        assert context["verification_scope"]["require_citations"] is True


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_retrieval_history(self):
        """测试空检索历史"""
        isolation = RetrievalIsolation()

        audit = isolation.get_audit_report()

        assert audit["total_retrievals"] == 0
        assert audit["retrieval_history"] == []

    def test_empty_state_minimization(self):
        """测试空状态最小化"""
        state = {}

        minimized = ContextMinimizer.minimize_for_agent(
            agent_type="fact_checker",
            state=state,
            current_step_index=0
        )

        assert minimized["user_topic"] == ""
        assert minimized["retrieved_docs"] == []
        assert minimized["current_fragment"] is None

    def test_out_of_bounds_step(self):
        """测试步骤越界"""
        state = {
            "outline": [{"step": 1}],
            "current_step_index": 10
        }

        minimized = ContextMinimizer.minimize_for_agent(
            agent_type="writer",
            state=state,
            current_step_index=10
        )

        assert minimized["current_step"] == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
