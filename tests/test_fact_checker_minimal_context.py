"""FactChecker 最小上下文模块测试

测试内容：
1. FactCheckerMinimalContext 功能
2. VerificationScope 范围控制
3. 幻觉控制场景
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.agents.fact_checker_minimal_context import (
    FactCheckerMinimalContext,
    VerificationScope,
)


class TestFactCheckerMinimalContext:
    """FactChecker 最小上下文测试类"""

    def setup_method(self):
        """每个测试前创建新的处理器"""
        self.handler = FactCheckerMinimalContext()

    def test_get_verification_context(self):
        """测试获取验证上下文"""
        state = {
            "user_topic": "Python async",
            "project_context": "async guide",
            "outline": [{"step": 1}, {"step": 2}],
            "last_retrieved_docs": [
                {"id": 1, "source": "doc1.md", "content": "async improves I/O", "citation": "[1]"},
                {"id": 2, "source": "doc2.md", "content": "async/await syntax", "citation": "[2]"}
            ],
            "fragments": [{"content": "fragment 1", "skill_used": "python"}],
            "current_step_index": 0
        }

        context = self.handler.get_verification_context(
            state=state,
            current_fragment_index=0
        )

        assert context["user_topic"] == "Python async"
        assert context["_metadata"]["context_type"] == "fact_checker_minimal"
        assert context["_metadata"]["isolation_enabled"] is True
        assert context["_metadata"]["can_access_history"] is False
        assert context["_metadata"]["can_access_retrieval_log"] is False

    def test_should_verify_pending(self):
        """测试应当验证（待验证）"""
        state = {
            "fragments": [
                {"content": "fragment 1", "fact_check_passed": False}
            ]
        }

        should_verify, reason = self.handler.should_verify(state, 0)

        assert should_verify is True
        assert reason == "pending_verification"

    def test_should_verify_already_verified(self):
        """测试不应验证（已验证）"""
        state = {
            "fragments": [
                {"content": "fragment 1", "fact_check_passed": True}
            ]
        }

        should_verify, reason = self.handler.should_verify(state, 0)

        assert should_verify is False
        assert reason == "already_verified"

    def test_should_verify_no_fragments(self):
        """测试不应验证（无片段）"""
        state = {"fragments": []}

        should_verify, reason = self.handler.should_verify(state, 0)

        assert should_verify is False
        assert reason == "no_fragments"

    def test_extract_retrieved_docs_for_verification(self):
        """测试提取验证用检索文档"""
        state = {
            "last_retrieved_docs": [
                {"id": 1, "source": "doc1.md", "content": "content 1", "citation": "[1]"},
                {"id": 2, "source": "doc2.md", "content": "content 2", "citation": "[2]"},
                {"id": 3, "source": "doc3.md", "content": "content 3", "citation": "[3]"}
            ]
        }

        docs = self.handler.extract_retrieved_docs_for_verification(state)

        assert len(docs) == 3
        assert docs[0]["citation"] == "[1]"
        assert docs[0]["content"] == "content 1"

    def test_extract_retrieved_docs_limit(self):
        """测试限制文档数量"""
        state = {
            "last_retrieved_docs": [
                {"id": i, "source": f"doc{i}.md", "content": f"content {i}"}
                for i in range(10)
            ]
        }

        docs = self.handler.extract_retrieved_docs_for_verification(state)

        assert len(docs) == 3

    def test_extract_retrieved_docs_empty(self):
        """测试空检索结果"""
        state = {"last_retrieved_docs": []}

        docs = self.handler.extract_retrieved_docs_for_verification(state)

        assert len(docs) == 0

    def test_format_sources_for_verification(self):
        """测试格式化源文档"""
        retrieved_docs = [
            {"source": "doc1.md", "content": "async improves I/O performance", "citation": "[1]"},
            {"source": "doc2.md", "content": "async/await syntax overview", "citation": "[2]"}
        ]

        formatted = self.handler.format_sources_for_verification(retrieved_docs)

        assert "[1] doc1.md:" in formatted
        assert "[2] doc2.md:" in formatted
        assert "async improves I/O" in formatted

    def test_format_sources_empty(self):
        """测试格式化空源文档"""
        formatted = self.handler.format_sources_for_verification([])

        assert "无检索文档" in formatted

    def test_create_verification_prompt(self):
        """测试创建验证提示"""
        fragment_content = "Use async to improve I/O performance"
        retrieved_docs = [
            {"source": "doc1.md", "content": "async improves I/O", "citation": "[1]"}
        ]

        messages = self.handler.create_verification_prompt(
            fragment_content=fragment_content,
            retrieved_docs=retrieved_docs,
            verification_scope=VerificationScope.DEFAULT_SCOPE
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "async improves I/O" in messages[1]["content"]
        assert "待验证的片段内容" in messages[1]["content"]

    def test_parse_verification_result_valid(self):
        """测试解析有效结果"""
        result = self.handler.parse_verification_result("VALID")

        assert result[0] is True
        assert result[1] == []

    def test_parse_verification_result_invalid(self):
        """测试解析无效结果"""
        response = """INVALID
- 幻觉: 函数 async_handler 未在源文档中找到
- 幻觉: 参数 timeout 类型错误"""

        is_valid, hallucinations = self.handler.parse_verification_result(response)

        assert is_valid is False
        assert len(hallucinations) == 2
        assert "async_handler" in hallucinations[0]

    def test_parse_verification_result_unparseable(self):
        """测试解析无法解析的结果"""
        response = "Some unclear response"

        is_valid, hallucinations = self.handler.parse_verification_result(response)

        assert is_valid is True
        assert hallucinations == []

    def test_create_verification_record(self):
        """测试创建验证记录"""
        record = self.handler.create_verification_record(
            fragment_index=0,
            is_valid=False,
            hallucinations=["function not found"],
            doc_count=2,
            context_metadata={"isolation_enabled": True}
        )

        assert record["fragment_index"] == 0
        assert record["is_valid"] is False
        assert record["hallucination_count"] == 1
        assert record["documents_used"] == 2
        assert record["isolation_enabled"] is True


class TestVerificationScope:
    """验证范围测试类"""

    def test_get_default_scope(self):
        """测试获取默认范围"""
        scope = VerificationScope.get_scope("default")

        assert scope["can_access_history"] is False
        assert scope["can_access_retrieval_log"] is False
        assert scope["max_retrieval_docs"] == 3

    def test_get_strict_scope(self):
        """测试获取严格范围"""
        scope = VerificationScope.get_scope("strict")

        assert scope["max_retrieval_docs"] == 2
        assert scope["max_fragment_length"] == 5000

    def test_apply_scope_default(self):
        """测试应用默认范围"""
        state = {
            "user_topic": "test",
            "project_context": "context",
            "current_step_query": "query",
            "fragments": [{"id": 1}, {"id": 2}, {"id": 3}],
            "last_retrieved_docs": [{"id": i} for i in range(10)],
            "outline": [{"step": 1}, {"step": 2}],
            "current_step_index": 1
        }

        scope = VerificationScope.DEFAULT_SCOPE
        filtered = VerificationScope.apply_scope(scope, state)

        assert len(filtered["fragments"]) == 1
        assert len(filtered["last_retrieved_docs"]) == 3
        assert "current_step" in filtered

    def test_apply_scope_with_history_access(self):
        """测试允许访问历史"""
        state = {
            "user_topic": "test",
            "project_context": "context",
            "fragments": [{"id": 1}, {"id": 2}],
            "last_retrieved_docs": [{"id": 1}],
            "outline": [{"step": 1}, {"step": 2}],
            "current_step_index": 0
        }

        scope = {
            "can_access_history": True,
            "can_access_other_fragments": True,
            "max_retrieval_docs": 3
        }

        filtered = VerificationScope.apply_scope(scope, state)

        assert len(filtered["fragments"]) == 2
        assert "outline" in filtered


class TestHallucinationControlIntegration:
    """幻觉控制集成测试"""

    def test_fact_checker_cannot_access_history(self):
        """测试 FactChecker 无法访问历史"""
        handler = FactCheckerMinimalContext()

        state = {
            "user_topic": "test",
            "last_retrieved_docs": [{"id": 1, "content": "current"}],
            "fragments": [{"content": "current fragment"}],
            "current_step_index": 5,
            "execution_log": [{"action": "old error", "step": 0}],
            "retrieval_history": [{"query": "old query", "step": 0}]
        }

        context = handler.get_verification_context(state, 5)

        assert context["_metadata"]["can_access_history"] is False
        assert context["_metadata"]["can_access_retrieval_log"] is False
        assert "execution_log" not in context
        assert "retrieval_history" not in context

    def test_isolated_retrieval_per_step(self):
        """测试每步隔离检索"""
        handler = FactCheckerMinimalContext()

        state = {
            "user_topic": "test",
            "last_retrieved_docs": [{"id": 1, "content": "step 1 result"}],
            "fragments": [{"content": "step 1 fragment"}],
            "current_step_index": 1
        }

        docs = handler.extract_retrieved_docs_for_verification(state)

        assert len(docs) == 1
        assert docs[0]["content"] == "step 1 result"

    def test_context_reduction_for_fact_checker(self):
        """测试 FactChecker 上下文精简"""
        handler = FactCheckerMinimalContext()

        large_state = {
            "user_topic": "Python async",
            "project_context": "comprehensive guide",
            "outline": [{"step": i} for i in range(50)],
            "last_retrieved_docs": [{"id": i, "content": f"doc {i}" * 100} for i in range(20)],
            "fragments": [{"content": f"fragment {i}" * 50} for i in range(100)],
            "execution_log": [{"action": f"log {i}"} for i in range(500)],
            "current_step_index": 10
        }

        context = handler.get_verification_context(large_state, 10)

        assert context["current_fragment"] is not None
        assert len(context["retrieved_docs"]) == 3
        assert "execution_log" not in context


class TestEdgeCases:
    """边界情况测试"""

    def test_empty_state_context(self):
        """测试空状态上下文"""
        handler = FactCheckerMinimalContext()

        state = {}

        should_verify, reason = handler.should_verify(state, 0)

        assert should_verify is False
        assert reason == "no_fragments"

    def test_none_values_in_state(self):
        """测试状态中的 None 值"""
        handler = FactCheckerMinimalContext()

        state = {
            "user_topic": None,
            "last_retrieved_docs": [None, {"content": "valid"}],
            "fragments": [{"content": None, "fact_check_passed": False}]
        }

        docs = handler.extract_retrieved_docs_for_verification(state)

        assert len(docs) == 1
        assert docs[0]["content"] == "valid"

    def test_fragment_index_out_of_range(self):
        """测试片段索引越界"""
        handler = FactCheckerMinimalContext()

        state = {
            "fragments": [{"content": "fragment 1"}]
        }

        should_verify, reason = handler.should_verify(state, 5)

        assert should_verify is False
        assert reason == "fragment_index_out_of_range"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
