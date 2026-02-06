"""质量评估集成测试

本模块测试 QualityEvalAgent 与 Navigator 的集成。
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Any, Optional

from src.domain.agents.quality_eval import (
    QualityEvalAgent,
    QualityEvaluation,
    QualityLevel,
    RetrievalStatus,
    AdaptiveAction
)
from src.domain.agents.navigator import retrieve_content
from src.domain.models import SharedState, RetrievedDocument, IntentAnalysis, OutlineStep


class TestNavigatorWithQualityEvaluation:
    """测试带质量评估的 Navigator"""

    @pytest.fixture
    def sample_state(self) -> SharedState:
        """创建示例状态"""
        return SharedState(
            user_topic="Python 异步编程",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="介绍异步编程",
                    description="Python 异步编程的基本概念"
                )
            ]
        )

    @pytest.fixture
    def sample_documents(self) -> List[RetrievedDocument]:
        """创建示例文档"""
        return [
            RetrievedDocument(
                content="Python 异步编程使用 async 和 await 关键字实现协程",
                source="docs/python/async.md",
                confidence=0.9,
                metadata={"type": "documentation"},
                summary="异步编程基础"
            ),
            RetrievedDocument(
                content="asyncio 是 Python 标准库中的异步 IO 模块",
                source="docs/python/asyncio.md",
                confidence=0.85,
                metadata={"type": "documentation"},
                summary="asyncio 模块介绍"
            )
        ]

    @pytest.mark.asyncio
    async def test_retrieve_with_quality_evaluation_enabled(
        self,
        sample_state,
        sample_documents
    ):
        """测试启用质量评估的检索"""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.85,
    "relevance_score": 0.9,
    "completeness_score": 0.8,
    "accuracy_score": 0.85,
    "strengths": ["内容高度相关"],
    "weaknesses": [],
    "suggestions": [],
    "needs_refinement": false
}
```""")

        mock_retrieval = MagicMock()
        mock_retrieval.hybrid_retrieve = AsyncMock(return_value=sample_documents)
        mock_retrieval.parallel_retrieve = AsyncMock(return_value=sample_documents)

        mock_parser = MagicMock()
        mock_parser.parse_file = AsyncMock(return_value="parsed content")

        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(return_value="summary")

        result = await retrieve_content(
            state=sample_state,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarizer,
                        enable_intent_parsing=False,
            enable_parallel=False,
            llm_service=mock_llm
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_quality_evaluation_logs_transitions(
        self,
        sample_state,
        sample_documents
    ):
        """测试质量评估日志记录"""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.75,
    "relevance_score": 0.8,
    "completeness_score": 0.7,
    "accuracy_score": 0.75,
    "strengths": ["相关"],
    "weaknesses": [],
    "suggestions": [],
    "needs_refinement": false
}
```""")

        mock_retrieval = MagicMock()
        mock_retrieval.hybrid_retrieve = AsyncMock(return_value=sample_documents)
        mock_retrieval.parallel_retrieve = AsyncMock(return_value=sample_documents)

        mock_parser = MagicMock()
        mock_parser.parse_file = AsyncMock(return_value={"content": "parsed"})

        mock_summarizer = MagicMock()
        mock_summarizer.summarize = AsyncMock(return_value="summary")

        with patch('src.domain.agents.navigator.agent_logger') as mock_logger:
            await retrieve_content(
                state=sample_state,
                retrieval_service=mock_retrieval,
                parser_service=mock_parser,
                summarization_service=mock_summarizer,
                                enable_intent_parsing=False,
                enable_parallel=False,
                llm_service=mock_llm
            )

            assert mock_logger.log_agent_transition.called


class TestAdaptiveActionIntegration:
    """测试自适应行动的集成"""

    def test_adaptive_action_with_intent(self):
        """测试带意图的自适应行动"""
        mock_llm = MagicMock()
        agent = QualityEvalAgent(mock_llm)

        intent = IntentAnalysis(
            primary_intent="学习 Python 异步",
            keywords=["async", "await", "asyncio"],
            search_sources=["rag"],
            confidence=0.9,
            intent_type="informational"
        )

        evaluation = QualityEvaluation(
            overall_score=0.3,
            relevance_score=0.4,
            completeness_score=0.2,
            accuracy_score=0.3,
            quality_level=QualityLevel.POOR,
            retrieval_status=RetrievalStatus.NEEDS_IMPROVEMENT,
            strengths=[],
            weaknesses=["相关性不足"],
            suggestions=["扩大搜索"],
            needs_refinement=True,
            refinement_strategy="broaden_search"
        )

        action = agent.determine_adaptive_action(evaluation, "test query", intent)

        assert action.action_type == "retry"
        assert action.parameters["alternative_keywords"] == intent.keywords
        assert action.parameters["alternative_sources"] == intent.search_sources


class TestQualityEvaluationWithIntent:
    """测试带意图的质量评估"""

    @pytest.fixture
    def mock_llm_service(self):
        """创建模拟 LLM 服务"""
        service = MagicMock()
        service.chat_completion = AsyncMock()
        return service

    @pytest.mark.asyncio
    async def test_evaluate_with_detailed_intent(self, mock_llm_service):
        """测试带详细意图的评估"""
        mock_llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.88,
    "relevance_score": 0.92,
    "completeness_score": 0.85,
    "accuracy_score": 0.87,
    "strengths": ["关键词匹配准确", "数据源正确"],
    "weaknesses": ["缺少最新版本信息"],
    "suggestions": ["补充 Python 3.11+ 的异步改进"],
    "needs_refinement": false
}
```""")

        agent = QualityEvalAgent(mock_llm_service)

        intent = IntentAnalysis(
            primary_intent="了解 Python 异步编程的实现原理",
            keywords=["async", "await", "coroutine", "event_loop"],
            search_sources=["rag", "web"],
            confidence=0.95,
            alternative_intents=[
                {"intent": "async/await 语法", "keywords": ["async", "await", "syntax"]},
                {"intent": "asyncio 库使用", "keywords": ["asyncio", "库", "API"]}
            ],
            intent_type="informational",
            language="zh"
        )

        documents = [
            RetrievedDocument(
                content="Python 协程是异步编程的核心",
                source="docs/async.md",
                confidence=0.9,
                metadata={"category": "programming"}
            )
        ]

        result = await agent.evaluate_quality("Python 异步编程", documents, intent)

        assert result.overall_score == 0.88
        assert result.relevance_score == 0.92
        assert "关键词匹配准确" in result.strengths
        assert "数据源正确" in result.strengths


class TestParallelRetrievalWithQualityEval:
    """测试并行检索与质量评估"""

    @pytest.mark.asyncio
    async def test_parallel_retrieve_with_quality_check(self):
        """测试并行检索与质量检查"""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.82,
    "relevance_score": 0.85,
    "completeness_score": 0.8,
    "accuracy_score": 0.81,
    "strengths": ["检索结果丰富"],
    "weaknesses": [],
    "suggestions": [],
    "needs_refinement": false
}
```""")

        agent = QualityEvalAgent(mock_llm)

        documents = [
            RetrievedDocument(content=f"文档 {i}", source=f"doc{i}.md", confidence=0.85, metadata={})
            for i in range(5)
        ]

        evaluation = await agent.evaluate_quality("测试查询", documents)

        assert evaluation.overall_score == 0.82
        assert evaluation.quality_level == QualityLevel.EXCELLENT
        assert evaluation.retrieval_status == RetrievalStatus.SUCCESS


class TestEmptyRetrievalWithQualityEval:
    """测试空检索结果的质量评估"""

    @pytest.mark.asyncio
    async def test_empty_result_triggers_refinement(self):
        """测试空结果触发改进"""
        mock_llm = MagicMock()
        agent = QualityEvalAgent(mock_llm)

        evaluation = await agent.evaluate_quality("Python 异步", [])

        assert evaluation.overall_score == 0.0
        assert evaluation.quality_level == QualityLevel.INSUFFICIENT
        assert evaluation.retrieval_status == RetrievalStatus.FAILED
        assert evaluation.needs_refinement is True
        assert "broaden_search" in evaluation.refinement_strategy

        action = agent.determine_adaptive_action(evaluation, "Python 异步", None)

        assert action.action_type == "retry"
        assert action.parameters["next_step"] == "retry_retrieval"


class TestRetrievalRetryWithQualityEval:
    """测试带质量评估的检索重试"""

    @pytest.mark.asyncio
    async def test_retry_improves_quality(self):
        """测试重试提高质量"""
        call_count = 0

        mock_llm = MagicMock()
        
        async def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return """
```json
{
    "overall_score": 0.45,
    "relevance_score": 0.5,
    "completeness_score": 0.4,
    "accuracy_score": 0.45,
    "strengths": [],
    "weaknesses": ["相关性不足"],
    "suggestions": ["使用更多关键词"],
    "needs_refinement": true,
    "refinement_strategy": "modify_keywords"
}
```"""
            else:
                return """
```json
{
    "overall_score": 0.85,
    "relevance_score": 0.9,
    "completeness_score": 0.8,
    "accuracy_score": 0.85,
    "strengths": ["质量改善"],
    "weaknesses": [],
    "suggestions": [],
    "needs_refinement": false
}
```"""
        
        mock_llm.chat_completion = mock_chat

        agent = QualityEvalAgent(mock_llm)

        initial_docs = [
            RetrievedDocument(content="无关内容", source="irrelevant.md", confidence=0.3, metadata={})
        ]

        async def retry_retrieval(query, sources=None):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return [RetrievedDocument(content="better", source="better.md", confidence=0.9, metadata={})]
            return [RetrievedDocument(content="best", source="best.md", confidence=0.95, metadata={})]

        final_docs, final_eval = await agent.adaptive_retrieve(
            query="Python 异步编程",
            documents=initial_docs,
            intent=None,
            retrieval_fn=retry_retrieval,
            max_retries=3
        )

        assert call_count > 1
        assert final_eval.needs_refinement is False


class TestQualityEvaluationLogging:
    """测试质量评估日志记录"""

    @pytest.mark.asyncio
    async def test_quality_eval_logs_transitions(self):
        """测试质量评估日志"""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.8,
    "relevance_score": 0.85,
    "completeness_score": 0.75,
    "accuracy_score": 0.8,
    "strengths": ["好"],
    "weaknesses": [],
    "suggestions": [],
    "needs_refinement": false
}
```""")

        agent = QualityEvalAgent(mock_llm)

        documents = [
            RetrievedDocument(content="test", source="s.md", confidence=0.9, metadata={})
        ]

        with patch('src.domain.agents.quality_eval.agent_logger') as mock_logger:
            await agent.evaluate_quality("test", documents)

            assert mock_logger.log_agent_transition.called


class TestComplexRetrievalScenarios:
    """测试复杂检索场景"""

    @pytest.mark.asyncio
    async def test_multi_source_retrieval_quality(self):
        """测试多源检索质量"""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.78,
    "relevance_score": 0.82,
    "completeness_score": 0.75,
    "accuracy_score": 0.77,
    "strengths": ["多源覆盖", "内容互补"],
    "weaknesses": ["深度不足"],
    "suggestions": ["增加专业文档"],
    "needs_refinement": true,
    "refinement_strategy": "augment_with_experts"
}
```""")

        agent = QualityEvalAgent(mock_llm)

        documents = [
            RetrievedDocument(content="Python async 基础", source="docs/python/async.md", confidence=0.9, metadata={}),
            RetrievedDocument(content="asyncio 示例", source="tutorials/asyncio.md", confidence=0.85, metadata={}),
            RetrievedDocument(content="异步最佳实践", source="best-practices.md", confidence=0.8, metadata={})
        ]

        result = await agent.evaluate_quality("Python 异步编程教程", documents)

        assert result.overall_score == 0.78
        assert result.quality_level == QualityLevel.GOOD
        assert "多源覆盖" in result.strengths
        assert "内容互补" in result.strengths

    @pytest.mark.asyncio
    async def test_low_confidence_documents(self):
        """测试低置信度文档"""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.35,
    "relevance_score": 0.4,
    "completeness_score": 0.3,
    "accuracy_score": 0.35,
    "strengths": [],
    "weaknesses": ["来源可信度低", "内容过时"],
    "suggestions": ["使用官方文档", "查找最新资源"],
    "needs_refinement": true,
    "refinement_strategy": "use_different_sources"
}
```""")

        agent = QualityEvalAgent(mock_llm)

        documents = [
            RetrievedDocument(content="可能是这样", source="forum/post.md", confidence=0.4, metadata={}),
            RetrievedDocument(content="旧版本信息", source="old-docs.md", confidence=0.3, metadata={})
        ]

        result = await agent.evaluate_quality("Python 异步", documents)

        assert result.overall_score == 0.35
        assert "来源可信度低" in result.weaknesses
        assert "use_different_sources" in result.refinement_strategy


class TestConvenienceFunctionIntegration:
    """测试便捷函数集成"""

    @pytest.mark.asyncio
    async def test_evaluate_quality_with_all_params(self):
        """测试完整参数的 evaluate_quality 函数"""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.88,
    "relevance_score": 0.92,
    "completeness_score": 0.85,
    "accuracy_score": 0.87,
    "strengths": ["高质量"],
    "weaknesses": [],
    "suggestions": [],
    "needs_refinement": false
}
```""")

        documents = [
            RetrievedDocument(content="test", source="s.md", confidence=0.9, metadata={})
        ]

        intent = IntentAnalysis(
            primary_intent="test",
            keywords=["test"],
            search_sources=["rag"],
            confidence=0.9
        )

        from src.domain.agents.quality_eval import evaluate_quality

        result = await evaluate_quality(
            query="test query",
            documents=documents,
            llm_service=mock_llm,
            intent=intent
        )

        assert result.overall_score == 0.88
        assert result.quality_level == QualityLevel.EXCELLENT

    @pytest.mark.asyncio
    async def test_adaptive_retrieve_with_max_retries(self):
        """测试带最大重试次数的 adaptive_retrieve"""
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.42,
    "relevance_score": 0.45,
    "completeness_score": 0.4,
    "accuracy_score": 0.41,
    "strengths": [],
    "weaknesses": ["相关性差"],
    "suggestions": ["修改关键词"],
    "needs_refinement": true,
    "refinement_strategy": "retry"
}
```""")

        agent = QualityEvalAgent(mock_llm)

        initial_docs = [
            RetrievedDocument(content="bad", source="bad.md", confidence=0.2, metadata={})
        ]

        retry_count = 0

        async def mock_retrieval(query, sources=None, expand=False):
            nonlocal retry_count
            retry_count += 1
            return [RetrievedDocument(content="retry", source=f"retry{retry_count}.md", confidence=0.3, metadata={})]

        final_docs, final_eval = await agent.adaptive_retrieve(
            query="test",
            documents=initial_docs,
            intent=None,
            retrieval_fn=mock_retrieval,
            max_retries=2
        )

        assert retry_count == 2
