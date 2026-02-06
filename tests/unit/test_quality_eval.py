"""Quality Evaluation Agent 单元测试

本模块测试 QualityEvalAgent 智能体的所有功能。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.domain.agents.quality_eval import (
    QualityEvalAgent,
    QualityEvaluation,
    QualityLevel,
    RetrievalStatus,
    AdaptiveAction,
    evaluate_quality,
    adaptive_retrieve
)
from src.domain.models import RetrievedDocument, IntentAnalysis


class TestQualityLevel:
    def test_quality_level_values(self):
        assert QualityLevel.EXCELLENT.value == "excellent"
        assert QualityLevel.GOOD.value == "good"
        assert QualityLevel.ACCEPTABLE.value == "acceptable"
        assert QualityLevel.POOR.value == "poor"
        assert QualityLevel.INSUFFICIENT.value == "insufficient"

    def test_retrieval_status_values(self):
        assert RetrievalStatus.SUCCESS.value == "success"
        assert RetrievalStatus.PARTIAL.value == "partial"
        assert RetrievalStatus.NEEDS_IMPROVEMENT.value == "needs_improvement"
        assert RetrievalStatus.FAILED.value == "failed"


class TestQualityEvaluationModel:
    def test_quality_evaluation_creation(self):
        evaluation = QualityEvaluation(
            overall_score=0.85,
            relevance_score=0.9,
            completeness_score=0.8,
            accuracy_score=0.85,
            quality_level=QualityLevel.GOOD,
            retrieval_status=RetrievalStatus.SUCCESS,
            strengths=["内容高度相关", "来源可靠"],
            weaknesses=["缺少最新信息"],
            suggestions=["补充官方文档"],
            needs_refinement=False
        )
        
        assert evaluation.overall_score == 0.85
        assert evaluation.quality_level == QualityLevel.GOOD
        assert len(evaluation.strengths) == 2

    def test_quality_evaluation_insufficient(self):
        evaluation = QualityEvaluation(
            overall_score=0.0,
            relevance_score=0.0,
            completeness_score=0.0,
            accuracy_score=0.0,
            quality_level=QualityLevel.INSUFFICIENT,
            retrieval_status=RetrievalStatus.FAILED,
            strengths=[],
            weaknesses=["未检索到任何文档"],
            suggestions=["尝试修改查询关键词"],
            needs_refinement=True,
            refinement_strategy="broaden_search"
        )
        
        assert evaluation.overall_score == 0.0
        assert evaluation.quality_level == QualityLevel.INSUFFICIENT


class TestAdaptiveAction:
    def test_adaptive_action_proceed(self):
        action = AdaptiveAction(
            action_type="proceed",
            reason="检索质量优秀",
            parameters={"next_step": "generation"}
        )
        
        assert action.action_type == "proceed"
        assert action.parameters["next_step"] == "generation"

    def test_adaptive_action_retry(self):
        action = AdaptiveAction(
            action_type="retry",
            reason="需要重新检索",
            parameters={
                "next_step": "retry_retrieval",
                "alternative_keywords": ["async", "await"]
            }
        )
        
        assert action.action_type == "retry"
        assert "alternative_keywords" in action.parameters


class TestQualityEvalAgent:
    @pytest.fixture
    def mock_llm_service(self):
        service = MagicMock()
        service.chat_completion = AsyncMock()
        return service

    @pytest.fixture
    def sample_documents(self):
        return [
            RetrievedDocument(
                content="Python 异步编程使用 async 和 await 关键字",
                source="docs/python/async.md",
                confidence=0.9,
                metadata={"type": "documentation"},
                summary="异步编程基础"
            ),
            RetrievedDocument(
                content="asyncio 是 Python 的异步 IO 库",
                source="docs/python/asyncio.md",
                confidence=0.85,
                metadata={"type": "documentation"},
                summary="asyncio 库介绍"
            )
        ]

    @pytest.mark.asyncio
    async def test_evaluate_quality_success(self, mock_llm_service, sample_documents):
        mock_llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.85,
    "relevance_score": 0.9,
    "completeness_score": 0.8,
    "accuracy_score": 0.85,
    "strengths": ["文档内容高度相关"],
    "weaknesses": [],
    "suggestions": [],
    "needs_refinement": false
}
```""")
        
        agent = QualityEvalAgent(mock_llm_service)
        result = await agent.evaluate_quality("Python 异步编程", sample_documents)
        
        assert result.overall_score == 0.85
        assert result.relevance_score == 0.9
        assert result.quality_level == QualityLevel.EXCELLENT

    @pytest.mark.asyncio
    async def test_evaluate_empty_documents(self, mock_llm_service):
        agent = QualityEvalAgent(mock_llm_service)
        result = await agent.evaluate_quality("Python 异步编程", [])
        
        assert result.overall_score == 0.0
        assert result.quality_level == QualityLevel.INSUFFICIENT
        assert result.needs_refinement is True

    @pytest.mark.asyncio
    async def test_evaluate_llm_failure(self, mock_llm_service, sample_documents):
        mock_llm_service.chat_completion = AsyncMock(side_effect=Exception("LLM error"))
        
        agent = QualityEvalAgent(mock_llm_service)
        result = await agent.evaluate_quality("Python 异步编程", sample_documents)
        
        assert result.overall_score > 0


class TestDetermineAdaptiveAction:
    def test_action_excellent_quality(self):
        agent = QualityEvalAgent(MagicMock())
        
        evaluation = QualityEvaluation(
            overall_score=0.9, relevance_score=0.95, completeness_score=0.85,
            accuracy_score=0.9, quality_level=QualityLevel.EXCELLENT,
            retrieval_status=RetrievalStatus.SUCCESS, strengths=["好"],
            weaknesses=[], suggestions=[], needs_refinement=False
        )
        
        action = agent.determine_adaptive_action(evaluation, "test", None)
        
        assert action.action_type == "proceed"

    def test_action_retry_poor_quality(self):
        agent = QualityEvalAgent(MagicMock())
        
        intent = IntentAnalysis(
            primary_intent="test", keywords=["k1"], search_sources=["rag"], confidence=0.8
        )
        
        evaluation = QualityEvaluation(
            overall_score=0.25, relevance_score=0.3, completeness_score=0.2,
            accuracy_score=0.25, quality_level=QualityLevel.POOR,
            retrieval_status=RetrievalStatus.NEEDS_IMPROVEMENT,
            strengths=[], weaknesses=["差"], suggestions=[], needs_refinement=True
        )
        
        action = agent.determine_adaptive_action(evaluation, "test", intent)
        
        assert action.action_type == "retry"
        assert action.parameters["alternative_keywords"] == intent.keywords

    def test_action_refine_acceptable_quality(self):
        agent = QualityEvalAgent(MagicMock())
        
        evaluation = QualityEvaluation(
            overall_score=0.5, relevance_score=0.55, completeness_score=0.45,
            accuracy_score=0.5, quality_level=QualityLevel.ACCEPTABLE,
            retrieval_status=RetrievalStatus.PARTIAL, strengths=[],
            weaknesses=["一般"], suggestions=["优化"], needs_refinement=True,
            refinement_strategy="modify_keywords"
        )
        
        action = agent.determine_adaptive_action(evaluation, "test", None)
        
        assert action.action_type == "refine_retrieval"


class TestFallbackEvaluation:
    def test_fallback_single_document(self):
        agent = QualityEvalAgent(MagicMock())
        
        documents = [
            RetrievedDocument(content="test", source="s1", confidence=0.9, metadata={})
        ]
        
        result = agent._fallback_evaluation(documents)
        
        assert result.overall_score > 0.6
        assert result.quality_level in [QualityLevel.EXCELLENT, QualityLevel.GOOD]

    def test_fallback_empty(self):
        agent = QualityEvalAgent(MagicMock())
        
        result = agent._fallback_evaluation([])
        
        assert result.overall_score == 0.0
        assert result.quality_level == QualityLevel.INSUFFICIENT


class TestQualityLevelDetermination:
    def test_excellent_level(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_quality_level(0.9) == QualityLevel.EXCELLENT

    def test_good_level(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_quality_level(0.75) == QualityLevel.GOOD

    def test_acceptable_level(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_quality_level(0.55) == QualityLevel.ACCEPTABLE

    def test_poor_level(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_quality_level(0.35) == QualityLevel.POOR

    def test_insufficient_level(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_quality_level(0.1) == QualityLevel.INSUFFICIENT


class TestRetrievalStatusDetermination:
    def test_success_status(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_retrieval_status(0.8) == RetrievalStatus.SUCCESS

    def test_partial_status(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_retrieval_status(0.6) == RetrievalStatus.PARTIAL

    def test_needs_improvement_status(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_retrieval_status(0.4) == RetrievalStatus.NEEDS_IMPROVEMENT

    def test_failed_status(self):
        agent = QualityEvalAgent(MagicMock())
        assert agent._determine_retrieval_status(0.2) == RetrievalStatus.FAILED


class TestAdaptiveRetrieve:
    @pytest.mark.asyncio
    async def test_adaptive_no_retry_needed(self):
        mock_llm = MagicMock()
        mock_llm.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.9,
    "relevance_score": 0.95,
    "completeness_score": 0.85,
    "accuracy_score": 0.9,
    "strengths": ["优秀"],
    "weaknesses": [],
    "suggestions": [],
    "needs_refinement": false
}
```""")
        
        agent = QualityEvalAgent(mock_llm)
        
        documents = [
            RetrievedDocument(content="test", source="s1", confidence=0.9, metadata={})
        ]
        
        async def mock_retrieval(**kwargs):
            return []
        
        result_docs, result_eval = await agent.adaptive_retrieve(
            "test", documents, None, mock_retrieval, max_retries=2
        )
        
        assert result_eval.overall_score == 0.9
        assert len(result_docs) == 1


class TestConvenienceFunctions:
    @pytest.mark.asyncio
    async def test_evaluate_function(self):
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
        
        documents = [
            RetrievedDocument(content="test", source="s1", confidence=0.9, metadata={})
        ]
        
        result = await evaluate_quality("test", documents, mock_llm)
        
        assert result.overall_score == 0.8
        assert result.quality_level == QualityLevel.EXCELLENT


class TestDocumentProcessing:
    def test_prepare_with_summary(self):
        agent = QualityEvalAgent(MagicMock())
        
        documents = [
            RetrievedDocument(
                content="Long content...", source="test.py",
                confidence=0.9, metadata={}, summary="Summary"
            )
        ]
        
        processed = agent._prepare_documents_for_eval(documents)
        
        assert len(processed) == 1
        assert "[摘要]" in processed[0][1]

    def test_prepare_truncation(self):
        agent = QualityEvalAgent(MagicMock())
        
        documents = [
            RetrievedDocument(
                content="A" * 5000, source="test.py", confidence=0.9, metadata={}
            )
        ]
        
        processed = agent._prepare_documents_for_eval(documents)
        
        assert "[内容截断]" in processed[0][1]


class TestJSONParsing:
    def test_parse_with_markdown(self):
        agent = QualityEvalAgent(MagicMock())
        
        response = """```json
{
    "overall_score": 0.85,
    "relevance_score": 0.9,
    "completeness_score": 0.8,
    "accuracy_score": 0.85,
    "strengths": ["test"],
    "weaknesses": [],
    "suggestions": []
}
```"""
        
        result = agent._parse_evaluation_response(response)
        
        assert result["overall_score"] == 0.85

    def test_parse_score_normalization(self):
        agent = QualityEvalAgent(MagicMock())
        
        response = """{
    "overall_score": 1.5,
    "relevance_score": -0.2,
    "completeness_score": 0.8,
    "accuracy_score": 2.0,
    "strengths": ["test"],
    "weaknesses": [],
    "suggestions": []
}"""
        
        result = agent._parse_evaluation_response(response)
        
        assert result["overall_score"] == 1.0
        assert result["relevance_score"] == 0.0
