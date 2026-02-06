"""
Integration tests for Orchestrator with Agentic RAG

测试编排器集成 Agentic RAG 功能：
- 意图解析节点集成
- 质量评估节点集成
- 自适应检索循环
- 条件边路由
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from src.application.orchestrator import WorkflowOrchestrator
from src.domain.models import SharedState, RetrievedDocument, OutlineStep, IntentAnalysis
from src.domain.state_types import GlobalState


async def mock_retrieve_with_strategy(*args, **kwargs):
    """Mock retrieval function for parallel search"""
    return [
        {"content": "Python 异步编程示例代码", "source": "test_file.py", "confidence": 0.95}
    ]


async def mock_hybrid_retrieve(*args, **kwargs):
    """Mock hybrid retrieval function"""
    return [
        {"content": "Python 异步编程示例代码", "source": "test_file.py", "confidence": 0.95}
    ]


class TestOrchestratorAgenticRAGNodes:
    """测试 Orchestrator Agentic RAG 节点"""
    
    @pytest.fixture
    def mock_services(self):
        """创建模拟服务"""
        llm_service = MagicMock()
        retrieval_service = MagicMock()
        retrieval_service.retrieve_with_strategy = mock_retrieve_with_strategy
        retrieval_service.hybrid_retrieve = mock_hybrid_retrieve
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        return {
            "llm_service": llm_service,
            "retrieval_service": retrieval_service,
            "parser_service": parser_service,
            "summarization_service": summarization_service
        }
    
    @pytest.mark.asyncio
    async def test_orchestrator_with_agentic_rag_enabled(self, mock_services):
        """测试启用 Agentic RAG 的编排器"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True,
            max_retrieval_retries=2
        )
        
        assert orchestrator.enable_agentic_rag is True
        assert orchestrator.intent_parser is not None
        assert orchestrator.quality_eval_agent is not None
        assert orchestrator.max_retrieval_retries == 2
    
    @pytest.mark.asyncio
    async def test_orchestrator_without_agentic_rag(self, mock_services):
        """测试禁用 Agentic RAG 的编排器"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=False
        )
        
        assert orchestrator.enable_agentic_rag is False
        assert orchestrator.intent_parser is None
        assert orchestrator.quality_eval_agent is None


class TestIntentParserNode:
    """测试意图解析节点"""
    
    @pytest.fixture
    def mock_services(self):
        """创建模拟服务"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解 Python 异步编程的实现方式",
    "keywords": ["async", "await", "asyncio", "异步编程"],
    "search_sources": ["rag"],
    "confidence": 0.95,
    "intent_type": "informational",
    "reasoning": "用户想了解异步编程的技术细节"
}
```""")
        
        retrieval_service = MagicMock()
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        return {
            "llm_service": llm_service,
            "retrieval_service": retrieval_service,
            "parser_service": parser_service,
            "summarization_service": summarization_service
        }
    
    @pytest.mark.asyncio
    async def test_intent_parser_node_execution(self, mock_services):
        """测试意图解析节点执行"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True
        )
        
        state = {
            "outline": [
                {"step_id": 0, "title": "步骤1", "description": "Python 异步编程怎么实现"}
            ],
            "current_step_index": 0,
            "execution_log": []
        }
        
        result = await orchestrator._intent_parser_node(state)
        
        assert "current_intent" in result
        intent = result["current_intent"]
        assert intent["primary_intent"] == "了解 Python 异步编程的实现方式"
        assert "async" in intent["keywords"]
        assert intent["confidence"] == 0.95
        assert "execution_log" in result
        
        mock_services["llm_service"].chat_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_intent_parser_node_no_step(self, mock_services):
        """测试没有步骤时的意图解析"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True
        )
        
        state = {
            "outline": [],
            "current_step_index": 0,
            "execution_log": []
        }
        
        result = await orchestrator._intent_parser_node(state)
        
        assert result == {}


class TestQualityEvalNode:
    """测试质量评估节点"""
    
    @pytest.fixture
    def mock_services(self):
        """创建模拟服务"""
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "overall_score": 0.85,
    "relevance_score": 0.9,
    "completeness_score": 0.8,
    "accuracy_score": 0.85,
    "strengths": ["内容高度相关"],
    "weaknesses": [],
    "suggestions": ["可以添加更多示例"],
    "needs_refinement": false
}
```""")
        
        retrieval_service = MagicMock()
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        return {
            "llm_service": llm_service,
            "retrieval_service": retrieval_service,
            "parser_service": parser_service,
            "summarization_service": summarization_service
        }
    
    @pytest.mark.asyncio
    async def test_quality_eval_node_execution(self, mock_services):
        """测试质量评估节点执行"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True
        )
        
        state = {
            "outline": [
                {"step_id": 0, "title": "步骤1", "description": "Python 异步编程"}
            ],
            "current_step_index": 0,
            "retrieved_docs": [
                RetrievedDocument(
                    content="Python 异步编程使用 async 和 await",
                    source="docs/async.md",
                    confidence=0.9,
                    metadata={}
                )
            ],
            "current_intent": {
                "primary_intent": "了解异步编程",
                "keywords": ["async"],
                "search_sources": ["rag"],
                "confidence": 0.95,
                "intent_type": "informational"
            },
            "enhanced_query": "了解异步编程",
            "execution_log": []
        }
        
        result = await orchestrator._quality_eval_node(state)
        
        assert "quality_evaluation" in result
        quality_eval = result["quality_evaluation"]
        assert quality_eval["overall_score"] == 0.85
        assert quality_eval["quality_level"] == "excellent"
        assert quality_eval["needs_refinement"] is False
        assert "quality_suggestions" in result
        assert "execution_log" in result
        
        mock_services["llm_service"].chat_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_quality_eval_node_no_documents(self, mock_services):
        """测试没有文档时的质量评估"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True
        )
        
        state = {
            "outline": [
                {"step_id": 0, "title": "步骤1", "description": "Python 异步编程"}
            ],
            "current_step_index": 0,
            "retrieved_docs": [],
            "execution_log": []
        }
        
        result = await orchestrator._quality_eval_node(state)
        
        assert "quality_evaluation" in result
        quality_eval = result["quality_evaluation"]
        assert quality_eval["overall_score"] == 0.0
        assert quality_eval["quality_level"] == "insufficient"
        assert quality_eval["needs_refinement"] is True
        assert "retrieval_status" in quality_eval
        assert quality_eval["retrieval_status"] == "no_results"
        
        mock_services["llm_service"].chat_completion.assert_not_called()


class TestQualityEvalRouting:
    """测试质量评估路由"""
    
    @pytest.fixture
    def orchestrator(self, mock_services):
        """创建编排器"""
        return WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True,
            max_retrieval_retries=2
        )
    
    def test_route_quality_eval_good(self, orchestrator):
        """测试质量评估路由 - 良好"""
        state = {
            "quality_evaluation": {
                "overall_score": 0.85,
                "quality_level": "excellent",
                "needs_refinement": False
            },
            "retrieval_retry_count": 0
        }
        
        result = orchestrator._route_quality_eval_decision(state)
        
        assert result == "good"
    
    def test_route_quality_eval_needs_refinement_with_retry(self, orchestrator):
        """测试质量评估路由 - 需要改进且可重试"""
        state = {
            "quality_evaluation": {
                "overall_score": 0.55,
                "quality_level": "poor",
                "needs_refinement": True
            },
            "retrieval_retry_count": 0
        }
        
        result = orchestrator._route_quality_eval_decision(state)
        
        assert result == "retry"
    
    def test_route_quality_eval_max_retries_exceeded(self, orchestrator):
        """测试质量评估路由 - 超过最大重试次数"""
        state = {
            "quality_evaluation": {
                "overall_score": 0.55,
                "quality_level": "poor",
                "needs_refinement": True
            },
            "retrieval_retry_count": 2
        }
        
        result = orchestrator._route_quality_eval_decision(state)
        
        assert result == "failed"
    
    def test_route_quality_eval_no_evaluation(self, orchestrator):
        """测试质量评估路由 - 无评估结果"""
        state = {
            "retrieval_retry_count": 0
        }
        
        result = orchestrator._route_quality_eval_decision(state)
        
        assert result == "failed"


class TestNavigatorNodeWithAgenticRAG:
    """测试 Navigator 节点与 Agentic RAG 集成"""
    
    @pytest.fixture
    def mock_services(self):
        """创建模拟服务"""
        llm_service = MagicMock()
        
        retrieval_service = MagicMock()
        retrieval_service.retrieve_with_strategy = AsyncMock(return_value=[
            MagicMock(
                file_path="docs/async.md",
                content="Python 异步编程",
                confidence=0.9,
                similarity=0.85,
                strategy_name="vector_search"
            )
        ])
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="docs/async.md",
                content="Python 异步编程",
                confidence=0.9,
                similarity=0.85,
                strategy_name="hybrid"
            )
        ])
        
        parser_service = MagicMock()
        parser_service.parse = MagicMock(return_value=MagicMock(
            has_deprecated=False,
            has_fixme=False,
            has_todo=False,
            has_security=False,
            language="python",
            elements=[],
            metadata={}
        ))
        
        summarization_service = MagicMock()
        summarization_service.check_size = MagicMock(return_value=False)
        
        return {
            "llm_service": llm_service,
            "retrieval_service": retrieval_service,
            "parser_service": parser_service,
            "summarization_service": summarization_service
        }
    
    @pytest.mark.asyncio
    async def test_navigator_node_with_intent(self, mock_services):
        """测试带意图的 Navigator 节点"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True
        )
        
        state = {
            "outline": [
                {"step_id": 0, "title": "步骤1", "description": "Python 异步编程"}
            ],
            "current_step_index": 0,
            "current_intent": {
                "primary_intent": "了解异步编程",
                "keywords": ["async"],
                "search_sources": ["rag"],
                "confidence": 0.95,
                "intent_type": "informational"
            },
            "retrieved_docs": [],
            "execution_log": []
        }
        
        result = await orchestrator._navigator_node(state)
        
        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) == 1
        assert "enhanced_query" in result
        assert result["enhanced_query"] == "了解异步编程"
        
        mock_services["retrieval_service"].retrieve_with_strategy.assert_called()
    
    @pytest.mark.asyncio
    async def test_navigator_node_workflow_complete(self, mock_services):
        """测试 Navigator 节点工作流完成"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True
        )
        
        state = {
            "outline": [
                {"step_id": 0, "title": "步骤1", "description": "步骤1"}
            ],
            "current_step_index": 1,
            "current_intent": None,
            "retrieved_docs": [],
            "execution_log": []
        }
        
        result = await orchestrator._navigator_node(state)
        
        assert result["workflow_complete"] is True
        
        mock_services["retrieval_service"].hybrid_retrieve.assert_not_called()


class TestOrchestratorGraphBuilding:
    """测试编排器图构建"""
    
    @pytest.fixture
    def mock_services(self):
        """创建模拟服务"""
        llm_service = MagicMock()
        retrieval_service = MagicMock()
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        return {
            "llm_service": llm_service,
            "retrieval_service": retrieval_service,
            "parser_service": parser_service,
            "summarization_service": summarization_service
        }
    
    def test_graph_builds_with_agentic_rag(self, mock_services):
        """测试启用 Agentic RAG 时图构建成功"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True
        )
        
        assert orchestrator.graph is not None
    
    def test_graph_builds_without_agentic_rag(self, mock_services):
        """测试禁用 Agentic RAG 时图构建成功"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=False
        )
        
        assert orchestrator.graph is not None
    
    def test_graph_builds_with_dynamic_adjustment(self, mock_services):
        """测试启用动态调整时图构建成功"""
        orchestrator = WorkflowOrchestrator(
            llm_service=mock_services["llm_service"],
            retrieval_service=mock_services["retrieval_service"],
            parser_service=mock_services["parser_service"],
            summarization_service=mock_services["summarization_service"],
            workspace_id="test-workspace",
            enable_agentic_rag=True,
            enable_dynamic_adjustment=True
        )
        
        assert orchestrator.graph is not None
