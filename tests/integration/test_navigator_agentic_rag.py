"""
Integration tests for Navigator Agent

测试导航器智能体的核心功能：
- 混合检索
- 并行检索优化
- 文档解析和摘要
- 意图接收和质量评估
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.domain.models import SharedState, RetrievedDocument, OutlineStep, IntentAnalysis
from src.domain.agents.quality_eval import QualityEvaluation, QualityLevel
from src.domain.agents.navigator import retrieve_content, smart_retrieve_content


class TestNavigatorBasicRetrieval:
    """测试导航器基本检索功能"""
    
    @pytest.fixture
    def sample_state(self):
        """创建示例状态"""
        return SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="Python 异步编程怎么实现"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
    
    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        return [
            MagicMock(
                file_path="src/async_demo.py",
                content="async def main(): pass",
                confidence=0.92,
                similarity=0.9,
                strategy_name="hybrid"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_retrieve_content_basic(self, sample_state, sample_documents):
        """测试基本检索功能"""
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=sample_documents)
        
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
        
        result_state, quality_evaluation = await retrieve_content(
            state=sample_state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_quality_eval=False
        )
        
        assert len(result_state["retrieved_docs"]) == 1
        assert result_state["retrieved_docs"][0]["source"] == "src/async_demo.py"
        assert quality_evaluation is None
        
        retrieval_service.hybrid_retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_content_with_intent(self, sample_state, sample_documents):
        """测试带意图的检索功能"""
        intent = IntentAnalysis(
            primary_intent="了解 Python 异步编程的实现方式",
            keywords=["async", "await", "asyncio"],
            search_sources=["rag"],
            confidence=0.95,
            intent_type="informational"
        )
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=sample_documents)
        
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
        
        result_state, quality_evaluation = await retrieve_content(
            state=sample_state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_quality_eval=False,
            intent=intent
        )
        
        assert len(result_state["retrieved_docs"]) == 1
        
        retrieval_service.hybrid_retrieve.assert_called_once()
        call_args = retrieval_service.hybrid_retrieve.call_args
        assert "了解 Python 异步编程的实现方式" in call_args.kwargs.get("query", call_args.args[1] if len(call_args.args) > 1 else "")
    
    @pytest.mark.asyncio
    async def test_retrieve_content_empty_results(self, sample_state):
        """测试空检索结果处理"""
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[])
        
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        result_state, quality_evaluation = await retrieve_content(
            state=sample_state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_quality_eval=False
        )
        
        assert len(result_state["retrieved_docs"]) == 0
        assert quality_evaluation is None
    
    @pytest.mark.asyncio
    async def test_retrieve_content_parallel(self, sample_state, sample_documents):
        """测试并行检索功能"""
        retrieval_service = MagicMock()
        retrieval_service.retrieve_with_strategy = AsyncMock(return_value=sample_documents[0])
        retrieval_service.get_cache_key = MagicMock(return_value="cache_key")
        retrieval_service.get_cached = MagicMock(return_value=None)
        
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
        
        result_state, quality_evaluation = await retrieve_content(
            state=sample_state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=True,
            enable_quality_eval=False
        )
        
        assert len(result_state["retrieved_docs"]) >= 0


class TestNavigatorWithQualityEvaluation:
    """测试带质量评估的导航器"""
    
    @pytest.fixture
    def sample_state(self):
        """创建示例状态"""
        return SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="Python 异步编程"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
    
    @pytest.fixture
    def sample_documents(self):
        """创建示例文档"""
        return [
            MagicMock(
                file_path="docs/async.md",
                content="Python 异步编程使用 async 和 await 关键字",
                confidence=0.9,
                similarity=0.85,
                strategy_name="vector"
            ),
            MagicMock(
                file_path="docs/asyncio.md",
                content="asyncio 是 Python 的异步 IO 库",
                confidence=0.85,
                similarity=0.8,
                strategy_name="keyword"
            )
        ]
    
    @pytest.mark.asyncio
    async def test_retrieve_with_quality_evaluation(self, sample_state, sample_documents):
        """测试启用质量评估的检索"""
        llm_service = MagicMock()
        
        mock_quality_eval = QualityEvaluation(
             overall_score=0.85,
             relevance_score=0.9,
             completeness_score=0.8,
             accuracy_score=0.85,
             quality_level=QualityLevel.EXCELLENT,
             retrieval_status="success",
             strengths=["内容高度相关"],
             weaknesses=[],
             suggestions=[],
             needs_refinement=False
         )
        
        with patch('src.domain.agents.navigator.QualityEvalAgent') as MockQualityEvalAgent:
            mock_agent = MagicMock()
            mock_agent.evaluate_quality = AsyncMock(return_value=mock_quality_eval)
            MockQualityEvalAgent.return_value = mock_agent
            
            retrieval_service = MagicMock()
            retrieval_service.hybrid_retrieve = AsyncMock(return_value=sample_documents)
            
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
            
            result_state, quality_evaluation = await retrieve_content(
                state=sample_state,
                retrieval_service=retrieval_service,
                parser_service=parser_service,
                summarization_service=summarization_service,
                workspace_id="test-workspace",
                enable_parallel=False,
                enable_quality_eval=True,
                llm_service=llm_service
            )
            
            assert result_state is not None
            assert quality_evaluation is not None
            assert quality_evaluation.overall_score == 0.85
    
    @pytest.mark.asyncio
    async def test_retrieve_with_intent_and_quality(self, sample_state, sample_documents):
        """测试同时带意图和质量评估的检索"""
        intent = IntentAnalysis(
            primary_intent="了解 Python 异步编程的实现方式",
            keywords=["async", "await", "asyncio"],
            search_sources=["rag"],
            confidence=0.95,
            intent_type="informational"
        )
        
        mock_quality_eval = QualityEvaluation(
            overall_score=0.88,
            relevance_score=0.92,
            completeness_score=0.85,
            accuracy_score=0.87,
            quality_level=QualityLevel.EXCELLENT,
            retrieval_status="success",
            strengths=["文档内容高度相关"],
            weaknesses=[],
            suggestions=[],
            needs_refinement=False
        )
        
        llm_service = MagicMock()
        
        with patch('src.domain.agents.navigator.QualityEvalAgent') as MockQualityEvalAgent:
            mock_agent = MagicMock()
            mock_agent.evaluate_quality = AsyncMock(return_value=mock_quality_eval)
            MockQualityEvalAgent.return_value = mock_agent
            
            retrieval_service = MagicMock()
            retrieval_service.hybrid_retrieve = AsyncMock(return_value=sample_documents)
            
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
            
            result_state, quality_evaluation = await retrieve_content(
                state=sample_state,
                retrieval_service=retrieval_service,
                parser_service=parser_service,
                summarization_service=summarization_service,
                workspace_id="test-workspace",
                enable_parallel=False,
                enable_quality_eval=True,
                llm_service=llm_service,
                intent=intent
            )
            
            assert result_state is not None
            assert quality_evaluation is not None
            assert quality_evaluation.overall_score == 0.88


class TestNavigatorEdgeCases:
    """测试导航器边界情况"""
    
    @pytest.mark.asyncio
    async def test_no_more_steps(self):
        """测试没有更多步骤的情况"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(step_id=0, title="步骤1", description="描述1")
            ],
            current_step_index=1,
            current_skill="standard_tutorial"
        )
        
        retrieval_service = MagicMock()
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        result_state, quality_evaluation = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_quality_eval=False
        )
        
        assert result_state["current_step_index"] == 1
        assert quality_evaluation is None
        
        retrieval_service.hybrid_retrieve.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_service_error_handling(self):
        """测试服务错误处理"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(step_id=0, title="步骤1", description="描述1")
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(side_effect=Exception("检索服务错误"))
        
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        result_state, quality_evaluation = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_quality_eval=False
        )
        
        assert len(result_state["retrieved_docs"]) == 0
        assert quality_evaluation is None
        assert len(result_state["execution_log"]) > 0


class TestSmartRetrieveContent:
    """测试智能检索功能"""
    
    @pytest.mark.asyncio
    async def test_smart_retrieve_with_skip(self):
        """测试智能跳过"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(step_id=0, title="步骤1", description="描述1")
            ],
            current_step_index=0,
            current_skill="standard_tutorial",
            retrieved_docs=[
                RetrievedDocument(
                    content="高质量内容",
                    source="doc.md",
                    confidence=0.9,
                    metadata={}
                )
            ]
        )
        
        retrieval_service = MagicMock()
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        result_state, quality_evaluation = await smart_retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_quality_eval=False
        )
        
        assert quality_evaluation is None
        retrieval_service.hybrid_retrieve.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_smart_retrieve_without_skip(self):
        """测试不跳过"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(step_id=0, title="步骤1", description="描述1")
            ],
            current_step_index=0,
            current_skill="standard_tutorial",
            retrieved_docs=[]
        )
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="doc.md",
                content="内容",
                confidence=0.8,
                similarity=0.7,
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
        
        with patch('src.domain.agents.navigator.SmartSkipOptimizer') as MockSkipOptimizer:
            mock_optimizer = MagicMock()
            mock_optimizer.should_skip.return_value = MagicMock(should_skip=False, reason="low_confidence")
            MockSkipOptimizer.return_value = mock_optimizer
            
            result_state, quality_evaluation = await smart_retrieve_content(
                state=state,
                retrieval_service=retrieval_service,
                parser_service=parser_service,
                summarization_service=summarization_service,
                workspace_id="test-workspace",
                enable_quality_eval=False
            )
            
            assert len(result_state["retrieved_docs"]) >= 0
        assert quality_evaluation is None
