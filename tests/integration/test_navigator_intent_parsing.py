"""
Integration tests for Navigator Agent with Intent Parsing

测试导航器智能体在意图解析启用时的功能：
- 意图解析集成
- 检索结果增强
- 日志记录
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.domain.models import SharedState, RetrievedDocument, OutlineStep, IntentAnalysis
from src.domain.agents.navigator import retrieve_content, smart_retrieve_content


class TestNavigatorWithIntentParsing:
    """测试带意图解析的导航器"""
    
    @pytest.mark.asyncio
    async def test_retrieve_content_with_intent(self):
        """测试带意图的检索功能"""
        state = SharedState(
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

        intent = IntentAnalysis(
            primary_intent="了解 Python 异步编程的实现方式",
            keywords=["async", "await", "asyncio", "异步编程"],
            search_sources=["rag"],
            confidence=0.95,
            intent_type="informational",
            language="zh"
        )

        llm_service = MagicMock()

        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="src/async_demo.py",
                content="async def main(): pass",
                confidence=0.92,
                similarity=0.9,
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

        result_state, quality_evaluation = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            enable_parallel=False,
            llm_service=llm_service,
            intent=intent
        )

        assert len(result_state.retrieved_docs) == 1
        assert result_state.retrieved_docs[0].source == "src/async_demo.py"
        
        retrieval_service.hybrid_retrieve.assert_called_once()


class TestNavigatorEmptyRetrieval:
    """测试空检索结果处理"""
    
    @pytest.mark.asyncio
    async def test_retrieve_content_empty_results(self):
        """测试空检索结果处理"""
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
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[])
        
        parser_service = MagicMock()
        summarization_service = MagicMock()
        
        result_state, quality_evaluation = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            enable_parallel=False,
            enable_quality_eval=False
        )
        
        assert len(result_state.retrieved_docs) == 0
        assert quality_evaluation is None


class TestNavigatorBasicFunctionality:
    """测试导航器基本功能"""
    
    @pytest.mark.asyncio
    async def test_retrieve_content_basic(self):
        """测试基本检索功能"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="如何实现 Python 异步编程"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="src/async_demo.py",
                content="async def main(): pass",
                confidence=0.92,
                similarity=0.9,
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
        
        result_state, quality_evaluation = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            enable_parallel=False,
            enable_quality_eval=False
        )
        
        assert len(result_state.retrieved_docs) == 1
        assert quality_evaluation is None
        
        retrieval_service.hybrid_retrieve.assert_called_once()


class TestNavigatorEdgeCases:
    """测试导航器边界情况"""
    
    @pytest.mark.asyncio
    async def test_skip_already_retrieved(self):
        """测试跳过已检索的步骤"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(step_id=0, title="步骤1", description="描述1"),
                OutlineStep(step_id=1, title="步骤2", description="描述2")
            ],
            current_step_index=1,
            current_skill="standard_tutorial",
            retrieved_docs=[
                RetrievedDocument(
                    content="已检索内容",
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
            enable_quality_eval=False
        )
        
        assert result_state.current_step_index == 1
        assert quality_evaluation is None
        
        retrieval_service.hybrid_retrieve.assert_not_called()


class TestNavigatorLogging:
    """测试导航器日志记录"""
    
    @pytest.mark.asyncio
    async def test_log_entry_created(self):
        """测试日志条目创建"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(step_id=0, title="步骤1", description="描述1")
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )

        intent = IntentAnalysis(
            primary_intent="测试意图",
            keywords=["测试"],
            search_sources=["rag"],
            confidence=0.9,
            intent_type="informational",
            language="zh"
        )

        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="src/test.py",
                content="test content",
                confidence=0.85,
                similarity=0.8,
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

        result_state, quality_evaluation = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            enable_parallel=False,
            llm_service=MagicMock(),
            intent=intent
        )

        assert len(result_state.execution_log) > 0
