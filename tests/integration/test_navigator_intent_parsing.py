"""
Integration tests for Navigator Agent with Intent Parsing

测试导航器智能体在意图解析启用时的功能：
- 意图解析集成
- 检索结果增强
- 日志记录
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.domain.models import SharedState, RetrievedDocument, OutlineStep
from src.domain.agents.navigator import retrieve_content, smart_retrieve_content


class TestNavigatorWithIntentParsing:
    """测试带意图解析的导航器"""
    
    @pytest.mark.asyncio
    async def test_retrieve_content_with_intent_parsing_enabled(self):
        """测试启用意图解析时的检索"""
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
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解 Python 异步编程的实现方式",
    "keywords": ["async", "await", "asyncio", "异步编程"],
    "search_sources": ["rag"],
    "confidence": 0.95,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
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
        
        result_state = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_intent_parsing=True,
            llm_service=llm_service
        )
        
        assert len(result_state.retrieved_docs) == 1
        assert result_state.retrieved_docs[0].source == "src/async_demo.py"
        
        llm_service.chat_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_retrieve_content_with_intent_parsing_disabled(self):
        """测试禁用意图解析时的检索"""
        state = SharedState(
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
        
        llm_service = MagicMock()
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="src/demo.py",
                content="def main(): pass",
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
        
        result_state = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_intent_parsing=False,
            llm_service=llm_service
        )
        
        llm_service.chat_completion.assert_not_called()
        assert len(result_state.retrieved_docs) == 1
    
    @pytest.mark.asyncio
    async def test_retrieve_content_without_llm_service(self):
        """测试没有 LLM 服务时的检索"""
        state = SharedState(
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
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="src/demo.py",
                content="async def main(): pass",
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
        
        result_state = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_intent_parsing=True,
            llm_service=None
        )
        
        assert len(result_state.retrieved_docs) == 1
        assert result_state.current_intent is None
    
    @pytest.mark.asyncio
    async def test_retrieve_content_enhanced_query(self):
        """测试增强查询的使用"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="这个复杂的认证系统是怎么工作的"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解认证系统的工作原理和实现细节",
    "keywords": ["认证", "auth", "OAuth", "JWT"],
    "search_sources": ["rag"],
    "confidence": 0.92,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        enhanced_query = None
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(side_effect=lambda query, **kwargs: (
            setattr(globals(), 'enhanced_query', query),
            [
                MagicMock(
                    file_path="src/auth.py",
                    content="class Authenticator: pass",
                    confidence=0.88,
                    similarity=0.85,
                    strategy_name="hybrid"
                )
            ]
        ))
        
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
        
        result_state = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_intent_parsing=True,
            llm_service=llm_service
        )
        
        assert result_state.current_intent is not None
        assert "认证系统" in result_state.current_intent.primary_intent


class TestSmartRetrieveWithIntentParsing:
    """测试智能检索与意图解析的集成"""
    
    @pytest.mark.asyncio
    async def test_smart_retrieve_with_intent_parsing(self):
        """测试智能检索启用意图解析"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="如何实现 REST API"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial",
            retrieved_docs=[
                RetrievedDocument(
                    content="def api(): pass",
                    source="src/api.py",
                    confidence=0.95,
                    metadata={}
                )
            ]
        )
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解 REST API 的实现方式",
    "keywords": ["REST", "API", "endpoint", "HTTP"],
    "search_sources": ["rag"],
    "confidence": 0.90,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="src/rest_api.py",
                content="@app.route('/')",
                confidence=0.92,
                similarity=0.88,
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
        
        result_state = await smart_retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_intent_parsing=True,
            llm_service=llm_service
        )
        
        assert result_state.current_intent is not None
        assert "REST" in result_state.current_intent.primary_intent or "API" in result_state.current_intent.primary_intent
    
    @pytest.mark.asyncio
    async def test_smart_retrieve_intent_parsing_disabled(self):
        """测试智能检索禁用意图解析"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="数据库查询优化"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial",
            retrieved_docs=[
                RetrievedDocument(
                    content="SELECT * FROM users",
                    source="src/db.py",
                    confidence=0.95,
                    metadata={}
                )
            ]
        )
        
        llm_service = MagicMock()
        
        retrieval_service = MagicMock()
        
        parser_service = MagicMock()
        
        summarization_service = MagicMock()
        
        result_state = await smart_retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_intent_parsing=False,
            llm_service=llm_service
        )
        
        llm_service.chat_completion.assert_not_called()


class TestNavigatorLoggingWithIntent:
    """测试导航器在意图解析时的日志记录"""
    
    @pytest.mark.asyncio
    async def test_log_entry_contains_intent_analysis(self):
        """测试日志条目包含意图分析信息"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="机器学习模型训练"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解机器学习模型的训练流程",
    "keywords": ["ML", "machine learning", "训练", "model"],
    "search_sources": ["rag"],
    "confidence": 0.93,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="src/ml/train.py",
                content="model.fit(X, y)",
                confidence=0.91,
                similarity=0.88,
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
        
        result_state = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_intent_parsing=True,
            llm_service=llm_service
        )
        
        log_entry = result_state.execution_log[-1]
        assert log_entry["intent_parsing_enabled"] is True
        assert "intent_analysis" in log_entry
        assert log_entry["intent_analysis"]["confidence"] == 0.93
        assert "machine learning" in log_entry["intent_analysis"]["keywords"] or "ML" in log_entry["intent_analysis"]["keywords"]
    
    @pytest.mark.asyncio
    async def test_log_entry_intent_disabled(self):
        """测试禁用意图解析时的日志"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="日志测试"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
        
        llm_service = MagicMock()
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[
            MagicMock(
                file_path="src/log.py",
                content="logger.info('test')",
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
        
        result_state = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_intent_parsing=False,
            llm_service=llm_service
        )
        
        log_entry = result_state.execution_log[-1]
        assert log_entry["intent_parsing_enabled"] is False
        assert "intent_analysis" not in log_entry


class TestNavigatorEmptyRetrievalWithIntent:
    """测试空检索结果时的意图解析处理"""
    
    @pytest.mark.asyncio
    async def test_empty_retrieval_with_intent_parsing(self):
        """测试空检索结果时的处理"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="一个不存在的主题"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "查询不存在的主题",
    "keywords": ["nonexistent"],
    "search_sources": ["rag"],
    "confidence": 0.8,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        retrieval_service = MagicMock()
        retrieval_service.hybrid_retrieve = AsyncMock(return_value=[])
        
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
        
        result_state = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=False,
            enable_intent_parsing=True,
            llm_service=llm_service
        )
        
        assert len(result_state.retrieved_docs) == 0
        assert result_state.current_intent is not None


class TestNavigatorParallelWithIntent:
    """测试并行检索与意图解析"""
    
    @pytest.mark.asyncio
    async def test_parallel_retrieve_with_intent_parsing(self):
        """测试并行检索启用意图解析"""
        state = SharedState(
            user_topic="测试主题",
            project_context="测试项目",
            outline=[
                OutlineStep(
                    step_id=0,
                    title="测试步骤",
                    description="Docker 容器化部署"
                )
            ],
            current_step_index=0,
            current_skill="standard_tutorial"
        )
        
        llm_service = MagicMock()
        llm_service.chat_completion = AsyncMock(return_value="""
```json
{
    "primary_intent": "了解 Docker 容器化部署的流程",
    "keywords": ["Docker", "容器", "container", "部署"],
    "search_sources": ["rag"],
    "confidence": 0.91,
    "intent_type": "informational",
    "language": "zh"
}
```""")
        
        vector_results = [
            MagicMock(
                file_path="src/docker/Dockerfile",
                content="FROM python:3.10",
                confidence=0.92,
                similarity=0.89,
                strategy_name="vector_search"
            )
        ]
        
        keyword_results = [
            MagicMock(
                file_path="docker-compose.yml",
                content="version: '3.8'",
                confidence=0.88,
                similarity=0.85,
                strategy_name="keyword_search"
            )
        ]
        
        retrieval_service = MagicMock()
        retrieval_service.retrieve_with_strategy = AsyncMock(side_effect=[
            vector_results,
            keyword_results
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
        
        result_state = await retrieve_content(
            state=state,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id="test-workspace",
            enable_parallel=True,
            enable_intent_parsing=True,
            llm_service=llm_service
        )
        
        assert result_state.current_intent is not None
        assert "Docker" in result_state.current_intent.primary_intent or "容器" in result_state.current_intent.primary_intent
        assert len(result_state.retrieved_docs) >= 1
