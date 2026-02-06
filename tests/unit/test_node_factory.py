"""NodeFactory 单元测试

测试 v2.1 架构的节点工厂功能，包括：
- NodeFactory 初始化
- 各节点函数的正确性
- 日志创建函数
"""

import pytest
import pytest_asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any
import asyncio

from src.domain.state_types import GlobalState, create_success_log, create_error_log, get_error_message
from src.domain.agents.node_factory import NodeFactory, create_node_factory


class TestNodeFactoryInitialization:
    """NodeFactory 初始化测试"""

    def test_create_node_factory_basic(self):
        """测试基本工厂创建"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        factory = create_node_factory(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization
                    )

        assert factory.llm_service is mock_llm
        assert factory.retrieval_service is mock_retrieval
        assert factory.parser_service is mock_parser
        assert factory.summarization_service is mock_summarization

    def test_node_factory_direct_instantiation(self):
        """测试直接实例化 NodeFactory"""
        mock_llm = Mock(spec=["agenerate"])
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        factory = NodeFactory(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization
                    )

        assert hasattr(factory, "llm_service")
        assert hasattr(factory, "retrieval_service")


class TestLogCreationFunctions:
    """日志创建函数测试"""

    def test_create_success_log_basic(self):
        """测试成功日志创建"""
        log = create_success_log(
            agent="test_agent",
            action="test_action",
            details={"key": "value"}
        )

        assert log["agent"] == "test_agent"
        assert log["action"] == "test_action"
        assert log["status"] == "success"
        assert log["details"]["key"] == "value"
        assert "timestamp" in log

    def test_create_error_log_basic(self):
        """测试错误日志创建"""
        log = create_error_log(
            agent="test_agent",
            action="test_action",
            error_message="Something went wrong",
            details={"error": "details"}
        )

        assert log["agent"] == "test_agent"
        assert log["action"] == "test_action"
        assert log["status"] == "error"
        assert log["error_message"] == "Something went wrong"
        assert log["details"]["error"] == "details"
        assert "timestamp" in log

    def test_get_error_message(self):
        """测试错误消息获取"""
        msg = get_error_message("validation_error")
        assert msg == "数据验证失败"

        msg = get_error_message("unknown_error")
        assert msg == "未知错误"


class TestPlannerNode:
    """规划器节点测试"""

    @pytest.mark.asyncio
    async def test_planner_node_generates_outline(self):
        """测试规划器生成大纲"""
        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value=Mock(
            content='[{"step_id": "1", "description": "第一步", "type": "scene"}]'
        ))

        factory = create_node_factory(
            llm_service=mock_llm,
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock()
                    )

        state: GlobalState = {
            "user_topic": "测试主题",
            "project_context": "测试上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = await factory.planner_node(state)

        assert "outline" in result
        assert len(result["outline"]) > 0
        assert "execution_log" in result

    @pytest.mark.asyncio
    async def test_planner_node_empty_topic(self):
        """测试空主题处理"""
        factory = create_node_factory(
            llm_service=Mock(),
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock()
                    )

        state: GlobalState = {
            "user_topic": "",
            "project_context": "测试上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = await factory.planner_node(state)

        assert "error_flag" in result
        assert result["error_flag"] == "validation_error"


class TestNavigatorNode:
    """导航器节点测试"""

    @pytest.mark.asyncio
    async def test_navigator_node_with_content(self):
        """测试有内容时的检索"""
        mock_retrieval = Mock()
        mock_retrieval.hybrid_retrieve = AsyncMock(return_value=[
            {"id": "doc1", "content": "测试内容1", "source": "test_source"},
            {"id": "doc2", "content": "测试内容2", "source": "test_source"}
        ])
        mock_llm = Mock()
        mock_llm.chat_completion = AsyncMock(return_value='[{"id": "doc1", "content": "测试内容1"}]')

        factory = create_node_factory(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=Mock(),
            summarization_service=Mock()
                    )

        state: GlobalState = {
            "user_topic": "测试主题",
            "project_context": "测试上下文",
            "outline": [{"title": "第一步", "description": "描述"}],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = await factory.navigator_node(state)

        assert "retrieved_docs" in result
        assert len(result["retrieved_docs"]) > 0

    @pytest.mark.asyncio
    async def test_navigator_node_boundary_error(self):
        """测试索引越界错误"""
        factory = create_node_factory(
            llm_service=Mock(),
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock()
                    )

        state: GlobalState = {
            "user_topic": "测试主题",
            "project_context": "测试上下文",
            "outline": [{"title": "第一步"}],
            "current_step_index": 5,
            "fragments": [],
            "retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = await factory.navigator_node(state)

        assert "error_flag" in result
        assert result["error_flag"] == "boundary_error"


class TestStepAdvancerNode:
    """步骤推进器节点测试"""

    def test_step_advancer_basic(self):
        """测试基本步骤推进"""
        factory = create_node_factory(
            llm_service=Mock(),
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock()
                    )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"title": "第一步"}, {"title": "第二步"}],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = factory.step_advancer_node(state)

        assert result["current_step_index"] == 1
        assert "execution_log" in result

    def test_step_advancer_completion(self):
        """测试步骤完成"""
        factory = create_node_factory(
            llm_service=Mock(),
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock()
                    )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"title": "第一步"}],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = factory.step_advancer_node(state)

        assert result["current_step_index"] == 1
        assert 1 >= len(state["outline"])


class TestDirectorNode:
    """导演节点测试"""

    @pytest.mark.asyncio
    async def test_director_node_evaluates(self):
        """测试导演节点评估"""
        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value=Mock(
            content='{"decision": "write", "quality_score": 0.8, "feedback": "继续"}'
        ))

        factory = create_node_factory(
            llm_service=mock_llm,
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock()
                    )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [{"id": "doc1", "content": "测试"}],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = await factory.director_node(state)

        assert "director_feedback" in result
        assert "execution_log" in result


class TestWriterNode:
    """编剧节点测试"""

    @pytest.mark.asyncio
    async def test_writer_node_generates_fragment(self):
        """测试编剧节点生成片段"""
        mock_llm = Mock()
        mock_llm.agenerate = AsyncMock(return_value=Mock(
            content="这是生成的剧本片段内容。"
        ))

        factory = create_node_factory(
            llm_service=mock_llm,
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock()
                    )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"title": "第一步"}],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [{"id": "doc1", "content": "参考内容"}],
            "director_feedback": {"decision": "write"},
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = await factory.writer_node(state)

        assert "fragments" in result
        assert "execution_log" in result
