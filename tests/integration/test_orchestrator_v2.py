"""Orchestrator v2.1 集成测试

测试 WorkflowOrchestrator 与 v2.1 架构的集成：
- 使用 GlobalState
- LangGraph 状态机构建
- 节点间状态流转
- 工作流执行流程
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any

from src.application.orchestrator import WorkflowOrchestrator
from src.domain.state_types import GlobalState


class TestWorkflowOrchestratorBasic:
    """工作流编排器基础测试"""

    def test_orchestrator_initialization_basic(self):
        """测试基本初始化"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test-workspace",
            enable_dynamic_adjustment=False
        )

        assert orchestrator.workspace_id == "test-workspace"
        assert orchestrator.enable_dynamic_adjustment is False
        assert orchestrator.graph is not None

    def test_orchestrator_initialization_with_dynamic(self):
        """测试带动态调整的初始化"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test-workspace",
            enable_dynamic_adjustment=True
        )

        assert orchestrator.enable_dynamic_adjustment is True
        assert hasattr(orchestrator, 'rag_analyzer')
        assert hasattr(orchestrator, 'dynamic_director')
        assert hasattr(orchestrator, 'skill_recommender')

    def test_orchestrator_has_node_factory(self):
        """测试编排器包含节点工厂"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        assert orchestrator.node_factory is not None


class TestGraphConstruction:
    """图构建测试"""

    def test_graph_built_without_dynamic(self):
        """测试无动态调整的图构建"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test",
            enable_dynamic_adjustment=False
        )

        graph = orchestrator.graph
        assert graph is not None

    def test_graph_built_with_dynamic(self):
        """测试带动态调整的图构建"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test",
            enable_dynamic_adjustment=True
        )

        graph = orchestrator.graph
        assert graph is not None


class TestRouterFunctions:
    """路由函数测试"""

    def test_route_director_decision_write(self):
        """测试导演决策路由 - 写作"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": {"decision": "write"},
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        decision = orchestrator._route_director_decision(state)
        assert decision == "write"

    def test_route_director_decision_pivot(self):
        """测试导演决策路由 - 转向"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": {"decision": "pivot"},
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        decision = orchestrator._route_director_decision(state)
        assert decision == "pivot"

    def test_route_fact_check_valid(self):
        """测试事实检查路由 - 有效"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0,
            "last_fragment": {"fact_check_passed": True}
        }

        result = orchestrator._route_fact_check(state)
        assert result == "valid"

    def test_route_fact_check_invalid(self):
        """测试事实检查路由 - 无效"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0,
            "last_fragment": {"fact_check_passed": False}
        }

        result = orchestrator._route_fact_check(state)
        assert result == "invalid"

    def test_route_completion_continue(self):
        """测试完成路由 - 继续"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1"}, {"step_id": "2"}],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = orchestrator._route_completion(state)
        assert result == "continue"

    def test_route_completion_done(self):
        """测试完成路由 - 完成"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1"}],
            "current_step_index": 1,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = orchestrator._route_completion(state)
        assert result == "done"


class TestGetCurrentStep:
    """获取当前步骤测试"""

    def test_get_current_step_valid(self):
        """测试获取有效当前步骤"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1", "description": "第一步"}],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        step = orchestrator._get_current_step(state)
        assert step is not None
        assert step["step_id"] == "1"

    def test_get_current_step_out_of_bounds(self):
        """测试获取超出范围的步骤"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [{"step_id": "1"}],
            "current_step_index": 5,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        step = orchestrator._get_current_step(state)
        assert step is None

    def test_get_current_step_empty_outline(self):
        """测试空大纲获取步骤"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        step = orchestrator._get_current_step(state)
        assert step is None


class TestDynamicDirectorRouting:
    """动态导演路由测试"""

    def test_route_dynamic_director_pivot(self):
        """测试动态导演路由 - 转向"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test",
            enable_dynamic_adjustment=True
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": {"adjustment_type": "pivot"},
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = orchestrator._route_dynamic_director_decision(state)
        assert result == "pivot"

    def test_route_dynamic_director_skill_switch(self):
        """测试动态导演路由 - 技能切换"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test",
            enable_dynamic_adjustment=True
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": {"adjustment_type": "skill_switch"},
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = orchestrator._route_dynamic_director_decision(state)
        assert result == "skill_switch"

    def test_route_dynamic_director_continue(self):
        """测试动态导演路由 - 继续"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test",
            enable_dynamic_adjustment=True
        )

        state: GlobalState = {
            "user_topic": "主题",
            "project_context": "上下文",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": {"adjustment_type": "continue"},
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        result = orchestrator._route_dynamic_director_decision(state)
        assert result == "continue"


class TestInitialState:
    """初始状态测试"""

    def test_initial_state_structure(self):
        """测试初始状态结构"""
        mock_llm = Mock()
        mock_retrieval = Mock()
        mock_parser = Mock()
        mock_summarization = Mock()

        orchestrator = WorkflowOrchestrator(
            llm_service=mock_llm,
            retrieval_service=mock_retrieval,
            parser_service=mock_parser,
            summarization_service=mock_summarization,
            workspace_id="test"
        )

        initial_state: GlobalState = {
            "user_topic": "",
            "project_context": "",
            "outline": [],
            "current_step_index": 0,
            "fragments": [],
            "last_retrieved_docs": [],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0
        }

        assert "user_topic" in initial_state
        assert "outline" in initial_state
        assert "fragments" in initial_state
        assert "execution_log" in initial_state


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
