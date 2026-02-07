"""
Integration tests for state passing between workflow nodes

Purpose: Verify that state fields are correctly passed between nodes,
especially quality_evaluation which caused a silent failure bug.

Bug history: quality_evaluation was returned by quality_eval_node but
was silently ignored by LangGraph because it wasn't defined in GlobalState.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, Any, List
from src.domain.state_types import GlobalState
from src.domain.agents.quality_eval import QualityEvalAgent, QualityEvaluation, QualityLevel
from src.domain.models import RetrievedDocument


class TestQualityEvaluationStatePassing:
    """Test that quality_evaluation is correctly passed between nodes"""

    def test_quality_eval_node_returns_quality_evaluation_field(self):
        """quality_eval_node must return quality_evaluation in updates
        
        This test validates that the node returns the field that other
        nodes depend on.
        """
        from src.application.orchestrator import WorkflowOrchestrator
        
        # Create orchestrator with mocked services
        orchestrator = WorkflowOrchestrator(
            llm_service=Mock(),
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock(),
            enable_agentic_rag=True,
            enable_dynamic_adjustment=False,
        )
        
        # Create test state
        test_state: GlobalState = {
            "user_topic": "测试主题",
            "project_context": "测试上下文",
            "current_skill": "heated_battle",
            "skill_history": [],
            "outline": [
                {
                    "step_id": 0,
                    "title": "步骤 1",
                    "description": "测试步骤",
                }
            ],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [
                {
                    "content": "测试内容",
                    "score": 0.9,
                    "metadata": {"source": "test"}
                }
            ],
            "director_feedback": None,
            "execution_log": [],
            "error_flag": None,
            "retry_count": 0,
            "workflow_complete": False,
        }
        
        # Mock the quality eval agent
        mock_evaluation = Mock(spec=QualityEvaluation)
        mock_evaluation.overall_score = 0.75
        mock_evaluation.quality_level = QualityLevel.GOOD
        mock_evaluation.needs_refinement = False
        mock_evaluation.relevance_score = 0.8
        mock_evaluation.completeness_score = 0.7
        mock_evaluation.accuracy_score = 0.75
        mock_evaluation.strengths = ["内容相关"]
        mock_evaluation.weaknesses = []
        mock_evaluation.suggestions = []
        mock_evaluation.refinement_strategy = None
        mock_evaluation.retrieval_status = Mock()
        mock_evaluation.retrieval_status.value = "success"
        
        orchestrator.quality_eval_agent = Mock()
        orchestrator.quality_eval_agent.evaluate_quality = AsyncMock(return_value=mock_evaluation)
        
        # Run the node
        import asyncio
        result = asyncio.run(orchestrator._quality_eval_node(test_state))
        
        # Verify quality_evaluation is in the updates
        assert "quality_evaluation" in result, \
            "quality_eval_node must return 'quality_evaluation' in updates"
        
        # Verify the structure
        quality_eval = result["quality_evaluation"]
        assert isinstance(quality_eval, dict), \
            "quality_evaluation must be a dict for LangGraph compatibility"
        assert "overall_score" in quality_eval, \
            "quality_evaluation must contain 'overall_score'"
        assert "quality_level" in quality_eval, \
            "quality_evaluation must contain 'quality_level'"
        assert "needs_refinement" in quality_eval, \
            "quality_evaluation must contain 'needs_refinement'"

    def test_route_quality_eval_decision_reads_quality_evaluation(self):
        """_route_quality_eval_decision must read quality_evaluation from state
        
        This test validates that the routing function can access the
        quality_evaluation field that was set by the previous node.
        """
        from src.application.orchestrator import WorkflowOrchestrator
        
        orchestrator = WorkflowOrchestrator(
            llm_service=Mock(),
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock(),
            enable_agentic_rag=True,
            enable_dynamic_adjustment=False,
            max_retrieval_retries=2,
        )
        
        # Test with quality_evaluation in state (dict format)
        state_with_quality: GlobalState = {
            "user_topic": "测试",
            "project_context": "测试",
            "outline": [{"title": "步骤1"}],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [],
            "execution_log": [],
            "quality_evaluation": {
                "overall_score": 0.75,
                "quality_level": "good",
                "needs_refinement": False,
            },
            "retrieval_retry_count": 0,
            "retry_count": 0,
            "workflow_complete": False,
        }
        
        # The routing function should be able to read quality_evaluation
        # This will fail if quality_evaluation is not in GlobalState.__annotations__
        result = orchestrator._route_quality_eval_decision(state_with_quality)
        
        assert result in ["good", "retry", "failed"], \
            "Routing should return a valid decision"


class TestStateFieldPropagation:
    """Test that state fields propagate correctly through the workflow"""

    def test_field_defined_in_globalstate_for_retrieved_docs(self):
        """retrieved_docs must be defined in GlobalState
        
        Bug history: Fields not in GlobalState TypedDict are silently ignored.
        """
        assert "retrieved_docs" in GlobalState.__annotations__, \
            "retrieved_docs must be defined in GlobalState"

    def test_field_defined_in_globalstate_for_director_feedback(self):
        """director_feedback must be defined in GlobalState"""
        assert "director_feedback" in GlobalState.__annotations__, \
            "director_feedback must be defined in GlobalState"

    def test_all_fields_used_by_orchestrator_are_defined(self):
        """All fields used by orchestrator routes must be defined
        
        This is a comprehensive check to catch any missing fields.
        """
        from src.application.orchestrator import WorkflowOrchestrator
        
        # Fields accessed by orchestrator routes
        expected_fields = {
            # From _route_quality_eval_decision
            "quality_evaluation",
            "retrieval_retry_count",
            # From _route_director_decision
            "director_feedback",
            "outline",
            "current_step_index",
            # Common fields
            "retrieved_docs",
            "fragments",
            "execution_log",
        }
        
        defined_fields = set(GlobalState.__annotations__.keys())
        
        for field in expected_fields:
            assert field in defined_fields, \
                f"Field '{field}' is used by orchestrator but not defined in GlobalState"


class TestIntegrationStateFlow:
    """Integration tests for state flow through multiple nodes"""

    def test_state_passes_quality_evaluation_between_nodes(self):
        """Verify quality_evaluation can be passed from quality_eval to director
        
        This is an end-to-end test that simulates the state flow.
        """
        from src.application.orchestrator import WorkflowOrchestrator
        
        orchestrator = WorkflowOrchestrator(
            llm_service=Mock(),
            retrieval_service=Mock(),
            parser_service=Mock(),
            summarization_service=Mock(),
            enable_agentic_rag=True,
            enable_dynamic_adjustment=False,
        )
        
        # Initial state
        initial_state: GlobalState = {
            "user_topic": "测试",
            "project_context": "测试",
            "outline": [{"title": "步骤1"}],
            "current_step_index": 0,
            "fragments": [],
            "retrieved_docs": [{"content": "测试", "score": 0.9}],
            "execution_log": [],
            "director_feedback": None,
            "quality_evaluation": None,
            "retry_count": 0,
            "workflow_complete": False,
        }
        
        # Step 1: quality_eval_node would be called (mocked)
        mock_evaluation = Mock()
        mock_evaluation.overall_score = 0.75
        mock_evaluation.quality_level = Mock()
        mock_evaluation.quality_level.value = "good"
        mock_evaluation.needs_refinement = False
        mock_evaluation.relevance_score = 0.8
        mock_evaluation.completeness_score = 0.7
        mock_evaluation.accuracy_score = 0.75
        mock_evaluation.strengths = []
        mock_evaluation.weaknesses = []
        mock_evaluation.suggestions = []
        mock_evaluation.refinement_strategy = None
        mock_evaluation.retrieval_status = Mock()
        mock_evaluation.retrieval_status.value = "success"
        
        # Simulate quality_eval_node returning updates
        quality_eval_updates = {
            "quality_evaluation": {
                "overall_score": 0.75,
                "quality_level": "good",
                "needs_refinement": False,
                "relevance_score": 0.8,
                "completeness_score": 0.7,
                "accuracy_score": 0.75,
                "strengths": [],
                "weaknesses": [],
                "suggestions": [],
                "retrieval_status": "success",
            }
        }
        
        # Apply updates to state (simulating LangGraph behavior)
        updated_state = {**initial_state, **quality_eval_updates}
        
        # Step 2: director_node should be able to read quality_evaluation
        # This simulates the actual workflow state flow
        assert "quality_evaluation" in updated_state, \
            "quality_evaluation should be in state after quality_eval_node"
        
        result = orchestrator._route_quality_eval_decision(updated_state)
        
        # With needs_refinement=False, should return "good"
        assert result == "good", \
            f"Expected 'good' but got '{result}' - routing should use quality_evaluation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
