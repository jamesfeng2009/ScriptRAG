"""
Unit tests for GlobalState field validation

Purpose: Ensure all workflow nodes use fields that are properly defined
in the GlobalState TypedDict. This prevents silent failures where
LangGraph ignores undefined fields.
"""

import pytest
from src.domain.state_types import GlobalState
from src.application.orchestrator import WorkflowOrchestrator


class TestGlobalStateFieldValidation:
    """Validate that all workflow-required fields are defined in GlobalState"""

    def test_quality_evaluation_field_is_defined(self):
        """quality_evaluation field must be defined for Agentic RAG workflow
        
        Bug history: Without this field, LangGraph silently ignores the field,
        causing quality_eval results to be lost between nodes.
        """
        assert "quality_evaluation" in GlobalState.__annotations__, \
            "quality_evaluation must be defined in GlobalState for Agentic RAG workflow"

    def test_all_retrieval_fields_are_defined(self):
        """All retrieval-related fields must be defined in GlobalState"""
        retrieval_fields = [
            "retrieved_docs",
        ]
        for field in retrieval_fields:
            assert field in GlobalState.__annotations__, \
                f"{field} must be defined in GlobalState"

    def test_all_navigation_fields_are_defined(self):
        """All navigation-related fields must be defined in GlobalState"""
        navigation_fields = [
            "director_feedback",
            "current_step_index",
            "outline",
        ]
        for field in navigation_fields:
            assert field in GlobalState.__annotations__, \
                f"{field} must be defined in GlobalState"

    def test_all_execution_fields_are_defined(self):
        """All execution-related fields must be defined in GlobalState"""
        execution_fields = [
            "fragments",
            "execution_log",
            "error_flag",
            "retry_count",
            "workflow_complete",
        ]
        for field in execution_fields:
            assert field in GlobalState.__annotations__, \
                f"{field} must be defined in GlobalState"


class TestStateFieldConsistency:
    """Test consistency between orchestrator and GlobalState"""

    def test_orchestrator_routes_use_defined_fields(self):
        """Routes should only access fields defined in GlobalState"""
        orchestrator = WorkflowOrchestrator(
            llm_service=None,
            retrieval_service=None,
            parser_service=None,
            summarization_service=None,
            enable_agentic_rag=True,
            enable_dynamic_adjustment=False,
        )

        # Fields accessed by _route_quality_eval_decision
        quality_eval_route_fields = ["quality_evaluation", "retrieval_retry_count"]

        for field in quality_eval_route_fields:
            assert field in GlobalState.__annotations__, \
                f"_route_quality_eval_decision accesses '{field}' which is not defined in GlobalState"

    def test_node_updates_use_defined_fields(self):
        """Node updates should only modify fields defined in GlobalState"""
        # This test validates that nodes don't create fields that
        # will be silently ignored by LangGraph

        # Known node update patterns
        node_update_fields = {
            "_quality_eval_node": ["quality_evaluation", "quality_suggestions", "quality_issues"],
            "_navigator_node": ["retrieved_docs", "intent_analysis"],
            "_director_node": ["director_feedback"],
        }

        for node_name, fields in node_update_fields.items():
            for field in fields:
                assert field in GlobalState.__annotations__, \
                    f"{node_name} updates '{field}' which is not defined in GlobalState"


class TestStateReducerCompatibility:
    """Test that state updates use compatible reducers"""

    def test_quality_evaluation_uses_overwrite_reducer(self):
        """quality_evaluation should use overwrite_reducer for transient data"""
        annotation = GlobalState.__annotations__.get("quality_evaluation", "")
        
        assert "overwrite_reducer" in str(annotation) or "overwrite" in str(annotation).lower(), \
            "quality_evaluation should use overwrite_reducer (transient data)"

    def test_execution_log_uses_audit_reducer(self):
        """execution_log should use audit_log_reducer for append-only"""
        annotation = GlobalState.__annotations__.get("execution_log", "")
        
        assert "audit_log_reducer" in str(annotation), \
            "execution_log should use audit_log_reducer (append-only)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
