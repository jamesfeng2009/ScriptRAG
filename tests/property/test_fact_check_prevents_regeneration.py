"""Property-based tests for fact check validation preventing regeneration

This module tests that when the mock fact_checker returns "VALID",
fragments are not removed and regeneration loops are prevented.

Feature: fix-integration-test-mock-data
"""

import pytest
from hypothesis import given, strategies as st, settings, HealthCheck
from unittest.mock import AsyncMock
from src.domain.models import SharedState
from src.application.orchestrator import WorkflowOrchestrator
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_mock_retrieval_service,
    create_mock_parser_service
)


def create_mock_summarization_service():
    """Create mock summarization service for testing"""
    from unittest.mock import Mock
    
    mock_service = Mock()
    mock_service.check_size = Mock(return_value=False)
    return mock_service


@pytest.mark.asyncio
@settings(
    max_examples=3,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow]
)
@given(
    user_topic=st.text(min_size=10, max_size=100),
    project_context=st.text(min_size=10, max_size=100)
)
async def test_valid_fact_check_prevents_regeneration(user_topic, project_context):
    """Property 12: Valid fact check prevents regeneration
    
    For any workflow execution where the mock fact_checker returns "VALID",
    the state.fact_check_passed flag should be True and fragments should
    not be removed from state.fragments, preventing regeneration loops.
    
    **Validates: Requirements 4.4**
    """
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = create_mock_summarization_service()
    
    initial_state = SharedState(
        user_topic=user_topic,
        project_context=project_context,
        outline=[],
        current_step_index=0,
        retrieved_docs=[],
        fragments=[],
        current_skill="standard_tutorial",
        global_tone="professional",
        pivot_triggered=False,
        pivot_reason=None,
        max_retries=3,
        awaiting_user_input=False,
        user_input_prompt=None,
        execution_log=[],
        fact_check_passed=True
    )
    
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        workspace_id="test-workspace"
    )
    
    result = await orchestrator.execute(initial_state, recursion_limit=50)
    
    final_state = result["state"]
    
    fact_check_passed = final_state.get("fact_check_passed")
    assert fact_check_passed is True, \
        f"Fact check failed when mock always returns VALID: {fact_check_passed}"
    
    if result.get("success"):
        fragments = final_state.get("fragments", [])
        assert len(fragments) > 0, \
            "No fragments found when fact check passed - they may have been removed"
        
        execution_log = final_state.get("execution_log", [])
        fact_check_events = [
            log for log in execution_log
            if log.get("agent_name") == "fact_checker"
        ]
        
        num_fragments = len(fragments)
        num_fact_checks = len(fact_check_events)
        
        assert num_fact_checks <= num_fragments * 2, \
            f"Too many fact checks ({num_fact_checks}) for {num_fragments} fragments - suggests regeneration loops"
