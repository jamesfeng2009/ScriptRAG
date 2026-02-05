"""Property-based tests for director approval behavior preventing pivots

This module tests that when the mock director always returns "approved",
the workflow never triggers a pivot, ensuring linear execution without loops.

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
async def test_director_always_approves_prevents_pivot(user_topic, project_context):
    """Property 11: Director always approves prevents pivot
    
    For any workflow execution using the mock director that always returns
    "approved", the state.pivot_triggered flag should remain False throughout
    execution, preventing pivot loops.
    
    **Validates: Requirements 4.3**
    """
    from src.domain.models import OutlineStep
    
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = create_mock_summarization_service()
    
    outline = [
        OutlineStep(
            step_id=0,
            title="第一步",
            description=f"关于 {user_topic} 的第一步内容",
            status="pending",
            retry_count=0
        ),
        OutlineStep(
            step_id=1,
            title="第二步",
            description=f"关于 {user_topic} 的第二步内容",
            status="pending",
            retry_count=0
        )
    ]
    
    initial_state = SharedState(
        user_topic=user_topic,
        project_context=project_context,
        outline=outline,
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
    
    pivot_triggered = final_state.get("pivot_triggered")
    assert pivot_triggered is False, \
        f"Pivot was triggered when director always approves. Reason: {final_state.get('pivot_reason')}"
    
    execution_log = final_state.get("execution_log", [])
    pivot_events = [
        log for log in execution_log
        if log.get("agent_name") == "pivot_manager"
    ]
    
    assert len(pivot_events) == 0, \
        f"Pivot manager was executed {len(pivot_events)} times when it shouldn't have been"
