"""Unit tests for WorkflowOrchestrator recursion_limit parameter

Tests verify:
- execute() accepts recursion_limit parameter
- Default value is 25
- Parameter is passed to LangGraph config
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from src.application.orchestrator import WorkflowOrchestrator
from src.domain.models import SharedState


@pytest.fixture
def mock_services():
    """Create mock services for orchestrator"""
    return {
        "llm_service": Mock(),
        "retrieval_service": Mock(),
        "parser_service": Mock(),
        "summarization_service": Mock(),
        "workspace_id": "test_workspace"
    }


@pytest.fixture
def orchestrator(mock_services):
    """Create orchestrator instance with mock services"""
    return WorkflowOrchestrator(**mock_services)


@pytest.fixture
def initial_state():
    """Create initial shared state for testing"""
    return SharedState(
        user_topic="Test topic",
        project_context="Test context"
    )


@pytest.mark.asyncio
async def test_execute_accepts_recursion_limit_parameter(orchestrator, initial_state):
    """
    Test that execute() method accepts recursion_limit parameter.
    """
    # Mock the graph.ainvoke to avoid actual execution
    mock_state_dict = {
        "user_topic": initial_state.user_topic,
        "project_context": initial_state.project_context,
        "outline": [],
        "fragments": [],
        "last_retrieved_docs": [],
        "execution_log": [],
        "current_step_index": 0,
        "error_flag": None,
        "retry_count": 0
    }
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state_dict)
    
    # Call execute with recursion_limit parameter
    result = await orchestrator.execute(initial_state, recursion_limit=50)
    
    # Verify the method accepts the parameter and completes
    assert result is not None
    assert isinstance(result, dict)
    assert "user_topic" in result


@pytest.mark.asyncio
async def test_execute_default_recursion_limit_is_25(orchestrator, initial_state):
    """
    Test that execute() method has default recursion_limit of 25.
    """
    # Mock the graph.ainvoke to capture config
    mock_state_dict = {
        "user_topic": initial_state.user_topic,
        "project_context": initial_state.project_context,
        "outline": [],
        "fragments": [],
        "last_retrieved_docs": [],
        "execution_log": [],
        "current_step_index": 0,
        "error_flag": None,
        "retry_count": 0
    }
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state_dict)
    
    # Call execute without recursion_limit parameter
    await orchestrator.execute(initial_state)
    
    # Verify ainvoke was called with default recursion_limit=25
    orchestrator.graph.ainvoke.assert_called_once()
    call_args = orchestrator.graph.ainvoke.call_args
    
    # Check that config parameter contains recursion_limit=25
    assert call_args[1]["config"]["recursion_limit"] == 25


@pytest.mark.asyncio
async def test_execute_passes_recursion_limit_to_langgraph(orchestrator, initial_state):
    """
    Test that recursion_limit parameter is passed to LangGraph config.
    """
    # Mock the graph.ainvoke to capture config
    mock_state_dict = {
        "user_topic": initial_state.user_topic,
        "project_context": initial_state.project_context,
        "outline": [],
        "fragments": [],
        "last_retrieved_docs": [],
        "execution_log": [],
        "current_step_index": 0,
        "error_flag": None,
        "retry_count": 0
    }
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state_dict)
    
    # Test with custom recursion_limit
    custom_limit = 100
    await orchestrator.execute(initial_state, recursion_limit=custom_limit)
    
    # Verify ainvoke was called with correct config
    orchestrator.graph.ainvoke.assert_called_once()
    call_args = orchestrator.graph.ainvoke.call_args
    
    # Check that config parameter contains the custom recursion_limit
    assert call_args[1]["config"]["recursion_limit"] == custom_limit


@pytest.mark.asyncio
async def test_execute_with_various_recursion_limits(orchestrator, initial_state):
    """
    Test that execute() works with various recursion_limit values.
    """
    # Mock the graph.ainvoke to return a valid state dict
    mock_state_dict = {
        "user_topic": initial_state.user_topic,
        "project_context": initial_state.project_context,
        "outline": [],
        "fragments": [],
        "retrieved_docs": [],
        "execution_log": [],
        "current_step_index": 0,
        "pivot_triggered": False,
        "pivot_reason": None,
        "fact_check_passed": True,
        "current_skill": "standard_tutorial",
        "skill_history": [],
        "retry_count": {},
        "updated_at": initial_state.updated_at
    }
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state_dict)
    
    # Test with different recursion_limit values
    test_limits = [10, 25, 50, 100, 200]
    
    for limit in test_limits:
        # Reset mock
        orchestrator.graph.ainvoke.reset_mock()
        orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state_dict)
        
        # Call execute with specific limit
        result = await orchestrator.execute(initial_state, recursion_limit=limit)
        
        # Verify success
        assert result is not None
        assert isinstance(result, dict)
        
        # Verify correct limit was passed
        call_args = orchestrator.graph.ainvoke.call_args
        assert call_args[1]["config"]["recursion_limit"] == limit
