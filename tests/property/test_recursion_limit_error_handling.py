"""Property-based tests for recursion limit error handling

Property 10: Exceeding recursion limit raises clear error
For any workflow that exceeds the configured recursion_limit,
the Orchestrator should return a failure result with an error
message containing "recursion limit" and the limit value.
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import Mock, AsyncMock
from src.application.orchestrator import WorkflowOrchestrator
from src.domain.models import SharedState


def create_mock_orchestrator():
    """Create orchestrator instance with mock services"""
    mock_services = {
        "llm_service": Mock(),
        "retrieval_service": Mock(),
        "parser_service": Mock(),
        "summarization_service": Mock(),
        "workspace_id": "test_workspace"
    }
    return WorkflowOrchestrator(**mock_services)


@pytest.mark.asyncio
@given(recursion_limit=st.integers(min_value=1, max_value=50))
@settings(max_examples=100, deadline=None)
async def test_recursion_limit_exceeded_raises_clear_error(recursion_limit: int):
    """
    Property 10: Exceeding recursion limit raises clear error
    
    For any recursion_limit value, when the workflow exceeds that limit,
    the orchestrator should return a failure result with a clear error
    message containing "recursion limit" and the limit value.
    
    Validates: Requirement 3.5
    """
    # Create orchestrator for this test iteration
    orchestrator = create_mock_orchestrator()
    
    # Create initial state
    initial_state = SharedState(
        user_topic="Test topic for recursion error",
        project_context="Testing recursion limit error handling"
    )
    
    # Mock the graph.ainvoke to raise RecursionError
    orchestrator.graph.ainvoke = AsyncMock(
        side_effect=RecursionError("maximum recursion depth exceeded")
    )
    
    # Execute with the generated recursion_limit
    result = await orchestrator.execute(initial_state, recursion_limit=recursion_limit)
    
    # Verify execution failed
    assert result["success"] is False
    
    # Verify error message contains "recursion limit"
    error_msg = result["error"].lower()
    assert "recursion limit" in error_msg or "recursion" in error_msg
    
    # Verify error message contains the limit value
    assert str(recursion_limit) in result["error"]
    
    # Verify state is included in result
    assert "state" in result
    assert result["state"] == initial_state
    
    # Verify execution_log is included
    assert "execution_log" in result


@pytest.mark.asyncio
@given(
    recursion_limit=st.integers(min_value=5, max_value=100),
    error_message=st.text(min_size=10, max_size=100)
)
@settings(max_examples=100, deadline=None)
async def test_recursion_error_handling_with_various_messages(
    recursion_limit: int,
    error_message: str
):
    """
    Property 10 (Extended): Recursion error handling works with various error messages
    
    For any recursion_limit and RecursionError message,
    the orchestrator should consistently return a clear error result.
    
    Validates: Requirement 3.5
    """
    # Create orchestrator for this test iteration
    orchestrator = create_mock_orchestrator()
    
    # Create initial state
    initial_state = SharedState(
        user_topic="Test topic",
        project_context="Test context"
    )
    
    # Mock the graph.ainvoke to raise RecursionError with custom message
    orchestrator.graph.ainvoke = AsyncMock(
        side_effect=RecursionError(error_message)
    )
    
    # Execute with the generated recursion_limit
    result = await orchestrator.execute(initial_state, recursion_limit=recursion_limit)
    
    # Verify execution failed
    assert result["success"] is False
    
    # Verify error message is clear and contains recursion limit info
    assert "error" in result
    assert str(recursion_limit) in result["error"]
    
    # Verify the error message follows expected format
    expected_prefix = f"Workflow exceeded recursion limit of {recursion_limit}"
    assert result["error"].startswith(expected_prefix)


@pytest.mark.asyncio
async def test_recursion_error_preserves_execution_state():
    """
    Property 10 (State preservation): Recursion error preserves execution state
    
    When a RecursionError occurs, the orchestrator should preserve
    the current state and execution log for debugging.
    
    Validates: Requirement 3.5
    """
    # Create orchestrator
    orchestrator = create_mock_orchestrator()
    
    # Create initial state with some execution log entries
    initial_state = SharedState(
        user_topic="Test topic",
        project_context="Test context"
    )
    initial_state.add_log_entry(
        agent_name="test_agent",
        action="test_action",
        details={"test": "data"}
    )
    
    # Mock the graph.ainvoke to raise RecursionError
    orchestrator.graph.ainvoke = AsyncMock(
        side_effect=RecursionError("maximum recursion depth exceeded")
    )
    
    # Execute with a low recursion_limit
    result = await orchestrator.execute(initial_state, recursion_limit=10)
    
    # Verify execution failed
    assert result["success"] is False
    
    # Verify state is preserved
    assert result["state"] == initial_state
    
    # Verify execution_log is preserved
    assert len(result["execution_log"]) > 0
    assert result["execution_log"][0]["agent_name"] == "test_agent"


@pytest.mark.asyncio
async def test_non_recursion_errors_handled_differently():
    """
    Property 10 (Error differentiation): Non-recursion errors are handled differently
    
    When a non-RecursionError exception occurs, the orchestrator should
    handle it with a generic error message, not the recursion limit message.
    
    Validates: Requirement 3.5
    """
    # Create orchestrator
    orchestrator = create_mock_orchestrator()
    
    # Create initial state
    initial_state = SharedState(
        user_topic="Test topic",
        project_context="Test context"
    )
    
    # Mock the graph.ainvoke to raise a different exception
    orchestrator.graph.ainvoke = AsyncMock(
        side_effect=ValueError("Some other error")
    )
    
    # Execute
    result = await orchestrator.execute(initial_state, recursion_limit=25)
    
    # Verify execution failed
    assert result["success"] is False
    
    # Verify error message does NOT contain "recursion limit"
    error_msg = result["error"].lower()
    assert "recursion limit" not in error_msg
    
    # Verify error message contains the actual error
    assert "some other error" in error_msg.lower()
