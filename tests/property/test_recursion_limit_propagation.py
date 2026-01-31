"""Property-based tests for recursion limit propagation

Property 9: Recursion limit propagates to LangGraph
For any recursion_limit value provided to Orchestrator.execute(),
when the workflow is executed, the LangGraph configuration should
receive that limit value.
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
@given(recursion_limit=st.integers(min_value=10, max_value=100))
@settings(max_examples=100, deadline=None)
async def test_recursion_limit_propagates_to_langgraph(recursion_limit: int):
    """
    Property 9: Recursion limit propagates to LangGraph
    
    For any recursion_limit value between 10 and 100,
    when execute() is called, the LangGraph ainvoke should
    receive that exact limit in its config parameter.
    
    Validates: Requirement 3.2
    """
    # Create orchestrator for this test iteration
    orchestrator = create_mock_orchestrator()
    
    # Create initial state
    initial_state = SharedState(
        user_topic="Test topic for recursion limit",
        project_context="Testing recursion limit propagation"
    )
    
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
        "current_skill": "standard_tutorial",  # Use valid skill name
        "skill_history": [],
        "retry_count": {},
        "updated_at": initial_state.updated_at
    }
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state_dict)
    
    # Execute with the generated recursion_limit
    result = await orchestrator.execute(initial_state, recursion_limit=recursion_limit)
    
    # Verify ainvoke was called
    assert orchestrator.graph.ainvoke.called
    
    # Extract the config parameter from the call
    call_args = orchestrator.graph.ainvoke.call_args
    assert call_args is not None
    
    # Verify config contains the correct recursion_limit
    config = call_args[1].get("config", {})
    assert "recursion_limit" in config
    assert config["recursion_limit"] == recursion_limit
    
    # Verify execution succeeded
    assert result["success"] is True


@pytest.mark.asyncio
@given(
    recursion_limit=st.integers(min_value=1, max_value=200),
    user_topic=st.text(min_size=5, max_size=50)
)
@settings(max_examples=100, deadline=None)
async def test_recursion_limit_propagation_with_various_topics(
    recursion_limit: int,
    user_topic: str
):
    """
    Property 9 (Extended): Recursion limit propagates regardless of state content
    
    For any recursion_limit and user_topic combination,
    the recursion_limit should always propagate correctly to LangGraph.
    
    Validates: Requirement 3.2
    """
    # Create orchestrator for this test iteration
    orchestrator = create_mock_orchestrator()
    
    # Create state with generated topic
    state = SharedState(
        user_topic=user_topic.strip() or "default topic",
        project_context="Test context"
    )
    
    # Mock the graph.ainvoke to return a valid state dict
    mock_state_dict = {
        "user_topic": state.user_topic,
        "project_context": state.project_context,
        "outline": [],
        "fragments": [],
        "retrieved_docs": [],
        "execution_log": [],
        "current_step_index": 0,
        "pivot_triggered": False,
        "pivot_reason": None,
        "fact_check_passed": True,
        "current_skill": "standard_tutorial",  # Use valid skill name
        "skill_history": [],
        "retry_count": {},
        "updated_at": state.updated_at
    }
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state_dict)
    
    # Execute with the generated recursion_limit
    await orchestrator.execute(state, recursion_limit=recursion_limit)
    
    # Verify the recursion_limit was passed correctly
    call_args = orchestrator.graph.ainvoke.call_args
    config = call_args[1].get("config", {})
    assert config["recursion_limit"] == recursion_limit


@pytest.mark.asyncio
async def test_recursion_limit_default_propagation():
    """
    Property 9 (Default case): Default recursion limit propagates correctly
    
    When no recursion_limit is provided, the default value of 25
    should propagate to LangGraph.
    
    Validates: Requirement 3.2, 3.3
    """
    # Create orchestrator
    orchestrator = create_mock_orchestrator()
    
    # Create initial state
    initial_state = SharedState(
        user_topic="Test topic",
        project_context="Test context"
    )
    
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
        "current_skill": "standard_tutorial",  # Use valid skill name
        "skill_history": [],
        "retry_count": {},
        "updated_at": initial_state.updated_at
    }
    orchestrator.graph.ainvoke = AsyncMock(return_value=mock_state_dict)
    
    # Execute without recursion_limit parameter
    await orchestrator.execute(initial_state)
    
    # Verify default recursion_limit=25 was passed
    call_args = orchestrator.graph.ainvoke.call_args
    config = call_args[1].get("config", {})
    assert config["recursion_limit"] == 25
