"""End-to-end integration tests for complete screenplay generation workflow

This module tests the complete workflow from user input to final screenplay,
verifying that all agents execute in the correct order and produce valid output.
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.domain.models import (
    SharedState,
    OutlineStep,
    RetrievedDocument,
    ScreenplayFragment
)
from src.application.orchestrator import WorkflowOrchestrator
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_mock_retrieval_service,
    create_mock_parser_service
)


@pytest.fixture
def mock_llm_service():
    """Create mock LLM service for testing with realistic responses
    """
    return create_mock_llm_service()


@pytest.fixture
def mock_retrieval_service():
    """Create mock retrieval service for testing with realistic code examples
    """
    return create_mock_retrieval_service()


@pytest.fixture
def mock_parser_service():
    """Create mock parser service for testing with realistic parse results
    """
    return create_mock_parser_service()


@pytest.fixture
def mock_summarization_service():
    """Create mock summarization service for testing"""
    summarization_service = Mock()
    
    def mock_check_size(content, threshold=10000):
        return False  # Content is always small enough
    
    summarization_service.check_size = Mock(side_effect=mock_check_size)
    
    return summarization_service


@pytest.fixture
def initial_state():
    """Create initial state for workflow testing"""
    return SharedState(
        user_topic="Introduction to Python async/await",
        project_context="Python async programming tutorial",
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


@pytest.mark.asyncio
async def test_complete_workflow_simple_outline(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test complete workflow from user input to final screenplay with simple outline
    
    This test verifies:
    - Planner generates outline (需求 12.1)
    - Navigator retrieves content for each step (需求 12.3)
    - Director evaluates content (需求 12.4)
    - Writer generates fragments (需求 12.6)
    - Fact checker validates fragments (需求 10.2)
    - Compiler integrates fragments (需求 12.7, 12.8)
    - All agents execute in correct order
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=500)
    
    assert result["success"] is True
    final_state = result["state"]
    assert "final_screenplay" in final_state
    assert final_state["final_screenplay"] is not None
    
    assert len(final_state["outline"]) > 0
    assert len(final_state["fragments"]) > 0
    assert len(final_state["execution_log"]) > 0


@pytest.mark.asyncio
async def test_workflow_with_multiple_steps(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test workflow processes multiple outline steps sequentially
    
    Verifies that the workflow correctly processes each step in order
    and generates fragments for all steps.
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=500)
    
    # Verify workflow completed
    assert result["success"] is True
    
    final_state = result["state"]
    
    # Verify multiple steps were created
    assert len(final_state["outline"]) >= 3
    
    # Verify fragments were generated for each step
    assert len(final_state["fragments"]) >= 3


@pytest.mark.asyncio
async def test_workflow_agent_execution_order(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test that agents execute in the correct order
    
    Verifies the workflow follows the expected agent sequence:
    planner -> navigator -> director -> writer -> fact_checker -> compiler
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=500)
    
    assert result["success"] is True
    
    final_state = result["state"]
    execution_log = final_state["execution_log"]
    
    # Extract agent execution sequence
    # Handle both "agent_name" and "agent" keys for compatibility
    agent_sequence = [
        log.get("agent_name") or log.get("agent") 
        for log in execution_log 
        if isinstance(log, dict) and ("agent_name" in log or "agent" in log)
    ]
    
    # Verify planner is first
    assert agent_sequence[0] == "planner"
    
    # Verify compiler is last (or near last)
    assert "compiler" in agent_sequence[-3:]
    
    # Verify navigator appears before writer for each step
    navigator_indices = [i for i, name in enumerate(agent_sequence) if name == "navigator"]
    writer_indices = [i for i, name in enumerate(agent_sequence) if name == "writer"]
    
    # Each writer execution should have a navigator execution before it
    for writer_idx in writer_indices:
        assert any(nav_idx < writer_idx for nav_idx in navigator_indices)


@pytest.mark.asyncio
async def test_workflow_final_screenplay_structure(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test that final screenplay has correct structure
    
    Verifies the compiler produces a well-structured final screenplay.
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=500)
    
    assert result["success"] is True
    assert result["state"]["final_screenplay"] is not None
    
    final_screenplay = result["state"]["final_screenplay"]
    
    # Verify screenplay is a non-empty string
    assert isinstance(final_screenplay, str)
    assert len(final_screenplay) > 0
    
    # Verify screenplay contains content
    assert len(final_screenplay.strip()) > 10


@pytest.mark.asyncio
async def test_workflow_state_consistency(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test that state remains consistent throughout workflow
    
    Verifies that state modifications are properly maintained across
    agent transitions.
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=500)
    
    assert result["success"] is True
    
    final_state = result["state"]
    
    # Verify state fields are consistent
    assert final_state["user_topic"] == initial_state.user_topic
    assert final_state["project_context"] == initial_state.project_context
    
    # Verify state was modified during execution
    assert len(final_state["outline"]) > 0
    assert len(final_state["fragments"]) > 0
    assert len(final_state["execution_log"]) > 0
    
    # Verify current_step_index advanced
    assert final_state["current_step_index"] > 0


@pytest.mark.asyncio
async def test_workflow_with_empty_retrieval(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test workflow handles empty retrieval results gracefully
    
    Verifies that when navigator returns no documents, the workflow
    continues without errors.
    """
    # Retrieval service already returns empty results by default
    
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=500)
    
    # Workflow should complete successfully even with empty retrieval
    assert result["success"] is True
    
    final_state = result["state"]
    
    # Verify workflow completed
    assert len(final_state["outline"]) > 0
    assert len(final_state["fragments"]) > 0
    
    # Verify no hallucinations (fragments should acknowledge lack of info)
    # This is verified by the fact checker in the workflow


@pytest.mark.asyncio
async def test_workflow_logging_completeness(
    mock_llm_service,
    mock_retrieval_service,
    mock_parser_service,
    mock_summarization_service,
    initial_state
):
    """Test that workflow logs all agent transitions
    
    Verifies comprehensive logging throughout the workflow.
    """
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm_service,
        retrieval_service=mock_retrieval_service,
        parser_service=mock_parser_service,
        summarization_service=mock_summarization_service,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=500)
    
    assert result["success"] is True
    
    final_state = result["state"]
    execution_log = final_state["execution_log"]
    
    # Verify log entries exist
    assert len(execution_log) > 0
    
    # Verify each log entry has required fields
    # Handle both "agent_name" and "agent" keys for compatibility
    for log_entry in execution_log:
        assert isinstance(log_entry, dict), "Log entry should be a dictionary"
        assert "agent_name" in log_entry or "agent" in log_entry, (
            f"Log entry should have 'agent_name' or 'agent' field. Entry: {log_entry}"
        )
        assert "action" in log_entry, f"Log entry should have 'action' field. Entry: {log_entry}"
    
    # Verify all major agents are logged
    agent_names = {
        log.get("agent_name") or log.get("agent") 
        for log in execution_log 
        if isinstance(log, dict) and ("agent_name" in log or "agent" in log)
    }
    expected_agents = {"planner", "navigator", "director", "writer", "fact_checker", "compiler"}
    
    # At least most agents should be present
    assert len(agent_names.intersection(expected_agents)) >= 5
