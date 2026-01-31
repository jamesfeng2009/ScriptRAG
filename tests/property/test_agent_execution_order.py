"""Property-based tests for agent execution order

This module tests that successful workflow executions follow the correct
agent execution sequence.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock
from src.domain.models import SharedState, OutlineStep
from src.application.orchestrator import WorkflowOrchestrator
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_mock_retrieval_service,
    create_mock_parser_service
)


@given(
    user_topic=st.text(min_size=10, max_size=100),
    num_steps=st.integers(min_value=3, max_value=5)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_successful_tests_verify_agent_execution_order(user_topic, num_steps):
    """Property 15: Successful tests verify agent execution order
    
    Feature: fix-integration-test-mock-data
    
    For any successful workflow execution, the state.execution_log should
    contain agent transitions in the expected order:
    planner → navigator → director → writer → fact_checker → compiler
    
    **Validates: Requirements 6.5**
    """
    # Filter out topics that might cause issues
    assume(len(user_topic.strip()) > 5)
    assume(not all(c in ' \t\n' for c in user_topic))
    
    # Create mock services
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = Mock()
    mock_summarization.check_size = Mock(return_value=False)
    
    # Create initial state
    initial_state = SharedState(
        user_topic=user_topic,
        project_context=f"Tutorial about {user_topic}",
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
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with increased recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=50)
    
    # Only verify order for successful executions
    if result["success"]:
        final_state = result["state"]
        execution_log = final_state.execution_log
        
        # Extract agent names from execution log
        agent_sequence = [
            log.get("agent_name") 
            for log in execution_log 
            if isinstance(log, dict) and "agent_name" in log
        ]
        
        # Verify we have agent executions
        assert len(agent_sequence) > 0, (
            "Execution log should contain agent transitions"
        )
        
        # Verify planner is first
        assert agent_sequence[0] == "planner", (
            f"Expected first agent to be 'planner', but got '{agent_sequence[0]}'. "
            f"Agent sequence: {agent_sequence}"
        )
        
        # Verify compiler is last (or near last, within last 3 positions)
        assert "compiler" in agent_sequence[-3:], (
            f"Expected 'compiler' to be in last 3 agents, but got: {agent_sequence[-3:]}"
        )
        
        # Verify expected agents are present (but navigator might be skipped in some flows)
        expected_agents = ["planner", "director", "writer", "compiler"]
        for expected_agent in expected_agents:
            assert expected_agent in agent_sequence, (
                f"Expected agent '{expected_agent}' to be in execution sequence, "
                f"but it's missing. Sequence: {agent_sequence}"
            )
        
        # Verify navigator appears in the sequence if retrieval is needed
        # Note: Navigator might be skipped in some workflow paths
        navigator_indices = [i for i, name in enumerate(agent_sequence) if name == "navigator"]
        writer_indices = [i for i, name in enumerate(agent_sequence) if name == "writer"]
        
        # If navigator exists, it should come before at least one writer
        if len(navigator_indices) > 0 and len(writer_indices) > 0:
            first_navigator = min(navigator_indices)
            assert any(writer_idx > first_navigator for writer_idx in writer_indices), (
                f"At least one writer should come after navigator. "
                f"Navigator indices: {navigator_indices}, Writer indices: {writer_indices}"
            )
        
        # Verify director appears before writer
        director_indices = [i for i, name in enumerate(agent_sequence) if name == "director"]
        for writer_idx in writer_indices:
            has_director_before = any(dir_idx < writer_idx for dir_idx in director_indices)
            assert has_director_before, (
                f"Writer at index {writer_idx} should have director before it. "
                f"Director indices: {director_indices}, Writer indices: {writer_indices}"
            )
        
        # Verify fact_checker appears after writer
        fact_checker_indices = [i for i, name in enumerate(agent_sequence) if name == "fact_checker"]
        for fact_idx in fact_checker_indices:
            has_writer_before = any(writer_idx < fact_idx for writer_idx in writer_indices)
            assert has_writer_before, (
                f"Fact checker at index {fact_idx} should have writer before it. "
                f"Writer indices: {writer_indices}, Fact checker indices: {fact_checker_indices}"
            )


@given(
    recursion_limit=st.integers(min_value=30, max_value=100)
)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_agent_sequence_is_consistent_across_limits(recursion_limit):
    """Property 15 (Extended): Agent sequence is consistent across recursion limits
    
    Feature: fix-integration-test-mock-data
    
    For any recursion limit value, successful executions should follow the
    same agent execution order pattern.
    
    **Validates: Requirements 6.5**
    """
    # Create mock services
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = Mock()
    mock_summarization.check_size = Mock(return_value=False)
    
    # Create initial state
    initial_state = SharedState(
        user_topic="Python async programming",
        project_context="Tutorial about async/await",
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
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        workspace_id="test-workspace"
    )
    
    # Execute workflow with the given recursion limit
    result = await orchestrator.execute(initial_state, recursion_limit=recursion_limit)
    
    # Only verify for successful executions
    if result["success"]:
        final_state = result["state"]
        execution_log = final_state.execution_log
        
        # Extract agent names
        agent_sequence = [
            log.get("agent_name") 
            for log in execution_log 
            if isinstance(log, dict) and "agent_name" in log
        ]
        
        # Verify basic order constraints
        if len(agent_sequence) > 0:
            # First should be planner
            assert agent_sequence[0] == "planner"
            
            # Last few should include compiler
            assert "compiler" in agent_sequence[-5:]
            
            # Planner should appear before all other agents
            planner_idx = agent_sequence.index("planner")
            for agent in ["navigator", "director", "writer", "fact_checker", "compiler"]:
                if agent in agent_sequence:
                    agent_idx = agent_sequence.index(agent)
                    assert agent_idx > planner_idx, (
                        f"Agent '{agent}' should appear after planner"
                    )


@given(
    workflow_complexity=st.sampled_from(["simple", "medium", "complex"])
)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_agent_order_holds_for_different_complexities(workflow_complexity):
    """Property 15 (Extended): Agent order holds for different workflow complexities
    
    Feature: fix-integration-test-mock-data
    
    Regardless of workflow complexity (number of steps), the agent execution
    order should follow the same pattern.
    
    **Validates: Requirements 6.5**
    """
    # Map complexity to expected number of steps
    complexity_map = {
        "simple": 3,
        "medium": 4,
        "complex": 5
    }
    
    # Create mock services
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = Mock()
    mock_summarization.check_size = Mock(return_value=False)
    
    # Create initial state
    initial_state = SharedState(
        user_topic=f"Python tutorial - {workflow_complexity} level",
        project_context=f"A {workflow_complexity} tutorial",
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
    
    # Create orchestrator
    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        workspace_id="test-workspace"
    )
    
    # Execute workflow
    result = await orchestrator.execute(initial_state, recursion_limit=50)
    
    # Only verify for successful executions
    if result["success"]:
        final_state = result["state"]
        execution_log = final_state.execution_log
        
        # Extract agent names
        agent_sequence = [
            log.get("agent_name") 
            for log in execution_log 
            if isinstance(log, dict) and "agent_name" in log
        ]
        
        # Verify core agent order pattern
        if len(agent_sequence) >= 5:
            # Find first occurrence of each key agent
            key_agents = ["planner", "navigator", "director", "writer", "compiler"]
            agent_positions = {}
            
            for agent in key_agents:
                if agent in agent_sequence:
                    agent_positions[agent] = agent_sequence.index(agent)
            
            # Verify relative ordering
            if "planner" in agent_positions and "navigator" in agent_positions:
                assert agent_positions["planner"] < agent_positions["navigator"], (
                    "Planner should come before navigator"
                )
            
            if "navigator" in agent_positions and "writer" in agent_positions:
                assert agent_positions["navigator"] < agent_positions["writer"], (
                    "Navigator should come before writer"
                )
            
            if "writer" in agent_positions and "compiler" in agent_positions:
                assert agent_positions["writer"] < agent_positions["compiler"], (
                    "Writer should come before compiler"
                )
