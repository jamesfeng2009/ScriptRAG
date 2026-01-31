"""Property-based tests for test error message clarity

This module tests that test failures produce clear, specific error messages
that help developers understand what went wrong.
"""

import pytest
from hypothesis import given, strategies as st, settings
import re
from unittest.mock import Mock, AsyncMock
from src.domain.models import (
    SharedState,
    OutlineStep,
    ScreenplayFragment,
    RetrievedDocument
)
from src.application.orchestrator import WorkflowOrchestrator
from tests.fixtures.realistic_mock_data import (
    create_mock_llm_service,
    create_mock_retrieval_service,
    create_mock_parser_service
)


@given(
    failure_scenario=st.sampled_from([
        "missing_outline",
        "empty_fragments",
        "wrong_agent_order",
        "missing_final_screenplay",
        "incomplete_steps",
        "invalid_fragment_structure",
        "missing_execution_log"
    ])
)
@settings(max_examples=100, deadline=None)
@pytest.mark.asyncio
async def test_error_messages_are_clear_and_specific(failure_scenario):
    """Property 14: Test error messages are clear
    
    Feature: fix-integration-test-mock-data
    
    For any test failure scenario, the error message should contain specific
    information about what failed (e.g., "expected X but got Y", "missing
    function Z", "format mismatch").
    
    **Validates: Requirements 6.2**
    """
    # Create mock services
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = Mock()
    mock_summarization.check_size = Mock(return_value=False)
    
    # Create initial state
    initial_state = SharedState(
        user_topic="Test topic",
        project_context="Test context",
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
    
    # Simulate different failure scenarios and verify error messages
    error_message = None
    
    try:
        if failure_scenario == "missing_outline":
            # Simulate assertion failure for missing outline
            final_state = result["state"]
            assert len(final_state.outline) > 0, (
                "Expected outline to be generated, but outline is empty. "
                f"Outline length: {len(final_state.outline)}"
            )
        
        elif failure_scenario == "empty_fragments":
            # Simulate assertion failure for empty fragments
            final_state = result["state"]
            assert len(final_state.fragments) > 0, (
                "Expected fragments to be generated, but fragments list is empty. "
                f"Fragments count: {len(final_state.fragments)}, "
                f"Outline steps: {len(final_state.outline)}"
            )
        
        elif failure_scenario == "wrong_agent_order":
            # Simulate assertion failure for wrong agent order
            final_state = result["state"]
            agent_sequence = [log.get("agent_name") for log in final_state.execution_log if "agent_name" in log]
            if len(agent_sequence) > 0:
                assert agent_sequence[0] == "planner", (
                    f"Expected first agent to be 'planner', but got '{agent_sequence[0]}'. "
                    f"Agent sequence: {agent_sequence}"
                )
        
        elif failure_scenario == "missing_final_screenplay":
            # Simulate assertion failure for missing final screenplay
            assert "final_screenplay" in result, (
                "Expected 'final_screenplay' in result, but key is missing. "
                f"Available keys: {list(result.keys())}"
            )
            assert result["final_screenplay"] is not None, (
                "Expected final_screenplay to be non-None, but got None"
            )
        
        elif failure_scenario == "incomplete_steps":
            # Simulate assertion failure for incomplete steps
            final_state = result["state"]
            if len(final_state.outline) > 0:
                completed_steps = [s for s in final_state.outline if s.status == "completed"]
                assert len(completed_steps) == len(final_state.outline), (
                    f"Expected all {len(final_state.outline)} steps to be completed, "
                    f"but only {len(completed_steps)} steps are completed. "
                    f"Step statuses: {[s.status for s in final_state.outline]}"
                )
        
        elif failure_scenario == "invalid_fragment_structure":
            # Simulate assertion failure for invalid fragment structure
            final_state = result["state"]
            if len(final_state.fragments) > 0:
                for i, fragment in enumerate(final_state.fragments):
                    assert isinstance(fragment, ScreenplayFragment), (
                        f"Expected fragment {i} to be ScreenplayFragment, "
                        f"but got {type(fragment).__name__}"
                    )
                    assert hasattr(fragment, "content"), (
                        f"Expected fragment {i} to have 'content' attribute, "
                        f"but attribute is missing. Available attributes: {dir(fragment)}"
                    )
        
        elif failure_scenario == "missing_execution_log":
            # Simulate assertion failure for missing execution log
            final_state = result["state"]
            assert len(final_state.execution_log) > 0, (
                "Expected execution_log to contain agent transitions, "
                f"but execution_log is empty. Log length: {len(final_state.execution_log)}"
            )
    
    except AssertionError as e:
        error_message = str(e)
    
    # If we caught an error, verify it's clear and specific
    if error_message:
        # Error message should be non-empty
        assert len(error_message) > 0, "Error message should not be empty"
        
        # Error message should contain specific information
        # Check for common patterns of clear error messages:
        # 1. Contains "expected" or "Expected"
        # 2. Contains actual values (numbers, strings, lists)
        # 3. Contains comparison information ("but got", "but", "missing")
        
        has_expectation = "expected" in error_message.lower()
        has_actual_value = any(word in error_message.lower() for word in ["got", "but", "missing", "is", "are"])
        has_specific_info = any(char in error_message for char in [":", "[", "{", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
        
        # At least 2 of these patterns should be present for a clear error message
        clarity_score = sum([has_expectation, has_actual_value, has_specific_info])
        
        assert clarity_score >= 2, (
            f"Error message lacks clarity. Message: {error_message}\n"
            f"Has expectation: {has_expectation}, "
            f"Has actual value: {has_actual_value}, "
            f"Has specific info: {has_specific_info}"
        )
        
        # Error message should not be too short (at least 20 characters)
        assert len(error_message) >= 20, (
            f"Error message is too short ({len(error_message)} chars): {error_message}"
        )


@given(
    missing_key=st.sampled_from([
        "final_screenplay",
        "state",
        "success",
        "execution_log"
    ])
)
@settings(max_examples=100, deadline=None)
def test_missing_key_error_messages_are_specific(missing_key):
    """Property 14 (Extended): Missing key error messages are specific
    
    Feature: fix-integration-test-mock-data
    
    When a required key is missing from a result dictionary, the error
    message should clearly state which key is missing and what keys are
    available.
    
    **Validates: Requirements 6.2**
    """
    # Simulate a result dictionary with a missing key
    result = {
        "final_screenplay": "Test screenplay",
        "state": SharedState(
            user_topic="Test",
            project_context="Test",
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
        ),
        "success": True
    }
    
    # Remove the key we're testing
    if missing_key in result:
        del result[missing_key]
    
    # Try to access the missing key and verify error message
    error_message = None
    try:
        assert missing_key in result, (
            f"Expected '{missing_key}' in result, but key is missing. "
            f"Available keys: {list(result.keys())}"
        )
    except AssertionError as e:
        error_message = str(e)
    
    # Verify error message is clear
    assert error_message is not None
    assert missing_key in error_message, (
        f"Error message should mention the missing key '{missing_key}'"
    )
    assert "Available keys" in error_message or "keys" in error_message, (
        "Error message should list available keys"
    )


@given(
    expected_value=st.sampled_from(["planner", "director", "writer", "fact_checker", "compiler"]),
    actual_value=st.sampled_from(["navigator", "unknown", "invalid", "", None])
)
@settings(max_examples=100, deadline=None)
def test_value_mismatch_error_messages_show_both_values(expected_value, actual_value):
    """Property 14 (Extended): Value mismatch errors show expected and actual
    
    Feature: fix-integration-test-mock-data
    
    When an assertion fails due to value mismatch, the error message should
    clearly show both the expected value and the actual value.
    
    **Validates: Requirements 6.2**
    """
    # Simulate a value mismatch assertion
    error_message = None
    try:
        assert actual_value == expected_value, (
            f"Expected value to be '{expected_value}', but got '{actual_value}'"
        )
    except AssertionError as e:
        error_message = str(e)
    
    # Verify error message contains both values
    if error_message:
        assert str(expected_value) in error_message, (
            f"Error message should contain expected value '{expected_value}'"
        )
        assert str(actual_value) in error_message, (
            f"Error message should contain actual value '{actual_value}'"
        )
        assert "expected" in error_message.lower(), (
            "Error message should use 'expected' keyword"
        )
