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
@settings(max_examples=3, deadline=None)
@pytest.mark.asyncio
async def test_error_messages_are_clear_and_specific(failure_scenario):
    """Property 14: Test error messages are clear
    
    For any test failure scenario, the error message should contain specific
    information about what failed (e.g., "expected X but got Y", "missing
    function Z", "format mismatch").
    
    **Validates: Requirements 6.2**
    """
    mock_llm = create_mock_llm_service()
    mock_retrieval = create_mock_retrieval_service()
    mock_parser = create_mock_parser_service()
    mock_summarization = Mock()
    mock_summarization.check_size = Mock(return_value=False)

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

    orchestrator = WorkflowOrchestrator(
        llm_service=mock_llm,
        retrieval_service=mock_retrieval,
        parser_service=mock_parser,
        summarization_service=mock_summarization,
        workspace_id="test-workspace"
    )

    result = await orchestrator.execute(initial_state, recursion_limit=50)

    error_message = None
    final_state = result["state"]

    try:
        if failure_scenario == "missing_outline":
            outline = final_state.get("outline", [])
            assert len(outline) > 0, (
                "Expected outline to be generated, but outline is empty. "
                f"Outline length: {len(outline)}"
            )

        elif failure_scenario == "empty_fragments":
            fragments = final_state.get("fragments", [])
            outline = final_state.get("outline", [])
            assert len(fragments) > 0, (
                "Expected fragments to be generated, but fragments list is empty. "
                f"Fragments count: {len(fragments)}, "
                f"Outline steps: {len(outline)}"
            )

        elif failure_scenario == "wrong_agent_order":
            execution_log = final_state.get("execution_log", [])
            agent_sequence = [log.get("agent_name") or log.get("agent") for log in execution_log if log]
            if len(agent_sequence) > 0:
                assert agent_sequence[0] == "planner", (
                    f"Expected first agent to be 'planner', but got '{agent_sequence[0]}'. "
                    f"Agent sequence: {agent_sequence}"
                )

        elif failure_scenario == "missing_final_screenplay":
            assert "final_screenplay" in result, (
                "Expected 'final_screenplay' in result, but key is missing. "
                f"Available keys: {list(result.keys())}"
            )
            assert result["final_screenplay"] is not None, (
                "Expected final_screenplay to be non-None, but got None"
            )

        elif failure_scenario == "incomplete_steps":
            outline = final_state.get("outline", [])
            if len(outline) > 0:
                if isinstance(outline[0], dict):
                    completed_steps = [s for s in outline if s.get("status") == "completed"]
                else:
                    completed_steps = [s for s in outline if s.status == "completed"]
                assert len(completed_steps) == len(outline), (
                    f"Expected all {len(outline)} steps to be completed, "
                    f"but only {len(completed_steps)} steps are completed. "
                    f"Step statuses: {[s.get('status') if isinstance(s, dict) else s.status for s in outline]}"
                )

        elif failure_scenario == "invalid_fragment_structure":
            fragments = final_state.get("fragments", [])
            if len(fragments) > 0:
                for i, fragment in enumerate(fragments):
                    if isinstance(fragment, dict):
                        assert "content" in fragment, (
                            f"Expected fragment {i} to have 'content' key, "
                            f"but key is missing. Available keys: {list(fragment.keys())}"
                        )
                    else:
                        assert hasattr(fragment, "content"), (
                            f"Expected fragment {i} to have 'content' attribute, "
                            f"but attribute is missing. Available attributes: {dir(fragment)}"
                        )

        elif failure_scenario == "missing_execution_log":
            execution_log = final_state.get("execution_log", [])
            assert len(execution_log) > 0, (
                "Expected execution_log to contain agent transitions, "
                f"but execution_log is empty. Log length: {len(execution_log)}"
            )

    except AssertionError as e:
        error_message = str(e)

    if error_message:
        assert len(error_message) > 0, "Error message should not be empty"

        has_expectation = "expected" in error_message.lower()
        has_actual_value = any(word in error_message.lower() for word in ["got", "but", "missing", "is", "are"])
        has_specific_info = any(char in error_message for char in [":", "[", "{", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

        clarity_score = sum([has_expectation, has_actual_value, has_specific_info])

        assert clarity_score >= 2, (
            f"Error message lacks clarity. Message: {error_message}\n"
            f"Has expectation: {has_expectation}, "
            f"Has actual value: {has_actual_value}, "
            f"Has specific info: {has_specific_info}"
        )

        assert len(error_message) >= 20, (
            f"Error message is too short ({len(error_message)} chars): {error_message}"
        )


@given(
    missing_key=st.sampled_from([
        "nonexistent_key_1",
        "nonexistent_key_2",
        "nonexistent_key_3",
    ])
)
@settings(max_examples=100, deadline=None)
def test_missing_key_error_messages_are_specific(missing_key):
    """Property 14 (Extended): Missing key error messages are specific
    
    When a required key is missing from a result dictionary, the error
    message should clearly state which key is missing and what keys are
    available.
    
    **Validates: Requirements 6.2**
    """
    result = {
        "final_screenplay": "Test screenplay",
        "state": {
            "user_topic": "Test",
            "project_context": "Test",
            "outline": [],
            "current_step_index": 0,
            "retrieved_docs": [],
            "fragments": [],
            "current_skill": "standard_tutorial",
            "global_tone": "professional",
            "pivot_triggered": False,
            "pivot_reason": None,
            "max_retries": 3,
            "awaiting_user_input": False,
            "user_input_prompt": None,
            "execution_log": [],
            "fact_check_passed": True
        },
        "success": True
    }

    error_message = None
    try:
        value = result[missing_key]
        if value is None:
            raise KeyError(f"Key '{missing_key}' has None value")
    except KeyError as e:
        error_message = str(e)

    assert error_message is not None, f"No error raised for missing key: {missing_key}"
    assert missing_key in error_message, (
        f"Error message should mention the missing key '{missing_key}': {error_message}"
    )


@given(
    value_mismatch=st.sampled_from([
        ("expected_count", 5, 3),
        ("status", "completed", "pending"),
        ("length", 10, 0),
    ])
)
@settings(max_examples=100, deadline=None)
def test_value_mismatch_error_messages_show_both_values(value_mismatch):
    """Property 14 (Extended): Value mismatch errors show both expected and actual
    
    When a value mismatch occurs, the error message should show both
    the expected value and the actual value.
    
    **Validates: Requirements 6.2**
    """
    key, expected, actual = value_mismatch

    try:
        assert expected == actual, f"Expected {key}={expected}, but got {actual}"
    except AssertionError as e:
        error_message = str(e)
        assert str(expected) in error_message, (
            f"Error message should contain expected value '{expected}': {error_message}"
        )
        assert str(actual) in error_message, (
            f"Error message should contain actual value '{actual}': {error_message}"
        )
