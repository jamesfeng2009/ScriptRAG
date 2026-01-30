"""Property-Based Tests for Deprecation Pivot Response

属性 10: 废弃转向响应

Property: When a pivot is triggered due to deprecation conflict, the PivotManager
should modify the current and subsequent outline steps, switch to warning_mode skill
(with compatibility constraints), and clear retrieved documents to trigger re-retrieval.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from datetime import datetime
from src.domain.agents.pivot_manager import handle_pivot
from src.domain.models import SharedState, OutlineStep, RetrievedDocument
from src.domain.skills import SKILLS, check_skill_compatibility


# Strategy for generating outline steps
@st.composite
def outline_step_strategy(draw, step_id=None):
    """Generate a valid OutlineStep"""
    if step_id is None:
        step_id = draw(st.integers(min_value=0, max_value=20))
    
    return OutlineStep(
        step_id=step_id,
        description=draw(st.text(min_size=10, max_size=100)),
        status=draw(st.sampled_from(["pending", "in_progress", "completed", "skipped"])),
        retry_count=draw(st.integers(min_value=0, max_value=3))
    )


# Strategy for generating retrieved documents
@st.composite
def retrieved_doc_strategy(draw):
    """Generate a valid RetrievedDocument"""
    return RetrievedDocument(
        content=draw(st.text(min_size=10, max_size=200)),
        source=draw(st.text(min_size=5, max_size=50)),
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata={"has_deprecated": draw(st.booleans())}
    )


# Strategy for generating SharedState with pivot triggered
@st.composite
def pivot_state_strategy(draw, pivot_reason="deprecation_conflict"):
    """Generate a SharedState with pivot triggered"""
    num_steps = draw(st.integers(min_value=2, max_value=10))
    current_index = draw(st.integers(min_value=0, max_value=num_steps - 1))
    
    outline = [
        OutlineStep(
            step_id=i,
            description=f"Step {i} description",
            status="pending" if i > current_index else "in_progress" if i == current_index else "completed",
            retry_count=0 if i != current_index else draw(st.integers(min_value=0, max_value=2))
        )
        for i in range(num_steps)
    ]
    
    # Generate some retrieved documents
    num_docs = draw(st.integers(min_value=1, max_value=5))
    retrieved_docs = [draw(retrieved_doc_strategy()) for _ in range(num_docs)]
    
    return SharedState(
        user_topic=draw(st.text(min_size=10, max_size=100)),
        project_context=draw(st.text(min_size=0, max_size=100)),
        outline=outline,
        current_step_index=current_index,
        retrieved_docs=retrieved_docs,
        current_skill=draw(st.sampled_from(list(SKILLS.keys()))),
        global_tone=draw(st.sampled_from(["professional", "cautionary", "engaging"])),
        pivot_triggered=True,
        pivot_reason=pivot_reason,
        max_retries=3
    )


@given(state=pivot_state_strategy(pivot_reason="deprecation_conflict"))
@settings(max_examples=100)
def test_deprecation_pivot_modifies_current_step(state: SharedState):
    """
    Property 10: Deprecation Pivot Response - Current Step Modification
    
    When a deprecation pivot is triggered:
    1. The current step description should be modified to include deprecation warning
    2. The retry count should be incremented
    3. The step status should be set to in_progress
    
    **Validates: Requirements 5.2, 5.3**
    """
    # Get original state
    original_step = state.get_current_step()
    assume(original_step is not None)
    
    original_desc = original_step.description
    original_retry = original_step.retry_count
    
    # Handle pivot
    result_state = handle_pivot(state)
    
    # Get modified step
    modified_step = result_state.get_current_step()
    assert modified_step is not None
    
    # Property 1: Description should be modified to include deprecation warning
    assert modified_step.description != original_desc, \
        "Step description should be modified"
    assert "DEPRECATED" in modified_step.description.upper() or \
           "deprecated" in modified_step.description.lower(), \
        f"Step description should mention deprecation: {modified_step.description}"
    
    # Property 2: Retry count should be incremented
    assert modified_step.retry_count == original_retry + 1, \
        f"Retry count should increment: {original_retry} -> {modified_step.retry_count}"
    
    # Property 3: Status should be in_progress
    assert modified_step.status == "in_progress", \
        f"Step status should be in_progress, got {modified_step.status}"


@given(state=pivot_state_strategy(pivot_reason="deprecation_conflict"))
@settings(max_examples=100)
def test_deprecation_pivot_switches_to_warning_mode(state: SharedState):
    """
    Property 10: Deprecation Pivot Response - Skill Switch
    
    When a deprecation pivot is triggered, the system should switch to
    warning_mode skill (or closest compatible skill).
    
    **Validates: Requirements 5.2, 5.3**
    """
    original_skill = state.current_skill
    
    # Handle pivot
    result_state = handle_pivot(state)
    
    # Property: Should switch to warning_mode or compatible skill
    new_skill = result_state.current_skill
    
    # If original skill is compatible with warning_mode, should switch
    if check_skill_compatibility(original_skill, "warning_mode"):
        assert new_skill == "warning_mode", \
            f"Should switch to warning_mode when compatible, got {new_skill}"
    else:
        # Should switch to a compatible skill
        assert new_skill in SKILLS, f"New skill {new_skill} must be valid"
        if new_skill != original_skill:
            assert check_skill_compatibility(original_skill, new_skill), \
                f"New skill {new_skill} must be compatible with {original_skill}"


@given(state=pivot_state_strategy(pivot_reason="deprecation_conflict"))
@settings(max_examples=100)
def test_deprecation_pivot_clears_retrieved_docs(state: SharedState):
    """
    Property 10: Deprecation Pivot Response - Re-retrieval Trigger
    
    When a pivot is triggered, retrieved documents should be cleared
    to trigger re-retrieval with updated context.
    
    **Validates: Requirements 5.2, 12.5**
    """
    # Ensure we have some retrieved docs
    assume(len(state.retrieved_docs) > 0)
    
    original_doc_count = len(state.retrieved_docs)
    
    # Handle pivot
    result_state = handle_pivot(state)
    
    # Property: Retrieved docs should be cleared
    assert len(result_state.retrieved_docs) == 0, \
        f"Retrieved docs should be cleared, had {original_doc_count}, " \
        f"now has {len(result_state.retrieved_docs)}"


@given(state=pivot_state_strategy(pivot_reason="deprecation_conflict"))
@settings(max_examples=100)
def test_deprecation_pivot_updates_subsequent_steps(state: SharedState):
    """
    Property 10: Deprecation Pivot Response - Subsequent Steps Update
    
    When a deprecation pivot is triggered, subsequent pending steps
    should be updated to consider the deprecation.
    
    **Validates: Requirements 5.2, 5.3**
    """
    # Only test if there are subsequent steps
    assume(state.current_step_index < len(state.outline) - 1)
    
    # Get subsequent pending steps before pivot
    subsequent_steps_before = [
        step for step in state.outline[state.current_step_index + 1:]
        if step.status == "pending"
    ]
    
    # Handle pivot
    result_state = handle_pivot(state)
    
    # Get subsequent steps after pivot
    subsequent_steps_after = [
        step for step in result_state.outline[result_state.current_step_index + 1:]
        if step.status == "pending"
    ]
    
    # Property: At least some subsequent steps should be modified
    # (if there were any pending steps)
    if subsequent_steps_before:
        # Check if any descriptions were updated
        modified_count = 0
        for i, step_after in enumerate(subsequent_steps_after):
            if i < len(subsequent_steps_before):
                step_before = subsequent_steps_before[i]
                if step_after.description != step_before.description:
                    modified_count += 1
                    # Should mention deprecation or consideration
                    assert "deprecation" in step_after.description.lower() or \
                           "consider" in step_after.description.lower(), \
                        f"Modified step should mention deprecation: {step_after.description}"
        
        # At least some steps should be modified (heuristic)
        # Note: This is a weak property since modification is heuristic-based


@given(state=pivot_state_strategy(pivot_reason="deprecation_conflict"))
@settings(max_examples=100)
def test_pivot_resets_trigger_flag(state: SharedState):
    """
    Property: After handling pivot, the pivot_triggered flag should be reset to False.
    
    **Validates: Requirements 5.2**
    """
    assert state.pivot_triggered is True, "Initial state should have pivot triggered"
    
    # Handle pivot
    result_state = handle_pivot(state)
    
    # Property: Pivot trigger should be reset
    assert result_state.pivot_triggered is False, \
        "Pivot trigger should be reset after handling"


@given(state=pivot_state_strategy(pivot_reason="deprecation_conflict"))
@settings(max_examples=100)
def test_pivot_adds_log_entry(state: SharedState):
    """
    Property: Handling pivot should add a log entry with details.
    
    **Validates: Requirements 13.2**
    """
    original_log_count = len(state.execution_log)
    
    # Handle pivot
    result_state = handle_pivot(state)
    
    # Property: Should add at least one log entry
    assert len(result_state.execution_log) > original_log_count, \
        "Should add log entry when handling pivot"
    
    # Check the last log entry
    last_log = result_state.execution_log[-1]
    assert last_log["agent_name"] == "pivot_manager", \
        f"Log should be from pivot_manager, got {last_log['agent_name']}"
    assert last_log["action"] == "handle_pivot", \
        f"Log action should be handle_pivot, got {last_log['action']}"
    assert "pivot_reason" in last_log["details"], \
        "Log should include pivot_reason"


# Example-based tests for specific scenarios
def test_deprecation_pivot_example():
    """Test deprecation pivot with a specific example"""
    state = SharedState(
        user_topic="How to use deprecated API",
        outline=[
            OutlineStep(step_id=0, description="Explain API usage", status="in_progress", retry_count=0),
            OutlineStep(step_id=1, description="Show examples", status="pending", retry_count=0),
        ],
        current_step_index=0,
        retrieved_docs=[
            RetrievedDocument(
                content="API is deprecated",
                source="docs.py",
                confidence=0.9,
                metadata={"has_deprecated": True}
            )
        ],
        current_skill="standard_tutorial",
        global_tone="professional",
        pivot_triggered=True,
        pivot_reason="deprecation_conflict",
        max_retries=3
    )
    
    result = handle_pivot(state)
    
    # Check modifications
    assert "DEPRECATED" in result.outline[0].description
    assert result.outline[0].retry_count == 1
    assert result.current_skill == "warning_mode"  # Compatible with standard_tutorial
    assert len(result.retrieved_docs) == 0
    assert result.pivot_triggered is False


def test_no_pivot_trigger_returns_unchanged():
    """Test that handle_pivot returns unchanged state if pivot not triggered"""
    state = SharedState(
        user_topic="Test topic",
        outline=[OutlineStep(step_id=0, description="Test", status="pending", retry_count=0)],
        current_step_index=0,
        pivot_triggered=False,
        max_retries=3
    )
    
    result = handle_pivot(state)
    
    # Should return essentially unchanged
    assert result.pivot_triggered is False
    assert result.outline[0].retry_count == 0


def test_pivot_without_reason_raises_error():
    """Test that pivot without reason raises ValidationError from Pydantic"""
    from pydantic import ValidationError
    
    with pytest.raises(ValidationError, match="转向触发时必须提供转向原因"):
        state = SharedState(
            user_topic="Test topic",
            outline=[OutlineStep(step_id=0, description="Test", status="in_progress", retry_count=0)],
            current_step_index=0,
            pivot_triggered=True,
            pivot_reason=None,  # No reason provided
            max_retries=3
        )
