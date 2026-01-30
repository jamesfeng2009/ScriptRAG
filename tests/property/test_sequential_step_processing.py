"""Property-Based Tests for Sequential Step Processing

Feature: rag-screenplay-multi-agent
Property 17: Sequential Step Processing

For any outline with N steps, the system should process steps sequentially
from step 0 to step N-1, incrementing current_step_index after each step
completion, ensuring no steps are skipped or processed out of order.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from src.domain.models import SharedState, OutlineStep, ScreenplayFragment


# Strategy for generating outline steps
@st.composite
def outline_steps_strategy(draw):
    """Generate a list of outline steps"""
    num_steps = draw(st.integers(min_value=2, max_value=10))
    steps = []
    for i in range(num_steps):
        step = OutlineStep(
            step_id=i,
            description=draw(st.text(min_size=10, max_size=100)),
            status="pending",
            retry_count=0
        )
        steps.append(step)
    return steps


@st.composite
def shared_state_with_outline_strategy(draw):
    """Generate a SharedState with an outline"""
    outline = draw(outline_steps_strategy())
    
    state = SharedState(
        user_topic=draw(st.text(min_size=5, max_size=50)),
        project_context=draw(st.text(min_size=5, max_size=100)),
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
    
    return state


@given(state=shared_state_with_outline_strategy())
@settings(max_examples=100, deadline=None)
def test_sequential_step_processing_property(state):
    """
    Property 17: Sequential Step Processing
    
    For any outline with N steps, the system should process steps sequentially
    from step 0 to step N-1, incrementing current_step_index after each step
    completion.
    
    验证需求: 12.2
    """
    # Assume we have at least 2 steps
    assume(len(state.outline) >= 2)
    
    # Record initial state
    initial_step_index = state.current_step_index
    num_steps = len(state.outline)
    
    # Simulate processing each step
    processed_steps = []
    
    for expected_step_id in range(num_steps):
        # Verify current step index matches expected
        assert state.current_step_index == expected_step_id, (
            f"Expected current_step_index to be {expected_step_id}, "
            f"but got {state.current_step_index}"
        )
        
        # Get current step
        current_step = state.get_current_step()
        assert current_step is not None, (
            f"Expected to get current step at index {expected_step_id}, "
            f"but got None"
        )
        
        # Verify step_id matches index
        assert current_step.step_id == expected_step_id, (
            f"Expected step_id to be {expected_step_id}, "
            f"but got {current_step.step_id}"
        )
        
        # Record processed step
        processed_steps.append(current_step.step_id)
        
        # Simulate step completion by creating a fragment
        fragment = ScreenplayFragment(
            step_id=current_step.step_id,
            content=f"Fragment for step {current_step.step_id}",
            skill_used=state.current_skill,
            sources=[]
        )
        state.fragments.append(fragment)
        current_step.status = "completed"
        
        # Move to next step (simulating completion check routing)
        state.current_step_index += 1
    
    # Verify all steps were processed in order
    assert processed_steps == list(range(num_steps)), (
        f"Steps were not processed sequentially. "
        f"Expected {list(range(num_steps))}, got {processed_steps}"
    )
    
    # Verify final step index is beyond the last step
    assert state.current_step_index == num_steps, (
        f"Expected final current_step_index to be {num_steps}, "
        f"but got {state.current_step_index}"
    )
    
    # Verify all steps are marked as completed
    for step in state.outline:
        assert step.status == "completed", (
            f"Expected step {step.step_id} to be completed, "
            f"but status is {step.status}"
        )
    
    # Verify we have a fragment for each step
    assert len(state.fragments) == num_steps, (
        f"Expected {num_steps} fragments, but got {len(state.fragments)}"
    )
    
    # Verify fragments are in order
    fragment_step_ids = [f.step_id for f in state.fragments]
    assert fragment_step_ids == list(range(num_steps)), (
        f"Fragments are not in sequential order. "
        f"Expected {list(range(num_steps))}, got {fragment_step_ids}"
    )


@given(state=shared_state_with_outline_strategy())
@settings(max_examples=100, deadline=None)
def test_no_step_skipping_property(state):
    """
    Property 17 (variant): No Step Skipping
    
    For any outline, no steps should be skipped during sequential processing
    unless explicitly marked as skipped by retry protection.
    
    验证需求: 12.2
    """
    # Assume we have at least 3 steps
    assume(len(state.outline) >= 3)
    
    num_steps = len(state.outline)
    
    # Process first half of steps normally
    mid_point = num_steps // 2
    
    for i in range(mid_point):
        current_step = state.get_current_step()
        assert current_step is not None
        
        # Create fragment
        fragment = ScreenplayFragment(
            step_id=current_step.step_id,
            content=f"Fragment for step {current_step.step_id}",
            skill_used=state.current_skill,
            sources=[]
        )
        state.fragments.append(fragment)
        current_step.status = "completed"
        
        # Move to next step
        state.current_step_index += 1
    
    # Verify we're at the mid point
    assert state.current_step_index == mid_point
    
    # Verify no steps were skipped in the first half
    for i in range(mid_point):
        step = state.outline[i]
        assert step.status == "completed", (
            f"Step {i} should be completed but is {step.status}"
        )
    
    # Verify remaining steps are still pending
    for i in range(mid_point, num_steps):
        step = state.outline[i]
        assert step.status == "pending", (
            f"Step {i} should be pending but is {step.status}"
        )


@given(state=shared_state_with_outline_strategy())
@settings(max_examples=100, deadline=None)
def test_step_index_never_decreases_property(state):
    """
    Property 17 (variant): Step Index Never Decreases
    
    For any outline, current_step_index should never decrease during
    normal forward processing (excluding pivot loops which reset to
    re-process a step).
    
    验证需求: 12.2
    """
    # Assume we have at least 2 steps
    assume(len(state.outline) >= 2)
    
    num_steps = len(state.outline)
    previous_index = state.current_step_index
    
    # Process all steps
    for _ in range(num_steps):
        current_step = state.get_current_step()
        if current_step is None:
            break
        
        # Record current index
        current_index = state.current_step_index
        
        # Verify index didn't decrease
        assert current_index >= previous_index, (
            f"Step index decreased from {previous_index} to {current_index}"
        )
        
        # Create fragment
        fragment = ScreenplayFragment(
            step_id=current_step.step_id,
            content=f"Fragment for step {current_step.step_id}",
            skill_used=state.current_skill,
            sources=[]
        )
        state.fragments.append(fragment)
        current_step.status = "completed"
        
        # Move to next step
        state.current_step_index += 1
        previous_index = current_index


@given(state=shared_state_with_outline_strategy())
@settings(max_examples=100, deadline=None)
def test_fragment_order_matches_step_order_property(state):
    """
    Property 17 (variant): Fragment Order Matches Step Order
    
    For any outline processed sequentially, the order of fragments
    should match the order of outline steps.
    
    验证需求: 12.2
    """
    # Assume we have at least 2 steps
    assume(len(state.outline) >= 2)
    
    num_steps = len(state.outline)
    
    # Process all steps
    for i in range(num_steps):
        current_step = state.get_current_step()
        assert current_step is not None
        
        # Create fragment
        fragment = ScreenplayFragment(
            step_id=current_step.step_id,
            content=f"Fragment for step {current_step.step_id}",
            skill_used=state.current_skill,
            sources=[]
        )
        state.fragments.append(fragment)
        current_step.status = "completed"
        
        # Move to next step
        state.current_step_index += 1
    
    # Verify fragment order
    for i, fragment in enumerate(state.fragments):
        assert fragment.step_id == i, (
            f"Fragment at position {i} has step_id {fragment.step_id}, "
            f"expected {i}"
        )
    
    # Verify we have exactly one fragment per step
    assert len(state.fragments) == num_steps, (
        f"Expected {num_steps} fragments, got {len(state.fragments)}"
    )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
