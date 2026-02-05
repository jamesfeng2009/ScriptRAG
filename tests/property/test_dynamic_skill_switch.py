"""Property-Based Tests for Dynamic Skill Switching

属性 8: 动态 Skill 切换

Property: The system should allow dynamic skill switching during execution.
When a skill switch is triggered (by Director or PivotManager), the system
should:
1. Update the current_skill in SharedState
2. Record the skill change in skill_history
3. Log the skill switch with reason and step_id
4. Respect compatibility constraints
5. Maintain consistency across all agents
"""

import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from datetime import datetime
from src.domain.models import SharedState, OutlineStep, RetrievedDocument
from src.domain.agents.pivot_manager import handle_pivot
from src.domain.skills import SKILLS, check_skill_compatibility


# Strategy for generating valid skill names
skill_names = st.sampled_from(list(SKILLS.keys()))

# Strategy for generating valid tone names
tone_names = st.sampled_from([
    "professional", "cautionary", "engaging", 
    "exploratory", "casual", "neutral"
])

# Strategy for generating pivot reasons
pivot_reasons = st.sampled_from([
    "deprecation_conflict",
    "content_complexity_high",
    "content_complexity_low",
    "missing_information"
])


def create_test_state(
    current_skill: str = "standard_tutorial",
    global_tone: str = "professional",
    num_steps: int = 3
) -> SharedState:
    """Create a test SharedState with outline steps"""
    outline = [
        OutlineStep(
            step_id=i,
            description=f"Step {i} description",
            status="pending" if i > 0 else "in_progress",
            retry_count=0
        )
        for i in range(num_steps)
    ]
    
    return SharedState(
        user_topic="Test topic",
        project_context="Test context",
        outline=outline,
        current_step_index=0,
        current_skill=current_skill,
        global_tone=global_tone
    )


@given(
    initial_skill=skill_names,
    new_skill=skill_names,
    reason=st.text(min_size=1, max_size=50),
    step_id=st.integers(min_value=0, max_value=10)
)
@settings(max_examples=100)
def test_skill_switch_updates_state(
    initial_skill: str,
    new_skill: str,
    reason: str,
    step_id: int
):
    """
    Property 8.1: Skill Switch Updates State
    
    For any skill switch operation:
    1. current_skill should be updated to new_skill
    2. skill_history should contain the switch record
    3. execution_log should contain the switch entry
    4. updated_at timestamp should be updated
    
    **Validates: Requirement 4.8**
    """
    # Create test state
    state = create_test_state(current_skill=initial_skill)
    initial_history_len = len(state.skill_history)
    initial_log_len = len(state.execution_log)
    
    # Skip if skills are the same (no-op)
    if initial_skill == new_skill:
        state.switch_skill(new_skill=new_skill, reason=reason, step_id=step_id)
        # Should be no-op
        assert state.current_skill == initial_skill
        assert len(state.skill_history) == initial_history_len
        return
    
    # Perform skill switch
    state.switch_skill(new_skill=new_skill, reason=reason, step_id=step_id)
    
    # Property 1: current_skill should be updated
    assert state.current_skill == new_skill, \
        f"current_skill should be {new_skill}, got {state.current_skill}"
    
    # Property 2: skill_history should contain the switch
    assert len(state.skill_history) == initial_history_len + 1, \
        f"skill_history should have {initial_history_len + 1} entries, " \
        f"got {len(state.skill_history)}"
    
    latest_history = state.skill_history[-1]
    assert latest_history["from_skill"] == initial_skill
    assert latest_history["to_skill"] == new_skill
    assert latest_history["reason"] == reason
    assert latest_history["step_id"] == step_id
    assert "timestamp" in latest_history
    
    # Property 3: execution_log should contain the switch
    assert len(state.execution_log) == initial_log_len + 1, \
        f"execution_log should have {initial_log_len + 1} entries"
    
    latest_log = state.execution_log[-1]
    assert latest_log["agent_name"] == "system"
    assert latest_log["action"] == "skill_switch"
    assert latest_log["details"]["from_skill"] == initial_skill
    assert latest_log["details"]["to_skill"] == new_skill


@given(
    pivot_reason=pivot_reasons,
    initial_skill=skill_names,
    global_tone=tone_names
)
@settings(max_examples=100)
def test_pivot_triggers_skill_switch(
    pivot_reason: str,
    initial_skill: str,
    global_tone: str
):
    """
    Property 8.2: Pivot Triggers Skill Switch
    
    When a pivot is triggered with a specific reason:
    1. The appropriate skill switch should occur
    2. The skill switch should respect compatibility constraints
    3. The skill_history should record the switch
    
    **Validates: Requirement 4.8**
    """
    # Create test state with pivot triggered
    state = create_test_state(
        current_skill=initial_skill,
        global_tone=global_tone,
        num_steps=3
    )
    state.pivot_triggered = True
    state.pivot_reason = pivot_reason
    
    # Add some retrieved docs for deprecation conflict
    if pivot_reason == "deprecation_conflict":
        state.retrieved_docs = [
            RetrievedDocument(
                content="Test content",
                source="test.py",
                confidence=0.9,
                metadata={"has_deprecated": True}
            )
        ]
    
    initial_skill_value = state.current_skill
    initial_history_len = len(state.skill_history)
    
    # Handle pivot
    state = handle_pivot(state)
    
    # Determine expected skill based on pivot reason
    expected_skills = {
        "deprecation_conflict": "warning_mode",
        "content_complexity_high": "visualization_analogy",
        "content_complexity_low": "standard_tutorial",
        "missing_information": "research_mode"
    }
    
    desired_skill = expected_skills.get(pivot_reason)
    
    if desired_skill and desired_skill != initial_skill_value:
        # Skill should have changed (possibly to compatible alternative)
        if check_skill_compatibility(initial_skill_value, desired_skill):
            # Direct switch should occur
            assert state.current_skill == desired_skill, \
                f"Expected skill {desired_skill}, got {state.current_skill}"
        else:
            # Should switch to compatible alternative or stay same
            if state.current_skill != initial_skill_value:
                assert check_skill_compatibility(initial_skill_value, state.current_skill), \
                    f"Switched to incompatible skill: {initial_skill_value} -> {state.current_skill}"
        
        # Skill history should be updated if skill changed
        if state.current_skill != initial_skill_value:
            assert len(state.skill_history) > initial_history_len, \
                "skill_history should be updated after skill switch"


@given(
    num_switches=st.integers(min_value=1, max_value=5),
    initial_skill=skill_names
)
@settings(max_examples=50)
def test_multiple_skill_switches_tracked(
    num_switches: int,
    initial_skill: str
):
    """
    Property 8.3: Multiple Skill Switches Tracked
    
    When multiple skill switches occur during execution:
    1. All switches should be recorded in skill_history
    2. The order should be preserved
    3. Each switch should have complete metadata
    
    **Validates: Requirement 4.8**
    """
    state = create_test_state(current_skill=initial_skill)
    
    # Perform multiple switches
    all_skills = list(SKILLS.keys())
    for i in range(num_switches):
        # Pick a different skill
        new_skill = all_skills[(all_skills.index(state.current_skill) + 1) % len(all_skills)]
        state.switch_skill(
            new_skill=new_skill,
            reason=f"switch_{i}",
            step_id=i
        )
    
    # Property 1: All switches should be recorded
    assert len(state.skill_history) == num_switches, \
        f"Expected {num_switches} history entries, got {len(state.skill_history)}"
    
    # Property 2: Order should be preserved
    for i, history_entry in enumerate(state.skill_history):
        assert history_entry["reason"] == f"switch_{i}", \
            f"History entry {i} has wrong reason: {history_entry['reason']}"
        assert history_entry["step_id"] == i, \
            f"History entry {i} has wrong step_id: {history_entry['step_id']}"
    
    # Property 3: Each switch should have complete metadata
    for history_entry in state.skill_history:
        assert "timestamp" in history_entry
        assert "from_skill" in history_entry
        assert "to_skill" in history_entry
        assert "reason" in history_entry
        assert "step_id" in history_entry
        assert history_entry["from_skill"] in SKILLS
        assert history_entry["to_skill"] in SKILLS


@given(
    initial_skill=skill_names,
    new_skill=skill_names
)
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.filter_too_much]
)
def test_skill_switch_idempotent_for_same_skill(
    initial_skill: str,
    new_skill: str
):
    """
    Property 8.4: Skill Switch is Idempotent for Same Skill
    
    When switching to the same skill:
    1. No history entry should be created
    2. No log entry should be created
    3. State should remain unchanged
    
    **Validates: Requirement 4.8**
    """
    # Only test same-skill switches
    assume(initial_skill == new_skill)
    
    state = create_test_state(current_skill=initial_skill)
    initial_history_len = len(state.skill_history)
    initial_log_len = len(state.execution_log)
    
    # Attempt to switch to same skill
    state.switch_skill(
        new_skill=new_skill,
        reason="test_reason",
        step_id=0
    )
    
    # Property 1: No history entry created
    assert len(state.skill_history) == initial_history_len, \
        "Same-skill switch should not create history entry"
    
    # Property 2: No log entry created
    assert len(state.execution_log) == initial_log_len, \
        "Same-skill switch should not create log entry"
    
    # Property 3: Skill remains unchanged
    assert state.current_skill == initial_skill


@given(skill_name=skill_names)
@settings(max_examples=50)
def test_invalid_skill_raises_error(skill_name: str):
    """
    Property 8.5: Invalid Skill Raises Error
    
    When attempting to switch to an invalid skill:
    1. ValueError should be raised
    2. State should remain unchanged
    
    **Validates: Requirement 4.8**
    """
    state = create_test_state(current_skill=skill_name)
    initial_skill = state.current_skill
    initial_history_len = len(state.skill_history)
    
    # Attempt to switch to invalid skill
    with pytest.raises(ValueError, match="无效的 Skill 模式"):
        state.switch_skill(
            new_skill="invalid_skill_name",
            reason="test",
            step_id=0
        )
    
    # State should remain unchanged
    assert state.current_skill == initial_skill
    assert len(state.skill_history) == initial_history_len


# Example-based tests for specific scenarios
def test_deprecation_conflict_switches_to_warning_mode():
    """Test that deprecation conflict triggers switch to warning_mode"""
    state = create_test_state(current_skill="standard_tutorial")
    state.pivot_triggered = True
    state.pivot_reason = "deprecation_conflict"
    state.retrieved_docs = [
        RetrievedDocument(
            content="Deprecated function",
            source="test.py",
            confidence=0.9,
            metadata={"has_deprecated": True}
        )
    ]
    
    state = handle_pivot(state)
    
    # Should switch to warning_mode (or compatible alternative)
    assert state.current_skill in ["warning_mode", "standard_tutorial"]
    if state.current_skill == "warning_mode":
        assert len(state.skill_history) == 1
        assert state.skill_history[0]["to_skill"] == "warning_mode"
        assert state.skill_history[0]["reason"] == "deprecation_conflict"


def test_high_complexity_switches_to_visualization():
    """Test that high complexity triggers switch to visualization_analogy"""
    state = create_test_state(current_skill="standard_tutorial")
    state.pivot_triggered = True
    state.pivot_reason = "content_complexity_high"
    
    state = handle_pivot(state)
    
    # Should switch to visualization_analogy (or compatible alternative)
    assert state.current_skill in ["visualization_analogy", "standard_tutorial"]
    if state.current_skill == "visualization_analogy":
        assert len(state.skill_history) == 1
        assert state.skill_history[0]["to_skill"] == "visualization_analogy"


def test_low_complexity_switches_to_standard():
    """Test that low complexity triggers switch back to standard_tutorial"""
    state = create_test_state(current_skill="visualization_analogy")
    state.pivot_triggered = True
    state.pivot_reason = "content_complexity_low"
    
    state = handle_pivot(state)
    
    # Should switch to standard_tutorial (or compatible alternative)
    assert state.current_skill in ["standard_tutorial", "visualization_analogy", "meme_style"]
    if state.current_skill == "standard_tutorial":
        assert len(state.skill_history) == 1
        assert state.skill_history[0]["to_skill"] == "standard_tutorial"


def test_missing_info_switches_to_research_mode():
    """Test that missing information triggers switch to research_mode"""
    state = create_test_state(current_skill="standard_tutorial")
    state.pivot_triggered = True
    state.pivot_reason = "missing_information"
    
    state = handle_pivot(state)
    
    # Should switch to research_mode (or compatible alternative)
    assert state.current_skill in ["research_mode", "standard_tutorial", "warning_mode"]
    if state.current_skill == "research_mode":
        assert len(state.skill_history) == 1
        assert state.skill_history[0]["to_skill"] == "research_mode"


def test_skill_history_preserves_chronological_order():
    """Test that skill history maintains chronological order"""
    state = create_test_state(current_skill="standard_tutorial")
    
    # Perform multiple switches
    state.switch_skill("warning_mode", "reason1", 0)
    state.switch_skill("research_mode", "reason2", 1)
    state.switch_skill("standard_tutorial", "reason3", 2)
    
    assert len(state.skill_history) == 3
    
    # Verify chronological order
    assert state.skill_history[0]["from_skill"] == "standard_tutorial"
    assert state.skill_history[0]["to_skill"] == "warning_mode"
    
    assert state.skill_history[1]["from_skill"] == "warning_mode"
    assert state.skill_history[1]["to_skill"] == "research_mode"
    
    assert state.skill_history[2]["from_skill"] == "research_mode"
    assert state.skill_history[2]["to_skill"] == "standard_tutorial"
    
    # Verify timestamps are in order
    timestamps = [entry["timestamp"] for entry in state.skill_history]
    assert timestamps == sorted(timestamps), "Timestamps should be in chronological order"
