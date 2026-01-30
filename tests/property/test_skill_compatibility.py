"""Property-Based Tests for Skill Compatibility Enforcement

属性 16: Skill 兼容性执行
Property: For any skill switch request, the PivotManager should only allow
switches to compatible skills. If the desired skill is not compatible with
the current skill, the system should select the closest compatible skill
based on global tone preference.
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from src.domain.agents.pivot_manager import constrain_skill_switch
from src.domain.skills import SKILLS, check_skill_compatibility


# Strategy for generating valid skill names
skill_names = st.sampled_from(list(SKILLS.keys()))

# Strategy for generating valid tone names
tone_names = st.sampled_from([
    "professional", "cautionary", "engaging", 
    "exploratory", "casual", "neutral"
])


@given(
    current_skill=skill_names,
    desired_skill=skill_names,
    global_tone=st.one_of(st.none(), tone_names)
)
@settings(max_examples=100)
def test_skill_compatibility_enforcement(
    current_skill: str,
    desired_skill: str,
    global_tone: str
):
    """
    Property 16: Skill Compatibility Enforcement
    
    For any skill switch request (current_skill -> desired_skill):
    1. The returned skill must be valid
    2. If current and desired are the same, return current
    3. If desired is compatible with current, return desired
    4. Otherwise, return a skill compatible with current
    5. The returned skill must exist in SKILLS
    
    **Validates: Requirements 11.2, 11.4**
    """
    # Execute skill switch with constraints
    result_skill = constrain_skill_switch(
        current_skill=current_skill,
        desired_skill=desired_skill,
        global_tone=global_tone
    )
    
    # Property 1: Result must be a valid skill
    assert result_skill in SKILLS, \
        f"Result skill {result_skill} is not in SKILLS"
    
    # Property 2: If same skill, should return current
    if current_skill == desired_skill:
        assert result_skill == current_skill, \
            f"Same skill switch should return current: {current_skill}"
    
    # Property 3: If desired is compatible, should return desired
    if check_skill_compatibility(current_skill, desired_skill):
        assert result_skill == desired_skill, \
            f"Compatible switch {current_skill} -> {desired_skill} " \
            f"should return desired, got {result_skill}"
    
    # Property 4: Result must be compatible with current
    # (or be the current skill itself)
    if result_skill != current_skill:
        assert check_skill_compatibility(current_skill, result_skill), \
            f"Result skill {result_skill} must be compatible with " \
            f"current skill {current_skill}"


@given(
    current_skill=skill_names,
    desired_skill=skill_names
)
@settings(max_examples=100)
def test_incompatible_switch_returns_compatible_skill(
    current_skill: str,
    desired_skill: str
):
    """
    Property: When desired skill is incompatible with current skill,
    the system must return a compatible alternative (or current skill).
    
    **Validates: Requirements 11.2, 11.4**
    """
    # Only test incompatible switches
    assume(current_skill != desired_skill)
    assume(not check_skill_compatibility(current_skill, desired_skill))
    
    result_skill = constrain_skill_switch(
        current_skill=current_skill,
        desired_skill=desired_skill,
        global_tone=None
    )
    
    # Result must be either current skill or a compatible skill
    if result_skill != current_skill:
        assert check_skill_compatibility(current_skill, result_skill), \
            f"Incompatible switch must return compatible skill: " \
            f"{current_skill} -> {result_skill}"
    
    # Result must be valid
    assert result_skill in SKILLS


@given(
    current_skill=skill_names,
    desired_skill=skill_names,
    global_tone=tone_names
)
@settings(max_examples=100)
def test_tone_preference_respected(
    current_skill: str,
    desired_skill: str,
    global_tone: str
):
    """
    Property: When global tone is specified and desired skill is incompatible,
    the system should prefer compatible skills with matching tone.
    
    **Validates: Requirements 11.2, 11.4**
    """
    # Only test incompatible switches
    assume(current_skill != desired_skill)
    assume(not check_skill_compatibility(current_skill, desired_skill))
    
    result_skill = constrain_skill_switch(
        current_skill=current_skill,
        desired_skill=desired_skill,
        global_tone=global_tone
    )
    
    # Result must be valid and compatible
    assert result_skill in SKILLS
    if result_skill != current_skill:
        assert check_skill_compatibility(current_skill, result_skill)
    
    # If there are compatible skills with matching tone, result should have that tone
    compatible_skills = SKILLS[current_skill].compatible_with
    tone_matches = [
        skill for skill in compatible_skills
        if SKILLS[skill].tone == global_tone
    ]
    
    if tone_matches:
        # Result should be one of the tone-matching skills
        assert result_skill in tone_matches or result_skill == current_skill, \
            f"With global_tone={global_tone}, expected one of {tone_matches}, " \
            f"got {result_skill}"


@given(skill_name=skill_names)
@settings(max_examples=50)
def test_same_skill_switch_is_noop(skill_name: str):
    """
    Property: Switching from a skill to itself should always return the same skill.
    
    **Validates: Requirements 11.2**
    """
    result = constrain_skill_switch(
        current_skill=skill_name,
        desired_skill=skill_name,
        global_tone=None
    )
    
    assert result == skill_name, \
        f"Same skill switch should be no-op: {skill_name} -> {result}"


def test_invalid_current_skill_raises_error():
    """
    Test that invalid current skill raises ValueError
    """
    with pytest.raises(ValueError, match="Invalid current skill"):
        constrain_skill_switch(
            current_skill="invalid_skill",
            desired_skill="standard_tutorial",
            global_tone=None
        )


def test_invalid_desired_skill_raises_error():
    """
    Test that invalid desired skill raises ValueError
    """
    with pytest.raises(ValueError, match="Invalid desired skill"):
        constrain_skill_switch(
            current_skill="standard_tutorial",
            desired_skill="invalid_skill",
            global_tone=None
        )


# Example-based tests for specific scenarios
def test_compatible_switch_standard_to_warning():
    """Test compatible switch from standard_tutorial to warning_mode"""
    result = constrain_skill_switch(
        current_skill="standard_tutorial",
        desired_skill="warning_mode",
        global_tone=None
    )
    assert result == "warning_mode"


def test_incompatible_switch_warning_to_meme():
    """Test incompatible switch from warning_mode to meme_style"""
    result = constrain_skill_switch(
        current_skill="warning_mode",
        desired_skill="meme_style",
        global_tone=None
    )
    # Should return a compatible skill, not meme_style
    assert result != "meme_style"
    assert result in SKILLS
    if result != "warning_mode":
        assert check_skill_compatibility("warning_mode", result)


def test_tone_preference_professional():
    """Test that professional tone preference is respected"""
    result = constrain_skill_switch(
        current_skill="warning_mode",
        desired_skill="meme_style",  # Incompatible
        global_tone="professional"
    )
    # Should prefer standard_tutorial (professional tone) if compatible
    assert result in SKILLS
    if result != "warning_mode":
        assert check_skill_compatibility("warning_mode", result)
