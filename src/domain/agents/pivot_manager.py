"""Pivot Manager Agent - Corrects outline when conflicts occur

This module implements the PivotManager agent, which is responsible for:
1. Handling pivot triggers from the Director
2. Modifying outline steps based on pivot reasons
3. Applying skill switches with compatibility constraints
4. Updating retry counters and step status
"""

from typing import Optional, List
import logging
from ..models import SharedState, OutlineStep
from ..skills import SKILLS, check_skill_compatibility, find_closest_compatible_skill
from ...infrastructure.logging import get_agent_logger

logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


def constrain_skill_switch(
    current_skill: str,
    desired_skill: str,
    global_tone: Optional[str] = None
) -> str:
    """Constrain skill switch based on compatibility rules
    
    This function ensures that skill switches respect compatibility constraints.
    If the desired skill is not directly compatible with the current skill,
    it finds the closest compatible skill based on global tone preference.
    
    Args:
        current_skill: Current active skill name
        desired_skill: Desired target skill name
        global_tone: Optional global tone preference for selection
        
    Returns:
        Name of the skill to switch to (may be different from desired_skill)
        
    Raises:
        ValueError: If skill names are invalid
        
    验证需求: 11.2, 11.4
    """
    if current_skill not in SKILLS:
        raise ValueError(f"Invalid current skill: {current_skill}")
    if desired_skill not in SKILLS:
        raise ValueError(f"Invalid desired skill: {desired_skill}")
    
    # If skills are the same, no switch needed
    if current_skill == desired_skill:
        logger.info(f"Skill switch not needed: already using {current_skill}")
        return current_skill
    
    # Check if direct switch is compatible
    if check_skill_compatibility(current_skill, desired_skill):
        logger.info(
            f"Skill switch allowed: {current_skill} -> {desired_skill} "
            f"(directly compatible)"
        )
        return desired_skill
    
    # Find closest compatible skill
    compatible_skill = find_closest_compatible_skill(
        current_skill=current_skill,
        desired_skill=desired_skill,
        global_tone=global_tone
    )
    
    if compatible_skill != desired_skill:
        logger.warning(
            f"Skill switch constrained: {current_skill} -> {desired_skill} "
            f"not compatible. Using {compatible_skill} instead "
            f"(global_tone={global_tone})"
        )
    
    return compatible_skill


def handle_pivot(state: SharedState) -> SharedState:
    """Handle pivot trigger and modify outline accordingly
    
    This function is called when the Director triggers a pivot. It:
    1. Analyzes the pivot reason
    2. Modifies the current and subsequent outline steps
    3. Applies skill switches with compatibility constraints
    4. Updates retry counters and step status
    5. Clears retrieved documents to trigger re-retrieval
    
    Args:
        state: Current shared state with pivot_triggered=True
        
    Returns:
        Updated shared state with modified outline
        
    Raises:
        ValueError: If pivot is triggered without a reason
    """
    if not state.pivot_triggered:
        logger.warning("handle_pivot called but pivot_triggered is False")
        return state
    
    if not state.pivot_reason:
        raise ValueError("Pivot triggered without a reason")
    
    logger.info(
        f"Handling pivot at step {state.current_step_index}: "
        f"reason={state.pivot_reason}"
    )
    
    # Get current step
    current_step = state.get_current_step()
    if not current_step:
        logger.error("No current step found for pivot handling")
        state.pivot_triggered = False
        state.pivot_reason = None
        return state
    
    # Log agent transition
    agent_logger.log_agent_transition(
        from_agent="director",
        to_agent="pivot_manager",
        step_id=current_step.step_id,
        reason=state.pivot_reason
    )
    
    # Increment retry counter for current step
    current_step.retry_count += 1
    logger.info(
        f"Incremented retry count for step {current_step.step_id}: "
        f"{current_step.retry_count}"
    )
    
    # Log retry attempt
    agent_logger.log_retry_attempt(
        step_id=current_step.step_id,
        retry_count=current_step.retry_count,
        max_retries=state.max_retries,
        reason=state.pivot_reason
    )
    
    # Handle different pivot reasons
    if state.pivot_reason == "deprecation_conflict":
        state = _handle_deprecation_pivot(state, current_step)
    elif "complexity" in state.pivot_reason:
        # Handle both content_complexity_high and content_complexity_low
        state = _handle_complexity_pivot(state, current_step)
    elif state.pivot_reason == "missing_information":
        state = _handle_missing_info_pivot(state, current_step)
    else:
        logger.warning(f"Unknown pivot reason: {state.pivot_reason}")
        # Generic handling: just update step status
        current_step.status = "in_progress"
    
    # Clear retrieved documents to trigger re-retrieval
    state.retrieved_docs = []
    logger.info("Cleared retrieved documents to trigger re-retrieval")
    
    # Add log entry
    state.add_log_entry(
        agent_name="pivot_manager",
        action="handle_pivot",
        details={
            "pivot_reason": state.pivot_reason,
            "step_id": current_step.step_id,
            "retry_count": current_step.retry_count,
            "new_skill": state.current_skill
        }
    )
    
    # Reset pivot trigger (but keep reason for logging)
    state.pivot_triggered = False
    
    return state


def _handle_deprecation_pivot(
    state: SharedState,
    current_step: OutlineStep
) -> SharedState:
    """Handle pivot triggered by deprecation conflict
    
    When a deprecation conflict is detected:
    1. Modify current step to focus on deprecation warning
    2. Switch to warning_mode skill (with compatibility constraints)
    3. Update subsequent steps that depend on deprecated feature
    
    Args:
        state: Current shared state
        current_step: Current outline step
        
    Returns:
        Updated shared state
    """
    logger.info(f"Handling deprecation conflict for step {current_step.step_id}")
    
    # Modify current step description to focus on deprecation
    original_desc = current_step.description
    current_step.description = (
        f"[DEPRECATED WARNING] {original_desc} "
        f"(Note: This feature is deprecated. Explain alternatives.)"
    )
    current_step.status = "in_progress"
    
    logger.info(
        f"Modified step {current_step.step_id} description: "
        f"{current_step.description}"
    )
    
    # Switch to warning_mode skill with compatibility constraints
    desired_skill = "warning_mode"
    new_skill = constrain_skill_switch(
        current_skill=state.current_skill,
        desired_skill=desired_skill,
        global_tone=state.global_tone
    )
    
    if new_skill != state.current_skill:
        logger.info(f"Switching skill: {state.current_skill} -> {new_skill}")
        state.switch_skill(
            new_skill=new_skill,
            reason="deprecation_conflict",
            step_id=current_step.step_id
        )
    
    # Update subsequent steps that might depend on deprecated feature
    # (This is a simplified heuristic - in production, would use more sophisticated analysis)
    for i in range(state.current_step_index + 1, len(state.outline)):
        step = state.outline[i]
        if step.status == "pending":
            # Add note about deprecation to subsequent steps
            if "deprecated" not in step.description.lower():
                step.description = (
                    f"{step.description} "
                    f"(Consider deprecation of previous feature)"
                )
                logger.info(
                    f"Updated subsequent step {step.step_id} to consider deprecation"
                )
    
    return state


def _handle_complexity_pivot(
    state: SharedState,
    current_step: OutlineStep
) -> SharedState:
    """Handle pivot triggered by complexity
    
    When content is too complex:
    1. Switch to visualization_analogy skill (with compatibility constraints)
    2. Modify current step to emphasize simplification
    
    Args:
        state: Current shared state
        current_step: Current outline step
        
    Returns:
        Updated shared state
    """
    logger.info(f"Handling complexity trigger for step {current_step.step_id}")
    
    # Determine if this is high or low complexity
    is_high_complexity = "high" in state.pivot_reason.lower()
    is_low_complexity = "low" in state.pivot_reason.lower()
    
    # If neither high nor low is specified, treat as generic complexity trigger
    if not is_high_complexity and not is_low_complexity:
        # Generic complexity trigger - modify description
        if "[COMPLEXITY]" not in current_step.description:
            current_step.description = (
                f"[COMPLEXITY] {current_step.description} "
                f"(Adjust presentation for complexity)"
            )
            current_step.status = "in_progress"
        return state
    
    if is_high_complexity:
        # Modify current step to emphasize simplification
        if "[SIMPLIFIED]" not in current_step.description:
            current_step.description = (
                f"[SIMPLIFIED] {current_step.description} "
                f"(Use analogies and visualizations)"
            )
            current_step.status = "in_progress"
        
        # Switch to visualization_analogy skill with compatibility constraints
        desired_skill = "visualization_analogy"
        new_skill = constrain_skill_switch(
            current_skill=state.current_skill,
            desired_skill=desired_skill,
            global_tone=state.global_tone
        )
        
        if new_skill != state.current_skill:
            logger.info(f"Switching skill: {state.current_skill} -> {new_skill}")
            state.switch_skill(
                new_skill=new_skill,
                reason="content_complexity_high",
                step_id=current_step.step_id
            )
    
    elif is_low_complexity:
        # Content is simple, switch back to standard tutorial
        # Modify current step to indicate simplification
        if "[STANDARD]" not in current_step.description:
            current_step.description = (
                f"[STANDARD] {current_step.description} "
                f"(Use standard tutorial format)"
            )
            current_step.status = "in_progress"
        
        desired_skill = "standard_tutorial"
        new_skill = constrain_skill_switch(
            current_skill=state.current_skill,
            desired_skill=desired_skill,
            global_tone=state.global_tone
        )
        
        if new_skill != state.current_skill:
            logger.info(f"Switching skill: {state.current_skill} -> {new_skill}")
            state.switch_skill(
                new_skill=new_skill,
                reason="content_complexity_low",
                step_id=current_step.step_id
            )
    
    return state


def _handle_missing_info_pivot(
    state: SharedState,
    current_step: OutlineStep
) -> SharedState:
    """Handle pivot triggered by missing information
    
    When information is insufficient:
    1. Switch to research_mode skill (with compatibility constraints)
    2. Modify current step to acknowledge information gap
    
    Args:
        state: Current shared state
        current_step: Current outline step
        
    Returns:
        Updated shared state
    """
    logger.info(f"Handling missing information for step {current_step.step_id}")
    
    # Modify current step to acknowledge information gap
    if "[RESEARCH NEEDED]" not in current_step.description:
        current_step.description = (
            f"[RESEARCH NEEDED] {current_step.description} "
            f"(Acknowledge information gaps)"
        )
        current_step.status = "in_progress"
    
    # Switch to research_mode skill with compatibility constraints
    desired_skill = "research_mode"
    new_skill = constrain_skill_switch(
        current_skill=state.current_skill,
        desired_skill=desired_skill,
        global_tone=state.global_tone
    )
    
    if new_skill != state.current_skill:
        logger.info(f"Switching skill: {state.current_skill} -> {new_skill}")
        state.switch_skill(
            new_skill=new_skill,
            reason="missing_information",
            step_id=current_step.step_id
        )
    
    return state
