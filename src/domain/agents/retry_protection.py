"""Retry Protection - Prevents infinite loops and enforces retry limits

This module implements retry limit checking and loop protection mechanisms:
1. Check if current step exceeds max_retries
2. Force degradation by skipping steps or marking as failed
3. Add placeholder fragments for skipped steps
4. Prevent infinite pivot loops
"""

import logging
from typing import Optional
from ..models import SharedState, OutlineStep, ScreenplayFragment
from ...infrastructure.logging import get_agent_logger

logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


def check_retry_limit(state: SharedState) -> SharedState:
    """Check retry limit and force degradation if exceeded
    
    This function implements loop protection by:
    1. Checking if current step's retry_count exceeds max_retries
    2. Forcing degradation by skipping the step
    3. Adding a placeholder fragment explaining the skip
    4. Advancing to the next step
    
    Args:
        state: Current shared state
        
    Returns:
        Updated shared state with step potentially skipped
    """
    # Get current step
    current_step = state.get_current_step()
    if not current_step:
        logger.warning("check_retry_limit: No current step found")
        return state
    
    # Check if retry limit exceeded
    if current_step.retry_count >= state.max_retries:
        logger.warning(
            f"Retry limit exceeded for step {current_step.step_id}: "
            f"retry_count={current_step.retry_count}, max_retries={state.max_retries}"
        )
        
        # Log degradation action
        agent_logger.log_degradation(
            step_id=current_step.step_id,
            degradation_type="retry_limit_exceeded",
            reason=f"Retry count {current_step.retry_count} >= max_retries {state.max_retries}",
            action_taken="skip_step_with_placeholder"
        )
        
        # Force degradation: skip the step
        current_step.status = "skipped"
        
        # Add placeholder fragment explaining the skip
        placeholder_fragment = ScreenplayFragment(
            step_id=current_step.step_id,
            content=(
                f"[SKIPPED] Step {current_step.step_id}: {current_step.description}\n\n"
                f"This step was skipped after {current_step.retry_count} retry attempts. "
                f"The system was unable to generate satisfactory content for this section. "
                f"Manual review and completion may be required."
            ),
            skill_used=state.current_skill,
            sources=[]
        )
        
        state.fragments.append(placeholder_fragment)
        
        logger.info(
            f"Added placeholder fragment for skipped step {current_step.step_id}"
        )
        
        # Add log entry
        state.add_log_entry(
            agent_name="retry_protection",
            action="force_skip",
            details={
                "step_id": current_step.step_id,
                "retry_count": current_step.retry_count,
                "max_retries": state.max_retries,
                "reason": "retry_limit_exceeded"
            }
        )
        
        # Clear pivot trigger to prevent further retries
        state.pivot_triggered = False
        state.pivot_reason = None
        
        # Advance to next step
        advanced = state.advance_step()
        if advanced:
            logger.info(f"Advanced to next step: {state.current_step_index}")
        else:
            logger.info("Reached end of outline after skipping step")
    
    else:
        logger.debug(
            f"Retry limit check passed for step {current_step.step_id}: "
            f"retry_count={current_step.retry_count}, max_retries={state.max_retries}"
        )
    
    return state


def is_in_infinite_loop(state: SharedState, window_size: int = 10) -> bool:
    """Detect if the system is stuck in an infinite loop
    
    This function analyzes recent execution logs to detect patterns
    that indicate an infinite loop (e.g., same step being retried repeatedly).
    
    Args:
        state: Current shared state
        window_size: Number of recent log entries to analyze
        
    Returns:
        True if infinite loop detected, False otherwise
    """
    if len(state.execution_log) < window_size:
        return False
    
    # Get recent log entries
    recent_logs = state.execution_log[-window_size:]
    
    # Count pivot triggers in recent logs
    pivot_count = sum(
        1 for log in recent_logs
        if log.get("action") == "handle_pivot"
    )
    
    # If more than 50% of recent actions are pivots, likely in a loop
    if pivot_count > window_size * 0.5:
        logger.warning(
            f"Potential infinite loop detected: {pivot_count} pivots "
            f"in last {window_size} actions"
        )
        return True
    
    # Check if same step is being retried excessively
    current_step = state.get_current_step()
    if current_step:
        same_step_actions = sum(
            1 for log in recent_logs
            if log.get("details", {}).get("step_id") == current_step.step_id
        )
        
        if same_step_actions > window_size * 0.7:
            logger.warning(
                f"Potential infinite loop detected: step {current_step.step_id} "
                f"appears in {same_step_actions} of last {window_size} actions"
            )
            return True
    
    return False


def reset_retry_counter(state: SharedState, step_id: int) -> SharedState:
    """Reset retry counter for a specific step
    
    This is useful when a step successfully completes or when
    manual intervention resolves an issue.
    
    Args:
        state: Current shared state
        step_id: ID of the step to reset
        
    Returns:
        Updated shared state
    """
    for step in state.outline:
        if step.step_id == step_id:
            old_count = step.retry_count
            step.retry_count = 0
            
            logger.info(
                f"Reset retry counter for step {step_id}: "
                f"{old_count} -> 0"
            )
            
            state.add_log_entry(
                agent_name="retry_protection",
                action="reset_retry_counter",
                details={
                    "step_id": step_id,
                    "old_retry_count": old_count
                }
            )
            
            break
    
    return state
