"""重试保护机制 - 防止无限循环并强制执行重试限制

本模块实现重试限制检查和循环保护机制：
1. 检查当前步骤是否超过最大重试次数
2. 通过跳过步骤或标记为失败来强制降级
3. 为跳过的步骤添加占位符片段
4. 防止无限转向循环
"""

import logging
from typing import Optional
from ..models import SharedState, OutlineStep, ScreenplayFragment
from ...infrastructure.logging import get_agent_logger

logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


def check_retry_limit(state: SharedState) -> SharedState:
    """检查重试限制，超出则强制降级
    
    本函数实现循环保护机制：
    1. 检查当前步骤的重试次数是否超过 max_retries
    2. 强制降级：跳过该步骤
    3. 添加占位符片段解释跳过原因
    4. 推进到下一个步骤
    
    Args:
        state: 共享状态对象
        
    Returns:
        更新后的共享状态，步骤可能被跳过
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
    """检测系统是否陷入无限循环
    
    本函数分析最近的执行日志，检测表示无限循环的模式
    （例如：同一步骤被反复重试）。
    
    Args:
        state: 共享状态对象
        window_size: 要分析的最新日志条目数量
        
    Returns:
        检测到无限循环返回 True，否则返回 False
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
    """重置特定步骤的重试计数器
    
    当步骤成功完成或手动干预解决问题时，此功能很有用。
    
    Args:
        state: 共享状态对象
        step_id: 要重置的步骤 ID
        
    Returns:
        更新后的共享状态
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
