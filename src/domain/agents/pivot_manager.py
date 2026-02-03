"""转向管理器智能体 - 在发生冲突时修正大纲

本模块实现转向管理器智能体，负责：
1. 处理来自导演的转向触发
2. 根据转向原因修改大纲步骤
3. 在兼容性约束下应用技能切换
4. 更新重试计数器和步骤状态
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
    """基于兼容性规则约束技能切换
    
    本函数确保技能切换遵守兼容性约束。
    如果所需技能与当前技能不直接兼容，
    它会根据全局语气偏好找到最接近的兼容技能。
    
    Args:
        current_skill: 当前活动技能名称
        desired_skill: 期望的目标技能名称
        global_tone: 可选的全局语气偏好用于选择
        
    Returns:
        要切换到的技能名称（可能与 desired_skill 不同）
        
    Raises:
        ValueError: 技能名称无效
    
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
    """处理转向触发并相应修改大纲
    
    本函数在导演触发转向时调用。它：
    1. 分析转向原因
    2. 修改当前和后续的大纲步骤
    3. 在兼容性约束下应用技能切换
    4. 更新重试计数器和步骤状态
    5. 清除检索到的文档以触发重新检索
    
    Args:
        state: 当前共享状态，pivot_triggered=True
        
    Returns:
        更新后的大纲已修改的共享状态
        
    Raises:
        ValueError: 如果触发转向但没有原因
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
    """处理由复杂度触发的转向
    
    当内容过于复杂时：
    1. 切换到 visualization_analogy 技能（带兼容性约束）
    2. 修改当前步骤以强调简化
    
    Args:
        state: 当前共享状态
        current_step: 当前大纲步骤
        
    Returns:
        更新后的共享状态
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
    """处理由信息缺失触发的转向
    
    当信息不足时：
    1. 切换到 research_mode 技能（带兼容性约束）
    2. 修改当前步骤以承认信息缺口
    
    Args:
        state: 当前共享状态
        current_step: 当前大纲步骤
        
    Returns:
        更新后的共享状态
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
