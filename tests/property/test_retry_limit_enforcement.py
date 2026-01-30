"""
属性测试：重试限制执行

属性 13: 重试限制执行
当步骤的重试次数达到或超过 max_retries 时，系统应强制降级，
跳过该步骤并添加占位符片段，防止无限循环。

Feature: rag-screenplay-multi-agent
Property 13: 重试限制执行
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from copy import deepcopy

from src.domain.models import SharedState, OutlineStep, ScreenplayFragment
from src.domain.agents.retry_protection import check_retry_limit


# 策略生成器
@st.composite
def outline_step_strategy(draw):
    """生成有效的 OutlineStep"""
    description = draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    return OutlineStep(
        step_id=draw(st.integers(min_value=0, max_value=100)),
        description=description,
        status=draw(st.sampled_from(["pending", "in_progress", "completed", "skipped"])),
        retry_count=draw(st.integers(min_value=0, max_value=10))
    )


@st.composite
def shared_state_with_retry_strategy(draw):
    """生成带有重试计数的 SharedState"""
    valid_skills = [
        "standard_tutorial",
        "warning_mode",
        "visualization_analogy",
        "research_mode",
        "meme_style",
        "fallback_summary"
    ]
    
    # 生成至少有一个步骤的大纲
    outline_size = draw(st.integers(min_value=1, max_value=10))
    outline = [draw(outline_step_strategy()) for _ in range(outline_size)]
    
    # 确保 step_id 唯一且连续
    for i, step in enumerate(outline):
        step.step_id = i
    
    # 生成当前步骤索引（在有效范围内）
    current_step_index = draw(st.integers(min_value=0, max_value=outline_size - 1))
    
    user_topic = draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    global_tone = draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    
    # 生成 max_retries（1-10）
    max_retries = draw(st.integers(min_value=1, max_value=10))
    
    return SharedState(
        user_topic=user_topic,
        project_context=draw(st.text(max_size=500)),
        outline=outline,
        current_step_index=current_step_index,
        current_skill=draw(st.sampled_from(valid_skills)),
        global_tone=global_tone,
        max_retries=max_retries,
        execution_log=[]
    )


@st.composite
def state_with_exceeded_retry_strategy(draw):
    """生成重试次数已超过限制的 SharedState"""
    state = draw(shared_state_with_retry_strategy())
    
    # 确保当前步骤的重试次数超过或等于 max_retries
    current_step = state.outline[state.current_step_index]
    current_step.retry_count = draw(st.integers(
        min_value=state.max_retries,
        max_value=state.max_retries + 5
    ))
    
    return state


@st.composite
def state_with_below_retry_strategy(draw):
    """生成重试次数未超过限制的 SharedState"""
    state = draw(shared_state_with_retry_strategy())
    
    # 确保当前步骤的重试次数低于 max_retries
    current_step = state.outline[state.current_step_index]
    current_step.retry_count = draw(st.integers(
        min_value=0,
        max_value=max(0, state.max_retries - 1)
    ))
    
    return state


class TestRetryLimitEnforcement:
    """测试重试限制执行属性"""
    
    @given(state=state_with_exceeded_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_retry_limit_forces_skip(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 超过限制强制跳过
        
        当步骤的 retry_count >= max_retries 时，check_retry_limit 应：
        1. 将步骤状态设置为 "skipped"
        2. 添加占位符片段
        3. 清除 pivot 触发器
        """
        # 记录初始状态
        current_step = state.get_current_step()
        initial_step_id = current_step.step_id
        initial_retry_count = current_step.retry_count
        initial_fragments_count = len(state.fragments)
        
        # 验证前提条件：重试次数已超过限制
        assert initial_retry_count >= state.max_retries
        
        # 执行重试限制检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤被跳过
        skipped_step = None
        for step in updated_state.outline:
            if step.step_id == initial_step_id:
                skipped_step = step
                break
        
        assert skipped_step is not None
        assert skipped_step.status == "skipped"
        
        # 验证添加了占位符片段
        assert len(updated_state.fragments) == initial_fragments_count + 1
        
        # 验证占位符片段内容
        placeholder = updated_state.fragments[-1]
        assert placeholder.step_id == initial_step_id
        assert "[SKIPPED]" in placeholder.content
        assert "retry attempts" in placeholder.content.lower()
        
        # 验证 pivot 触发器被清除
        assert updated_state.pivot_triggered is False
        assert updated_state.pivot_reason is None
    
    @given(state=state_with_exceeded_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_retry_limit_advances_step(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 超过限制后前进到下一步
        
        当步骤被跳过后，系统应前进到下一步（如果存在）。
        """
        # 确保不是最后一步
        assume(state.current_step_index < len(state.outline) - 1)
        
        # 记录初始步骤索引
        initial_step_index = state.current_step_index
        
        # 执行重试限制检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤索引已前进
        assert updated_state.current_step_index == initial_step_index + 1
    
    @given(state=state_with_exceeded_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_retry_limit_logs_action(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 记录强制跳过动作
        
        当步骤被强制跳过时，应在执行日志中记录该动作。
        """
        # 记录初始日志数量
        initial_log_count = len(state.execution_log)
        current_step = state.get_current_step()
        initial_step_id = current_step.step_id
        
        # 执行重试限制检查
        updated_state = check_retry_limit(state)
        
        # 验证添加了日志条目
        assert len(updated_state.execution_log) == initial_log_count + 1
        
        # 验证日志内容
        log_entry = updated_state.execution_log[-1]
        assert log_entry["agent_name"] == "retry_protection"
        assert log_entry["action"] == "force_skip"
        assert log_entry["details"]["step_id"] == initial_step_id
        assert log_entry["details"]["reason"] == "retry_limit_exceeded"
    
    @given(state=state_with_below_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_below_limit_no_skip(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 未超过限制不跳过
        
        当步骤的 retry_count < max_retries 时，check_retry_limit 不应：
        1. 修改步骤状态
        2. 添加占位符片段
        3. 前进步骤索引
        """
        # 记录初始状态
        current_step = state.get_current_step()
        initial_status = current_step.status
        initial_step_index = state.current_step_index
        initial_fragments_count = len(state.fragments)
        
        # 验证前提条件：重试次数未超过限制
        assert current_step.retry_count < state.max_retries
        
        # 执行重试限制检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤状态未改变
        updated_step = updated_state.get_current_step()
        assert updated_step.status == initial_status
        
        # 验证未添加占位符片段
        assert len(updated_state.fragments) == initial_fragments_count
        
        # 验证步骤索引未改变
        assert updated_state.current_step_index == initial_step_index
    
    @given(state=state_with_exceeded_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_placeholder_fragment_structure(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 占位符片段结构正确
        
        占位符片段应包含：
        1. 正确的 step_id
        2. 说明性内容
        3. 当前使用的 skill
        4. 空的 sources 列表
        """
        # 记录初始状态
        current_step = state.get_current_step()
        initial_step_id = current_step.step_id
        current_skill = state.current_skill
        
        # 执行重试限制检查
        updated_state = check_retry_limit(state)
        
        # 获取占位符片段
        placeholder = updated_state.fragments[-1]
        
        # 验证片段结构
        assert placeholder.step_id == initial_step_id
        assert isinstance(placeholder.content, str)
        assert len(placeholder.content) > 0
        assert placeholder.skill_used == current_skill
        assert isinstance(placeholder.sources, list)
        assert len(placeholder.sources) == 0
    
    @given(state=state_with_exceeded_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_multiple_steps_exceed_limit(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 多个步骤超过限制
        
        当多个步骤的重试次数超过限制时，每个步骤应独立处理。
        """
        # 确保至少有 3 个步骤
        assume(len(state.outline) >= 3)
        
        # 设置多个步骤的重试次数超过限制
        for i in range(min(3, len(state.outline))):
            state.outline[i].retry_count = state.max_retries + 1
        
        # 从第一个步骤开始
        state.current_step_index = 0
        
        skipped_count = 0
        max_iterations = len(state.outline)
        
        # 处理所有超过限制的步骤
        for _ in range(max_iterations):
            current_step = state.get_current_step()
            if not current_step:
                break
            
            if current_step.retry_count >= state.max_retries:
                initial_step_id = current_step.step_id
                state = check_retry_limit(state)
                
                # 验证该步骤被跳过
                for step in state.outline:
                    if step.step_id == initial_step_id:
                        assert step.status == "skipped"
                        skipped_count += 1
                        break
            else:
                # 如果当前步骤未超过限制，手动前进
                if not state.advance_step():
                    break
        
        # 验证至少有一个步骤被跳过
        assert skipped_count >= 1
    
    @given(state=state_with_exceeded_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_last_step_exceeded(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 最后一步超过限制
        
        当最后一步的重试次数超过限制时，应正确处理而不崩溃。
        """
        # 设置为最后一步
        state.current_step_index = len(state.outline) - 1
        
        # 确保最后一步超过限制
        last_step = state.get_current_step()
        last_step.retry_count = state.max_retries + 1
        
        # 执行重试限制检查（不应崩溃）
        updated_state = check_retry_limit(state)
        
        # 验证最后一步被跳过
        assert last_step.status == "skipped"
        
        # 验证添加了占位符片段
        assert len(updated_state.fragments) > 0
        assert updated_state.fragments[-1].step_id == last_step.step_id
    
    @given(state=state_with_exceeded_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_pivot_cleared_on_skip(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 跳过时清除 pivot 状态
        
        当步骤被跳过时，应清除 pivot_triggered 和 pivot_reason。
        """
        # 设置 pivot 触发器
        state.pivot_triggered = True
        state.pivot_reason = "test_reason"
        
        # 执行重试限制检查
        updated_state = check_retry_limit(state)
        
        # 验证 pivot 状态被清除
        assert updated_state.pivot_triggered is False
        assert updated_state.pivot_reason is None
    
    @given(state=shared_state_with_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_idempotent_check(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 幂等性检查
        
        对同一状态多次调用 check_retry_limit 应产生一致的结果。
        """
        # 创建状态副本
        state_copy = deepcopy(state)
        
        # 第一次检查
        result1 = check_retry_limit(state)
        
        # 第二次检查（使用副本）
        result2 = check_retry_limit(state_copy)
        
        # 验证结果一致
        current_step1 = result1.get_current_step()
        current_step2 = result2.get_current_step()
        
        if current_step1 and current_step2:
            assert current_step1.status == current_step2.status
            assert len(result1.fragments) == len(result2.fragments)
            assert result1.current_step_index == result2.current_step_index
    
    @given(state=state_with_exceeded_retry_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_13_retry_count_preserved(self, state: SharedState):
        """
        属性 13: 重试限制执行 - 重试计数保留
        
        当步骤被跳过时，其 retry_count 应保留（不重置）。
        """
        # 记录初始重试计数
        current_step = state.get_current_step()
        initial_retry_count = current_step.retry_count
        initial_step_id = current_step.step_id
        
        # 执行重试限制检查
        updated_state = check_retry_limit(state)
        
        # 查找被跳过的步骤
        skipped_step = None
        for step in updated_state.outline:
            if step.step_id == initial_step_id:
                skipped_step = step
                break
        
        # 验证重试计数未被重置
        assert skipped_step is not None
        assert skipped_step.retry_count == initial_retry_count
