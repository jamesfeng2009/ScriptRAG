"""
属性测试：大纲修改持久性

属性 2: 大纲修改持久性
对于任何大纲修改操作（添加、删除、更新步骤），修改应在共享状态中持久化，
并在后续访问中保持一致。

Feature: rag-screenplay-multi-agent
Property 2: 大纲修改持久性
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from copy import deepcopy

from src.domain.models import SharedState, OutlineStep


# 重用策略生成器
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
def shared_state_with_outline_strategy(draw):
    """生成带有大纲的 SharedState"""
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
    
    return SharedState(
        user_topic=user_topic,
        project_context=draw(st.text(max_size=500)),
        outline=outline,
        current_step_index=current_step_index,
        current_skill=draw(st.sampled_from(valid_skills)),
        global_tone=global_tone,
        max_retries=draw(st.integers(min_value=1, max_value=10)),
        execution_log=[]
    )


class TestOutlineModificationPersistence:
    """测试大纲修改持久性属性"""
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_add_step_persistence(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 添加步骤持久性
        
        当添加新步骤到大纲时，该步骤应持久化并在后续访问中可见。
        """
        # 记录初始大纲大小
        initial_outline_size = len(state.outline)
        
        # 创建新步骤
        new_step = OutlineStep(
            step_id=initial_outline_size,
            description="New test step",
            status="pending",
            retry_count=0
        )
        
        # 添加新步骤
        state.outline.append(new_step)
        
        # 验证步骤已添加并持久化
        assert len(state.outline) == initial_outline_size + 1
        assert state.outline[-1].step_id == initial_outline_size
        assert state.outline[-1].description == "New test step"
        assert state.outline[-1].status == "pending"
        
        # 再次访问验证持久性
        retrieved_step = state.outline[-1]
        assert retrieved_step.step_id == new_step.step_id
        assert retrieved_step.description == new_step.description
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_update_step_persistence(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 更新步骤持久性
        
        当更新大纲中的步骤时，更新应持久化并在后续访问中可见。
        """
        # 选择要更新的步骤
        target_index = 0
        initial_description = state.outline[target_index].description
        initial_status = state.outline[target_index].status
        initial_retry_count = state.outline[target_index].retry_count
        
        # 更新步骤
        state.outline[target_index].description = "Updated description"
        state.outline[target_index].status = "in_progress"
        state.outline[target_index].retry_count = initial_retry_count + 1
        
        # 验证更新已持久化
        assert state.outline[target_index].description == "Updated description"
        assert state.outline[target_index].description != initial_description
        assert state.outline[target_index].status == "in_progress"
        assert state.outline[target_index].retry_count == initial_retry_count + 1
        
        # 再次访问验证持久性
        retrieved_step = state.outline[target_index]
        assert retrieved_step.description == "Updated description"
        assert retrieved_step.status == "in_progress"
        assert retrieved_step.retry_count == initial_retry_count + 1
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_remove_step_persistence(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 删除步骤持久性
        
        当从大纲中删除步骤时，删除应持久化并在后续访问中反映。
        """
        # 确保至少有两个步骤
        assume(len(state.outline) >= 2)
        
        # 记录初始状态
        initial_outline_size = len(state.outline)
        step_to_remove = state.outline[0]
        removed_step_id = step_to_remove.step_id
        
        # 删除第一个步骤
        state.outline.pop(0)
        
        # 验证删除已持久化
        assert len(state.outline) == initial_outline_size - 1
        
        # 验证被删除的步骤不再存在
        remaining_step_ids = [step.step_id for step in state.outline]
        assert removed_step_id not in remaining_step_ids
        
        # 再次访问验证持久性
        assert len(state.outline) == initial_outline_size - 1
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_multiple_modifications_persistence(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 多次修改持久性
        
        当对大纲进行多次修改时，所有修改应累积持久化。
        """
        # 记录初始状态
        initial_outline_size = len(state.outline)
        
        # 执行多次修改
        # 1. 更新第一个步骤
        state.outline[0].status = "completed"
        
        # 2. 添加新步骤
        new_step = OutlineStep(
            step_id=initial_outline_size,
            description="Additional step",
            status="pending",
            retry_count=0
        )
        state.outline.append(new_step)
        
        # 3. 更新最后一个步骤的重试计数
        state.outline[-1].retry_count = 1
        
        # 验证所有修改都已持久化
        assert state.outline[0].status == "completed"
        assert len(state.outline) == initial_outline_size + 1
        assert state.outline[-1].description == "Additional step"
        assert state.outline[-1].retry_count == 1
        
        # 再次访问验证持久性
        assert state.outline[0].status == "completed"
        assert len(state.outline) == initial_outline_size + 1
        assert state.outline[-1].retry_count == 1
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_step_status_transition_persistence(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 步骤状态转换持久性
        
        当步骤状态经过多次转换时，每次转换都应持久化。
        """
        target_index = 0
        
        # 状态转换序列
        status_sequence = ["pending", "in_progress", "completed"]
        
        for expected_status in status_sequence:
            state.outline[target_index].status = expected_status
            
            # 验证状态已更新并持久化
            assert state.outline[target_index].status == expected_status
            
            # 再次访问验证持久性
            retrieved_status = state.outline[target_index].status
            assert retrieved_status == expected_status
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_retry_count_increment_persistence(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 重试计数增量持久性
        
        当步骤的重试计数递增时，每次递增都应持久化。
        """
        target_index = 0
        initial_retry_count = state.outline[target_index].retry_count
        
        # 递增重试计数多次
        for i in range(1, 4):
            state.outline[target_index].retry_count += 1
            
            # 验证递增已持久化
            expected_count = initial_retry_count + i
            assert state.outline[target_index].retry_count == expected_count
            
            # 再次访问验证持久性
            retrieved_count = state.outline[target_index].retry_count
            assert retrieved_count == expected_count
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_outline_replacement_persistence(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 大纲替换持久性
        
        当整个大纲被替换时，新大纲应持久化。
        """
        # 创建新大纲
        new_outline = [
            OutlineStep(
                step_id=0,
                description="New step 1",
                status="pending",
                retry_count=0
            ),
            OutlineStep(
                step_id=1,
                description="New step 2",
                status="pending",
                retry_count=0
            )
        ]
        
        # 替换大纲
        state.outline = new_outline
        
        # 验证新大纲已持久化
        assert len(state.outline) == 2
        assert state.outline[0].description == "New step 1"
        assert state.outline[1].description == "New step 2"
        
        # 再次访问验证持久性
        assert len(state.outline) == 2
        assert state.outline[0].step_id == 0
        assert state.outline[1].step_id == 1
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_pivot_triggered_outline_modification(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 转向触发的大纲修改
        
        当转向触发导致大纲修改时，修改应持久化（模拟需求 5.2）。
        """
        # 模拟转向触发
        state.pivot_triggered = True
        state.pivot_reason = "deprecation_conflict"
        
        # 模拟转向管理器修改当前步骤
        current_step = state.outline[state.current_step_index]
        original_description = current_step.description
        
        # 修改当前步骤以反映废弃警告
        current_step.description = f"{original_description} (废弃警告)"
        current_step.retry_count += 1
        
        # 验证修改已持久化
        assert state.pivot_triggered is True
        assert state.pivot_reason == "deprecation_conflict"
        assert "(废弃警告)" in state.outline[state.current_step_index].description
        assert state.outline[state.current_step_index].retry_count > 0
        
        # 再次访问验证持久性
        retrieved_step = state.outline[state.current_step_index]
        assert "(废弃警告)" in retrieved_step.description
    
    @given(state=shared_state_with_outline_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_2_outline_modification_with_state_copy(self, state: SharedState):
        """
        属性 2: 大纲修改持久性 - 状态拷贝后的大纲修改
        
        验证深拷贝后的状态修改不影响原始状态的大纲。
        """
        # 创建深拷贝
        state_copy = deepcopy(state)
        
        # 修改拷贝的大纲
        state_copy.outline[0].status = "completed"
        state_copy.outline[0].description = "Modified in copy"
        
        # 验证原始状态未受影响
        assert state.outline[0].status != "completed" or state.outline[0].description != "Modified in copy"
        
        # 验证拷贝的修改已持久化
        assert state_copy.outline[0].status == "completed"
        assert state_copy.outline[0].description == "Modified in copy"
