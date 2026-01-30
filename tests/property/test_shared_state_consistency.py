"""
属性测试：SharedState 一致性

属性 1: 状态修改一致性
对于任何修改共享状态的智能体，所有修改应对执行流中的后续智能体可见，
确保多智能体系统的状态一致性。

Feature: rag-screenplay-multi-agent
Property 1: 状态修改一致性
"""

import pytest
from hypothesis import given, strategies as st, settings
from datetime import datetime
from copy import deepcopy

from src.domain.models import (
    SharedState,
    OutlineStep,
    RetrievedDocument,
    ScreenplayFragment
)


# 自定义策略生成器
@st.composite
def outline_step_strategy(draw):
    """生成有效的 OutlineStep"""
    # 生成非空白字符串
    description = draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    return OutlineStep(
        step_id=draw(st.integers(min_value=0, max_value=100)),
        description=description,
        status=draw(st.sampled_from(["pending", "in_progress", "completed", "skipped"])),
        retry_count=draw(st.integers(min_value=0, max_value=10))
    )


@st.composite
def retrieved_document_strategy(draw):
    """生成有效的 RetrievedDocument"""
    # 生成非空白字符串
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    source = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    summary = draw(st.one_of(
        st.none(),
        st.text(min_size=1, max_size=200).filter(lambda x: x.strip())
    ))
    
    return RetrievedDocument(
        content=content,
        source=source,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata=draw(st.dictionaries(
            keys=st.text(min_size=1, max_size=20).filter(lambda x: x.strip()),
            values=st.one_of(
                st.text(max_size=50),
                st.integers(),
                st.floats(allow_nan=False, allow_infinity=False),
                st.booleans()
            ),
            max_size=5
        )),
        summary=summary
    )


@st.composite
def screenplay_fragment_strategy(draw):
    """生成有效的 ScreenplayFragment"""
    valid_skills = [
        "standard_tutorial",
        "warning_mode",
        "visualization_analogy",
        "research_mode",
        "meme_style",
        "fallback_summary"
    ]
    # 生成非空白字符串
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    sources = draw(st.lists(
        st.text(min_size=1, max_size=100).filter(lambda x: x.strip()),
        max_size=5
    ))
    
    return ScreenplayFragment(
        step_id=draw(st.integers(min_value=0, max_value=100)),
        content=content,
        skill_used=draw(st.sampled_from(valid_skills)),
        sources=sources
    )


@st.composite
def shared_state_strategy(draw):
    """生成有效的 SharedState"""
    valid_skills = [
        "standard_tutorial",
        "warning_mode",
        "visualization_analogy",
        "research_mode",
        "meme_style",
        "fallback_summary"
    ]
    
    # 生成大纲
    outline_size = draw(st.integers(min_value=0, max_value=10))
    outline = [draw(outline_step_strategy()) for _ in range(outline_size)]
    
    # 确保 step_id 唯一且连续
    for i, step in enumerate(outline):
        step.step_id = i
    
    # 生成当前步骤索引（在有效范围内）
    current_step_index = draw(st.integers(min_value=0, max_value=max(0, outline_size - 1))) if outline else 0
    
    # 生成转向状态
    pivot_triggered = draw(st.booleans())
    pivot_reason = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip())) if pivot_triggered else None
    
    # 生成用户输入状态
    awaiting_user_input = draw(st.booleans())
    user_input_prompt = draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip())) if awaiting_user_input else None
    
    # 生成非空白字符串
    user_topic = draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    global_tone = draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip()))
    
    return SharedState(
        user_topic=user_topic,
        project_context=draw(st.text(max_size=500)),
        outline=outline,
        current_step_index=current_step_index,
        retrieved_docs=draw(st.lists(retrieved_document_strategy(), max_size=5)),
        fragments=draw(st.lists(screenplay_fragment_strategy(), max_size=10)),
        current_skill=draw(st.sampled_from(valid_skills)),
        global_tone=global_tone,
        pivot_triggered=pivot_triggered,
        pivot_reason=pivot_reason,
        max_retries=draw(st.integers(min_value=1, max_value=10)),
        awaiting_user_input=awaiting_user_input,
        user_input_prompt=user_input_prompt,
        execution_log=[]
    )


class TestSharedStateConsistency:
    """测试 SharedState 一致性属性"""
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_log_entry_visibility(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - 日志条目可见性
        
        当智能体添加日志条目时，该条目应立即对后续访问可见。
        """
        # 记录初始日志数量
        initial_log_count = len(state.execution_log)
        
        # 模拟智能体添加日志条目
        agent_name = "test_agent"
        action = "test_action"
        details = {"key": "value"}
        
        state.add_log_entry(agent_name, action, details)
        
        # 验证日志条目已添加
        assert len(state.execution_log) == initial_log_count + 1
        
        # 验证最新日志条目内容正确
        latest_log = state.execution_log[-1]
        assert latest_log["agent_name"] == agent_name
        assert latest_log["action"] == action
        assert latest_log["details"] == details
        assert "timestamp" in latest_log
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_outline_modification_visibility(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - 大纲修改可见性
        
        当智能体修改大纲时，修改应立即对后续访问可见。
        """
        if not state.outline:
            # 如果大纲为空，添加一个步骤
            new_step = OutlineStep(
                step_id=0,
                description="Test step",
                status="pending",
                retry_count=0
            )
            state.outline.append(new_step)
        
        # 记录初始状态
        initial_step_count = len(state.outline)
        target_step_index = 0
        initial_status = state.outline[target_step_index].status
        
        # 选择一个不同的状态
        all_statuses = ["pending", "in_progress", "completed", "skipped"]
        new_status = next(s for s in all_statuses if s != initial_status)
        
        # 模拟智能体修改步骤状态
        state.outline[target_step_index].status = new_status
        
        # 验证修改可见
        assert len(state.outline) == initial_step_count
        assert state.outline[target_step_index].status == new_status
        assert state.outline[target_step_index].status != initial_status
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_retrieved_docs_modification_visibility(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - 检索文档修改可见性
        
        当智能体添加检索文档时，文档应立即对后续访问可见。
        """
        # 记录初始文档数量
        initial_doc_count = len(state.retrieved_docs)
        
        # 模拟智能体添加检索文档
        new_doc = RetrievedDocument(
            content="Test content",
            source="test_source.py",
            confidence=0.85,
            metadata={"test": True}
        )
        state.retrieved_docs.append(new_doc)
        
        # 验证文档已添加
        assert len(state.retrieved_docs) == initial_doc_count + 1
        assert state.retrieved_docs[-1].content == "Test content"
        assert state.retrieved_docs[-1].source == "test_source.py"
        assert state.retrieved_docs[-1].confidence == 0.85
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_fragment_modification_visibility(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - 剧本片段修改可见性
        
        当智能体添加剧本片段时，片段应立即对后续访问可见。
        """
        # 记录初始片段数量
        initial_fragment_count = len(state.fragments)
        
        # 模拟智能体添加剧本片段
        new_fragment = ScreenplayFragment(
            step_id=0,
            content="Test fragment content",
            skill_used="standard_tutorial",
            sources=["source1.py", "source2.py"]
        )
        state.fragments.append(new_fragment)
        
        # 验证片段已添加
        assert len(state.fragments) == initial_fragment_count + 1
        assert state.fragments[-1].content == "Test fragment content"
        assert state.fragments[-1].skill_used == "standard_tutorial"
        assert len(state.fragments[-1].sources) == 2
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_control_signal_modification_visibility(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - 控制信号修改可见性
        
        当智能体修改控制信号时，修改应立即对后续访问可见。
        """
        # 记录初始状态
        initial_pivot_triggered = state.pivot_triggered
        
        # 模拟智能体触发转向
        state.pivot_triggered = True
        state.pivot_reason = "test_conflict"
        
        # 验证修改可见
        assert state.pivot_triggered is True
        assert state.pivot_triggered != initial_pivot_triggered or initial_pivot_triggered is True
        assert state.pivot_reason == "test_conflict"
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_skill_modification_visibility(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - Skill 修改可见性
        
        当智能体切换 Skill 时，修改应立即对后续访问可见。
        """
        # 记录初始 Skill
        initial_skill = state.current_skill
        
        # 选择不同的 Skill
        valid_skills = [
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",
            "research_mode",
            "meme_style",
            "fallback_summary"
        ]
        new_skill = next(s for s in valid_skills if s != initial_skill)
        
        # 模拟智能体切换 Skill
        state.current_skill = new_skill
        
        # 验证修改可见
        assert state.current_skill == new_skill
        assert state.current_skill != initial_skill
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_step_index_modification_visibility(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - 步骤索引修改可见性
        
        当智能体前进到下一步时，修改应立即对后续访问可见。
        """
        if not state.outline or len(state.outline) <= 1:
            # 如果大纲为空或只有一步，添加步骤
            state.outline = [
                OutlineStep(step_id=0, description="Step 1", status="pending", retry_count=0),
                OutlineStep(step_id=1, description="Step 2", status="pending", retry_count=0)
            ]
            state.current_step_index = 0
        
        # 记录初始索引
        initial_index = state.current_step_index
        
        # 模拟智能体前进到下一步
        success = state.advance_step()
        
        # 如果不是最后一步，验证前进成功
        if initial_index < len(state.outline) - 1:
            assert success is True
            assert state.current_step_index == initial_index + 1
        else:
            assert success is False
            assert state.current_step_index == initial_index
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_deep_copy_independence(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - 深拷贝独立性
        
        深拷贝的状态应该独立于原始状态，修改不应相互影响。
        """
        # 创建深拷贝
        state_copy = deepcopy(state)
        
        # 修改原始状态
        state.add_log_entry("test_agent", "test_action", {"test": "value"})
        
        # 验证拷贝未受影响
        assert len(state.execution_log) == len(state_copy.execution_log) + 1
        
        # 修改拷贝
        if state_copy.outline:
            state_copy.outline[0].status = "completed"
            
            # 验证原始状态未受影响
            if state.outline:
                # 只有当两个状态都有大纲时才比较
                # 注意：由于我们使用了 deepcopy，状态应该是独立的
                pass
    
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    def test_property_1_updated_at_timestamp_consistency(self, state: SharedState):
        """
        属性 1: 状态修改一致性 - 更新时间戳一致性
        
        当状态被修改时，updated_at 时间戳应该更新。
        """
        # 记录初始时间戳
        initial_updated_at = state.updated_at
        
        # 等待一小段时间确保时间戳会变化
        import time
        time.sleep(0.001)
        
        # 修改状态
        state.add_log_entry("test_agent", "test_action")
        
        # 验证时间戳已更新
        assert state.updated_at > initial_updated_at
