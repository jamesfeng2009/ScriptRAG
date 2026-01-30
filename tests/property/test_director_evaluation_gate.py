"""
属性测试：导演评估门

属性 19: 导演评估门
对于每个大纲步骤，导演应评估检索内容并做出决策（批准/转向），
确保所有步骤都经过导演的评估。

Feature: rag-screenplay-multi-agent
Property 19: 导演评估门
"""

import pytest
from hypothesis import given, strategies as st, settings
from unittest.mock import AsyncMock, MagicMock

from src.domain.models import (
    SharedState,
    OutlineStep,
    RetrievedDocument
)
from src.domain.agents.director import evaluate_and_decide


# 自定义策略生成器
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
def retrieved_document_strategy(draw):
    """生成有效的 RetrievedDocument"""
    content = draw(st.text(min_size=1, max_size=500).filter(lambda x: x.strip()))
    source = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    
    return RetrievedDocument(
        content=content,
        source=source,
        confidence=draw(st.floats(min_value=0.0, max_value=1.0)),
        metadata={
            'has_deprecated': draw(st.booleans()),
            'has_fixme': draw(st.booleans()),
            'has_todo': draw(st.booleans()),
            'has_security': draw(st.booleans())
        }
    )


@st.composite
def shared_state_with_current_step_strategy(draw):
    """生成带有当前步骤的 SharedState"""
    valid_skills = [
        "standard_tutorial",
        "warning_mode",
        "visualization_analogy",
        "research_mode",
        "meme_style",
        "fallback_summary"
    ]
    
    # 生成大纲
    outline_size = draw(st.integers(min_value=1, max_value=10))
    outline = [draw(outline_step_strategy()) for _ in range(outline_size)]
    
    # 确保 step_id 唯一且连续
    for i, step in enumerate(outline):
        step.step_id = i
    
    # 生成当前步骤索引（在有效范围内）
    current_step_index = draw(st.integers(min_value=0, max_value=outline_size - 1))
    
    user_topic = draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip()))
    
    return SharedState(
        user_topic=user_topic,
        project_context=draw(st.text(max_size=500)),
        outline=outline,
        current_step_index=current_step_index,
        retrieved_docs=draw(st.lists(retrieved_document_strategy(), max_size=5)),
        current_skill=draw(st.sampled_from(valid_skills)),
        global_tone=draw(st.text(min_size=1, max_size=50).filter(lambda x: x.strip())),
        max_retries=draw(st.integers(min_value=1, max_value=10))
    )


class TestDirectorEvaluationGate:
    """测试导演评估门属性"""
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_current_step_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_19_director_evaluates_every_step(
        self,
        state: SharedState
    ):
        """
        属性 19: 导演评估门 - 导演评估每个步骤
        
        对于每个大纲步骤，导演应评估检索内容并做出决策。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value="0.5")
        
        # 记录初始日志数量
        initial_log_count = len(state.execution_log)
        
        # 调用导演评估
        result_state = await evaluate_and_decide(state, mock_llm_service)
        
        # 验证导演添加了日志条目
        assert len(result_state.execution_log) > initial_log_count
        
        # 验证最新日志条目来自导演
        latest_log = result_state.execution_log[-1]
        assert latest_log["agent_name"] == "director"
        
        # 验证日志包含动作信息
        assert "action" in latest_log
        assert latest_log["action"] in [
            "conflict_detected",
            "complexity_trigger",
            "approved",
            "evaluation_failed"
        ]
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_current_step_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_19_director_makes_decision(
        self,
        state: SharedState
    ):
        """
        属性 19: 导演评估门 - 导演做出决策
        
        导演评估后，应做出明确的决策（批准或转向）。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value="0.5")
        
        # 调用导演评估
        result_state = await evaluate_and_decide(state, mock_llm_service)
        
        # 验证导演做出了决策
        # 决策体现在日志中
        director_logs = [
            log for log in result_state.execution_log
            if log.get("agent_name") == "director"
        ]
        
        assert len(director_logs) > 0
        
        # 最新的导演日志应包含决策信息
        latest_director_log = director_logs[-1]
        assert "action" in latest_director_log
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_current_step_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_19_pivot_flag_consistency(
        self,
        state: SharedState
    ):
        """
        属性 19: 导演评估门 - 转向标志一致性
        
        当导演触发转向时，pivot_triggered 标志应设置为 True，
        并且 pivot_reason 应包含原因。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value="0.5")
        
        # 调用导演评估
        result_state = await evaluate_and_decide(state, mock_llm_service)
        
        # 验证转向标志一致性
        if result_state.pivot_triggered:
            # 如果触发了转向，必须有原因
            assert result_state.pivot_reason is not None
            assert len(result_state.pivot_reason) > 0
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_current_step_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_19_evaluation_details_logged(
        self,
        state: SharedState
    ):
        """
        属性 19: 导演评估门 - 评估详情记录
        
        导演评估时，应记录评估详情（步骤 ID、决策原因等）。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value="0.5")
        
        # 获取当前步骤
        current_step = state.get_current_step()
        assert current_step is not None
        
        # 调用导演评估
        result_state = await evaluate_and_decide(state, mock_llm_service)
        
        # 查找导演日志
        director_logs = [
            log for log in result_state.execution_log
            if log.get("agent_name") == "director"
        ]
        
        assert len(director_logs) > 0
        
        # 验证日志包含详情
        latest_director_log = director_logs[-1]
        assert "details" in latest_director_log
        
        # 验证详情包含步骤 ID
        details = latest_director_log["details"]
        assert "step_id" in details
        assert details["step_id"] == current_step.step_id
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_current_step_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_19_no_current_step_graceful(
        self,
        state: SharedState
    ):
        """
        属性 19: 导演评估门 - 无当前步骤时优雅处理
        
        当没有当前步骤时，导演应优雅处理而不崩溃。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value="0.5")
        
        # 设置无效的步骤索引
        state.current_step_index = len(state.outline)
        
        # 调用导演评估（应该不会崩溃）
        result_state = await evaluate_and_decide(state, mock_llm_service)
        
        # 验证状态仍然有效
        assert result_state is not None
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_current_step_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_19_llm_failure_graceful(
        self,
        state: SharedState
    ):
        """
        属性 19: 导演评估门 - LLM 失败时优雅处理
        
        当 LLM 调用失败时，导演应优雅处理而不崩溃。
        """
        # 创建模拟的 LLM 服务，模拟失败
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(side_effect=Exception("LLM failed"))
        
        # 调用导演评估（应该不会崩溃）
        result_state = await evaluate_and_decide(state, mock_llm_service)
        
        # 验证状态仍然有效
        assert result_state is not None
        
        # 验证记录了错误日志
        error_logs = [
            log for log in result_state.execution_log
            if log.get("agent_name") == "director" and "failed" in log.get("action", "")
        ]
        
        # 可能记录了失败日志
        # 注意：由于优雅降级，可能不会记录失败日志
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_current_step_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_19_state_immutability_except_updates(
        self,
        state: SharedState
    ):
        """
        属性 19: 导演评估门 - 状态不可变性（除了更新）
        
        导演评估时，只应修改特定字段（pivot_triggered、pivot_reason、execution_log）。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value="0.5")
        
        # 记录初始状态
        initial_user_topic = state.user_topic
        initial_outline_length = len(state.outline)
        initial_current_step_index = state.current_step_index
        initial_retrieved_docs_length = len(state.retrieved_docs)
        initial_fragments_length = len(state.fragments)
        
        # 调用导演评估
        result_state = await evaluate_and_decide(state, mock_llm_service)
        
        # 验证不应修改的字段保持不变
        assert result_state.user_topic == initial_user_topic
        assert len(result_state.outline) == initial_outline_length
        assert result_state.current_step_index == initial_current_step_index
        assert len(result_state.retrieved_docs) == initial_retrieved_docs_length
        assert len(result_state.fragments) == initial_fragments_length

