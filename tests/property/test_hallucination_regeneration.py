"""
属性测试：幻觉重新生成

属性 15: 幻觉重新生成
当事实检查器检测到幻觉时，系统应移除无效片段并触发重新生成。

Feature: rag-screenplay-multi-agent
Property 15: 幻觉重新生成
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import AsyncMock, MagicMock
import asyncio

from src.domain.models import (
    SharedState,
    OutlineStep,
    ScreenplayFragment,
    RetrievedDocument
)
from src.domain.agents.fact_checker import verify_fragment_node


# 自定义策略生成器
@st.composite
def shared_state_with_invalid_fragment_strategy(draw):
    """生成包含无效片段的 SharedState"""
    # 创建基本状态
    user_topic = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    
    # 创建大纲步骤
    step = OutlineStep(
        step_id=0,
        description=draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip())),
        status="in_progress",
        retry_count=draw(st.integers(min_value=0, max_value=2))
    )
    
    # 创建包含幻觉的片段（引用不存在的函数）
    fake_func = "nonexistent_function"
    fragment = ScreenplayFragment(
        step_id=0,
        content=f"使用 `{fake_func}()` 函数来处理数据。",
        skill_used="standard_tutorial",
        sources=["test.py"]
    )
    
    # 创建不包含该函数的检索文档
    doc = RetrievedDocument(
        content="这是一个示例文档，不包含 nonexistent_function。",
        source="test.py",
        confidence=0.9,
        metadata={}
    )
    
    state = SharedState(
        user_topic=user_topic,
        outline=[step],
        current_step_index=0,
        retrieved_docs=[doc],
        fragments=[fragment],
        current_skill="standard_tutorial"
    )
    
    return state


@st.composite
def shared_state_with_valid_fragment_strategy(draw):
    """生成包含有效片段的 SharedState"""
    # 创建基本状态
    user_topic = draw(st.text(min_size=1, max_size=100).filter(lambda x: x.strip()))
    
    # 创建大纲步骤
    step = OutlineStep(
        step_id=0,
        description=draw(st.text(min_size=1, max_size=200).filter(lambda x: x.strip())),
        status="in_progress",
        retry_count=draw(st.integers(min_value=0, max_value=2))
    )
    
    # 创建有效的片段（引用存在的函数）
    real_func = "real_function"
    fragment = ScreenplayFragment(
        step_id=0,
        content=f"使用 `{real_func}()` 函数来处理数据。",
        skill_used="standard_tutorial",
        sources=["test.py"]
    )
    
    # 创建包含该函数的检索文档
    doc = RetrievedDocument(
        content=f"这是一个示例文档。\ndef {real_func}():\n    pass\n使用 {real_func} 函数。",
        source="test.py",
        confidence=0.9,
        metadata={}
    )
    
    state = SharedState(
        user_topic=user_topic,
        outline=[step],
        current_step_index=0,
        retrieved_docs=[doc],
        fragments=[fragment],
        current_skill="standard_tutorial"
    )
    
    return state


class TestHallucinationRegeneration:
    """测试幻觉重新生成属性"""
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_invalid_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_15_invalid_fragment_removed(self, state):
        """
        属性 15: 幻觉重新生成 - 无效片段被移除
        
        当检测到幻觉时，无效片段应该被移除。
        """
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="INVALID\n- 函数 'nonexistent_function' 未在源文档中找到")
        
        # 记录初始片段数量
        initial_fragment_count = len(state.fragments)
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证片段被移除
        assert len(updated_state.fragments) < initial_fragment_count
        assert len(updated_state.fragments) == 0  # 应该移除了唯一的片段
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_invalid_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_15_retry_count_incremented(self, state):
        """
        属性 15: 幻觉重新生成 - 重试计数器增加
        
        当检测到幻觉时，当前步骤的重试计数器应该增加。
        """
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="INVALID\n- 函数 'nonexistent_function' 未在源文档中找到")
        
        # 记录初始重试计数
        initial_retry_count = state.outline[0].retry_count
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证重试计数增加
        assert updated_state.outline[0].retry_count == initial_retry_count + 1
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_invalid_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_15_fact_check_passed_flag_set(self, state):
        """
        属性 15: 幻觉重新生成 - fact_check_passed 标志设置
        
        当检测到幻觉时，fact_check_passed 标志应该设置为 False。
        """
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="INVALID\n- 函数 'nonexistent_function' 未在源文档中找到")
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证标志设置为 False
        assert updated_state.fact_check_passed is False
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_valid_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_15_valid_fragment_kept(self, state):
        """
        属性 15: 幻觉重新生成 - 有效片段保留
        
        当片段有效时，片段应该被保留。
        """
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="VALID")
        
        # 记录初始片段数量
        initial_fragment_count = len(state.fragments)
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证片段保留
        assert len(updated_state.fragments) == initial_fragment_count
        assert len(updated_state.fragments) > 0
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_valid_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_15_valid_fragment_no_retry_increment(self, state):
        """
        属性 15: 幻觉重新生成 - 有效片段不增加重试计数
        
        当片段有效时，重试计数器不应该增加。
        """
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="VALID")
        
        # 记录初始重试计数
        initial_retry_count = state.outline[0].retry_count
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证重试计数不变
        assert updated_state.outline[0].retry_count == initial_retry_count
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_valid_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_15_valid_fragment_fact_check_passed(self, state):
        """
        属性 15: 幻觉重新生成 - 有效片段通过检查
        
        当片段有效时，fact_check_passed 标志应该设置为 True。
        """
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="VALID")
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证标志设置为 True
        assert updated_state.fact_check_passed is True
    
    @pytest.mark.asyncio
    async def test_property_15_empty_fragments_handling(self):
        """
        属性 15: 幻觉重新生成 - 空片段列表处理
        
        当没有片段时，事实检查器应该优雅处理。
        """
        # 创建没有片段的状态
        state = SharedState(
            user_topic="测试主题",
            outline=[OutlineStep(step_id=0, description="测试步骤", status="pending")],
            current_step_index=0,
            fragments=[]  # 空片段列表
        )
        
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证没有崩溃，状态保持不变
        assert len(updated_state.fragments) == 0
        assert updated_state is not None
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_invalid_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_15_step_status_reset(self, state):
        """
        属性 15: 幻觉重新生成 - 步骤状态重置
        
        当检测到幻觉时，步骤状态应该重置为 in_progress 以便重新生成。
        """
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="INVALID\n- 函数 'nonexistent_function' 未在源文档中找到")
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证步骤状态重置
        assert updated_state.outline[0].status == "in_progress"
    
    @pytest.mark.asyncio
    @given(state=shared_state_with_invalid_fragment_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_15_logging(self, state):
        """
        属性 15: 幻觉重新生成 - 日志记录
        
        当检测到幻觉时，应该记录验证失败的日志。
        """
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="INVALID\n- 函数 'nonexistent_function' 未在源文档中找到")
        
        # 记录初始日志数量
        initial_log_count = len(state.execution_log)
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证添加了日志
        assert len(updated_state.execution_log) > initial_log_count
        
        # 验证日志包含正确的信息
        latest_log = updated_state.execution_log[-1]
        assert latest_log["agent_name"] == "fact_checker"
        assert latest_log["action"] == "verification_failed"
        assert "hallucinations" in latest_log["details"]
    
    @pytest.mark.asyncio
    async def test_property_15_llm_error_graceful_degradation(self):
        """
        属性 15: 幻觉重新生成 - LLM 错误优雅降级
        
        当 LLM 调用失败时，系统应该优雅降级（假设片段有效）。
        """
        # 创建包含片段的状态
        state = SharedState(
            user_topic="测试主题",
            outline=[OutlineStep(step_id=0, description="测试步骤", status="in_progress")],
            current_step_index=0,
            retrieved_docs=[
                RetrievedDocument(
                    content="测试文档内容",
                    source="test.py",
                    confidence=0.9,
                    metadata={}
                )
            ],
            fragments=[
                ScreenplayFragment(
                    step_id=0,
                    content="测试内容",
                    skill_used="standard_tutorial",
                    sources=[]
                )
            ]
        )
        
        # 创建会抛出异常的模拟 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(side_effect=Exception("LLM 调用失败"))
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证优雅降级：片段保留，标志设置为 True
        assert len(updated_state.fragments) == 1
        assert updated_state.fact_check_passed is True
        
        # 验证没有崩溃，系统继续运行
        assert updated_state is not None
    
    @pytest.mark.asyncio
    @given(
        state=shared_state_with_invalid_fragment_strategy(),
        num_retries=st.integers(min_value=0, max_value=2)
    )
    @settings(max_examples=100, deadline=None)
    async def test_property_15_retry_count_accumulation(self, state, num_retries):
        """
        属性 15: 幻觉重新生成 - 重试计数累积
        
        多次检测到幻觉时，重试计数应该累积增加。
        """
        # 设置初始重试计数
        state.outline[0].retry_count = num_retries
        
        # 创建模拟的 LLM 服务
        llm_service = AsyncMock()
        llm_service.chat_completion = AsyncMock(return_value="INVALID\n- 幻觉检测")
        
        # 调用事实检查器节点
        updated_state = await verify_fragment_node(state, llm_service)
        
        # 验证重试计数增加
        assert updated_state.outline[0].retry_count == num_retries + 1
