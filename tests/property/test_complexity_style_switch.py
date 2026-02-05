"""
属性测试：基于复杂度的风格切换

属性 11: 基于复杂度的风格切换
当检索内容极其枯燥或复杂时（复杂度 > 0.7），导演应触发风格切换，
推荐使用 visualization_analogy Skill。

Feature: rag-screenplay-multi-agent
Property 11: 基于复杂度的风格切换
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import AsyncMock, MagicMock, patch

from src.domain.models import (
    SharedState,
    OutlineStep,
    RetrievedDocument
)
from src.domain.agents.director import assess_complexity, evaluate_and_decide
from src.domain.skills import SKILLS


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
def shared_state_strategy(draw):
    """生成有效的 SharedState"""
    valid_skills = list(SKILLS.keys())
    
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


class TestComplexityStyleSwitch:
    """测试基于复杂度的风格切换属性"""
    
    @pytest.mark.asyncio
    @given(
        step=outline_step_strategy(),
        docs=st.lists(
            retrieved_document_strategy(),
            min_size=1,
            max_size=5
        ),
        complexity_score=st.floats(min_value=0.71, max_value=1.0)
    )
    @settings(max_examples=100, deadline=None)
    async def test_property_11_high_complexity_triggers_switch(
        self,
        step: OutlineStep,
        docs: list,
        complexity_score: float
    ):
        """
        属性 11: 基于复杂度的风格切换 - 高复杂度触发切换
        
        当复杂度分数 > 0.7 时，应推荐切换到 visualization_analogy。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value=str(complexity_score))
        
        # 调用复杂度评估
        result_score = await assess_complexity(step, docs, mock_llm_service)
        
        # 验证复杂度分数在预期范围内
        assert 0.0 <= result_score <= 1.0
        
        # 如果复杂度 > 0.7，验证会触发风格切换
        if result_score > 0.7:
            # 创建共享状态，确保文档没有冲突标记
            clean_docs = [
                RetrievedDocument(
                    content=doc.content,
                    source=doc.source,
                    confidence=doc.confidence,
                    metadata={
                        'has_deprecated': False,
                        'has_fixme': False,
                        'has_todo': False,
                        'has_security': False
                    }
                )
                for doc in docs
            ]
            
            state = SharedState(
                user_topic="Test topic",
                outline=[step],
                current_step_index=0,
                retrieved_docs=clean_docs,
                current_skill="standard_tutorial"
            )
            
            # 使用模拟的 LLM 服务调用导演决策
            result_state = await evaluate_and_decide(state, mock_llm_service)
            
            # 验证触发了转向
            assert result_state.pivot_triggered is True
            assert result_state.pivot_reason is not None
            assert "complexity" in result_state.pivot_reason.lower()
    
    @pytest.mark.asyncio
    @given(
        step=outline_step_strategy(),
        docs=st.lists(
            retrieved_document_strategy(),
            min_size=1,
            max_size=5
        ),
        complexity_score=st.floats(min_value=0.0, max_value=0.7)
    )
    @settings(max_examples=100, deadline=None)
    async def test_property_11_low_complexity_no_switch(
        self,
        step: OutlineStep,
        docs: list,
        complexity_score: float
    ):
        """
        属性 11: 基于复杂度的风格切换 - 低复杂度不触发切换
        
        当复杂度分数 <= 0.7 时，不应因复杂度触发风格切换。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value=str(complexity_score))
        
        # 调用复杂度评估
        result_score = await assess_complexity(step, docs, mock_llm_service)
        
        # 验证复杂度分数在预期范围内
        assert 0.0 <= result_score <= 1.0
        
        # 如果复杂度 <= 0.7，验证不会因复杂度触发风格切换
        if result_score <= 0.7:
            # 创建共享状态（确保没有废弃标记）
            clean_docs = [
                RetrievedDocument(
                    content=doc.content,
                    source=doc.source,
                    confidence=doc.confidence,
                    metadata={
                        'has_deprecated': False,
                        'has_fixme': False,
                        'has_todo': False,
                        'has_security': False
                    }
                )
                for doc in docs
            ]
            
            state = SharedState(
                user_topic="Test topic",
                outline=[step],
                current_step_index=0,
                retrieved_docs=clean_docs,
                current_skill="standard_tutorial"
            )
            
            # 使用模拟的 LLM 服务调用导演决策
            result_state = await evaluate_and_decide(state, mock_llm_service)
            
            # 验证没有因复杂度触发转向
            if result_state.pivot_triggered:
                assert "complexity" not in result_state.pivot_reason.lower()
    
    @pytest.mark.asyncio
    @given(
        step=outline_step_strategy(),
        docs=st.lists(
            retrieved_document_strategy(),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    async def test_property_11_complexity_score_range(
        self,
        step: OutlineStep,
        docs: list
    ):
        """
        属性 11: 基于复杂度的风格切换 - 复杂度分数范围正确
        
        复杂度评估应返回 0.0 到 1.0 之间的分数。
        """
        # 创建模拟的 LLM 服务，返回随机分数
        import random
        random_score = random.uniform(0.0, 1.0)
        
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value=str(random_score))
        
        # 调用复杂度评估
        result_score = await assess_complexity(step, docs, mock_llm_service)
        
        # 验证分数在有效范围内
        assert 0.0 <= result_score <= 1.0
    
    @pytest.mark.asyncio
    @given(step=outline_step_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_11_empty_docs_zero_complexity(
        self,
        step: OutlineStep
    ):
        """
        属性 11: 基于复杂度的风格切换 - 空文档列表复杂度为 0
        
        当检索文档列表为空时，复杂度应为 0.0。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        
        # 调用复杂度评估
        result_score = await assess_complexity(step, [], mock_llm_service)
        
        # 验证复杂度为 0
        assert result_score == 0.0
    
    @pytest.mark.asyncio
    @given(
        step=outline_step_strategy(),
        docs=st.lists(
            retrieved_document_strategy(),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    async def test_property_11_llm_failure_fallback(
        self,
        step: OutlineStep,
        docs: list
    ):
        """
        属性 11: 基于复杂度的风格切换 - LLM 失败时回退到启发式
        
        当 LLM 调用失败时，应回退到启发式复杂度评估。
        """
        # 创建模拟的 LLM 服务，模拟失败
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(side_effect=Exception("LLM failed"))
        
        # 调用复杂度评估
        result_score = await assess_complexity(step, docs, mock_llm_service)
        
        # 验证仍然返回有效的复杂度分数
        assert 0.0 <= result_score <= 1.0
    
    @pytest.mark.asyncio
    @given(
        step=outline_step_strategy(),
        docs=st.lists(
            retrieved_document_strategy(),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=100, deadline=None)
    async def test_property_11_invalid_llm_response_fallback(
        self,
        step: OutlineStep,
        docs: list
    ):
        """
        属性 11: 基于复杂度的风格切换 - 无效 LLM 响应时回退
        
        当 LLM 返回无效响应时，应回退到启发式复杂度评估。
        """
        # 创建模拟的 LLM 服务，返回无效响应
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value="invalid response")
        
        # 调用复杂度评估
        result_score = await assess_complexity(step, docs, mock_llm_service)
        
        # 验证仍然返回有效的复杂度分数
        assert 0.0 <= result_score <= 1.0
    
    @pytest.mark.asyncio
    @given(state=shared_state_strategy())
    @settings(max_examples=100, deadline=None)
    async def test_property_11_director_decision_logging(
        self,
        state: SharedState
    ):
        """
        属性 11: 基于复杂度的风格切换 - 导演决策日志记录
        
        导演做出决策时，应记录日志条目。
        """
        # 创建模拟的 LLM 服务
        mock_llm_service = MagicMock()
        mock_llm_service.chat_completion = AsyncMock(return_value="0.5")
        
        # 记录初始日志数量
        initial_log_count = len(state.execution_log)
        
        # 调用导演决策
        result_state = await evaluate_and_decide(state, mock_llm_service)
        
        # 验证添加了日志条目
        assert len(result_state.execution_log) > initial_log_count
        
        # 验证日志条目包含导演信息
        latest_log = result_state.execution_log[-1]
        assert latest_log["agent_name"] == "director"
        assert "action" in latest_log

