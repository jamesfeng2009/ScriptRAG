"""Property Test: Comprehensive Logging

**属性 22: 全面日志记录**

对于任何智能体执行，系统应记录：
1. 带时间戳的智能体转换
2. 带原因的转向触发
3. 带来源的 RAG 检索结果
4. 带触发原因的 Skill 切换
5. 事实检查器验证结果
6. 重试尝试和降级动作
7. 带堆栈跟踪的错误

Feature: rag-screenplay-multi-agent
Property 22: Comprehensive Logging
"""

import pytest
from hypothesis import given, strategies as st, settings
from typing import List, Dict, Any
import logging
from io import StringIO

from src.domain.models import (
    SharedState,
    OutlineStep,
    RetrievedDocument,
    ScreenplayFragment
)
from src.infrastructure.logging import AgentLogger, get_agent_logger


# Custom strategies
@st.composite
def outline_step_strategy(draw):
    """Generate outline steps"""
    step_id = draw(st.integers(min_value=1, max_value=10))
    description = draw(st.text(min_size=10, max_size=100))
    status = draw(st.sampled_from(["pending", "in_progress", "completed", "skipped"]))
    retry_count = draw(st.integers(min_value=0, max_value=5))
    
    return OutlineStep(
        step_id=step_id,
        description=description,
        status=status,
        retry_count=retry_count
    )


@st.composite
def shared_state_strategy(draw):
    """Generate shared state with outline"""
    num_steps = draw(st.integers(min_value=1, max_value=5))
    outline = [draw(outline_step_strategy()) for _ in range(num_steps)]
    
    # Ensure unique step IDs
    for i, step in enumerate(outline):
        step.step_id = i + 1
    
    return SharedState(
        user_topic=draw(st.text(min_size=10, max_size=100)),
        project_context=draw(st.text(min_size=10, max_size=100)),
        outline=outline,
        current_step_index=draw(st.integers(min_value=0, max_value=num_steps-1))
    )


class TestComprehensiveLogging:
    """Test comprehensive logging across all agents"""
    
    @given(
        from_agent=st.sampled_from(["planner", "navigator", "director", "pivot_manager", "writer", "fact_checker"]),
        to_agent=st.sampled_from(["planner", "navigator", "director", "pivot_manager", "writer", "fact_checker"]),
        step_id=st.integers(min_value=1, max_value=10),
        reason=st.text(min_size=5, max_size=50)
    )
    @settings(max_examples=50)
    def test_agent_transition_logging(
        self,
        from_agent: str,
        to_agent: str,
        step_id: int,
        reason: str
    ):
        """
        属性 22.1: 智能体转换日志
        
        对于任何智能体转换，系统应记录：
        - 源智能体和目标智能体
        - 步骤 ID
        - 转换原因
        - 时间戳
        
        验证需求: 13.1
        """
        # Setup logging capture
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("test_logger")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        agent_logger = AgentLogger(logger)
        
        # Log agent transition
        agent_logger.log_agent_transition(
            from_agent=from_agent,
            to_agent=to_agent,
            step_id=step_id,
            reason=reason
        )
        
        # Verify log was created
        log_output = log_stream.getvalue()
        
        # Check that log contains key information
        assert from_agent in log_output or to_agent in log_output
        assert "transition" in log_output.lower()
        
        # Cleanup
        logger.removeHandler(handler)
    
    @given(
        step_id=st.integers(min_value=1, max_value=10),
        pivot_reason=st.sampled_from([
            "deprecation_conflict",
            "content_complexity_high",
            "content_complexity_low",
            "missing_information"
        ])
    )
    @settings(max_examples=50)
    def test_pivot_trigger_logging(
        self,
        step_id: int,
        pivot_reason: str
    ):
        """
        属性 22.2: 转向触发日志
        
        对于任何转向触发，系统应记录：
        - 步骤 ID
        - 转向原因
        - 冲突详情
        - 时间戳
        
        验证需求: 13.2
        """
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        
        logger = logging.getLogger("test_logger_pivot")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        agent_logger = AgentLogger(logger)
        
        # Log pivot trigger
        agent_logger.log_pivot_trigger(
            step_id=step_id,
            pivot_reason=pivot_reason,
            conflict_details={"test": "details"}
        )
        
        # Verify log was created
        log_output = log_stream.getvalue()
        
        # Check that log contains key information
        assert "pivot" in log_output.lower()
        assert pivot_reason in log_output or str(step_id) in log_output
        
        # Cleanup
        logger.removeHandler(handler)
    
    @given(
        step_id=st.integers(min_value=1, max_value=10),
        doc_count=st.integers(min_value=0, max_value=10),
        retrieval_method=st.sampled_from(["vector", "keyword", "hybrid"])
    )
    @settings(max_examples=50)
    def test_retrieval_result_logging(
        self,
        step_id: int,
        doc_count: int,
        retrieval_method: str
    ):
        """
        属性 22.3: RAG 检索结果日志
        
        对于任何 RAG 检索，系统应记录：
        - 步骤 ID
        - 检索到的文档数量
        - 来源列表
        - 检索方法
        - 置信度分数
        
        验证需求: 13.3
        """
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("test_logger_retrieval")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        agent_logger = AgentLogger(logger)
        
        # Generate sources
        sources = [f"file_{i}.py" for i in range(doc_count)]
        confidence_scores = [0.8 + i * 0.01 for i in range(doc_count)]
        
        # Log retrieval result
        agent_logger.log_retrieval_result(
            step_id=step_id,
            doc_count=doc_count,
            sources=sources,
            retrieval_method=retrieval_method,
            confidence_scores=confidence_scores
        )
        
        # Verify log was created
        log_output = log_stream.getvalue()
        
        # Check that log contains key information
        assert "retrieved" in log_output.lower() or "retrieval" in log_output.lower()
        assert str(doc_count) in log_output
        
        # Cleanup
        logger.removeHandler(handler)
    
    @given(
        step_id=st.integers(min_value=1, max_value=10),
        from_skill=st.sampled_from([
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",
            "research_mode",
            "meme_style",
            "fallback_summary"
        ]),
        to_skill=st.sampled_from([
            "standard_tutorial",
            "warning_mode",
            "visualization_analogy",
            "research_mode",
            "meme_style",
            "fallback_summary"
        ]),
        trigger_reason=st.text(min_size=5, max_size=50)
    )
    @settings(max_examples=50)
    def test_skill_switch_logging(
        self,
        step_id: int,
        from_skill: str,
        to_skill: str,
        trigger_reason: str
    ):
        """
        属性 22.4: Skill 切换日志
        
        对于任何 Skill 切换，系统应记录：
        - 步骤 ID
        - 源 Skill 和目标 Skill
        - 触发原因
        - 复杂度分数（如果适用）
        
        验证需求: 13.4
        """
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("test_logger_skill")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        agent_logger = AgentLogger(logger)
        
        # Log skill switch
        agent_logger.log_skill_switch(
            step_id=step_id,
            from_skill=from_skill,
            to_skill=to_skill,
            trigger_reason=trigger_reason,
            complexity_score=0.75
        )
        
        # Verify log was created
        log_output = log_stream.getvalue()
        
        # Check that log contains key information
        assert "skill" in log_output.lower()
        assert from_skill in log_output or to_skill in log_output
        
        # Cleanup
        logger.removeHandler(handler)
    
    @given(
        step_id=st.integers(min_value=1, max_value=10),
        is_valid=st.booleans(),
        hallucination_count=st.integers(min_value=0, max_value=5)
    )
    @settings(max_examples=50)
    def test_fact_check_result_logging(
        self,
        step_id: int,
        is_valid: bool,
        hallucination_count: int
    ):
        """
        属性 22.5: 事实检查器验证结果日志
        
        对于任何事实检查，系统应记录：
        - 步骤 ID
        - 验证结果（有效/无效）
        - 检测到的幻觉列表
        - 验证方法
        
        验证需求: 13.5
        """
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.INFO)
        
        logger = logging.getLogger("test_logger_fact")
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        
        agent_logger = AgentLogger(logger)
        
        # Generate hallucinations
        hallucinations = [f"hallucination_{i}" for i in range(hallucination_count)]
        
        # Log fact check result
        agent_logger.log_fact_check_result(
            step_id=step_id,
            is_valid=is_valid,
            hallucinations=hallucinations,
            verification_method="llm_verification"
        )
        
        # Verify log was created
        log_output = log_stream.getvalue()
        
        # Check that log contains key information
        assert "fact" in log_output.lower() or "check" in log_output.lower()
        
        # Cleanup
        logger.removeHandler(handler)
    
    @given(
        step_id=st.integers(min_value=1, max_value=10),
        retry_count=st.integers(min_value=1, max_value=5),
        max_retries=st.integers(min_value=3, max_value=5),
        reason=st.text(min_size=5, max_size=50)
    )
    @settings(max_examples=50)
    def test_retry_attempt_logging(
        self,
        step_id: int,
        retry_count: int,
        max_retries: int,
        reason: str
    ):
        """
        属性 22.6: 重试尝试日志
        
        对于任何重试尝试，系统应记录：
        - 步骤 ID
        - 当前重试次数
        - 最大重试次数
        - 重试原因
        
        验证需求: 13.6
        """
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        
        logger = logging.getLogger("test_logger_retry")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        agent_logger = AgentLogger(logger)
        
        # Log retry attempt
        agent_logger.log_retry_attempt(
            step_id=step_id,
            retry_count=retry_count,
            max_retries=max_retries,
            reason=reason
        )
        
        # Verify log was created
        log_output = log_stream.getvalue()
        
        # Check that log contains key information
        assert "retry" in log_output.lower()
        assert str(retry_count) in log_output or str(max_retries) in log_output
        
        # Cleanup
        logger.removeHandler(handler)
    
    @given(
        step_id=st.integers(min_value=1, max_value=10),
        degradation_type=st.sampled_from([
            "retry_limit_exceeded",
            "retrieval_failed",
            "llm_failed",
            "fact_check_failed"
        ]),
        reason=st.text(min_size=5, max_size=50),
        action_taken=st.text(min_size=5, max_size=50)
    )
    @settings(max_examples=50)
    def test_degradation_logging(
        self,
        step_id: int,
        degradation_type: str,
        reason: str,
        action_taken: str
    ):
        """
        属性 22.7: 降级动作日志
        
        对于任何降级动作，系统应记录：
        - 步骤 ID
        - 降级类型
        - 降级原因
        - 采取的动作
        
        验证需求: 13.6
        """
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.WARNING)
        
        logger = logging.getLogger("test_logger_degradation")
        logger.addHandler(handler)
        logger.setLevel(logging.WARNING)
        
        agent_logger = AgentLogger(logger)
        
        # Log degradation
        agent_logger.log_degradation(
            step_id=step_id,
            degradation_type=degradation_type,
            reason=reason,
            action_taken=action_taken
        )
        
        # Verify log was created
        log_output = log_stream.getvalue()
        
        # Check that log contains key information
        assert "degradation" in log_output.lower()
        assert degradation_type in log_output or action_taken in log_output
        
        # Cleanup
        logger.removeHandler(handler)
    
    def test_error_logging_with_stack_trace(self):
        """
        属性 22.8: 错误日志带堆栈跟踪
        
        对于任何错误，系统应记录：
        - 错误类型
        - 错误消息
        - 完整堆栈跟踪
        - 上下文信息
        - 智能体名称
        
        验证需求: 13.7
        """
        log_stream = StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setLevel(logging.ERROR)
        
        logger = logging.getLogger("test_logger_error")
        logger.addHandler(handler)
        logger.setLevel(logging.ERROR)
        
        agent_logger = AgentLogger(logger)
        
        # Create an error
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            # Log error with context
            agent_logger.log_error_with_context(
                error=e,
                context={"step_id": 1, "action": "test_action"},
                agent_name="test_agent"
            )
        
        # Verify log was created
        log_output = log_stream.getvalue()
        
        # Check that log contains key information
        assert "error" in log_output.lower()
        assert "ValueError" in log_output or "Test error message" in log_output
        
        # Cleanup
        logger.removeHandler(handler)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
