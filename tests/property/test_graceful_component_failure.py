"""Property Test: Graceful Component Failure

Feature: rag-screenplay-multi-agent
Property 27: 优雅组件失败

属性描述:
对于任何单个组件失败（RAG 检索、事实检查器、摘要器、编剧、超时），
系统应记录失败，应用适当的回退行为，并继续执行而不崩溃。

测试策略:
- 使用 Hypothesis 生成各种失败场景
- 模拟不同组件的失败
- 验证系统不会崩溃
- 验证回退行为正确应用
- 验证失败被正确记录
"""

import pytest
from hypothesis import given, strategies as st, settings, assume
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from src.domain.models import SharedState, OutlineStep, RetrievedDocument
from src.infrastructure.error_handler import (
    ErrorHandler,
    PostgreSQLConnectionError,
    EmbeddingGenerationError,
    FactCheckerError,
    SummarizerError,
    WriterError,
    TimeoutError as CustomTimeoutError
)
from src.services.llm.service import LLMService
from src.services.retrieval_service import RetrievalService
from src.services.summarization_service import SummarizationService
from src.domain.agents.fact_checker import verify_fragment_node
from src.domain.agents.writer import generate_fragment


# Hypothesis strategies
@st.composite
def shared_state_strategy(draw):
    """生成 SharedState 对象"""
    num_steps = draw(st.integers(min_value=1, max_value=5))
    
    outline = [
        OutlineStep(
            step_id=i+1,
            description=f"Step {i+1}",
            status="pending",
            retry_count=0
        )
        for i in range(num_steps)
    ]
    
    return SharedState(
        user_topic="Test topic",
        project_context="Test context",
        outline=outline,
        current_step_index=0,
        retrieved_docs=[],
        fragments=[],
        current_skill="standard_tutorial",
        global_tone="professional",
        pivot_triggered=False,
        pivot_reason=None,
        max_retries=3,
        awaiting_user_input=False,
        user_input_prompt=None,
        execution_log=[]
    )


@st.composite
def failure_type_strategy(draw):
    """生成失败类型"""
    return draw(st.sampled_from([
        "postgresql_connection",
        "embedding_generation",
        "fact_checker",
        "summarizer",
        "writer",
        "timeout"
    ]))


class TestGracefulComponentFailure:
    """测试优雅组件失败"""
    
    @given(
        state=shared_state_strategy(),
        failure_type=failure_type_strategy()
    )
    @settings(max_examples=100, deadline=None)
    @pytest.mark.asyncio
    async def test_component_failure_does_not_crash(
        self,
        state: SharedState,
        failure_type: str
    ):
        """
        属性 27: 优雅组件失败
        
        验证需求: 18.1, 18.2, 18.3, 18.4, 18.5
        
        对于任何单个组件失败，系统应：
        1. 不崩溃（不抛出未捕获的异常）
        2. 记录失败
        3. 应用适当的回退行为
        4. 继续执行
        """
        # 根据失败类型模拟不同的失败场景
        if failure_type == "postgresql_connection":
            # 测试 PostgreSQL 连接失败
            await self._test_postgresql_failure(state)
        
        elif failure_type == "embedding_generation":
            # 测试嵌入生成失败
            await self._test_embedding_failure(state)
        
        elif failure_type == "fact_checker":
            # 测试事实检查器失败
            await self._test_fact_checker_failure(state)
        
        elif failure_type == "summarizer":
            # 测试摘要器失败
            await self._test_summarizer_failure(state)
        
        elif failure_type == "writer":
            # 测试编剧失败
            await self._test_writer_failure(state)
        
        elif failure_type == "timeout":
            # 测试超时
            await self._test_timeout_failure(state)
    
    async def _test_postgresql_failure(self, state: SharedState):
        """测试 PostgreSQL 连接失败（需求 18.1）"""
        # 模拟 PostgreSQL 连接失败
        async def failing_retrieval(*args, **kwargs):
            raise PostgreSQLConnectionError("Connection failed")
        
        # 使用错误处理器
        result = await ErrorHandler.handle_retrieval_error(
            failing_retrieval,
            fallback_to_keyword_only=True
        )
        
        # 验证：应返回空结果而不是崩溃
        assert result == []
        
        # 验证：系统应继续执行
        assert True  # 如果到达这里，说明没有崩溃
    
    async def _test_embedding_failure(self, state: SharedState):
        """测试嵌入生成失败（需求 18.1）"""
        # 模拟嵌入生成失败
        async def failing_embedding(*args, **kwargs):
            raise EmbeddingGenerationError("Embedding generation failed")
        
        # 使用错误处理器
        result = await ErrorHandler.handle_retrieval_error(
            failing_embedding,
            fallback_to_keyword_only=True
        )
        
        # 验证：应返回空结果而不是崩溃
        assert result == []
        
        # 验证：系统应继续执行
        assert True
    
    async def _test_fact_checker_failure(self, state: SharedState):
        """测试事实检查器失败（需求 18.2）"""
        # 模拟事实检查器失败
        async def failing_fact_checker(*args, **kwargs):
            raise FactCheckerError("Fact checker failed")
        
        # 使用错误处理器
        result = await ErrorHandler.handle_component_failure(
            failing_fact_checker,
            component_name="fact_checker",
            fallback_value=state,
            log_level="warning"
        )
        
        # 验证：应返回回退值（原始 state）
        assert result == state
        
        # 验证：系统应继续执行
        assert True
    
    async def _test_summarizer_failure(self, state: SharedState):
        """测试摘要器失败（需求 18.3）"""
        # 模拟摘要器失败
        async def failing_summarizer(*args, **kwargs):
            raise SummarizerError("Summarizer failed")
        
        # 使用错误处理器（带截断回退）
        content = "A" * 20000  # 大内容
        result = await ErrorHandler.handle_component_failure(
            failing_summarizer,
            component_name="summarizer",
            fallback_value=None,
            log_level="warning",
            content=content,
            max_length=10000
        )
        
        # 验证：应返回截断的内容
        assert result is not None
        assert len(result) <= 10003  # 10000 + "..."
        
        # 验证：系统应继续执行
        assert True
    
    async def _test_writer_failure(self, state: SharedState):
        """测试编剧失败（需求 18.4）"""
        # 模拟编剧失败
        async def failing_writer(*args, **kwargs):
            raise WriterError("Writer failed")
        
        # 使用错误处理器（带 Skill 回退）
        result = await ErrorHandler.handle_component_failure(
            failing_writer,
            component_name="writer",
            fallback_value=state,
            log_level="warning",
            skill="standard_tutorial"
        )
        
        # 验证：应返回回退值
        assert result == state
        
        # 验证：系统应继续执行
        assert True
    
    async def _test_timeout_failure(self, state: SharedState):
        """测试超时（需求 18.5）"""
        # 模拟超时
        async def slow_function(*args, **kwargs):
            await asyncio.sleep(2.0)
            return "result"
        
        # 使用超时保护
        with pytest.raises(CustomTimeoutError):
            await ErrorHandler.with_timeout(
                slow_function,
                timeout_seconds=0.1
            )
        
        # 验证：超时应抛出 TimeoutError
        # 在实际使用中，这个异常会被捕获并处理
        assert True
    
    @given(state=shared_state_strategy())
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_multiple_failures_do_not_crash(self, state: SharedState):
        """
        测试多个连续失败不会导致系统崩溃
        
        验证需求: 18.1, 18.2, 18.3, 18.4
        """
        # 模拟多个组件连续失败
        failures = [
            PostgreSQLConnectionError("DB failed"),
            FactCheckerError("Fact checker failed"),
            SummarizerError("Summarizer failed"),
            WriterError("Writer failed")
        ]
        
        for failure in failures:
            async def failing_component(*args, **kwargs):
                raise failure
            
            # 使用错误处理器
            result = await ErrorHandler.handle_component_failure(
                failing_component,
                component_name="test_component",
                fallback_value=state,
                log_level="warning"
            )
            
            # 验证：每次失败都应返回回退值
            assert result == state
        
        # 验证：系统在多次失败后仍然运行
        assert True
    
    @given(state=shared_state_strategy())
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_failure_logging(self, state: SharedState):
        """
        测试失败被正确记录
        
        验证需求: 13.7
        """
        # 模拟组件失败
        async def failing_component(*args, **kwargs):
            raise FactCheckerError("Test failure")
        
        # 使用错误处理器
        with patch('src.infrastructure.error_handler.logger') as mock_logger:
            result = await ErrorHandler.handle_component_failure(
                failing_component,
                component_name="test_component",
                fallback_value=state,
                log_level="warning"
            )
            
            # 验证：失败应被记录
            assert mock_logger.warning.called or mock_logger.error.called
        
        # 验证：系统应继续执行
        assert result == state
    
    @given(
        state=shared_state_strategy(),
        retry_count=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=50, deadline=None)
    @pytest.mark.asyncio
    async def test_exponential_backoff_retry(
        self,
        state: SharedState,
        retry_count: int
    ):
        """
        测试指数退避重试机制
        
        验证需求: 15.9
        """
        call_count = 0
        
        async def failing_then_succeeding(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < retry_count:
                from src.infrastructure.error_handler import RateLimitError
                raise RateLimitError("Rate limit")
            return "success"
        
        # 使用指数退避重试
        result = await ErrorHandler.retry_with_exponential_backoff(
            failing_then_succeeding,
            max_retries=retry_count,
            initial_delay=0.01,  # 使用小延迟以加快测试
            max_delay=0.1,
            exponential_base=2.0
        )
        
        # 验证：最终应成功
        assert result == "success"
        
        # 验证：应重试正确的次数
        assert call_count == retry_count


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
