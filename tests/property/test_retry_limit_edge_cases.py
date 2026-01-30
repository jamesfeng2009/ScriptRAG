"""
边界情况测试：最大重试边界

边界情况 2: 最大重试边界
测试重试计数在边界值（max_retries - 1, max_retries, max_retries + 1）
时的行为，确保边界条件正确处理。

Feature: rag-screenplay-multi-agent
Edge Case 2: 最大重试边界
"""

import pytest
from hypothesis import given, strategies as st, settings

from src.domain.models import SharedState, OutlineStep, ScreenplayFragment
from src.domain.agents.retry_protection import check_retry_limit


class TestRetryLimitEdgeCases:
    """测试最大重试边界的边界情况"""
    
    def test_edge_case_2_exactly_at_limit(self):
        """
        边界情况 2: retry_count == max_retries
        
        当 retry_count 恰好等于 max_retries 时，应触发跳过。
        """
        # 创建状态，retry_count == max_retries
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=3  # 等于 max_retries
                )
            ],
            current_step_index=0,
            max_retries=3
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤被跳过
        assert updated_state.outline[0].status == "skipped"
        assert len(updated_state.fragments) == 1
        assert "[SKIPPED]" in updated_state.fragments[0].content
    
    def test_edge_case_2_one_below_limit(self):
        """
        边界情况 2: retry_count == max_retries - 1
        
        当 retry_count 比 max_retries 少 1 时，不应触发跳过。
        """
        # 创建状态，retry_count == max_retries - 1
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=2  # max_retries - 1
                )
            ],
            current_step_index=0,
            max_retries=3
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤未被跳过
        assert updated_state.outline[0].status == "in_progress"
        assert len(updated_state.fragments) == 0
        assert updated_state.current_step_index == 0
    
    def test_edge_case_2_one_above_limit(self):
        """
        边界情况 2: retry_count == max_retries + 1
        
        当 retry_count 比 max_retries 多 1 时，应触发跳过。
        """
        # 创建状态，retry_count == max_retries + 1
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=4  # max_retries + 1
                )
            ],
            current_step_index=0,
            max_retries=3
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤被跳过
        assert updated_state.outline[0].status == "skipped"
        assert len(updated_state.fragments) == 1
        assert "[SKIPPED]" in updated_state.fragments[0].content
    
    def test_edge_case_2_zero_retries(self):
        """
        边界情况 2: retry_count == 0
        
        当 retry_count 为 0 时，不应触发跳过。
        """
        # 创建状态，retry_count == 0
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="pending",
                    retry_count=0
                )
            ],
            current_step_index=0,
            max_retries=3
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤未被跳过
        assert updated_state.outline[0].status == "pending"
        assert len(updated_state.fragments) == 0
        assert updated_state.current_step_index == 0
    
    def test_edge_case_2_max_retries_one(self):
        """
        边界情况 2: max_retries == 1
        
        当 max_retries 为最小值 1 时，retry_count >= 1 应触发跳过。
        """
        # 创建状态，max_retries == 1, retry_count == 1
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=1
                )
            ],
            current_step_index=0,
            max_retries=1
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤被跳过
        assert updated_state.outline[0].status == "skipped"
        assert len(updated_state.fragments) == 1
    
    def test_edge_case_2_max_retries_ten(self):
        """
        边界情况 2: max_retries == 10
        
        当 max_retries 为最大值 10 时，retry_count >= 10 应触发跳过。
        """
        # 创建状态，max_retries == 10, retry_count == 10
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=10
                )
            ],
            current_step_index=0,
            max_retries=10
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤被跳过
        assert updated_state.outline[0].status == "skipped"
        assert len(updated_state.fragments) == 1
    
    def test_edge_case_2_boundary_transition(self):
        """
        边界情况 2: 从未超过到超过的转换
        
        测试 retry_count 从 max_retries - 1 增加到 max_retries 的转换。
        """
        # 创建状态，retry_count == max_retries - 1
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=2  # max_retries - 1
                )
            ],
            current_step_index=0,
            max_retries=3
        )
        
        # 第一次检查：不应跳过
        updated_state = check_retry_limit(state)
        assert updated_state.outline[0].status == "in_progress"
        assert len(updated_state.fragments) == 0
        
        # 增加 retry_count 到 max_retries
        updated_state.outline[0].retry_count = 3
        
        # 第二次检查：应跳过
        final_state = check_retry_limit(updated_state)
        assert final_state.outline[0].status == "skipped"
        assert len(final_state.fragments) == 1
    
    def test_edge_case_2_multiple_steps_at_boundary(self):
        """
        边界情况 2: 多个步骤在边界值
        
        测试多个步骤的 retry_count 在不同边界值时的行为。
        """
        # 创建状态，多个步骤在不同边界值
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Step at limit - 1",
                    status="in_progress",
                    retry_count=2  # max_retries - 1
                ),
                OutlineStep(
                    step_id=1,
                    description="Step at limit",
                    status="in_progress",
                    retry_count=3  # max_retries
                ),
                OutlineStep(
                    step_id=2,
                    description="Step above limit",
                    status="in_progress",
                    retry_count=4  # max_retries + 1
                )
            ],
            current_step_index=0,
            max_retries=3
        )
        
        # 检查第一个步骤（未超过限制）
        state = check_retry_limit(state)
        assert state.outline[0].status == "in_progress"
        assert state.current_step_index == 0
        
        # 移动到第二个步骤并检查（恰好在限制）
        state.current_step_index = 1
        state = check_retry_limit(state)
        assert state.outline[1].status == "skipped"
        assert len(state.fragments) == 1
        
        # 检查第三个步骤（超过限制）
        state = check_retry_limit(state)
        assert state.outline[2].status == "skipped"
        assert len(state.fragments) == 2
    
    def test_edge_case_2_very_high_retry_count(self):
        """
        边界情况 2: 非常高的 retry_count
        
        测试 retry_count 远超 max_retries 时的行为。
        """
        # 创建状态，retry_count 远超 max_retries（但在模型限制内）
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=10  # 远超 max_retries，但在模型限制内
                )
            ],
            current_step_index=0,
            max_retries=3
        )
        
        # 执行检查（不应崩溃）
        updated_state = check_retry_limit(state)
        
        # 验证步骤被跳过
        assert updated_state.outline[0].status == "skipped"
        assert len(updated_state.fragments) == 1
        
        # 验证日志记录了正确的 retry_count
        log_entry = updated_state.execution_log[-1]
        assert log_entry["details"]["retry_count"] == 10
    
    def test_edge_case_2_boundary_with_pivot_triggered(self):
        """
        边界情况 2: 边界值时有 pivot 触发
        
        测试在边界值时，如果有 pivot 触发，应被清除。
        """
        # 创建状态，retry_count == max_retries，且有 pivot 触发
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=3  # 等于 max_retries
                )
            ],
            current_step_index=0,
            max_retries=3,
            pivot_triggered=True,
            pivot_reason="test_reason"
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证步骤被跳过
        assert updated_state.outline[0].status == "skipped"
        
        # 验证 pivot 触发被清除
        assert updated_state.pivot_triggered is False
        assert updated_state.pivot_reason is None
    
    def test_edge_case_2_boundary_last_step_in_outline(self):
        """
        边界情况 2: 最后一步在边界值
        
        测试最后一步的 retry_count 恰好等于 max_retries 时的行为。
        """
        # 创建状态，最后一步 retry_count == max_retries
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="First step",
                    status="completed",
                    retry_count=0
                ),
                OutlineStep(
                    step_id=1,
                    description="Last step",
                    status="in_progress",
                    retry_count=3  # 等于 max_retries
                )
            ],
            current_step_index=1,
            max_retries=3
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证最后一步被跳过
        assert updated_state.outline[1].status == "skipped"
        assert len(updated_state.fragments) == 1
        
        # 验证步骤索引未改变（因为已经是最后一步）
        assert updated_state.current_step_index == 1
    
    def test_edge_case_2_sequential_boundary_checks(self):
        """
        边界情况 2: 连续的边界检查
        
        测试对同一步骤连续多次调用 check_retry_limit 的行为。
        """
        # 创建状态，retry_count == max_retries
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=3
                ),
                OutlineStep(
                    step_id=1,
                    description="Next step",
                    status="pending",
                    retry_count=0
                )
            ],
            current_step_index=0,
            max_retries=3
        )
        
        # 第一次检查：应跳过并前进
        state = check_retry_limit(state)
        assert state.outline[0].status == "skipped"
        assert state.current_step_index == 1
        assert len(state.fragments) == 1
        
        # 第二次检查：当前步骤已改变，不应再次跳过
        state = check_retry_limit(state)
        assert state.outline[1].status == "pending"
        assert state.current_step_index == 1
        assert len(state.fragments) == 1  # 没有新增片段
    
    @pytest.mark.parametrize("max_retries,retry_count,should_skip", [
        (1, 0, False),   # 未达到限制
        (1, 1, True),    # 恰好达到限制
        (1, 2, True),    # 超过限制
        (3, 2, False),   # 未达到限制
        (3, 3, True),    # 恰好达到限制
        (3, 4, True),    # 超过限制
        (5, 4, False),   # 未达到限制
        (5, 5, True),    # 恰好达到限制
        (5, 6, True),    # 超过限制
        (10, 9, False),  # 未达到限制
        (10, 10, True),  # 恰好达到限制
    ])
    def test_edge_case_2_parametrized_boundaries(
        self,
        max_retries: int,
        retry_count: int,
        should_skip: bool
    ):
        """
        边界情况 2: 参数化边界测试
        
        使用参数化测试验证各种 max_retries 和 retry_count 组合。
        """
        # 创建状态
        state = SharedState(
            user_topic="Test topic",
            outline=[
                OutlineStep(
                    step_id=0,
                    description="Test step",
                    status="in_progress",
                    retry_count=retry_count
                )
            ],
            current_step_index=0,
            max_retries=max_retries
        )
        
        # 执行检查
        updated_state = check_retry_limit(state)
        
        # 验证结果
        if should_skip:
            assert updated_state.outline[0].status == "skipped"
            assert len(updated_state.fragments) == 1
        else:
            assert updated_state.outline[0].status == "in_progress"
            assert len(updated_state.fragments) == 0
