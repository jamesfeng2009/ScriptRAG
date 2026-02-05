"""Unit tests for task stack edge cases

测试任务堆栈的边界情况，确保在各种异常情况下正确处理。

测试场景：
1. 空堆栈的 pop 操作行为
2. 精确深度限制（3 层）
3. 无效任务上下文的拒绝
4. 空堆栈的 peek 操作

验证：Requirements 1.2, 1.3, 1.4, 1.6
"""

import pytest
from datetime import datetime
from unittest.mock import Mock

from src.domain.task_stack import (
    TaskStackManager,
    TaskContext,
    TaskStackDepthError,
    TaskStackEmptyError,
    InvalidTaskContextError,
)


def create_empty_state():
    """创建初始空状态"""
    return {}


class TestEmptyStackOperations:
    """空堆栈操作测试类"""

    def test_empty_stack_pop_raises_error(self):
        """
        测试空堆栈的 pop 操作应抛出 TaskStackEmptyError
        
        场景：直接对空堆栈执行 pop 操作
        预期：抛出 TaskStackEmptyError 异常
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        with pytest.raises(TaskStackEmptyError) as exc_info:
            manager.pop(state)
        
        assert exc_info.value.operation == "pop"

    def test_empty_stack_pop_with_initialized_stack(self):
        """
        测试已初始化空堆栈的 pop 操作
        
        场景：先初始化堆栈，然后执行 pop
        预期：抛出 TaskStackEmptyError 异常
        """
        manager = TaskStackManager()
        state = manager.initialize_stack(create_empty_state())
        
        with pytest.raises(TaskStackEmptyError):
            manager.pop(state)

    def test_empty_stack_peek_returns_none(self):
        """
        测试空堆栈的 peek 操作应返回 None
        
        场景：直接对空堆栈执行 peek 操作
        预期：返回 None，不修改堆栈
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        result = manager.peek(state)
        
        assert result is None

    def test_empty_stack_get_depth_returns_zero(self):
        """
        测试空堆栈的 get_depth 操作应返回 0
        
        场景：直接对空堆栈获取深度
        预期：返回 0
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        depth = manager.get_depth(state)
        
        assert depth == 0

    def test_empty_stack_is_empty_returns_true(self):
        """
        测试空堆栈的 is_empty 操作应返回 True
        
        场景：直接对空堆栈检查是否为空
        预期：返回 True
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        assert manager.is_empty(state) is True

    def test_empty_stack_get_current_task_id_returns_none(self):
        """
        测试空堆栈的 get_current_task_id 操作应返回 None
        
        场景：直接对空堆栈获取当前任务 ID
        预期：返回 None
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        result = manager.get_current_task_id(state)
        
        assert result is None

    def test_empty_stack_get_parent_context_returns_none(self):
        """
        测试空堆栈的 get_parent_context 操作应返回 None
        
        场景：直接对空堆栈获取父任务上下文
        预期：返回 None
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        result = manager.get_parent_context(state)
        
        assert result is None


class TestDepthLimitEdgeCases:
    """深度限制边界测试类"""

    def test_exact_depth_limit_rejection(self):
        """
        测试精确深度限制（3 层）
        
        场景：推送 3 个任务后尝试推送第 4 个
        预期：第 4 次推送应抛出 TaskStackDepthError
        """
        manager = TaskStackManager(max_depth=3)
        state = create_empty_state()
        
        for i in range(3):
            context = TaskContext(
                task_id=f"task_{i}",
                parent_id=f"task_{i-1}" if i > 0 else None,
                depth=0,
                creation_timestamp=datetime.now(),
                task_data={"level": i}
            )
            state = manager.push(state, context)
        
        assert manager.get_depth(state) == 3
        
        fourth_context = TaskContext(
            task_id="task_3",
            parent_id="task_2",
            depth=3,
            creation_timestamp=datetime.now(),
            task_data={"level": 3}
        )
        
        with pytest.raises(TaskStackDepthError) as exc_info:
            manager.push(state, fourth_context)
        
        assert exc_info.value.depth == 3
        assert exc_info.value.max_depth == 3

    def test_depth_limit_one_less_than_max(self):
        """
        测试深度限制边界（最大深度减一）
        
        场景：推送 2 个任务（最大深度为 3）
        预期：成功推送，深度为 2
        """
        manager = TaskStackManager(max_depth=3)
        state = create_empty_state()
        
        for i in range(2):
            context = TaskContext(
                task_id=f"task_{i}",
                parent_id=f"task_{i-1}" if i > 0 else None,
                depth=0,
                creation_timestamp=datetime.now(),
                task_data={}
            )
            state = manager.push(state, context)
        
        assert manager.get_depth(state) == 2

    def test_depth_after_pop(self):
        """
        测试 pop 操作后的深度减少
        
        场景：推送 3 个任务后弹出 1 个
        预期：深度减少到 2，可以继续推送
        """
        manager = TaskStackManager(max_depth=3)
        state = create_empty_state()
        
        for i in range(3):
            context = TaskContext(
                task_id=f"task_{i}",
                parent_id=None,
                depth=0,
                creation_timestamp=datetime.now(),
                task_data={}
            )
            state = manager.push(state, context)
        
        assert manager.get_depth(state) == 3
        
        state, popped = manager.pop(state)
        
        assert manager.get_depth(state) == 2
        
        new_context = TaskContext(
            task_id="new_task",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={}
        )
        
        state = manager.push(state, new_context)
        
        assert manager.get_depth(state) == 3

    def test_custom_max_depth(self):
        """
        测试自定义最大深度
        
        场景：设置最大深度为 2
        预期：推送 2 个任务后，第 3 个应被拒绝
        """
        manager = TaskStackManager(max_depth=2)
        state = create_empty_state()
        
        for i in range(2):
            context = TaskContext(
                task_id=f"task_{i}",
                parent_id=None,
                depth=0,
                creation_timestamp=datetime.now(),
                task_data={}
            )
            state = manager.push(state, context)
        
        with pytest.raises(TaskStackDepthError):
            manager.push(state, {
                "task_id": "task_2",
                "parent_id": "task_1",
                "depth": 2,
                "creation_timestamp": datetime.now(),
                "task_data": {}
            })

    def test_default_max_depth(self):
        """
        测试默认最大深度
        
        场景：使用默认配置
        预期：最大深度应为 3
        """
        manager = TaskStackManager()
        
        assert manager.max_depth == 3
        assert manager.DEFAULT_MAX_DEPTH == 3


class TestInvalidTaskContextRejection:
    """无效任务上下文拒绝测试类"""

    def test_missing_task_id(self):
        """
        测试缺少 task_id 的拒绝
        
        场景：推送缺少 task_id 的上下文
        预期：抛出 InvalidTaskContextError
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        with pytest.raises(InvalidTaskContextError) as exc_info:
            manager.push(state, {
                "parent_id": None,
                "depth": 0,
                "creation_timestamp": datetime.now(),
                "task_data": {}
            })
        
        assert "task_id" in exc_info.value.reason

    def test_empty_task_id(self):
        """
        测试空字符串 task_id 的拒绝
        
        场景：推送 task_id 为空字符串的上下文
        预期：抛出 InvalidTaskContextError
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        with pytest.raises(InvalidTaskContextError):
            manager.push(state, {
                "task_id": "",
                "parent_id": None,
                "depth": 0,
                "creation_timestamp": datetime.now(),
                "task_data": {}
            })

    def test_invalid_parent_id_type(self):
        """
        测试无效 parent_id 类型的拒绝
        
        场景：推送 parent_id 为整数（非字符串或 None）的上下文
        预期：抛出 InvalidTaskContextError
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        with pytest.raises(InvalidTaskContextError):
            manager.push(state, {
                "task_id": "test",
                "parent_id": 123,
                "depth": 0,
                "creation_timestamp": datetime.now(),
                "task_data": {}
            })

    def test_negative_depth(self):
        """
        测试负数 depth 的拒绝
        
        场景：推送 depth 为负数的上下文
        预期：抛出 InvalidTaskContextError
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        with pytest.raises(InvalidTaskContextError):
            manager.push(state, {
                "task_id": "test",
                "parent_id": None,
                "depth": -1,
                "creation_timestamp": datetime.now(),
                "task_data": {}
            })

    def test_non_dict_context(self):
        """
        测试非字典类型的拒绝
        
        场景：推送非字典类型的上下文
        预期：抛出 InvalidTaskContextError
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        with pytest.raises(InvalidTaskContextError):
            manager.push(state, "not a dict")

    def test_invalid_depth_type(self):
        """
        测试无效 depth 类型的拒绝
        
        场景：推送 depth 为字符串的上下文
        预期：抛出 InvalidTaskContextError
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        with pytest.raises(InvalidTaskContextError):
            manager.push(state, {
                "task_id": "test",
                "parent_id": None,
                "depth": "zero",
                "creation_timestamp": datetime.now(),
                "task_data": {}
            })


class TestPeekEdgeCases:
    """Peek 边界测试类"""

    def test_peek_empty_stack(self):
        """
        测试空堆栈的 peek 操作
        
        场景：对空堆栈执行 peek
        预期：返回 None
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        result = manager.peek(state)
        
        assert result is None

    def test_peek_single_element_stack(self):
        """
        测试单元素堆栈的 peek 操作
        
        场景：堆栈只有 1 个元素时执行 peek
        预期：返回该元素
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        context = TaskContext(
            task_id="single_task",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={}
        )
        state = manager.push(state, context)
        
        result = manager.peek(state)
        
        assert result is not None
        assert result["task_id"] == "single_task"

    def test_peek_does_not_modify_stack(self):
        """
        测试 peek 操作不修改堆栈
        
        场景：多次执行 peek
        预期：堆栈深度不变
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        for i in range(3):
            context = TaskContext(
                task_id=f"task_{i}",
                parent_id=None,
                depth=0,
                creation_timestamp=datetime.now(),
                task_data={}
            )
            state = manager.push(state, context)
        
        original_depth = manager.get_depth(state)
        
        for _ in range(5):
            peek_result = manager.peek(state)
            assert peek_result is not None
        
        final_depth = manager.get_depth(state)
        
        assert original_depth == final_depth

    def test_peek_after_pop(self):
        """
        测试 pop 后的 peek 操作
        
        场景：弹出元素后执行 peek
        预期：返回新的栈顶元素
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        for i in range(2):
            context = TaskContext(
                task_id=f"task_{i}",
                parent_id=None,
                depth=0,
                creation_timestamp=datetime.now(),
                task_data={}
            )
            state = manager.push(state, context)
        
        state, _ = manager.pop(state)
        
        peek_result = manager.peek(state)
        
        assert peek_result is not None
        assert peek_result["task_id"] == "task_0"


class TestStackInitialization:
    """堆栈初始化测试类"""

    def test_initialize_empty_stack(self):
        """
        测试初始化空堆栈
        
        场景：对空状态初始化堆栈
        预期：创建 task_stack 字段
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        result = manager.initialize_stack(state)
        
        assert "task_stack" in result
        assert result["task_stack"]["max_depth"] == 3
        assert result["task_stack"]["stack"] == []

    def test_initialize_preserves_existing_stack(self):
        """
        测试初始化保留已存在的堆栈
        
        场景：对已有堆栈的状态初始化
        预期：保留原有堆栈内容
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        context = TaskContext(
            task_id="existing_task",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={}
        )
        state = manager.push(state, context)
        
        result = manager.initialize_stack(state)
        
        assert len(result["task_stack"]["stack"]) == 1
        assert result["task_stack"]["stack"][0]["task_id"] == "existing_task"


class TestStackClear:
    """堆栈清空测试类"""

    def test_clear_non_empty_stack(self):
        """
        测试清空非空堆栈
        
        场景：推送多个任务后清空堆栈
        预期：堆栈变为空
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        for i in range(3):
            context = TaskContext(
                task_id=f"task_{i}",
                parent_id=None,
                depth=0,
                creation_timestamp=datetime.now(),
                task_data={}
            )
            state = manager.push(state, context)
        
        assert manager.get_depth(state) == 3
        
        result = manager.clear(state)
        
        assert manager.get_depth(result) == 0
        assert manager.is_empty(result)

    def test_clear_empty_stack(self):
        """
        测试清空空堆栈
        
        场景：对空堆栈执行清空
        预期：无错误，堆栈仍为空
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        result = manager.clear(state)
        
        assert manager.is_empty(result)


class TestMultipleOperations:
    """多操作组合测试类"""

    def test_push_pop_push_sequence(self):
        """
        测试 push-pop-push 序列
        
        场景：推送、弹出、推送的序列操作
        预期：堆栈状态正确
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        state = manager.push(state, TaskContext(
            task_id="task_1",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={}
        ))
        
        assert manager.get_current_task_id(state) == "task_1"
        
        state, popped = manager.pop(state)
        assert popped["task_id"] == "task_1"
        
        state = manager.push(state, TaskContext(
            task_id="task_2",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={}
        ))
        
        assert manager.get_current_task_id(state) == "task_2"

    def test_get_parent_context_with_single_element(self):
        """
        测试单元素堆栈获取父上下文
        
        场景：堆栈只有 1 个元素时获取父上下文
        预期：返回 None
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        state = manager.push(state, TaskContext(
            task_id="only_task",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={}
        ))
        
        parent = manager.get_parent_context(state)
        
        assert parent is None

    def test_get_parent_context_with_two_elements(self):
        """
        测试双元素堆栈获取父上下文
        
        场景：堆栈有 2 个元素时获取父上下文
        预期：返回栈顶的下一个元素
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        state = manager.push(state, TaskContext(
            task_id="task_1",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={}
        ))
        
        state = manager.push(state, TaskContext(
            task_id="task_2",
            parent_id="task_1",
            depth=1,
            creation_timestamp=datetime.now(),
            task_data={}
        ))
        
        parent = manager.get_parent_context(state)
        
        assert parent is not None
        assert parent["task_id"] == "task_1"
