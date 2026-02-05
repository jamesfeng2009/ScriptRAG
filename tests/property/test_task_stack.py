"""Property-based tests for task stack operations

使用 Hypothesis 框架验证任务堆栈的正确性属性。

属性测试核心原则：
- 为每个属性生成 100+ 随机测试用例
- 验证属性在所有有效输入下都成立
- 发现边界情况和潜在 bug

验证的属性：
1. Push-Pop Round Trip：push 后 pop 应恢复原始状态
2. Depth Limit Enforcement：深度不应超过 3 层
3. Peek Idempotence：peek 不应修改堆栈
4. Task Metadata Completeness：所有元数据字段应正确保存
"""

import pytest
from hypothesis import given, settings, Verbosity
from hypothesis.strategies import integers, lists, dictionaries, text, one_of, none
from datetime import datetime

from src.domain.task_stack import (
    TaskStackManager,
    TaskContext,
    TaskStackDepthError,
    TaskStackEmptyError,
    InvalidTaskContextError,
)


def create_valid_task_context():
    """生成有效的任务上下文策略"""
    return dictionaries(
        keys=text(min_size=1, max_size=20),
        values=one_of(text(), integers(), none())
    ).map(lambda data: {
        "task_id": f"task_{data.get('id', len(data))}",
        "parent_id": None,
        "depth": 0,
        "creation_timestamp": datetime.now(),
        "task_data": data
    })


def create_empty_state():
    """创建初始空状态"""
    return {}


class TestTaskStackProperties:
    """任务堆栈属性测试类"""
    
    @given(
        task_contexts=lists(
            dictionaries(
                keys=text(min_size=1, max_size=20),
                values=one_of(text(), integers(), none())
            ).map(lambda data: {
                "task_id": f"task_{len(data)}_{id(data) % 10000}",
                "parent_id": None,
                "depth": 0,
                "creation_timestamp": datetime.now(),
                "task_data": data
            }),
            max_size=10
        )
    )
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_push_pop_round_trip(self, task_contexts):
        """
        Property 1: Push-Pop Round Trip
        
        对于任意任务堆栈和任意有效任务上下文，
        将任务上下文推入堆栈然后立即弹出，
        应该返回相同的任务上下文并恢复原始堆栈状态。
        
        验证：Requirements 1.1, 1.2
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        for context_dict in task_contexts:
            task_context = TaskContext(**context_dict)
            
            original_depth = manager.get_depth(state)
            
            pushed_state = manager.push(state.copy(), task_context)
            pushed_depth = manager.get_depth(pushed_state)
            
            assert pushed_depth == original_depth + 1
            
            popped_state, popped_context = manager.pop(pushed_state)
            popped_depth = manager.get_depth(popped_state)
            
            assert popped_depth == original_depth
            assert popped_context["task_id"] == task_context["task_id"]
            assert popped_context["parent_id"] == task_context["parent_id"]
            assert popped_context["depth"] == task_context["depth"]
    
    @given(
        contexts=lists(
            dictionaries(
                keys=text(min_size=1, max_size=20),
                values=one_of(text(), integers(), none())
            ).map(lambda data: {
                "task_id": f"task_{len(data)}_{id(data) % 10000}_{len(data)}",
                "parent_id": None,
                "depth": 0,
                "creation_timestamp": datetime.now(),
                "task_data": data
            }),
            max_size=20
        )
    )
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_depth_limit_enforcement(self, contexts):
        """
        Property 2: Depth Limit Enforcement
        
        对于任意推送操作序列，
        堆栈深度不应超过 3 层，
        任何超过此限制的推送尝试都应被拒绝并返回错误。
        
        验证：Requirements 1.3, 1.4
        """
        manager = TaskStackManager(max_depth=3)
        state = create_empty_state()
        
        pushed_count = 0
        for context_dict in contexts:
            if pushed_count >= 10:
                break
                
            task_context = TaskContext(**context_dict)
            
            try:
                state = manager.push(state, task_context)
                pushed_count += 1
                
                depth = manager.get_depth(state)
                assert depth <= 3
                
            except TaskStackDepthError:
                depth = manager.get_depth(state)
                assert depth == 3
    
    @given(
        task_contexts=lists(
            dictionaries(
                keys=text(min_size=1, max_size=20),
                values=one_of(text(), integers(), none())
            ).map(lambda data: {
                "task_id": f"task_{len(data)}_{id(data) % 10000}",
                "parent_id": None,
                "depth": 0,
                "creation_timestamp": datetime.now(),
                "task_data": data
            }),
            max_size=3
        )
    )
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_peek_idempotence(self, task_contexts):
        """
        Property 3: Peek Idempotence
        
        对于任意任务堆栈状态，
        多次调用 peek 应返回相同的任务上下文，
        且不修改堆栈。
        
        验证：Requirements 1.6
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        for context_dict in task_contexts:
            task_context = TaskContext(**context_dict)
            state = manager.push(state, task_context)
        
        if manager.is_empty(state):
            return
        
        peek1 = manager.peek(state)
        peek2 = manager.peek(state)
        depth_after = manager.get_depth(state)
        
        assert peek1 == peek2
        assert peek1 is not None
        assert peek1["task_id"] == peek2["task_id"]
    
    @given(
        contexts=lists(
            dictionaries(
                keys=text(min_size=1, max_size=20),
                values=one_of(text(), integers(), none())
            ).map(lambda data: {
                "task_id": f"task_{len(data)}_{id(data) % 10000}_{len(data)}",
                "parent_id": None if len(data) == 0 else f"parent_{len(data) - 1}",
                "depth": 0,
                "creation_timestamp": datetime.now(),
                "task_data": data
            }),
            max_size=3
        )
    )
    @settings(max_examples=100, verbosity=Verbosity.verbose)
    def test_metadata_completeness(self, contexts):
        """
        Property 4: Task Metadata Completeness
        
        对于任意堆栈上的任务上下文，
        所有必需的元数据字段（task_id, parent_id, depth, creation_timestamp）
        都应存在且有效。
        
        验证：Requirements 1.7
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        for context_dict in contexts:
            task_context = TaskContext(**context_dict)
            state = manager.push(state, task_context)
        
        depth = manager.get_depth(state)
        
        for i in range(depth):
            stack = state["task_stack"]["stack"]
            context = stack[i]
            
            assert "task_id" in context
            assert isinstance(context["task_id"], str)
            assert len(context["task_id"]) > 0
            
            assert "parent_id" in context
            assert context["parent_id"] is None or isinstance(context["parent_id"], str)
            
            assert "depth" in context
            assert isinstance(context["depth"], int)
            assert context["depth"] == i
            
            assert "creation_timestamp" in context
            assert isinstance(context["creation_timestamp"], datetime)
            
            assert "task_data" in context
            assert isinstance(context["task_data"], dict)


class TestTaskStackIntegration:
    """任务堆栈集成测试类"""
    
    def test_nested_subtask_creation(self):
        """
        测试嵌套子任务的创建和上下文传递
        
        场景：
        1. 创建主任务（深度 0）
        2. 创建子任务（深度 1，parent_id 指向主任务）
        3. 创建孙任务（深度 2，parent_id 指向子任务）
        4. 验证嵌套层次正确
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        main_task = manager.create_subtask_context(
            state,
            "main_task",
            {"title": "主任务", "description": "主要任务描述"}
        )
        state = manager.push(state, main_task)
        
        assert manager.get_depth(state) == 1
        assert manager.peek(state)["parent_id"] is None
        
        subtask = manager.create_subtask_context(
            state,
            "subtask_1",
            {"title": "子任务", "description": "子任务描述"}
        )
        state = manager.push(state, subtask)
        
        assert manager.get_depth(state) == 2
        assert manager.peek(state)["parent_id"] == "main_task"
        
        grandchild_task = manager.create_subtask_context(
            state,
            "grandchild_1",
            {"title": "孙任务", "description": "孙任务描述"}
        )
        state = manager.push(state, grandchild_task)
        
        assert manager.get_depth(state) == 3
        assert manager.peek(state)["parent_id"] == "subtask_1"
        
        parent_context = manager.get_parent_context(state)
        assert parent_context is not None
        assert parent_context["task_id"] == "subtask_1"
    
    def test_subtask_completion_restores_parent(self):
        """
        测试子任务完成后恢复父任务上下文
        
        场景：
        1. 创建并推送主任务
        2. 创建并推送子任务
        3. 完成子任务（pop）
        4. 验证当前任务恢复为主任务
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        main_context = manager.create_subtask_context(
            state, "main", {"title": "Main"}
        )
        state = manager.push(state, main_context)
        
        subtask_context = manager.create_subtask_context(
            state, "subtask", {"title": "Subtask"}
        )
        state = manager.push(state, subtask_context)
        
        assert manager.peek(state)["task_id"] == "subtask"
        
        state, popped = manager.pop(state)
        
        assert manager.peek(state)["task_id"] == "main"
        assert popped["task_id"] == "subtask"
    
    def test_max_depth_rejection(self):
        """
        测试超过最大深度时的拒绝行为
        
        场景：
        1. 尝试推送第 4 个任务（深度 3）
        2. 验证抛出 TaskStackDepthError
        """
        manager = TaskStackManager(max_depth=3)
        state = create_empty_state()
        
        for i in range(3):
            context = TaskContext(
                task_id=f"task_{i}",
                parent_id=f"task_{i-1}" if i > 0 else None,
                depth=i,
                creation_timestamp=datetime.now(),
                task_data={"level": i}
            )
            state = manager.push(state, context)
        
        assert manager.get_depth(state) == 3
        
        extra_context = TaskContext(
            task_id="task_3",
            parent_id="task_2",
            depth=3,
            creation_timestamp=datetime.now(),
            task_data={"level": 3}
        )
        
        with pytest.raises(TaskStackDepthError) as exc_info:
            manager.push(state, extra_context)
        
        assert exc_info.value.depth == 3
        assert exc_info.value.max_depth == 3
    
    def test_empty_stack_operations(self):
        """
        测试空堆栈的操作行为
        
        场景：
        1. peek 空堆栈应返回 None
        2. pop 空堆栈应抛出 TaskStackEmptyError
        3. get_depth 空堆栈应返回 0
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        assert manager.peek(state) is None
        assert manager.get_depth(state) == 0
        assert manager.is_empty(state)
        
        with pytest.raises(TaskStackEmptyError):
            manager.pop(state)
    
    def test_stack_clear(self):
        """
        测试堆栈清空操作
        
        场景：
        1. 推送多个任务
        2. 清空堆栈
        3. 验证堆栈为空
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
        
        state = manager.clear(state)
        
        assert manager.get_depth(state) == 0
        assert manager.is_empty(state)
    
    def test_invalid_task_context_rejection(self):
        """
        测试无效任务上下文的拒绝
        
        场景：
        1. 缺少 task_id
        2. 无效的 parent_id 类型
        3. 负数 depth
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        with pytest.raises(InvalidTaskContextError):
            manager.push(state, {
                "parent_id": None,
                "depth": 0,
                "task_data": {}
            })
        
        with pytest.raises(InvalidTaskContextError):
            manager.push(state, {
                "task_id": "test",
                "parent_id": 123,
                "depth": 0,
                "task_data": {}
            })
        
        with pytest.raises(InvalidTaskContextError):
            manager.push(state, {
                "task_id": "test",
                "parent_id": None,
                "depth": -1,
                "task_data": {}
            })
    
    def test_current_task_id_retrieval(self):
        """
        测试获取当前任务 ID
        
        场景：
        1. 空堆栈应返回 None
        2. 非空堆栈应返回栈顶任务的 task_id
        """
        manager = TaskStackManager()
        state = create_empty_state()
        
        assert manager.get_current_task_id(state) is None
        
        context = TaskContext(
            task_id="current_task",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={}
        )
        state = manager.push(state, context)
        
        assert manager.get_current_task_id(state) == "current_task"
