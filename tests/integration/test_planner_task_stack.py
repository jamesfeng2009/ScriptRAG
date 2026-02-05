"""Integration tests for Planner with Task Stack

集成测试：验证 Planner Agent 与 Task Stack 的集成

测试场景：
1. 创建嵌套子任务（1、2、3 层深度）
2. 完成子任务并恢复父任务上下文
3. Task Stack 禁用时的 Planner 行为
4. 向后兼容性验证

验证：Requirements 1.1, 1.2, 4.4
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from src.domain.agents.task_aware_planner import (
    TaskAwarePlanner,
    create_planner_with_task_stack,
)
from src.domain.task_stack import TaskStackManager


def create_test_state(
    user_topic: str = "测试主题",
    project_context: str = "测试上下文",
    outline: list = None,
    current_step_index: int = 0
) -> dict:
    """创建测试状态"""
    return {
        "user_topic": user_topic,
        "project_context": project_context,
        "outline": outline or [],
        "current_step_index": current_step_index,
        "fragments": [],
        "execution_log": [],
    }


class TestPlannerWithTaskStack:
    """Planner with Task Stack 集成测试类"""

    @pytest.mark.asyncio
    async def test_planner_with_task_stack_enabled(self):
        """
        测试启用 Task Stack 的 Planner
        
        场景：
        1. 创建启用了 Task Stack 的 Planner
        2. 执行规划
        3. 验证 Task Stack 被正确初始化
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        assert planner.use_task_stack is True
        assert planner.stack_manager is not None
        assert planner.stack_manager.max_depth == 3

    @pytest.mark.asyncio
    async def test_planner_without_task_stack(self):
        """
        测试禁用 Task Stack 的 Planner
        
        场景：
        1. 创建禁用 Task Stack 的 Planner
        2. 验证 Task Stack 相关操作为空操作
        """
        planner = create_planner_with_task_stack(
            use_task_stack=False
        )
        
        assert planner.use_task_stack is False
        assert planner.stack_manager is None
        
        state = create_test_state()
        
        result = planner.create_subtask(state, {"test": "data"})
        assert result == state
        
        result, popped = planner.complete_subtask(state)
        assert result == state
        assert popped is None
        
        assert planner.get_current_task(state) is None
        assert planner.get_task_depth(state) == 0

    @pytest.mark.asyncio
    async def test_nested_subtask_creation_single_level(self):
        """
        测试单层子任务创建
        
        场景：
        1. 规划生成大纲
        2. 创建一级子任务
        3. 验证 Task Stack 深度为 1
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state(
            outline=[{"step_id": 0, "description": "第一步"}]
        )
        
        state = planner.stack_manager.push(state, {
            "task_id": "main_task",
            "parent_id": None,
            "depth": 0,
            "creation_timestamp": datetime.now(),
            "task_data": {"level": "main"}
        })
        
        subtask_data = {
            "title": "子任务1",
            "parent_step": 0,
            "description": "这是一个子任务"
        }
        
        state = planner.create_subtask(state, subtask_data)
        
        assert planner.get_task_depth(state) == 2
        
        current = planner.get_current_task(state)
        assert current is not None
        assert current["parent_id"] == "main_task"

    @pytest.mark.asyncio
    async def test_nested_subtask_creation_two_levels(self):
        """
        测试两层嵌套子任务创建
        
        场景：
        1. 创建主任务（深度 0）
        2. 创建一级子任务（深度 1）
        3. 创建二级子任务（深度 2）
        4. 验证深度为 3
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        for level in range(3):
            subtask_data = {"level": level, "description": f"级别 {level}"}
            state = planner.create_subtask(state, subtask_data)
            
            expected_depth = level + 1
            assert planner.get_task_depth(state) == expected_depth
        
        current = planner.get_current_task(state)
        assert current is not None
        assert current["depth"] == 2

    @pytest.mark.asyncio
    async def test_nested_subtask_creation_three_levels(self):
        """
        测试三层嵌套子任务创建（最大深度）
        
        场景：
        1. 创建三层嵌套任务
        2. 尝试创建第四层
        3. 验证被拒绝
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        for level in range(3):
            subtask_data = {"level": level}
            state = planner.create_subtask(state, subtask_data)
        
        assert planner.get_task_depth(state) == 3
        
        with pytest.raises(Exception):
            planner.create_subtask(state, {"level": 3})

    @pytest.mark.asyncio
    async def test_subtask_completion_restores_parent(self):
        """
        测试子任务完成恢复父任务
        
        场景：
        1. 创建嵌套任务（主任务 -> 子任务1 -> 子任务2）
        2. 完成子任务2
        3. 验证当前任务恢复为子任务1
        4. 完成子任务1
        5. 验证当前任务恢复为主任务
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        state = planner.create_subtask(state, {"title": "主任务"})
        state = planner.create_subtask(state, {"title": "子任务1"})
        state = planner.create_subtask(state, {"title": "子任务2"})
        
        current = planner.get_current_task(state)
        assert current["task_data"]["title"] == "子任务2"
        
        state, popped = planner.complete_subtask(state)
        
        current = planner.get_current_task(state)
        assert current["task_data"]["title"] == "子任务1"
        assert popped["task_data"]["title"] == "子任务2"
        
        state, popped = planner.complete_subtask(state)
        
        current = planner.get_current_task(state)
        assert current["task_data"]["title"] == "主任务"
        assert popped["task_data"]["title"] == "子任务1"

    @pytest.mark.asyncio
    async def test_subtask_completion_empty_stack(self):
        """
        测试空堆栈时完成子任务
        
        场景：
        1. 在空堆栈上尝试完成子任务
        2. 验证返回原状态和 None
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        result_state, popped = planner.complete_subtask(state)
        
        assert result_state == state
        assert popped is None

    @pytest.mark.asyncio
    async def test_get_parent_task(self):
        """
        测试获取父任务
        
        场景：
        1. 创建嵌套任务
        2. 获取父任务
        3. 验证父任务信息正确
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        state = planner.create_subtask(state, {"title": "父任务"})
        state = planner.create_subtask(state, {"title": "当前任务"})
        
        parent = planner.get_parent_task(state)
        
        assert parent is not None
        assert parent["task_data"]["title"] == "父任务"

    @pytest.mark.asyncio
    async def test_has_parent_task(self):
        """
        测试检查是否存在父任务
        
        场景：
        1. 只有一个任务时
        2. 有多个任务时
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        assert planner.has_parent_task(state) is False
        
        state = planner.create_subtask(state, {"title": "主任务"})
        
        assert planner.has_parent_task(state) is False
        
        state = planner.create_subtask(state, {"title": "子任务"})
        
        assert planner.has_parent_task(state) is True

    @pytest.mark.asyncio
    async def test_get_current_task_id(self):
        """
        测试获取当前任务 ID
        
        场景：
        1. 空堆栈
        2. 非空堆栈
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        assert planner.get_current_task_id(state) is None
        
        state = planner.create_subtask(state, {"title": "任务1"})
        task_id = planner.get_current_task_id(state)
        
        assert task_id is not None
        assert task_id.startswith("subtask_")

    @pytest.mark.asyncio
    async def test_restore_task(self):
        """
        测试恢复任务数据
        
        场景：
        1. 修改状态数据
        2. 恢复任务数据
        3. 验证数据恢复
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        original_outline = [{"step_id": 0, "description": "原始步骤"}]
        modified_outline = [{"step_id": 0, "description": "修改步骤"}]
        
        state = create_test_state(outline=original_outline)
        state["outline"] = modified_outline
        
        task_data = {
            "outline": original_outline,
            "current_step_index": 0
        }
        
        state = planner.restore_task(state, task_data)
        
        assert state["outline"] == original_outline

    @pytest.mark.asyncio
    async def test_backward_compatibility(self):
        """
        测试向后兼容性
        
        场景：
        1. 使用原有 plan_outline 函数
        2. 验证行为与原 Planner 一致
        """
        from src.domain.agents.task_aware_planner import plan_outline
        
        mock_llm_service = Mock()
        mock_llm_service.chat_completion = AsyncMock(
            return_value="""步骤1: 介绍测试主题 | 关键词: 测试, 主题
步骤2: 详细说明 | 关键词: 说明, 详细
步骤3: 实践示例 | 关键词: 示例, 实践
步骤4: 总结 | 关键词: 总结
步骤5: 注意事项 | 关键词: 注意"""
        )
        
        state = create_test_state()
        
        result = await plan_outline(state, mock_llm_service)
        
        assert "outline" in result
        assert len(result["outline"]) >= 5
        assert result["current_step_index"] == 0

    @pytest.mark.asyncio
    async def test_task_stack_depth_tracking(self):
        """
        测试 Task Stack 深度跟踪
        
        场景：
        1. 推送和弹出操作
        2. 验证深度正确跟踪
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        depths = []
        for i in range(3):
            subtask_data = {"step": i}
            state = planner.create_subtask(state, subtask_data)
            depths.append(planner.get_task_depth(state))
        
        assert depths == [1, 2, 3]
        
        popped_contexts = []
        while planner.get_task_depth(state) > 0:
            state, popped = planner.complete_subtask(state)
            if popped:
                popped_contexts.append(popped)
        
        assert len(popped_contexts) == 3
        assert popped_contexts[0]["task_data"]["step"] == 2
        assert popped_contexts[1]["task_data"]["step"] == 1
        assert popped_contexts[2]["task_data"]["step"] == 0

    @pytest.mark.asyncio
    async def test_callback_hooks(self):
        """
        测试回调钩子
        
        场景：
        1. 设置 on_subtask_created 回调
        2. 设置 on_subtask_completed 回调
        3. 验证回调被调用
        """
        created_tasks = []
        completed_tasks = []
        
        def on_created(task, state):
            created_tasks.append(task["task_id"])
        
        def on_completed(task, state):
            completed_tasks.append(task["task_id"])
        
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3,
            on_subtask_created=on_created,
            on_subtask_completed=on_completed
        )
        
        state = create_test_state()
        state = planner.create_subtask(state, {"title": "任务1"})
        state = planner.create_subtask(state, {"title": "任务2"})
        
        assert len(created_tasks) == 2
        
        state, _ = planner.complete_subtask(state)
        state, _ = planner.complete_subtask(state)
        
        assert len(completed_tasks) == 2


class TestPlannerSubtaskFlow:
    """Planner 子任务流程测试类"""

    @pytest.mark.asyncio
    async def test_complex_nested_task_flow(self):
        """
        测试复杂嵌套任务流程
        
        完整流程：
        1. 规划主大纲
        2. 创建子任务处理复杂章节
        3. 在子任务中创建孙任务处理细节
        4. 完成孙任务
        5. 完成子任务
        6. 恢复主任务
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state(
            outline=[
                {"step_id": 0, "description": "引言"},
                {"step_id": 1, "description": "核心概念（需要详细展开）"},
                {"step_id": 2, "description": "实践"},
            ]
        )
        
        assert planner.get_task_depth(state) == 0
        
        state = planner.create_subtask(state, {
            "title": "处理核心概念章节",
            "parent_step": 1,
            "subtasks": ["概念A", "概念B", "概念C"]
        })
        
        assert planner.get_task_depth(state) == 1
        assert planner.has_parent_task(state) is False
        
        state = planner.create_subtask(state, {
            "title": "展开概念A",
            "parent_subtask": "处理核心概念章节"
        })
        
        assert planner.get_task_depth(state) == 2
        assert planner.has_parent_task(state) is True
        
        parent = planner.get_parent_task(state)
        assert parent["task_data"]["title"] == "处理核心概念章节"
        
        state, completed = planner.complete_subtask(state)
        assert completed["task_data"]["title"] == "展开概念A"
        
        current = planner.get_current_task(state)
        assert current["task_data"]["title"] == "处理核心概念章节"
        
        state, completed = planner.complete_subtask(state)
        assert completed["task_data"]["title"] == "处理核心概念章节"
        
        assert planner.get_task_depth(state) == 0

    @pytest.mark.asyncio
    async def test_max_depth_error_handling(self):
        """
        测试最大深度错误处理
        
        场景：
        1. 达到最大深度时创建子任务
        2. 验证抛出异常
        3. 验证状态未被修改
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=2
        )
        
        state = create_test_state()
        
        for i in range(2):
            state = planner.create_subtask(state, {"level": i})
        
        original_depth = planner.get_task_depth(state)
        
        with pytest.raises(Exception):
            planner.create_subtask(state, {"level": 2})
        
        assert planner.get_task_depth(state) == original_depth

    @pytest.mark.asyncio
    async def test_empty_task_data(self):
        """
        测试空任务数据
        
        场景：
        1. 创建无数据的子任务
        2. 验证操作成功
        """
        planner = create_planner_with_task_stack(
            use_task_stack=True,
            max_depth=3
        )
        
        state = create_test_state()
        
        state = planner.create_subtask(state, {})
        
        assert planner.get_task_depth(state) == 1
        
        current = planner.get_current_task(state)
        assert current is not None
        assert current["task_data"] == {}


class TestBackwardCompatibility:
    """向后兼容性测试类"""

    @pytest.mark.asyncio
    async def test_original_interface_unchanged(self):
        """
        测试原有接口保持不变
        
        验证：
        1. plan_outline 函数签名相同
        2. 返回类型兼容
        """
        from src.domain.agents.task_aware_planner import plan_outline
        import inspect
        
        sig = inspect.signature(plan_outline)
        params = list(sig.parameters.keys())
        
        assert "state" in params
        assert "llm_service" in params

    @pytest.mark.asyncio
    async def test_output_format_compatibility(self):
        """
        测试输出格式兼容
        
        场景：
        1. 使用原有接口
        2. 验证输出格式与原 Planner 一致
        """
        from src.domain.agents.task_aware_planner import plan_outline
        
        mock_llm_service = Mock()
        mock_llm_service.chat_completion = AsyncMock(
            return_value="""步骤1: 第一步 | 关键词: a, b
步骤2: 第二步 | 关键词: c, d"""
        )
        
        state = create_test_state()
        
        result = await plan_outline(state, mock_llm_service)
        
        assert "outline" in result
        assert isinstance(result["outline"], list)
        assert len(result["outline"]) >= 2
        
        first_step = result["outline"][0]
        assert "step_id" in first_step
        assert "description" in first_step
        assert "status" in first_step

    @pytest.mark.asyncio
    async def test_disabled_task_stack_preserves_state(self):
        """
        测试禁用 Task Stack 时保留状态
        
        场景：
        1. 使用 TaskStack 禁用模式
        2. 验证状态不会被意外修改
        """
        planner = create_planner_with_task_stack(
            use_task_stack=False
        )
        
        state = create_test_state(
            outline=[{"step_id": 0, "description": "测试步骤"}]
        )
        
        result = planner.create_subtask(state, {"test": "data"})
        
        assert result == state
        assert "task_stack" not in result or result.get("task_stack") is None

    @pytest.mark.asyncio
    async def test_fallback_outline_generation(self):
        """
        测试回退大纲生成
        
        场景：
        1. LLM 调用失败
        2. 验证生成回退大纲
        """
        mock_llm_service = Mock()
        mock_llm_service.chat_completion = AsyncMock(
            side_effect=Exception("LLM Error")
        )
        
        planner = create_planner_with_task_stack(
            use_task_stack=True
        )
        
        state = create_test_state(user_topic="失败主题")
        
        result = await planner.plan_outline(state, mock_llm_service)
        
        assert "outline" in result
        assert len(result["outline"]) == 5
        assert result["outline"][0]["description"] == "介绍主题：失败主题"
