"""
任务堆栈管理 - 支持嵌套任务层次结构

功能：
1. TaskStack：管理任务层次结构的堆栈数据结构
2. TaskStackManager：提供 push/pop/peek/get_depth 等操作
3. 支持最大 3 层嵌套深度
4. 与 GlobalState 集成

使用示例：
    from src.domain.task_stack import TaskStackManager, TaskContext

    manager = TaskStackManager()
    state = create_initial_state()
    
    # 创建任务上下文
    context = TaskContext(
        task_id="task_1",
        parent_id=None,
        depth=0,
        creation_timestamp=datetime.now(),
        task_data={"title": "主任务"}
    )
    
    # 推入堆栈
    state = manager.push(state, context)
    
    # 查看栈顶（不修改）
    current = manager.peek(state)
    
    # 弹出堆栈
    state, popped = manager.pop(state)
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, TypedDict

logger = logging.getLogger(__name__)


class TaskStackDepthError(Exception):
    """任务堆栈深度超出限制"""
    def __init__(self, depth: int, max_depth: int):
        self.depth = depth
        self.max_depth = max_depth
        super().__init__(
            f"任务堆栈深度 {depth} 超出最大限制 {max_depth}"
        )


class TaskStackEmptyError(Exception):
    """任务堆栈为空时的操作错误"""
    def __init__(self, operation: str):
        self.operation = operation
        super().__init__(
            f"无法在空堆栈上执行 {operation} 操作"
        )


class InvalidTaskContextError(Exception):
    """无效的任务上下文错误"""
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"无效的任务上下文: {reason}")


class TaskContext(TypedDict):
    """
    任务上下文
    
    表示堆栈中的单个任务，包含任务标识、层次信息和元数据。
    
    字段：
        task_id：唯一任务标识符
        parent_id：父任务标识符（顶层任务为 None）
        depth：当前深度（0 起始）
        creation_timestamp：创建时间戳
        task_data：任务特定数据（可扩展）
    """
    task_id: str
    parent_id: Optional[str]
    depth: int
    creation_timestamp: datetime
    task_data: Dict[str, Any]


class TaskStack(TypedDict):
    """
    任务堆栈
    
    GlobalState 中存储的任务堆栈结构。
    
    字段：
        stack：任务上下文列表（栈顶在末尾）
        max_depth：最大允许深度（默认 3）
    """
    stack: List[TaskContext]
    max_depth: int


class TaskStackManager:
    """
    任务堆栈管理器
    
    提供任务堆栈的核心操作：push、pop、peek、get_depth。
    所有操作都返回新的 GlobalState（或修改后的状态），符合不可变设计。
    """
    
    DEFAULT_MAX_DEPTH: int = 3
    
    def __init__(self, max_depth: Optional[int] = None):
        """
        初始化任务堆栈管理器
        
        Args:
            max_depth：最大堆栈深度（默认 3）
        """
        self.max_depth = max_depth or self.DEFAULT_MAX_DEPTH
    
    def initialize_stack(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        初始化任务堆栈（如果不存在）
        
        Args:
            state：当前 GlobalState
            
        Returns：
            添加了 task_stack 字段的状态
        """
        if "task_stack" not in state:
            state["task_stack"] = TaskStack(
                stack=[],
                max_depth=self.max_depth
            )
        return state
    
    def push(
        self,
        state: Dict[str, Any],
        task_context: TaskContext
    ) -> Dict[str, Any]:
        """
        将任务上下文推入堆栈
        
        Args:
            state：当前 GlobalState
            task_context：任务上下文
            
        Returns：
            更新后的状态
            
        Raises：
            TaskStackDepthError：超出最大深度
            InvalidTaskContextError：无效的任务上下文
        """
        self._validate_task_context(task_context)
        
        state = self.initialize_stack(state)
        task_stack = state["task_stack"]
        current_depth = len(task_stack["stack"])
        
        if current_depth >= task_stack["max_depth"]:
            raise TaskStackDepthError(
                depth=current_depth,
                max_depth=task_stack["max_depth"]
            )
        
        new_context = TaskContext(
            task_id=task_context["task_id"],
            parent_id=task_context["parent_id"],
            depth=current_depth,
            creation_timestamp=task_context.get("creation_timestamp") or datetime.now(),
            task_data=task_context.get("task_data", {})
        )
        
        task_stack["stack"].append(new_context)
        
        logger.debug(
            f"任务推入堆栈: {new_context['task_id']} "
            f"(深度: {new_context['depth']})"
        )
        
        return state
    
    def pop(self, state: Dict[str, Any]) -> tuple[Dict[str, Any], Optional[TaskContext]]:
        """
        从堆栈顶部弹出任务上下文
        
        Args:
            state：当前 GlobalState
            
        Returns：
            (更新后的状态, 弹出的任务上下文)
            
        Raises：
            TaskStackEmptyError：堆栈为空
        """
        state = self.initialize_stack(state)
        task_stack = state["task_stack"]
        
        if not task_stack["stack"]:
            raise TaskStackEmptyError("pop")
        
        popped_context = task_stack["stack"].pop()
        
        logger.debug(
            f"任务弹出堆栈: {popped_context['task_id']} "
            f"(原深度: {popped_context['depth']})"
        )
        
        return state, popped_context
    
    def peek(self, state: Dict[str, Any]) -> Optional[TaskContext]:
        """
        查看堆栈顶部的任务上下文（不修改堆栈）
        
        Args:
            state：当前 GlobalState
            
        Returns：
            栈顶任务上下文，如果为空则返回 None
        """
        self.initialize_stack(state)
        task_stack = state["task_stack"]
        
        if not task_stack["stack"]:
            return None
        
        return task_stack["stack"][-1]
    
    def get_depth(self, state: Dict[str, Any]) -> int:
        """
        获取当前堆栈深度
        
        Args:
            state：当前 GlobalState
            
        Returns：
            当前堆栈深度（0 表示空堆栈）
        """
        self.initialize_stack(state)
        return len(state["task_stack"]["stack"])
    
    def is_empty(self, state: Dict[str, Any]) -> bool:
        """
        检查堆栈是否为空
        
        Args:
            state：当前 GlobalState
            
        Returns：
            如果堆栈为空返回 True
        """
        return self.get_depth(state) == 0
    
    def clear(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        清空堆栈
        
        Args:
            state：当前 GlobalState
            
        Returns：
            更新后的状态（空堆栈）
        """
        self.initialize_stack(state)
        state["task_stack"]["stack"] = []
        return state
    
    def get_current_task_id(self, state: Dict[str, Any]) -> Optional[str]:
        """
        获取当前任务 ID（栈顶任务的 task_id）
        
        Args:
            state：当前 GlobalState
            
        Returns：
            当前任务 ID，如果为空则返回 None
        """
        top_context = self.peek(state)
        if top_context:
            return top_context["task_id"]
        return None
    
    def get_parent_context(self, state: Dict[str, Any]) -> Optional[TaskContext]:
        """
        获取父任务上下文（栈顶的下一个任务）
        
        Args:
            state：当前 GlobalState
            
        Returns：
            父任务上下文，如果不存在则返回 None
        """
        self.initialize_stack(state)
        task_stack = state["task_stack"]
        
        stack = task_stack["stack"]
        if len(stack) >= 2:
            return stack[-2]
        return None
    
    def _validate_task_context(self, task_context: TaskContext) -> None:
        """
        验证任务上下文的有效性
        
        Args:
            task_context：待验证的任务上下文
            
        Raises：
            InvalidTaskContextError：验证失败
        """
        if not isinstance(task_context, dict):
            raise InvalidTaskContextError(
                "task_context 必须是字典类型"
            )
        
        if not task_context.get("task_id"):
            raise InvalidTaskContextError(
                "task_id 是必需字段"
            )
        
        if not isinstance(task_context.get("task_id"), str):
            raise InvalidTaskContextError(
                "task_id 必须是字符串类型"
            )
        
        if task_context.get("parent_id") is not None:
            if not isinstance(task_context["parent_id"], str):
                raise InvalidTaskContextError(
                    "parent_id 必须是字符串或 None"
                )
        
        depth = task_context.get("depth", 0)
        if not isinstance(depth, int):
            raise InvalidTaskContextError(
                "depth 必须是整数类型"
            )
        
        if depth < 0:
            raise InvalidTaskContextError(
                "depth 不能为负数"
            )
    
    def create_subtask_context(
        self,
        state: Dict[str, Any],
        subtask_id: str,
        subtask_data: Dict[str, Any]
    ) -> TaskContext:
        """
        创建子任务上下文
        
        自动从当前栈顶获取 parent_id 和 depth。
        
        Args:
            state：当前 GlobalState
            subtask_id：子任务 ID
            subtask_data：子任务数据
            
        Returns：
            新创建的子任务上下文
        """
        current_depth = self.get_depth(state)
        
        parent_context = self.peek(state)
        parent_id = parent_context["task_id"] if parent_context else None
        
        return TaskContext(
            task_id=subtask_id,
            parent_id=parent_id,
            depth=current_depth,
            creation_timestamp=datetime.now(),
            task_data=subtask_data
        )
