"""
Planner Agent with Task Stack Support - Enhanced Task Decomposition

功能：
1. 生成初始剧本大纲
2. 支持递归任务分解（嵌套子任务）
3. 使用 TaskStack 管理任务层次结构
4. 可选启用 Task Stack 支持

使用示例：
```python
from src.domain.agents.task_aware_planner import create_planner_with_task_stack

planner = create_planner_with_task_stack(
    use_task_stack=True,
    max_depth=3
)

result_state = await planner.plan_outline(state, llm_service)
```
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from ..task_stack import TaskStackManager, TaskContext, TaskStackDepthError
from ...services.llm.service import LLMService

logger = logging.getLogger(__name__)


def create_planner_with_task_stack(
    use_task_stack: bool = False,
    max_depth: int = 3,
    on_subtask_created: Optional[Callable] = None,
    on_subtask_completed: Optional[Callable] = None
) -> 'TaskAwarePlanner':
    """
    创建支持 Task Stack 的 Planner 实例
    
    Args:
        use_task_stack：是否启用 Task Stack
        max_depth：最大嵌套深度
        on_subtask_created：子任务创建时的回调
        on_subtask_completed：子任务完成时的回调
        
    Returns:
        TaskAwarePlanner 实例
    """
    return TaskAwarePlanner(
        use_task_stack=use_task_stack,
        max_depth=max_depth,
        on_subtask_created=on_subtask_created,
        on_subtask_completed=on_subtask_completed
    )


class TaskAwarePlanner:
    """
    支持 Task Stack 的 Planner Agent
    
    功能：
    1. 生成初始剧本大纲
    2. 递归分解复杂任务为子任务
    3. 使用 TaskStack 管理任务上下文
    4. 可选模式：启用/禁用 Task Stack
    
    向后兼容性：
    - use_task_stack=False 时行为与原 Planner 一致
    - Task Stack 操作失败时静默降级
    """
    
    def __init__(
        self,
        use_task_stack: bool = False,
        max_depth: int = 3,
        on_subtask_created: Optional[Callable] = None,
        on_subtask_completed: Optional[Callable] = None
    ):
        """
        初始化 TaskAwarePlanner
        
        Args:
            use_task_stack：是否启用 Task Stack
            max_depth：最大嵌套深度
            on_subtask_created：子任务创建回调
            on_subtask_completed：子任务完成回调
        """
        self.use_task_stack = use_task_stack
        self.max_depth = max_depth
        self.stack_manager = TaskStackManager(max_depth=max_depth) if use_task_stack else None
        self.on_subtask_created = on_subtask_created
        self.on_subtask_completed = on_subtask_completed
        
        logger.info(
            f"TaskAwarePlanner initialized: "
            f"use_task_stack={use_task_stack}, max_depth={max_depth}"
        )
    
    async def plan_outline(
        self,
        state: Dict[str, Any],
        llm_service: LLMService
    ) -> Dict[str, Any]:
        """
        规划器主函数 - 生成初始剧本大纲
        
        功能：
        1. 分析用户主题和项目上下文
        2. 生成包含 5-10 步的结构化大纲
        3. 如果启用 TaskStack，创建初始任务上下文
        
        Args:
            state: GlobalState 字典
            llm_service: LLM 服务实例
            
        Returns:
            更新后的 GlobalState
        """
        logger.info("Planner agent started: Generating outline")
        
        state = self._add_log_entry(
            state,
            "planner",
            "start_planning",
            {
                "user_topic": state.get("user_topic"),
                "project_context": state.get("project_context"),
                "use_task_stack": self.use_task_stack
            }
        )
        
        try:
            prompt = self._build_planning_prompt(
                state.get("user_topic", ""),
                state.get("project_context", "")
            )
            
            messages = [
                {
                    "role": "system",
                    "content": "你是一个专业的剧本规划师，擅长将复杂主题分解为清晰的步骤。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            response = await llm_service.chat_completion(
                messages=messages,
                task_type="high_performance",
                temperature=0.7,
                max_tokens=2000
            )
            
            outline_steps = self._parse_outline_response(response)
            
            if len(outline_steps) < 5:
                logger.warning(f"Generated outline has only {len(outline_steps)} steps, padding to 5")
                outline_steps = self._pad_outline(outline_steps, 5)
            elif len(outline_steps) > 10:
                logger.warning(f"Generated outline has {len(outline_steps)} steps, truncating to 10")
                outline_steps = outline_steps[:10]
            
            state["outline"] = outline_steps
            state["current_step_index"] = 0
            
            if self.use_task_stack and self.stack_manager:
                main_task_context = self._create_main_task_context(
                    state.get("user_topic", ""),
                    state.get("project_context", ""),
                    outline_steps
                )
                state = self.stack_manager.push(state, main_task_context)
            
            state = self._add_log_entry(
                state,
                "planner",
                "outline_generated",
                {
                    "num_steps": len(outline_steps),
                    "task_stack_depth": self.stack_manager.get_depth(state) if self.stack_manager else 0
                }
            )
            
            logger.info(f"Planner agent completed: Generated {len(outline_steps)} steps")
            
            return state
            
        except Exception as e:
            logger.error(f"Planner agent failed: {str(e)}")
            
            state = self._add_log_entry(
                state,
                "planner",
                "planning_failed",
                {"error": str(e)}
            )
            
            state["outline"] = self._create_fallback_outline(state.get("user_topic", ""))
            state["current_step_index"] = 0
            
            logger.warning("Using fallback outline due to planning failure")
            
            return state
    
    def create_subtask(
        self,
        state: Dict[str, Any],
        subtask_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        创建子任务并推入堆栈
        
        功能：
        1. 创建子任务上下文
        2. 推入 TaskStack
        3. 触发 on_subtask_created 回调
        
        Args:
            state: 当前 GlobalState
            subtask_data：子任务数据
            
        Returns:
            更新后的 GlobalState
            
        Raises:
            TaskStackDepthError：超出最大深度
        """
        if not self.use_task_stack or not self.stack_manager:
            logger.debug("Task Stack disabled, skipping subtask creation")
            return state
        
        try:
            subtask_context = self.stack_manager.create_subtask_context(
                state,
                f"subtask_{uuid.uuid4().hex[:8]}",
                subtask_data
            )
            
            state = self.stack_manager.push(state, subtask_context)
            
            logger.info(
                f"Subtask created: {subtask_context['task_id']} "
                f"(depth: {subtask_context['depth']})"
            )
            
            if self.on_subtask_created:
                self.on_subtask_created(subtask_context, state)
            
            return state
            
        except TaskStackDepthError:
            logger.warning(
                f"Cannot create subtask: max depth {self.max_depth} reached"
            )
            raise
    
    def complete_subtask(
        self,
        state: Dict[str, Any]
    ) -> tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
        """
        完成当前子任务并恢复父任务上下文
        
        功能：
        1. 弹出 TaskStack
        2. 恢复父任务为当前任务
        3. 触发 on_subtask_completed 回调
        
        Args:
            state: 当前 GlobalState
            
        Returns:
            (更新后的 GlobalState, 弹出的子任务上下文)
        """
        if not self.use_task_stack or not self.stack_manager:
            logger.debug("Task Stack disabled, skipping subtask completion")
            return state, None
        
        if self.stack_manager.is_empty(state):
            logger.warning("Cannot complete subtask: task stack is empty")
            return state, None
        
        state, popped_context = self.stack_manager.pop(state)
        
        logger.info(
            f"Subtask completed: {popped_context['task_id']} "
            f"(restored to parent)"
        )
        
        if self.on_subtask_completed:
            self.on_subtask_completed(popped_context, state)
        
        return state, popped_context
    
    def get_current_task(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        获取当前任务上下文
        
        Args:
            state: 当前 GlobalState
            
        Returns:
            当前任务上下文，如果堆栈为空则返回 None
        """
        if not self.use_task_stack or not self.stack_manager:
            return None
        
        return self.stack_manager.peek(state)
    
    def get_current_task_id(self, state: Dict[str, Any]) -> Optional[str]:
        """
        获取当前任务 ID
        
        Args:
            state: 当前 GlobalState
            
        Returns:
            当前任务 ID
        """
        if not self.use_task_stack or not self.stack_manager:
            return None
        
        return self.stack_manager.get_current_task_id(state)
    
    def get_task_depth(self, state: Dict[str, Any]) -> int:
        """
        获取当前任务深度
        
        Args:
            state: 当前 GlobalState
            
        Returns:
            当前深度
        """
        if not self.use_task_stack or not self.stack_manager:
            return 0
        
        return self.stack_manager.get_depth(state)
    
    def has_parent_task(self, state: Dict[str, Any]) -> bool:
        """
        检查是否存在父任务
        
        Args:
            state: 当前 GlobalState
            
        Returns:
            如果有父任务返回 True
        """
        if not self.use_task_stack or not self.stack_manager:
            return False
        
        return self.stack_manager.get_depth(state) > 1
    
    def get_parent_task(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        获取父任务上下文
        
        Args:
            state: 当前 GlobalState
            
        Returns:
            父任务上下文
        """
        if not self.use_task_stack or not self.stack_manager:
            return None
        
        return self.stack_manager.get_parent_context(state)
    
    def restore_task(
        self,
        state: Dict[str, Any],
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        恢复任务数据到状态
        
        功能：
        将任务数据恢复到 GlobalState，
        用于子任务完成后恢复父任务状态。
        
        Args:
            state: 当前 GlobalState
            task_data：任务数据
            
        Returns:
            更新后的 GlobalState
        """
        if "outline" in task_data:
            state["outline"] = task_data["outline"]
        if "current_step_index" in task_data:
            state["current_step_index"] = task_data["current_step_index"]
        
        return state
    
    def _create_main_task_context(self, state: Dict[str, Any]) -> TaskContext:
        """
        创建主任务上下文
        
        Args:
            state: GlobalState
            
        Returns:
            主任务上下文
        """
        return TaskContext(
            task_id=f"main_{uuid.uuid4().hex[:8]}",
            parent_id=None,
            depth=0,
            creation_timestamp=datetime.now(),
            task_data={
                "user_topic": state.get("user_topic", ""),
                "project_context": state.get("project_context", ""),
                "outline": state.get("outline", []),
                "current_step_index": state.get("current_step_index", 0)
            }
        )
    
    def _add_log_entry(
        self,
        state: Dict[str, Any],
        agent_name: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        添加日志条目
        
        Args:
            state: 当前 GlobalState
            agent_name：智能体名称
            action：执行的动作
            details：详情
            
        Returns:
            更新后的 GlobalState
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "agent_name": agent_name,
            "action": action,
            "details": details or {}
        }
        
        if "execution_log" not in state:
            state["execution_log"] = []
        
        state["execution_log"].append(log_entry)
        
        return state
    
    def _build_planning_prompt(self, user_topic: str, project_context: str) -> str:
        """构建规划提示词"""
        prompt = f"""请为以下主题生成一个结构化的剧本大纲：

主题：{user_topic}

项目上下文：{project_context if project_context else "无特定上下文"}

要求：
1. 生成 5-10 个步骤，每个步骤应该是一个独立的章节或主题
2. 每个步骤应该有清晰的描述，说明该步骤要讲解的内容
3. 步骤之间应该有逻辑连贯性，形成完整的叙述流程
4. 为每个步骤估算可能需要检索的关键词（用于后续 RAG 检索）

请按以下格式输出（每个步骤一行）：
步骤1: [步骤描述] | 关键词: [关键词1, 关键词2, ...]
步骤2: [步骤描述] | 关键词: [关键词1, 关键词2, ...]
...

示例：
步骤1: 介绍 FastAPI 框架的基本概念和优势 | 关键词: FastAPI, ASGI, 异步框架
步骤2: 创建第一个 FastAPI 应用程序 | 关键词: FastAPI, 路由, 装饰器
"""
        return prompt
    
    def _parse_outline_response(self, response: str) -> List[Dict[str, Any]]:
        """解析 LLM 响应生成大纲步骤"""
        outline_steps = []
        lines = response.strip().split('\n')
        
        step_id = 0
        for line in lines:
            line = line.strip()
            
            if not line or not line.startswith('步骤'):
                continue
            
            try:
                if ':' in line:
                    content = line.split(':', 1)[1].strip()
                    
                    if '|' in content:
                        description = content.split('|')[0].strip()
                    else:
                        description = content
                    
                    outline_step = {
                        "step_id": step_id,
                        "description": description,
                        "status": "pending",
                        "retry_count": 0,
                        "title": "",
                        "estimated_keywords": self._extract_keywords(content)
                    }
                    
                    outline_steps.append(outline_step)
                    step_id += 1
                    
            except Exception as e:
                logger.warning(f"Failed to parse outline line: {line}, error: {str(e)}")
                continue
        
        return outline_steps
    
    def _extract_keywords(self, content: str) -> List[str]:
        """从内容中提取关键词"""
        keywords = []
        
        if '关键词:' in content:
            keyword_part = content.split('关键词:')[-1].strip()
            keywords = [k.strip() for k in keyword_part.split(',') if k.strip()]
        
        return keywords
    
    def _pad_outline(self, outline_steps: List[Dict[str, Any]], min_steps: int) -> List[Dict[str, Any]]:
        """填充大纲步骤到最小数量"""
        while len(outline_steps) < min_steps:
            step_id = len(outline_steps)
            outline_steps.append({
                "step_id": step_id,
                "description": f"补充步骤 {step_id + 1}：进一步探讨相关主题",
                "status": "pending",
                "retry_count": 0,
                "title": "",
                "estimated_keywords": []
            })
        
        return outline_steps
    
    def _create_fallback_outline(self, user_topic: str) -> List[Dict[str, Any]]:
        """创建回退大纲"""
        logger.info("Creating fallback outline")
        
        return [
            {
                "step_id": 0,
                "description": f"介绍主题：{user_topic}",
                "status": "pending",
                "retry_count": 0,
                "title": "",
                "estimated_keywords": []
            },
            {
                "step_id": 1,
                "description": "核心概念和基础知识",
                "status": "pending",
                "retry_count": 0,
                "title": "",
                "estimated_keywords": []
            },
            {
                "step_id": 2,
                "description": "实践示例和应用场景",
                "status": "pending",
                "retry_count": 0,
                "title": "",
                "estimated_keywords": []
            },
            {
                "step_id": 3,
                "description": "常见问题和解决方案",
                "status": "pending",
                "retry_count": 0,
                "title": "",
                "estimated_keywords": []
            },
            {
                "step_id": 4,
                "description": "总结和最佳实践",
                "status": "pending",
                "retry_count": 0,
                "title": "",
                "estimated_keywords": []
            }
        ]


async def plan_outline(state: Dict[str, Any], llm_service: LLMService) -> Dict[str, Any]:
    """
    兼容性函数：原有 Planner 接口
    
    此函数保持原有接口不变，确保向后兼容。
    内部使用 TaskAwarePlanner 但默认禁用 Task Stack。
    
    Args:
        state: GlobalState 字典
        llm_service: LLM 服务实例
        
    Returns:
        更新后的 GlobalState
    """
    planner = create_planner_with_task_stack(use_task_stack=False)
    return await planner.plan_outline(state, llm_service)


def _build_planning_prompt(user_topic: str, project_context: str) -> str:
    """构建规划提示词"""
    prompt = f"""请为以下主题生成一个结构化的剧本大纲：

主题：{user_topic}

项目上下文：{project_context if project_context else "无特定上下文"}

要求：
1. 生成 5-10 个步骤，每个步骤应该是一个独立的章节或主题
2. 每个步骤应该有清晰的描述，说明该步骤要讲解的内容
3. 步骤之间应该有逻辑连贯性，形成完整的叙述流程
4. 为每个步骤估算可能需要检索的关键词（用于后续 RAG 检索）

请按以下格式输出（每个步骤一行）：
步骤1: [步骤描述] | 关键词: [关键词1, 关键词2, ...]
步骤2: [步骤描述] | 关键词: [关键词1, 关键词2, ...]
...

示例：
步骤1: 介绍 FastAPI 框架的基本概念和优势 | 关键词: FastAPI, ASGI, 异步框架
步骤2: 创建第一个 FastAPI 应用程序 | 关键词: FastAPI, 路由, 装饰器
"""
    return prompt


def _extract_keywords(content: str) -> List[str]:
    """从内容中提取关键词"""
    keywords = []
    
    if '关键词:' in content:
        keyword_part = content.split('关键词:')[-1].strip()
        keywords = [k.strip() for k in keyword_part.split(',') if k.strip()]
    
    return keywords


def _parse_outline_response(response: str) -> List[Dict[str, Any]]:
    """解析 LLM 响应生成大纲步骤"""
    outline_steps = []
    lines = response.strip().split('\n')
    
    step_id = 0
    for line in lines:
        line = line.strip()
        
        if not line or not line.startswith('步骤'):
            continue
        
        try:
            if ':' in line:
                content = line.split(':', 1)[1].strip()
                
                if '|' in content:
                    description = content.split('|')[0].strip()
                else:
                    description = content
                
                outline_step = {
                    "step_id": step_id,
                    "description": description,
                    "status": "pending",
                    "retry_count": 0,
                    "title": "",
                    "estimated_keywords": _extract_keywords(content)
                }
                
                outline_steps.append(outline_step)
                step_id += 1
                
        except Exception as e:
            logger.warning(f"Failed to parse outline line: {line}, error: {str(e)}")
            continue
    
    return outline_steps


def _pad_outline(outline_steps: List[Dict[str, Any]], min_steps: int) -> List[Dict[str, Any]]:
    """填充大纲步骤到最小数量"""
    while len(outline_steps) < min_steps:
        step_id = len(outline_steps)
        outline_steps.append({
            "step_id": step_id,
            "description": f"补充步骤 {step_id + 1}：进一步探讨相关主题",
            "status": "pending",
            "retry_count": 0,
            "title": "",
            "estimated_keywords": []
        })
    
    return outline_steps


def _create_fallback_outline(user_topic: str) -> List[Dict[str, Any]]:
    """创建回退大纲"""
    logger.info("Creating fallback outline")
    
    return [
        {
            "step_id": 0,
            "description": f"介绍主题：{user_topic}",
            "status": "pending",
            "retry_count": 0,
            "title": "",
            "estimated_keywords": []
        },
        {
            "step_id": 1,
            "description": "核心概念和基础知识",
            "status": "pending",
            "retry_count": 0,
            "title": "",
            "estimated_keywords": []
        },
        {
            "step_id": 2,
            "description": "实践示例和应用场景",
            "status": "pending",
            "retry_count": 0,
            "title": "",
            "estimated_keywords": []
        },
        {
            "step_id": 3,
            "description": "常见问题和解决方案",
            "status": "pending",
            "retry_count": 0,
            "title": "",
            "estimated_keywords": []
        },
        {
            "step_id": 4,
            "description": "总结和最佳实践",
            "status": "pending",
            "retry_count": 0,
            "title": "",
            "estimated_keywords": []
        }
    ]
