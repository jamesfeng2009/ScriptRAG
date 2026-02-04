"""工作流编排器基类 - 共享节点实现"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from ..domain.state_types import GlobalState
from ..domain.agents.node_factory import NodeFactory
from ..infrastructure.langgraph_error_handler import with_error_handling

logger = logging.getLogger(__name__)


class BaseWorkflowOrchestrator:
    """
    工作流编排器基类 - 提供共享的节点实现
    
    职责：
    1. 定义所有工作流节点的默认实现
    2. 提供状态访问的公共方法
    3. 被 WorkflowOrchestrator 和 InteractiveWorkflowOrchestrator 继承
    """
    
    def __init__(self, node_factory: NodeFactory):
        """
        初始化基类
        
        Args:
            node_factory: 节点工厂实例
        """
        self.node_factory = node_factory
    
    def _get_state_value(self, state: GlobalState, key: str, default: Any = None) -> Any:
        """安全地从状态中获取值"""
        if isinstance(state, dict):
            return state.get(key, default)
        return getattr(state, key, default)
    
    @with_error_handling(agent_name="planner", action_name="plan_outline")
    async def _planner_node(self, state: GlobalState) -> Dict[str, Any]:
        """规划器节点 - 生成剧本大纲"""
        user_topic = self._get_state_value(state, "user_topic", "")
        project_context = self._get_state_value(state, "project_context", "")
        
        outline = await self.node_factory.planner_node(user_topic, project_context)
        
        return {
            "outline": outline,
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "planner",
                "action": "created outline",
                "step_count": len(outline)
            }]
        }
    
    @with_error_handling(agent_name="navigator", action_name="navigate_step")
    async def _navigator_node(self, state: GlobalState) -> Dict[str, Any]:
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        
        if current_step_index >= len(outline):
            return {
                "workflow_complete": True,
                "execution_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "node": "navigator",
                    "action": f"navigation complete - all {len(outline)} steps processed"
                }]
            }
        
        try:
            navigation = await self.node_factory.navigator_node(state)
        except Exception:
            navigation = {
                "execution_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "node": "navigator",
                    "action": f"navigated to step {current_step_index}"
                }]
            }
        
        return {
            "navigation": navigation,
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "navigator",
                "action": f"navigated to step {current_step_index}"
            }]
        }
    
    @with_error_handling(agent_name="director", action_name="direct_step")
    async def _director_node(self, state: GlobalState) -> Dict[str, Any]:
        """导演节点 - 评估并决定下一步"""
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        
        if current_step_index >= len(outline):
            return {}
        
        feedback = await self.node_factory.director_node(state)
        
        decision = feedback.get("decision", "write")
        
        if decision == "continue":
            decision = "write"
        
        return {
            "director_feedback": feedback,
            "decision": decision,
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "director",
                "action": f"directed step {current_step_index}",
                "decision": decision
            }]
        }
    
    @with_error_handling(agent_name="retry_protection", action_name="check_retry")
    async def _retry_protection_node(self, state: GlobalState) -> Dict[str, Any]:
        """重试保护节点 - 防止无限重试"""
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        max_retries = self._get_state_value(state, "max_retries", 3)
        
        retry_count = self._get_state_value(state, f"step_{current_step_index}_retry_count", 0)
        
        if retry_count >= max_retries:
            return {
                "skip_current_step": True,
                "execution_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "node": "retry_protection",
                    "action": f"step {current_step_index} skipped due to max retries"
                }]
            }
        
        return {
            "skip_current_step": False,
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "retry_protection",
                "action": f"step {current_step_index} allowed to proceed"
            }]
        }
    
    @with_error_handling(agent_name="writer", action_name="write_fragment")
    async def _writer_node(self, state: GlobalState) -> Dict[str, Any]:
        """编剧节点 - 生成剧本片段"""
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        skip_current_step = self._get_state_value(state, "skip_current_step", False)
        
        if current_step_index >= len(outline) or skip_current_step:
            return {}
        
        result = await self.node_factory.writer_node(state)
        
        fragment = result.get("fragment", "")
        fragments = self._get_state_value(state, "fragments", [])
        fragments.append(fragment)
        
        return {
            "fragments": fragments,
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "writer",
                "action": f"wrote fragment for step {current_step_index}"
            }]
        }
    
    @with_error_handling(agent_name="fact_checker", action_name="verify_content")
    async def _fact_checker_node(self, state: GlobalState) -> Dict[str, Any]:
        """事实检查器节点 - 验证内容准确性"""
        fragments = self._get_state_value(state, "fragments", [])
        
        if not fragments:
            return {"fact_check_passed": True}
        
        result = await self.node_factory.fact_checker_node(state)
        
        is_valid = result.get("fact_check_passed", True)
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        
        if not is_valid:
            retry_key = f"step_{current_step_index}_retry_count"
            retry_count = self._get_state_value(state, retry_key, 0)
            state[retry_key] = retry_count + 1
        
        return {
            "fact_check_passed": is_valid,
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "fact_checker",
                "action": f"fact check {'passed' if is_valid else 'failed'} for step {current_step_index}"
            }]
        }
    
    @with_error_handling(agent_name="step_advancer", action_name="advance_step")
    async def _step_advancer_node(self, state: GlobalState) -> Dict[str, Any]:
        """步骤推进器节点 - 推进到下一步，并标记工作流完成"""
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        skip_current_step = self._get_state_value(state, "skip_current_step", False)
        
        is_done = current_step_index >= len(outline)
        
        if skip_current_step or is_done:
            return {
                "current_step_index": current_step_index,
                "execution_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "node": "step_advancer",
                    "action": f"workflow complete at step {current_step_index}"
                }],
                "workflow_complete": True
            }
        
        next_index = current_step_index + 1
        is_done_after_advance = next_index >= len(outline)
        
        updates = {
            "current_step_index": next_index,
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "step_advancer",
                "action": f"advanced from step {current_step_index} to {next_index}"
            }]
        }
        
        if is_done_after_advance:
            updates["workflow_complete"] = True
        
        return updates
    
    @with_error_handling(agent_name="compiler", action_name="compile_script")
    async def _compiler_node(self, state: GlobalState) -> Dict[str, Any]:
        """编译器节点 - 编译最终剧本"""
        logger.info("[COMPILER] 编译器节点被调用!")
        result = await self.node_factory.compiler_node(state)
        
        script = result.get("script", "")
        
        logger.info(f"[COMPILER] 生成剧本长度: {len(script)}")
        
        return {
            "final_screenplay": script,
            "execution_log": [{
                "timestamp": datetime.now().isoformat(),
                "node": "compiler",
                "action": "compiled final script"
            }]
        }
