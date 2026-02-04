"""编辑器智能体 - 处理用户交互和工具调用"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

from ..tools.tool_service import ToolService
from ..tools.tool_executor import ToolExecutor
from ..state_types import GlobalState

logger = logging.getLogger(__name__)


class EditorAgent:
    """
    编辑器智能体 - 处理用户交互和工具调用
    
    职责：
    1. 理解用户意图
    2. 协调工具调用
    3. 生成自然回复
    4. 管理状态更新
    
    v2.1 架构：
    - 使用 ToolService 执行工具调用
    - 使用 GlobalState 进行状态管理
    - 支持多轮对话
    """
    
    def __init__(
        self,
        tool_service: ToolService,
        system_prompt: Optional[str] = None
    ):
        """
        初始化编辑器智能体
        
        Args:
            tool_service: 工具服务实例
            system_prompt: 系统提示词
        """
        self.tool_service = tool_service
        self.system_prompt = system_prompt or self._create_default_system_prompt()
    
    def _create_default_system_prompt(self) -> str:
        """创建默认系统提示词"""
        return """你是剧本生成的智能编辑器助手。你的核心职责是帮助用户创建和完善剧本大纲。

## 你的能力

### 1. 理解用户意图
- **添加步骤**：当用户说"在XX之后添加..."时，调用 add_step
- **修改步骤**：当用户说"把XX改成..."或"修改第X步"时，调用 modify_step
- **删除步骤**：当用户说"删除第X步"时，调用 delete_step
- **重新生成**：当用户说"重新生成第X步"时，调用 regenerate_fragment
- **查询状态**：当用户问"现在进度如何"时，调用 get_current_status
- **闲聊**：当用户问候或闲聊时，自然回应

### 2. 工具使用规则
- add_step: 需要指定位置、标题、描述和技能
- modify_step: 需要指定步骤索引和新值（至少一个）
- delete_step: 需要指定步骤索引
- regenerate_fragment: 需要指定步骤索引和原因
- get_current_status: 无参数，返回当前完整状态

### 3. 回复原则
- 执行工具后，向用户确认操作结果
- 如果工具执行失败，解释原因并建议解决方案
- 保持回复简洁、专业、有帮助
- 使用自然语言，像在与真人对话

请根据用户输入判断是否需要调用工具。"""
    
    async def process_message(
        self,
        user_message: str,
        state: GlobalState,
        chat_history: Optional[List[Dict[str, str]]] = None,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        处理用户消息
        
        Args:
            user_message: 用户消息
            state: 当前全局状态
            chat_history: 对话历史
            include_context: 是否在消息中包含上下文
            
        Returns:
            {
                "response": str,  # 系统回复
                "state": GlobalState,  # 更新后的状态
                "tool_calls": List[Dict],  # 执行的工具调用
                "requires_user_input": bool,  # 是否需要用户输入
                "updated_chat_history": List[Dict]  # 更新后的对话历史
            }
        """
        context_info = ""
        
        if include_context:
            context_info = self._build_context_info(state)
        
        if chat_history is None:
            chat_history = []
        
        system_messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        if context_info:
            system_messages.append({
                "role": "system",
                "content": f"\n当前上下文信息：\n{context_info}"
            })
        
        full_messages = system_messages + chat_history + [{"role": "user", "content": user_message}]
        
        result = await self.tool_service.execute_tool_loop(
            messages=full_messages,
            state=state,
            task_type="high_performance"
        )
        
        final_response = result["final_response"]
        
        if not result["tool_results"]:
            if "现在进度" in user_message or "状态" in user_message:
                status_info = self._format_status_info(state)
                final_response = status_info
        
        chat_history = result["messages"]
        chat_history.append({"role": "assistant", "content": final_response})
        
        return {
            "response": final_response,
            "state": result["updated_state"],
            "tool_calls": result["tool_results"],
            "requires_user_input": result["requires_user_input"],
            "updated_chat_history": chat_history,
            "exceeded_max_iterations": result["exceeded_max_iterations"]
        }
    
    def _build_context_info(self, state: GlobalState) -> str:
        """构建上下文信息"""
        outline = state.get("outline", [])
        fragments = state.get("fragments", [])
        current_step_index = state.get("current_step_index", 0)
        
        context_parts = []
        
        context_parts.append(f"总步骤数：{len(outline)}")
        context_parts.append(f"已完成：{len(fragments)}")
        context_parts.append(f"当前步骤索引：{current_step_index}")
        
        if outline:
            recent_steps = outline[-3:] if len(outline) > 3 else outline
            steps_summary = "\n".join([
                f"  {s.get('step_id', i)}: {s.get('title', 'Untitled')}"
                for i, s in enumerate(recent_steps)
            ])
            context_parts.append(f"最近步骤：\n{steps_summary}")
        
        return "\n".join(context_parts)
    
    def _format_status_info(self, state: GlobalState) -> str:
        """格式化状态信息"""
        outline = state.get("outline", [])
        fragments = state.get("fragments", [])
        current_step_index = state.get("current_step_index", 0)
        
        total_steps = len(outline)
        completed_steps = len(fragments)
        progress = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        status_lines = [
            "=== 当前状态 ===",
            f"总步骤数：{total_steps}",
            f"已完成：{completed_steps}",
            f"进行中：{current_step_index}",
            f"进度：{progress:.1f}%"
        ]
        
        if outline:
            status_lines.append("\n大纲预览：")
            for i, step in enumerate(outline[:5]):
                status = "✓" if i < completed_steps else "○"
                status_lines.append(f"  {status} 步骤{i}: {step.get('title', 'Untitled')}")
            
            if len(outline) > 5:
                status_lines.append(f"  ... 还有 {len(outline) - 5} 个步骤")
        
        awaiting = state.get("awaiting_user_input", False)
        if awaiting:
            status_lines.append("\n⚠️ 等待用户输入")
        
        return "\n".join(status_lines)
    
    def create_outline_summary(self, state: GlobalState) -> str:
        """创建大纲摘要"""
        outline = state.get("outline", [])
        fragments = state.get("fragments", [])
        
        if not outline:
            return "目前还没有大纲。"
        
        summary_lines = [f"**剧本大纲（共 {len(outline)} 步）**\n"]
        
        for i, step in enumerate(outline):
            fragment = next((f for f in fragments if f.get("step_id") == i), None)
            status = "✅" if fragment else "⏳"
            
            step_line = f"{status} **步骤 {i}**: {step.get('title', 'Untitled')}"
            
            if step.get("dynamically_added"):
                step_line += " *(动态添加)*"
            
            summary_lines.append(step_line)
            
            if fragment and len(fragment.get("content", "")) > 100:
                preview = fragment["content"][:100] + "..."
                summary_lines.append(f"   > {preview}")
        
        return "\n".join(summary_lines)
