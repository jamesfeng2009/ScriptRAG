"""工具服务 - 管理 Function Calling 循环和工具执行"""

import json
import logging
from typing import List, Dict, Any, Optional, Literal
from datetime import datetime

from .tool_executor import ToolExecutor
from .tool_definitions import TOOLS
from ..state_types import GlobalState, ToolCall
from ...services.llm.service import LLMService

logger = logging.getLogger(__name__)


class ToolService:
    """
    工具服务 - 管理 Function Calling 循环
    
    功能：
    1. 管理 LLM 和工具之间的调用循环
    2. 执行工具并合并结果到状态
    3. 处理用户干预请求
    4. 维护对话历史
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        tool_executor: ToolExecutor,
        max_iterations: int = 10
    ):
        """
        初始化工具服务
        
        Args:
            llm_service: LLM 服务实例
            tool_executor: 工具执行器实例
            max_iterations: 最大工具调用迭代次数
        """
        self.llm_service = llm_service
        self.tool_executor = tool_executor
        self.max_iterations = max_iterations
        self.available_tools = TOOLS
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """获取可用的工具 Schema"""
        return self.available_tools
    
    async def execute_tool_loop(
        self,
        messages: List[Dict[str, str]],
        state: GlobalState,
        task_type: str = "high_performance"
    ) -> Dict[str, Any]:
        """
        执行工具调用循环
        
        Args:
            messages: 对话消息历史
            state: 当前全局状态
            task_type: 任务类型
            
        Returns:
            {
                "final_response": str,  # LLM 的最终回复
                "messages": List[Dict],  # 更新后的消息历史
                "tool_results": List[Dict],  # 所有工具执行结果
                "exceeded_max_iterations": bool,  # 是否超过最大迭代次数
                "requires_user_input": bool,  # 是否需要用户输入
                "updated_state": GlobalState  # 更新后的状态
            }
        """
        tool_results = []
        exceeded_max_iterations = False
        requires_user_input = False
        
        adapter = self.llm_service._get_adapter(task_type)
        model_name = self.llm_service._get_model_name(task_type)
        
        for iteration in range(self.max_iterations):
            logger.info(f"Tool call iteration {iteration + 1}/{self.max_iterations}")
            
            response = await adapter.chat_completion_with_tools(
                messages=messages,
                model=model_name,
                tools=self.available_tools,
                tool_choice={"type": "function"},
                temperature=0.7,
                max_tokens=4096
            )
            
            tool_calls = response.get("tool_calls", [])
            finish_reason = response.get("finish_reason", "stop")
            
            if not tool_calls or finish_reason == "stop":
                return {
                    "final_response": response.get("content", ""),
                    "messages": messages,
                    "tool_results": tool_results,
                    "exceeded_max_iterations": False,
                    "requires_user_input": False,
                    "updated_state": state
                }
            
            logger.info(f"Executing {len(tool_calls)} tool calls")
            
            for call in tool_calls:
                result = await self.tool_executor.execute_tool_call(call, state)
                tool_results.append(result)
                
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": call["function"]["name"],
                    "content": json.dumps(result.get("content", {}), ensure_ascii=False)
                }
                messages.append(tool_message)
                
                if result.get("success"):
                    state = self.tool_executor._merge_tool_result(state, result)
                    
                    if call["function"]["name"] == "request_user_input":
                        requires_user_input = True
                        break
                else:
                    logger.warning(f"Tool execution failed: {result.get('error')}")
            
            if requires_user_input:
                break
        
        exceeded_max_iterations = (iteration + 1) >= self.max_iterations
        
        if exceeded_max_iterations:
            logger.warning("Exceeded maximum tool call iterations")
            messages.append({
                "role": "system",
                "content": "警告：已达到最大工具调用迭代次数，请直接回复用户，不要再调用工具。"
            })
        
        final_response = await adapter.chat_completion(
            messages=messages,
            model=model_name,
            temperature=0.7,
            max_tokens=2048
        )
        
        return {
            "final_response": final_response,
            "messages": messages,
            "tool_results": tool_results,
            "exceeded_max_iterations": exceeded_max_iterations,
            "requires_user_input": requires_user_input,
            "updated_state": state
        }
    
    async def execute_single_turn(
        self,
        user_message: str,
        state: GlobalState,
        chat_history: Optional[List[Dict[str, str]]] = None,
        task_type: str = "high_performance"
    ) -> Dict[str, Any]:
        """
        执行单轮对话（包含工具调用）
        
        Args:
            user_message: 用户消息
            state: 当前全局状态
            chat_history: 可选的对话历史
            task_type: 任务类型
            
        Returns:
            {
                "response": str,  # 系统回复
                "state": GlobalState,  # 更新后的状态
                "tool_calls_executed": List[Dict],  # 执行的工具调用
                "requires_user_input": bool  # 是否需要用户输入
            }
        """
        messages = chat_history.copy() if chat_history else []
        messages.append({"role": "user", "content": user_message})
        
        result = await self.execute_tool_loop(messages, state, task_type)
        
        messages.append({"role": "assistant", "content": result["final_response"]})
        
        return {
            "response": result["final_response"],
            "state": result["updated_state"],
            "tool_calls_executed": result["tool_results"],
            "requires_user_input": result["requires_user_input"],
            "chat_history": messages
        }
    
    def create_system_prompt(self, include_tools: bool = True) -> str:
        """
        创建系统提示词
        
        Args:
            include_tools: 是否包含工具说明
            
        Returns:
            系统提示词字符串
        """
        system_prompt = """你是剧本生成的智能助手。你的职责是帮助用户创建、编辑和完善剧本大纲。

你的核心能力：
1. 理解用户的修改意图（添加、删除、修改步骤）
2. 通过工具执行用户的指令
3. 生成高质量的剧本内容
4. 提供专业的建议和反馈

交互原则：
- 当用户表达修改意图时，主动调用相应工具
- 当用户询问状态时，提供准确的信息
- 当用户闲聊时，自然回应但保持专业
- 在执行工具前，确保参数正确
- 每次工具调用后，向用户确认操作结果

请根据用户输入决定是否需要调用工具。"""
        
        if include_tools:
            tool_descriptions = []
            for tool in self.available_tools:
                func = tool.get("function", {})
                tool_descriptions.append(
                    f"- {func.get('name')}: {func.get('description')}"
                )
            
            tools_section = f"\n\n可用工具：\n" + "\n".join(tool_descriptions)
            system_prompt += tools_section
        
        return system_prompt
