"""工具执行器 - 执行 LLM 调用的工具"""

import json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from .tool_definitions import (
    TOOL_RETRIEVE,
    TOOL_WRITE_FRAGMENT,
    TOOL_ADD_STEP,
    TOOL_MODIFY_STEP,
    TOOL_DELETE_STEP,
    TOOL_REGENERATE_FRAGMENT,
    TOOL_GET_STATUS,
    TOOL_REQUEST_USER_INPUT,
    RetrieveArgs,
    WriteFragmentArgs,
    AddStepArgs,
    ModifyStepArgs,
    DeleteStepArgs,
    RegenerateFragmentArgs,
    UserInputArgs,
)
from ..state_types import GlobalState, ToolResult, ToolCall

logger = logging.getLogger(__name__)


class ToolExecutor:
    """工具执行器 - 执行 LLM 选择的工具"""
    
    def __init__(
        self,
        llm_service: Any,
        retrieval_service: Any,
        node_factory: Any,
        workspace_id: str = ""
    ):
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.node_factory = node_factory
        self.workspace_id = workspace_id
    
    async def execute_tool_call(self, call: ToolCall, state: GlobalState) -> ToolResult:
        """执行单个工具调用"""
        tool_name = call.get("name") or call.get("function", {}).get("name")
        if not tool_name:
            return ToolResult(
                tool_call_id=call.get("id", ""),
                name="unknown",
                content=None,
                success=False,
                error="Tool call missing name"
            )
        
        args_str = call.get("arguments") or call.get("function", {}).get("arguments", "{}")
        
        try:
            if isinstance(args_str, str):
                args = json.loads(args_str)
            else:
                args = args_str
        except json.JSONDecodeError:
            args = {}
        
        try:
            if tool_name == TOOL_RETRIEVE:
                return await self._execute_retrieve(args, state)
            
            elif tool_name == TOOL_WRITE_FRAGMENT:
                return await self._execute_write_fragment(args, state)
            
            elif tool_name == TOOL_ADD_STEP:
                return await self._execute_add_step(args, state)
            
            elif tool_name == TOOL_MODIFY_STEP:
                return await self._execute_modify_step(args, state)
            
            elif tool_name == TOOL_DELETE_STEP:
                return await self._execute_delete_step(args, state)
            
            elif tool_name == TOOL_REGENERATE_FRAGMENT:
                return await self._execute_regenerate_fragment(args, state)
            
            elif tool_name == TOOL_GET_STATUS:
                return await self._execute_get_status(args, state)
            
            elif tool_name == TOOL_REQUEST_USER_INPUT:
                return await self._execute_request_user_input(args, state)
            
            else:
                return ToolResult(
                    tool_call_id=call["id"],
                    name=tool_name,
                    content=None,
                    success=False,
                    error=f"Unknown tool: {tool_name}"
                )
        
        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name}, error: {e}")
            return ToolResult(
                tool_call_id=call["id"],
                name=tool_name,
                content=None,
                success=False,
                error=str(e)
            )
    
    async def execute_all_tools(
        self,
        calls: List[ToolCall],
        state: GlobalState
    ) -> List[ToolResult]:
        """执行所有工具调用"""
        results = []
        
        for call in calls:
            result = await self.execute_tool_call(call, state)
            results.append(result)
            
            # 更新状态
            if result["success"]:
                state = self._merge_tool_result(state, result)
        
        return results
    
    async def _execute_retrieve(self, args: Dict, state: GlobalState) -> ToolResult:
        """执行检索"""
        parsed = RetrieveArgs(**args)
        
        retrieved_docs = await self.retrieval_service.hybrid_retrieve(
            query=parsed.query,
            workspace_id=self.workspace_id,
            top_k=parsed.top_k,
            filters=parsed.filters
        )
        
        docs_as_dicts = [
            {
                "id": doc.metadata.get("chunk_id", str(i)),
                "content": doc.content,
                "source": doc.metadata.get("source", ""),
                "score": doc.score,
                "metadata": doc.metadata
            }
            for i, doc in enumerate(retrieved_docs)
        ]
        
        return ToolResult(
            tool_call_id="",  # 填充
            name=TOOL_RETRIEVE,
            content={"documents": docs_as_dicts},
            success=True,
            error=None
        )
    
    async def _execute_write_fragment(self, args: Dict, state: GlobalState) -> ToolResult:
        """执行写作"""
        parsed = WriteFragmentArgs(**args)
        
        outline = state.get("outline", [])
        if parsed.outline_index >= len(outline):
            return ToolResult(
                tool_call_id="",
                name=TOOL_WRITE_FRAGMENT,
                content=None,
                success=False,
                error=f"Outline index out of bounds: {parsed.outline_index}"
            )
        
        step = outline[parsed.outline_index]
        retrieved_docs = state.get("last_retrieved_docs", [])
        
        fragment_content = await self.node_factory._generate_fragment_content(
            step=step,
            documents=retrieved_docs,
            skill=parsed.skill
        )
        
        fragment = {
            "step_id": parsed.step_id,
            "content": fragment_content,
            "references": [doc.get("source", "") for doc in retrieved_docs],
            "skill_used": parsed.skill
        }
        
        return ToolResult(
            tool_call_id="",
            name=TOOL_WRITE_FRAGMENT,
            content={"fragment": fragment},
            success=True,
            error=None
        )
    
    async def _execute_add_step(self, args: Dict, state: GlobalState) -> ToolResult:
        """动态添加步骤"""
        parsed = AddStepArgs(**args)
        
        outline = state.get("outline", [])
        new_step = {
            "step_id": len(outline),
            "title": parsed.title,
            "description": parsed.description,
            "skill": parsed.skill,
            "status": "pending",
            "dynamically_added": True,
            "added_at": datetime.now().isoformat()
        }
        
        # 插入到指定位置
        insert_index = parsed.after_step_id + 1
        updated_outline = outline[:insert_index] + [new_step] + outline[insert_index:]
        
        # 重新编号
        for i, step in enumerate(updated_outline):
            step["step_id"] = i
        
        return ToolResult(
            tool_call_id="",
            name=TOOL_ADD_STEP,
            content={"outline": updated_outline, "added_step": new_step},
            success=True,
            error=None
        )
    
    async def _execute_modify_step(self, args: Dict, state: GlobalState) -> ToolResult:
        """修改步骤"""
        parsed = ModifyStepArgs(**args)
        
        outline = state.get("outline", [])
        if parsed.step_index >= len(outline):
            return ToolResult(
                tool_call_id="",
                name=TOOL_MODIFY_STEP,
                content=None,
                success=False,
                error=f"Step index out of bounds: {parsed.step_index}"
            )
        
        step = outline[parsed.step_index]
        
        if parsed.new_title:
            step["title"] = parsed.new_title
        if parsed.new_description:
            step["description"] = parsed.new_description
        if parsed.new_skill:
            step["skill"] = parsed.new_skill
        
        step["modified_at"] = datetime.now().isoformat()
        
        return ToolResult(
            tool_call_id="",
            name=TOOL_MODIFY_STEP,
            content={"outline": outline, "modified_step": step},
            success=True,
            error=None
        )
    
    async def _execute_delete_step(self, args: Dict, state: GlobalState) -> ToolResult:
        """删除步骤"""
        parsed = DeleteStepArgs(**args)
        
        outline = state.get("outline", [])
        if parsed.step_index >= len(outline):
            return ToolResult(
                tool_call_id="",
                name=TOOL_DELETE_STEP,
                content=None,
                success=False,
                error=f"Step index out of bounds: {parsed.step_index}"
            )
        
        deleted_step = outline[parsed.step_index]
        updated_outline = outline[:parsed.step_index] + outline[parsed.step_index + 1:]
        
        for i, step in enumerate(updated_outline):
            step["step_id"] = i
        
        deleted_step["deleted_at"] = datetime.now().isoformat()
        
        return ToolResult(
            tool_call_id="",
            name=TOOL_DELETE_STEP,
            content={"outline": updated_outline, "deleted_step": deleted_step},
            success=True,
            error=None
        )
    
    async def _execute_regenerate_fragment(self, args: Dict, state: GlobalState) -> ToolResult:
        """重新生成片段"""
        parsed = RegenerateFragmentArgs(**args)
        
        outline = state.get("outline", [])
        fragments = state.get("fragments", [])
        
        if parsed.step_index >= len(outline):
            return ToolResult(
                tool_call_id="",
                name=TOOL_REGENERATE_FRAGMENT,
                content=None,
                success=False,
                error=f"Step index out of bounds: {parsed.step_index}"
            )
        
        step = outline[parsed.step_index]
        retrieved_docs = state.get("last_retrieved_docs", [])
        
        new_content = await self.node_factory._generate_fragment_content(
            step=step,
            documents=retrieved_docs,
            skill=step.get("skill", "standard_tutorial")
        )
        
        fragment = {
            "step_id": parsed.step_index,
            "content": new_content,
            "references": [doc.get("source", "") for doc in retrieved_docs],
            "skill_used": step.get("skill", "standard_tutorial"),
            "regenerated_at": datetime.now().isoformat(),
            "reason": parsed.reason
        }
        
        updated_fragments = [f for f in fragments if f.get("step_id") != parsed.step_index]
        updated_fragments.append(fragment)
        
        step["status"] = "pending"
        
        return ToolResult(
            tool_call_id="",
            name=TOOL_REGENERATE_FRAGMENT,
            content={"fragments": updated_fragments, "regenerated_fragment": fragment},
            success=True,
            error=None
        )
    
    async def _execute_get_status(self, args: Dict, state: GlobalState) -> ToolResult:
        """获取当前状态"""
        outline = state.get("outline", [])
        fragments = state.get("fragments", [])
        current_step_index = state.get("current_step_index", 0)
        
        completed_steps = len(fragments)
        total_steps = len(outline)
        progress = (completed_steps / total_steps * 100) if total_steps > 0 else 0
        
        status_info = {
            "total_steps": total_steps,
            "completed_steps": completed_steps,
            "current_step_index": current_step_index,
            "progress_percentage": round(progress, 1),
            "outline": outline,
            "last_fragment_preview": fragments[-1]["content"][:200] if fragments else None,
            "awaiting_user_input": state.get("awaiting_user_input", False),
            "human_intervention": state.get("human_intervention", None)
        }
        
        return ToolResult(
            tool_call_id="",
            name=TOOL_GET_STATUS,
            content=status_info,
            success=True,
            error=None
        )
    
    async def _execute_request_user_input(self, args: Dict, state: GlobalState) -> ToolResult:
        """请求用户输入"""
        parsed = UserInputArgs(**args)
        
        # 设置中断状态
        state["awaiting_user_input"] = True
        state["human_intervention"] = {
            "type": "request",
            "prompt": parsed.prompt,
            "choices": parsed.choices,
            "requested_at": datetime.now().isoformat(),
            "completed_at": None
        }
        
        return ToolResult(
            tool_call_id="",
            name=TOOL_REQUEST_USER_INPUT,
            content={"intervention_requested": True},
            success=True,
            error=None
        )
    
    def _merge_tool_result(self, state: GlobalState, result: ToolResult) -> GlobalState:
        """将工具结果合并到状态"""
        tool_name = result["name"]
        content = result.get("content", {})
        
        if tool_name == TOOL_RETRIEVE and "documents" in content:
            state["last_retrieved_docs"] = content["documents"]
        
        elif tool_name == TOOL_WRITE_FRAGMENT and "fragment" in content:
            fragments = state.get("fragments", [])
            fragments.append(content["fragment"])
            state["fragments"] = fragments
        
        elif tool_name == TOOL_ADD_STEP and "outline" in content:
            state["outline"] = content["outline"]
        
        elif tool_name == TOOL_MODIFY_STEP and "outline" in content:
            state["outline"] = content["outline"]
        
        elif tool_name == TOOL_DELETE_STEP and "outline" in content:
            state["outline"] = content["outline"]
        
        elif tool_name == TOOL_REGENERATE_FRAGMENT and "fragments" in content:
            state["fragments"] = content["fragments"]
        
        return state
