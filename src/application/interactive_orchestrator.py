"""交互式工作流编排器 - 支持 Function Calling 和用户干预"""

import logging
from typing import Dict, Any, Optional, List
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from ..domain.state_types import GlobalState
from ..domain.tools.tool_service import ToolService
from ..domain.tools.tool_executor import ToolExecutor
from ..domain.agents.editor_agent import EditorAgent
from ..domain.agents.node_factory import NodeFactory
from .base_orchestrator import BaseWorkflowOrchestrator

logger = logging.getLogger(__name__)


class InteractiveWorkflowOrchestrator(BaseWorkflowOrchestrator):
    """
    交互式工作流编排器 - 支持 Function Calling 和用户干预
    
    继承自 BaseWorkflowOrchestrator，使用共享的节点实现。
    扩展功能：
    1. Editor Agent 节点：处理用户输入和工具调用
    2. 状态机循环：在 Editor 和工作流之间切换
    3. 用户干预点：暂停工作流等待用户输入
    
    模式：
    - EDITOR_MODE：编辑器模式，处理用户对话
    - WORKFLOW_MODE：工作流模式，执行剧本生成
    """
    
    MODE_EDITOR = "editor"
    MODE_WORKFLOW = "workflow"
    
    def __init__(
        self,
        llm_service: Any,
        retrieval_service: Any,
        parser_service: Any,
        summarization_service: Any,
        workspace_id: str,
        enable_checkpointer: bool = True
    ):
        """
        初始化交互式工作流编排器
        
        Args:
            llm_service: LLM 服务实例
            retrieval_service: 检索服务实例
            parser_service: 解析服务实例
            summarization_service: 摘要服务实例
            workspace_id: 工作空间 ID
            enable_checkpointer: 是否启用状态持久化
        """
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.parser_service = parser_service
        self.summarization_service = summarization_service
        self.workspace_id = workspace_id
        
        self.node_factory = NodeFactory(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id=workspace_id
        )
        
        super().__init__(self.node_factory)
        
        self.tool_executor = ToolExecutor(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            node_factory=self.node_factory,
            workspace_id=workspace_id
        )
        
        self.tool_service = ToolService(
            llm_service=llm_service,
            tool_executor=self.tool_executor,
            max_iterations=10
        )
        
        self.editor_agent = EditorAgent(tool_service=self.tool_service)
        
        self.graph = self._build_graph()
        
        checkpointer = MemorySaver() if enable_checkpointer else None
        self.config = {"checkpointer": checkpointer} if checkpointer else None
        
        logger.info("InteractiveWorkflowOrchestrator 初始化完成")
    
    def _build_graph(self):
        """构建 LangGraph 状态图"""
        workflow = StateGraph(GlobalState)
        
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("navigator", self._navigator_node)
        workflow.add_node("director", self._director_node)
        workflow.add_node("retry_protection", self._retry_protection_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("fact_checker", self._fact_checker_node)
        workflow.add_node("step_advancer", self._step_advancer_node)
        workflow.add_node("compiler", self._compiler_node)
        workflow.add_node("editor", self._editor_node)
        
        workflow.set_entry_point("planner")
        
        workflow.add_edge("planner", "navigator")
        workflow.add_edge("navigator", "director")
        
        workflow.add_conditional_edges(
            "director",
            self._route_director_decision,
            {
                "pivot": "navigator",
                "navigate": "navigator",
                "write": "retry_protection",
                "editor": "editor"
            }
        )
        
        workflow.add_edge("retry_protection", "writer")
        workflow.add_edge("writer", "fact_checker")
        
        workflow.add_conditional_edges(
            "fact_checker",
            self._route_fact_check,
            {"invalid": "retry_protection", "valid": "step_advancer"}
        )
        
        workflow.add_conditional_edges(
            "step_advancer",
            self._route_completion,
            {"continue": "navigator", "done": "editor"}
        )
        
        workflow.add_conditional_edges(
            "editor",
            self._route_editor_decision,
            {
                "continue_workflow": "navigator",
                "stay_in_editor": "editor",
                "finish": "compiler"
            }
        )
        
        workflow.add_edge("compiler", END)
        
        return workflow.compile()
    
    def _route_director_decision(self, state: GlobalState) -> str:
        """路由导演决策"""
        director_feedback = self._get_state_value(state, "director_feedback", {})
        decision = director_feedback.get("decision", "write")
        
        if decision == "editor":
            return "editor"
        elif decision == "continue":
            return "write"
        return decision
    
    def _route_fact_check(self, state: GlobalState) -> str:
        """路由事实检查结果"""
        fact_check_passed = self._get_state_value(state, "fact_check_passed", False)
        return "valid" if fact_check_passed else "invalid"
    
    def _route_completion(self, state: GlobalState) -> str:
        """路由完成状态"""
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        
        if current_step_index >= len(outline):
            return "done"
        return "continue"
    
    def _route_editor_decision(self, state: GlobalState) -> str:
        """路由编辑器决策"""
        awaiting_user_input = self._get_state_value(state, "awaiting_user_input", False)
        
        if awaiting_user_input:
            return "stay_in_editor"
        
        user_intervention = self._get_state_value(state, "human_intervention", None)
        
        if user_intervention and user_intervention.get("completed_at") is None:
            return "stay_in_editor"
        
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        
        if current_step_index >= len(outline):
            return "finish"
        
        return "continue_workflow"
    
    async def _editor_node(self, state: GlobalState) -> Dict[str, Any]:
        """编辑器节点 - 处理用户输入和工具调用"""
        user_message = self._get_state_value(state, "user_message", "")
        chat_history = self._get_state_value(state, "chat_history", [])
        
        if not user_message:
            return {
                "awaiting_user_input": True,
                "editor_response": "请输入您想要执行的修改或操作。"
            }
        
        result = await self.editor_agent.process_message(
            user_message=user_message,
            state=state,
            chat_history=chat_history,
            include_context=True
        )
        
        return {
            "editor_response": result["response"],
            "chat_history": result["updated_chat_history"],
            "awaiting_user_input": result["requires_user_input"],
            "human_intervention": state.get("human_intervention"),
            "exceeded_max_iterations": result.get("exceeded_max_iterations", False)
        }
    
    async def execute_workflow(
        self,
        initial_state: GlobalState,
        user_inputs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        执行工作流
        
        Args:
            initial_state: 初始状态
            user_inputs: 可选的用户输入列表
            
        Returns:
            最终状态
        """
        state = initial_state.copy()
        
        if user_inputs:
            state["user_inputs"] = user_inputs
            state["current_input_index"] = 0
        
        final_state = None
        
        async for chunk in self.graph.astream(state, config=self.config):
            for node_name, node_output in chunk.items():
                logger.info(f"Node {node_name} completed")
                
                if node_name == "editor" and "editor_response" in node_output:
                    response = node_output["editor_response"]
                    print(f"\n🤖 Editor: {response}")
                    
                    if node_output.get("awaiting_user_input"):
                        continue
                
                if node_name == "compiler" and "final_script" in node_output:
                    final_state = {**state, **node_output}
                    print(f"\n✅ 剧本生成完成！")
                    print(f"📄 脚本长度: {len(node_output['final_script'])} 字符")
        
        return final_state or state
    
    async def process_user_message(
        self,
        state: GlobalState,
        user_message: str
    ) -> Dict[str, Any]:
        """
        处理用户消息
        
        Args:
            state: 当前状态
            user_message: 用户消息
            
        Returns:
            更新后的状态
        """
        state["user_message"] = user_message
        
        async for chunk in self.graph.astream(state, config=self.config):
            for node_name, node_output in chunk.items():
                if node_name == "editor":
                    return {**state, **node_output}
        
        return state
