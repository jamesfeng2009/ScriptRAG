"""Workflow Orchestrator - LangGraph state machine management

This module implements the LangGraph state machine that orchestrates
the multi-agent screenplay generation workflow.
"""

import logging
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END

from ..domain.models import SharedState
from ..domain.agents.planner import plan_outline
from ..domain.agents.navigator import retrieve_content
from ..domain.agents.director import evaluate_and_decide
from ..domain.agents.pivot_manager import handle_pivot
from ..domain.agents.writer import generate_fragment
from ..domain.agents.fact_checker import verify_fragment_node
from ..domain.agents.compiler import compile_screenplay
from ..domain.agents.retry_protection import check_retry_limit
from ..services.llm.service import LLMService
from ..services.retrieval_service import RetrievalService
from ..services.parser.tree_sitter_parser import IParserService
from ..services.summarization_service import SummarizationService
from ..infrastructure.error_handler import (
    ErrorHandler,
    TimeoutError as CustomTimeoutError,
    with_timeout
)


logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    工作流编排器 - 管理 LangGraph 状态机
    
    职责：
    1. 构建和编译 LangGraph 状态图
    2. 定义智能体节点和边
    3. 实现路由函数
    4. 执行完整的剧本生成工作流
    
    验证需求: 14.2, 14.3, 14.4, 14.5, 14.6
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        parser_service: IParserService,
        summarization_service: SummarizationService,
        workspace_id: str
    ):
        """
        初始化工作流编排器
        
        Args:
            llm_service: LLM 服务实例
            retrieval_service: 检索服务实例
            parser_service: 解析服务实例
            summarization_service: 摘要服务实例
            workspace_id: 工作空间 ID
        """
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.parser_service = parser_service
        self.summarization_service = summarization_service
        self.workspace_id = workspace_id
        
        # 构建状态图
        self.graph = self._build_graph()
        
        logger.info("WorkflowOrchestrator initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态图
        
        定义所有智能体节点、边和路由逻辑。
        
        Returns:
            编译后的状态图
            
        验证需求: 14.2, 14.3, 14.4, 14.5
        """
        logger.info("Building LangGraph state machine")
        
        # 创建状态图
        workflow = StateGraph(SharedState)
        
        # 添加节点（需求 14.2）
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("navigator", self._navigator_node)
        workflow.add_node("director", self._director_node)
        workflow.add_node("pivot_manager", self._pivot_manager_node)
        workflow.add_node("retry_protection", self._retry_protection_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("fact_checker", self._fact_checker_node)
        workflow.add_node("compiler", self._compiler_node)
        
        # 设置入口点（需求 14.3）
        workflow.set_entry_point("planner")
        
        # 添加边（需求 14.3, 14.4）
        # 规划器 -> 导航器
        workflow.add_edge("planner", "navigator")
        
        # 导航器 -> 导演
        workflow.add_edge("navigator", "director")
        
        # 导演决策逻辑（条件边）（需求 14.5）
        workflow.add_conditional_edges(
            "director",
            self._route_director_decision,
            {
                "pivot": "pivot_manager",
                "write": "retry_protection"
            }
        )
        
        # 转向管理器 -> 导航器（转向循环）（需求 14.4）
        workflow.add_edge("pivot_manager", "navigator")
        
        # 重试保护 -> 编剧
        workflow.add_edge("retry_protection", "writer")
        
        # 编剧 -> 事实检查器
        workflow.add_edge("writer", "fact_checker")
        
        # 事实检查器决策逻辑（条件边）（需求 14.5）
        # 完成检查集成在这里（需求 14.4, 14.5）
        workflow.add_conditional_edges(
            "fact_checker",
            self._route_fact_check_and_completion,
            {
                "invalid": "retry_protection",
                "continue": "navigator",
                "done": "compiler"
            }
        )
        
        # 编译器 -> 结束
        workflow.add_edge("compiler", END)
        
        # 编译图（需求 14.6）
        compiled_graph = workflow.compile()
        
        logger.info("LangGraph state machine built and compiled successfully")
        
        return compiled_graph
    
    # Node wrapper functions
    
    async def _planner_node(self, state: SharedState) -> SharedState:
        """规划器节点包装函数（带超时保护）"""
        logger.info("Executing planner node")
        try:
            return await ErrorHandler.with_timeout(
                plan_outline,
                timeout_seconds=60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Planner node timed out: {str(e)}")
            # 超时时跳过步骤
            state.add_log_entry(
                agent_name="planner",
                action="timeout",
                details={"error": str(e)}
            )
            return state
    
    async def _navigator_node(self, state: SharedState) -> SharedState:
        """导航器节点包装函数（带超时保护）"""
        logger.info("Executing navigator node")
        try:
            return await ErrorHandler.with_timeout(
                retrieve_content,
                timeout_seconds=60.0,
                state,
                self.retrieval_service,
                self.parser_service,
                self.summarization_service,
                self.workspace_id
            )
        except CustomTimeoutError as e:
            logger.error(f"Navigator node timed out: {str(e)}")
            # 超时时跳过步骤，返回空检索结果
            state.retrieved_docs = []
            state.add_log_entry(
                agent_name="navigator",
                action="timeout",
                details={"error": str(e)}
            )
            return state
    
    async def _director_node(self, state: SharedState) -> SharedState:
        """导演节点包装函数（带超时保护）"""
        logger.info("Executing director node")
        try:
            return await ErrorHandler.with_timeout(
                evaluate_and_decide,
                timeout_seconds=60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Director node timed out: {str(e)}")
            # 超时时假设批准继续
            state.pivot_triggered = False
            state.add_log_entry(
                agent_name="director",
                action="timeout",
                details={"error": str(e), "fallback": "approved"}
            )
            return state
    
    async def _pivot_manager_node(self, state: SharedState) -> SharedState:
        """转向管理器节点包装函数（带超时保护）"""
        logger.info("Executing pivot_manager node")
        try:
            return await ErrorHandler.with_timeout(
                handle_pivot,
                timeout_seconds=30.0,
                state
            )
        except CustomTimeoutError as e:
            logger.error(f"Pivot manager node timed out: {str(e)}")
            # 超时时跳过转向
            state.pivot_triggered = False
            state.add_log_entry(
                agent_name="pivot_manager",
                action="timeout",
                details={"error": str(e)}
            )
            return state
    
    async def _retry_protection_node(self, state: SharedState) -> SharedState:
        """重试保护节点包装函数（带超时保护）"""
        logger.info("Executing retry_protection node")
        try:
            return await ErrorHandler.with_timeout(
                check_retry_limit,
                timeout_seconds=10.0,
                state
            )
        except CustomTimeoutError as e:
            logger.error(f"Retry protection node timed out: {str(e)}")
            # 超时时强制降级
            current_step = state.get_current_step()
            if current_step:
                current_step.status = "skipped"
            state.add_log_entry(
                agent_name="retry_protection",
                action="timeout",
                details={"error": str(e), "fallback": "skipped"}
            )
            return state
    
    async def _writer_node(self, state: SharedState) -> SharedState:
        """编剧节点包装函数（带超时保护）"""
        logger.info("Executing writer node")
        try:
            return await ErrorHandler.with_timeout(
                generate_fragment,
                timeout_seconds=60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Writer node timed out: {str(e)}")
            # 超时时跳过步骤
            current_step = state.get_current_step()
            if current_step:
                current_step.status = "skipped"
            state.add_log_entry(
                agent_name="writer",
                action="timeout",
                details={"error": str(e)}
            )
            return state
    
    async def _fact_checker_node(self, state: SharedState) -> SharedState:
        """事实检查器节点包装函数（带超时保护）"""
        logger.info("Executing fact_checker node")
        try:
            return await ErrorHandler.with_timeout(
                verify_fragment_node,
                timeout_seconds=60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Fact checker node timed out: {str(e)}")
            # 超时时假设片段有效
            state.fact_check_passed = True
            state.add_log_entry(
                agent_name="fact_checker",
                action="timeout",
                details={"error": str(e), "fallback": "assumed_valid"}
            )
            return state
    
    async def _compiler_node(self, state: SharedState) -> str:
        """编译器节点包装函数（带超时保护）"""
        logger.info("Executing compiler node")
        try:
            return await ErrorHandler.with_timeout(
                compile_screenplay,
                timeout_seconds=60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Compiler node timed out: {str(e)}")
            # 超时时返回简单的编译结果
            state.add_log_entry(
                agent_name="compiler",
                action="timeout",
                details={"error": str(e)}
            )
            # 返回简单的片段拼接
            return "\n\n".join([f.content for f in state.fragments])
    
    # Routing functions (需求 14.5)
    
    def _route_director_decision(
        self,
        state: SharedState
    ) -> Literal["pivot", "write"]:
        """
        导演决策路由函数
        
        根据导演的评估结果决定下一步动作：
        - pivot: 检测到冲突或触发条件，需要转向
        - write: 批准内容，继续生成
        
        Args:
            state: 共享状态
            
        Returns:
            路由目标节点名称
            
        验证需求: 14.5
        """
        # 检查是否触发转向
        if state.pivot_triggered:
            logger.info("Director decision: pivot (conflict or trigger detected)")
            return "pivot"
        
        # 默认：批准继续
        logger.info("Director decision: write (approved)")
        return "write"
    
    def _route_fact_check_and_completion(
        self,
        state: SharedState
    ) -> Literal["invalid", "continue", "done"]:
        """
        事实检查器和完成检查组合路由函数
        
        根据事实检查结果和完成状态决定下一步动作：
        - invalid: 片段包含幻觉，需要重新生成
        - continue: 片段有效，还有步骤需要处理，继续循环
        - done: 片段有效，所有步骤完成，进入编译阶段
        
        Args:
            state: 共享状态
            
        Returns:
            路由目标节点名称
            
        验证需求: 14.5
        """
        # 首先检查事实检查结果
        if not state.fact_check_passed:
            logger.info("Fact check decision: invalid (regeneration needed)")
            return "invalid"
        
        # 事实检查通过，检查是否完成
        # 移动到下一步
        state.current_step_index += 1
        
        # 检查是否还有步骤
        if state.current_step_index < len(state.outline):
            logger.info(
                f"Completion check: continue (step {state.current_step_index}/{len(state.outline)})"
            )
            return "continue"
        
        # 所有步骤完成
        logger.info("Completion check: done (all steps completed)")
        return "done"
    
    async def execute(self, state: SharedState) -> Dict[str, Any]:
        """
        执行完整的剧本生成工作流
        
        Args:
            state: 初始共享状态
            
        Returns:
            包含最终剧本和执行日志的字典
            
        验证需求: 14.6
        """
        logger.info("Starting screenplay generation workflow")
        
        try:
            # 执行状态图
            result = await self.graph.ainvoke(state)
            
            logger.info("Screenplay generation workflow completed successfully")
            
            return {
                "success": True,
                "final_screenplay": result if isinstance(result, str) else None,
                "state": state,
                "execution_log": state.execution_log
            }
        
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "state": state,
                "execution_log": state.execution_log
            }
