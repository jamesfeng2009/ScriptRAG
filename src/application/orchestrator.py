"""Workflow Orchestrator - LangGraph state machine management

This module implements the LangGraph state machine that orchestrates
the multi-agent screenplay generation workflow.

Enhanced Features (when enable_dynamic_adjustment=True):
1. RAGContentAnalyzer integration for semantic content analysis
2. DynamicDirector for real-time direction adjustment
3. SkillRecommender for intelligent skill selection
"""

import logging
from typing import List, Literal, Dict, Any, Optional
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
from ..domain.agents.enhanced_agents import (
    RAGContentAnalyzer,
    DynamicDirector,
    SkillRecommender,
    StructuredScreenplayWriter,
    ContentAnalysis,
    DirectionAdjustment,
    DirectionAdjustmentType
)
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
    
    支持两种模式：
    - 基础模式（enable_dynamic_adjustment=False）：标准工作流
    - 增强模式（enable_dynamic_adjustment=True）：支持 RAG 内容分析和动态方向调整
    
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
        workspace_id: str,
        enable_dynamic_adjustment: bool = False
    ):
        """
        初始化工作流编排器
        
        Args:
            llm_service: LLM 服务实例
            retrieval_service: 检索服务实例
            parser_service: 解析服务实例
            summarization_service: 摘要服务实例
            workspace_id: 工作空间 ID
            enable_dynamic_adjustment: 是否启用动态方向调整（默认关闭以保持兼容性）
        """
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.parser_service = parser_service
        self.summarization_service = summarization_service
        self.workspace_id = workspace_id
        self.enable_dynamic_adjustment = enable_dynamic_adjustment
        
        if enable_dynamic_adjustment:
            self.rag_analyzer = RAGContentAnalyzer(llm_service)
            self.dynamic_director = DynamicDirector(llm_service)
            self.skill_recommender = SkillRecommender(llm_service)
            self.screenplay_writer = StructuredScreenplayWriter(llm_service)
        
        self.graph = self._build_graph()
        
        logger.info(f"WorkflowOrchestrator initialized (dynamic_adjustment={enable_dynamic_adjustment})")
    
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态图
        
        根据 enable_dynamic_adjustment 参数决定是否启用增强功能。
        
        Returns:
            编译后的状态图
            
        验证需求: 14.2, 14.3, 14.4, 14.5
        """
        logger.info(f"Building LangGraph state machine (dynamic_adjustment={self.enable_dynamic_adjustment})")
        
        workflow = StateGraph(SharedState)
        
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("navigator", self._navigator_node)
        workflow.add_node("director", self._director_node)
        workflow.add_node("pivot_manager", self._pivot_manager_node)
        workflow.add_node("retry_protection", self._retry_protection_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("fact_checker", self._fact_checker_node)
        workflow.add_node("step_advancer", self._step_advancer_node)
        workflow.add_node("compiler", self._compiler_node)
        
        if self.enable_dynamic_adjustment:
            workflow.add_node("rag_analyzer", self._rag_analyzer_node)
            workflow.add_node("dynamic_director", self._dynamic_director_node)
            workflow.add_node("skill_recommender", self._skill_recommender_node)
        
        workflow.set_entry_point("planner")
        
        workflow.add_edge("planner", "navigator")
        
        if self.enable_dynamic_adjustment:
            workflow.add_edge("navigator", "rag_analyzer")
            workflow.add_edge("rag_analyzer", "dynamic_director")
            
            workflow.add_conditional_edges(
                "dynamic_director",
                self._route_dynamic_director_decision,
                {
                    "pivot": "pivot_manager",
                    "skill_switch": "skill_recommender",
                    "continue": "retry_protection"
                }
            )
            
            workflow.add_edge("skill_recommender", "retry_protection")
        else:
            workflow.add_edge("navigator", "director")
            
            workflow.add_conditional_edges(
                "director",
                self._route_director_decision,
                {"pivot": "pivot_manager", "write": "retry_protection"}
            )
        
        workflow.add_edge("pivot_manager", "navigator")
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
            {"continue": "navigator", "done": "compiler"}
        )
        
        workflow.add_edge("compiler", END)
        
        compiled_graph = workflow.compile()
        
        logger.info("LangGraph state machine built and compiled successfully")
        
        return compiled_graph
    
    async def _rag_analyzer_node(self, state: SharedState) -> SharedState:
        """RAG内容分析器节点"""
        if not hasattr(self, 'rag_analyzer'):
            return state
            
        logger.info("Executing RAG analyzer node")
        try:
            current_step = state.get_current_step()
            if not current_step or not state.retrieved_docs:
                logger.warning("No content to analyze")
                return state
            
            query = current_step.description
            analysis = await self.rag_analyzer.analyze(
                query=query,
                retrieved_docs=state.retrieved_docs
            )
            
            state.rag_analysis = {
                "content_types": [ct.value for ct in analysis.content_types],
                "main_topic": analysis.main_topic,
                "sub_topics": analysis.sub_topics,
                "difficulty_level": analysis.difficulty_level,
                "tone_style": analysis.tone_style.value if analysis.tone_style else None,
                "key_concepts": analysis.key_concepts,
                "warnings": analysis.warnings,
                "prerequisites": analysis.prerequisites,
                "suggested_skill": analysis.suggested_skill,
                "confidence": analysis.confidence
            }
            
            state.add_log_entry(
                agent_name="rag_analyzer",
                action="analyze_content",
                details={
                    "step_id": current_step.step_id,
                    "content_types": analysis.content_types,
                    "difficulty": analysis.difficulty_level,
                    "suggested_skill": analysis.suggested_skill
                }
            )
            
            logger.info(f"RAG analysis completed: types={analysis.content_types}, difficulty={analysis.difficulty_level}")
            
            return state
            
        except Exception as e:
            logger.error(f"RAG analyzer failed: {str(e)}")
            state.add_log_entry(
                agent_name="rag_analyzer",
                action="analysis_failed",
                details={"error": str(e)}
            )
            return state
    
    async def _dynamic_director_node(self, state: SharedState) -> SharedState:
        """动态导演节点 - 基于RAG分析做出方向调整决策"""
        if not hasattr(self, 'dynamic_director'):
            return state
            
        logger.info("Executing dynamic director node")
        try:
            if not state.rag_analysis:
                logger.warning("No RAG analysis available")
                return state
            
            analysis = ContentAnalysis(
                content_types=[],
                main_topic=state.rag_analysis.get("main_topic", ""),
                sub_topics=state.rag_analysis.get("sub_topics", []),
                difficulty_level=state.rag_analysis.get("difficulty_level", 0.5),
                tone_style=None,
                key_concepts=state.rag_analysis.get("key_concepts", []),
                warnings=state.rag_analysis.get("warnings", []),
                prerequisites=state.rag_analysis.get("prerequisites", []),
                suggested_skill=state.rag_analysis.get("suggested_skill"),
                confidence=state.rag_analysis.get("confidence", 0.5)
            )
            
            state, adjustment = await self.dynamic_director.evaluate_and_adjust(
                state=state,
                content_analysis=analysis
            )
            
            state.add_log_entry(
                agent_name="dynamic_director",
                action="direction_adjustment",
                details={
                    "adjustment_type": adjustment.adjustment_type.value if adjustment else "no_change",
                    "reason": adjustment.reason if adjustment else "",
                    "confidence": adjustment.confidence if adjustment else 0
                }
            )
            
            logger.info(f"Dynamic director decision: {adjustment.adjustment_type.value if adjustment else 'no_change'}")
            
            return state
            
        except Exception as e:
            logger.error(f"Dynamic director failed: {str(e)}")
            state.add_log_entry(
                agent_name="dynamic_director",
                action="decision_failed",
                details={"error": str(e)}
            )
            return state
    
    async def _skill_recommender_node(self, state: SharedState) -> SharedState:
        """技能推荐器节点 - 根据内容分析推荐并切换技能"""
        if not hasattr(self, 'skill_recommender'):
            return state
            
        logger.info("Executing skill recommender node")
        try:
            current_step = state.get_current_step()
            if not current_step:
                return state
            
            rag_analysis = getattr(state, 'rag_analysis', None)
            
            recommendation = await self.skill_recommender.recommend(
                topic=state.user_topic,
                context=state.project_context,
                content_analysis=rag_analysis
            )
            
            recommended_skill = recommendation.get("recommended_skill")
            confidence = recommendation.get("confidence", 0)
            reasoning = recommendation.get("reasoning", "")
            
            if recommended_skill and confidence > 0.7 and recommended_skill != state.current_skill:
                old_skill = state.current_skill
                state.current_skill = recommended_skill
                
                if not hasattr(state, 'skill_history') or state.skill_history is None:
                    state.skill_history = []
                
                state.skill_history.append({
                    "step_id": current_step.step_id,
                    "reason": f"Auto-switch: {reasoning}",
                    "from_skill": old_skill,
                    "to_skill": recommended_skill,
                    "triggered_by": "skill_recommender",
                    "confidence": confidence
                })
                
                logger.info(f"Skill auto-switched: {old_skill} -> {recommended_skill} (confidence: {confidence})")
            
            state.add_log_entry(
                agent_name="skill_recommender",
                action="recommend_skill",
                details={
                    "step_id": current_step.step_id,
                    "recommended_skill": recommended_skill,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "current_skill": state.current_skill
                }
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Skill recommender failed: {str(e)}")
            state.add_log_entry(
                agent_name="skill_recommender",
                action="recommendation_failed",
                details={"error": str(e)}
            )
            return state
    
    async def _planner_node(self, state: SharedState) -> SharedState:
        """规划器节点包装函数（带超时保护）"""
        logger.info("Executing planner node")
        try:
            return await ErrorHandler.with_timeout(
                plan_outline,
                60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Planner node timed out: {str(e)}")
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
                60.0,
                state,
                self.retrieval_service,
                self.parser_service,
                self.summarization_service,
                self.workspace_id
            )
        except CustomTimeoutError as e:
            logger.error(f"Navigator node timed out: {str(e)}")
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
                60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Director node timed out: {str(e)}")
            state.pivot_triggered = False
            state.add_log_entry(
                agent_name="director",
                action="timeout",
                details={"error": str(e), "fallback": "approved"}
            )
            return state
    
    def _pivot_manager_node(self, state: SharedState) -> SharedState:
        """转向管理器节点包装函数（带超时保护）"""
        logger.info("Executing pivot_manager node")
        try:
            return handle_pivot(state)
        except Exception as e:
            logger.error(f"Pivot manager node failed: {str(e)}")
            state.pivot_triggered = False
            state.add_log_entry(
                agent_name="pivot_manager",
                action="error",
                details={"error": str(e)}
            )
            return state
    
    def _retry_protection_node(self, state: SharedState) -> SharedState:
        """重试保护节点包装函数（带超时保护）"""
        logger.info("Executing retry_protection node")
        try:
            return check_retry_limit(state)
        except Exception as e:
            logger.error(f"Retry protection node failed: {str(e)}")
            current_step = state.get_current_step()
            if current_step:
                current_step.status = "skipped"
            state.add_log_entry(
                agent_name="retry_protection",
                action="error",
                details={"error": str(e), "fallback": "skipped"}
            )
            return state
    
    async def _writer_node(self, state: SharedState) -> SharedState:
        """编剧节点包装函数（带超时保护）"""
        logger.info("Executing writer node")
        try:
            return await ErrorHandler.with_timeout(
                generate_fragment,
                60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Writer node timed out: {str(e)}")
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
                60.0,
                state,
                self.llm_service
            )
        except CustomTimeoutError as e:
            logger.error(f"Fact checker node timed out: {str(e)}")
            state.fact_check_passed = True
            state.add_log_entry(
                agent_name="fact_checker",
                action="timeout",
                details={"error": str(e), "fallback": "assumed_valid"}
            )
            return state
    
    def _step_advancer_node(self, state: SharedState) -> SharedState:
        """步骤推进器节点 - 将 current_step_index 推进到下一步"""
        old_index = state.current_step_index
        state.current_step_index += 1
        logger.info(f"Advancing from step {old_index} to {state.current_step_index}")
        return state
    
    async def _compiler_node(self, state: SharedState) -> SharedState:
        """编译器节点包装函数（带超时保护）"""
        logger.info("Executing compiler node")
        try:
            final_screenplay = await ErrorHandler.with_timeout(
                compile_screenplay,
                60.0,
                state,
                self.llm_service
            )
            
            state.add_log_entry(
                agent_name="compiler",
                action="screenplay_compiled",
                details={"screenplay_length": len(final_screenplay)}
            )
            
            state.execution_log.append({
                "agent_name": "compiler",
                "action": "final_screenplay",
                "timestamp": state.execution_log[-1]["timestamp"] if state.execution_log else None,
                "details": {"screenplay": final_screenplay}
            })
            
            return state
            
        except CustomTimeoutError as e:
            logger.error(f"Compiler node timed out: {str(e)}")
            state.add_log_entry(
                agent_name="compiler",
                action="timeout",
                details={"error": str(e)}
            )
            fallback_screenplay = "\n\n".join([f.content for f in state.fragments])
            state.execution_log.append({
                "agent_name": "compiler",
                "action": "final_screenplay",
                "timestamp": state.execution_log[-1]["timestamp"] if state.execution_log else None,
                "details": {"screenplay": fallback_screenplay}
            })
            return state
    
    def _route_director_decision(
        self,
        state: SharedState
    ) -> Literal["pivot", "write"]:
        """导演决策路由函数（基础模式）"""
        if state.pivot_triggered:
            logger.info("Director decision: pivot")
            return "pivot"
        logger.info("Director decision: write")
        return "write"
    
    def _route_dynamic_director_decision(
        self,
        state: SharedState
    ) -> Literal["pivot", "skill_switch", "continue"]:
        """
        动态导演决策路由函数（增强模式）
        
        根据动态导演的方向调整决策决定下一步动作：
        - pivot: 检测到冲突或触发条件，需要转向
        - skill_switch: 建议切换技能
        - continue: 正常继续
        """
        if state.pivot_triggered:
            logger.info("Dynamic director decision: pivot")
            return "pivot"
        
        rag_analysis = getattr(state, 'rag_analysis', None)
        if rag_analysis and rag_analysis.get('suggested_skill'):
            if rag_analysis['suggested_skill'] != state.current_skill:
                confidence = rag_analysis.get('confidence', 0)
                if confidence > 0.7:
                    logger.info(f"Dynamic director decision: skill_switch to {rag_analysis['suggested_skill']}")
                    return "skill_switch"
        
        logger.info("Dynamic director decision: continue")
        return "continue"
    
    def _route_fact_check(
        self,
        state: SharedState
    ) -> Literal["invalid", "valid"]:
        """事实检查器路由函数"""
        if not state.fact_check_passed:
            logger.info("Fact check decision: invalid")
            return "invalid"
        logger.info("Fact check decision: valid")
        return "valid"
    
    def _route_completion(
        self,
        state: SharedState
    ) -> Literal["continue", "done"]:
        """完成检查路由函数"""
        if state.current_step_index < len(state.outline):
            logger.info(f"Completion check: continue (step {state.current_step_index}/{len(state.outline)})")
            return "continue"
        logger.info("Completion check: done")
        return "done"
    
    async def execute(
        self,
        state: SharedState,
        recursion_limit: int = 25
    ) -> Dict[str, Any]:
        """
        执行完整的剧本生成工作流
        
        Args:
            state: 初始共享状态
            recursion_limit: LangGraph 最大递归迭代次数（默认 25）
            
        Returns:
            包含最终剧本和执行日志的字典
            
        验证需求: 14.6, 3.1, 3.2, 3.3, 3.5
        """
        logger.info(f"Starting screenplay generation workflow (dynamic_adjustment={self.enable_dynamic_adjustment})")
        
        try:
            result_state_dict = await self.graph.ainvoke(
                state,
                config={"recursion_limit": recursion_limit}
            )
            
            logger.info("Screenplay generation workflow completed successfully")
            
            final_state = SharedState(**result_state_dict)
            
            final_screenplay = None
            for log_entry in reversed(final_state.execution_log):
                if (log_entry.get("agent_name") == "compiler" and 
                    log_entry.get("action") == "final_screenplay"):
                    final_screenplay = log_entry.get("details", {}).get("screenplay")
                    break
            
            return {
                "success": True,
                "final_screenplay": final_screenplay,
                "state": final_state,
                "execution_log": final_state.execution_log
            }
        
        except RecursionError as e:
            error_msg = f"Workflow exceeded recursion limit of {recursion_limit}"
            logger.error(f"{error_msg}: {str(e)}")
            
            return {
                "success": False,
                "error": error_msg,
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


EnhancedWorkflowOrchestrator = WorkflowOrchestrator
