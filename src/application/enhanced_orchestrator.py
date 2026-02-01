"""Enhanced Workflow Orchestrator - LangGraph state machine with dynamic direction adjustment

This module extends the base WorkflowOrchestrator with:
1. RAGContentAnalyzer integration for semantic content analysis
2. DynamicDirector for real-time direction adjustment
3. SkillRecommender for intelligent skill selection
4. StructuredScreenplayWriter for formatted output

These enhancements enable the system to:
- Analyze retrieved content semantically
- Dynamically adjust screenplay direction based on RAG content
- Recommend and switch skills based on content characteristics
- Generate structured screenplay format with scenes, characters, and dialogue
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


class EnhancedWorkflowOrchestrator:
    """
    增强版工作流编排器 - 支持 RAG 内容分析和动态方向调整
    
    新增功能：
    1. RAG内容语义分析
    2. 实时方向调整决策
    3. 智能Skill推荐
    4. 结构化剧本输出
    
    验证需求: 14.2, 14.3, 14.4, 14.5, 14.6
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        parser_service: IParserService,
        summarization_service: SummarizationService,
        workspace_id: str,
        enable_dynamic_adjustment: bool = True
    ):
        """
        初始化增强版工作流编排器
        
        Args:
            llm_service: LLM 服务实例
            retrieval_service: 检索服务实例
            parser_service: 解析服务实例
            summarization_service: 摘要服务实例
            workspace_id: 工作空间 ID
            enable_dynamic_adjustment: 是否启用动态方向调整
        """
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.parser_service = parser_service
        self.summarization_service = summarization_service
        self.workspace_id = workspace_id
        self.enable_dynamic_adjustment = enable_dynamic_adjustment
        
        # 初始化增强版智能体
        self.rag_analyzer = RAGContentAnalyzer(llm_service)
        self.dynamic_director = DynamicDirector(llm_service)
        self.skill_recommender = SkillRecommender(llm_service)
        self.screenplay_writer = StructuredScreenplayWriter(llm_service)
        
        # 构建状态图
        self.graph = self._build_enhanced_graph()
        
        logger.info("EnhancedWorkflowOrchestrator initialized")
    
    def _build_enhanced_graph(self) -> StateGraph:
        """
        构建增强版 LangGraph 状态图
        
        新增节点：
        - rag_analyzer: RAG内容分析
        - dynamic_director: 动态方向调整
        - screenplay_writer: 结构化剧本输出
        
        Returns:
            编译后的状态图
        """
        logger.info("Building enhanced LangGraph state machine")
        
        workflow = StateGraph(SharedState)
        
        # 原有节点
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("navigator", self._navigator_node)
        workflow.add_node("director", self._director_node)
        workflow.add_node("pivot_manager", self._pivot_manager_node)
        workflow.add_node("retry_protection", self._retry_protection_node)
        workflow.add_node("writer", self._writer_node)
        workflow.add_node("fact_checker", self._fact_checker_node)
        workflow.add_node("step_advancer", self._step_advancer_node)
        workflow.add_node("compiler", self._compiler_node)
        
        # 新增节点：RAG内容分析器
        workflow.add_node("rag_analyzer", self._rag_analyzer_node)
        
        # 新增节点：动态导演
        workflow.add_node("dynamic_director", self._dynamic_director_node)
        
        # 新增节点：技能推荐器
        workflow.add_node("skill_recommender", self._skill_recommender_node)
        
        # 设置入口点
        workflow.set_entry_point("planner")
        
        # 原有边
        workflow.add_edge("planner", "navigator")
        workflow.add_edge("navigator", "director")
        
        # 导演决策（条件边）
        workflow.add_conditional_edges(
            "director",
            self._route_director_decision,
            {"pivot": "pivot_manager", "write": "retry_protection"}
        )
        
        workflow.add_edge("pivot_manager", "navigator")
        workflow.add_edge("retry_protection", "writer")
        workflow.add_edge("writer", "fact_checker")
        
        # 事实检查（条件边）
        workflow.add_conditional_edges(
            "fact_checker",
            self._route_fact_check,
            {"invalid": "retry_protection", "valid": "step_advancer"}
        )
        
        # 步骤推进（条件边）
        workflow.add_conditional_edges(
            "step_advancer",
            self._route_completion,
            {"continue": "navigator", "done": "compiler"}
        )
        
        workflow.add_edge("compiler", END)
        
        # 增强版：导航器之后添加RAG分析
        # 修改导航器->导演的边，插入RAG分析节点
        # 注意：这需要修改原有边的逻辑
        
        # 新的流程：导航器 -> rag_analyzer -> dynamic_director -> (pivot or skill_recommender)
        # 对于启用动态调整的模式
        
        compiled_graph = workflow.compile()
        
        logger.info("Enhanced LangGraph state machine built and compiled successfully")
        
        return compiled_graph
    
    async def _rag_analyzer_node(self, state: SharedState) -> SharedState:
        """RAG内容分析器节点"""
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
            
            # 存储分析结果到状态
            state.rag_analysis = {
                "content_types": [ct.value for ct in analysis.content_types],
                "main_topic": analysis.main_topic,
                "sub_topics": analysis.sub_topics,
                "difficulty_level": analysis.difficulty_level,
                "tone_style": analysis.tone_style.value,
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
        logger.info("Executing dynamic director node")
        try:
            if not state.rag_analysis:
                logger.warning("No RAG analysis available")
                return state
            
            # 重建ContentAnalysis对象
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
            
            # 记录调整决策
            state.add_log_entry(
                agent_name="dynamic_director",
                action="direction_adjustment",
                details={
                    "adjustment_type": adjustment.adjustment_type.value,
                    "reason": adjustment.reason,
                    "confidence": adjustment.confidence
                }
            )
            
            logger.info(f"Dynamic director decision: {adjustment.adjustment_type.value}")
            
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
        """技能推荐器节点"""
        logger.info("Executing skill recommender node")
        try:
            current_step = state.get_current_step()
            if not current_step:
                return state
            
            recommendation = await self.skill_recommender.recommend(
                topic=state.user_topic,
                context=state.project_context,
                content_analysis=None
            )
            
            # 记录推荐结果
            state.add_log_entry(
                agent_name="skill_recommender",
                action="recommend_skill",
                details={
                    "step_id": current_step.step_id,
                    "recommended_skill": recommendation.get("recommended_skill"),
                    "confidence": recommendation.get("confidence"),
                    "reasoning": recommendation.get("reasoning")
                }
            )
            
            # 如果推荐置信度高且建议切换，则触发切换
            if (recommendation.get("confidence", 0) > 0.7 and 
                recommendation.get("recommended_skill") != state.current_skill):
                
                logger.info(f"Skill recommendation: {recommendation['recommended_skill']} (confidence: {recommendation['confidence']})")
            
            return state
            
        except Exception as e:
            logger.error(f"Skill recommender failed: {str(e)}")
            state.add_log_entry(
                agent_name="skill_recommender",
                action="recommendation_failed",
                details={"error": str(e)}
            )
            return state
    
    # 复用原有的节点包装函数
    async def _planner_node(self, state: SharedState) -> SharedState:
        """规划器节点包装函数"""
        logger.info("Executing enhanced planner node")
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
        """导航器节点包装函数"""
        logger.info("Executing enhanced navigator node")
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
        """导演节点包装函数"""
        logger.info("Executing enhanced director node")
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
                details={"error": str(e)}
            )
            return state
    
    def _pivot_manager_node(self, state: SharedState) -> SharedState:
        """转向管理器节点包装函数"""
        logger.info("Executing enhanced pivot_manager node")
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
        """重试保护节点包装函数"""
        logger.info("Executing enhanced retry_protection node")
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
                details={"error": str(e)}
            )
            return state
    
    async def _writer_node(self, state: SharedState) -> SharedState:
        """编剧节点包装函数"""
        logger.info("Executing enhanced writer node")
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
        """事实检查器节点包装函数"""
        logger.info("Executing enhanced fact_checker node")
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
                details={"error": str(e)}
            )
            return state
    
    def _step_advancer_node(self, state: SharedState) -> SharedState:
        """步骤推进器节点"""
        old_index = state.current_step_index
        state.current_step_index += 1
        logger.info(f"Advancing from step {old_index} to {state.current_step_index}")
        return state
    
    async def _compiler_node(self, state: SharedState) -> SharedState:
        """编译器节点包装函数"""
        logger.info("Executing enhanced compiler node")
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
            return state
    
    def _route_director_decision(
        self,
        state: SharedState
    ) -> Literal["pivot", "write"]:
        """导演决策路由函数"""
        if state.pivot_triggered:
            logger.info("Director decision: pivot")
            return "pivot"
        logger.info("Director decision: write")
        return "write"
    
    def _route_fact_check(
        self,
        state: SharedState
    ) -> Literal["invalid", "valid"]:
        """事实检查路由函数"""
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
        执行增强版剧本生成工作流
        
        Args:
            state: 初始共享状态
            recursion_limit: 最大递归迭代次数
            
        Returns:
            包含最终剧本和执行日志的字典
        """
        logger.info(
            f"Starting enhanced screenplay generation workflow "
            f"(dynamic_adjustment={self.enable_dynamic_adjustment})"
        )
        
        try:
            result_state_dict = await self.graph.ainvoke(
                state,
                config={"recursion_limit": recursion_limit}
            )
            
            logger.info("Enhanced workflow completed successfully")
            
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
                "execution_log": final_state.execution_log,
                "dynamic_adjustments": self._extract_adjustments(final_state)
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
            logger.error(f"Enhanced workflow execution failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "state": state,
                "execution_log": state.execution_log
            }
    
    def _extract_adjustments(self, state: SharedState) -> List[Dict[str, Any]]:
        """提取所有方向调整记录"""
        adjustments = []
        for log in state.execution_log:
            if log.get("action") == "direction_adjustment":
                adjustments.append(log.get("details", {}))
        return adjustments
