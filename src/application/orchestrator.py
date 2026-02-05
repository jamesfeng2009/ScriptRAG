"""
工作流编排器 - LangGraph 状态机管理

该模块实现了 LangGraph 状态机，用于编排多智能体剧本生成工作流。

增强功能（当 enable_dynamic_adjustment=True 时）：
1. RAGContentAnalyzer 集成用于语义内容分析
2. DynamicDirector 用于实时方向调整
3. SkillRecommender 用于智能技能选择

v2.1 架构变更：
- 使用 GlobalState TypedDict 替代 SharedState
- 采用 Reducer 模式进行状态更新
- 函数级隔离和错误处理标准化
"""

import logging
import operator
from datetime import datetime
from typing import List, Literal, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END

from ..domain.state_types import GlobalState
from ..domain.agents.node_factory import NodeFactory, create_success_log, create_error_log
from ..domain.agents.enhanced_agents import (
    RAGContentAnalyzer,
    DynamicDirector,
    SkillRecommender,
    StructuredScreenplayWriter,
    ContentAnalysis,
    DirectionAdjustment,
    DirectionAdjustmentType
)
from ..domain.agents.agent_collaboration import (
    AgentNegotiator,
    ParallelAgentExecutor,
    AgentExecutionTracer,
    AgentReflection,
    CollaborationManager
)
from ..services.llm.service import LLMService
from ..services.retrieval_service import RetrievalService
from ..services.parser.tree_sitter_parser import IParserService
from ..services.summarization_service import SummarizationService
from ..services.session_manager import SessionManager
from ..infrastructure.langgraph_error_handler import (
    with_error_handling,
)
from .base_orchestrator import BaseWorkflowOrchestrator


logger = logging.getLogger(__name__)


class WorkflowOrchestrator(BaseWorkflowOrchestrator):
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
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        parser_service: IParserService,
        summarization_service: SummarizationService,
        workspace_id: str,
        enable_dynamic_adjustment: bool = False,
        session_manager: Optional[SessionManager] = None,
        auto_save_sessions: bool = False,
        save_interval_steps: int = 5,
        enable_task_stack: bool = False,
        max_task_depth: int = 3
    ):
        """
        初始化工作流编排器

        参数:
            llm_service: LLM 服务实例
            retrieval_service: 检索服务实例
            parser_service: 解析服务实例
            summarization_service: 摘要服务实例
            workspace_id: 工作空间 ID
            enable_dynamic_adjustment: 是否启用动态方向调整（默认关闭以保持兼容性）
            session_manager: 会话管理器实例（可选，用于断点续传）
            auto_save_sessions: 是否自动保存会话状态
            save_interval_steps: 自动保存间隔（步数）
            enable_task_stack: 是否启用 Task Stack（用于嵌套任务管理，默认关闭）
            max_task_depth: Task Stack 最大嵌套深度
        """
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.parser_service = parser_service
        self.summarization_service = summarization_service
        self.workspace_id = workspace_id
        self.enable_dynamic_adjustment = enable_dynamic_adjustment
        self.session_manager = session_manager
        self.auto_save_sessions = auto_save_sessions
        self.save_interval_steps = save_interval_steps
        self.enable_task_stack = enable_task_stack
        
        self.node_factory = NodeFactory(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            workspace_id=workspace_id,
            use_task_stack=enable_task_stack,
            max_task_depth=max_task_depth
        )
        
        super().__init__(self.node_factory)
        
        if enable_dynamic_adjustment:
            self.rag_analyzer = RAGContentAnalyzer(llm_service)
            self.dynamic_director = DynamicDirector(llm_service)
            self.skill_recommender = SkillRecommender(llm_service)
            self.screenplay_writer = StructuredScreenplayWriter(llm_service)

        self.collaboration_manager = CollaborationManager(llm_service)
        self.execution_tracer = AgentExecutionTracer()
        self.agent_reflection = AgentReflection(llm_service)

        self.graph = self._build_graph()

        logger.info(
            f"WorkflowOrchestrator 初始化完成 "
            f"(dynamic_adjustment={enable_dynamic_adjustment}, "
            f"task_stack={enable_task_stack}, max_depth={max_task_depth})"
        )
    
    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态图
        
        根据 enable_dynamic_adjustment 参数决定是否启用增强功能。
        
        返回:
            编译后的状态图
            
        验证需求: 14.2, 14.3, 14.4, 14.5
        """
        logger.info(f"构建 LangGraph 状态机 (dynamic_adjustment={self.enable_dynamic_adjustment})")
        
        workflow = StateGraph(GlobalState)
        
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

        workflow.add_node("collaboration_manager", self._collaboration_manager_node)
        workflow.add_node("execution_tracer", self._execution_tracer_node)
        workflow.add_node("agent_reflection", self._agent_reflection_node)
        
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
                {
                    "pivot": "pivot_manager", 
                    "navigate": "navigator",
                    "write": "retry_protection"
                }
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
        
        logger.info("LangGraph 状态机构建并编译成功")
        
        return compiled_graph
    
    def _get_current_step(self, state: GlobalState) -> Optional[Dict[str, Any]]:
        """获取当前步骤"""
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        
        if 0 <= current_step_index < len(outline):
            return outline[current_step_index]
        return None
    
    def _get_state_value(self, state: GlobalState, key: str, default: Any = None) -> Any:
        """安全地从状态中获取值，支持字典和Pydantic模型"""
        if isinstance(state, dict):
            return state.get(key, default)
        else:
            return getattr(state, key, default)
    
    def _get_director_feedback(self, state: GlobalState) -> Dict[str, Any]:
        """安全地获取导演反馈"""
        feedback = self._get_state_value(state, "director_feedback", {})
        if isinstance(feedback, dict):
            return feedback
        return {}
    
    def _route_director_decision(self, state: GlobalState) -> str:
        """
        路由导演决策
        
        参数:
            state: 当前全局状态
            
        返回:
            路由目标节点标识
        """
        director_feedback = self._get_director_feedback(state)
        trigger_retrieval = director_feedback.get("trigger_retrieval", False)
        
        if trigger_retrieval:
            return "navigate"
        
        decision = director_feedback.get("decision", None)
        if decision:
            if decision == "continue":
                decision = "write"
            elif decision == "retry":
                decision = "navigate"
            return decision
        
        pivot_triggered = self._get_state_value(state, "pivot_triggered", False)
        if pivot_triggered:
            return "pivot"
        
        return "write"
    
    def _route_dynamic_director_decision(self, state: GlobalState) -> str:
        """
        路由动态导演决策
        
        参数:
            state: 当前全局状态
            
        返回:
            路由目标节点标识
        """
        director_feedback = self._get_director_feedback(state)
        adjustment_type = director_feedback.get("adjustment_type", "continue")
        
        if adjustment_type == "pivot":
            return "pivot"
        elif adjustment_type == "skill_switch":
            return "skill_switch"
        else:
            return "continue"
    
    def _route_fact_check(self, state: GlobalState) -> str:
        """
        路由事实检查结果
        
        参数:
            state: 当前全局状态
            
        返回:
            路由目标节点标识
        """
        fact_check_passed = self._get_state_value(state, "fact_check_passed", False)
        logger.info(f"route_fact_check: fact_check_passed={fact_check_passed}")
        return "valid" if fact_check_passed else "invalid"
    
    def _route_completion(self, state: GlobalState) -> str:
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        workflow_complete = self._get_state_value(state, "workflow_complete", False)
        
        logger.info(f"route_completion: index={current_step_index}, outline_len={len(outline)}, complete={workflow_complete}")
        
        if workflow_complete or current_step_index >= len(outline):
            logger.info("route_completion: returning 'done'")
            return "done"
        logger.info("route_completion: returning 'continue'")
        return "continue"
    
    @with_error_handling(agent_name="rag_analyzer", action_name="analyze_content")
    async def _rag_analyzer_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        RAG内容分析器节点
        
        职责:
            - 分析检索到的内容语义
            - 识别内容类型、难度级别
            - 提供写作方向建议
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行 RAG 分析器节点")
        
        current_step = self._get_current_step(state)
        retrieved_docs = self._get_state_value(state, "last_retrieved_docs", [])
        
        if not current_step or not retrieved_docs:
            logger.warning("没有内容可分析")
            return {}
        
        query = current_step.get("description", "")
        analysis = await self.rag_analyzer.analyze(
            query=query,
            retrieved_docs=retrieved_docs
        )
        
        rag_analysis = {
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
        
        return {
            "director_feedback": {"rag_analysis": rag_analysis},
            "execution_log": create_success_log(
                agent="rag_analyzer",
                action_name="analyze_content",
                details={
                    "step_id": current_step.get("step_id"),
                    "content_types": analysis.content_types,
                    "difficulty": analysis.difficulty_level,
                    "suggested_skill": analysis.suggested_skill
                }
            )
        }
    
    @with_error_handling(agent_name="dynamic_director", action_name="direction_adjustment")
    async def _dynamic_director_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        动态导演节点 - 基于RAG分析做出方向调整决策
        
        职责:
            - 评估 RAG 分析结果
            - 决定是否需要调整写作方向
            - 生成具体的调整指令
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行动态导演节点")
        
        director_feedback = self._get_director_feedback(state)
        rag_analysis = director_feedback.get("rag_analysis", {})
        
        if not rag_analysis:
            logger.warning("没有 RAG 分析结果可用")
            return {}
        
        analysis = ContentAnalysis(
            content_types=[],
            main_topic=rag_analysis.get("main_topic", ""),
            sub_topics=rag_analysis.get("sub_topics", []),
            difficulty_level=rag_analysis.get("difficulty_level", 0.5),
            tone_style=None,
            key_concepts=rag_analysis.get("key_concepts", []),
            warnings=rag_analysis.get("warnings", []),
            prerequisites=rag_analysis.get("prerequisites", []),
            suggested_skill=rag_analysis.get("suggested_skill"),
            confidence=rag_analysis.get("confidence", 0.5)
        )
        
        current_step = self._get_current_step(state)
        adjustment = await self.dynamic_director.evaluate_and_adjust(
            state=state,
            content_analysis=analysis
        )
        
        return {
            "director_feedback": {
                "adjustment_type": adjustment.adjustment_type.value if adjustment else "no_change",
                "reason": adjustment.reason if adjustment else "",
                "confidence": adjustment.confidence if adjustment else 0
            },
            "execution_log": create_success_log(
                agent="dynamic_director",
                action_name="direction_adjustment",
                details={
                    "adjustment_type": adjustment.adjustment_type.value if adjustment else "no_change",
                    "reason": adjustment.reason if adjustment else "",
                    "confidence": adjustment.confidence if adjustment else 0
                }
            )
        }
    
    @with_error_handling(agent_name="skill_recommender", action_name="recommend_skill")
    async def _skill_recommender_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        技能推荐器节点 - 根据内容分析推荐并切换技能
        
        职责:
            - 分析当前上下文和内容特征
            - 推荐最适合的写作技能
            - 自动切换技能（如需要）
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行技能推荐器节点")
        
        current_step = self._get_current_step(state)
        if not current_step:
            return {}
        
        user_topic = self._get_state_value(state, "user_topic", "")
        project_context = self._get_state_value(state, "project_context", "")
        director_feedback = self._get_director_feedback(state)
        rag_analysis = director_feedback.get("rag_analysis", {})
        
        recommendation = await self.skill_recommender.recommend(
            topic=user_topic,
            context=project_context,
            content_analysis=rag_analysis
        )
        
        recommended_skill = recommendation.get("recommended_skill")
        confidence = recommendation.get("confidence", 0)
        reasoning = recommendation.get("reasoning", "")
        
        updates = {}
        logs = []
        
        if recommended_skill and confidence > 0.7:
            updates["current_skill"] = recommended_skill
            
            skill_history = self._get_state_value(state, "skill_history", [])
            skill_history.append({
                "step_id": current_step.get("step_id"),
                "reason": f"自动切换: {reasoning}",
                "to_skill": recommended_skill,
                "triggered_by": "skill_recommender",
                "confidence": confidence
            })
            updates["skill_history"] = skill_history
            
            logger.info(f"技能自动切换: -> {recommended_skill} (置信度: {confidence})")
        
        logs.append(create_success_log(
            agent="skill_recommender",
            action_name="recommend_skill",
            details={
                "step_id": current_step.get("step_id"),
                "recommended_skill": recommended_skill,
                "confidence": confidence,
                "reasoning": reasoning
            }
        ))
        
        updates["execution_log"] = logs
        return updates
    
    @with_error_handling(agent_name="planner", action_name="generate_outline")
    async def _planner_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        规划器节点 - 生成剧本大纲
        
        职责:
            - 分析用户主题和项目上下文
            - 生成结构化的大纲
            - 设置初始步骤索引
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典（包含 outline 和 execution_log）
        """
        logger.info("执行规划器节点")
        
        user_topic = self._get_state_value(state, "user_topic", "")
        project_context = self._get_state_value(state, "project_context", "")
        
        return await self.node_factory.planner_node(state)
    
    @with_error_handling(agent_name="navigator", action_name="retrieve_content")
    async def _navigator_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        导航器节点 - 检索相关内容
        
        职责:
            - 根据当前步骤检索相关内容
            - 解析和摘要检索结果
            - 更新检索文档列表
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行导航器节点")
        
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
        
        return await self.node_factory.navigator_node(state)
    
    @with_error_handling(agent_name="director", action_name="evaluate_and_decide")
    async def _director_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        导演节点 - 评估并决定下一步行动
        
        职责:
            - 评估当前步骤和检索内容
            - 决定是转向、写作还是继续
            - 设置导演反馈标志
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行导演节点")
        
        return await self.node_factory.director_node(state)
    
    @with_error_handling(agent_name="pivot_manager", action_name="handle_pivot")
    async def _pivot_manager_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        转向管理器节点 - 处理步骤转向
        
        职责:
            - 执行转向逻辑
            - 重置转向标志
            - 记录转向历史
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行转向管理器节点")
        
        return await self.node_factory.pivot_manager_node(state)
    
    @with_error_handling(agent_name="retry_protection", action_name="check_retry_limit")
    async def _retry_protection_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        重试保护节点 - 检查重试次数限制
        
        职责:
            - 检查当前步骤重试次数
            - 阻止超过限制的重试
            - 标记超限步骤
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行重试保护节点")
        
        return await self.node_factory.retry_protection_node(state)
    
    @with_error_handling(agent_name="writer", action_name="generate_fragment")
    async def _writer_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        编剧节点 - 生成剧本片段
        
        职责:
            - 根据大纲步骤生成剧本内容
            - 追加到片段列表
            - 设置最后生成的片段
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典（包含 fragments 和 execution_log）
        """
        logger.info("执行编剧节点")
        
        return await self.node_factory.writer_node(state)
    
    @with_error_handling(agent_name="fact_checker", action_name="verify_fragment")
    async def _fact_checker_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        事实检查器节点 - 验证生成内容的准确性
        
        职责:
            - 检查片段中的事实准确性
            - 标记验证结果
            - 记录检查日志
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行事实检查器节点")
        
        return await self.node_factory.fact_checker_node(state)
    
    def _step_advancer_node(self, state: GlobalState) -> Dict[str, Any]:
        old_index = self._get_state_value(state, "current_step_index", 0)
        outline = self._get_state_value(state, "outline", [])
        skip_current_step = self._get_state_value(state, "skip_current_step", False)
        
        is_complete = old_index >= len(outline)
        
        if skip_current_step or is_complete:
            logger.info(f"推进步骤: {old_index} -> {old_index}, complete=True (workflow done)")
            result = {
                "current_step_index": old_index,
                "execution_log": create_success_log(
                    agent="step_advancer",
                    action="advance_step",
                    details={"from_index": old_index, "to_index": old_index, "is_complete": True}
                ),
                "workflow_complete": True
            }
            return result
        
        new_index = old_index + 1
        is_complete_after = new_index >= len(outline)
        
        logger.info(f"推进步骤: {old_index} -> {new_index}, complete={is_complete_after}")
        
        result = {
            "current_step_index": new_index,
            "execution_log": create_success_log(
                agent="step_advancer",
                action="advance_step",
                details={"from_index": old_index, "to_index": new_index, "is_complete": is_complete_after}
            )
        }
        
        if is_complete_after:
            result["workflow_complete"] = True
            
        return result
    
    @with_error_handling(agent_name="compiler", action_name="compile_screenplay")
    async def _compiler_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        编译器节点 - 编译最终剧本
        
        职责:
            - 编译所有片段成最终剧本
            - 添加审计日志
            - 返回最终结果
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行编译器节点")
        
        return await self.node_factory.compiler_node(state)
    
    async def execute(
        self,
        initial_state: Dict[str, Any],
        session_id: Optional[str] = None,
        recursion_limit: int = 25
    ) -> Dict[str, Any]:
        """
        执行工作流

        支持断点续传功能：
        1. 如果提供了 session_id，尝试恢复之前的会话状态
        2. 执行过程中自动保存会话状态（如果启用）
        3. 执行完成后保存最终状态

        参数:
            initial_state: 初始状态字典
            session_id: 会话 ID（用于恢复之前的会话）
            recursion_limit: 递归限制（默认25）

        返回:
            包含success、session_id和最终状态的字典

        使用示例:
            # 启动新会话
            result = await orchestrator.execute(initial_state)
            print(f"Session ID: {result['session_id']}")

            # 恢复之前的会话
            restored = await orchestrator.execute(
                initial_state,
                session_id=result['session_id']
            )
        """
        current_session_id = session_id

        try:
            if current_session_id and self.session_manager:
                restored_state = await self.session_manager.load_session(current_session_id)
                if restored_state:
                    logger.info(f"从会话 {current_session_id} 恢复，继续执行")
                    initial_state = restored_state
                    initial_state["resumed_from_session"] = current_session_id
                else:
                    logger.warning(f"会话 {current_session_id} 不存在，将创建新会话")
                    current_session_id = None

            logger.info(f"开始执行工作流，recursion_limit={recursion_limit}")

            config = {
                "recursion_limit": recursion_limit,
                "configurable": {
                    "thread_id": self.workspace_id
                }
            }

            final_state = await self.graph.ainvoke(initial_state, config=config)

            if self.session_manager and current_session_id:
                await self.session_manager.save_session(
                    current_session_id,
                    final_state,
                    self.workspace_id
                )

            logger.info("工作流执行完成")
            return {
                "success": True,
                "session_id": current_session_id,
                "state": final_state
            }

        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")

            if self.session_manager and current_session_id:
                try:
                    await self.session_manager.save_session(
                        current_session_id,
                        initial_state,
                        self.workspace_id
                    )
                    logger.info(f"已保存失败状态到会话 {current_session_id}")
                except Exception as save_error:
                    logger.error(f"保存失败状态失败: {str(save_error)}")

            return {
                "success": False,
                "session_id": current_session_id,
                "error": str(e),
                "state": initial_state
            }

    async def save_checkpoint(
        self,
        state: Dict[str, Any],
        session_id: str
    ) -> bool:
        """
        保存检查点

        用于在执行过程中手动保存状态。

        Args:
            state: 当前状态
            session_id: 会话 ID

        Returns:
            是否保存成功
        """
        if not self.session_manager:
            logger.warning("未配置 session_manager，无法保存检查点")
            return False

        return await self.session_manager.save_session(
            session_id,
            state,
            self.workspace_id
        )
    
    async def execute_streaming(self, initial_state: Dict[str, Any]):
        """
        流式执行工作流
        
        参数:
            initial_state: 初始状态字典
            
        返回:
            状态更新流
        """
        logger.info("开始流式执行工作流")
        
        config = {
            "configurable": {
                "thread_id": self.workspace_id
            }
        }
        
        async for state_update in self.graph.astream(initial_state, config=config):
            yield state_update
        
        logger.info("流式执行工作流完成")

    @with_error_handling(agent_name="collaboration_manager", action_name="manage_collaboration")
    async def _collaboration_manager_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        协作管理器节点

        职责：
            - 管理 Agent 之间的协作
            - 处理技能切换协商
            - 执行并行任务
        """
        try:
            negotiation_result = await self.collaboration_manager.negotiate_skill_switch(
                director_recommendation=state.get("director_recommendation", ""),
                writer_preference=state.get("writer_preference", ""),
                content_analysis=state.get("content_analysis", {}),
                query=state.get("query", ""),
                current_state=dict(state)
            )

            return {
                "collaboration_result": {
                    "negotiation_status": negotiation_result.status.value,
                    "negotiated_skill": negotiation_result.negotiated_skill,
                    "decision_type": negotiation_result.decision_type.value,
                    "reason": negotiation_result.reason
                },
                "negotiation_history": state.get("negotiation_history", []) + [dict(negotiation_result)]
            }

        except Exception as e:
            logger.error(f"协作管理失败: {str(e)}")
            return {"collaboration_error": str(e)}

    @with_error_handling(agent_name="execution_tracer", action_name="trace_execution")
    async def _execution_tracer_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        执行追踪器节点

        职责：
            - 记录 Agent 执行链
            - 追踪决策过程
            - 分析执行模式
        """
        try:
            current_agent = state.get("current_agent", "unknown")
            input_state = {
                "query": state.get("query", ""),
                "step_index": state.get("current_step_index", 0)
            }

            execution_node = await self.execution_tracer.trace_decision(
                agent_name=current_agent,
                input_state=input_state,
                output_state=dict(state),
                decision_reason=state.get("director_feedback", {}).get("reason", "执行完成"),
                execution_time_ms=state.get("execution_time_ms", 0)
            )

            chain_visualization = self.execution_tracer.visualize_chain()
            stats = self.execution_tracer.get_stats()

            return {
                "execution_node": execution_node,
                "execution_chain_visualization": chain_visualization,
                "execution_stats": stats
            }

        except Exception as e:
            logger.error(f"执行追踪失败: {str(e)}")
            return {"tracing_error": str(e)}

    @with_error_handling(agent_name="agent_reflection", action_name="reflect_on_failure")
    async def _agent_reflection_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        自我反思节点

        职责：
            - 从失败中学习
            - 分析失败模式
            - 生成调整建议
        """
        try:
            recent_failures = state.get("recent_failures", [])
            current_agent = state.get("current_agent", "unknown")

            if not recent_failures:
                return {"reflection_result": None}

            reflection_result = await self.agent_reflection.reflect_on_failure(
                agent_name=current_agent,
                failure_context={
                    "failures": recent_failures,
                    "query": state.get("query", ""),
                    "step_index": state.get("current_step_index", 0)
                },
                state=dict(state)
            )

            return {
                "reflection_result": {
                    "reflection_id": reflection_result.reflection_id,
                    "failure_pattern": reflection_result.failure_pattern,
                    "suggested_adjustments": reflection_result.suggested_adjustments,
                    "confidence": reflection_result.confidence,
                    "reason": reflection_result.reason
                },
                "agent_reflection_history": state.get("agent_reflection_history", []) + [dict(reflection_result)]
            }

        except Exception as e:
            logger.error(f"自我反思失败: {str(e)}")
            return {"reflection_error": str(e)}
