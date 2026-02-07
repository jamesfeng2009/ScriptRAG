"""
工作流编排器 - LangGraph 状态机管理

该模块实现了 LangGraph 状态机，用于编排多智能体剧本生成工作流。

增强功能（Agentic RAG 集成）：
1. IntentParserAgent 集成用于意图解析和查询增强
2. QualityEvalAgent 集成用于质量评估和自适应检索
3. 基于质量评估的条件重试循环

v2.1 架构变更：
- 使用 GlobalState TypedDict 替代 SharedState
- 采用 Reducer 模式进行状态更新
- 函数级隔离和错误处理标准化
- Agentic RAG：意图解析 + 质量评估 + 自适应检索
"""

import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END

from ..domain.state_types import GlobalState
from ..domain.agents.node_factory import NodeFactory, create_success_log
from ..domain.agents.navigator import RetrievedDocument
from ..domain.tools.tool_service import ToolService
from ..domain.tools.tool_executor import ToolExecutor
from ..domain.agents.editor_agent import EditorAgent
from ..domain.agents.enhanced_agents import (
    RAGContentAnalyzer,
    DynamicDirector,
    SkillRecommender,
    StructuredScreenplayWriter,
    ContentAnalysis
)
from ..domain.agents.agent_collaboration import (
    AgentExecutionTracer,
    AgentReflection,
    CollaborationManager
)
from ..domain.agents.intent_parser import IntentParserAgent, IntentAnalysis
from ..domain.agents.quality_eval import QualityEvalAgent
from ..domain.agents.navigator import retrieve_content
from ..services.llm.service import LLMService
from ..services.retrieval_service import RetrievalService
from ..services.parser.tree_sitter_parser import IParserService
from ..services.core.summarization_service import SummarizationService
from ..infrastructure.langgraph_error_handler import (
    with_error_handling,
)
from ..infrastructure.logging import get_agent_logger
from ..services.persistence.agent_execution_persistence_service import (
    agent_execution_service,
    AgentExecutionRecord
)
from .base_orchestrator import BaseWorkflowOrchestrator


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


class WorkflowOrchestrator(BaseWorkflowOrchestrator):
    """
    工作流编排器 - 管理 LangGraph 状态机
    
    支持三种模式：
    - 基础模式（enable_agentic_rag=False）：标准工作流
    - Agentic RAG 模式（enable_agentic_rag=True）：意图解析 + 质量评估 + 自适应检索
    - 增强模式（enable_dynamic_adjustment=True）：RAG 内容分析和动态方向调整
    
    Agentic RAG 工作流：
        planner → intent_parser → navigator → quality_eval → director
                                            ↘              ↗
                                              如果质量差 → 重试
    
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
        enable_dynamic_adjustment: bool = False,
        enable_agentic_rag: bool = True,
        enable_task_stack: bool = False,
        max_task_depth: int = 3,
        enable_tools: bool = False,
        tool_max_iterations: int = 10,
        max_retrieval_retries: int = 2
    ):
        """
        初始化工作流编排器

        参数:
            llm_service: LLM 服务实例
            retrieval_service: 检索服务实例
            parser_service: 解析服务实例
            summarization_service: 摘要服务实例
            enable_dynamic_adjustment: 是否启用动态方向调整（默认关闭以保持兼容性）
            enable_agentic_rag: 是否启用 Agentic RAG（默认启用：意图解析 + 质量评估）
            enable_task_stack: 是否启用 Task Stack（用于嵌套任务管理，默认关闭）
            max_task_depth: Task Stack 最大嵌套深度
            enable_tools: 是否启用工具服务（用于 Function Calling，默认关闭）
            tool_max_iterations: 工具调用最大迭代次数
            max_retrieval_retries: 最大检索重试次数（Agentic RAG）
        """
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.parser_service = parser_service
        self.summarization_service = summarization_service
        self.enable_dynamic_adjustment = enable_dynamic_adjustment
        self.enable_agentic_rag = enable_agentic_rag
        self.max_retrieval_retries = max_retrieval_retries
        
        self.node_factory = NodeFactory(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service,
            use_task_stack=enable_task_stack,
            max_task_depth=max_task_depth
        )
        
        super().__init__(self.node_factory)
        
        if enable_agentic_rag:
            self.intent_parser = IntentParserAgent(llm_service)
            self.quality_eval_agent = QualityEvalAgent(llm_service)
            logger.info(
                f"Agentic RAG 初始化完成 "
                f"(intent_parser=True, quality_eval=True, max_retries={max_retrieval_retries})"
            )
        
        if enable_dynamic_adjustment:
            self.rag_analyzer = RAGContentAnalyzer(llm_service)
            self.dynamic_director = DynamicDirector(llm_service)
            self.skill_recommender = SkillRecommender(llm_service)
            self.screenplay_writer = StructuredScreenplayWriter(llm_service)

        self.collaboration_manager = CollaborationManager(llm_service)
        self.execution_tracer = AgentExecutionTracer()
        self.agent_reflection = AgentReflection(llm_service)
        self.enable_tools = enable_tools
        
        if enable_tools:
            self.tool_executor = ToolExecutor(
                llm_service=llm_service,
                retrieval_service=retrieval_service,
                node_factory=self.node_factory
            )
            self.tool_service = ToolService(
                llm_service=llm_service,
                tool_executor=self.tool_executor,
                max_iterations=tool_max_iterations
            )
            self.editor_agent = EditorAgent(tool_service=self.tool_service)
            logger.info(
                f"ToolService 和 ToolExecutor 初始化完成 "
                f"(enable_tools=True, max_iterations={tool_max_iterations})"
            )

        self.graph = self._build_graph()

        logger.info(
            f"WorkflowOrchestrator 初始化完成 "
            f"(agentic_rag={enable_agentic_rag}, dynamic_adjustment={enable_dynamic_adjustment}, "
            f"task_stack={enable_task_stack}, max_depth={max_task_depth}, "
            f"tools={enable_tools})"
        )

    async def _record_agent_execution(
        self,
        agent_name: str,
        node_name: str,
        state: GlobalState,
        updates: Dict[str, Any],
        execution_time_ms: Optional[float] = None,
        error: Optional[str] = None
    ):
        """记录 agent 执行情况到数据库"""
        try:
            task_id = self._get_state_value(state, "task_id", None)
            chat_session_id = self._get_state_value(state, "chat_session_id", None)
            current_step_index = self._get_state_value(state, "current_step_index", 0)
            outline = self._get_state_value(state, "outline", [])
            current_step = outline[current_step_index] if current_step_index < len(outline) else {}
            step_id_raw = current_step.get("step_id", None)
            step_id = str(step_id_raw) if step_id_raw is not None else None

            retry_count = self._get_state_value(state, "retrieval_retry_count", 0)

            execution_id = f"exec_{uuid.uuid4().hex[:12]}"

            record = AgentExecutionRecord(
                execution_id=execution_id,
                task_id=task_id,
                chat_session_id=chat_session_id,
                agent_name=agent_name,
                node_name=node_name,
                step_id=step_id,
                step_index=current_step_index,
                action=agent_name,
                input_data={
                    "user_topic": self._get_state_value(state, "user_topic", ""),
                    "current_step_index": current_step_index,
                    "retry_count": retry_count
                },
                output_data=self._serialize_updates(updates),
                status="error" if error else "success",
                error_message=error,
                execution_time_ms=execution_time_ms,
                retry_count=retry_count,
                extra_data={
                    "agentic_rag_enabled": self.enable_agentic_rag,
                    "dynamic_adjustment_enabled": self.enable_dynamic_adjustment
                }
            )

            await agent_execution_service.create(record)
            logger.info(f"[WorkflowOrchestrator] Agent 执行记录已落库: {agent_name} - {execution_id}")

        except Exception as e:
            logger.error(f"[WorkflowOrchestrator] Agent 执行记录落库失败: {e}")

    def _serialize_updates(self, updates: Dict[str, Any]) -> Dict[str, Any]:
        """序列化 updates 字典，处理 Pydantic 对象"""
        def serialize_value(value: Any) -> Any:
            if isinstance(value, RetrievedDocument):
                return {
                    "content": getattr(value, "content", ""),
                    "source": getattr(value, "source", ""),
                    "confidence": getattr(value, "confidence", 0.0),
                    "metadata": getattr(value, "metadata", {})
                }
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            else:
                return value
        
        return serialize_value(updates)

    def _build_graph(self) -> StateGraph:
        """
        构建 LangGraph 状态图
        
        根据 enable_agentic_rag 参数决定是否启用 Agentic RAG 增强功能。
        
        Agentic RAG 工作流：
            planner → intent_parser → navigator → quality_eval → director
                                                ↘              ↗
                                                  如果质量差 → 重试
        
        返回:
            编译后的状态图
        """
        logger.info(f"构建 LangGraph 状态机 (agentic_rag={self.enable_agentic_rag}, dynamic_adjustment={self.enable_dynamic_adjustment})")
        
        workflow = StateGraph(GlobalState)
        
        workflow.add_node("planner", self._planner_node)
        
        if self.enable_agentic_rag:
            workflow.add_node("intent_parser", self._intent_parser_node)
        
        workflow.add_node("navigator", self._navigator_node)
        
        if self.enable_agentic_rag:
            workflow.add_node("quality_eval", self._quality_eval_node)
        
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
        
        if self.enable_tools:
            workflow.add_node("editor", self._editor_node)
        
        workflow.set_entry_point("planner")
        
        if self.enable_agentic_rag:
            workflow.add_edge("planner", "intent_parser")
            workflow.add_edge("intent_parser", "navigator")
            workflow.add_edge("navigator", "quality_eval")
            
            workflow.add_conditional_edges(
                "quality_eval",
                self._route_quality_eval_decision,
                {
                    "good": "director",
                    "retry": "navigator",
                    "failed": "director"
                }
            )
            
            if self.enable_dynamic_adjustment:
                workflow.add_edge("quality_eval", "rag_analyzer")
                workflow.add_edge("rag_analyzer", "dynamic_director")
                
                workflow.add_conditional_edges(
                    "dynamic_director",
                    self._route_dynamic_director_decision,
                    {
                        "pivot": "pivot_manager",
                        "skill_switch": "skill_recommender",
                        "continue": "retry_protection",
                        "retry_protection": "retry_protection",
                        "write": "writer"
                    }
                )
                
                workflow.add_edge("skill_recommender", "retry_protection")
            else:
                workflow.add_edge("quality_eval", "director")
        else:
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
                        "continue": "retry_protection",
                        "retry_protection": "retry_protection",
                        "write": "writer"
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
                "navigate": "intent_parser" if self.enable_agentic_rag else "navigator",
                "retry_protection": "retry_protection",
                "write": "writer"
            }
        )
        
        workflow.add_edge("pivot_manager", "intent_parser" if self.enable_agentic_rag else "navigator")
        workflow.add_edge("retry_protection", "navigator")
        workflow.add_edge("writer", "fact_checker")
        
        workflow.add_conditional_edges(
            "fact_checker",
            self._route_fact_check,
            {"invalid": "retry_protection", "valid": "step_advancer"}
        )
        
        workflow.add_conditional_edges(
            "step_advancer",
            self._route_completion,
            {"continue": "intent_parser" if self.enable_agentic_rag else "navigator", "done": "compiler"}
        )
        
        if self.enable_tools:
            workflow.add_edge("editor", "fact_checker")
            workflow.add_conditional_edges(
                "editor",
                self._route_editor_result,
                {"continue": "fact_checker", "done": "compiler"}
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
    
    def _route_quality_eval_decision(self, state: GlobalState) -> str:
        """
        路由质量评估决策
        
        根据质量评估结果决定下一步：
        - good: 质量可接受，继续到 director
        - retry: 质量差，需要重新检索
        - failed: 评估失败，继续到 director
        
        参数:
            state: 当前全局状态
            
        返回:
            路由目标节点标识
        """
        quality_evaluation = self._get_state_value(state, "quality_evaluation", None)
        
        if not quality_evaluation:
            logger.warning("没有质量评估结果，跳转到 director")
            return "failed"
        
        if isinstance(quality_evaluation, dict):
            needs_refinement = quality_evaluation.get("needs_refinement", False)
            quality_level = quality_evaluation.get("quality_level", "unknown")
            overall_score = quality_evaluation.get("overall_score", 0.0)
        else:
            needs_refinement = quality_evaluation.needs_refinement
            quality_level = quality_evaluation.quality_level.value if hasattr(quality_evaluation.quality_level, 'value') else str(quality_evaluation.quality_level)
            overall_score = quality_evaluation.overall_score
        
        retry_count = self._get_state_value(state, "retrieval_retry_count", 0)
        
        logger.info(
            f"route_quality_eval: level={quality_level}, score={overall_score:.2f}, "
            f"needs_refinement={needs_refinement}, retry_count={retry_count}"
        )
        
        if needs_refinement and retry_count < self.max_retrieval_retries:
            logger.info(f"质量评估需要改进，触发重试 (尝试 {retry_count + 1}/{self.max_retrieval_retries})")
            return "retry"
        elif needs_refinement and retry_count >= self.max_retrieval_retries:
            logger.warning(f"已达到最大重试次数 ({self.max_retrieval_retries})，继续处理")
            return "failed"
        else:
            logger.info(f"质量评估通过 (level={quality_level})")
            return "good"
    
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
        decision = director_feedback.get("decision", None)
        
        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)
        current_step = outline[current_step_index] if current_step_index < len(outline) else {}
        
        if isinstance(current_step, dict):
            retry_count = current_step.get("retry_count", 0)
        else:
            retry_count = getattr(current_step, "retry_count", 0)
        
        logger.info(f"route_director: trigger_retrieval={trigger_retrieval}, decision={decision}, retry_count={retry_count}")
        logger.info(f"route_director: director_feedback keys={list(director_feedback.keys()) if director_feedback else None}")
        
        if decision == "pivot":
            return "pivot"
        
        if decision == "retry":
            outline = self._get_state_value(state, "outline", [])
            current_step_index = self._get_state_value(state, "current_step_index", 0)
            current_step = outline[current_step_index] if current_step_index < len(outline) else {}
            
            if isinstance(current_step, dict):
                retry_count = current_step.get("retry_count", 0)
            else:
                retry_count = getattr(current_step, "retry_count", 0)
            
            if retry_count >= self.max_retrieval_retries:
                logger.warning(f"已达到最大重试次数 ({self.max_retrieval_retries})，强制继续到 writer")
                return "write"
            logger.info(f"触发重试: retry_count={retry_count}/{self.max_retrieval_retries}，经过 retry_protection 递增计数")
            return "retry_protection"
        
        if trigger_retrieval:
            return "navigate"
        
        if decision == "continue":
            return "write"
        
        if decision is None:
            quality_evaluation = self._get_state_value(state, "quality_evaluation", None)
            logger.info(f"route_director: quality_evaluation exists={quality_evaluation is not None}")
            if quality_evaluation:
                if isinstance(quality_evaluation, dict):
                    q_level = quality_evaluation.get("quality_level", "unknown")
                else:
                    q_level = getattr(quality_evaluation, "quality_level", "unknown")
                
                logger.info(f"route_director: q_level={q_level}")
                if q_level in ["good", "excellent"]:
                    logger.info(f"route_director: 信任质量评估结果 (level={q_level})，直接跳转到 writer")
                    return "write"
        
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
    
    def _route_director_decision_with_editor(self, state: GlobalState) -> str:
        """
        路由导演决策（包含编辑器模式）
        
        参数:
            state: 当前全局状态
            
        返回:
            路由目标节点标识
        """
        awaiting_user_input = self._get_state_value(state, "awaiting_user_input", False)
        human_intervention = self._get_state_value(state, "human_intervention", None)
        
        if awaiting_user_input or human_intervention:
            return "editor"
        
        return self._route_director_decision(state)
    
    def _route_editor_result(self, state: GlobalState) -> str:
        """
        路由编辑器结果
        
        参数:
            state: 当前全局状态
            
        返回:
            路由目标节点标识
        """
        requires_user_input = self._get_state_value(state, "requires_user_input", False)
        exceeded_max_iterations = self._get_state_value(state, "exceeded_max_iterations", False)
        
        if requires_user_input or exceeded_max_iterations:
            return "done"
        return "continue"
    
    @with_error_handling(agent_name="editor", action_name="process_user_input")
    async def _editor_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        编辑器节点 - 处理用户输入和工具调用
        
        职责:
            - 处理用户消息
            - 执行工具调用
            - 更新对话历史
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行编辑器节点")

        await self._record_agent_execution(
            agent_name="editor",
            node_name="editor",
            state=state,
            updates={}
        )

        if not self.enable_tools:
            return {"editor_response": "工具未启用"}
        
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

        updates = {
            "editor_response": result["response"],
            "chat_history": result["updated_chat_history"],
            "awaiting_user_input": result["requires_user_input"],
            "human_intervention": state.get("human_intervention"),
            "exceeded_max_iterations": result.get("exceeded_max_iterations", False)
        }

        return updates

    @with_error_handling(agent_name="intent_parser", action_name="parse_intent")
    async def _intent_parser_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        意图解析节点 - Agentic RAG 核心组件
        
        职责:
            - 分析当前步骤的查询意图
            - 提取关键词和数据源建议
            - 生成增强查询
            - 更新意图分析结果
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行意图解析节点")

        await self._record_agent_execution(
            agent_name="intent_parser",
            node_name="intent_parser",
            state=state,
            updates={}
        )

        current_step = self._get_current_step(state)
        if not current_step:
            return {}
        
        query = current_step.get("description", "")
        previous_intent = self._get_state_value(state, "current_intent", None)
        
        retry_count = self._get_state_value(state, "retrieval_retry_count", 0)
        
        context = None
        if previous_intent and retry_count > 0:
            previous_suggestions = self._get_state_value(state, "quality_suggestions", [])
            context = {
                "retry_count": retry_count,
                "previous_intent": previous_intent if isinstance(previous_intent, dict) else {
                    "primary_intent": previous_intent.primary_intent if hasattr(previous_intent, 'primary_intent') else str(previous_intent),
                    "keywords": previous_intent.keywords if hasattr(previous_intent, 'keywords') else []
                },
                "quality_suggestions": previous_suggestions,
                "quality_issues": self._get_state_value(state, "quality_issues", [])
            }
            
            logger.info(
                f"意图解析重试上下文: retry_count={retry_count}, "
                f"suggestions={len(previous_suggestions)}"
            )
        
        agent_logger.log_agent_transition(
            from_agent="planner",
            to_agent="intent_parser",
            step_id=current_step.get("step_id"),
            reason="parse_query_intent"
        )
        
        if context:
            intent = await self.intent_parser.parse_intent_with_context(query, context)
        else:
            intent = await self.intent_parser.parse_intent(query)
        
        agent_logger.log_agent_transition(
            from_agent="intent_parser",
            to_agent="navigator",
            step_id=current_step.get("step_id"),
            reason=f"intent_parsed: {intent.intent_type}"
        )
        
        logger.info(
            f"意图解析完成: intent={intent.primary_intent[:50]}..., "
            f"keywords={intent.keywords[:3]}, sources={intent.search_sources}, "
            f"confidence={intent.confidence:.2f}"
        )
        
        updates = {
            "current_intent": {
                "primary_intent": intent.primary_intent,
                "keywords": intent.keywords,
                "search_sources": intent.search_sources,
                "confidence": intent.confidence,
                "intent_type": intent.intent_type,
                "reasoning": intent.reasoning if hasattr(intent, 'reasoning') else ""
            }
        }
        
        updates["execution_log"] = create_success_log(
            agent="intent_parser",
            action="parse_intent",
            details={
                "step_id": current_step.get("step_id"),
                "query": query[:100],
                "primary_intent": intent.primary_intent[:100],
                "keywords": intent.keywords,
                "search_sources": intent.search_sources,
                "confidence": intent.confidence,
                "intent_type": intent.intent_type,
                "retry_count": retry_count
            }
        )

        return updates
    
    @with_error_handling(agent_name="quality_eval", action_name="evaluate_quality")
    async def _quality_eval_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        质量评估节点 - Agentic RAG 核心组件
        
        职责:
            - 评估检索结果的质量
            - 生成改进建议
            - 决定是否需要重新检索
            - 记录质量评估日志
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行质量评估节点")

        await self._record_agent_execution(
            agent_name="quality_eval",
            node_name="quality_eval",
            state=state,
            updates={}
        )

        current_step = self._get_current_step(state)
        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        current_intent = self._get_state_value(state, "current_intent", None)
        enhanced_query = self._get_state_value(state, "enhanced_query", "")
        
        if not current_step:
            return {}
        
        if not retrieved_docs:
            logger.warning("没有检索结果可供评估")
            
            agent_logger.log_agent_transition(
                from_agent="navigator",
                to_agent="quality_eval",
                step_id=current_step.get("step_id"),
                reason="no_documents_to_evaluate"
            )
            
            return {
                "quality_evaluation": {
                    "overall_score": 0.0,
                    "relevance_score": 0.0,
                    "completeness_score": 0.0,
                    "accuracy_score": 0.0,
                    "quality_level": "insufficient",
                    "retrieval_status": "no_results",
                    "strengths": [],
                    "weaknesses": ["没有检索到相关文档"],
                    "suggestions": ["尝试修改查询关键词", "检查数据源配置"],
                    "needs_refinement": True,
                    "refinement_strategy": None
                },
                "execution_log": create_success_log(
                    agent="quality_eval",
                    action="evaluate_quality",
                    details={
                        "step_id": current_step.get("step_id"),
                        "doc_count": 0,
                        "quality_level": "insufficient",
                        "needs_refinement": True
                    }
                )
            }
        
        intent_obj = None
        if current_intent:
            if isinstance(current_intent, dict):
                intent_obj = IntentAnalysis(
                    primary_intent=current_intent.get("primary_intent", ""),
                    keywords=current_intent.get("keywords", []),
                    search_sources=current_intent.get("search_sources", []),
                    confidence=current_intent.get("confidence", 0.5),
                    intent_type=current_intent.get("intent_type", "informational")
                )
            else:
                intent_obj = current_intent
        
        agent_logger.log_agent_transition(
            from_agent="navigator",
            to_agent="quality_eval",
            step_id=current_step.get("step_id"),
            reason="evaluate_retrieval_quality"
        )
        
        quality_evaluation = await self.quality_eval_agent.evaluate_quality(
            query=enhanced_query or current_step.get("description", ""),
            documents=retrieved_docs,
            intent=intent_obj
        )
        
        retry_count = self._get_state_value(state, "retrieval_retry_count", 0)
        
        agent_logger.log_agent_transition(
            from_agent="quality_eval",
            to_agent=("director" if not quality_evaluation.needs_refinement else "navigator"),
            step_id=current_step.get("step_id"),
            reason=f"quality_{quality_evaluation.quality_level.value}_decision"
        )
        
        logger.info(
            f"质量评估完成: score={quality_evaluation.overall_score:.2f}, "
            f"level={quality_evaluation.quality_level.value}, "
            f"needs_refinement={quality_evaluation.needs_refinement}, "
            f"retry_count={retry_count}"
        )
        
        updates = {
            "quality_evaluation": {
                "overall_score": quality_evaluation.overall_score,
                "relevance_score": quality_evaluation.relevance_score,
                "completeness_score": quality_evaluation.completeness_score,
                "accuracy_score": quality_evaluation.accuracy_score,
                "quality_level": quality_evaluation.quality_level.value,
                "retrieval_status": quality_evaluation.retrieval_status.value if hasattr(quality_evaluation.retrieval_status, 'value') else str(quality_evaluation.retrieval_status),
                "strengths": quality_evaluation.strengths,
                "weaknesses": quality_evaluation.weaknesses,
                "suggestions": quality_evaluation.suggestions,
                "needs_refinement": quality_evaluation.needs_refinement,
                "refinement_strategy": quality_evaluation.refinement_strategy,
                "retry_count": retry_count
            },
            "quality_suggestions": quality_evaluation.suggestions,
            "quality_issues": quality_evaluation.weaknesses
        }
        
        if isinstance(quality_evaluation, dict):
            level = quality_evaluation.get('quality_level', 'unknown')
            score = quality_evaluation.get('overall_score', 0.0)
        else:
            level = quality_evaluation.quality_level.value if hasattr(quality_evaluation.quality_level, 'value') else str(quality_evaluation.quality_level)
            score = quality_evaluation.overall_score
        
        logger.info(f"质量评估结果: level={level}, score={score:.2f}")
        
        if quality_evaluation.needs_refinement and retry_count < self.max_retrieval_retries:
            updates["retrieval_retry_count"] = retry_count + 1
            logger.info(f"触发重试机制: retry_count={retry_count + 1}/{self.max_retrieval_retries}")
        
        updates["execution_log"] = create_success_log(
            agent="quality_eval",
            action="evaluate_quality",
            details={
                "step_id": current_step.get("step_id"),
                "doc_count": len(retrieved_docs),
                "overall_score": quality_evaluation.overall_score,
                "relevance_score": quality_evaluation.relevance_score,
                "completeness_score": quality_evaluation.completeness_score,
                "accuracy_score": quality_evaluation.accuracy_score,
                "quality_level": quality_evaluation.quality_level.value,
                "needs_refinement": quality_evaluation.needs_refinement,
                "retry_count": retry_count,
                "strengths_count": len(quality_evaluation.strengths),
                "weaknesses_count": len(quality_evaluation.weaknesses),
                "suggestions_count": len(quality_evaluation.suggestions)
            }
        )

        return updates

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

        await self._record_agent_execution(
            agent_name="rag_analyzer",
            node_name="rag_analyzer",
            state=state,
            updates={}
        )

        current_step = self._get_current_step(state)
        retrieved_docs = self._get_state_value(state, "retrieved_docs", [])
        
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
                action="analyze_content",
                details={
                    "step_id": current_step.get("step_id"),
                    "content_types": analysis.content_types,
                    "difficulty": analysis.difficulty_level,
                    "suggested_skill": analysis.suggested_skill
                }
            )
        }

        return updates

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

        await self._record_agent_execution(
            agent_name="dynamic_director",
            node_name="dynamic_director",
            state=state,
            updates={}
        )

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
                action="direction_adjustment",
                details={
                    "adjustment_type": adjustment.adjustment_type.value if adjustment else "no_change",
                    "reason": adjustment.reason if adjustment else "",
                    "confidence": adjustment.confidence if adjustment else 0
                }
            )
        }

        return updates

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

        await self._record_agent_execution(
            agent_name="skill_recommender",
            node_name="skill_recommender",
            state=state,
            updates={}
        )

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
            action="recommend_skill",
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

        await self._record_agent_execution(
            agent_name="planner",
            node_name="planner",
            state=state,
            updates={}
        )

        user_topic = self._get_state_value(state, "user_topic", "")

        updates = await self.node_factory.planner_node(state)

        if self.enable_agentic_rag:
            agent_logger.log_agent_transition(
                from_agent="entry",
                to_agent="planner",
                step_id=None,
                reason="generate_outline"
            )

        return updates
    
    @with_error_handling(agent_name="navigator", action_name="retrieve_content")
    async def _navigator_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        导航器节点 - 检索相关内容
        
        职责:
            - 根据当前步骤检索相关内容
            - 解析和摘要检索结果
            - 更新检索文档列表
            - 支持意图解析后的增强查询
        
        参数:
            state: 当前全局状态
            
        返回:
            状态更新字典
        """
        logger.info("执行导航器节点")

        outline = self._get_state_value(state, "outline", [])
        current_step_index = self._get_state_value(state, "current_step_index", 0)

        await self._record_agent_execution(
            agent_name="navigator",
            node_name="navigator",
            state=state,
            updates={}
        )

        current_intent = self._get_state_value(state, "current_intent", None)
        retry_count = self._get_state_value(state, "retrieval_retry_count", 0)

        if current_step_index >= len(outline):
            updates = {
                "workflow_complete": True,
                "execution_log": [{
                    "timestamp": datetime.now().isoformat(),
                    "node": "navigator",
                    "action": f"navigation complete - all {len(outline)} steps processed"
                }]
            }
            return updates
        
        intent_obj = None
        if current_intent and isinstance(current_intent, dict):
            intent_obj = IntentAnalysis(
                primary_intent=current_intent.get("primary_intent", ""),
                keywords=current_intent.get("keywords", []),
                search_sources=current_intent.get("search_sources", []),
                confidence=current_intent.get("confidence", 0.5),
                intent_type=current_intent.get("intent_type", "informational")
            )
        elif current_intent:
            intent_obj = current_intent
        
        current_step = outline[current_step_index]
        
        if self.enable_agentic_rag:
            agent_logger.log_agent_transition(
                from_agent=("quality_eval" if retry_count > 0 else "intent_parser"),
                to_agent="navigator",
                step_id=current_step.get("step_id"),
                reason=("retry_retrieval" if retry_count > 0 else "retrieve_content")
            )
        
        updated_state, quality_evaluation = await retrieve_content(
            state=state,
            retrieval_service=self.retrieval_service,
            parser_service=self.parser_service,
            summarization_service=self.summarization_service,
            enable_parallel=True,
            enable_quality_eval=False,
            llm_service=self.llm_service,
            intent=intent_obj
        )
        
        enhanced_query = intent_obj.primary_intent if intent_obj else current_step.get("description", "")
        
        if isinstance(updated_state, dict):
            updates = {
                "retrieved_docs": updated_state.get("retrieved_docs", []),
                "enhanced_query": enhanced_query,
                "execution_log": updated_state.get("execution_log", [])
            }
        else:
            updates = {
                "retrieved_docs": updated_state.retrieved_docs,
                "enhanced_query": enhanced_query,
                "execution_log": updated_state.execution_log
            }

        await self._record_agent_execution(
            agent_name="navigator",
            node_name="navigator",
            state=state,
            updates=updates
        )

        return updates
    
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

        await self._record_agent_execution(
            agent_name="director",
            node_name="director",
            state=state,
            updates={}
        )

        updates = await self.node_factory.director_node(state)

        return updates
    
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

        await self._record_agent_execution(
            agent_name="pivot_manager",
            node_name="pivot_manager",
            state=state,
            updates={}
        )

        updates = await self.node_factory.pivot_manager_node(state)

        return updates
    
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

        await self._record_agent_execution(
            agent_name="retry_protection",
            node_name="retry_protection",
            state=state,
            updates={}
        )

        updates = await self.node_factory.retry_protection_node(state)

        return updates
    
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

        await self._record_agent_execution(
            agent_name="writer",
            node_name="writer",
            state=state,
            updates={}
        )

        updates = await self.node_factory.writer_node(state)

        return updates
    
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

        await self._record_agent_execution(
            agent_name="fact_checker",
            node_name="fact_checker",
            state=state,
            updates={}
        )

        updates = await self.node_factory.fact_checker_node(state)

        return updates
    
    async def _step_advancer_node(self, state: GlobalState) -> Dict[str, Any]:
        old_index = self._get_state_value(state, "current_step_index", 0)
        outline = self._get_state_value(state, "outline", [])

        await self._record_agent_execution(
            agent_name="step_advancer",
            node_name="step_advancer",
            state=state,
            updates={}
        )

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

        await self._record_agent_execution(
            agent_name="compiler",
            node_name="compiler",
            state=state,
            updates={}
        )

        updates = await self.node_factory.compiler_node(state)

        return updates
    
    async def execute(
        self,
        initial_state: Dict[str, Any],
        recursion_limit: int = 25
    ) -> Dict[str, Any]:
        """
        执行工作流

        参数:
            initial_state: 初始状态字典
            recursion_limit: 递归限制（默认25）

        返回:
            包含 success 和最终状态的字典
        """
        thread_id = f"run_{uuid.uuid4().hex[:8]}"
        
        try:
            logger.info(f"开始执行工作流，recursion_limit={recursion_limit}")
            logger.info(f"Agentic RAG 模式: {self.enable_agentic_rag}")

            config = {
                "recursion_limit": recursion_limit,
                "configurable": {
                    "thread_id": thread_id
                }
            }

            final_state = await self.graph.ainvoke(initial_state, config=config)

            logger.info("工作流执行完成")
            return {
                "success": True,
                "state": final_state
            }

        except Exception as e:
            logger.error(f"工作流执行失败: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "state": initial_state
            }
    
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
                "thread_id": "default"
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
        await self._record_agent_execution(
            agent_name="collaboration_manager",
            node_name="collaboration_manager",
            state=state,
            updates={}
        )

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
        await self._record_agent_execution(
            agent_name="execution_tracer",
            node_name="execution_tracer",
            state=state,
            updates={}
        )

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
    
    @with_error_handling(agent_name="agent_reflection", action_name="agent_reflection")
    async def _agent_reflection_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        Agent 反思节点

        职责：
            - 从失败中学习
            - 调整策略
            - 记录反思结果
        """
        await self._record_agent_execution(
            agent_name="agent_reflection",
            node_name="agent_reflection",
            state=state,
            updates={}
        )

        try:
            reflection_result = await self.agent_reflection.reflect_on_failure(
                agent_name=state.get("current_agent", "unknown"),
                failure_context=state.get("last_error", {}),
                state=state
            )

            return {
                "reflection_result": reflection_result,
                "strategy_adjustment": reflection_result.get("adjustments", {}) if reflection_result else {}
            }

        except Exception as e:
            logger.error(f"Agent 反思失败: {str(e)}")
            return {"reflection_error": str(e)}
