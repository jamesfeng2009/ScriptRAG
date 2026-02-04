"""
v2.1 风格节点函数（LangGraph Native）

本模块定义了遵循 v2.1 架构规范的节点函数。
特点：
- 接受 GlobalState (TypedDict)
- 返回 Diff (Dict[str, Any])，LangGraph 自动合并
- 使用 Reducer 实现追加保护
- 完整的错误处理（使用 langgraph_error_handler）

日志规范：
    所有节点必须：
    1. 使用 @with_error_handling 装饰器
    2. 成功时返回 execution_log (success)
    3. 失败时返回 error_flag
    4. 使用 audit_log_reducer 自动追加日志

使用示例：
    from src.domain.state_types import GlobalState
    from src.domain.agents.node_factory import NodeFactory, create_node_factory
    
    factory = create_node_factory(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        parser_service=parser_service,
        summarization_service=summarization_service
    )
    
    workflow.add_node("planner", factory.planner_node)
    workflow.add_node("navigator", factory.navigator_node)

迁移指南：
    https://docs/architecture/v2.1_architecture_spec.md#73-迁移指南

错误处理文档：
    src/infrastructure/langgraph_error_handler.py
"""

import logging
from typing import Dict, Any, List, Optional, Union, Set, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

from ..state_types import (
    GlobalState,
    create_error_log,
    create_success_log,
    get_error_message,
)
from ...services.llm.service import LLMService
from ...services.retrieval_service import RetrievalService
from ...services.parser.tree_sitter_parser import IParserService
from ...services.summarization_service import SummarizationService
from ...infrastructure.langgraph_error_handler import (
    with_error_handling,
    ErrorCategory,
    ErrorRecovery,
)
from ..data_access_control import DataAccessControl


# ============================================================================
# 节点函数工厂（支持依赖注入）
# ============================================================================


class NodeFactory:
    """节点函数工厂
    
    用于创建带有依赖注入的 v2.1 风格节点函数。
    
    使用示例：
        factory = NodeFactory(
            llm_service=llm_service,
            retrieval_service=retrieval_service,
            parser_service=parser_service,
            summarization_service=summarization_service
        )
        
        workflow.add_node("planner", factory.planner_node)
        workflow.add_node("navigator", factory.navigator_node)
    """
    
    def __init__(
        self,
        llm_service: LLMService,
        retrieval_service: RetrievalService,
        parser_service: IParserService,
        summarization_service: SummarizationService,
        workspace_id: str = ""
    ):
        self.llm_service = llm_service
        self.retrieval_service = retrieval_service
        self.parser_service = parser_service
        self.summarization_service = summarization_service
        self.workspace_id = workspace_id
    
    # =========================================================================
    # Planner 节点
    # =========================================================================

    @DataAccessControl.agent_access(
        agent_name="planner",
        reads={"user_topic", "project_context"},
        writes={"outline", "execution_log"},
        description="生成大纲 - 分析用户主题并创建步骤列表"
    )
    @with_error_handling(agent_name="planner", action_name="generate_outline")
    async def planner_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        规划器节点
        
        职责：
        - 分析用户主题和项目上下文
        - 生成包含 5-10 步的结构化大纲
        - 返回 outline 更新（覆盖模式）
        
        数据流向：
        - 输入：user_topic, project_context
        - 输出：outline (Overwrite)
        
        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            user_topic = state.get("user_topic", "")
            project_context = state.get("project_context", "")
            
            if not user_topic:
                return {
                    "execution_log": create_error_log(
                        agent="planner",
                        action="empty_topic",
                        error_message="user_topic 为空"
                    ),
                    "error_flag": "validation_error"
                }
            
            logger.info(f"Planner: Generating outline for topic: {user_topic}")
            
            outline = await self._generate_outline_async(user_topic, project_context)
            
            return {
                "outline": outline,
                "execution_log": create_success_log(
                    agent="planner",
                    action="outline_generated",
                    details={
                        "topic": user_topic,
                        "step_count": len(outline)
                    }
                )
            }
        
        except Exception as e:
            logger.error(f"Planner node error: {str(e)}", exc_info=True)
            fallback_outline = self._create_fallback_outline(user_topic)
            return {
                "outline": fallback_outline,
                "execution_log": create_error_log(
                    agent="planner",
                    action="fallback_used",
                    error_message=str(e),
                    details={"topic": user_topic}
                )
            }
    
    async def _generate_outline_async(
        self,
        user_topic: str,
        project_context: str
    ) -> List[Dict[str, Any]]:
        """生成大纲（异步内部方法）"""
        import json
        
        prompt = f"""请为以下主题生成一个结构化的剧本大纲：

主题：{user_topic}
上下文：{project_context or '无特定上下文'}

要求：
1. 生成 5-10 个步骤
2. 每个步骤包含标题和描述
3. 步骤之间有逻辑连贯性

请按以下 JSON 格式输出：
{{
    "steps": [
        {{"step_id": 0, "title": "步骤标题", "description": "步骤描述"}}
    ]
}}
"""
        
        response = await self.llm_service.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        import json
        try:
            result = json.loads(response)
            # Handle both dict with "steps" key and direct list format
            if isinstance(result, dict) and "steps" in result:
                steps_data = result["steps"]
            elif isinstance(result, list):
                steps_data = result
            else:
                logger.warning(f"Unexpected outline format: {type(result)}")
                return self._create_fallback_outline(user_topic)
            
            outline = [
                {
                    "step_id": step.get("step_id", idx) if isinstance(step, dict) else getattr(step, "step_id", idx),
                    "title": step.get("title", f"步骤 {idx+1}") if isinstance(step, dict) else getattr(step, "title", f"步骤 {idx+1}"),
                    "description": step.get("description", step.get("title", f"步骤 {idx+1} 的内容")) if isinstance(step, dict) else getattr(step, "description", getattr(step, "title", f"步骤 {idx+1} 的内容")),
                    "status": "pending"
                }
                for idx, step in enumerate(steps_data)
            ]
            return outline
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse outline response: {e}")
            return self._create_fallback_outline(user_topic)
    
    def _create_fallback_outline(self, user_topic: str) -> List[Dict[str, Any]]:
        """创建回退大纲"""
        return [
            {
                "step_id": i,
                "title": f"第 {i+1} 部分",
                "description": f"关于 {user_topic} 的第 {i+1} 部分内容",
                "status": "pending"
            }
            for i in range(5)
        ]
    
    # =========================================================================
    # Navigator 节点
    # =========================================================================

    @DataAccessControl.agent_access(
        agent_name="navigator",
        reads={"outline", "current_step_index", "project_context"},
        writes={"last_retrieved_docs", "execution_log"},
        description="检索相关文档 - 根据当前步骤检索参考资料"
    )
    @with_error_handling(agent_name="navigator", action_name="retrieve", error_category=ErrorCategory.RETRIEVAL)
    async def navigator_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        导航器节点
        
        职责：
        - 根据当前步骤检索相关文档
        - 将检索结果写入 last_retrieved_docs（覆盖）
        - 记录执行日志
        
        数据流向：
        - 输入：current_step_index, outline, project_context
        - 输出：last_retrieved_docs (Overwrite)
        
        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            step_index = state.get("current_step_index", 0)
            outline = state.get("outline", [])
            
            if step_index >= len(outline):
                return {
                    "execution_log": create_error_log(
                        agent="navigator",
                        action="boundary_error",
                        error_message=f"步骤索引越界: {step_index} >= {len(outline)}",
                        details={"step_index": step_index, "outline_length": len(outline)}
                    ),
                    "error_flag": "boundary_error"
                }
            
            current_step = outline[step_index]
            step_title = current_step.get("title") if isinstance(current_step, dict) else getattr(current_step, "title", "")
            step_desc = current_step.get("description") if isinstance(current_step, dict) else getattr(current_step, "description", "")
            query = f"{state.get('user_topic', '')} {step_title} {step_desc}"
            
            logger.info(f"Navigator: Retrieving for step {step_index}: {step_title}")
            
            retrieved_docs = await self._retrieve_documents_async(query)
            
            docs_as_dicts = []
            for i, doc in enumerate(retrieved_docs):
                if isinstance(doc, dict):
                    docs_as_dicts.append({
                        "id": doc.get("id", str(i)),
                        "content": doc.get("content", ""),
                        "source": doc.get("source", doc.get("file_path", "")),
                        "score": doc.get("score", doc.get("similarity", 0.0)),
                        "metadata": doc.get("metadata", {})
                    })
                else:
                    docs_as_dicts.append({
                        "id": getattr(doc, "id", str(i)),
                        "content": getattr(doc, "content", ""),
                        "source": getattr(doc, "file_path", getattr(doc, "source", "")),
                        "score": getattr(doc, "similarity", getattr(doc, "score", 0.0)),
                        "metadata": getattr(doc, "metadata", {})
                    })
            
            return {
                "last_retrieved_docs": docs_as_dicts,
                "execution_log": create_success_log(
                    agent="navigator",
                    action="retrieve_completed",
                    details={
                        "step_index": step_index,
                        "step_title": step_title,
                        "doc_count": len(docs_as_dicts)
                    }
                )
            }
        
        except Exception as e:
            logger.error(f"Navigator node error: {str(e)}", exc_info=True)
            return {
                "execution_log": create_error_log(
                    agent="navigator",
                    action="error",
                    error_message=str(e),
                    details={"step_index": state.get("current_step_index", 0)}
                ),
                "error_flag": "retrieval_error"
            }
    
    async def navigator_node_async(self, state: GlobalState) -> Dict[str, Any]:
        """导航器节点（异步版本）"""
        try:
            step_index = state.get("current_step_index", 0)
            outline = state.get("outline", [])
            
            if step_index >= len(outline):
                return {
                    "execution_log": create_error_log(
                        agent="navigator",
                        action="boundary_error",
                        error_message=f"步骤索引越界: {step_index} >= {len(outline)}"
                    ),
                    "error_flag": "boundary_error"
                }
            
            current_step = outline[step_index]
            query = f"{state.get('user_topic', '')} {current_step.title}"
            
            retrieved_docs = await self.retrieval_service.hybrid_retrieve(
                query=query,
                top_k=5,
                enable_reranking=True
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
            
            return {
                "last_retrieved_docs": docs_as_dicts,
                "execution_log": create_success_log(
                    agent="navigator",
                    action="retrieve_completed",
                    details={
                        "step_index": step_index,
                        "doc_count": len(docs_as_dicts)
                    }
                )
            }
        
        except Exception as e:
            logger.error(f"Navigator async error: {str(e)}", exc_info=True)
            return {
                "execution_log": create_error_log(
                    agent="navigator",
                    action="error",
                    error_message=str(e)
                ),
                "error_flag": "retrieval_error"
            }
    
    async def _retrieve_documents_async(self, query: str) -> List[Dict[str, Any]]:
        """检索文档（异步版本，内部方法）"""
        import json
        
        try:
            response = await self.llm_service.chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": "根据查询返回相关的文档信息，返回 JSON 数组格式。"
                    },
                    {"role": "user", "content": f"查询: {query}"}
                ],
                task_type="high_performance",
                temperature=0.3,
                max_tokens=1000
            )
            
            result = json.loads(response)
            return result if isinstance(result, list) else []
        
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Document retrieval failed: {e}")
            return []
    
    # =========================================================================
    # Director 节点
    # =========================================================================

    @DataAccessControl.agent_access(
        agent_name="director",
        reads={"outline", "fragments", "current_step_index", "last_retrieved_docs"},
        writes={"director_feedback", "execution_log"},
        description="评估方向 - 分析内容质量并提供反馈"
    )
    @with_error_handling(agent_name="director", action_name="evaluate", error_category=ErrorCategory.LLM)
    async def director_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        导演节点
        
        职责：
        - 评估检索结果质量
        - 决定是否继续或转向
        - 返回结构化反馈
        
        数据流向：
        - 输入：last_retrieved_docs, current_step_index
        - 输出：director_feedback (Overwrite)
        
        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            retrieved_docs = state.get("last_retrieved_docs", [])
            step_index = state.get("current_step_index", 0)
            
            quality_score = self._evaluate_retrieval_quality(retrieved_docs)
            
            if quality_score < 0.5:
                feedback = {
                    "decision": "retry",
                    "reason": f"检索质量不足（分数：{quality_score:.2f}），需要补充检索",
                    "confidence": quality_score,
                    "suggested_skill": None,
                    "trigger_retrieval": True,
                    "metadata": {
                        "quality_score": quality_score,
                        "doc_count": len(retrieved_docs),
                        "retry_reason": "low_quality"
                    }
                }
                
                return {
                    "director_feedback": feedback,
                    "execution_log": create_success_log(
                        agent="director",
                        action="retry_triggered",
                        details={
                            "step_index": step_index,
                            "quality_score": quality_score,
                            "reason": feedback["reason"]
                        }
                    )
                }
            
            elif quality_score < 0.7:
                feedback = {
                    "decision": "retry",
                    "reason": f"检索质量一般（分数：{quality_score:.2f}），建议优化查询",
                    "confidence": quality_score,
                    "suggested_skill": None,
                    "metadata": {"quality_score": quality_score}
                }
                
                return {
                    "director_feedback": feedback,
                    "execution_log": create_success_log(
                        agent="director",
                        action="retry_recommended",
                        details={"step_index": step_index, "quality_score": quality_score}
                    )
                }
            
            else:
                suggested_skill = self._recommend_skill(retrieved_docs)
                feedback = {
                    "decision": "continue",
                    "reason": "检索结果充足且质量良好",
                    "confidence": quality_score,
                    "suggested_skill": suggested_skill,
                    "metadata": {"quality_score": quality_score}
                }
                
                return {
                    "director_feedback": feedback,
                    "execution_log": create_success_log(
                        agent="director",
                        action="approved",
                        details={
                            "step_index": step_index,
                            "quality_score": quality_score,
                            "suggested_skill": suggested_skill
                        }
                    )
                }
        
        except Exception as e:
            logger.error(f"Director node error: {str(e)}", exc_info=True)
            return {
                "execution_log": create_error_log(
                    agent="director",
                    action="error",
                    error_message=str(e)
                ),
                "error_flag": "llm_error"
            }
    
    def _evaluate_retrieval_quality(self, documents: List[Dict]) -> float:
        """评估检索质量"""
        if not documents:
            return 0.0
        
        doc_count = len(documents)
        avg_score = sum(doc.get("score", 0) for doc in documents) / doc_count
        
        count_score = min(doc_count / 5, 1.0)
        return 0.3 * count_score + 0.7 * avg_score
    
    def _recommend_skill(self, documents: List[Dict]) -> str:
        """推荐技能"""
        return "standard_tutorial"
    
    # =========================================================================
    # Writer 节点
    # =========================================================================

    @DataAccessControl.agent_access(
        agent_name="writer",
        reads={"outline", "current_step_index", "last_retrieved_docs", "director_feedback", "current_skill"},
        writes={"fragments", "execution_log"},
        description="生成剧本片段 - 根据步骤和检索内容生成剧本"
    )
    @with_error_handling(agent_name="writer", action_name="generate_fragment", error_category=ErrorCategory.LLM)
    async def writer_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        编剧节点
        
        职责：
        - 根据当前步骤、检索结果和导演反馈生成剧本片段
        - 将片段追加到 fragments（追加保护 ⭐）
        - 记录执行日志
        
        数据流向：
        - 输入：current_step_index, outline, last_retrieved_docs, director_feedback
        - 输出：fragments (Append Only ⭐)
        
        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            error_flag = state.get("error_flag")
            if error_flag:
                return {
                    "execution_log": create_success_log(
                        agent="writer",
                        action="skipped",
                        details={"reason": f"error_flag: {error_flag}"}
                    )
                }
            
            feedback = state.get("director_feedback")
            if feedback:
                if feedback.get("trigger_retrieval"):
                    return {
                        "execution_log": create_success_log(
                            agent="writer",
                            action="waiting_for_retrieval",
                            details={
                                "reason": feedback.get("reason"),
                                "quality_score": feedback.get("confidence")
                            }
                        )
                    }
                if feedback.get("decision") == "pivot":
                    return {
                        "execution_log": create_success_log(
                            agent="writer",
                            action="waiting_for_pivot",
                            details={"reason": feedback.get("reason")}
                        )
                    }
            
            step_index = state.get("current_step_index", 0)
            outline = state.get("outline", [])
            
            if step_index >= len(outline):
                return {
                    "execution_log": create_error_log(
                        agent="writer",
                        action="boundary_error",
                        error_message=f"步骤索引越界: {step_index} >= {len(outline)}"
                    ),
                    "error_flag": "boundary_error"
                }
            
            current_step = outline[step_index]
            
            # Handle both dict and Pydantic object formats
            if isinstance(current_step, dict):
                step_title = current_step.get("title", f"步骤 {step_index}")
            else:
                step_title = getattr(current_step, "title", f"步骤 {step_index}")
            
            retrieved_docs = state.get("last_retrieved_docs", [])
            current_skill = state.get("current_skill", "standard_tutorial")
            
            logger.info(f"Writer: Generating fragment for step {step_index}")
            
            fragment_content = await self._generate_fragment_content(
                step=current_step,
                documents=retrieved_docs,
                skill=current_skill
            )
            
            fragment = {
                "step_id": step_index,
                "content": fragment_content,
                "references": [doc.get("source", "") for doc in retrieved_docs],
                "skill_used": current_skill
            }
            
            return {
                "fragments": [fragment],
                "execution_log": create_success_log(
                    agent="writer",
                    action="fragment_completed",
                    details={
                        "step_index": step_index,
                        "step_title": step_title,
                        "skill_used": current_skill,
                        "reference_count": len(retrieved_docs)
                    }
                )
            }
        
        except Exception as e:
            logger.error(f"Writer node error: {str(e)}", exc_info=True)
            return {
                "execution_log": create_error_log(
                    agent="writer",
                    action="error",
                    error_message=str(e)
                ),
                "error_flag": "llm_error"
            }
    
    async def _generate_fragment_content(
        self,
        step: Union[Dict[str, Any], Any],
        documents: List[Dict[str, Any]],
        skill: str
    ) -> str:
        """生成片段内容"""
        from ...services.llm.service import LLMService
        
        if isinstance(step, dict):
            step_title = step.get("title", "未知步骤")
            step_description = step.get("description", "")
        else:
            step_title = getattr(step, "title", "未知步骤")
            step_description = getattr(step, "description", "")
        
        content_parts = []
        for doc in documents[:3]:
            content_parts.append(doc.get("content", "")[:500])
        
        retrieved_text = "\n\n".join(content_parts)
        
        prompt = f"""请根据以下信息生成剧本片段：

步骤：{step_title} - {step_description}
技能：{skill}
参考内容：
{retrieved_text}

请生成清晰、结构化的内容。"""
        
        try:
            response = await self.llm_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000
            )
            return f"# {step_title}\n\n{response}"
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}")
            return f"# {step_title}\n\n[内容生成失败，请参考：\n{retrieved_text[:500]}...]"
    
    # =========================================================================
    # Step Advancer 节点
    # =========================================================================
    
    @with_error_handling(agent_name="step_advancer", action_name="advance_step")
    def step_advancer_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        步骤推进器节点

        职责：
        - 将 current_step_index 增加 1
        - 当所有步骤完成时，设置 workflow_complete=True

        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            current_index = state.get("current_step_index", 0)
            outline = state.get("outline", [])

            new_index = current_index + 1
            is_done = new_index >= len(outline)

            updates = {
                "current_step_index": new_index,
                "execution_log": create_success_log(
                    agent="step_advancer",
                    action="step_advanced",
                    details={
                        "from_index": current_index,
                        "to_index": new_index,
                        "is_complete": is_done
                    }
                )
            }

            if is_done:
                updates["workflow_complete"] = True

            return updates

        except Exception as e:
                logger.error(f"Step advancer error: {str(e)}")
                return {
                    "execution_log": create_error_log(
                        agent="step_advancer",
                        action="error",
                        error_message=str(e)
                    )
                }
    
    # =========================================================================
    # Pivot Manager 节点
    # =========================================================================

    @DataAccessControl.agent_access(
        agent_name="pivot_manager",
        reads={"outline", "fragments", "current_step_index", "director_feedback"},
        writes={"outline", "current_skill", "current_step_index", "execution_log"},
        description="处理转向 - 修改大纲或切换技能"
    )
    @with_error_handling(agent_name="pivot_manager", action_name="handle_pivot")
    async def pivot_manager_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        转向管理器节点
        
        职责：
        - 处理来自导演的转向触发
        - 根据转向原因修改大纲步骤
        - 在兼容性约束下应用技能切换
        - 清除检索到的文档以触发重新检索
        
        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            from ..agents.pivot_manager import handle_pivot
            
            state_dict = dict(state)
            from ..models import SharedState
            shared_state = SharedState(**state_dict)
            
            result_state = handle_pivot(shared_state)
            
            updates = {
                "pivot_triggered": False,
                "pivot_reason": None,
                "execution_log": create_success_log(
                    agent="pivot_manager",
                    action="pivot_handled",
                    details={"handled": True}
                )
            }
            
            if result_state.outline:
                updates["outline"] = result_state.outline
            
            return updates
        
        except Exception as e:
            logger.error(f"Pivot manager node error: {str(e)}", exc_info=True)
            return {
                "execution_log": create_error_log(
                    agent="pivot_manager",
                    action="error",
                    error_message=str(e)
                ),
                "error_flag": "pivot_error"
            }
    
    # =========================================================================
    # Retry Protection 节点
    # =========================================================================

    @DataAccessControl.agent_access(
        agent_name="retry_protection",
        reads={"outline", "current_step_index", "retry_count"},
        writes={"retry_count", "error_flag", "execution_log"},
        description="重试保护 - 检查重试次数并决定是否降级"
    )
    @with_error_handling(agent_name="retry_protection", action_name="check_retry_limit")
    async def retry_protection_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        重试保护节点
        
        职责：
        - 检查当前步骤重试次数
        - 阻止超过限制的重试
        - 标记超限步骤
        
        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            current_step_index = state.get("current_step_index", 0)
            outline = state.get("outline", [])
            max_retries = state.get("max_retries", 3)
            
            if current_step_index >= len(outline):
                return {
                    "execution_log": create_success_log(
                        agent="retry_protection",
                        action="no_retry_needed",
                        details={"reason": "no more steps"}
                    )
                }
            
            current_step = outline[current_step_index]
            
            # Handle both dict and Pydantic object formats
            if isinstance(current_step, dict):
                retry_count = current_step.get("retry_count", 0)
                step_title = current_step.get("title", f"步骤 {current_step_index}")
            else:
                retry_count = getattr(current_step, "retry_count", 0)
                step_title = getattr(current_step, "title", f"步骤 {current_step_index}")
            
            if retry_count >= max_retries:
                return {
                    "execution_log": create_error_log(
                        agent="retry_protection",
                        action="retry_limit_exceeded",
                        error_message=f"步骤 {current_step_index} ({step_title}) 已达到最大重试次数 {max_retries}"
                    ),
                    "error_flag": "retry_limit_exceeded",
                    "max_retries_reached": True
                }
            
            return {
                "execution_log": create_success_log(
                    agent="retry_protection",
                    action="retry_allowed",
                    details={
                        "step_index": current_step_index,
                        "retry_count": retry_count,
                        "max_retries": max_retries
                    }
                )
            }
        
        except Exception as e:
            logger.error(f"Retry protection node error: {str(e)}", exc_info=True)
            return {
                "execution_log": create_error_log(
                    agent="retry_protection",
                    action="error",
                    error_message=str(e)
                ),
                "error_flag": "retry_error"
            }
    
    # =========================================================================
    # Fact Checker 节点
    # =========================================================================

    @DataAccessControl.agent_access(
        agent_name="fact_checker",
        reads={"fragments", "last_retrieved_docs"},
        writes={"fragments", "execution_log"},
        description="事实检查 - 验证片段内容与源文档一致性"
    )
    @with_error_handling(agent_name="fact_checker", action_name="verify_fragment", error_category=ErrorCategory.VALIDATION)
    async def fact_checker_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        事实检查器节点
        
        职责：
        - 检查片段中的事实准确性
        - 标记验证结果
        - 记录检查日志
        
        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            fragments = state.get("fragments", [])
            retrieved_docs = state.get("last_retrieved_docs", [])
            
            if not fragments:
                return {
                    "fact_check_passed": True,
                    "execution_log": create_success_log(
                        agent="fact_checker",
                        action="no_fragments_to_check",
                        details={"fragments_count": 0}
                    )
                }
            
            last_fragment = fragments[-1]
            fragment_content = last_fragment.get("content", "") if isinstance(last_fragment, dict) else str(last_fragment)
            
            fact_check_result = await self._verify_facts(fragment_content, retrieved_docs)
            
            if fact_check_result["is_valid"]:
                return {
                    "fact_check_passed": True,
                    "execution_log": create_success_log(
                        agent="fact_checker",
                        action="fact_check_passed",
                        details={
                            "fragment_index": len(fragments) - 1,
                            "issues_found": 0
                        }
                    )
                }
            else:
                issues = fact_check_result.get("issues", [])
                return {
                    "fact_check_passed": False,
                    "execution_log": create_error_log(
                        agent="fact_checker",
                        action="fact_check_failed",
                        error_message=f"发现 {len(issues)} 个事实问题",
                        details={
                            "fragment_index": len(fragments) - 1,
                            "issues": issues
                        }
                    ),
                    "error_flag": "fact_check_failed"
                }
        
        except Exception as e:
            logger.error(f"Fact checker node error: {str(e)}", exc_info=True)
            return {
                "fact_check_passed": False,
                "execution_log": create_error_log(
                    agent="fact_checker",
                    action="error",
                    error_message=str(e)
                ),
                "error_flag": "fact_check_error"
            }
    
    async def _verify_facts(self, content: str, docs: List[Dict]) -> Dict[str, Any]:
        """验证事实（异步内部方法）"""
        if not docs:
            return {"is_valid": True, "issues": []}
        
        prompt = f"""请验证以下内容是否与提供的参考文档一致：

内容：
{content}

参考文档：
{docs}

如果内容与参考文档一致，返回 "VALID"。
如果发现不一致，列出所有问题。
"""
        
        try:
            response = await self.llm_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            
            if "VALID" in response.upper() and not response.strip().startswith("INVALID"):
                return {"is_valid": True, "issues": []}
            
            issues = [line.strip() for line in response.split("\n") if line.strip()]
            return {"is_valid": False, "issues": issues}
        
        except Exception as e:
            logger.warning(f"Fact verification failed: {e}")
            return {"is_valid": True, "issues": []}
    
    # =========================================================================
    # Compiler 节点
    # =========================================================================

    @DataAccessControl.agent_access(
        agent_name="compiler",
        reads={"fragments", "outline"},
        writes={"final_screenplay", "execution_log"},
        description="编译剧本 - 整合所有片段成最终剧本"
    )
    @with_error_handling(agent_name="compiler", action_name="compile")
    async def compiler_node(self, state: GlobalState) -> Dict[str, Any]:
        """
        编译器节点
        
        职责：
        - 编译所有片段成最终剧本
        - 添加审计日志
        - 返回最终结果
        
        Returns:
            Dict[str, Any]: Diff 更新
        """
        try:
            fragments = state.get("fragments", [])
            outline = state.get("outline", [])
            user_topic = state.get("user_topic", "")
            project_context = state.get("project_context", "")
            
            if not fragments:
                return {
                    "final_screenplay": f"# {user_topic}\n\n[未生成任何内容]",
                    "execution_log": create_error_log(
                        agent="compiler",
                        action="no_fragments",
                        error_message="没有可编译的片段"
                    ),
                    "error_flag": "no_fragments"
                }
            
            content_parts = []
            for i, fragment in enumerate(fragments):
                if isinstance(fragment, dict):
                    content_parts.append(fragment.get("content", ""))
                else:
                    content_parts.append(str(fragment))
            
            full_content = "\n\n".join(content_parts)
            
            prompt = f"""请将以下剧本片段编译成一个完整的、结构化的文档：

主题：{user_topic}
上下文：{project_context}

大纲结构：
{[s.get('title', '') for s in outline]}

片段内容：
{full_content}

请提供一个格式良好的最终文档。"""
            
            compiled = await self.llm_service.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=5000
            )
            
            return {
                "final_screenplay": compiled,
                "execution_log": create_success_log(
                    agent="compiler",
                    action="compilation_complete",
                    details={
                        "fragments_count": len(fragments),
                        "total_length": len(compiled)
                    }
                )
            }
        
        except Exception as e:
            logger.error(f"Compiler node error: {str(e)}", exc_info=True)
            return {
                "final_screenplay": f"# {user_topic}\n\n[编译失败]",
                "execution_log": create_error_log(
                    agent="compiler",
                    action="error",
                    error_message=str(e)
                ),
                "error_flag": "compiler_error"
            }
    
    # =========================================================================
    # 路由函数
    # =========================================================================
    
    def route_director_decision(self, state: GlobalState) -> str:
        """导演决策路由"""
        feedback = state.get("director_feedback")
        if feedback:
            if feedback.get("trigger_retrieval"):
                return "navigate"
            if feedback.get("decision") == "pivot":
                return "pivot"
        return "write"
    
    def route_fact_check(self, state: GlobalState) -> str:
        """事实检查路由"""
        fact_check_passed = state.get("fact_check_passed", True)
        return "valid" if fact_check_passed else "invalid"
    
    def route_completion(self, state: GlobalState) -> str:
        """完成路由"""
        step_index = state.get("current_step_index", 0)
        outline = state.get("outline", [])
        
        if step_index >= len(outline):
            return "done"
        return "continue"


# ============================================================================
# 便捷函数（无需工厂）
# ============================================================================


def create_node_factory(
    llm_service: LLMService,
    retrieval_service: RetrievalService,
    parser_service: IParserService,
    summarization_service: SummarizationService,
    workspace_id: str = ""
) -> NodeFactory:
    """创建节点工厂的便捷函数"""
    return NodeFactory(
        llm_service=llm_service,
        retrieval_service=retrieval_service,
        parser_service=parser_service,
        summarization_service=summarization_service,
        workspace_id=workspace_id
    )
