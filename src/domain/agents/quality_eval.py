"""Quality Evaluation Agent - 质量评估智能体

本模块实现质量评估智能体，负责：
1. 评估检索结果质量
2. 生成置信度评分
3. 提供改进建议
4. 实现自适应检索循环

用于 Agentic RAG 系统的质量控制层。
"""

import json
import logging
import re
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from ...services.llm.service import LLMService
from ...infrastructure.logging import get_agent_logger
from ...domain.models import RetrievedDocument, IntentAnalysis


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


class QualityLevel(Enum):
    """质量等级枚举"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    INSUFFICIENT = "insufficient"


class RetrievalStatus(Enum):
    """检索状态枚举"""
    SUCCESS = "success"
    PARTIAL = "partial"
    NEEDS_IMPROVEMENT = "needs_improvement"
    FAILED = "failed"


@dataclass
class QualityEvaluation:
    """质量评估结果"""
    overall_score: float
    relevance_score: float
    completeness_score: float
    accuracy_score: float
    quality_level: QualityLevel
    retrieval_status: RetrievalStatus
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    needs_refinement: bool
    refinement_strategy: Optional[str] = None


@dataclass
class AdaptiveAction:
    """自适应行动"""
    action_type: str
    reason: str
    parameters: Dict[str, Any]


class QualityEvalAgent:
    """质量评估智能体
    
    使用 LLM 评估检索结果的质量，并提供改进建议。
    
    功能：
    1. 评估检索文档的相关性、完整性和准确性
    2. 生成综合质量分数
    3. 提供检索改进建议
    4. 实现自适应检索循环（必要时重新检索）
    
    示例:
        query = "Python 异步编程怎么实现"
        docs = [retrieved_doc1, retrieved_doc2]
        result = await agent.evaluate_quality(query, docs)
        # result.overall_score = 0.85
        # result.quality_level = QualityLevel.GOOD
        # result.needs_refinement = False
    """
    
    def __init__(self, llm_service: LLMService):
        """
        初始化质量评估智能体
        
        Args:
            llm_service: LLM 服务实例
        """
        self.llm_service = llm_service
    
    async def evaluate_quality(
        self,
        query: str,
        documents: List[RetrievedDocument],
        intent: Optional[IntentAnalysis] = None
    ) -> QualityEvaluation:
        """
        评估检索结果质量
        
        Args:
            query: 用户查询
            documents: 检索到的文档列表
            intent: 意图分析结果（可选）
            
        Returns:
            QualityEvaluation: 质量评估结果
        """
        logger.info(f"Evaluating quality for {len(documents)} documents")
        
        if not documents:
            return self._create_insufficient_result(query)
        
        try:
            doc_contents = self._prepare_documents_for_eval(documents)
            
            doc_summaries = []
            for i, (doc, content) in enumerate(doc_contents):
                doc_summaries.append({
                    "index": i + 1,
                    "source": doc.source,
                    "confidence": doc.confidence,
                    "content_preview": content[:500],
                    "metadata": doc.metadata
                })
            
            messages = self._build_evaluation_prompt(query, doc_summaries, intent)
            
            response = await self.llm_service.chat_completion(
                messages=messages,
                task_type="high_performance",
                temperature=0.3,
                max_tokens=1500
            )
            
            evaluation_data = self._parse_evaluation_response(response)
            
            quality_evaluation = QualityEvaluation(
                overall_score=evaluation_data["overall_score"],
                relevance_score=evaluation_data["relevance_score"],
                completeness_score=evaluation_data["completeness_score"],
                accuracy_score=evaluation_data["accuracy_score"],
                quality_level=self._determine_quality_level(evaluation_data["overall_score"]),
                retrieval_status=self._determine_retrieval_status(evaluation_data["overall_score"]),
                strengths=evaluation_data["strengths"],
                weaknesses=evaluation_data["weaknesses"],
                suggestions=evaluation_data["suggestions"],
                needs_refinement=evaluation_data["overall_score"] < 0.7,
                refinement_strategy=evaluation_data.get("refinement_strategy")
            )
            
            agent_logger.log_agent_transition(
                from_agent="retrieval",
                to_agent="quality_eval",
                reason=f"quality_evaluated: {quality_evaluation.quality_level.value}"
            )
            
            logger.info(
                f"Quality evaluated: score={quality_evaluation.overall_score:.2f}, "
                f"level={quality_evaluation.quality_level.value}, "
                f"needs_refinement={quality_evaluation.needs_refinement}"
            )
            
            return quality_evaluation
            
        except Exception as e:
            logger.error(f"Quality evaluation failed: {str(e)}")
            return self._fallback_evaluation(documents)
    
    def _prepare_documents_for_eval(
        self,
        documents: List[RetrievedDocument]
    ) -> List[Tuple[RetrievedDocument, str]]:
        """
        准备文档用于评估
        
        Args:
            documents: 检索到的文档列表
            
        Returns:
            处理后的文档列表
        """
        processed = []
        
        for doc in documents:
            content = doc.content
            if doc.summary:
                content = f"[摘要] {doc.summary}\n\n[完整内容] {content}"
            
            if len(content) > 3000:
                content = content[:3000] + "...\n[内容截断]"
            
            processed.append((doc, content))
        
        return processed
    
    def _build_evaluation_prompt(
        self,
        query: str,
        doc_summaries: List[Dict[str, Any]],
        intent: Optional[IntentAnalysis]
    ) -> List[Dict[str, str]]:
        """
        构建质量评估的 prompt
        
        Args:
            query: 用户查询
            doc_summaries: 文档摘要列表
            intent: 意图分析结果
            
        Returns:
            Messages 列表
        """
        intent_context = ""
        if intent:
            intent_context = f"""
用户查询意图分析：
- 主要意图：{intent.primary_intent}
- 建议关键词：{', '.join(intent.keywords)}
- 建议数据源：{', '.join(intent.search_sources)}
- 置信度：{intent.confidence:.2f}

请结合意图分析结果评估文档与用户真实需求的匹配程度。
"""
        
        docs_json = json.dumps(doc_summaries, ensure_ascii=False, indent=2)
        
        return [
            {
                "role": "system",
                "content": f"""你是一个检索质量评估专家。你的任务是评估检索结果的质量，并提供改进建议。

评估维度：
1. **相关性（relevance）**: 文档内容与用户查询的匹配程度
2. **完整性（completeness）**: 文档是否涵盖查询所需的完整信息
3. **准确性（accuracy）**: 文档内容的准确性和可靠性

评分标准：
- 0.0-0.4: 质量差（poor）
- 0.4-0.6: 可接受（acceptable）
- 0.6-0.8: 良好（good）
- 0.8-1.0: 优秀（excellent）

请分析每个文档，计算综合质量分数，并提供：
1. 总体质量分数（0.0-1.0）
2. 各维度分数（相关性、完整性、准确性）
3. 优势列表
4. 劣势列表
5. 改进建议
6. 是否需要重新检索（分数 < 0.7 时建议）
7. 重新检索策略（如需要）

{intent_context}请返回 JSON 格式的分析结果：

```json
{{
    "overall_score": 0.85,
    "relevance_score": 0.9,
    "completeness_score": 0.8,
    "accuracy_score": 0.85,
    "strengths": ["文档内容高度相关", "来源可靠"],
    "weaknesses": ["缺少最新信息"],
    "suggestions": ["补充官方文档", "增加示例代码"],
    "needs_refinement": false,
    "refinement_strategy": null
}}
```"""
            },
            {
                "role": "user",
                "content": f"""请评估以下检索结果的质量：

用户查询：{query}

检索到的文档：
{docs_json}

请进行质量评估并返回 JSON 结果。"""
            }
        ]
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的评估响应
        
        Args:
            response: LLM 原始响应
            
        Returns:
            解析后的评估数据
        """
        try:
            json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = response.strip()
            
            data = json.loads(json_str)
            
            required_fields = ["overall_score", "relevance_score", "completeness_score", 
                             "accuracy_score", "strengths", "weaknesses", "suggestions"]
            
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            
            return {
                "overall_score": max(0.0, min(1.0, float(data["overall_score"]))),
                "relevance_score": max(0.0, min(1.0, float(data["relevance_score"]))),
                "completeness_score": max(0.0, min(1.0, float(data["completeness_score"]))),
                "accuracy_score": max(0.0, min(1.0, float(data["accuracy_score"]))),
                "strengths": data.get("strengths", []),
                "weaknesses": data.get("weaknesses", []),
                "suggestions": data.get("suggestions", []),
                "needs_refinement": data.get("needs_refinement", data["overall_score"] < 0.7),
                "refinement_strategy": data.get("refinement_strategy")
            }
            
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Failed to parse evaluation response: {e}")
            raise ValueError(f"Invalid evaluation response format: {e}")
    
    def _determine_quality_level(self, score: float) -> QualityLevel:
        """
        根据分数确定质量等级
        
        Args:
            score: 质量分数
            
        Returns:
            QualityLevel: 质量等级
        """
        if score >= 0.8:
            return QualityLevel.EXCELLENT
        elif score >= 0.6:
            return QualityLevel.GOOD
        elif score >= 0.4:
            return QualityLevel.ACCEPTABLE
        elif score >= 0.2:
            return QualityLevel.POOR
        else:
            return QualityLevel.INSUFFICIENT
    
    def _determine_retrieval_status(self, score: float) -> RetrievalStatus:
        """
        根据分数确定检索状态
        
        Args:
            score: 质量分数
            
        Returns:
            RetrievalStatus: 检索状态
        """
        if score >= 0.7:
            return RetrievalStatus.SUCCESS
        elif score >= 0.5:
            return RetrievalStatus.PARTIAL
        elif score >= 0.3:
            return RetrievalStatus.NEEDS_IMPROVEMENT
        else:
            return RetrievalStatus.FAILED
    
    def _create_insufficient_result(self, query: str) -> QualityEvaluation:
        """
        创建结果不足时的评估结果
        
        Args:
            query: 用户查询
            
        Returns:
            QualityEvaluation: 评估结果
        """
        return QualityEvaluation(
            overall_score=0.0,
            relevance_score=0.0,
            completeness_score=0.0,
            accuracy_score=0.0,
            quality_level=QualityLevel.INSUFFICIENT,
            retrieval_status=RetrievalStatus.FAILED,
            strengths=[],
            weaknesses=["未检索到任何文档"],
            suggestions=["尝试修改查询关键词", "扩大搜索范围", "使用不同的数据源"],
            needs_refinement=True,
            refinement_strategy="broaden_search"
        )
    
    def _fallback_evaluation(self, documents: List[RetrievedDocument]) -> QualityEvaluation:
        """
        评估失败时的回退方法
        
        使用简单的规则计算质量分数
        
        Args:
            documents: 检索到的文档列表
            
        Returns:
            QualityEvaluation: 基本评估结果
        """
        logger.warning("Using fallback quality evaluation")
        
        if not documents:
            return self._create_insufficient_result("")
        
        avg_confidence = sum(doc.confidence for doc in documents) / len(documents)
        
        doc_count_score = min(1.0, len(documents) / 5.0)
        
        overall_score = (avg_confidence * 0.7 + doc_count_score * 0.3)
        
        return QualityEvaluation(
            overall_score=overall_score,
            relevance_score=overall_score,
            completeness_score=doc_count_score,
            accuracy_score=avg_confidence,
            quality_level=self._determine_quality_level(overall_score),
            retrieval_status=self._determine_retrieval_status(overall_score),
            strengths=[f"检索到 {len(documents)} 个相关文档"],
            weaknesses=[],
            suggestions=[],
            needs_refinement=overall_score < 0.7,
            refinement_strategy=None
        )
    
    def determine_adaptive_action(
        self,
        evaluation: QualityEvaluation,
        query: str,
        intent: Optional[IntentAnalysis]
    ) -> AdaptiveAction:
        """
        根据评估结果确定自适应行动
        
        Args:
            evaluation: 质量评估结果
            query: 用户查询
            intent: 意图分析结果
            
        Returns:
            AdaptiveAction: 自适应行动
        """
        if evaluation.overall_score >= 0.8:
            return AdaptiveAction(
                action_type="proceed",
                reason="检索质量优秀，可以继续生成",
                parameters={"next_step": "generation"}
            )
        elif evaluation.overall_score >= 0.6:
            return AdaptiveAction(
                action_type="augment",
                reason="检索质量良好，建议补充更多信息",
                parameters={
                    "next_step": "generation_with_augmentation",
                    "suggestions": evaluation.suggestions[:3]
                }
            )
        elif evaluation.overall_score >= 0.4:
            return AdaptiveAction(
                action_type="refine_retrieval",
                reason="检索质量一般，建议优化检索策略",
                parameters={
                    "next_step": "refine_retrieval",
                    "strategy": evaluation.refinement_strategy or "modify_keywords",
                    "suggestions": evaluation.suggestions[:3]
                }
            )
        else:
            return AdaptiveAction(
                action_type="retry",
                reason="检索质量差，需要重新检索",
                parameters={
                    "next_step": "retry_retrieval",
                    "alternative_keywords": intent.keywords if intent else [],
                    "alternative_sources": intent.search_sources if intent else ["rag"]
                }
            )
    
    async def adaptive_retrieve(
        self,
        query: str,
        documents: List[RetrievedDocument],
        intent: Optional[IntentAnalysis],
        retrieval_fn: callable,
        max_retries: int = 2
    ) -> Tuple[List[RetrievedDocument], QualityEvaluation]:
        """
        自适应检索循环
        
        如果初始检索质量不佳，自动调整策略重新检索
        
        Args:
            query: 用户查询
            documents: 初始检索结果
            intent: 意图分析结果
            retrieval_fn: 检索函数
            max_retries: 最大重试次数
            
        Returns:
            Tuple[最终文档列表, 最终质量评估结果]
        """
        logger.info(f"Starting adaptive retrieval for query: {query[:50]}...")
        
        current_docs = documents
        evaluation = await self.evaluate_quality(query, current_docs, intent)
        
        attempt = 0
        
        while evaluation.needs_refinement and attempt < max_retries:
            attempt += 1
            
            logger.info(f"Adaptive retrieval attempt {attempt}/{max_retries}")
            
            action = self.determine_adaptive_action(evaluation, query, intent)
            
            if action.action_type == "retry":
                alternative_keywords = action.parameters.get("alternative_keywords", [])
                alternative_sources = action.parameters.get("alternative_sources", [])
                
                enhanced_query = query
                if alternative_keywords:
                    enhanced_query = f"{query} {' '.join(alternative_keywords)}"
                
                try:
                    current_docs = await retrieval_fn(
                        query=enhanced_query,
                        sources=alternative_sources
                    )
                    
                    evaluation = await self.evaluate_quality(query, current_docs, intent)
                    
                    agent_logger.log_agent_transition(
                        from_agent="quality_eval",
                        to_agent="retrieval",
                        step_id=attempt,
                        reason=f"retry_with_improved_strategy: {action.parameters.get('strategy')}"
                    )
                    
                except Exception as e:
                    logger.error(f"Retry retrieval failed: {str(e)}")
                    break
                    
            elif action.action_type == "refine_retrieval":
                strategy = action.parameters.get("strategy", "modify_keywords")
                
                try:
                    if strategy == "broaden_search":
                        current_docs = await retrieval_fn(query=query, expand=True)
                    elif strategy == "modify_keywords" and intent:
                        refined_query = f"{intent.primary_intent} {' '.join(intent.keywords[:5])}"
                        current_docs = await retrieval_fn(query=refined_query)
                    else:
                        current_docs = await retrieval_fn(query=query)
                    
                    evaluation = await self.evaluate_quality(query, current_docs, intent)
                    
                except Exception as e:
                    logger.error(f"Refine retrieval failed: {str(e)}")
                    break
                    
            else:
                break
        
        if attempt > 0:
            logger.info(
                f"Adaptive retrieval completed: attempts={attempt}, "
                f"final_score={evaluation.overall_score:.2f}, "
                f"docs_count={len(current_docs)}"
            )
        
        return current_docs, evaluation


async def evaluate_quality(
    query: str,
    documents: List[RetrievedDocument],
    llm_service: LLMService,
    intent: Optional[IntentAnalysis] = None
) -> QualityEvaluation:
    """
    便捷的质量评估函数
    
    Args:
        query: 用户查询
        documents: 检索到的文档列表
        llm_service: LLM 服务
        intent: 意图分析结果
        
    Returns:
        QualityEvaluation: 质量评估结果
    """
    agent = QualityEvalAgent(llm_service)
    return await agent.evaluate_quality(query, documents, intent)


async def adaptive_retrieve(
    query: str,
    documents: List[RetrievedDocument],
    llm_service: LLMService,
    retrieval_fn: callable,
    intent: Optional[IntentAnalysis] = None,
    max_retries: int = 2
) -> Tuple[List[RetrievedDocument], QualityEvaluation]:
    """
    便捷的自适应检索函数
    
    Args:
        query: 用户查询
        documents: 初始检索结果
        llm_service: LLM 服务
        retrieval_fn: 检索函数
        intent: 意图分析结果
        max_retries: 最大重试次数
        
    Returns:
        Tuple[最终文档列表, 最终质量评估结果]
    """
    agent = QualityEvalAgent(llm_service)
    return await agent.adaptive_retrieve(query, documents, intent, retrieval_fn, max_retries)
