"""导演智能体 - 评估内容并做出决策

本模块实现导演智能体，负责：
1. 检测冲突（例如：已废弃的功能）
2. 评估内容复杂度
3. 做出决策（批准/转向）
4. 推荐技能切换
5. 必要时触发转向
6. 智能跳过优化（基于质量和复杂度）
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from ..models import SharedState, RetrievedDocument, OutlineStep
from ...services.llm.service import LLMService
from ...services.optimization import (
    QualityAssessor,
    ComplexityBasedSkipper,
    CacheBasedSkipper,
    SmartSkipOptimizer
)
from ...infrastructure.logging import get_agent_logger


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


def detect_conflicts(
    current_step: OutlineStep,
    retrieved_docs: List[RetrievedDocument]
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    检测事实冲突
    - 将大纲意图与检索的元数据进行比较
    - 检测废弃功能冲突
    - 返回冲突类型和详情
    
    Args:
        current_step: 当前大纲步骤
        retrieved_docs: 检索到的文档列表
        
    Returns:
        (has_conflict, conflict_type, conflict_details) 元组
    """
    # 检查是否有文档标记为废弃
    deprecated_docs = [
        doc for doc in retrieved_docs
        if doc.metadata.get('has_deprecated', False)
    ]
    
    if deprecated_docs:
        # 检测到废弃冲突
        deprecated_sources = [doc.source for doc in deprecated_docs]
        conflict_details = (
            f"步骤 {current_step.step_id} 计划解释的功能在以下文档中被标记为废弃: "
            f"{', '.join(deprecated_sources)}"
        )
        
        logger.warning(f"Deprecation conflict detected: {conflict_details}")
        
        return True, "deprecation_conflict", conflict_details
    
    # 检查是否有安全问题标记
    security_docs = [
        doc for doc in retrieved_docs
        if doc.metadata.get('has_security', False)
    ]
    
    if security_docs:
        # 检测到安全问题
        security_sources = [doc.source for doc in security_docs]
        conflict_details = (
            f"步骤 {current_step.step_id} 涉及的代码在以下文档中有安全标记: "
            f"{', '.join(security_sources)}"
        )
        
        logger.warning(f"Security issue detected: {conflict_details}")
        
        return True, "security_issue", conflict_details
    
    # 检查是否有 FIXME 标记
    fixme_docs = [
        doc for doc in retrieved_docs
        if doc.metadata.get('has_fixme', False)
    ]
    
    if fixme_docs:
        # 检测到需要修复的问题
        fixme_sources = [doc.source for doc in fixme_docs]
        conflict_details = (
            f"步骤 {current_step.step_id} 涉及的代码在以下文档中有 FIXME 标记: "
            f"{', '.join(fixme_sources)}"
        )
        
        logger.info(f"FIXME markers detected: {conflict_details}")
        
        return True, "fixme_issue", conflict_details
    
    # 没有检测到冲突
    return False, None, None


async def assess_complexity(
    current_step: OutlineStep,
    retrieved_docs: List[RetrievedDocument],
    llm_service: LLMService
) -> float:
    """
    评估内容复杂度
    - 使用 LLM 或启发式评估复杂度
    - 评估技术术语、嵌套结构、抽象概念
    - 返回复杂度分数（0-1）
    
    Args:
        current_step: 当前大纲步骤
        retrieved_docs: 检索到的文档列表
        llm_service: LLM 服务
        
    Returns:
        复杂度分数（0.0-1.0）
    """
    try:
        # 如果没有检索到文档，复杂度为 0
        if not retrieved_docs:
            return 0.0
        
        # 构建评估提示
        docs_summary = "\n\n".join([
            f"文档 {i+1} ({doc.source}):\n{doc.content[:500]}..."
            for i, doc in enumerate(retrieved_docs[:3])  # 只使用前 3 个文档
        ])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个内容复杂度评估专家。请评估给定内容的复杂度。\n"
                    "复杂度评分标准：\n"
                    "- 0.0-0.3: 简单内容，易于理解\n"
                    "- 0.3-0.5: 中等复杂度，需要一定技术背景\n"
                    "- 0.5-0.7: 较复杂，包含抽象概念或嵌套结构\n"
                    "- 0.7-1.0: 极其复杂，包含大量技术术语和抽象概念\n"
                    "请只返回一个 0.0 到 1.0 之间的数字。"
                )
            },
            {
                "role": "user",
                "content": (
                    f"步骤描述: {current_step.description}\n\n"
                    f"检索到的文档内容:\n{docs_summary}\n\n"
                    f"请评估这些内容的复杂度（0.0-1.0）："
                )
            }
        ]
        
        # 调用 LLM 评估复杂度
        response = await llm_service.chat_completion(
            messages=messages,
            task_type="high_performance",
            temperature=0.3,
            max_tokens=10
        )
        
        # 解析响应
        try:
            complexity_score = float(response.strip())
            # 确保分数在 0-1 范围内
            complexity_score = max(0.0, min(1.0, complexity_score))
            
            logger.info(f"Complexity assessment for step {current_step.step_id}: {complexity_score}")
            
            return complexity_score
            
        except ValueError:
            # 如果无法解析，使用启发式方法
            logger.warning(f"Failed to parse complexity score from LLM response: {response}")
            return _heuristic_complexity_assessment(retrieved_docs)
    
    except Exception as e:
        logger.error(f"Complexity assessment failed: {str(e)}")
        # 回退到启发式方法
        return _heuristic_complexity_assessment(retrieved_docs)


def _heuristic_complexity_assessment(retrieved_docs: List[RetrievedDocument]) -> float:
    """
    启发式复杂度评估（回退方法）
    
    Args:
        retrieved_docs: 检索到的文档列表
        
    Returns:
        复杂度分数（0.0-1.0）
    """
    if not retrieved_docs:
        return 0.0
    
    complexity_score = 0.0
    
    # 因素 1: 文档数量（更多文档 = 更复杂）
    doc_count_factor = min(len(retrieved_docs) / 10.0, 0.3)
    complexity_score += doc_count_factor
    
    # 因素 2: 平均文档长度（更长 = 更复杂）
    avg_length = sum(len(doc.content) for doc in retrieved_docs) / len(retrieved_docs)
    length_factor = min(avg_length / 10000.0, 0.3)
    complexity_score += length_factor
    
    # 因素 3: 元数据复杂度（有问题标记 = 更复杂）
    metadata_issues = sum(
        1 for doc in retrieved_docs
        if any([
            doc.metadata.get('has_deprecated', False),
            doc.metadata.get('has_fixme', False),
            doc.metadata.get('has_security', False)
        ])
    )
    metadata_factor = min(metadata_issues / len(retrieved_docs), 0.4)
    complexity_score += metadata_factor
    
    # 确保分数在 0-1 范围内
    complexity_score = max(0.0, min(1.0, complexity_score))
    
    logger.info(f"Heuristic complexity assessment: {complexity_score}")
    
    return complexity_score


async def evaluate_and_decide(
    state: SharedState,
    llm_service: LLMService
) -> SharedState:
    """
    导演决策逻辑
    - 检测冲突并触发转向
    - 评估复杂度并推荐 Skill 切换
    - 在状态中设置 pivot_triggered 标志和 pivot_reason
    - 集成智能跳过优化

    Args:
        state: 共享状态
        llm_service: LLM 服务

    Returns:
        更新后的共享状态
    """
    try:
        current_step = state.get_current_step()
        if not current_step:
            logger.warning("No current step to evaluate")
            return state

        logger.info(f"Director: Evaluating step {current_step.step_id}: {current_step.description}")

        skip_optimizer = SmartSkipOptimizer(
            enable_quality_skip=True,
            enable_complexity_skip=False,
            enable_cache_skip=True
        )

        if state.retrieved_docs:
            content_for_quality = str(state.retrieved_docs[0].content)[:500]

            quality_decisions = skip_optimizer.evaluate_skip_decision(
                content=content_for_quality,
                complexity_score=None,
                cache_key=f"director:{current_step.step_id}",
                context=current_step.description
            )

            overall_decision = skip_optimizer.get_overall_skip_decision(quality_decisions)

            if overall_decision.should_skip:
                logger.info(
                    f"Director: Skipping detailed evaluation - "
                    f"reason={overall_decision.reason}, "
                    f"confidence={overall_decision.confidence:.2f}"
                )

                state.execution_log.append({
                    'agent': 'director',
                    'action': 'smart_skip',
                    'skip_reason': overall_decision.reason,
                    'confidence': overall_decision.confidence,
                    'details': overall_decision.details
                })

                quality_assessor = QualityAssessor()
                quality_score = quality_assessor.assess_quality(content_for_quality)

                if quality_score >= 0.9:
                    state.pivot_triggered = False
                    state.add_log_entry(
                        agent_name="director",
                        action="approved_high_quality",
                        details={
                            "step_id": current_step.step_id,
                            "quality_score": quality_score,
                            "skip_reason": overall_decision.reason
                        }
                    )
                    logger.info(
                        f"Director: High quality content (score={quality_score:.2f}), "
                        f"approved without detailed check"
                    )
                    return state

        has_conflict, conflict_type, conflict_details = detect_conflicts(
            current_step=current_step,
            retrieved_docs=state.retrieved_docs
        )
        
        if has_conflict:
            # 触发转向
            state.pivot_triggered = True
            state.pivot_reason = conflict_type
            
            # Log pivot trigger with reason
            agent_logger.log_pivot_trigger(
                step_id=current_step.step_id,
                pivot_reason=conflict_type,
                conflict_details={"details": conflict_details}
            )
            
            # 记录日志
            state.add_log_entry(
                agent_name="director",
                action="conflict_detected",
                details={
                    "step_id": current_step.step_id,
                    "conflict_type": conflict_type,
                    "conflict_details": conflict_details
                }
            )
            
            logger.warning(f"Director: Conflict detected, triggering pivot: {conflict_type}")
            
            return state
        
        # 2. 评估复杂度
        complexity_score = await assess_complexity(
            current_step=current_step,
            retrieved_docs=state.retrieved_docs,
            llm_service=llm_service
        )
        
        # 3. 基于复杂度推荐 Skill 切换
        recommended_skill = None
        
        if complexity_score > 0.7:
            # 极其复杂，推荐使用可视化类比
            recommended_skill = "visualization_analogy"
            reason = f"content_complexity_high"
            
            state.pivot_triggered = True
            state.pivot_reason = reason
            
            logger.info(f"Director: High complexity detected ({complexity_score:.2f}), recommending skill switch to {recommended_skill}")
            
            # Log skill switch recommendation
            agent_logger.log_skill_switch(
                step_id=current_step.step_id,
                from_skill=state.current_skill,
                to_skill=recommended_skill,
                trigger_reason=reason,
                complexity_score=complexity_score
            )
            
            # 记录日志
            state.add_log_entry(
                agent_name="director",
                action="complexity_trigger",
                details={
                    "step_id": current_step.step_id,
                    "complexity_score": complexity_score,
                    "recommended_skill": recommended_skill,
                    "reason": reason
                }
            )
            
            return state
        elif complexity_score < 0.3 and state.current_skill != "standard_tutorial":
            # 内容简单，可以切换回标准教程模式
            recommended_skill = "standard_tutorial"
            reason = f"content_complexity_low"
            
            state.pivot_triggered = True
            state.pivot_reason = reason
            
            logger.info(f"Director: Low complexity detected ({complexity_score:.2f}), recommending skill switch to {recommended_skill}")
            
            # Log skill switch recommendation
            agent_logger.log_skill_switch(
                step_id=current_step.step_id,
                from_skill=state.current_skill,
                to_skill=recommended_skill,
                trigger_reason=reason,
                complexity_score=complexity_score
            )
            
            # 记录日志
            state.add_log_entry(
                agent_name="director",
                action="complexity_trigger",
                details={
                    "step_id": current_step.step_id,
                    "complexity_score": complexity_score,
                    "recommended_skill": recommended_skill,
                    "reason": reason,
                    "trigger_type": "low_complexity"
                }
            )
            
            return state
        
        # 4. 没有冲突或触发条件，批准继续
        logger.info(f"Director: Approved step {current_step.step_id}, complexity: {complexity_score:.2f}")
        
        # 记录日志
        state.add_log_entry(
            agent_name="director",
            action="approved",
            details={
                "step_id": current_step.step_id,
                "complexity_score": complexity_score
            }
        )
        
        return state
        
    except Exception as e:
        logger.error(f"Director evaluation failed: {str(e)}")
        # 优雅降级：批准继续而不是崩溃
        state.add_log_entry(
            agent_name="director",
            action="evaluation_failed",
            details={"error": str(e)}
        )
        return state

