"""事实检查智能体 - 根据检索到的来源验证生成的内容

本模块实现事实检查智能体，负责：
1. 根据检索到的文档验证剧本片段
2. 检测幻觉（不存在的代码、函数、参数）
3. 验证所有陈述是否基于来源
4. 检测到幻觉时触发重新生成
5. 记录验证结果
6. 细粒度幻觉检测（句子级别）
7. 幻觉类型分类和严重程度评估
8. 幻觉预防和修复
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from ..models import SharedState, ScreenplayFragment, RetrievedDocument
from ...services.llm.service import LLMService
from ...services.hallucination_detection import (
    GranularHallucinationDetector,
    HallucinationClassifier,
    HallucinationPrevention,
    HallucinationRepairer,
    UnifiedHallucinationService,
    FragmentHallucinationResult,
    HallucinationType
)
from ...infrastructure.logging import get_agent_logger
from ...infrastructure.error_handler import (
    FactCheckerError,
    handle_component_errors
)


logger = logging.getLogger(__name__)
agent_logger = get_agent_logger(__name__)


async def verify_fragment(
    fragment: ScreenplayFragment,
    retrieved_docs: List[RetrievedDocument],
    llm_service: LLMService
) -> Tuple[bool, List[str]]:
    """
    验证剧本片段与源文档的一致性
    - 使用 LLM 将片段与源文档进行比较
    - 检测不存在的代码、函数、参数
    - 返回验证结果和幻觉列表
    
    Args:
        fragment: 要验证的剧本片段
        retrieved_docs: 检索到的源文档列表
        llm_service: LLM 服务
        
    Returns:
        (is_valid, hallucinations) 元组
        - is_valid: 片段是否有效（无幻觉）
        - hallucinations: 检测到的幻觉列表
    """
    try:
        # 如果没有检索文档，无法验证
        if not retrieved_docs:
            logger.warning(
                f"No retrieved documents to verify fragment for step {fragment.step_id}"
            )
            # 没有源文档时，假设片段有效（可能是 research_mode）
            return True, []
        
        # 构建源文档摘要
        sources_summary = "\n\n".join([
            f"源文档 {i+1} ({doc.source}):\n{doc.content[:1000]}..."
            for i, doc in enumerate(retrieved_docs[:5])  # 最多使用前 5 个文档
        ])
        
        # 构建验证提示
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个事实检查专家。你的任务是验证生成的内容是否与提供的源文档一致。\n"
                    "请仔细检查以下内容：\n"
                    "1. 代码示例是否存在于源文档中\n"
                    "2. 函数名、类名、参数名是否准确\n"
                    "3. 技术细节是否与源文档匹配\n"
                    "4. 是否有编造的信息（幻觉）\n\n"
                    "如果发现幻觉，请列出具体的幻觉内容。\n"
                    "如果内容完全基于源文档，请回答 'VALID'。\n"
                    "如果发现幻觉，请回答 'INVALID' 并列出幻觉，格式为：\n"
                    "INVALID\n"
                    "- 幻觉1: 描述\n"
                    "- 幻觉2: 描述\n"
                )
            },
            {
                "role": "user",
                "content": (
                    f"源文档内容:\n{sources_summary}\n\n"
                    f"生成的片段内容:\n{fragment.content}\n\n"
                    f"请验证生成的片段是否与源文档一致："
                )
            }
        ]
        
        # 调用 LLM 进行验证
        logger.info(f"Fact Checker: Verifying fragment for step {fragment.step_id}")
        
        response = await llm_service.chat_completion(
            messages=messages,
            task_type="high_performance",  # 事实检查使用高性能模型
            temperature=0.1,  # 低温度以获得更确定的结果
            max_tokens=1000
        )
        
        # 解析响应
        response_text = response.strip()
        
        logger.info(f"Fact Checker: LLM response: {response_text[:100]}")
        
        if response_text.startswith("VALID"):
            logger.info(f"Fact Checker: Fragment for step {fragment.step_id} is valid")
            return True, []
        
        elif response_text.startswith("INVALID"):
            # 提取幻觉列表
            hallucinations = []
            lines = response_text.split('\n')
            
            for line in lines[1:]:  # 跳过第一行 "INVALID"
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    # 移除列表标记
                    hallucination = line.lstrip('-•').strip()
                    if hallucination:
                        hallucinations.append(hallucination)
            
            logger.warning(
                f"Fact Checker: Fragment for step {fragment.step_id} contains "
                f"{len(hallucinations)} hallucinations"
            )
            
            return False, hallucinations
        
        else:
            # 无法解析响应，使用启发式方法
            logger.warning(
                f"Fact Checker: Unable to parse LLM response, using heuristic method"
            )
            return _heuristic_verification(fragment, retrieved_docs)
    
    except Exception as e:
        logger.error(f"Fact Checker verification failed: {str(e)}")
        # 验证失败时，假设片段有效以避免阻塞流程
        return True, []


def _heuristic_verification(
    fragment: ScreenplayFragment,
    retrieved_docs: List[RetrievedDocument]
) -> Tuple[bool, List[str]]:
    """
    启发式验证方法（回退方法）
    
    使用简单的启发式规则检测明显的幻觉。
    
    Args:
        fragment: 要验证的剧本片段
        retrieved_docs: 检索到的源文档列表
        
    Returns:
        (is_valid, hallucinations) 元组
    """
    hallucinations = []
    
    # 如果没有检索文档，无法验证，假设有效（可能是 research_mode）
    if not retrieved_docs:
        logger.info("Heuristic verification: No documents to verify against, assuming valid")
        return True, []
    
    # 合并所有源文档内容
    all_source_content = " ".join([doc.content for doc in retrieved_docs])
    
    # 规则 1: 检查代码块中的函数定义
    # 提取片段中的代码块
    code_blocks = re.findall(r'```[\s\S]*?```', fragment.content)
    
    for code_block in code_blocks:
        # 提取函数定义（Python 和 JavaScript 风格）
        # Python: def function_name(
        function_defs = re.findall(r'def\s+(\w+)\s*\(', code_block)
        # JavaScript/TypeScript: function function_name( 或 const function_name = 
        function_defs.extend(re.findall(r'function\s+(\w+)\s*\(', code_block))
        function_defs.extend(re.findall(r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\()', code_block))
        
        for func_name in function_defs:
            # 检查函数名是否在源文档中（使用单词边界）
            if not re.search(r'\b' + re.escape(func_name) + r'\b', all_source_content):
                hallucinations.append(
                    f"函数 '{func_name}' 未在源文档中找到"
                )
    
    # 规则 2: 检查明确的函数调用引用
    # 查找类似 `function_name()` 的模式
    function_calls = re.findall(r'`(\w+)\(\)`', fragment.content)
    
    for func_call in function_calls:
        # 检查函数名是否在源文档中（使用单词边界）
        if not re.search(r'\b' + re.escape(func_call) + r'\b', all_source_content):
            hallucinations.append(
                f"函数调用 '{func_call}()' 未在源文档中找到"
            )
    
    # 规则 3: 检查类名引用
    # 查找类似 `ClassName` 类 的模式
    class_refs = re.findall(r'`(\w+)`\s*类', fragment.content)
    
    for class_name in class_refs:
        # 检查类名是否在源文档中（使用单词边界）
        if not re.search(r'\b' + re.escape(class_name) + r'\b', all_source_content):
            hallucinations.append(
                f"类 '{class_name}' 未在源文档中找到"
            )
    
    # 如果检测到幻觉，返回无效
    if hallucinations:
        logger.warning(
            f"Heuristic verification detected {len(hallucinations)} potential hallucinations"
        )
        return False, hallucinations
    
    # 没有检测到明显的幻觉
    logger.info("Heuristic verification: No obvious hallucinations detected")
    return True, []


async def verify_fragment_node(
    state: SharedState,
    llm_service: LLMService
) -> SharedState:
    """
    事实检查器节点函数（带错误处理）
    - 根据检索的文档验证最新片段
    - 移除无效片段并增加重试计数器
    - 为路由设置 fact_check_passed 标志
    - 使用警告处理事实检查器失败
    
    Args:
        state: 共享状态
        llm_service: LLM 服务
        
    Returns:
        更新后的共享状态
    """
    @handle_component_errors(
        component_name="fact_checker",
        fallback_value=state,
        log_level="warning"
    )
    async def _verify_with_error_handling():
        # 获取最新的片段
        if not state.fragments:
            logger.warning("Fact Checker: No fragments to verify")
            # 假设有效以继续流程
            state.fact_check_passed = True
            return state
        
        latest_fragment = state.fragments[-1]
        
        logger.info(
            f"Fact Checker: Verifying fragment for step {latest_fragment.step_id}"
        )
        
        # Log agent transition
        agent_logger.log_agent_transition(
            from_agent="writer",
            to_agent="fact_checker",
            step_id=latest_fragment.step_id,
            reason="verify_fragment"
        )
        
        # 验证片段
        try:
            is_valid, hallucinations = await verify_fragment(
                fragment=latest_fragment,
                retrieved_docs=state.retrieved_docs,
                llm_service=llm_service
            )
        except Exception as e:
            # 如果验证失败，抛出 FactCheckerError 以触发错误处理
            logger.error(f"Fact Checker verification failed: {str(e)}")
            raise FactCheckerError(f"Verification failed: {str(e)}")
        
        # Log fact check result
        agent_logger.log_fact_check_result(
            step_id=latest_fragment.step_id,
            is_valid=is_valid,
            hallucinations=hallucinations,
            verification_method="llm_verification"
        )
        
        if is_valid:
            # 片段有效，设置标志
            state.fact_check_passed = True
            
            # 标记当前步骤为完成
            current_step = state.get_current_step()
            if current_step:
                current_step.status = "completed"
            
            # 记录日志
            state.add_log_entry(
                agent_name="fact_checker",
                action="verification_passed",
                details={
                    "step_id": latest_fragment.step_id,
                    "fragment_length": len(latest_fragment.content),
                    "num_sources": len(latest_fragment.sources)
                }
            )
            
            logger.info(
                f"Fact Checker: Fragment for step {latest_fragment.step_id} passed verification"
            )
        
        else:
            # 片段无效，包含幻觉
            state.fact_check_passed = False
            
            # 移除无效片段
            state.fragments.pop()
            
            # 获取当前步骤并增加重试计数器
            current_step = state.get_current_step()
            if current_step:
                current_step.retry_count += 1
                current_step.status = "in_progress"  # 重置状态以便重新生成
                
                logger.warning(
                    f"Fact Checker: Fragment for step {current_step.step_id} failed verification, "
                    f"retry count: {current_step.retry_count}"
                )
            
            # 记录日志
            state.add_log_entry(
                agent_name="fact_checker",
                action="verification_failed",
                details={
                    "step_id": latest_fragment.step_id,
                    "hallucinations": hallucinations,
                    "retry_count": current_step.retry_count if current_step else 0
                }
            )
            
            logger.warning(
                f"Fact Checker: Detected {len(hallucinations)} hallucinations in "
                f"fragment for step {latest_fragment.step_id}"
            )
        
        return state
    
    # 执行带错误处理的验证
    result = await _verify_with_error_handling()
    
    # 如果错误处理返回了回退值（原始 state），假设片段有效
    if result == state and not hasattr(state, 'fact_check_passed'):
        state.fact_check_passed = True
        logger.warning("Fact Checker: Error occurred, assuming fragment is valid")
    
    return result


async def granular_verify_fragment(
    fragment: ScreenplayFragment,
    retrieved_docs: List[RetrievedDocument],
    llm_service: LLMService,
    enable_repair: bool = True
) -> Dict[str, Any]:
    """
    细粒度幻觉检测和修复

    功能：
    1. 句子级别的幻觉检测
    2. 幻觉类型分类
    3. 严重程度评估
    4. 幻觉修复（可选）

    Args:
        fragment: 要验证的剧本片段
        retrieved_docs: 检索到的源文档列表
        llm_service: LLM 服务
        enable_repair: 是否启用自动修复

    Returns:
        包含检测结果和修复建议的字典
    """
    detector = GranularHallucinationDetector(llm_service)
    classifier = HallucinationClassifier()
    repairer = HallucinationRepairer(llm_service)

    try:
        logger.info(
            f"Granular Fact Checker: Starting detailed verification for "
            f"step {fragment.step_id}"
        )

        result = await detector.detect_sentence_level(
            content=fragment.content,
            retrieved_docs=[
                {"id": doc.source, "content": doc.content}
                for doc in retrieved_docs
            ],
            fragment_id=fragment.step_id
        )

        hallucinations = [
            sentence for sentence in result.sentences
            if sentence.is_hallucination
        ]

        classified_hallucinations = []
        for hallucination in hallucinations:
            hallucination_type = classifier.classify_hallucination(
                hallucination.sentence,
                context=fragment.content
            )
            severity = classifier.get_severity(hallucination_type.value)

            classified_hallucinations.append({
                "sentence": hallucination.sentence,
                "is_hallucination": hallucination.is_hallucination,
                "confidence": hallucination.confidence,
                "type": hallucination_type.value,
                "severity": severity,
                "reason": hallucination.reason,
                "supporting_source": hallucination.supporting_source
            })

        repair_result = None
        if enable_repair and hallucinations:
            repair_result = await repairer.repair_hallucination(
                fragment=fragment,
                hallucinations=[
                    {
                        "sentence": h.sentence,
                        "type": classifier.classify_hallucination(
                            h.sentence, fragment.content
                        ).value,
                        "reason": h.reason
                    }
                    for h in hallucinations
                ],
                retrieved_docs=[
                    {"id": doc.source, "content": doc.content}
                    for doc in retrieved_docs
                ]
            )

        is_valid = len(hallucinations) == 0

        return {
            "is_valid": is_valid,
            "fragment_id": fragment.step_id,
            "total_sentences": len(result.sentences),
            "hallucination_count": len(hallucinations),
            "hallucinations": classified_hallucinations,
            "overall_confidence": result.overall_confidence,
            "hallucination_rate": result.hallucination_rate,
            "repair_result": repair_result,
            "recommendations": result.recommendations
        }

    except Exception as e:
        logger.error(f"Granular fact check failed: {str(e)}")
        return {
            "is_valid": True,
            "error": str(e),
            "fragment_id": fragment.step_id,
            "fallback_to_basic": True
        }


async def enhance_prompt_with_prevention(
    original_prompt: str,
    retrieved_docs: List[RetrievedDocument],
    llm_service: LLMService
) -> str:
    """
    使用预防机制增强 prompt

    功能：
    1. 提取可用函数和类列表
    2. 添加约束防止幻觉
    3. 添加正确/错误示例

    Args:
        original_prompt: 原始 prompt
        retrieved_docs: 检索到的文档
        llm_service: LLM 服务

    Returns:
        增强后的 prompt
    """
    prevention = HallucinationPrevention(llm_service)

    enhanced_prompt = prevention.enhance_prompt_with_constraints(
        original_prompt,
        [{"id": doc.source, "content": doc.content} for doc in retrieved_docs]
    )

    enhanced_prompt = prevention.add_verification_examples(enhanced_prompt)

    return enhanced_prompt
