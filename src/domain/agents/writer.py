"""Writer Agent - Generates screenplay fragments

The Writer agent is responsible for:
1. Applying skill-specific generation strategies
2. Generating screenplay fragments based on retrieved content
3. Grounding all statements in retrieved sources
4. Handling empty retrievals with research mode
5. Creating fragments with source attribution
"""

import logging
from typing import List, Dict, Any, Optional

from ..models import SharedState, ScreenplayFragment, RetrievedDocument, OutlineStep
from ..skills import SKILLS
from ...services.llm.service import LLMService
from ...infrastructure.error_handler import (
    WriterError,
    handle_component_errors
)


logger = logging.getLogger(__name__)


# Skill-specific prompts and guidelines
SKILL_PROMPTS = {
    "standard_tutorial": {
        "system_prompt": (
            "你是一个专业的技术教程编写专家。你的任务是生成清晰、结构化的教程内容。\n"
            "要求：\n"
            "- 使用专业、正式的语调\n"
            "- 提供清晰的步骤说明\n"
            "- 包含代码示例和解释\n"
            "- 所有陈述必须基于提供的检索内容\n"
            "- 不要编造不存在的代码、函数或参数\n"
            "- 引用来源时使用 [来源: 文件路径] 格式"
        ),
        "user_template": (
            "步骤描述: {step_description}\n\n"
            "检索到的相关内容:\n{retrieved_content}\n\n"
            "请基于以上内容生成这一步骤的剧本片段。确保所有陈述都有来源支持。"
        )
    },
    
    "warning_mode": {
        "system_prompt": (
            "你是一个技术风险评估专家。你的任务是突出显示废弃功能、安全问题和潜在风险。\n"
            "要求：\n"
            "- 使用警示性、谨慎的语调\n"
            "- 明确指出废弃的功能和替代方案\n"
            "- 强调安全问题和 FIXME 标记\n"
            "- 提供迁移建议和最佳实践\n"
            "- 所有警告必须基于检索内容中的标记\n"
            "- 使用 ⚠️ 符号标记重要警告"
        ),
        "user_template": (
            "步骤描述: {step_description}\n\n"
            "检索到的内容（包含警告标记）:\n{retrieved_content}\n\n"
            "请生成突出显示风险和废弃内容的剧本片段。"
        )
    },
    
    "visualization_analogy": {
        "system_prompt": (
            "你是一个善于用类比和可视化解释复杂概念的教育专家。\n"
            "要求：\n"
            "- 使用生动、引人入胜的语调\n"
            "- 将复杂的技术概念转化为易懂的类比\n"
            "- 使用比喻、故事和日常例子\n"
            "- 建议可视化图表或示意图\n"
            "- 保持技术准确性的同时提高可理解性\n"
            "- 所有类比必须基于检索内容的实际功能"
        ),
        "user_template": (
            "步骤描述: {step_description}\n\n"
            "检索到的技术内容:\n{retrieved_content}\n\n"
            "请使用类比和可视化方法生成易于理解的剧本片段。"
        )
    },
    
    "research_mode": {
        "system_prompt": (
            "你是一个研究顾问，擅长识别知识缺口并建议研究方向。\n"
            "要求：\n"
            "- 使用探索性、开放的语调\n"
            "- 明确指出信息不足的地方\n"
            "- 不要编造或猜测缺失的信息\n"
            "- 建议可能的研究方向和资源\n"
            "- 提出需要进一步调查的问题\n"
            "- 使用 '需要进一步研究' 或 '信息不足' 等明确表述"
        ),
        "user_template": (
            "步骤描述: {step_description}\n\n"
            "检索到的内容（可能不完整）:\n{retrieved_content}\n\n"
            "请生成承认信息缺口的剧本片段，并建议研究方向。"
        )
    },
    
    "meme_style": {
        "system_prompt": (
            "你是一个技术内容创作者，擅长用轻松幽默的方式讲解技术。\n"
            "要求：\n"
            "- 使用轻松、幽默、非正式的语调\n"
            "- 可以使用网络流行语和梗\n"
            "- 保持技术准确性\n"
            "- 让内容有趣且易于记忆\n"
            "- 适当使用表情符号\n"
            "- 所有技术细节必须基于检索内容"
        ),
        "user_template": (
            "步骤描述: {step_description}\n\n"
            "检索到的技术内容:\n{retrieved_content}\n\n"
            "请用轻松幽默的方式生成剧本片段。"
        )
    },
    
    "fallback_summary": {
        "system_prompt": (
            "你是一个技术摘要专家，擅长提供高层次的概述。\n"
            "要求：\n"
            "- 使用中性、客观的语调\n"
            "- 提供简洁的高层次概述\n"
            "- 避免深入技术细节\n"
            "- 关注核心概念和关键要点\n"
            "- 承认详细信息不可用时的限制\n"
            "- 基于可用的检索内容提供摘要"
        ),
        "user_template": (
            "步骤描述: {step_description}\n\n"
            "可用的内容:\n{retrieved_content}\n\n"
            "请提供高层次的概述性剧本片段。"
        )
    }
}


def apply_skill(
    skill_name: str,
    step: OutlineStep,
    retrieved_docs: List[RetrievedDocument],
    llm_service: LLMService
) -> Dict[str, Any]:
    """
    应用特定 Skill 的生成策略

    - 为每种 Skill 模式定义提示和指南
    - 确保 research_mode 明确指出信息缺口
    
    Args:
        skill_name: Skill 名称
        step: 当前大纲步骤
        retrieved_docs: 检索到的文档列表
        llm_service: LLM 服务
        
    Returns:
        包含 messages 和 metadata 的字典
    """
    if skill_name not in SKILL_PROMPTS:
        logger.warning(f"Unknown skill: {skill_name}, using standard_tutorial")
        skill_name = "standard_tutorial"
    
    skill_config = SKILL_PROMPTS[skill_name]
    
    # 构建检索内容摘要
    if not retrieved_docs:
        retrieved_content = "[无检索内容]"
    else:
        retrieved_content = "\n\n".join([
            f"文档 {i+1} (来源: {doc.source}, 置信度: {doc.confidence:.2f}):\n{doc.content[:1000]}..."
            for i, doc in enumerate(retrieved_docs[:5])  # 最多使用前 5 个文档
        ])
    
    # 构建消息
    system_prompt = skill_config["system_prompt"]
    user_prompt = skill_config["user_template"].format(
        step_description=step.description,
        retrieved_content=retrieved_content
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # 返回消息和元数据
    return {
        "messages": messages,
        "metadata": {
            "skill_used": skill_name,
            "num_docs": len(retrieved_docs),
            "sources": [doc.source for doc in retrieved_docs]
        }
    }


async def generate_fragment(
    state: SharedState,
    llm_service: LLMService
) -> SharedState:
    """
    编剧主函数（带错误处理）
    
    实现需求 12.6, 18.4:
    - 应用活动 Skill 生成剧本片段
    - 将所有陈述基于检索内容
    - 创建带来源的 ScreenplayFragment
    - 使用更简单的 Skill 处理编剧失败
    
    实现需求 7.2, 7.3, 7.5, 7.6, 7.7:
    - 通过切换到 research_mode 处理空检索
    - 生成承认信息缺口的片段
    - 设置 awaiting_user_input 状态请求用户输入
    
    Args:
        state: 共享状态
        llm_service: LLM 服务
        
    Returns:
        更新后的共享状态
    """
    @handle_component_errors(
        component_name="writer",
        fallback_value=state,
        log_level="warning"
    )
    async def _generate_with_error_handling():
        # 获取当前步骤
        current_step = state.get_current_step()
        if not current_step:
            logger.warning("No current step to generate fragment for")
            return state
        
        logger.info(
            f"Writer: Generating fragment for step {current_step.step_id}: "
            f"{current_step.description}"
        )
        
        # 检查是否有检索内容
        if not state.retrieved_docs:
            logger.warning(
                f"No retrieved documents for step {current_step.step_id}, "
                f"switching to research_mode"
            )
            
            # 切换到 research_mode（需求 7.2）
            original_skill = state.current_skill
            state.current_skill = "research_mode"
            
            # 设置等待用户输入状态（需求 7.5, 7.6）
            state.awaiting_user_input = True
            state.user_input_prompt = (
                f"步骤 {current_step.step_id} 缺少相关信息。"
                f"请提供关于 '{current_step.description}' 的额外上下文或资源。"
            )
            
            # 记录日志
            state.add_log_entry(
                agent_name="writer",
                action="empty_retrieval_research_mode",
                details={
                    "step_id": current_step.step_id,
                    "original_skill": original_skill,
                    "new_skill": "research_mode",
                    "user_input_prompt": state.user_input_prompt
                }
            )
            
            # 生成承认信息缺口的片段（需求 7.3）
            fragment_content = (
                f"## 步骤 {current_step.step_id}: {current_step.description}\n\n"
                f"**信息不足**: 当前没有足够的信息来详细说明这一步骤。\n\n"
                f"**需要进一步研究**:\n"
                f"- 查找关于 '{current_step.description}' 的官方文档\n"
                f"- 检查项目代码库中的相关实现\n"
                f"- 咨询团队成员或技术专家\n\n"
                f"**建议**: 在继续之前，请提供更多上下文信息。"
            )
            
            # 创建片段
            fragment = ScreenplayFragment(
                step_id=current_step.step_id,
                content=fragment_content,
                skill_used="research_mode",
                sources=[]
            )
            
            state.fragments.append(fragment)
            
            logger.info(
                f"Writer: Generated research_mode fragment for step {current_step.step_id}, "
                f"awaiting user input"
            )
            
            return state
        
        # 应用当前 Skill 生成策略
        skill_data = apply_skill(
            skill_name=state.current_skill,
            step=current_step,
            retrieved_docs=state.retrieved_docs,
            llm_service=llm_service
        )
        
        # 调用 LLM 生成片段
        logger.info(f"Writer: Calling LLM with skill '{state.current_skill}'")
        
        try:
            fragment_content = await llm_service.chat_completion(
                messages=skill_data["messages"],
                task_type="lightweight",  # 编剧使用轻量级模型
                temperature=0.7,
                max_tokens=2000
            )
        except Exception as e:
            # 如果 LLM 调用失败，抛出 WriterError 以触发错误处理
            logger.error(f"LLM generation failed: {str(e)}")
            raise WriterError(f"LLM generation failed: {str(e)}")
        
        # 创建 ScreenplayFragment
        fragment = ScreenplayFragment(
            step_id=current_step.step_id,
            content=fragment_content,
            skill_used=state.current_skill,
            sources=skill_data["metadata"]["sources"]
        )
        
        # 添加到状态
        state.fragments.append(fragment)
        
        # 更新步骤状态
        current_step.status = "completed"
        
        # 记录日志
        state.add_log_entry(
            agent_name="writer",
            action="generate_fragment",
            details={
                "step_id": current_step.step_id,
                "skill_used": state.current_skill,
                "num_sources": len(skill_data["metadata"]["sources"]),
                "sources": skill_data["metadata"]["sources"],
                "fragment_length": len(fragment_content)
            }
        )
        
        logger.info(
            f"Writer: Successfully generated fragment for step {current_step.step_id} "
            f"using skill '{state.current_skill}'"
        )
        
        return state
    
    # 执行带错误处理的生成
    result = await _generate_with_error_handling()
    
    # 如果错误处理返回了回退值（原始 state），尝试使用 fallback_summary
    if result == state and (not state.fragments or state.fragments[-1].step_id != state.get_current_step().step_id):
        logger.info("Writer: Error occurred, attempting fallback to fallback_summary skill")
        
        # Switch to fallback_summary skill
        if state.current_skill != "fallback_summary":
            state.current_skill = "fallback_summary"
            state.add_log_entry(
                agent_name="writer",
                action="skill_switch",
                details={
                    "reason": "generation_failure",
                    "new_skill": "fallback_summary"
                }
            )
        
        current_step = state.get_current_step()
        if current_step:
            # 生成简单的回退片段
            fallback_content = (
                f"## 步骤 {current_step.step_id}: {current_step.description}\n\n"
                f"**概述**: 由于技术问题，无法生成详细内容。\n\n"
                f"**要点**: 请参考相关文档了解更多信息。"
            )
            
            fragment = ScreenplayFragment(
                step_id=current_step.step_id,
                content=fallback_content,
                skill_used="fallback_summary",
                sources=[]
            )
            
            state.fragments.append(fragment)
            current_step.status = "completed"
            
            # 记录日志
            state.add_log_entry(
                agent_name="writer",
                action="generate_fragment_fallback",
                details={
                    "step_id": current_step.step_id,
                    "skill_used": "fallback_summary"
                }
            )
            
            logger.info(f"Writer: Generated fallback fragment for step {current_step.step_id}")
    
    return result


async def handle_user_input_resume(
    state: SharedState,
    user_input: str,
    llm_service: LLMService
) -> SharedState:
    """
    处理用户输入后恢复执行
    
    实现需求 7.7:
    - 当用户提供输入后，系统应恢复执行并继续处理
    
    Args:
        state: 共享状态
        user_input: 用户提供的输入
        llm_service: LLM 服务
        
    Returns:
        更新后的共享状态
    """
    logger.info("Writer: Resuming execution with user input")
    
    # 清除等待状态
    state.awaiting_user_input = False
    state.user_input_prompt = None
    
    # 获取当前步骤
    current_step = state.get_current_step()
    if not current_step:
        logger.warning("No current step to resume")
        return state
    
    # 将用户输入作为额外上下文
    # 创建一个虚拟的检索文档
    from ..models import RetrievedDocument
    
    user_doc = RetrievedDocument(
        content=user_input,
        source="user_input",
        confidence=1.0,
        metadata={"type": "user_provided"}
    )
    
    # 添加到检索文档列表
    state.retrieved_docs.append(user_doc)
    
    # 切换回原来的 Skill（如果之前不是 research_mode）
    if state.current_skill == "research_mode":
        state.current_skill = "standard_tutorial"
    
    # 重新生成片段
    state = await generate_fragment(state, llm_service)
    
    # 记录日志
    state.add_log_entry(
        agent_name="writer",
        action="resume_with_user_input",
        details={
            "step_id": current_step.step_id,
            "user_input_length": len(user_input)
        }
    )
    
    logger.info(f"Writer: Resumed execution for step {current_step.step_id}")
    
    return state
