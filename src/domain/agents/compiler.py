"""
Compiler Agent - Integrates fragments into final screenplay

编译器智能体负责将所有生成的剧本片段整合为最终的连贯剧本。
它使用轻量级 LLM 模型来平滑章节之间的过渡，添加引言和结论，
并格式化最终输出。
"""

import logging
from typing import List, Dict, Any

from ..models import SharedState, ScreenplayFragment
from ...services.llm.service import LLMService


logger = logging.getLogger(__name__)


async def compile_screenplay(
    state: SharedState,
    llm_service: LLMService
) -> str:
    """
    编译剧本 - 将所有片段整合为最终剧本
    
    功能：
    1. 按顺序集成所有剧本片段
    2. 平滑章节之间的过渡
    3. 添加引言和结论
    4. 格式化最终输出
    
    Args:
        state: 共享状态对象
        llm_service: LLM 服务实例
        
    Returns:
        最终整合的剧本文本
    """
    logger.info("Compiler: Starting screenplay compilation")
    
    # 记录日志
    state.add_log_entry(
        agent_name="compiler",
        action="start_compilation",
        details={
            "fragment_count": len(state.fragments),
            "outline_steps": len(state.outline)
        }
    )
    
    # 检查是否有片段
    if not state.fragments:
        logger.warning("Compiler: No fragments to compile")
        state.add_log_entry(
            agent_name="compiler",
            action="compilation_skipped",
            details={"reason": "no_fragments"}
        )
        return _generate_empty_screenplay(state)
    
    # 按 step_id 排序片段
    sorted_fragments = sorted(state.fragments, key=lambda f: f.step_id)
    
    # 构建编译提示
    compilation_prompt = _build_compilation_prompt(state, sorted_fragments)
    
    try:
        # 使用轻量级模型进行编译
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专业的剧本编译器。你的任务是将多个剧本片段整合为一个连贯、"
                    "流畅的完整剧本。你需要：\n"
                    "1. 保持每个片段的核心内容和风格\n"
                    "2. 在章节之间添加自然的过渡\n"
                    "3. 添加引人入胜的引言\n"
                    "4. 添加总结性的结论\n"
                    "5. 确保整体叙述连贯一致\n"
                    "6. 使用清晰的 Markdown 格式\n\n"
                    "不要改变片段的核心内容，只需要改善连贯性和可读性。"
                )
            },
            {
                "role": "user",
                "content": compilation_prompt
            }
        ]
        
        final_screenplay = await llm_service.chat_completion(
            messages=messages,
            task_type="lightweight",
            temperature=0.7,
            max_tokens=4000
        )
        
        logger.info("Compiler: Successfully compiled screenplay")
        state.add_log_entry(
            agent_name="compiler",
            action="compilation_completed",
            details={
                "final_length": len(final_screenplay),
                "fragments_integrated": len(sorted_fragments)
            }
        )
        
        return final_screenplay
        
    except Exception as e:
        logger.error(f"Compiler: Failed to compile screenplay: {str(e)}")
        state.add_log_entry(
            agent_name="compiler",
            action="compilation_failed",
            details={"error": str(e)}
        )
        
        # 降级：简单拼接片段
        logger.info("Compiler: Falling back to simple concatenation")
        return _fallback_compilation(state, sorted_fragments)


def _build_compilation_prompt(
    state: SharedState,
    fragments: List[ScreenplayFragment]
) -> str:
    """
    构建编译提示
    
    Args:
        state: 共享状态
        fragments: 排序后的片段列表
        
    Returns:
        编译提示文本
    """
    prompt_parts = [
        f"# 剧本主题\n{state.user_topic}\n",
        f"\n# 项目上下文\n{state.project_context}\n" if state.project_context else "",
        f"\n# 全局语调\n{state.global_tone}\n",
        "\n# 剧本片段\n"
    ]
    
    # 添加每个片段
    for i, fragment in enumerate(fragments, 1):
        # 获取对应的大纲步骤描述
        step_desc = "未知步骤"
        for step in state.outline:
            if step.step_id == fragment.step_id:
                step_desc = step.description
                break
        
        prompt_parts.append(
            f"\n## 片段 {i} (步骤 {fragment.step_id}: {step_desc})\n"
            f"**使用的风格**: {fragment.skill_used}\n"
            f"**来源数量**: {len(fragment.sources)}\n\n"
            f"{fragment.content}\n"
        )
    
    prompt_parts.append(
        "\n---\n\n"
        "请将以上片段整合为一个完整、连贯的剧本。要求：\n"
        "1. 添加一个引言，介绍剧本主题和目标\n"
        "2. 在片段之间添加自然的过渡句\n"
        "3. 保持每个片段的核心内容和风格\n"
        "4. 添加一个结论，总结关键要点\n"
        "5. 使用清晰的 Markdown 格式（标题、列表、代码块等）\n"
        "6. 确保整体叙述流畅、逻辑清晰\n\n"
        "请直接输出最终剧本，不要添加额外的说明或注释。"
    )
    
    return "".join(prompt_parts)


def _generate_empty_screenplay(state: SharedState) -> str:
    """
    生成空剧本（当没有片段时）
    
    Args:
        state: 共享状态
        
    Returns:
        空剧本文本
    """
    return f"""# {state.user_topic}

## 概述

本剧本旨在探讨 "{state.user_topic}"。

{f"**项目上下文**: {state.project_context}" if state.project_context else ""}

## 内容

由于缺少足够的信息或检索内容，无法生成详细的剧本内容。

建议：
- 提供更多的项目上下文
- 确保代码库已正确索引
- 检查检索配置是否正确

## 总结

请提供更多信息以生成完整的剧本。
"""


def _fallback_compilation(
    state: SharedState,
    fragments: List[ScreenplayFragment]
) -> str:
    """
    降级编译 - 简单拼接片段
    
    当 LLM 编译失败时使用此方法。
    
    Args:
        state: 共享状态
        fragments: 排序后的片段列表
        
    Returns:
        简单拼接的剧本文本
    """
    parts = [
        f"# {state.user_topic}\n",
        "\n## 概述\n",
        f"本剧本探讨 \"{state.user_topic}\"。\n"
    ]
    
    if state.project_context:
        parts.append(f"\n**项目上下文**: {state.project_context}\n")
    
    parts.append("\n---\n")
    
    # 添加每个片段
    for i, fragment in enumerate(fragments, 1):
        # 获取对应的大纲步骤描述
        step_desc = "未知步骤"
        for step in state.outline:
            if step.step_id == fragment.step_id:
                step_desc = step.description
                break
        
        parts.append(f"\n## 第 {i} 部分: {step_desc}\n\n")
        parts.append(f"{fragment.content}\n")
        
        # 添加来源信息（如果有）
        if fragment.sources:
            parts.append(f"\n*来源: {', '.join(fragment.sources[:3])}")
            if len(fragment.sources) > 3:
                parts.append(f" 等 {len(fragment.sources)} 个来源")
            parts.append("*\n")
    
    # 添加结论
    parts.append("\n---\n\n## 总结\n\n")
    parts.append(
        f"本剧本涵盖了 {len(fragments)} 个主要部分，"
        f"基于 {sum(len(f.sources) for f in fragments)} 个来源文档生成。\n"
    )
    
    return "".join(parts)
